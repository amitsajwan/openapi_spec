import asyncio
import logging
from typing import Any, Callable, Awaitable, Dict, Optional
from collections import defaultdict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import StateSnapshot # For type hinting current_snapshot

from models import BotState, ExecutionGraphState, Node as PlanNode # Import PlanNode for type hint
# We need access to the graph definition to get node details for the UI
from execution_graph_definition import ExecutionGraphDefinition


logger = logging.getLogger(__name__)

class GraphExecutionManager:
    def __init__(
        self,
        runnable_graph: Any, # This is the compiled LangGraph
        graph_definition: ExecutionGraphDefinition, # Pass the definition object
        websocket_callback: Callable[[str, Dict[str, Any], Optional[str]], Awaitable[None]],
        planning_checkpointer: Optional[BaseCheckpointSaver],
        main_planning_session_id: str
    ):
        self.runnable_graph: Any = runnable_graph
        self.graph_definition = graph_definition # Store for fetching node details
        self.websocket_callback = websocket_callback
        self.planning_checkpointer = planning_checkpointer
        self.main_planning_session_id = main_planning_session_id
        self.resume_queues: Dict[str, asyncio.Queue[Any]] = defaultdict(asyncio.Queue)
        logger.info(f"GraphExecutionManager initialized for Main Planning Session ID: {self.main_planning_session_id}")

    async def submit_resume_data(self, main_session_id: str, resume_data: Any) -> bool:
        # ... (submit_resume_data remains the same)
        if main_session_id not in self.resume_queues:
             self.resume_queues[main_session_id] = asyncio.Queue()
             logger.info(f"Initialized resume queue for Main Session ID: {main_session_id} on first submit.")
        if main_session_id in self.resume_queues:
            try:
                await self.resume_queues[main_session_id].put(resume_data)
                logger.info(f"Resume data submitted for Main Session ID: {main_session_id}. Data: {str(resume_data)[:100]}...")
                return True
            except Exception as e:
                logger.error(f"Error putting resume data onto queue for Main Session ID {main_session_id}: {e}")
                return False
        else: 
            logger.warning(f"No active resume queue found for Main Session ID: {main_session_id}.")
            return False


    async def execute_workflow(
        self,
        initial_graph_values: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        graph2_thread_id = config.get("configurable", {}).get("thread_id")
        if not graph2_thread_id:
            await self.websocket_callback("execution_error", {"error": "Config error: thread_id missing."}, None)
            raise ValueError("thread_id missing for ExecutionGraph.")

        logger.info(f"--- [Graph 2] Starting Workflow for ThreadID: {graph2_thread_id} ---")
        current_input_for_astream: Optional[Dict[str, Any]] = initial_graph_values
        graph_has_ended = False

        while not graph_has_ended:
            try:
                logger.debug(f"ThreadID '{graph2_thread_id}': Calling astream. Input: {str(current_input_for_astream)[:100]}...")
                
                # Collect all events from this astream iteration
                events_in_iteration = []
                async for event in self.runnable_graph.astream(current_input_for_astream, config=config):
                    events_in_iteration.append(event)
                    logger.debug(f"ThreadID '{graph2_thread_id}': Event received: {list(event.keys())}")
                    if "__end__" in event:
                        logger.info(f"ThreadID '{graph2_thread_id}': __end__ node reached.")
                        graph_has_ended = True
                        break 
                
                if graph_has_ended:
                    logger.info(f"ThreadID '{graph2_thread_id}': Graph ended via __end__ node.")
                    break # Exit while loop

                # If astream finished without hitting __end__, it means an interrupt occurred.
                logger.info(f"ThreadID '{graph2_thread_id}': astream iteration completed. Graph paused due to interrupt_before.")
                
                # Get current state to find out what's next and prepare UI data
                current_snapshot: StateSnapshot = self.runnable_graph.get_state(config)
                if not current_snapshot:
                    logger.error(f"ThreadID '{graph2_thread_id}': Failed to get state after interrupt.")
                    await self.websocket_callback("execution_error", {"error": "Failed to get state after interrupt."}, graph2_thread_id)
                    return {"error": "Failed to get state after interrupt."}

                # Determine the interrupted node. `next` usually holds the ID(s) of interrupted node(s).
                interrupted_node_id: Optional[str] = None
                if current_snapshot.next: # 'next' can be a string or a list of strings
                    interrupted_node_id = current_snapshot.next[0] if isinstance(current_snapshot.next, list) and current_snapshot.next else \
                                          current_snapshot.next if isinstance(current_snapshot.next, str) else None
                
                if not interrupted_node_id:
                    logger.error(f"ThreadID '{graph2_thread_id}': Could not determine interrupted node from state.next: {current_snapshot.next}")
                    await self.websocket_callback("execution_error", {"error": "Could not determine interrupted node."}, graph2_thread_id)
                    return {"error": "Could not determine interrupted node."}

                logger.info(f"ThreadID '{graph2_thread_id}': Graph interrupted before node '{interrupted_node_id}'.")

                # Get the node definition from the plan to prepare UI details
                node_def_for_ui: Optional[PlanNode] = self.graph_definition.get_node_definition(interrupted_node_id)
                if not node_def_for_ui:
                    logger.error(f"ThreadID '{graph2_thread_id}': Could not find definition for interrupted node '{interrupted_node_id}'.")
                    await self.websocket_callback("execution_error", {"error": f"Definition for node {interrupted_node_id} not found."}, graph2_thread_id)
                    return {"error": f"Definition for node {interrupted_node_id} not found."}

                # Prepare data for the UI confirmation modal
                # The node hasn't run yet, so we use its definition and current state to make the payload
                temp_state_for_payload_prep = ExecutionGraphState.model_validate(current_snapshot.values)
                _, _, prepared_payload, _ = self.graph_definition._prepare_api_request_components(node_def_for_ui, temp_state_for_payload_prep)

                details_for_ui = {
                    "type": "api_call_confirmation", # Consistent with frontend
                    "operationId": node_def_for_ui.operationId,
                    "method": node_def_for_ui.method,
                    "path": node_def_for_ui.path, # Path template
                    "payload_to_confirm": prepared_payload, # Payload based on current state
                    "prompt": node_def_for_ui.confirmation_prompt or f"Confirm API call: {node_def_for_ui.method} {node_def_for_ui.path}?",
                    "confirmation_key": f"confirmed_{node_def_for_ui.effective_id}"
                }
                
                await self.websocket_callback("human_intervention_required", {"node_name": interrupted_node_id, "details_for_ui": details_for_ui}, graph2_thread_id)
                
                try:
                    if self.main_planning_session_id not in self.resume_queues:
                        self.resume_queues[self.main_planning_session_id] = asyncio.Queue()
                    resume_data = await asyncio.wait_for(self.resume_queues[self.main_planning_session_id].get(), timeout=600.0)
                    
                    confirmation_key = resume_data.get("confirmation_key")
                    decision = resume_data.get("decision", False) # Default to False if not specified

                    if not confirmation_key:
                        raise ValueError("Resume data missing 'confirmation_key'.")

                    update_for_resume_state = {
                        "confirmed_data": {
                            confirmation_key: decision,
                            f"{confirmation_key}_details": resume_data 
                        }
                        # No need to manage pending_confirmation_data in state with this interrupt pattern
                    }
                    if not decision: # If user cancelled
                        logger.info(f"ThreadID '{graph2_thread_id}': User cancelled operation for node '{interrupted_node_id}'. Stopping workflow.")
                        # To truly stop, we might need to route to an error/end state or raise specific exception
                        # For now, update state and let it error out or end if no path forward.
                        self.runnable_graph.update_state(config, {"error": f"User cancelled operation: {interrupted_node_id}", **update_for_resume_state})
                        graph_has_ended = True # Treat as ended
                        continue # Go to final state retrieval

                    self.runnable_graph.update_state(config, update_for_resume_state)
                    current_input_for_astream = None # Resume from checkpoint
                    logger.info(f"ThreadID '{graph2_thread_id}': State updated for resume. Continuing.")

                except asyncio.TimeoutError:
                    error_msg = f"ThreadID '{graph2_thread_id}': Timeout waiting for resume for '{interrupted_node_id}'."
                    logger.error(error_msg)
                    await self.websocket_callback("workflow_timeout", {"message": error_msg, "node_name": interrupted_node_id}, graph2_thread_id)
                    self.runnable_graph.update_state(config, {"error": "ConfirmationTimeout"})
                    return self.runnable_graph.get_state(config).values
                
                except Exception as e_resume:
                    error_msg = f"ThreadID '{graph2_thread_id}': Error processing resume: {e_resume}"
                    logger.error(error_msg, exc_info=True)
                    await self.websocket_callback("execution_error", {"error": error_msg}, graph2_thread_id)
                    self.runnable_graph.update_state(config, {"error": error_msg})
                    return self.runnable_graph.get_state(config).values

            except Exception as e_loop:
                error_msg = f"Critical error in workflow for ThreadID '{graph2_thread_id}': {e_loop}"
                logger.critical(error_msg, exc_info=True)
                await self.websocket_callback("execution_error", {"error": error_msg}, graph2_thread_id)
                try: self.runnable_graph.update_state(config, {"error": error_msg})
                except: pass # Best effort
                return self.runnable_graph.get_state(config).values if self.runnable_graph else {"error": error_msg}

        # After while loop (graph_has_ended is True)
        try:
            logger.info(f"ThreadID '{graph2_thread_id}': Workflow ended. Getting final state.")
            final_snapshot = self.runnable_graph.get_state(config)
            final_state_dict = final_snapshot.values if final_snapshot else {}
            if final_state_dict.get("error"):
                await self.websocket_callback("execution_failed", {"final_state": final_state_dict}, graph2_thread_id)
            else:
                await self.websocket_callback("execution_completed", {"final_state": final_state_dict}, graph2_thread_id)
            return final_state_dict
        except Exception as e_final:
            error_msg = f"Error getting final state for ThreadID '{graph2_thread_id}': {e_final}"
            logger.error(error_msg, exc_info=True)
            await self.websocket_callback("execution_error", {"error": error_msg}, graph2_thread_id)
            return {"error": error_msg}
