import asyncio
import logging
from typing import Any, Callable, Awaitable, Dict, Optional
from collections import defaultdict

from langgraph.checkpoint.base import BaseCheckpointSaver

from models import BotState, ExecutionGraphState

logger = logging.getLogger(__name__)

class GraphExecutionManager:
    """
    Manages the runtime execution of a compiled LangGraph (Graph 2 - Execution Graph).
    Handles streaming of graph execution, and manages resumption for human-in-the-loop steps
    by checking for 'pending_confirmation_data' in the graph state.
    """
    def __init__(
        self,
        runnable_graph: Any,
        websocket_callback: Callable[[str, Dict[str, Any], Optional[str]], Awaitable[None]],
        planning_checkpointer: Optional[BaseCheckpointSaver],
        main_planning_session_id: str
    ):
        self.runnable_graph: Any = runnable_graph
        self.websocket_callback = websocket_callback
        self.planning_checkpointer = planning_checkpointer
        self.main_planning_session_id = main_planning_session_id
        self.resume_queues: Dict[str, asyncio.Queue[Any]] = defaultdict(asyncio.Queue)
        logger.info(f"GraphExecutionManager initialized for Main Planning Session ID: {self.main_planning_session_id}")

    async def submit_resume_data(self, main_session_id: str, resume_data: Any) -> bool:
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
            await self.websocket_callback(
                "execution_error",
                {"error": "ConfigurationError: thread_id is missing for ExecutionGraph."},
                None
            )
            raise ValueError("thread_id is missing in config['configurable']['thread_id'] for ExecutionGraph (Graph 2).")

        final_state_dict: Optional[Dict[str, Any]] = None
        logger.info(f"--- [Graph 2] Starting Workflow Execution for ThreadID: {graph2_thread_id} (Main Session: {self.main_planning_session_id}) ---")

        current_input_for_astream: Optional[Dict[str, Any]] = initial_graph_values
        graph_has_ended_via_END_node = False

        while not graph_has_ended_via_END_node:
            try:
                logger.debug(f"Graph 2 ThreadID '{graph2_thread_id}': Starting/continuing astream. Input: {str(current_input_for_astream)[:200]}...")
                
                async for event in self.runnable_graph.astream(current_input_for_astream, config=config):
                    for node_name, node_output_state_update in event.items():
                        if node_name == "__end__":
                            logger.info(f"Graph 2 ThreadID '{graph2_thread_id}' | Graph execution reached END state.")
                            graph_has_ended_via_END_node = True
                            break
                        logger.info(f"Graph 2 ThreadID '{graph2_thread_id}' | Node '{node_name}' executed. Update keys: {list(node_output_state_update.keys()) if isinstance(node_output_state_update, dict) else type(node_output_state_update)}")
                    if graph_has_ended_via_END_node:
                        break

                if graph_has_ended_via_END_node:
                    logger.info(f"Graph 2 ThreadID '{graph2_thread_id}': Graph explicitly ended via __end__ node.")
                    break # Exit the while loop, proceed to get final state

                # If graph hasn't ended, check the state for pending confirmation
                logger.debug(f"Graph 2 ThreadID '{graph2_thread_id}': astream iteration finished. Checking state for pending confirmation.")
                current_snapshot = self.runnable_graph.get_state(config)
                current_state_values = current_snapshot.values if current_snapshot and hasattr(current_snapshot, 'values') else {}
                
                pending_conf_data_from_state = current_state_values.get("pending_confirmation_data")

                if pending_conf_data_from_state:
                    logger.info(f"Graph 2 ThreadID '{graph2_thread_id}': Detected pending confirmation. Data: {str(pending_conf_data_from_state)[:200]}")
                    interrupted_node_name = pending_conf_data_from_state.get("operationId", "UnknownNode")
                    
                    await self.websocket_callback(
                        "human_intervention_required",
                        {"node_name": interrupted_node_name, "details_for_ui": pending_conf_data_from_state},
                        graph2_thread_id
                    )
                    
                    logger.info(f"Graph 2 ThreadID '{graph2_thread_id}': Waiting for resume data from queue...")
                    try:
                        if self.main_planning_session_id not in self.resume_queues:
                            self.resume_queues[self.main_planning_session_id] = asyncio.Queue()

                        resume_data = await asyncio.wait_for(self.resume_queues[self.main_planning_session_id].get(), timeout=600.0)
                        logger.info(f"Graph 2 ThreadID '{graph2_thread_id}': Resuming with data: {str(resume_data)[:100]}")

                        confirmation_key_from_ui = resume_data.get("confirmation_key")
                        if not confirmation_key_from_ui:
                            logger.error(f"Graph 2 ThreadID '{graph2_thread_id}': Resume data missing 'confirmation_key'. Cannot process.")
                            raise ValueError("Resume data is missing the 'confirmation_key'.")

                        update_for_resume_state = {
                            "confirmed_data": {
                                confirmation_key_from_ui: resume_data.get("decision", True),
                                f"{confirmation_key_from_ui}_details": resume_data
                            },
                            "pending_confirmation_data": None
                        }
                        
                        await self.runnable_graph.update_state(config, update_for_resume_state)
                        current_input_for_astream = None # LangGraph will use checkpointed state
                        logger.info(f"Graph 2 ThreadID '{graph2_thread_id}': State updated for resume. Continuing workflow.")
                        # The while loop will continue, and astream will be called again.
                    
                    except asyncio.TimeoutError:
                        error_msg = f"Graph 2 ThreadID '{graph2_thread_id}': Timeout waiting for resume data for node '{interrupted_node_name}'."
                        logger.error(error_msg)
                        await self.websocket_callback("workflow_timeout", {"message": error_msg, "node_name": interrupted_node_name}, graph2_thread_id)
                        try: 
                            self.runnable_graph.update_state(config, {"error": "ConfirmationTimeout", "pending_confirmation_data": None})
                        except Exception as e_up: logger.error(f"Failed to update state on timeout: {e_up}")
                        error_snapshot_on_timeout = self.runnable_graph.get_state(config)
                        return error_snapshot_on_timeout.values if error_snapshot_on_timeout else {"error": error_msg}
                    
                    except Exception as e_resume_proc:
                        error_msg = f"Graph 2 ThreadID '{graph2_thread_id}': Error processing resume data: {type(e_resume_proc).__name__} - {e_resume_proc}"
                        logger.error(error_msg, exc_info=True)
                        await self.websocket_callback("execution_error", {"error": error_msg}, graph2_thread_id)
                        try: 
                            self.runnable_graph.update_state(config, {"error": error_msg, "pending_confirmation_data": None})
                        except Exception as e_up: logger.error(f"Failed to update state on resume error: {e_up}")
                        error_snapshot_on_resume_error = self.runnable_graph.get_state(config)
                        return error_snapshot_on_resume_error.values if error_snapshot_on_resume_error else {"error": error_msg}
                
                # REMOVED else block that prematurely set graph_has_ended_via_END_node = True
                # If no pending confirmation and not __end__, the loop continues naturally.
                # current_input_for_astream will be None (if a node just ran) or initial_graph_values (first run).
                # LangGraph's astream(None, ...) will correctly pick up the next node.
                else:
                    logger.debug(f"Graph 2 ThreadID '{graph2_thread_id}': No pending confirmation. Loop continues for next node if graph not ended.")
                    current_input_for_astream = None # Ensure next astream call uses checkpoint

            except Exception as e_stream_loop:
                error_message = f"Critical error in ExecutionGraph (Graph 2) stream loop for ThreadID '{graph2_thread_id}': {type(e_stream_loop).__name__} - {str(e_stream_loop)}"
                logger.critical(error_message, exc_info=True)
                await self.websocket_callback("execution_error", {"error": error_message}, graph2_thread_id)
                try: 
                    self.runnable_graph.update_state(config, {"error": error_message, "pending_confirmation_data": None})
                except Exception as e_up: logger.error(f"Failed to update state on critical error: {e_up}")
                critical_error_snapshot = self.runnable_graph.get_state(config)
                return critical_error_snapshot.values if critical_error_snapshot else {"error": error_message}
            
        # After the while loop (graph_has_ended_via_END_node is True)
        try:
            logger.info(f"Graph 2 ThreadID '{graph2_thread_id}': Workflow processing loop ended. Getting final state.")
            final_state_snapshot = self.runnable_graph.get_state(config)
            if final_state_snapshot and hasattr(final_state_snapshot, 'values'):
                final_state_dict = final_state_snapshot.values
                if final_state_dict.get("error"):
                     await self.websocket_callback("execution_failed", {"final_state": final_state_dict}, graph2_thread_id)
                     logger.error(f"--- [Graph 2] Workflow FAILED for ThreadID: {graph2_thread_id}. Error: {final_state_dict['error']} ---")
                else:
                    await self.websocket_callback("execution_completed", {"final_state": final_state_dict}, graph2_thread_id)
                    logger.info(f"--- [Graph 2] Workflow COMPLETED SUCCESSFULLY for ThreadID: {graph2_thread_id} ---")
            else:
                await self.websocket_callback("execution_warning", {"message": "Execution finished, but no final state snapshot retrieved."}, graph2_thread_id)
                logger.warning(f"--- [Graph 2] Workflow FINISHED (No Final State Snapshot) for ThreadID: {graph2_thread_id} ---")
        
        except Exception as e_get_final_state:
            error_message = f"Error retrieving final state for Graph 2 ThreadID '{graph2_thread_id}': {type(e_get_final_state).__name__} - {str(e_get_final_state)}"
            logger.error(error_message, exc_info=True)
            await self.websocket_callback("execution_error", {"error": error_message, "detail": "Failed to get final state."}, graph2_thread_id)
            final_state_dict = {"error": error_message}
            
        return final_state_dict
