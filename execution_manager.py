import asyncio
import logging
from typing import Any, Callable, Awaitable, Dict, Optional
from collections import defaultdict
import json # For serializing complex objects in events

from langgraph.checkpoint.base import BaseCheckpointSaver # Keep for type hint, though not used if checkpointer is None
from langgraph.graph.state import StateSnapshot

from models import BotState, ExecutionGraphState, Node as PlanNode
from execution_graph_definition import ExecutionGraphDefinition


logger = logging.getLogger(__name__)

class GraphExecutionManager:
    def __init__(
        self,
        runnable_graph: Any,
        graph_definition: ExecutionGraphDefinition,
        websocket_callback: Callable[[str, Dict[str, Any], Optional[str]], Awaitable[None]],
        planning_checkpointer: Optional[BaseCheckpointSaver], # For LangGraph config if used
        main_planning_session_id: str
    ):
        self.runnable_graph: Any = runnable_graph
        self.graph_definition = graph_definition
        self.websocket_callback = websocket_callback
        self.planning_checkpointer = planning_checkpointer # Checkpointer for the Graph 2 instance
        self.main_planning_session_id = main_planning_session_id # Main G1 session ID, for context
        self.resume_queues: Dict[str, asyncio.Queue[Any]] = defaultdict(asyncio.Queue)
        logger.info(f"GraphExecutionManager initialized for Main Planning Session ID: {self.main_planning_session_id}")

    async def submit_resume_data(self, graph2_thread_id: str, resume_data: Any) -> bool:
        """
        Submits resume data to the queue for the specified Graph 2 thread ID.
        """
        if graph2_thread_id not in self.resume_queues:
             # This case should ideally not happen if queue is created when interrupt occurs
             self.resume_queues[graph2_thread_id] = asyncio.Queue()
             logger.warning(f"Resume queue for G2 Thread ID: {graph2_thread_id} was not pre-initialized. Created now.")

        try:
            await self.resume_queues[graph2_thread_id].put(resume_data)
            logger.info(f"Resume data submitted for G2 Thread ID: {graph2_thread_id}. Data: {str(resume_data)[:100]}...")
            return True
        except Exception as e:
            logger.error(f"Error putting resume data onto queue for G2 Thread ID {graph2_thread_id}: {e}")
            return False


    async def execute_workflow(
        self,
        initial_graph_values: Dict[str, Any], # This is the initial ExecutionGraphState dict
        config: Dict[str, Any], # This config MUST contain {"configurable": {"thread_id": "some_g2_thread_id"}}
    ) -> Optional[Dict[str, Any]]:
        graph2_thread_id = config.get("configurable", {}).get("thread_id")
        if not graph2_thread_id:
            # Use main_planning_session_id for the callback if g2_thread_id is somehow missing
            await self.websocket_callback("execution_error", {"error": "Config error: Graph 2 thread_id missing for execution."}, self.main_planning_session_id)
            raise ValueError("thread_id missing for ExecutionGraph execution.")

        logger.info(f"--- [Graph 2 Manager] Starting Workflow for G2 ThreadID: {graph2_thread_id} ---")
        current_input_for_stream: Optional[Dict[str, Any]] = initial_graph_values
        graph_has_ended_flag = False # More explicit flag

        while not graph_has_ended_flag:
            try:
                logger.debug(f"G2 ThreadID '{graph2_thread_id}': Calling astream_events. Input: {str(current_input_for_stream)[:200]}...")
                
                async for event in self.runnable_graph.astream_events(current_input_for_stream, config=config, version="v2"):
                    event_name = event["event"]
                    event_data = event.get("data", {})
                    event_tags = event.get("tags", [])
                    node_name = event.get("name", "") # Populated for tool/llm events

                    logger.debug(f"G2 ThreadID '{graph2_thread_id}': Event: {event_name}, Node: {node_name}, DataKeys: {list(event_data.keys())}")

                    # Forward relevant events to the UI
                    if event_name == "on_tool_start": # 'tool' in LangGraph corresponds to our API call nodes
                        tool_input = event_data.get("input")
                        # The input to a node in ExecutionGraphState is the full state.
                        # We need to derive what the actual API call parameters would be for a good preview.
                        # This is complex here. The node_def itself is better for this.
                        node_def_preview = self.graph_definition.get_node_definition(node_name)
                        preview_payload = "Input not readily available from event"
                        if node_def_preview and node_def_preview.payload: # Get template
                            preview_payload = node_def_preview.payload
                        elif isinstance(tool_input, dict) and tool_input.get('messages'): # If it's an LLM call input
                            preview_payload = tool_input['messages']


                        await self.websocket_callback("tool_start", { # Renamed from api_call_start for clarity
                            "node_name": node_name,
                            "input_preview": str(preview_payload)[:200] + "..." # Simplified preview
                        }, graph2_thread_id)

                    elif event_name == "on_tool_end":
                        output_data = event_data.get("output", {}) # Output is usually a dict to update state
                        api_results_for_node = output_data.get("api_results", {}).get(node_name, {})
                        status = api_results_for_node.get("status_code", "N/A")
                        error_msg = api_results_for_node.get("error")
                        exec_time = api_results_for_node.get("execution_time", "N/A")
                        response_preview = str(api_results_for_node.get("response_body", ""))[:200] + "..."

                        await self.websocket_callback("tool_end", { # Renamed from api_call_end
                            "node_name": node_name,
                            "status_code": status,
                            "error": error_msg,
                            "execution_time": exec_time,
                            "response_preview": response_preview,
                            "raw_output": str(output_data)[:200]+"..." # For debugging UI
                        }, graph2_thread_id)
                    
                    elif event_name == "on_graph_end": # LangGraph v0.1.x specific
                        logger.info(f"G2 ThreadID '{graph2_thread_id}': on_graph_end event received. Workflow finished.")
                        graph_has_ended_flag = True
                        break # Exit the event processing loop for this stream

                    # Check for __end__ node explicitly (LangGraph v0.0.x style, less common now with on_graph_end)
                    if isinstance(event_data, dict) and "__end__" in event_data: # Should be caught by on_graph_end typically
                         logger.info(f"G2 ThreadID '{graph2_thread_id}': __end__ marker found in event data keys. Workflow finished.")
                         graph_has_ended_flag = True
                         break


                if graph_has_ended_flag:
                    logger.info(f"G2 ThreadID '{graph2_thread_id}': Graph processing loop ended because graph_has_ended_flag is true.")
                    break # Exit the outer while loop

                # If astream_events finished an iteration without graph_has_ended_flag being set,
                # it implies an interruption (or an issue).
                logger.info(f"G2 ThreadID '{graph2_thread_id}': astream_events iteration completed. Checking for interrupt.")
                
                current_snapshot: Optional[StateSnapshot] = self.runnable_graph.get_state(config)
                if not current_snapshot:
                    logger.error(f"G2 ThreadID '{graph2_thread_id}': Failed to get state after astream_events iteration.")
                    await self.websocket_callback("execution_error", {"error": "Failed to get state after stream iteration."}, graph2_thread_id)
                    return {"error": "Failed to get state post-stream."}

                if not current_snapshot.next: # No next node means the graph truly ended or is stuck
                    if not any(call.return_value is None for call in current_snapshot.inflight): # Check if any node is still running
                        logger.info(f"G2 ThreadID '{graph2_thread_id}': No next node and no inflight operations. Graph has ended.")
                        graph_has_ended_flag = True
                        break # Exit the while loop

                interrupted_node_id: Optional[str] = None
                if current_snapshot.next:
                    interrupted_node_id = current_snapshot.next[0] if isinstance(current_snapshot.next, list) and current_snapshot.next else \
                                          current_snapshot.next if isinstance(current_snapshot.next, str) else None
                
                if not interrupted_node_id:
                    # This could happen if the graph legitimately finished and `on_graph_end` wasn't the last event processed,
                    # or if there's an issue. If snapshot.next is empty, it usually means the graph is at an end state.
                    logger.info(f"G2 ThreadID '{graph2_thread_id}': No specific interrupted node ID found in snapshot.next. Assuming graph ended or requires no further input.")
                    graph_has_ended_flag = True # Assume end if no clear next step for interruption
                    break


                logger.info(f"G2 ThreadID '{graph2_thread_id}': Graph interrupted before node '{interrupted_node_id}'.")

                node_def_for_ui: Optional[PlanNode] = self.graph_definition.get_node_definition(interrupted_node_id)
                if not node_def_for_ui:
                    err_msg = f"Definition for interrupted node {interrupted_node_id} not found."
                    logger.error(f"G2 ThreadID '{graph2_thread_id}': {err_msg}")
                    await self.websocket_callback("execution_error", {"error": err_msg}, graph2_thread_id)
                    return {"error": err_msg}

                # Prepare data for UI confirmation
                temp_state_for_payload_prep = ExecutionGraphState.model_validate(current_snapshot.values)
                _, _, prepared_payload, _ = self.graph_definition._prepare_api_request_components(node_def_for_ui, temp_state_for_payload_prep)

                details_for_ui = {
                    "type": "api_call_confirmation",
                    "operationId": node_def_for_ui.operationId,
                    "method": node_def_for_ui.method,
                    "path": node_def_for_ui.path,
                    "payload_to_confirm": prepared_payload,
                    "prompt": node_def_for_ui.confirmation_prompt or f"Confirm API call: {node_def_for_ui.method} {node_def_for_ui.path}?",
                    "confirmation_key": f"confirmed_{node_def_for_ui.effective_id}"
                }
                
                await self.websocket_callback("human_intervention_required", {"node_name": interrupted_node_id, "details_for_ui": details_for_ui}, graph2_thread_id)
                
                try:
                    if graph2_thread_id not in self.resume_queues: # Ensure queue exists for this G2 thread
                        self.resume_queues[graph2_thread_id] = asyncio.Queue()

                    resume_data = await asyncio.wait_for(self.resume_queues[graph2_thread_id].get(), timeout=600.0) # 10 min timeout
                    
                    confirmation_key = resume_data.get("confirmation_key")
                    decision = resume_data.get("decision", False)

                    if not confirmation_key:
                        raise ValueError("Resume data missing 'confirmation_key'.")

                    update_for_resume_state = {
                        "confirmed_data": {
                            confirmation_key: decision,
                            f"{confirmation_key}_details": resume_data 
                        }
                    }
                    if not decision:
                        logger.info(f"G2 ThreadID '{graph2_thread_id}': User cancelled operation for node '{interrupted_node_id}'. Updating state with error and ending.")
                        # Update state to reflect cancellation, then end.
                        error_update = {"error": f"User cancelled operation: {interrupted_node_id}"}
                        self.runnable_graph.update_state(config, {**update_for_resume_state, **error_update})
                        graph_has_ended_flag = True # Treat as ended due to cancellation
                        continue # Go to final state retrieval at the end of the while loop

                    self.runnable_graph.update_state(config, update_for_resume_state)
                    current_input_for_stream = None # Resume from checkpoint by passing None
                    logger.info(f"G2 ThreadID '{graph2_thread_id}': State updated for resume. Continuing workflow.")

                except asyncio.TimeoutError:
                    error_msg = f"Timeout waiting for resume for '{interrupted_node_id}' on G2 ThreadID '{graph2_thread_id}'."
                    logger.error(error_msg)
                    await self.websocket_callback("workflow_timeout", {"message": error_msg, "node_name": interrupted_node_id}, graph2_thread_id)
                    self.runnable_graph.update_state(config, {"error": "ConfirmationTimeout"})
                    graph_has_ended_flag = True # End due to timeout
                
                except Exception as e_resume:
                    error_msg = f"Error processing resume for G2 ThreadID '{graph2_thread_id}': {e_resume}"
                    logger.error(error_msg, exc_info=True)
                    await self.websocket_callback("execution_error", {"error": error_msg}, graph2_thread_id)
                    self.runnable_graph.update_state(config, {"error": error_msg})
                    graph_has_ended_flag = True # End due to resume processing error

            except Exception as e_loop: # Catch errors in the main while loop
                error_msg = f"Critical error in workflow execution for G2 ThreadID '{graph2_thread_id}': {e_loop}"
                logger.critical(error_msg, exc_info=True)
                await self.websocket_callback("execution_error", {"error": error_msg}, graph2_thread_id)
                try:
                    self.runnable_graph.update_state(config, {"error": error_msg})
                except Exception as e_state_update:
                    logger.error(f"Failed to update state with error: {e_state_update}")
                graph_has_ended_flag = True # End due to critical error

        # After while loop (graph_has_ended_flag is True)
        try:
            logger.info(f"G2 ThreadID '{graph2_thread_id}': Workflow ended. Getting final state.")
            final_snapshot = self.runnable_graph.get_state(config)
            final_state_dict = final_snapshot.values if final_snapshot else {}
            
            if final_state_dict.get("error"):
                await self.websocket_callback("execution_failed", {"final_state": final_state_dict}, graph2_thread_id)
            else:
                await self.websocket_callback("execution_completed", {"final_state": final_state_dict}, graph2_thread_id)
            return final_state_dict
        except Exception as e_final:
            error_msg = f"Error getting final state for G2 ThreadID '{graph2_thread_id}': {e_final}"
            logger.error(error_msg, exc_info=True)
            await self.websocket_callback("execution_error", {"error": error_msg}, graph2_thread_id)
            return {"error": error_msg}
        finally:
            # Clean up the resume queue for this specific G2 thread ID
            self.resume_queues.pop(graph2_thread_id, None)
            logger.info(f"Cleaned up resume queue for G2 ThreadID: {graph2_thread_id}")
