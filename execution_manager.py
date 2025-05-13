import asyncio
import logging
from typing import Any, Callable, Awaitable, Dict, Optional
from collections import defaultdict
import json 

from langgraph.graph import START, END # ADDED THIS IMPORT
from langgraph.checkpoint.base import BaseCheckpointSaver 
from langgraph.types import StateSnapshot 

from models import BotState, ExecutionGraphState, Node as PlanNode
from execution_graph_definition import ExecutionGraphDefinition


logger = logging.getLogger(__name__)

class GraphExecutionManager:
    def __init__(
        self,
        runnable_graph: Any,
        graph_definition: ExecutionGraphDefinition,
        websocket_callback: Callable[[str, Dict[str, Any], Optional[str]], Awaitable[None]],
        planning_checkpointer: Optional[BaseCheckpointSaver], 
        main_planning_session_id: str
    ):
        self.runnable_graph: Any = runnable_graph
        self.graph_definition = graph_definition
        self.websocket_callback = websocket_callback
        self.planning_checkpointer = planning_checkpointer 
        self.main_planning_session_id = main_planning_session_id 
        self.resume_queues: Dict[str, asyncio.Queue[Any]] = defaultdict(asyncio.Queue)
        logger.info(f"GraphExecutionManager initialized for Main Planning Session ID: {self.main_planning_session_id}")

    async def submit_resume_data(self, graph2_thread_id: str, resume_data: Any) -> bool:
        if graph2_thread_id not in self.resume_queues:
             self.resume_queues[graph2_thread_id] = asyncio.Queue()
             logger.warning(f"Resume queue for G2 Thread ID: {graph2_thread_id} was not pre-initialized. Created now.")

        try:
            await self.resume_queues[graph2_thread_id].put(resume_data)
            logger.info(f"Resume data submitted for G2 Thread ID: {graph2_thread_id}. Queue size: {self.resume_queues[graph2_thread_id].qsize()}. Data: {str(resume_data)[:100]}...")
            return True
        except Exception as e:
            logger.error(f"Error putting resume data onto queue for G2 Thread ID {graph2_thread_id}: {e}")
            return False


    async def execute_workflow(
        self,
        initial_graph_values: Dict[str, Any], 
        config: Dict[str, Any], 
    ) -> Optional[Dict[str, Any]]:
        graph2_thread_id = config.get("configurable", {}).get("thread_id")
        if not graph2_thread_id:
            await self.websocket_callback("execution_error", {"error": "Config error: Graph 2 thread_id missing for execution."}, self.main_planning_session_id)
            return {"error": "thread_id missing for ExecutionGraph execution."}

        logger.info(f"--- [Graph 2 Manager] Starting Workflow for G2 ThreadID: {graph2_thread_id} ---")
        current_input_for_stream: Optional[Dict[str, Any]] = initial_graph_values
        graph_has_ended_flag = False 

        while not graph_has_ended_flag:
            try:
                logger.debug(f"G2 ThreadID '{graph2_thread_id}': Calling astream_events. Input: {str(current_input_for_stream)[:200]}...")
                
                async for event in self.runnable_graph.astream_events(current_input_for_stream, config=config, version="v2"):
                    event_name = event["event"] 
                    event_data = event.get("data", {})
                    node_name_from_event = event.get("name", "") 

                    logger.debug(f"G2 ThreadID '{graph2_thread_id}': Event: {event_name}, Node: {node_name_from_event}, DataKeys: {list(event_data.keys())}")

                    if event_name == "on_chain_start": 
                        node_def_preview = self.graph_definition.get_node_definition(node_name_from_event)
                        # Check if it's an actual API node and not LangGraph's internal START/END constants
                        # or conventionally named internal nodes like "__start__", "__end__".
                        if node_def_preview and node_name_from_event not in [START, END, "__start__", "__end__"]: 
                            preview_payload_str = "Input details not readily available from this event."
                            if node_def_preview.payload: 
                                preview_payload_str = str(node_def_preview.payload)
                            
                            await self.websocket_callback("tool_start", { 
                                "node_name": node_name_from_event,
                                "input_preview": preview_payload_str[:200] + "..."
                            }, graph2_thread_id)

                    elif event_name == "on_chain_end":
                        node_def_preview = self.graph_definition.get_node_definition(node_name_from_event)
                        if node_def_preview and node_name_from_event not in [START, END, "__start__", "__end__"]: 
                            output_data = event_data.get("output", {}) 
                            api_results_for_node = {}
                            
                            if isinstance(output_data, dict) and "api_results" in output_data and isinstance(output_data["api_results"], dict):
                                api_results_for_node = output_data["api_results"].get(node_name_from_event, {})
                            
                            status = api_results_for_node.get("status_code", "N/A")
                            error_msg = api_results_for_node.get("error")
                            exec_time = api_results_for_node.get("execution_time", "N/A")
                            response_preview = str(api_results_for_node.get("response_body", ""))[:200] + "..."

                            await self.websocket_callback("tool_end", { 
                                "node_name": node_name_from_event,
                                "status_code": status,
                                "error": error_msg,
                                "execution_time": exec_time,
                                "response_preview": response_preview,
                                "raw_output_from_node": str(output_data)[:200]+"..." 
                            }, graph2_thread_id)
                    
                    elif event_name == "on_graph_end": 
                        logger.info(f"G2 ThreadID '{graph2_thread_id}': on_graph_end event received. Workflow finished.")
                        graph_has_ended_flag = True
                        break 

                if graph_has_ended_flag:
                    logger.info(f"G2 ThreadID '{graph2_thread_id}': Graph processing loop ended due to on_graph_end event.")
                    break 

                logger.info(f"G2 ThreadID '{graph2_thread_id}': astream_events iteration completed. Checking for interrupt or true end.")
                
                current_snapshot: Optional[StateSnapshot] = await self.runnable_graph.aget_state(config) 
                if not current_snapshot:
                    logger.error(f"G2 ThreadID '{graph2_thread_id}': Failed to get state after astream_events iteration.")
                    await self.websocket_callback("execution_error", {"error": "Failed to get state after stream iteration."}, graph2_thread_id)
                    return {"error": "Failed to get state post-stream."}

                if not current_snapshot.next:
                    if not any(call.return_value is None for call in getattr(current_snapshot, 'inflight', [])):
                        logger.info(f"G2 ThreadID '{graph2_thread_id}': Snapshot indicates no next node and no inflight operations. Graph has ended.")
                        graph_has_ended_flag = True
                        break 

                interrupted_node_id_str: Optional[str] = None
                if current_snapshot.next:
                    if isinstance(current_snapshot.next, (list, tuple)) and len(current_snapshot.next) > 0:
                        interrupted_node_id_str = current_snapshot.next[0]
                        logger.info(f"G2 ThreadID '{graph2_thread_id}': Potential interrupt before node(s): {current_snapshot.next}. Using first: '{interrupted_node_id_str}'.")
                    elif isinstance(current_snapshot.next, str): 
                        interrupted_node_id_str = current_snapshot.next
                        logger.info(f"G2 ThreadID '{graph2_thread_id}': Potential interrupt before node: '{interrupted_node_id_str}' (snapshot.next was str).")
                
                if not interrupted_node_id_str:
                    logger.info(f"G2 ThreadID '{graph2_thread_id}': No specific interrupted node ID found in snapshot.next. Assuming graph ended or requires no further input.")
                    graph_has_ended_flag = True 
                    break

                logger.info(f"G2 ThreadID '{graph2_thread_id}': Graph interrupted before node '{interrupted_node_id_str}'.")

                node_def_for_ui: Optional[PlanNode] = self.graph_definition.get_node_definition(interrupted_node_id_str) 
                if not node_def_for_ui:
                    err_msg = f"Definition for interrupted node '{interrupted_node_id_str}' not found. Cannot proceed with confirmation."
                    logger.error(f"G2 ThreadID '{graph2_thread_id}': {err_msg}")
                    await self.websocket_callback("execution_error", {"error": err_msg, "interrupted_node_id": interrupted_node_id_str}, graph2_thread_id)
                    await self.runnable_graph.aupdate_state(config, {"error": err_msg})
                    graph_has_ended_flag = True
                    continue 
                try:
                    temp_state_for_payload_prep = ExecutionGraphState.model_validate(current_snapshot.values)
                except Exception as e_val:
                    val_err_msg = f"Error validating current snapshot values for payload prep: {e_val}"
                    logger.error(f"G2 ThreadID '{graph2_thread_id}': {val_err_msg}. Snapshot values: {current_snapshot.values}")
                    await self.websocket_callback("execution_error", {"error": val_err_msg, "interrupted_node_id": interrupted_node_id_str}, graph2_thread_id)
                    await self.runnable_graph.aupdate_state(config, {"error": val_err_msg})
                    graph_has_ended_flag = True
                    continue
                
                _, _, prepared_payload, _ = self.graph_definition._prepare_api_request_components(node_def_for_ui, temp_state_for_payload_prep)

                details_for_ui = {
                    "type": "api_call_confirmation", 
                    "operationId": node_def_for_ui.operationId,
                    "effective_node_id": node_def_for_ui.effective_id, 
                    "method": node_def_for_ui.method,
                    "path": node_def_for_ui.path,
                    "payload_to_confirm": prepared_payload, 
                    "prompt": node_def_for_ui.confirmation_prompt or f"Confirm API call: {node_def_for_ui.method} {node_def_for_ui.path}?",
                    "confirmation_key": f"confirmed_{node_def_for_ui.effective_id}" 
                }
                
                await self.websocket_callback("human_intervention_required", {"node_name": interrupted_node_id_str, "details_for_ui": details_for_ui}, graph2_thread_id)
                
                try:
                    if graph2_thread_id not in self.resume_queues: 
                        self.resume_queues[graph2_thread_id] = asyncio.Queue()
                        logger.info(f"G2 ThreadID '{graph2_thread_id}': Resume queue re-initialized for interrupt.")

                    logger.info(f"G2 ThreadID '{graph2_thread_id}': Awaiting resume data on queue {id(self.resume_queues[graph2_thread_id])} for node '{interrupted_node_id_str}'.")
                    resume_data = await asyncio.wait_for(self.resume_queues[graph2_thread_id].get(), timeout=600.0) 
                    logger.info(f"G2 ThreadID '{graph2_thread_id}': Resume data received: {str(resume_data)[:100]}")
                    
                    confirmation_key_from_resume = resume_data.get("confirmation_key")
                    decision = resume_data.get("decision", False) 

                    if not confirmation_key_from_resume:
                        raise ValueError("Resume data missing 'confirmation_key'.")
                    update_for_resume_state = {
                        "confirmed_data": {
                            confirmation_key_from_resume: decision, 
                            f"{confirmation_key_from_resume}_details": resume_data 
                        }
                    }
                    
                    if not decision: 
                        logger.info(f"G2 ThreadID '{graph2_thread_id}': User cancelled/denied operation for node '{interrupted_node_id_str}'. Updating state with error and ending workflow.")
                        error_update = {"error": f"User cancelled/denied operation: {interrupted_node_id_str}"}
                        await self.runnable_graph.aupdate_state(config, {**update_for_resume_state, **error_update})
                        graph_has_ended_flag = True 
                        continue 

                    await self.runnable_graph.aupdate_state(config, update_for_resume_state)
                    current_input_for_stream = None 
                    logger.info(f"G2 ThreadID '{graph2_thread_id}': State updated for resume. Continuing workflow from '{interrupted_node_id_str}'.")

                except asyncio.TimeoutError:
                    error_msg = f"Timeout waiting for resume for '{interrupted_node_id_str}' on G2 ThreadID '{graph2_thread_id}'."
                    logger.error(error_msg)
                    await self.websocket_callback("workflow_timeout", {"message": error_msg, "node_name": interrupted_node_id_str}, graph2_thread_id)
                    await self.runnable_graph.aupdate_state(config, {"error": "ConfirmationTimeout"})
                    graph_has_ended_flag = True 
                
                except Exception as e_resume:
                    error_msg = f"Error processing resume for G2 ThreadID '{graph2_thread_id}', node '{interrupted_node_id_str}': {e_resume}"
                    logger.error(error_msg, exc_info=True)
                    await self.websocket_callback("execution_error", {"error": error_msg, "node_name": interrupted_node_id_str}, graph2_thread_id)
                    await self.runnable_graph.aupdate_state(config, {"error": error_msg})
                    graph_has_ended_flag = True 

            except Exception as e_loop: 
                error_msg = f"Critical error in workflow execution for G2 ThreadID '{graph2_thread_id}': {e_loop}"
                logger.critical(error_msg, exc_info=True)
                await self.websocket_callback("execution_error", {"error": error_msg}, graph2_thread_id)
                try:
                    await self.runnable_graph.aupdate_state(config, {"error": error_msg})
                except Exception as e_state_update:
                    logger.error(f"Failed to update state with critical error: {e_state_update}")
                graph_has_ended_flag = True 

        final_state_dict = {"error": "Workflow ended prematurely or state not retrieved."}
        try:
            logger.info(f"G2 ThreadID '{graph2_thread_id}': Workflow ended. Getting final state.")
            final_snapshot = await self.runnable_graph.aget_state(config) 
            final_state_dict = final_snapshot.values if final_snapshot else {}
            
            if final_state_dict.get("error"):
                await self.websocket_callback("execution_failed", {"final_state": final_state_dict, "message": f"Workflow failed with error: {final_state_dict.get('error')}"}, graph2_thread_id)
            else:
                await self.websocket_callback("execution_completed", {"final_state": final_state_dict, "message": "Workflow completed successfully."}, graph2_thread_id)
            return final_state_dict
        except Exception as e_final:
            error_msg = f"Error getting final state for G2 ThreadID '{graph2_thread_id}': {e_final}"
            logger.error(error_msg, exc_info=True)
            await self.websocket_callback("execution_error", {"error": error_msg, "message": "Failed to retrieve final workflow state."}, graph2_thread_id)
            final_state_dict["error"] = error_msg 
            return final_state_dict
        finally:
            if graph2_thread_id in self.resume_queues:
                while not self.resume_queues[graph2_thread_id].empty():
                    try:
                        self.resume_queues[graph2_thread_id].get_nowait()
                        self.resume_queues[graph2_thread_id].task_done() 
                    except asyncio.QueueEmpty:
                        break 
                    except Exception as e_drain:
                        logger.warning(f"Error draining resume queue for {graph2_thread_id}: {e_drain}")
                        break
                self.resume_queues.pop(graph2_thread_id, None)
                logger.info(f"Cleaned up resume queue for G2 ThreadID: {graph2_thread_id}")
