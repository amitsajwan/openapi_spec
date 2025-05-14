# websocket_helpers.py
import logging
import uuid
import json
import asyncio
from typing import Any, Dict, Optional, Callable, Awaitable

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from models import BotState, GraphOutput as PlanSchema, ExecutionGraphState
# Assuming these are correctly importable from their new locations if refactored
# from core_logic.spec_processor import SpecProcessor # Example, if needed directly
# from core_logic.graph_generator import GraphGenerator # Example

from execution_graph_definition import ExecutionGraphDefinition # Keep if used
from execution_manager import GraphExecutionManager # Keep if used
from api_executor import APIExecutor # For type hinting

logger = logging.getLogger(__name__)

# --- WebSocket Message Sending Helper ---
async def send_websocket_message_helper(
    websocket: WebSocket,
    msg_type: str,
    content: Any,
    session_id: str, 
    source_graph: str = "graph1_planning",
    graph2_thread_id: Optional[str] = None
):
    if websocket.client_state != WebSocketState.CONNECTED:
        logger.warning(f"[{session_id}] Attempted to send WS message but socket is not connected (State: {websocket.client_state}). Type: {msg_type}")
        return

    try:
        try:
            json.dumps(content) 
            payload_to_send = content
        except TypeError as te:
            logger.warning(f"Content for WS type {msg_type} not directly JSON serializable, attempting str(): {te}. Content snippet: {str(content)[:200]}")
            payload_to_send = {"raw_content": str(content)} 

        await websocket.send_json({
            "type": msg_type,
            "source": source_graph,
            "content": payload_to_send,
            "session_id": session_id, 
            "graph2_thread_id": graph2_thread_id or session_id 
        })
    except Exception as e:
        logger.error(f"[{session_id}] WebSocket send error (Type: {msg_type}, G2_Thread: {graph2_thread_id}): {e}", exc_info=False)


# --- Main WebSocket Connection Handler ---
async def handle_websocket_connection(
    websocket: WebSocket,
    session_id: str, 
    langgraph_planning_app: Any,
    api_executor_instance: APIExecutor, # Assuming this is for Graph 2
    active_graph2_executors: Dict[str, GraphExecutionManager],
    active_graph2_definitions: Dict[str, ExecutionGraphDefinition]
):
    await send_websocket_message_helper(
        websocket, "info", {"message": "Connection established."}, 
        session_id, "system"
    )

    _session_bot_state_store: Dict[str, Any] = {"workflow_execution_status": "idle"}
    current_graph2_thread_id_for_session: Optional[str] = None 

    async def graph2_ws_callback_for_this_session(
        event_type: str,
        data: Dict[str, Any],
        graph2_thread_id_param: Optional[str] 
    ):
        # ... (graph2_ws_callback_for_this_session implementation remains the same)
        await send_websocket_message_helper(
            websocket, event_type, data, session_id, "graph2_execution", graph2_thread_id_param
        )

        if event_type not in ["tool_start", "tool_end"]: 
            logger.info(f"[{session_id}] G2 Callback (G2_Thread: {graph2_thread_id_param}). Event: {event_type}. Store status before: {_session_bot_state_store.get('workflow_execution_status')}")
            new_status = _session_bot_state_store.get("workflow_execution_status", "idle")

            if event_type == "human_intervention_required":
                new_status = "paused_for_confirmation"
            elif event_type in ["execution_completed", "execution_failed", "workflow_timeout"]:
                new_status = "completed" if event_type == "execution_completed" else "failed"
                _session_bot_state_store["workflow_execution_results"] = data.get("final_state", {})
                if graph2_thread_id_param:
                    if graph2_thread_id_param in active_graph2_executors:
                        logger.info(f"[{session_id}] Workflow for G2 Thread ID {graph2_thread_id_param} ended. Removing its executor and definition.")
                        active_graph2_executors.pop(graph2_thread_id_param, None)
                        active_graph2_definitions.pop(graph2_thread_id_param, None)
                    else:
                        logger.warning(f"[{session_id}] G2 Thread ID {graph2_thread_id_param} ended but not found in active executors for cleanup.")
            
            _session_bot_state_store["workflow_execution_status"] = new_status
            logger.info(f"[{session_id}] G2 Callback. Store status after: {new_status}")


    try:
        while True:
            user_input_text = await websocket.receive_text()
            user_input_text = user_input_text.strip()
            if not user_input_text:
                continue
            logger.info(f"[{session_id}] RX: '{user_input_text[:100]}...'")

            current_bot_state = _initialize_bot_state_for_turn(session_id, user_input_text, _session_bot_state_store)

            if user_input_text.lower().startswith("resume_exec"):
                await _process_resume_command(
                    websocket, session_id, user_input_text,
                    _session_bot_state_store, active_graph2_executors
                )
                continue 

            current_bot_state = await _run_planning_graph(
                websocket, session_id, current_bot_state, langgraph_planning_app
            )

            _persist_bot_state_after_planning(_session_bot_state_store, current_bot_state)
            logger.debug(f"[{session_id}] Persisted BotState. Store status: {_session_bot_state_store.get('workflow_execution_status')}")

            if _session_bot_state_store.get("workflow_execution_status") == "pending_start":
                current_graph2_thread_id_for_session = await _initiate_execution_graph(
                    websocket, session_id, _session_bot_state_store,
                    api_executor_instance, active_graph2_executors,
                    active_graph2_definitions, graph2_ws_callback_for_this_session 
                )
                if not current_graph2_thread_id_for_session: 
                     _session_bot_state_store["workflow_execution_status"] = "failed"

    except WebSocketDisconnect:
        logger.info(f"[{session_id}] WebSocket disconnected by client.")
    except Exception as e_outer_loop:
        logger.critical(f"[{session_id}] Unhandled error in WebSocket loop: {e_outer_loop}", exc_info=True)
        await send_websocket_message_helper(
            websocket, "error", {"error": "An unexpected server error occurred."},
            session_id, "system_error"
        )
    finally:
        logger.info(f"[{session_id}] Cleaning up WebSocket connection resources.")
        if current_graph2_thread_id_for_session:
            logger.info(f"[{session_id}] Cleaning up executor and definition for last G2 thread: {current_graph2_thread_id_for_session}")
            active_graph2_executors.pop(current_graph2_thread_id_for_session, None)
            active_graph2_definitions.pop(current_graph2_thread_id_for_session, None)
        else:
            logger.info(f"[{session_id}] No specific G2 thread ID tracked for final cleanup.")

        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.close(code=1000) 
            except RuntimeError as e:
                logger.warning(f"[{session_id}] Error closing WebSocket (already closing?): {e}")
            except Exception as e:
                logger.error(f"[{session_id}] Exception during WebSocket close: {e}")
        logger.info(f"[{session_id}] WebSocket connection closed and resources cleaned up.")


# --- Helper Functions for WebSocket Logic ---

def _initialize_bot_state_for_turn(
    session_id: str, user_input: str, session_store: Dict[str, Any]
) -> BotState:
    # ... (_initialize_bot_state_for_turn implementation remains the same)
    exec_graph_data = session_store.get("execution_graph")
    execution_graph_model = None
    if isinstance(exec_graph_data, dict):
        try:
            execution_graph_model = PlanSchema.model_validate(exec_graph_data)
        except Exception as e:
            logger.error(f"[{session_id}] Failed to validate execution_graph from session_store: {e}")
            execution_graph_model = None 
    elif isinstance(exec_graph_data, PlanSchema):
        execution_graph_model = exec_graph_data

    state = BotState(
        session_id=session_id,
        user_input=user_input,
        openapi_spec_text=session_store.get("openapi_spec_text"),
        openapi_schema=session_store.get("openapi_schema"),
        schema_cache_key=session_store.get("schema_cache_key"),
        schema_summary=session_store.get("schema_summary"),
        identified_apis=session_store.get("identified_apis", []),
        payload_descriptions=session_store.get("payload_descriptions", {}),
        execution_graph=execution_graph_model,
        plan_generation_goal=session_store.get("plan_generation_goal"),
        input_is_spec=session_store.get("input_is_spec", False), 
        workflow_execution_status=session_store.get("workflow_execution_status", "idle"),
        workflow_execution_results=session_store.get("workflow_execution_results", {}),
        workflow_extracted_data=session_store.get("workflow_extracted_data", {})
    )
    state.scratchpad = session_store.get("scratchpad", {}) # Preserve scratchpad across turns if needed, or clear it
    return state


async def _process_resume_command(
    websocket: WebSocket, session_id: str, user_input_text: str,
    session_store: Dict[str, Any],
    active_graph2_executors: Dict[str, GraphExecutionManager]
):
    # ... (_process_resume_command implementation remains the same)
    try:
        parts = user_input_text.split(" ", 2)
        if len(parts) < 3:
            await send_websocket_message_helper(
                websocket, "warning", {"message": "Resume command format: resume_exec <graph2_thread_id> <json_payload>"},
                session_id, "system_warning"
            )
            return

        graph2_thread_id_to_resume = parts[1]
        payload_str = parts[2].strip()
        resume_payload_for_graph2 = json.loads(payload_str) if payload_str else None

        if not resume_payload_for_graph2:
            await send_websocket_message_helper(
                websocket, "warning", {"message": "Resume command is missing the JSON payload."},
                session_id, "system_warning", graph2_thread_id_to_resume
            )
            return

        exec_manager = active_graph2_executors.get(graph2_thread_id_to_resume)

        if session_store.get("workflow_execution_status") == "paused_for_confirmation":
            if exec_manager:
                submitted = await exec_manager.submit_resume_data(graph2_thread_id_to_resume, resume_payload_for_graph2)
                if submitted:
                    await send_websocket_message_helper(
                        websocket, "info", {"message": "Resume data submitted to execution manager."},
                        session_id, "graph2_execution", graph2_thread_id_to_resume
                    )
                    session_store["workflow_execution_status"] = "running" 
                else:
                    await send_websocket_message_helper(
                        websocket, "error", {"error": "Failed to submit resume data to the manager."},
                        session_id, "graph2_execution", graph2_thread_id_to_resume
                    )
            else:
                await send_websocket_message_helper(
                    websocket, "warning", {"message": f"Cannot resume: No active execution manager found for G2 Thread ID '{graph2_thread_id_to_resume}'. It might have already finished or failed."},
                    session_id, "system_warning"
                )
        else:
            status_msg = session_store.get("workflow_execution_status", "unknown")
            await send_websocket_message_helper(
                websocket, "warning", {"message": f"Cannot resume: Workflow is not currently paused for confirmation (current status: {status_msg})."},
                session_id, "system_warning"
            )
    except json.JSONDecodeError:
        await send_websocket_message_helper(
            websocket, "error", {"error": "Invalid JSON payload provided for resume_exec command."},
            session_id, "system_error"
        )
    except Exception as e_resume:
        logger.error(f"[{session_id}] Error processing 'resume_exec' command: {e_resume}", exc_info=True)
        await send_websocket_message_helper(
            websocket, "error", {"error": f"Error processing resume command: {str(e_resume)}"},
            session_id, "system_error"
        )


async def _run_planning_graph(
    websocket: WebSocket, session_id: str, bot_state: BotState, langgraph_planning_app: Any
) -> BotState:
    await send_websocket_message_helper(
        websocket, "status", {"message": "Planning workflow..."},
        session_id, "graph1_planning"
    )
    planning_config = {"configurable": {"thread_id": session_id}} 
    current_bot_state_after_planning = bot_state 
    last_intermediate_message_sent: Optional[str] = None 

    try:
        graph1_input_dict = current_bot_state_after_planning.model_dump(exclude_none=True)
        # Ensure scratchpad is part of the input if it's used by nodes and needs to be checkpointed by LangGraph
        if 'scratchpad' not in graph1_input_dict:
             graph1_input_dict['scratchpad'] = current_bot_state_after_planning.scratchpad


        async for event in langgraph_planning_app.astream_events(graph1_input_dict, config=planning_config, version="v2"): 
            event_name = event["event"]
            event_data = event.get("data", {})
            
            if event_name == "on_tool_end" or event_name == "on_chat_model_end" or event_name == "on_chain_end": 
                node_output_any = event_data.get("output")

                if isinstance(node_output_any, BotState):
                    current_bot_state_after_planning = node_output_any
                elif isinstance(node_output_any, dict):
                    try:
                        # Update only fields present in BotState to avoid Pydantic errors if output has extra keys
                        valid_fields_to_update = {k: v for k, v in node_output_any.items() if hasattr(BotState, k)}
                        current_bot_state_after_planning = current_bot_state_after_planning.model_copy(update=valid_fields_to_update)
                        # Ensure scratchpad is preserved/updated if it was in node_output_any
                        if 'scratchpad' in node_output_any and isinstance(node_output_any['scratchpad'], dict):
                            current_bot_state_after_planning.scratchpad.update(node_output_any['scratchpad'])

                    except Exception as e_val:
                        logger.error(f"[{session_id}] Error updating BotState from G1 node output dict: {e_val}, dict: {str(node_output_any)[:500]}")
                
                # --- MODIFIED SECTION for intermediate messages ---
                # Send queued intermediate messages first
                if current_bot_state_after_planning.scratchpad and 'intermediate_messages' in current_bot_state_after_planning.scratchpad:
                    message_queue = current_bot_state_after_planning.scratchpad.pop('intermediate_messages', [])
                    for queued_msg_content in message_queue:
                        if isinstance(queued_msg_content, str) and queued_msg_content and queued_msg_content != last_intermediate_message_sent:
                            await send_websocket_message_helper(
                                websocket, "intermediate", {"message": queued_msg_content},
                                session_id, "graph1_planning"
                            )
                            last_intermediate_message_sent = queued_msg_content
                        elif isinstance(queued_msg_content, dict) and queued_msg_content.get("message"): # If message is an object
                             msg_to_send = queued_msg_content.get("message")
                             if msg_to_send and msg_to_send != last_intermediate_message_sent:
                                await send_websocket_message_helper(
                                    websocket, "intermediate", {"message": msg_to_send},
                                    session_id, "graph1_planning"
                                )
                                last_intermediate_message_sent = msg_to_send


                # Send the primary response if it's different from the last queued one
                if current_bot_state_after_planning.response:
                    msg_to_send = current_bot_state_after_planning.response
                    if msg_to_send != last_intermediate_message_sent: 
                        await send_websocket_message_helper(
                            websocket, "intermediate", {"message": msg_to_send},
                            session_id, "graph1_planning"
                        )
                        last_intermediate_message_sent = msg_to_send
                    current_bot_state_after_planning.response = None # Clear after sending or if duplicate
                # --- END OF MODIFIED SECTION ---


                if current_bot_state_after_planning.scratchpad and 'graph_to_send' in current_bot_state_after_planning.scratchpad:
                    graph_json_str = current_bot_state_after_planning.scratchpad.pop('graph_to_send', None)
                    if graph_json_str:
                        try:
                            graph_content = json.loads(graph_json_str) 
                            await send_websocket_message_helper(
                                websocket, "graph_update", graph_content,
                                session_id, "graph1_planning"
                            )
                        except json.JSONDecodeError as e_json:
                            logger.error(f"[{session_id}] G1: 'graph_to_send' was not valid JSON: {e_json}. Content: {graph_json_str[:100]}")
                        except Exception as e_graph:
                            logger.error(f"[{session_id}] G1: Failed to send graph_update: {e_graph}")

            elif event_name == "on_graph_end":
                final_graph_output = event_data.get("output") 
                if isinstance(final_graph_output, dict):
                    try:
                        current_bot_state_after_planning = BotState.model_validate(final_graph_output)
                    except Exception as e_val:
                         logger.error(f"[{session_id}] Error validating final BotState from G1 output: {e_val}, dict: {str(final_graph_output)[:500]}")
                elif isinstance(final_graph_output, BotState): 
                    current_bot_state_after_planning = final_graph_output
                else:
                    logger.error(f"[{session_id}] G1: on_graph_end output was not a dict or BotState: {type(final_graph_output)}")
                break 

        final_msg_content = current_bot_state_after_planning.final_response
        if not final_msg_content and current_bot_state_after_planning.response: # Should be rare
            final_msg_content = current_bot_state_after_planning.response
            logger.warning(f"[{session_id}] G1: final_response was empty, using lingering .response for final message: '{str(final_msg_content)[:50]}...'.")

        if final_msg_content:
            if final_msg_content != last_intermediate_message_sent:
                await send_websocket_message_helper(
                    websocket, "final", {"message": final_msg_content},
                    session_id, "graph1_planning"
                )
            else: # Suppress duplicate final message
                 logger.info(f"[{session_id}] G1: Final message ('{str(final_msg_content)[:50]}...') is identical to last intermediate. Suppressing duplicate final message.")
        
        current_bot_state_after_planning.final_response = "" 
        current_bot_state_after_planning.response = None

    except Exception as e_plan:
        logger.critical(f"[{session_id}] Planning Graph (Graph 1) execution error: {e_plan}", exc_info=True)
        await send_websocket_message_helper(
            websocket, "error", {"error": f"Error during planning phase: {str(e_plan)[:200]}"}, 
            session_id, "system_error"
        )
        current_bot_state_after_planning.workflow_execution_status = "failed" 

    return current_bot_state_after_planning


def _persist_bot_state_after_planning(session_store: Dict[str, Any], bot_state: BotState):
    # ... (_persist_bot_state_after_planning implementation remains the same)
    session_store["openapi_spec_text"] = bot_state.openapi_spec_text
    session_store["openapi_schema"] = bot_state.openapi_schema
    session_store["schema_cache_key"] = bot_state.schema_cache_key
    session_store["schema_summary"] = bot_state.schema_summary
    session_store["identified_apis"] = bot_state.identified_apis
    session_store["payload_descriptions"] = bot_state.payload_descriptions
    if bot_state.execution_graph:
        session_store["execution_graph"] = bot_state.execution_graph.model_dump(exclude_none=True)
    else:
        session_store["execution_graph"] = None
    session_store["plan_generation_goal"] = bot_state.plan_generation_goal
    session_store["input_is_spec"] = bot_state.input_is_spec 
    session_store["workflow_execution_status"] = bot_state.workflow_execution_status
    session_store["workflow_execution_results"] = bot_state.workflow_execution_results
    session_store["workflow_extracted_data"] = bot_state.workflow_extracted_data
    session_store["scratchpad"] = bot_state.scratchpad # Persist scratchpad


async def _initiate_execution_graph(
    websocket: WebSocket,
    session_id: str, 
    session_store: Dict[str, Any],
    api_executor_instance: APIExecutor,
    active_graph2_executors: Dict[str, GraphExecutionManager],
    active_graph2_definitions: Dict[str, ExecutionGraphDefinition],
    graph2_ws_callback: Callable[[str, Dict[str, Any], Optional[str]], Awaitable[None]]
) -> Optional[str]: 
    # ... (_initiate_execution_graph implementation remains the same)
    logger.info(f"[{session_id}] Checking if Graph 2 initiation is pending...")
    exec_plan_dict = session_store.get("execution_graph")
    exec_plan_model = PlanSchema.model_validate(exec_plan_dict) if exec_plan_dict else None

    if not exec_plan_model:
        logger.warning(f"[{session_id}] Graph 2 initiation skipped: No execution plan found in session store.")
        await send_websocket_message_helper(
            websocket, "warning", {"message": "Cannot start execution: No valid plan available."},
            session_id, "system_warning"
        )
        return None
    if not api_executor_instance: 
        logger.error(f"[{session_id}] Graph 2 initiation failed: APIExecutor not available.")
        await send_websocket_message_helper(
            websocket, "error", {"error": "Cannot start execution: API executor service is not available."},
            session_id, "system_error"
        )
        return None

    graph2_thread_id = f"{session_id}_exec_{uuid.uuid4().hex[:8]}"
    logger.info(f"[{session_id}] Generated new G2 Thread ID for execution: {graph2_thread_id}")

    try:
        exec_graph_def = ExecutionGraphDefinition(
            graph_execution_plan=exec_plan_model,
            api_executor=api_executor_instance
        )
        active_graph2_definitions[graph2_thread_id] = exec_graph_def
        runnable_exec_graph = exec_graph_def.get_runnable_graph()

        exec_manager = GraphExecutionManager(
            runnable_graph=runnable_exec_graph,
            graph_definition=exec_graph_def,
            websocket_callback=graph2_ws_callback, 
            planning_checkpointer=None, 
            main_planning_session_id=session_id 
        )
        active_graph2_executors[graph2_thread_id] = exec_manager

        initial_exec_state_values = ExecutionGraphState(
            initial_input=session_store.get("workflow_extracted_data", {}),
        ).model_dump(exclude_none=True)

        exec_graph_config = {"configurable": {"thread_id": graph2_thread_id}}

        await send_websocket_message_helper(
            websocket, "info", {"message": f"Execution phase (Graph 2) starting with G2 Thread ID: {graph2_thread_id}."},
            session_id, "graph2_execution", graph2_thread_id 
        )
        session_store["workflow_execution_status"] = "running" 

        asyncio.create_task(exec_manager.execute_workflow(initial_exec_state_values, exec_graph_config))
        return graph2_thread_id

    except Exception as e_exec_setup:
        logger.error(f"[{session_id}] Failed to set up and start Graph 2 (G2 Thread ID: {graph2_thread_id}): {e_exec_setup}", exc_info=True)
        await send_websocket_message_helper(
            websocket, "error", {"error": f"Server error during execution setup: {str(e_exec_setup)[:150]}"},
            session_id, "system_error", graph2_thread_id
        )
        active_graph2_executors.pop(graph2_thread_id, None)
        active_graph2_definitions.pop(graph2_thread_id, None)
        return None

