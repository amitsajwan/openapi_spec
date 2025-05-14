# websocket_helpers.py
import logging
import uuid
import json
import asyncio
from typing import Any, Dict, Optional, Callable, Awaitable

from fastapi import WebSocket, WebSocketDisconnect
# Removed: from starlette.websockets import WebSocketState

from models import BotState, GraphOutput as PlanSchema, ExecutionGraphState
# Assuming these are correctly importable from their new locations if refactored
# from core_logic.spec_processor import SpecProcessor
# from core_logic.graph_generator import GraphGenerator

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
    """
    Sends a JSON payload to the connected WebSocket client.
    Relies on exception handling for disconnections or errors during sending.
    The explicit WebSocketState.CONNECTED check has been removed.
    """
    try:
        # Attempt to serialize content if it's not directly JSON serializable (basic attempt)
        try:
            # This line doesn't actually send, it just checks if json.dumps would work.
            # The actual serialization happens in websocket.send_json()
            json.dumps(content)
            payload_to_send = content
        except TypeError as te:
            logger.warning(f"Content for WS type {msg_type} not directly JSON serializable, attempting str(): {te}. Content snippet: {str(content)[:200]}")
            payload_to_send = {"raw_content": str(content), "original_type": str(type(content))}

        # Directly attempt to send the JSON payload.
        # If the websocket is not connected, this will raise an exception.
        await websocket.send_json({
            "type": msg_type,
            "source": source_graph,
            "content": payload_to_send,
            "session_id": session_id,
            "graph2_thread_id": graph2_thread_id or session_id # Default to session_id if no specific G2 thread ID
        })
    except WebSocketDisconnect:
        logger.warning(f"[{session_id}] WebSocket disconnected while trying to send message: {msg_type}")
    except RuntimeError as e:
        client_state_info = "unknown"
        if hasattr(websocket, 'client_state'): # Check if client_state attribute exists
            client_state_info = str(websocket.client_state)
        logger.warning(f"[{session_id}] RuntimeError sending message type {msg_type}: {e}. WebSocket state: {client_state_info}")
    except Exception as e:
        logger.error(f"[{session_id}] Unexpected error sending message type {msg_type}: {e}", exc_info=True)


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
        await send_websocket_message_helper(
            websocket, event_type, data, session_id, "graph2_execution", graph2_thread_id_param
        )

        if event_type not in ["tool_start", "tool_end", "llm_start", "llm_stream", "llm_end"]: # Avoid logging too frequently for minor events
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

            # Run the planning graph (Graph 1)
            current_bot_state = await _run_planning_graph(
                websocket, session_id, current_bot_state, langgraph_planning_app
            )

            # Persist state changes from Graph 1
            _persist_bot_state_after_planning(_session_bot_state_store, current_bot_state)
            logger.debug(f"[{session_id}] Persisted BotState. Store status: {_session_bot_state_store.get('workflow_execution_status')}")

            # Conditionally initiate Graph 2 if planning indicated it
            if _session_bot_state_store.get("workflow_execution_status") == "pending_start":
                current_graph2_thread_id_for_session = await _initiate_execution_graph(
                    websocket, session_id, _session_bot_state_store,
                    api_executor_instance, active_graph2_executors,
                    active_graph2_definitions, graph2_ws_callback_for_this_session
                )
                if not current_graph2_thread_id_for_session:
                     _session_bot_state_store["workflow_execution_status"] = "failed" # Mark as failed if G2 couldn't start

    except WebSocketDisconnect:
        logger.info(f"[{session_id}] WebSocket disconnected by client.")
    except Exception as e_outer_loop:
        logger.critical(f"[{session_id}] Unhandled error in WebSocket loop: {e_outer_loop}", exc_info=True)
        # Ensure error is sent to client before attempting to close
        await send_websocket_message_helper(
            websocket, "error", {"error": "An unexpected server error occurred in the main loop."},
            session_id, "system_error"
        )
    finally:
        logger.info(f"[{session_id}] Cleaning up WebSocket connection resources.")
        if current_graph2_thread_id_for_session and current_graph2_thread_id_for_session in active_graph2_executors:
            logger.info(f"[{session_id}] Cleaning up active executor and definition for G2 thread: {current_graph2_thread_id_for_session}")
            # Potentially call a cleanup method on the executor if it exists
            # executor_to_clean = active_graph2_executors.pop(current_graph2_thread_id_for_session, None)
            # if executor_to_clean and hasattr(executor_to_clean, 'cleanup'):
            #     await executor_to_clean.cleanup()
            active_graph2_executors.pop(current_graph2_thread_id_for_session, None)
            active_graph2_definitions.pop(current_graph2_thread_id_for_session, None)
        else:
            logger.info(f"[{session_id}] No specific G2 thread ID tracked or active for final cleanup.")

        # Attempt to gracefully close the WebSocket if it's still open
        # Check client_state without importing WebSocketState directly from starlette,
        # relying on the WebSocket object's properties.
        current_ws_state = "unknown"
        if hasattr(websocket, 'client_state'):
            current_ws_state = str(websocket.client_state)

        if current_ws_state == "WebSocketState.CONNECTED": # Compare with string representation
            try:
                logger.info(f"[{session_id}] Attempting to close WebSocket (current state: {current_ws_state}).")
                await websocket.close(code=1000)
            except RuntimeError as e: # Handles cases like "Cannot call .close() another time."
                logger.warning(f"[{session_id}] Error closing WebSocket (already closing or closed?): {e}")
            except Exception as e: # Catch any other exception during close
                logger.error(f"[{session_id}] Exception during WebSocket close: {e}")
        else:
            logger.info(f"[{session_id}] WebSocket already not in connected state (State: {current_ws_state}). No explicit close call.")
        logger.info(f"[{session_id}] WebSocket connection closed and resources cleaned up.")


# --- Helper Functions for WebSocket Logic ---

def _initialize_bot_state_for_turn(
    session_id: str, user_input: str, session_store: Dict[str, Any]
) -> BotState:
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

    # Ensure scratchpad is initialized as a dict if not present or not a dict
    scratchpad_data = session_store.get("scratchpad", {})
    if not isinstance(scratchpad_data, dict):
        logger.warning(f"[{session_id}] Scratchpad from session_store was not a dict, reinitializing. Type was: {type(scratchpad_data)}")
        scratchpad_data = {}


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
        workflow_extracted_data=session_store.get("workflow_extracted_data", {}),
        scratchpad=scratchpad_data # Use the (potentially reinitialized) scratchpad
    )
    # state.scratchpad = session_store.get("scratchpad", {}) # Preserve scratchpad
    return state


async def _process_resume_command(
    websocket: WebSocket, session_id: str, user_input_text: str,
    session_store: Dict[str, Any],
    active_graph2_executors: Dict[str, GraphExecutionManager]
):
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
                # Ensure resume_payload_for_graph2 is a dict if that's what submit_resume_data expects
                if not isinstance(resume_payload_for_graph2, dict):
                    await send_websocket_message_helper(
                        websocket, "error", {"error": "Resume payload must be a JSON object."},
                        session_id, "system_error", graph2_thread_id_to_resume
                    )
                    return

                submitted = await exec_manager.submit_resume_data(graph2_thread_id_to_resume, resume_payload_for_graph2)
                if submitted:
                    await send_websocket_message_helper(
                        websocket, "info", {"message": "Resume data submitted to execution manager."},
                        session_id, "graph2_execution", graph2_thread_id_to_resume
                    )
                    session_store["workflow_execution_status"] = "running"
                else:
                    await send_websocket_message_helper(
                        websocket, "error", {"error": "Failed to submit resume data to the manager (manager rejected)."},
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
    # This message indicates the start of Graph 1 processing.
    # The client-side script.js should have already started its "thinking" indicator
    # when the user sent the message that triggered this flow.
    await send_websocket_message_helper(
        websocket, "status", {"message": "Processing request with Planning Workflow..."}, # More generic status
        session_id, "graph1_planning"
    )
    planning_config = {"configurable": {"thread_id": session_id}}
    current_bot_state_after_planning = bot_state
    last_intermediate_message_sent: Optional[str] = None

    try:
        graph1_input_dict = current_bot_state_after_planning.model_dump(exclude_none=True)
        if 'scratchpad' not in graph1_input_dict: # Ensure scratchpad is included
             graph1_input_dict['scratchpad'] = current_bot_state_after_planning.scratchpad


        async for event in langgraph_planning_app.astream_events(graph1_input_dict, config=planning_config, version="v2"):
            event_name = event["event"]
            event_data = event.get("data", {})

            if event_name == "on_tool_end" or event_name == "on_chat_model_end" or event_name == "on_chain_end":
                node_output_any = event_data.get("output")

                # Update current_bot_state_after_planning from node output
                if isinstance(node_output_any, BotState):
                    current_bot_state_after_planning = node_output_any
                elif isinstance(node_output_any, dict):
                    try:
                        # Safely update BotState from dict
                        valid_fields_to_update = {k: v for k, v in node_output_any.items() if hasattr(BotState, k)}
                        current_bot_state_after_planning = current_bot_state_after_planning.model_copy(update=valid_fields_to_update)
                        if 'scratchpad' in node_output_any and isinstance(node_output_any['scratchpad'], dict):
                            current_bot_state_after_planning.scratchpad.update(node_output_any['scratchpad'])
                    except Exception as e_val:
                        logger.error(f"[{session_id}] Error updating BotState from G1 node output dict: {e_val}, dict: {str(node_output_any)[:500]}")

                # Send intermediate messages from scratchpad
                if current_bot_state_after_planning.scratchpad and 'intermediate_messages' in current_bot_state_after_planning.scratchpad:
                    message_queue = current_bot_state_after_planning.scratchpad.pop('intermediate_messages', [])
                    for queued_msg_content in message_queue:
                        msg_to_send_str = ""
                        if isinstance(queued_msg_content, str):
                            msg_to_send_str = queued_msg_content
                        elif isinstance(queued_msg_content, dict) and queued_msg_content.get("message"):
                            msg_to_send_str = queued_msg_content.get("message")
                        
                        if msg_to_send_str and msg_to_send_str != last_intermediate_message_sent:
                            await send_websocket_message_helper(
                                websocket, "intermediate", {"message": msg_to_send_str},
                                session_id, "graph1_planning"
                            )
                            last_intermediate_message_sent = msg_to_send_str
                
                # Send primary response if different
                if current_bot_state_after_planning.response:
                    msg_to_send = current_bot_state_after_planning.response
                    if msg_to_send != last_intermediate_message_sent:
                        await send_websocket_message_helper(
                            websocket, "intermediate", {"message": msg_to_send}, # Could also be "status" or "final" depending on context
                            session_id, "graph1_planning"
                        )
                        last_intermediate_message_sent = msg_to_send
                    current_bot_state_after_planning.response = None # Clear after sending

                # Send graph updates from scratchpad
                if current_bot_state_after_planning.scratchpad and 'graph_to_send' in current_bot_state_after_planning.scratchpad:
                    graph_json_str = current_bot_state_after_planning.scratchpad.pop('graph_to_send', None)
                    if graph_json_str:
                        try:
                            graph_content = json.loads(graph_json_str)
                            await send_websocket_message_helper(
                                websocket, "graph_update", graph_content, # Ensure content is the dict itself
                                session_id, "graph1_planning"
                            )
                        except json.JSONDecodeError as e_json:
                            logger.error(f"[{session_id}] G1: 'graph_to_send' was not valid JSON: {e_json}. Content: {graph_json_str[:100]}")
                        except Exception as e_graph:
                            logger.error(f"[{session_id}] G1: Failed to send graph_update: {e_graph}")
                
                # Send API responses from scratchpad (if Graph 1 makes API calls and stores results)
                if current_bot_state_after_planning.scratchpad and 'last_api_response' in current_bot_state_after_planning.scratchpad:
                    api_response_data = current_bot_state_after_planning.scratchpad.pop('last_api_response', None)
                    if api_response_data and isinstance(api_response_data, dict):
                        await send_websocket_message_helper(
                            websocket, "api_response", api_response_data,
                            session_id, "graph1_planning" # Or adjust source if applicable
                        )


            elif event_name == "on_graph_end": # This is the end of the entire Graph 1 execution
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
                
                # Send the final response for Graph 1
                # This is crucial for the client to stop its "thinking" indicator.
                final_msg_content = current_bot_state_after_planning.final_response
                if not final_msg_content and current_bot_state_after_planning.response:
                    final_msg_content = current_bot_state_after_planning.response
                    logger.warning(f"[{session_id}] G1: final_response was empty, using lingering .response for final message: '{str(final_msg_content)[:50]}...'.")
                
                if not final_msg_content and not (current_bot_state_after_planning.workflow_execution_status == "pending_start"):
                    # If no specific final message and not pending G2, send a generic completion.
                    final_msg_content = "Planning phase complete."
                    logger.info(f"[{session_id}] G1: No specific final_response from graph, sending generic: '{final_msg_content}'")


                if final_msg_content: # Only send if there's something to send
                    # Prepare data_payload for final_response
                    final_payload = {"message": final_msg_content}
                    # Include graph if it's part of the final state and hasn't been sent or needs re-sending
                    if current_bot_state_after_planning.execution_graph:
                        try:
                            final_payload["graph"] = current_bot_state_after_planning.execution_graph.model_dump(exclude_none=True)
                        except Exception as e_dump:
                            logger.error(f"[{session_id}] G1: Error dumping execution_graph for final response: {e_dump}")

                    await send_websocket_message_helper(
                        websocket, "final", final_payload, # Send the payload containing message and potentially graph
                        session_id, "graph1_planning"
                    )
                
                current_bot_state_after_planning.final_response = "" # Clear after sending
                current_bot_state_after_planning.response = None # Clear any lingering response
                break # Exit the astream_events loop for Graph 1

    except Exception as e_plan:
        logger.critical(f"[{session_id}] Planning Graph (Graph 1) execution error: {e_plan}", exc_info=True)
        # This error message will trigger the client to stop "thinking"
        await send_websocket_message_helper(
            websocket, "error", {"error": f"Error during planning phase: {str(e_plan)[:200]}"},
            session_id, "system_error"
        )
        current_bot_state_after_planning.workflow_execution_status = "failed" # Update status

    return current_bot_state_after_planning


def _persist_bot_state_after_planning(session_store: Dict[str, Any], bot_state: BotState):
    session_store["openapi_spec_text"] = bot_state.openapi_spec_text
    session_store["openapi_schema"] = bot_state.openapi_schema
    session_store["schema_cache_key"] = bot_state.schema_cache_key
    session_store["schema_summary"] = bot_state.schema_summary
    session_store["identified_apis"] = bot_state.identified_apis
    session_store["payload_descriptions"] = bot_state.payload_descriptions
    if bot_state.execution_graph:
        session_store["execution_graph"] = bot_state.execution_graph.model_dump(exclude_none=True)
    else:
        session_store["execution_graph"] = None # Ensure it's explicitly None if not present
    session_store["plan_generation_goal"] = bot_state.plan_generation_goal
    session_store["input_is_spec"] = bot_state.input_is_spec
    session_store["workflow_execution_status"] = bot_state.workflow_execution_status
    session_store["workflow_execution_results"] = bot_state.workflow_execution_results
    session_store["workflow_extracted_data"] = bot_state.workflow_extracted_data
    session_store["scratchpad"] = bot_state.scratchpad # Persist the entire scratchpad


async def _initiate_execution_graph(
    websocket: WebSocket,
    session_id: str,
    session_store: Dict[str, Any],
    api_executor_instance: APIExecutor,
    active_graph2_executors: Dict[str, GraphExecutionManager],
    active_graph2_definitions: Dict[str, ExecutionGraphDefinition],
    graph2_ws_callback: Callable[[str, Dict[str, Any], Optional[str]], Awaitable[None]]
) -> Optional[str]:
    logger.info(f"[{session_id}] Checking if Graph 2 initiation is pending...")
    exec_plan_dict = session_store.get("execution_graph") # This should be a dict from model_dump
    exec_plan_model = None
    if isinstance(exec_plan_dict, dict):
        try:
            exec_plan_model = PlanSchema.model_validate(exec_plan_dict)
        except Exception as e_val:
            logger.error(f"[{session_id}] Failed to validate execution_graph for G2 from session_store: {e_val}. Data: {str(exec_plan_dict)[:200]}")
            await send_websocket_message_helper(
                websocket, "error", {"error": "Execution plan is invalid and cannot be used for Graph 2."},
                session_id, "system_error"
            )
            return None
    else:
        logger.warning(f"[{session_id}] Graph 2 initiation skipped: No execution plan dictionary found in session store. Found type: {type(exec_plan_dict)}")
        await send_websocket_message_helper(
            websocket, "warning", {"message": "Cannot start execution: No valid plan available for Graph 2."},
            session_id, "system_warning"
        )
        return None


    if not api_executor_instance:
        logger.error(f"[{session_id}] Graph 2 initiation failed: APIExecutor not available.")
        await send_websocket_message_helper(
            websocket, "error", {"error": "Cannot start execution: API executor service is not available for Graph 2."},
            session_id, "system_error"
        )
        return None

    graph2_thread_id = f"{session_id}_exec_{uuid.uuid4().hex[:8]}"
    logger.info(f"[{session_id}] Generated new G2 Thread ID for execution: {graph2_thread_id}")

    try:
        exec_graph_def = ExecutionGraphDefinition(
            graph_execution_plan=exec_plan_model, # Use the validated model
            api_executor=api_executor_instance
        )
        active_graph2_definitions[graph2_thread_id] = exec_graph_def
        runnable_exec_graph = exec_graph_def.get_runnable_graph()

        exec_manager = GraphExecutionManager(
            runnable_graph=runnable_exec_graph,
            graph_definition=exec_graph_def,
            websocket_callback=graph2_ws_callback,
            planning_checkpointer=None, # No separate checkpointer for G2 state for now
            main_planning_session_id=session_id # Link back to G1 session
        )
        active_graph2_executors[graph2_thread_id] = exec_manager

        initial_exec_state_values = ExecutionGraphState(
            initial_input=session_store.get("workflow_extracted_data", {}),
            # Any other initial state fields for Graph 2 can be set here
        ).model_dump(exclude_none=True)

        exec_graph_config = {"configurable": {"thread_id": graph2_thread_id}}

        await send_websocket_message_helper(
            websocket, "info", {"message": f"Execution phase (Workflow Graph) starting with G2 Thread ID: {graph2_thread_id}."},
            session_id, "graph2_execution", graph2_thread_id
        )
        session_store["workflow_execution_status"] = "running"

        # Start Graph 2 execution as a background task
        asyncio.create_task(exec_manager.execute_workflow(initial_exec_state_values, exec_graph_config))
        return graph2_thread_id

    except Exception as e_exec_setup:
        logger.error(f"[{session_id}] Failed to set up and start Graph 2 (G2 Thread ID: {graph2_thread_id}): {e_exec_setup}", exc_info=True)
        await send_websocket_message_helper(
            websocket, "error", {"error": f"Server error during execution setup for Graph 2: {str(e_exec_setup)[:150]}"},
            session_id, "system_error", graph2_thread_id
        )
        active_graph2_executors.pop(graph2_thread_id, None)
        active_graph2_definitions.pop(graph2_thread_id, None)
        return None
