# websocket_helpers.py
import logging
import uuid
import json
import asyncio
from typing import Any, Dict, Optional, Callable, Awaitable

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from models import BotState, GraphOutput as PlanSchema, ExecutionGraphState
from execution_graph_definition import ExecutionGraphDefinition
from execution_manager import GraphExecutionManager
from api_executor import APIExecutor # For type hinting

logger = logging.getLogger(__name__)

# --- WebSocket Message Sending Helper ---
async def send_websocket_message_helper(
    websocket: WebSocket,
    msg_type: str,
    content: Any,
    session_id: str, # Graph 1 session ID
    source_graph: str = "graph1_planning",
    graph2_thread_id: Optional[str] = None
):
    """
    Helper function to send JSON messages over a WebSocket connection.
    Ensures content is JSON serializable.
    """
    if websocket.client_state != WebSocketState.CONNECTED:
        logger.warning(f"[{session_id}] Attempted to send WS message but socket is not connected (State: {websocket.client_state}). Type: {msg_type}")
        return

    try:
        # Ensure content is JSON serializable before sending
        try:
            # Attempt to serialize to catch errors early, then send the original object
            # if it's already a dict/list/primitive that send_json can handle.
            # If it's a Pydantic model, model_dump() should have been called before passing here.
            json.dumps(content)
            payload_to_send = content
        except TypeError as te:
            logger.warning(f"Content for WS type {msg_type} not directly JSON serializable, attempting str(): {te}. Content snippet: {str(content)[:200]}")
            payload_to_send = {"raw_content": str(content)} # Fallback

        await websocket.send_json({
            "type": msg_type,
            "source": source_graph,
            "content": payload_to_send,
            "session_id": session_id, # Always include the main G1 session ID
            "graph2_thread_id": graph2_thread_id or session_id # Use specific G2 ID if available
        })
    except Exception as e:
        # Catch broader exceptions during send_json itself (e.g., connection dropped mid-send)
        logger.error(f"[{session_id}] WebSocket send error (Type: {msg_type}, G2_Thread: {graph2_thread_id}): {e}", exc_info=False)


# --- Main WebSocket Connection Handler ---
async def handle_websocket_connection(
    websocket: WebSocket,
    session_id: str, # Graph 1 (planning) session ID
    langgraph_planning_app: Any,
    api_executor_instance: APIExecutor,
    active_graph2_executors: Dict[str, GraphExecutionManager],
    active_graph2_definitions: Dict[str, ExecutionGraphDefinition]
):
    """
    Manages an active WebSocket connection, processing incoming messages,
    orchestrating Graph 1 (planning) and Graph 2 (execution).
    """
    await send_websocket_message_helper(
        websocket, "info", {"message": "Connection established with helper."},
        session_id, "system"
    )

    # Session-specific store for BotState fields that persist across Graph 1 invocations
    _session_bot_state_store: Dict[str, Any] = {"workflow_execution_status": "idle"}
    current_graph2_thread_id_for_session: Optional[str] = None # Tracks the G2 ID for the current G1 turn

    # --- Graph 2 Callback (Nested to capture session-specific WebSocket & ID) ---
    async def graph2_ws_callback_for_this_session(
        event_type: str,
        data: Dict[str, Any],
        graph2_thread_id_param: Optional[str] # This is the G2 thread ID from the manager
    ):
        """
        Callback for GraphExecutionManager to send updates back to the client
        via this specific WebSocket session.
        """
        # Ensure messages from G2 are tagged with their specific G2 thread ID
        await send_websocket_message_helper(
            websocket, event_type, data, session_id, "graph2_execution", graph2_thread_id_param
        )

        # Update the main session's general workflow status based on G2 events
        # Exclude per-tool events from this high-level status update
        if event_type not in ["tool_start", "tool_end"]:
            logger.info(f"[{session_id}] G2 Callback (G2_Thread: {graph2_thread_id_param}). Event: {event_type}. Store status before: {_session_bot_state_store.get('workflow_execution_status')}")
            new_status = _session_bot_state_store.get("workflow_execution_status", "idle")

            if event_type == "human_intervention_required":
                new_status = "paused_for_confirmation"
            elif event_type in ["execution_completed", "execution_failed", "workflow_timeout"]:
                new_status = "completed" if event_type == "execution_completed" else "failed"
                _session_bot_state_store["workflow_execution_results"] = data.get("final_state", {})
                # Clean up the specific G2 executor and definition when it's truly finished/failed
                if graph2_thread_id_param:
                    if graph2_thread_id_param in active_graph2_executors:
                        logger.info(f"[{session_id}] Workflow for G2 Thread ID {graph2_thread_id_param} ended. Removing its executor and definition.")
                        active_graph2_executors.pop(graph2_thread_id_param, None)
                        active_graph2_definitions.pop(graph2_thread_id_param, None)
                    else:
                        logger.warning(f"[{session_id}] G2 Thread ID {graph2_thread_id_param} ended but not found in active executors for cleanup.")


            _session_bot_state_store["workflow_execution_status"] = new_status
            logger.info(f"[{session_id}] G2 Callback. Store status after: {new_status}")
    # --- End of Graph 2 Callback ---

    try:
        while True:
            user_input_text = await websocket.receive_text()
            user_input_text = user_input_text.strip()
            if not user_input_text:
                continue
            logger.info(f"[{session_id}] RX: '{user_input_text[:100]}...'")

            # --- Initialize BotState for the current turn ---
            current_bot_state = _initialize_bot_state_for_turn(session_id, user_input_text, _session_bot_state_store)

            # --- Handle "resume_exec" Command ---
            if user_input_text.lower().startswith("resume_exec"):
                await _process_resume_command(
                    websocket, session_id, user_input_text,
                    _session_bot_state_store, active_graph2_executors
                )
                continue # Skip further processing for this turn

            # --- Run Graph 1 (Planning) ---
            current_bot_state = await _run_planning_graph(
                websocket, session_id, current_bot_state, langgraph_planning_app
            )

            # --- Persist BotState after Graph 1 ---
            _persist_bot_state_after_planning(_session_bot_state_store, current_bot_state)
            logger.debug(f"[{session_id}] Persisted BotState. Store status: {_session_bot_state_store.get('workflow_execution_status')}")


            # --- Initiate Graph 2 (Execution) if pending ---
            if _session_bot_state_store.get("workflow_execution_status") == "pending_start":
                current_graph2_thread_id_for_session = await _initiate_execution_graph(
                    websocket, session_id, _session_bot_state_store,
                    api_executor_instance, active_graph2_executors,
                    active_graph2_definitions, graph2_ws_callback_for_this_session # Pass the nested callback
                )
                if not current_graph2_thread_id_for_session: # If initiation failed
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
        # Clean up any G2 executor specifically associated with the *last* G2 run started by this G1 session
        if current_graph2_thread_id_for_session:
            logger.info(f"[{session_id}] Cleaning up executor and definition for last G2 thread: {current_graph2_thread_id_for_session}")
            active_graph2_executors.pop(current_graph2_thread_id_for_session, None)
            active_graph2_definitions.pop(current_graph2_thread_id_for_session, None)
        else:
            # Fallback: if no specific G2 ID was tracked for the very last interaction,
            # this might indicate a G1-only session or an issue before G2 started.
            # No specific G2 cleanup needed here unless a broader session-based cleanup is intended,
            # which is risky if G2 threads could outlive the G1 session that spawned them.
            logger.info(f"[{session_id}] No specific G2 thread ID tracked for final cleanup.")


        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.close(code=1000) # Normal closure
            except RuntimeError as e:
                logger.warning(f"[{session_id}] Error closing WebSocket (already closing?): {e}")
            except Exception as e:
                logger.error(f"[{session_id}] Exception during WebSocket close: {e}")
        logger.info(f"[{session_id}] WebSocket connection closed and resources cleaned up.")


# --- Helper Functions for WebSocket Logic ---

def _initialize_bot_state_for_turn(
    session_id: str, user_input: str, session_store: Dict[str, Any]
) -> BotState:
    """Creates and initializes a BotState instance for the current processing turn."""
    # Ensure execution_graph is loaded as Pydantic model if present in store
    exec_graph_data = session_store.get("execution_graph")
    execution_graph_model = None
    if isinstance(exec_graph_data, dict):
        try:
            execution_graph_model = PlanSchema.model_validate(exec_graph_data)
        except Exception as e:
            logger.error(f"[{session_id}] Failed to validate execution_graph from session_store: {e}")
            execution_graph_model = None # Proceed without it if validation fails
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
        input_is_spec=session_store.get("input_is_spec", False), # Will be determined by router
        workflow_execution_status=session_store.get("workflow_execution_status", "idle"),
        workflow_execution_results=session_store.get("workflow_execution_results", {}),
        workflow_extracted_data=session_store.get("workflow_extracted_data", {})
        # response, final_response, next_step, intent are transient for the turn
    )
    # Ensure scratchpad is a dict for the new turn
    state.scratchpad = {}
    return state

async def _process_resume_command(
    websocket: WebSocket, session_id: str, user_input_text: str,
    session_store: Dict[str, Any],
    active_graph2_executors: Dict[str, GraphExecutionManager]
):
    """Handles the 'resume_exec' command from the user."""
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

        # Check general status first, then specific manager
        if session_store.get("workflow_execution_status") == "paused_for_confirmation":
            if exec_manager:
                submitted = await exec_manager.submit_resume_data(graph2_thread_id_to_resume, resume_payload_for_graph2)
                if submitted:
                    await send_websocket_message_helper(
                        websocket, "info", {"message": "Resume data submitted to execution manager."},
                        session_id, "graph2_execution", graph2_thread_id_to_resume
                    )
                    session_store["workflow_execution_status"] = "running" # Update general status
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
    """Executes Graph 1 (Planning) and updates the BotState."""
    await send_websocket_message_helper(
        websocket, "status", {"message": "Planning workflow..."},
        session_id, "graph1_planning"
    )
    planning_config = {"configurable": {"thread_id": session_id}} # Graph 1 uses main session_id
    current_bot_state_after_planning = bot_state # Initialize with incoming state

    try:
        graph1_input_dict = current_bot_state_after_planning.model_dump(exclude_none=True)
        async for event in langgraph_planning_app.astream_events(graph1_input_dict, config=planning_config, version="v2"): # Use v2 events
            event_name = event["event"]
            event_data = event.get("data", {})
            node_name = event.get("name", "") # For on_tool_end etc.

            # logger.debug(f"[{session_id}] G1 Event: {event_name}, Node: {node_name}, DataKeys: {list(event_data.keys())}")

            if event_name == "on_tool_end" or event_name == "on_chat_model_end" or event_name == "on_chain_end": # More specific event types
                # The output of a node is usually in event["data"]["output"]
                # For StateGraph, this output is merged into the state.
                # The full state snapshot is usually available at on_graph_end or by calling get_state.
                # For intermediate updates, we can try to parse if 'output' is a BotState or dict.
                node_output_any = event_data.get("output")

                if isinstance(node_output_any, BotState):
                    current_bot_state_after_planning = node_output_any
                elif isinstance(node_output_any, dict):
                    try:
                        # Merge output dict into existing state. Be careful with Pydantic.
                        # Create a copy and update, or use model_copy(update=...)
                        updated_fields = {k: v for k, v in node_output_any.items() if hasattr(BotState, k)} # Only update valid BotState fields
                        current_bot_state_after_planning = current_bot_state_after_planning.model_copy(update=updated_fields)
                    except Exception as e_val:
                        logger.error(f"[{session_id}] Error updating BotState from G1 node output dict: {e_val}, dict: {node_output_any}")

                # Send intermediate response if the node generated one
                if current_bot_state_after_planning.response:
                    await send_websocket_message_helper(
                        websocket, "intermediate", {"message": current_bot_state_after_planning.response},
                        session_id, "graph1_planning"
                    )
                    current_bot_state_after_planning.response = None # Clear after sending

                # Send graph update if available in scratchpad
                if current_bot_state_after_planning.scratchpad and 'graph_to_send' in current_bot_state_after_planning.scratchpad:
                    graph_json_str = current_bot_state_after_planning.scratchpad.pop('graph_to_send', None)
                    if graph_json_str:
                        try:
                            graph_content = json.loads(graph_json_str) # Expecting a JSON string
                            await send_websocket_message_helper(
                                websocket, "graph_update", graph_content,
                                session_id, "graph1_planning"
                            )
                        except json.JSONDecodeError as e_json:
                            logger.error(f"[{session_id}] G1: 'graph_to_send' was not valid JSON: {e_json}. Content: {graph_json_str[:100]}")
                        except Exception as e_graph:
                            logger.error(f"[{session_id}] G1: Failed to send graph_update: {e_graph}")


            elif event_name == "on_graph_end":
                final_graph_output = event_data.get("output") # This should be the final BotState dict
                if isinstance(final_graph_output, dict):
                    try:
                        current_bot_state_after_planning = BotState.model_validate(final_graph_output)
                    except Exception as e_val:
                         logger.error(f"[{session_id}] Error validating final BotState from G1 output: {e_val}, dict: {final_graph_output}")
                elif isinstance(final_graph_output, BotState): # Should ideally be dict from LangGraph
                    current_bot_state_after_planning = final_graph_output
                else:
                    logger.error(f"[{session_id}] G1: on_graph_end output was not a dict or BotState: {t
