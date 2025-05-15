# websocket_helpers.py
import logging
import uuid
import json
import asyncio
from typing import Any, Dict, Optional, Callable, Awaitable

from fastapi import WebSocket, WebSocketDisconnect

# Assuming models are correctly imported from models.py
from models import BotState, GraphOutput as PlanSchema, ExecutionGraphState, APIExecutor
# Assuming ExecutionGraphDefinition and GraphExecutionManager are correctly imported
from execution_graph_definition import ExecutionGraphDefinition
from execution_manager import GraphExecutionManager
# Import the new load test orchestrator
from load_test_orchestrator import execute_load_test

logger = logging.getLogger(__name__)

# --- WebSocket Message Sending Helper ---
async def send_websocket_message_helper(
    websocket: WebSocket,
    msg_type: str,
    content: Any,
    session_id: str, # This is the G1 session ID
    source_graph: str = "graph1_planning", # Default source
    graph2_thread_id: Optional[str] = None # Specific G2 thread ID if applicable
):
    """
    Sends a JSON payload to the connected WebSocket client.
    """
    try:
        # Attempt to serialize content if it's not directly JSON serializable
        try:
            json.dumps(content) # Check serializability
            payload_to_send = content
        except TypeError:
            logger.warning(f"Content for WS type {msg_type} not directly JSON serializable, attempting str(). Snippet: {str(content)[:200]}")
            payload_to_send = {"raw_content": str(content), "original_type": str(type(content))}

        await websocket.send_json({
            "type": msg_type,
            "source": source_graph,
            "content": payload_to_send,
            "session_id": session_id, # G1 session_id
            "graph2_thread_id": graph2_thread_id # G2 thread_id if message is from/for a G2 instance
        })
    except WebSocketDisconnect:
        logger.warning(f"[{session_id}] WebSocket disconnected while trying to send message: {msg_type}")
    except RuntimeError as e: # Handles cases like "Cannot call send while not connected."
        client_state_info = "unknown"
        if hasattr(websocket, 'client_state'): # Check if client_state attribute exists
            client_state_info = str(websocket.client_state)
        logger.warning(f"[{session_id}] RuntimeError sending message type {msg_type}: {e}. WebSocket state: {client_state_info}")
    except Exception as e:
        logger.error(f"[{session_id}] Unexpected error sending message type {msg_type}: {e}", exc_info=True)


# --- Helper Functions for WebSocket Logic ---

def _initialize_bot_state_for_turn(
    session_id: str, user_input: str, session_store: Dict[str, Any]
) -> BotState:
    """Initializes or retrieves BotState for the current turn."""
    exec_graph_data = session_store.get("execution_graph")
    execution_graph_model = None
    if isinstance(exec_graph_data, dict):
        try:
            execution_graph_model = PlanSchema.model_validate(exec_graph_data)
        except Exception as e:
            logger.error(f"[{session_id}] Failed to validate execution_graph from session_store: {e}")
            execution_graph_model = None # Ensure it's None if validation fails
    elif isinstance(exec_graph_data, PlanSchema): # If it's already a model instance
        execution_graph_model = exec_graph_data

    scratchpad_data = session_store.get("scratchpad", {})
    if not isinstance(scratchpad_data, dict):
        logger.warning(f"[{session_id}] Scratchpad from session_store was not a dict, reinitializing. Type was: {type(scratchpad_data)}")
        scratchpad_data = {}

    # Ensure extracted_params is also carried over or initialized
    extracted_params_data = session_store.get("extracted_params", {})
    if not isinstance(extracted_params_data, dict):
        logger.warning(f"[{session_id}] extracted_params from session_store was not a dict, reinitializing.")
        extracted_params_data = {}


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
        scratchpad=scratchpad_data,
        extracted_params=extracted_params_data # Use the (potentially reinitialized) extracted_params
    )
    return state

async def _process_resume_command(
    websocket: WebSocket, session_id: str, user_input_text: str,
    session_store: Dict[str, Any], # G1 session store
    active_graph2_executors: Dict[str, GraphExecutionManager] # For single G2 instances
):
    """Processes the 'resume_exec' command from the user."""
    try:
        parts = user_input_text.split(" ", 2)
        if len(parts) < 3:
            await send_websocket_message_helper(
                websocket, "warning", {"message": "Resume command format: resume_exec <g2_thread_id> <json_payload>"},
                session_id, "system_warning", None
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

        # This resume command is intended for the single G2 instance managed by this G1 session
        exec_manager = active_graph2_executors.get(graph2_thread_id_to_resume)

        # Check the G1 session's view of the single G2 instance's status
        if session_store.get("workflow_execution_status") == "paused_for_confirmation":
            if exec_manager:
                if not isinstance(resume_payload_for_graph2, dict):
                    await send_websocket_message_helper(websocket, "error", {"error": "Resume payload must be a JSON object."}, session_id, "system_error", graph2_thread_id_to_resume)
                    return

                submitted = await exec_manager.submit_resume_data(graph2_thread_id_to_resume, resume_payload_for_graph2)
                if submitted:
                    await send_websocket_message_helper(websocket, "info", {"message": "Resume data submitted to execution manager."}, session_id, "graph2_execution", graph2_thread_id_to_resume)
                    session_store["workflow_execution_status"] = "running" # Update G1's view
                else:
                    await send_websocket_message_helper(websocket, "error", {"error": "Failed to submit resume data to the manager (manager rejected)."}, session_id, "graph2_execution", graph2_thread_id_to_resume)
            else:
                await send_websocket_message_helper(websocket, "warning", {"message": f"Cannot resume: No active execution manager found for G2 Thread ID '{graph2_thread_id_to_resume}'. It might have already finished or failed."}, session_id, "system_warning", None)
        else:
            status_msg = session_store.get("workflow_execution_status", "unknown")
            await send_websocket_message_helper(websocket, "warning", {"message": f"Cannot resume: Workflow is not currently paused for confirmation (current status: {status_msg})."}, session_id, "system_warning", None)
    except json.JSONDecodeError:
        await send_websocket_message_helper(websocket, "error", {"error": "Invalid JSON payload provided for resume_exec command."}, session_id, "system_error", None)
    except Exception as e_resume:
        logger.error(f"[{session_id}] Error processing 'resume_exec' command: {e_resume}", exc_info=True)
        await send_websocket_message_helper(websocket, "error", {"error": f"Error processing resume command: {str(e_resume)}"}, session_id, "system_error", None)


async def _run_planning_graph(
    websocket: WebSocket, session_id: str, bot_state: BotState, langgraph_planning_app: Any
) -> BotState:
    """Runs the Graph 1 (Planning) and streams events back to the client."""
    await send_websocket_message_helper(
        websocket, "status", {"message": "Processing request with Planning Workflow..."},
        session_id, "graph1_planning", None
    )
    planning_config = {"configurable": {"thread_id": session_id}}
    current_bot_state_after_planning = bot_state # Start with the input state
    last_intermediate_message_sent: Optional[str] = None

    try:
        graph1_input_dict = current_bot_state_after_planning.model_dump(exclude_none=True)
        if 'scratchpad' not in graph1_input_dict: # Ensure scratchpad is included
             graph1_input_dict['scratchpad'] = current_bot_state_after_planning.scratchpad
        if 'extracted_params' not in graph1_input_dict: # Ensure extracted_params is included
            graph1_input_dict['extracted_params'] = current_bot_state_after_planning.extracted_params


        async for event in langgraph_planning_app.astream_events(graph1_input_dict, config=planning_config, version="v2"): # Use v2 for BotState
            event_name = event["event"]
            event_data = event.get("data", {})

            if event_name in ["on_tool_end", "on_chat_model_end", "on_chain_end"]: # Events that might output a BotState
                node_output_any = event_data.get("output")

                if isinstance(node_output_any, BotState):
                    current_bot_state_after_planning = node_output_any
                elif isinstance(node_output_any, dict): # If a node returns a dict, try to update BotState
                    try:
                        valid_fields_to_update = {k: v for k, v in node_output_any.items() if hasattr(BotState, k)}
                        current_bot_state_after_planning = current_bot_state_after_planning.model_copy(update=valid_fields_to_update)
                        # Merge scratchpad if returned
                        if 'scratchpad' in node_output_any and isinstance(node_output_any['scratchpad'], dict):
                            current_bot_state_after_planning.scratchpad.update(node_output_any['scratchpad'])
                        if 'extracted_params' in node_output_any and isinstance(node_output_any['extracted_params'], dict):
                            current_bot_state_after_planning.extracted_params.update(node_output_any['extracted_params'])
                    except Exception as e_val:
                        logger.error(f"[{session_id}] Error updating BotState from G1 node output dict: {e_val}, dict: {str(node_output_any)[:500]}")

                # Send intermediate messages from scratchpad
                if current_bot_state_after_planning.scratchpad and 'intermediate_messages' in current_bot_state_after_planning.scratchpad:
                    message_queue = current_bot_state_after_planning.scratchpad.pop('intermediate_messages', [])
                    for queued_msg_content in message_queue:
                        msg_to_send_str = ""
                        if isinstance(queued_msg_content, str): msg_to_send_str = queued_msg_content
                        elif isinstance(queued_msg_content, dict) and queued_msg_content.get("message"): msg_to_send_str = queued_msg_content.get("message")
                        if msg_to_send_str and msg_to_send_str != last_intermediate_message_sent:
                            await send_websocket_message_helper(websocket, "intermediate", {"message": msg_to_send_str}, session_id, "graph1_planning", None)
                            last_intermediate_message_sent = msg_to_send_str
                
                # Send primary response if different (some nodes set state.response directly)
                if current_bot_state_after_planning.response and current_bot_state_after_planning.response != last_intermediate_message_sent:
                    await send_websocket_message_helper(websocket, "intermediate", {"message": current_bot_state_after_planning.response}, session_id, "graph1_planning", None)
                    last_intermediate_message_sent = current_bot_state_after_planning.response
                    current_bot_state_after_planning.response = None # Clear after sending

                # Send graph updates from scratchpad
                if current_bot_state_after_planning.scratchpad and 'graph_to_send' in current_bot_state_after_planning.scratchpad:
                    graph_json_str = current_bot_state_after_planning.scratchpad.pop('graph_to_send', None)
                    if graph_json_str:
                        try:
                            graph_content = json.loads(graph_json_str)
                            await send_websocket_message_helper(websocket, "graph_update", graph_content, session_id, "graph1_planning", None)
                        except Exception as e_graph: logger.error(f"[{session_id}] G1: Failed to send graph_update: {e_graph}")
                
                # Send API responses from scratchpad (if Graph 1 makes API calls and stores results)
                if current_bot_state_after_planning.scratchpad and 'last_api_response' in current_bot_state_after_planning.scratchpad:
                    api_response_data = current_bot_state_after_planning.scratchpad.pop('last_api_response', None)
                    if api_response_data and isinstance(api_response_data, dict):
                        await send_websocket_message_helper(websocket, "api_response", api_response_data, session_id, "graph1_planning", None)


            elif event_name == "on_graph_end": # End of the entire Graph 1 execution
                final_graph_output = event_data.get("output")
                if isinstance(final_graph_output, dict): # Final output should be a BotState dict
                    try: current_bot_state_after_planning = BotState.model_validate(final_graph_output)
                    except Exception as e_val: logger.error(f"[{session_id}] Error validating final BotState from G1 output: {e_val}, dict: {str(final_graph_output)[:500]}")
                elif isinstance(final_graph_output, BotState): # Or a BotState object
                    current_bot_state_after_planning = final_graph_output
                else:
                    logger.error(f"[{session_id}] G1: on_graph_end output was not a dict or BotState: {type(final_graph_output)}")
                
                final_msg_content = current_bot_state_after_planning.final_response or current_bot_state_after_planning.response
                if not final_msg_content and not (current_bot_state_after_planning.workflow_execution_status in ["pending_start", "pending_load_test"]):
                    final_msg_content = "Planning phase complete." # Generic completion
                    logger.info(f"[{session_id}] G1: No specific final_response from graph, sending generic: '{final_msg_content}'")

                if final_msg_content:
                    final_payload = {"message": final_msg_content}
                    if current_bot_state_after_planning.execution_graph: # Include graph if available
                        try: final_payload["graph"] = current_bot_state_after_planning.execution_graph.model_dump(exclude_none=True)
                        except Exception as e_dump: logger.error(f"[{session_id}] G1: Error dumping execution_graph for final response: {e_dump}")
                    await send_websocket_message_helper(websocket, "final", final_payload, session_id, "graph1_planning", None)
                
                current_bot_state_after_planning.final_response = "" # Clear after sending
                current_bot_state_after_planning.response = None
                break # Exit the astream_events loop for Graph 1

    except Exception as e_plan:
        logger.critical(f"[{session_id}] Planning Graph (Graph 1) execution error: {e_plan}", exc_info=True)
        await send_websocket_message_helper(
            websocket, "error", {"error": f"Error during planning phase: {str(e_plan)[:200]}"},
            session_id, "system_error", None
        )
        current_bot_state_after_planning.workflow_execution_status = "failed"

    return current_bot_state_after_planning


def _persist_bot_state_after_planning(session_store: Dict[str, Any], bot_state: BotState):
    """Persists relevant fields from BotState to the session_store."""
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
    session_store["scratchpad"] = bot_state.scratchpad # Persist the entire scratchpad
    session_store["extracted_params"] = bot_state.extracted_params # Persist extracted params


async def _initiate_execution_graph(
    websocket: WebSocket,
    session_id: str, # G1 session ID
    session_store: Dict[str, Any],
    api_executor_instance: APIExecutor,
    active_graph2_executors: Dict[str, GraphExecutionManager], # For single G2 instance
    active_graph2_definitions: Dict[str, ExecutionGraphDefinition], # For single G2 instance
    graph2_ws_callback: Callable[[str, Dict[str, Any], Optional[str]], Awaitable[None]], # Unified callback
    disable_confirmation_prompts: bool = False # Parameter to control prompts
) -> Optional[str]: # Returns the G2 thread ID if successful
    """Initiates a single Graph 2 execution."""
    logger.info(f"[{session_id}] Initiating single Graph 2 execution. Confirmations disabled: {disable_confirmation_prompts}")
    exec_plan_dict = session_store.get("execution_graph")
    exec_plan_model = None
    if isinstance(exec_plan_dict, dict):
        try:
            exec_plan_model = PlanSchema.model_validate(exec_plan_dict)
        except Exception as e_val:
            logger.error(f"[{session_id}] Failed to validate execution_graph for G2: {e_val}. Data: {str(exec_plan_dict)[:200]}")
            await send_websocket_message_helper(websocket, "error", {"error": "Execution plan is invalid for G2."}, session_id, "system_error", None)
            return None
    else:
        logger.warning(f"[{session_id}] G2 initiation skipped: No execution plan dictionary. Type: {type(exec_plan_dict)}")
        await send_websocket_message_helper(websocket, "warning", {"message": "Cannot start execution: No valid plan for G2."}, session_id, "system_warning", None)
        return None

    if not api_executor_instance:
        logger.error(f"[{session_id}] G2 initiation failed: APIExecutor not available.")
        await send_websocket_message_helper(websocket, "error", {"error": "API executor service NA for G2."}, session_id, "system_error", None)
        return None

    # Generate a unique thread_id for this specific Graph 2 run
    graph2_thread_id = f"{session_id}_exec_{uuid.uuid4().hex[:8]}"
    logger.info(f"[{session_id}] Generated new G2 Thread ID for single execution: {graph2_thread_id}")

    try:
        exec_graph_def = ExecutionGraphDefinition(
            graph_execution_plan=exec_plan_model,
            api_executor=api_executor_instance,
            disable_confirmation_prompts=disable_confirmation_prompts # Pass the flag
        )
        active_graph2_definitions[graph2_thread_id] = exec_graph_def # Store for potential resume/debug
        runnable_exec_graph = exec_graph_def.get_runnable_graph()

        exec_manager = GraphExecutionManager(
            runnable_graph=runnable_exec_graph,
            graph_definition=exec_graph_def,
            websocket_callback=graph2_ws_callback, # Use the unified callback
            planning_checkpointer=None,
            main_planning_session_id=session_id # Link back to G1 session
        )
        active_graph2_executors[graph2_thread_id] = exec_manager # Store for resume

        initial_exec_state_values = ExecutionGraphState(
            initial_input=session_store.get("workflow_extracted_data", {}),
        ).model_dump(exclude_none=True)
        exec_graph_config = {"configurable": {"thread_id": graph2_thread_id}}

        await send_websocket_message_helper(
            websocket, "info",
            {"message": f"Execution phase (Workflow Graph) starting with G2 Thread ID: {graph2_thread_id}. Prompts disabled: {disable_confirmation_prompts}."},
            session_id, "graph2_execution", graph2_thread_id
        )
        # Update G1's view of its associated single G2 instance
        session_store["workflow_execution_status"] = "running"

        # Start Graph 2 execution as a background task
        asyncio.create_task(exec_manager.execute_workflow(initial_exec_state_values, exec_graph_config))
        return graph2_thread_id # Return the G2 thread ID

    except Exception as e_exec_setup:
        logger.error(f"[{session_id}] Failed to set up and start Graph 2 (G2 Thread ID: {graph2_thread_id}): {e_exec_setup}", exc_info=True)
        await send_websocket_message_helper(websocket, "error", {"error": f"Server error during G2 setup: {str(e_exec_setup)[:150]}"}, session_id, "system_error", graph2_thread_id)
        active_graph2_executors.pop(graph2_thread_id, None) # Clean up on failure
        active_graph2_definitions.pop(graph2_thread_id, None)
        return None


# --- Main WebSocket Connection Handler ---
async def handle_websocket_connection(
    websocket: WebSocket,
    session_id: str, # This is the G1 session ID
    langgraph_planning_app: Any,
    api_executor_instance: APIExecutor,
    active_graph2_executors: Dict[str, GraphExecutionManager], # Tracks single G2 instances
    active_graph2_definitions: Dict[str, ExecutionGraphDefinition] # Tracks single G2 defs
):
    await send_websocket_message_helper(
        websocket, "info", {"message": "Connection established."},
        session_id, "system", None
    )

    _session_bot_state_store: Dict[str, Any] = {"workflow_execution_status": "idle"}
    current_single_g2_thread_id: Optional[str] = None # Tracks the G2 thread_id for the *single* G2 run

    # Unified callback for messages from any Graph 2 instance (single or load test worker)
    async def graph2_instance_ws_callback(
        event_type: str,
        data: Dict[str, Any],
        g2_thread_id_param: Optional[str] # The specific G2 thread ID that generated this event
    ):
        is_load_test_worker = g2_thread_id_param and "loadtest_" in g2_thread_id_param

        # Filter noisy messages for load test workers
        if is_load_test_worker and event_type in ["tool_start", "llm_start", "llm_stream", "llm_end", "tool_end"]:
            # logger.debug(f"[{session_id}] LoadTestWorker G2 Event (G2_Thread: {g2_thread_id_param}): {event_type} - Skipped sending to client.")
            pass # Optionally, log these to server console only or a specific load test log
        else:
            await send_websocket_message_helper(
                websocket, event_type, data, session_id, "graph2_execution", g2_thread_id_param
            )

        # Update _session_bot_state_store ONLY for the single G2 instance associated with this G1 session
        if not is_load_test_worker and g2_thread_id_param == current_single_g2_thread_id:
            if event_type not in ["tool_start", "tool_end", "llm_start", "llm_stream", "llm_end"]: # Avoid logging too frequently
                logger.info(f"[{session_id}] Single G2 Callback (G2_Thread: {g2_thread_id_param}). Event: {event_type}.")
                new_status = _session_bot_state_store.get("workflow_execution_status", "idle")

                if event_type == "human_intervention_required":
                    new_status = "paused_for_confirmation"
                elif event_type in ["execution_completed", "execution_failed", "workflow_timeout"]:
                    new_status = "completed" if event_type == "execution_completed" else "failed"
                    _session_bot_state_store["workflow_execution_results"] = data.get("final_state", {})
                    if g2_thread_id_param: # Cleanup for this single G2 run
                        logger.info(f"[{session_id}] Single G2 run for {g2_thread_id_param} ended. Removing its executor and definition.")
                        active_graph2_executors.pop(g2_thread_id_param, None)
                        active_graph2_definitions.pop(g2_thread_id_param, None)
                        current_single_g2_thread_id = None # Clear the tracked ID for this G1 session

                _session_bot_state_store["workflow_execution_status"] = new_status
                logger.info(f"[{session_id}] Single G2. Session store status after: {new_status}")

    try:
        while True:
            user_input_text = await websocket.receive_text()
            user_input_text = user_input_text.strip()
            if not user_input_text: continue
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

            workflow_status_from_g1 = _session_bot_state_store.get("workflow_execution_status")

            if workflow_status_from_g1 == "pending_start":
                if current_single_g2_thread_id and current_single_g2_thread_id in active_graph2_executors:
                    logger.warning(f"[{session_id}] A single G2 thread {current_single_g2_thread_id} is already active. Not starting new one for 'pending_start'.")
                    await send_websocket_message_helper(websocket, "warning", {"message": f"Workflow {current_single_g2_thread_id} is already running or paused."}, session_id, "system_warning", None)
                else:
                    current_single_g2_thread_id = await _initiate_execution_graph(
                        websocket, session_id, _session_bot_state_store,
                        api_executor_instance, active_graph2_executors,
                        active_graph2_definitions, graph2_instance_ws_callback,
                        disable_confirmation_prompts=False # Normal single run
                    )
                    if not current_single_g2_thread_id:
                         _session_bot_state_store["workflow_execution_status"] = "failed"

            elif workflow_status_from_g1 == "pending_load_test":
                await execute_load_test( # Call the refactored orchestrator
                    websocket=websocket,
                    main_session_id=session_id,
                    session_store=_session_bot_state_store,
                    api_executor_instance=api_executor_instance,
                    graph2_instance_callback=graph2_instance_ws_callback,
                    send_overall_status_callback=send_websocket_message_helper # Pass the helper directly
                )
                _session_bot_state_store["workflow_execution_status"] = "idle" # Reset G1 status after load test initiated

    except WebSocketDisconnect:
        logger.info(f"[{session_id}] WebSocket disconnected by client.")
    except Exception as e_outer_loop:
        logger.critical(f"[{session_id}] Unhandled error in WebSocket loop: {e_outer_loop}", exc_info=True)
        await send_websocket_message_helper(
            websocket, "error", {"error": "An unexpected server error occurred in the main loop."},
            session_id, "system_critical", None
        )
    finally:
        logger.info(f"[{session_id}] Cleaning up WebSocket connection resources.")
        if current_single_g2_thread_id and current_single_g2_thread_id in active_graph2_executors:
            logger.info(f"[{session_id}] Cleaning up active G2 executor for single run: {current_single_g2_thread_id}")
            active_graph2_executors.pop(current_single_g2_thread_id, None)
            active_graph2_definitions.pop(current_single_g2_thread_id, None)
        # ... (rest of your finally block for closing websocket) ...
        current_ws_state = "unknown"
        if hasattr(websocket, 'client_state'): current_ws_state = str(websocket.client_state)
        if "WebSocketState.CONNECTED" in current_ws_state:
            try: await websocket.close(code=1000)
            except Exception as e_close: logger.warning(f"[{session_id}] Error closing WebSocket: {e_close}")
        logger.info(f"[{session_id}] WebSocket connection closed for G1 session.")

