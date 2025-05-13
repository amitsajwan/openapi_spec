# main.py
import logging
import uuid
import json
import os
import sys
import asyncio
from typing import Any, Dict, Optional, Callable, Awaitable, Literal

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

from models import BotState, GraphOutput as PlanSchema, ExecutionGraphState
from graph import build_graph
from llm_config import initialize_llms
from execution_graph_definition import ExecutionGraphDefinition
from execution_manager import GraphExecutionManager
from api_executor import APIExecutor as UserAPIExecutor
from utils import SCHEMA_CACHE

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s - %(name)s - %(levelname)s - %(threadName)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

langgraph_planning_app: Optional[Any] = None
api_executor_instance: Optional[UserAPIExecutor] = None
active_graph2_executors: Dict[str, GraphExecutionManager] = {}

@app.on_event("startup")
async def startup_event():
    global langgraph_planning_app, api_executor_instance
    logger.info("FastAPI startup: Initializing LLMs, API Executor, and Planning Graph (Graph 1)...")
    try:
        router_llm_instance, worker_llm_instance = initialize_llms()
        api_executor_instance = UserAPIExecutor(
            base_url=os.getenv("DEFAULT_API_BASE_URL", "http://localhost:8080"), # Added a default
            timeout=float(os.getenv("API_TIMEOUT", 30.0))
        )
        langgraph_planning_app = build_graph(
            router_llm_instance, worker_llm_instance, api_executor_instance, None
        )
        logger.info("Main Planning LangGraph (Graph 1) built and compiled (NO CHECKPOINTER).")
    except Exception as e:
        logger.critical(f"CRITICAL FAILURE during startup: {e}", exc_info=True)
        langgraph_planning_app = None
        if api_executor_instance and hasattr(api_executor_instance, 'close'):
            if asyncio.iscoroutinefunction(getattr(api_executor_instance, 'close')):
                await api_executor_instance.close()
            else:
                api_executor_instance.close() # type: ignore
        api_executor_instance = None

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("FastAPI shutdown initiated.")
    if api_executor_instance and hasattr(api_executor_instance, 'close'):
        try:
            if asyncio.iscoroutinefunction(getattr(api_executor_instance, 'close')):
                await api_executor_instance.close()
            else:
                api_executor_instance.close() # type: ignore
            logger.info("User's APIExecutor client closed successfully.")
        except Exception as e:
            logger.error(f"Error closing User's APIExecutor client: {e}", exc_info=True)
    if SCHEMA_CACHE and hasattr(SCHEMA_CACHE, 'close'):
        try: SCHEMA_CACHE.close()
        except Exception as e: logger.error(f"Error closing schema cache: {e}", exc_info=True)
    logger.info("FastAPI shutdown complete.")

async def send_websocket_message(
    websocket: WebSocket, msg_type: str, content: Any, session_id: str,
    source_graph: Optional[Literal["system", "graph1_planning", "graph2_execution"]] = "graph1_planning",
    graph2_thread_id: Optional[str] = None
):
    try:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json({
                "type": msg_type, "source": source_graph, "content": content,
                "session_id": session_id, "graph2_thread_id": graph2_thread_id or session_id
            })
    except WebSocketDisconnect:
        logger.warning(f"[{session_id}] WS TX fail (Type: {msg_type}): Client disconnected.")
    except Exception as e:
        logger.error(f"[{session_id}] WS TX error (Type: {msg_type}): {e}", exc_info=False)

@app.websocket("/ws/openapi_agent")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    logger.info(f"WebSocket connection accepted. Session ID (Graph 1): {session_id}")

    if langgraph_planning_app is None or api_executor_instance is None:
        await send_websocket_message(websocket, "error", "Backend agent not initialized.", session_id, "system")
        await websocket.close(code=1011); return

    await send_websocket_message(websocket, "info", {"message": "Connection established."}, session_id, "system")
    _session_bot_state_store: Dict[str, Any] = {}

    async def graph2_ws_callback_with_state_update(
        event_type: str, data: Dict[str, Any], graph2_thread_id_param: Optional[str],
        bot_state_instance_ref: BotState # Pass the current BotState instance by reference
    ):
        # Send message to client first
        await send_websocket_message(websocket, event_type, data, session_id, "graph2_execution", graph2_thread_id_param)

        # Update BotState based on Graph 2 events
        logger.info(f"[{session_id}] Graph 2 Callback. Event: {event_type}. BotState status before: {bot_state_instance_ref.workflow_execution_status}")
        if event_type == "human_intervention_required":
            bot_state_instance_ref.workflow_execution_status = "paused_for_confirmation"
            # The 'data' here is expected to be {'node_name': ..., 'details_for_ui': ...}
            # No further update to workflow_execution_results needed here for this event.
        elif event_type in ["execution_completed", "execution_failed", "workflow_timeout"]:
            bot_state_instance_ref.workflow_execution_status = "completed" if event_type == "execution_completed" else "failed"
            bot_state_instance_ref.workflow_execution_results = data.get("final_state", {})
            if data.get("final_state", {}).get("error"):
                 bot_state_instance_ref.response = bot_state_instance_ref.response or str(data["final_state"]["error"])
            if event_type == "workflow_timeout":
                bot_state_instance_ref.response = data.get("message", "Workflow timed out.")
        
        logger.info(f"[{session_id}] Graph 2 Callback. BotState status after: {bot_state_instance_ref.workflow_execution_status}")
        # Persisting the updated bot_state_instance_ref to _session_bot_state_store will happen
        # at the end of the main WebSocket message loop.

    try:
        while True:
            user_input_text = await websocket.receive_text()
            user_input_text = user_input_text.strip()
            logger.info(f"[{session_id}] WS RX User Input: '{user_input_text[:150]}...'")
            if not user_input_text: continue

            current_bot_state = BotState(
                session_id=session_id, user_input=user_input_text,
                openapi_spec_text=_session_bot_state_store.get("openapi_spec_text"),
                openapi_schema=_session_bot_state_store.get("openapi_schema"),
                schema_cache_key=_session_bot_state_store.get("schema_cache_key"),
                schema_summary=_session_bot_state_store.get("schema_summary"),
                identified_apis=_session_bot_state_store.get("identified_apis", []),
                payload_descriptions=_session_bot_state_store.get("payload_descriptions", {}),
                execution_graph=PlanSchema.model_validate(_session_bot_state_store["execution_graph"]) if _session_bot_state_store.get("execution_graph") else None,
                plan_generation_goal=_session_bot_state_store.get("plan_generation_goal"),
                input_is_spec=_session_bot_state_store.get("input_is_spec", False),
                # Crucially load the workflow status from the store
                workflow_execution_status=_session_bot_state_store.get("workflow_execution_status", "idle"),
                workflow_execution_results=_session_bot_state_store.get("workflow_execution_results", {}),
                workflow_extracted_data=_session_bot_state_store.get("workflow_extracted_data", {})
            )
            current_bot_state.response = None; current_bot_state.final_response = ""
            current_bot_state.next_step = None; current_bot_state.intent = None
            if current_bot_state.scratchpad: current_bot_state.scratchpad.pop('graph_to_send', None)

            is_resume_for_graph2 = False
            resume_payload_for_graph2: Optional[Dict[str, Any]] = None
            if user_input_text.lower().startswith("resume_exec"):
                try:
                    payload_str = user_input_text.split("resume_exec", 1)[-1].strip()
                    if payload_str: resume_payload_for_graph2 = json.loads(payload_str); is_resume_for_graph2 = True
                    else: await send_websocket_message(websocket, "warning", "Resume command missing payload.", session_id, "system")
                except Exception as e_resume_parse:
                    logger.error(f"[{session_id}] Error parsing resume_exec: {e_resume_parse}")
                    await send_websocket_message(websocket, "error", f"Invalid resume command: {e_resume_parse}", session_id, "system")

            if is_resume_for_graph2 and resume_payload_for_graph2:
                exec_manager = active_graph2_executors.get(session_id)
                # Check the status from the *current_bot_state* which was loaded from the store
                if current_bot_state.workflow_execution_status == "paused_for_confirmation":
                    if exec_manager:
                        logger.info(f"[{session_id}] Submitting resume data to Graph 2 Execution Manager.")
                        submitted = await exec_manager.submit_resume_data(session_id, resume_payload_for_graph2)
                        if submitted:
                            await send_websocket_message(websocket, "info", "Resume data submitted. Workflow will attempt to continue.", session_id, "graph2_execution")
                            current_bot_state.workflow_execution_status = "running" # Update current and will be saved to store
                        else: await send_websocket_message(websocket, "error", "Failed to submit resume data.", session_id, "graph2_execution")
                    else: await send_websocket_message(websocket, "warning", "Cannot resume: No active execution manager.", session_id, "system")
                else:
                    status_msg = current_bot_state.workflow_execution_status
                    logger.warning(f"[{session_id}] Resume rejected. Status: '{status_msg}', expected 'paused_for_confirmation'.")
                    await send_websocket_message(websocket, "warning", f"Cannot resume: Workflow not paused (status: {status_msg}).", session_id, "system")
                # After handling resume, save state and continue to next user input
                _session_bot_state_store["workflow_execution_status"] = current_bot_state.workflow_execution_status
                continue

            # --- Graph 1 (Planning) Execution ---
            await send_websocket_message(websocket, "status", "Processing with Planning Graph...", session_id, "graph1_planning")
            planning_graph_config = {"configurable": {"thread_id": session_id}}
            try:
                graph1_input = current_bot_state.model_dump(exclude_none=True)
                async for event in langgraph_planning_app.astream_events(graph1_input, config=planning_graph_config, version="v1"):
                    if event["event"] == "on_chain_end" or event["event"] == "on_tool_end":
                        node_output = event["data"].get("output")
                        if isinstance(node_output, BotState): current_bot_state = node_output
                        elif isinstance(node_output, dict):
                            current_bot_state = BotState.model_validate({**current_bot_state.model_dump(), **node_output})
                        if current_bot_state.response:
                            await send_websocket_message(websocket, "intermediate", current_bot_state.response, session_id, "graph1_planning")
                        if current_bot_state.scratchpad and 'graph_to_send' in current_bot_state.scratchpad:
                            graph_json = current_bot_state.scratchpad.pop('graph_to_send', None)
                            if graph_json:
                                try:
                                    content_to_send = json.loads(graph_json) if isinstance(graph_json, str) else graph_json
                                    await send_websocket_message(websocket, "graph_update", content_to_send, session_id, "graph1_planning")
                                except Exception as e_graph: logger.error(f"[{session_id}] Failed to send graph_update: {e_graph}")
                    if event["event"] == "on_graph_end":
                        final_output = event["data"].get("output")
                        if isinstance(final_output, BotState): current_bot_state = final_output
                        elif isinstance(final_output, dict):
                             current_bot_state = BotState.model_validate({**current_bot_state.model_dump(), **final_output})
                        logger.info(f"[{session_id}] Planning Graph finished. Final response: '{str(current_bot_state.final_response)[:100]}...'")
                        break
                
                final_resp_g1 = current_bot_state.final_response or current_bot_state.response
                if final_resp_g1:
                    await send_websocket_message(websocket, "final", final_resp_g1, session_id, "graph1_planning")

            except Exception as e_plan:
                logger.critical(f"[{session_id}] Planning Graph error: {e_plan}", exc_info=True)
                await send_websocket_message(websocket, "error", f"Planning error: {e_plan}", session_id, "system")
                current_bot_state.workflow_execution_status = "failed" # Mark as failed
                # Fall through to save state

            # --- Persist BotState to _session_bot_state_store AFTER Graph 1 run ---
            if current_bot_state.openapi_schema:
                for key in ["openapi_spec_text", "openapi_schema", "schema_cache_key", "schema_summary", "identified_apis", "payload_descriptions"]:
                    _session_bot_state_store[key] = getattr(current_bot_state, key)
            else: # Clear schema related if no schema
                for key in ["openapi_spec_text", "openapi_schema", "schema_cache_key", "schema_summary", "identified_apis", "payload_descriptions"]:
                    _session_bot_state_store.pop(key, None)
            
            _session_bot_state_store["execution_graph"] = current_bot_state.execution_graph.model_dump(exclude_none=True) if current_bot_state.execution_graph else None
            _session_bot_state_store["plan_generation_goal"] = current_bot_state.plan_generation_goal
            _session_bot_state_store["input_is_spec"] = current_bot_state.input_is_spec
            # Persist workflow status which might have been updated by graph2_ws_callback or Graph 1
            _session_bot_state_store["workflow_execution_status"] = current_bot_state.workflow_execution_status
            _session_bot_state_store["workflow_execution_results"] = current_bot_state.workflow_execution_results
            _session_bot_state_store["workflow_extracted_data"] = current_bot_state.workflow_extracted_data
            logger.debug(f"[{session_id}] Persisted BotState to session store. Workflow status: {_session_bot_state_store.get('workflow_execution_status')}")


            # --- Graph 2 (Execution) Initiation ---
            if current_bot_state.workflow_execution_status == "pending_start":
                logger.info(f"[{session_id}] Planning Graph signaled 'pending_start'. Initiating Execution Graph (Graph 2).")
                exec_plan = current_bot_state.execution_graph # Already a PlanSchema instance or None
                if not exec_plan or not api_executor_instance:
                    err_msg = "Execution plan missing." if not exec_plan else "API executor not ready."
                    logger.error(f"[{session_id}] Cannot start Execution Graph: {err_msg}")
                    await send_websocket_message(websocket, "error", f"Cannot start execution: {err_msg}", session_id, "system")
                    current_bot_state.workflow_execution_status = "failed"
                    _session_bot_state_store["workflow_execution_status"] = "failed" # Persist failure
                else:
                    try:
                        exec_graph_def = ExecutionGraphDefinition(exec_plan, api_executor_instance)
                        runnable_exec_graph = exec_graph_def.get_runnable_graph()
                        
                        # Pass current_bot_state which will be updated by the callback
                        bound_graph2_callback = lambda et, d, g2id: graph2_ws_callback_with_state_update(et, d, g2id, current_bot_state)

                        exec_manager = GraphExecutionManager(
                            runnable_exec_graph, bound_graph2_callback, None, session_id
                        )
                        active_graph2_executors[session_id] = exec_manager
                        exec_initial_vals = ExecutionGraphState(
                            initial_input=current_bot_state.workflow_extracted_data or {},
                            # api_results, extracted_ids, confirmed_data will use default_factory
                        ).model_dump(exclude_none=True)
                        
                        graph2_thread_id = f"{session_id}_exec_{uuid.uuid4().hex[:6]}"
                        exec_graph_config = {"configurable": {"thread_id": graph2_thread_id}}
                        
                        logger.info(f"[{session_id}] ExecutionManager (Graph 2) created for ThreadID '{graph2_thread_id}'. Starting workflow.")
                        await send_websocket_message(websocket, "info", f"Execution phase started (ID: {graph2_thread_id}).", session_id, "graph2_execution", graph2_thread_id)
                        current_bot_state.workflow_execution_status = "running"
                        _session_bot_state_store["workflow_execution_status"] = "running" # Persist running status

                        asyncio.create_task(exec_manager.execute_workflow(exec_initial_vals, exec_graph_config))
                        # After create_task, the state of current_bot_state (especially workflow_execution_status)
                        # might be further updated by the callback. We need to ensure this latest state is saved.
                        # The save happens at the end of the loop, which is correct.
                    except Exception as e_exec_setup:
                        logger.error(f"[{session_id}] Failed to create/start Execution Graph (Graph 2): {e_exec_setup}", exc_info=True)
                        await send_websocket_message(websocket, "error", f"Failed to start execution: {e_exec_setup}", session_id, "system")
                        current_bot_state.workflow_execution_status = "failed"
                        _session_bot_state_store["workflow_execution_status"] = "failed" # Persist failure
            
            # Ensure the very latest state of current_bot_state (potentially modified by callbacks or Graph 2 setup)
            # is saved to the session store before the loop waits for the next user message.
            # This is a repeat of the save block but ensures status from G2 setup is captured.
            _session_bot_state_store["workflow_execution_status"] = current_bot_state.workflow_execution_status
            # Other fields like execution_graph, plan_generation_goal etc., are typically set by Graph 1,
            # so the save after Graph 1 run should be sufficient for those. The most dynamic is workflow_status.


    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}.")
    except Exception as e_outer_loop:
        logger.critical(f"[{session_id}] Unhandled error in WebSocket main loop: {e_outer_loop}", exc_info=True)
        if websocket.client_state == WebSocketState.CONNECTED:
            try: await send_websocket_message(websocket, "error", "Critical server error.", session_id, "system")
            except Exception: pass
    finally:
        logger.info(f"[{session_id}] Cleaning up WebSocket connection resources.")
        if session_id in active_graph2_executors: del active_graph2_executors[session_id]
        if websocket.client_state == WebSocketState.CONNECTED:
            try: await websocket.close()
            except Exception: pass

@app.get("/", response_class=FileResponse)
async def get_index_page():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse("Frontend index.html not found in ./static directory.", status_code=404)
    return FileResponse(index_path)
