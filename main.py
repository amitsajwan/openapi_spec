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
from execution_graph_definition import ExecutionGraphDefinition # For type hint
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
if not os.path.exists(STATIC_DIR): os.makedirs(STATIC_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

langgraph_planning_app: Optional[Any] = None
api_executor_instance: Optional[UserAPIExecutor] = None
active_graph2_executors: Dict[str, GraphExecutionManager] = {}
# Store graph definitions to allow manager to access node details
active_graph2_definitions: Dict[str, ExecutionGraphDefinition] = {}


@app.on_event("startup")
async def startup_event():
    global langgraph_planning_app, api_executor_instance
    logger.info("FastAPI startup...")
    try:
        router_llm, worker_llm = initialize_llms()
        api_executor_instance = UserAPIExecutor(
            base_url=os.getenv("DEFAULT_API_BASE_URL", "http://localhost:8080"),
            timeout=float(os.getenv("API_TIMEOUT", 30.0))
        )
        langgraph_planning_app = build_graph(router_llm, worker_llm, api_executor_instance, None)
        logger.info("Main Planning LangGraph (Graph 1) built.")
    except Exception as e:
        logger.critical(f"CRITICAL FAILURE during startup: {e}", exc_info=True)
        # Handle cleanup if necessary
        langgraph_planning_app = None


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("FastAPI shutdown.")
    if api_executor_instance and hasattr(api_executor_instance, 'close'):
        if asyncio.iscoroutinefunction(getattr(api_executor_instance, 'close')): await api_executor_instance.close()
        else: api_executor_instance.close() # type: ignore
    if SCHEMA_CACHE: SCHEMA_CACHE.close()


async def send_websocket_message(
    websocket: WebSocket, msg_type: str, content: Any, session_id: str,
    source_graph: str = "graph1_planning", graph2_thread_id: Optional[str] = None
):
    try:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json({
                "type": msg_type, "source": source_graph, "content": content,
                "session_id": session_id, "graph2_thread_id": graph2_thread_id or session_id
            })
    except Exception as e:
        logger.error(f"[{session_id}] WS TX error (Type: {msg_type}): {e}", exc_info=False)


@app.websocket("/ws/openapi_agent")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    logger.info(f"WebSocket accepted. Session ID (Graph 1): {session_id}")

    if not langgraph_planning_app or not api_executor_instance:
        await send_websocket_message(websocket, "error", "Backend not initialized.", session_id, "system")
        await websocket.close(code=1011); return

    await send_websocket_message(websocket, "info", {"message": "Connection established."}, session_id, "system")
    _session_bot_state_store: Dict[str, Any] = {"workflow_execution_status": "idle"}

    async def graph2_ws_callback(event_type: str, data: Dict[str, Any], graph2_thread_id_param: Optional[str]):
        await send_websocket_message(websocket, event_type, data, session_id, "graph2_execution", graph2_thread_id_param)
        
        logger.info(f"[{session_id}] G2 Callback. Event: {event_type}. Store status before: {_session_bot_state_store.get('workflow_execution_status')}")
        new_status = _session_bot_state_store.get("workflow_execution_status", "idle")
        if event_type == "human_intervention_required":
            new_status = "paused_for_confirmation"
        elif event_type in ["execution_completed", "execution_failed", "workflow_timeout"]:
            new_status = "completed" if event_type == "execution_completed" else "failed"
            _session_bot_state_store["workflow_execution_results"] = data.get("final_state", {})
        
        _session_bot_state_store["workflow_execution_status"] = new_status
        logger.info(f"[{session_id}] G2 Callback. Store status after: {new_status}")

    try:
        while True:
            user_input_text = await websocket.receive_text()
            user_input_text = user_input_text.strip()
            if not user_input_text: continue
            logger.info(f"[{session_id}] RX: '{user_input_text[:100]}...'")

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
                workflow_execution_status=_session_bot_state_store.get("workflow_execution_status", "idle"),
                workflow_execution_results=_session_bot_state_store.get("workflow_execution_results", {}),
                workflow_extracted_data=_session_bot_state_store.get("workflow_extracted_data", {})
            )
            # Reset transient fields
            current_bot_state.response = None; current_bot_state.final_response = ""
            current_bot_state.next_step = None; current_bot_state.intent = None


            if user_input_text.lower().startswith("resume_exec"):
                # ... (resume logic remains largely the same, checking _session_bot_state_store status) ...
                try:
                    payload_str = user_input_text.split("resume_exec", 1)[-1].strip()
                    resume_payload_for_graph2 = json.loads(payload_str) if payload_str else None
                    if not resume_payload_for_graph2:
                        await send_websocket_message(websocket, "warning", "Resume command missing payload.", session_id, "system"); continue
                    
                    exec_manager = active_graph2_executors.get(session_id)
                    if _session_bot_state_store.get("workflow_execution_status") == "paused_for_confirmation":
                        if exec_manager:
                            submitted = await exec_manager.submit_resume_data(session_id, resume_payload_for_graph2)
                            if submitted:
                                await send_websocket_message(websocket, "info", "Resume data submitted.", session_id, "graph2_execution")
                                _session_bot_state_store["workflow_execution_status"] = "running" 
                            else: await send_websocket_message(websocket, "error", "Failed to submit resume data.", session_id, "graph2_execution")
                        else: await send_websocket_message(websocket, "warning", "Cannot resume: No active exec manager.", session_id, "system")
                    else:
                        status_msg = _session_bot_state_store.get("workflow_execution_status", "unknown")
                        await send_websocket_message(websocket, "warning", f"Cannot resume: Workflow not paused (status: {status_msg}).", session_id, "system")
                except Exception as e_resume:
                    logger.error(f"[{session_id}] Error parsing/processing resume_exec: {e_resume}")
                    await send_websocket_message(websocket, "error", f"Invalid resume: {e_resume}", session_id, "system")
                continue


            # --- Graph 1 (Planning) ---
            await send_websocket_message(websocket, "status", "Planning...", session_id, "graph1_planning")
            planning_config = {"configurable": {"thread_id": session_id}}
            try:
                graph1_input = current_bot_state.model_dump(exclude_none=True)
                async for event in langgraph_planning_app.astream_events(graph1_input, config=planning_config, version="v1"):
                    # ... (event handling for Graph 1 remains similar) ...
                    if event["event"] == "on_chain_end" or event["event"] == "on_tool_end":
                        node_output = event["data"].get("output")
                        if isinstance(node_output, BotState): current_bot_state = node_output
                        elif isinstance(node_output, dict):
                            current_bot_state = BotState.model_validate({**current_bot_state.model_dump(exclude_none=True), **node_output})
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
                             current_bot_state = BotState.model_validate({**current_bot_state.model_dump(exclude_none=True), **final_output})
                        break
                final_resp_g1 = current_bot_state.final_response or current_bot_state.response
                if final_resp_g1: await send_websocket_message(websocket, "final", final_resp_g1, session_id, "graph1_planning")
            except Exception as e_plan:
                logger.critical(f"[{session_id}] Planning Graph error: {e_plan}", exc_info=True)
                await send_websocket_message(websocket, "error", f"Planning error: {str(e_plan)[:150]}", session_id, "system")
                current_bot_state.workflow_execution_status = "failed"
            
            # --- Persist BotState to Store (after Graph 1, before Graph 2 init) ---
            store_keys = ["openapi_spec_text", "openapi_schema", "schema_cache_key", "schema_summary", "identified_apis", "payload_descriptions", "plan_generation_goal", "input_is_spec", "workflow_execution_status", "workflow_execution_results", "workflow_extracted_data"]
            for key in store_keys:
                if hasattr(current_bot_state, key):
                    value = getattr(current_bot_state, key)
                    if key == "execution_graph" and value:
                        _session_bot_state_store[key] = value.model_dump(exclude_none=True)
                    elif value is not None or key == "workflow_execution_status": # Always save status
                         _session_bot_state_store[key] = value
                    else:
                        _session_bot_state_store.pop(key, None)
            logger.debug(f"[{session_id}] Persisted BotState. Store status: {_session_bot_state_store.get('workflow_execution_status')}")


            # --- Graph 2 (Execution) Initiation ---
            if _session_bot_state_store.get("workflow_execution_status") == "pending_start":
                logger.info(f"[{session_id}] Store signals 'pending_start'. Initiating Graph 2.")
                exec_plan_dict = _session_bot_state_store.get("execution_graph")
                exec_plan = PlanSchema.model_validate(exec_plan_dict) if exec_plan_dict else None

                if exec_plan and api_executor_instance:
                    try:
                        exec_graph_def = ExecutionGraphDefinition(exec_plan, api_executor_instance)
                        active_graph2_definitions[session_id] = exec_graph_def # Store definition
                        runnable_exec_graph = exec_graph_def.get_runnable_graph()
                        
                        exec_manager = GraphExecutionManager(
                            runnable_graph=runnable_exec_graph,
                            graph_definition=exec_graph_def, # Pass definition to manager
                            websocket_callback=graph2_ws_callback, 
                            planning_checkpointer=None, 
                            main_planning_session_id=session_id
                        )
                        active_graph2_executors[session_id] = exec_manager
                        exec_initial_vals = ExecutionGraphState(
                            initial_input=_session_bot_state_store.get("workflow_extracted_data", {}),
                        ).model_dump(exclude_none=True)
                        
                        graph2_thread_id = f"{session_id}_exec_{uuid.uuid4().hex[:6]}"
                        exec_graph_config = {"configurable": {"thread_id": graph2_thread_id}}
                        
                        await send_websocket_message(websocket, "info", f"Execution phase starting (ID: {graph2_thread_id}).", session_id, "graph2_execution", graph2_thread_id)
                        _session_bot_state_store["workflow_execution_status"] = "running" 

                        asyncio.create_task(exec_manager.execute_workflow(exec_initial_vals, exec_graph_config))
                    except Exception as e_exec_setup:
                        logger.error(f"[{session_id}] Failed to start Graph 2: {e_exec_setup}", exc_info=True)
                        await send_websocket_message(websocket, "error", f"Execution setup error: {str(e_exec_setup)[:150]}", session_id, "system")
                        _session_bot_state_store["workflow_execution_status"] = "failed"
                else:
                    err = "Plan missing" if not exec_plan else "API executor missing"
                    logger.error(f"[{session_id}] Cannot start Graph 2: {err}.")
                    await send_websocket_message(websocket, "error", f"Cannot start execution: {err}.", session_id, "system")
                    _session_bot_state_store["workflow_execution_status"] = "failed"
            
    except WebSocketDisconnect: logger.info(f"WS disconnected: {session_id}.")
    except Exception as e_outer: logger.critical(f"[{session_id}] Unhandled WS loop error: {e_outer}", exc_info=True)
    finally:
        logger.info(f"[{session_id}] Cleaning up WS resources.")
        active_graph2_executors.pop(session_id, None)
        active_graph2_definitions.pop(session_id, None)
        if websocket.client_state == WebSocketState.CONNECTED:
            try: await websocket.close()
            except: pass

@app.get("/", response_class=FileResponse)
async def get_index_page(): # Serve frontend
    index_path = os.path.join(STATIC_DIR, "index.html")
    return FileResponse(index_path) if os.path.exists(index_path) else HTMLResponse("Frontend not found.", 404)
