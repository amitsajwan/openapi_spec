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
if not os.path.exists(STATIC_DIR): os.makedirs(STATIC_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

langgraph_planning_app: Optional[Any] = None
api_executor_instance: Optional[UserAPIExecutor] = None
active_graph2_executors: Dict[str, GraphExecutionManager] = {}
active_graph2_definitions: Dict[str, ExecutionGraphDefinition] = {}


@app.on_event("startup")
async def startup_event():
    global langgraph_planning_app, api_executor_instance
    logger.info("FastAPI startup...")
    try:
        router_llm, worker_llm = initialize_llms()
        api_executor_instance = UserAPIExecutor(
            base_url=os.getenv("DEFAULT_API_BASE_URL", None),
            timeout=float(os.getenv("API_TIMEOUT", 30.0))
        )
        # Pass None for checkpointer to build_graph if not using one for Graph 1
        langgraph_planning_app = build_graph(router_llm, worker_llm, api_executor_instance, None) 
        logger.info("Main Planning LangGraph (Graph 1) built.")
    except Exception as e:
        logger.critical(f"CRITICAL FAILURE during startup: {e}", exc_info=True)
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
            # Ensure content is JSON serializable
            try:
                json_content_for_send = json.loads(json.dumps(content))
            except TypeError as te:
                logger.warning(f"Content for WS type {msg_type} not JSON serializable, sending as string: {te}. Content: {str(content)[:200]}")
                json_content_for_send = {"raw_content": str(content)}


            await websocket.send_json({
                "type": msg_type, "source": source_graph, "content": json_content_for_send,
                "session_id": session_id, "graph2_thread_id": graph2_thread_id or session_id
            })
    except Exception as e:
        logger.error(f"[{session_id}] WS TX error (Type: {msg_type}, Thread: {graph2_thread_id}): {e}", exc_info=False)


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


    async def graph2_ws_callback_for_session(event_type: str, data: Dict[str, Any], graph2_thread_id_param: Optional[str]):
        target_thread_id_for_msg = graph2_thread_id_param # This ID comes from GraphExecutionManager
        
        await send_websocket_message(websocket, event_type, data, session_id, "graph2_execution", target_thread_id_for_msg)
        
        # Only update high-level status from specific manager events
        # Per-tool events don't change the overall workflow status directly
        if event_type not in ["tool_start", "tool_end"]: # Exclude per-tool events from this general status update
            logger.info(f"[{session_id}] G2 Callback (Thread: {target_thread_id_for_msg}). Event: {event_type}. Store status before: {_session_bot_state_store.get('workflow_execution_status')}")
            new_status = _session_bot_state_store.get("workflow_execution_status", "idle")
            
            if event_type == "human_intervention_required":
                new_status = "paused_for_confirmation"
            elif event_type in ["execution_completed", "execution_failed", "workflow_timeout"]:
                new_status = "completed" if event_type == "execution_completed" else "failed"
                _session_bot_state_store["workflow_execution_results"] = data.get("final_state", {})
                if target_thread_id_for_msg and target_thread_id_for_msg in active_graph2_executors:
                    logger.info(f"[{session_id}] Workflow for G2 Thread ID {target_thread_id_for_msg} ended. Removing its executor.")
                    active_graph2_executors.pop(target_thread_id_for_msg, None)
                    active_graph2_definitions.pop(target_thread_id_for_msg, None)

            _session_bot_state_store["workflow_execution_status"] = new_status
            logger.info(f"[{session_id}] G2 Callback. Store status after: {new_status}")


    current_graph2_thread_id_for_session: Optional[str] = None 

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
            current_bot_state.response = None; current_bot_state.final_response = ""
            current_bot_state.next_step = None; current_bot_state.intent = None

            if user_input_text.lower().startswith("resume_exec"):
                try:
                    parts = user_input_text.split(" ", 2) 
                    if len(parts) < 3:
                        await send_websocket_message(websocket, "warning", "Resume command format: resume_exec <graph2_thread_id> <json_payload>", session_id, "system"); continue
                    
                    graph2_thread_id_to_resume = parts[1]
                    payload_str = parts[2].strip()
                    resume_payload_for_graph2 = json.loads(payload_str) if payload_str else None

                    if not resume_payload_for_graph2:
                        await send_websocket_message(websocket, "warning", "Resume command missing payload.", session_id, "system", graph2_thread_id_to_resume); continue
                    
                    exec_manager = active_graph2_executors.get(graph2_thread_id_to_resume)
                    
                    if _session_bot_state_store.get("workflow_execution_status") == "paused_for_confirmation": # General check
                        if exec_manager: # Specific check for this G2 thread
                            submitted = await exec_manager.submit_resume_data(graph2_thread_id_to_resume, resume_payload_for_graph2)
                            if submitted:
                                await send_websocket_message(websocket, "info", {"message":"Resume data submitted."}, session_id, "graph2_execution", graph2_thread_id_to_resume)
                                _session_bot_state_store["workflow_execution_status"] = "running" 
                            else: await send_websocket_message(websocket, "error", {"error":"Failed to submit resume data to manager."}, session_id, "graph2_execution", graph2_thread_id_to_resume)
                        else: await send_websocket_message(websocket, "warning", {"message":f"Cannot resume: No active exec manager for G2 Thread ID {graph2_thread_id_to_resume}."}, session_id, "system")
                    else:
                        status_msg = _session_bot_state_store.get("workflow_execution_status", "unknown")
                        await send_websocket_message(websocket, "warning", {"message":f"Cannot resume: Workflow not paused (status: {status_msg})."}, session_id, "system")
                except Exception as e_resume:
                    logger.error(f"[{session_id}] Error parsing/processing resume_exec: {e_resume}", exc_info=True)
                    await send_websocket_message(websocket, "error", {"error":f"Invalid resume command: {e_resume}"}, session_id, "system")
                continue

            # --- Graph 1 (Planning) ---
            await send_websocket_message(websocket, "status", {"message":"Planning..."}, session_id, "graph1_planning")
            planning_config = {"configurable": {"thread_id": session_id}}
            try:
                graph1_input = current_bot_state.model_dump(exclude_none=True)
                async for event in langgraph_planning_app.astream_events(graph1_input, config=planning_config, version="v2"): # Using v2 for consistency
                    event_name = event["event"]
                    event_data = event.get("data", {})
                    
                    if event_name == "on_chain_end" or event_name == "on_tool_end": 
                        node_output_any = event_data.get("output")
                        
                        # Attempt to update current_bot_state carefully
                        if isinstance(node_output_any, BotState):
                            current_bot_state = node_output_any
                        elif isinstance(node_output_any, dict):
                            try:
                                # Merge output dict into existing state, preferring keys from output
                                updated_fields = {k: v for k, v in node_output_any.items() if v is not None}
                                current_bot_state = current_bot_state.model_copy(update=updated_fields)
                            except Exception as e_val:
                                logger.error(f"Error updating BotState from dict: {e_val}, dict: {node_output_any}")
                        
                        if current_bot_state.response: # Check if node set an intermediate response
                            await send_websocket_message(websocket, "intermediate", {"message":current_bot_state.response}, session_id, "graph1_planning")
                        
                        if current_bot_state.scratchpad and 'graph_to_send' in current_bot_state.scratchpad:
                            graph_json_str = current_bot_state.scratchpad.pop('graph_to_send', None)
                            if graph_json_str:
                                try:
                                    # graph_to_send should already be a JSON string from model_dump_json
                                    graph_content = json.loads(graph_json_str) 
                                    await send_websocket_message(websocket, "graph_update", graph_content, session_id, "graph1_planning")
                                except Exception as e_graph:
                                    logger.error(f"[{session_id}] Failed to send graph_update (not valid JSON string?): {e_graph}. String was: {graph_json_str[:100]}")
                    
                    elif event_name == "on_graph_end":
                        final_output_any = event_data.get("output")
                        if isinstance(final_output_any, BotState):
                            current_bot_state = final_output_any
                        elif isinstance(final_output_any, dict) :
                            try:
                                updated_fields = {k: v for k, v in final_output_any.items() if v is not None}
                                current_bot_state = current_bot_state.model_copy(update=updated_fields)
                            except Exception as e_val:
                                logger.error(f"Error updating BotState from dict at graph end: {e_val}, dict: {final_output_any}")
                        break 
                
                final_resp_g1 = current_bot_state.final_response or current_bot_state.response
                if final_resp_g1:
                    await send_websocket_message(websocket, "final", {"message":final_resp_g1}, session_id, "graph1_planning")
            
            except Exception as e_plan:
                logger.critical(f"[{session_id}] Planning Graph error: {e_plan}", exc_info=True)
                await send_websocket_message(websocket, "error", {"error":f"Planning error: {str(e_plan)[:150]}"}, session_id, "system")
                current_bot_state.workflow_execution_status = "failed"
            
            _session_bot_state_store["openapi_spec_text"] = current_bot_state.openapi_spec_text
            _session_bot_state_store["openapi_schema"] = current_bot_state.openapi_schema
            # ... (persist other BotState fields as before) ...
            _session_bot_state_store["schema_cache_key"] = current_bot_state.schema_cache_key
            _session_bot_state_store["schema_summary"] = current_bot_state.schema_summary
            _session_bot_state_store["identified_apis"] = current_bot_state.identified_apis
            _session_bot_state_store["payload_descriptions"] = current_bot_state.payload_descriptions
            if current_bot_state.execution_graph:
                _session_bot_state_store["execution_graph"] = current_bot_state.execution_graph.model_dump(exclude_none=True)
            else:
                 _session_bot_state_store["execution_graph"] = None
            _session_bot_state_store["plan_generation_goal"] = current_bot_state.plan_generation_goal
            _session_bot_state_store["input_is_spec"] = current_bot_state.input_is_spec
            _session_bot_state_store["workflow_execution_status"] = current_bot_state.workflow_execution_status
            _session_bot_state_store["workflow_execution_results"] = current_bot_state.workflow_execution_results
            _session_bot_state_store["workflow_extracted_data"] = current_bot_state.workflow_extracted_data


            logger.debug(f"[{session_id}] Persisted BotState. Store status: {_session_bot_state_store.get('workflow_execution_status')}")

            if _session_bot_state_store.get("workflow_execution_status") == "pending_start":
                logger.info(f"[{session_id}] Store signals 'pending_start'. Initiating Graph 2.")
                exec_plan_dict = _session_bot_state_store.get("execution_graph")
                exec_plan = PlanSchema.model_validate(exec_plan_dict) if exec_plan_dict else None

                if exec_plan and api_executor_instance:
                    try:
                        current_graph2_thread_id_for_session = f"{session_id}_exec_{uuid.uuid4().hex[:8]}"
                        logger.info(f"[{session_id}] Generated G2 Thread ID for this run: {current_graph2_thread_id_for_session}")

                        # REMOVED: websocket_callback and graph2_thread_id from ExecutionGraphDefinition constructor
                        exec_graph_def = ExecutionGraphDefinition(
                            graph_execution_plan=exec_plan,
                            api_executor=api_executor_instance
                        )
                        active_graph2_definitions[current_graph2_thread_id_for_session] = exec_graph_def
                        runnable_exec_graph = exec_graph_def.get_runnable_graph()
                        
                        exec_manager = GraphExecutionManager(
                            runnable_graph=runnable_exec_graph,
                            graph_definition=exec_graph_def, 
                            websocket_callback=graph2_ws_callback_for_session,
                            planning_checkpointer=None, # Checkpointer for Graph 2 if needed
                            main_planning_session_id=session_id 
                        )
                        active_graph2_executors[current_graph2_thread_id_for_session] = exec_manager
                        
                        exec_initial_vals = ExecutionGraphState(
                            initial_input=_session_bot_state_store.get("workflow_extracted_data", {}),
                            # Other fields like api_results, extracted_ids, confirmed_data start empty via default_factory
                        ).model_dump(exclude_none=True)
                        
                        exec_graph_config = {"configurable": {"thread_id": current_graph2_thread_id_for_session}}
                        
                        await send_websocket_message(websocket, "info", {"message":f"Execution phase starting (G2 Thread ID: {current_graph2_thread_id_for_session})."}, session_id, "graph2_execution", current_graph2_thread_id_for_session)
                        _session_bot_state_store["workflow_execution_status"] = "running" 

                        asyncio.create_task(exec_manager.execute_workflow(exec_initial_vals, exec_graph_config))
                    
                    except Exception as e_exec_setup:
                        g2_tid_err = current_graph2_thread_id_for_session or "UNKNOWN_G2_TID"
                        logger.error(f"[{session_id}] Failed to start Graph 2 (G2 Thread ID: {g2_tid_err}): {e_exec_setup}", exc_info=True)
                        await send_websocket_message(websocket, "error", {"error":f"Execution setup error: {str(e_exec_setup)[:150]}"}, session_id, "system", g2_tid_err)
                        _session_bot_state_store["workflow_execution_status"] = "failed"
                else:
                    err = "Plan missing" if not exec_plan else "API executor missing"
                    logger.error(f"[{session_id}] Cannot start Graph 2: {err}.")
                    await send_websocket_message(websocket, "error", {"error":f"Cannot start execution: {err}."}, session_id, "system")
                    _session_bot_state_store["workflow_execution_status"] = "failed"
    
