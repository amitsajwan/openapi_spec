# main.py
import logging
import uuid
import json
import os
import sys
import asyncio
from typing import Any, Dict, Optional, Callable, Awaitable, Literal
from datetime import datetime, timezone

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

# --- Models ---
from models import BotState, GraphOutput as PlanSchema, ExecutionGraphState

# --- Graph 1 (Planning) Components ---
from graph import build_graph # build_graph will need to be adjusted if checkpointer is removed
from llm_config import initialize_llms 

# --- Graph 2 (Execution) Components ---
from execution_graph_definition import ExecutionGraphDefinition
from execution_manager import GraphExecutionManager

# --- API Executor ---
from api_executor import APIExecutor as UserAPIExecutor

# --- Utilities & Checkpointing ---
# from langgraph.checkpoint.memory import MemorySaver # No longer used for Graph 1 directly here
# from langgraph.checkpoint.base import BaseCheckpointSaver # No longer used for Graph 1 directly here
from utils import SCHEMA_CACHE

# Logging Setup
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
# planning_checkpointer: Optional[BaseCheckpointSaver] = None # REMOVED
active_graph2_executors: Dict[str, GraphExecutionManager] = {}


@app.on_event("startup")
async def startup_event():
    global langgraph_planning_app, api_executor_instance # REMOVED planning_checkpointer
    logger.info("FastAPI startup: Initializing LLMs, API Executor, and Planning Graph (Graph 1)...")
    try:
        router_llm_instance, worker_llm_instance = initialize_llms()
        
        api_executor_instance = UserAPIExecutor(
            base_url=os.getenv("DEFAULT_API_BASE_URL"),
            timeout=float(os.getenv("API_TIMEOUT", 30.0))
        )
        logger.info("User's APIExecutor instance created successfully.")

        # REMOVED planning_checkpointer initialization

        langgraph_planning_app = build_graph(
            router_llm_instance,
            worker_llm_instance,
            api_executor_instance,
            None # Pass None for checkpointer to build_graph
        )
        logger.info("Main Planning LangGraph (Graph 1) built and compiled (NO CHECKPOINTER).")
    except Exception as e:
        logger.critical(f"CRITICAL FAILURE during startup: {e}", exc_info=True)
        langgraph_planning_app = None
        if api_executor_instance and hasattr(api_executor_instance, 'close'):
            await api_executor_instance.close()
        api_executor_instance = None
        # planning_checkpointer = None # REMOVED

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("FastAPI shutdown initiated.")
    if api_executor_instance and hasattr(api_executor_instance, 'close'):
        try:
            await api_executor_instance.close()
            logger.info("User's APIExecutor client closed successfully.")
        except Exception as e:
            logger.error(f"Error closing User's APIExecutor client: {e}", exc_info=True)
    
    if SCHEMA_CACHE and hasattr(SCHEMA_CACHE, 'close'):
        try:
            SCHEMA_CACHE.close() 
            logger.info("Schema cache (DiskCache) closed.")
        except Exception as e:
            logger.error(f"Error closing schema cache: {e}", exc_info=True)
    logger.info("FastAPI shutdown complete.")

async def send_websocket_message(
    websocket: WebSocket,
    msg_type: str,
    content: Any,
    session_id: str, 
    source_graph: Optional[Literal["system", "graph1_planning", "graph2_execution"]] = "graph1_planning",
    graph2_thread_id: Optional[str] = None 
):
    try:
        if websocket.client_state == WebSocketState.CONNECTED:
            payload = {
                "type": msg_type,
                "source": source_graph,
                "content": content,
                "session_id": session_id, 
                "graph2_thread_id": graph2_thread_id if graph2_thread_id else session_id 
            }
            await websocket.send_json(payload)
    except WebSocketDisconnect:
        logger.warning(f"[{session_id}] WS TX fail (Type: {msg_type}, Source: {source_graph}): Client disconnected.")
    except Exception as e:
        logger.error(f"[{session_id}] WS TX error (Type: {msg_type}, Source: {source_graph}): {e}", exc_info=False)


@app.websocket("/ws/openapi_agent")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4()) 
    logger.info(f"WebSocket connection accepted. Session ID (Graph 1): {session_id}")

    # MODIFIED: Check only for langgraph_planning_app and api_executor_instance
    if langgraph_planning_app is None or api_executor_instance is None:
        await send_websocket_message(websocket, "error", "Backend agent or API executor not initialized. Check server logs.", session_id, "system")
        await websocket.close(code=1011) 
        return

    await send_websocket_message(websocket, "info", {"message": "Connection established. Ready for OpenAPI spec or queries."}, session_id, "system")
    
    # This BotState will be reset for each new message if no checkpointer is used for Graph 1
    # However, we can maintain it for the duration of processing a single user message that might
    # trigger Graph 2.
    current_bot_state: BotState = BotState(session_id=session_id) 

    # default_put_metadata no longer needed for Graph 1 checkpointing here

    async def graph2_ws_callback_with_state_update(
        event_type: str,
        data: Dict[str, Any],
        graph2_thread_id_param: Optional[str] 
    ):
        # This callback still sends messages
        await send_websocket_message(websocket, event_type, data, session_id, "graph2_execution", graph2_thread_id_param)

        # If Graph 2 has finished, we can update the *in-memory* current_bot_state for this websocket connection,
        # but it won't be persisted by Graph 1's (now non-existent) checkpointer.
        if event_type in ["execution_completed", "execution_failed", "workflow_timeout"]:
            logger.info(f"[{session_id}] Graph 2 (ThreadID: {graph2_thread_id_param}) reported terminal state: {event_type}. Updating in-memory BotState for current request.")
            
            # Update the current_bot_state instance directly
            if event_type == "execution_completed":
                current_bot_state.workflow_execution_status = "completed"
            elif event_type == "workflow_timeout":
                current_bot_state.workflow_execution_status = "failed" 
                current_bot_state.response = data.get("message", "Workflow timed out.")
            else: 
                current_bot_state.workflow_execution_status = "failed"
            
            current_bot_state.workflow_execution_results = data.get("final_state", {})
            if data.get("final_state", {}).get("error"):
                 current_bot_state.response = current_bot_state.response or str(data["final_state"]["error"])
            
            logger.info(f"[{session_id}] In-memory BotState updated with Graph 2 final status: {current_bot_state.workflow_execution_status}")
            # No planning_checkpointer.put() call here

    try:
        while True:
            user_input_text = await websocket.receive_text()
            user_input_text = user_input_text.strip()
            logger.info(f"[{session_id}] WS RX User Input: '{user_input_text[:150]}...'")

            if not user_input_text:
                continue
            
            # Initialize or reset current_bot_state for each new user message cycle
            # If you want some continuity within a single WebSocket session but not true checkpointing,
            # you could initialize current_bot_state outside the loop and only reset certain fields.
            # For full statelessness per message (simpler to bypass checkpointer issues):
            if current_bot_state.session_id != session_id: # Should not happen with current setup
                 current_bot_state = BotState(session_id=session_id)
            
            current_bot_state.user_input = user_input_text
            current_bot_state.response = None
            current_bot_state.final_response = "" 
            current_bot_state.next_step = None
            current_bot_state.intent = None 
            # If openapi_spec_text was set from a previous message in this session, it might persist
            # depending on how current_bot_state is managed. For full statelessness per message,
            # you'd also reset schema-related fields here if not parsing a new spec.
            # For now, let's assume the router handles spec detection and resetting relevant fields.
            if current_bot_state.scratchpad: current_bot_state.scratchpad.pop('graph_to_send', None)


            is_resume_for_graph2 = False
            resume_payload_for_graph2: Optional[Dict[str, Any]] = None

            if user_input_text.lower().startswith("resume_exec"):
                try:
                    payload_str = user_input_text.split("resume_exec", 1)[-1].strip()
                    if payload_str:
                        resume_payload_for_graph2 = json.loads(payload_str)
                        is_resume_for_graph2 = True
                    else:
                        await send_websocket_message(websocket, "warning", "Resume command 'resume_exec' received, but no payload data followed.", session_id, "system")
                except json.JSONDecodeError:
                    logger.error(f"[{session_id}] Could not parse JSON from resume_exec command: '{payload_str}'")
                    await send_websocket_message(websocket, "error", "Invalid JSON format for resume_exec payload.", session_id, "system")
                except Exception as e_resume_parse:
                    logger.error(f"[{session_id}] Error parsing resume_exec command: {e_resume_parse}")
                    await send_websocket_message(websocket, "error", f"Error parsing resume command: {str(e_resume_parse)}", session_id, "system")

            if is_resume_for_graph2 and resume_payload_for_graph2:
                execution_manager_instance = active_graph2_executors.get(session_id)
                if execution_manager_instance:
                    if current_bot_state.workflow_execution_status == "paused_for_confirmation":
                        logger.info(f"[{session_id}] User provided resume data for Graph 2. Submitting to Execution Manager.")
                        submitted = await execution_manager_instance.submit_resume_data(
                            main_session_id=session_id, 
                            resume_data=resume_payload_for_graph2
                        )
                        if submitted:
                            await send_websocket_message(websocket, "info", "Resume data submitted to execution workflow. It will attempt to continue.", session_id, "graph2_execution")
                            current_bot_state.workflow_execution_status = "running"
                        else:
                             await send_websocket_message(websocket, "error", "Failed to submit resume data to the execution workflow.", session_id, "graph2_execution")
                    else:
                        status_msg = current_bot_state.workflow_execution_status
                        logger.warning(f"[{session_id}] Received resume data, but workflow status is '{status_msg}', not 'paused_for_confirmation'.")
                        await send_websocket_message(websocket, "warning", f"Cannot resume: Workflow is not currently paused for confirmation (current status: {status_msg}).", session_id, "system")
                else:
                    logger.warning(f"[{session_id}] Received resume data, but no active Execution Manager found for this session.")
                    await send_websocket_message(websocket, "warning", "Cannot resume: No active execution workflow found for this session.", session_id, "system")
                continue 

            await send_websocket_message(websocket, "status", "Processing with Planning Graph...", session_id, "graph1_planning")
            # Config for astream_events still needs thread_id for internal LangGraph routing/logging if used by the graph.
            # No checkpointing means no loading/saving of state between astream_events calls for different user inputs.
            planning_graph_config = {"configurable": {"thread_id": session_id}} 
            

            try:
                graph1_input_for_astream = current_bot_state.model_dump(exclude_none=True)
                # pending_sends is not relevant if not using a checkpointer that manages it.

                async for event in langgraph_planning_app.astream_events(graph1_input_for_astream, config=planning_graph_config, version="v1"): 
                    if event["event"] == "on_chain_end" or event["event"] == "on_tool_end": 
                        node_output_data = event["data"].get("output")
                        if isinstance(node_output_data, BotState): # If node returns full state
                            current_bot_state.update(node_output_data.model_dump(exclude_none=True)) # Merge update
                        elif isinstance(node_output_data, dict): # If node returns partial state dict
                            current_bot_state = BotState(**{**current_bot_state.model_dump(exclude_none=True), **node_output_data})
                        
                        if current_bot_state.response:
                             await send_websocket_message(websocket, "intermediate", current_bot_state.response, session_id, "graph1_planning")
                        if current_bot_state.scratchpad and 'graph_to_send' in current_bot_state.scratchpad:
                            graph_json_plan_to_send = current_bot_state.scratchpad.pop('graph_to_send', None)
                            if graph_json_plan_to_send:
                                try:
                                    await send_websocket_message(websocket, "graph_update", json.loads(graph_json_plan_to_send), session_id, "graph1_planning")
                                except json.JSONDecodeError:
                                    logger.error(f"[{session_id}] Failed to parse graph_json_plan_to_send for UI.")
                    
                    if event["event"] == "on_graph_end": 
                        final_planning_output_data = event["data"].get("output") 
                        if final_planning_output_data and isinstance(final_planning_output_data, dict):
                             current_bot_state = BotState(**{**current_bot_state.model_dump(exclude_none=True), **final_planning_output_data})
                        elif isinstance(final_planning_output_data, BotState): 
                             current_bot_state = final_planning_output_data # Replace if full state is output
                        logger.info(f"[{session_id}] Planning Graph (Graph 1) finished run. Final response: '{str(current_bot_state.final_response)[:100]}...'")
                        break 
                
                final_response_from_graph1 = current_bot_state.final_response or current_bot_state.response 
                if final_response_from_graph1:
                    await send_websocket_message(websocket, "final", final_response_from_graph1, session_id, "graph1_planning")

            except Exception as e_planning_graph:
                logger.critical(f"[{session_id}] Planning Graph (Graph 1) execution error: {e_planning_graph}", exc_info=True)
                await send_websocket_message(websocket, "error", f"Planning Agent error: {str(e_planning_graph)[:150]}", session_id, "system")
                # No checkpointing to update on error
                continue

            if current_bot_state.workflow_execution_status == "pending_start":
                logger.info(f"[{session_id}] Planning Graph signaled 'pending_start'. Initiating Execution Graph (Graph 2).")
                execution_plan_from_graph1: Optional[PlanSchema] = current_bot_state.execution_graph

                if not execution_plan_from_graph1 or not isinstance(execution_plan_from_graph1, PlanSchema):
                    logger.error(f"[{session_id}] Cannot start Execution Graph: 'execution_graph' (plan) is missing or invalid in BotState.")
                    await send_websocket_message(websocket, "error", "Execution plan from planning phase is missing or invalid.", session_id, "system")
                    current_bot_state.workflow_execution_status = "failed"
                elif not api_executor_instance: 
                    logger.error(f"[{session_id}] Cannot start Execution Graph: api_executor_instance is not available.")
                    await send_websocket_message(websocket, "error", "Internal Server Error: API Executor not ready.", session_id, "system")
                    current_bot_state.workflow_execution_status = "failed"
                else:
                    try:
                        exec_graph_def = ExecutionGraphDefinition(execution_plan_from_graph1, api_executor_instance)
                        runnable_exec_graph = exec_graph_def.get_runnable_graph()
                        
                        exec_manager = GraphExecutionManager(
                            runnable_graph=runnable_exec_graph,
                            websocket_callback=graph2_ws_callback_with_state_update, 
                            planning_checkpointer=None, # Graph 2 doesn't need Graph 1's checkpointer
                            main_planning_session_id=session_id 
                        )
                        active_graph2_executors[session_id] = exec_manager

                        exec_initial_values = ExecutionGraphState(
                            initial_input=current_bot_state.workflow_extracted_data or {},
                            extracted_ids={},
                            api_results={},
                            confirmed_data={}
                        ).model_dump(exclude_none=True)

                        graph2_thread_id = f"{session_id}_exec_{uuid.uuid4().hex[:6]}" 
                        # Graph 2 uses its own MemorySaver, so its config doesn't need these extra keys for that.
                        execution_graph_config = {"configurable": {"thread_id": graph2_thread_id}} 
                        
                        logger.info(f"[{session_id}] ExecutionManager (Graph 2) created for its ThreadID '{graph2_thread_id}'. Starting workflow stream as a background task.")
                        await send_websocket_message(websocket, "info", f"Execution phase started (ID: {graph2_thread_id}). Monitoring API calls...", session_id, "graph2_execution", graph2_thread_id)
                        current_bot_state.workflow_execution_status = "running"

                        asyncio.create_task(
                            exec_manager.execute_workflow(exec_initial_values, execution_graph_config)
                        )
                    except ValueError as ve: 
                        logger.error(f"[{session_id}] Value error setting up Execution Graph (Graph 2): {ve}", exc_info=True)
                        await send_websocket_message(websocket, "error", f"Setup error for execution: {str(ve)[:150]}", session_id, "system")
                        current_bot_state.workflow_execution_status = "failed"
                    except Exception as e_exec_setup:
                        logger.error(f"[{session_id}] Failed to create/start Execution Graph (Graph 2): {e_exec_setup}", exc_info=True)
                        await send_websocket_message(websocket, "error", f"Failed to start execution phase: {str(e_exec_setup)[:150]}", session_id, "system")
                        current_bot_state.workflow_execution_status = "failed"
            
            # No planning_checkpointer.put() at the end of the loop for Graph 1

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}.")
    except Exception as e_outer_loop:
        logger.critical(f"[{session_id}] Unhandled error in WebSocket main loop: {e_outer_loop}", exc_info=True)
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await send_websocket_message(websocket, "error", "A critical server error occurred. Please try reconnecting.", session_id, "system")
            except Exception: pass 
    finally:
        logger.info(f"[{session_id}] Cleaning up WebSocket connection resources.")
        if session_id in active_graph2_executors:
            del active_graph2_executors[session_id]
            logger.info(f"[{session_id}] Removed GraphExecutionManager instance.")
        
        if websocket.client_state == WebSocketState.CONNECTED:
            try: await websocket.close()
            except Exception: pass

@app.get("/", response_class=FileResponse)
async def get_index_page():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        html_content = """
        <html><head><title>OpenAPI Agent</title></head>
        <body><h1>OpenAPI Agent Backend</h1>
        <p>Static frontend (index.html) not found in ./static directory.</p>
        <p>Please create an index.html file to interact with the WebSocket endpoint at /ws/openapi_agent.</p>
        </body></html>
        """
        return HTMLResponse(html_content, status_code=404)
    return FileResponse(index_path)
