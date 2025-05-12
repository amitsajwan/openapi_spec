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

# --- Models ---
from models import BotState, GraphOutput as PlanSchema, ExecutionGraphState

# --- Graph 1 (Planning) Components ---
from graph import build_graph
from llm_config import initialize_llms # Assuming you have this file

# --- Graph 2 (Execution) Components ---
from execution_graph_definition import ExecutionGraphDefinition
from execution_manager import GraphExecutionManager

# --- API Executor ---
from api_executor import APIExecutor as UserAPIExecutor

# --- Utilities & Checkpointing ---
from langgraph.checkpoint.memory import MemorySaver # Using MemorySaver for Graph 1
from langgraph.checkpoint.base import BaseCheckpointSaver # For type hinting
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

# Global instances, initialized on startup
langgraph_planning_app: Optional[Any] = None # Compiled LangGraph for Graph 1
api_executor_instance: Optional[UserAPIExecutor] = None
planning_checkpointer: Optional[BaseCheckpointSaver] = None # Checkpointer for Graph 1
# Stores active Graph 2 execution managers, keyed by Graph 1's session_id
active_graph2_executors: Dict[str, GraphExecutionManager] = {}


@app.on_event("startup")
async def startup_event():
    global langgraph_planning_app, api_executor_instance, planning_checkpointer
    logger.info("FastAPI startup: Initializing LLMs, API Executor, and Planning Graph (Graph 1)...")
    try:
        router_llm_instance, worker_llm_instance = initialize_llms()
        
        api_executor_instance = UserAPIExecutor(
            base_url=os.getenv("DEFAULT_API_BASE_URL"),
            timeout=float(os.getenv("API_TIMEOUT", 30.0))
        )
        logger.info("User's APIExecutor instance created successfully.")

        planning_checkpointer = MemorySaver() # Initialize checkpointer for Graph 1
        logger.info("Planning Graph (Graph 1) checkpointer (MemorySaver) initialized.")

        langgraph_planning_app = build_graph(
            router_llm_instance,
            worker_llm_instance,
            api_executor_instance,
            planning_checkpointer # Pass the checkpointer to Graph 1 builder
        )
        logger.info("Main Planning LangGraph (Graph 1) built and compiled successfully.")
    except Exception as e:
        logger.critical(f"CRITICAL FAILURE during startup: {e}", exc_info=True)
        langgraph_planning_app = None
        if api_executor_instance and hasattr(api_executor_instance, 'close'):
            await api_executor_instance.close()
        api_executor_instance = None
        planning_checkpointer = None # Ensure it's None if setup fails

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
            SCHEMA_CACHE.close() # Assuming SCHEMA_CACHE is the diskcache.Cache instance
            logger.info("Schema cache (DiskCache) closed.")
        except Exception as e:
            logger.error(f"Error closing schema cache: {e}", exc_info=True)
    logger.info("FastAPI shutdown complete.")

async def send_websocket_message(
    websocket: WebSocket,
    msg_type: str,
    content: Any,
    session_id: str, # This is the main session_id (Graph 1's thread_id)
    source_graph: Optional[Literal["system", "graph1_planning", "graph2_execution"]] = "graph1_planning",
    graph2_thread_id: Optional[str] = None # Specific thread_id for Graph 2 if message originates there
):
    """
    Helper function to send JSON messages over WebSocket.
    `session_id` is always Graph 1's session.
    `graph2_thread_id` is used if the message is from a specific Graph 2 instance.
    """
    try:
        if websocket.client_state == WebSocketState.CONNECTED:
            payload = {
                "type": msg_type,
                "source": source_graph,
                "content": content,
                "session_id": session_id, # Main session ID for client tracking
                # If graph2_thread_id is provided, client might use it to associate with a specific execution flow
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
    session_id = str(uuid.uuid4()) # This is Graph 1's session_id / thread_id
    logger.info(f"WebSocket connection accepted. Session ID (Graph 1): {session_id}")

    if langgraph_planning_app is None or api_executor_instance is None or planning_checkpointer is None:
        await send_websocket_message(websocket, "error", "Backend agent, API executor, or checkpointer not initialized. Check server logs.", session_id, "system")
        await websocket.close(code=1011) # Internal server error
        return

    await send_websocket_message(websocket, "info", {"message": "Connection established. Ready for OpenAPI spec or queries."}, session_id, "system")
    
    current_bot_state: Optional[BotState] = None # Holds the state for Graph 1

    # --- Callback for GraphExecutionManager (Graph 2) ---
    async def graph2_ws_callback_with_state_update(
        event_type: str,
        data: Dict[str, Any],
        graph2_thread_id_param: Optional[str] # Graph 2's own thread_id
    ):
        """
        Callback for GraphExecutionManager to send messages via WebSocket
        AND update Graph 1's BotState upon Graph 2 completion/failure.
        """
        # Send message to client
        await send_websocket_message(websocket, event_type, data, session_id, "graph2_execution", graph2_thread_id_param)

        # If Graph 2 has finished (completed or failed), update Graph 1's BotState
        if event_type in ["execution_completed", "execution_failed", "workflow_timeout"]:
            logger.info(f"[{session_id}] Graph 2 (ThreadID: {graph2_thread_id_param}) reported terminal state: {event_type}. Updating Graph 1 BotState.")
            graph1_config = {"configurable": {"thread_id": session_id}}
            try:
                # Fetch the latest BotState for Graph 1
                checkpoint = planning_checkpointer.get(graph1_config)
                if checkpoint:
                    raw_state_values = checkpoint.get("channel_values", checkpoint) # LangGraph structure
                    bot_state_to_update = BotState.model_validate(raw_state_values)
                    
                    # Update status and results from Graph 2
                    if event_type == "execution_completed":
                        bot_state_to_update.workflow_execution_status = "completed"
                    elif event_type == "workflow_timeout":
                        bot_state_to_update.workflow_execution_status = "failed" # Or a specific "timeout" status
                        bot_state_to_update.response = data.get("message", "Workflow timed out.")
                    else: # execution_failed
                        bot_state_to_update.workflow_execution_status = "failed"
                    
                    bot_state_to_update.workflow_execution_results = data.get("final_state", {})
                    if data.get("final_state", {}).get("error"):
                         bot_state_to_update.response = bot_state_to_update.response or str(data["final_state"]["error"])


                    # Save the updated BotState back to the checkpointer
                    planning_checkpointer.put(graph1_config, bot_state_to_update.model_dump(exclude_none=True))
                    logger.info(f"[{session_id}] Graph 1 BotState updated and checkpointed with Graph 2 final status: {bot_state_to_update.workflow_execution_status}")
                    
                    # Optionally send a final consolidated message from Graph 1 if needed,
                    # or let the Graph 2 message be the primary indicator.
                    # For now, we assume the Graph 2 message is sufficient.
                    # If Graph 1 needs to react immediately, this would be the place.
                    
                else:
                    logger.warning(f"[{session_id}] Could not retrieve Graph 1 BotState for update after Graph 2 finished.")
            except Exception as e_update:
                logger.error(f"[{session_id}] Error updating Graph 1 BotState after Graph 2 finished: {e_update}", exc_info=True)
    # --- End Callback ---

    try:
        while True:
            user_input_text = await websocket.receive_text()
            user_input_text = user_input_text.strip()
            logger.info(f"[{session_id}] WS RX User Input: '{user_input_text[:150]}...'")

            if not user_input_text:
                continue

            is_resume_for_graph2 = False
            resume_payload_for_graph2: Optional[Dict[str, Any]] = None

            # Check for "resume_exec" command
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

            # --- Handle Resume for Graph 2 ---
            if is_resume_for_graph2 and resume_payload_for_graph2:
                execution_manager_instance = active_graph2_executors.get(session_id)
                if execution_manager_instance:
                    # It's important to get the LATEST current_bot_state here, as Graph 2 might have
                    # updated it via the callback if it timed out waiting for this resume.
                    graph1_config_for_resume_check = {"configurable": {"thread_id": session_id}}
                    latest_checkpoint = planning_checkpointer.get(graph1_config_for_resume_check)
                    temp_bot_state_for_resume: Optional[BotState] = None
                    if latest_checkpoint:
                        try:
                            temp_bot_state_for_resume = BotState.model_validate(latest_checkpoint.get("channel_values", latest_checkpoint))
                        except Exception: pass
                    
                    if temp_bot_state_for_resume and temp_bot_state_for_resume.workflow_execution_status == "paused_for_confirmation":
                        logger.info(f"[{session_id}] User provided resume data for Graph 2. Submitting to Execution Manager.")
                        submitted = await execution_manager_instance.submit_resume_data(
                            main_session_id=session_id, # Graph 1's session_id
                            resume_data=resume_payload_for_graph2
                        )
                        if submitted:
                            await send_websocket_message(websocket, "info", "Resume data submitted to execution workflow. It will attempt to continue.", session_id, "graph2_execution")
                            # Tentatively update local current_bot_state; actual state update happens via callback or next Graph 1 cycle
                            if current_bot_state: current_bot_state.workflow_execution_status = "running"
                        else:
                             await send_websocket_message(websocket, "error", "Failed to submit resume data to the execution workflow.", session_id, "graph2_execution")
                    else:
                        status_msg = temp_bot_state_for_resume.workflow_execution_status if temp_bot_state_for_resume else (current_bot_state.workflow_execution_status if current_bot_state else 'Unknown')
                        logger.warning(f"[{session_id}] Received resume data, but workflow status is '{status_msg}', not 'paused_for_confirmation'.")
                        await send_websocket_message(websocket, "warning", f"Cannot resume: Workflow is not currently paused for confirmation (current status: {status_msg}).", session_id, "system")
                else:
                    logger.warning(f"[{session_id}] Received resume data, but no active Execution Manager found for this session.")
                    await send_websocket_message(websocket, "warning", "Cannot resume: No active execution workflow found for this session.", session_id, "system")
                continue # Skip Graph 1 processing for this input

            # --- Process with Planning Graph (Graph 1) ---
            await send_websocket_message(websocket, "status", "Processing with Planning Graph...", session_id, "graph1_planning")
            planning_graph_config = {"configurable": {"thread_id": session_id}}
            
            # Load or initialize BotState for Graph 1
            checkpoint = planning_checkpointer.get(planning_graph_config)
            if checkpoint:
                try:
                    raw_state_values = checkpoint.get("channel_values", checkpoint)
                    current_bot_state = BotState.model_validate(raw_state_values)
                except Exception as e_load_state:
                    logger.warning(f"[{session_id}] Error loading BotState from checkpoint: {e_load_state}. Initializing new.", exc_info=False)
                    current_bot_state = BotState(session_id=session_id)
            else:
                current_bot_state = BotState(session_id=session_id)

            # Update state with new user input and reset transient fields for Graph 1 cycle
            current_bot_state.user_input = user_input_text
            current_bot_state.response = None
            current_bot_state.final_response = "" # Ensure final_response is reset
            current_bot_state.next_step = None
            current_bot_state.intent = None # Router will set this
            if current_bot_state.scratchpad: current_bot_state.scratchpad.pop('graph_to_send', None)


            try:
                # Stream events from Graph 1
                graph1_input_state_dict = current_bot_state.model_dump(exclude_none=True)
                async for event in langgraph_planning_app.astream_events(graph1_input_state_dict, config=planning_graph_config, version="v1"): # Use "v1" or a specific version
                    if event["event"] == "on_chain_end" or event["event"] == "on_tool_end": # LangGraph event names
                        node_output_data = event["data"].get("output")
                        if isinstance(node_output_data, BotState):
                            current_bot_state = node_output_data
                        elif isinstance(node_output_data, dict): # Output might be a dict representation of BotState
                            try:
                                current_bot_state = BotState.model_validate(node_output_data)
                            except Exception as e_val:
                                logger.error(f"[{session_id}] Error validating Graph 1 node output into BotState: {e_val}")
                        
                        # Send intermediate responses and graph updates
                        if current_bot_state and current_bot_state.response:
                             await send_websocket_message(websocket, "intermediate", current_bot_state.response, session_id, "graph1_planning")
                        if current_bot_state and current_bot_state.scratchpad and 'graph_to_send' in current_bot_state.scratchpad:
                            graph_json_plan_to_send = current_bot_state.scratchpad.pop('graph_to_send', None)
                            if graph_json_plan_to_send:
                                try:
                                    await send_websocket_message(websocket, "graph_update", json.loads(graph_json_plan_to_send), session_id, "graph1_planning")
                                except json.JSONDecodeError:
                                    logger.error(f"[{session_id}] Failed to parse graph_json_plan_to_send for UI.")
                    
                    if event["event"] == "on_graph_end": # LangGraph event name
                        final_planning_output_dict = event["data"].get("output")
                        if final_planning_output_dict and isinstance(final_planning_output_dict, dict):
                             current_bot_state = BotState.model_validate(final_planning_output_dict)
                        logger.info(f"[{session_id}] Planning Graph (Graph 1) finished run. Final response: '{str(current_bot_state.final_response if current_bot_state else 'N/A')[:100]}...'")
                        break # Exit astream_events loop for Graph 1
                
                if not current_bot_state: # Should not happen if graph runs
                    logger.error(f"[{session_id}] CRITICAL: BotState is None after Planning Graph (Graph 1) stream. This indicates a severe issue.");
                    await send_websocket_message(websocket, "error", "Critical error in planning agent.", session_id, "system")
                    continue
                
                # Send final response from Graph 1
                final_response_from_graph1 = current_bot_state.final_response or current_bot_state.response # Fallback if final_response wasn't set
                if final_response_from_graph1:
                    await send_websocket_message(websocket, "final", final_response_from_graph1, session_id, "graph1_planning")

            except Exception as e_planning_graph:
                logger.critical(f"[{session_id}] Planning Graph (Graph 1) execution error: {e_planning_graph}", exc_info=True)
                await send_websocket_message(websocket, "error", f"Planning Agent error: {str(e_planning_graph)[:150]}", session_id, "system")
                # Save current (potentially partial) state on error
                if current_bot_state:
                    planning_checkpointer.put(planning_graph_config, current_bot_state.model_dump(exclude_none=True))
                continue

            # --- Initiate Execution Graph (Graph 2) if pending_start ---
            if current_bot_state and current_bot_state.workflow_execution_status == "pending_start":
                logger.info(f"[{session_id}] Planning Graph signaled 'pending_start'. Initiating Execution Graph (Graph 2).")
                execution_plan_from_graph1: Optional[PlanSchema] = getattr(current_bot_state, 'execution_graph', None)

                if not execution_plan_from_graph1 or not isinstance(execution_plan_from_graph1, PlanSchema):
                    logger.error(f"[{session_id}] Cannot start Execution Graph: 'execution_graph' (plan) is missing or invalid in BotState.")
                    await send_websocket_message(websocket, "error", "Execution plan from planning phase is missing or invalid.", session_id, "system")
                    if current_bot_state: current_bot_state.workflow_execution_status = "failed"
                elif not api_executor_instance: # Should have been caught at startup
                    logger.error(f"[{session_id}] Cannot start Execution Graph: api_executor_instance is not available.")
                    await send_websocket_message(websocket, "error", "Internal Server Error: API Executor not ready.", session_id, "system")
                    if current_bot_state: current_bot_state.workflow_execution_status = "failed"
                else:
                    try:
                        exec_graph_def = ExecutionGraphDefinition(execution_plan_from_graph1, api_executor_instance)
                        runnable_exec_graph = exec_graph_def.get_runnable_graph()
                        
                        # Instantiate GraphExecutionManager with the new callback and planning_checkpointer
                        exec_manager = GraphExecutionManager(
                            runnable_graph=runnable_exec_graph,
                            websocket_callback=graph2_ws_callback_with_state_update, # Use the enhanced callback
                            planning_checkpointer=planning_checkpointer, # Pass Graph 1's checkpointer
                            main_planning_session_id=session_id # Pass Graph 1's session_id
                        )
                        active_graph2_executors[session_id] = exec_manager

                        # Prepare initial state for Graph 2
                        exec_initial_values = ExecutionGraphState(
                            initial_input=current_bot_state.workflow_extracted_data or {},
                            extracted_ids={},
                            api_results={},
                            confirmed_data={}
                        ).model_dump(exclude_none=True)

                        graph2_thread_id = f"{session_id}_exec_{uuid.uuid4().hex[:6]}" # Unique ID for this Graph 2 instance
                        execution_graph_config = {"configurable": {"thread_id": graph2_thread_id}}
                        
                        logger.info(f"[{session_id}] ExecutionManager (Graph 2) created for its ThreadID '{graph2_thread_id}'. Starting workflow stream as a background task.")
                        await send_websocket_message(websocket, "info", f"Execution phase started (ID: {graph2_thread_id}). Monitoring API calls...", session_id, "graph2_execution", graph2_thread_id)
                        if current_bot_state: current_bot_state.workflow_execution_status = "running"

                        # Run Graph 2 execution as a background task
                        asyncio.create_task(
                            exec_manager.execute_workflow(exec_initial_values, execution_graph_config)
                        )
                        # Graph 1's turn ends here for now. Graph 2 runs in background.
                        # BotState (with status "running") will be saved below.
                        # Updates from Graph 2 will come via its callback.

                    except ValueError as ve: # Errors from ExecutionGraphDefinition or Manager setup
                        logger.error(f"[{session_id}] Value error setting up Execution Graph (Graph 2): {ve}", exc_info=True)
                        await send_websocket_message(websocket, "error", f"Setup error for execution: {str(ve)[:150]}", session_id, "system")
                        if current_bot_state: current_bot_state.workflow_execution_status = "failed"
                    except Exception as e_exec_setup:
                        logger.error(f"[{session_id}] Failed to create/start Execution Graph (Graph 2): {e_exec_setup}", exc_info=True)
                        await send_websocket_message(websocket, "error", f"Failed to start execution phase: {str(e_exec_setup)[:150]}", session_id, "system")
                        if current_bot_state: current_bot_state.workflow_execution_status = "failed"
            
            # Save Graph 1's state at the end of its processing cycle
            if current_bot_state:
                planning_checkpointer.put(planning_graph_config, current_bot_state.model_dump(exclude_none=True))
                logger.debug(f"[{session_id}] Graph 1 BotState checkpointed. Workflow status: {current_bot_state.workflow_execution_status}")

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}.")
    except Exception as e_outer_loop:
        logger.critical(f"[{session_id}] Unhandled error in WebSocket main loop: {e_outer_loop}", exc_info=True)
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await send_websocket_message(websocket, "error", "A critical server error occurred. Please try reconnecting.", session_id, "system")
            except Exception: pass # Avoid error in error handling
    finally:
        logger.info(f"[{session_id}] Cleaning up WebSocket connection resources.")
        # Clean up the Graph 2 executor if it exists for this session
        if session_id in active_graph2_executors:
            # Potentially add cleanup logic to GraphExecutionManager if needed (e.g., cancel tasks)
            del active_graph2_executors[session_id]
            logger.info(f"[{session_id}] Removed GraphExecutionManager instance.")
        
        if websocket.client_state == WebSocketState.CONNECTED:
            try: await websocket.close()
            except Exception: pass

@app.get("/", response_class=FileResponse)
async def get_index_page():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        # Provide a simple HTML response if index.html is missing
        html_content = """
        <html><head><title>OpenAPI Agent</title></head>
        <body><h1>OpenAPI Agent Backend</h1>
        <p>Static frontend (index.html) not found in ./static directory.</p>
        <p>Please create an index.html file to interact with the WebSocket endpoint at /ws/openapi_agent.</p>
        </body></html>
        """
        return HTMLResponse(html_content, status_code=404)
    return FileResponse(index_path)

# To run with uvicorn: uvicorn main:app --reload
# Ensure .env has GOOGLE_API_KEY, DEFAULT_API_BASE_URL, API_TIMEOUT as needed.
# And llm_config.py is present and correctly configured.
