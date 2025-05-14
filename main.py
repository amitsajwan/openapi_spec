# main.py
import logging
import uuid
# import json # Not directly used here, but often useful
import os
import sys
import asyncio
from typing import Any, Dict, Optional # Removed Callable, Awaitable, Literal as they are not directly used here

from fastapi import FastAPI, WebSocket # WebSocketDisconnect also imported but not directly used
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
# from starlette.websockets import WebSocketState # Not directly used here

# Models and Core Components
# from models import BotState, GraphOutput as PlanSchema, ExecutionGraphState # Not directly used here
from graph import build_graph
from llm_config import initialize_llms # Now returns three LLMs
from execution_graph_definition import ExecutionGraphDefinition # For type hinting if needed
from execution_manager import GraphExecutionManager # For type hinting if needed
from api_executor import APIExecutor as UserAPIExecutor
from utils import SCHEMA_CACHE

# Refactored WebSocket handling logic
from websocket_helpers import handle_websocket_connection, send_websocket_message_helper

# --- Logging Configuration ---
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(), # Allow configuring log level via ENV
    stream=sys.stdout,
    format='%(asctime)s - %(name)s - %(levelname)s - %(threadName)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- FastAPI Application Setup ---
app = FastAPI()
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(STATIC_DIR):
    logger.warning(f"Static directory {STATIC_DIR} does not exist. Frontend might not load.")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --- Global Application State ---
langgraph_planning_app: Optional[Any] = None
api_executor_instance: Optional[UserAPIExecutor] = None
# No need to store utility_llm globally if it's only passed down once at startup

active_graph2_executors: Dict[str, GraphExecutionManager] = {}
active_graph2_definitions: Dict[str, ExecutionGraphDefinition] = {}


# --- FastAPI Startup and Shutdown Events ---
@app.on_event("startup")
async def startup_event():
    """Initializes global components when the FastAPI application starts."""
    global langgraph_planning_app, api_executor_instance
    logger.info("FastAPI application startup...")
    try:
        # initialize_llms now returns router_llm, worker_llm, utility_llm
        router_llm, worker_llm, utility_llm = initialize_llms()
        
        api_executor_instance = UserAPIExecutor(
            base_url=os.getenv("DEFAULT_API_BASE_URL", None),
            timeout=float(os.getenv("API_TIMEOUT", "30.0"))
        )
        
        # Pass all three LLMs to build_graph
        langgraph_planning_app = build_graph(
            router_llm=router_llm,
            worker_llm=worker_llm,
            utility_llm=utility_llm, # New utility LLM
            api_executor_instance=api_executor_instance,
            checkpointer=None # No checkpointer for Graph 1 state persistence
        )
        logger.info("Main Planning LangGraph (Graph 1) built successfully with all LLMs.")
    except Exception as e:
        logger.critical(f"CRITICAL FAILURE during application startup: {e}", exc_info=True)
        langgraph_planning_app = None
        api_executor_instance = None


@app.on_event("shutdown")
async def shutdown_event():
    """Cleans up resources when the FastAPI application shuts down."""
    logger.info("FastAPI application shutdown...")
    if api_executor_instance and hasattr(api_executor_instance, 'close'):
        if asyncio.iscoroutinefunction(getattr(api_executor_instance, 'close')):
            await api_executor_instance.close()
        else:
            getattr(api_executor_instance, 'close')() # type: ignore
        logger.info("APIExecutor closed.")
    if SCHEMA_CACHE:
        SCHEMA_CACHE.close()
        logger.info("Schema cache closed.")
    
    logger.info(f"Cleaning up {len(active_graph2_executors)} active Graph 2 executors (if any)...")
    for exec_id, executor in list(active_graph2_executors.items()):
        logger.info(f"Potentially cleaning up executor for G2_Thread_ID: {exec_id}")
        # Add executor.cleanup() or similar if GraphExecutionManager needs explicit cleanup
        active_graph2_executors.pop(exec_id, None)
        active_graph2_definitions.pop(exec_id, None)
    logger.info("Graph 2 executor cleanup attempt complete.")


# --- WebSocket Endpoint ---
@app.websocket("/ws/openapi_agent")
async def websocket_endpoint(websocket: WebSocket):
    """Handles new WebSocket connections for the OpenAPI agent."""
    await websocket.accept()
    session_id = str(uuid.uuid4()) 
    logger.info(f"WebSocket connection accepted. Assigned Session ID (Graph 1): {session_id}")

    if not langgraph_planning_app or not api_executor_instance:
        logger.error(
            f"[{session_id}] Cannot handle WebSocket connection: Backend services are not initialized."
        )
        # Use the imported send_websocket_message_helper
        await send_websocket_message_helper(
            websocket=websocket,
            msg_type="error",
            content={"error": "Backend services are not initialized. Please try again later or contact support."},
            session_id=session_id,
            source_graph="system_critical"
        )
        await websocket.close(code=1011) 
        return

    await handle_websocket_connection(
        websocket=websocket,
        session_id=session_id,
        langgraph_planning_app=langgraph_planning_app,
        api_executor_instance=api_executor_instance,
        active_graph2_executors=active_graph2_executors,
        active_graph2_definitions=active_graph2_definitions
    )


# --- Static File Serving for Frontend ---
@app.get("/", response_class=FileResponse)
async def get_index_page():
    """Serves the main HTML page for the frontend."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        logger.error(f"Frontend index.html not found at expected path: {index_path}")
        return HTMLResponse(
            "<h1>Frontend not found.</h1><p>Please ensure static files are correctly placed and "
            f"the STATIC_DIR '{STATIC_DIR}' is correctly configured and accessible.</p>",
            status_code=404
        )

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server for local development...")
    # Ensure the app is run as "main:app" where "main" is the filename and "app" is the FastAPI instance.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
