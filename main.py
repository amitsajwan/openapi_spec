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

# Models and Core Components
from models import BotState, GraphOutput as PlanSchema, ExecutionGraphState
from graph import build_graph
from llm_config import initialize_llms
from execution_graph_definition import ExecutionGraphDefinition
from execution_manager import GraphExecutionManager
from api_executor import APIExecutor as UserAPIExecutor
from utils import SCHEMA_CACHE

# Refactored WebSocket handling logic
from websocket_helpers import handle_websocket_connection, send_websocket_message_helper

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s - %(name)s - %(levelname)s - %(threadName)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- FastAPI Application Setup ---
app = FastAPI()
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --- Global Application State ---
# These are initialized at startup
langgraph_planning_app: Optional[Any] = None
api_executor_instance: Optional[UserAPIExecutor] = None

# These manage active Graph 2 instances and their definitions, keyed by G2 Thread ID
active_graph2_executors: Dict[str, GraphExecutionManager] = {}
active_graph2_definitions: Dict[str, ExecutionGraphDefinition] = {}


# --- FastAPI Startup and Shutdown Events ---
@app.on_event("startup")
async def startup_event():
    """Initializes global components when the FastAPI application starts."""
    global langgraph_planning_app, api_executor_instance
    logger.info("FastAPI application startup...")
    try:
        router_llm, worker_llm = initialize_llms()
        api_executor_instance = UserAPIExecutor(
            base_url=os.getenv("DEFAULT_API_BASE_URL", None),
            timeout=float(os.getenv("API_TIMEOUT", 30.0))
        )
        # Graph 1 (Planning Graph) is built here
        # Pass None for checkpointer if not using persistence for Graph 1
        langgraph_planning_app = build_graph(router_llm, worker_llm, api_executor_instance, None)
        logger.info("Main Planning LangGraph (Graph 1) built successfully.")
    except Exception as e:
        logger.critical(f"CRITICAL FAILURE during application startup: {e}", exc_info=True)
        # Consider how to handle this - e.g., prevent app from starting or run in a degraded mode
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
            api_executor_instance.close() # type: ignore
        logger.info("APIExecutor closed.")
    if SCHEMA_CACHE:
        SCHEMA_CACHE.close()
        logger.info("Schema cache closed.")
    # Consider any other cleanup for active_graph2_executors if needed,
    # though they should ideally be cleaned up when their respective WebSockets close.


# --- WebSocket Endpoint ---
@app.websocket("/ws/openapi_agent")
async def websocket_endpoint(websocket: WebSocket):
    """
    Handles new WebSocket connections for the OpenAPI agent.
    Delegates connection management to the `handle_websocket_connection` function.
    """
    await websocket.accept()
    session_id = str(uuid.uuid4()) # Main planning session ID for this connection
    logger.info(f"WebSocket connection accepted. Assigned Session ID (Graph 1): {session_id}")

    if not langgraph_planning_app or not api_executor_instance:
        # Critical components failed to initialize during startup
        await send_websocket_message_helper(
            websocket, "error", {"error": "Backend services are not initialized. Please try again later or contact support."},
            session_id, "system"
        )
        await websocket.close(code=1011) # Internal error
        return

    # Pass all necessary global/shared components to the handler
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
        logger.error("Frontend index.html not found at expected path.")
        return HTMLResponse("<h1>Frontend not found.</h1><p>Please ensure static files are correctly placed.</p>", status_code=404)

if __name__ == "__main__":
    # This block is for local development and debugging, typically Uvicorn is run directly.
    import uvicorn
    logger.info("Starting Uvicorn server for local development...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
