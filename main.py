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
from api_executor import APIExecutor as UserAPIExecutor # Renamed to avoid conflict if APIExecutor name is used elsewhere
from utils import SCHEMA_CACHE

# Refactored WebSocket handling logic
from websocket_helpers import handle_websocket_connection, send_websocket_message_helper

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO, # Consider making this configurable (e.g., from ENV for DEBUG)
    stream=sys.stdout,
    format='%(asctime)s - %(name)s - %(levelname)s - %(threadName)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- FastAPI Application Setup ---
app = FastAPI()
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(STATIC_DIR):
    # This might be an issue if the static dir is essential and doesn't get created by other means.
    # For robustness, ensure this directory exists or handle the case where it might not.
    logger.warning(f"Static directory {STATIC_DIR} does not exist. Frontend might not load.")
    # os.makedirs(STATIC_DIR) # Or create it if it's expected to be empty initially
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
            timeout=float(os.getenv("API_TIMEOUT", "30.0")) # Ensure default is a string for float()
        )
        # Graph 1 (Planning Graph) is built here using the refactored build_graph
        # The signature of build_graph (router_llm, worker_llm, api_executor_instance, checkpointer)
        # has remained the same, so this call should still be valid.
        langgraph_planning_app = build_graph(
            router_llm,
            worker_llm,
            api_executor_instance,
            None # No checkpointer specified for Graph 1 state persistence in this setup
        )
        logger.info("Main Planning LangGraph (Graph 1) built successfully using refactored components.")
    except Exception as e:
        logger.critical(f"CRITICAL FAILURE during application startup: {e}", exc_info=True)
        # In a production scenario, you might want to exit or have a fallback.
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
            # Ensure close is callable if not a coroutine (though httpx.AsyncClient.aclose is)
            # This branch might be unnecessary if APIExecutor.close is always async.
            getattr(api_executor_instance, 'close')() # type: ignore
        logger.info("APIExecutor closed.")
    if SCHEMA_CACHE:
        SCHEMA_CACHE.close()
        logger.info("Schema cache closed.")
    # Consider cleanup for active_graph2_executors if they hold resources
    # that aren't cleaned up by WebSocket disconnections.
    # For example, if they spawn background tasks that aren't managed by the WebSocket lifecycle.
    logger.info(f"Cleaning up {len(active_graph2_executors)} active Graph 2 executors (if any)...")
    for exec_id, executor in list(active_graph2_executors.items()): # Iterate over a copy for safe removal
        # Implement a cleanup method in GraphExecutionManager if needed
        # e.g., await executor.cleanup_resources()
        logger.info(f"Potentially cleaning up executor for G2_Thread_ID: {exec_id}")
        active_graph2_executors.pop(exec_id, None)
        active_graph2_definitions.pop(exec_id, None)
    logger.info("Graph 2 executor cleanup attempt complete.")


# --- WebSocket Endpoint ---
@app.websocket("/ws/openapi_agent")
async def websocket_endpoint(websocket: WebSocket):
    """
    Handles new WebSocket connections for the OpenAPI agent.
    Delegates connection management to the `handle_websocket_connection` function.
    """
    await websocket.accept()
    session_id = str(uuid.uuid4()) # Main planning session ID (Graph 1) for this WebSocket connection
    logger.info(f"WebSocket connection accepted. Assigned Session ID (Graph 1): {session_id}")

    if not langgraph_planning_app or not api_executor_instance:
        # This indicates a critical failure during startup.
        logger.error(
            f"[{session_id}] Cannot handle WebSocket connection: Backend services (Planning Graph or API Executor) are not initialized."
        )
        await send_websocket_message_helper(
            websocket=websocket,
            msg_type="error",
            content={"error": "Backend services are not initialized. Please try again later or contact support."},
            session_id=session_id, # Use the generated session_id for this message
            source_graph="system_critical"
        )
        await websocket.close(code=1011) # Internal error preventing operation
        return

    # Pass all necessary global/shared components to the handler.
    # handle_websocket_connection now uses these to interact with Graph 1 and manage Graph 2 instances.
    await handle_websocket_connection(
        websocket=websocket,
        session_id=session_id, # This is the Graph 1 session_id
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
    # This block is for local development and debugging.
    # In production, Uvicorn (or another ASGI server) would typically be run directly against main:app.
    import uvicorn
    logger.info("Starting Uvicorn server for local development...")
    # Example: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)

