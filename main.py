# main.py
import logging
import uuid
import os
import sys
import asyncio
from typing import Any, Dict, Optional

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# Models and Core Components
from models import APIExecutor # Ensure APIExecutor is imported
from graph import build_graph
from llm_config import initialize_llms
from execution_graph_definition import ExecutionGraphDefinition # For type hinting
from execution_manager import GraphExecutionManager # For type hinting
# from api_executor import APIExecutor as UserAPIExecutor # Already imported as APIExecutor
from utils import SCHEMA_CACHE # For shutdown

# Refactored WebSocket handling logic
from websocket_helpers import handle_websocket_connection #, send_websocket_message_helper (not used directly in main)

# --- Logging Configuration ---
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    stream=sys.stdout,
    format='%(asctime)s - %(name)s - %(levelname)s - %(threadName)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- FastAPI Application Setup ---
app = FastAPI()
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(STATIC_DIR):
    logger.warning(f"Static directory {STATIC_DIR} does not exist. Frontend might not load.")
else:
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --- Global Application State ---
langgraph_planning_app: Optional[Any] = None
api_executor_instance: Optional[APIExecutor] = None # Use the imported APIExecutor

# These dictionaries track single Graph 2 instances, not load test workers directly
active_graph2_executors: Dict[str, GraphExecutionManager] = {}
active_graph2_definitions: Dict[str, ExecutionGraphDefinition] = {}


# --- FastAPI Startup and Shutdown Events ---
@app.on_event("startup")
async def startup_event():
    """Initializes global components when the FastAPI application starts."""
    global langgraph_planning_app, api_executor_instance
    logger.info("FastAPI application startup...")
    try:
        router_llm, worker_llm, utility_llm = initialize_llms()

        api_executor_instance = APIExecutor( # Use the imported APIExecutor
            base_url=os.getenv("DEFAULT_API_BASE_URL", None),
            timeout=float(os.getenv("API_TIMEOUT", "30.0"))
        )

        langgraph_planning_app = build_graph(
            router_llm=router_llm,
            worker_llm=worker_llm,
            utility_llm=utility_llm,
            api_executor_instance=api_executor_instance, # Pass the created instance
            checkpointer=None # No checkpointer for Graph 1 state persistence in this setup
        )
        logger.info("Main Planning LangGraph (Graph 1) built successfully.")
    except Exception as e:
        logger.critical(f"CRITICAL FAILURE during application startup: {e}", exc_info=True)
        # Ensure these are None if startup fails, so WebSocket handler knows
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
            # This path might not be hit if close is always async, but good for robustness
            getattr(api_executor_instance, 'close')() # type: ignore
        logger.info("APIExecutor closed.")

    if SCHEMA_CACHE and hasattr(SCHEMA_CACHE, 'close'): # Check if SCHEMA_CACHE is not None and has close
        try:
            SCHEMA_CACHE.close() # For diskcache or similar
            logger.info("Schema cache closed.")
        except Exception as e_cache_close:
            logger.warning(f"Error closing schema cache: {e_cache_close}")


    logger.info(f"Cleaning up {len(active_graph2_executors)} active single Graph 2 executors (if any)...")
    # This cleans up single G2 instances, load test workers are managed by the orchestrator
    for exec_id, executor in list(active_graph2_executors.items()):
        logger.info(f"Potentially cleaning up single G2 executor for G2_Thread_ID: {exec_id}")
        # If GraphExecutionManager has a cleanup method, call it here
        # e.g., if executor.cleanup(): await executor.cleanup()
        active_graph2_executors.pop(exec_id, None)
        active_graph2_definitions.pop(exec_id, None) # Also clear its definition
    logger.info("Single Graph 2 executor cleanup attempt complete.")


# --- WebSocket Endpoint ---
@app.websocket("/ws/openapi_agent")
async def websocket_endpoint(websocket: WebSocket):
    """Handles new WebSocket connections for the OpenAPI agent."""
    await websocket.accept()
    # G1 session ID, distinct from any G2 thread IDs
    g1_session_id = str(uuid.uuid4())
    logger.info(f"WebSocket connection accepted. Assigned G1 Session ID: {g1_session_id}")

    if not langgraph_planning_app or not api_executor_instance:
        logger.error(
            f"[{g1_session_id}] Cannot handle WebSocket connection: Backend services (Planning Graph or API Executor) are not initialized."
        )
        # Use a simple send_json here as send_websocket_message_helper is in websocket_helpers
        await websocket.send_json({
            "type": "error",
            "source": "system_critical",
            "content": {"error": "Backend services are not initialized. Please try again later or contact support."},
            "session_id": g1_session_id,
        })
        await websocket.close(code=1011) # Internal Error
        return

    # Pass the global dictionaries for single G2 instance management
    await handle_websocket_connection(
        websocket=websocket,
        session_id=g1_session_id, # This is the G1 session ID
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
    # Standard way to run Uvicorn for FastAPI
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
