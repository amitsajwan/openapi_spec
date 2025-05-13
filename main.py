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

            Gemini Advanced

2.5 Pro (preview)

Conversation with Gemini

api_executor
PY

core_logic
PY

execution_...definition
PY

graph
PY

llm_config
PY

main
PY

models
PY

requirements
TXT

router
PY

index.html
HTML
Understand this, keep it I will ask




Show thinking
Okay, I have understood the files you've provided. I'm ready for your questions.


Sources and related content

execution_manager
PY
Take this also




Show thinking
Okay, I've taken that file as well. I'm ready for your questions.


Sources and related content
When I run execute workflow, I don't get infront ui side what api are executed




Show thinking
It seems the information about individual APIs being executed during a workflow isn't being explicitly sent to the user interface (UI) in real-time. Here's a breakdown of why that might be happening and where to look in the code:

How Workflow Execution Information Flows (or Doesn't Flow) to the UI:

Backend Logging vs. UI Messages:

The api_executor.py file, which handles the direct execution of API calls, includes detailed logging about each request (URL, method, payload, response). Similarly, execution_graph_definition.py logs the start and end of each node (which typically corresponds to an API call) in the execution graph.   
However, these are primarily backend logs and are not automatically translated into messages sent to the UI.
WebSocket Communication (main.py):

The main.py file uses WebSockets to communicate with the UI. The send_websocket_message function is key here.
The graph2_ws_callback function in main.py is designed to receive updates from the GraphExecutionManager (which runs the workflow defined in Graph 2) and forward them to the UI.
Currently, the event types handled by this callback and sent to the UI seem to be higher-level workflow events such as:
human_intervention_required (when the workflow pauses for user confirmation)
execution_completed
execution_failed
workflow_timeout
When the execution phase starts, an "info" message like "Execution phase starting..." is sent.
Execution Management (execution_manager.py):

The GraphExecutionManager in execution_manager.py is responsible for running the sequence of API calls defined in the execution plan.
It uses the websocket_callback to send the high-level events mentioned above.
While it processes nodes one by one, it doesn't appear to send a specific WebSocket message to the UI each time an individual API call within the workflow starts or finishes with its result, beyond the backend logging.
User Interface (index.html):

The index.html file defines a "Workflow Execution Log" section (<div id="workflowLogMessages">) where such messages would be displayed.
The corresponding JavaScript (script.js, linked in index.html) would handle incoming WebSocket messages and update this log. It would currently be set up to display messages for the event types that main.py and execution_manager.py are sending.
Why API Execution Details Might Not Appear on the UI:

Based on the code structure:

There isn't a specific mechanism implemented to send a WebSocket message from the backend to the frontend detailing the start, progress, or completion of each individual API call during the workflow execution. The communication is focused on broader workflow state changes.
Potential Areas for Modification to Display API Execution Details:

To show each API call as it's executed on the UI, you would likely need to:

Backend Changes:

In execution_graph_definition.py, within the _make_node_runnable method (which wraps the execution of each API call), you could add calls to self.websocket_callback (passed down or accessed appropriately) before and after the await self.api_executor.execute_api(...) line.
Before: Send a message like {"type": "api_call_start", "content": {"operationId": node_definition.operationId, "method": node_definition.method, "path": node_definition.path}}.
After: Send a message like {"type": "api_call_end", "content": {"operationId": node_definition.operationId, "status_code": api_call_result.get("status_code"), "response_preview": str(api_call_result.get("response_body"))[:100]}}.
Alternatively, the GraphExecutionManager in execution_manager.py could be modified if the LangGraph framework provides events for individual node transitions that could be captured and sent via the websocket_callback.
Frontend Changes (in script.js which is referenced by index.html):

The JavaScript code that handles incoming WebSocket messages would need to be updated to recognize these new message types (api_call_start, api_call_end).
Upon receiving these messages, it should format the information and append it to the workflowLogMessages div in index.html.
By implementing these changes, you can create a more granular log of API executions on the UI.


Sources and related content
Ok give files with updates




Show thinking
Okay, I'll provide the updated files with the necessary changes to display API execution details on the UI.

Here's a summary of the modifications:

execution_graph_definition.py:

The ExecutionGraphDefinition class will now store the websocket_callback and the graph2_thread_id.
The _make_node_runnable method will use this callback to send api_call_start and api_call_end messages to the UI before and after an API call is made.
main.py:

When an ExecutionGraphDefinition instance is created, it will be provided with the graph2_ws_callback and the specific graph2_thread_id for the current execution.
index.html:

I will add comments within a <script> tag in index.html outlining the JavaScript changes needed in your static/script.js file to handle these new WebSocket message types (api_call_start, api_call_end) and display them in the "Workflow Execution Log".
Here are the updated files:

Python

# execution_graph_definition.py
import asyncio
import logging
from typing import Any, Callable, Awaitable, Dict, Optional, Tuple, Union

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
# Interrupt is not raised by nodes anymore in this pattern, but imported for clarity if needed elsewhere.
# from langgraph.types import Interrupt

from models import GraphOutput, Node, Edge, InputMapping, OutputMapping, ExecutionGraphState
from api_executor import APIExecutor

logger = logging.getLogger(__name__)

# _get_value_from_path and _set_value_by_path remain the same

def _get_value_from_path(data_dict: Optional[Dict[str, Any]], path: str) -> Any:
    if not path or not isinstance(data_dict, dict):
        return None
    if path.startswith("$."):
        path = path[2:]
    keys = path.split('.')
    current_val = data_dict
    for key_part in keys:
        if isinstance(current_val, dict):
            if key_part not in current_val:
                return None
            current_val = current_val.get(key_part)
        elif isinstance(current_val, list):
            try:
                idx = int(key_part)
                if 0 <= idx < len(current_val):
                    current_val = current_val[idx]
                else:
                    return None
            except ValueError:
                return None
        else:
            return None
    return current_val

def _set_value_by_path(data_dict: Dict[str, Any], path: str, value: Any):
    if not path:
        return
    keys = path.split('.')
    current_level = data_dict
    for i, key in enumerate(keys[:-1]):
        if key not in current_level or not isinstance(current_level.get(key), dict):
            current_level[key] = {}
        current_level = current_level[key]
    if keys:
        current_level[keys[-1]] = value

class ExecutionGraphDefinition:
    def __init__(
        self,
        graph_execution_plan: GraphOutput,
        api_executor: APIExecutor,
        websocket_callback: Optional[Callable[[str, Dict[str, Any], Optional[str]], Awaitable[None]]] = None, # MODIFIED
        graph2_thread_id: Optional[str] = None # MODIFIED
    ):
        self.graph_plan = graph_execution_plan
        self.api_executor = api_executor
        self.websocket_callback = websocket_callback # MODIFIED
        self.graph2_thread_id = graph2_thread_id # MODIFIED
        self.runnable_graph: Any = self._build_and_compile_graph()
        logger.info("ExecutionGraphDefinition initialized and Graph 2 compiled.")

    async def _send_ws_update(self, event_type: str, data: Dict[str, Any]): # MODIFIED_ADD
        """Helper to send websocket updates if callback is configured."""
        if self.websocket_callback and self.graph2_thread_id:
            try:
                await self.websocket_callback(event_type, data, self.graph2_thread_id)
            except Exception as e:
                logger.error(f"Error sending WS update ({event_type}) for thread {self.graph2_thread_id}: {e}")

    def _resolve_placeholders(self, template_string: str, state: ExecutionGraphState) -> str:
        if not isinstance(template_string, str):
            return str(template_string)
        resolved_string = template_string
        extracted_ids_dict = state.extracted_ids
        initial_input_dict = state.initial_input if isinstance(state.initial_input, dict) else {}
        for key, value in extracted_ids_dict.items():
            if value is not None:
                resolved_string = resolved_string.replace(f"{{{key}}}", str(value))
        for key, value in initial_input_dict.items():
            if value is not None:
                 resolved_string = resolved_string.replace(f"{{{key}}}", str(value))
        return resolved_string

    def _apply_input_mappings(
        self,
        node_definition: Node,
        state: ExecutionGraphState,
        resolved_path_params: Dict[str, Any],
        resolved_query_params: Dict[str, Any],
        resolved_body_payload_for_field_mapping: Dict[str, Any],
        resolved_headers: Dict[str, Any]
    ) -> None:
        if not node_definition.input_mappings:
            return
        current_extracted_ids = state.extracted_ids
        for mapping in node_definition.input_mappings:
            source_value = _get_value_from_path(current_extracted_ids, mapping.source_data_path)
            if source_value is None:
                logger.warning(f"Node '{node_definition.effective_id}': InputMapping - Could not find source data for path '{mapping.source_data_path}'. Skipping.")
                continue
            if mapping.target_parameter_in == "path":
                resolved_path_params[mapping.target_parameter_name] = str(source_value)
            elif mapping.target_parameter_in == "query":
                resolved_query_params[mapping.target_parameter_name] = source_value
            elif mapping.target_parameter_in == "header":
                resolved_headers[mapping.target_parameter_name] = str(source_value)
            elif mapping.target_parameter_in.startswith("body."):
                field_path_in_body = mapping.target_parameter_in.split("body.", 1)[1]
                _set_value_by_path(resolved_body_payload_for_field_mapping, field_path_in_body, source_value)

    def _prepare_api_request_components(
        self, node_definition: Node, state: ExecutionGraphState
    ) -> Tuple[str, Dict[str, Any], Any, Dict[str, Any]]:
        path_template = node_definition.path or ""
        payload_template = node_definition.payload
        if isinstance(payload_template, dict):
            payload_template = payload_template.copy()

        resolved_path_from_template = self._resolve_placeholders(path_template, state)
        body_after_placeholder_resolution: Any
        if isinstance(payload_template, dict):
            body_after_placeholder_resolution = {
                key: (self._resolve_placeholders(value, state) if isinstance(value, str) else value)
                for key, value in payload_template.items()
            }
        elif payload_template is not None:
            body_after_placeholder_resolution = self._resolve_placeholders(str(payload_template), state)
        else:
            body_after_placeholder_resolution = {}

        final_path_params: Dict[str, Any] = {}
        final_query_params: Dict[str, Any] = {}
        final_headers: Dict[str, Any] = {}
        body_for_field_mappings = body_after_placeholder_resolution if isinstance(body_after_placeholder_resolution, dict) else {}

        self._apply_input_mappings(
            node_definition, state, final_path_params, final_query_params,
            body_for_field_mappings, final_headers
        )

        final_body_payload = body_for_field_mappings
        if not isinstance(body_after_placeholder_resolution, dict) and body_for_field_mappings:
            final_body_payload = body_for_field_mappings
        elif isinstance(body_after_placeholder_resolution, dict):
             final_body_payload = body_for_field_mappings
        else:
            final_body_payload = body_after_placeholder_resolution

        for mapping in node_definition.input_mappings or []:
            if mapping.target_parameter_in == "body":
                current_extracted_ids = state.extracted_ids
                source_value = _get_value_from_path(current_extracted_ids, mapping.source_data_path)
                if source_value is not None:
                    final_body_payload = source_value
                    break
        
        final_api_path = resolved_path_from_template
        if final_path_params:
            temp_extracted_ids_for_path = state.extracted_ids.copy()
            temp_extracted_ids_for_path.update(final_path_params)
            initial_input_dict = state.initial_input if isinstance(state.initial_input, dict) else {}
            for k,v in initial_input_dict.items():
                if k not in temp_extracted_ids_for_path:
                    temp_extracted_ids_for_path[k] = v
            temp_state_for_path_re_resolution = ExecutionGraphState(
                extracted_ids=temp_extracted_ids_for_path,
                api_results={}, confirmed_data={}, initial_input={}
            )
            final_api_path = self._resolve_placeholders(path_template, temp_state_for_path_re_resolution)
        return final_api_path, final_query_params, final_body_payload, final_headers

    def _apply_confirmed_data_to_request(
        self, node_definition: Node, state: ExecutionGraphState,
        current_body_payload: Any, current_query_params: Dict[str, Any], current_headers: Dict[str, Any]
    ) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        operationId = node_definition.effective_id
        confirmation_key = f"confirmed_{operationId}"
        updated_body = current_body_payload.copy() if isinstance(current_body_payload, dict) else current_body_payload
        updated_params = current_query_params.copy()
        updated_headers = current_headers.copy()
        current_confirmed_data = state.confirmed_data
        
        # Check if this specific operation was confirmed (decision is True)
        if current_confirmed_data.get(confirmation_key): # This implies decision was True
            confirmed_details = current_confirmed_data.get(f"{confirmation_key}_details", {})
            if "modified_payload" in confirmed_details: # User might have modified the payload
                updated_body = confirmed_details["modified_payload"]
                logger.info(f"Node '{operationId}': Applied modified payload from user confirmation.")
            # Add similar checks for modified_query_params, modified_headers if UI supports modifying them
        return updated_body, updated_params, updated_headers

    async def _execute_api_and_process_outputs(
        self, node_definition: Node, api_path: str,
        query_params: Dict[str, Any], body_payload: Any, headers: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        effective_id = node_definition.effective_id
        api_call_method = node_definition.method or "GET"

        # MODIFIED_ADD: Send api_call_start event
        await self._send_ws_update("api_call_start", {
            "operationId": node_definition.operationId,
            "effective_id": effective_id,
            "method": api_call_method,
            "path": api_path, # Resolved path
            "payload_preview": str(body_payload)[:200] + "..." if body_payload and len(str(body_payload)) > 200 else body_payload,
            "query_params": query_params,
        })

        api_call_result_dict = await self.api_executor.execute_api(
            operationId=effective_id, method=api_call_method, endpoint=api_path,
            payload=body_payload, query_params=query_params, headers=headers
        )
        
        # MODIFIED_ADD: Send api_call_end event
        status_code = api_call_result_dict.get("status_code")
        response_body_preview = str(api_call_result_dict.get("response_body"))[:200] + "..." \
            if api_call_result_dict.get("response_body") and len(str(api_call_result_dict.get("response_body"))) > 200 \
            else api_call_result_dict.get("response_body")

        await self._send_ws_update("api_call_end", {
            "operationId": node_definition.operationId,
            "effective_id": effective_id,
            "method": api_call_method,
            "path": api_path,
            "status_code": status_code,
            "response_preview": response_body_preview,
            "error": api_call_result_dict.get("error"),
            "execution_time": api_call_result_dict.get("execution_time")
        })

        extracted_data_for_state = {}
        is_successful = isinstance(status_code, int) and 200 <= status_code < 300
        if is_successful and node_definition.output_mappings:
            response_body = api_call_result_dict.get("response_body")
            if isinstance(response_body, dict):
                for mapping in node_definition.output_mappings:
                    extracted_value = _get_value_from_path(response_body, mapping.source_data_path)
                    if extracted_value is not None:
                        extracted_data_for_state[mapping.target_data_key] = extracted_value
            elif response_body is not None:
                 logger.warning(f"Node '{effective_id}': Response body type {type(response_body)} not dict. Cannot apply output mappings.")
        return api_call_result_dict, (extracted_data_for_state if extracted_data_for_state else None)

    def _make_node_runnable(
        self, node_definition: Node
    ) -> Callable[[ExecutionGraphState], Awaitable[Dict[str, Any]]]:
        """
        Creates an awaitable function for a given API node definition.
        The graph will interrupt *before* this node if requires_confirmation is true.
        This node, when run, checks if it was confirmed.
        """
        async def node_executor(state: ExecutionGraphState) -> Dict[str, Any]:
            effective_id = node_definition.effective_id
            logger.info(f"--- [Graph 2] Node Start: {effective_id} (OpID: {node_definition.operationId}) ---")
            
            output_state_update: Dict[str, Any] = {} 

            try:
                api_path, query_params, body_payload, headers = self._prepare_api_request_components(node_definition, state)
                
                if node_definition.requires_confirmation:
                    confirmation_key = f"confirmed_{effective_id}"
                    if not state.confirmed_data.get(confirmation_key): 
                        skip_message = f"Node '{effective_id}' requires confirmation, but it was not found or was negative in confirmed_data. Skipping execution."
                        logger.warning(skip_message)
                        output_state_update["error"] = skip_message 
                        output_state_update["api_results"] = {effective_id: {"status_code": "SKIPPED_NO_CONFIRMATION", "error": skip_message}}
                        # MODIFIED_ADD: Send skip event
                        await self._send_ws_update("api_call_skipped", {
                            "operationId":
