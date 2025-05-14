# main.py (Conceptual - showing only the WebSocket endpoint modification)
import logging
import json
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Depends
from starlette.websockets import WebSocketState

# Your existing imports
# from models import BotState
# from config import get_graph_for_session
# from core_logic.graph_definition import create_graph_with_streaming_context # Or your compiled graph
# from core_logic.websocket_helpers import send_intermediate_ws_messages

# Import websocket helpers (send_thinking_started/finished are now removed from it)
from core_logic.websocket_helpers import (
    send_error_message,
    send_intermediate_message,
    send_graph_update,
    send_final_response,
    send_api_response 
    # send_status_update (if used for other statuses)
)

logger = logging.getLogger(__name__)
# app = FastAPI() # Your FastAPI app
# compiled_graph = create_graph_with_streaming_context() # Your compiled LangGraph app

# @app.websocket("/ws/openai_spec_agent/{session_id}") # Your WebSocket route
async def websocket_endpoint(websocket: WebSocket, session_id: str): # Adjust params
    await websocket.accept()
    logger.info(f"WebSocket connection accepted for session_id: {session_id}")

    thread_id = f"ws_session_{session_id}_{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}
    logger.info(f"Using LangGraph config: {config}")

    # current_graph_app = compiled_graph # Assuming 'compiled_graph' is your app

    try:
        while True:
            raw_data_from_client = await websocket.receive_text()
            client_data = json.loads(raw_data_from_client)
            message_type = client_data.get("type")
            
            logger.info(f"Received message type '{message_type}' from session '{session_id}'")

            try:
                if message_type == "process_openapi_spec" or message_type == "user_interaction" or message_type == "run_workflow":
                    # REMOVED: await send_thinking_started(websocket, "Processing your request...")

                    graph_input_dict = {}
                    if message_type == "process_openapi_spec":
                        graph_input_dict = {
                            "user_input": client_data.get("openapi_spec_string") or client_data.get("openapi_spec_url"),
                            "openapi_spec_string": client_data.get("openapi_spec_string"),
                            "openapi_spec_url": client_data.get("openapi_spec_url"),
                            "spec_source": client_data.get("source"),
                            "spec_file_name": client_data.get("file_name"),
                            "plan_generation_goal": client_data.get("goal", "Generate a plan to interact with this API."),
                            "chat_history": [], 
                            "scratchpad": {"intermediate_messages": []},
                            "websocket_session_id": session_id,
                            "current_graph_thread_id": thread_id
                        }
                    elif message_type == "user_interaction":
                        graph_input_dict = {
                            "user_input": client_data.get("text"),
                            "chat_history": [], 
                            "execution_graph": client_data.get("current_graph"), 
                            "scratchpad": {"intermediate_messages": []},
                            "websocket_session_id": session_id,
                            "current_graph_thread_id": thread_id
                        }
                    elif message_type == "run_workflow": # Handle run_workflow
                        graph_input_dict = {
                            "user_input": client_data.get("goal", "Execute the current workflow."), # The router can see this
                            "execution_graph": client_data.get("graph"), # Pass the graph to be executed
                            "intent_override": "setup_workflow_execution", # Hint for the router
                            "chat_history": [],
                            "scratchpad": {"intermediate_messages": []},
                            "websocket_session_id": session_id,
                            "current_graph_thread_id": thread_id
                        }
                    
                    final_state_data = None
                    # Replace `compiled_graph` with your actual compiled LangGraph application
                    async for event in compiled_graph.astream_events(graph_input_dict, config=config, version="v2"):
                        event_type = event["event"]
                        event_name = event["name"]
                        event_data = event["data"]

                        if event_type == "on_chain_end" or event_type == "on_tool_end":
                            output_state = event_data.get("output")
                            if isinstance(output_state, dict):
                                final_state_data = output_state # Keep track of the latest state
                                new_ui_messages = output_state.get("scratchpad", {}).get("intermediate_messages", [])
                                for msg_content in new_ui_messages:
                                    await send_intermediate_message(websocket, msg_content)
                                if "scratchpad" in output_state and "intermediate_messages" in output_state["scratchpad"]:
                                    output_state["scratchpad"]["intermediate_messages"] = [] # Clear after sending
                                
                                current_graph_payload = output_state.get("execution_graph") # This should be a GraphOutput model dump
                                if current_graph_payload:
                                     await send_graph_update(websocket, current_graph_payload)

                                api_call_result = output_state.get("scratchpad", {}).get("last_api_response")
                                if api_call_result and isinstance(api_call_result, dict):
                                    await send_api_response(websocket, api_call_result)
                                    if "scratchpad" in output_state and "last_api_response" in output_state["scratchpad"]:
                                        del output_state["scratchpad"]["last_api_response"] # Clear after sending


                        if event_type == "on_chain_end" and (event_name == "__root__" or event_name.lower() == "agent" or event_name.lower() == "workflow"):
                            final_output_map = event_data.get("output")
                            if isinstance(final_output_map, dict):
                                final_response_text = final_output_map.get("response", "Processing complete.")
                                # Send the final response along with any other relevant data from final_output_map
                                await send_final_response(websocket, final_response_text, data_payload=final_output_map)
                                logger.info(f"Graph execution finished for {session_id}. Final response sent.")
                            else: # If final output is not a dict, send a generic final response
                                await send_final_response(websocket, "Processing complete.", data_payload={"output": final_output_map})


                    # REMOVED: The explicit send_thinking_finished call.
                    # The client will stop thinking upon receiving 'final_response' or 'error'.

                # ... (other message_type handling like execute_api_call if you have it) ...
                
                else:
                    logger.warning(f"Unknown message type from session {session_id}: {message_type}")
                    await send_error_message(websocket, f"Unknown message type received: {message_type}")

            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected by client {session_id}.")
                break 
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received from session {session_id}.")
                await send_error_message(websocket, "Invalid JSON format in message.")
                # Client will stop thinking on this error message.
            except Exception as e:
                logger.error(f"Error during WebSocket communication or graph processing for session {session_id}: {e}", exc_info=True)
                await send_error_message(websocket, f"An server error occurred: {str(e)}")
                # Client will stop thinking on this error message.
            # REMOVED: The finally block that used to send send_thinking_finished.

    except Exception as e:
        logger.error(f"Unhandled exception in WebSocket handler for session {session_id}: {e}", exc_info=True)
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await send_error_message(websocket, "A critical server error occurred with the WebSocket connection.")
            except Exception: 
                pass
    finally:
        logger.info(f"Closing WebSocket connection for session_id: {session_id}.")

