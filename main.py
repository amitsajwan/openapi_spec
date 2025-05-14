# main.py (Conceptual - showing only the WebSocket endpoint modification)
import logging
import json
import uuid # For unique thread IDs

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langgraph.errors import GraphRecursionError

# Your existing imports
# from models import BotState # Assuming BotState is correctly defined
# from config import get_graph_for_session # Your function to get/create graph instances
# from core_logic.graph_definition import create_graph_with_streaming_context # Or however your graph is created/compiled
# from core_logic.websocket_helpers import send_intermediate_ws_messages # Your existing helper

# Import the new websocket helpers for thinking status
from core_logic.websocket_helpers import (
    send_thinking_started,
    send_thinking_finished,
    send_error_message,
    send_intermediate_message # If you need to send other messages directly
    # ... other helpers like send_graph_update, send_final_response
)

# Your FastAPI app and other routes
# app = FastAPI()
# ...

# Mount static files and templates if you do it here
# app.mount("/static", StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="templates")


# Assume 'compiled_graph' is your main LangGraph application instance
# compiled_graph = create_graph_with_streaming_context() # Or however you get it

logger = logging.getLogger(__name__)

# Your existing WebSocket endpoint
# @app.websocket("/ws/openai_spec_agent/{session_id}") # Or your specific path
async def websocket_endpoint(websocket: WebSocket, session_id: str): # Adjust params as per your setup
    await websocket.accept()
    logger.info(f"WebSocket connection accepted for session_id: {session_id}")

    # Create a unique LangGraph config for this WebSocket session
    # This is crucial for isolating state between concurrent users/sessions.
    # Using a new thread_id for each connection, or manage sessions more robustly
    thread_id = f"ws_session_{session_id}_{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}
    logger.info(f"Using LangGraph config: {config}")

    # Get a graph instance, possibly session-specific if your setup requires it
    # current_graph_app = get_graph_for_session(session_id) # Example
    # For this example, let's assume 'compiled_graph' is the globally available, compiled LangGraph app
    # from main import compiled_graph # (Ensure it's imported or available)


    try:
        while True:
            raw_data_from_client = await websocket.receive_text()
            client_data = json.loads(raw_data_from_client)
            message_type = client_data.get("type")
            
            logger.info(f"Received message type '{message_type}' from session '{session_id}'")

            # Initialize success flag for thinking_finished
            operation_successful = False
            thinking_message_sent = False

            try:
                if message_type == "process_openapi_spec" or message_type == "user_interaction":
                    # --- Signal Thinking Started ---
                    await send_thinking_started(websocket, "Processing your request...")
                    thinking_message_sent = True

                    # Prepare input for the graph based on message_type
                    # This needs to match how your LangGraph expects its initial state/input
                    graph_input_dict = {}
                    if message_type == "process_openapi_spec":
                        graph_input_dict = {
                            "user_input": client_data.get("openapi_spec_string") or client_data.get("openapi_spec_url"), # The router node will handle this
                            "openapi_spec_string": client_data.get("openapi_spec_string"),
                            "openapi_spec_url": client_data.get("openapi_spec_url"),
                            "spec_source": client_data.get("source"),
                            "spec_file_name": client_data.get("file_name"),
                            "plan_generation_goal": client_data.get("goal", "Generate a plan to interact with this API."),
                            "chat_history": [], # Initialize or load history
                            "scratchpad": {"intermediate_messages": []}, # For intermediate updates
                            "websocket_session_id": session_id, # Pass session ID if BotState needs it
                            "current_graph_thread_id": thread_id # Pass thread_id for graph execution context
                        }
                    elif message_type == "user_interaction":
                        graph_input_dict = {
                            "user_input": client_data.get("text"),
                            # Load relevant parts of state: chat_history, execution_graph, etc.
                            # This depends on how your graph manages state continuity for interactions.
                            # For simplicity, assuming user_input is the primary driver for the router.
                            "chat_history": [], # Placeholder - manage history properly
                            "execution_graph": client_data.get("current_graph"), # If client sends it
                            "scratchpad": {"intermediate_messages": []},
                            "websocket_session_id": session_id,
                            "current_graph_thread_id": thread_id
                        }
                    
                    # Ensure the input structure matches what your graph's BotState/input schema expects.
                    # The 'user_input' key is typically what the OpenAPIRouter node would look for.

                    # --- Invoke LangGraph and Stream Events ---
                    # Replace `compiled_graph` with your actual compiled LangGraph application
                    # from main import compiled_graph # Ensure it's accessible
                    
                    final_state = None
                    async for event in compiled_graph.astream_events(graph_input_dict, config=config, version="v2"):
                        event_type = event["event"]
                        event_name = event["name"]
                        event_data = event["data"]
                        # logger.debug(f"Graph Event ({session_id}): {event_type} - {event_name}")

                        # Your existing logic to handle different event types
                        # (e.g., on_tool_end, on_chat_model_stream)
                        # This is where send_intermediate_ws_messages would be called if you use StreamingContext
                        # For example, if a node updates BotState.scratchpad.intermediate_messages:
                        if event_type == "on_chain_end" or event_type == "on_tool_end":
                            output_state = event_data.get("output")
                            if isinstance(output_state, dict): # LangGraph v2 passes state as dict
                                final_state = output_state # Keep track of the latest state
                                # Check for intermediate messages in the output state's scratchpad
                                new_ui_messages = output_state.get("scratchpad", {}).get("intermediate_messages", [])
                                for msg_content in new_ui_messages:
                                    await send_intermediate_message(websocket, msg_content)
                                # Clear them from the state if they are meant to be transient for UI
                                if "scratchpad" in output_state and "intermediate_messages" in output_state["scratchpad"]:
                                    output_state["scratchpad"]["intermediate_messages"] = []
                                
                                # Send graph updates if present in the output state
                                current_graph_json = output_state.get("execution_graph")
                                if current_graph_json: # Assuming execution_graph is the JSON serializable graph
                                     # from core_logic.websocket_helpers import send_graph_update (ensure imported)
                                     # await send_graph_update(websocket, current_graph_json) # If it's already a dict
                                     pass # Your script.js handles graph updates from 'graph_update' type messages

                        # If it's the end of the root graph execution
                        if event_type == "on_chain_end" and (event_name == "__root__" or event_name.lower() == "agent" or event_name.lower() == "workflow"): # Adjust to your root node/graph name
                            final_output_map = event_data.get("output")
                            if isinstance(final_output_map, dict):
                                final_response_text = final_output_map.get("response", "Processing complete.")
                                # from core_logic.websocket_helpers import send_final_response (ensure imported)
                                # await send_final_response(websocket, final_response_text, data_payload=final_output_map)
                                logger.info(f"Graph execution finished for {session_id}. Final response: {final_response_text[:100]}")
                    
                    operation_successful = True # Mark as successful if stream completes
                    # The final "thinking_finished" will be sent in the outer finally block.

                elif message_type == "execute_api_call": # Example for direct API call from UI
                    # This would be a separate flow, not directly part of the main agent graph perhaps
                    # Or it could be an input to a specific node.
                    # For now, just an example placeholder.
                    await send_thinking_started(websocket, "Executing API call...")
                    thinking_message_sent = True
                    # ... logic to execute API call ...
                    # from core_logic.websocket_helpers import send_api_response (ensure imported)
                    # await send_api_response(websocket, {"status_code": 200, "body": {"message": "API call successful"}})
                    operation_successful = True
                
                else:
                    logger.warning(f"Unknown message type from session {session_id}: {message_type}")
                    await send_error_message(websocket, f"Unknown message type received: {message_type}")
                    # No thinking started, so no need to finish thinking for this specific case.

            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected by client {session_id}.")
                break # Exit the while True loop for this connection
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received from session {session_id}.")
                await send_error_message(websocket, "Invalid JSON format in message.")
                # If thinking was started for a previous valid message, ensure it's stopped.
                if thinking_message_sent: # Check if send_thinking_started was called for this iteration
                    await send_thinking_finished(websocket, "Failed due to invalid input.", success=False)
            except GraphRecursionError as gre: # Specific LangGraph error
                logger.error(f"Graph recursion error for session {session_id}: {gre}", exc_info=True)
                await send_error_message(websocket, f"Critical error: The workflow entered a loop. {str(gre)}")
                if thinking_message_sent:
                    await send_thinking_finished(websocket, "Processing failed due to a workflow loop.", success=False)
            except Exception as e:
                logger.error(f"Error during WebSocket communication or graph processing for session {session_id}: {e}", exc_info=True)
                await send_error_message(websocket, f"An server error occurred: {str(e)}")
                if thinking_message_sent:
                    await send_thinking_finished(websocket, "Processing failed due to an error.", success=False)
                # Depending on the error, you might want to break or continue.
            finally:
                # --- Ensure Thinking Finished is always signaled if it was started ---
                if thinking_message_sent: # Only send if thinking_started was called in this iteration
                    if operation_successful:
                        await send_thinking_finished(websocket, "Processing complete.", success=True)
                    else:
                        # Error specific message would have been sent by the except block
                        # This ensures the indicator is turned off.
                        await send_thinking_finished(websocket, "Processing finished with issues.", success=False)

    except Exception as e:
        # This catches errors outside the inner try/except, like initial accept or loop setup.
        logger.error(f"Unhandled exception in WebSocket handler for session {session_id}: {e}", exc_info=True)
        if websocket.client_state == WebSocketState.CONNECTED: # Check before sending
            try:
                await send_error_message(websocket, "A critical server error occurred with the WebSocket connection.")
                # Also ensure thinking is stopped if it was somehow globally active for this socket
                await send_thinking_finished(websocket, "Connection error.", success=False)
            except Exception: # If sending itself fails
                pass
    finally:
        logger.info(f"Closing WebSocket connection for session_id: {session_id}.")
        # FastAPI handles closing on context exit.
        # Any other cleanup for the session can go here.
