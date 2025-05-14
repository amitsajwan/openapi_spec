# core_logic/websocket_helpers.py
import logging
from fastapi import WebSocket, WebSocketDisconnect, WebSocketState
from typing import Dict, Any, Optional
import json

logger = logging.getLogger(__name__)

# Centralized function to send JSON messages to a WebSocket client.
async def _send_json_to_client(websocket: WebSocket, payload: Dict[str, Any]):
    """
    Sends a JSON payload to the connected WebSocket client.
    Handles potential disconnections or errors during sending.
    """
    if websocket.client_state == WebSocketState.CONNECTED:
        try:
            await websocket.send_json(payload)
        except WebSocketDisconnect:
            logger.warning(f"WebSocket disconnected while trying to send message: {payload.get('type')}")
        except RuntimeError as e:
            # This can happen if send is called on a closed/closing connection
            logger.warning(f"RuntimeError sending message type {payload.get('type')}: {e}. WebSocket state: {websocket.application_state}, {websocket.client_state}")
        except Exception as e:
            logger.error(f"Error sending message type {payload.get('type')}: {e}", exc_info=True)
    else:
        logger.warning(f"WebSocket not connected. State: {websocket.client_state}. Cannot send message: {payload.get('type')}")


async def send_status_update(websocket: WebSocket, status_event: str, message: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
    """
    Sends a general status update to the client.
    (No longer used for 'thinking_started' or 'thinking_finished')
    Args:
        websocket: The WebSocket connection.
        status_event: The specific status event.
        message: An optional message to display to the user related to this status.
        details: Optional additional details to include in the payload.
    """
    payload_content = {"event": status_event}
    if message:
        payload_content["message"] = message
    if details:
        payload_content.update(details)
    logger.debug(f"Sending status_update: event='{status_event}', message='{message}'")
    await _send_json_to_client(websocket, {"type": "status_update", "data": payload_content})

async def send_intermediate_message(websocket: WebSocket, message: str, source: str = "assistant"):
    """Sends an intermediate textual message to be displayed in the chat."""
    logger.debug(f"Sending intermediate_message from '{source}': {message[:100]}...") # Log snippet
    await _send_json_to_client(websocket, {"type": "intermediate_message", "data": {"message": message, "source": source}})

async def send_graph_update(websocket: WebSocket, graph_data: Dict[str, Any]):
    """Sends updated graph data to the client for rendering."""
    logger.debug("Sending graph_update") # Ensure graph_data is a dict ready for JSON
    await _send_json_to_client(websocket, {"type": "graph_update", "data": {"graph": graph_data}})

async def send_api_response(websocket: WebSocket, response_data: Dict[str, Any]):
    """Sends the result of an API call execution to the client."""
    logger.debug(f"Sending api_response for operation: {response_data.get('operation_id')}")
    await _send_json_to_client(websocket, {"type": "api_response", "data": response_data})

async def send_error_message(websocket: WebSocket, error_message: str, details: Optional[Dict[str, Any]] = None):
    """Sends an error message to the client."""
    logger.error(f"Sending error_message: {error_message}") # Log as error on backend
    payload_content = {"message": error_message}
    if details:
        payload_content.update(details)
    await _send_json_to_client(websocket, {"type": "error", "data": payload_content})

async def send_final_response(websocket: WebSocket, message: str, data_payload: Optional[Dict[str, Any]] = None):
    """Sends a final, conclusive response to the client."""
    logger.debug(f"Sending final_response: {message[:100]}...") # Log snippet
    payload_content = {"message": message}
    if data_payload: # This could include the final state, graph, etc.
        payload_content.update(data_payload)
    await _send_json_to_client(websocket, {"type": "final_response", "data": payload_content})

