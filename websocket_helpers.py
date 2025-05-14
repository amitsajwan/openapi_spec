# core_logic/websocket_helpers.py
import logging
from fastapi import WebSocket, WebSocketDisconnect # WebSocketState removed from here
# If WebSocketState was ever needed directly, it would be:
# from starlette.websockets import WebSocketState
from typing import Dict, Any, Optional
import json

logger = logging.getLogger(__name__)

# Centralized function to send JSON messages to a WebSocket client.
async def _send_json_to_client(websocket: WebSocket, payload: Dict[str, Any]):
    """
    Sends a JSON payload to the connected WebSocket client.
    Relies on exception handling for disconnections or errors during sending.
    """
    try:
        # Directly attempt to send the JSON payload.
        # If the websocket is not connected, this will raise an exception.
        await websocket.send_json(payload)
    except WebSocketDisconnect:
        # This specific exception is caught if the client actively disconnected.
        logger.warning(f"WebSocket disconnected while trying to send message: {payload.get('type')}")
    except RuntimeError as e:
        # RuntimeError can occur for various reasons, e.g., sending on a closing socket.
        # The websocket.client_state attribute can still be useful for logging here if available.
        client_state_info = "unknown"
        if hasattr(websocket, 'client_state'):
            client_state_info = str(websocket.client_state)
        logger.warning(f"RuntimeError sending message type {payload.get('type')}: {e}. WebSocket state: {client_state_info}")
    except Exception as e:
        # Catch any other unexpected errors during sending.
        logger.error(f"Unexpected error sending message type {payload.get('type')}: {e}", exc_info=True)


async def send_status_update(websocket: WebSocket, status_event: str, message: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
    """
    Sends a general status update to the client.
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
    logger.debug(f"Sending intermediate_message from '{source}': {message[:100]}...")
    await _send_json_to_client(websocket, {"type": "intermediate_message", "data": {"message": message, "source": source}})

async def send_graph_update(websocket: WebSocket, graph_data: Dict[str, Any]):
    """Sends updated graph data to the client for rendering."""
    logger.debug("Sending graph_update")
    await _send_json_to_client(websocket, {"type": "graph_update", "data": {"graph": graph_data}})

async def send_api_response(websocket: WebSocket, response_data: Dict[str, Any]):
    """Sends the result of an API call execution to the client."""
    logger.debug(f"Sending api_response for operation: {response_data.get('operation_id')}")
    await _send_json_to_client(websocket, {"type": "api_response", "data": response_data})

async def send_error_message(websocket: WebSocket, error_message: str, details: Optional[Dict[str, Any]] = None):
    """Sends an error message to the client."""
    logger.error(f"Sending error_message: {error_message}")
    payload_content = {"message": error_message}
    if details:
        payload_content.update(details)
    await _send_json_to_client(websocket, {"type": "error", "data": payload_content})

async def send_final_response(websocket: WebSocket, message: str, data_payload: Optional[Dict[str, Any]] = None):
    """Sends a final, conclusive response to the client."""
    logger.debug(f"Sending final_response: {message[:100]}...")
    payload_content = {"message": message}
    if data_payload:
        payload_content.update(data_payload)
    await _send_json_to_client(websocket, {"type": "final_response", "data": payload_content})

