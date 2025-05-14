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
    Sends a status update to the client, typically for UI state changes like thinking indicators.
    Args:
        websocket: The WebSocket connection.
        status_event: The specific status event (e.g., "thinking_started", "thinking_finished").
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

async def send_thinking_started(websocket: WebSocket, message: Optional[str] = "Processing your request..."):
    """Signals the client that a potentially long operation has started."""
    await send_status_update(websocket, "thinking_started", message)

async def send_thinking_finished(websocket: WebSocket, message: Optional[str] = "Processing complete.", success: Optional[bool] = None):
    """
    Signals the client that a potentially long operation has finished.
    Args:
        success: Optional boolean to indicate if the operation was successful.
    """
    details = {}
    if success is not None:
        details["success"] = success
    await send_status_update(websocket, "thinking_finished", message, details=details)

async def send_intermediate_message(websocket: WebSocket, message: str, source: str = "assistant"):
    """Sends an intermediate textual message to be displayed in the chat."""
    logger.debug(f"Sending intermediate_message from '{source}': {message[:100]}...") # Log snippet
    await _send_json_to_client(websocket, {"type": "intermediate_message", "data": {"message": message, "source": source}})

async def send_graph_update(websocket: WebSocket, graph_data: Dict[str, Any]):
    """Sends updated graph data to the client for rendering."""
    logger.debug("Sending graph_update")
    await _send_json_to_client(websocket, {"type": "graph_update", "data": {"graph": graph_data}})

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
    if data_payload:
        payload_content.update(data_payload)
    await _send_json_to_client(websocket, {"type": "final_response", "data": payload_content})

# Optional: ConnectionManager if you need to manage multiple connections centrally
# class ConnectionManager:
#     def __init__(self):
#         self.active_connections: Dict[str, WebSocket] = {} # client_id: WebSocket
#
#     async def connect(self, websocket: WebSocket, client_id: str):
#         await websocket.accept()
#         self.active_connections[client_id] = websocket
#         logger.info(f"Client {client_id} connected. Total: {len(self.active_connections)}")
#
#     def disconnect(self, client_id: str):
#         if client_id in self.active_connections:
#             # Ensure the websocket is closed before deleting
#             # ws = self.active_connections.pop(client_id)
#             # try:
#             #     # await ws.close() # This might error if already closed
#             # except Exception:
#             #     pass
#             del self.active_connections[client_id] # Simpler removal
#             logger.info(f"Client {client_id} disconnected. Total: {len(self.active_connections)}")
#
#     async def get_connection(self, client_id: str) -> Optional[WebSocket]:
#         return self.active_connections.get(client_id)

# manager = ConnectionManager() # Instantiate if used globally
