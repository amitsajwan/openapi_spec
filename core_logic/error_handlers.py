# core_logic/error_handlers.py
import logging
from models import BotState

logger = logging.getLogger(__name__)

class ErrorHandlers:
    """
    Contains methods to handle common error states or unexpected situations
    in the planning graph, such as unknown user input or detected loops.
    """

    def __init__(self):
        """Initializes the ErrorHandlers."""
        logger.info("ErrorHandlers initialized.")

    def _queue_intermediate_message(self, state: BotState, msg: str):
        """
        Helper to queue messages for the UI and set the current response in BotState.
        Ensures that the user gets updates.
        """
        if "intermediate_messages" not in state.scratchpad:
            state.scratchpad["intermediate_messages"] = []
        # Avoid queuing exact same consecutive message if state.response was already it
        if (
            not state.scratchpad["intermediate_messages"]
            or state.scratchpad["intermediate_messages"][-1] != msg
        ):
            state.scratchpad["intermediate_messages"].append(msg)
        state.response = msg # Update current response for logging or if it's the last one


    def handle_unknown(self, state: BotState) -> BotState:
        """
        Handles situations where the user's input is unclear or the system
        doesn't know how to proceed.
        Sets a generic error message and routes to the responder.
        """
        tool_name = "handle_unknown"
        
        # Check if a more specific error response has already been set by a previous node
        # that decided to route to handle_unknown.
        current_response = state.response
        if not current_response or "error" not in str(current_response).lower():
            # If no specific error message is present, set a generic one.
            self._queue_intermediate_message(
                state,
                "I'm not sure how to process that request. Could you please rephrase it, "
                "or provide an OpenAPI specification if you haven't already?",
            )
        else:
            # If state.response already contains an error message, preserve it.
            # This allows other nodes to set a specific error and then route here.
            self._queue_intermediate_message(state, current_response)

        state.update_scratchpad_reason(
            tool_name,
            f"Handling unknown input or situation. Final response to be: {state.response}",
        )
        state.next_step = "responder"
        return state

    def handle_loop(self, state: BotState) -> BotState:
        """
        Handles detected processing loops within the graph.
        Sets a message informing the user about the loop and routes to the responder.
        """
        tool_name = "handle_loop"
        self._queue_intermediate_message(
            state,
            "It seems we're stuck in a processing loop. Please try rephrasing your request "
            "or starting over with the OpenAPI specification.",
        )
        state.loop_counter = 0  # Reset loop counter after handling
        state.update_scratchpad_reason(
            tool_name, "Loop detected, routing to responder with a loop message."
        )
        state.next_step = "responder"
        return state
