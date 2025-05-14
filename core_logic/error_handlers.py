# core_logic/error_handlers.py
import logging
from models import BotState

logger = logging.getLogger(__name__)

class ErrorHandlers:
    def handle_unknown(self, state: BotState) -> BotState:
        tool_name = "handle_unknown"
        if not state.response or "error" not in str(state.response).lower():
            state.response = "I'm not sure how to process that request. Could you please rephrase it, or provide an OpenAPI specification if you haven't already?"
        state.update_scratchpad_reason(tool_name, f"Handling unknown input or situation. Final response to be: {state.response}")
        state.next_step = "responder"
        return state

    def handle_loop(self, state: BotState) -> BotState:
        tool_name = "handle_loop"
        state.response = "It seems we're stuck in a processing loop. Please try rephrasing your request or starting over with the OpenAPI specification."
        state.loop_counter = 0 
        state.update_scratchpad_reason(tool_name, "Loop detected, routing to responder with a loop message.")
        state.next_step = "responder"
        return state
