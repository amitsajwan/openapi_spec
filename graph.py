# graph.py
import logging
from typing import Any, Dict, Literal

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver # For type hinting checkpointer

# Assuming models.py is in the same directory or accessible in PYTHONPATH
from models import BotState
# Assuming core_logic.py and router.py are accessible
from core_logic import OpenAPICoreLogic # OpenAPICoreLogic now correctly imports APIExecutor
from router import OpenAPIRouter

# For type hinting the api_executor_instance parameter in build_graph
# APIExecutor is defined in api_executor.py
try:
    from api_executor import APIExecutor # MODIFIED: Import directly from api_executor.py
except ImportError:
    # This fallback is less likely to be hit if api_executor.py is present
    logging.warning("graph.py: Could not import APIExecutor from api_executor.py for type hinting. Using Any.")
    APIExecutor = Any # Fallback type


logger = logging.getLogger(__name__)

def finalize_response(state: BotState) -> BotState:
    """
    Sets final_response from state.response if available, clears intermediate response,
    and prepares the state for ending the current graph turn.
    This node acts as the responder, ensuring a clean hand-off or end of a processing cycle.
    """
    tool_name = "responder"
    # Log entry point with current response values for debugging
    logger.info(f"Responder ({tool_name}): Entered. state.response='{str(state.response)[:100]}...', state.final_response='{str(state.final_response)[:100]}...'")
    state.update_scratchpad_reason(tool_name, f"Finalizing response. Initial state.response: '{str(state.response)[:100]}...'")

    if state.response: # If an intermediate response was set by the last node
        state.final_response = state.response
        logger.info(f"Responder ({tool_name}): Set final_response from state.response: '{str(state.final_response)[:200]}...'")
    elif not state.final_response: # Only set a default if final_response isn't already set (e.g., by an error message directly in BotState)
        state.final_response = "Processing complete. How can I help you further?"
        logger.warning(f"Responder ({tool_name}): state.response was empty/None. Using default final_response: '{state.final_response}'")
    else:
        # This case means state.response was falsey, but state.final_response already had a value.
        # This can happen if a node directly sets final_response due to a critical error.
        logger.info(f"Responder ({tool_name}): state.response was falsey, but final_response was already set. No change to final_response: '{str(state.final_response)[:200]}...'")

    # Clear fields for the next turn to avoid carry-over issues
    state.response = None         # Clear intermediate response, as it's now in final_response
    state.next_step = None      # Clear routing directive from the previous node, router will decide next
    state.intent = None         # Clear current intent, will be re-evaluated by router on new input
    # state.user_input is typically updated by the main loop receiving new input for the next cycle.
    # Scratchpad items like 'graph_to_send' are managed by the nodes that set them and cleared by main.py if needed.

    state.update_scratchpad_reason(tool_name, f"Final response set in state: {str(state.final_response)[:200]}...")
    logger.info(f"Responder ({tool_name}): Exiting. state.final_response='{str(state.final_response)[:100]}...', state.response='{state.response}'")
    return state


def build_graph(
    router_llm: Any,
    worker_llm: Any,
    api_executor_instance: APIExecutor, # Type hint should now work correctly
    checkpointer: BaseCheckpointSaver
) -> StateGraph: # Return type is StateGraph, but compiled it becomes a CompiledGraph or similar
    """
    Builds and compiles the LangGraph StateGraph for the OpenAPI agent (Planning Graph - Graph 1).

    Args:
        router_llm: The language model instance for routing user intent.
        worker_llm: The language model instance for core logic tasks (summarization, generation).
        api_executor_instance: An instance of APIExecutor (from api_executor.py) to be used by core logic.
        checkpointer: A LangGraph checkpointer instance (e.g., MemorySaver).

    Returns:
        A compiled LangGraph application.
    """
    logger.info("Building LangGraph graph (Planning Graph - Graph 1) with APIExecutor integration...")

    # Initialize core logic and router with their respective LLMs and dependencies
    # OpenAPICoreLogic now correctly receives the APIExecutor instance.
    core_logic = OpenAPICoreLogic(worker_llm=worker_llm, api_executor_instance=api_executor_instance)
    router_instance = OpenAPIRouter(router_llm=router_llm)

    builder = StateGraph(BotState)

    # --- Add All Nodes for the Planning Graph ---
    # Each node corresponds to a method in OpenAPICoreLogic or OpenAPIRouter, or the finalize_response function.
    builder.add_node("router", router_instance.route)
    builder.add_node("parse_openapi_spec", core_logic.parse_openapi_spec)
    builder.add_node("process_schema_pipeline", core_logic.process_schema_pipeline)
    builder.add_node("_generate_execution_graph", core_logic._generate_execution_graph)
    builder.add_node("verify_graph", core_logic.verify_graph)
    builder.add_node("refine_api_graph", core_logic.refine_api_graph)
    builder.add_node("describe_graph", core_logic.describe_graph)
    builder.add_node("get_graph_json", core_logic.get_graph_json)
    builder.add_node("answer_openapi_query", core_logic.answer_openapi_query)
    builder.add_node("interactive_query_planner", core_logic.interactive_query_planner)
    builder.add_node("interactive_query_executor", core_logic.interactive_query_executor)
    builder.add_node("setup_workflow_execution", core_logic.setup_workflow_execution)
    # Note: resume_workflow_with_payload is handled within interactive_query_planner/executor logic
    builder.add_node("handle_unknown", core_logic.handle_unknown)
    builder.add_node("handle_loop", core_logic.handle_loop)
    builder.add_node("responder", finalize_response) # The final node before ending the graph turn

    # --- Define Edges for the Planning Graph ---

    # Entry point: All interactions start at the router.
    builder.add_edge(START, "router")

    # Conditional edges from the router based on classified intent.
    # The router sets state.intent, which is used here for conditional routing.
    # Ensure all intents in OpenAPIRouter.AVAILABLE_INTENTS that map to nodes are actual nodes.
    router_targetable_intents: Dict[str, str] = {
        intent_val: intent_val for intent_val in OpenAPIRouter.AVAILABLE_INTENTS.__args__ # type: ignore
        if intent_val in builder.nodes # Ensure the intent name matches a defined node
    }
    for intent_val in OpenAPIRouter.AVAILABLE_INTENTS.__args__: # type: ignore
        if intent_val not in router_targetable_intents:
            # This warning helps catch configuration mismatches between router and graph definition.
            logger.warning(f"Router intent '{intent_val}' is defined in AVAILABLE_INTENTS but not added as a node in the graph.")

    builder.add_conditional_edges(
        "router", # Source node
        lambda state: state.intent, # Function to determine the condition (which intent was set)
        router_targetable_intents # Mapping from intent string to target node name
    )

    # Define how internal nodes (core logic nodes) decide the next step.
    # These nodes should set state.next_step to the name of the next node.
    def route_from_internal_node_state(state: BotState) -> str:
        """
        Determines the next node based on state.next_step set by an internal processing node.
        Provides fallback to 'responder' or 'handle_unknown' if next_step is invalid.
        """
        next_node_name = state.next_step
        current_node_info = state.intent or "Unknown (routing from internal node)" # Get context from intent if available
        
        if not next_node_name:
            logger.warning(f"Node '{current_node_info}' did not set state.next_step. Defaulting to 'responder'.")
            return "responder" # Default to responder if no next step is specified
        
        if next_node_name not in builder.nodes:
            logger.error(f"Node '{current_node_info}' tried to route to non-existent node '{next_node_name}'. Defaulting to 'handle_unknown'.")
            state.response = f"Error: System tried to navigate to an invalid internal step ('{next_node_name}')." # Inform user of internal error
            return "handle_unknown" # Route to error handling if target node doesn't exist
            
        logger.debug(f"Routing from internal node '{current_node_info}'. Next step decided by node: '{next_node_name}'")
        return next_node_name

    # List of nodes that determine their own next step by setting state.next_step
    nodes_that_set_next_step = [
        "parse_openapi_spec", "process_schema_pipeline", "_generate_execution_graph",
        "verify_graph", "refine_api_graph", "describe_graph", "get_graph_json",
        "answer_openapi_query", "interactive_query_planner", "interactive_query_executor",
        "setup_workflow_execution", "handle_unknown", "handle_loop"
        # "responder" is a terminal node for this graph's cycle, leading to END.
    ]

    # Create a mapping for all possible target nodes for these internal conditional edges.
    # This ensures that any node can, in principle, route to any other valid node if logic dictates.
    all_graph_nodes_as_targets: Dict[str, str] = {
        node_name: node_name for node_name in builder.nodes
    }
    # Ensure common fallbacks are always in the target map, even if not explicitly listed as nodes that set next_step.
    if "responder" not in all_graph_nodes_as_targets: all_graph_nodes_as_targets["responder"] = "responder"
    if "handle_unknown" not in all_graph_nodes_as_targets: all_graph_nodes_as_targets["handle_unknown"] = "handle_unknown"

    for source_node_name in nodes_that_set_next_step:
        if source_node_name in builder.nodes:
            builder.add_conditional_edges(
                source_node_name,
                route_from_internal_node_state, # Uses state.next_step
                all_graph_nodes_as_targets # Allows routing to any valid node
            )
        else:
            # This error indicates a mismatch in graph definition, crucial for debugging.
            logger.error(f"Configuration error: Node '{source_node_name}' listed in 'nodes_that_set_next_step' was not added to the graph builder.")

    # The "responder" node is the designated end point for a cycle of this planning graph.
    builder.add_edge("responder", END)

    try:
        # Compile the graph with the provided checkpointer.
        # debug=True can provide more verbose logging from LangGraph during execution.
        app = builder.compile(checkpointer=checkpointer, debug=True)
        logger.info("LangGraph graph (Planning Graph - Graph 1) compiled successfully with checkpointer and debug mode.")
        return app
    except Exception as e:
        logger.critical(f"LangGraph (Planning Graph - Graph 1) compilation failed: {e}", exc_info=True)
        raise # Re-raise the exception as compilation is critical.

