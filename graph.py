# graph.py
import logging
from typing import Any, Dict, Optional

from langgraph.graph import StateGraph, START, END
# No longer importing BaseCheckpointSaver here if Graph 1 doesn't use persistence directly in this file.

from models import BotState
from router import OpenAPIRouter # Assumes router.py is in the same top-level directory
from api_executor import APIExecutor # For type hinting

# Import the new refactored core logic classes
from core_logic.spec_processor import SpecProcessor
from core_logic.graph_generator import GraphGenerator
from core_logic.interaction_handler import InteractionHandler
from core_logic.error_handlers import ErrorHandlers
# If you create a WorkflowLifecycleManager, it would be imported here too.

logger = logging.getLogger(__name__)

def finalize_response(state: BotState) -> BotState:
    """
    Ensures a final response is set for the user and cleans up state for the next turn.
    This node is typically the last step before END for most paths in the graph.
    It promotes state.response to state.final_response if no specific final_response
    was set by a preceding node.
    """
    tool_name = "responder"
    logger.info(
        f"Responder ({tool_name}): Entered. "
        f"Current state.response (from prev node): '{str(state.response)[:100]}', "
        f"state.final_response (incoming): '{str(state.final_response)[:100]}'"
    )
    # The scratchpad reason for entering this node should have been set by the node *before* it.
    # This node will set its own reason for the actions it takes.
    state.update_scratchpad_reason(
        tool_name,
        f"Finalizing response. Initial state.response: '{str(state.response)[:100]}...'",
    )

    # If a final_response is already set by a previous node and is meaningful, respect it.
    # Otherwise, promote state.response or set a default.
    if not state.final_response or \
       state.final_response.strip() == "" or \
       state.final_response == "Processing complete. How can I help you further?": # Check against default
        if state.response and state.response.strip():
            state.final_response = state.response
            logger.info(
                f"Responder: Promoted state.response ('{state.response[:100]}...') to final_response."
            )
        else:
            # If state.response is also empty or None, set a generic default final message.
            state.final_response = "Processing complete. How can I help you further?"
            logger.info("Responder: Setting default final_response.")
    else:
        logger.info(
            f"Responder: Using pre-existing meaningful state.final_response ('{state.final_response[:100]}...')."
        )

    # Clear intermediate response, intent, and next_step for the new turn.
    # This ensures that the state is clean for the next user input.
    state.response = None
    state.intent = None
    state.next_step = None # Crucial to prevent unintended routing in the next turn.

    state.update_scratchpad_reason(
        tool_name, f"Final response set in state: {str(state.final_response)[:200]}..."
    )
    logger.info(
        f"Responder ({tool_name}): Exiting. "
        f"state.final_response='{str(state.final_response)[:100]}', "
        f"state.response (outgoing from this node)='{state.response}'"
    )
    return state


def build_graph(
    router_llm: Any,
    worker_llm: Any,
    api_executor_instance: APIExecutor, # Used by InteractionHandler for workflow setup context
    checkpointer: Optional[Any] # For Graph 1's own state persistence, if enabled
) -> Any:
    """
    Builds the main planning LangGraph (Graph 1) using the refactored core logic components.

    Args:
        router_llm: The language model for the OpenAPIRouter.
        worker_llm: The language model for core logic tasks (summaries, generation, etc.).
        api_executor_instance: Instance of APIExecutor, passed for context (e.g., to workflow setup).
        checkpointer: Optional checkpointer for saving/resuming Graph 1's state.

    Returns:
        A compiled LangGraph application.
    """
    logger.info("Building LangGraph (Planning Graph - Graph 1) with refactored core logic components...")
    if checkpointer:
        logger.info("Compiling Planning Graph WITH checkpointer.")
    else:
        logger.info("Compiling Planning Graph WITHOUT checkpointer.")

    # Instantiate the refactored core logic components
    spec_processor = SpecProcessor(worker_llm=worker_llm)
    graph_generator = GraphGenerator(worker_llm=worker_llm)
    # InteractionHandler needs instances of other components it might delegate to.
    interaction_handler = InteractionHandler(
        worker_llm=worker_llm,
        graph_generator_instance=graph_generator,
        spec_processor_instance=spec_processor,
        api_executor_instance=api_executor_instance # For context, not direct execution by IH
    )
    error_handlers = ErrorHandlers()
    router_instance = OpenAPIRouter(router_llm=router_llm)

    # Initialize the StateGraph with BotState
    builder = StateGraph(BotState)

    # --- Add nodes to the graph, mapping them to methods in the new classes ---
    builder.add_node("router", router_instance.route)

    # Specification Processing Nodes
    builder.add_node("parse_openapi_spec", spec_processor.parse_openapi_spec)
    # process_schema_pipeline needs the graph_generator's method for graph creation.
    # We use a lambda to pass the correct method from the graph_generator instance.
    builder.add_node(
        "process_schema_pipeline",
        lambda st: spec_processor.process_schema_pipeline(
            st, graph_generator_func=graph_generator._generate_execution_graph
        ),
    )

    # Graph Generation and Management Nodes
    builder.add_node("_generate_execution_graph", graph_generator._generate_execution_graph)
    builder.add_node("verify_graph", graph_generator.verify_graph)
    builder.add_node("refine_api_graph", graph_generator.refine_api_graph)

    # Interaction and Querying Nodes
    builder.add_node("describe_graph", interaction_handler.describe_graph)
    builder.add_node("get_graph_json", interaction_handler.get_graph_json)
    builder.add_node("answer_openapi_query", interaction_handler.answer_openapi_query)
    builder.add_node("interactive_query_planner", interaction_handler.interactive_query_planner)
    builder.add_node("interactive_query_executor", interaction_handler.interactive_query_executor)

    # Workflow Control Nodes (handled by InteractionHandler)
    builder.add_node("setup_workflow_execution", interaction_handler.setup_workflow_execution)
    # Note: resume_workflow_with_payload is typically part of the interactive_query_executor flow,
    # triggered by user input containing payload data, rather than a direct node from the router.

    # Error Handling Nodes
    builder.add_node("handle_unknown", error_handlers.handle_unknown)
    builder.add_node("handle_loop", error_handlers.handle_loop)

    # Responder Node (finalizes the response for the turn)
    builder.add_node("responder", finalize_response)

    # --- Define Edges for the graph ---
    builder.add_edge(START, "router") # Always start with the router

    # Conditional edges from the router based on determined intent
    # Ensure all intents in OpenAPIRouter.AVAILABLE_INTENTS are valid nodes in the graph
    router_targetable_intents: Dict[str, str] = {
        # Ensure intent_val is a string, as __args__ might give Literal types
        str(intent_val): str(intent_val)
        for intent_val in OpenAPIRouter.AVAILABLE_INTENTS.__args__ # type: ignore
        if str(intent_val) in builder.nodes # Check if the intent name matches a defined node
    }

    # Log any intents defined in the router that don't have a corresponding node
    for intent_val_literal in OpenAPIRouter.AVAILABLE_INTENTS.__args__: # type: ignore
        intent_val_str = str(intent_val_literal)
        # 'responder' is a common terminal node, not usually a direct router target for starting new logic.
        if intent_val_str not in router_targetable_intents and intent_val_str != "responder":
            logger.warning(
                f"Router intent '{intent_val_str}' is defined in AVAILABLE_INTENTS "
                "but not added as a node in the graph or explicitly handled as a target."
            )
    builder.add_conditional_edges(
        "router",
        lambda state: state.intent, # Route based on the 'intent' field in BotState
        router_targetable_intents,
    )

    # Conditional edges from nodes that determine their own next step via state.next_step
    def route_from_internal_node_state(state: BotState) -> str:
        """Determines the next node based on state.next_step set by the previous node."""
        next_node_name = state.next_step
        
        if not next_node_name:
            logger.warning(
                "A node (previous to this routing decision) did not set state.next_step. "
                "Defaulting to 'responder' to prevent graph from stalling."
            )
            return "responder"

        if next_node_name not in builder.nodes:
            logger.error(
                f"A node tried to route to a non-existent next_step '{next_node_name}'. "
                "Defaulting to 'handle_unknown' to manage the error."
            )
            # It's good practice for the node itself to set state.response if it detects this,
            # but we can set a generic error here as a fallback.
            state.response = (
                state.response or # Preserve existing response if it's an error
                f"Error: System tried to navigate to an invalid internal step ('{next_node_name}')."
            )
            state.final_response = state.response # Ensure this error is seen by user
            return "handle_unknown"

        logger.debug(f"Routing from internal node. Next step decided by node: '{next_node_name}'")
        return next_node_name

    # List of nodes that internally decide their next_step
    nodes_that_set_next_step = [
        "parse_openapi_spec", "process_schema_pipeline",
        "_generate_execution_graph", "verify_graph", "refine_api_graph",
        "describe_graph", "get_graph_json", "answer_openapi_query",
        "interactive_query_planner", "interactive_query_executor",
        "setup_workflow_execution",
        "handle_unknown", "handle_loop",
        # "responder" is terminal, so it does not set a next_step to another logic node.
    ]

    # All nodes in the graph are potential targets for conditional routing
    all_graph_nodes_as_targets: Dict[str, str] = {
        node_name: node_name for node_name in builder.nodes
    }
    # Ensure common fallbacks are in the target map if not already explicitly nodes (though they are)
    if "responder" not in all_graph_nodes_as_targets:
        all_graph_nodes_as_targets["responder"] = "responder"
    if "handle_unknown" not in all_graph_nodes_as_targets:
        all_graph_nodes_as_targets["handle_unknown"] = "handle_unknown"

    for source_node_name in nodes_that_set_next_step:
        if source_node_name in builder.nodes:
            builder.add_conditional_edges(
                source_node_name,
                route_from_internal_node_state, # Routing function
                all_graph_nodes_as_targets, # Possible destinations
            )
        else:
            # This indicates a configuration mismatch between the list and actual nodes.
            logger.error(
                f"Configuration error: Node '{source_node_name}' listed in "
                "'nodes_that_set_next_step' was not added to the graph builder."
            )

    # Terminal edge for the responder node, leading to the end of the graph flow for the turn.
    builder.add_edge("responder", END)

    # Compile the graph
    try:
        compiled_app = builder.compile(checkpointer=checkpointer, debug=True) # Enable debug for more logs
        logger.info(
            "LangGraph (Planning Graph - Graph 1) compiled successfully "
            "WITH checkpointer (if provided) and debug mode, using refactored core logic."
        )
        return compiled_app
    except Exception as e:
        logger.critical(
            f"LangGraph (Planning Graph - Graph 1) compilation failed with refactored logic: {e}",
            exc_info=True,
        )
        raise # Re-raise the exception to halt startup if compilation fails.
