# graph.py
import logging
from typing import Any, Dict, Optional

from langgraph.graph import StateGraph, START, END
# from langgraph.checkpoint.base import BaseCheckpointSaver # No longer needed if not using checkpointer for Graph 1

from models import BotState
from router import OpenAPIRouter
from api_executor import APIExecutor # For type hinting

# Import the new refactored core logic classes
from core_logic.spec_processor import SpecProcessor
from core_logic.graph_generator import GraphGenerator # Assuming this will be created
from core_logic.interaction_handler import InteractionHandler # Assuming this will be created/consolidated
from core_logic.error_handlers import ErrorHandlers
# Assuming WorkflowLifecycleManager might be separate or part of InteractionHandler
# from core_logic.workflow_manager import WorkflowLifecycleManager

logger = logging.getLogger(__name__)

def finalize_response(state: BotState) -> BotState:
    """
    Ensures a final response is set for the user and cleans up state for the next turn.
    This node is typically the last step before END for most paths in the graph.
    """
    tool_name = "responder"
    logger.info(f"Responder ({tool_name}): Entered. Current state.response (from prev node): '{str(state.response)[:100]}', state.final_response: '{str(state.final_response)[:100]}'")
    state.update_scratchpad_reason(tool_name, f"Finalizing response. Initial state.response: '{str(state.response)[:100]}...'")

    # If a final_response is already set and meaningful, respect it.
    # Otherwise, promote state.response or set a default.
    if not state.final_response or state.final_response.strip() == "" or state.final_response == "Processing complete. How can I help you further?":
        if state.response and state.response.strip():
            state.final_response = state.response
            logger.info(f"Responder: Promoted state.response ('{state.response[:100]}...') to final_response.")
        else:
            state.final_response = "Processing complete. How can I help you further?"
            logger.info(f"Responder: Setting default final_response.")
    else:
        logger.info(f"Responder: Using pre-existing meaningful state.final_response ('{state.final_response[:100]}...').")

    # Clear intermediate response, intent, and next_step for the new turn.
    state.response = None
    state.intent = None
    state.next_step = None # Crucial to prevent loops if a node forgot to clear it

    state.update_scratchpad_reason(tool_name, f"Final response set in state: {str(state.final_response)[:200]}...")
    logger.info(f"Responder ({tool_name}): Exiting. state.final_response='{str(state.final_response)[:100]}', state.response (outgoing from this node)='{state.response}'")
    return state


def build_graph(
    router_llm: Any,
    worker_llm: Any,
    api_executor_instance: APIExecutor,
    checkpointer: Optional[Any] # For Graph 1's own state, if needed
) -> Any:
    logger.info("Building LangGraph (Planning Graph - Graph 1) with refactored core logic...")
    if checkpointer:
        logger.info("Compiling Planning Graph WITH checkpointer.")
    else:
        logger.info("Compiling Planning Graph WITHOUT checkpointer.")

    # Instantiate the refactored core logic components
    # These instances will hold the methods that were previously part of OpenAPICoreLogic
    spec_processor = SpecProcessor(worker_llm=worker_llm)
    graph_generator = GraphGenerator(worker_llm=worker_llm) # api_executor might not be needed here directly
    # InteractionHandler might need graph_generator, spec_processor, and a workflow_manager
    # workflow_manager = WorkflowLifecycleManager(api_executor_instance=api_executor_instance) # If created
    interaction_handler = InteractionHandler(
        worker_llm=worker_llm,
        graph_generator_instance=graph_generator,
        # workflow_control_instance=workflow_manager, # Pass if WorkflowManager is separate
        spec_processor_instance=spec_processor,
        api_executor_instance=api_executor_instance # If setup/resume are in InteractionHandler
    )
    error_handlers = ErrorHandlers()
    router_instance = OpenAPIRouter(router_llm=router_llm)

    builder = StateGraph(BotState)

    # Add nodes, mapping them to methods in the new classes
    builder.add_node("router", router_instance.route)

    # Spec Processing Nodes
    builder.add_node("parse_openapi_spec", spec_processor.parse_openapi_spec)
    # process_schema_pipeline now needs the graph_generator's generate method
    builder.add_node(
        "process_schema_pipeline",
        lambda state: spec_processor.process_schema_pipeline(state, graph_generator_func=graph_generator._generate_execution_graph)
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

    # Workflow Control Nodes (assuming they are in InteractionHandler for now)
    builder.add_node("setup_workflow_execution", interaction_handler.setup_workflow_execution)
    # resume_workflow_with_payload is typically triggered by user input via router -> interactive_query_planner -> interactive_query_executor
    # So, it might not be a direct node here unless there's another path to it.
    # For now, let's assume interactive_query_executor handles the call to a resume method.

    # Error Handling Nodes
    builder.add_node("handle_unknown", error_handlers.handle_unknown)
    builder.add_node("handle_loop", error_handlers.handle_loop)

    # Responder Node
    builder.add_node("responder", finalize_response)

    # --- Define Edges ---
    builder.add_edge(START, "router")

    # Conditional edges from the router based on intent
    # Ensure all intents in OpenAPIRouter.AVAILABLE_INTENTS are mapped if they are nodes
    router_targetable_intents: Dict[str, str] = {
        intent_val: intent_val for intent_val in OpenAPIRouter.AVAILABLE_INTENTS.__args__ # type: ignore
        if intent_val in builder.nodes
    }
    for intent_val_literal in OpenAPIRouter.AVAILABLE_INTENTS.__args__: # type: ignore
        intent_val_str = str(intent_val_literal)
        if intent_val_str not in router_targetable_intents and intent_val_str not in ["responder"]: # responder is a common end
            logger.warning(f"Router intent '{intent_val_str}' is defined in AVAILABLE_INTENTS but not added as a node in the graph or explicitly handled.")

    builder.add_conditional_edges(
        "router",
        lambda state: state.intent,
        router_targetable_intents
    )

    # Conditional edges from nodes that determine their own next step
    def route_from_internal_node_state(state: BotState) -> str:
        next_node_name = state.next_step
        # current_node_info = state.intent or "Unknown (routing from internal node)" # state.intent is cleared by responder
        
        # It's better to get the current node name from the graph's internal state if possible,
        # or rely on the fact that this function is called *after* a node has run.
        # For now, we'll assume state.next_step is reliably set.

        if not next_node_name:
            # This case should ideally be minimized. Nodes should set next_step or be terminal.
            logger.warning(f"A node (previous to this routing decision) did not set state.next_step. Defaulting to 'responder'.")
            return "responder"

        if next_node_name not in builder.nodes:
            logger.error(f"A node tried to route to a non-existent next_step '{next_node_name}'. Defaulting to 'handle_unknown'.")
            state.response = f"Error: System tried to navigate to an invalid internal step ('{next_node_name}')."
            # Ensure final_response is also set if this is a critical error path
            state.final_response = state.response
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
        "handle_unknown", "handle_loop"
        # "responder" is terminal, so not listed here.
    ]

    all_graph_nodes_as_targets: Dict[str, str] = {
        node_name: node_name for node_name in builder.nodes
    }
    # Ensure common fallbacks are in the target map if not already explicitly nodes
    if "responder" not in all_graph_nodes_as_targets: all_graph_nodes_as_targets["responder"] = "responder"
    if "handle_unknown" not in all_graph_nodes_as_targets: all_graph_nodes_as_targets["handle_unknown"] = "handle_unknown"


    for source_node_name in nodes_that_set_next_step:
        if source_node_name in builder.nodes:
            builder.add_conditional_edges(
                source_node_name,
                route_from_internal_node_state,
                all_graph_nodes_as_targets # All nodes are potential targets from these
            )
        else:
            logger.error(f"Configuration error: Node '{source_node_name}' listed in 'nodes_that_set_next_step' was not added to the graph builder.")

    # Terminal edge for the responder
    builder.add_edge("responder", END)

    # Compile the graph
    try:
        if checkpointer:
            app = builder.compile(checkpointer=checkpointer, debug=True)
            logger.info("LangGraph (Planning Graph - Graph 1) compiled successfully WITH checkpointer and debug mode using refactored logic.")
        else:
            app = builder.compile(debug=True)
            logger.info("LangGraph (Planning Graph - Graph 1) compiled successfully WITHOUT checkpointer and debug mode using refactored logic.")
        return app
    except Exception as e:
        logger.critical(f"LangGraph (Planning Graph - Graph 1) compilation failed with refactored logic: {e}", exc_info=True)
        raise
