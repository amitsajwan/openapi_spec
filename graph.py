# graph.py
import logging
from typing import Any, Dict, Optional

from langgraph.graph import StateGraph, START, END
import asyncio # Ensure asyncio is imported for async def

from models import BotState
from router import OpenAPIRouter # Ensure OpenAPIRouter is imported
from api_executor import APIExecutor # Ensure APIExecutor is imported

# Ensure core_logic components are correctly imported
from core_logic.spec_processor import SpecProcessor
from core_logic.graph_generator import GraphGenerator
from core_logic.interaction_handler import InteractionHandler
from core_logic.error_handlers import ErrorHandlers

logger = logging.getLogger(__name__)

def finalize_response(state: BotState) -> BotState:
    """Ensures a final response is set and cleans up state for the next turn."""
    tool_name = "responder"
    logger.info(
        f"Responder ({tool_name}): Entered. "
        f"Current state.response (from prev node): '{str(state.response)[:100]}', "
        f"state.final_response (incoming): '{str(state.final_response)[:100]}'"
    )
    state.update_scratchpad_reason(
        tool_name,
        f"Finalizing response. Initial state.response: '{str(state.response)[:100]}...'",
    )

    if not state.final_response or \
       state.final_response.strip() == "" or \
       state.final_response == "Processing complete. How can I help you further?":
        if state.response and state.response.strip():
            state.final_response = state.response
            logger.info(
                f"Responder: Promoted state.response ('{state.response[:100]}...') to final_response."
            )
        else:
            state.final_response = "Processing complete. How can I help you further?"
            logger.info("Responder: Setting default final_response.")
    else:
        logger.info(
            f"Responder: Using pre-existing meaningful state.final_response ('{state.final_response[:100]}...')."
        )

    state.response = None
    state.intent = None # Clear intent for the next turn
    state.next_step = None # Clear next_step

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
    utility_llm: Any,
    api_executor_instance: APIExecutor,
    checkpointer: Optional[Any]
) -> Any:
    logger.info("Building LangGraph (Planning Graph - Graph 1) with refactored core logic and utility LLM...")
    if checkpointer:
        logger.info("Compiling Planning Graph WITH checkpointer.")
    else:
        logger.info("Compiling Planning Graph WITHOUT checkpointer.")

    # Instantiate core logic components
    spec_processor = SpecProcessor(worker_llm=worker_llm, utility_llm=utility_llm)
    graph_generator = GraphGenerator(worker_llm=worker_llm)
    # Ensure InteractionHandler is initialized with all its required instances
    interaction_handler = InteractionHandler(
        worker_llm=worker_llm,
        graph_generator_instance=graph_generator,
        spec_processor_instance=spec_processor,
        api_executor_instance=api_executor_instance # Passed for context or future use by IH
    )
    error_handlers = ErrorHandlers()
    router_instance = OpenAPIRouter(router_llm=router_llm)

    builder = StateGraph(BotState)

    # Add nodes
    builder.add_node("router", router_instance.route)
    builder.add_node("parse_openapi_spec", spec_processor.parse_openapi_spec)

    # Async wrapper for process_schema_pipeline
    async def _process_schema_pipeline_wrapper(state: BotState) -> BotState:
        return await spec_processor.process_schema_pipeline(
            state, graph_generator_func=graph_generator._generate_execution_graph
        )
    builder.add_node("process_schema_pipeline", _process_schema_pipeline_wrapper)

    builder.add_node("_generate_execution_graph", graph_generator._generate_execution_graph)
    builder.add_node("verify_graph", graph_generator.verify_graph)
    builder.add_node("refine_api_graph", graph_generator.refine_api_graph)
    builder.add_node("describe_graph", interaction_handler.describe_graph)
    builder.add_node("get_graph_json", interaction_handler.get_graph_json)
    builder.add_node("answer_openapi_query", interaction_handler.answer_openapi_query)

    # Async wrapper for interactive_query_planner (if it needs to call async sub-steps)
    async def _interactive_query_planner_wrapper(state: BotState) -> BotState:
        return await interaction_handler.interactive_query_planner(state)
    builder.add_node("interactive_query_planner", _interactive_query_planner_wrapper)

    # Async wrapper for interactive_query_executor
    async def _interactive_query_executor_wrapper(state: BotState) -> BotState:
        return await interaction_handler.interactive_query_executor(state)
    builder.add_node("interactive_query_executor", _interactive_query_executor_wrapper)

    builder.add_node("setup_workflow_execution", interaction_handler.setup_workflow_execution)

    # Add new node for initiating load test
    async def _initiate_load_test_wrapper(state: BotState) -> BotState:
        return await interaction_handler.handle_initiate_load_test(state)
    builder.add_node("initiate_load_test", _initiate_load_test_wrapper)

    builder.add_node("handle_unknown", error_handlers.handle_unknown)
    builder.add_node("handle_loop", error_handlers.handle_loop)
    builder.add_node("responder", finalize_response)

    # Define Edges
    builder.add_edge(START, "router")

    # Ensure all intents in OpenAPIRouter.AVAILABLE_INTENTS that are actual nodes
    # are targetable by the router.
    router_targetable_intents: Dict[str, str] = {
        str(intent_val): str(intent_val)
        for intent_val in OpenAPIRouter.AVAILABLE_INTENTS.__args__ # type: ignore
        if str(intent_val) in builder.nodes # Check if intent name is an added node name
    }
    # Log any intents defined in AVAILABLE_INTENTS but not added as nodes (excluding 'responder' which is special)
    for intent_val_literal in OpenAPIRouter.AVAILABLE_INTENTS.__args__: # type: ignore
        intent_val_str = str(intent_val_literal)
        if intent_val_str not in router_targetable_intents and intent_val_str != "responder":
            logger.warning(
                f"Router intent '{intent_val_str}' is defined in OpenAPIRouter.AVAILABLE_INTENTS "
                "but not added as a node in the graph or explicitly handled as a target by the router."
            )
    builder.add_conditional_edges("router", lambda state: state.intent, router_targetable_intents)


    # Define how nodes that set `state.next_step` should route
    def route_from_internal_node_state(state: BotState) -> str:
        next_node_name = state.next_step
        if not next_node_name:
            logger.warning(f"Node '{state.intent or 'Unknown Last Intent'}' did not set state.next_step. Defaulting to 'responder'.")
            return "responder"
        if next_node_name not in builder.nodes:
            logger.error(f"Node '{state.intent or 'Unknown Last Intent'}' tried to route to non-existent next_step '{next_node_name}'. Defaulting to 'handle_unknown'.")
            # Update state to reflect this system error to the user
            state.response = (state.response or "") + f" Critical system error: Tried to navigate to an invalid internal step ('{next_node_name}'). Please report this."
            state.final_response = state.response
            return "handle_unknown"
        logger.debug(f"Routing from internal node. Last intent: '{state.intent}', Next step decided by node: '{next_node_name}'")
        return next_node_name

    # List of nodes that are expected to set `state.next_step` to control flow
    nodes_that_set_next_step = [
        "parse_openapi_spec", "process_schema_pipeline",
        "_generate_execution_graph", "verify_graph", "refine_api_graph",
        "describe_graph", "get_graph_json", "answer_openapi_query",
        "interactive_query_planner", "interactive_query_executor",
        "setup_workflow_execution",
        "initiate_load_test", # New node added to this list
        "handle_unknown", "handle_loop",
    ]

    # Ensure all nodes in the graph are potential targets for conditional routing
    all_graph_nodes_as_targets: Dict[str, str] = { node_name: node_name for node_name in builder.nodes }
    # Ensure 'responder' and 'handle_unknown' are always valid targets, even if not explicitly in every node's next_step list
    if "responder" not in all_graph_nodes_as_targets: all_graph_nodes_as_targets["responder"] = "responder"
    if "handle_unknown" not in all_graph_nodes_as_targets: all_graph_nodes_as_targets["handle_unknown"] = "handle_unknown"


    for source_node_name in nodes_that_set_next_step:
        if source_node_name in builder.nodes:
            builder.add_conditional_edges(
                source_node_name,
                route_from_internal_node_state,
                all_graph_nodes_as_targets # All nodes are potential next steps
            )
        else:
            # This should not happen if lists are kept in sync with add_node calls
            logger.error(f"Configuration error: Node '{source_node_name}' in 'nodes_that_set_next_step' list was not added to the graph builder.")

    # Final sink node
    builder.add_edge("responder", END)

    try:
        compiled_app = builder.compile(checkpointer=checkpointer, debug=True)
        logger.info("LangGraph (Planning Graph - Graph 1) compiled successfully.")
        return compiled_app
    except Exception as e:
        logger.critical(f"LangGraph (Planning Graph - Graph 1) compilation failed: {e}", exc_info=True)
        raise
