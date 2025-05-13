import asyncio
import logging
from typing import Any, Callable, Awaitable, Dict, Optional, Tuple, Union # Added Any

from langgraph.graph import StateGraph, START, END # Removed CompiledGraph import
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Interrupt

# Assuming models.py is in the same directory or accessible via PYTHONPATH
from models import GraphOutput, Node, Edge, InputMapping, OutputMapping, ExecutionGraphState
# Assuming api_executor.py is in the same directory or accessible
from api_executor import APIExecutor # This should be your APIExecutor class

logger = logging.getLogger(__name__)

def _get_value_from_path(data_dict: Optional[Dict[str, Any]], path: str) -> Any:
    """
    Simple helper to get a value from a nested dictionary using dot-notation.
    Example: "user.address.city" or "items.0.name"
    Returns None if path is invalid, data_dict is None, or value not found.
    """
    if not path or not isinstance(data_dict, dict):
        return None

    if path.startswith("$."):
        path = path[2:] # Remove $. prefix if present

    keys = path.split('.')
    current_val = data_dict
    for key_part in keys:
        if isinstance(current_val, dict):
            if key_part not in current_val:
                logger.debug(f"Path '{path}' not found at key '{key_part}' in dict: {str(current_val)[:100]}...")
                return None
            current_val = current_val.get(key_part)
        elif isinstance(current_val, list):
            try:
                idx = int(key_part)
                if 0 <= idx < len(current_val):
                    current_val = current_val[idx]
                else:
                    logger.debug(f"Path '{path}' index '{idx}' out of bounds for list: {str(current_val)[:100]}...")
                    return None
            except ValueError: # If key_part is not a valid integer for list index
                logger.debug(f"Path '{path}' expected integer index for list, got '{key_part}'. List: {str(current_val)[:100]}...")
                return None
        else: # Current_val is not a dict or list, so cannot traverse further
            logger.debug(f"Path '{path}' encountered non-traversable type '{type(current_val)}' at key part '{key_part}'. Value: {str(current_val)[:100]}...")
            return None
    return current_val

def _set_value_by_path(data_dict: Dict[str, Any], path: str, value: Any):
    """
    Simple helper to set a value in a nested dictionary using dot-notation (e.g., "user.address.city").
    Creates intermediate dicts if they don't exist.
    Assumes `path` does not start with "body." anymore, as it's relative to `data_dict`.
    """
    if not path: # Cannot set value if path is empty
        logger.warning("Cannot set value with empty path within _set_value_by_path.")
        return

    keys = path.split('.')
    current_level = data_dict
    for i, key in enumerate(keys[:-1]): # Iterate up to the second to last key
        if key not in current_level or not isinstance(current_level.get(key), dict):
            current_level[key] = {} # Create a new dict if key doesn't exist or not a dict
        current_level = current_level[key]

    if keys: # Ensure there's at least one key
        current_level[keys[-1]] = value # Set the value at the last key
    else:
        logger.warning(f"Path was effectively empty for _set_value_by_path after splitting. Original dict might be directly modified if it was the target.")


class ExecutionGraphDefinition:
    """
    Defines and compiles "Graph 2" (Execution LangGraph) based on an execution plan.
    It takes a plan (GraphOutput from models.py), an APIExecutor, and builds a runnable LangGraph.
    """
    def __init__(self, graph_execution_plan: GraphOutput, api_executor: APIExecutor):
        self.graph_plan = graph_execution_plan
        self.api_executor = api_executor
        self.runner: Any = self._build_and_compile_graph()
        logger.info("ExecutionGraphDefinition initialized and Graph 2 compiled.")

    def _resolve_placeholders(self, template_string: str, state: ExecutionGraphState) -> str:
        """Resolves placeholders like {{key}} in a string using data from ExecutionGraphState."""
        if not isinstance(template_string, str):
            return str(template_string)

        resolved_string = template_string
        extracted_ids_dict = state.extracted_ids if isinstance(state.extracted_ids, dict) else {}
        initial_input_dict = state.initial_input if isinstance(state.initial_input, dict) else {}

        for key, value in extracted_ids_dict.items():
            if value is not None:
                resolved_string = resolved_string.replace(f"{{{key}}}", str(value))

        for key, value in initial_input_dict.items():
            if value is not None:
                 resolved_string = resolved_string.replace(f"{{{key}}}", str(value))
        return resolved_string

    def _apply_input_mappings(
        self,
        node_definition: Node,
        state: ExecutionGraphState,
        resolved_path_params: Dict[str, Any],
        resolved_query_params: Dict[str, Any],
        resolved_body_payload_for_field_mapping: Dict[str, Any],
        resolved_headers: Dict[str, Any]
    ) -> None:
        """Applies input mappings to prepare parameters for an API call."""
        if not node_definition.input_mappings:
            return

        current_extracted_ids = state.extracted_ids if isinstance(state.extracted_ids, dict) else {}
        logger.debug(f"Node '{node_definition.effective_id}': Applying {len(node_definition.input_mappings)} input mappings. Current extracted_ids: {current_extracted_ids}")

        for mapping in node_definition.input_mappings:
            source_value = _get_value_from_path(current_extracted_ids, mapping.source_data_path)

            if source_value is None:
                logger.warning(f"Node '{node_definition.effective_id}': InputMapping - Could not find source data for path '{mapping.source_data_path}' from extracted_ids. Skipping mapping for '{mapping.target_parameter_name}'.")
                continue

            logger.debug(f"Node '{node_definition.effective_id}': InputMapping - Found value for source path '{mapping.source_data_path}'. Target: '{mapping.target_parameter_name}' in '{mapping.target_parameter_in}'. Value: {str(source_value)[:100]}...")

            if mapping.target_parameter_in == "path":
                resolved_path_params[mapping.target_parameter_name] = str(source_value)
            elif mapping.target_parameter_in == "query":
                resolved_query_params[mapping.target_parameter_name] = source_value
            elif mapping.target_parameter_in == "header":
                resolved_headers[mapping.target_parameter_name] = str(source_value)
            elif mapping.target_parameter_in == "body":
                logger.debug(f"Node '{node_definition.effective_id}': InputMapping with target_parameter_in='body' will be processed by _prepare_api_request_components for whole body replacement.")
                pass
            elif mapping.target_parameter_in.startswith("body."):
                field_path_in_body = mapping.target_parameter_in.split("body.", 1)[1]
                _set_value_by_path(resolved_body_payload_for_field_mapping, field_path_in_body, source_value)
            else:
                logger.warning(f"Node '{node_definition.effective_id}': Unsupported target_parameter_in '{mapping.target_parameter_in}' for input mapping.")


    def _prepare_api_request_components(
        self, node_definition: Node, state: ExecutionGraphState
    ) -> Tuple[str, Dict[str, Any], Any, Dict[str, Any]]:
        """
        Prepares all components for an API request: URL path, query parameters, body payload, and headers.
        This involves resolving placeholders and applying input mappings.
        """
        path_template = node_definition.path or ""
        payload_template = node_definition.payload
        if isinstance(payload_template, dict):
            payload_template = payload_template.copy()

        resolved_path_from_template = self._resolve_placeholders(path_template, state)

        body_after_placeholder_resolution: Any
        if isinstance(payload_template, dict):
            body_after_placeholder_resolution = {}
            for key, value in payload_template.items():
                body_after_placeholder_resolution[key] = self._resolve_placeholders(value, state) if isinstance(value, str) else value
        elif payload_template is not None:
            body_after_placeholder_resolution = self._resolve_placeholders(str(payload_template), state)
        else:
            body_after_placeholder_resolution = {}

        final_path_params: Dict[str, Any] = {}
        final_query_params: Dict[str, Any] = {}
        final_headers: Dict[str, Any] = {}
        body_for_field_mappings = body_after_placeholder_resolution if isinstance(body_after_placeholder_resolution, dict) else {}

        self._apply_input_mappings(
            node_definition, state,
            final_path_params, final_query_params,
            body_for_field_mappings,
            final_headers
        )

        if not isinstance(body_after_placeholder_resolution, dict) and body_for_field_mappings:
            logger.info(f"Node '{node_definition.effective_id}': Original string/None payload template was superseded by 'body.*' input mappings, resulting in a dictionary payload.")
            final_body_payload = body_for_field_mappings
        elif isinstance(body_after_placeholder_resolution, dict):
            final_body_payload = body_for_field_mappings
        else:
            final_body_payload = body_after_placeholder_resolution

        for mapping in node_definition.input_mappings or []:
            if mapping.target_parameter_in == "body":
                current_extracted_ids = state.extracted_ids if isinstance(state.extracted_ids, dict) else {}
                source_value = _get_value_from_path(current_extracted_ids, mapping.source_data_path)
                if source_value is not None:
                    final_body_payload = source_value
                    logger.info(f"Node '{node_definition.effective_id}': Entire body was replaced by input mapping from source '{mapping.source_data_path}'. New body type: {type(final_body_payload)}.")
                    break

        final_api_path = resolved_path_from_template
        if final_path_params:
            temp_extracted_ids_for_path = (state.extracted_ids.copy() if isinstance(state.extracted_ids, dict) else {})
            temp_extracted_ids_for_path.update(final_path_params)
            initial_input_dict = state.initial_input if isinstance(state.initial_input, dict) else {}
            for k,v in initial_input_dict.items():
                if k not in temp_extracted_ids_for_path:
                    temp_extracted_ids_for_path[k] = v
            temp_state_for_path_re_resolution = ExecutionGraphState(
                extracted_ids=temp_extracted_ids_for_path,
                initial_input={}
            )
            final_api_path = self._resolve_placeholders(path_template, temp_state_for_path_re_resolution)

        logger.debug(f"Node '{node_definition.effective_id}': Prepared API components: Path='{final_api_path}', Query={final_query_params}, BodyType='{type(final_body_payload)}', Headers={final_headers}")
        return final_api_path, final_query_params, final_body_payload, final_headers

    async def _handle_node_confirmation_interrupt_if_needed(
        self, node_definition: Node, state: ExecutionGraphState,
        api_path: str, query_params: Dict[str, Any], body_payload: Any, headers: Dict[str, Any]
    ):
        """Raises an Interrupt if the node requires confirmation and hasn't been confirmed yet."""
        operationId = node_definition.effective_id
        confirmation_key = f"confirmed_{operationId}"
        current_confirmed_data = state.confirmed_data if isinstance(state.confirmed_data, dict) else {}
        is_already_confirmed = current_confirmed_data.get(confirmation_key)

        if node_definition.requires_confirmation and not is_already_confirmed:
            logger.info(f"Node '{operationId}' requires confirmation (Path: {api_path}). Raising Interrupt.")
            interrupt_data_for_ui = {
                "type": "api_call_confirmation",
                "operationId": operationId,
                "method": node_definition.method,
                "path": api_path,
                "query_params_to_confirm": query_params,
                "payload_to_confirm": body_payload,
                "headers_to_confirm": headers,
                "prompt": node_definition.confirmation_prompt or \
                          f"Confirm API call: {node_definition.method} {api_path}?",
                "confirmation_key": confirmation_key
            }
            raise Interrupt(interrupt_data_for_ui)


    def _apply_confirmed_data_to_request(
        self, node_definition: Node, state: ExecutionGraphState,
        current_body_payload: Any, current_query_params: Dict[str, Any], current_headers: Dict[str, Any]
    ) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        """
        If the node was confirmed and the user provided modified data,
        this function applies those modifications to the request components.
        """
        operationId = node_definition.effective_id
        confirmation_key = f"confirmed_{operationId}"

        updated_body = current_body_payload.copy() if isinstance(current_body_payload, dict) else current_body_payload
        updated_params = current_query_params.copy()
        updated_headers = current_headers.copy()

        current_confirmed_data = state.confirmed_data if isinstance(state.confirmed_data, dict) else {}

        if current_confirmed_data.get(confirmation_key):
            confirmed_details = current_confirmed_data.get(f"{confirmation_key}_details", {})
            if "modified_payload" in confirmed_details:
                updated_body = confirmed_details["modified_payload"]
                logger.info(f"Node '{operationId}': Applied modified payload from user confirmation.")
            if "modified_query_params" in confirmed_details and isinstance(confirmed_details["modified_query_params"], dict):
                updated_params = confirmed_details["modified_query_params"]
                logger.info(f"Node '{operationId}': Applied modified query parameters from user confirmation.")
            if "modified_headers" in confirmed_details and isinstance(confirmed_details["modified_headers"], dict):
                updated_headers = confirmed_details["modified_headers"]
                logger.info(f"Node '{operationId}': Applied modified headers from user confirmation.")
        return updated_body, updated_params, updated_headers

    async def _execute_api_and_process_outputs(
        self, node_definition: Node, api_path: str,
        query_params: Dict[str, Any], body_payload: Any, headers: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Executes the API call using APIExecutor and processes its output_mappings.
        Returns the API call result and any data extracted for the shared state.
        """
        effective_id = node_definition.effective_id
        api_call_method = node_definition.method or "GET"

        api_call_result_dict = await self.api_executor.execute_api(
            operationId=effective_id, method=api_call_method, endpoint=api_path,
            payload=body_payload, query_params=query_params, headers=headers
        )

        extracted_data_for_state = {}
        status_code = api_call_result_dict.get("status_code")
        is_successful = isinstance(status_code, int) and 200 <= status_code < 300

        if is_successful and node_definition.output_mappings:
            response_body = api_call_result_dict.get("response_body")
            if isinstance(response_body, dict):
                for mapping in node_definition.output_mappings:
                    extracted_value = _get_value_from_path(response_body, mapping.source_data_path)
                    if extracted_value is not None:
                        extracted_data_for_state[mapping.target_data_key] = extracted_value
                        logger.debug(f"Node '{effective_id}': OutputMapping - Extracted '{mapping.target_data_key}' from response path '{mapping.source_data_path}'. Value: {str(extracted_value)[:100]}...")
                    else:
                        logger.warning(f"Node '{effective_id}': OutputMapping - Could not extract from response path '{mapping.source_data_path}' for key '{mapping.target_data_key}'.")
            elif response_body is not None:
                 logger.warning(f"Node '{effective_id}': Response body (type: {type(response_body)}) is not a dict. Cannot apply output mappings. Preview: {str(response_body)[:100]}")
        return api_call_result_dict, (extracted_data_for_state if extracted_data_for_state else None)

    def _make_node_runnable(
        self, node_definition: Node
    ) -> Callable[[ExecutionGraphState], Awaitable[Dict[str, Any]]]:
        """
        Creates an awaitable function for a given API node definition, suitable for LangGraph.
        This function handles request preparation, confirmation, execution, and output processing.
        """
        async def node_executor(state: ExecutionGraphState) -> Dict[str, Any]:
            effective_id = node_definition.effective_id
            logger.info(f"--- [Graph 2] Node Start: {effective_id} (OpID: {node_definition.operationId}) ---")
            try:
                api_path, query_params, body_payload, headers = self._prepare_api_request_components(node_definition, state)
                await self._handle_node_confirmation_interrupt_if_needed(
                    node_definition, state, api_path, query_params, body_payload, headers
                )
                final_body, final_params, final_headers = self._apply_confirmed_data_to_request(
                    node_definition, state, body_payload, query_params, headers
                )
                api_call_result, extracted_data = await self._execute_api_and_process_outputs(
                    node_definition, api_path, final_params, final_body, final_headers
                )
                logger.info(f"--- [Graph 2] Node End: {effective_id} ---")
                return {
                    "api_results": {effective_id: api_call_result},
                    "extracted_ids": extracted_data,
                    "error": None
                }
            except Interrupt:
                logger.info(f"Node '{effective_id}' raising Interrupt for confirmation to be handled by manager.")
                raise
            except Exception as e:
                error_message = f"Error in node {effective_id}: {type(e).__name__} - {e}"
                logger.error(error_message, exc_info=True)
                return {
                    "error": error_message,
                    "api_results": {effective_id: {"error": error_message, "status_code": "NODE_EXCEPTION"}}
                }
        return node_executor

    def _build_and_compile_graph(self) -> Any:
        """
        Builds the LangGraph StateGraph from the graph_plan (nodes and edges)
        and compiles it.
        """
        logger.info(f"ExecutionGraphDefinition: Building graph. Plan description: {self.graph_plan.description or 'N/A'}")
        if logger.isEnabledFor(logging.DEBUG): # Avoid expensive model_dump_json if not debugging
            try:
                logger.debug(f"Plan nodes for Graph 2 build: {[n.model_dump_json(indent=None) for n in self.graph_plan.nodes]}")
                logger.debug(f"Plan edges for Graph 2 build: {[e.model_dump_json(indent=None) for e in self.graph_plan.edges]}")
            except Exception as e_dump:
                logger.warning(f"Could not dump plan nodes/edges for debug logging: {e_dump}")


        builder = StateGraph(ExecutionGraphState)
        if not self.graph_plan.nodes:
            raise ValueError("Execution plan (GraphOutput) must contain at least one node to build Graph 2.")

        # Store definitions of nodes that will actually be added to the LangGraph builder
        actual_api_nodes_defs = []
        for node_def in self.graph_plan.nodes:
            node_effective_id_str = str(node_def.effective_id).strip() if node_def.effective_id else ""
            
            # Skip conceptual START_NODE/END_NODE from the plan; they don't become executable LangGraph nodes.
            if node_effective_id_str.upper() in ["START_NODE", "END_NODE"]:
                logger.debug(f"Skipping conceptual plan node: '{node_effective_id_str}' for LangGraph builder.")
                continue
            
            if not node_def.method or not node_def.path: # Should be caught by Pydantic if method/path are not Optional
                raise ValueError(f"API Node '{node_effective_id_str}' in plan must have 'method' and 'path' defined.")
            
            logger.debug(f"Adding executable node to LangGraph builder: '{node_effective_id_str}'")
            builder.add_node(node_effective_id_str, self._make_node_runnable(node_def))
            actual_api_nodes_defs.append(node_def)

        # These are the string IDs of nodes actually added to the LangGraph builder
        executable_node_ids_in_builder = {str(n.effective_id).strip() for n in actual_api_nodes_defs}
        logger.debug(f"Executable node IDs added to LangGraph builder: {executable_node_ids_in_builder}")

        has_start_edge = False # Flag to check if an explicit entry point from LangGraph's START is set
        if not self.graph_plan.edges:
            logger.warning("Execution plan has no edges defined. Graph 2 might be empty or disconnected.")

        for edge_idx, edge in enumerate(self.graph_plan.edges):
            # Get original node IDs from the plan, strip whitespace for robustness
            plan_from_node_original_case = str(edge.from_node).strip() if edge.from_node else ""
            plan_to_node_original_case = str(edge.to_node).strip() if edge.to_node else ""

            # For comparison, use uppercase
            plan_from_node_upper = plan_from_node_original_case.upper()
            plan_to_node_upper = plan_to_node_original_case.upper()
            
            is_plan_source_start_node = plan_from_node_upper == "START_NODE"
            is_plan_target_end_node = plan_to_node_upper == "END_NODE"

            logger.debug(
                f"Processing edge {edge_idx + 1}/{len(self.graph_plan.edges)}: "
                f"from_original='{edge.from_node}', from_processed_original_case='{plan_from_node_original_case}', from_upper='{plan_from_node_upper}', is_plan_source_start={is_plan_source_start_node}; "
                f"to_original='{edge.to_node}', to_processed_original_case='{plan_to_node_original_case}', to_upper='{plan_to_node_upper}', is_plan_target_end={is_plan_target_end_node}"
            )

            # Determine the source and target for LangGraph's builder.add_edge
            # If plan source is "START_NODE", use LangGraph's START constant. Otherwise, use the processed node ID.
            source_for_builder = START if is_plan_source_start_node else plan_from_node_original_case
            # If plan target is "END_NODE", use LangGraph's END constant. Otherwise, use the processed node ID.
            target_for_builder = END if is_plan_target_end_node else plan_to_node_original_case

            # Validate that if source/target are actual node IDs (not START/END constants), they exist in the builder
            if source_for_builder != START: # It's an actual node ID string
                if not plan_from_node_original_case: # Check if original ID was empty
                     raise ValueError(f"Edge {edge_idx + 1} has an effectively empty 'from_node' in the plan.")
                if plan_from_node_original_case not in executable_node_ids_in_builder:
                    raise ValueError(
                        f"Edge source '{plan_from_node_original_case}' from plan is not a defined executable node in the LangGraph builder. "
                        f"Executable nodes: {executable_node_ids_in_builder}"
                    )
            
            if target_for_builder != END: # It's an actual node ID string
                if not plan_to_node_original_case: # Check if original ID was empty
                    raise ValueError(f"Edge {edge_idx + 1} has an effectively empty 'to_node' in the plan.")
                if plan_to_node_original_case not in executable_node_ids_in_builder:
                    raise ValueError(
                        f"Edge target '{plan_to_node_original_case}' from plan is not a defined executable node in the LangGraph builder. "
                        f"Executable nodes: {executable_node_ids_in_builder}"
                    )
            try:
                # Log exactly what is being passed to builder.add_edge
                source_log_str = 'LANGGRAPH_START' if source_for_builder == START else source_for_builder
                target_log_str = 'LANGGRAPH_END' if target_for_builder == END else target_for_builder
                logger.debug(f"Attempting to add edge to LangGraph builder: from='{source_log_str}', to='{target_log_str}'")
                
                builder.add_edge(source_for_builder, target_for_builder)
                logger.debug("Successfully added edge to LangGraph builder.")
            except Exception as e_add_edge: # Catch errors from LangGraph's add_edge
                logger.error(
                    f"LangGraph builder.add_edge call failed. "
                    f"Plan edge: from='{edge.from_node}' -> to='{edge.to_node}'. "
                    f"Mapped for builder: from='{source_log_str}' -> to='{target_log_str}'. "
                    f"Error: {e_add_edge}", exc_info=True
                )
                # Re-raise with more context
                raise ValueError(
                    f"Failed to add edge from plan ('{edge.from_node}' -> '{edge.to_node}') to LangGraph builder. "
                    f"Mapped source: '{source_log_str}', Mapped target: '{target_log_str}'. "
                    f"Underlying error: {e_add_edge}"
                )

            if source_for_builder == START:
                has_start_edge = True
        
        # Set entry point for the LangGraph graph
        if not has_start_edge and actual_api_nodes_defs:
            # If no edge from "START_NODE" in plan, assume the first *executable* API node is the entry point.
            entry_point_candidate = str(actual_api_nodes_defs[0].effective_id).strip()
            if entry_point_candidate not in executable_node_ids_in_builder:
                 # This should not happen if actual_api_nodes_defs[0] is correctly derived
                 raise ValueError(f"Default entry point candidate '{entry_point_candidate}' is not in executable_node_ids_in_builder: {executable_node_ids_in_builder}")
            logger.info(f"No explicit START edge defined in plan that connects to an executable node. Setting LangGraph entry point to: '{entry_point_candidate}'.")
            builder.set_entry_point(entry_point_candidate)
        elif not actual_api_nodes_defs and not has_start_edge :
             logger.warning("Graph has no executable API nodes and no explicit entry point from START. This execution graph will be empty or unrunnable.")
             # An empty graph might be valid if the plan was just START_NODE -> END_NODE, though LangGraph might still error.
             # If actual_api_nodes_defs is empty, there are no nodes for set_entry_point.

        # Set finish points for any leaf executable nodes that are not already connected to plan's "END_NODE"
        if actual_api_nodes_defs:
            # Get all node IDs from the plan that are sources in an edge (excluding the conceptual START_NODE from the plan)
            all_source_node_ids_in_plan_edges = {
                str(e.from_node).strip() for e in self.graph_plan.edges
                if str(e.from_node).strip().upper() != "START_NODE" # Edges from plan's START_NODE don't make an API node a "source" for this logic
            }
            for node_def in actual_api_nodes_defs: # Iterate over nodes actually added to the builder
                node_id_str = str(node_def.effective_id).strip()
                
                # Check if this executable node is a source in any edge leading to another *executable* node or the plan's END_NODE
                is_source_to_another_node_or_end = any(
                    str(e.from_node).strip() == node_id_str and 
                    (str(e.to_node).strip().upper() == "END_NODE" or str(e.to_node).strip() in executable_node_ids_in_builder)
                    for e in self.graph_plan.edges
                )
                
                if not is_source_to_another_node_or_end:
                    # This node is an executable node, and it does not lead to any other executable node or the plan's END_NODE.
                    # So, it should be a finish point in the LangGraph graph.
                    logger.info(f"Executable node '{node_id_str}' is a leaf or only leads to non-executable nodes (excluding END_NODE). Setting as a LangGraph finish point.")
                    builder.set_finish_point(node_id_str)
        
        # Compile the graph with a checkpointer (MemorySaver for in-memory checkpointing)
        return builder.compile(checkpointer=MemorySaver())

    def get_runnable_graph(self) -> Any:
        """Returns the compiled, runnable LangGraph instance."""
        return self.runner

