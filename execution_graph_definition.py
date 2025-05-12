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
        # This case should ideally not be reached if path is not empty,
        # but added for robustness.
        logger.warning(f"Path was effectively empty for _set_value_by_path after splitting. Original dict might be directly modified if it was the target.")


class ExecutionGraphDefinition:
    """
    Defines and compiles "Graph 2" (Execution LangGraph) based on an execution plan.
    It takes a plan (GraphOutput from models.py), an APIExecutor, and builds a runnable LangGraph.
    """
    def __init__(self, graph_execution_plan: GraphOutput, api_executor: APIExecutor):
        self.graph_plan = graph_execution_plan
        self.api_executor = api_executor 
        self.runner: Any = self._build_and_compile_graph() # MODIFIED: Type hint to Any
        logger.info("ExecutionGraphDefinition initialized and Graph 2 compiled.")

    def _resolve_placeholders(self, template_string: str, state: ExecutionGraphState) -> str:
        """Resolves placeholders like {{key}} in a string using data from ExecutionGraphState."""
        if not isinstance(template_string, str): # If not a string, return its string representation
            return str(template_string) 

        resolved_string = template_string
        # Ensure extracted_ids and initial_input are dictionaries before iterating
        extracted_ids_dict = state.extracted_ids if isinstance(state.extracted_ids, dict) else {}
        initial_input_dict = state.initial_input if isinstance(state.initial_input, dict) else {}

        # Resolve from extracted_ids (data from previous nodes)
        for key, value in extracted_ids_dict.items():
            if value is not None: # Ensure value is not None before converting to string
                resolved_string = resolved_string.replace(f"{{{key}}}", str(value))
        
        # Resolve from initial_input (data provided at the start of Graph 2)
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
            # Get value from the shared data pool (extracted_ids)
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
                # This case means the entire body should be replaced by source_value.
                # It's handled in _prepare_api_request_components after initial placeholder resolution.
                logger.debug(f"Node '{node_definition.effective_id}': InputMapping with target_parameter_in='body' will be processed by _prepare_api_request_components for whole body replacement.")
                pass 
            elif mapping.target_parameter_in.startswith("body."):
                # Maps to a specific field within the body object.
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
        # If payload_template is a dict, copy it to avoid modifying the original Node model
        if isinstance(payload_template, dict):
            payload_template = payload_template.copy()

        # 1. Resolve placeholders in path and body from state (initial_input, extracted_ids)
        resolved_path_from_template = self._resolve_placeholders(path_template, state)
        
        body_after_placeholder_resolution: Any
        if isinstance(payload_template, dict):
            body_after_placeholder_resolution = {}
            for key, value in payload_template.items():
                body_after_placeholder_resolution[key] = self._resolve_placeholders(value, state) if isinstance(value, str) else value
        elif payload_template is not None: # If template is not dict but exists (e.g. string)
            body_after_placeholder_resolution = self._resolve_placeholders(str(payload_template), state)
        else: # No payload template
            body_after_placeholder_resolution = {} # Default to empty dict if no template, allows body.fieldName mappings

        # Initialize containers for resolved parameters
        final_path_params: Dict[str, Any] = {} # For path params that are mapped, not just {{placeholder}}
        final_query_params: Dict[str, Any] = {}
        final_headers: Dict[str, Any] = {}
        
        # This dictionary will hold the body as it's modified by "body.fieldName" input mappings.
        # Start with the placeholder-resolved body if it's a dict, or an empty dict.
        body_for_field_mappings = body_after_placeholder_resolution if isinstance(body_after_placeholder_resolution, dict) else {}
        
        # 2. Apply input mappings (this can modify final_query_params, body_for_field_mappings, final_headers, and final_path_params)
        self._apply_input_mappings(
            node_definition, state,
            final_path_params, final_query_params, 
            body_for_field_mappings, 
            final_headers
        )

        # Determine the final body payload
        # If the original template was not a dict, but "body.*" mappings created a dict, use that.
        if not isinstance(body_after_placeholder_resolution, dict) and body_for_field_mappings:
            logger.info(f"Node '{node_definition.effective_id}': Original string/None payload template was superseded by 'body.*' input mappings, resulting in a dictionary payload.")
            final_body_payload = body_for_field_mappings
        elif isinstance(body_after_placeholder_resolution, dict):
            # If original was a dict, body_for_field_mappings is that dict with fields updated/added.
            final_body_payload = body_for_field_mappings 
        else: # Original was not dict, and no "body.*" mappings created a dict (e.g. it was a string, or None)
            final_body_payload = body_after_placeholder_resolution


        # 3. Handle 'body' input mappings (replace entire body) - this takes precedence
        for mapping in node_definition.input_mappings or []:
            if mapping.target_parameter_in == "body":
                current_extracted_ids = state.extracted_ids if isinstance(state.extracted_ids, dict) else {}
                source_value = _get_value_from_path(current_extracted_ids, mapping.source_data_path)
                if source_value is not None: 
                    final_body_payload = source_value # Replace the entire body
                    logger.info(f"Node '{node_definition.effective_id}': Entire body was replaced by input mapping from source '{mapping.source_data_path}'. New body type: {type(final_body_payload)}.")
                    break # Assuming only one "body" replacement mapping is effective or intended
        
        # 4. Re-resolve path if path parameters were set by input_mappings
        final_api_path = resolved_path_from_template
        if final_path_params: # If _apply_input_mappings populated path parameters
            # Create a temporary state that includes these newly resolved path params for placeholder resolution
            temp_extracted_ids_for_path = (state.extracted_ids.copy() if isinstance(state.extracted_ids, dict) else {})
            temp_extracted_ids_for_path.update(final_path_params) # Add mapped path params
            
            initial_input_dict = state.initial_input if isinstance(state.initial_input, dict) else {}
            for k,v in initial_input_dict.items(): # Also consider initial_input for path placeholders
                if k not in temp_extracted_ids_for_path: 
                    temp_extracted_ids_for_path[k] = v
            
            # Create a temporary state object just for this re-resolution
            temp_state_for_path_re_resolution = ExecutionGraphState(
                extracted_ids=temp_extracted_ids_for_path,
                initial_input={} # Initial input already merged into temp_extracted_ids_for_path for this purpose
            )
            # Re-resolve the original path_template with these combined values
            final_api_path = self._resolve_placeholders(path_template, temp_state_for_path_re_resolution)

        logger.debug(f"Node '{node_definition.effective_id}': Prepared API components: Path='{final_api_path}', Query={final_query_params}, BodyType='{type(final_body_payload)}', Headers={final_headers}")
        return final_api_path, final_query_params, final_body_payload, final_headers

    async def _handle_node_confirmation_interrupt_if_needed(
        self, node_definition: Node, state: ExecutionGraphState,
        api_path: str, query_params: Dict[str, Any], body_payload: Any, headers: Dict[str, Any]
    ):
        """Raises an Interrupt if the node requires confirmation and hasn't been confirmed yet."""
        operationId = node_definition.effective_id
        confirmation_key = f"confirmed_{operationId}" # Key to check in state.confirmed_data
        
        current_confirmed_data = state.confirmed_data if isinstance(state.confirmed_data, dict) else {}
        is_already_confirmed = current_confirmed_data.get(confirmation_key) # Check if this specific confirmation exists

        if node_definition.requires_confirmation and not is_already_confirmed:
            logger.info(f"Node '{operationId}' requires confirmation (Path: {api_path}). Raising Interrupt.")
            # Data to send to the UI for the confirmation prompt
            interrupt_data_for_ui = {
                "type": "api_call_confirmation", # Standardized type for UI to recognize
                "operationId": operationId,
                "method": node_definition.method,
                "path": api_path,
                "query_params_to_confirm": query_params,
                "payload_to_confirm": body_payload, # Send the prepared payload
                "headers_to_confirm": headers,
                "prompt": node_definition.confirmation_prompt or \
                          f"Confirm API call: {node_definition.method} {api_path}?", # Default prompt
                "confirmation_key": confirmation_key # Key the UI should send back with the decision
            }
            raise Interrupt(interrupt_data_for_ui) # LangGraph will catch this


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
        
        # Start with copies of the current request components
        updated_body = current_body_payload.copy() if isinstance(current_body_payload, dict) else current_body_payload
        updated_params = current_query_params.copy()
        updated_headers = current_headers.copy()

        current_confirmed_data = state.confirmed_data if isinstance(state.confirmed_data, dict) else {}
        
        # Check if this specific operation was confirmed (value for confirmation_key is True)
        if current_confirmed_data.get(confirmation_key):
            # The resume data (including any modifications) is expected under a details key
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
        api_call_method = node_definition.method or "GET" # Default to GET if not specified (should be rare)
        
        # Make the API call
        api_call_result_dict = await self.api_executor.execute_api(
            operationId=effective_id, method=api_call_method, endpoint=api_path,
            payload=body_payload, query_params=query_params, headers=headers
        )

        extracted_data_for_state = {} # Data to add to ExecutionGraphState.extracted_ids
        status_code = api_call_result_dict.get("status_code")
        is_successful = isinstance(status_code, int) and 200 <= status_code < 300

        # Process output mappings if the call was successful and mappings exist
        if is_successful and node_definition.output_mappings:
            response_body = api_call_result_dict.get("response_body")
            if isinstance(response_body, dict): # Output mappings usually expect a JSON dict response
                for mapping in node_definition.output_mappings:
                    extracted_value = _get_value_from_path(response_body, mapping.source_data_path)
                    if extracted_value is not None:
                        extracted_data_for_state[mapping.target_data_key] = extracted_value
                        logger.debug(f"Node '{effective_id}': OutputMapping - Extracted '{mapping.target_data_key}' from response path '{mapping.source_data_path}'. Value: {str(extracted_value)[:100]}...")
                    else:
                        logger.warning(f"Node '{effective_id}': OutputMapping - Could not extract from response path '{mapping.source_data_path}' for key '{mapping.target_data_key}'.")
            elif response_body is not None: # If response body exists but isn't a dict
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
                # 1. Prepare request components (URL, params, body, headers)
                api_path, query_params, body_payload, headers = self._prepare_api_request_components(node_definition, state)
                
                # 2. Handle confirmation if required (raises Interrupt if not confirmed)
                await self._handle_node_confirmation_interrupt_if_needed(
                    node_definition, state, api_path, query_params, body_payload, headers
                )
                
                # 3. Apply any user-confirmed modifications to the request components
                # This step is reached if:
                #    a) Confirmation was not required.
                #    b) Confirmation was required, an Interrupt was raised, and then the graph resumed
                #       with `state.confirmed_data` updated for this node.
                final_body, final_params, final_headers = self._apply_confirmed_data_to_request(
                    node_definition, state, body_payload, query_params, headers
                )
                
                # 4. Execute the API call and process outputs
                api_call_result, extracted_data = await self._execute_api_and_process_outputs(
                    node_definition, api_path, final_params, final_body, final_headers
                )
                
                logger.info(f"--- [Graph 2] Node End: {effective_id} ---")
                # Return structure expected by LangGraph to update ExecutionGraphState
                return {
                    "api_results": {effective_id: api_call_result}, # Add this call's result
                    "extracted_ids": extracted_data, # Add extracted data to shared pool
                    "error": None # No error in this node's execution path
                }
            except Interrupt: # Re-raise Interrupt to be caught by GraphExecutionManager
                logger.info(f"Node '{effective_id}' raising Interrupt for confirmation to be handled by manager.")
                raise
            except Exception as e: # Catch all other errors during node execution
                error_message = f"Error in node {effective_id}: {type(e).__name__} - {e}"
                logger.error(error_message, exc_info=True)
                # Return error information in the state
                return {
                    "error": error_message,
                    "api_results": {effective_id: {"error": error_message, "status_code": "NODE_EXCEPTION"}}
                }
        return node_executor

    def _build_and_compile_graph(self) -> Any: # MODIFIED: Type hint to Any
        """
        Builds the LangGraph StateGraph from the graph_plan (nodes and edges)
        and compiles it.
        """
        builder = StateGraph(ExecutionGraphState) # Define the state schema for this graph
        
        if not self.graph_plan.nodes:
            raise ValueError("Execution plan (GraphOutput) must contain at least one node.")

        node_ids_in_plan = {n.effective_id for n in self.graph_plan.nodes}
        
        # Add API call nodes to the graph builder
        for node_def in self.graph_plan.nodes:
            # START_NODE and END_NODE are conceptual for planning; they don't become executable API nodes.
            # LangGraph uses its own START and END constants for graph structure.
            if node_def.effective_id.upper() in ["START_NODE", "END_NODE"]:
                continue 
            
            # Basic validation for executable nodes
            if not node_def.method or not node_def.path:
                raise ValueError(f"API Node '{node_def.effective_id}' in plan must have 'method' and 'path' defined.")
            
            builder.add_node(node_def.effective_id, self._make_node_runnable(node_def))

        # Add edges based on the plan
        has_start_edge = False # Flag to check if an explicit entry point is set
        for edge in self.graph_plan.edges:
            source_is_langgraph_start = edge.from_node.upper() == "START" # LangGraph's special START
            target_is_langgraph_end = edge.to_node.upper() == "END"     # LangGraph's special END

            # Validate edge source and target against nodes defined in the plan (unless START/END)
            if not source_is_langgraph_start and edge.from_node not in node_ids_in_plan: 
                logger.warning(f"Edge source '{edge.from_node}' not in plan nodes. Skipping edge.")
                continue
            if not target_is_langgraph_end and edge.to_node not in node_ids_in_plan:
                logger.warning(f"Edge target '{edge.to_node}' not in plan nodes. Skipping edge.")
                continue

            if source_is_langgraph_start:
                builder.add_edge(START, edge.to_node) # Connect LangGraph START to the first actual node
                has_start_edge = True
            elif target_is_langgraph_end:
                builder.add_edge(edge.from_node, END) # Connect a node to LangGraph END
            else: # Regular edge between two defined nodes
                builder.add_edge(edge.from_node, edge.to_node)
        
        # Identify actual API nodes (excluding conceptual START_NODE/END_NODE from the plan)
        actual_api_nodes = [n for n in self.graph_plan.nodes if n.effective_id.upper() not in ["START_NODE", "END_NODE"]]
        if not actual_api_nodes:
             raise ValueError("Execution plan contains no executable API nodes after filtering START_NODE/END_NODE.")

        # Set entry point if not explicitly defined by an edge from "START"
        if not has_start_edge and actual_api_nodes:
            # If no edge from "START", assume the first API node in the list is the entry point.
            # This might need more sophisticated logic based on graph structure if order isn't guaranteed.
            builder.set_entry_point(actual_api_nodes[0].effective_id)
            logger.info(f"No explicit START edge defined in plan. Set entry point to: '{actual_api_nodes[0].effective_id}'.")
        elif not actual_api_nodes and not has_start_edge : # No API nodes and no start edge
             raise ValueError("Graph has no API nodes and no explicit entry point from START.")


        # Set finish points for any leaf nodes that are not already connected to "END"
        if actual_api_nodes:
            all_from_nodes_in_edges = {e.from_node for e in self.graph_plan.edges if e.from_node.upper() != "START"}
            for node_def in actual_api_nodes:
                node_id = node_def.effective_id
                # A node is a leaf if it's not a source in any edge AND not connected to END
                is_leaf_not_ending = (node_id not in all_from_nodes_in_edges) and \
                                     (not any(e.from_node == node_id and e.to_node.upper() == "END" for e in self.graph_plan.edges))
                if is_leaf_not_ending:
                    logger.info(f"Node '{node_id}' is a leaf not connected to END. Setting as a finish point.")
                    builder.set_finish_point(node_id) # Mark such nodes as potential end points of execution paths
        
        # Compile the graph with a checkpointer (MemorySaver for in-memory checkpointing)
        return builder.compile(checkpointer=MemorySaver())

    def get_runnable_graph(self) -> Any: # MODIFIED: Type hint to Any
        """Returns the compiled, runnable LangGraph instance."""
        return self.runner
