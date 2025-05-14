import asyncio
import logging
from typing import Any, Callable, Awaitable, Dict, Optional, Tuple, Union, List

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from models import GraphOutput, Node, Edge, InputMapping, OutputMapping, ExecutionGraphState
from api_executor import APIExecutor

logger = logging.getLogger(__name__)

# _get_value_from_path and _set_value_by_path remain the same
def _get_value_from_path(data_dict: Optional[Dict[str, Any]], path: str) -> Any:
    if not path or not isinstance(data_dict, dict):
        return None
    if path.startswith("$."):
        path = path[2:]
    keys = path.split('.')
    current_val = data_dict
    for key_part in keys:
        if isinstance(current_val, dict):
            if key_part not in current_val:
                return None
            current_val = current_val.get(key_part)
        elif isinstance(current_val, list):
            try:
                idx = int(key_part)
                if 0 <= idx < len(current_val):
                    current_val = current_val[idx]
                else:
                    return None
            except ValueError:
                return None
        else:
            return None
    return current_val

def _set_value_by_path(data_dict: Dict[str, Any], path: str, value: Any):
    if not path:
        return
    keys = path.split('.')
    current_level = data_dict
    for i, key in enumerate(keys[:-1]):
        if key not in current_level or not isinstance(current_level.get(key), dict):
            current_level[key] = {}
        current_level = current_level[key]
    if keys:
        current_level[keys[-1]] = value


class ExecutionGraphDefinition:
    def __init__(
        self,
        graph_execution_plan: GraphOutput,
        api_executor: APIExecutor
    ):
        self.graph_plan = graph_execution_plan
        self.api_executor = api_executor
        self.runnable_graph: Any = self._build_and_compile_graph()
        logger.info("ExecutionGraphDefinition initialized and Graph 2 compiled.")

    def _resolve_placeholders(self, template_string: str, state: ExecutionGraphState) -> str:
        """
        Resolves placeholders in a string using the format {{key}}.
        Data for placeholders is sourced from state.initial_input and state.extracted_ids.
        """
        if not isinstance(template_string, str):
            logger.debug(f"Template string is not a string type: {type(template_string)}, returning as is.")
            return str(template_string)
        
        resolved_string = template_string
        # Combine initial_input and extracted_ids for placeholder data.
        # extracted_ids takes precedence in case of key collision, which is generally desired.
        placeholders_data = {**(state.initial_input or {}), **(state.extracted_ids or {})}
        
        if not placeholders_data:
            logger.debug("No placeholders_data available (initial_input and extracted_ids are empty/None) for {{...}} resolution.")
            return resolved_string

        logger.debug(f"Attempting to resolve {{...}} placeholders in: '{template_string}' with data: {placeholders_data}")

        for key, value in placeholders_data.items():
            if value is not None:
                placeholder_to_find = "{{" + str(key) + "}}" # Match {{key}}
                str_value = str(value) 

                if placeholder_to_find in resolved_string:
                    resolved_string = resolved_string.replace(placeholder_to_find, str_value)
                    logger.debug(f"Replaced '{{{{{key}}}}}' with '{str_value}'. New string: '{resolved_string}'")
        
        logger.debug(f"Final string after {{...}} resolution: '{resolved_string}'")
        return resolved_string

    def _apply_input_mappings(
        self,
        node_definition: Node,
        state: ExecutionGraphState,
        resolved_path_params: Dict[str, Any], # Output: Populated with resolved path param values
        resolved_query_params: Dict[str, Any], # Output: Populated with resolved query param values
        resolved_body_payload_for_field_mapping: Dict[str, Any], # Output: Body fields are set here
        resolved_headers: Dict[str, Any] # Output: Populated with resolved header values
    ) -> None:
        """
        Applies input mappings to populate path parameters, query parameters, body fields, and headers.
        It sources data from state.initial_input and state.extracted_ids.
        """
        if not node_definition.input_mappings:
            return
        
        # Data sources for input mappings. extracted_ids takes precedence.
        available_data_for_mapping = {**(state.initial_input or {}), **(state.extracted_ids or {})}

        for mapping in node_definition.input_mappings:
            source_value = None
            if mapping.source_data_path: 
                source_value = _get_value_from_path(available_data_for_mapping, mapping.source_data_path)

            if source_value is None:
                logger.warning(f"Node '{node_definition.effective_id}': InputMapping - Could not find source data for path '{mapping.source_data_path}' in available data {list(available_data_for_mapping.keys())}. Skipping.")
                continue

            target_param_name = mapping.target_parameter_name
            target_param_in = mapping.target_parameter_in

            if target_param_in == "path":
                resolved_path_params[target_param_name] = str(source_value) # Ensure string for path
            elif target_param_in == "query":
                resolved_query_params[target_param_name] = source_value
            elif target_param_in == "header":
                resolved_headers[target_param_name] = str(source_value) # Ensure string for headers
            elif target_param_in.startswith("body."):
                field_path_in_body = target_param_in.split("body.", 1)[1]
                _set_value_by_path(resolved_body_payload_for_field_mapping, field_path_in_body, source_value)
            elif target_param_in == "body":
                 # This case (full body replacement via input mapping) is handled in _prepare_api_request_components
                 logger.debug(f"Node '{node_definition.effective_id}': InputMapping for full body replacement noted for '{target_param_name}'. Will be applied in _prepare_api_request_components if applicable.")


    def _prepare_api_request_components(
        self, node_definition: Node, state: ExecutionGraphState
    ) -> Tuple[str, Dict[str, Any], Any, Dict[str, Any]]:
        """
        Prepares the API request components: path, query parameters, body payload, and headers.
        1. Resolves {{...}} placeholders in the path and body templates.
        2. Applies input mappings to populate path params, query params, body fields, and headers.
        3. Specifically substitutes {path_param_name} style placeholders in the API path.
        4. Handles full body replacement if specified by an input mapping.
        """
        path_template = node_definition.path or ""
        payload_template = node_definition.payload # This is the *template* from the Node definition
        
        # Make a mutable copy of the payload template if it's a dictionary
        current_payload_template_copy: Any
        if isinstance(payload_template, dict):
            current_payload_template_copy = {k: v for k, v in payload_template.items()} 
        else:
            current_payload_template_copy = payload_template # Could be None, str, list etc.

        # Step 1: Resolve {{...}} placeholders in the path and body templates
        # This uses data from state.initial_input and state.extracted_ids
        resolved_path_from_double_braces = self._resolve_placeholders(path_template, state)
        
        body_after_double_brace_resolution: Any
        if isinstance(current_payload_template_copy, dict):
            body_after_double_brace_resolution = {
                key: (self._resolve_placeholders(value, state) if isinstance(value, str) else value)
                for key, value in current_payload_template_copy.items()
            }
        elif current_payload_template_copy is not None: # If template is a non-dict (e.g. string, list)
            body_after_double_brace_resolution = self._resolve_placeholders(str(current_payload_template_copy), state)
        else: # If payload_template was None
            body_after_double_brace_resolution = {} # Default to an empty dict if no template, allows field mappings

        # Initialize containers for resolved parameters
        resolved_path_params_from_mapping: Dict[str, Any] = {} 
        resolved_query_params_from_mapping: Dict[str, Any] = {}
        resolved_headers_from_mapping: Dict[str, Any] = {}
        # Use the body (potentially modified by {{...}} resolution) as the base for field mappings
        body_for_field_mappings = body_after_double_brace_resolution if isinstance(body_after_double_brace_resolution, dict) else {}


        # Step 2: Apply input mappings (populates the resolved_*_params dicts and modifies body_for_field_mappings)
        self._apply_input_mappings(
            node_definition, state, 
            resolved_path_params_from_mapping, 
            resolved_query_params_from_mapping,
            body_for_field_mappings, # This dict will be modified by _apply_input_mappings for "body.fieldName"
            resolved_headers_from_mapping
        )

        # The body payload is now the one potentially modified by "body.fieldName" mappings
        final_body_payload = body_for_field_mappings 

        # Step 3: Handle full body replacement from input mapping (if any)
        # This overrides any previous body construction if a specific input mapping targets the entire body.
        if node_definition.input_mappings:
            for mapping in node_definition.input_mappings:
                if mapping.target_parameter_in == "body": # A mapping explicitly targets the entire body
                    current_data_sources = {**(state.initial_input or {}), **(state.extracted_ids or {})}
                    source_value_for_full_body = _get_value_from_path(current_data_sources, mapping.source_data_path)
                    if source_value_for_full_body is not None:
                        final_body_payload = source_value_for_full_body 
                        logger.info(f"Node '{node_definition.effective_id}': Entire request body replaced by input mapping from '{mapping.source_data_path}'.")
                        break # First full body mapping encountered takes precedence

        # Step 4: Substitute {path_param_name} style placeholders in the API path
        # Start with the path that has already had {{...}} resolved
        final_api_path = resolved_path_from_double_braces
        if resolved_path_params_from_mapping: 
            for param_name, param_value in resolved_path_params_from_mapping.items():
                openapi_placeholder = f"{{{param_name}}}"
                if openapi_placeholder in final_api_path:
                    final_api_path = final_api_path.replace(openapi_placeholder, str(param_value))
                    logger.debug(f"Node '{node_definition.effective_id}': Replaced OpenAPI path param '{openapi_placeholder}' with '{param_value}'. New path: '{final_api_path}'")
                else:
                    logger.warning(f"Node '{node_definition.effective_id}': Path parameter '{param_name}' (from input_mappings) with placeholder '{openapi_placeholder}' not found in path template '{final_api_path}'.")
        
        logger.debug(f"Node '{node_definition.effective_id}': Final prepared components - Path: {final_api_path}, Query: {resolved_query_params_from_mapping}, Body Type: {type(final_body_payload)}, Headers: {resolved_headers_from_mapping}")
        return final_api_path, resolved_query_params_from_mapping, final_body_payload, resolved_headers_from_mapping


    def _apply_confirmed_data_to_request(
        self, node_definition: Node, state: ExecutionGraphState,
        current_body_payload: Any, current_query_params: Dict[str, Any], current_headers: Dict[str, Any]
    ) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        """
        Applies data confirmed by the user (during an interrupt) to the request components.
        Currently, this primarily focuses on replacing the body payload if modified by the user.
        """
        operationId = node_definition.effective_id # Use effective_id for consistency with confirmation_key
        confirmation_key = f"confirmed_{operationId}"
        
        # Make copies to avoid modifying the original dicts if they are passed around
        updated_body = current_body_payload
        if isinstance(current_body_payload, dict): 
            updated_body = {k:v for k,v in current_body_payload.items()}
        
        updated_params = current_query_params.copy()
        updated_headers = current_headers.copy()
        
        current_confirmed_data = state.confirmed_data or {} 
        
        # Check if this operation was indeed confirmed (decision was true)
        if current_confirmed_data.get(confirmation_key) is True: # Explicitly check for True
            confirmed_details = current_confirmed_data.get(f"{confirmation_key}_details", {})
            
            # If user provided a modified payload during confirmation, use it
            if "modified_payload" in confirmed_details: 
                updated_body = confirmed_details["modified_payload"]
                logger.info(f"Node '{operationId}': Applied modified payload from user confirmation.")
            # Future: Could extend to apply modifications to query_params or headers if needed
            # e.g., if confirmed_details contained "modified_query_params": {...}
                
        return updated_body, updated_params, updated_headers


    async def _execute_api_and_process_outputs(
        self, node_definition: Node, api_path: str,
        query_params: Dict[str, Any], body_payload: Any, headers: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Executes the API call using APIExecutor and processes its output mappings.
        """
        effective_id = node_definition.effective_id
        api_call_method = node_definition.method or "GET" # Default to GET if not specified

        # Log the final components being sent to the API executor
        payload_log_preview = str(body_payload)[:200] + "..." if body_payload and len(str(body_payload)) > 200 else body_payload
        logger.debug(f"Node '{effective_id}': Executing API call. Method: {api_call_method}, Path: {api_path}, Query: {query_params}, Headers: {headers}, Body Preview: {payload_log_preview}")

        api_call_result_dict = await self.api_executor.execute_api(
            operationId=effective_id, # Pass effective_id for better context in APIExecutor logs
            method=api_call_method, 
            endpoint=api_path,
            payload=body_payload, 
            query_params=query_params, 
            headers=headers
        )
        
        extracted_data_for_state = {} # To store data extracted via output_mappings
        status_code = api_call_result_dict.get("status_code")
        is_successful = isinstance(status_code, int) and 200 <= status_code < 300

        if is_successful and node_definition.output_mappings:
            response_body = api_call_result_dict.get("response_body")
            if isinstance(response_body, (dict, list)): # Output mappings usually expect JSON
                for mapping in node_definition.output_mappings:
                    extracted_value = _get_value_from_path(response_body, mapping.source_data_path)
                    if extracted_value is not None:
                        extracted_data_for_state[mapping.target_data_key] = extracted_value
                        logger.debug(f"Node '{effective_id}': OutputMapping - Extracted '{mapping.target_data_key}' (value: {str(extracted_value)[:50]}...) from path '{mapping.source_data_path}'.")
                    else:
                        logger.warning(f"Node '{effective_id}': OutputMapping - Could not extract value for target_data_key '{mapping.target_data_key}' using path '{mapping.source_data_path}' from response. Response preview: {str(response_body)[:100]}")
            elif response_body is not None: # Response body is not dict/list (e.g., plain text, XML)
                 logger.warning(f"Node '{effective_id}': Response body type {type(response_body)} is not dict or list. Cannot apply standard JSONPath output mappings directly. Response preview: {str(response_body)[:100]}")
                 # Simple heuristic: if there's only one output mapping and its source_data_path is simple (e.g. no dots or brackets, implying direct use of the response)
                 if len(node_definition.output_mappings) == 1:
                     om = node_definition.output_mappings[0]
                     if not any(c in om.source_data_path for c in ['.', '[', ']']) or om.source_data_path == "$": # If path is just "data" or "$"
                         simple_key = om.target_data_key
                         extracted_data_for_state[simple_key] = response_body
                         logger.info(f"Node '{effective_id}': Applied non-dict/list response directly to target_data_key '{simple_key}' due to simple output mapping ('{om.source_data_path}').")


        return api_call_result_dict, (extracted_data_for_state if extracted_data_for_state else None)

    def _make_node_runnable(
        self, node_definition: Node
    ) -> Callable[[ExecutionGraphState], Awaitable[Dict[str, Any]]]:
        """
        Creates a runnable async function for a given API node definition.
        This function will be added to the LangGraph StateGraph.
        """
        async def node_executor(state: ExecutionGraphState) -> Dict[str, Any]:
            effective_id = node_definition.effective_id
            logger.info(f"--- [Graph 2 Node] Start: {effective_id} (OpID: {node_definition.operationId}) ---")
            
            output_state_update: Dict[str, Any] = {} # What this node will return to update the graph's state

            try:
                # Prepare API request components (path, query, body, headers) using current state
                api_path, query_params, body_payload, headers = self._prepare_api_request_components(node_definition, state)
                
                # Handle confirmation requirement
                if node_definition.requires_confirmation:
                    confirmation_key = f"confirmed_{effective_id}"
                    # Check if 'confirmed_data' exists and the specific key is True
                    if not (state.confirmed_data or {}).get(confirmation_key, False): # Default to False if key not found
                        skip_message = f"Node '{effective_id}' requires confirmation, but it was not found or was negative in confirmed_data. Skipping execution."
                        logger.warning(skip_message)
                        output_state_update["error"] = skip_message 
                        # Store a result indicating skip, to avoid breaking state structure for api_results
                        output_state_update["api_results"] = {effective_id: {"status_code": "SKIPPED_NO_CONFIRMATION", "error": skip_message, "path_template": node_definition.path, "method": node_definition.method}}
                        return output_state_update # Halt execution of this node

                # Apply any user-confirmed data (e.g., modified payload) to the request components
                final_body, final_params, final_headers = self._apply_confirmed_data_to_request(
                    node_definition, state, body_payload, query_params, headers
                )
                
                # Execute the API call and process its outputs
                api_call_result, extracted_data = await self._execute_api_and_process_outputs(
                    node_definition, api_path, final_params, final_body, final_headers
                )
                
                # Update api_results in the state
                current_api_results = (state.api_results or {}).copy() # Get existing or default to empty
                current_api_results[effective_id] = api_call_result
                output_state_update["api_results"] = current_api_results
                
                # Update extracted_ids in the state if any data was extracted
                if extracted_data:
                    current_extracted_ids = (state.extracted_ids or {}).copy() # Get existing or default
                    current_extracted_ids.update(extracted_data) # Merge new data
                    output_state_update["extracted_ids"] = current_extracted_ids
                
                # Clear any error from previous attempts for this node if successful now
                if "error" in output_state_update and not api_call_result.get("error"):
                    output_state_update.pop("error", None)
                elif api_call_result.get("error"): # Propagate error from API call if present
                    output_state_update["error"] = api_call_result.get("error")

                logger.info(f"--- [Graph 2 Node] End: {effective_id} (Status: {api_call_result.get('status_code', 'N/A')}) ---")
                return output_state_update

            except Exception as e:
                error_message = f"Critical error in node {effective_id} execution logic: {type(e).__name__} - {e}"
                logger.error(error_message, exc_info=True)
                output_state_update["error"] = error_message
                # Ensure api_results structure is maintained even on internal node error
                current_api_results_on_error = (state.api_results or {}).copy()
                current_api_results_on_error[effective_id] = {"error": error_message, "status_code": "NODE_INTERNAL_EXCEPTION", "path_template": node_definition.path, "method": node_definition.method}
                output_state_update["api_results"] = current_api_results_on_error
                return output_state_update
        return node_executor

    def _build_and_compile_graph(self) -> Any:
        """
        Builds and compiles the LangGraph StateGraph for API execution based on the graph_plan.
        """
        logger.info(f"ExecutionGraphDefinition: Building Graph 2. Plan description: {self.graph_plan.description or 'N/A'}")
        builder = StateGraph(ExecutionGraphState)
        
        if not self.graph_plan.nodes:
            raise ValueError("Execution plan (GraphOutput from Graph 1) must contain at least one node to build Graph 2.")

        actual_api_nodes_defs: List[Node] = [] # Stores Node definitions for actual API calls
        nodes_requiring_interrupt_before: list[str] = [] # For LangGraph's interrupt_before

        # Add nodes to the builder
        for node_def in self.graph_plan.nodes:
            node_effective_id_str = str(node_def.effective_id).strip() if node_def.effective_id else ""
            
            if not node_effective_id_str: # Should not happen if GraphOutput is validated
                logger.error(f"Node with operationId '{node_def.operationId}' has an empty or None effective_id. Skipping.")
                continue
            
            # Skip conceptual START_NODE and END_NODE from Graph 1's plan, as LangGraph has its own START/END
            if node_effective_id_str.upper() in ["START_NODE", "END_NODE"]:
                continue 
            
            # Validate that API nodes have essential fields
            if not node_def.method or not node_def.path:
                raise ValueError(f"API Node '{node_effective_id_str}' in plan must have 'method' and 'path' defined for Graph 2 execution.")
            
            builder.add_node(node_effective_id_str, self._make_node_runnable(node_def))
            actual_api_nodes_defs.append(node_def)
            
            if node_def.requires_confirmation:
                nodes_requiring_interrupt_before.append(node_effective_id_str)
                logger.info(f"Node '{node_effective_id_str}' marked for 'interrupt_before' due to requires_confirmation=True.")

        # Handle empty graph case (no actual API nodes after filtering START/END)
        if not actual_api_nodes_defs:
            logger.warning("No executable API nodes found in the plan (after filtering START_NODE/END_NODE). Building a minimal START -> END graph for Graph 2.")
            # LangGraph requires at least one node for compilation if not using START/END directly.
            # Add dummy nodes if no actual API nodes are present.
            builder.add_node("__compiler_dummy_start__", lambda x: x) 
            builder.add_node("__compiler_dummy_end__", lambda x: x)     
            builder.set_entry_point("__compiler_dummy_start__")
            builder.add_edge("__compiler_dummy_start__", "__compiler_dummy_end__")
            builder.set_finish_point("__compiler_dummy_end__") 
            return builder.compile(checkpointer=MemorySaver()) # Use MemorySaver for execution graph's checkpointer

        executable_node_ids_in_builder = {str(n.effective_id).strip() for n in actual_api_nodes_defs}
        has_start_edge_from_plan = False
        
        if not self.graph_plan.edges:
            logger.warning("Execution plan (Graph 1) has no edges defined. This may lead to an unconnected Graph 2 if not handled by entry/finish point logic.")

        # Add edges based on the plan from Graph 1
        for edge_idx, edge in enumerate(self.graph_plan.edges):
            plan_from_node_original_case = str(edge.from_node).strip() if edge.from_node else ""
            plan_to_node_original_case = str(edge.to_node).strip() if edge.to_node else ""
            
            if not plan_from_node_original_case: raise ValueError(f"Edge {edge_idx + 1} in Graph 1 plan has an empty 'from_node'.")
            if not plan_to_node_original_case: raise ValueError(f"Edge {edge_idx + 1} in Graph 1 plan has an empty 'to_node'.")

            is_plan_source_start_node = plan_from_node_original_case.upper() == "START_NODE"
            is_plan_target_end_node = plan_to_node_original_case.upper() == "END_NODE"

            # Determine source and target for LangGraph's builder
            source_for_builder = START if is_plan_source_start_node else plan_from_node_original_case
            target_for_builder = END if is_plan_target_end_node else plan_to_node_original_case

            # Validate that non-START/END nodes in edges exist in the builder
            if source_for_builder != START and plan_from_node_original_case not in executable_node_ids_in_builder:
                raise ValueError(f"Edge source '{plan_from_node_original_case}' (from Graph 1 plan) not found in Graph 2 builder nodes: {executable_node_ids_in_builder}")
            if target_for_builder != END and plan_to_node_original_case not in executable_node_ids_in_builder:
                raise ValueError(f"Edge target '{plan_to_node_original_case}' (from Graph 1 plan) not found in Graph 2 builder nodes: {executable_node_ids_in_builder}")
            
            try:
                builder.add_edge(source_for_builder, target_for_builder)
                logger.debug(f"Added edge to Graph 2 builder: {source_for_builder} --> {target_for_builder} (from plan edge: {plan_from_node_original_case} -> {plan_to_node_original_case})")
            except Exception as e_add_edge: # Catch LangGraph specific errors if any
                source_log = 'LANGGRAPH.START' if source_for_builder == START else source_for_builder
                target_log = 'LANGGRAPH.END' if target_for_builder == END else target_for_builder
                raise ValueError(f"LangGraph (Graph 2) failed to add edge ('{source_log}' -> '{target_log}'). Error: {e_add_edge}")

            if source_for_builder == START:
                has_start_edge_from_plan = True # Flag if Graph 1's plan defined entry points
        
        # Set entry point for Graph 2
        graph_entry_node_id_if_not_start: Optional[str] = None # Tracks if first API node is entry

        if has_start_edge_from_plan:
            # If edges from START_NODE are defined in Graph 1's plan, LangGraph infers START as the entry point.
            # No explicit builder.set_entry_point(START) needed.
            logger.info("Graph 2 entry point is LANGGRAPH.START (inferred from Graph 1 plan edges originating from START_NODE).")
        elif actual_api_nodes_defs: 
            # If no START_NODE edges in plan, set the first *actual API node* as the entry point
            entry_point_candidate_str = str(actual_api_nodes_defs[0].effective_id).strip()
            if entry_point_candidate_str not in executable_node_ids_in_builder: # Should not happen
                 raise ValueError(f"Default entry point candidate '{entry_point_candidate_str}' for Graph 2 is not among executable nodes: {executable_node_ids_in_builder}")
            builder.set_entry_point(entry_point_candidate_str)
            graph_entry_node_id_if_not_start = entry_point_candidate_str 
            logger.info(f"No explicit START_NODE edge in Graph 1 plan. Graph 2 entry point set to first API node: '{entry_point_candidate_str}'.")
        else: # Should be caught by the "no actual_api_nodes_defs" check earlier
            logger.critical("Graph 2 building logic error: Reached unexpected state for entry point setting.")
            raise RuntimeError("Failed to determine a valid entry point for Graph 2 (unexpected state).")

        # Set finish points for Graph 2: Nodes that don't lead to other API nodes or END_NODE in the plan
        if actual_api_nodes_defs:
            for node_d in actual_api_nodes_defs:
                node_id_s = str(node_d.effective_id).strip()
                
                # Check if this node_id_s is a source in any edge leading to another *executable API node* or to the conceptual END_NODE from the plan
                is_source_to_another_api_node_or_plan_end = any(
                    str(e.from_node).strip() == node_id_s and
                    ( (str(e.to_node).strip().upper() == "END_NODE") or (str(e.to_node).strip() in executable_node_ids_in_builder) )
                    for e in self.graph_plan.edges
                )
                
                if not is_source_to_another_api_node_or_plan_end:
                    # This node is a leaf in the API call sequence according to the plan
                    # (or it only leads to END_NODE, which is handled by add_edge(..., END))
                    if node_id_s == graph_entry_node_id_if_not_start and not any(str(e.from_node).strip() == node_id_s and str(e.to_node).strip().upper() == "END_NODE" for e in self.graph_plan.edges):
                        # If the entry point is also a leaf and doesn't explicitly go to END_NODE in plan, add edge to LangGraph's END
                        try:
                            builder.add_edge(node_id_s, END)
                            logger.info(f"Graph 2 entry point node '{node_id_s}' is also a plan leaf and had no explicit edge to END_NODE. Added edge to LANGGRAPH.END.")
                        except Exception as e_add_final: # e.g. if edge already exists due to plan "node_id_s -> END_NODE"
                            logger.warning(f"Could not add edge from Graph 2 entry leaf '{node_id_s}' to LANGGRAPH.END: {e_add_final}. It might already exist if plan had '{node_id_s} -> END_NODE'.")
                    elif node_id_s != graph_entry_node_id_if_not_start: # For non-entry leaf nodes
                         try:
                            # If a non-entry node is a leaf in the plan (doesn't go to another API node or END_NODE),
                            # and it doesn't have an explicit edge to END_NODE in the plan,
                            # it should be a finish point for LangGraph.
                            # However, if it *does* have an edge to END_NODE in the plan, that add_edge(..., END) call handles it.
                            # So, we only set_finish_point if there's no explicit edge to END_NODE.
                            if not any(str(e.from_node).strip() == node_id_s and str(e.to_node).strip().upper() == "END_NODE" for e in self.graph_plan.edges):
                                builder.set_finish_point(node_id_s)
                                logger.info(f"Node '{node_id_s}' (a plan leaf with no explicit edge to END_NODE) set as LangGraph finish point for Graph 2.")
                         except Exception as e_set_finish:
                             logger.error(f"Failed to set node '{node_id_s}' as finish point for Graph 2: {e_set_finish}", exc_info=True)
        
        logger.info(f"Compiling execution graph (Graph 2). Nodes marked for interrupt_before: {nodes_requiring_interrupt_before}")
        # Each execution graph (Graph 2) instance should have its own memory for checkpoints/interrupts.
        memory_saver_for_graph2 = MemorySaver() 
        
        try:
            compiled_graph = builder.compile(
                checkpointer=memory_saver_for_graph2, 
                interrupt_before=nodes_requiring_interrupt_before if nodes_requiring_interrupt_before else None,
                debug=True # Enable debug for more detailed logging from LangGraph itself
            )
            logger.info("Execution graph (Graph 2) compiled successfully.")
            return compiled_graph
        except Exception as e_compile:
            logger.critical(f"Execution graph (Graph 2) compilation failed: {e_compile}", exc_info=True)
            # Log details of the graph structure attempt
            logger.error(f"Graph 2 Builder details before crash: Nodes: {list(builder.nodes.keys())}, Edges: {builder.edges}, Entry: {builder.entry_point}, Finish: {list(builder.finish_points)}")
            raise

    def get_runnable_graph(self) -> Any:
        """Returns the compiled runnable LangGraph (Graph 2)."""
        return self.runnable_graph

    def get_node_definition(self, node_effective_id: Union[str, Tuple[str, ...], List[str]]) -> Optional[Node]:
        """
        Retrieves the original Node definition (from Graph 1's plan) for a given effective_id.
        This is useful for the UI or other parts of Graph 2 to get details about a node.
        """
        target_id_str: Optional[str] = None
        if isinstance(node_effective_id, str):
            target_id_str = node_effective_id.strip()
        elif isinstance(node_effective_id, (list, tuple)) and node_effective_id: # Handle cases where LangGraph might pass list of next nodes
            potential_id = node_effective_id[0]
            if isinstance(potential_id, str):
                target_id_str = potential_id.strip()
                logger.warning(
                    f"get_node_definition was called with a list/tuple: {node_effective_id}. "
                    f"Using its first element '{target_id_str}' as the target ID. "
                    f"This might occur if LangGraph's 'next' field contains multiple possible next nodes for an interrupt."
                )
            else:
                logger.error(f"get_node_definition received a list/tuple whose first element is not a string: {node_effective_id}")
                return None
        else: # Invalid type
            logger.error(f"get_node_definition received an invalid type for node_effective_id: {type(node_effective_id)}")
            return None
        
        if target_id_str is None: # Should not happen if logic above is correct
            return None

        for node in self.graph_plan.nodes: # Search in the original plan from Graph 1
            current_node_eff_id = str(node.effective_id).strip() if node.effective_id else ""
            if current_node_eff_id == target_id_str:
                return node
        
        logger.warning(f"Node definition not found in Graph 1 plan for effective_id: '{target_id_str}'")
        return None

