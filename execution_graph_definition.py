import asyncio
import logging
from typing import Any, Callable, Awaitable, Dict, Optional, Tuple, Union

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
        if not isinstance(template_string, str):
            return str(template_string)
        resolved_string = template_string
        extracted_ids_dict = state.extracted_ids
        initial_input_dict = state.initial_input if isinstance(state.initial_input, dict) else {}
        # Resolve from extracted_ids first
        for key, value in extracted_ids_dict.items():
            if value is not None:
                resolved_string = resolved_string.replace(f"{{{key}}}", str(value))
        # Then resolve from initial_input for any remaining placeholders
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
        if not node_definition.input_mappings:
            return
        current_extracted_ids = state.extracted_ids # Data comes from the shared pool
        initial_input_data = state.initial_input if isinstance(state.initial_input, dict) else {}

        for mapping in node_definition.input_mappings:
            source_value = None
            # Try to get value from extracted_ids first
            if mapping.source_data_path: # Ensure path is not empty
                source_value = _get_value_from_path(current_extracted_ids, mapping.source_data_path)

            # If not found in extracted_ids, try from initial_input (if path is simple key)
            if source_value is None and mapping.source_data_path and not any(c in mapping.source_data_path for c in ['.', '[', ']']): # Simple key
                 source_value = _get_value_from_path(initial_input_data, mapping.source_data_path)


            if source_value is None:
                logger.warning(f"Node '{node_definition.effective_id}': InputMapping - Could not find source data for path '{mapping.source_data_path}' in extracted_ids or initial_input. Skipping.")
                continue

            target_param_name = mapping.target_parameter_name
            target_param_in = mapping.target_parameter_in

            if target_param_in == "path":
                resolved_path_params[target_param_name] = str(source_value)
            elif target_param_in == "query":
                resolved_query_params[target_param_name] = source_value
            elif target_param_in == "header":
                resolved_headers[target_param_name] = str(source_value)
            elif target_param_in.startswith("body."):
                field_path_in_body = target_param_in.split("body.", 1)[1]
                _set_value_by_path(resolved_body_payload_for_field_mapping, field_path_in_body, source_value)
            elif target_param_in == "body": # To replace the entire body
                 # This case is handled after placeholder resolution in _prepare_api_request_components
                 # by checking input_mappings again. For now, we note it.
                 logger.debug(f"Node '{node_definition.effective_id}': InputMapping for full body replacement noted for '{target_param_name}'. Will be applied later.")


    def _prepare_api_request_components(
        self, node_definition: Node, state: ExecutionGraphState
    ) -> Tuple[str, Dict[str, Any], Any, Dict[str, Any]]:
        path_template = node_definition.path or ""
        payload_template = node_definition.payload
        # Deep copy the payload template if it's a dictionary to avoid modifying the original
        if isinstance(payload_template, dict):
            payload_template = {k: v for k, v in payload_template.items()} # Simple deep copy for one level

        # 1. Resolve placeholders in path and body templates
        resolved_path_from_template = self._resolve_placeholders(path_template, state)
        
        body_after_placeholder_resolution: Any
        if isinstance(payload_template, dict):
            body_after_placeholder_resolution = {
                key: (self._resolve_placeholders(value, state) if isinstance(value, str) else value)
                for key, value in payload_template.items()
            }
        elif payload_template is not None: # If payload is a string or other primitive
            body_after_placeholder_resolution = self._resolve_placeholders(str(payload_template), state)
        else: # No payload template
            body_after_placeholder_resolution = {}


        # Initialize containers for final request components
        final_path_params: Dict[str, Any] = {} # For path parameters that might need re-resolution
        final_query_params: Dict[str, Any] = {}
        final_headers: Dict[str, Any] = {}
        # Use the placeholder-resolved body as the base for field mappings
        body_for_field_mappings = body_after_placeholder_resolution if isinstance(body_after_placeholder_resolution, dict) else {}


        # 2. Apply input mappings (this populates final_query_params, final_headers, and modifies body_for_field_mappings)
        self._apply_input_mappings(
            node_definition, state, final_path_params, final_query_params,
            body_for_field_mappings, final_headers
        )

        # Determine the final body payload
        final_body_payload = body_for_field_mappings # Start with the (potentially modified by mapping) body

        # Check if any input mapping targets the entire body
        # This type of mapping should override any previous body content.
        if node_definition.input_mappings:
            for mapping in node_definition.input_mappings:
                if mapping.target_parameter_in == "body":
                    current_data_sources = {**state.initial_input, **state.extracted_ids} if isinstance(state.initial_input, dict) else state.extracted_ids
                    source_value = _get_value_from_path(current_data_sources, mapping.source_data_path)
                    if source_value is not None:
                        final_body_payload = source_value # Replace entire body
                        logger.info(f"Node '{node_definition.effective_id}': Entire request body replaced by input mapping from '{mapping.source_data_path}'.")
                        break # Assume only one mapping should replace the entire body

        # 3. Re-resolve path if path_params were populated by input_mappings
        final_api_path = resolved_path_from_template
        if final_path_params: # Path parameters were added/modified by input mappings
            # Create a temporary state with these path params for re-resolution
            # This ensures that placeholders in the path template (e.g., /items/{itemId})
            # can be filled by values derived from input mappings.
            temp_ids_for_path_resolution = {
                **(state.initial_input if isinstance(state.initial_input, dict) else {}),
                **state.extracted_ids,
                **final_path_params # Mapped path params take precedence
            }
            temp_state_for_path_re_resolution = ExecutionGraphState(
                extracted_ids=temp_ids_for_path_resolution,
                api_results={}, confirmed_data={}, initial_input={} # Other fields not needed for path re-resolution
            )
            final_api_path = self._resolve_placeholders(path_template, temp_state_for_path_re_resolution)
            logger.debug(f"Node '{node_definition.effective_id}': API path re-resolved to '{final_api_path}' after applying path params from input mappings.")

        return final_api_path, final_query_params, final_body_payload, final_headers


    def _apply_confirmed_data_to_request(
        self, node_definition: Node, state: ExecutionGraphState,
        current_body_payload: Any, current_query_params: Dict[str, Any], current_headers: Dict[str, Any]
    ) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        operationId = node_definition.effective_id
        confirmation_key = f"confirmed_{operationId}"
        
        updated_body = current_body_payload
        if isinstance(current_body_payload, dict): # Ensure deep copy for dicts
            updated_body = {k:v for k,v in current_body_payload.items()}
        
        updated_params = current_query_params.copy()
        updated_headers = current_headers.copy()
        
        current_confirmed_data = state.confirmed_data
        
        # Check if this operationId was part of a confirmation decision
        if current_confirmed_data.get(confirmation_key): 
            confirmed_details = current_confirmed_data.get(f"{confirmation_key}_details", {})
            
            # If user provided a modified payload during confirmation, use it
            if "modified_payload" in confirmed_details: 
                updated_body = confirmed_details["modified_payload"]
                logger.info(f"Node '{operationId}': Applied modified payload from user confirmation.")
            
            # Potentially extend this to apply modified query_params or headers if your UI supports it
            # For example:
            # if "modified_query_params" in confirmed_details:
            #     updated_params.update(confirmed_details["modified_query_params"])
            #     logger.info(f"Node '{operationId}': Applied modified query params from user confirmation.")
            # if "modified_headers" in confirmed_details:
            #     updated_headers.update(confirmed_details["modified_headers"])
            #     logger.info(f"Node '{operationId}': Applied modified headers from user confirmation.")
                
        return updated_body, updated_params, updated_headers


    async def _execute_api_and_process_outputs(
        self, node_definition: Node, api_path: str,
        query_params: Dict[str, Any], body_payload: Any, headers: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
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
            if isinstance(response_body, (dict, list)): # Allow list for top-level array responses
                for mapping in node_definition.output_mappings:
                    extracted_value = _get_value_from_path(response_body, mapping.source_data_path)
                    if extracted_value is not None:
                        extracted_data_for_state[mapping.target_data_key] = extracted_value
                    else:
                        logger.warning(f"Node '{effective_id}': OutputMapping - Could not extract value for path '{mapping.source_data_path}' from response.")
            elif response_body is not None: # Response body exists but isn't dict/list
                 logger.warning(f"Node '{effective_id}': Response body type {type(response_body)} is not dict or list. Cannot apply output mappings directly. Response preview: {str(response_body)[:100]}")
                 # If there's only one output mapping and it targets a simple key,
                 # and the response_body is a primitive, we could assign it directly.
                 if len(node_definition.output_mappings) == 1 and not any(c in node_definition.output_mappings[0].source_data_path for c in ['.', '[', ']']):
                     simple_key = node_definition.output_mappings[0].target_data_key
                     extracted_data_for_state[simple_key] = response_body
                     logger.info(f"Node '{effective_id}': Applied non-dict/list response directly to target_data_key '{simple_key}' due to simple output mapping.")


        return api_call_result_dict, (extracted_data_for_state if extracted_data_for_state else None)

    def _make_node_runnable(
        self, node_definition: Node
    ) -> Callable[[ExecutionGraphState], Awaitable[Dict[str, Any]]]:
        async def node_executor(state: ExecutionGraphState) -> Dict[str, Any]:
            effective_id = node_definition.effective_id
            logger.info(f"--- [Graph 2 Node] Start: {effective_id} (OpID: {node_definition.operationId}) ---")
            
            output_state_update: Dict[str, Any] = {} 

            try:
                api_path, query_params, body_payload, headers = self._prepare_api_request_components(node_definition, state)
                
                logger.debug(f"Node '{effective_id}': Prepared components - Path: {api_path}, Query: {query_params}, Body Type: {type(body_payload)}, Headers: {headers}")


                if node_definition.requires_confirmation:
                    confirmation_key = f"confirmed_{effective_id}"
                    # Check if confirmation was explicitly given and was True
                    if not state.confirmed_data.get(confirmation_key, False): # Default to False if key not present
                        skip_message = f"Node '{effective_id}' requires confirmation, but it was not found or was negative in confirmed_data. Skipping execution."
                        logger.warning(skip_message)
                        output_state_update["error"] = skip_message 
                        output_state_update["api_results"] = {effective_id: {"status_code": "SKIPPED_NO_CONFIRMATION", "error": skip_message, "path_template": node_definition.path, "method": node_definition.method}}
                        return output_state_update

                # Apply any data modifications from the confirmation step (e.g., user edited payload)
                final_body, final_params, final_headers = self._apply_confirmed_data_to_request(
                    node_definition, state, body_payload, query_params, headers
                )
                
                logger.debug(f"Node '{effective_id}': Proceeding with API call execution. Final Body Type: {type(final_body)}")
                api_call_result, extracted_data = await self._execute_api_and_process_outputs(
                    node_definition, api_path, final_params, final_body, final_headers
                )
                
                # Store the full API call result under the node's effective_id
                current_api_results = state.api_results.copy() if state.api_results else {}
                current_api_results[effective_id] = api_call_result
                output_state_update["api_results"] = current_api_results
                
                if extracted_data:
                    # Merge extracted data with existing extracted_ids
                    current_extracted_ids = state.extracted_ids.copy() if state.extracted_ids else {}
                    current_extracted_ids.update(extracted_data)
                    output_state_update["extracted_ids"] = current_extracted_ids
                
                logger.info(f"--- [Graph 2 Node] End: {effective_id} (Execution result captured) ---")
                return output_state_update

            except Exception as e:
                error_message = f"Error in node {effective_id}: {type(e).__name__} - {e}"
                logger.error(error_message, exc_info=True)
                output_state_update["error"] = error_message
                current_api_results_on_error = state.api_results.copy() if state.api_results else {}
                current_api_results_on_error[effective_id] = {"error": error_message, "status_code": "NODE_EXCEPTION", "path_template": node_definition.path, "method": node_definition.method}
                output_state_update["api_results"] = current_api_results_on_error
                return output_state_update
        return node_executor

    def _build_and_compile_graph(self) -> Any:
        logger.info(f"ExecutionGraphDefinition: Building graph. Plan: {self.graph_plan.description or 'N/A'}")
        builder = StateGraph(ExecutionGraphState)
        
        if not self.graph_plan.nodes:
            logger.error("Execution plan must contain at least one node. Compilation will likely fail or be empty.")
            # Depending on LangGraph version, compiling an empty graph might raise an error or return a non-runnable graph.
            # It's better to raise an error here if it's truly invalid.
            # For now, let it proceed and LangGraph will handle it.
            # Consider: raise ValueError("Execution plan must contain at least one node.")

        actual_api_nodes_defs = []
        nodes_requiring_interrupt_before: list[str] = [] 

        for node_def in self.graph_plan.nodes:
            node_id_str = str(node_def.effective_id).strip()
            # Skip conceptual START_NODE and END_NODE from being added as executable LangGraph nodes
            if node_id_str.upper() in ["START_NODE", "END_NODE"]:
                logger.debug(f"Skipping conceptual node '{node_id_str}' from direct addition to LangGraph builder.")
                continue
            
            # Validate essential fields for API nodes
            if not node_def.method or not node_def.path:
                # This should ideally be caught during Graph 1's verification.
                # If it reaches here, it's a problem with the plan.
                logger.error(f"API Node '{node_id_str}' (OpID: {node_def.operationId}) is missing 'method' or 'path'. This node will likely fail if called.")
                # Depending on strictness, you might raise ValueError here.
                # For now, add it and let it fail at runtime if called without these.

            builder.add_node(node_id_str, self._make_node_runnable(node_def))
            actual_api_nodes_defs.append(node_def) # Keep track of nodes actually added to builder
            
            if node_def.requires_confirmation:
                nodes_requiring_interrupt_before.append(node_id_str)
                logger.info(f"Node '{node_id_str}' marked for 'interrupt_before' due to requires_confirmation=True.")


        executable_node_ids_in_builder = {str(n.effective_id).strip() for n in actual_api_nodes_defs}
        has_start_edge_from_plan = False
        
        if not self.graph_plan.edges and actual_api_nodes_defs:
            logger.warning("Execution plan has no edges, but has API nodes. This might lead to an unconnected graph if not handled.")
        elif not self.graph_plan.edges and not actual_api_nodes_defs:
            logger.warning("Execution plan has no edges and no executable API nodes. Graph will be empty.")


        for edge in self.graph_plan.edges:
            plan_from_node_original_case = str(edge.from_node).strip()
            plan_to_node_original_case = str(edge.to_node).strip()

            # Determine source for LangGraph: START constant or actual node ID
            is_plan_source_start_node = plan_from_node_original_case.upper() == "START_NODE"
            source_for_builder = START if is_plan_source_start_node else plan_from_node_original_case
            
            # Determine target for LangGraph: END constant or actual node ID
            is_plan_target_end_node = plan_to_node_original_case.upper() == "END_NODE"
            target_for_builder = END if is_plan_target_end_node else plan_to_node_original_case

            # Validate that non-START/END nodes in edges exist in the builder
            if source_for_builder != START and plan_from_node_original_case not in executable_node_ids_in_builder:
                logger.error(f"Edge source node '{plan_from_node_original_case}' from plan is not an executable node in the builder. Skipping edge. Available: {executable_node_ids_in_builder}")
                continue
            if target_for_builder != END and plan_to_node_original_case not in executable_node_ids_in_builder:
                logger.error(f"Edge target node '{plan_to_node_original_case}' from plan is not an executable node in the builder. Skipping edge. Available: {executable_node_ids_in_builder}")
                continue
            
            try:
                builder.add_edge(source_for_builder, target_for_builder)
                logger.debug(f"Added edge: {source_for_builder} --> {target_for_builder}")
                if source_for_builder == START:
                    has_start_edge_from_plan = True
            except Exception as e_add_edge:
                 logger.error(f"Failed to add edge {source_for_builder} -> {target_for_builder}: {e_add_edge}", exc_info=True)

        # Set entry point
        if has_start_edge_from_plan:
            builder.set_entry_point(START) 
            logger.info("Entry point set to START (due to explicit edge from START_NODE in plan).")
        elif actual_api_nodes_defs: # No explicit START_NODE edge, but there are API nodes
            # Default to the first API node encountered in the plan as the entry point
            # This assumes nodes are somewhat ordered in the plan if no START_NODE edge.
            entry_point_candidate = str(actual_api_nodes_defs[0].effective_id).strip()
            builder.set_entry_point(entry_point_candidate) 
            logger.info(f"No explicit START_NODE edge. Entry point heuristically set to first API node: '{entry_point_candidate}'.")
        else: # No explicit START_NODE edge and no API nodes (empty or only conceptual nodes)
             logger.warning("Graph has no executable API nodes and no explicit entry point from START_NODE. LangGraph might default or error.")
             # LangGraph might require an entry point even for an empty graph if compiled.
             # If you intend to support empty runnable graphs, you might need a dummy entry that goes to END.
             # For now, this state is considered problematic.

        # Set finish points for nodes that are leaves or only point to END_NODE
        if actual_api_nodes_defs:
            all_source_ids_in_edges = {str(e.from_node).strip() for e in self.graph_plan.edges if str(e.from_node).strip().upper() != "START_NODE"}
            
            for node_def in actual_api_nodes_defs:
                node_id = str(node_def.effective_id).strip()
                
                # A node is a finish point if it's not a source for any edge leading to another API node
                is_source_for_api_edge = any(
                    str(e.from_node).strip() == node_id and 
                    str(e.to_node).strip().upper() != "END_NODE" and # Edge does not go to conceptual END
                    str(e.to_node).strip() in executable_node_ids_in_builder # Edge goes to another actual API node
                    for e in self.graph_plan.edges
                )
                
                # An explicit edge from this node to the conceptual "END_NODE" means LangGraph handles its termination.
                has_explicit_edge_to_end_node = any(
                    str(e.from_node).strip() == node_id and 
                    str(e.to_node).strip().upper() == "END_NODE" 
                    for e in self.graph_plan.edges
                )

                if not is_source_for_api_edge and not has_explicit_edge_to_end_node:
                    # This node is a leaf among API nodes and doesn't have an explicit edge to "END_NODE" in the plan.
                    # So, it should be a finish point in the LangGraph.
                    try:
                        builder.set_finish_point(node_id)
                        logger.info(f"Node '{node_id}' is a leaf or points only to END implicitly. Setting as LangGraph finish point.")
                    except Exception as e_set_finish:
                        logger.error(f"Failed to set node '{node_id}' as finish point: {e_set_finish}", exc_info=True)
        
        logger.info(f"Compiling execution graph. Nodes for interrupt_before: {nodes_requiring_interrupt_before}")
        memory_saver = MemorySaver() # Instantiate the checkpointer
        
        try:
            compiled_graph = builder.compile(
                checkpointer=memory_saver, 
                interrupt_before=nodes_requiring_interrupt_before if nodes_requiring_interrupt_before else None
                # Consider adding interrupt_after if needed for some nodes
            )
            logger.info("Execution graph compiled successfully.")
            return compiled_graph
        except Exception as e_compile:
            logger.critical(f"Execution graph compilation failed: {e_compile}", exc_info=True)
            # Depending on desired behavior, you might re-raise or return a non-runnable indicator
            raise # Re-raise for now, as a non-compiled graph is a critical failure


    def get_runnable_graph(self) -> Any:
        return self.runnable_graph

    def get_node_definition(self, node_effective_id: Union[str, Tuple[str, ...], List[str]]) -> Optional[Node]:
        """
        Retrieves a node definition by its effective_id.
        Handles cases where node_effective_id might be passed as a string or a list/tuple containing the string.
        """
        target_id_str: Optional[str] = None

        if isinstance(node_effective_id, str):
            target_id_str = node_effective_id.strip()
        elif isinstance(node_effective_id, (list, tuple)) and node_effective_id:
            # If it's a list/tuple, assume the first element is the ID.
            # This addresses the user's observation that node_effective_id was coming as node_effective_id[0].
            # It's still recommended to fix the caller to pass a simple string.
            potential_id = node_effective_id[0]
            if isinstance(potential_id, str):
                target_id_str = potential_id.strip()
                logger.warning(
                    f"get_node_definition was called with a list/tuple: {node_effective_id}. "
                    f"Using its first element '{target_id_str}'. "
                    f"Consider fixing the caller to pass a string ID."
                )
            else:
                logger.error(f"get_node_definition received a list/tuple whose first element is not a string: {node_effective_id}")
                return None
        else:
            logger.error(f"get_node_definition received an invalid type for node_effective_id: {type(node_effective_id)}")
            return None

        if target_id_str is None:
            return None

        for node in self.graph_plan.nodes:
            if str(node.effective_id).strip() == target_id_str:
                return node
        logger.warning(f"Node definition not found for effective_id: '{target_id_str}'")
        return None

