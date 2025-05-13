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
        if not isinstance(template_string, str):
            return str(template_string)
        resolved_string = template_string
        placeholders_data = {**(state.initial_input or {}), **(state.extracted_ids or {})}

        for key, value in placeholders_data.items():
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
        
        available_data_for_mapping = {**(state.initial_input or {}), **(state.extracted_ids or {})}

        for mapping in node_definition.input_mappings:
            source_value = None
            if mapping.source_data_path: 
                source_value = _get_value_from_path(available_data_for_mapping, mapping.source_data_path)

            if source_value is None:
                logger.warning(f"Node '{node_definition.effective_id}': InputMapping - Could not find source data for path '{mapping.source_data_path}' in available data. Skipping.")
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
            elif target_param_in == "body":
                 logger.debug(f"Node '{node_definition.effective_id}': InputMapping for full body replacement noted for '{target_param_name}'. Applied in _prepare_api_request_components.")


    def _prepare_api_request_components(
        self, node_definition: Node, state: ExecutionGraphState
    ) -> Tuple[str, Dict[str, Any], Any, Dict[str, Any]]:
        path_template = node_definition.path or ""
        payload_template = node_definition.payload
        
        if isinstance(payload_template, dict):
            current_payload_template = {k: v for k, v in payload_template.items()} 
        else:
            current_payload_template = payload_template

        resolved_path_from_template = self._resolve_placeholders(path_template, state)
        
        body_after_placeholder_resolution: Any
        if isinstance(current_payload_template, dict):
            body_after_placeholder_resolution = {
                key: (self._resolve_placeholders(value, state) if isinstance(value, str) else value)
                for key, value in current_payload_template.items()
            }
        elif current_payload_template is not None: 
            body_after_placeholder_resolution = self._resolve_placeholders(str(current_payload_template), state)
        else: 
            body_after_placeholder_resolution = {}

        final_path_params: Dict[str, Any] = {} 
        final_query_params: Dict[str, Any] = {}
        final_headers: Dict[str, Any] = {}
        body_for_field_mappings = body_after_placeholder_resolution if isinstance(body_after_placeholder_resolution, dict) else {}

        self._apply_input_mappings(
            node_definition, state, final_path_params, final_query_params,
            body_for_field_mappings, final_headers
        )

        final_body_payload = body_for_field_mappings 

        if node_definition.input_mappings:
            for mapping in node_definition.input_mappings:
                if mapping.target_parameter_in == "body":
                    current_data_sources = {**(state.initial_input or {}), **(state.extracted_ids or {})}
                    source_value = _get_value_from_path(current_data_sources, mapping.source_data_path)
                    if source_value is not None:
                        final_body_payload = source_value 
                        logger.info(f"Node '{node_definition.effective_id}': Entire request body replaced by input mapping from '{mapping.source_data_path}'.")
                        break 

        final_api_path = resolved_path_from_template
        if final_path_params: 
            temp_ids_for_path_resolution = {
                **(state.initial_input or {}),
                **(state.extracted_ids or {}),
                **final_path_params 
            }
            temp_state_for_path_re_resolution = ExecutionGraphState(
                extracted_ids=temp_ids_for_path_resolution,
                api_results={}, confirmed_data={}, initial_input=None 
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
        if isinstance(current_body_payload, dict): 
            updated_body = {k:v for k,v in current_body_payload.items()}
        
        updated_params = current_query_params.copy()
        updated_headers = current_headers.copy()
        
        current_confirmed_data = state.confirmed_data or {} 
        
        if current_confirmed_data.get(confirmation_key): 
            confirmed_details = current_confirmed_data.get(f"{confirmation_key}_details", {})
            
            if "modified_payload" in confirmed_details: 
                updated_body = confirmed_details["modified_payload"]
                logger.info(f"Node '{operationId}': Applied modified payload from user confirmation.")
                
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
            if isinstance(response_body, (dict, list)): 
                for mapping in node_definition.output_mappings:
                    extracted_value = _get_value_from_path(response_body, mapping.source_data_path)
                    if extracted_value is not None:
                        extracted_data_for_state[mapping.target_data_key] = extracted_value
                    else:
                        logger.warning(f"Node '{effective_id}': OutputMapping - Could not extract value for path '{mapping.source_data_path}' from response.")
            elif response_body is not None: 
                 logger.warning(f"Node '{effective_id}': Response body type {type(response_body)} is not dict or list. Cannot apply output mappings directly. Response preview: {str(response_body)[:100]}")
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
                    if not (state.confirmed_data or {}).get(confirmation_key, False): 
                        skip_message = f"Node '{effective_id}' requires confirmation, but it was not found or was negative in confirmed_data. Skipping execution."
                        logger.warning(skip_message)
                        output_state_update["error"] = skip_message 
                        output_state_update["api_results"] = {effective_id: {"status_code": "SKIPPED_NO_CONFIRMATION", "error": skip_message, "path_template": node_definition.path, "method": node_definition.method}}
                        return output_state_update

                final_body, final_params, final_headers = self._apply_confirmed_data_to_request(
                    node_definition, state, body_payload, query_params, headers
                )
                
                logger.debug(f"Node '{effective_id}': Proceeding with API call execution. Final Body Type: {type(final_body)}")
                api_call_result, extracted_data = await self._execute_api_and_process_outputs(
                    node_definition, api_path, final_params, final_body, final_headers
                )
                
                current_api_results = (state.api_results or {}).copy()
                current_api_results[effective_id] = api_call_result
                output_state_update["api_results"] = current_api_results
                
                if extracted_data:
                    current_extracted_ids = (state.extracted_ids or {}).copy()
                    current_extracted_ids.update(extracted_data)
                    output_state_update["extracted_ids"] = current_extracted_ids
                
                logger.info(f"--- [Graph 2 Node] End: {effective_id} (Execution result captured) ---")
                return output_state_update

            except Exception as e:
                error_message = f"Error in node {effective_id}: {type(e).__name__} - {e}"
                logger.error(error_message, exc_info=True)
                output_state_update["error"] = error_message
                current_api_results_on_error = (state.api_results or {}).copy()
                current_api_results_on_error[effective_id] = {"error": error_message, "status_code": "NODE_EXCEPTION", "path_template": node_definition.path, "method": node_definition.method}
                output_state_update["api_results"] = current_api_results_on_error
                return output_state_update
        return node_executor

    def _build_and_compile_graph(self) -> Any:
        logger.info(f"ExecutionGraphDefinition: Building graph. Plan description: {self.graph_plan.description or 'N/A'}")
        builder = StateGraph(ExecutionGraphState)
        
        if not self.graph_plan.nodes:
            # If the plan itself has no nodes, it's an invalid plan for execution.
            raise ValueError("Execution plan (GraphOutput) must contain at least one node to build Graph 2.")

        actual_api_nodes_defs = []
        nodes_requiring_interrupt_before: list[str] = []

        for node_def in self.graph_plan.nodes:
            # Ensure effective_id is a string and stripped. Handle if it's None.
            node_effective_id_str = str(node_def.effective_id).strip() if node_def.effective_id else ""
            if not node_effective_id_str: # Should not happen if plan is well-formed
                logger.error(f"Node with operationId '{node_def.operationId}' has an empty or None effective_id. Skipping.")
                continue

            if node_effective_id_str.upper() in ["START_NODE", "END_NODE"]:
                logger.debug(f"Skipping conceptual node '{node_effective_id_str}' from direct addition to LangGraph builder.")
                continue
            
            if not node_def.method or not node_def.path:
                # This is a critical error for an API node.
                raise ValueError(f"API Node '{node_effective_id_str}' in plan must have 'method' and 'path' defined.")
            
            builder.add_node(node_effective_id_str, self._make_node_runnable(node_def))
            actual_api_nodes_defs.append(node_def)
            
            if node_def.requires_confirmation:
                nodes_requiring_interrupt_before.append(node_effective_id_str)
                logger.info(f"Node '{node_effective_id_str}' marked for 'interrupt_before'.")

        # If, after filtering, there are no actual API nodes to execute
        if not actual_api_nodes_defs:
            logger.warning("No executable API nodes found in the plan. Building a minimal START -> END graph.")
            builder.set_entry_point(START)
            builder.add_edge(START, END)
            memory_saver = MemorySaver()
            return builder.compile(
                checkpointer=memory_saver,
                interrupt_before=nodes_requiring_interrupt_before if nodes_requiring_interrupt_before else None
            )

        executable_node_ids_in_builder = {str(n.effective_id).strip() for n in actual_api_nodes_defs}
        has_start_edge_from_plan = False
        
        if not self.graph_plan.edges:
            logger.warning("Execution plan has no edges defined. This might lead to an unconnected graph if not handled by entry/finish point logic.")

        for edge_idx, edge in enumerate(self.graph_plan.edges):
            plan_from_node_original_case = str(edge.from_node).strip() if edge.from_node else ""
            plan_to_node_original_case = str(edge.to_node).strip() if edge.to_node else ""
            
            if not plan_from_node_original_case:
                raise ValueError(f"Edge {edge_idx + 1} has an empty 'from_node'.")
            if not plan_to_node_original_case:
                raise ValueError(f"Edge {edge_idx + 1} has an empty 'to_node'.")

            plan_from_node_upper = plan_from_node_original_case.upper()
            plan_to_node_upper = plan_to_node_original_case.upper()
            
            is_plan_source_start_node = plan_from_node_upper == "START_NODE"
            is_plan_target_end_node = plan_to_node_upper == "END_NODE"

            source_for_builder = START if is_plan_source_start_node else plan_from_node_original_case
            target_for_builder = END if is_plan_target_end_node else plan_to_node_original_case

            if source_for_builder != START and plan_from_node_original_case not in executable_node_ids_in_builder:
                raise ValueError(f"Edge source '{plan_from_node_original_case}' not in builder nodes: {executable_node_ids_in_builder}")
            if target_for_builder != END and plan_to_node_original_case not in executable_node_ids_in_builder:
                raise ValueError(f"Edge target '{plan_to_node_original_case}' not in builder nodes: {executable_node_ids_in_builder}")
            
            try:
                builder.add_edge(source_for_builder, target_for_builder)
                logger.debug(f"Added edge: {source_for_builder} --> {target_for_builder}")
            except Exception as e_add_edge:
                source_log = 'LANGGRAPH_START' if source_for_builder == START else source_for_builder
                target_log = 'LANGGRAPH_END' if target_for_builder == END else target_for_builder
                raise ValueError(f"LangGraph failed to add edge ('{source_log}' -> '{target_log}'). Error: {e_add_edge}")

            if source_for_builder == START:
                has_start_edge_from_plan = True
        
        # Set entry point
        if has_start_edge_from_plan:
            builder.set_entry_point(START)
            logger.info("Entry point set to START (due to explicit edge from START_NODE in plan).")
        elif actual_api_nodes_defs: # Must have at least one actual node to be an entry point
            entry_point_candidate = str(actual_api_nodes_defs[0].effective_id).strip()
            # This check should pass if actual_api_nodes_defs is populated correctly
            if entry_point_candidate not in executable_node_ids_in_builder:
                 raise ValueError(f"Default entry point candidate '{entry_point_candidate}' is not among executable nodes: {executable_node_ids_in_builder}")
            builder.set_entry_point(entry_point_candidate)
            logger.info(f"No explicit START_NODE edge. Entry point set to first API node: '{entry_point_candidate}'.")
        else:
            # This case should be covered by the "no actual_api_nodes_defs" check earlier.
            # If it's reached, it's an unexpected state.
            logger.critical("Graph building: Reached unexpected state for entry point setting. No API nodes and no START edge, but not caught by empty graph logic.")
            raise RuntimeError("Failed to determine a valid entry point for the graph.")


        # Set finish points for nodes that are leaves or only point to the conceptual END_NODE
        if actual_api_nodes_defs:
            for node_def in actual_api_nodes_defs:
                node_id_str = str(node_def.effective_id).strip()
                
                # Check if this node is a source for any edge leading to another *executable* API node
                is_source_to_another_api_node = any(
                    str(e.from_node).strip() == node_id_str and
                    str(e.to_node).strip() in executable_node_ids_in_builder and # Target must be an API node
                    str(e.to_node).strip().upper() != "END_NODE" # And not the conceptual END_NODE
                    for e in self.graph_plan.edges
                )
                
                # Check if this node has an explicit edge to the conceptual "END_NODE" in the plan
                has_explicit_edge_to_plan_end_node = any(
                    str(e.from_node).strip() == node_id_str and
                    str(e.to_node).strip().upper() == "END_NODE"
                    for e in self.graph_plan.edges
                )

                # A node is a finish point if:
                # 1. It does not lead to any other executable API node.
                # 2. AND it does not have an explicit edge to the conceptual "END_NODE" in the plan
                #    (because if it did, LangGraph handles that edge to its `END` constant).
                if not is_source_to_another_api_node and not has_explicit_edge_to_plan_end_node:
                    # However, if this node is also the *only* node and the entry point,
                    # LangGraph might error if it's also set as a finish point directly without an edge to END.
                    # This is more subtly handled by ensuring single-node entry points have an edge to END if no other outgoing.
                    if len(actual_api_nodes_defs) == 1 and node_id_str == builder.entry_point and not has_explicit_edge_to_plan_end_node:
                        logger.info(f"Single node graph with entry point '{node_id_str}'. Ensuring it has a path to END.")
                        # If it doesn't already point to END via plan, add it.
                        # This check is a bit redundant if the above logic for single node graphs is perfect, but safe.
                        try:
                            builder.add_edge(node_id_str, END)
                            logger.info(f"Added explicit edge from single entry node '{node_id_str}' to END.")
                        except Exception as e: # Catch if edge already exists or other issues
                            logger.warning(f"Could not add edge from single entry node '{node_id_str}' to END: {e}. It might already exist or be handled.")
                    elif node_id_str != builder.entry_point or len(actual_api_nodes_defs) > 1 : # Not a single node entry point, or multiple nodes exist
                        try:
                            builder.set_finish_point(node_id_str)
                            logger.info(f"Node '{node_id_str}' set as finish point.")
                        except Exception as e_set_finish:
                            logger.error(f"Failed to set node '{node_id_str}' as finish point: {e_set_finish}", exc_info=True)

        logger.info(f"Compiling execution graph. Nodes for interrupt_before: {nodes_requiring_interrupt_before}")
        memory_saver = MemorySaver()
        
        try:
            compiled_graph = builder.compile(
                checkpointer=memory_saver, 
                interrupt_before=nodes_requiring_interrupt_before if nodes_requiring_interrupt_before else None
            )
            logger.info("Execution graph compiled successfully.")
            return compiled_graph
        except Exception as e_compile:
            logger.critical(f"Execution graph compilation failed: {e_compile}", exc_info=True)
            raise

    def get_runnable_graph(self) -> Any:
        return self.runnable_graph

    def get_node_definition(self, node_effective_id: Union[str, Tuple[str, ...], List[str]]) -> Optional[Node]:
        target_id_str: Optional[str] = None
        if isinstance(node_effective_id, str):
            target_id_str = node_effective_id.strip()
        elif isinstance(node_effective_id, (list, tuple)) and node_effective_id:
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

        if target_id_str is None: # Should be caught by above checks if input was not string or valid list/tuple
            return None

        for node in self.graph_plan.nodes:
            # Ensure comparison is robust if node.effective_id could be None
            current_node_eff_id = str(node.effective_id).strip() if node.effective_id else ""
            if current_node_eff_id == target_id_str:
                return node
        logger.warning(f"Node definition not found for effective_id: '{target_id_str}'")
        return None
