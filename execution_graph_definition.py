# execution_graph_definition.py
import asyncio
import logging
from typing import Any, Callable, Awaitable, Dict, Optional, Tuple, Union, List

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver # Using MemorySaver for Graph 2 instances

# Assuming models and APIExecutor are correctly imported
from models import GraphOutput, Node, Edge, InputMapping, OutputMapping, ExecutionGraphState
from api_executor import APIExecutor

logger = logging.getLogger(__name__)

# Helper functions _get_value_from_path and _set_value_by_path
# These should be robust as they are critical for data mapping.
def _get_value_from_path(data_dict: Optional[Dict[str, Any]], path: str) -> Any:
    if not path or not isinstance(data_dict, dict):
        return None
    if path.startswith("$."): # Support basic JSONPath-like start
        path = path[2:]
    keys = path.split('.')
    current_val = data_dict
    for key_part in keys:
        if isinstance(current_val, dict):
            if key_part not in current_val:
                # Try to see if key_part is an index for a list if the dict value is a list
                # This is a simple heuristic, full JSONPath is more complex
                if isinstance(current_val.get(keys[0] if keys[0] in current_val else None), list):
                    try:
                        idx = int(key_part)
                        list_val = current_val.get(keys[0])
                        if list_val and 0 <= idx < len(list_val):
                            current_val = list_val[idx]
                            continue
                        else: return None
                    except ValueError: return None # key_part is not an int for list access
                return None # key_part not in dict
            current_val = current_val.get(key_part)
        elif isinstance(current_val, list):
            try:
                idx = int(key_part)
                if 0 <= idx < len(current_val):
                    current_val = current_val[idx]
                else:
                    return None # Index out of bounds
            except ValueError:
                return None # key_part is not a valid integer index
        else:
            return None # Cannot traverse further
    return current_val

def _set_value_by_path(data_dict: Dict[str, Any], path: str, value: Any):
    if not path:
        return
    keys = path.split('.')
    current_level = data_dict
    for i, key in enumerate(keys[:-1]):
        # If a key is numeric, it might imply a list index, but this simple _set_value_by_path
        # primarily targets dicts. For lists within dicts, the path should ideally be handled
        # by the caller ensuring the list exists and the index is valid.
        # This function will create nested dicts if they don't exist.
        if key not in current_level or not isinstance(current_level.get(key), dict):
            current_level[key] = {}
        current_level = current_level[key] # type: ignore
    if keys:
        current_level[keys[-1]] = value


class ExecutionGraphDefinition:
    def __init__(
        self,
        graph_execution_plan: GraphOutput,
        api_executor: APIExecutor,
        disable_confirmation_prompts: bool = False # New parameter
    ):
        if not isinstance(graph_execution_plan, GraphOutput):
            raise TypeError("graph_execution_plan must be an instance of GraphOutput.")
        if not isinstance(api_executor, APIExecutor):
            raise TypeError("api_executor must be an instance of APIExecutor.")

        self.graph_plan = graph_execution_plan
        self.api_executor = api_executor
        self.disable_confirmation_prompts = disable_confirmation_prompts # Store the flag
        self.runnable_graph: Any = self._build_and_compile_graph()
        logger.info(
            f"ExecutionGraphDefinition initialized. Confirmations disabled: {self.disable_confirmation_prompts}. "
            f"Graph 2 compiled for plan: {self.graph_plan.graph_id or 'N/A'}."
        )

    def _resolve_placeholders(self, template_string: str, state: ExecutionGraphState) -> str:
        if not isinstance(template_string, str):
            logger.debug(f"Template string is not a string type: {type(template_string)}, returning as is.")
            return str(template_string)
        resolved_string = template_string
        placeholders_data = {**(state.initial_input or {}), **(state.extracted_ids or {})}
        if not placeholders_data:
            return resolved_string
        for key, value in placeholders_data.items():
            if value is not None:
                placeholder_to_find = "{{" + str(key) + "}}"
                str_value = str(value)
                if placeholder_to_find in resolved_string:
                    resolved_string = resolved_string.replace(placeholder_to_find, str_value)
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
            source_value = _get_value_from_path(available_data_for_mapping, mapping.source_data_path)
            if source_value is None:
                logger.warning(f"Node '{node_definition.effective_id}': InputMapping - Could not find source data for path '{mapping.source_data_path}'. Skipping.")
                continue
            target_param_name = mapping.target_parameter_name
            target_param_in = mapping.target_parameter_in
            if target_param_in == "path": resolved_path_params[target_param_name] = str(source_value)
            elif target_param_in == "query": resolved_query_params[target_param_name] = source_value
            elif target_param_in == "header": resolved_headers[target_param_name] = str(source_value)
            elif target_param_in.startswith("body."):
                field_path_in_body = target_param_in.split("body.", 1)[1]
                _set_value_by_path(resolved_body_payload_for_field_mapping, field_path_in_body, source_value)
            elif target_param_in == "body":
                 logger.debug(f"Node '{node_definition.effective_id}': Full body replacement noted for '{target_param_name}'.")


    def _prepare_api_request_components(
        self, node_definition: Node, state: ExecutionGraphState
    ) -> Tuple[str, Dict[str, Any], Any, Dict[str, Any]]:
        path_template = node_definition.path or ""
        payload_template = node_definition.payload
        current_payload_template_copy: Any = json.loads(json.dumps(payload_template)) if isinstance(payload_template, (dict, list)) else payload_template

        resolved_path_from_double_braces = self._resolve_placeholders(path_template, state)
        body_after_double_brace_resolution: Any
        if isinstance(current_payload_template_copy, dict):
            body_after_double_brace_resolution = {
                key: (self._resolve_placeholders(value, state) if isinstance(value, str) else value)
                for key, value in current_payload_template_copy.items()
            }
        elif current_payload_template_copy is not None:
            body_after_double_brace_resolution = self._resolve_placeholders(str(current_payload_template_copy), state)
        else:
            body_after_double_brace_resolution = {}

        resolved_path_params_from_mapping: Dict[str, Any] = {}
        resolved_query_params_from_mapping: Dict[str, Any] = {}
        resolved_headers_from_mapping: Dict[str, Any] = {}
        body_for_field_mappings = body_after_double_brace_resolution if isinstance(body_after_double_brace_resolution, dict) else {}

        self._apply_input_mappings(
            node_definition, state,
            resolved_path_params_from_mapping,
            resolved_query_params_from_mapping,
            body_for_field_mappings,
            resolved_headers_from_mapping
        )
        final_body_payload = body_for_field_mappings
        if node_definition.input_mappings:
            for mapping in node_definition.input_mappings:
                if mapping.target_parameter_in == "body":
                    current_data_sources = {**(state.initial_input or {}), **(state.extracted_ids or {})}
                    source_value_for_full_body = _get_value_from_path(current_data_sources, mapping.source_data_path)
                    if source_value_for_full_body is not None:
                        final_body_payload = source_value_for_full_body
                        logger.info(f"Node '{node_definition.effective_id}': Entire request body replaced by input mapping from '{mapping.source_data_path}'.")
                        break
        final_api_path = resolved_path_from_double_braces
        if resolved_path_params_from_mapping:
            for param_name, param_value in resolved_path_params_from_mapping.items():
                openapi_placeholder = f"{{{param_name}}}"
                if openapi_placeholder in final_api_path:
                    final_api_path = final_api_path.replace(openapi_placeholder, str(param_value))
        return final_api_path, resolved_query_params_from_mapping, final_body_payload, resolved_headers_from_mapping

    def _apply_confirmed_data_to_request(
        self, node_definition: Node, state: ExecutionGraphState,
        current_body_payload: Any, current_query_params: Dict[str, Any], current_headers: Dict[str, Any]
    ) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        operation_id = node_definition.effective_id
        confirmation_key = f"confirmed_{operation_id}"
        updated_body = json.loads(json.dumps(current_body_payload)) if isinstance(current_body_payload, (dict, list)) else current_body_payload
        updated_params = current_query_params.copy()
        updated_headers = current_headers.copy()
        current_confirmed_data = state.confirmed_data or {}
        if current_confirmed_data.get(confirmation_key) is True:
            confirmed_details = current_confirmed_data.get(f"{confirmation_key}_details", {})
            if "modified_payload" in confirmed_details:
                updated_body = confirmed_details["modified_payload"]
                logger.info(f"Node '{operation_id}': Applied modified payload from user confirmation.")
        return updated_body, updated_params, updated_headers

    async def _execute_api_and_process_outputs(
        self, node_definition: Node, api_path: str,
        query_params: Dict[str, Any], body_payload: Any, headers: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        effective_id = node_definition.effective_id
        api_call_method = node_definition.method or "GET"
        payload_log_preview = str(body_payload)[:200] + "..." if body_payload and len(str(body_payload)) > 200 else body_payload
        logger.debug(f"Node '{effective_id}': Executing API. Method: {api_call_method}, Path: {api_path}, Query: {query_params}, Headers: {headers}, Body Preview: {payload_log_preview}")
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
                        logger.warning(f"Node '{effective_id}': OutputMapping - Could not extract '{mapping.target_data_key}' from path '{mapping.source_data_path}'.")
            elif response_body is not None:
                 logger.warning(f"Node '{effective_id}': Response body type {type(response_body)} not dict/list. Cannot apply JSONPath output mappings.")
                 if len(node_definition.output_mappings) == 1:
                     om = node_definition.output_mappings[0]
                     if not any(c in om.source_data_path for c in ['.', '[', ']']) or om.source_data_path == "$":
                         extracted_data_for_state[om.target_data_key] = response_body
                         logger.info(f"Node '{effective_id}': Applied non-dict/list response to '{om.target_data_key}'.")
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
                # MODIFIED: Check disable_confirmation_prompts flag
                if not self.disable_confirmation_prompts and node_definition.requires_confirmation:
                    confirmation_key = f"confirmed_{effective_id}"
                    if not (state.confirmed_data or {}).get(confirmation_key, False):
                        skip_message = f"Node '{effective_id}' requires confirmation, but it was not found/negative. Skipping."
                        logger.warning(skip_message)
                        output_state_update["error"] = skip_message
                        output_state_update["api_results"] = {effective_id: {"status_code": "SKIPPED_NO_CONFIRMATION", "error": skip_message, "path_template": node_definition.path, "method": node_definition.method}}
                        return output_state_update
                elif self.disable_confirmation_prompts and node_definition.requires_confirmation:
                    logger.info(f"Node '{effective_id}' requires confirmation, but prompts are disabled for this run. Proceeding automatically.")


                final_body, final_params, final_headers = self._apply_confirmed_data_to_request(
                    node_definition, state, body_payload, query_params, headers
                )
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
                if "error" in output_state_update and not api_call_result.get("error"):
                    output_state_update.pop("error", None)
                elif api_call_result.get("error"):
                    output_state_update["error"] = api_call_result.get("error")
                logger.info(f"--- [Graph 2 Node] End: {effective_id} (Status: {api_call_result.get('status_code', 'N/A')}) ---")
                return output_state_update
            except Exception as e:
                error_message = f"Critical error in node {effective_id} execution: {type(e).__name__} - {e}"
                logger.error(error_message, exc_info=True)
                output_state_update["error"] = error_message
                current_api_results_on_error = (state.api_results or {}).copy()
                current_api_results_on_error[effective_id] = {"error": error_message, "status_code": "NODE_INTERNAL_EXCEPTION", "path_template": node_definition.path, "method": node_definition.method}
                output_state_update["api_results"] = current_api_results_on_error
                return output_state_update
        return node_executor

    def _build_and_compile_graph(self) -> Any:
        logger.info(f"ExecutionGraphDefinition: Building Graph 2. Plan: {self.graph_plan.graph_id or 'N/A'}. Confirmations disabled: {self.disable_confirmation_prompts}")
        builder = StateGraph(ExecutionGraphState)
        if not self.graph_plan.nodes:
            raise ValueError("Execution plan (GraphOutput) must contain nodes to build Graph 2.")

        actual_api_nodes_defs: List[Node] = []
        nodes_requiring_interrupt_before: list[str] = []

        for node_def in self.graph_plan.nodes:
            node_effective_id_str = str(node_def.effective_id).strip() if node_def.effective_id else ""
            if not node_effective_id_str or node_effective_id_str.upper() in ["START_NODE", "END_NODE"]:
                continue
            if not node_def.method or not node_def.path: # Should be validated by GraphOutput model
                raise ValueError(f"API Node '{node_effective_id_str}' in plan must have 'method' and 'path'.")
            builder.add_node(node_effective_id_str, self._make_node_runnable(node_def))
            actual_api_nodes_defs.append(node_def)

            # MODIFIED: Only add to interrupt list if confirmations are NOT globally disabled
            if not self.disable_confirmation_prompts and node_def.requires_confirmation:
                nodes_requiring_interrupt_before.append(node_effective_id_str)
                logger.info(f"Node '{node_effective_id_str}' marked for 'interrupt_before' (confirmations enabled for this graph instance).")
            elif self.disable_confirmation_prompts and node_def.requires_confirmation:
                logger.info(f"Node '{node_effective_id_str}' requires confirmation, but prompts are globally disabled for this graph instance. Will not interrupt.")

        if not actual_api_nodes_defs:
            logger.warning("No executable API nodes in plan. Building minimal START -> END graph for Graph 2.")
            builder.add_node("__compiler_dummy_start__", lambda x: x)
            builder.add_node("__compiler_dummy_end__", lambda x: x)
            builder.set_entry_point("__compiler_dummy_start__")
            builder.add_edge("__compiler_dummy_start__", "__compiler_dummy_end__")
            builder.set_finish_point("__compiler_dummy_end__")
            return builder.compile(checkpointer=MemorySaver())

        executable_node_ids_in_builder = {str(n.effective_id).strip() for n in actual_api_nodes_defs}
        has_start_edge_from_plan = False
        if not self.graph_plan.edges: logger.warning("Execution plan has no edges defined.")

        for edge_idx, edge in enumerate(self.graph_plan.edges):
            plan_from_node = str(edge.from_node).strip()
            plan_to_node = str(edge.to_node).strip()
            if not plan_from_node: raise ValueError(f"Edge {edge_idx+1} has empty 'from_node'.")
            if not plan_to_node: raise ValueError(f"Edge {edge_idx+1} has empty 'to_node'.")

            is_source_start = plan_from_node.upper() == "START_NODE"
            is_target_end = plan_to_node.upper() == "END_NODE"
            source_for_builder = START if is_source_start else plan_from_node
            target_for_builder = END if is_target_end else plan_to_node

            if source_for_builder != START and plan_from_node not in executable_node_ids_in_builder:
                raise ValueError(f"Edge source '{plan_from_node}' not in Graph 2 builder nodes: {executable_node_ids_in_builder}")
            if target_for_builder != END and plan_to_node not in executable_node_ids_in_builder:
                raise ValueError(f"Edge target '{plan_to_node}' not in Graph 2 builder nodes: {executable_node_ids_in_builder}")
            try:
                builder.add_edge(source_for_builder, target_for_builder)
            except Exception as e_add_edge:
                raise ValueError(f"LangGraph (Graph 2) failed to add edge ('{source_for_builder}' -> '{target_for_builder}'). Error: {e_add_edge}")
            if source_for_builder == START: has_start_edge_from_plan = True

        graph_entry_node_id_if_not_start: Optional[str] = None
        if has_start_edge_from_plan:
            logger.info("Graph 2 entry point is LANGGRAPH.START (inferred from plan).")
        elif actual_api_nodes_defs:
            entry_candidate = str(actual_api_nodes_defs[0].effective_id).strip()
            if entry_candidate not in executable_node_ids_in_builder:
                 raise ValueError(f"Default entry candidate '{entry_candidate}' not in executable nodes.")
            builder.set_entry_point(entry_candidate)
            graph_entry_node_id_if_not_start = entry_candidate
            logger.info(f"No START_NODE edge in plan. Graph 2 entry point set to first API node: '{entry_candidate}'.")
        else:
            raise RuntimeError("Failed to determine valid entry point for Graph 2.")

        if actual_api_nodes_defs:
            for node_d in actual_api_nodes_defs:
                node_id_s = str(node_d.effective_id).strip()
                is_source_to_another_api_or_plan_end = any(
                    str(e.from_node).strip() == node_id_s and
                    ((str(e.to_node).strip().upper() == "END_NODE") or (str(e.to_node).strip() in executable_node_ids_in_builder))
                    for e in self.graph_plan.edges
                )
                if not is_source_to_another_api_or_plan_end:
                    if node_id_s == graph_entry_node_id_if_not_start and not any(str(e.from_node).strip() == node_id_s and str(e.to_node).strip().upper() == "END_NODE" for e in self.graph_plan.edges):
                        try: builder.add_edge(node_id_s, END); logger.info(f"Graph 2 entry leaf '{node_id_s}' (no explicit END_NODE edge in plan) connected to LANGGRAPH.END.")
                        except Exception as e_add_final: logger.warning(f"Could not add edge from G2 entry leaf '{node_id_s}' to END: {e_add_final}.")
                    elif node_id_s != graph_entry_node_id_if_not_start:
                         if not any(str(e.from_node).strip() == node_id_s and str(e.to_node).strip().upper() == "END_NODE" for e in self.graph_plan.edges):
                             try: builder.set_finish_point(node_id_s); logger.info(f"Node '{node_id_s}' (plan leaf, no explicit END_NODE edge) set as LangGraph finish point.")
                             except Exception as e_set_finish: logger.error(f"Failed to set '{node_id_s}' as finish point: {e_set_finish}", exc_info=True)

        interrupt_list_to_use = nodes_requiring_interrupt_before if nodes_requiring_interrupt_before else None
        logger.info(f"Compiling execution graph (Graph 2). Nodes for interrupt_before: {interrupt_list_to_use or 'None'}")
        memory_saver_for_graph2 = MemorySaver()
        try:
            compiled_graph = builder.compile(
                checkpointer=memory_saver_for_graph2,
                interrupt_before=interrupt_list_to_use,
                debug=True
            )
            logger.info("Execution graph (Graph 2) compiled successfully.")
            return compiled_graph
        except Exception as e_compile:
            logger.critical(f"Execution graph (Graph 2) compilation failed: {e_compile}", exc_info=True)
            logger.error(f"Graph 2 Builder details: Nodes: {list(builder.nodes.keys())}, Edges: {builder.edges}, Entry: {builder.entry_point}, Finish: {list(builder.finish_points)}")
            raise

    def get_runnable_graph(self) -> Any:
        return self.runnable_graph

    def get_node_definition(self, node_effective_id: Union[str, Tuple[str, ...], List[str]]) -> Optional[Node]:
        target_id_str: Optional[str] = None
        if isinstance(node_effective_id, str): target_id_str = node_effective_id.strip()
        elif isinstance(node_effective_id, (list, tuple)) and node_effective_id:
            potential_id = node_effective_id[0]
            if isinstance(potential_id, str): target_id_str = potential_id.strip()
            else: logger.error(f"get_node_definition: list/tuple item not str: {node_effective_id}"); return None
        else: logger.error(f"get_node_definition: invalid type for node_id: {type(node_effective_id)}"); return None
        if target_id_str is None: return None
        for node in self.graph_plan.nodes:
            current_node_eff_id = str(node.effective_id).strip() if node.effective_id else ""
            if current_node_eff_id == target_id_str: return node
        logger.warning(f"Node definition not found in Graph 1 plan for effective_id: '{target_id_str}'")
        return None
