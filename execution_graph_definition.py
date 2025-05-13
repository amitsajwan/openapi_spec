import asyncio
import logging
# Make sure LangGraphInterrupt is imported if you intend to use it, though we are moving away from returning it from nodes
from langgraph.types import Interrupt as LangGraphInterrupt
from typing import Any, Callable, Awaitable, Dict, Optional, Tuple, Union

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


from models import GraphOutput, Node, Edge, InputMapping, OutputMapping, ExecutionGraphState
from api_executor import APIExecutor

logger = logging.getLogger(__name__)

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
    def __init__(self, graph_execution_plan: GraphOutput, api_executor: APIExecutor):
        self.graph_plan = graph_execution_plan
        self.api_executor = api_executor
        self.runner: Any = self._build_and_compile_graph()
        logger.info("ExecutionGraphDefinition initialized and Graph 2 compiled.")

    def _resolve_placeholders(self, template_string: str, state: ExecutionGraphState) -> str:
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
        if not node_definition.input_mappings:
            return
        current_extracted_ids = state.extracted_ids
        for mapping in node_definition.input_mappings:
            source_value = _get_value_from_path(current_extracted_ids, mapping.source_data_path)
            if source_value is None:
                logger.warning(f"Node '{node_definition.effective_id}': InputMapping - Could not find source data for path '{mapping.source_data_path}'. Skipping.")
                continue
            if mapping.target_parameter_in == "path":
                resolved_path_params[mapping.target_parameter_name] = str(source_value)
            elif mapping.target_parameter_in == "query":
                resolved_query_params[mapping.target_parameter_name] = source_value
            elif mapping.target_parameter_in == "header":
                resolved_headers[mapping.target_parameter_name] = str(source_value)
            elif mapping.target_parameter_in.startswith("body."):
                field_path_in_body = mapping.target_parameter_in.split("body.", 1)[1]
                _set_value_by_path(resolved_body_payload_for_field_mapping, field_path_in_body, source_value)

    def _prepare_api_request_components(
        self, node_definition: Node, state: ExecutionGraphState
    ) -> Tuple[str, Dict[str, Any], Any, Dict[str, Any]]:
        path_template = node_definition.path or ""
        payload_template = node_definition.payload
        if isinstance(payload_template, dict):
            payload_template = payload_template.copy()

        resolved_path_from_template = self._resolve_placeholders(path_template, state)
        body_after_placeholder_resolution: Any
        if isinstance(payload_template, dict):
            body_after_placeholder_resolution = {
                key: (self._resolve_placeholders(value, state) if isinstance(value, str) else value)
                for key, value in payload_template.items()
            }
        elif payload_template is not None:
            body_after_placeholder_resolution = self._resolve_placeholders(str(payload_template), state)
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
        if not isinstance(body_after_placeholder_resolution, dict) and body_for_field_mappings:
            final_body_payload = body_for_field_mappings
        elif isinstance(body_after_placeholder_resolution, dict): # body_for_field_mappings is the updated dict
             final_body_payload = body_for_field_mappings
        else: # Original was not dict, and no "body.*" mappings created a dict
            final_body_payload = body_after_placeholder_resolution


        for mapping in node_definition.input_mappings or []:
            if mapping.target_parameter_in == "body":
                current_extracted_ids = state.extracted_ids
                source_value = _get_value_from_path(current_extracted_ids, mapping.source_data_path)
                if source_value is not None:
                    final_body_payload = source_value
                    break
        
        final_api_path = resolved_path_from_template
        if final_path_params:
            temp_extracted_ids_for_path = state.extracted_ids.copy()
            temp_extracted_ids_for_path.update(final_path_params)
            initial_input_dict = state.initial_input if isinstance(state.initial_input, dict) else {}
            for k,v in initial_input_dict.items():
                if k not in temp_extracted_ids_for_path:
                    temp_extracted_ids_for_path[k] = v
            temp_state_for_path_re_resolution = ExecutionGraphState(
                extracted_ids=temp_extracted_ids_for_path,
                api_results={}, confirmed_data={}, initial_input={} # Ensure other required fields are present
            )
            final_api_path = self._resolve_placeholders(path_template, temp_state_for_path_re_resolution)
        return final_api_path, final_query_params, final_body_payload, final_headers

    def _apply_confirmed_data_to_request(
        self, node_definition: Node, state: ExecutionGraphState,
        current_body_payload: Any, current_query_params: Dict[str, Any], current_headers: Dict[str, Any]
    ) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        operationId = node_definition.effective_id
        confirmation_key = f"confirmed_{operationId}"
        updated_body = current_body_payload.copy() if isinstance(current_body_payload, dict) else current_body_payload
        updated_params = current_query_params.copy()
        updated_headers = current_headers.copy()
        current_confirmed_data = state.confirmed_data
        if current_confirmed_data.get(confirmation_key):
            confirmed_details = current_confirmed_data.get(f"{confirmation_key}_details", {})
            if "modified_payload" in confirmed_details:
                updated_body = confirmed_details["modified_payload"]
            if "modified_query_params" in confirmed_details and isinstance(confirmed_details["modified_query_params"], dict):
                updated_params = confirmed_details["modified_query_params"]
            if "modified_headers" in confirmed_details and isinstance(confirmed_details["modified_headers"], dict):
                updated_headers = confirmed_details["modified_headers"]
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
            if isinstance(response_body, dict):
                for mapping in node_definition.output_mappings:
                    extracted_value = _get_value_from_path(response_body, mapping.source_data_path)
                    if extracted_value is not None:
                        extracted_data_for_state[mapping.target_data_key] = extracted_value
            elif response_body is not None:
                 logger.warning(f"Node '{effective_id}': Response body type {type(response_body)} not dict. Cannot apply output mappings.")
        return api_call_result_dict, (extracted_data_for_state if extracted_data_for_state else None)

    def _make_node_runnable(
        self, node_definition: Node
    ) -> Callable[[ExecutionGraphState], Awaitable[Dict[str, Any]]]: # Node now always returns a state update dict
        """
        Creates an awaitable function for a given API node definition.
        If confirmation is needed, it sets 'pending_confirmation_data' in the state and returns.
        Otherwise, it executes the API call and returns its results.
        """
        async def node_executor(state: ExecutionGraphState) -> Dict[str, Any]: # Always returns a dict
            effective_id = node_definition.effective_id
            logger.info(f"--- [Graph 2] Node Start: {effective_id} (OpID: {node_definition.operationId}) ---")
            
            # Default state update, ensures pending_confirmation_data is cleared if not set later
            output_state_update: Dict[str, Any] = {"pending_confirmation_data": None}

            try:
                # 1. Prepare request components (URL, params, body, headers)
                # This step is done regardless of confirmation, as details are needed for the prompt.
                api_path, query_params, body_payload, headers = self._prepare_api_request_components(node_definition, state)
                
                # 2. Check if confirmation is required and not yet given
                operationId = node_definition.effective_id # Same as effective_id
                confirmation_key = f"confirmed_{operationId}"
                # state.confirmed_data is guaranteed to be a dict by Pydantic model (default_factory=dict)
                is_already_confirmed = state.confirmed_data.get(confirmation_key)

                if node_definition.requires_confirmation and not is_already_confirmed:
                    logger.info(f"Node '{operationId}' requires confirmation. Path: {api_path}. Setting state for manager.")
                    
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
                        "confirmation_key": confirmation_key # Key for UI to send back
                    }
                    
                    output_state_update["pending_confirmation_data"] = interrupt_data_for_ui
                    logger.info(f"--- [Graph 2] Node '{effective_id}' anwaiting confirmation. Returning state update. ---")
                    return output_state_update # Return state update; Manager will detect pause

                # 3. If here, either no confirmation needed, or it was already confirmed.
                # Apply any user-confirmed modifications from state.confirmed_data
                final_body, final_params, final_headers = self._apply_confirmed_data_to_request(
                    node_definition, state, body_payload, query_params, headers
                )
                
                # 4. Execute the API call and process outputs
                logger.debug(f"Node '{effective_id}': Proceeding with API call execution.")
                api_call_result, extracted_data = await self._execute_api_and_process_outputs(
                    node_definition, api_path, final_params, final_body, final_headers
                )
                
                output_state_update["api_results"] = {effective_id: api_call_result}
                if extracted_data: # Only add if there's data to avoid None values if operator.ior expects dicts
                    output_state_update["extracted_ids"] = extracted_data
                
                # Ensure pending_confirmation_data is cleared if we proceeded successfully past confirmation stage
                output_state_update["pending_confirmation_data"] = None 
                # If it was confirmed, the confirmed_data for this op_id remains.
                # It will be ignored in subsequent runs of this node unless cleared by manager after successful use.
                # Or, _apply_confirmed_data_to_request could clear it after applying.
                # For now, let's assume manager handles clearing confirmed_data if needed after full node success.

                logger.info(f"--- [Graph 2] Node End: {effective_id} (Execution successful) ---")
                return output_state_update

            except Exception as e:
                error_message = f"Error in node {effective_id}: {type(e).__name__} - {e}"
                logger.error(error_message, exc_info=True)
                output_state_update["error"] = error_message
                output_state_update["api_results"] = {effective_id: {"error": error_message, "status_code": "NODE_EXCEPTION"}}
                output_state_update["pending_confirmation_data"] = None # Clear on error too
                return output_state_update
        return node_executor

    def _build_and_compile_graph(self) -> Any:
        logger.info(f"ExecutionGraphDefinition: Building graph. Plan description: {self.graph_plan.description or 'N/A'}")
        builder = StateGraph(ExecutionGraphState)
        if not self.graph_plan.nodes:
            raise ValueError("Execution plan (GraphOutput) must contain at least one node to build Graph 2.")

        actual_api_nodes_defs = []
        for node_def in self.graph_plan.nodes:
            node_effective_id_str = str(node_def.effective_id).strip() if node_def.effective_id else ""
            if node_effective_id_str.upper() in ["START_NODE", "END_NODE"]:
                continue
            if not node_def.method or not node_def.path:
                raise ValueError(f"API Node '{node_effective_id_str}' in plan must have 'method' and 'path' defined.")
            builder.add_node(node_effective_id_str, self._make_node_runnable(node_def))
            actual_api_nodes_defs.append(node_def)

        executable_node_ids_in_builder = {str(n.effective_id).strip() for n in actual_api_nodes_defs}
        has_start_edge = False
        if not self.graph_plan.edges:
            logger.warning("Execution plan has no edges defined.")

        for edge_idx, edge in enumerate(self.graph_plan.edges):
            plan_from_node_original_case = str(edge.from_node).strip() if edge.from_node else ""
            plan_to_node_original_case = str(edge.to_node).strip() if edge.to_node else ""
            plan_from_node_upper = plan_from_node_original_case.upper()
            plan_to_node_upper = plan_to_node_original_case.upper()
            is_plan_source_start_node = plan_from_node_upper == "START_NODE"
            is_plan_target_end_node = plan_to_node_upper == "END_NODE"

            source_for_builder = START if is_plan_source_start_node else plan_from_node_original_case
            target_for_builder = END if is_plan_target_end_node else plan_to_node_original_case

            if source_for_builder != START:
                if not plan_from_node_original_case:
                     raise ValueError(f"Edge {edge_idx + 1} has an empty 'from_node'.")
                if plan_from_node_original_case not in executable_node_ids_in_builder:
                    raise ValueError(f"Edge source '{plan_from_node_original_case}' not in builder: {executable_node_ids_in_builder}")
            if target_for_builder != END:
                if not plan_to_node_original_case:
                    raise ValueError(f"Edge {edge_idx + 1} has an empty 'to_node'.")
                if plan_to_node_original_case not in executable_node_ids_in_builder:
                    raise ValueError(f"Edge target '{plan_to_node_original_case}' not in builder: {executable_node_ids_in_builder}")
            try:
                builder.add_edge(source_for_builder, target_for_builder)
            except Exception as e_add_edge:
                source_log = 'LANGGRAPH_START' if source_for_builder == START else source_for_builder
                target_log = 'LANGGRAPH_END' if target_for_builder == END else target_for_builder
                raise ValueError(f"Failed to add edge ('{source_log}' -> '{target_log}'). Error: {e_add_edge}")

            if source_for_builder == START:
                has_start_edge = True
        
        if not has_start_edge and actual_api_nodes_defs:
            entry_point_candidate = str(actual_api_nodes_defs[0].effective_id).strip()
            if entry_point_candidate not in executable_node_ids_in_builder:
                 raise ValueError(f"Default entry point '{entry_point_candidate}' not in builder: {executable_node_ids_in_builder}")
            builder.set_entry_point(entry_point_candidate)
        elif not actual_api_nodes_defs and not has_start_edge :
             logger.warning("Graph has no executable API nodes and no explicit entry point from START.")

        if actual_api_nodes_defs:
            for node_def in actual_api_nodes_defs:
                node_id_str = str(node_def.effective_id).strip()
                is_source_to_another_node_or_end = any(
                    str(e.from_node).strip() == node_id_str and
                    (str(e.to_node).strip().upper() == "END_NODE" or str(e.to_node).strip() in executable_node_ids_in_builder)
                    for e in self.graph_plan.edges
                )
                if not is_source_to_another_node_or_end:
                    builder.set_finish_point(node_id_str)
        return builder.compile(checkpointer=MemorySaver())

    def get_runnable_graph(self) -> Any:
        return self.runner
