import asyncio
import logging
from typing import Any, Callable, Awaitable, Dict, Optional, Tuple, Union

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
# Interrupt is not raised by nodes anymore in this pattern, but imported for clarity if needed elsewhere.
# from langgraph.types import Interrupt

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
    def __init__(self, graph_execution_plan: GraphOutput, api_executor: APIExecutor):
        self.graph_plan = graph_execution_plan # Keep a reference for the manager
        self.api_executor = api_executor
        self.runnable_graph: Any = self._build_and_compile_graph()
        logger.info("ExecutionGraphDefinition initialized and Graph 2 compiled.")

    def _resolve_placeholders(self, template_string: str, state: ExecutionGraphState) -> str:
        if not isinstance(template_string, str):
            return str(template_string)
        resolved_string = template_string
        extracted_ids_dict = state.extracted_ids
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
        elif isinstance(body_after_placeholder_resolution, dict):
             final_body_payload = body_for_field_mappings
        else:
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
                api_results={}, confirmed_data={}, initial_input={}
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
        
        # Check if this specific operation was confirmed (decision is True)
        if current_confirmed_data.get(confirmation_key): # This implies decision was True
            confirmed_details = current_confirmed_data.get(f"{confirmation_key}_details", {})
            if "modified_payload" in confirmed_details: # User might have modified the payload
                updated_body = confirmed_details["modified_payload"]
                logger.info(f"Node '{operationId}': Applied modified payload from user confirmation.")
            # Add similar checks for modified_query_params, modified_headers if UI supports modifying them
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
    ) -> Callable[[ExecutionGraphState], Awaitable[Dict[str, Any]]]:
        """
        Creates an awaitable function for a given API node definition.
        The graph will interrupt *before* this node if requires_confirmation is true.
        This node, when run, checks if it was confirmed.
        """
        async def node_executor(state: ExecutionGraphState) -> Dict[str, Any]:
            effective_id = node_definition.effective_id
            logger.info(f"--- [Graph 2] Node Start: {effective_id} (OpID: {node_definition.operationId}) ---")
            
            output_state_update: Dict[str, Any] = {} # Start with an empty update

            try:
                # Prepare request components first. These might be needed by the manager
                # if it needs to show them to the user during an interruption *before* this node.
                api_path, query_params, body_payload, headers = self._prepare_api_request_components(node_definition, state)
                
                # If this node requires confirmation, it means the graph interrupted *before* it.
                # Now we check if the confirmation was provided in state.confirmed_data.
                if node_definition.requires_confirmation:
                    confirmation_key = f"confirmed_{effective_id}"
                    # state.confirmed_data is guaranteed to be a dict
                    if not state.confirmed_data.get(confirmation_key): # Decision was False or not present
                        # This means user cancelled or something went wrong with resume.
                        # The graph should ideally not have resumed to this node if confirmation was false.
                        # For safety, we can skip execution or raise an error.
                        # Let's log and skip, returning minimal state update.
                        skip_message = f"Node '{effective_id}' requires confirmation, but it was not found or was negative in confirmed_data. Skipping execution."
                        logger.warning(skip_message)
                        output_state_update["error"] = skip_message 
                        output_state_update["api_results"] = {effective_id: {"status_code": "SKIPPED_NO_CONFIRMATION", "error": skip_message}}
                        return output_state_update

                # If here, either no confirmation needed, or it was confirmed (checked above).
                # Apply any user-confirmed modifications from state.confirmed_data
                final_body, final_params, final_headers = self._apply_confirmed_data_to_request(
                    node_definition, state, body_payload, query_params, headers
                )
                
                logger.debug(f"Node '{effective_id}': Proceeding with API call execution.")
                api_call_result, extracted_data = await self._execute_api_and_process_outputs(
                    node_definition, api_path, final_params, final_body, final_headers
                )
                
                output_state_update["api_results"] = {effective_id: api_call_result}
                if extracted_data:
                    output_state_update["extracted_ids"] = extracted_data
                
                # Clear the specific confirmation for this node now that it has been processed
                # This prevents it from being "sticky" if the node were somehow re-run without a new interruption.
                if node_definition.requires_confirmation:
                    confirmation_key = f"confirmed_{effective_id}"
                    # We need a way to update confirmed_data to remove this key or its details.
                    # LangGraph's state merging (operator.ior) adds/updates keys.
                    # To "remove" a confirmation, the manager would need to update state with a version of
                    # confirmed_data that omits this key *after* this node successfully runs.
                    # For now, the node itself cannot easily "remove" from the merged state.
                    # The manager will handle clearing pending_confirmation_data.
                    pass


                logger.info(f"--- [Graph 2] Node End: {effective_id} (Execution successful) ---")
                return output_state_update

            except Exception as e:
                error_message = f"Error in node {effective_id}: {type(e).__name__} - {e}"
                logger.error(error_message, exc_info=True)
                output_state_update["error"] = error_message
                output_state_update["api_results"] = {effective_id: {"error": error_message, "status_code": "NODE_EXCEPTION"}}
                return output_state_update
        return node_executor

    def _build_and_compile_graph(self) -> Any:
        logger.info(f"ExecutionGraphDefinition: Building graph. Plan: {self.graph_plan.description or 'N/A'}")
        builder = StateGraph(ExecutionGraphState)
        if not self.graph_plan.nodes:
            raise ValueError("Execution plan must contain at least one node.")

        actual_api_nodes_defs = []
        nodes_requiring_interrupt_before: List[str] = []

        for node_def in self.graph_plan.nodes:
            node_id_str = str(node_def.effective_id).strip()
            if node_id_str.upper() in ["START_NODE", "END_NODE"]:
                continue
            if not node_def.method or not node_def.path:
                raise ValueError(f"API Node '{node_id_str}' must have 'method' and 'path'.")
            
            builder.add_node(node_id_str, self._make_node_runnable(node_def))
            actual_api_nodes_defs.append(node_def)
            if node_def.requires_confirmation:
                nodes_requiring_interrupt_before.append(node_id_str)
                logger.info(f"Node '{node_id_str}' marked for 'interrupt_before'.")


        executable_node_ids_in_builder = {str(n.effective_id).strip() for n in actual_api_nodes_defs}
        has_start_edge = False
        if not self.graph_plan.edges: logger.warning("Execution plan has no edges.")

        for edge in self.graph_plan.edges:
            # (Edge processing logic remains the same as in execution_graph_definition_fix)
            plan_from_node_original_case = str(edge.from_node).strip()
            plan_to_node_original_case = str(edge.to_node).strip()
            is_plan_source_start_node = plan_from_node_original_case.upper() == "START_NODE"
            is_plan_target_end_node = plan_to_node_original_case.upper() == "END_NODE"
            source_for_builder = START if is_plan_source_start_node else plan_from_node_original_case
            target_for_builder = END if is_plan_target_end_node else plan_to_node_original_case

            if source_for_builder != START and plan_from_node_original_case not in executable_node_ids_in_builder:
                raise ValueError(f"Edge source '{plan_from_node_original_case}' not in builder: {executable_node_ids_in_builder}")
            if target_for_builder != END and plan_to_node_original_case not in executable_node_ids_in_builder:
                raise ValueError(f"Edge target '{plan_to_node_original_case}' not in builder: {executable_node_ids_in_builder}")
            builder.add_edge(source_for_builder, target_for_builder)
            if source_for_builder == START: has_start_edge = True
        
        if not has_start_edge and actual_api_nodes_defs:
            entry_point = str(actual_api_nodes_defs[0].effective_id).strip()
            builder.set_entry_point(entry_point)
            logger.info(f"No explicit START edge. Entry point set to: '{entry_point}'.")
        elif not actual_api_nodes_defs and not has_start_edge:
             logger.warning("Graph has no executable API nodes and no explicit entry point.")

        if actual_api_nodes_defs: # Set finish points
            all_source_ids_in_edges = {str(e.from_node).strip() for e in self.graph_plan.edges if str(e.from_node).strip().upper() != "START_NODE"}
            for node_def in actual_api_nodes_defs:
                node_id = str(node_def.effective_id).strip()
                is_source_to_another = any(
                    str(e.from_node).strip() == node_id and
                    (str(e.to_node).strip().upper() == "END_NODE" or str(e.to_node).strip() in executable_node_ids_in_builder)
                    for e in self.graph_plan.edges
                )
                if not is_source_to_another:
                    logger.info(f"Node '{node_id}' is a leaf. Setting as finish point.")
                    builder.set_finish_point(node_id)
        
        logger.info(f"Compiling execution graph. Nodes for interrupt_before: {nodes_requiring_interrupt_before}")
        return builder.compile(
            checkpointer=MemorySaver(),
            interrupt_before=nodes_requiring_interrupt_before if nodes_requiring_interrupt_before else None
        )

    def get_runnable_graph(self) -> Any:
        return self.runnable_graph

    def get_node_definition(self, node_effective_id: str) -> Optional[Node]:
        """Helper to get node definition from the plan, used by manager."""
        for node in self.graph_plan.nodes:
            if str(node.effective_id).strip() == node_effective_id:
                return node
        return None

