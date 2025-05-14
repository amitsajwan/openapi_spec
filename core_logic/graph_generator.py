# core_logic/graph_generator.py
import json
import logging
import os
from typing import Any, Dict, Optional, List # Added List

from pydantic import ValidationError as PydanticValidationError

from models import BotState, GraphOutput, Node, Edge # Added Node, Edge
from utils import (
    llm_call_helper,
    parse_llm_json_output_with_model,
    check_for_cycles,
)

logger = logging.getLogger(__name__)

# --- Configurable Limits from environment variables ---
MAX_APIS_IN_PROMPT_SUMMARY_LONG = int(
    os.getenv("MAX_APIS_IN_PROMPT_SUMMARY_LONG", "20")
)
MAX_TOTAL_APIS_THRESHOLD_FOR_TRUNCATION_LONG = int(
    os.getenv("MAX_TOTAL_APIS_THRESHOLD_FOR_TRUNCATION_LONG", "25")
)
MAX_APIS_IN_PROMPT_SUMMARY_SHORT = int(
    os.getenv("MAX_APIS_IN_PROMPT_SUMMARY_SHORT", "10")
)
MAX_TOTAL_APIS_THRESHOLD_FOR_TRUNCATION_SHORT = int(
    os.getenv("MAX_TOTAL_APIS_THRESHOLD_FOR_TRUNCATION_SHORT", "15")
)


class GraphGenerator:
    """
    Handles the generation, verification, and refinement of API execution graphs.
    """

    def __init__(self, worker_llm: Any):
        if not hasattr(worker_llm, "invoke"):
            raise TypeError("worker_llm must have an 'invoke' method.")
        self.worker_llm = worker_llm
        logger.info("GraphGenerator initialized.")

    def _queue_intermediate_message(self, state: BotState, msg: str):
        if "intermediate_messages" not in state.scratchpad:
            state.scratchpad["intermediate_messages"] = []
        if (
            not state.scratchpad["intermediate_messages"]
            or state.scratchpad["intermediate_messages"][-1] != msg
        ):
            state.scratchpad["intermediate_messages"].append(msg)
        state.response = msg

    def _ensure_start_end_nodes_and_basic_connectivity(self, graph: GraphOutput) -> GraphOutput:
        """
        Programmatically ensures START_NODE and END_NODE exist and are minimally connected
        if other API nodes are present. This is a fallback for LLM inconsistencies.
        """
        if not isinstance(graph, GraphOutput):
            logger.warning("_ensure_start_end_nodes: Input is not a GraphOutput object. Skipping.")
            return graph # Should not happen if parse_llm_json_output_with_model works

        node_ids = {node.effective_id for node in graph.nodes}
        has_start_node = "START_NODE" in node_ids
        has_end_node = "END_NODE" in node_ids

        # Ensure START_NODE exists
        if not has_start_node:
            logger.warning("START_NODE missing from LLM output. Adding it programmatically.")
            graph.nodes.append(Node(operationId="START_NODE", method="SYSTEM", summary="Workflow Start"))
            has_start_node = True # Assume addition is successful for subsequent logic

        # Ensure END_NODE exists
        if not has_end_node:
            logger.warning("END_NODE missing from LLM output. Adding it programmatically.")
            graph.nodes.append(Node(operationId="END_NODE", method="SYSTEM", summary="Workflow End"))
            has_end_node = True # Assume addition is successful

        api_nodes = [node for node in graph.nodes if node.effective_id not in ["START_NODE", "END_NODE"]]

        if api_nodes: # Only add default connections if there are actual API nodes
            # Ensure START_NODE has an outgoing edge if not already present
            start_has_outgoing = any(edge.from_node == "START_NODE" for edge in graph.edges)
            if has_start_node and not start_has_outgoing:
                first_api_node_id = api_nodes[0].effective_id
                logger.warning(f"START_NODE has no outgoing edges. Connecting to first API node: {first_api_node_id}")
                graph.edges.append(Edge(from_node="START_NODE", to_node=first_api_node_id, description="Default start connection"))

            # Ensure END_NODE has an incoming edge if not already present
            end_has_incoming = any(edge.to_node == "END_NODE" for edge in graph.edges)
            if has_end_node and not end_has_incoming:
                last_api_node_id = api_nodes[-1].effective_id
                logger.warning(f"END_NODE has no incoming edges. Connecting from last API node: {last_api_node_id}")
                graph.edges.append(Edge(from_node=last_api_node_id, to_node="END_NODE", description="Default end connection"))
        elif not api_nodes and has_start_node and has_end_node: # Only START and END nodes
            # Ensure a direct edge from START to END if no other nodes exist
            start_to_end_exists = any(edge.from_node == "START_NODE" and edge.to_node == "END_NODE" for edge in graph.edges)
            if not start_to_end_exists:
                logger.warning("Graph has only START/END nodes. Ensuring direct connection.")
                graph.edges.append(Edge(from_node="START_NODE", to_node="END_NODE", description="Direct flow for empty plan"))
        return graph


    def _generate_execution_graph(
        self, state: BotState, goal: Optional[str] = None
    ) -> BotState:
        tool_name = "_generate_execution_graph"
        current_goal = goal or state.plan_generation_goal or "General API workflow overview"
        self._queue_intermediate_message(
            state, f"Building API workflow graph for goal: '{current_goal[:70]}...'"
        )
        state.update_scratchpad_reason(tool_name, f"Generating graph. Goal: {current_goal}")

        if not state.identified_apis:
            self._queue_intermediate_message(state, "Cannot generate graph: No API operations identified.")
            state.execution_graph = None 
            state.update_scratchpad_reason(tool_name, state.response)
            state.next_step = "responder" 
            return state

        # ... (api_summaries_for_prompt and feedback_str construction as before) ...
        api_summaries_for_prompt = []
        num_apis_to_summarize = MAX_APIS_IN_PROMPT_SUMMARY_LONG
        truncate_threshold = MAX_TOTAL_APIS_THRESHOLD_FOR_TRUNCATION_LONG
        for idx, api in enumerate(state.identified_apis):
            if (idx >= num_apis_to_summarize and len(state.identified_apis) > truncate_threshold):
                api_summaries_for_prompt.append(f"- ...and {len(state.identified_apis) - num_apis_to_summarize} more operations.")
                break
            likely_confirmation = api["method"].upper() in ["POST", "PUT", "DELETE", "PATCH"]
            params_str_parts = []
            if api.get("parameters"):
                for p_idx, p_detail in enumerate(api["parameters"]):
                    if p_idx >= 3: params_str_parts.append("..."); break 
                    param_name = p_detail.get("name", "N/A"); param_in = p_detail.get("in", "N/A")
                    param_schema = p_detail.get("schema", {}); param_type = (param_schema.get("type", "unknown") if isinstance(param_schema, dict) else "unknown")
                    params_str_parts.append(f"{param_name}({param_in}, {param_type})")
            params_str = (f"Params: {', '.join(params_str_parts)}" if params_str_parts else "No explicit params listed.")
            req_body_info = ""
            if api.get("requestBody") and isinstance(api["requestBody"], dict):
                content = api["requestBody"].get("content", {}); json_schema = content.get("application/json", {}).get("schema", {})
                if json_schema and isinstance(json_schema, dict) and json_schema.get("properties"):
                    props = list(json_schema.get("properties", {}).keys())[:3] 
                    req_body_info = (f" ReqBody fields (sample from schema): {', '.join(props)}{'...' if len(json_schema.get('properties', {})) > 3 else ''}.")
            api_summaries_for_prompt.append(f"- operationId: {api['operationId']} ({api['method']} {api['path']}), summary: {api.get('summary', 'N/A')[:80]}. {params_str}{req_body_info} likely_requires_confirmation: {'yes' if likely_confirmation else 'no'}")
        apis_str = "\n".join(api_summaries_for_prompt)
        feedback_str = (f"Refinement Feedback: {state.graph_regeneration_reason}" if state.graph_regeneration_reason else "")
        
        prompt = f"""
        Goal: "{current_goal}". {feedback_str}
        Available API Operations (summary with parameters and sample request body fields from validated schemas):\n{apis_str}

        Design a logical and runnable API execution graph as a JSON object. The graph must achieve the specified Goal.
        Consider typical API workflow patterns.

        The graph must adhere to the Pydantic models:
        InputMapping: {{"source_operation_id": "str_effective_id_of_source_node", "source_data_path": "str_jsonpath_to_value_in_extracted_ids (e.g., '$.createdItemIdFromStep1')", "target_parameter_name": "str_param_name_in_target_node (e.g., 'itemId')", "target_parameter_in": "Literal['path', 'query', 'body', 'body.fieldName']"}}
        OutputMapping: {{"source_data_path": "str_jsonpath_to_value_in_THIS_NODE_RESPONSE (e.g., '$.id', '$.data.token')", "target_data_key": "str_UNIQUE_key_for_shared_data_pool (e.g., 'createdItemId', 'userAuthToken')"}}
        Node: {{ ... "payload": {{ "template_key": "realistic_example_value or {{{{placeholder_from_output_mapping}}}}" }} ... }} 
        
        CRITICAL INSTRUCTIONS FOR `payload` FIELD in Nodes (for POST, PUT, PATCH), using the schema information provided:
        1.  Accuracy is Key: The `payload` dictionary MUST ONLY contain fields that are actually defined by the specific API's request body schema.
        2.  Do Not Invent Fields: Do NOT include any fields in the `payload` that are not part of the API's expected request body.
        3.  Realistic Values: Use realistic example values.
        4.  Placeholders for Dynamic Data: Use `{{{{key_from_output_mapping}}}}` for values from previous steps.
        5.  Optional Fields: OMIT optional fields if no value is known or relevant.

        CRITICAL INSTRUCTIONS FOR DATA FLOW:
        1.  Create Node (e.g., POST /products): MUST have an `OutputMapping` to extract the ID (e.g., `{{"source_data_path": "$.id", "target_data_key": "newProductId"}}`).
        2.  Get/Update/Delete Node (e.g., GET /products/{{{{newProductId}}}}): Path MUST use the placeholder from the Create Node's `target_data_key`.

        MANDATORY General Instructions:
        - ALWAYS include a "START_NODE" (method: "SYSTEM", summary: "Workflow Start") and an "END_NODE" (method: "SYSTEM", summary: "Workflow End").
        - Select 2-5 relevant API operations between START_NODE and END_NODE.
        - Set `requires_confirmation: true` for POST, PUT, DELETE, PATCH.
        - Connect nodes with `edges`. START_NODE MUST connect to the first API operation(s). The last API operation(s) MUST connect to END_NODE. All API nodes must be part of a path from START_NODE to END_NODE.
        - Ensure logical sequence.
        - Provide overall `description` and `refinement_summary`.

        Output ONLY the JSON object for GraphOutput. Ensure valid JSON.
        """
        
        try:
            llm_response = llm_call_helper(self.worker_llm, prompt)
            graph_output_candidate = parse_llm_json_output_with_model(llm_response, expected_model=GraphOutput)

            if graph_output_candidate:
                # Programmatically ensure START/END nodes and basic connectivity
                graph_output_candidate = self._ensure_start_end_nodes_and_basic_connectivity(graph_output_candidate)
                
                state.execution_graph = graph_output_candidate
                self._queue_intermediate_message(state, "API workflow graph generated.")
                logger.info(f"Graph generated. Description: {graph_output_candidate.description or 'N/A'}")
                if graph_output_candidate.refinement_summary:
                    logger.info(f"LLM summary for graph: {graph_output_candidate.refinement_summary}")
                state.graph_regeneration_reason = None 
                state.graph_refinement_iterations = 0 
                state.next_step = "verify_graph" 
                state.update_scratchpad_reason(tool_name, f"Graph gen success. Next: {state.next_step}")
                return state 

            error_msg = "LLM failed to produce a valid GraphOutput JSON, or it was structurally incomplete."
            logger.error(error_msg + f" Raw LLM output snippet: {llm_response[:300]}...")
            self._queue_intermediate_message(state, "Failed to generate a valid execution graph (AI output format or structure issue).")
            state.execution_graph = None 
            state.graph_regeneration_reason = state.graph_regeneration_reason or "LLM output was not a valid GraphOutput object or missed key structural elements."
            
            current_attempts = state.scratchpad.get('graph_gen_attempts', 0)
            if current_attempts < 1: 
                state.scratchpad['graph_gen_attempts'] = current_attempts + 1
                logger.info("Retrying initial graph generation once due to validation/parsing failure.")
                state.next_step = "_generate_execution_graph" 
            else:
                logger.error("Max initial graph generation attempts reached. Routing to handle_unknown.")
                state.next_step = "handle_unknown" 
                state.scratchpad['graph_gen_attempts'] = 0 

        except Exception as e:
            logger.error(f"Error during graph generation LLM call or processing: {e}", exc_info=False)
            self._queue_intermediate_message(state, f"Error generating graph: {str(e)[:150]}...")
            state.execution_graph = None
            state.graph_regeneration_reason = f"LLM call/processing error: {str(e)[:100]}..."
            state.next_step = "handle_unknown" 

        state.update_scratchpad_reason(tool_name, f"Graph gen status: {'Success' if state.execution_graph else 'Failed'}. Resp: {state.response}")
        return state

    def verify_graph(self, state: BotState) -> BotState:
        tool_name = "verify_graph"
        self._queue_intermediate_message(state, "Verifying API workflow graph...")
        state.update_scratchpad_reason(tool_name, "Verifying graph structure and integrity.")

        if not state.execution_graph or not isinstance(state.execution_graph, GraphOutput):
            # ... (existing logic for no graph to verify) ...
            current_response = state.response or ""
            self._queue_intermediate_message(state, current_response + " No execution graph to verify (possibly due to generation error or wrong type).")
            state.graph_regeneration_reason = (state.graph_regeneration_reason or "No graph was generated to verify.")
            logger.warning(f"verify_graph: No graph found or invalid type. Reason: {state.graph_regeneration_reason}. Routing to _generate_execution_graph for regeneration.")
            state.next_step = "_generate_execution_graph"; return state

        issues = []
        try:
            GraphOutput.model_validate(state.execution_graph.model_dump()) 
            is_dag, cycle_msg = check_for_cycles(state.execution_graph)
            if not is_dag: issues.append(cycle_msg or "Graph contains cycles.")
            
            node_ids = {node.effective_id for node in state.execution_graph.nodes}
            if "START_NODE" not in node_ids: issues.append("CRITICAL: START_NODE is missing from the graph nodes list.")
            if "END_NODE" not in node_ids: issues.append("CRITICAL: END_NODE is missing from the graph nodes list.")

            api_nodes_exist = any(node.effective_id not in ["START_NODE", "END_NODE"] for node in state.execution_graph.nodes)

            if "START_NODE" in node_ids:
                start_outgoing = any(edge.from_node == "START_NODE" for edge in state.execution_graph.edges)
                start_incoming = any(edge.to_node == "START_NODE" for edge in state.execution_graph.edges)
                if api_nodes_exist and not start_outgoing : issues.append("START_NODE has no outgoing edges to any API operations.")
                if start_incoming: issues.append("START_NODE should not have incoming edges.")
            
            if "END_NODE" in node_ids:
                end_incoming = any(edge.to_node == "END_NODE" for edge in state.execution_graph.edges)
                end_outgoing = any(edge.from_node == "END_NODE" for edge in state.execution_graph.edges)
                if api_nodes_exist and not end_incoming : issues.append("END_NODE has no incoming edges from any API operations.")
                if end_outgoing: issues.append("END_NODE should not have outgoing edges.")
            
            if not api_nodes_exist and ("START_NODE" in node_ids and "END_NODE" in node_ids): # Only START and END
                if not any(edge.from_node == "START_NODE" and edge.to_node == "END_NODE" for edge in state.execution_graph.edges):
                    issues.append("Graph with only START_NODE and END_NODE is missing a direct edge between them.")


            for node in state.execution_graph.nodes:
                if node.effective_id.upper() not in ["START_NODE", "END_NODE"]: 
                    if not node.method or not node.path: issues.append(f"Node '{node.effective_id}' is missing 'method' or 'path'.")
        except PydanticValidationError as ve: 
            logger.error(f"Graph Pydantic validation failed during verify_graph: {ve}"); issues.append(f"Graph structure is invalid (Pydantic): {str(ve)[:200]}...") 
        except Exception as e: 
            logger.error(f"Unexpected error during graph verification: {e}", exc_info=True); issues.append(f"Unexpected error during verification: {str(e)[:100]}.")
        
        if not issues:
            # ... (existing success logic) ...
            self._queue_intermediate_message(state, "Graph verification successful (Structure, DAG, START/END nodes, basic execution fields).")
            state.update_scratchpad_reason(tool_name, "Graph verification successful."); logger.info("Graph verification successful.")
            state.graph_regeneration_reason = None; state.scratchpad['refinement_validation_failures'] = 0 
            try: state.scratchpad['graph_to_send'] = state.execution_graph.model_dump_json(indent=2); logger.info("Graph marked to be sent to UI after verification.")
            except Exception as e: logger.error(f"Error serializing graph for sending after verification: {e}")
            logger.info("Graph verified. Proceeding to describe graph."); state.next_step = "describe_graph"
            if state.input_is_spec: 
                api_title = state.openapi_schema.get('info', {}).get('title', 'the API') if state.openapi_schema else 'the API'
                self._queue_intermediate_message(state, f"Successfully processed the OpenAPI specification for '{api_title}'. Identified {len(state.identified_apis)} API operations, generated example payloads, and created an API workflow graph with {len(state.execution_graph.nodes)} steps. The graph is verified. You can now ask questions, request specific plan refinements, or try to execute the workflow.")
                state.input_is_spec = False 
        else: 
            # ... (existing failure logic, routing to refine or regenerate) ...
            error_details = " ".join(issues); self._queue_intermediate_message(state, f"Graph verification failed: {error_details}."); state.graph_regeneration_reason = f"Verification failed: {error_details}."; logger.warning(f"Graph verification failed: {error_details}.")
            if state.graph_refinement_iterations < state.max_refinement_iterations: 
                logger.info(f"Verification failed. Attempting graph refinement (iteration {state.graph_refinement_iterations + 1})."); 
                state.next_step = "refine_api_graph"
            else: 
                logger.warning("Max refinement iterations reached, but graph still has verification issues. Attempting full regeneration."); 
                state.next_step = "_generate_execution_graph"; state.graph_refinement_iterations = 0; state.scratchpad['graph_gen_attempts'] = 0 
        state.update_scratchpad_reason(tool_name, f"Verification result: {state.response[:200]}...")
        return state

    def refine_api_graph(self, state: BotState) -> BotState:
        tool_name = "refine_api_graph"; iteration = state.graph_refinement_iterations + 1
        self._queue_intermediate_message(state, f"Refining API workflow graph (Attempt {iteration}/{state.max_refinement_iterations})...")
        state.update_scratchpad_reason(tool_name, f"Refining graph. Iteration: {iteration}. Reason: {state.graph_regeneration_reason or 'General refinement request.'}")
        # ... (check for no graph or max iterations as before) ...
        if not state.execution_graph or not isinstance(state.execution_graph, GraphOutput): self._queue_intermediate_message(state, "No graph to refine or invalid graph type. Please generate a graph first."); logger.warning("refine_api_graph: No execution_graph found or invalid type."); state.next_step = "_generate_execution_graph"; return state
        if iteration > state.max_refinement_iterations: self._queue_intermediate_message(state, f"Max refinement iterations ({state.max_refinement_iterations}) reached. Using current graph."); logger.warning("Max refinement iterations reached."); state.next_step = "describe_graph"; return state
        try: current_graph_json = state.execution_graph.model_dump_json(indent=2)
        except Exception as e: logger.error(f"Error serializing current graph for refinement prompt: {e}"); self._queue_intermediate_message(state, "Error preparing current graph for refinement. Cannot proceed."); state.next_step = "handle_unknown"; return state
        
        # ... (api_summaries_for_prompt construction as before) ...
        api_summaries_for_prompt = []
        num_apis_to_summarize = MAX_APIS_IN_PROMPT_SUMMARY_SHORT; truncate_threshold = MAX_TOTAL_APIS_THRESHOLD_FOR_TRUNCATION_SHORT
        for idx, api in enumerate(state.identified_apis): 
            if idx >= num_apis_to_summarize and len(state.identified_apis) > truncate_threshold: api_summaries_for_prompt.append(f"- ...and {len(state.identified_apis) - num_apis_to_summarize} more operations."); break
            likely_confirmation = api['method'].upper() in ["POST", "PUT", "DELETE", "PATCH"]
            api_summaries_for_prompt.append(f"- opId: {api['operationId']} ({api['method']} {api['path']}), summary: {api.get('summary', 'N/A')[:70]}, confirm: {'yes' if likely_confirmation else 'no'}")
        apis_ctx = "\n".join(api_summaries_for_prompt)

        prompt = f"""
        User's Overall Goal: "{state.plan_generation_goal or 'General workflow'}"
        Feedback for Refinement: "{state.graph_regeneration_reason or 'General request to improve the graph.'}"
        Current Graph (JSON to be refined, based on validated schemas):\n```json\n{current_graph_json}\n```
        Available API Operations (sample for context, from validated schemas):\n{apis_ctx}

        Task: Refine the current graph based on the feedback. Ensure the refined graph:
        1.  Strictly adheres to Pydantic models (GraphOutput, Node, Edge, etc.).
            - For `payload` in Nodes: ONLY include fields defined by the API's request body schema.
        2.  MANDATORY: ALWAYS include "START_NODE" and "END_NODE" (method: "SYSTEM").
        3.  MANDATORY: START_NODE MUST connect to the first API operation(s). The last API operation(s) MUST connect to END_NODE. All API nodes must be part of a path from START_NODE to END_NODE.
        4.  All node `operationId`s in edges must exist in the `nodes` list.
        5.  Nodes for execution have `method` and `path`.
        6.  Mappings (`input_mappings`, `output_mappings`) are logical.
        7.  `requires_confirmation` is set appropriately.
        8.  Addresses the specific feedback and ensures logical dependencies.
        9.  Provide a concise `refinement_summary` in the JSON.

        Output ONLY the refined GraphOutput JSON object.
        """
        try:
            llm_response_str = llm_call_helper(self.worker_llm, prompt)
            refined_graph_candidate = parse_llm_json_output_with_model(llm_response_str, expected_model=GraphOutput)
            if refined_graph_candidate:
                # Programmatically ensure START/END nodes and basic connectivity
                refined_graph_candidate = self._ensure_start_end_nodes_and_basic_connectivity(refined_graph_candidate)

                logger.info(f"Refinement attempt (iter {iteration}) produced a structurally valid GraphOutput.")
                state.execution_graph = refined_graph_candidate
                refinement_summary = refined_graph_candidate.refinement_summary or "AI provided no specific summary for this refinement."
                state.update_scratchpad_reason(tool_name, f"LLM Refinement Summary (Iter {iteration}): {refinement_summary}")
                self._queue_intermediate_message(state, f"Graph refined (Iteration {iteration}). Summary: {refinement_summary}")
                state.graph_refinement_iterations = iteration
                state.graph_regeneration_reason = None
                state.scratchpad['refinement_validation_failures'] = 0
                state.next_step = "verify_graph" 
            else:
                # ... (existing error handling for invalid refinement output) ...
                error_msg = "LLM refinement failed to produce a GraphOutput JSON that is valid or self-consistent."; logger.error(error_msg + f" Raw LLM output snippet for refinement: {llm_response_str[:300]}..."); self._queue_intermediate_message(state, f"Error during graph refinement (iteration {iteration}): AI output was invalid. Will retry refinement or regenerate graph."); state.graph_regeneration_reason = state.graph_regeneration_reason or "LLM output for refinement was not a valid GraphOutput object or had structural issues."
                state.scratchpad['refinement_validation_failures'] = state.scratchpad.get('refinement_validation_failures', 0) + 1
                if iteration < state.max_refinement_iterations:
                    if state.scratchpad.get('refinement_validation_failures',0) >= 2: logger.warning(f"Multiple consecutive refinement validation failures (iter {iteration}). Escalating to full graph regeneration."); self._queue_intermediate_message(state, state.response + " Attempting full regeneration due to persistent refinement issues."); state.next_step = "_generate_execution_graph"; state.graph_refinement_iterations = 0; state.scratchpad['refinement_validation_failures'] = 0; state.scratchpad['graph_gen_attempts'] = 0 
                    else: state.next_step = "refine_api_graph" 
                else: logger.warning(f"Max refinement iterations reached after LLM output error during refinement. Describing last valid graph or failing."); state.next_step = "describe_graph" 
        except Exception as e:
            # ... (existing exception handling for refinement) ...
            logger.error(f"Error during graph refinement LLM call or processing (iter {iteration}): {e}", exc_info=False); self._queue_intermediate_message(state, f"Error refining graph (iter {iteration}): {str(e)[:150]}..."); state.graph_regeneration_reason = state.graph_regeneration_reason or f"Refinement LLM call/processing error (iter {iteration}): {str(e)[:100]}..."
            if iteration < state.max_refinement_iterations: state.next_step = "refine_api_graph" 
            else: logger.warning(f"Max refinement iterations reached after exception. Describing graph or failing."); state.next_step = "describe_graph"
        return state
