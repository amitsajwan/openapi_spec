# core_logic/graph_generator.py
import logging
from typing import Any, Dict, Optional, List
import json 
import os
import re # Added for _extract_json_from_payload_description

from models import BotState, GraphOutput, Node 
from utils import (
    llm_call_helper, parse_llm_json_output_with_model, check_for_cycles
)
from pydantic import ValidationError as PydanticValidationError

logger = logging.getLogger(__name__)

# Configurable Limits
MAX_APIS_IN_PROMPT_SUMMARY_SHORT = int(os.getenv("MAX_APIS_IN_PROMPT_SUMMARY_SHORT", "10"))
MAX_APIS_IN_PROMPT_SUMMARY_LONG = int(os.getenv("MAX_APIS_IN_PROMPT_SUMMARY_LONG", "20")) 
MAX_TOTAL_APIS_THRESHOLD_FOR_TRUNCATION_SHORT = int(os.getenv("MAX_TOTAL_APIS_THRESHOLD_FOR_TRUNCATION_SHORT", "15"))
MAX_TOTAL_APIS_THRESHOLD_FOR_TRUNCATION_LONG = int(os.getenv("MAX_TOTAL_APIS_THRESHOLD_FOR_TRUNCATION_LONG", "25"))

class GraphGenerator:
    def __init__(self, worker_llm: Any):
        self.worker_llm = worker_llm
        logger.info("GraphGenerator initialized.")

    def _extract_json_from_payload_description(self, payload_desc_text: str) -> Optional[Dict[str, Any]]:
        if not payload_desc_text or not isinstance(payload_desc_text, str):
            return None
        try:
            match = re.search(r"Request Payload Example:\s*```json\s*([\s\S]*?)\s*```", payload_desc_text, re.DOTALL | re.IGNORECASE)
            if match:
                json_str = match.group(1).strip()
                return json.loads(json_str)
            match = re.search(r"```json\s*([\s\S]*?)\s*```", payload_desc_text, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
                return json.loads(json_str)
            if payload_desc_text.strip().startswith("{") and payload_desc_text.strip().endswith("}"):
                 return json.loads(payload_desc_text.strip())
        except json.JSONDecodeError:
            logger.warning(f"Could not extract JSON from payload_description: {payload_desc_text[:100]}...")
        return None

    def _generate_execution_graph(self, state: BotState, goal: Optional[str] = None) -> BotState:
        tool_name = "_generate_execution_graph"
        current_goal = goal or state.plan_generation_goal or "General API workflow overview"
        state.response = f"Building API workflow graph for goal: '{current_goal[:70]}...'"
        state.update_scratchpad_reason(tool_name, f"Generating graph. Goal: {current_goal}")

        if not state.identified_apis:
            state.response = "Cannot generate graph: No API operations identified."
            state.execution_graph = None 
            state.update_scratchpad_reason(tool_name, state.response)
            state.next_step = "responder" 
            return state

        api_summaries_for_prompt = []
        # ... (rest of api_summaries_for_prompt generation, including example_payload_str from previous version)
        for idx, api in enumerate(state.identified_apis):
            if idx >= MAX_APIS_IN_PROMPT_SUMMARY_LONG and len(state.identified_apis) > MAX_TOTAL_APIS_THRESHOLD_FOR_TRUNCATION_LONG:
                api_summaries_for_prompt.append(f"- ...and {len(state.identified_apis) - MAX_APIS_IN_PROMPT_SUMMARY_LONG} more operations.")
                break
            
            likely_confirmation = api['method'].upper() in ["POST", "PUT", "DELETE", "PATCH"]
            params_str_parts = []
            if api.get('parameters'):
                for p_idx, p_detail in enumerate(api['parameters']):
                    if p_idx >= 3: params_str_parts.append("..."); break 
                    param_name = p_detail.get('name', 'N/A'); param_in = p_detail.get('in', 'N/A')
                    param_schema = p_detail.get('schema', {}) 
                    param_type = param_schema.get('type', 'unknown') if isinstance(param_schema, dict) else 'unknown'
                    params_str_parts.append(f"{param_name}({param_in}, {param_type})")
            params_str = f"Params: {', '.join(params_str_parts)}" if params_str_parts else "No explicit params listed."
            
            req_body_info = ""
            if api.get('requestBody') and isinstance(api['requestBody'], dict):
                content = api['requestBody'].get('content', {})
                json_schema = content.get('application/json', {}).get('schema', {})
                if json_schema and isinstance(json_schema, dict) and json_schema.get('properties'):
                    props = list(json_schema.get('properties', {}).keys())[:3] 
                    req_body_info = f" ReqBody fields (sample from schema): {', '.join(props)}{'...' if len(json_schema.get('properties', {})) > 3 else ''}."
            
            example_payload_str = "Not available."
            if state.payload_descriptions and api['operationId'] in state.payload_descriptions:
                extracted_example = self._extract_json_from_payload_description(state.payload_descriptions[api['operationId']])
                if extracted_example:
                    example_payload_str = json.dumps(extracted_example) 
                elif "No request payload needed" in state.payload_descriptions[api['operationId']]:
                    example_payload_str = "No request payload typically needed."
                else:
                    example_payload_str = "Example structure description available (see payload_descriptions)."

            api_summaries_for_prompt.append(
                f"- operationId: {api['operationId']} ({api['method']} {api['path']}), "
                f"summary: {api.get('summary', 'N/A')[:80]}. {params_str}{req_body_info} " 
                f"Example Request Payload (from payload_descriptions, use as a strong reference for Node.payload if applicable): {example_payload_str}. "
                f"likely_requires_confirmation: {'yes' if likely_confirmation else 'no'}"
            )
        apis_str = "\n".join(api_summaries_for_prompt)
        feedback_str = f"Refinement Feedback: {state.graph_regeneration_reason}" if state.graph_regeneration_reason else ""
        
        prompt = f"""
        Goal: "{current_goal}". {feedback_str}
        Available API Operations (summary with parameters, sample request body fields from validated schemas, and example request payloads from payload_descriptions):\n{apis_str}

        Design a logical and runnable API execution graph as a JSON object. The graph must achieve the specified Goal.
        The entire output MUST be a single, valid JSON object adhering to the Pydantic models for GraphOutput, Node, Edge, InputMapping, and OutputMapping.

        **Key Pydantic Model Fields (Ensure ALL required fields are present):**
        - **GraphOutput:** `nodes` (List[Node]), `edges` (List[Edge]), `description` (Optional[str]), `refinement_summary` (Optional[str])
        - **Node:**
            - **REQUIRED:** `operationId` (str).
            - For API calls (not START_NODE/END_NODE): **REQUIRED:** `method` (str), `path` (str).
            - Optional: `display_name` (str), `summary` (str), `description` (str), `payload` (Dict), `payload_description` (str), `input_mappings` (List[InputMapping]), `output_mappings` (List[OutputMapping]), `requires_confirmation` (bool), `confirmation_prompt` (str).
        - **Edge:** **REQUIRED:** `from_node` (str), `to_node` (str). Optional: `description` (str).
        - **InputMapping:** **REQUIRED:** `source_operation_id` (str), `source_data_path` (str), `target_parameter_name` (str), `target_parameter_in` (Literal['path', 'query', 'header', 'cookie', 'body', 'body.fieldName']).
        - **OutputMapping:** **REQUIRED:** `source_data_path` (str), `target_data_key` (str).

        **CRITICAL INSTRUCTIONS for `payload` FIELD in Nodes (for POST, PUT, PATCH):**
        1.  **Use Provided Example:** For each API operation you include in the graph, REFER STRONGLY to its "Example Request Payload (from payload_descriptions)" provided above. Use this example as the primary template for the `Node.payload`.
        2.  **Adapt Example:** Adapt the example payload if necessary to fit the specific Goal of this graph. For instance, if the example has generic values but the Goal implies specific ones, use the specific ones.
        3.  **Schema Adherence:** The `payload` dictionary MUST ONLY contain fields that are actually defined by the specific API's request body schema.
        4.  **Do Not Invent Fields:** Do NOT include any fields in the `payload` that are not part of the API's expected request body.
        5.  **Placeholders for Dynamic Data:** If a field's value in the adapted example payload should come from a previous step's output, REPLACE the static example value with a placeholder like `{{{{key_from_output_mapping}}}}`. Ensure this placeholder matches a `target_data_key` from an `OutputMapping` of a preceding node.
        6.  **Optional Fields:** If a field is optional and the provided example payload omits it, or if its value is not relevant to the Goal, you can also omit it from the `Node.payload`.

        **CRITICAL INSTRUCTIONS for Graph Structure and Logic:**
        1.  **START_NODE and END_NODE:** ALWAYS include "START_NODE" and "END_NODE" (method: "SYSTEM", path: "/start" or "/end"). All other API nodes MUST be connected directly or indirectly between START_NODE and END_NODE. START_NODE should have outgoing edges to the first API(s). The last API(s) should have outgoing edges to END_NODE.
        2.  **Logical Sequencing (CRUD):** Ensure a logical sequence of operations. For example, a 'create' operation (POST) for a resource MUST precede any 'get by ID' (GET), 'update' (PUT/PATCH), or 'delete' (DELETE) operations for that *same specific resource instance*. Data created (like an ID) MUST be mapped via `OutputMapping` and used in subsequent relevant nodes.
        3.  **Data Flow:** If an API call (e.g., GET /items/{{{{itemId}}}}) depends on an ID from a previous step (e.g., a POST /items that created the item), its `path` MUST use a placeholder (e.g., `/items/{{{{createdItemId}}}}`) that matches a `target_data_key` from an `OutputMapping` of the preceding node. Alternatively, use an `InputMapping` for path parameters.
        4.  **Node Uniqueness & Connectivity:** `operationId` for Nodes MUST be unique within the graph unless `display_name` is used to differentiate. All API nodes must be part of a connected path from START_NODE to END_NODE. No orphaned nodes.
        5.  **Relevance:** Select 2-5 API operations most relevant to the Goal.
        6.  **Confirmation:** Set `requires_confirmation: true` for POST, PUT, DELETE, PATCH.
        7.  **Edges:** `from_node` and `to_node` in Edges MUST match `effective_id` of nodes.

        Provide overall `description` and `refinement_summary`.
        Output ONLY the JSON object for GraphOutput. Ensure valid JSON.
        """
        
        try:
            llm_response = llm_call_helper(self.worker_llm, prompt)
            graph_output_candidate = parse_llm_json_output_with_model(llm_response, expected_model=GraphOutput)

            if graph_output_candidate: 
                if not any(node.operationId == "START_NODE" for node in graph_output_candidate.nodes) or \
                   not any(node.operationId == "END_NODE" for node in graph_output_candidate.nodes):
                    logger.error("LLM generated graph is missing START_NODE or END_NODE.")
                    state.graph_regeneration_reason = "Generated graph missing START_NODE or END_NODE. Please ensure they are included."
                else:
                    missing_method_path = []
                    for node in graph_output_candidate.nodes:
                        if node.operationId.upper() not in ["START_NODE", "END_NODE"]:
                            if not node.method or not node.path:
                                missing_method_path.append(f"Node '{node.effective_id}' (OpID: {node.operationId or 'MISSING'}) is missing required 'method' or 'path'.")
                    if missing_method_path:
                        logger.error(f"LLM generated graph has API nodes with missing method/path: {missing_method_path}")
                        state.graph_regeneration_reason = "Generated graph has API nodes missing 'method' or 'path'. These are required. " + " ".join(missing_method_path)
                    else:
                        state.execution_graph = graph_output_candidate
                        state.response = "API workflow graph generated."
                        logger.info(f"Graph generated. Description: {graph_output_candidate.description or 'N/A'}")
                        if graph_output_candidate.refinement_summary:
                            logger.info(f"LLM summary for graph: {graph_output_candidate.refinement_summary}")
                        state.graph_regeneration_reason = None 
                        state.graph_refinement_iterations = 0 
                        state.next_step = "verify_graph" 
                        state.update_scratchpad_reason(tool_name, f"Graph gen success. Next: {state.next_step}")
                        return state 

            error_msg_detail = state.graph_regeneration_reason or "LLM failed to produce a valid GraphOutput JSON according to Pydantic models, or it was structurally incomplete."
            logger.error(error_msg_detail + f" Raw LLM output snippet: {llm_response[:300]}...")
            state.response = f"Failed to generate a valid execution graph: {error_msg_detail}"
            state.execution_graph = None 
            state.graph_regeneration_reason = error_msg_detail 
            
            current_attempts = state.scratchpad.get('graph_gen_attempts', 0)
            if current_attempts < 1: 
                state.scratchpad['graph_gen_attempts'] = current_attempts + 1
                logger.info("Retrying initial graph generation once due to validation/parsing failure or structural incompleteness.")
                state.next_step = "_generate_execution_graph" 
            else:
                logger.error("Max initial graph generation attempts reached. Routing to handle_unknown.")
                state.next_step = "handle_unknown" 
                state.scratchpad['graph_gen_attempts'] = 0 

        except Exception as e: 
            logger.error(f"Error during graph generation LLM call or processing: {e}", exc_info=True)
            state.response = f"Error generating graph: {str(e)[:150]}..."
            state.execution_graph = None
            state.graph_regeneration_reason = f"LLM call/processing error: {str(e)[:100]}..."
            state.next_step = "handle_unknown" 

        state.update_scratchpad_reason(tool_name, f"Graph gen status: {'Success' if state.execution_graph else 'Failed'}. Resp: {state.response}")
        return state

    def verify_graph(self, state: BotState) -> BotState:
        tool_name = "verify_graph"; state.response = "Verifying API workflow graph..."; state.update_scratchpad_reason(tool_name, "Verifying graph structure and integrity.")
        if not state.execution_graph or not isinstance(state.execution_graph, GraphOutput): 
            state.response = state.response or "No execution graph to verify (possibly due to generation error or wrong type)."
            state.graph_regeneration_reason = state.graph_regeneration_reason or "No graph was generated to verify."
            logger.warning(f"verify_graph: No graph found or invalid type. Reason: {state.graph_regeneration_reason}. Routing to _generate_execution_graph for regeneration.")
            state.next_step = "_generate_execution_graph"
            return state
        
        issues = []
        try:
            GraphOutput.model_validate(state.execution_graph.model_dump()) 
            is_dag, cycle_msg = check_for_cycles(state.execution_graph)
            if not is_dag: issues.append(cycle_msg or "Graph contains cycles.")
            
            node_ids = {node.effective_id for node in state.execution_graph.nodes}
            api_node_ids = {node.effective_id for node in state.execution_graph.nodes if node.operationId.upper() not in ["START_NODE", "END_NODE"]}

            if "START_NODE" not in node_ids: issues.append("START_NODE is missing.")
            if "END_NODE" not in node_ids: issues.append("END_NODE is missing.")
            
            if "START_NODE" in node_ids:
                start_outgoing_edges = [edge for edge in state.execution_graph.edges if edge.from_node == "START_NODE"]
                start_incoming = any(edge.to_node == "START_NODE" for edge in state.execution_graph.edges)
                if not start_outgoing_edges and api_node_ids : 
                    issues.append("START_NODE has no outgoing edges to any API operations.")
                if start_incoming: issues.append("START_NODE should not have incoming edges.")
                for edge in start_outgoing_edges:
                    if edge.to_node not in api_node_ids and edge.to_node != "END_NODE": # Allow START -> END if no API nodes
                        issues.append(f"START_NODE has an outgoing edge to '{edge.to_node}', which is not a recognized API node or END_NODE.")
            
            if "END_NODE" in node_ids:
                end_incoming_edges = [edge for edge in state.execution_graph.edges if edge.to_node == "END_NODE"]
                end_outgoing = any(edge.from_node == "END_NODE" for edge in state.execution_graph.edges)
                if not end_incoming_edges and api_node_ids: 
                    issues.append("END_NODE has no incoming edges from any API operations.")
                if end_outgoing: issues.append("END_NODE should not have outgoing edges.")
                for edge in end_incoming_edges:
                    if edge.from_node not in api_node_ids and edge.from_node != "START_NODE": # Allow START -> END
                         issues.append(f"END_NODE has an incoming edge from '{edge.from_node}', which is not a recognized API node or START_NODE.")
            
            # Check reachability for all API nodes from START_NODE and to END_NODE
            if api_node_ids:
                # Build adjacency list for graph traversal
                adj: Dict[str, List[str]] = {nid: [] for nid in node_ids}
                for edge in state.execution_graph.edges:
                    if edge.from_node in adj: # Ensure from_node is valid before adding
                        adj[edge.from_node].append(edge.to_node)

                # Reachability from START
                q = ["START_NODE"] if "START_NODE" in node_ids else []
                reachable_from_start = set(q)
                head = 0
                while head < len(q):
                    curr = q[head]; head += 1
                    for neighbor in adj.get(curr, []):
                        if neighbor not in reachable_from_start:
                            reachable_from_start.add(neighbor); q.append(neighbor)
                
                unreachable_apis = api_node_ids - reachable_from_start
                if unreachable_apis:
                    issues.append(f"Unreachable API nodes from START_NODE: {', '.join(list(unreachable_apis)[:3])}{'...' if len(unreachable_apis) > 3 else ''}.")

                # Reachability to END (reverse graph traversal)
                # This is more complex; for now, we rely on prompt and START_NODE checks.
                # A simple check: if an API node has no outgoing edges and is not END_NODE, it's a problem.
                for api_id in api_node_ids:
                    if not any(edge.from_node == api_id for edge in state.execution_graph.edges):
                        issues.append(f"API Node '{api_id}' has no outgoing edges and is not connected to END_NODE.")


            for node in state.execution_graph.nodes:
                if node.effective_id.upper() not in ["START_NODE", "END_NODE"]: 
                    if not node.operationId: 
                        issues.append(f"Node with effective_id '{node.effective_id}' is missing mand
