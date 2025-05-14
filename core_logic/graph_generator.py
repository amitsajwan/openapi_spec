# core_logic/graph_generator.py
import json
import logging
import os
from typing import Any, Dict, Optional

from pydantic import ValidationError as PydanticValidationError

from models import BotState, GraphOutput
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
        """
        Initializes the GraphGenerator.

        Args:
            worker_llm: The language model used for generation and refinement tasks.
        """
        if not hasattr(worker_llm, "invoke"):
            raise TypeError("worker_llm must have an 'invoke' method.")
        self.worker_llm = worker_llm
        logger.info("GraphGenerator initialized.")

    def _queue_intermediate_message(self, state: BotState, msg: str):
        """
        Helper to queue messages for the UI and set the current response in BotState.
        This ensures that the user gets updates during long-running processes.
        """
        if "intermediate_messages" not in state.scratchpad:
            state.scratchpad["intermediate_messages"] = []
        # Avoid queuing exact same consecutive message if state.response was already it
        if (
            not state.scratchpad["intermediate_messages"]
            or state.scratchpad["intermediate_messages"][-1] != msg
        ):
            state.scratchpad["intermediate_messages"].append(msg)
        state.response = msg  # Update current response for logging or if it's the last one

    def _generate_execution_graph(
        self, state: BotState, goal: Optional[str] = None
    ) -> BotState:
        """
        Generates an API execution graph based on the user's goal and available APIs.
        This is the primary method for creating the initial graph.
        """
        tool_name = "_generate_execution_graph"
        current_goal = goal or state.plan_generation_goal or "General API workflow overview"
        self._queue_intermediate_message(
            state, f"Building API workflow graph for goal: '{current_goal[:70]}...'"
        )
        state.update_scratchpad_reason(tool_name, f"Generating graph. Goal: {current_goal}")

        if not state.identified_apis:
            self._queue_intermediate_message(
                state, "Cannot generate graph: No API operations identified."
            )
            state.execution_graph = None
            state.update_scratchpad_reason(tool_name, state.response)
            state.next_step = "responder"
            return state

        # Prepare a summary of identified APIs for the LLM prompt
        api_summaries_for_prompt = []
        num_apis_to_summarize = MAX_APIS_IN_PROMPT_SUMMARY_LONG
        truncate_threshold = MAX_TOTAL_APIS_THRESHOLD_FOR_TRUNCATION_LONG

        for idx, api in enumerate(state.identified_apis):
            if (
                idx >= num_apis_to_summarize
                and len(state.identified_apis) > truncate_threshold
            ):
                api_summaries_for_prompt.append(
                    f"- ...and {len(state.identified_apis) - num_apis_to_summarize} more operations."
                )
                break

            likely_confirmation = api["method"].upper() in [
                "POST",
                "PUT",
                "DELETE",
                "PATCH",
            ]
            params_str_parts = []
            if api.get("parameters"):
                for p_idx, p_detail in enumerate(api["parameters"]):
                    if p_idx >= 3: # Limit parameter details for brevity
                        params_str_parts.append("...")
                        break
                    param_name = p_detail.get("name", "N/A")
                    param_in = p_detail.get("in", "N/A")
                    param_schema = p_detail.get("schema", {})
                    param_type = (
                        param_schema.get("type", "unknown")
                        if isinstance(param_schema, dict)
                        else "unknown"
                    )
                    params_str_parts.append(f"{param_name}({param_in}, {param_type})")
            params_str = (
                f"Params: {', '.join(params_str_parts)}"
                if params_str_parts
                else "No explicit params listed."
            )

            req_body_info = ""
            if api.get("requestBody") and isinstance(api["requestBody"], dict):
                content = api["requestBody"].get("content", {})
                json_schema = content.get("application/json", {}).get("schema", {})
                if json_schema and isinstance(json_schema, dict) and json_schema.get(
                    "properties"
                ):
                    props = list(json_schema.get("properties", {}).keys())[:3]
                    req_body_info = (
                        f" ReqBody fields (sample from schema): {', '.join(props)}"
                        f"{'...' if len(json_schema.get('properties', {})) > 3 else ''}."
                    )

            api_summaries_for_prompt.append(
                f"- operationId: {api['operationId']} ({api['method']} {api['path']}), "
                f"summary: {api.get('summary', 'N/A')[:80]}. {params_str}{req_body_info} "
                f"likely_requires_confirmation: {'yes' if likely_confirmation else 'no'}"
            )
        apis_str = "\n".join(api_summaries_for_prompt)
        feedback_str = (
            f"Refinement Feedback: {state.graph_regeneration_reason}"
            if state.graph_regeneration_reason
            else ""
        )

        # Construct the prompt for the LLM
        prompt = f"""
        Goal: "{current_goal}". {feedback_str}
        Available API Operations (summary with parameters and sample request body fields from validated schemas):\n{apis_str}

        Design a logical and runnable API execution graph as a JSON object. The graph must achieve the specified Goal.
        Consider typical API workflow patterns. For example:
        - A 'create' operation (e.g., POST /items) should usually precede 'get by ID' (e.g., GET /items/{{{{itemId}}}}).
        - Data created in one step (e.g., an ID from a POST response) MUST be mapped via `OutputMapping` and then used in subsequent steps.

        The graph must adhere to the Pydantic models:
        InputMapping: {{"source_operation_id": "str_effective_id_of_source_node", "source_data_path": "str_jsonpath_to_value_in_extracted_ids (e.g., '$.createdItemIdFromStep1')", "target_parameter_name": "str_param_name_in_target_node (e.g., 'itemId')", "target_parameter_in": "Literal['path', 'query', 'body', 'body.fieldName']"}}
        OutputMapping: {{"source_data_path": "str_jsonpath_to_value_in_THIS_NODE_RESPONSE (e.g., '$.id', '$.data.token')", "target_data_key": "str_UNIQUE_key_for_shared_data_pool (e.g., 'createdItemId', 'userAuthToken')"}}
        Node: {{ ... "payload": {{ "template_key": "realistic_example_value or {{{{placeholder_from_output_mapping}}}}" }} ... }}

        CRITICAL INSTRUCTIONS FOR `payload` FIELD in Nodes (for POST, PUT, PATCH), using the schema information provided:
        1.  **Accuracy is Key:** The `payload` dictionary MUST ONLY contain fields that are actually defined by the specific API's request body schema (as hinted in 'ReqBody fields (sample from schema)' or from your knowledge of the API).
        2.  **Do Not Invent Fields:** Do NOT include any fields in the `payload` that are not part of the API's expected request body.
        3.  **Realistic Values:** Use realistic example values for fields (e.g., for "name": "Example Product", for "email": "test@example.com").
        4.  **Placeholders for Dynamic Data:** If a field's value should come from a previous step's output (via `OutputMapping`), use a placeholder like `{{{{key_from_output_mapping}}}}`. Ensure this placeholder matches a `target_data_key` from an `OutputMapping` of a preceding node.
        5.  **Optional Fields:** If a field is optional according to the API spec and no value is known or relevant to the goal, OMIT it from the payload rather than inventing a value or using a generic placeholder. If a default is sensible and known, use it.

        CRITICAL INSTRUCTIONS FOR DATA FLOW (e.g., Create Product then Get Product by ID):
        1.  **Create Node (e.g., POST /products):**
            * MUST have an `OutputMapping` to extract the ID of the newly created product from its response. Example: `{{"source_data_path": "$.id", "target_data_key": "newProductId"}}`. (Adjust `source_data_path` based on the actual API response structure for the ID).
        2.  **Get/Update/Delete Node (e.g., GET /products/{{{{some_id_placeholder}}}}):**
            * Its `path` MUST use a placeholder for the ID. This placeholder MUST exactly match the `target_data_key` from the "Create Node's" `OutputMapping`. Example Path: `/products/{{{{newProductId}}}}`.
            * Alternatively, if the path is `/products/{{pathParamName}}`, an `InputMapping` is needed: `{{ "source_operation_id": "effective_id_of_create_node", "source_data_path": "$.newProductId", "target_parameter_name": "pathParamName", "target_parameter_in": "path" }}`.

        General Instructions:
        - Create "START_NODE" and "END_NODE" (method: "SYSTEM").
        - Select 2-5 relevant API operations.
        - Set `requires_confirmation: true` for POST, PUT, DELETE, PATCH.
        - Connect nodes with `edges`. START_NODE to first API(s), last API(s) to END_NODE.
        - Ensure logical sequence.
        - Provide overall `description` and `refinement_summary`.

        Output ONLY the JSON object for GraphOutput. Ensure valid JSON.
        """

        try:
            llm_response = llm_call_helper(self.worker_llm, prompt)
            # Attempt to parse the LLM's JSON output and validate against GraphOutput model
            graph_output_candidate = parse_llm_json_output_with_model(
                llm_response, expected_model=GraphOutput
            )

            if graph_output_candidate:
                # Basic check for essential START_NODE and END_NODE
                if not any(
                    node.operationId == "START_NODE" for node in graph_output_candidate.nodes
                ) or not any(
                    node.operationId == "END_NODE" for node in graph_output_candidate.nodes
                ):
                    logger.error(
                        "LLM generated graph is missing START_NODE or END_NODE."
                    )
                    state.graph_regeneration_reason = (
                        "Generated graph missing START_NODE or END_NODE. "
                        "Please ensure they are included."
                    )
                    # Fall through to retry logic without setting execution_graph
                else:
                    state.execution_graph = graph_output_candidate
                    self._queue_intermediate_message(state, "API workflow graph generated.")
                    logger.info(
                        f"Graph generated. Description: {graph_output_candidate.description or 'N/A'}"
                    )
                    if graph_output_candidate.refinement_summary:
                        logger.info(
                            f"LLM summary for graph: {graph_output_candidate.refinement_summary}"
                        )
                    state.graph_regeneration_reason = None # Clear reason on successful generation
                    state.graph_refinement_iterations = 0 # Reset refinement attempts
                    state.next_step = "verify_graph" # Proceed to verification
                    state.update_scratchpad_reason(
                        tool_name, f"Graph gen success. Next: {state.next_step}"
                    )
                    return state # Successfully generated and validated graph

            # If graph_output_candidate is None or START/END nodes are missing
            error_msg = (
                "LLM failed to produce a valid GraphOutput JSON, or it was structurally incomplete "
                "(e.g., missing START/END nodes)."
            )
            logger.error(error_msg + f" Raw LLM output snippet: {llm_response[:300]}...")
            self._queue_intermediate_message(
                state,
                "Failed to generate a valid execution graph (AI output format, structure, or missing critical nodes like START/END).",
            )
            state.execution_graph = None # Ensure graph is None on failure
            state.graph_regeneration_reason = (
                state.graph_regeneration_reason
                or "LLM output was not a valid GraphOutput object or missed key structural elements."
            )

            # Retry logic for initial generation
            current_attempts = state.scratchpad.get("graph_gen_attempts", 0)
            if current_attempts < 1: # Allow one retry for initial generation
                state.scratchpad["graph_gen_attempts"] = current_attempts + 1
                logger.info(
                    "Retrying initial graph generation once due to validation/parsing failure."
                )
                state.next_step = "_generate_execution_graph" # Loop back to this node
            else:
                logger.error(
                    "Max initial graph generation attempts reached. Routing to handle_unknown."
                )
                state.next_step = "handle_unknown" # Give up after retry
                state.scratchpad["graph_gen_attempts"] = 0 # Reset for future attempts if any

        except Exception as e:
            logger.error(
                f"Error during graph generation LLM call or processing: {e}",
                exc_info=False, # Set to True for full traceback if needed for debugging
            )
            self._queue_intermediate_message(state, f"Error generating graph: {str(e)[:150]}...")
            state.execution_graph = None
            state.graph_regeneration_reason = (
                f"LLM call/processing error: {str(e)[:100]}..."
            )
            state.next_step = "handle_unknown" # Fallback on unexpected error

        state.update_scratchpad_reason(
            tool_name,
            f"Graph gen status: {'Success' if state.execution_graph else 'Failed'}. Resp: {state.response}",
        )
        return state

    def verify_graph(self, state: BotState) -> BotState:
        """
        Verifies the structural integrity and basic executability of the generated graph.
        Checks for cycles, presence of START/END nodes, and essential fields.
        """
        tool_name = "verify_graph"
        self._queue_intermediate_message(state, "Verifying API workflow graph...")
        state.update_scratchpad_reason(
            tool_name, "Verifying graph structure and integrity."
        )

        if not state.execution_graph or not isinstance(state.execution_graph, GraphOutput):
            current_response = state.response or ""
            self._queue_intermediate_message(
                state,
                current_response
                + " No execution graph to verify (possibly due to generation error or wrong type).",
            )
            state.graph_regeneration_reason = (
                state.graph_regeneration_reason or "No graph was generated to verify."
            )
            logger.warning(
                f"verify_graph: No graph found or invalid type. Reason: {state.graph_regeneration_reason}. "
                "Routing to _generate_execution_graph for regeneration."
            )
            state.next_step = "_generate_execution_graph"
            return state

        issues = []
        try:
            # Validate against Pydantic model (already done by parse_llm_json_output_with_model if successful)
            # but can be re-validated here for safety or if graph was modified.
            GraphOutput.model_validate(state.execution_graph.model_dump())

            # Check for cycles
            is_dag, cycle_msg = check_for_cycles(state.execution_graph)
            if not is_dag:
                issues.append(cycle_msg or "Graph contains cycles.")

            # Check for START_NODE and END_NODE presence
            node_ids = {node.effective_id for node in state.execution_graph.nodes}
            if "START_NODE" not in node_ids:
                issues.append("START_NODE is missing.")
            if "END_NODE" not in node_ids:
                issues.append("END_NODE is missing.")

            # Check START_NODE/END_NODE connectivity (basic checks)
            if "START_NODE" in node_ids:
                start_outgoing = any(
                    edge.from_node == "START_NODE" for edge in state.execution_graph.edges
                )
                start_incoming = any(
                    edge.to_node == "START_NODE" for edge in state.execution_graph.edges
                )
                if not start_outgoing and len(state.execution_graph.nodes) > 2: # More than just START/END
                    issues.append(
                        "START_NODE has no outgoing edges to actual API operations."
                    )
                if start_incoming:
                    issues.append("START_NODE should not have incoming edges.")

            if "END_NODE" in node_ids:
                end_incoming = any(
                    edge.to_node == "END_NODE" for edge in state.execution_graph.edges
                )
                end_outgoing = any(
                    edge.from_node == "END_NODE" for edge in state.execution_graph.edges
                )
                if not end_incoming and len(state.execution_graph.nodes) > 2: # More than just START/END
                    issues.append(
                        "END_NODE has no incoming edges from actual API operations."
                    )
                if end_outgoing:
                    issues.append("END_NODE should not have outgoing edges.")
            
            # Check if API nodes (non-START/END) have method and path
            for node in state.execution_graph.nodes:
                if node.effective_id.upper() not in ["START_NODE", "END_NODE"]:
                    if not node.method or not node.path:
                        issues.append(f"Node '{node.effective_id}' is missing 'method' or 'path', required for execution.")


        except PydanticValidationError as ve:
            logger.error(f"Graph Pydantic validation failed during verify_graph: {ve}")
            issues.append(f"Graph structure is invalid (Pydantic): {str(ve)[:200]}...")
        except Exception as e:
            logger.error(f"Unexpected error during graph verification: {e}", exc_info=True)
            issues.append(
                f"An unexpected error occurred during verification: {str(e)[:100]}."
            )

        if not issues:
            self._queue_intermediate_message(
                state,
                "Graph verification successful (Structure, DAG, START/END nodes, basic execution fields).",
            )
            state.update_scratchpad_reason(tool_name, "Graph verification successful.")
            logger.info("Graph verification successful.")
            state.graph_regeneration_reason = None # Clear reason on success
            state.scratchpad["refinement_validation_failures"] = 0 # Reset on successful verification

            # Send graph to UI
            try:
                state.scratchpad["graph_to_send"] = state.execution_graph.model_dump_json(indent=2)
                logger.info("Graph marked to be sent to UI after verification.")
            except Exception as e:
                logger.error(f"Error serializing graph for sending after verification: {e}")

            logger.info("Graph verified. Proceeding to describe graph.")
            state.next_step = "describe_graph"

            # If this verification followed spec parsing, provide a comprehensive success message
            if state.input_is_spec: # This flag is set by the router
                api_title = state.openapi_schema.get('info', {}).get('title', 'the API') if state.openapi_schema else 'the API'
                self._queue_intermediate_message(
                    state,
                    f"Successfully processed the OpenAPI specification for '{api_title}'. "
                    f"Identified {len(state.identified_apis)} API operations, "
                    f"generated example payloads, and created an API workflow graph with "
                    f"{len(state.execution_graph.nodes)} steps. The graph is verified. "
                    f"You can now ask questions, request specific plan refinements, or try to execute the workflow."
                )
                state.input_is_spec = False # Reset flag

        else:
            error_details = " ".join(issues)
            self._queue_intermediate_message(state, f"Graph verification failed: {error_details}.")
            state.graph_regeneration_reason = f"Verification failed: {error_details}."
            logger.warning(f"Graph verification failed: {error_details}.")

            if state.graph_refinement_iterations < state.max_refinement_iterations:
                logger.info(
                    f"Verification failed. Attempting graph refinement "
                    f"(iteration {state.graph_refinement_iterations + 1})."
                )
                state.next_step = "refine_api_graph"
            else:
                logger.warning(
                    "Max refinement iterations reached, but graph still has verification issues. "
                    "Attempting full regeneration."
                )
                state.next_step = "_generate_execution_graph"
                state.graph_refinement_iterations = 0 # Reset for full regeneration
                state.scratchpad["graph_gen_attempts"] = 0 # Reset for full regeneration

        state.update_scratchpad_reason(
            tool_name, f"Verification result: {state.response[:200]}..."
        )
        return state

    def refine_api_graph(self, state: BotState) -> BotState:
        """
        Attempts to refine the existing API execution graph based on feedback or verification failures.
        """
        tool_name = "refine_api_graph"
        iteration = state.graph_refinement_iterations + 1
        self._queue_intermediate_message(
            state,
            f"Refining API workflow graph (Attempt {iteration}/{state.max_refinement_iterations})...",
        )
        state.update_scratchpad_reason(
            tool_name,
            f"Refining graph. Iteration: {iteration}. Reason: {state.graph_regeneration_reason or 'General refinement request.'}",
        )

        if not state.execution_graph or not isinstance(state.execution_graph, GraphOutput):
            self._queue_intermediate_message(
                state, "No graph to refine or invalid graph type. Please generate a graph first."
            )
            logger.warning("refine_api_graph: No execution_graph found or invalid type.")
            state.next_step = "_generate_execution_graph" # Try to generate one
            return state

        if iteration > state.max_refinement_iterations:
            self._queue_intermediate_message(
                state,
                f"Max refinement iterations ({state.max_refinement_iterations}) reached. "
                f"Using current graph (description: {state.execution_graph.description or 'N/A'}). "
                "Please try a new goal or manually edit if needed.",
            )
            logger.warning("Max refinement iterations reached. Proceeding with current graph.")
            state.next_step = "describe_graph" # Describe the last valid graph
            return state

        try:
            current_graph_json = state.execution_graph.model_dump_json(indent=2)
        except Exception as e:
            logger.error(f"Error serializing current graph for refinement prompt: {e}")
            self._queue_intermediate_message(
                state, "Error preparing current graph for refinement. Cannot proceed."
            )
            state.next_step = "handle_unknown"
            return state

        # Prepare context for the LLM refinement prompt
        api_summaries_for_prompt = []
        num_apis_to_summarize = MAX_APIS_IN_PROMPT_SUMMARY_SHORT
        truncate_threshold = MAX_TOTAL_APIS_THRESHOLD_FOR_TRUNCATION_SHORT
        for idx, api in enumerate(state.identified_apis):
            if (
                idx >= num_apis_to_summarize
                and len(state.identified_apis) > truncate_threshold
            ):
                api_summaries_for_prompt.append(
                    f"- ...and {len(state.identified_apis) - num_apis_to_summarize} more operations."
                )
                break
            likely_confirmation = api["method"].upper() in [
                "POST",
                "PUT",
                "DELETE",
                "PATCH",
            ]
            api_summaries_for_prompt.append(
                f"- opId: {api['operationId']} ({api['method']} {api['path']}), "
                f"summary: {api.get('summary', 'N/A')[:70]}, "
                f"confirm: {'yes' if likely_confirmation else 'no'}"
            )
        apis_ctx = "\n".join(api_summaries_for_prompt)

        prompt = f"""
        User's Overall Goal: "{state.plan_generation_goal or 'General workflow'}"
        Feedback for Refinement: "{state.graph_regeneration_reason or 'General request to improve the graph.'}"
        Current Graph (JSON to be refined, based on validated schemas):\n```json\n{current_graph_json}\n```
        Available API Operations (sample for context, from validated schemas):\n{apis_ctx}

        Task: Refine the current graph based on the feedback. Ensure the refined graph:
        1.  Strictly adheres to the Pydantic model structure for GraphOutput, Node, Edge, InputMapping, OutputMapping.
            - For `payload` in Nodes: ONLY include fields defined by the API's request body schema. DO NOT invent fields. Use realistic example values or placeholders like `{{{{key_from_output_mapping}}}}` if data comes from a prior step. Omit optional fields if value is unknown.
        2.  Includes "START_NODE" and "END_NODE" correctly linked.
        3.  All node `operationId`s (or `display_name` if used as `effective_id`) in edges must exist in the `nodes` list.
        4.  Nodes intended for execution have `method` and `path` attributes.
        5.  `input_mappings` and `output_mappings` are logical for data flow. `source_data_path` should be plausible JSON paths. `target_data_key` in output_mappings should be unique and descriptive.
        6.  `requires_confirmation` is set appropriately.
        7.  Addresses the specific feedback. Ensure logical dependencies (e.g., create before get/update).
        8.  Provide a concise `refinement_summary` field in the JSON explaining what was changed or attempted.

        Output ONLY the refined GraphOutput JSON object.
        """
        try:
            llm_response_str = llm_call_helper(self.worker_llm, prompt)
            refined_graph_candidate = parse_llm_json_output_with_model(
                llm_response_str, expected_model=GraphOutput
            )

            if refined_graph_candidate:
                logger.info(
                    f"Refinement attempt (iter {iteration}) produced a structurally valid GraphOutput."
                )
                state.execution_graph = refined_graph_candidate
                refinement_summary = (
                    refined_graph_candidate.refinement_summary
                    or "AI provided no specific summary for this refinement."
                )
                state.update_scratchpad_reason(
                    tool_name,
                    f"LLM Refinement Summary (Iter {iteration}): {refinement_summary}",
                )
                self._queue_intermediate_message(
                    state, f"Graph refined (Iteration {iteration}). Summary: {refinement_summary}"
                )
                state.graph_refinement_iterations = iteration
                state.graph_regeneration_reason = None # Clear reason on successful refinement
                state.scratchpad["refinement_validation_failures"] = 0 # Reset on successful refinement
                state.next_step = "verify_graph" # Verify the refined graph
            else:
                error_msg = "LLM refinement failed to produce a GraphOutput JSON that is valid or self-consistent."
                logger.error(
                    error_msg
                    + f" Raw LLM output snippet for refinement: {llm_response_str[:300]}..."
                )
                self._queue_intermediate_message(
                    state,
                    f"Error during graph refinement (iteration {iteration}): AI output was invalid. Will retry refinement or regenerate graph.",
                )
                state.graph_regeneration_reason = (
                    state.graph_regeneration_reason
                    or "LLM output for refinement was not a valid GraphOutput object or had structural issues."
                )
                state.scratchpad["refinement_validation_failures"] = state.scratchpad.get("refinement_validation_failures", 0) + 1

                if iteration < state.max_refinement_iterations:
                    if state.scratchpad.get("refinement_validation_failures",0) >= 2: # If 2 consecutive refinement attempts fail validation
                        logger.warning(f"Multiple consecutive refinement validation failures (iter {iteration}). Escalating to full graph regeneration.")
                        self._queue_intermediate_message(state, state.response + " Attempting full regeneration due to persistent refinement issues.")
                        state.next_step = "_generate_execution_graph"
                        state.graph_refinement_iterations = 0 # Reset for full regeneration
                        state.scratchpad['refinement_validation_failures'] = 0
                        state.scratchpad['graph_gen_attempts'] = 0 # Reset for full regeneration
                    else:
                        state.next_step = "refine_api_graph" # Retry refinement
                else:
                    logger.warning(
                        "Max refinement iterations reached after LLM output error during refinement. "
                        "Describing last valid graph or failing."
                    )
                    state.next_step = "describe_graph" # Give up on refinement

        except Exception as e:
            logger.error(
                f"Error during graph refinement LLM call or processing (iter {iteration}): {e}",
                exc_info=False,
            )
            self._queue_intermediate_message(
                state, f"Error refining graph (iter {iteration}): {str(e)[:150]}..."
            )
            state.graph_regeneration_reason = (
                state.graph_regeneration_reason
                or f"Refinement LLM call/processing error (iter {iteration}): {str(e)[:100]}..."
            )
            if iteration < state.max_refinement_iterations:
                state.next_step = "refine_api_graph" # Retry refinement
            else:
                logger.warning(
                    "Max refinement iterations reached after exception. Describing graph or failing."
                )
                state.next_step = "describe_graph"
        return state
