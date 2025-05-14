# core_logic/interaction_handler.py
import logging
import json # For preparing schema snippet
import os
from typing import Any, Dict, Optional

from models import BotState, GraphOutput
from utils import llm_call_helper, parse_llm_json_output_with_model
# To avoid circular imports, InteractionHandler will call methods on instances
# of GraphGenerator and SpecProcessor passed to its constructor.

logger = logging.getLogger(__name__)

# --- Configurable Limits from environment variables ---
MAX_APIS_IN_PROMPT_SUMMARY_SHORT = int(
    os.getenv("MAX_APIS_IN_PROMPT_SUMMARY_SHORT", "10")
)
# Max characters for the schema overview snippet in the prompt
MAX_SCHEMA_OVERVIEW_LENGTH = int(os.getenv("MAX_SCHEMA_OVERVIEW_LENGTH", "2000"))


class InteractionHandler:
    """
    Handles user interactions like querying API details, describing the graph,
    planning and executing multi-step interactive queries, and managing workflow execution setup.
    """

    def __init__(
        self,
        worker_llm: Any,
        graph_generator_instance: Any, # Instance of GraphGenerator
        spec_processor_instance: Any,   # Instance of SpecProcessor
        api_executor_instance: Any # Instance of APIExecutor
    ):
        """
        Initializes the InteractionHandler.

        Args:
            worker_llm: The language model for performing tasks.
            graph_generator_instance: An instance of GraphGenerator for graph-related actions.
            spec_processor_instance: An instance of SpecProcessor for spec-related actions.
            api_executor_instance: Instance of APIExecutor, primarily for context or future use
                                   in workflow setup, not for direct API calls from Graph 1.
        """
        if not hasattr(worker_llm, "invoke"):
            raise TypeError("worker_llm must have an 'invoke' method.")
        self.worker_llm = worker_llm
        self.graph_generator = graph_generator_instance
        self.spec_processor = spec_processor_instance
        self.api_executor = api_executor_instance
        logger.info("InteractionHandler initialized.")

    def _queue_intermediate_message(self, state: BotState, msg: str):
        """Helper to queue messages for the UI and set the current response in BotState."""
        if "intermediate_messages" not in state.scratchpad:
            state.scratchpad["intermediate_messages"] = []
        if (
            not state.scratchpad["intermediate_messages"]
            or state.scratchpad["intermediate_messages"][-1] != msg
        ):
            state.scratchpad["intermediate_messages"].append(msg)
        state.response = msg

    def describe_graph(self, state: BotState) -> BotState:
        """
        Generates or retrieves a natural language description of the current execution graph.
        """
        tool_name = "describe_graph"
        self._queue_intermediate_message(state, "Preparing graph description...")
        state.update_scratchpad_reason(
            tool_name, "Preparing to describe the current execution graph."
        )

        if not state.execution_graph or not isinstance(
            state.execution_graph, GraphOutput
        ):
            self._queue_intermediate_message(
                state,
                (state.response or "")
                + " No execution graph is currently available to describe or graph is invalid.",
            )
            logger.warning(
                "describe_graph: No execution_graph found in state or invalid type."
            )
        else:
            graph_desc = state.execution_graph.description
            final_desc_for_user = ""

            if not graph_desc or len(graph_desc) < 20:
                logger.info("Graph description is short or missing, generating a dynamic one.")
                node_summaries = []
                for node in state.execution_graph.nodes:
                    node_summaries.append(
                        f"- {node.effective_id}: {node.summary or node.operationId[:50]}"
                    )

                nodes_str = "\n".join(node_summaries[:5])
                if len(node_summaries) > 5:
                    nodes_str += f"\n- ... and {len(node_summaries) - 5} more nodes."

                prompt = (
                    f"The following API execution graph has been generated for the goal: '{state.plan_generation_goal or 'general use'}'.\n"
                    f"Nodes in the graph ({len(state.execution_graph.nodes)} total, sample):\n{nodes_str}\n\n"
                    f"Please provide a concise, user-friendly natural language description of this workflow. "
                    f"Explain its overall purpose and the general sequence of operations. "
                    f"Use Markdown for readability (e.g., a brief introductory sentence, then bullet points for key stages if appropriate)."
                )
                try:
                    dynamic_desc = llm_call_helper(self.worker_llm, prompt)
                    if graph_desc and graph_desc != dynamic_desc:
                        final_desc_for_user = (
                            f"**Overall Workflow Plan for: '{state.plan_generation_goal or 'General Use'}'**\n\n"
                            f"{dynamic_desc}\n\n"
                            f"*Original AI-generated graph description: {graph_desc}*"
                        )
                    else:
                        final_desc_for_user = (
                            f"**Current API Workflow for: '{state.plan_generation_goal or 'General Use'}'**\n\n"
                            f"{dynamic_desc}"
                        )
                except Exception as e:
                    logger.error(f"Error generating dynamic graph description: {e}")
                    default_node_preview = ', '.join([n.effective_id for n in state.execution_graph.nodes[:3]]) + ('...' if len(state.execution_graph.nodes) > 3 else '')
                    final_desc_for_user = (
                        f"**Current API Workflow for: '{state.plan_generation_goal or 'General Use'}'**\n\n"
                        f"{graph_desc or f'No detailed description available. The graph includes nodes like {default_node_preview}'}"
                    )
            else:
                final_desc_for_user = (
                    f"**Current API Workflow for: '{state.plan_generation_goal or 'General Use'}'**\n\n"
                    f"{graph_desc}"
                )

            if state.execution_graph.refinement_summary:
                final_desc_for_user += f"\n\n**Last Refinement Note:** {state.execution_graph.refinement_summary}"

            self._queue_intermediate_message(state, final_desc_for_user)
            if 'graph_to_send' not in state.scratchpad and state.execution_graph:
                 try:
                     state.scratchpad['graph_to_send'] = state.execution_graph.model_dump_json(indent=2)
                 except Exception as e:
                     logger.error(f"Error serializing graph for sending during describe_graph: {e}")

        state.update_scratchpad_reason(
            tool_name,
            f"Graph description generated/retrieved. Response set: {state.response[:100]}...",
        )
        state.next_step = "responder"
        return state

    def get_graph_json(self, state: BotState) -> BotState:
        """
        Provides the JSON representation of the current execution graph to the UI.
        """
        tool_name = "get_graph_json"
        self._queue_intermediate_message(state, "Fetching graph JSON...")
        state.update_scratchpad_reason(tool_name, "Attempting to provide graph JSON.")

        if not state.execution_graph or not isinstance(
            state.execution_graph, GraphOutput
        ):
            self._queue_intermediate_message(
                state, "No execution graph is currently available or graph is invalid."
            )
        else:
            try:
                graph_json_str = state.execution_graph.model_dump_json(indent=2)
                state.scratchpad["graph_to_send"] = graph_json_str
                self._queue_intermediate_message(
                    state,
                    "The current API workflow graph is available in the graph view. You can also copy the JSON from there if needed.",
                )
                logger.info("Provided graph JSON to scratchpad for UI.")
            except Exception as e:
                logger.error(f"Error serializing execution_graph to JSON: {e}")
                self._queue_intermediate_message(state, f"Error serializing graph to JSON: {str(e)}")

        state.next_step = "responder"
        return state

    def answer_openapi_query(self, state: BotState) -> BotState:
        """
        Answers user questions based on the loaded OpenAPI specification, generated graph,
        and a snippet of the processed schema.
        """
        tool_name = "answer_openapi_query"
        self._queue_intermediate_message(state, "Thinking about your question...")
        state.update_scratchpad_reason(
            tool_name,
            f"Attempting to answer user query: {state.user_input[:100] if state.user_input else 'N/A'}",
        )

        if not state.openapi_schema and not (
            state.execution_graph and isinstance(state.execution_graph, GraphOutput)
        ):
            self._queue_intermediate_message(
                state,
                "I don't have an OpenAPI specification loaded or a graph generated yet. Please provide one first.",
            )
            state.next_step = "responder"
            return state

        context_parts = []
        if state.user_input:
            context_parts.append(f"User Question: \"{state.user_input}\"")

        # Add OpenAPI Schema Overview (truncated)
        if state.openapi_schema:
            try:
                limited_schema_for_prompt = {
                    "info": state.openapi_schema.get("info", {}),
                    "servers": state.openapi_schema.get("servers", []),
                }
                paths_overview = {}
                paths_count = 0
                for path_url, path_item in state.openapi_schema.get("paths", {}).items():
                    if paths_count >= 5:  # Limit to first 5 paths for overview
                        paths_overview["... (more paths exist)"] = "..."
                        break
                    path_methods_summary = {}
                    if isinstance(path_item, dict):
                        for method, op_details in path_item.items():
                            if isinstance(op_details, dict):
                                path_methods_summary[method] = {
                                    "summary": (op_details.get("summary", "N/A")[:70] + "...") if op_details.get("summary") else "N/A",
                                    "operationId": op_details.get("operationId", "N/A")
                                }
                    paths_overview[path_url] = path_methods_summary
                    paths_count += 1
                if paths_overview:
                    limited_schema_for_prompt["paths_overview"] = paths_overview
                
                component_schemas_overview = {}
                schemas_count = 0
                if "components" in state.openapi_schema and isinstance(state.openapi_schema["components"], dict):
                    for schema_name, schema_def in state.openapi_schema["components"].get("schemas", {}).items():
                        if schemas_count >= 5: # Limit to first 5 schema names
                            component_schemas_overview["... (more schemas exist)"] = "..."
                            break
                        component_schemas_overview[schema_name] = {"type": schema_def.get("type", "object/ref")} # Indicate type or if it's a ref (though v3 refs should be resolved)
                        schemas_count += 1
                if component_schemas_overview:
                     limited_schema_for_prompt["components_schemas_overview"] = component_schemas_overview

                spec_detail_str = json.dumps(limited_schema_for_prompt, indent=2)
                if len(spec_detail_str) > MAX_SCHEMA_OVERVIEW_LENGTH:
                    spec_detail_str = spec_detail_str[:MAX_SCHEMA_OVERVIEW_LENGTH] + "\n... (specification details truncated)"
                
                context_parts.append(f"\n### OpenAPI Specification Structure Overview (JSON):\n```json\n{spec_detail_str}\n```")
            except Exception as e:
                logger.error(f"Error preparing openapi_schema detail for prompt: {e}")
                context_parts.append("\n### OpenAPI Specification Structure Overview:\nError preparing schema details for display.")


        if state.schema_summary:
            context_parts.append(
                f"\n### API Specification Summary (AI Generated):\n{state.schema_summary}"
            )

        identified_apis_md = "\n### Identified API Operations (Sample):\n"
        if state.identified_apis:
            num_apis_to_list = MAX_APIS_IN_PROMPT_SUMMARY_SHORT
            for i, api in enumerate(state.identified_apis[:num_apis_to_list]):
                identified_apis_md += (
                    f"- **Operation ID:** `{api.get('operationId', 'N/A')}`\n"
                    f"  - **Method & Path:** `{api.get('method', '?')} {api.get('path', '?')}`\n"
                    f"  - **Summary:** _{api.get('summary', 'No summary')[:100]}..._\n"
                )
            if len(state.identified_apis) > num_apis_to_list:
                identified_apis_md += (
                    f"- ... and {len(state.identified_apis) - num_apis_to_list} more.\n"
                )
        else:
            identified_apis_md += "No specific API operations identified yet.\n"
        context_parts.append(identified_apis_md)

        if (
            state.execution_graph
            and isinstance(state.execution_graph, GraphOutput)
            and state.execution_graph.description
        ):
            graph_desc_md = (
                f"\n### Current Workflow Graph ('{state.plan_generation_goal or 'General Purpose'}') Description:\n"
                f"{state.execution_graph.description}"
            )
            if state.execution_graph.refinement_summary:
                graph_desc_md += (
                    f"\nLast Refinement: {state.execution_graph.refinement_summary}"
                )
            context_parts.append(graph_desc_md)
        elif state.execution_graph and isinstance(state.execution_graph, GraphOutput):
            context_parts.append(
                f"\n### Current Workflow Graph ('{state.plan_generation_goal or 'General Purpose'}') exists but has no detailed description."
            )

        payload_descriptions_md = (
            "\n### Available Payload/Response Examples (AI Generated, for reference):\n"
        )
        if state.payload_descriptions:
            for op_id, desc_text in list(state.payload_descriptions.items())[:3]: # Show first 3 examples
                payload_descriptions_md += (
                    f"**For Operation ID `{op_id}`:**\n```text\n{desc_text[:300]}...\n```\n\n"
                )
            if len(state.payload_descriptions) > 3:
                payload_descriptions_md += "... and more examples exist for other operations.\n"
        else:
            payload_descriptions_md += (
                "No detailed payload/response examples have been generated yet.\n"
            )
        context_parts.append(payload_descriptions_md)

        full_context = "\n".join(context_parts)
        if not full_context.strip():
            full_context = (
                "No specific API context available, but an OpenAPI spec might be loaded."
            )

        prompt = f"""You are an expert API assistant. Your task is to answer the user's question based on the provided context.
        The context includes:
        1.  An "OpenAPI Specification Structure Overview" (a truncated JSON snippet of the actual schema, potentially with resolved $refs for OpenAPI 3.x).
        2.  An "API Specification Summary" (AI-generated high-level summary).
        3.  A list of "Identified API Operations" with their summaries.
        4.  A "Current Workflow Graph Description" if a plan exists.
        5.  "Available Payload/Response Examples" (AI-generated examples for some operations).

        Context:
        {full_context}

        User Question: "{state.user_input}"

        **Instructions for Answering:**
        1.  **Prioritize Information:** Use the "OpenAPI Specification Structure Overview" for direct schema details (like exact field names, data types if visible, or structure of paths and components). Supplement this with information from other context sections.
        2.  **Understand the Question:** Determine if the user is asking for general information, details about a specific API, or "how-to" perform an action.
        3.  **For "How-To" Questions:**
            a.  Identify relevant API operation(s) from the "Identified API Operations" or "OpenAPI Specification Structure Overview".
            b.  State the identified operation: its Operation ID, Method, and Path.
            c.  Refer to "Payload/Response Examples" or infer from the "OpenAPI Specification Structure Overview" for request body structure and key fields.
            d.  List important path or query parameters.
            e.  Describe the expected successful response.
        4.  **For Questions about Specific APIs:**
            a.  Find the API in "Identified API Operations" or "OpenAPI Specification Structure Overview".
            b.  Provide its summary, method, path. Use the "OpenAPI Specification Structure Overview" for more precise details if needed.
        5.  **Clarity and Conciseness:** Provide a clear, concise, and helpful answer.
        6.  **Formatting:** Use Markdown (headings, lists, bolding, code blocks for JSON examples or API paths).
        7.  **Unavailable Information:** If the information is not available in the context, state that clearly. Do not invent details.
        8.  **No Conversational Fluff:** Focus only on answering the question directly.

        Answer:
        """
        try:
            answer = llm_call_helper(self.worker_llm, prompt)
            self._queue_intermediate_message(state, answer)
            logger.info("Successfully generated answer for OpenAPI query.")
        except Exception as e:
            logger.error(
                f"Error generating answer for OpenAPI query: {e}", exc_info=False
            )
            self._queue_intermediate_message(
                state,
                f"### Error Answering Query\nSorry, I encountered an error while trying to answer your question: {str(e)[:100]}...",
            )

        state.update_scratchpad_reason(
            tool_name, f"Answered query. Response snippet: {state.response[:100]}..."
        )
        state.next_step = "responder"
        return state

    def interactive_query_planner(self, state: BotState) -> BotState:
        """
        Plans a sequence of internal actions to address complex user queries
        that may involve modifying state or generating new artifacts.
        """
        tool_name = "interactive_query_planner"
        self._queue_intermediate_message(
            state, "Planning how to address your interactive query..."
        )
        state.update_scratchpad_reason(
            tool_name,
            f"Entering interactive query planner for input: {state.user_input[:100] if state.user_input else 'N/A'}",
        )

        state.scratchpad.pop("interactive_action_plan", None)
        state.scratchpad.pop("current_interactive_action_idx", None)
        state.scratchpad.pop("current_interactive_results", None)

        graph_summary = (
            state.execution_graph.description[:150] + "..."
            if state.execution_graph
            and isinstance(state.execution_graph, GraphOutput)
            and state.execution_graph.description
            else "No graph currently generated."
        )
        payload_keys_sample = list(state.payload_descriptions.keys())[:3]

        prompt = f"""User Query: "{state.user_input}"

Current State Context:
- API Spec Summary: {'Available' if state.schema_summary else 'Not available.'}
- Identified APIs count: {len(state.identified_apis) if state.identified_apis else 0}. Example OpIDs: {", ".join([api['operationId'] for api in state.identified_apis[:3]])}...
- Example Payload Descriptions available for OpIDs (sample): {payload_keys_sample}...
- Current Execution Graph Goal: {state.plan_generation_goal or 'Not set.'}
- Current Graph Description: {graph_summary}
- Workflow Execution Status: {state.workflow_execution_status}

Available Internal Actions (choose one or more in sequence, output as a JSON list):
1.  `rerun_payload_generation`: Regenerate payload/response examples for specific APIs, possibly with new user-provided context.
    Params: {{ "operation_ids_to_update": ["opId1", "opId2"], "new_context": "User's new context string for generation" }}
2.  `contextualize_graph_descriptions`: Rewrite descriptions within the *existing* graph (overall, nodes, edges) to reflect new user context or focus. This does NOT change graph structure.
    Params: {{ "new_context_for_graph": "User's new context/focus for descriptions" }}
3.  `regenerate_graph_with_new_goal`: Create a *new* graph if the user states a completely different high-level goal OR requests a significant structural change (add/remove/reorder API steps).
    Params: {{ "new_goal_string": "User's new goal, incorporating the structural change (e.g., 'Workflow to X, then Y, and then Z as the last step')" }}
4.  `refine_existing_graph_structure`: For minor structural adjustments to the existing graph (e.g., "add API Z after Y but before END_NODE", "remove API X"). This implies the overall goal is similar but the sequence/nodes need adjustment. The LLM will be asked to refine the current graph JSON.
    Params: {{ "refinement_instructions_for_structure": "User's specific feedback for structural refinement (e.g., 'Add operation Z after Y', 'Ensure X comes before Y')" }}
5.  `answer_query_directly`: If the query can be answered using existing information (API summary, API list, current graph description, existing payload examples, schema overview) without modifications to artifacts.
    Params: {{ "query_for_synthesizer": "The original user query or a rephrased one for direct answering." }}
6.  `setup_workflow_execution_interactive`: If the user asks to run/execute the current graph. This action prepares the system for execution.
    Params: {{ "initial_parameters": {{ "param1": "value1" }} }} (Optional initial parameters for the workflow, if provided by user)
7.  `resume_workflow_with_payload_interactive`: If the workflow is 'paused_for_confirmation' and the user provides the necessary payload/confirmation to continue.
    Params: {{ "confirmed_payload": {{...}} }} (The JSON payload confirmed or provided by the user)
8.  `synthesize_final_answer`: (Usually the last step of a plan) Formulate a comprehensive answer to the user based on the outcomes of previous internal actions or if no other action is suitable.
    Params: {{ "synthesis_prompt_instructions": "Instructions for the LLM on what to include in the final answer, summarizing actions taken or information gathered." }}

Task:
1. Analyze the user's query in the context of the current system state.
2. Create a short, logical "interactive_action_plan" (a list of action objects, max 3-4 steps).
   - For requests to run the graph, use `setup_workflow_execution_interactive`.
   - If the graph is paused and user provides data, use `resume_workflow_with_payload_interactive`.
   - For structural changes like "add X at the end", prefer `regenerate_graph_with_new_goal` or `refine_existing_graph_structure`.
3. Provide a brief "user_query_understanding" (1-2 sentences).

Output ONLY a JSON object with this structure:
{{
  "user_query_understanding": "Brief interpretation of user's need.",
  "interactive_action_plan": [
    {{"action_name": "action_enum_value", "action_params": {{...}}, "description": "Briefly, why this action is chosen."}}
  ]
}}
If the query is very simple and can be answered directly, the plan might just be one "answer_query_directly" or "synthesize_final_answer" action.
If the query is ambiguous or cannot be handled by available actions, use "synthesize_final_answer" with instructions to inform the user."""
        try:
            llm_response_str = llm_call_helper(self.worker_llm, prompt)
            parsed_plan_data = parse_llm_json_output_with_model(llm_response_str)

            if (
                parsed_plan_data
                and isinstance(parsed_plan_data, dict)
                and "interactive_action_plan" in parsed_plan_data
                and isinstance(parsed_plan_data["interactive_action_plan"], list)
                and "user_query_understanding" in parsed_plan_data
            ):
                state.scratchpad["user_query_understanding"] = parsed_plan_data[
                    "user_query_understanding"
                ]
                state.scratchpad["interactive_action_plan"] = parsed_plan_data[
                    "interactive_action_plan"
                ]
                state.scratchpad["current_interactive_action_idx"] = 0
                state.scratchpad["current_interactive_results"] = []
                self._queue_intermediate_message(
                    state,
                    f"Understood query: {state.scratchpad['user_query_understanding']}. Starting internal actions...",
                )
                logger.info(
                    f"Interactive plan generated: {state.scratchpad['interactive_action_plan']}"
                )
                state.next_step = "interactive_query_executor"
            else:
                logger.error(
                    f"LLM failed to produce a valid interactive plan. Raw: {llm_response_str[:300]}"
                )
                raise ValueError(
                    "LLM failed to produce a valid interactive plan JSON structure with required keys."
                )
        except Exception as e:
            logger.error(f"Error in interactive_query_planner: {e}", exc_info=False)
            self._queue_intermediate_message(
                state,
                f"Sorry, I encountered an error while planning how to address your request: {str(e)[:100]}...",
            )
            state.next_step = "answer_openapi_query"

        state.update_scratchpad_reason(
            tool_name,
            f"Interactive plan generated. Next: {state.next_step}. Response: {state.response[:100]}",
        )
        return state

    def _internal_contextualize_graph_descriptions(
        self, state: BotState, new_context: str
    ) -> str:
        """Internal helper to rewrite graph descriptions based on new context."""
        tool_name = "_internal_contextualize_graph_descriptions"
        if not state.execution_graph or not isinstance(
            state.execution_graph, GraphOutput
        ):
            return "No graph to contextualize or graph is invalid."
        if not new_context:
            return "No new context provided for contextualization."

        logger.info(
            f"Attempting to contextualize graph descriptions with context: {new_context[:100]}..."
        )
        original_graph_desc = state.execution_graph.description

        if state.execution_graph.description:
            prompt_overall = (
                f"Current overall graph description: \"{state.execution_graph.description}\"\n"
                f"New User Context/Focus: \"{new_context}\"\n\n"
                f"Rewrite the graph description to incorporate this new context/focus, keeping it concise. Output only the new description text."
            )
            try:
                state.execution_graph.description = llm_call_helper(
                    self.worker_llm, prompt_overall
                )
                logger.info(
                    f"Overall graph description contextualized: {state.execution_graph.description[:100]}..."
                )
            except Exception as e:
                logger.error(f"Error contextualizing overall graph description: {e}")
                state.execution_graph.description = original_graph_desc

        nodes_to_update = [
            n
            for n in state.execution_graph.nodes
            if n.operationId not in ["START_NODE", "END_NODE"]
        ][:3]
        for node in nodes_to_update:
            original_node_desc = node.description
            if node.description:
                prompt_node = (
                    f"Current description for node '{node.effective_id}' ({node.summary}): \"{node.description}\"\n"
                    f"Overall User Context/Focus for the graph: \"{new_context}\"\n\n"
                    f"Rewrite this node's description to align with the new context, focusing on its role in the workflow under this context. Output only the new description text for this node."
                )
                try:
                    node.description = llm_call_helper(self.worker_llm, prompt_node)
                    logger.info(
                        f"Node '{node.effective_id}' description contextualized: {node.description[:100]}..."
                    )
                except Exception as e:
                    logger.error(
                        f"Error contextualizing node '{node.effective_id}' description: {e}"
                    )
                    node.description = original_node_desc
        if state.execution_graph:
            state.scratchpad["graph_to_send"] = state.execution_graph.model_dump_json(
                indent=2
            )
        
        state.update_scratchpad_reason(
            tool_name, f"Graph descriptions contextualized with context: {new_context[:70]}."
        )
        return f"Graph descriptions have been updated to reflect the context: '{new_context[:70]}...'."

    def interactive_query_executor(self, state: BotState) -> BotState:
        """Executes the planned sequence of internal actions."""
        tool_name = "interactive_query_executor"
        plan = state.scratchpad.get("interactive_action_plan", [])
        idx = state.scratchpad.get("current_interactive_action_idx", 0)
        results = state.scratchpad.get("current_interactive_results", [])

        if not plan or idx >= len(plan):
            final_response_message = "Finished interactive processing. "
            if results:
                last_result_str = str(results[-1])
                final_response_message += (
                    last_result_str[:200] + "..."
                    if len(last_result_str) > 200
                    else last_result_str
                )
            else:
                final_response_message += "No specific actions were taken or results to report."
            
            if not state.response or state.response.startswith("Executing internal step"):
                 self._queue_intermediate_message(state, final_response_message)

            logger.info("Interactive plan execution completed or no plan.")
            state.next_step = "responder"
            state.update_scratchpad_reason(
                tool_name, "Interactive plan execution completed or no plan."
            )
            return state

        action = plan[idx]
        action_name = action.get("action_name")
        action_params = action.get("action_params", {})
        action_description = action.get("description", "No description for action.")

        self._queue_intermediate_message(
            state, f"Executing internal step ({idx + 1}/{len(plan)}): {action_description[:70]}..."
        )
        state.update_scratchpad_reason(
            tool_name,
            f"Executing action ({idx + 1}/{len(plan)}): {action_name} - {action_description}",
        )
        action_result_message = f"Action '{action_name}' completed."

        try:
            if action_name == "rerun_payload_generation":
                op_ids = action_params.get("operation_ids_to_update", [])
                new_ctx = action_params.get("new_context", "")
                if op_ids and isinstance(op_ids, list) and new_ctx:
                    self.spec_processor._generate_payload_descriptions(
                        state, target_apis=op_ids, context_override=new_ctx
                    )
                    action_result_message = (
                        f"Payload examples update requested for {op_ids} with context '{new_ctx[:30]}...'."
                    )
                else:
                    action_result_message = "Skipped rerun_payload_generation: Missing operation_ids or new_context, or invalid format."
                state.next_step = "interactive_query_executor"

            elif action_name == "contextualize_graph_descriptions":
                new_ctx_graph = action_params.get("new_context_for_graph", "")
                if new_ctx_graph:
                    action_result_message = self._internal_contextualize_graph_descriptions(
                        state, new_ctx_graph
                    )
                else:
                    action_result_message = "Skipped contextualize_graph_descriptions: Missing new_context_for_graph."
                state.next_step = "interactive_query_executor"

            elif action_name == "regenerate_graph_with_new_goal":
                new_goal = action_params.get("new_goal_string")
                if new_goal:
                    state.plan_generation_goal = new_goal
                    state.execution_graph = None
                    state.graph_refinement_iterations = 0
                    state.scratchpad['graph_gen_attempts'] = 0
                    state.scratchpad['refinement_validation_failures'] = 0
                    state = self.graph_generator._generate_execution_graph(
                        state, goal=new_goal
                    )
                    action_result_message = (
                        f"Graph regeneration started for new goal: {new_goal[:50]}..."
                    )
                else:
                    action_result_message = "Skipped regenerate_graph_with_new_goal: Missing new_goal_string."
                    state.next_step = "interactive_query_executor"

            elif action_name == "refine_existing_graph_structure":
                refinement_instr = action_params.get(
                    "refinement_instructions_for_structure"
                )
                if (
                    refinement_instr
                    and state.execution_graph
                    and isinstance(state.execution_graph, GraphOutput)
                ):
                    state.graph_regeneration_reason = refinement_instr
                    state.scratchpad['refinement_validation_failures'] = 0
                    state = self.graph_generator.refine_api_graph(state)
                    action_result_message = (
                        f"Graph refinement (structure) started with instructions: {refinement_instr[:50]}..."
                    )
                elif not state.execution_graph or not isinstance(state.execution_graph, GraphOutput):
                    action_result_message = "Skipped refine_existing_graph_structure: No graph exists or invalid type."
                    state.next_step = "interactive_query_executor"
                else:
                    action_result_message = "Skipped refine_existing_graph_structure: Missing refinement_instructions_for_structure."
                    state.next_step = "interactive_query_executor"

            elif action_name == "answer_query_directly":
                query_to_answer = action_params.get(
                    "query_for_synthesizer", state.user_input or ""
                )
                original_user_input = state.user_input
                state.user_input = query_to_answer
                state = self.answer_openapi_query(state)
                state.user_input = original_user_input
                action_result_message = (
                    f"Direct answer generated for: {query_to_answer[:50]}..."
                )

            elif action_name == "setup_workflow_execution_interactive":
                state = self.setup_workflow_execution(state)
                action_result_message = (
                    f"Workflow execution setup initiated. Status: {state.workflow_execution_status}."
                )
                if idx + 1 < len(plan):
                     logger.warning("More actions planned after setup_workflow_execution_interactive. These will likely be skipped as setup routes to responder.")

            elif action_name == "resume_workflow_with_payload_interactive":
                confirmed_payload = action_params.get("confirmed_payload")
                if confirmed_payload and isinstance(confirmed_payload, dict):
                    state = self.resume_workflow_with_payload(state, confirmed_payload)
                    action_result_message = (
                        f"Workflow resumption with payload prepared. Status: {state.workflow_execution_status}."
                    )
                else:
                    action_result_message = "Skipped resume_workflow: Missing or invalid confirmed_payload in action_params."
                state.next_step = "responder"

            elif action_name == "synthesize_final_answer":
                synthesis_instr = action_params.get(
                    "synthesis_prompt_instructions",
                    "Summarize actions and provide a final response.",
                )
                current_action_summary = results[-1] if results and results[-1] != f"Action '{action_name}' completed." else action_result_message
                all_prior_results_summary = "; ".join(
                    [str(r)[:150] for r in results[:-1]] + [str(current_action_summary)[:150]]
                )
                final_synthesis_prompt = (
                    f"User's original query: '{state.user_input}'.\n"
                    f"My understanding of the query: '{state.scratchpad.get('user_query_understanding', 'N/A')}'.\n"
                    f"Internal actions taken and their results (summary): {all_prior_results_summary if all_prior_results_summary else 'No specific actions taken or results to summarize.'}\n"
                    f"Additional instructions for synthesis: {synthesis_instr}\n\n"
                    f"Based on all the above, formulate a comprehensive and helpful final answer for the user in Markdown format."
                )
                try:
                    final_answer = llm_call_helper(self.worker_llm, final_synthesis_prompt)
                    self._queue_intermediate_message(state, final_answer)
                    action_result_message = "Final answer synthesized."
                except Exception as e:
                    logger.error(f"Error synthesizing final answer: {e}")
                    self._queue_intermediate_message(
                        state,
                        f"Sorry, I encountered an error while synthesizing the final answer: {str(e)[:100]}",
                    )
                    action_result_message = "Error during final answer synthesis."
                state.next_step = "responder"
            else:
                action_result_message = f"Unknown or unhandled action: {action_name}."
                logger.warning(action_result_message)
                state.next_step = "interactive_query_executor"
        except Exception as e_action:
            logger.error(
                f"Error executing action '{action_name}': {e_action}", exc_info=True
            )
            action_result_message = (
                f"Error during action '{action_name}': {str(e_action)[:100]}..."
            )
            self._queue_intermediate_message(state, action_result_message)
            state.next_step = "interactive_query_executor"

        results.append(action_result_message)
        state.scratchpad["current_interactive_action_idx"] = idx + 1
        state.scratchpad["current_interactive_results"] = results

        if state.next_step == "interactive_query_executor":
            if state.scratchpad.get("current_interactive_action_idx", 0) >= len(plan):
                if action_name not in [
                    "synthesize_final_answer",
                    "answer_query_directly",
                    "setup_workflow_execution_interactive",
                    "resume_workflow_with_payload_interactive"
                ]:
                    logger.info(f"Interactive plan finished after action '{action_name}'. Finalizing with synthesis.")
                    all_results_summary_for_final_synth = "; ".join(
                        [str(r)[:100] + ('...' if len(str(r)) > 100 else '') for r in results]
                    )
                    final_synthesis_instr_auto = (
                        f"The user's query was: '{state.user_input}'. My understanding was: '{state.scratchpad.get('user_query_understanding', 'N/A')}'. "
                        f"The following internal actions were taken with these results: {all_results_summary_for_final_synth}. "
                        f"Please formulate a comprehensive final answer to the user based on these actions and results."
                    )
                    try:
                        final_answer_auto = llm_call_helper(self.worker_llm, final_synthesis_instr_auto)
                        self._queue_intermediate_message(state, final_answer_auto)
                    except Exception as e_synth:
                        logger.error(f"Error during final synthesis in interactive_query_executor: {e_synth}")
                        self._queue_intermediate_message(state, "Processed your request. " + (str(results[-1])[:100] if results else ""))
                state.next_step = "responder"
        return state

    def setup_workflow_execution(self, state: BotState) -> BotState:
        """Prepares the system to execute the current workflow graph."""
        tool_name = "setup_workflow_execution"
        logger.info(f"[{state.session_id}] Setting up workflow execution based on current graph.")
        state.update_scratchpad_reason(tool_name, "Preparing for workflow execution.")

        if not state.execution_graph or not isinstance(
            state.execution_graph, GraphOutput
        ):
            self._queue_intermediate_message(
                state,
                "No execution graph is available to run or graph is invalid. Please generate or load one first.",
            )
            state.workflow_execution_status = "failed"
            state.next_step = "responder"
            return state

        if state.workflow_execution_status in [
            "running",
            "paused_for_confirmation",
            "pending_start",
        ]:
            self._queue_intermediate_message(
                state,
                "A workflow is already running, paused, or pending. Please wait for it to complete or address the current state.",
            )
            state.next_step = "responder"
            return state

        try:
            state.workflow_execution_status = "pending_start"
            self._queue_intermediate_message(
                state,
                (
                    "Workflow execution has been prepared. The system will now attempt to start running the defined API calls. "
                    "You should receive updates on its progress shortly."
                ),
            )
            logger.info(
                f"[{state.session_id}] BotState prepared for workflow execution. Status set to 'pending_start'."
            )
        except Exception as e:
            logger.error(
                f"[{state.session_id}] Error during workflow setup preparation: {e}",
                exc_info=True,
            )
            self._queue_intermediate_message(
                state, f"Critical error preparing workflow execution: {str(e)[:150]}"
            )
            state.workflow_execution_status = "failed"

        state.next_step = "responder"
        return state

    def resume_workflow_with_payload(
        self, state: BotState, confirmed_payload: Dict[str, Any]
    ) -> BotState:
        """Handles data provided by the user to resume a paused workflow."""
        tool_name = "resume_workflow_with_payload"
        logger.info(
            f"[{state.session_id}] Preparing to resume workflow with confirmed_payload."
        )
        state.update_scratchpad_reason(
            tool_name,
            f"Payload received for workflow resumption: {str(confirmed_payload)[:100]}...",
        )

        if state.workflow_execution_status != "paused_for_confirmation":
            self._queue_intermediate_message(
                state,
                (
                    f"Workflow is not currently paused for confirmation (current status: {state.workflow_execution_status}). "
                    "Cannot process resume payload at this time."
                ),
            )
            state.next_step = "responder"
            return state
        
        state.scratchpad["pending_resume_payload_from_interactive_action"] = confirmed_payload
        state.workflow_execution_status = "running"
        self._queue_intermediate_message(
            state,
            "Confirmation payload received. System will attempt to resume workflow execution.",
        )
        logger.info(
            f"[{state.session_id}] Confirmed payload noted in scratchpad. Workflow status set to 'running' by Graph 1 "
            f"(pending actual resume by Graph 2 execution manager for its specific thread)."
        )

        state.next_step = "responder"
        return state
