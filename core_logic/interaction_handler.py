# core_logic/interaction_handler.py
import logging
from typing import Any, Dict, Optional

from models import BotState, GraphOutput # GraphOutput for type hints
from utils import llm_call_helper, parse_llm_json_output_with_model # parse_llm_json_output_with_model for interactive_query_planner
import os 

logger = logging.getLogger(__name__)

# Configurable Limits
MAX_APIS_IN_PROMPT_SUMMARY_SHORT = int(os.getenv("MAX_APIS_IN_PROMPT_SUMMARY_SHORT", "10"))

class InteractionHandler:
    def __init__(self, worker_llm: Any, graph_generator_instance: Any, workflow_control_instance: Any, spec_processor_instance: Any):
        """
        Initializes the InteractionHandler.
        
        Args:
            worker_llm: The language model for performing tasks.
            graph_generator_instance: An instance of GraphGenerator for graph-related actions.
            workflow_control_instance: An instance of WorkflowLifecycleManager for workflow actions.
            spec_processor_instance: An instance of SpecProcessor for spec-related actions.
        """
        self.worker_llm = worker_llm
        self.graph_generator = graph_generator_instance
        self.workflow_control = workflow_control_instance
        self.spec_processor = spec_processor_instance
        logger.info("InteractionHandler initialized.")

    def describe_graph(self, state: BotState) -> BotState:
        tool_name = "describe_graph"
        state.response = "Preparing graph description..."
        state.update_scratchpad_reason(tool_name, "Preparing to describe the current execution graph.")

        if not state.execution_graph or not isinstance(state.execution_graph, GraphOutput):
            state.response = (state.response or "") + " No execution graph is currently available to describe or graph is invalid."
            logger.warning("describe_graph: No execution_graph found in state or invalid type.")
        else:
            graph_desc = state.execution_graph.description
            if not graph_desc or len(graph_desc) < 20: 
                logger.info("Graph description is short or missing, generating a dynamic one.")
                node_summaries = []
                for node in state.execution_graph.nodes:
                    node_summaries.append(f"- {node.effective_id}: {node.summary or node.operationId[:50]}") 
                
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
            
            state.response = final_desc_for_user
            if 'graph_to_send' not in state.scratchpad and state.execution_graph:
                 try:
                     state.scratchpad['graph_to_send'] = state.execution_graph.model_dump_json(indent=2)
                 except Exception as e:
                     logger.error(f"Error serializing graph for sending during describe_graph: {e}")

        state.update_scratchpad_reason(tool_name, f"Graph description generated/retrieved. Response set: {state.response[:100]}...")
        state.next_step = "responder"
        return state

    def get_graph_json(self, state: BotState) -> BotState:
        tool_name = "get_graph_json"
        state.response = "Fetching graph JSON..."
        state.update_scratchpad_reason(tool_name, "Attempting to provide graph JSON.")

        if not state.execution_graph or not isinstance(state.execution_graph, GraphOutput):
            state.response = "No execution graph is currently available or graph is invalid."
        else:
            try:
                graph_json_str = state.execution_graph.model_dump_json(indent=2)
                state.scratchpad['graph_to_send'] = graph_json_str
                state.response = "The current API workflow graph is available in the graph view. You can also copy the JSON from there if needed."
                logger.info("Provided graph JSON to scratchpad for UI.")
            except Exception as e:
                logger.error(f"Error serializing execution_graph to JSON: {e}")
                state.response = f"Error serializing graph to JSON: {str(e)}"
        
        state.next_step = "responder"
        return state

    def answer_openapi_query(self, state: BotState) -> BotState:
        tool_name = "answer_openapi_query"
        state.response = "Thinking about your question..."
        state.update_scratchpad_reason(tool_name, f"Attempting to answer user query: {state.user_input[:100] if state.user_input else 'N/A'}")

        if not state.openapi_schema and not (state.execution_graph and isinstance(state.execution_graph, GraphOutput)):
            state.response = "I don't have an OpenAPI specification loaded or a graph generated yet. Please provide one first."
            state.next_step = "responder"
            return state

        context_parts = []
        if state.user_input:
            context_parts.append(f"User Question: \"{state.user_input}\"")

        if state.schema_summary:
            context_parts.append(f"\n### API Specification Summary (from validated schema)\n{state.schema_summary}")

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
                identified_apis_md += f"- ... and {len(state.identified_apis) - num_apis_to_list} more.\n"
        else:
            identified_apis_md += "No specific API operations identified yet.\n"
        context_parts.append(identified_apis_md)

        if state.execution_graph and isinstance(state.execution_graph, GraphOutput) and state.execution_graph.description:
            graph_desc_md = f"\n### Current Workflow Graph ('{state.plan_generation_goal or 'General Purpose'}') Description:\n{state.execution_graph.description}"
            if state.execution_graph.refinement_summary:
                graph_desc_md += f"\nLast Refinement: {state.execution_graph.refinement_summary}"
            context_parts.append(graph_desc_md)
        elif state.execution_graph and isinstance(state.execution_graph, GraphOutput):
            context_parts.append(f"\n### Current Workflow Graph ('{state.plan_generation_goal or 'General Purpose'}') exists but has no detailed description.")

        payload_descriptions_md = "\n### Available Payload/Response Examples (for reference):\n"
        if state.payload_descriptions:
            for op_id, desc_text in state.payload_descriptions.items():
                payload_descriptions_md += f"**For Operation ID `{op_id}`:**\n```text\n{desc_text[:300]}...\n```\n\n" 
        else:
            payload_descriptions_md += "No detailed payload/response examples have been generated yet.\n"
        context_parts.append(payload_descriptions_md)
        
        full_context = "\n".join(context_parts)
        if not full_context.strip():
            full_context = "No specific API context available, but an OpenAPI spec might be loaded."

        prompt = f"""You are an expert API assistant. Your task is to answer the user's question based on the provided context. Use Markdown for formatting.

        Context:
        {full_context}

        User Question: "{state.user_input}"

        **Instructions for Answering:**
        1.  **Understand the Question:** Determine if the user is asking for general information, details about a specific API, or "how-to" perform an action (e.g., "how to create a product").
        2.  **For "How-To" Questions (e.g., "how to create X", "how do I update Y?"):**
            a.  Identify the most relevant API operation(s) from the "Identified API Operations" list that would achieve the user's goal (e.g., a POST to `/products` for "create product").
            b.  State the identified operation: its Operation ID, Method, and Path.
            c.  Refer to the "Available Payload/Response Examples" for the identified Operation ID. Describe the necessary request body structure based on its example. Mention key fields and their expected data types or example values.
            d.  List any important path or query parameters.
            e.  Briefly describe the expected successful response.
            f.  If multiple operations seem relevant, briefly mention them.
        3.  **For Questions about Specific APIs (e.g., "what does 'getUser' do?"):**
            a.  Find the API in the "Identified API Operations" list by its Operation ID or path.
            b.  Provide its summary, method, path.
            c.  If available, use its "Payload/Response Example" to describe its request/response.
        4.  **For General Questions:** Use the API Specification Summary and other relevant context.
        5.  **Clarity and Conciseness:** Provide a clear, concise, and helpful answer.
        6.  **Formatting:** Use Markdown (headings, lists, bolding, code blocks for JSON examples or API paths).
        7.  **Unavailable Information:** If the information is not available in the context, state that clearly. Do not invent details.
        8.  **No Conversational Fluff:** Focus only on answering the question directly.

        Answer:
        """
        try:
            state.response = llm_call_helper(self.worker_llm, prompt)
            logger.info("Successfully generated answer for OpenAPI query.")
        except Exception as e:
            logger.error(f"Error generating answer for OpenAPI query: {e}", exc_info=False)
            state.response = f"### Error Answering Query\nSorry, I encountered an error while trying to answer your question: {str(e)[:100]}..."
        
        state.update_scratchpad_reason(tool_name, f"Answered query. Response snippet: {state.response[:100]}...")
        state.next_step = "responder"
        return state

    def interactive_query_planner(self, state: BotState) -> BotState:
        tool_name = "interactive_query_planner"
        state.response = "Planning how to address your interactive query..."
        state.update_scratchpad_reason(tool_name, f"Entering interactive query planner for input: {state.user_input[:100] if state.user_input else 'N/A'}")
        
        state.scratchpad.pop('interactive_action_plan', None)
        state.scratchpad.pop('current_interactive_action_idx', None)
        state.scratchpad.pop('current_interactive_results', None)
        
        graph_summary = state.execution_graph.description[:150] + "..." if state.execution_graph and isinstance(state.execution_graph, GraphOutput) and state.execution_graph.description else "No graph currently generated."
        payload_keys_sample = list(state.payload_descriptions.keys())[:3]
        
        prompt = f"""User Query: "{state.user_input}"\n\nCurrent State Context:\n- API Spec Summary: {'Available (from validated spec)' if state.schema_summary else 'Not available.'}\n- Identified APIs count: {len(state.identified_apis) if state.identified_apis else 0}. Example OpIDs: {", ".join([api['operationId'] for api in state.identified_apis[:3]])}...\n- Example Payload Descriptions available for OpIDs (sample, from validated spec): {payload_keys_sample}...\n- Current Execution Graph Goal: {state.plan_generation_goal or 'Not set.'}\n- Current Graph Description: {graph_summary}\n- Workflow Execution Status: {state.workflow_execution_status}\n\nAvailable Internal Actions (choose one or more in sequence, output as a JSON list):\n1.  `rerun_payload_generation`: Regenerate payload/response examples for specific APIs, possibly with new user-provided context.\n    Params: {{ "operation_ids_to_update": ["opId1", "opId2"], "new_context": "User's new context string for generation" }}\n2.  `contextualize_graph_descriptions`: Rewrite descriptions within the *existing* graph (overall, nodes, edges) to reflect new user context or focus. This does NOT change graph structure.\n    Params: {{ "new_context_for_graph": "User's new context/focus for descriptions" }}\n3.  `regenerate_graph_with_new_goal`: Create a *new* graph if the user states a completely different high-level goal OR requests a significant structural change (add/remove/reorder API steps).\n    Params: {{ "new_goal_string": "User's new goal, incorporating the structural change (e.g., 'Workflow to X, then Y, and then Z as the last step')" }}\n4.  `refine_existing_graph_structure`: For minor structural adjustments to the existing graph (e.g., "add API Z after Y but before END_NODE", "remove API X"). This implies the overall goal is similar but the sequence/nodes need adjustment. The LLM will be asked to refine the current graph JSON.\n    Params: {{ "refinement_instructions_for_structure": "User's specific feedback for structural refinement (e.g., 'Add operation Z after Y', 'Ensure X comes before Y')" }}\n5.  `answer_query_directly`: If the query can be answered using existing information (API summary, API list, current graph description, existing payload examples) without modifications to artifacts.\n    Params: {{ "query_for_synthesizer": "The original user query or a rephrased one for direct answering." }}\n6.  `setup_workflow_execution_interactive`: If the user asks to run/execute the current graph. This action prepares the system for execution.\n    Params: {{ "initial_parameters": {{ "param1": "value1" }} }} (Optional initial parameters for the workflow, if provided by user)\n7.  `resume_workflow_with_payload_interactive`: If the workflow is 'paused_for_confirmation' and the user provides the necessary payload/confirmation to continue.\n    Params: {{ "confirmed_payload": {{...}} }} (The JSON payload confirmed or provided by the user)\n8.  `synthesize_final_answer`: (Usually the last step of a plan) Formulate a comprehensive answer to the user based on the outcomes of previous internal actions or if no other action is suitable.\n    Params: {{ "synthesis_prompt_instructions": "Instructions for the LLM on what to include in the final answer, summarizing actions taken or information gathered." }}\n\nTask:\n1. Analyze the user's query in the context of the current system state.\n2. Create a short, logical "interactive_action_plan" (a list of action objects, max 3-4 steps).\n   - For requests to run the graph, use `setup_workflow_execution_interactive`.\n   - If the graph is paused and user provides data, use `resume_workflow_with_payload_interactive`.\n   - For structural changes like "add X at the end", prefer `regenerate_graph_with_new_goal` or `refine_existing_graph_structure`.\n3. Provide a brief "user_query_understanding" (1-2 sentences).\n\nOutput ONLY a JSON object with this structure:\n{{\n  "user_query_understanding": "Brief interpretation of user's need.",\n  "interactive_action_plan": [\n    {{"action_name": "action_enum_value", "action_params": {{...}}, "description": "Briefly, why this action is chosen."}}\n  ]\n}}\nIf the query is very simple and can be answered directly, the plan might just be one "answer_query_directly" or "synthesize_final_answer" action.\nIf the query is ambiguous or cannot be handled by available actions, use "synthesize_final_answer" with instructions to inform the user."""
        try:
            llm_response_str = llm_call_helper(self.worker_llm, prompt)
            parsed_plan_data = parse_llm_json_output_with_model(llm_response_str) # This utility handles JSON parsing
            if parsed_plan_data and isinstance(parsed_plan_data, dict) and \
               "interactive_action_plan" in parsed_plan_data and \
               isinstance(parsed_plan_data["interactive_action_plan"], list) and \
               "user_query_understanding" in parsed_plan_data:
                state.scratchpad['user_query_understanding'] = parsed_plan_data["user_query_understanding"]
                state.scratchpad['interactive_action_plan'] = parsed_plan_data["interactive_action_plan"]
                state.scratchpad['current_interactive_action_idx'] = 0
                state.scratchpad['current_interactive_results'] = [] 
                state.response = f"Understood query: {state.scratchpad['user_query_understanding']}. Starting internal actions..."
                logger.info(f"Interactive plan generated: {state.scratchpad['interactive_action_plan']}")
                state.next_step = "interactive_query_executor"
            else:
                logger.error(f"LLM failed to produce a valid interactive plan. Raw: {llm_response_str[:300]}")
                raise ValueError("LLM failed to produce a valid interactive plan JSON structure with required keys.")
        except Exception as e:
            logger.error(f"Error in interactive_query_planner: {e}", exc_info=False)
            state.response = f"Sorry, I encountered an error while planning how to address your request: {str(e)[:100]}..."
            state.next_step = "answer_openapi_query" # Fallback to trying a direct answer
        
        state.update_scratchpad_reason(tool_name, f"Interactive plan generated. Next: {state.next_step}. Response: {state.response[:100]}")
        return state

    def _internal_contextualize_graph_descriptions(self, state: BotState, new_context: str) -> str:
        tool_name = "_internal_contextual
