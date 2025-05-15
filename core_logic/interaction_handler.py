# core_logic/interaction_handler.py
import logging
import json
import os
from typing import Any, Dict, Optional, List # Added List

# Assuming models are correctly imported from the parent directory or models.py
from models import BotState, GraphOutput # GraphOutput might be PlanSchema elsewhere
# Assuming utils are correctly imported
from utils import llm_call_helper # parse_llm_json_output_with_model might be needed for interactive_query_planner

# To avoid circular imports, InteractionHandler will call methods on instances
# of GraphGenerator and SpecProcessor passed to its constructor.
# from .graph_generator import GraphGenerator # Example if they were in the same package
# from .spec_processor import SpecProcessor

logger = logging.getLogger(__name__)

MAX_APIS_IN_PROMPT_SUMMARY_SHORT = int(os.getenv("MAX_APIS_IN_PROMPT_SUMMARY_SHORT", "10"))
MAX_SCHEMA_OVERVIEW_LENGTH = int(os.getenv("MAX_SCHEMA_OVERVIEW_LENGTH", "2000"))


class InteractionHandler:
    def __init__(
        self,
        worker_llm: Any,
        graph_generator_instance: Any, # Instance of GraphGenerator
        spec_processor_instance: Any,   # Instance of SpecProcessor
        api_executor_instance: Any      # Instance of APIExecutor
    ):
        if not hasattr(worker_llm, "invoke"):
            raise TypeError("worker_llm must have an 'invoke' method.")
        self.worker_llm = worker_llm
        self.graph_generator = graph_generator_instance
        self.spec_processor = spec_processor_instance
        self.api_executor = api_executor_instance # For context or future use
        logger.info("InteractionHandler initialized.")

    def _queue_intermediate_message(self, state: BotState, msg: str):
        if "intermediate_messages" not in state.scratchpad:
            state.scratchpad["intermediate_messages"] = []
        if (
            not state.scratchpad["intermediate_messages"]
            or state.scratchpad["intermediate_messages"][-1] != msg
        ):
            state.scratchpad["intermediate_messages"].append(msg)
        state.response = msg

    def describe_graph(self, state: BotState) -> BotState:
        tool_name = "describe_graph"
        self._queue_intermediate_message(state, "Preparing graph description...")
        state.update_scratchpad_reason(tool_name, "Preparing to describe the current execution graph.")

        if not state.execution_graph or not isinstance(state.execution_graph, GraphOutput):
            self._queue_intermediate_message(state, (state.response or "") + " No execution graph is currently available or graph is invalid.")
            logger.warning("describe_graph: No execution_graph found or invalid type.")
        else:
            graph_desc = state.execution_graph.description
            final_desc_for_user = ""
            if not graph_desc or len(graph_desc) < 20:
                logger.info("Graph description is short/missing, generating a dynamic one.")
                node_summaries = [f"- {node.effective_id}: {node.summary or node.operationId[:50]}" for node in state.execution_graph.nodes]
                nodes_str = "\n".join(node_summaries[:5])
                if len(node_summaries) > 5: nodes_str += f"\n- ... and {len(node_summaries) - 5} more nodes."
                prompt = (
                    f"The API execution graph for goal: '{state.plan_generation_goal or 'general use'}' includes:\n"
                    f"Nodes ({len(state.execution_graph.nodes)} total, sample):\n{nodes_str}\n\n"
                    f"Provide a concise, user-friendly natural language description of this workflow. Explain its purpose and general sequence. Use Markdown."
                )
                try:
                    dynamic_desc = llm_call_helper(self.worker_llm, prompt)
                    final_desc_for_user = f"**Current API Workflow for: '{state.plan_generation_goal or 'General Use'}'**\n\n{dynamic_desc}"
                    if graph_desc and graph_desc != dynamic_desc: final_desc_for_user += f"\n\n*Original AI-generated graph description: {graph_desc}*"
                except Exception as e:
                    logger.error(f"Error generating dynamic graph description: {e}")
                    default_node_preview = ', '.join([n.effective_id for n in state.execution_graph.nodes[:3]]) + ('...' if len(state.execution_graph.nodes) > 3 else '')
                    final_desc_for_user = f"**Current API Workflow for: '{state.plan_generation_goal or 'General Use'}'**\n\n{graph_desc or f'No detailed description. Nodes: {default_node_preview}'}"
            else:
                final_desc_for_user = f"**Current API Workflow for: '{state.plan_generation_goal or 'General Use'}'**\n\n{graph_desc}"
            if state.execution_graph.refinement_summary:
                final_desc_for_user += f"\n\n**Last Refinement Note:** {state.execution_graph.refinement_summary}"
            self._queue_intermediate_message(state, final_desc_for_user)
            if 'graph_to_send' not in state.scratchpad and state.execution_graph:
                 try: state.scratchpad['graph_to_send'] = state.execution_graph.model_dump_json(indent=2)
                 except Exception as e: logger.error(f"Error serializing graph for sending during describe_graph: {e}")
        state.update_scratchpad_reason(tool_name, f"Graph description generated/retrieved. Response set: {state.response[:100]}...")
        state.next_step = "responder"
        return state

    def get_graph_json(self, state: BotState) -> BotState:
        tool_name = "get_graph_json"
        self._queue_intermediate_message(state, "Fetching graph JSON...")
        state.update_scratchpad_reason(tool_name, "Attempting to provide graph JSON.")
        if not state.execution_graph or not isinstance(state.execution_graph, GraphOutput):
            self._queue_intermediate_message(state, "No execution graph is currently available or graph is invalid.")
        else:
            try:
                graph_json_str = state.execution_graph.model_dump_json(indent=2)
                state.scratchpad["graph_to_send"] = graph_json_str
                self._queue_intermediate_message(state, "Current API workflow graph (JSON) is available in the graph view.")
                logger.info("Provided graph JSON to scratchpad for UI.")
            except Exception as e:
                logger.error(f"Error serializing execution_graph to JSON: {e}")
                self._queue_intermediate_message(state, f"Error serializing graph to JSON: {str(e)}")
        state.next_step = "responder"
        return state

    def answer_openapi_query(self, state: BotState) -> BotState:
        tool_name = "answer_openapi_query"
        self._queue_intermediate_message(state, "Thinking about your question...")
        state.update_scratchpad_reason(tool_name, f"Attempting to answer user query: {state.user_input[:100] if state.user_input else 'N/A'}")
        if not state.openapi_schema and not (state.execution_graph and isinstance(state.execution_graph, GraphOutput)):
            self._queue_intermediate_message(state, "I don't have an OpenAPI spec or graph yet. Please provide one.")
            state.next_step = "responder"; return state

        context_parts = [f"User Question: \"{state.user_input}\""] if state.user_input else []
        if state.openapi_schema:
            try:
                limited_schema = {"info": state.openapi_schema.get("info", {}), "servers": state.openapi_schema.get("servers", [])}
                paths_overview = {p: {m: {"summary": op.get("summary", "N/A")[:70], "operationId": op.get("operationId", "N/A")} for m, op in pi.items() if isinstance(op, dict)} for p, pi in list(state.openapi_schema.get("paths", {}).items())[:5] if isinstance(pi, dict)}
                if paths_overview: limited_schema["paths_overview"] = paths_overview
                if len(state.openapi_schema.get("paths", {})) > 5: limited_schema["paths_overview"]["... (more paths)"] = "..."
                spec_detail_str = json.dumps(limited_schema, indent=2)
                if len(spec_detail_str) > MAX_SCHEMA_OVERVIEW_LENGTH: spec_detail_str = spec_detail_str[:MAX_SCHEMA_OVERVIEW_LENGTH] + "\n... (spec details truncated)"
                context_parts.append(f"\n### OpenAPI Spec Overview (JSON):\n```json\n{spec_detail_str}\n```")
            except Exception as e: logger.error(f"Error preparing openapi_schema detail for prompt: {e}"); context_parts.append("\n### OpenAPI Spec Overview:\nError preparing schema details.")
        if state.schema_summary: context_parts.append(f"\n### API Spec Summary (AI Generated):\n{state.schema_summary}")
        if state.identified_apis:
            apis_md = "\n### Identified API Operations (Sample):\n" + "".join([f"- **OpID:** `{api.get('operationId', 'N/A')}` (`{api.get('method', '?')} {api.get('path', '?')}`) _{api.get('summary', 'No summary')[:70]}..._\n" for api in state.identified_apis[:MAX_APIS_IN_PROMPT_SUMMARY_SHORT]])
            if len(state.identified_apis) > MAX_APIS_IN_PROMPT_SUMMARY_SHORT: apis_md += f"- ... and {len(state.identified_apis) - MAX_APIS_IN_PROMPT_SUMMARY_SHORT} more.\n"
            context_parts.append(apis_md)
        if state.execution_graph and isinstance(state.execution_graph, GraphOutput) and state.execution_graph.description:
            graph_desc_md = f"\n### Current Workflow Graph ('{state.plan_generation_goal or 'General'}') Description:\n{state.execution_graph.description}"
            if state.execution_graph.refinement_summary: graph_desc_md += f"\nLast Refinement: {state.execution_graph.refinement_summary}"
            context_parts.append(graph_desc_md)
        if state.payload_descriptions:
            payload_md = "\n### Payload/Response Examples (AI Generated, Sample):\n" + "".join([f"**OpID `{op_id}`:**\n```text\n{desc_text[:200]}...\n```\n" for op_id, desc_text in list(state.payload_descriptions.items())[:2]])
            if len(state.payload_descriptions) > 2: payload_md += "... and more examples exist.\n"
            context_parts.append(payload_md)

        full_context = "\n".join(context_parts) or "No specific API context available."
        prompt = f"""Context:\n{full_context}\n\nUser Question: "{state.user_input}"\n\nAnswer the user's question based ONLY on the provided context. Be concise. Use Markdown for formatting. If info is unavailable, state that.
        Answer:"""
        try:
            answer = llm_call_helper(self.worker_llm, prompt)
            self._queue_intermediate_message(state, answer)
            logger.info("Successfully generated answer for OpenAPI query.")
        except Exception as e:
            logger.error(f"Error generating answer for OpenAPI query: {e}", exc_info=False)
            self._queue_intermediate_message(state, f"### Error Answering Query\nSorry, an error occurred: {str(e)[:100]}...")
        state.update_scratchpad_reason(tool_name, f"Answered query. Response snippet: {state.response[:100]}...")
        state.next_step = "responder"
        return state

    async def handle_initiate_load_test(self, state: BotState) -> BotState:
        """
        Handles the 'initiate_load_test' intent.
        Sets up the state for the WebSocket handler to orchestrate the load test.
        """
        tool_name = "handle_initiate_load_test"
        self._queue_intermediate_message(state, "Preparing for load test execution...")
        state.update_scratchpad_reason(tool_name, "Handling 'initiate_load_test' intent from router.")

        if not state.execution_graph or not isinstance(state.execution_graph, GraphOutput):
            self._queue_intermediate_message(
                state,
                "Cannot start load test: No execution graph is available. Please generate or load a plan first.",
            )
            state.workflow_execution_status = "failed" # Indicate failure to set up
            state.next_step = "responder"
            return state

        num_users = state.extracted_params.get("num_users", 1) # Default to 1 if not found
        if not isinstance(num_users, int) or num_users <= 0:
            logger.warning(f"Invalid number of users for load test: {num_users}. Defaulting to 1.")
            num_users = 1
            self._queue_intermediate_message(state, f"Warning: Invalid number of users for load test. Defaulting to {num_users} user(s).")


        # Store load test parameters in scratchpad for the WebSocket handler
        state.scratchpad["load_test_config"] = {
            "num_concurrent_users": num_users,
            # "duration_minutes": duration_minutes, # Placeholder for future duration logic
            "disable_confirmation_prompts": True    # Key setting for load tests
        }
        # Set a specific status that the WebSocket handler will look for
        state.workflow_execution_status = "pending_load_test"

        self._queue_intermediate_message(
            state,
            f"Load test initiated for {num_users} concurrent virtual user(s). "
            f"During the test, payload confirmations will be automatically bypassed. "
            f"The system will now attempt to start the test execution."
        )
        logger.info(
            f"[{state.session_id}] BotState prepared for load test. Config: {state.scratchpad['load_test_config']}. "
            f"Status set to '{state.workflow_execution_status}'."
        )

        # Graph 1's role for this request is done. The WebSocket handler will take over.
        state.next_step = "responder"
        return state

    def interactive_query_planner(self, state: BotState) -> BotState:
        # This method would need to be async if any of its sub-actions become async
        tool_name = "interactive_query_planner"
        self._queue_intermediate_message(state, "Planning how to address your interactive query...")
        state.update_scratchpad_reason(tool_name, f"Entering interactive query planner for input: {state.user_input[:100] if state.user_input else 'N/A'}")
        state.scratchpad.pop("interactive_action_plan", None); state.scratchpad.pop("current_interactive_action_idx", None); state.scratchpad.pop("current_interactive_results", None)

        graph_summary = (state.execution_graph.description[:150] + "..." if state.execution_graph and isinstance(state.execution_graph, GraphOutput) and state.execution_graph.description else "No graph currently generated.")
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
1.  `rerun_payload_generation`: Regenerate payload/response examples for specific APIs. Params: {{ "operation_ids_to_update": ["opId1"], "new_context": "User context" }}
2.  `contextualize_graph_descriptions`: Rewrite descriptions within the *existing* graph. Params: {{ "new_context_for_graph": "User context" }}
3.  `regenerate_graph_with_new_goal`: Create a *new* graph for a different goal or major structural change. Params: {{ "new_goal_string": "User's new goal" }}
4.  `refine_existing_graph_structure`: Minor structural adjustments to the existing graph. Params: {{ "refinement_instructions_for_structure": "User feedback" }}
5.  `answer_query_directly`: Answer using existing information. Params: {{ "query_for_synthesizer": "Original user query" }}
6.  `setup_workflow_execution_interactive`: If user asks to run/execute current graph. Params: {{ "initial_parameters": {{"param1": "value1"}} }}
7.  `resume_workflow_with_payload_interactive`: If workflow is paused and user provides data. Params: {{ "confirmed_payload": {{...}} }}
8.  `synthesize_final_answer`: Formulate a final answer based on previous actions. Params: {{ "synthesis_prompt_instructions": "Instructions for LLM" }}

Task:
1. Analyze user query and current state.
2. Create "interactive_action_plan" (list of action objects, max 3-4 steps).
3. Provide "user_query_understanding" (1-2 sentences).
Output ONLY a JSON object: {{"user_query_understanding": "...", "interactive_action_plan": [{{"action_name": "...", "action_params": {{...}}, "description": "..."}}]}}
If ambiguous, use "synthesize_final_answer" to inform user."""
        try:
            # Assuming parse_llm_json_output_with_model is available or imported
            from utils import parse_llm_json_output_with_model
            llm_response_str = llm_call_helper(self.worker_llm, prompt)
            parsed_plan_data = parse_llm_json_output_with_model(llm_response_str) # This function needs to exist

            if parsed_plan_data and isinstance(parsed_plan_data, dict) and "interactive_action_plan" in parsed_plan_data and isinstance(parsed_plan_data["interactive_action_plan"], list) and "user_query_understanding" in parsed_plan_data:
                state.scratchpad["user_query_understanding"] = parsed_plan_data["user_query_understanding"]
                state.scratchpad["interactive_action_plan"] = parsed_plan_data["interactive_action_plan"]
                state.scratchpad["current_interactive_action_idx"] = 0
                state.scratchpad["current_interactive_results"] = []
                self._queue_intermediate_message(state, f"Understood query: {state.scratchpad['user_query_understanding']}. Starting internal actions...")
                logger.info(f"Interactive plan generated: {state.scratchpad['interactive_action_plan']}")
                state.next_step = "interactive_query_executor"
            else:
                logger.error(f"LLM failed to produce a valid interactive plan. Raw: {llm_response_str[:300]}")
                raise ValueError("LLM failed to produce a valid interactive plan JSON structure.")
        except Exception as e:
            logger.error(f"Error in interactive_query_planner: {e}", exc_info=False)
            self._queue_intermediate_message(state, f"Sorry, error planning how to address your request: {str(e)[:100]}...")
            state.next_step = "answer_openapi_query" # Fallback to direct answer
        state.update_scratchpad_reason(tool_name, f"Interactive plan generated. Next: {state.next_step}. Response: {state.response[:100]}")
        return state

    def _internal_contextualize_graph_descriptions(self, state: BotState, new_context: str) -> str:
        tool_name = "_internal_contextualize_graph_descriptions"
        if not state.execution_graph or not isinstance(state.execution_graph, GraphOutput): return "No graph to contextualize or invalid."
        if not new_context: return "No new context provided."
        logger.info(f"Attempting to contextualize graph descriptions with context: {new_context[:100]}...")
        original_graph_desc = state.execution_graph.description
        if state.execution_graph.description:
            prompt_overall = (f"Current graph desc: \"{state.execution_graph.description}\"\nNew User Context: \"{new_context}\"\nRewrite graph desc. Output only new desc.")
            try: state.execution_graph.description = llm_call_helper(self.worker_llm, prompt_overall); logger.info(f"Overall graph desc contextualized: {state.execution_graph.description[:100]}...")
            except Exception as e: logger.error(f"Error contextualizing overall graph desc: {e}"); state.execution_graph.description = original_graph_desc
        nodes_to_update = [n for n in state.execution_graph.nodes if n.operationId not in ["START_NODE", "END_NODE"]][:3]
        for node in nodes_to_update:
            original_node_desc = node.description
            if node.description:
                prompt_node = (f"Node '{node.effective_id}' ({node.summary}) desc: \"{node.description}\"\nOverall User Context: \"{new_context}\"\nRewrite node desc. Output only new desc.")
                try: node.description = llm_call_helper(self.worker_llm, prompt_node); logger.info(f"Node '{node.effective_id}' desc contextualized: {node.description[:100]}...")
                except Exception as e: logger.error(f"Error contextualizing node '{node.effective_id}' desc: {e}"); node.description = original_node_desc
        if state.execution_graph: state.scratchpad["graph_to_send"] = state.execution_graph.model_dump_json(indent=2)
        state.update_scratchpad_reason(tool_name, f"Graph descriptions contextualized with context: {new_context[:70]}.")
        return f"Graph descriptions updated for context: '{new_context[:70]}...'."

    async def interactive_query_executor(self, state: BotState) -> BotState: # Made async
        tool_name = "interactive_query_executor"
        plan = state.scratchpad.get("interactive_action_plan", [])
        idx = state.scratchpad.get("current_interactive_action_idx", 0)
        results = state.scratchpad.get("current_interactive_results", [])

        if not plan or idx >= len(plan):
            final_response_message = "Finished interactive processing. "
            if results: final_response_message += (str(results[-1])[:200] + "..." if len(str(results[-1])) > 200 else str(results[-1]))
            else: final_response_message += "No specific actions taken or results to report."
            if not state.response or state.response.startswith("Executing internal step"): self._queue_intermediate_message(state, final_response_message)
            logger.info("Interactive plan execution completed or no plan."); state.next_step = "responder"
            state.update_scratchpad_reason(tool_name, "Interactive plan execution completed or no plan."); return state

        action = plan[idx]; action_name = action.get("action_name"); action_params = action.get("action_params", {}); action_description = action.get("description", "No description.")
        self._queue_intermediate_message(state, f"Executing internal step ({idx + 1}/{len(plan)}): {action_description[:70]}...")
        state.update_scratchpad_reason(tool_name, f"Executing action ({idx + 1}/{len(plan)}): {action_name} - {action_description}")
        action_result_message = f"Action '{action_name}' completed."

        try:
            if action_name == "rerun_payload_generation":
                op_ids = action_params.get("operation_ids_to_update", []); new_ctx = action_params.get("new_context", "")
                if op_ids and isinstance(op_ids, list): # new_ctx can be empty
                    # This needs to be awaited if _generate_payload_descriptions_parallel is async
                    await self.spec_processor._generate_payload_descriptions_parallel(state, target_apis=op_ids, context_override=new_ctx if new_ctx else None)
                    action_result_message = f"Payload examples update requested for {op_ids} with context '{new_ctx[:30]}...'."
                else: action_result_message = "Skipped rerun_payload_generation: Missing operation_ids or invalid format."
                state.next_step = "interactive_query_executor"
            elif action_name == "contextualize_graph_descriptions":
                new_ctx_graph = action_params.get("new_context_for_graph", "")
                if new_ctx_graph: action_result_message = self._internal_contextualize_graph_descriptions(state, new_ctx_graph)
                else: action_result_message = "Skipped contextualize_graph_descriptions: Missing new_context_for_graph."
                state.next_step = "interactive_query_executor"
            elif action_name == "regenerate_graph_with_new_goal":
                new_goal = action_params.get("new_goal_string")
                if new_goal:
                    state.plan_generation_goal = new_goal; state.execution_graph = None; state.graph_refinement_iterations = 0; state.scratchpad['graph_gen_attempts'] = 0; state.scratchpad['refinement_validation_failures'] = 0
                    state = self.graph_generator._generate_execution_graph(state, goal=new_goal) # This is synchronous
                    action_result_message = f"Graph regeneration started for new goal: {new_goal[:50]}..."
                    # _generate_execution_graph sets its own next_step, usually to verify_graph
                else: action_result_message = "Skipped regenerate_graph_with_new_goal: Missing new_goal_string."; state.next_step = "interactive_query_executor"
            elif action_name == "refine_existing_graph_structure":
                refinement_instr = action_params.get("refinement_instructions_for_structure")
                if refinement_instr and state.execution_graph and isinstance(state.execution_graph, GraphOutput):
                    state.graph_regeneration_reason = refinement_instr; state.scratchpad['refinement_validation_failures'] = 0
                    state = self.graph_generator.refine_api_graph(state) # This is synchronous
                    action_result_message = f"Graph refinement (structure) started with instructions: {refinement_instr[:50]}..."
                    # refine_api_graph sets its own next_step
                elif not state.execution_graph or not isinstance(state.execution_graph, GraphOutput): action_result_message = "Skipped refine_existing_graph_structure: No graph exists."; state.next_step = "interactive_query_executor"
                else: action_result_message = "Skipped refine_existing_graph_structure: Missing refinement_instructions."; state.next_step = "interactive_query_executor"
            elif action_name == "answer_query_directly":
                query_to_answer = action_params.get("query_for_synthesizer", state.user_input or ""); original_user_input = state.user_input; state.user_input = query_to_answer
                state = self.answer_openapi_query(state); state.user_input = original_user_input # answer_openapi_query is sync and sets next_step to responder
                action_result_message = f"Direct answer generated for: {query_to_answer[:50]}..."
            elif action_name == "setup_workflow_execution_interactive":
                state = self.setup_workflow_execution(state) # This is synchronous
                action_result_message = f"Workflow execution setup initiated. Status: {state.workflow_execution_status}."
                # setup_workflow_execution sets next_step to responder
            elif action_name == "resume_workflow_with_payload_interactive":
                # This path might not be hit if resume_exec command is handled directly by websocket_helpers
                confirmed_payload = action_params.get("confirmed_payload")
                if confirmed_payload and isinstance(confirmed_payload, dict):
                    state = self.resume_workflow_with_payload(state, confirmed_payload) # This is synchronous
                    action_result_message = f"Workflow resumption with payload prepared. Status: {state.workflow_execution_status}."
                else: action_result_message = "Skipped resume_workflow: Missing or invalid confirmed_payload."
                state.next_step = "responder" # resume_workflow_with_payload sets next_step
            elif action_name == "synthesize_final_answer":
                synthesis_instr = action_params.get("synthesis_prompt_instructions", "Summarize actions and provide a final response.")
                current_action_summary = results[-1] if results and results[-1] != f"Action '{action_name}' completed." else action_result_message
                all_prior_results_summary = "; ".join([str(r)[:150] for r in results[:-1]] + [str(current_action_summary)[:150]])
                final_synthesis_prompt = (f"User query: '{state.user_input}'.\nUnderstanding: '{state.scratchpad.get('user_query_understanding', 'N/A')}'.\nActions & Results: {all_prior_results_summary or 'N/A'}\nInstructions: {synthesis_instr}\nFormulate a comprehensive final answer.")
                try: final_answer = llm_call_helper(self.worker_llm, final_synthesis_prompt); self._queue_intermediate_message(state, final_answer); action_result_message = "Final answer synthesized."
                except Exception as e: logger.error(f"Error synthesizing final answer: {e}"); self._queue_intermediate_message(state, f"Sorry, error synthesizing final answer: {str(e)[:100]}"); action_result_message = "Error during final answer synthesis."
                state.next_step = "responder"
            else:
                action_result_message = f"Unknown or unhandled action: {action_name}."; logger.warning(action_result_message)
                state.next_step = "interactive_query_executor" # Loop back to try next action or finish
        except Exception as e_action:
            logger.error(f"Error executing action '{action_name}': {e_action}", exc_info=True)
            action_result_message = f"Error during action '{action_name}': {str(e_action)[:100]}..."
            self._queue_intermediate_message(state, action_result_message)
            state.next_step = "interactive_query_executor" # Try next action or finish

        results.append(action_result_message)
        state.scratchpad["current_interactive_action_idx"] = idx + 1
        state.scratchpad["current_interactive_results"] = results

        # If the current action didn't set a specific next_step (like _generate_execution_graph does),
        # and we are still in 'interactive_query_executor', check if plan is done.
        if state.next_step == "interactive_query_executor":
            if state.scratchpad.get("current_interactive_action_idx", 0) >= len(plan): # Plan finished
                if action_name not in ["synthesize_final_answer", "answer_query_directly", "setup_workflow_execution_interactive", "resume_workflow_with_payload_interactive"]:
                    logger.info(f"Interactive plan finished after action '{action_name}'. Finalizing with synthesis.")
                    all_results_summary_for_final_synth = "; ".join([str(r)[:100] + ('...' if len(str(r)) > 100 else '') for r in results])
                    final_synthesis_instr_auto = (f"User query: '{state.user_input}'. Understanding: '{state.scratchpad.get('user_query_understanding', 'N/A')}'. Actions & Results: {all_results_summary_for_final_synth}. Formulate a final answer.")
                    try: final_answer_auto = llm_call_helper(self.worker_llm, final_synthesis_instr_auto); self._queue_intermediate_message(state, final_answer_auto)
                    except Exception as e_synth: logger.error(f"Error during final synthesis in executor: {e_synth}"); self._queue_intermediate_message(state, "Processed your request. " + (str(results[-1])[:100] if results else ""))
                state.next_step = "responder"
        return state

    def setup_workflow_execution(self, state: BotState) -> BotState:
        tool_name = "setup_workflow_execution"
        logger.info(f"[{state.session_id}] Setting up workflow execution based on current graph.")
        state.update_scratchpad_reason(tool_name, "Preparing for workflow execution.")
        if not state.execution_graph or not isinstance(state.execution_graph, GraphOutput):
            self._queue_intermediate_message(state, "No execution graph available to run or graph is invalid. Please generate one first.")
            state.workflow_execution_status = "failed"; state.next_step = "responder"; return state
        if state.workflow_execution_status in ["running", "paused_for_confirmation", "pending_start"]:
            self._queue_intermediate_message(state, "A workflow is already running, paused, or pending. Please wait or address current state.")
            state.next_step = "responder"; return state
        try:
            state.workflow_execution_status = "pending_start" # This status will be picked up by websocket_helpers
            self._queue_intermediate_message(state, "Workflow execution prepared. System will attempt to start running API calls. Updates will follow.")
            logger.info(f"[{state.session_id}] BotState prepared for workflow execution. Status set to 'pending_start'.")
        except Exception as e:
            logger.error(f"[{state.session_id}] Error during workflow setup preparation: {e}", exc_info=True)
            self._queue_intermediate_message(state, f"Critical error preparing workflow execution: {str(e)[:150]}")
            state.workflow_execution_status = "failed"
        state.next_step = "responder"
        return state

    def resume_workflow_with_payload(self, state: BotState, confirmed_payload: Dict[str, Any]) -> BotState:
        tool_name = "resume_workflow_with_payload"
        logger.info(f"[{state.session_id}] Preparing to resume workflow with confirmed_payload.")
        state.update_scratchpad_reason(tool_name, f"Payload received for workflow resumption: {str(confirmed_payload)[:100]}...")
        if state.workflow_execution_status != "paused_for_confirmation":
            self._queue_intermediate_message(state, f"Workflow not paused for confirmation (status: {state.workflow_execution_status}). Cannot process resume.")
            state.next_step = "responder"; return state
        state.scratchpad["pending_resume_payload_from_interactive_action"] = confirmed_payload
        state.workflow_execution_status = "running" # Update G1's view
        self._queue_intermediate_message(state, "Confirmation payload received. System will attempt to resume workflow execution.")
        logger.info(f"[{state.session_id}] Confirmed payload noted in scratchpad. Workflow status set to 'running' by G1.")
        state.next_step = "responder"
        return state

