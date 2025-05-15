# router.py
import logging
import re
from typing import Dict, Any, Literal, Optional
import json
import yaml

from models import BotState # Ensure BotState is imported
# from utils import llm_call_helper # If used for LLM-based routing

logger = logging.getLogger(__name__)

class OpenAPIRouter:
    AVAILABLE_INTENTS = Literal[
        "parse_openapi_spec",
        "process_schema_pipeline",
        "verify_graph",
        "_generate_execution_graph",
        "refine_api_graph",
        "describe_graph",
        "get_graph_json",
        "answer_openapi_query",
        "interactive_query_planner",
        "setup_workflow_execution",
        "initiate_load_test", # New intent for load testing
        "handle_unknown",
        "handle_loop",
        "responder"
    ]

    SPEC_START_REGEX = re.compile(
        r"^\s*(\"openapi\":|\'openapi\':|openapi:|swagger:|{|-|\binfo:|\bpaths:|\bcomponents:)",
        re.IGNORECASE | re.MULTILINE
    )

    # Regex for load test command
    LOAD_TEST_REGEX = re.compile(r"run load test with (\d+)\s*(?:users|flows|instances)", re.IGNORECASE)

    SCHEMA_LOADED_COMMANDS: Dict[str, AVAILABLE_INTENTS] = {
        "describe graph": "describe_graph", "show graph": "describe_graph", "what is the plan": "describe_graph",
        "get graph json": "get_graph_json", "show graph json": "get_graph_json",
        "generate new graph for": "_generate_execution_graph",
        "create new plan for": "_generate_execution_graph",
        "refine graph": "refine_api_graph", "improve plan": "refine_api_graph",
        "verify graph": "verify_graph",
        "run workflow": "setup_workflow_execution",
        "execute workflow": "setup_workflow_execution",
        "start workflow": "setup_workflow_execution",
        "run the plan": "setup_workflow_execution",
        "execute the plan": "setup_workflow_execution",
        # "run load test with N users" will be handled by LOAD_TEST_REGEX
    }

    def __init__(self, router_llm: Any): # Assuming router_llm is passed for LLM-based routing
        if not hasattr(router_llm, 'invoke'): # Check if it's an LLM-like object
            logger.warning("router_llm passed to OpenAPIRouter does not have 'invoke'. LLM-based routing might fail.")
            # raise TypeError("router_llm must have an 'invoke' method.") # Or handle more gracefully
        self.router_llm = router_llm
        logger.info("OpenAPIRouter initialized.")

    def _is_new_spec(self, state: BotState, user_input: str) -> bool:
        if not user_input or len(user_input) < 50:
            return False
        if not self.SPEC_START_REGEX.search(user_input):
            return False
        if state.openapi_spec_text:
            try:
                current_hash = hashlib.sha256(state.openapi_spec_text.strip().encode('utf-8')).hexdigest()
                new_hash = hashlib.sha256(user_input.strip().encode('utf-8')).hexdigest()
                if current_hash == new_hash:
                    return False
            except Exception as e:
                logger.warning(f"Error comparing spec hashes: {e}. Assuming different.")
        try:
            parsed_content = json.loads(user_input) if user_input.strip().startswith("{") else yaml.safe_load(user_input)
            if isinstance(parsed_content, dict) and \
               ('openapi' in parsed_content or 'swagger' in parsed_content) and \
               'info' in parsed_content and 'paths' in parsed_content:
                return True
            return False
        except (json.JSONDecodeError, yaml.YAMLError, TypeError):
            return False
        except Exception:
            return False # Fallback

    def route(self, state: BotState) -> BotState:
        user_input = state.user_input
        if not user_input:
            logger.warning("Router: Empty user input received. Routing to handle_unknown.")
            state.intent = "handle_unknown"
            return state

        user_input_lower = user_input.lower().strip()
        previous_intent = state.intent

        state.input_is_spec = False
        state.extracted_params = {} # Initialize/clear extracted_params for the turn

        if self._is_new_spec(state, user_input):
            logger.info("Router: Detected potential new OpenAPI spec. Routing to parse_openapi_spec.")
            state.openapi_spec_string = user_input
            state.input_is_spec = True
            state.intent = "parse_openapi_spec"
            state.execution_graph = None
            state.plan_generation_goal = None
            state.graph_refinement_iterations = 0
            state.payload_descriptions = {}
            state.identified_apis = []
            state.schema_summary = None
            state.workflow_execution_status = "idle"
            state.workflow_extracted_data = {}
            state.workflow_execution_results = {}
            state.scratchpad.pop('workflow_executor_instance', None)
            return state

        determined_intent: Optional[OpenAPIRouter.AVAILABLE_INTENTS] = None

        # Check for load test command first if a schema and graph exist
        if state.openapi_schema and state.execution_graph: # Load test requires an existing plan
            load_test_match = self.LOAD_TEST_REGEX.match(user_input) # Match on original case for consistency, regex is case-insensitive
            if load_test_match:
                try:
                    num_users = int(load_test_match.group(1))
                    if num_users > 0:
                        logger.info(f"Router: Matched load test command for {num_users} users.")
                        determined_intent = "initiate_load_test"
                        state.extracted_params = {"num_users": num_users} # Store extracted number of users
                        # Duration can be added here if regex is updated
                    else:
                        logger.warning(f"Router: Load test command with invalid number of users ({num_users}). Treating as unknown.")
                        # Optionally, set response here to inform user
                except ValueError:
                    logger.warning("Router: Load test command with non-integer number of users. Treating as unknown.")

        # If not a load test, check other simple commands
        if determined_intent is None and state.openapi_schema:
            for command_prefix, intent_val in self.SCHEMA_LOADED_COMMANDS.items():
                if user_input_lower.startswith(command_prefix):
                    logger.info(f"Router: Matched command '{command_prefix}'. Routing to {intent_val}.")
                    determined_intent = intent_val
                    if intent_val == "_generate_execution_graph":
                        state.plan_generation_goal = user_input[len(command_prefix):].strip() or "Generate a relevant workflow."
                        state.execution_graph = None; state.graph_refinement_iterations = 0
                    elif intent_val == "refine_api_graph":
                        state.graph_regeneration_reason = user_input
                    # No specific params for setup_workflow_execution here
                    break # Important: exit after first match

        # LLM-based intent classification if no command match and schema exists
        if determined_intent is None and state.openapi_schema and hasattr(self.router_llm, 'invoke'):
            logger.debug("Router: No simple command match. Using LLM for intent classification.")
            graph_status_info = "No API execution graph exists."
            if state.execution_graph: graph_status_info = f"An API execution graph for goal '{state.plan_generation_goal or 'unknown'}' exists."
            workflow_status_info = f"Workflow status: {state.workflow_execution_status}."
            classification_prompt = f"""
            An OpenAPI spec is loaded. {graph_status_info} {workflow_status_info}
            User input: "{user_input}"
            Classify intent. Choose ONE: "answer_openapi_query", "setup_workflow_execution", "interactive_query_planner", "unknown".
            Chosen Classification:""" # Simplified prompt for brevity
            try:
                # Assuming llm_call_helper is available or self.router_llm.invoke is used directly
                # For this example, direct use if llm_call_helper is not in this file's scope
                # from utils import llm_call_helper # Would be needed if used
                # llm_response_raw = llm_call_helper(self.router_llm, classification_prompt)
                response_obj = self.router_llm.invoke(classification_prompt) # Direct invoke
                llm_response_raw = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)

                llm_response = llm_response_raw.strip().lower().replace("\"", "").replace("'", "")
                if "initiate_load_test" in llm_response: determined_intent = "initiate_load_test" # LLM could also suggest this
                elif "setup_workflow_execution" in llm_response: determined_intent = "setup_workflow_execution"
                elif "interactive_query_planner" in llm_response: determined_intent = "interactive_query_planner"
                elif "answer_openapi_query" in llm_response: determined_intent = "answer_openapi_query"
                else: determined_intent = "handle_unknown"
                logger.info(f"Router LLM classified intent as: {determined_intent} (raw: '{llm_response_raw}')")
            except Exception as e:
                logger.error(f"Router LLM classification failed: {e}", exc_info=True)
                determined_intent = "handle_unknown"

        # Default if no intent determined
        if determined_intent is None:
            logger.debug("Router: No intent determined. Defaulting to handle_unknown.")
            determined_intent = "handle_unknown"
            if state.openapi_schema is None and len(user_input) > 30:
                 state.response = "I don't have an OpenAPI specification loaded. Please provide one first."

        # Loop detection and final assignment
        final_intent_str = str(determined_intent)
        non_looping_intents = ["handle_unknown", "handle_loop", "parse_openapi_spec", "responder", "interactive_query_planner"]
        if final_intent_str == previous_intent and final_intent_str not in non_looping_intents:
            state.loop_counter += 1
            if state.loop_counter >= 2:
                logger.warning(f"Router: Potential loop with intent '{final_intent_str}'. Routing to handle_loop.")
                final_intent_str = "handle_loop"
                state.loop_counter = 0
        else:
            state.loop_counter = 0

        state.intent = final_intent_str
        logger.info(f"Router final routing decision: '{state.intent}' for input: '{user_input[:100]}...' with params: {state.extracted_params}")
        return state

# Ensure hashlib is imported if used by _is_new_spec
import hashlib
