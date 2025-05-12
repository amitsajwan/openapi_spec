# router.py
import logging
import re
import hashlib
from typing import Dict, Any, Literal, Optional
import json
import yaml

# Assuming models.py is in the same directory or accessible in PYTHONPATH
from models import BotState
from utils import llm_call_helper

logger = logging.getLogger(__name__)

class OpenAPIRouter:
    """
    Routes user input to appropriate nodes in the LangGraph agent.
    Handles initial spec detection, subsequent interactive query classification,
    and requests to execute workflows.
    """
    AVAILABLE_INTENTS = Literal[
        "parse_openapi_spec",
        "process_schema_pipeline",
        "verify_graph",
        "_generate_execution_graph",
        "refine_api_graph",
        "describe_graph",
        "get_graph_json",
        "answer_openapi_query",
        "interactive_query_planner", # This planner can dispatch to core_logic methods
        "setup_workflow_execution",  # New: To initiate workflow execution
        # "resume_workflow_with_payload", # This will be handled by interactive_query_planner
        "handle_unknown",
        "handle_loop",
        "responder"
    ]

    # Regex to heuristically detect start of an OpenAPI/Swagger spec
    SPEC_START_REGEX = re.compile(
        r"^\s*(\"openapi\":|\'openapi\':|openapi:|swagger:|{|-|\binfo:|\bpaths:|\bcomponents:)",
        re.IGNORECASE | re.MULTILINE
    )

    # Simple commands that can be directly mapped to intents when a schema is loaded
    SCHEMA_LOADED_COMMANDS: Dict[str, AVAILABLE_INTENTS] = {
        "describe graph": "describe_graph", "show graph": "describe_graph", "what is the plan": "describe_graph",
        "get graph json": "get_graph_json", "show graph json": "get_graph_json",
        "generate new graph for": "_generate_execution_graph", # Expects goal after prefix
        "create new plan for": "_generate_execution_graph",    # Expects goal after prefix
        "refine graph": "refine_api_graph", "improve plan": "refine_api_graph", # Expects refinement instruction
        "verify graph": "verify_graph",
        # New commands for workflow execution
        "run workflow": "setup_workflow_execution",
        "execute workflow": "setup_workflow_execution",
        "start workflow": "setup_workflow_execution",
        "run the plan": "setup_workflow_execution",
        "execute the plan": "setup_workflow_execution",
    }

    def __init__(self, router_llm: Any):
        if not hasattr(router_llm, 'invoke'):
            raise TypeError("router_llm must have an 'invoke' method.")
        self.router_llm = router_llm
        logger.info("OpenAPIRouter initialized.")

    def _is_new_spec(self, state: BotState, user_input: str) -> bool:
        """Heuristically checks if input is a new OpenAPI spec."""
        if not user_input or len(user_input) < 50: # Basic length check
            return False
        
        # Check if input starts with common OpenAPI/Swagger keywords or structures
        if not self.SPEC_START_REGEX.search(user_input):
            return False
        
        # If a spec is already loaded, compare hashes to see if it's different
        if state.openapi_spec_text:
            try:
                # Normalize by stripping whitespace before hashing
                current_hash = hashlib.sha256(state.openapi_spec_text.strip().encode('utf-8')).hexdigest()
                new_hash = hashlib.sha256(user_input.strip().encode('utf-8')).hexdigest()
                if current_hash == new_hash:
                    logger.debug("Input content matches the currently loaded spec. Not treating as new.")
                    return False
            except Exception as e:
                logger.warning(f"Error comparing spec hashes: {e}. Assuming different for safety.", exc_info=False)
        
        # Attempt basic parsing to confirm structure (JSON or YAML)
        try:
            if user_input.strip().startswith("{"): # Likely JSON
                parsed_content = json.loads(user_input)
            else: # Likely YAML
                parsed_content = yaml.safe_load(user_input)
            
            # Check for essential OpenAPI root keys
            if isinstance(parsed_content, dict) and \
               ('openapi' in parsed_content or 'swagger' in parsed_content) and \
               'info' in parsed_content and 'paths' in parsed_content:
                logger.debug("Input is potentially a new, parsable OpenAPI/Swagger spec.")
                return True
            else:
                logger.debug("Input parsed but doesn't look like a valid OpenAPI/Swagger spec (missing key fields).")
                return False
        except (json.JSONDecodeError, yaml.YAMLError, TypeError): # Added TypeError for yaml.safe_load with bad input
            logger.debug("Input starts like a spec but failed basic JSON/YAML parsing. Not treating as new spec via heuristic.")
            return False
        except Exception as e: # Catch any other unexpected error during parsing attempt
            logger.warning(f"Unexpected error during spec parsing check: {e}", exc_info=False)
            return False


    def route(self, state: BotState) -> BotState:
        user_input = state.user_input
        if not user_input: # Should ideally not happen if input is validated before router
            logger.warning("Router: Empty user input received. Routing to handle_unknown.")
            state.intent = "handle_unknown"
            return state

        user_input_lower = user_input.lower().strip()
        previous_intent = state.intent
        
        # Reset flags/fields for the new turn
        state.input_is_spec = False
        # state.response = None # Intermediate responses are set by nodes, not router

        # 1. Check for new OpenAPI specification
        if self._is_new_spec(state, user_input): # Pass the original user_input
            logger.info("Router: Detected potential new OpenAPI spec. Routing to parse_openapi_spec.")
            state.openapi_spec_string = user_input # Store the raw string for parsing
            state.input_is_spec = True
            state.intent = "parse_openapi_spec"
            # Reset related state fields as a new spec is being processed
            state.execution_graph = None
            state.plan_generation_goal = None
            state.graph_refinement_iterations = 0
            state.payload_descriptions = {}
            state.identified_apis = []
            state.schema_summary = None
            state.workflow_execution_status = "idle" # Reset workflow status
            state.workflow_extracted_data = {}
            state.workflow_execution_results = {}
            state.scratchpad.pop('workflow_executor_instance', None) # Clear any old executor
            return state

        determined_intent: Optional[OpenAPIRouter.AVAILABLE_INTENTS] = None

        # 2. Check for simple commands if a schema is loaded
        if state.openapi_schema: # Check if a schema (and thus identified_apis, graph etc.) might exist
            for command_prefix, intent_val in self.SCHEMA_LOADED_COMMANDS.items():
                if user_input_lower.startswith(command_prefix):
                    logger.info(f"Router: Matched command '{command_prefix}'. Routing to {intent_val}.")
                    determined_intent = intent_val
                    
                    if intent_val == "_generate_execution_graph":
                        # Extract the goal from the user input after the command prefix
                        state.plan_generation_goal = user_input[len(command_prefix):].strip()
                        if not state.plan_generation_goal: # If no specific goal, use a generic one
                             state.plan_generation_goal = "Generate a relevant workflow based on the API capabilities."
                             logger.warning(f"Command '{command_prefix}' matched but no specific goal provided. Using generic goal.")
                        # Reset graph-specific fields for new generation
                        state.execution_graph = None 
                        state.graph_refinement_iterations = 0
                    elif intent_val == "refine_api_graph":
                        # The rest of the input is considered refinement instructions
                        state.graph_regeneration_reason = user_input # Pass full input as reason
                    elif intent_val == "setup_workflow_execution":
                        # No specific params extracted here; core_logic will handle setup
                        logger.info("Router: Intent set to setup_workflow_execution based on command.")
                    break
        
        # 3. If no simple command match and schema is loaded, use LLM for intent classification
        if determined_intent is None and state.openapi_schema:
            logger.debug("Router: No simple command match with loaded schema. Using LLM for intent classification.")
            
            graph_status_info = "No API execution graph currently exists."
            if state.execution_graph:
                graph_status_info = f"An API execution graph for goal '{state.plan_generation_goal or 'unknown'}' has been generated."
            
            workflow_status_info = f"Current workflow execution status: {state.workflow_execution_status}."

            classification_prompt = f"""
            An OpenAPI specification is loaded. {graph_status_info} {workflow_status_info}
            User input: "{user_input}"

            Classify the user's intent. Choose ONE of the following:
            - "answer_openapi_query": If the user is asking a straightforward question about the API spec, its operations, the current graph, or existing payload examples. (e.g., "List all GET APIs.", "What does 'createUser' do?", "Describe the current plan.")
            - "setup_workflow_execution": If the user explicitly asks to run, execute, or start the current workflow/plan. (e.g., "Run this workflow.", "Execute the plan.")
            - "interactive_query_planner": If the user's request implies a need to *modify context for generation*, *regenerate parts of the plan/payloads with new information*, *set a new goal for the graph*, make structural changes to the graph, or requires multi-step reasoning beyond a simple lookup or direct execution. This is also used if the user provides data to resume a paused workflow. (e.g., "Focus the plan on 'X' products.", "What if I want to add a notification step?", "Regenerate the graph for goal Y.", "Here is the confirmed payload: {{...}}").
            - "unknown": If the intent is unclear or doesn't fit the above.

            Chosen Classification:
            """
            try:
                llm_response_raw = llm_call_helper(self.router_llm, classification_prompt)
                llm_response = llm_response_raw.strip().lower().replace("\"", "").replace("'", "") # Clean response
                
                # Match against available intents robustly
                if "setup_workflow_execution" in llm_response:
                    determined_intent = "setup_workflow_execution"
                elif "interactive_query_planner" in llm_response:
                    determined_intent = "interactive_query_planner"
                elif "answer_openapi_query" in llm_response:
                    determined_intent = "answer_openapi_query"
                else: # Fallback for unclear LLM responses or if it literally says "unknown"
                    determined_intent = "handle_unknown"
                logger.info(f"Router LLM classified intent as: {determined_intent} (raw LLM response: '{llm_response_raw}')")

            except Exception as e:
                logger.error(f"Router LLM classification failed: {e}", exc_info=True)
                determined_intent = "handle_unknown" # Fallback on error

        # 4. If no intent determined yet (e.g., no schema loaded and no spec input)
        if determined_intent is None:
            logger.debug("Router: No schema loaded, not a new spec, and no LLM classification done (or failed). Defaulting to handle_unknown.")
            determined_intent = "handle_unknown"
            if state.openapi_schema is None and len(user_input) > 30: # Heuristic: if input is somewhat long but not a spec
                 state.response = "I don't have an OpenAPI specification loaded. Please provide one first so I can assist with that query."


        # 5. Loop detection and final intent assignment
        final_intent_str = str(determined_intent) # Ensure it's a string
        
        # Check for repetitive intents that might indicate a loop
        # Exclude intents that are naturally part of a sequence or terminal
        non_looping_intents = [
            "handle_unknown", "handle_loop", "parse_openapi_spec", "responder",
            "interactive_query_planner" # Planner itself has sub-steps, so less likely to be a direct loop source
        ]
        if final_intent_str == previous_intent and final_intent_str not in non_looping_intents:
            state.loop_counter += 1
            if state.loop_counter >= 2: # If the same non-terminal intent is hit twice in a row
                logger.warning(f"Router: Potential loop detected with intent '{final_intent_str}'. Routing to handle_loop.")
                final_intent_str = "handle_loop"
                state.loop_counter = 0 # Reset counter after handling loop
        else:
            state.loop_counter = 0 # Reset counter if intent changes or is non-looping

        state.intent = final_intent_str
        logger.info(f"Router final routing decision: '{state.intent}' for input: '{user_input[:100]}...'")
        return state

