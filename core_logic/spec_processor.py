# core_logic/spec_processor.py
import json
import logging
from typing import Any, Dict, List, Optional, Callable
import yaml
import os

from models import BotState, GraphOutput # GraphOutput for type hints
from utils import (
    llm_call_helper,
    load_cached_schema,
    save_schema_to_cache,
    get_cache_key,
    SCHEMA_CACHE,
)
# Removed openapi-spec-validator imports:
# from openapi_spec_validator import (
#     openapi_v2_spec_validator,
#     openapi_v30_spec_validator,
#     openapi_v31_spec_validator,
# )
# from openapi_spec_validator.validation.exceptions import OpenAPIValidationError

logger = logging.getLogger(__name__)

# --- Configurable Limits from environment variables ---
MAX_PAYLOAD_SAMPLES_TO_GENERATE_INITIAL = int(
    os.getenv("MAX_PAYLOAD_SAMPLES_TO_GENERATE_INITIAL", "3")
)
MAX_PAYLOAD_SAMPLES_TO_GENERATE_SINGLE_PASS = int(
    os.getenv("MAX_PAYLOAD_SAMPLES_TO_GENERATE_SINGLE_PASS", "2")
)


class SpecProcessor:
    """
    Handles parsing and initial analysis of OpenAPI specifications,
    including generating summaries, identifying APIs, and creating payload descriptions.
    Schema validation via openapi-spec-validator has been removed.
    """

    def __init__(self, worker_llm: Any):
        """
        Initializes the SpecProcessor.

        Args:
            worker_llm: The language model used for tasks like summarization and description generation.
        """
        if not hasattr(worker_llm, "invoke"):
            raise TypeError("worker_llm must have an 'invoke' method.")
        self.worker_llm = worker_llm
        logger.info("SpecProcessor initialized.")

    def _queue_intermediate_message(self, state: BotState, msg: str):
        """
        Helper to queue messages for the UI and set the current response in BotState.
        """
        if "intermediate_messages" not in state.scratchpad:
            state.scratchpad["intermediate_messages"] = []
        if (
            not state.scratchpad["intermediate_messages"]
            or state.scratchpad["intermediate_messages"][-1] != msg
        ):
            state.scratchpad["intermediate_messages"].append(msg)
        state.response = msg

    def parse_openapi_spec(self, state: BotState) -> BotState:
        """
        Parses the OpenAPI specification string provided in BotState.
        Updates BotState with the parsed schema, cache key, and status.
        Does NOT perform schema validation using openapi-spec-validator.
        Handles caching of successfully parsed specifications.
        """
        tool_name = "parse_openapi_spec"
        self._queue_intermediate_message(
            state, "Parsing OpenAPI specification (validation step removed)..."
        )
        state.update_scratchpad_reason(
            tool_name, "Attempting to parse OpenAPI spec (validation removed)."
        )
        spec_text = state.openapi_spec_string

        if not spec_text:
            self._queue_intermediate_message(state, "No OpenAPI specification text provided.")
            state.update_scratchpad_reason(tool_name, "No spec text in state.")
            state.next_step = "responder"
            state.openapi_spec_string = None
            return state

        cache_key = get_cache_key(spec_text)
        # Adjusted cache key to reflect that this is parsed, not externally validated
        cached_full_analysis_key = f"{cache_key}_full_analysis_parsed_only"

        if SCHEMA_CACHE:
            cached_schema_artifacts = load_cached_schema(cached_full_analysis_key)
            if cached_schema_artifacts and isinstance(cached_schema_artifacts, dict):
                try:
                    state.openapi_schema = cached_schema_artifacts.get("openapi_schema")
                    state.schema_summary = cached_schema_artifacts.get("schema_summary")
                    state.identified_apis = cached_schema_artifacts.get(
                        "identified_apis", []
                    )
                    state.payload_descriptions = cached_schema_artifacts.get(
                        "payload_descriptions", {}
                    )
                    graph_dict = cached_schema_artifacts.get("execution_graph")
                    if graph_dict:
                        state.execution_graph = (
                            GraphOutput.model_validate(graph_dict)
                            if isinstance(graph_dict, dict)
                            else graph_dict
                        )
                    
                    state.schema_cache_key = cache_key
                    state.openapi_spec_text = spec_text
                    state.openapi_spec_string = None

                    logger.info(
                        f"Loaded parsed OpenAPI data and analysis from cache: {cached_full_analysis_key}"
                    )
                    self._queue_intermediate_message(
                        state,
                        "OpenAPI specification and derived analysis (parsed only) loaded from cache.",
                    )
                    if state.execution_graph and isinstance(state.execution_graph, GraphOutput):
                        state.scratchpad['graph_to_send'] = state.execution_graph.model_dump_json(indent=2)
                    state.next_step = "responder"
                    return state
                except Exception as e:
                    logger.warning(
                        f"Error rehydrating state from cached full analysis (key: {cached_full_analysis_key}): {e}. "
                        "Proceeding with fresh parsing."
                    )
                    state.openapi_schema = None
                    state.schema_summary = None
                    state.identified_apis = []
                    state.payload_descriptions = {}
                    state.execution_graph = None
        
        parsed_spec_dict: Optional[Dict[str, Any]] = None
        error_message: Optional[str] = None

        try:
            parsed_spec_dict = json.loads(spec_text)
        except json.JSONDecodeError:
            try:
                parsed_spec_dict = yaml.safe_load(spec_text)
            except yaml.YAMLError as yaml_e:
                error_message = f"YAML parsing failed: {yaml_e}"
            except Exception as e_yaml_other:
                 error_message = f"Unexpected error during YAML parsing: {e_yaml_other}"
        except Exception as e_json_other:
            error_message = f"Unexpected error during JSON parsing: {e_json_other}"

        if error_message:
            state.openapi_schema = None
            state.openapi_spec_string = None
            self._queue_intermediate_message(
                state, f"Failed to parse specification: {error_message}"
            )
            logger.error(
                f"Parsing failed: {error_message}. Input snippet: {spec_text[:200]}..."
            )
            state.next_step = "responder"
            state.update_scratchpad_reason(
                tool_name, f"Parsing failed. Response: {state.response}"
            )
            return state

        if not parsed_spec_dict or not isinstance(parsed_spec_dict, dict):
            state.openapi_schema = None
            state.openapi_spec_string = None
            self._queue_intermediate_message(
                state, "Parsed content is not a valid dictionary structure."
            )
            logger.error(
                f"Parsed content is not a dictionary. Type: {type(parsed_spec_dict)}. Input snippet: {spec_text[:200]}..."
            )
            state.next_step = "responder"
            state.update_scratchpad_reason(tool_name, "Parsed content not a dict.")
            return state

        # --- Validation Block Removed ---
        # The openapi-spec-validator logic was here.
        # Without it, we assume the parsed_spec_dict is usable if parsing succeeded.
        # The primary risk is that $ref pointers are not resolved and the schema might be structurally invalid
        # according to OpenAPI rules, which could cause issues downstream.

        # Check for basic OpenAPI structure heuristically
        if not (parsed_spec_dict.get("openapi") or parsed_spec_dict.get("swagger")) or \
           not parsed_spec_dict.get("info") or \
           not parsed_spec_dict.get("paths"):
            state.openapi_schema = None
            state.openapi_spec_string = None
            self._queue_intermediate_message(
                state,
                "Parsed specification appears to be missing essential OpenAPI fields (openapi/swagger, info, paths). "
                "Proceeding with caution, but analysis might be incomplete or fail.",
            )
            logger.warning(
                "Parsed spec is missing key OpenAPI fields. Downstream processing might fail."
            )
            # Allow proceeding to the pipeline, but the user is warned.
            # Alternatively, could route to responder here if strictness is desired.
            # state.next_step = "responder"
            # For now, let's try to process it.
            state.openapi_schema = parsed_spec_dict # Store the parsed schema
            state.schema_cache_key = cache_key
            state.openapi_spec_text = spec_text
            state.openapi_spec_string = None
            state.next_step = "process_schema_pipeline"

        else:
            # If basic fields are present, proceed.
            state.openapi_schema = parsed_spec_dict
            state.schema_cache_key = cache_key
            state.openapi_spec_text = spec_text
            state.openapi_spec_string = None

            logger.info(
                "Successfully parsed OpenAPI spec. Validation step has been removed. "
                "References ($ref) within the spec may not be resolved."
            )
            self._queue_intermediate_message(
                state,
                "OpenAPI specification parsed (validation removed). Starting analysis pipeline...",
            )
            state.next_step = "process_schema_pipeline"
        
        state.update_scratchpad_reason(
            tool_name,
            f"Parsing status: {'Success (basic fields check)' if state.openapi_schema else 'Failed or missing key fields'}. Response: {state.response}",
        )
        return state

    def _generate_llm_schema_summary(self, state: BotState):
        """
        Generates a natural language summary of the loaded OpenAPI schema using an LLM.
        """
        tool_name = "_generate_llm_schema_summary"
        self._queue_intermediate_message(state, "Generating API summary...")
        state.update_scratchpad_reason(tool_name, "Generating schema summary.")

        if not state.openapi_schema:
            state.schema_summary = "Could not generate summary: No schema loaded."
            logger.warning(state.schema_summary)
            self._queue_intermediate_message(state, state.schema_summary)
            return

        spec_info = state.openapi_schema.get("info", {})
        title = spec_info.get("title", "N/A")
        version = spec_info.get("version", "N/A")
        description = spec_info.get("description", "N/A")
        num_paths = len(state.openapi_schema.get("paths", {}))

        paths_preview_list = []
        for p, m_dict in list(state.openapi_schema.get("paths", {}).items())[:3]:
            methods = list(m_dict.keys()) if isinstance(m_dict, dict) else '[methods not parsable]'
            paths_preview_list.append(f"  {p}: {methods}")
        paths_preview = "\n".join(paths_preview_list)

        summary_prompt = (
            f"Summarize the following API specification. Focus on its main purpose, "
            f"key resources/capabilities, and any mentioned authentication schemes. Be concise (around 100-150 words).\n\n"
            f"Title: {title}\nVersion: {version}\nDescription: {description[:500]}...\n"
            f"Number of paths: {num_paths}\nExample Paths (first 3):\n{paths_preview}\n\n"
            f"Note: The schema was parsed but not validated by an external tool, so $ref pointers might be unresolved.\n"
            f"Concise Summary:"
        )
        try:
            state.schema_summary = llm_call_helper(self.worker_llm, summary_prompt)
            logger.info("Schema summary generated.")
            self._queue_intermediate_message(state, "API summary created.")
        except Exception as e:
            logger.error(f"Error generating schema summary: {e}", exc_info=False)
            state.schema_summary = f"Error generating summary: {str(e)[:150]}..."
            self._queue_intermediate_message(state, state.schema_summary)
        
        state.update_scratchpad_reason(
            tool_name,
            f"Summary status: {'Success' if state.schema_summary and not state.schema_summary.startswith('Error') else 'Failed'}",
        )

    def _identify_apis_from_schema(self, state: BotState):
        """
        Identifies and extracts details of API operations from the loaded schema.
        """
        tool_name = "_identify_apis_from_schema"
        self._queue_intermediate_message(state, "Identifying API operations...")
        state.update_scratchpad_reason(tool_name, "Identifying APIs.")

        if not state.openapi_schema:
            state.identified_apis = []
            logger.warning("No schema to identify APIs from.")
            self._queue_intermediate_message(state, "Cannot identify APIs: No schema loaded.")
            return

        apis = []
        paths = state.openapi_schema.get("paths", {})
        for path_url, path_item in paths.items():
            if not isinstance(path_item, dict):
                logger.warning(f"Skipping non-dictionary path item at '{path_url}'")
                continue
            for method, operation_details in path_item.items():
                if method.lower() not in {
                    "get", "post", "put", "delete", "patch", "options", "head", "trace"
                } or not isinstance(operation_details, dict):
                    continue
                
                op_id_suffix = path_url.replace('/', '_').replace('{', '').replace('}', '').strip('_')
                default_op_id = f"{method.lower()}_{op_id_suffix or 'root'}"

                api_info = {
                    "operationId": operation_details.get("operationId", default_op_id),
                    "path": path_url,
                    "method": method.upper(),
                    "summary": operation_details.get("summary", ""),
                    "description": operation_details.get("description", ""),
                    "parameters": operation_details.get("parameters", []), # These might contain $refs
                    "requestBody": operation_details.get("requestBody", {}), # This might contain $refs
                    "responses": operation_details.get("responses", {}), # These might contain $refs
                }
                apis.append(api_info)
        state.identified_apis = apis
        logger.info(f"Identified {len(apis)} API operations.")
        self._queue_intermediate_message(state, f"Identified {len(apis)} API operations.")
        state.update_scratchpad_reason(tool_name, f"Identified {len(apis)} APIs.")

    def _generate_payload_descriptions(
        self,
        state: BotState,
        target_apis: Optional[List[str]] = None,
        context_override: Optional[str] = None,
    ):
        """
        Generates example request payloads and response structure descriptions for API operations.
        Warns that schemas might contain unresolved $refs.
        """
        tool_name = "_generate_payload_descriptions"
        self._queue_intermediate_message(
            state, "Creating payload and response examples..."
        )
        state.update_scratchpad_reason(
            tool_name,
            f"Generating payload descriptions. Targets: {target_apis or 'subset'}. Context: {bool(context_override)}",
        )

        if not state.identified_apis:
            logger.warning("No APIs identified, cannot generate payload descriptions.")
            self._queue_intermediate_message(
                state, "Cannot create payload examples: No APIs identified."
            )
            return

        payload_descs = state.payload_descriptions or {}
        apis_to_process = []
        if target_apis:
            apis_to_process = [
                api for api in state.identified_apis if api["operationId"] in target_apis
            ]
        else:
            apis_with_payload_info = [
                api
                for api in state.identified_apis
                if api.get("requestBody")
                or any(
                    p.get("in") in ["body", "formData"]
                    for p in api.get("parameters", [])
                )
            ]
            unprocessed_apis = [
                api
                for api in apis_with_payload_info
                if api["operationId"] not in payload_descs
            ]
            if unprocessed_apis:
                apis_to_process = unprocessed_apis[:MAX_PAYLOAD_SAMPLES_TO_GENERATE_INITIAL]
            else:
                apis_to_process = apis_with_payload_info[:MAX_PAYLOAD_SAMPLES_TO_GENERATE_SINGLE_PASS]

        logger.info(
            f"Attempting to generate payload descriptions for {len(apis_to_process)} APIs."
        )
        processed_count = 0
        for api_op in apis_to_process:
            op_id = api_op["operationId"]
            if op_id in payload_descs and not context_override and not target_apis and processed_count > 0:
                continue

            self._queue_intermediate_message(state, f"Generating payload example for '{op_id}'...")
            params_summary_list = []
            for p_idx, p_detail in enumerate(api_op.get("parameters", [])):
                if p_idx >= 5:
                    params_summary_list.append("...")
                    break
                param_name = p_detail.get("name", "N/A")
                param_in = p_detail.get("in", "N/A")
                param_type = "N/A"
                # Schemas might contain $refs here
                if 'schema' in p_detail and isinstance(p_detail['schema'], dict):
                    param_type = p_detail['schema'].get('type', str(p_detail['schema'])) # Show $ref if type missing
                    if param_type == 'array' and 'items' in p_detail['schema'] and isinstance(p_detail['schema']['items'], dict):
                        param_type = f"array of {p_detail['schema']['items'].get('type', str(p_detail['schema']['items']))}"
                params_summary_list.append(f"{param_name}({param_in}, type: {param_type})")
            params_summary_str = ", ".join(params_summary_list) if params_summary_list else "None"
            
            request_body_schema_str = "N/A (or may contain $refs)"
            if api_op.get('requestBody') and isinstance(api_op['requestBody'], dict):
                content = api_op['requestBody'].get('content', {})
                json_content = content.get('application/json', {})
                schema = json_content.get('schema', {}) # This schema might contain $refs
                if schema:
                    request_body_schema_str = json.dumps(schema, indent=2)[:500] + "..."

            success_response_schema_str = "N/A (or may contain $refs)"
            responses = api_op.get("responses", {})
            for status_code, resp_details in responses.items():
                if status_code.startswith("2") and isinstance(resp_details, dict):
                    content = resp_details.get("content", {})
                    json_content = content.get("application/json", {})
                    schema = json_content.get("schema", {}) # This schema might contain $refs
                    if schema:
                        success_response_schema_str = json.dumps(schema, indent=2)[:300] + "..."
                        break

            context_str = f" User Context: {context_override}." if context_override else ""
            prompt = (
                f"API Operation: {op_id} ({api_op['method']} {api_op['path']})\n"
                f"Summary: {api_op.get('summary', 'N/A')}\n{context_str}\n"
                f"Parameters: {params_summary_str}\n"
                f"Request Body Schema (if application/json; may contain unresolved $ref pointers):\n```json\n{request_body_schema_str}\n```\n"
                f"Successful (2xx) Response Schema (sample, if application/json; may contain unresolved $ref pointers):\n```json\n{success_response_schema_str}\n```\n\n"
                f"Task: Provide a concise, typical, and REALISTIC JSON example for the request payload (if applicable for this method and API design). "
                f"Use plausible, real-world example values based on the parameter names, types, and the API schema. Be aware that the provided schemas might contain $ref pointers that are not resolved. "
                f"If a schema is just a $ref (e.g., {{\"'$ref'\": \"#/components/schemas/User\"}}), assume a plausible structure for that referenced object based on its name. "
                f"For example, if a field is 'email', use 'user@example.com'. If 'count', use a number like 5. "
                f"Also, provide a brief description of the expected JSON response structure for a successful call, based on the schema. "
                f"Focus on key fields. If no request payload is typically needed (e.g., for GET with only path/query params), state 'No request payload needed.' clearly. "
                f"Format clearly:\n"
                f"Request Payload Example:\n```json\n{{\"key\": \"realistic_value\", \"another_key\": 123}}\n```\n"
                f"Expected Response Structure:\nBrief description of response fields (e.g., 'Returns an object with id, name, and status. The 'status' field indicates processing outcome.')."
            )

            try:
                description = llm_call_helper(self.worker_llm, prompt)
                payload_descs[op_id] = description
                processed_count += 1
            except Exception as e:
                logger.error(
                    f"Error generating payload description for {op_id}: {e}",
                    exc_info=False,
                )
                payload_descs[op_id] = f"Error generating description: {str(e)[:100]}..."
                self._queue_intermediate_message(
                    state, f"Error creating payload example for '{op_id}': {str(e)[:100]}..."
                )
                if "quota" in str(e).lower() or "429" in str(e):
                    logger.warning(f"Quota error during payload description for {op_id}. Stopping further payload generation for this turn.")
                    self._queue_intermediate_message(state, state.response + " Hit API limits during generation.")
                    break
        state.payload_descriptions = payload_descs
        
        final_payload_msg = ""
        if processed_count > 0:
            final_payload_msg = f"Generated payload examples for {processed_count} API operation(s)."
        elif not apis_to_process and not target_apis:
            final_payload_msg = "No relevant APIs found requiring new payload examples at this time."
        elif target_apis and not apis_to_process:
             final_payload_msg = f"Could not find the specified API(s) ({target_apis}) to generate payload examples."
        
        if final_payload_msg:
            self._queue_intermediate_message(state, final_payload_msg)
        
        state.update_scratchpad_reason(
            tool_name,
            f"Payload descriptions updated for {processed_count} of {len(apis_to_process)} targeted APIs.",
        )

    def process_schema_pipeline(
        self, state: BotState, graph_generator_func: Callable[[BotState, Optional[str]], BotState]
    ) -> BotState:
        """
        Orchestrates the full pipeline for processing an OpenAPI schema.
        """
        tool_name = "process_schema_pipeline"
        self._queue_intermediate_message(state, "Starting API analysis pipeline...")
        state.update_scratchpad_reason(tool_name, "Starting schema pipeline.")
        
        if not state.openapi_schema:
            self._queue_intermediate_message(
                state, "Cannot run pipeline: No schema loaded (parsing may have failed)."
            )
            state.next_step = "handle_unknown"
            return state
        
        state.schema_summary = None
        state.identified_apis = []
        state.payload_descriptions = {}
        state.execution_graph = None
        state.graph_refinement_iterations = 0
        state.plan_generation_goal = state.plan_generation_goal or "Provide a general overview workflow."
        state.scratchpad['graph_gen_attempts'] = 0
        state.scratchpad['refinement_validation_failures'] = 0

        self._generate_llm_schema_summary(state)
        if state.schema_summary and ("Error generating summary: 429" in state.schema_summary or "quota" in state.schema_summary.lower()):
            logger.warning("API limit hit during schema summary. Stopping pipeline.")
            state.next_step = "responder"
            return state
            
        self._identify_apis_from_schema(state)
        if not state.identified_apis:
            msg = (state.response or "") + " No API operations were identified from the schema. Cannot generate payload examples or an execution graph."
            self._queue_intermediate_message(state, msg)
            state.next_step = "responder"
            return state
            
        self._generate_payload_descriptions(state)
        if any("Error generating description: 429" in desc for desc in state.payload_descriptions.values()) or \
           any("quota" in desc.lower() for desc in state.payload_descriptions.values()):
            logger.warning("API limit hit during payload description generation.")

        state = graph_generator_func(state, state.plan_generation_goal)
        
        if (
            state.openapi_schema and state.schema_cache_key and SCHEMA_CACHE and
            state.execution_graph and 
            state.next_step not in ["handle_unknown", "responder_with_error_from_pipeline"]
        ):
            full_analysis_data = {
                "openapi_schema": state.openapi_schema,
                "schema_summary": state.schema_summary,
                "identified_apis": state.identified_apis,
                "payload_descriptions": state.payload_descriptions,
                "execution_graph": state.execution_graph.model_dump() if state.execution_graph and isinstance(state.execution_graph, GraphOutput) else None,
                "plan_generation_goal": state.plan_generation_goal,
            }
            # Use the adjusted cache key
            cached_full_analysis_key = f"{state.schema_cache_key}_full_analysis_parsed_only"
            save_schema_to_cache(cached_full_analysis_key, full_analysis_data)
            logger.info(
                f"Saved parsed data and analysis to cache: {cached_full_analysis_key}"
            )
            
        state.update_scratchpad_reason(
            tool_name,
            f"Schema processing pipeline completed. Next step determined by graph generation: {state.next_step}",
        )
        return state
