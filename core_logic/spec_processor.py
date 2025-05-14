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
# Re-add openapi-spec-validator imports for v3.x
from openapi_spec_validator import (
    openapi_v30_spec_validator,
    openapi_v31_spec_validator,
)
from openapi_spec_validator.validation.exceptions import OpenAPIValidationError

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
    Handles parsing, validation (for OpenAPI 3.0.x, 3.1.x), and initial analysis
    of OpenAPI specifications.
    """

    def __init__(self, worker_llm: Any):
        if not hasattr(worker_llm, "invoke"):
            raise TypeError("worker_llm must have an 'invoke' method.")
        self.worker_llm = worker_llm
        logger.info("SpecProcessor initialized.")

    def _queue_intermediate_message(self, state: BotState, msg: str):
        if "intermediate_messages" not in state.scratchpad:
            state.scratchpad["intermediate_messages"] = []
        if (
            not state.scratchpad["intermediate_messages"]
            or state.scratchpad["intermediate_messages"][-1] != msg
        ):
            state.scratchpad["intermediate_messages"].append(msg)
        state.response = msg

    def parse_openapi_spec(self, state: BotState) -> BotState:
        tool_name = "parse_openapi_spec"
        self._queue_intermediate_message(
            state, "Parsing OpenAPI specification (with v3.x validation)..."
        )
        state.update_scratchpad_reason(
            tool_name, "Attempting to parse and validate (v3.x) OpenAPI spec."
        )
        spec_text = state.openapi_spec_string
        
        # Enhanced Log: Check initial spec_text type and content
        logger.debug(f"[{tool_name}] Initial spec_text type: {type(spec_text)}")
        if isinstance(spec_text, str):
            logger.debug(f"[{tool_name}] Initial spec_text (first 200 chars): {spec_text[:200]}")
        else:
            logger.warning(f"[{tool_name}] Initial spec_text is not a string.")


        if not spec_text:
            self._queue_intermediate_message(state, "No OpenAPI specification text provided.")
            state.update_scratchpad_reason(tool_name, "No spec text in state.")
            state.next_step = "responder"
            state.openapi_spec_string = None
            return state
        
        # Explicitly check if spec_text is a string before proceeding
        if not isinstance(spec_text, str):
            error_msg = f"Invalid type for OpenAPI specification: expected a string, but got {type(spec_text)}. Cannot proceed."
            logger.error(f"[{tool_name}] {error_msg}")
            self._queue_intermediate_message(state, error_msg)
            state.openapi_schema = None
            state.openapi_spec_string = None # Clear invalid input
            state.next_step = "responder"
            return state

        cache_key = get_cache_key(spec_text)
        cached_full_analysis_key = f"{cache_key}_full_analysis_v3_validated_or_parsed"

        if SCHEMA_CACHE:
            cached_schema_artifacts = load_cached_schema(cached_full_analysis_key)
            if cached_schema_artifacts and isinstance(cached_schema_artifacts, dict):
                try:
                    # ... (cache loading logic as before) ...
                    state.openapi_schema = cached_schema_artifacts.get("openapi_schema")
                    state.schema_summary = cached_schema_artifacts.get("schema_summary")
                    state.identified_apis = cached_schema_artifacts.get("identified_apis", [])
                    state.payload_descriptions = cached_schema_artifacts.get("payload_descriptions", {})
                    graph_dict = cached_schema_artifacts.get("execution_graph")
                    if graph_dict:
                        state.execution_graph = GraphOutput.model_validate(graph_dict) if isinstance(graph_dict, dict) else graph_dict
                    state.schema_cache_key = cache_key
                    state.openapi_spec_text = spec_text 
                    state.openapi_spec_string = None
                    logger.info(f"Loaded processed OpenAPI data and analysis from cache: {cached_full_analysis_key}")
                    self._queue_intermediate_message(state, "OpenAPI specification and derived analysis (v3 validated or parsed) loaded from cache.")
                    if state.execution_graph and isinstance(state.execution_graph, GraphOutput):
                        state.scratchpad['graph_to_send'] = state.execution_graph.model_dump_json(indent=2)
                    state.next_step = "responder"
                    return state
                except Exception as e:
                    logger.warning(f"Error rehydrating state from cache (key: {cached_full_analysis_key}): {e}. Proceeding with fresh parsing.")
                    state.openapi_schema = None; state.schema_summary = None; state.identified_apis = []; state.payload_descriptions = {}; state.execution_graph = None
        
        parsed_spec_dict: Optional[Dict[str, Any]] = None
        error_message: Optional[str] = None
        
        logger.debug(f"[{tool_name}] Attempting to parse spec_text (length: {len(spec_text)}) as JSON/YAML.")
        try:
            if spec_text.strip().startswith("{"):
                logger.debug(f"[{tool_name}] Trying to parse as JSON.")
                parsed_spec_dict = json.loads(spec_text)
                logger.debug(f"[{tool_name}] Successfully parsed as JSON.")
            else:
                logger.debug(f"[{tool_name}] Trying to parse as YAML.")
                parsed_spec_dict = yaml.safe_load(spec_text) # This expects a stream (string or file)
                logger.debug(f"[{tool_name}] Successfully parsed as YAML.")
        except json.JSONDecodeError as json_e:
            error_message = f"JSON parsing failed: {json_e.msg} (at line {json_e.lineno} column {json_e.colno})"
            logger.error(f"[{tool_name}] {error_message}")
        except yaml.YAMLError as yaml_e:
            error_message = f"YAML parsing failed: {yaml_e}"
            logger.error(f"[{tool_name}] {error_message}")
        except Exception as e_parse:
            error_message = f"Unexpected error during spec parsing: {type(e_parse).__name__} - {e_parse}"
            logger.error(f"[{tool_name}] {error_message}", exc_info=True)

        if error_message:
            state.openapi_schema = None; state.openapi_spec_string = None
            self._queue_intermediate_message(state, f"Failed to parse specification: {error_message}")
            logger.error(f"Parsing failed. Input snippet: {spec_text[:200]}...")
            state.next_step = "responder"
            state.update_scratchpad_reason(tool_name, f"Parsing failed. Response: {state.response}")
            return state

        if not parsed_spec_dict or not isinstance(parsed_spec_dict, dict):
            state.openapi_schema = None; state.openapi_spec_string = None
            self._queue_intermediate_message(state, "Parsed content is not a valid dictionary structure.")
            logger.error(f"Parsed content is not a dictionary. Type: {type(parsed_spec_dict)}. Input snippet: {spec_text[:200]}...")
            state.next_step = "responder"
            state.update_scratchpad_reason(tool_name, "Parsed content not a dict.")
            return state
        
        logger.debug(f"[{tool_name}] Successfully parsed spec into a dictionary. Top-level keys: {list(parsed_spec_dict.keys())}")

        spec_version_str = str(parsed_spec_dict.get("openapi", "")).strip()
        is_swagger_v2 = "swagger" in parsed_spec_dict and str(parsed_spec_dict.get("swagger", "")).strip().startswith("2")
        
        validation_performed = False
        validation_successful = False

        try:
            logger.debug(f"[{tool_name}] Detected OpenAPI version string: '{spec_version_str}', is_swagger_v2: {is_swagger_v2}")
            
            # Ensure parsed_spec_dict is not None before passing to validator
            if parsed_spec_dict is None: # Should have been caught earlier, but defensive check
                raise ValueError("parsed_spec_dict became None before validation, this should not happen.")

            if spec_version_str.startswith("3.0"):
                logger.info(f"[{tool_name}] Attempting to validate as OpenAPI v3.0.x (Version: {spec_version_str}).")
                logger.debug(f"[{tool_name}] Data type passed to openapi_v30_spec_validator: {type(parsed_spec_dict)}")
                # logger.debug(f"[{tool_name}] Data snippet: {str(parsed_spec_dict)[:1000]}...") # Potentially very long
                validated_spec_dict = openapi_v30_spec_validator.validate(parsed_spec_dict)
                parsed_spec_dict = validated_spec_dict 
                validation_performed = True; validation_successful = True
                logger.info(f"[{tool_name}] OpenAPI 3.0.x validated successfully.")
            elif spec_version_str.startswith("3.1"):
                logger.info(f"[{tool_name}] Attempting to validate as OpenAPI v3.1.x (Version: {spec_version_str}).")
                logger.debug(f"[{tool_name}] Data type passed to openapi_v31_spec_validator: {type(parsed_spec_dict)}")
                validated_spec_dict = openapi_v31_spec_validator.validate(parsed_spec_dict)
                parsed_spec_dict = validated_spec_dict
                validation_performed = True; validation_successful = True
                logger.info(f"[{tool_name}] OpenAPI 3.1.x validated successfully.")
            elif is_swagger_v2:
                logger.warning(f"[{tool_name}] Swagger 2.0 spec detected. Specific v2 validation not active. $refs may not be resolved by validator.")
            elif spec_version_str:
                 logger.warning(f"[{tool_name}] Unsupported OpenAPI version for validation: '{spec_version_str}'. $refs may not be resolved by validator.")
            else:
                logger.warning(f"[{tool_name}] Could not determine OpenAPI version for validation. $refs may not be resolved by validator.")

            state.openapi_schema = parsed_spec_dict
            state.schema_cache_key = cache_key
            state.openapi_spec_text = spec_text
            state.openapi_spec_string = None

            if validation_performed and validation_successful:
                self._queue_intermediate_message(state, "OpenAPI 3.x specification parsed and validated. Starting analysis pipeline...")
            else:
                 self._queue_intermediate_message(state, "OpenAPI specification parsed (v3 validation not applicable/performed or skipped). Starting analysis pipeline...")
            state.next_step = "process_schema_pipeline"

        except OpenAPIValidationError as val_e:
            state.openapi_schema = None; state.openapi_spec_string = None
            error_detail = str(val_e.message if hasattr(val_e, 'message') else val_e)
            if hasattr(val_e, 'instance') and hasattr(val_e, 'schema_path'):
                 error_detail_path = "->".join(map(str, val_e.schema_path)) # type: ignore
                 error_detail = f"Validation error at path '{error_detail_path}' for instance segment '{str(val_e.instance)[:100]}...': {val_e.message}"
            self._queue_intermediate_message(state, f"OpenAPI 3.x specification is invalid: {error_detail[:500]}")
            logger.error(f"[{tool_name}] OpenAPI 3.x Validation failed: {error_detail}", exc_info=False)
            state.next_step = "responder"
        except TypeError as te: # Catch the specific TypeError
            logger.error(f"[{tool_name}] TypeError during validation or processing: {te}", exc_info=True)
            logger.error(f"[{tool_name}] This often means a None value was passed to a function expecting a string, possibly from an unresolved or null $ref in the spec, or an issue within the validator library when processing the spec structure.")
            self._queue_intermediate_message(
                state, f"Error processing OpenAPI spec: A None value was encountered where a string was expected. This might be due to an issue with $ref resolution or a null value in your spec. Specific error: {te}"
            )
            state.openapi_schema = None; state.openapi_spec_string = None
            state.next_step = "responder"
        except Exception as e_general_processing:
            state.openapi_schema = None; state.openapi_spec_string = None
            self._queue_intermediate_message(state, f"Error during OpenAPI parsing or validation attempt: {type(e_general_processing).__name__} - {str(e_general_processing)[:200]}")
            logger.error(f"[{tool_name}] Unexpected error during parsing/validation: {e_general_processing}", exc_info=True)
            state.next_step = "responder"
        
        state.update_scratchpad_reason(
            tool_name,
            f"Parsing and Validation (v3.x) status: {'Success' if state.openapi_schema else 'Failed'}. Response: {state.response}",
        )
        return state

    def _generate_llm_schema_summary(self, state: BotState):
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

        validation_note = ""
        spec_version_str = str(state.openapi_schema.get("openapi", "")).strip()
        if spec_version_str.startswith("3.0") or spec_version_str.startswith("3.1"):
            validation_note = "The schema was validated for OpenAPI 3.x, so $ref pointers should be resolved."
        else:
            validation_note = "The schema was parsed; $ref pointers might be unresolved if not OpenAPI 3.x or if validation was skipped for other versions."


        summary_prompt = (
            f"Summarize the following API specification. Focus on its main purpose, "
            f"key resources/capabilities, and any mentioned authentication schemes. Be concise (around 100-150 words).\n\n"
            f"Title: {title}\nVersion: {version}\nDescription: {description[:500]}...\n"
            f"Number of paths: {num_paths}\nExample Paths (first 3):\n{paths_preview}\n\n"
            f"Note: {validation_note}\n"
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
                    "parameters": operation_details.get("parameters", []), 
                    "requestBody": operation_details.get("requestBody", {}), 
                    "responses": operation_details.get("responses", {}), 
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
        context_override: Optional[str] = None, # Corrected: removed extra ']'
    ):
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
            self._queue_intermediate_message(state, "Cannot create payload examples: No APIs identified.")
            return

        payload_descs = state.payload_descriptions or {}
        apis_to_process = []
        # ... (logic for selecting apis_to_process remains the same) ...
        if target_apis:
            apis_to_process = [api for api in state.identified_apis if api["operationId"] in target_apis]
        else:
            apis_with_payload_info = [api for api in state.identified_apis if api.get("requestBody") or any(p.get("in") in ["body", "formData"] for p in api.get("parameters", []))]
            unprocessed_apis = [api for api in apis_with_payload_info if api["operationId"] not in payload_descs]
            if unprocessed_apis:
                apis_to_process = unprocessed_apis[:MAX_PAYLOAD_SAMPLES_TO_GENERATE_INITIAL]
            else:
                apis_to_process = apis_with_payload_info[:MAX_PAYLOAD_SAMPLES_TO_GENERATE_SINGLE_PASS]

        logger.info(f"Attempting to generate payload descriptions for {len(apis_to_process)} APIs.")
        processed_count = 0
        for api_op in apis_to_process:
            # ... (prompt construction and LLM call logic remains the same, ensuring validation_note_for_payload is set correctly) ...
            op_id = api_op["operationId"]
            if op_id in payload_descs and not context_override and not target_apis and processed_count > 0:
                continue
            self._queue_intermediate_message(state, f"Generating payload example for '{op_id}'...")
            params_summary_list = []
            for p_idx, p_detail in enumerate(api_op.get("parameters", [])):
                if p_idx >= 5: params_summary_list.append("..."); break
                param_name = p_detail.get("name", "N/A"); param_in = p_detail.get("in", "N/A"); param_type = "N/A"
                if 'schema' in p_detail and isinstance(p_detail['schema'], dict):
                    param_type = p_detail['schema'].get('type', str(p_detail['schema'])) 
                    if param_type == 'array' and 'items' in p_detail['schema'] and isinstance(p_detail['schema']['items'], dict):
                        param_type = f"array of {p_detail['schema']['items'].get('type', str(p_detail['schema']['items']))}"
                params_summary_list.append(f"{param_name}({param_in}, type: {param_type})")
            params_summary_str = ", ".join(params_summary_list) if params_summary_list else "None"
            request_body_schema_str = "N/A"
            if api_op.get('requestBody') and isinstance(api_op['requestBody'], dict):
                content = api_op['requestBody'].get('content', {}); json_content = content.get('application/json', {}); schema = json_content.get('schema', {}) 
                if schema: request_body_schema_str = json.dumps(schema, indent=2)[:500] + "..."
            success_response_schema_str = "N/A"
            responses = api_op.get("responses", {});
            for status_code, resp_details in responses.items():
                if status_code.startswith("2") and isinstance(resp_details, dict):
                    content = resp_details.get("content", {}); json_content = content.get("application/json", {}); schema = json_content.get("schema", {}) 
                    if schema: success_response_schema_str = json.dumps(schema, indent=2)[:300] + "..."; break
            validation_note_for_payload = ""
            spec_version_str_payload = str(state.openapi_schema.get("openapi", "")).strip() if state.openapi_schema else ""
            if spec_version_str_payload.startswith("3.0") or spec_version_str_payload.startswith("3.1"):
                validation_note_for_payload = "The schema was validated for OpenAPI 3.x, so $ref pointers in the provided schemas below should be resolved."
            else:
                validation_note_for_payload = "The schema was parsed; $ref pointers in the provided schemas below might be unresolved if not OpenAPI 3.x or if validation was skipped for other versions."
            context_str = f" User Context: {context_override}." if context_override else ""
            prompt = (f"API Operation: {op_id} ({api_op['method']} {api_op['path']})\nSummary: {api_op.get('summary', 'N/A')}\n{context_str}\nParameters: {params_summary_str}\nNote on Schemas: {validation_note_for_payload}\nRequest Body Schema (if application/json):\n```json\n{request_body_schema_str}\n```\nSuccessful (2xx) Response Schema (sample, if application/json):\n```json\n{success_response_schema_str}\n```\n\nTask: Provide a concise, typical, and REALISTIC JSON example for the request payload (if applicable for this method and API design). Use plausible, real-world example values based on the parameter names, types, and the API schema. If a schema was just a $ref (e.g., {{\"'$ref'\": \"#/components/schemas/User\"}}) before validation, the validator should have resolved it if it was OpenAPI 3.x. Base your example on the (potentially resolved) schema provided. For example, if a field is 'email', use 'user@example.com'. If 'count', use a number like 5. Also, provide a brief description of the expected JSON response structure for a successful call, based on the schema. Focus on key fields. If no request payload is typically needed (e.g., for GET with only path/query params), state 'No request payload needed.' clearly. Format clearly:\nRequest Payload Example:\n```json\n{{\"key\": \"realistic_value\", \"another_key\": 123}}\n```\nExpected Response Structure:\nBrief description of response fields (e.g., 'Returns an object with id, name, and status. The 'status' field indicates processing outcome.').")
            try:
                description = llm_call_helper(self.worker_llm, prompt)
                payload_descs[op_id] = description; processed_count += 1
            except Exception as e:
                logger.error(f"Error generating payload description for {op_id}: {e}", exc_info=False)
                payload_descs[op_id] = f"Error generating description: {str(e)[:100]}..."
                self._queue_intermediate_message(state, f"Error creating payload example for '{op_id}': {str(e)[:100]}...")
                if "quota" in str(e).lower() or "429" in str(e):
                    logger.warning(f"Quota error during payload description for {op_id}. Stopping further payload generation for this turn.")
                    self._queue_intermediate_message(state, state.response + " Hit API limits during generation."); break
        state.payload_descriptions = payload_descs
        final_payload_msg = ""
        if processed_count > 0: final_payload_msg = f"Generated payload examples for {processed_count} API operation(s)."
        elif not apis_to_process and not target_apis: final_payload_msg = "No relevant APIs found requiring new payload examples at this time."
        elif target_apis and not apis_to_process: final_payload_msg = f"Could not find the specified API(s) ({target_apis}) to generate payload examples."
        if final_payload_msg: self._queue_intermediate_message(state, final_payload_msg)
        state.update_scratchpad_reason(tool_name, f"Payload descriptions updated for {processed_count} of {len(apis_to_process)} targeted APIs.")


    def process_schema_pipeline(
        self, state: BotState, graph_generator_func: Callable[[BotState, Optional[str]], BotState]
    ) -> BotState:
        tool_name = "process_schema_pipeline"
        # ... (rest of the method remains the same, ensuring it calls the updated sub-methods) ...
        self._queue_intermediate_message(state, "Starting API analysis pipeline...")
        state.update_scratchpad_reason(tool_name, "Starting schema pipeline.")
        if not state.openapi_schema:
            self._queue_intermediate_message(state, "Cannot run pipeline: No schema loaded (parsing or validation may have failed).")
            state.next_step = "handle_unknown"; return state
        state.schema_summary = None; state.identified_apis = []; state.payload_descriptions = {}; state.execution_graph = None
        state.graph_refinement_iterations = 0; state.plan_generation_goal = state.plan_generation_goal or "Provide a general overview workflow."
        state.scratchpad['graph_gen_attempts'] = 0; state.scratchpad['refinement_validation_failures'] = 0
        self._generate_llm_schema_summary(state)
        if state.schema_summary and ("Error generating summary: 429" in state.schema_summary or "quota" in state.schema_summary.lower()):
            logger.warning("API limit hit during schema summary. Stopping pipeline."); state.next_step = "responder"; return state
        self._identify_apis_from_schema(state)
        if not state.identified_apis:
            msg = (state.response or "") + " No API operations were identified from the schema. Cannot generate payload examples or an execution graph."
            self._queue_intermediate_message(state, msg); state.next_step = "responder"; return state
        self._generate_payload_descriptions(state)
        if any("Error generating description: 429" in desc for desc in state.payload_descriptions.values()) or any("quota" in desc.lower() for desc in state.payload_descriptions.values()):
            logger.warning("API limit hit during payload description generation.")
        state = graph_generator_func(state, state.plan_generation_goal)
        if (state.openapi_schema and state.schema_cache_key and SCHEMA_CACHE and state.execution_graph and state.next_step not in ["handle_unknown", "responder_with_error_from_pipeline"]):
            full_analysis_data = {"openapi_schema": state.openapi_schema, "schema_summary": state.schema_summary, "identified_apis": state.identified_apis, "payload_descriptions": state.payload_descriptions, "execution_graph": state.execution_graph.model_dump() if state.execution_graph and isinstance(state.execution_graph, GraphOutput) else None, "plan_generation_goal": state.plan_generation_goal}
            cached_full_analysis_key = f"{state.schema_cache_key}_full_analysis_v3_validated_or_parsed"
            save_schema_to_cache(cached_full_analysis_key, full_analysis_data)
            logger.info(f"Saved processed data and analysis to cache: {cached_full_analysis_key}")
        state.update_scratchpad_reason(tool_name, f"Schema processing pipeline completed. Next step determined by graph generation: {state.next_step}")
        return state

