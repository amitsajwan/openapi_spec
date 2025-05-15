# core_logic/spec_processor.py
import json
import logging
from typing import Any, Dict, List, Optional, Callable, Tuple
import yaml
import os
import re
import asyncio # For parallelism in payload descriptions

from models import BotState, GraphOutput
from utils import (
    llm_call_helper,
    load_cached_schema,
    save_schema_to_cache,
    get_cache_key,
    SCHEMA_CACHE,
)

logger = logging.getLogger(__name__)

MAX_PAYLOAD_SAMPLES_TO_GENERATE_INITIAL = int(os.getenv("MAX_PAYLOAD_SAMPLES_TO_GENERATE_INITIAL", "3"))
MAX_PAYLOAD_SAMPLES_TO_GENERATE_SINGLE_PASS = int(os.getenv("MAX_PAYLOAD_SAMPLES_TO_GENERATE_SINGLE_PASS", "2"))
MAX_CONCURRENT_PAYLOAD_DESC_LLMS = int(os.getenv("MAX_CONCURRENT_PAYLOAD_DESC_LLMS", "3"))


class SpecProcessor:
    """
    Handles parsing and initial analysis of OpenAPI specifications.
    Includes an LLM-based pre-cleanup step and parallel processing for some LLM-dependent tasks.
    Formal validation using openapi-spec-validator has been removed.
    """
    def __init__(self, worker_llm: Any, utility_llm: Any):
        """
        Initializes the SpecProcessor.

        Args:
            worker_llm: The primary language model for complex tasks.
            utility_llm: A potentially smaller/faster LLM for simpler tasks like cleanup or basic summaries.
        """
        if not hasattr(worker_llm, "invoke"):
            raise TypeError("worker_llm must have an 'invoke' method.")
        if not hasattr(utility_llm, "invoke"):
            raise TypeError("utility_llm must have an 'invoke' method.")
        self.worker_llm = worker_llm
        self.utility_llm = utility_llm
        logger.info("SpecProcessor initialized with worker_llm and utility_llm. Formal spec validation is disabled.")

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

    def _strip_markdown_code_block(self, text: str) -> str:
        """
        Strips markdown code block delimiters (```language ... ``` or ``` ... ```)
        from the beginning and end of a string, handling potentially nested fences iteratively.
        """
        current_text = text.strip()
        previous_text = None

        while current_text and current_text != previous_text:
            previous_text = current_text
            match = re.match(r"^\s*```(?:[a-zA-Z0-9_+-]+)?\s*([\s\S]*?)\s*```\s*$", current_text, re.DOTALL)
            if match:
                current_text = match.group(1).strip()
            else:
                lines = current_text.splitlines()
                stripped_leading = False
                if lines and lines[0].strip().startswith("```"):
                    lines.pop(0)
                    stripped_leading = True

                stripped_trailing = False
                if lines and lines[-1].strip() == "```":
                    lines.pop(-1)
                    stripped_trailing = True

                if stripped_leading or stripped_trailing:
                    current_text = "\n".join(lines).strip()
                else:
                    break
        return current_text

    def _llm_cleanup_spec_text(self, spec_text: str, tool_name_for_log: str) -> str:
        """
        Uses the utility LLM to perform minor cleanup of the OpenAPI spec text.
        Focuses on correcting common syntax issues like unquoted strings or numbers
        where strings are expected, especially for $ref, description, summary, and keys.
        Also strips markdown code blocks from the LLM's response using the iterative stripper.
        """
        logger.info(f"[{tool_name_for_log}] Attempting LLM-based cleanup of the spec text using UTILITY_LLM.")
        prompt = f"""
The following text is an OpenAPI specification, potentially in JSON or YAML format.
It might contain minor syntax errors. The most critical errors to fix are type mismatches where an integer or boolean is used instead of a string.

Your task is to:
1.  Carefully review the specification.
2.  Identify and correct ONLY minor syntax issues. Your ABSOLUTE HIGHEST PRIORITY is to fix cases where an integer or boolean is used for a field that MUST be a string according to the OpenAPI specification. Examples:
    a.  `description: 123` should become `description: "123"` (JSON) or `description: '123'` (YAML).
    b.  `summary: 456` should become `summary: "456"` or `summary: '456'`.
    c.  `$ref: 123` (an integer) is INVALID. It MUST be a string like `"$ref": "#/components/schemas/MyObject"`. If you see a numeric `$ref`, convert it to a placeholder string like `"$ref": "INVALID_REF_WAS_NUMBER_123"`.
    d.  Keys in objects/mappings MUST be strings. For example, in the `responses` object, status codes like `200` (integer) should be represented as string keys like `"200"` (JSON) or `'200'` (YAML).
    e.  Fields like `version` (e.g., `version: 1`) should be strings (e.g., `version: "1.0"`).
    f.  `operationId`, `title`, `format` (when its value indicates a string format like 'date-time'), and `type` (when its value is "string", "integer", etc.) must have string values.
3.  The output MUST be a valid YAML or JSON representation of the specification. Prefer YAML for readability if the input format is ambiguous, otherwise try to match the input format.
4.  Do NOT alter the semantic meaning or overall structure of the API specification. Only fix obvious, minor syntax and type errors, focusing on string conversions.
5.  If the input text appears to be largely correct or if you are unsure about a change that might alter semantics, return it as close to the original as possible but ensuring basic well-formedness.

Original OpenAPI specification text:
```
{spec_text}
```

Return ONLY the cleaned-up OpenAPI specification text. Do not add any explanations, apologies, or introductory/concluding remarks.
        """
        try:
            cleaned_spec_text_from_llm_raw = llm_call_helper(self.utility_llm, prompt)

            if not isinstance(cleaned_spec_text_from_llm_raw, str):
                logger.warning(f"[{tool_name_for_log}] LLM cleanup (utility) did not return a string (got {type(cleaned_spec_text_from_llm_raw)}). Using original spec text.")
                return spec_text

            cleaned_spec_text_stripped_md = self._strip_markdown_code_block(cleaned_spec_text_from_llm_raw)
            cleaned_spec_text = cleaned_spec_text_stripped_md.strip()

            if cleaned_spec_text and len(cleaned_spec_text) > 0.3 * len(spec_text):
                logger.info(f"[{tool_name_for_log}] LLM cleanup (utility) performed. Original length: {len(spec_text)}, Cleaned length: {len(cleaned_spec_text)}")
                if cleaned_spec_text != spec_text.strip():
                    logger.debug(f"[{tool_name_for_log}] Spec changed by LLM (utility).")
                return cleaned_spec_text
            else:
                logger.warning(f"[{tool_name_for_log}] LLM cleanup (utility) returned an unexpectedly short or empty response (after stripping MD). Length: {len(cleaned_spec_text)}. Using original spec text.")
                return spec_text
        except Exception as e:
            logger.error(f"[{tool_name_for_log}] Error during LLM cleanup (utility): {e}. Using original spec text.", exc_info=True)
            return spec_text

    def parse_openapi_spec(self, state: BotState) -> BotState:
        """
        Parses the OpenAPI specification. Includes an LLM pre-cleanup step and
        iterative markdown stripping. Formal validation is bypassed.
        """
        tool_name = "parse_openapi_spec"
        self._queue_intermediate_message(state, "Preprocessing and parsing OpenAPI specification...")
        state.update_scratchpad_reason(tool_name, "Attempting to preprocess and parse OpenAPI spec (formal validation disabled).")

        original_spec_text_from_state = state.openapi_spec_string

        logger.debug(f"[{tool_name}] Initial spec_text type: {type(original_spec_text_from_state)}")
        if isinstance(original_spec_text_from_state, str): logger.debug(f"[{tool_name}] Initial spec_text (first 200 chars): {original_spec_text_from_state[:200]}")
        else: logger.warning(f"[{tool_name}] Initial spec_text is not a string: {type(original_spec_text_from_state)}")

        if not original_spec_text_from_state:
            self._queue_intermediate_message(state, "No OpenAPI specification text provided."); state.update_scratchpad_reason(tool_name, "No spec text in state.")
            state.next_step = "responder"; state.openapi_spec_string = None; return state

        if not isinstance(original_spec_text_from_state, str):
            error_msg = f"Invalid type for OpenAPI specification: expected a string, but got {type(original_spec_text_from_state)}. Cannot proceed."
            logger.error(f"[{tool_name}] {error_msg}"); self._queue_intermediate_message(state, error_msg)
            state.openapi_schema = None; state.openapi_spec_string = None; state.next_step = "responder"; return state

        spec_text_initially_stripped = self._strip_markdown_code_block(original_spec_text_from_state)
        if spec_text_initially_stripped != original_spec_text_from_state.strip():
            logger.info(f"[{tool_name}] Initial markdown fences stripped from input before LLM cleanup.")
            self._queue_intermediate_message(state, "Initial markdown code blocks stripped from input.")

        self._queue_intermediate_message(state, "Attempting minor cleanup of the specification using AI...")
        spec_text_after_llm_cleanup = self._llm_cleanup_spec_text(spec_text_initially_stripped, tool_name)
        spec_text_to_parse = self._strip_markdown_code_block(spec_text_after_llm_cleanup)

        if spec_text_to_parse.strip() != original_spec_text_from_state.strip():
            if spec_text_after_llm_cleanup.strip() != spec_text_initially_stripped.strip():
                 self._queue_intermediate_message(state, "AI performed minor adjustments to the specification text before parsing.")
                 logger.info(f"[{tool_name}] Spec text was modified by LLM pre-cleanup.")
            elif spec_text_to_parse.strip() != spec_text_initially_stripped.strip():
                 logger.info(f"[{tool_name}] Spec text was modified by final stripping after LLM cleanup (LLM itself made no changes to content).")
        else:
            self._queue_intermediate_message(state, "No significant adjustments made to specification text by AI or stripping, or cleanup was skipped/failed.")
            logger.info(f"[{tool_name}] Spec text remained unchanged after all cleanup and stripping attempts.")

        if not spec_text_to_parse.strip():
            error_msg = "Specification text became empty after cleanup and stripping. Cannot parse."
            logger.error(f"[{tool_name}] {error_msg} Original spec (snippet): {original_spec_text_from_state[:200]}...")
            self._queue_intermediate_message(state, error_msg)
            state.openapi_schema = None; state.openapi_spec_string = None; state.next_step = "responder"; return state

        cache_key = get_cache_key(spec_text_to_parse); cached_full_analysis_key = f"{cache_key}_full_analysis_parsed_only_v1"

        if SCHEMA_CACHE:
            cached_schema_artifacts = load_cached_schema(cached_full_analysis_key)
            if cached_schema_artifacts and isinstance(cached_schema_artifacts, dict):
                try:
                    state.openapi_schema = cached_schema_artifacts.get("openapi_schema"); state.schema_summary = cached_schema_artifacts.get("schema_summary")
                    state.identified_apis = cached_schema_artifacts.get("identified_apis", []); state.payload_descriptions = cached_schema_artifacts.get("payload_descriptions", {})
                    graph_dict = cached_schema_artifacts.get("execution_graph")
                    if graph_dict: state.execution_graph = GraphOutput.model_validate(graph_dict) if isinstance(graph_dict, dict) else graph_dict
                    state.schema_cache_key = cache_key; state.openapi_spec_text = spec_text_to_parse; state.openapi_spec_string = None
                    logger.info(f"Loaded parsed OpenAPI data and analysis from cache: {cached_full_analysis_key}")
                    self._queue_intermediate_message(state, "OpenAPI specification and derived analysis (parsed only) loaded from cache.")
                    if state.execution_graph and isinstance(state.execution_graph, GraphOutput): state.scratchpad['graph_to_send'] = state.execution_graph.model_dump_json(indent=2)
                    state.next_step = "responder"; return state
                except Exception as e:
                    logger.warning(f"Error rehydrating state from cache (key: {cached_full_analysis_key}): {e}. Proceeding with fresh parsing.")
                    state.openapi_schema = None; state.schema_summary = None; state.identified_apis = []; state.payload_descriptions = {}; state.execution_graph = None

        parsed_spec_dict: Optional[Dict[str, Any]] = None; error_message: Optional[str] = None
        logger.debug(f"[{tool_name}] Attempting to parse spec_text (length: {len(spec_text_to_parse)}) as JSON/YAML after all cleanup.")
        try:
            final_cleaned_spec_text_for_parsing = spec_text_to_parse.strip()
            if not final_cleaned_spec_text_for_parsing:
                 raise ValueError("Cannot parse empty specification string.")

            if final_cleaned_spec_text_for_parsing.startswith("{") and final_cleaned_spec_text_for_parsing.endswith("}"):
                logger.debug(f"[{tool_name}] Trying to parse final cleaned spec as JSON."); parsed_spec_dict = json.loads(final_cleaned_spec_text_for_parsing); logger.debug(f"[{tool_name}] Successfully parsed final cleaned spec as JSON.")
            else:
                logger.debug(f"[{tool_name}] Trying to parse final cleaned spec as YAML."); parsed_spec_dict = yaml.safe_load(final_cleaned_spec_text_for_parsing); logger.debug(f"[{tool_name}] Successfully parsed final cleaned spec as YAML.")

        except (json.JSONDecodeError, yaml.YAMLError) as parse_err:
            err_type = "JSON" if isinstance(parse_err, json.JSONDecodeError) else "YAML"
            err_details = ""
            if isinstance(parse_err, json.JSONDecodeError):
                err_details = f" (at line {parse_err.lineno} column {parse_err.colno})"
            elif hasattr(parse_err, 'problem_mark'):
                err_details = f" (near line {parse_err.problem_mark.line + 1} column {parse_err.problem_mark.column + 1})"
            error_message = f"{err_type} parsing failed{err_details}: {parse_err}"
            logger.error(f"[{tool_name}] {error_message}. Original spec (snippet): {original_spec_text_from_state[:200]}... Final cleaned spec (snippet): {spec_text_to_parse[:200]}...")
        except ValueError as ve:
            error_message = str(ve); logger.error(f"[{tool_name}] {error_message}")
        except Exception as e_parse:
            error_message = f"Unexpected error during spec parsing: {type(e_parse).__name__} - {e_parse}"; logger.error(f"[{tool_name}] {error_message}", exc_info=True)

        if error_message:
            state.openapi_schema = None; state.openapi_spec_string = None; self._queue_intermediate_message(state, f"Failed to parse specification: {error_message}")
            state.next_step = "responder"; state.update_scratchpad_reason(tool_name, f"Parsing failed. Response: {state.response}"); return state

        if not parsed_spec_dict or not isinstance(parsed_spec_dict, dict):
            state.openapi_schema = None; state.openapi_spec_string = None; self._queue_intermediate_message(state, "Parsed content is not a valid dictionary structure.")
            logger.error(f"Parsed content is not a dictionary. Type: {type(parsed_spec_dict)}. Input snippet: {spec_text_to_parse[:200]}...")
            state.next_step = "responder"; state.update_scratchpad_reason(tool_name, "Parsed content not a dict."); return state

        logger.debug(f"[{tool_name}] Successfully parsed spec into a dictionary. Top-level keys: {list(parsed_spec_dict.keys())}")

        try:
            spec_version_str = str(parsed_spec_dict.get("openapi", "")).strip()
            is_swagger_v2 = "swagger" in parsed_spec_dict and str(parsed_spec_dict.get("swagger", "")).strip().startswith("2")

            logger.info(f"[{tool_name}] Spec parsed successfully. OpenAPI version string: '{spec_version_str}', Swagger 2.0 detected: {is_swagger_v2}.")
            if not spec_version_str and not is_swagger_v2:
                logger.warning(f"[{tool_name}] Could not determine OpenAPI/Swagger version from parsed content. Proceeding with caution.")

            if logger.isEnabledFor(logging.DEBUG):
                try:
                    debug_snippet = {
                        "info_title": parsed_spec_dict.get("info", {}).get("title"),
                        "info_version": parsed_spec_dict.get("info", {}).get("version"),
                        "paths_count": len(parsed_spec_dict.get("paths", {})),
                        "components_schemas_count": len(parsed_spec_dict.get("components", {}).get("schemas", {}))
                    }
                    logger.debug(f"[{tool_name}] Snippet of parsed spec structure: {json.dumps(debug_snippet, indent=2)}")
                except Exception as e_debug_log:
                    logger.debug(f"[{tool_name}] Could not create debug snippet for parsed spec: {e_debug_log}")

            state.openapi_schema = parsed_spec_dict
            state.schema_cache_key = cache_key
            state.openapi_spec_text = spec_text_to_parse
            state.openapi_spec_string = None

            self._queue_intermediate_message(state, "OpenAPI specification parsed (formal validation skipped). Starting analysis pipeline...")
            state.next_step = "process_schema_pipeline"

        except TypeError as te:
            type_error_arg = "unknown"
            try:
                if te.__cause__ and hasattr(te.__cause__, 'args') and te.__cause__.args: type_error_arg = type(te.__cause__.args[0]).__name__
            except: pass
            logger.error(f"[{tool_name}] TypeError during basic processing of parsed spec: {te}", exc_info=True)
            logger.error(f"[{tool_name}] This indicates an issue with the structure of the parsed data (e.g., got '{type_error_arg}'). Original spec snippet: {original_spec_text_from_state[:200]}... Cleaned spec snippet: {spec_text_to_parse[:200]}...")
            self._queue_intermediate_message(state, f"Error processing parsed OpenAPI spec: A data type mismatch occurred (e.g., got '{type_error_arg}' where a dict/string might be expected). Error: {te}")
            state.openapi_schema = None; state.openapi_spec_string = None; state.next_step = "responder"
        except Exception as e_general_processing:
            state.openapi_schema = None; state.openapi_spec_string = None
            self._queue_intermediate_message(state, f"Error during post-parsing setup: {type(e_general_processing).__name__} - {str(e_general_processing)[:200]}")
            logger.error(f"[{tool_name}] Unexpected error after parsing: {e_general_processing}", exc_info=True); state.next_step = "responder"

        state.update_scratchpad_reason(tool_name, f"Parsing status: {'Success' if state.openapi_schema else 'Failed'}. Formal validation skipped. Response: {state.response}"); return state

    def _generate_llm_schema_summary(self, state: BotState):
        tool_name = "_generate_llm_schema_summary"
        self._queue_intermediate_message(state, "Generating API summary (using Utility LLM)...")
        state.update_scratchpad_reason(tool_name, "Generating schema summary with Utility LLM.")
        if not state.openapi_schema:
            state.schema_summary = "Could not generate summary: No schema loaded."
            logger.warning(state.schema_summary)
            self._queue_intermediate_message(state, state.schema_summary)
            return
        spec_info = state.openapi_schema.get("info", {}); title = spec_info.get("title", "N/A"); version = spec_info.get("version", "N/A"); description = spec_info.get("description", "N/A"); num_paths = len(state.openapi_schema.get("paths", {}))
        paths_preview_list = [];
        for p, m_dict in list(state.openapi_schema.get("paths", {}).items())[:3]: methods = list(m_dict.keys()) if isinstance(m_dict, dict) else '[methods not parsable]'; paths_preview_list.append(f"  {p}: {methods}")
        paths_preview = "\n".join(paths_preview_list)

        validation_note = "The schema was parsed without formal validation. $ref pointers might be unresolved."

        summary_prompt = (f"Summarize the following API specification. Focus on its main purpose, key resources/capabilities, and any mentioned authentication schemes. Be concise (around 100-150 words).\n\nTitle: {title}\nVersion: {version}\nDescription: {description[:500]}...\nNumber of paths: {num_paths}\nExample Paths (first 3):\n{paths_preview}\n\nNote: {validation_note}\nConcise Summary:")
        try:
            state.schema_summary = llm_call_helper(self.utility_llm, summary_prompt)
            logger.info("Schema summary generated by Utility LLM.")
            self._queue_intermediate_message(state, "API summary created.")
        except Exception as e:
            logger.error(f"Error generating schema summary (utility_llm): {e}", exc_info=False)
            state.schema_summary = f"Error generating summary: {str(e)[:150]}..."
            self._queue_intermediate_message(state, state.schema_summary)
        state.update_scratchpad_reason(tool_name, f"Summary status: {'Success' if state.schema_summary and not state.schema_summary.startswith('Error') else 'Failed'}")

    def _identify_apis_from_schema(self, state: BotState):
        tool_name = "_identify_apis_from_schema"; self._queue_intermediate_message(state, "Identifying API operations..."); state.update_scratchpad_reason(tool_name, "Identifying APIs.")
        if not state.openapi_schema: state.identified_apis = []; logger.warning("No schema to identify APIs from."); self._queue_intermediate_message(state, "Cannot identify APIs: No schema loaded."); return
        apis = []; paths = state.openapi_schema.get("paths", {})
        for path_url, path_item in paths.items():
            if not isinstance(path_item, dict): logger.warning(f"Skipping non-dictionary path item at '{path_url}'"); continue
            for method, operation_details in path_item.items():
                if method.lower() not in {"get", "post", "put", "delete", "patch", "options", "head", "trace"} or not isinstance(operation_details, dict): continue
                op_id_suffix = path_url.replace('/', '_').replace('{', '').replace('}', '').strip('_'); default_op_id = f"{method.lower()}_{op_id_suffix or 'root'}"
                api_info = {"operationId": operation_details.get("operationId", default_op_id), "path": path_url, "method": method.upper(), "summary": operation_details.get("summary", ""), "description": operation_details.get("description", ""), "parameters": operation_details.get("parameters", []), "requestBody": operation_details.get("requestBody", {}), "responses": operation_details.get("responses", {})}
                apis.append(api_info)
        state.identified_apis = apis; logger.info(f"Identified {len(apis)} API operations."); self._queue_intermediate_message(state, f"Identified {len(apis)} API operations."); state.update_scratchpad_reason(tool_name, f"Identified {len(apis)} APIs.")

    async def _generate_single_payload_desc(self, api_op: Dict[str, Any], state_openapi_schema: Dict[str, Any], context_override: Optional[str]) -> Tuple[str, str]:
        op_id = api_op["operationId"]
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
                content = resp_details.get("content", {}); json_content = content.get('application/json', {}); schema = json_content.get("schema", {})
                if schema: success_response_schema_str = json.dumps(schema, indent=2)[:300] + "..."; break

        validation_note_for_payload = "The schema was parsed without formal validation. $ref pointers in the provided schemas below might be unresolved. The LLM should attempt to interpret them if possible."

        context_str = f" User Context: {context_override}." if context_override else ""
        prompt = (f"API Operation: {op_id} ({api_op['method']} {api_op['path']})\nSummary: {api_op.get('summary', 'N/A')}\n{context_str}\nParameters: {params_summary_str}\nNote on Schemas: {validation_note_for_payload}\nRequest Body Schema (if application/json):\n```json\n{request_body_schema_str}\n```\nSuccessful (2xx) Response Schema (sample, if application/json):\n```json\n{success_response_schema_str}\n```\n\nTask: Provide a concise, typical, and REALISTIC JSON example for the request payload (if applicable for this method and API design). Use plausible, real-world example values based on the parameter names, types, and the API schema. If a schema contains a $ref (e.g., {{\"'$ref'\": \"#/components/schemas/User\"}}), interpret it based on the overall specification context if possible. Base your example on the provided schema. For example, if a field is 'email', use 'user@example.com'. If 'count', use a number like 5. Also, provide a brief description of the expected JSON response structure for a successful call, based on the schema. Focus on key fields. If no request payload is typically needed (e.g., for GET with only path/query params), state 'No request payload needed.' clearly. Format clearly:\nRequest Payload Example:\n```json\n{{\"key\": \"realistic_value\", \"another_key\": 123}}\n```\nExpected Response Structure:\nBrief description of response fields (e.g., 'Returns an object with id, name, and status. The 'status' field indicates processing outcome.').")
        try:
            if hasattr(self.worker_llm, 'ainvoke'):
                 response_obj = await self.worker_llm.ainvoke(prompt)
                 description = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)
            else:
                 logger.warning(f"LLM object for op_id {op_id} does not have 'ainvoke'. Using synchronous 'invoke'. Parallelism might be limited.")
                 description = llm_call_helper(self.worker_llm, prompt)
            return op_id, description
        except Exception as e:
            logger.error(f"Error generating payload description for {op_id} (parallel task): {e}", exc_info=False)
            return op_id, f"Error generating description: {str(e)[:100]}..."

    async def _generate_payload_descriptions_parallel(self, state: BotState, target_apis: Optional[List[str]] = None, context_override: Optional[str] = None):
        tool_name = "_generate_payload_descriptions_parallel"
        self._queue_intermediate_message(state, "Creating payload and response examples (in parallel)...")
        state.update_scratchpad_reason(tool_name, f"Generating payload descriptions in parallel. Targets: {target_apis or 'subset'}. Context: {bool(context_override)}")
        if not state.identified_apis:
            logger.warning("No APIs identified, cannot generate payload descriptions."); self._queue_intermediate_message(state, "Cannot create payload examples: No APIs identified."); return

        payload_descs = state.payload_descriptions or {}
        apis_to_process_candidates = []
        if target_apis:
            apis_to_process_candidates = [api for api in state.identified_apis if api["operationId"] in target_apis]
        else:
            apis_with_payload_info = [api for api in state.identified_apis if api.get("requestBody") or any(p.get("in") in ["body", "formData"] for p in api.get("parameters", []))]
            unprocessed_apis = [api for api in apis_with_payload_info if api["operationId"] not in payload_descs]
            if unprocessed_apis:
                apis_to_process_candidates = unprocessed_apis[:MAX_PAYLOAD_SAMPLES_TO_GENERATE_INITIAL * 2]
            else:
                apis_to_process_candidates = apis_with_payload_info[:MAX_PAYLOAD_SAMPLES_TO_GENERATE_SINGLE_PASS * 2]

        apis_to_actually_process = []
        for api_op in apis_to_process_candidates:
            if api_op["operationId"] not in payload_descs or context_override or target_apis:
                apis_to_actually_process.append(api_op)

        if not apis_to_actually_process:
            self._queue_intermediate_message(state, "No new APIs require payload description generation at this time."); return

        logger.info(f"Attempting to generate payload descriptions in parallel for {len(apis_to_actually_process)} APIs.")

        tasks = []
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_PAYLOAD_DESC_LLMS)

        async def _call_with_semaphore(api_op_task: Dict[str, Any], schema_task: Dict[str, Any], context_task: Optional[str]) -> Tuple[str, str]:
            async with semaphore:
                return await self._generate_single_payload_desc(api_op_task, schema_task, context_task)

        for api_op in apis_to_actually_process:
            current_openapi_schema = state.openapi_schema if state.openapi_schema else {}
            tasks.append(_call_with_semaphore(api_op, current_openapi_schema, context_override))

        processed_count = 0
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                original_api_op = apis_to_actually_process[i]
                op_id_for_result = original_api_op["operationId"]

                if isinstance(result, Exception):
                    logger.error(f"Exception in parallel payload generation for {op_id_for_result}: {result}")
                    payload_descs[op_id_for_result] = f"Error generating description (parallel): {str(result)[:100]}..."
                    self._queue_intermediate_message(state, f"Error creating payload example for '{op_id_for_result}': {str(result)[:100]}...")
                elif isinstance(result, tuple) and len(result) == 2:
                    op_id_succ, description = result
                    payload_descs[op_id_succ] = description
                    if "Error generating description" not in description:
                        processed_count += 1
                    else:
                         self._queue_intermediate_message(state, f"Partial error for '{op_id_succ}': {description}")
                else:
                    logger.error(f"Unexpected result type from parallel payload generation for {op_id_for_result}: {type(result)}")
                    payload_descs[op_id_for_result] = "Unexpected error during generation."

        state.payload_descriptions.update(payload_descs)

        final_payload_msg = ""
        if processed_count > 0: final_payload_msg = f"Generated payload examples for {processed_count} API operation(s) (parallel execution)."
        elif not apis_to_actually_process : final_payload_msg = "No relevant APIs found requiring new payload examples at this time."
        else: final_payload_msg = "Payload generation tasks completed with no new successful descriptions."

        if final_payload_msg: self._queue_intermediate_message(state, final_payload_msg)
        state.update_scratchpad_reason(tool_name, f"Parallel payload descriptions updated for {processed_count} APIs.")

    async def process_schema_pipeline_async(self, state: BotState, graph_generator_func: Callable[[BotState, Optional[str]], BotState]) -> BotState:
        tool_name = "process_schema_pipeline_async"
        self._queue_intermediate_message(state, "Starting API analysis pipeline (async steps)...")
        state.update_scratchpad_reason(tool_name, "Starting schema pipeline with async steps.")
        if not state.openapi_schema:
            self._queue_intermediate_message(state, "Cannot run pipeline: No schema loaded (parsing may have failed).")
            state.next_step = "handle_unknown"; return state
        state.schema_summary = None; state.identified_apis = []; state.payload_descriptions = {}
        state.execution_graph = None; state.graph_refinement_iterations = 0
        state.plan_generation_goal = state.plan_generation_goal or "Provide a general overview workflow."
        state.scratchpad['graph_gen_attempts'] = 0; state.scratchpad['refinement_validation_failures'] = 0

        self._generate_llm_schema_summary(state)
        if state.schema_summary and ("Error generating summary: 429" in state.schema_summary or "quota" in state.schema_summary.lower()):
            logger.warning("API limit hit during schema summary. Stopping pipeline."); state.next_step = "responder"; return state

        self._identify_apis_from_schema(state)
        if not state.identified_apis:
            msg = (state.response or "") + " No API operations identified. Cannot generate graph."
            self._queue_intermediate_message(state, msg); state.next_step = "responder"; return state

        # This is the crucial part for payload descriptions
        await self._generate_payload_descriptions_parallel(state)
        if any("Error generating description: 429" in desc for desc in (state.payload_descriptions or {}).values()) or \
           any("quota" in desc.lower() for desc in (state.payload_descriptions or {}).values()):
            logger.warning("API limit hit during payload description generation.")
            # Decide if this should halt the pipeline or just proceed without full descriptions
            # For now, it continues, but a message is logged.

        state = graph_generator_func(state, state.plan_generation_goal)

        if (state.openapi_schema and state.schema_cache_key and SCHEMA_CACHE and \
            state.execution_graph and state.next_step not in ["handle_unknown", "responder_with_error_from_pipeline"]):
            full_analysis_data = {
                "openapi_schema": state.openapi_schema, "schema_summary": state.schema_summary,
                "identified_apis": state.identified_apis, "payload_descriptions": state.payload_descriptions,
                "execution_graph": state.execution_graph.model_dump() if isinstance(state.execution_graph, GraphOutput) else None,
                "plan_generation_goal": state.plan_generation_goal
            }
            cached_full_analysis_key = f"{state.schema_cache_key}_full_analysis_parsed_only_v1"
            save_schema_to_cache(cached_full_analysis_key, full_analysis_data)
            logger.info(f"Saved processed data and analysis (parsed only) to cache: {cached_full_analysis_key}")

        state.update_scratchpad_reason(tool_name, f"Schema processing pipeline completed. Next step: {state.next_step}")
        return state

    # MODIFIED: process_schema_pipeline is now async
    async def process_schema_pipeline(self, state: BotState, graph_generator_func: Callable[[BotState, Optional[str]], BotState]) -> BotState:
        """
        Asynchronously processes the schema pipeline.
        This method is now async and directly awaits the async helper.
        """
        tool_name = "process_schema_pipeline"
        try:
            logger.info(f"[{tool_name}] Starting asynchronous schema processing pipeline.")
            # Directly await the async processing method
            state = await self.process_schema_pipeline_async(state, graph_generator_func)
            logger.info(f"[{tool_name}] Asynchronous schema processing pipeline completed. Next step: {state.next_step}")

        except Exception as e:
            # This catch block is a general fallback for unexpected errors during the async execution.
            # Specific errors within process_schema_pipeline_async should be handled there.
            logger.error(f"[{tool_name}] Unexpected error during async schema processing: {e}", exc_info=True)
            self._queue_intermediate_message(state, f"Unexpected critical error during schema processing: {str(e)[:150]}")
            state.next_step = "handle_unknown" # Route to a safe error handler

        return state
