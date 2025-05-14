# core_logic.py
import json
import logging
import asyncio 
from typing import Any, Dict, List, Optional
import yaml
import os 

from models import BotState, GraphOutput, Node, InputMapping, OutputMapping 
from utils import (
    llm_call_helper, load_cached_schema, save_schema_to_cache,
    get_cache_key, check_for_cycles, parse_llm_json_output_with_model,
    SCHEMA_CACHE
)
from pydantic import ValidationError as PydanticValidationError # Alias to avoid confusion
from api_executor import APIExecutor

# Import specific validators for different OpenAPI versions
from openapi_spec_validator import (
    openapi_v2_spec_validator, 
    openapi_v30_spec_validator, 
    openapi_v31_spec_validator
)
# Import the specific exception class from its module path
from openapi_spec_validator.validation.exceptions import OpenAPIValidationError 

logger = logging.getLogger(__name__)

# --- Configurable Limits ---
MAX_PAYLOAD_SAMPLES_TO_GENERATE_INITIAL = int(os.getenv("MAX_PAYLOAD_SAMPLES_TO_GENERATE_INITIAL", "3")) 
MAX_PAYLOAD_SAMPLES_TO_GENERATE_SINGLE_PASS = int(os.getenv("MAX_PAYLOAD_SAMPLES_TO_GENERATE_SINGLE_PASS", "2"))
MAX_APIS_IN_PROMPT_SUMMARY_SHORT = int(os.getenv("MAX_APIS_IN_PROMPT_SUMMARY_SHORT", "10"))
MAX_APIS_IN_PROMPT_SUMMARY_LONG = int(os.getenv("MAX_APIS_IN_PROMPT_SUMMARY_LONG", "20")) 
MAX_TOTAL_APIS_THRESHOLD_FOR_TRUNCATION_SHORT = int(os.getenv("MAX_TOTAL_APIS_THRESHOLD_FOR_TRUNCATION_SHORT", "15"))
MAX_TOTAL_APIS_THRESHOLD_FOR_TRUNCATION_LONG = int(os.getenv("MAX_TOTAL_APIS_THRESHOLD_FOR_TRUNCATION_LONG", "25"))


class OpenAPICoreLogic:
    def __init__(self, worker_llm: Any, api_executor_instance: APIExecutor):
        if not hasattr(worker_llm, 'invoke'):
            raise TypeError("worker_llm must have an 'invoke' method.")
        if not isinstance(api_executor_instance, APIExecutor):
            logger.warning("api_executor_instance does not seem to be a valid APIExecutor. Workflow execution might fail.")
        self.worker_llm = worker_llm
        self.api_executor = api_executor_instance
        logger.info("OpenAPICoreLogic initialized with worker_llm and api_executor.")

    def parse_openapi_spec(self, state: BotState) -> BotState:
        tool_name = "parse_openapi_spec"
        state.response = "Parsing and validating OpenAPI specification..." 
        state.update_scratchpad_reason(tool_name, "Attempting to parse and validate OpenAPI spec.")
        spec_text = state.openapi_spec_string
        
        if not spec_text:
            state.response = "No OpenAPI specification text provided."
            state.update_scratchpad_reason(tool_name, "No spec text in state.")
            state.next_step = "responder"
            state.openapi_spec_string = None 
            return state

        cache_key = get_cache_key(spec_text)
        # Using a more descriptive cache key for the fully processed (validated) schema
        cached_full_analysis_key = f"{cache_key}_full_analysis_validated" 
        cached_schema_artifacts = load_cached_schema(cached_full_analysis_key)

        if cached_schema_artifacts and isinstance(cached_schema_artifacts, dict):
            try:
                state.openapi_schema = cached_schema_artifacts.get('openapi_schema') 
                state.schema_summary = cached_schema_artifacts.get('schema_summary')
                state.identified_apis = cached_schema_artifacts.get('identified_apis', [])
                state.payload_descriptions = cached_schema_artifacts.get('payload_descriptions', {})
                graph_dict = cached_schema_artifacts.get('execution_graph')
                if graph_dict:
                    state.execution_graph = GraphOutput.model_validate(graph_dict) if isinstance(graph_dict, dict) else graph_dict
                
                state.schema_cache_key = cache_key 
                state.openapi_spec_text = spec_text 
                state.openapi_spec_string = None 

                logger.info(f"Loaded fully analyzed and validated OpenAPI data from cache: {cached_full_analysis_key}")
                state.response = "OpenAPI specification and derived analysis (validated) loaded from cache."
                if state.execution_graph and isinstance(state.execution_graph, GraphOutput):
                     state.scratchpad['graph_to_send'] = state.execution_graph.model_dump_json(indent=2)
                state.next_step = "responder" 
                return state
            except Exception as e:
                logger.warning(f"Error rehydrating state from cached full analysis (key: {cached_full_analysis_key}): {e}. Proceeding with fresh parsing and validation.")
                state.openapi_schema = None; state.schema_summary = None; state.identified_apis = []; state.payload_descriptions = {}; state.execution_graph = None

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
            state.response = f"Failed to parse specification: {error_message}"
            logger.error(f"Parsing failed: {error_message}. Input snippet: {spec_text[:200]}...")
            state.next_step = "responder"
            state.update_scratchpad_reason(tool_name, f"Parsing failed. Response: {state.response}")
            return state

        if not parsed_spec_dict or not isinstance(parsed_spec_dict, dict):
            state.openapi_schema = None
            state.openapi_spec_string = None
            state.response = "Parsed content is not a valid dictionary structure."
            logger.error(f"Parsed content is not a dictionary. Type: {type(parsed_spec_dict)}. Input snippet: {spec_text[:200]}...")
            state.next_step = "responder"
            state.update_scratchpad_reason(tool_name, "Parsed content not a dict.")
            return state

        try:
            spec_version_str = parsed_spec_dict.get('openapi', parsed_spec_dict.get('swagger'))
            
            # The .validate() methods raise an error on failure and return None on success.
            # They operate on the parsed_spec_dict, resolving references internally for validation.
            if spec_version_str and spec_version_str.startswith('2.'):
                logger.info(f"Attempting to validate as OpenAPI v2 (Swagger) specification (Version: {spec_version_str}).")
                openapi_v2_spec_validator.validate(parsed_spec_dict)
            elif spec_version_str and spec_version_str.startswith('3.0'):
                logger.info(f"Attempting to validate as OpenAPI v3.0.x specification (Version: {spec_version_str}).")
                openapi_v30_spec_validator.validate(parsed_spec_dict)
            elif spec_version_str and spec_version_str.startswith('3.1'):
                logger.info(f"Attempting to validate as OpenAPI v3.1.x specification (Version: {spec_version_str}).")
                openapi_v31_spec_validator.validate(parsed_spec_dict)
            else:
                logger.warning(f"Unknown or unsupported OpenAPI version string: '{spec_version_str}'. Attempting with v3.0 validator as a default.")
                openapi_v30_spec_validator.validate(parsed_spec_dict) 
            
            # If validation passed, parsed_spec_dict is the (effectively) dereferenced schema for our purposes.
            # The validator resolves refs to perform validation.
            state.openapi_schema = parsed_spec_dict 
            state.schema_cache_key = cache_key 
            state.openapi_spec_text = spec_text 
            state.openapi_spec_string = None 

            logger.info("Successfully parsed and validated OpenAPI spec (references resolved for validation).")
            state.response = "OpenAPI specification parsed and validated. Starting analysis pipeline..."
            state.next_step = "process_schema_pipeline"

        except OpenAPIValidationError as val_e: 
            state.openapi_schema = None 
            state.openapi_spec_string = None
            error_detail = str(val_e.message if hasattr(val_e, 'message') else val_e)
            # Provide more context if available from the error object
            if hasattr(val_e, 'instance') and hasattr(val_e, 'schema_path'):
                 error_detail_path = "->".join(map(str, val_e.schema_path))
                 error_detail = f"Validation error at path '{error_detail_path}' for instance segment '{str(val_e.instance)[:100]}...': {val_e.message}"
            
            state.response = f"OpenAPI specification is invalid: {error_detail[:500]}"
            logger.error(f"OpenAPI Validation failed: {error_detail}", exc_info=False) 
            state.next_step = "responder"
        except Exception as e_general_processing: 
            state.openapi_schema = None
            state.openapi_spec_string = None
            state.response = f"Error during OpenAPI validation/processing: {str(e_general_processing)[:200]}"
            logger.error(f"Unexpected error during validation/processing: {e_general_processing}", exc_info=True)
            state.next_step = "responder"
        
        state.update_scratchpad_reason(tool_name, f"Parsing & Validation status: {'Success' if state.openapi_schema else 'Failed'}. Response: {state.response}")
        return state

    def _generate_llm_schema_summary(self, state: BotState):
        tool_name = "_generate_llm_schema_summary"; state.response = "Generating API summary..."; state.update_scratchpad_reason(tool_name, "Generating schema summary.")
        if not state.openapi_schema: state.schema_summary = "Could not generate summary: No schema loaded."; logger.warning(state.schema_summary); state.response = state.schema_summary; return
        spec_info = state.openapi_schema.get('info', {}); title = spec_info.get('title', 'N/A'); version = spec_info.get('version', 'N/A'); description = spec_info.get('description', 'N/A'); num_paths = len(state.openapi_schema.get('paths', {}))
        paths_preview_list = []
        for p, m_dict in list(state.openapi_schema.get('paths', {}).items())[:3]: methods = list(m_dict.keys()) if isinstance(m_dict, dict) else '[methods not parsable]'; paths_preview_list.append(f"  {p}: {methods}")
        paths_preview = "\n".join(paths_preview_list)
        summary_prompt = (f"Summarize the following API specification. Focus on its main purpose, key resources/capabilities, and any mentioned authentication schemes. Be concise (around 100-150 words).\n\nTitle: {title}\nVersion: {version}\nDescription: {description[:500]}...\nNumber of paths: {num_paths}\nExample Paths (first 3):\n{paths_preview}\n\nConcise Summary:")
        try: state.schema_summary = llm_call_helper(self.worker_llm, summary_prompt); logger.info("Schema summary generated."); state.response = "API summary created."
        except Exception as e: logger.error(f"Error generating schema summary: {e}", exc_info=False); state.schema_summary = f"Error generating summary: {str(e)[:150]}..."; state.response = state.schema_summary
        state.update_scratchpad_reason(tool_name, f"Summary status: {'Success' if state.schema_summary and not state.schema_summary.startswith('Error') else 'Failed'}")

    def _identify_apis_from_schema(self, state: BotState):
        tool_name = "_identify_apis_from_schema"; state.response = "Identifying API operations..."; state.update_scratchpad_reason(tool_name, "Identifying APIs.")
        if not state.openapi_schema: state.identified_apis = []; logger.warning("No schema to identify APIs from."); state.response = "Cannot identify APIs: No schema loaded."; return
        apis = []; paths = state.openapi_schema.get('paths', {})
        for path_url, path_item in paths.items():
            if not isinstance(path_item, dict): logger.warning(f"Skipping non-dictionary path item at '{path_url}'"); continue
            for method, operation_details in path_item.items():
                if method.lower() not in {'get', 'post', 'put', 'delete', 'patch', 'options', 'head', 'trace'} or not isinstance(operation_details, dict): continue
                op_id_suffix = path_url.replace('/', '_').replace('{', '').replace('}', '').strip('_'); default_op_id = f"{method.lower()}_{op_id_suffix or 'root'}" 
                api_info = {'operationId': operation_details.get('operationId', default_op_id), 'path': path_url, 'method': method.upper(), 'summary': operation_details.get('summary', ''), 'description': operation_details.get('description', ''), 'parameters': operation_details.get('parameters', []), 'requestBody': operation_details.get('requestBody', {}), 'responses': operation_details.get('responses', {})}
                apis.append(api_info)
        state.identified_apis = apis; logger.info(f"Identified {len(apis)} API operations."); state.response = f"Identified {len(apis)} API operations."; state.update_scratchpad_reason(tool_name, f"Identified {len(apis)} APIs.")

    def _generate_payload_descriptions(self, state: BotState, target_apis: Optional[List[str]] = None, context_override: Optional[str] = None):
        tool_name = "_generate_payload_descriptions"; state.response = "Creating payload and response examples..."; state.update_scratchpad_reason(tool_name, f"Generating payload descriptions. Targets: {target_apis or 'subset'}. Context: {bool(context_override)}")
        if not state.identified_apis: logger.warning("No APIs identified, cannot generate payload descriptions."); state.response = "Cannot create payload examples: No APIs identified."; return
        payload_descs = state.payload_descriptions or {} 
        if target_apis: apis_to_process = [api for api in state.identified_apis if api['operationId'] in target_apis]
        else: 
            apis_with_payload_info = [api for api in state.identified_apis if api.get('requestBody') or any(p.get('in') in ['body', 'formData'] for p in api.get('parameters', []))]; unprocessed_apis = [api for api in apis_with_payload_info if api['operationId'] not in payload_descs]
            if unprocessed_apis: apis_to_process = unprocessed_apis[:MAX_PAYLOAD_SAMPLES_TO_GENERATE_INITIAL]
            else: apis_to_process = apis_with_payload_info[:MAX_PAYLOAD_SAMPLES_TO_GENERATE_SINGLE_PASS]
        logger.info(f"Attempting to generate payload descriptions for {len(apis_to_process)} APIs."); processed_count = 0
        for api_op in apis_to_process:
            op_id = api_op['operationId']
            if op_id in payload_descs and not context_override and not target_apis and processed_count > 0: continue
            state.response = f"Generating payload example for '{op_id}'..." 
            params_summary_list = []; 
            for p_idx, p_detail in enumerate(api_op.get('parameters', [])):
                if p_idx >= 5: params_summary_list.append("..."); break
                param_name = p_detail.get('name', 'N/A'); param_in = p_detail.get('in', 'N/A'); param_type = "N/A"
                if 'schema' in p_detail and isinstance(p_detail['schema'], dict): param_type = p_detail['schema'].get('type', 'object'); 
                if param_type == 'array' and 'items' in p_detail['schema'] and isinstance(p_detail['schema']['items'], dict): param_type = f"array of {p_detail['schema']['items'].get('type', 'object')}"
                params_summary_list.append(f"{param_name}({param_in}, type: {param_type})")
            params_summary_str = ", ".join(params_summary_list) if params_summary_list else "None"
            
            request_body_schema_str = "N/A"
            if api_op.get('requestBody') and isinstance(api_op['requestBody'], dict): 
                content = api_op['requestBody'].get('content', {}); 
                json_content = content.get('application/json', {}); 
                schema = json_content.get('schema', {}); 
                if schema: request_body_schema_str = json.dumps(schema, indent=2)[:500] + "..." 
            
            success_response_schema_str = "N/A"; responses = api_op.get('responses', {})
            for status_code, resp_details in responses.items():
                if status_code.startswith('2') and isinstance(resp_details, dict): 
                    content = resp_details.get('content', {}); 
                    json_content = content.get('application/json', {}); 
                    schema = json_content.get('schema', {}); 
                    if schema: success_response_schema_str = json.dumps(schema, indent=2)[:300] + "..."; break 
            
            context_str = f" User Context: {context_override}." if context_override else ""
            prompt = (
                f"API Operation: {op_id} ({api_op['method']} {api_op['path']})\n"
                f"Summary: {api_op.get('summary', 'N/A')}\n{context_str}\n"
                f"Parameters: {params_summary_str}\n"
                f"Request Body Schema (if application/json, effectively resolved for validation):\n```json\n{request_body_schema_str}\n```\n"
                f"Successful (2xx) Response Schema (sample, if application/json, effectively resolved for validation):\n```json\n{success_response_schema_str}\n```\n\n"
                f"Task: Provide a concise, typical, and REALISTIC JSON example for the request payload (if applicable for this method and API design). "
                f"Use plausible, real-world example values based on the parameter names, types, and the API schema (which has had its references resolved for validation purposes). For example, if a field is 'email', use 'user@example.com'. If 'count', use a number like 5. "
                f"Also, provide a brief description of the expected JSON response structure for a successful call, based on the schema. "
                f"Focus on key fields. If no request payload is typically needed (e.g., for GET with only path/query params), state 'No request payload needed.' clearly. "
                f"Format clearly:\n"
                f"Request Payload Example:\n```json\n{{\"key\": \"realistic_value\", \"another_key\": 123}}\n```\n"
                f"Expected Response Structure:\nBrief description of response fields (e.g., 'Returns an object with id, name, and status. The 'status' field indicates processing outcome.')."
            )
            try:
                description = llm_call_helper(self.worker_llm, prompt)
                payload_descs[op_id] = description; processed_count += 1
            except Exception as e:
                logger.error(f"Error generating payload description for {op_id}: {e}", exc_info=False)
                payload_descs[op_id] = f"Error generating description: {str(e)[:100]}..."
                state.response = f"Error creating payload example for '{op_id}': {str(e)[:100]}..."
                if "quota" in str(e).lower() or "429" in str(e): logger.warning(f"Quota error during payload description for {op_id}. Stopping further payload generation for this turn."); state.response += " Hit API limits during generation."; break 
        state.payload_descriptions = payload_descs
        if processed_count > 0: state.response = f"Generated payload examples for {processed_count} API operation(s)."
        elif not apis_to_process and not target_apis: state.response = "No relevant APIs found requiring new payload examples at this time."
        elif target_apis and not apis_to_process: state.response = f"Could not find the specified API(s) ({target_apis}) to generate payload examples."
        state.update_scratchpad_reason(tool_name, f"Payload descriptions updated for {processed_count} of {len(apis_to_process)} targeted APIs.")

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
        num_apis_to_summarize = MAX_APIS_IN_PROMPT_SUMMARY_LONG
        truncate_threshold = MAX_TOTAL_APIS_THRESHOLD_FOR_TRUNCATION_LONG

        for idx, api in enumerate(state.identified_apis):
            if idx >= num_apis_to_summarize and len(state.identified_apis) > truncate_threshold:
                api_summaries_for_prompt.append(f"- ...and {len(state.identified_apis) - num_apis_to_summarize} more operations.")
                break
            
            likely_confirmation = api['method'].upper() in ["POST", "PUT", "DELETE", "PATCH"]
            params_str_parts = []
            if api.get('parameters'):
                for p_idx, p_detail in enumerate(api['parameters']):
                    if p_idx >= 3: params_str_parts.append("..."); break 
                    param_name = p_detail.get('name', 'N/A')
                    param_in = p_detail.get('in', 'N/A')
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

            api_summaries_for_prompt.append(
                f"- operationId: {api['operationId']} ({api['method']} {api['path']}), "
                f"summary: {api.get('summary', 'N/A')[:80]}. {params_str}{req_body_info} " 
                f"likely_requires_confirmation: {'yes' if likely_confirmation else 'no'}"
            )
        apis_str = "\n".join(api_summaries_for_prompt)
        feedback_str = f"Refinement Feedback: {state.graph_regeneration_reason}" if state.graph_regeneration_reason else ""
        
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
            graph_output_candidate = parse_llm_json_output_with_model(llm_response, expected_model=GraphOutput)

            if graph_output_candidate:
                if not any(node.operationId == "START_NODE" for node in graph_output_candidate.nodes) or \
                   not any(node.operationId == "END_NODE" for node in graph_output_candidate.nodes):
                    logger.error("LLM generated graph is missing START_NODE or END_NODE.")
                    state.graph_regeneration_reason = "Generated graph missing START_NODE or END_NODE. Please ensure they are included."
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

            error_msg = "LLM failed to produce a valid GraphOutput JSON, or it was structurally incomplete (e.g., missing START/END nodes)."
            logger.error(error_msg + f" Raw LLM output snippet: {llm_response[:300]}...")
            state.response = "Failed to generate a valid execution graph (AI output format, structure, or missing critical nodes like START/END)."
            state.execution_graph = None 
            state.graph_regeneration_reason = state.graph_regeneration_reason or "LLM output was not a valid GraphOutput object or missed key structural elements."
            
            current_attempts = state.scratchpad.get('graph_gen_attempts', 0)
            if current_attempts < 1: 
                state.scratchpad['graph_gen_attempts'] = current_attempts + 1
                logger.info("Retrying initial graph generation once due to validation/parsing failure.")
                state.next_step = "_generate_execution_graph" 
            else:
                logger.error("Max initial graph generation attempts reached. Routing to handle_unknown.")
                state.next_step = "handle_unknown" 
                state.scratchpad['graph_gen_attempts'] = 0 

        except Exception as e:
            logger.error(f"Error during graph generation LLM call or processing: {e}", exc_info=False)
            state.response = f"Error generating graph: {str(e)[:150]}..."
            state.execution_graph = None
            state.graph_regeneration_reason = f"LLM call/processing error: {str(e)[:100]}..."
            state.next_step = "handle_unknown" 

        state.update_scratchpad_reason(tool_name, f"Graph gen status: {'Success' if state.execution_graph else 'Failed'}. Resp: {state.response}")
        return state

    def process_schema_pipeline(self, state: BotState) -> BotState:
        tool_name = "process_schema_pipeline"; state.response = "Starting API analysis pipeline..."; state.update_scratchpad_reason(tool_name, "Starting schema pipeline.")
        if not state.openapi_schema: 
            state.response = "Cannot run pipeline: No valid (or validated) schema loaded."
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
            state.response = state.schema_summary
            state.next_step = "responder"
            return state
            
        self._identify_apis_from_schema(state) 
        if not state.identified_apis: 
            state.response = (state.response or "") + " No API operations were identified from the schema. Cannot generate payload examples or an execution graph."
            state.next_step = "responder"
            return state
            
        self._generate_payload_descriptions(state) 
        if any("Error generating description: 429" in desc for desc in state.payload_descriptions.values()) or \
           any("quota" in desc.lower() for desc in state.payload_descriptions.values()):
            logger.warning("API limit hit during payload description generation.")
            if "Hit API limits" not in (state.response or ""):
                 state.response = (state.response or "") + " Partial success: Hit API limits while generating some payload examples."

        self._generate_execution_graph(state, goal=state.plan_generation_goal) 
        
        if state.openapi_schema and state.schema_cache_key and SCHEMA_CACHE and \
           state.execution_graph and state.next_step not in ["handle_unknown", "responder_with_error_from_pipeline"]: 
            
            full_analysis_data = {
                'openapi_schema': state.openapi_schema, 
                'schema_summary': state.schema_summary, 
                'identified_apis': state.identified_apis, 
                'payload_descriptions': state.payload_descriptions, 
                'execution_graph': state.execution_graph.model_dump() if state.execution_graph and isinstance(state.execution_graph, GraphOutput) else None, 
                'plan_generation_goal': state.plan_generation_goal
            }
            cached_full_analysis_key = f"{state.schema_cache_key}_full_analysis_validated" # reflect that schema is validated
            save_schema_to_cache(cached_full_analysis_key, full_analysis_data)
            logger.info(f"Saved fully analyzed and validated data to cache: {cached_full_analysis_key}")
            
        state.update_scratchpad_reason(tool_name, f"Schema processing pipeline initiated. Next step determined by _generate_execution_graph: {state.next_step}")
        return state

    def verify_graph(self, state: BotState) -> BotState:
        tool_name = "verify_graph"; state.response = "Verifying API workflow graph..."; state.update_scratchpad_reason(tool_name, "Verifying graph structure and integrity.")
        if not state.execution_graph or not isinstance(state.execution_graph, GraphOutput): state.response = state.response or "No execution graph to verify (possibly due to generation error or wrong type)."; state.graph_regeneration_reason = state.graph_regeneration_reason or "No graph was generated to verify."; logger.warning(f"verify_graph: No graph found or invalid type. Reason: {state.graph_regeneration_reason}. Routing to _generate_execution_graph for regeneration."); state.next_step = "_generate_execution_graph"; return state
        issues = []
        try:
            # Validate against Pydantic model
            GraphOutput.model_validate(state.execution_graph.model_dump()) 
            # Check for cycles
            is_dag, cycle_msg = check_for_cycles(state.execution_graph)
            if not is_dag: issues.append(cycle_msg or "Graph contains cycles.")
            # Check for START_NODE and END_NODE presence
            node_ids = {node.effective_id for node in state.execution_graph.nodes}
            if "START_NODE" not in node_ids: issues.append("START_NODE is missing.")
            if "END_NODE" not in node_ids: issues.append("END_NODE is missing.")
            # Check START_NODE/END_NODE connectivity
            if "START_NODE" in node_ids:
                start_outgoing = any(edge.from_node == "START_NODE" for edge in state.execution_graph.edges); start_incoming = any(edge.to_node == "START_NODE" for edge in state.execution_graph.edges)
                if not start_outgoing and len(state.execution_graph.nodes) > 2 : issues.append("START_NODE has no outgoing edges to actual API operations.")
                if start_incoming: issues.append("START_NODE should not have incoming edges.")
            if "END_NODE" in node_ids:
                end_incoming = any(edge.to_node == "END_NODE" for edge in state.execution_graph.edges); end_outgoing = any(edge.from_node == "END_NODE" for edge in state.execution_graph.edges)
                if not end_incoming and len(state.execution_graph.nodes) > 2: issues.append("END_NODE has no incoming edges from actual API operations.")
                if end_outgoing: issues.append("END_NODE should not have outgoing edges.")
            # Check if API nodes have method and path
            for node in state.execution_graph.nodes:
                if node.effective_id.upper() not in ["START_NODE", "END_NODE"]: 
                    if not node.method or not node.path: issues.append(f"Node '{node.effective_id}' is missing 'method' or 'path', required for execution.")
        except PydanticValidationError as ve: # Catch Pydantic validation errors
            logger.error(f"Graph Pydantic validation failed during verify_graph: {ve}"); issues.append(f"Graph structure is invalid (Pydantic): {str(ve)[:200]}...") 
        except Exception as e: 
            logger.error(f"Unexpected error during graph verification: {e}", exc_info=True); issues.append(f"An unexpected error occurred during verification: {str(e)[:100]}.")
        
        if not issues:
            state.response = "Graph verification successful (Structure, DAG, START/END nodes, basic execution fields)."; state.update_scratchpad_reason(tool_name, "Graph verification successful."); logger.info("Graph verification successful."); state.graph_regeneration_reason = None; state.scratchpad['refinement_validation_failures'] = 0 
            try: state.scratchpad['graph_to_send'] = state.execution_graph.model_dump_json(indent=2); logger.info("Graph marked to be sent to UI after verification.")
            except Exception as e: logger.error(f"Error serializing graph for sending after verification: {e}")
            logger.info("Graph verified. Proceeding to describe graph."); state.next_step = "describe_graph"
            if state.input_is_spec: 
                api_title = state.openapi_schema.get('info', {}).get('title', 'the API') if state.openapi_schema else 'the API'
                state.response = (f"Successfully processed the OpenAPI specification for '{api_title}'. Identified {len(state.identified_apis)} API operations, generated example payloads, and created an API workflow graph with {len(state.execution_graph.nodes)} steps. The graph is verified. You can now ask questions, request specific plan refinements, or try to execute the workflow.")
                state.input_is_spec = False 
        else: 
            error_details = " ".join(issues); state.response = f"Graph verification failed: {error_details}."; state.graph_regeneration_reason = f"Verification failed: {error_details}."; logger.warning(f"Graph verification failed: {error_details}.")
            if state.graph_refinement_iterations < state.max_refinement_iterations: 
                logger.info(f"Verification failed. Attempting graph refinement (iteration {state.graph_refinement_iterations + 1})."); 
                state.next_step = "refine_api_graph"
            else: 
                logger.warning("Max refinement iterations reached, but graph still has verification issues. Attempting full regeneration."); 
                state.next_step = "_generate_execution_graph"; state.graph_refinement_iterations = 0; state.scratchpad['graph_gen_attempts'] = 0 
        state.update_scratchpad_reason(tool_name, f"Verification result: {state.response[:200]}...")
        return state

    def refine_api_graph(self, state: BotState) -> BotState:
        tool_name = "refine_api_graph"; iteration = state.graph_refinement_iterations + 1; state.response = f"Refining API workflow graph (Attempt {iteration}/{state.max_refinement_iterations})..."; state.update_scratchpad_reason(tool_name, f"Refining graph. Iteration: {iteration}. Reason: {state.graph_regeneration_reason or 'General refinement request.'}")
        if not state.execution_graph or not isinstance(state.execution_graph, GraphOutput): state.response = "No graph to refine or invalid graph type. Please generate a graph first."; logger.warning("refine_api_graph: No execution_graph found or invalid type."); state.next_step = "_generate_execution_graph"; return state
        if iteration > state.max_refinement_iterations: state.response = (f"Max refinement iterations ({state.max_refinement_iterations}) reached. Using current graph (description: {state.execution_graph.description or 'N/A'}). Please try a new goal or manually edit if needed."); logger.warning("Max refinement iterations reached. Proceeding with current graph."); state.next_step = "describe_graph"; return state
        try: current_graph_json = state.execution_graph.model_dump_json(indent=2)
        except Exception as e: logger.error(f"Error serializing current graph for refinement prompt: {e}"); state.response = "Error preparing current graph for refinement. Cannot proceed."; state.next_step = "handle_unknown"; return state
        
        api_summaries_for_prompt = []
        num_apis_to_summarize = MAX_APIS_IN_PROMPT_SUMMARY_SHORT; truncate_threshold = MAX_TOTAL_APIS_THRESHOLD_FOR_TRUNCATION_SHORT
        for idx, api in enumerate(state.identified_apis): 
            if idx >= num_apis_to_summarize and len(state.identified_apis) > truncate_threshold: api_summaries_for_prompt.append(f"- ...and {len(state.identified_apis) - num_apis_to_summarize} more operations."); break
            likely_confirmation = api['method'].upper() in ["POST", "PUT", "DELETE", "PATCH"]
            api_summaries_for_prompt.append(f"- opId: {api['operationId']} ({api['method']} {api['path']}), summary: {api.get('summary', 'N/A')[:70]}, confirm: {'yes' if likely_confirmation else 'no'}")
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
            refined_graph_candidate = parse_llm_json_output_with_model(llm_response_str, expected_model=GraphOutput)
            if refined_graph_candidate:
                logger.info(f"Refinement attempt (iter {iteration}) produced a structurally valid GraphOutput.")
                state.execution_graph = refined_graph_candidate; refinement_summary = refined_graph_candidate.refinement_summary or "AI provided no specific summary for this refinement."; state.update_scratchpad_reason(tool_name, f"LLM Refinement Summary (Iter {iteration}): {refinement_summary}")
                state.graph_refinement_iterations = iteration; state.response = f"Graph refined (Iteration {iteration}). Summary: {refinement_summary}"; state.graph_regeneration_reason = None; state.scratchpad['refinement_validation_failures'] = 0; state.next_step = "verify_graph" 
            else:
                error_msg = "LLM refinement failed to produce a GraphOutput JSON that is valid or self-consistent."; logger.error(error_msg + f" Raw LLM output snippet for refinement: {llm_response_str[:300]}..."); state.response = f"Error during graph refinement (iteration {iteration}): AI output was invalid. Will retry refinement or regenerate graph."; state.graph_regeneration_reason = state.graph_regeneration_reason or "LLM output for refinement was not a valid GraphOutput object or had structural issues."
                state.scratchpad['refinement_validation_failures'] = state.scratchpad.get('refinement_validation_failures', 0) + 1
                if iteration < state.max_refinement_iterations:
                    if state.scratchpad['refinement_validation_failures'] >= 2: logger.warning(f"Multiple consecutive refinement validation failures (iter {iteration}). Escalating to full graph regeneration."); state.response += " Attempting full regeneration due to persistent refinement issues."; state.next_step = "_generate_execution_graph"; state.graph_refinement_iterations = 0; state.scratchpad['refinement_validation_failures'] = 0; state.scratchpad['graph_gen_attempts'] = 0 
                    else: state.next_step = "refine_api_graph" 
                else: logger.warning(f"Max refinement iterations reached after LLM output error during refinement. Describing last valid graph or failing."); state.next_step = "describe_graph" 
        except Exception as e:
            logger.error(f"Error during graph refinement LLM call or processing (iter {iteration}): {e}", exc_info=False); state.response = f"Error refining graph (iter {iteration}): {str(e)[:150]}..."; state.graph_regeneration_reason = state.graph_regeneration_reason or f"Refinement LLM call/processing error (iter {iteration}): {str(e)[:100]}..."
            if iteration < state.max_refinement_iterations: state.next_step = "refine_api_graph" 
            else: logger.warning(f"Max refinement iterations reached after exception. Describing graph or failing."); state.next_step = "describe_graph"
        return state

    def describe_graph(self, state: BotState) -> BotState:
        tool_name = "describe_graph"; state.response = "Preparing graph description..."; state.update_scratchpad_reason(tool_name, "Preparing to describe the current execution graph.")
        if not state.execution_graph or not isinstance(state.execution_graph, GraphOutput): state.response = (state.response or "") + " No execution graph is currently available to describe or graph is invalid."; logger.warning("describe_graph: No execution_graph found in state or invalid type.")
        else:
            graph_desc = state.execution_graph.description
            if not graph_desc or len(graph_desc) < 20: 
                logger.info("Graph description is short or missing, generating a dynamic one."); node_summaries = []
                for node in state.execution_graph.nodes: node_summaries.append(f"- {node.effective_id}: {node.summary or node.operationId[:50]}") 
                nodes_str = "\n".join(node_summaries[:5]); 
                if len(node_summaries) > 5: nodes_str += f"\n- ... and {len(node_summaries) - 5} more nodes."
                prompt = (f"The following API execution graph has been generated for the goal: '{state.plan_generation_goal or 'general use'}'.\nNodes in the graph ({len(state.execution_graph.nodes)} total, sample):\n{nodes_str}\n\nPlease provide a concise, user-friendly natural language description of this workflow. Explain its overall purpose and the general sequence of operations. Use Markdown for readability (e.g., a brief introductory sentence, then bullet points for key stages if appropriate).")
                try:
                    dynamic_desc = llm_call_helper(self.worker_llm, prompt)
                    if graph_desc and graph_desc != dynamic_desc: final_desc_for_user = f"**Overall Workflow Plan for: '{state.plan_generation_goal or 'General Use'}'**\n\n{dynamic_desc}\n\n*Original AI-generated graph description: {graph_desc}*"
                    else: final_desc_for_user = f"**Current API Workflow for: '{state.plan_generation_goal or 'General Use'}'**\n\n{dynamic_desc}"
                except Exception as e: logger.error(f"Error generating dynamic graph description: {e}"); final_desc_for_user = (f"**Current API Workflow for: '{state.plan_generation_goal or 'General Use'}'**\n\n{graph_desc or 'No detailed description available. The graph includes nodes like ' + ', '.join([n.effective_id for n in state.execution_graph.nodes[:3]]) + '...'}")
            else: final_desc_for_user = f"**Current API Workflow for: '{state.plan_generation_goal or 'General Use'}'**\n\n{graph_desc}"
            if state.execution_graph.refinement_summary: final_desc_for_user += f"\n\n**Last Refinement Note:** {state.execution_graph.refinement_summary}"
            state.response = final_desc_for_user
            if 'graph_to_send' not in state.scratchpad and state.execution_graph:
                 try: state.scratchpad['graph_to_send'] = state.execution_graph.model_dump_json(indent=2)
                 except Exception as e: logger.error(f"Error serializing graph for sending during describe_graph: {e}")
        state.update_scratchpad_reason(tool_name, f"Graph description generated/retrieved. Response set: {state.response[:100]}..."); state.next_step = "responder"; return state

    def get_graph_json(self, state: BotState) -> BotState:
        tool_name = "get_graph_json"; state.response = "Fetching graph JSON..."; state.update_scratchpad_reason(tool_name, "Attempting to provide graph JSON.")
        if not state.execution_graph or not isinstance(state.execution_graph, GraphOutput): state.response = "No execution graph is currently available or graph is invalid."
        else:
            try: graph_json_str = state.execution_graph.model_dump_json(indent=2); state.scratchpad['graph_to_send'] = graph_json_str; state.response = f"The current API workflow graph is available in the graph view. You can also copy the JSON from there if needed."; logger.info("Provided graph JSON to scratchpad for UI.")
            except Exception as e: logger.error(f"Error serializing execution_graph to JSON: {e}"); state.response = f"Error serializing graph to JSON: {str(e)}"
        state.next_step = "responder"; return state

    def answer_openapi_query(self, state: BotState) -> BotState:
        tool_name = "answer_openapi_query"; state.response = "Thinking about your question..."; state.update_scratchpad_reason(tool_name, f"Attempting to answer user query: {state.user_input[:100] if state.user_input else 'N/A'}")
        if not state.openapi_schema and not (state.execution_graph and isinstance(state.execution_graph, GraphOutput)): state.response = "I don't have an OpenAPI specification loaded or a graph generated yet. Please provide one first."; state.next_step = "responder"; return state
        context_parts = []
        if state.user_input: context_parts.append(f"User Question: \"{state.user_input}\"")
        if state.schema_summary: context_parts.append(f"\n### API Specification Summary (from validated schema)\n{state.schema_summary}")
        if state.identified_apis:
            api_list_md = "\n### Identified API Operations (Sample - first few, from validated schema):\n"; num_apis_to_list = MAX_APIS_IN_PROMPT_SUMMARY_SHORT 
            for i, api in enumerate(state.identified_apis[:num_apis_to_list]): api_list_md += f"- **{api.get('operationId', 'N/A')}**: {api.get('method', '?')} {api.get('path', '?')} - _{api.get('summary', 'No summary')[:70]}..._\n"
            if len(state.identified_apis) > num_apis_to_list: api_list_md += f"- ... and {len(state.identified_apis) - num_apis_to_list} more.\n"
            context_parts.append(api_list_md)
        if state.execution_graph and isinstance(state.execution_graph, GraphOutput) and state.execution_graph.description:
            graph_desc_md = f"\n### Current Workflow Graph ('{state.plan_generation_goal or 'General Purpose'}') Description:\n{state.execution_graph.description}"
            if state.execution_graph.refinement_summary: graph_desc_md += f"\nLast Refinement: {state.execution_graph.refinement_summary}"
            context_parts.append(graph_desc_md)
        elif state.execution_graph and isinstance(state.execution_graph, GraphOutput): context_parts.append(f"\n### Current Workflow Graph ('{state.plan_generation_goal or 'General Purpose'}') exists but has no detailed description.")
        payload_info_md = ""
        if state.user_input and state.payload_descriptions: 
            for op_id, desc_text in state.payload_descriptions.items():
                if op_id.lower() in state.user_input.lower(): payload_info_md = f"\n### Payload/Response Example for '{op_id}' (from schema):\n{desc_text}\n"; context_parts.append(payload_info_md); break 
        full_context = "\n".join(context_parts)
        if not full_context.strip(): full_context = "No specific API context available, but an OpenAPI spec might be loaded."
        prompt = f"""You are an expert API assistant. Answer the user's question based on the provided context (which is derived from a validated OpenAPI schema). Use Markdown for formatting (e.g., headings, lists, bolding, italics, and code blocks for JSON snippets).\n\n{full_context}\n\nPlease provide a clear, concise, and helpful answer to the User Question. If the information is not available in the context, state that clearly. If listing multiple items (like API operations), use bullet points. If showing example JSON, ensure it is in a Markdown code block (e.g., ```json ... ```). Focus only on answering the question. Do not add conversational fluff beyond the answer."""
        try: state.response = llm_call_helper(self.worker_llm, prompt); logger.info("Successfully generated answer for OpenAPI query.")
        except Exception as e: logger.error(f"Error generating answer for OpenAPI query: {e}", exc_info=False); state.response = f"### Error Answering Query\nSorry, I encountered an error while trying to answer your question: {str(e)[:100]}..."
        state.update_scratchpad_reason(tool_name, f"Answered query. Response snippet: {state.response[:100]}..."); state.next_step = "responder"; return state

    def interactive_query_planner(self, state: BotState) -> BotState:
        tool_name = "interactive_query_planner"; state.response = "Planning how to address your interactive query..."; state.update_scratchpad_reason(tool_name, f"Entering interactive query planner for input: {state.user_input[:100] if state.user_input else 'N/A'}")
        state.scratchpad.pop('interactive_action_plan', None); state.scratchpad.pop('current_interactive_action_idx', None); state.scratchpad.pop('current_interactive_results', None)
        graph_summary = state.execution_graph.description[:150] + "..." if state.execution_graph and isinstance(state.execution_graph, GraphOutput) and state.execution_graph.description else "No graph currently generated."; payload_keys_sample = list(state.payload_descriptions.keys())[:3]
        prompt = f"""User Query: "{state.user_input}"\n\nCurrent State Context:\n- API Spec Summary: {'Available (from validated spec)' if state.schema_summary else 'Not available.'}\n- Identified APIs count: {len(state.identified_apis) if state.identified_apis else 0}. Example OpIDs: {", ".join([api['operationId'] for api in state.identified_apis[:3]])}...\n- Example Payload Descriptions available for OpIDs (sample, from validated spec): {payload_keys_sample}...\n- Current Execution Graph Goal: {state.plan_generation_goal or 'Not set.'}\n- Current Graph Description: {graph_summary}\n- Workflow Execution Status: {state.workflow_execution_status}\n\nAvailable Internal Actions (choose one or more in sequence, output as a JSON list):\n1.  `rerun_payload_generation`: Regenerate payload/response examples for specific APIs, possibly with new user-provided context.\n    Params: {{ "operation_ids_to_update": ["opId1", "opId2"], "new_context": "User's new context string for generation" }}\n2.  `contextualize_graph_descriptions`: Rewrite descriptions within the *existing* graph (overall, nodes, edges) to reflect new user context or focus. This does NOT change graph structure.\n    Params: {{ "new_context_for_graph": "User's new context/focus for descriptions" }}\n3.  `regenerate_graph_with_new_goal`: Create a *new* graph if the user states a completely different high-level goal OR requests a significant structural change (add/remove/reorder API steps).\n    Params: {{ "new_goal_string": "User's new goal, incorporating the structural change (e.g., 'Workflow to X, then Y, and then Z as the last step')" }}\n4.  `refine_existing_graph_structure`: For minor structural adjustments to the existing graph (e.g., "add API Z after Y but before END_NODE", "remove API X"). This implies the overall goal is similar but the sequence/nodes need adjustment. The LLM will be asked to refine the current graph JSON.\n    Params: {{ "refinement_instructions_for_structure": "User's specific feedback for structural refinement (e.g., 'Add operation Z after Y', 'Ensure X comes before Y')" }}\n5.  `answer_query_directly`: If the query can be answered using existing information (API summary, API list, current graph description, existing payload examples) without modifications to artifacts.\n    Params: {{ "query_for_synthesizer": "The original user query or a rephrased one for direct answering." }}\n6.  `setup_workflow_execution_interactive`: If the user asks to run/execute the current graph. This action prepares the system for execution.\n    Params: {{ "initial_parameters": {{ "param1": "value1" }} }} (Optional initial parameters for the workflow, if provided by user)\n7.  `resume_workflow_with_payload_interactive`: If the workflow is 'paused_for_confirmation' and the user provides the necessary payload/confirmation to continue.\n    Params: {{ "confirmed_payload": {{...}} }} (The JSON payload confirmed or provided by the user)\n8.  `synthesize_final_answer`: (Usually the last step of a plan) Formulate a comprehensive answer to the user based on the outcomes of previous internal actions or if no other action is suitable.\n    Params: {{ "synthesis_prompt_instructions": "Instructions for the LLM on what to include in the final answer, summarizing actions taken or information gathered." }}\n\nTask:\n1. Analyze the user's query in the context of the current system state.\n2. Create a short, logical "interactive_action_plan" (a list of action objects, max 3-4 steps).\n   - For requests to run the graph, use `setup_workflow_execution_interactive`.\n   - If the graph is paused and user provides data, use `resume_workflow_with_payload_interactive`.\n   - For structural changes like "add X at the end", prefer `regenerate_graph_with_new_goal` or `refine_existing_graph_structure`.\n3. Provide a brief "user_query_understanding" (1-2 sentences).\n\nOutput ONLY a JSON object with this structure:\n{{\n  "user_query_understanding": "Brief interpretation of user's need.",\n  "interactive_action_plan": [\n    {{"action_name": "action_enum_value", "action_params": {{...}}, "description": "Briefly, why this action is chosen."}}\n  ]\n}}\nIf the query is very simple and can be answered directly, the plan might just be one "answer_query_directly" or "synthesize_final_answer" action.\nIf the query is ambiguous or cannot be handled by available actions, use "synthesize_final_answer" with instructions to inform the user."""
        try:
            llm_response_str = llm_call_helper(self.worker_llm, prompt)
            parsed_plan_data = parse_llm_json_output_with_model(llm_response_str) 
            if parsed_plan_data and isinstance(parsed_plan_data, dict) and "interactive_action_plan" in parsed_plan_data and isinstance(parsed_plan_data["interactive_action_plan"], list) and "user_query_understanding" in parsed_plan_data:
                state.scratchpad['user_query_understanding'] = parsed_plan_data["user_query_understanding"]; state.scratchpad['interactive_action_plan'] = parsed_plan_data["interactive_action_plan"]; state.scratchpad['current_interactive_action_idx'] = 0; state.scratchpad['current_interactive_results'] = [] 
                state.response = f"Understood query: {state.scratchpad['user_query_understanding']}. Starting internal actions..."; logger.info(f"Interactive plan generated: {state.scratchpad['interactive_action_plan']}"); state.next_step = "interactive_query_executor"
            else: logger.error(f"LLM failed to produce a valid interactive plan. Raw: {llm_response_str[:300]}"); raise ValueError("LLM failed to produce a valid interactive plan JSON structure with required keys.")
        except Exception as e: logger.error(f"Error in interactive_query_planner: {e}", exc_info=False); state.response = f"Sorry, I encountered an error while planning how to address your request: {str(e)[:100]}..."; state.next_step = "answer_openapi_query" 
        state.update_scratchpad_reason(tool_name, f"Interactive plan generated. Next: {state.next_step}. Response: {state.response[:100]}")
        return state

    def _internal_contextualize_graph_descriptions(self, state: BotState, new_context: str) -> str:
        tool_name = "_internal_contextualize_graph_descriptions"
        if not state.execution_graph or not isinstance(state.execution_graph, GraphOutput): return "No graph to contextualize or graph is invalid."
        if not new_context: return "No new context provided for contextualization."
        logger.info(f"Attempting to contextualize graph descriptions with context: {new_context[:100]}...")
        if state.execution_graph.description:
            prompt_overall = (f"Current overall graph description: \"{state.execution_graph.description}\"\nNew User Context/Focus: \"{new_context}\"\n\nRewrite the graph description to incorporate this new context/focus, keeping it concise. Output only the new description text.")
            try: state.execution_graph.description = llm_call_helper(self.worker_llm, prompt_overall); logger.info(f"Overall graph description contextualized: {state.execution_graph.description[:100]}...")
            except Exception as e: logger.error(f"Error contextualizing overall graph description: {e}")
        nodes_to_update = [n for n in state.execution_graph.nodes if n.operationId not in ["START_NODE", "END_NODE"]][:3]
        for node in nodes_to_update:
            if node.description: 
                prompt_node = (f"Current description for node '{node.effective_id}' ({node.summary}): \"{node.description}\"\nOverall User Context/Focus for the graph: \"{new_context}\"\n\nRewrite this node's description to align with the new context, focusing on its role in the workflow under this context. Output only the new description text for this node.")
                try: node.description = llm_call_helper(self.worker_llm, prompt_node); logger.info(f"Node '{node.effective_id}' description contextualized: {node.description[:100]}...")
                except Exception as e: logger.error(f"Error contextualizing node '{node.effective_id}' description: {e}")
        if state.execution_graph: state.scratchpad['graph_to_send'] = state.execution_graph.model_dump_json(indent=2) 
        state.update_scratchpad_reason(tool_name, f"Graph descriptions contextualized with context: {new_context[:70]}.")
        return f"Graph descriptions have been updated to reflect the context: '{new_context[:70]}...'."

    def interactive_query_executor(self, state: BotState) -> BotState:
        tool_name = "interactive_query_executor"; plan = state.scratchpad.get('interactive_action_plan', []); idx = state.scratchpad.get('current_interactive_action_idx', 0); results = state.scratchpad.get('current_interactive_results', []) 
        if not plan or idx >= len(plan):
            final_response_message = "Finished interactive processing. "; 
            if results: final_response_message += (str(results[-1])[:200] + "..." if len(str(results[-1])) > 200 else str(results[-1]))
            else: final_response_message += "No specific actions were taken or results to report."
            if not state.response: state.response = final_response_message
            logger.info("Interactive plan execution completed or no plan."); state.next_step = "responder"; state.update_scratchpad_reason(tool_name, "Interactive plan execution completed or no plan."); return state
        action = plan[idx]; action_name = action.get("action_name"); action_params = action.get("action_params", {}); action_description = action.get("description", "No description for action.") 
        state.response = f"Executing internal step ({idx + 1}/{len(plan)}): {action_description[:70]}..."; state.update_scratchpad_reason(tool_name, f"Executing action ({idx + 1}/{len(plan)}): {action_name} - {action_description}")
        action_result_message = f"Action '{action_name}' completed." 
        try:
            if action_name == "rerun_payload_generation":
                op_ids = action_params.get("operation_ids_to_update", []); new_ctx = action_params.get("new_context", "")
                if op_ids and new_ctx: self._generate_payload_descriptions(state, target_apis=op_ids, context_override=new_ctx); action_result_message = f"Payload examples updated for {op_ids} with context '{new_ctx[:30]}...'."
                else: action_result_message = "Skipped rerun_payload_generation: Missing operation_ids or new_context."
                results.append(action_result_message); state.next_step = "interactive_query_executor" 
            elif action_name == "contextualize_graph_descriptions":
                new_ctx_graph = action_params.get("new_context_for_graph", "")
                if new_ctx_graph: action_result_message = self._internal_contextualize_graph_descriptions(state, new_ctx_graph)
                else: action_result_message = "Skipped contextualize_graph_descriptions: Missing new_context_for_graph."
                results.append(action_result_message); state.next_step = "interactive_query_executor"
            elif action_name == "regenerate_graph_with_new_goal":
                new_goal = action_params.get("new_goal_string")
                if new_goal: state.plan_generation_goal = new_goal; state.execution_graph = None; state.graph_refinement_iterations = 0; state.scratchpad['graph_gen_attempts'] = 0; state.scratchpad['refinement_validation_failures'] = 0; self._generate_execution_graph(state, goal=new_goal); action_result_message = f"Graph regeneration started for new goal: {new_goal[:50]}..."
                else: action_result_message = "Skipped regenerate_graph_with_new_goal: Missing new_goal_string."; state.next_step = "interactive_query_executor" 
                results.append(action_result_message)
            elif action_name == "refine_existing_graph_structure":
                refinement_instr = action_params.get("refinement_instructions_for_structure")
                if refinement_instr and state.execution_graph and isinstance(state.execution_graph, GraphOutput): state.graph_regeneration_reason = refinement_instr; state.scratchpad['refinement_validation_failures'] = 0; self.refine_api_graph(state); action_result_message = f"Graph refinement (structure) started with instructions: {refinement_instr[:50]}..."
                elif not state.execution_graph or not isinstance(state.execution_graph, GraphOutput): action_result_message = "Skipped refine_existing_graph_structure: No graph exists or invalid type."; state.next_step = "interactive_query_executor"
                else: action_result_message = "Skipped refine_existing_graph_structure: Missing refinement_instructions_for_structure."; state.next_step = "interactive_query_executor"
                results.append(action_result_message)
            elif action_name == "answer_query_directly":
                query_to_answer = action_params.get("query_for_synthesizer", state.user_input or ""); original_user_input = state.user_input; state.user_input = query_to_answer; self.answer_openapi_query(state); state.user_input = original_user_input; action_result_message = f"Direct answer generated for: {query_to_answer[:50]}..."; results.append(action_result_message)
            elif action_name == "setup_workflow_execution_interactive":
                self.setup_workflow_execution(state); action_result_message = f"Workflow execution setup initiated. Status: {state.workflow_execution_status}."; results.append(action_result_message)
                if idx + 1 < len(plan): logger.warning("More actions planned after setup_workflow_execution_interactive. These will likely be skipped as setup routes to responder.")
            elif action_name == "resume_workflow_with_payload_interactive":
                confirmed_payload = action_params.get("confirmed_payload")
                if confirmed_payload and isinstance(confirmed_payload, dict): state.scratchpad['pending_resume_payload'] = confirmed_payload; state.response = "Received payload to resume workflow. System will attempt to continue."; state.workflow_execution_status = "running"; action_result_message = f"Workflow resumption with payload prepared. Status: {state.workflow_execution_status}."
                else: action_result_message = "Skipped resume_workflow: Missing or invalid confirmed_payload."
                results.append(action_result_message); state.next_step = "responder" 
            elif action_name == "synthesize_final_answer":
                synthesis_instr = action_params.get("synthesis_prompt_instructions", "Summarize actions and provide a final response."); all_prior_results_summary = "; ".join([str(r)[:150] for r in results])
                final_synthesis_prompt = (f"User's original query: '{state.user_input}'.\nMy understanding of the query: '{state.scratchpad.get('user_query_understanding', 'N/A')}'.\nInternal actions taken and their results (summary): {all_prior_results_summary if all_prior_results_summary else 'No specific actions taken or results to summarize.'}\nAdditional instructions for synthesis: {synthesis_instr}\n\nBased on all the above, formulate a comprehensive and helpful final answer for the user in Markdown format.")
                try: state.response = llm_call_helper(self.worker_llm, final_synthesis_prompt); action_result_message = "Final answer synthesized."
                except Exception as e: logger.error(f"Error synthesizing final answer: {e}"); state.response = f"Sorry, I encountered an error while synthesizing the final answer: {str(e)[:100]}"; action_result_message = "Error during final answer synthesis."
                results.append(action_result_message); state.next_step = "responder" 
            else: action_result_message = f"Unknown or unhandled action: {action_name}."; logger.warning(action_result_message); results.append(action_result_message); state.next_step = "interactive_query_executor" 
        except Exception as e_action: logger.error(f"Error executing action '{action_name}': {e_action}", exc_info=True); action_result_message = f"Error during action '{action_name}': {str(e_action)[:100]}..."; results.append(action_result_message); state.response = action_result_message; state.next_step = "interactive_query_executor" 
        state.scratchpad['current_interactive_action_idx'] = idx + 1; state.scratchpad['current_interactive_results'] = results 
        if state.next_step == "interactive_query_executor": 
            if state.scratchpad['current_interactive_action_idx'] >= len(plan): 
                if action_name not in ["synthesize_final_answer", "answer_query_directly", "setup_workflow_execution_interactive", "resume_workflow_with_payload_interactive"]:
                    logger.info(f"Interactive plan finished after action '{action_name}'. Finalizing with synthesis.")
                    final_synthesis_instr = (f"The user's query was: '{state.user_input}'. My understanding was: '{state.scratchpad.get('user_query_understanding', 'N/A')}'. The following internal actions were taken with these results: {'; '.join([str(r)[:100] + '...' for r in results])}. Please formulate a comprehensive final answer to the user based on these actions and results.")
                    try: state.response = llm_call_helper(self.worker_llm, final_synthesis_instr)
                    except Exception as e_synth: logger.error(f"Error during final synthesis in interactive_query_executor: {e_synth}"); state.response = "Processed your request. " + (str(results[-1])[:100] if results else "")
                state.next_step = "responder"
        return state

    def handle_unknown(self, state: BotState) -> BotState:
        tool_name = "handle_unknown"
        if not state.response or "error" not in str(state.response).lower(): state.response = "I'm not sure how to process that request. Could you please rephrase it, or provide an OpenAPI specification if you haven't already?"
        state.update_scratchpad_reason(tool_name, f"Handling unknown input or situation. Final response to be: {state.response}"); state.next_step = "responder"; return state

    def handle_loop(self, state: BotState) -> BotState:
        tool_name = "handle_loop"; state.response = "It seems we're stuck in a processing loop. Please try rephrasing your request or starting over with the OpenAPI specification."; state.loop_counter = 0; state.update_scratchpad_reason(tool_name, "Loop detected, routing to responder with a loop message."); state.next_step = "responder"; return state

    def setup_workflow_execution(self, state: BotState) -> BotState:
        tool_name = "setup_workflow_execution"; logger.info(f"[{state.session_id}] Setting up workflow execution based on current graph."); state.update_scratchpad_reason(tool_name, "Preparing for workflow execution.")
        if not state.execution_graph or not isinstance(state.execution_graph, GraphOutput): state.response = "No execution graph is available to run or graph is invalid. Please generate or load one first."; state.workflow_execution_status = "failed"; state.next_step = "responder"; return state
        if state.workflow_execution_status in ["running", "paused_for_confirmation", "pending_start"]: state.response = "A workflow is already running, paused, or pending. Please wait for it to complete or address the current state."; state.next_step = "responder"; return state
        try: state.workflow_execution_status = "pending_start"; state.response = ("Workflow execution has been prepared. The system will now attempt to start running the defined API calls. You should receive updates on its progress shortly."); logger.info(f"[{state.session_id}] BotState prepared for workflow execution. Status set to 'pending_start'.")
        except Exception as e: logger.error(f"[{state.session_id}] Error during workflow setup preparation: {e}", exc_info=True); state.response = f"Critical error preparing workflow execution: {str(e)[:150]}"; state.workflow_execution_status = "failed"
        state.next_step = "responder"; return state

    def resume_workflow_with_payload(self, state: BotState, confirmed_payload: Dict[str, Any]) -> BotState:
        tool_name = "resume_workflow_with_payload"; logger.info(f"[{state.session_id}] Preparing to resume workflow with confirmed_payload."); state.update_scratchpad_reason(tool_name, f"Payload received for workflow resumption: {str(confirmed_payload)[:100]}...")
        if state.workflow_execution_status != "paused_for_confirmation": state.response = (f"Workflow is not currently paused for confirmation (current status: {state.workflow_execution_status}). Cannot process resume payload at this time."); state.next_step = "responder"; return state
        state.scratchpad['pending_resume_payload'] = confirmed_payload; state.workflow_execution_status = "running"; state.response = "Confirmation payload received. System will attempt to resume workflow execution."; logger.info(f"[{state.session_id}] Confirmed payload stored in scratchpad. Workflow status set to 'running' (pending actual resume by main.py).")
        state.next_step = "responder"; return state
