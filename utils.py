# utils.py
import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Type
from pydantic import BaseModel, ValidationError
import diskcache

# Attempt to import from models.py, with fallbacks for standalone use/linting
try:
    from models import GraphOutput, Node, Edge # BaseModel is implicitly imported by Pydantic models
except ImportError:
    logging.warning("utils.py: Could not import full models from models.py. Using dummy classes for basic type hinting.")
    # Basic dummy classes if models.py isn't found (primarily for linting)
    class BaseModel:
        @classmethod
        def model_validate(cls, data): return data
        def model_dump_json(self, indent=None): return json.dumps(self.__dict__, indent=indent if indent else 2)

    class GraphOutput(BaseModel): pass
    class Node(BaseModel):
        @property
        def effective_id(self) -> str: return getattr(self, 'display_name', None) or getattr(self, 'operationId', 'unknown_node')
    class Edge(BaseModel): pass


logger = logging.getLogger(__name__)

# --- Persistent Caching Setup ---
CACHE_DIR = os.path.join(os.getcwd(), ".openapi_cache")
SCHEMA_CACHE: Optional[diskcache.Cache] = None
try:
    os.makedirs(CACHE_DIR, exist_ok=True)
    SCHEMA_CACHE = diskcache.Cache(CACHE_DIR)
    logger.info(f"Initialized schema cache at: {CACHE_DIR}")
except Exception as e:
    logger.error(f"Failed to initialize disk cache at {CACHE_DIR}: {e}. Caching disabled.", exc_info=True)

def get_cache_key(spec_text: str) -> str:
    """Generates a SHA256 cache key for the spec text."""
    return hashlib.sha256(spec_text.encode('utf-8')).hexdigest()

def load_cached_schema(cache_key: str) -> Optional[Dict[str, Any]]:
    """Loads a parsed schema from cache."""
    if SCHEMA_CACHE is None: return None
    try:
        schema = SCHEMA_CACHE.get(cache_key)
        if schema:
            logger.debug(f"Cache hit for key: {cache_key}")
            return schema
        return None
    except Exception as e:
        logger.error(f"Error loading schema from cache (key: {cache_key}): {e}", exc_info=True)
        return None

def save_schema_to_cache(cache_key: str, schema: Dict[str, Any]):
    """Saves a parsed schema to cache."""
    if SCHEMA_CACHE is None: return
    try:
        SCHEMA_CACHE.set(cache_key, schema)
        logger.debug(f"Saved schema to cache with key: {cache_key}")
    except Exception as e:
        logger.error(f"Error saving schema to cache (key: {cache_key}): {e}", exc_info=True)

# --- Graph Utilities ---
def check_for_cycles(graph: GraphOutput) -> Tuple[bool, str]:
    """Checks if the graph is a DAG. Returns (is_dag, message)."""
    if not isinstance(graph, GraphOutput) or not isinstance(graph.nodes, list):
        return False, "Invalid graph structure."

    adj: Dict[str, List[str]] = {node.effective_id: [] for node in graph.nodes if isinstance(node, Node)}
    node_ids = set(adj.keys())

    if not node_ids:
        return True, "Graph has no valid nodes."

    if not isinstance(graph.edges, list):
        return False, "Invalid graph edges structure."

    for edge in graph.edges:
        if isinstance(edge, Edge) and edge.from_node in adj and edge.to_node in node_ids:
            if edge.to_node not in adj[edge.from_node]: # Avoid duplicates if input has them
                 adj[edge.from_node].append(edge.to_node)
        else:
            logger.warning(f"Skipping invalid edge in cycle check: {getattr(edge, 'from_node', 'N/A')}->{getattr(edge, 'to_node', 'N/A')}")


    visited: Dict[str, int] = {node_id: 0 for node_id in node_ids} # 0: unvisited, 1: visiting, 2: visited
    path: List[str] = []

    def has_cycle_util(node_id: str) -> bool:
        visited[node_id] = 1 # Mark as visiting
        path.append(node_id)

        for neighbor_id in adj.get(node_id, []):
            if visited[neighbor_id] == 1: # Cycle detected
                # Construct cycle path for message
                try:
                    cycle_start_index = path.index(neighbor_id)
                    nonlocal cycle_message_detail
                    cycle_message_detail = " -> ".join(path[cycle_start_index:] + [neighbor_id])
                except ValueError:
                    cycle_message_detail = f"Involving {neighbor_id} and {node_id}"
                return True
            if visited[neighbor_id] == 0:
                if has_cycle_util(neighbor_id):
                    return True
        
        visited[node_id] = 2 # Mark as visited
        path.pop()
        return False

    cycle_message_detail = ""
    for node_id in node_ids:
        if visited[node_id] == 0:
            if has_cycle_util(node_id):
                return False, f"Cycle detected: {cycle_message_detail}"
    return True, "No cycles detected."


# --- LLM Call Helper ---
def llm_call_helper(llm: Any, prompt: Any, attempt: int = 1, max_attempts: int = 2) -> str:
    """Helper for LLM calls with logging and basic error handling."""
    prompt_repr = str(prompt)[:500] + '...' if len(str(prompt)) > 500 else str(prompt)
    logger.debug(f"LLM call (Attempt {attempt}/{max_attempts}) Prompt: {prompt_repr}")
    try:
        response_obj = llm.invoke(prompt)
        content = ""
        if hasattr(response_obj, 'content'):
            content = response_obj.content
        elif isinstance(response_obj, str):
            content = response_obj
        else:
            logger.warning(f"LLM response object type ({type(response_obj)}) has no 'content' and is not str. Trying str().")
            content = str(response_obj)
        
        if not isinstance(content, str):
            logger.warning(f"LLM content is not a string ({type(content)}). Converting to string.")
            content = str(content)

        logger.debug(f"LLM call successful. Response: {content[:500]}...")
        return content
    except Exception as e:
        logger.error(f"LLM call failed (Attempt {attempt}/{max_attempts}): {e}", exc_info=True)
        if attempt < max_attempts:
            logger.info(f"Retrying LLM call for: {prompt_repr.splitlines()[0] if isinstance(prompt_repr, str) else 'Structured Prompt'}")
            # Consider adding a small delay before retrying if appropriate
            return llm_call_helper(llm, prompt, attempt + 1, max_attempts)
        raise # Re-raise after max attempts

# --- JSON Parsing Helper ---
def parse_llm_json_output_with_model(llm_output: str, expected_model: Optional[Type[BaseModel]] = None) -> Any:
    """
    Parses JSON output from LLM string, handles markdown, and optional Pydantic validation.
    Returns parsed data (dict, list, or Pydantic model instance), or None on failure.
    """
    if not isinstance(llm_output, str):
        logger.error(f"Cannot parse non-string LLM output as JSON. Type: {type(llm_output)}")
        return None

    json_block = llm_output.strip()
    logger.debug(f"Attempting to parse LLM JSON output (first 500 chars): {json_block[:500]}...")

    # Extract from markdown code fences if present
    if json_block.startswith("```"):
        match = re.search(r"```(?:json|JSON)?\s*([\s\S]*?)\s*```", json_block, re.DOTALL)
        if match:
            json_block = match.group(1).strip()
            logger.debug("Extracted JSON content from markdown fence.")
        else: # Fallback for simple fence removal if regex fails
            if json_block.startswith("```json"): json_block = json_block[7:]
            elif json_block.startswith("```JSON"): json_block = json_block[7:]
            elif json_block.startswith("```"): json_block = json_block[3:]
            if json_block.endswith("```"): json_block = json_block[:-3]
            json_block = json_block.strip()


    try:
        parsed_data = json.loads(json_block)
        logger.debug("Successfully parsed JSON string.")

        if expected_model and issubclass(expected_model, BaseModel) and hasattr(expected_model, 'model_validate'):
            logger.debug(f"Validating parsed JSON against Pydantic model: {expected_model.__name__}")
            try:
                validated_data = expected_model.model_validate(parsed_data)
                logger.debug("Pydantic validation successful.")
                return validated_data
            except ValidationError as ve:
                logger.error(f"Pydantic validation failed for {expected_model.__name__}: {ve}")
                logger.debug(f"Data that failed Pydantic validation: {parsed_data}")
                return None # Validation failed
        return parsed_data # No Pydantic validation requested or model was not suitable
    except json.JSONDecodeError as jde:
        # Try to find the problematic character and context
        # jde.doc contains the full string, jde.pos is the character index
        context_start = max(0, jde.pos - 30)
        context_end = min(len(jde.doc), jde.pos + 30)
        problem_snippet = jde.doc[context_start:context_end]
        logger.error(f"JSON parsing failed: {jde.msg}. At char {jde.pos}. Snippet: '{problem_snippet}'")
        logger.debug(f"Full text attempted for JSON parsing: {json_block}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during JSON parsing/validation: {e}", exc_info=True)
        return None

# Ensure regex is imported if not already at the top
import re
