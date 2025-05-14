# utils.py
import logging
import json
import yaml # Keep yaml import if used elsewhere or for future use
import re
from typing import Any, Dict, Optional
from langchain_core.language_models.base import BaseLanguageModel # For type hinting
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage # For type hinting

logger = logging.getLogger(__name__)

# --- LLM Call Helper ---
def llm_call_helper(llm: BaseLanguageModel, prompt: str, system_prompt: Optional[str] = None) -> str:
    """
    Helper function to make a call to an LLM and return the string content.
    Includes basic error handling.
    """
    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=prompt))
    
    try:
        response = llm.invoke(messages)
        if isinstance(response, AIMessage):
            return response.content
        elif isinstance(response, str): # Some LLMs might return str directly
            return response
        else:
            logger.error(f"LLM call returned unexpected type: {type(response)}. Content: {str(response)[:200]}")
            raise ValueError(f"LLM response was not an AIMessage or string: {type(response)}")
    except Exception as e:
        logger.error(f"Error during LLM call: {e}", exc_info=True)
        # Depending on policy, you might re-raise, return a default, or handle specific errors
        raise # Re-raise the exception to be handled by the caller


# --- LLM Output Fence Stripping Utility ---
def strip_llm_json_output_fences(llm_output_str: str) -> str:
    """
    Strips markdown code block fences (e.g., ```json ... ``` or ``` ... ```)
    from a string, typically an LLM output expected to be JSON.
    It handles optional language specifiers like 'json'.
    Iteratively strips to handle potential nested or repeated fences.
    """
    if not isinstance(llm_output_str, str):
        logger.warning(f"strip_llm_json_output_fences received non-string input type: {type(llm_output_str)}. Returning as is.")
        return llm_output_str # Or raise TypeError

    current_text = llm_output_str.strip()
    previous_text = None
    
    # Regex to find markdown code blocks, capturing the content inside.
    # It matches ``` optionally followed by a language (e.g., json, yaml), then content, then ```.
    # Ensures it matches the outermost pair for each iteration.
    # Language specifier allows letters, numbers, underscore, plus, hyphen.
    # Using re.IGNORECASE for the language specifier (e.g. ```JSON is same as ```json)
    regex_pattern = r"^\s*```(?:[a-zA-Z0-9_+\-]+)?\s*([\s\S]*?)\s*```\s*$"

    while current_text and current_text != previous_text:
        previous_text = current_text
        match = re.match(regex_pattern, current_text, re.DOTALL | re.IGNORECASE)
        if match:
            current_text = match.group(1).strip() # Get the content and strip it
        else:
            # Fallback for simpler cases or malformed fences not caught by the main regex.
            # This handles cases like just ``` at the start or end without full block structure,
            # or if the content itself starts/ends with ``` after a previous strip.
            lines = current_text.splitlines()
            stripped_leading = False
            if lines and lines[0].strip().lower().startswith("```"): # Check for ```json, ```yaml etc.
                # More robustly remove the first line if it's a fence start
                first_line_content = lines[0].strip()
                if re.match(r"```(?:[a-zA-Z0-9_+\-]+)?\s*$", first_line_content, re.IGNORECASE):
                    lines.pop(0)
                    stripped_leading = True
            
            stripped_trailing = False
            if lines and lines[-1].strip() == "```":
                lines.pop(-1)
                stripped_trailing = True
            
            if stripped_leading or stripped_trailing:
                current_text = "\n".join(lines).strip()
            else:
                # No standard markdown block found by regex, and no simple leading/trailing ``` lines removed.
                break # Exit loop as no changes were made in this iteration.
    
    # Final check: if the result is still wrapped in quotes (e.g. LLM returns '"{\\"key\\": \\"value\\"}"')
    # This can happen if the LLM tries to escape a JSON string within a string.
    if len(current_text) >= 2 and current_text.startswith('"') and current_text.endswith('"'):
        try:
            # Attempt to parse the inner string as JSON. If successful, it means the outer quotes were extraneous.
            # E.g. "\"{\\\"key\\\": \\\"value\\\"}\"" -> "{\"key\": \"value\"}"
            potential_json_inner = json.loads(current_text)
            if isinstance(potential_json_inner, str): # If un-quoting resulted in another string that is valid JSON
                 current_text = potential_json_inner
        except json.JSONDecodeError:
            pass # Not a double-quoted JSON string, leave as is.


    return current_text


# --- Schema Caching (Example, adjust as needed) ---
SCHEMA_CACHE: Optional[Any] = None # Placeholder for a cache object (e.g., from cachetools)
# Example using a simple dict for non-persistent cache if cachetools is not available
# SCHEMA_CACHE = {} 

def initialize_schema_cache(maxsize=10, ttl=3600):
    """Initializes the schema cache if cachetools is available."""
    global SCHEMA_CACHE
    try:
        from cachetools import TTLCache
        SCHEMA_CACHE = TTLCache(maxsize=maxsize, ttl=ttl)
        logger.info(f"Schema cache initialized with TTLCache (maxsize={maxsize}, ttl={ttl}).")
    except ImportError:
        SCHEMA_CACHE = {} # Fallback to simple dict if cachetools not installed
        logger.warning("cachetools library not found. Using a simple dictionary for schema caching (non-persistent, no TTL).")

def get_cache_key(spec_text: str) -> str:
    """Generates a cache key from the spec text (e.g., using a hash)."""
    import hashlib
    return hashlib.sha256(spec_text.encode('utf-8')).hexdigest()

def load_cached_schema(cache_key: str) -> Optional[Dict[str, Any]]:
    """Loads a schema from the cache."""
    if SCHEMA_CACHE is not None:
        try:
            cached_data = SCHEMA_CACHE.get(cache_key)
            if cached_data:
                logger.info(f"Schema found in cache for key: {cache_key[:10]}...")
                return cached_data
        except Exception as e: # Handle potential errors with cache interaction
            logger.warning(f"Error accessing schema cache for loading: {e}")
    return None

def save_schema_to_cache(cache_key: str, schema_data: Dict[str, Any]):
    """Saves a schema to the cache."""
    if SCHEMA_CACHE is not None:
        try:
            SCHEMA_CACHE[cache_key] = schema_data
            logger.info(f"Schema saved to cache with key: {cache_key[:10]}...")
        except Exception as e: # Handle potential errors with cache interaction
            logger.warning(f"Error accessing schema cache for saving: {e}")

# Initialize cache on module load (optional, or call explicitly in main.py startup)
# initialize_schema_cache()
