# llm_config.py
import os
import logging
from typing import Tuple, Any, Dict
import json # Added for mock graph generation

# Attempt to import the ChatGoogleGenerativeAI class
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    LANGCHAIN_GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    LANGCHAIN_GOOGLE_GENAI_AVAILABLE = False
    ChatGoogleGenerativeAI = None # Define as None if not available
    logging.warning(
        "Failed to import ChatGoogleGenerativeAI from langchain_google_genai. "
        "Real LLM functionality will be unavailable unless 'langchain-google-genai' is installed. "
        "You can still use mock LLMs by setting USE_MOCK_LLMS=true."
    )

logger = logging.getLogger(__name__)

# --- Environment Variable Names ---
GOOGLE_API_KEY_ENV_VAR = "GOOGLE_API_KEY"
USE_MOCK_LLMS_ENV_VAR = "USE_MOCK_LLMS" # New environment variable

# --- Model Names ---
ROUTER_LLM_MODEL_NAME = os.getenv("ROUTER_LLM_MODEL_NAME", "gemini-1.5-flash-latest")
WORKER_LLM_MODEL_NAME = os.getenv("WORKER_LLM_MODEL_NAME", "gemini-1.5-pro-latest")

# --- LLM Configuration Parameters ---
ROUTER_LLM_TEMPERATURE = float(os.getenv("ROUTER_LLM_TEMPERATURE", "0.2"))
WORKER_LLM_TEMPERATURE = float(os.getenv("WORKER_LLM_TEMPERATURE", "0.7"))

# --- Mock LLM Implementation ---
class MockLLMContent:
    """A simple class to mimic the 'content' attribute of an LLM response."""
    def __init__(self, content: str):
        self.content = content

class MockLLM:
    """
    A mock LLM class that mimics the behavior of a LangChain LLM.
    It returns predefined responses based on simple keyword matching in the prompt.
    """
    def __init__(self, name: str = "MockLLM"):
        self.name = name
        # Predefined responses for the router
        self.router_responses: Dict[str, str] = {
            "classify this intent: 'show me the user details api.'": "answer_openapi_query",
            "classify the user's intent. choose one of the following:": "answer_openapi_query", # Default for router
            "run this workflow": "setup_workflow_execution",
            "execute the plan": "setup_workflow_execution",
            "focus the plan on": "interactive_query_planner",
            "regenerate the graph for goal": "interactive_query_planner", # Simpler match
            "add a notification step": "interactive_query_planner",
            "what if i want to change": "interactive_query_planner",
            "here is the confirmed payload": "interactive_query_planner",
        }
        # Predefined responses for the worker
        self.worker_responses: Dict[str, str] = {
            "summarize the following api specification": "This is a mock API summary. It describes a set of endpoints for managing items and users.",
            "provide a concise, typical json example": json.dumps({
                "Request Payload Example": "```json\n{\"name\": \"mockItem\", \"value\": 123}\n```",
                "Expected Response Structure": "Returns an object with id, name, and status."
            }),
            "design an api execution graph as a json object": self._generate_mock_graph_output(),
            "refine the current graph based on the feedback": self._generate_mock_graph_output(refined=True),
            "provide a concise, user-friendly natural language description of this workflow": "This is a mock workflow description: It starts, gets some data, processes it, and then ends.",
            "answer the user's question based on the provided context": "This is a mock answer based on the provided context. The API seems to support creating and retrieving users.",
            "create a short, logical \"interactive_action_plan\"": json.dumps({
                "user_query_understanding": "User wants to do something specific.",
                "interactive_action_plan": [
                    {"action_name": "answer_query_directly", "action_params": {"query_for_synthesizer": "User's original query"}, "description": "Attempting to answer directly."}
                ]
            }),
            "rewrite the graph description to incorporate this new context/focus": "Mock contextualized graph description.",
            "rewrite this node's description to align with the new context": "Mock contextualized node description.",
            "formulate a comprehensive and helpful final answer for the user": "This is a mock synthesized final answer based on the actions taken."

        }
        logger.info(f"{self.name} initialized.")

    def _generate_mock_graph_output(self, refined: bool = False) -> str:
        """Generates a consistent mock GraphOutput JSON string."""
        graph_desc = "Initial mock graph."
        refinement_summary = "Initial graph with 2 steps."
        if refined:
            graph_desc = "Refined mock graph with an extra step."
            refinement_summary = "Refined the graph to include a logging step."

        return json.dumps({
            "graph_id": "mock-graph-123",
            "description": graph_desc,
            "nodes": [
                {"operationId": "START_NODE", "method": "SYSTEM", "path": "/start", "summary": "Start of the workflow"},
                {"operationId": "mockOpGetItems", "method": "GET", "path": "/items", "summary": "Get all items", "description": "Retrieves a list of items.", "output_mappings": [{"source_data_path": "$.items", "target_data_key": "retrieved_items"}]},
                {"operationId": "mockOpProcessItem", "method": "POST", "path": "/items/{{itemId}}/process", "summary": "Process an item", "description": "Processes a specific item.", "input_mappings": [{"source_operation_id": "mockOpGetItems", "source_data_path": "$.retrieved_items[0].id", "target_parameter_name": "itemId", "target_parameter_in": "path"}], "requires_confirmation": True, "payload_description": "Payload: {'action': 'process'}"},
                {"operationId": "END_NODE", "method": "SYSTEM", "path": "/end", "summary": "End of the workflow"}
            ],
            "edges": [
                {"from_node": "START_NODE", "to_node": "mockOpGetItems"},
                {"from_node": "mockOpGetItems", "to_node": "mockOpProcessItem"},
                {"from_node": "mockOpProcessItem", "to_node": "END_NODE"}
            ],
            "refinement_summary": refinement_summary
        })

    def invoke(self, prompt: Any) -> MockLLMContent:
        """
        Mimics the LLM's invoke method.
        Returns a MockLLMContent object containing a predefined response.
        """
        prompt_str = str(prompt).lower()
        logger.debug(f"{self.name} received prompt (first 200 chars): {prompt_str[:200]}...")

        response_dict = self.router_responses if "classify" in prompt_str else self.worker_responses

        for keyword, response in response_dict.items():
            if keyword.lower() in prompt_str:
                logger.info(f"{self.name} matched keyword '{keyword}' and will return: {str(response)[:100]}...")
                return MockLLMContent(response)

        # Default fallback response
        default_response = "I'm a mock LLM and I don't have a specific answer for that."
        if "json" in prompt_str or "graphoutput" in prompt_str.lower(): # If JSON is expected
            default_response = self._generate_mock_graph_output()
        elif "classify" in prompt_str: # Default for router if no match
             default_response = "handle_unknown"


        logger.warning(f"{self.name} did not find a keyword match. Returning default response: {default_response[:100]}...")
        return MockLLMContent(default_response)

def initialize_llms() -> Tuple[Any, Any]:
    """
    Initializes and returns the router and worker LLMs.
    Uses mock LLMs if USE_MOCK_LLMS environment variable is set to 'true'.
    Otherwise, retrieves the GOOGLE_API_KEY and configures ChatGoogleGenerativeAI.

    Returns:
        A tuple containing (router_llm, worker_llm).

    Raises:
        ValueError: If GOOGLE_API_KEY is not set (and not using mocks).
        RuntimeError: If ChatGoogleGenerativeAI could not be imported (and not using mocks).
    """
    use_mocks_str = os.getenv(USE_MOCK_LLMS_ENV_VAR, "false").lower()
    use_mocks = use_mocks_str == "true"

    if use_mocks:
        logger.info("Using Mock LLMs as USE_MOCK_LLMS is set to true.")
        mock_router_llm = MockLLM(name="MockRouterLLM")
        mock_worker_llm = MockLLM(name="MockWorkerLLM")
        return mock_router_llm, mock_worker_llm

    # --- Real LLM Initialization ---
    if not LANGCHAIN_GOOGLE_GENAI_AVAILABLE:
        logger.error(
            "Real LLMs requested, but ChatGoogleGenerativeAI class not available. "
            "Please ensure 'langchain-google-genai' is installed or set USE_MOCK_LLMS=true."
        )
        raise RuntimeError(
            "ChatGoogleGenerativeAI class not available. Cannot initialize real LLMs."
        )

    google_api_key = os.getenv(GOOGLE_API_KEY_ENV_VAR)
    if not google_api_key:
        logger.error(f"{GOOGLE_API_KEY_ENV_VAR} environment variable not found.")
        raise ValueError(
            f"Missing Google API Key. Please set the {GOOGLE_API_KEY_ENV_VAR} environment variable (or use mock LLMs)."
        )

    try:
        router_llm = ChatGoogleGenerativeAI(
            model=ROUTER_LLM_MODEL_NAME,
            temperature=ROUTER_LLM_TEMPERATURE,
            google_api_key=google_api_key,
            convert_system_message_to_human=True,
        )
        logger.info(f"Router LLM initialized successfully with model: {ROUTER_LLM_MODEL_NAME}")

        worker_llm = ChatGoogleGenerativeAI(
            model=WORKER_LLM_MODEL_NAME,
            temperature=WORKER_LLM_TEMPERATURE,
            google_api_key=google_api_key,
            convert_system_message_to_human=True,
        )
        logger.info(f"Worker LLM initialized successfully with model: {WORKER_LLM_MODEL_NAME}")

        return router_llm, worker_llm

    except Exception as e:
        logger.critical(f"Failed to initialize Google Generative AI models: {e}", exc_info=True)
        raise RuntimeError(f"Could not initialize LLMs: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test with Mock LLMs
    os.environ[USE_MOCK_LLMS_ENV_VAR] = "true"
    logger.info("--- Testing with Mock LLMs ---")
    try:
        mock_router, mock_worker = initialize_llms()
        logger.info("Mock LLMs initialized for testing.")

        router_prompt_example = "System: You are a router. User: Classify this intent: 'run this workflow'"
        mock_router_response = mock_router.invoke(router_prompt_example)
        logger.info(f"Mock Router LLM test response content: {mock_router_response.content}")
        assert mock_router_response.content == "setup_workflow_execution"


        worker_prompt_example = "Design an API execution graph as a JSON object for getting user details."
        mock_worker_response = mock_worker.invoke(worker_prompt_example)
        logger.info(f"Mock Worker LLM test response (graph) content: {mock_worker_response.content[:200]}...")
        try:
            json.loads(mock_worker_response.content)
            logger.info("Mock worker graph response is valid JSON.")
        except json.JSONDecodeError:
            logger.error("Mock worker graph response is NOT valid JSON.")


        worker_summary_prompt = "Summarize the following API specification: openapi: 3.0..."
        mock_summary_response = mock_worker.invoke(worker_summary_prompt)
        logger.info(f"Mock Worker LLM summary response: {mock_summary_response.content}")
        assert "mock API summary" in mock_summary_response.content


    except Exception as e:
        logger.error(f"An error occurred during mock LLM testing: {e}", exc_info=True)
    finally:
        del os.environ[USE_MOCK_LLMS_ENV_VAR] # Clean up env var

    logger.info("\n--- Testing with Real LLMs (if GOOGLE_API_KEY is set) ---")
    # Test with Real LLMs (requires GOOGLE_API_KEY to be set in environment)
    if os.getenv(GOOGLE_API_KEY_ENV_VAR) and LANGCHAIN_GOOGLE_GENAI_AVAILABLE:
        try:
            real_router, real_worker = initialize_llms()
            logger.info("Real LLMs initialized for testing.")
            
            # Example actual invoke (can be costly/slow, so commented out by default in general use)
            # real_router_response = real_router.invoke("Classify this intent: 'Show me the user details API.'")
            # logger.info(f"Real Router LLM test response: {real_router_response.content}")

            # real_worker_response = real_worker.invoke("Explain the concept of an API in one sentence.")
            # logger.info(f"Real Worker LLM test response: {real_worker_response.content}")
            logger.info("Real LLM initialization test completed. (Actual invokes can be un-commented for full test)")
        except ValueError as ve:
            logger.error(f"Configuration error during real LLM test: {ve}")
        except RuntimeError as rte:
            logger.error(f"Runtime error during real LLM test: {rte}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during the real LLM test: {e}", exc_info=True)
    else:
        logger.warning("GOOGLE_API_KEY not set or langchain-google-genai not available. Skipping real LLM test.")

    logger.info("Direct test of llm_config.py completed.")