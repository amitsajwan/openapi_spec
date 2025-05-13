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
        "The system will use mock LLMs if real ones cannot be initialized."
    )

logger = logging.getLogger(__name__)

# --- Environment Variable Names ---
GOOGLE_API_KEY_ENV_VAR = "GOOGLE_API_KEY"
USE_MOCK_LLMS_ENV_VAR = "USE_MOCK_LLMS"

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
            "regenerate the graph for goal": "interactive_query_planner",
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

        default_response = "I'm a mock LLM and I don't have a specific answer for that."
        if "json" in prompt_str or "graphoutput" in prompt_str.lower():
            default_response = self._generate_mock_graph_output()
        elif "classify" in prompt_str:
             default_response = "handle_unknown"

        logger.warning(f"{self.name} did not find a keyword match. Returning default response: {default_response[:100]}...")
        return MockLLMContent(default_response)

def _get_mock_llms() -> Tuple[MockLLM, MockLLM]:
    """Helper function to instantiate and return mock LLMs."""
    logger.info("Instantiating Mock LLMs.")
    mock_router_llm = MockLLM(name="MockRouterLLM")
    mock_worker_llm = MockLLM(name="MockWorkerLLM")
    return mock_router_llm, mock_worker_llm

def initialize_llms() -> Tuple[Any, Any]:
    """
    Initializes and returns the router and worker LLMs.
    - If USE_MOCK_LLMS is 'true', uses mock LLMs.
    - Otherwise, tries to use real Google LLMs if 'langchain-google-genai' is installed AND GOOGLE_API_KEY is set.
    - Falls back to mock LLMs if real LLM initialization fails or prerequisites are missing.

    Returns:
        A tuple containing (router_llm, worker_llm).
    """
    use_mocks_explicitly = os.getenv(USE_MOCK_LLMS_ENV_VAR, "false").lower() == "true"

    if use_mocks_explicitly:
        logger.info("Explicitly using Mock LLMs as USE_MOCK_LLMS is set to true.")
        return _get_mock_llms()

    # --- Attempt Real LLM Initialization ---
    if not LANGCHAIN_GOOGLE_GENAI_AVAILABLE:
        logger.warning(
            "Real LLMs implicitly requested (USE_MOCK_LLMS is not 'true'), but 'langchain-google-genai' is not installed. "
            "Falling back to Mock LLMs. Please install 'langchain-google-genai' for real LLM functionality."
        )
        return _get_mock_llms()

    google_api_key = os.getenv(GOOGLE_API_KEY_ENV_VAR)
    if not google_api_key:
        logger.warning(
            f"{GOOGLE_API_KEY_ENV_VAR} environment variable not found. Real LLMs require an API key. "
            "Falling back to Mock LLMs. Please set the GOOGLE_API_KEY environment variable for real LLM functionality."
        )
        return _get_mock_llms()

    try:
        logger.info("Attempting to initialize real Google Generative AI models...")
        router_llm = ChatGoogleGenerativeAI(
            model=ROUTER_LLM_MODEL_NAME,
            temperature=ROUTER_LLM_TEMPERATURE,
            google_api_key=google_api_key,
            convert_system_message_to_human=True,
        )
        logger.info(f"Real Router LLM initialized successfully with model: {ROUTER_LLM_MODEL_NAME}")

        worker_llm = ChatGoogleGenerativeAI(
            model=WORKER_LLM_MODEL_NAME,
            temperature=WORKER_LLM_TEMPERATURE,
            google_api_key=google_api_key,
            convert_system_message_to_human=True,
        )
        logger.info(f"Real Worker LLM initialized successfully with model: {WORKER_LLM_MODEL_NAME}")

        return router_llm, worker_llm

    except Exception as e:
        logger.error(f"Failed to initialize real Google Generative AI models: {e}. Falling back to Mock LLMs.", exc_info=True)
        return _get_mock_llms()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    # Scenario 1: Explicitly use mocks
    logger.info("\n--- Scenario 1: Testing with USE_MOCK_LLMS=true ---")
    os.environ[USE_MOCK_LLMS_ENV_VAR] = "true"
    # Unset API key to ensure it's not accidentally used
    if GOOGLE_API_KEY_ENV_VAR in os.environ:
        del os.environ[GOOGLE_API_KEY_ENV_VAR]
    try:
        router, worker = initialize_llms()
        logger.info(f"Initialized LLMs: Router type = {type(router).__name__}, Worker type = {type(worker).__name__}")
        assert isinstance(router, MockLLM) and isinstance(worker, MockLLM), "Should be MockLLM instances"
        logger.info("Scenario 1 PASSED: Explicitly used mock LLMs.")
    except Exception as e:
        logger.error(f"Error in Scenario 1: {e}", exc_info=True)
    finally:
        del os.environ[USE_MOCK_LLMS_ENV_VAR]

    # Scenario 2: Implicitly request real LLMs, but no API key (should fall back to mocks)
    logger.info("\n--- Scenario 2: Testing with no API key (should fall back to mocks) ---")
    # Ensure USE_MOCK_LLMS is not true
    if USE_MOCK_LLMS_ENV_VAR in os.environ:
        del os.environ[USE_MOCK_LLMS_ENV_VAR]
    # Ensure API key is not set
    if GOOGLE_API_KEY_ENV_VAR in os.environ:
        del os.environ[GOOGLE_API_KEY_ENV_VAR]
    try:
        router, worker = initialize_llms()
        logger.info(f"Initialized LLMs: Router type = {type(router).__name__}, Worker type = {type(worker).__name__}")
        if LANGCHAIN_GOOGLE_GENAI_AVAILABLE: # Only assert mock if the lib is there, otherwise it's expected to be mock
            assert isinstance(router, MockLLM) and isinstance(worker, MockLLM), "Should fall back to MockLLM instances if no API key"
            logger.info("Scenario 2 PASSED: Fell back to mock LLMs due to missing API key.")
        else:
            assert isinstance(router, MockLLM) and isinstance(worker, MockLLM), "Should use MockLLM if langchain-google-genai not available"
            logger.info("Scenario 2 PASSED: Used mock LLMs as langchain-google-genai is not available.")

    except Exception as e:
        logger.error(f"Error in Scenario 2: {e}", exc_info=True)


    # Scenario 3: Attempt real LLMs with API key (if key is provided externally and lib is installed)
    logger.info("\n--- Scenario 3: Testing with API key (if GOOGLE_API_KEY is set externally) ---")
    # Ensure USE_MOCK_LLMS is not true
    if USE_MOCK_LLMS_ENV_VAR in os.environ:
        del os.environ[USE_MOCK_LLMS_ENV_VAR]
    # User must set GOOGLE_API_KEY in their environment for this to run with real LLMs
    if os.getenv(GOOGLE_API_KEY_ENV_VAR) and LANGCHAIN_GOOGLE_GENAI_AVAILABLE:
        try:
            router, worker = initialize_llms()
            logger.info(f"Initialized LLMs: Router type = {type(router).__name__}, Worker type = {type(worker).__name__}")
            assert not isinstance(router, MockLLM), "Should be a real LLM instance if API key is provided"
            logger.info("Scenario 3 PASSED: Initialized real LLMs (GOOGLE_API_KEY was set).")
            # Example invoke (optional, uncomment to test further)
            # response = router.invoke("Hello")
            # logger.info(f"Real router response: {response.content}")
        except Exception as e:
            logger.error(f"Error in Scenario 3: {e}", exc_info=True)
    elif not LANGCHAIN_GOOGLE_GENAI_AVAILABLE:
        logger.warning("Scenario 3 SKIPPED: langchain-google-genai not installed. Cannot test real LLMs.")
    else:
        logger.warning(f"Scenario 3 SKIPPED: {GOOGLE_API_KEY_ENV_VAR} not set in environment. Cannot test real LLMs.")

    logger.info("\nDirect test of llm_config.py completed.")
    