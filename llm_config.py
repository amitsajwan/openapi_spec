# llm_config.py
import os
import logging
from typing import Tuple, Any, Dict 
import json # Added for mock graph generation
import asyncio # For potential async operations in mock or real LLMs

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
UTILITY_LLM_MODEL_NAME = os.getenv("UTILITY_LLM_MODEL_NAME", "gemini-1.5-flash-latest") # Smaller model for utility tasks

# --- LLM Configuration Parameters ---
ROUTER_LLM_TEMPERATURE = float(os.getenv("ROUTER_LLM_TEMPERATURE", "0.2"))
WORKER_LLM_TEMPERATURE = float(os.getenv("WORKER_LLM_TEMPERATURE", "0.7"))
UTILITY_LLM_TEMPERATURE = float(os.getenv("UTILITY_LLM_TEMPERATURE", "0.3")) # Temp for utility tasks

# --- Mock LLM Implementation ---
class MockLLMContent:
    """A simple class to mimic the 'content' attribute of an LLM response."""
    def __init__(self, content: str):
        self.content = content

class MockLLM:
    """
    A mock LLM class that mimics the behavior of a LangChain LLM.
    It returns predefined responses based on simple keyword matching in the prompt.
    Can simulate async behavior for testing asyncio.gather.
    """
    def __init__(self, name: str = "MockLLM", simulate_async_delay: float = 0.01):
        self.name = name
        self.simulate_async_delay = simulate_async_delay
        # Predefined responses for the router
        self.router_responses: Dict[str, str] = {
            "classify this intent: 'show me the user details api.'": "answer_openapi_query",
            "classify the user's intent. choose one of the following:": "answer_openapi_query",
            "run this workflow": "setup_workflow_execution",
            "execute the plan": "setup_workflow_execution",
            "focus the plan on": "interactive_query_planner",
            "regenerate the graph for goal": "interactive_query_planner",
            "add a notification step": "interactive_query_planner",
            "what if i want to change": "interactive_query_planner",
            "here is the confirmed payload": "interactive_query_planner",
        }
        # Predefined responses for the worker (more complex tasks)
        self.worker_responses: Dict[str, str] = {
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
        # Predefined responses for the utility LLM (simpler tasks)
        self.utility_responses: Dict[str, str] = {
            "the following text is an openapi specification, potentially in json or yaml format": "openapi: 3.0.3\ninfo:\n  title: Cleaned API by Mock Utility LLM\n  version: '1.0'\npaths:\n  /items:\n    get:\n      summary: Get items\n      responses:\n        '200':\n          description: A list of items.", # Mock cleanup response
            "summarize the following api specification": "This is a mock API summary from Utility LLM. It describes a set of endpoints.",
            "provide a concise, typical json example": json.dumps({ # For payload descriptions
                "Request Payload Example": "```json\n{\"name\": \"mockItemFromUtility\", \"value\": 456}\n```",
                "Expected Response Structure": "Utility LLM: Returns an object with id and name."
            }),
        }
        logger.info(f"{self.name} initialized. Async delay: {self.simulate_async_delay}s")

    def _generate_mock_graph_output(self, refined: bool = False) -> str:
        """Generates a consistent mock GraphOutput JSON string."""
        graph_desc = "Initial mock graph by Worker LLM."
        refinement_summary = "Initial graph with 2 steps by Worker LLM."
        if refined:
            graph_desc = "Refined mock graph with an extra step by Worker LLM."
            refinement_summary = "Refined the graph to include a logging step by Worker LLM."
        # ... (rest of the mock graph output as before)
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

    async def ainvoke(self, prompt: Any, **kwargs) -> MockLLMContent: # Langchain uses ainvoke for async
        """ Mimics async invoke for testing asyncio.gather """
        await asyncio.sleep(self.simulate_async_delay)
        return self.invoke(prompt) # Delegate to synchronous logic after delay

    def invoke(self, prompt: Any, **kwargs) -> MockLLMContent:
        """
        Mimics the LLM's invoke method.
        Returns a MockLLMContent object containing a predefined response.
        """
        prompt_str = str(prompt).lower()
        logger.debug(f"{self.name} received prompt (first 200 chars): {prompt_str[:200]}...")

        response_dict = {}
        if "Router" in self.name:
            response_dict = self.router_responses
        elif "Worker" in self.name: # Default to worker if not specified, or if name matches
            response_dict = {**self.worker_responses, **self.utility_responses} # Worker can also do utility tasks if needed
        elif "Utility" in self.name:
            response_dict = self.utility_responses
        
        # More specific matching for utility tasks if this is the utility LLM
        if "Utility" in self.name:
            if "the following text is an openapi specification" in prompt_str: # Cleanup task
                 return MockLLMContent(self.utility_responses["the following text is an openapi specification, potentially in json or yaml format"])
            if "summarize the following api specification" in prompt_str: # Summary task
                 return MockLLMContent(self.utility_responses["summarize the following api specification"])
            if "provide a concise, typical json example" in prompt_str: # Payload desc task
                 return MockLLMContent(self.utility_responses["provide a concise, typical json example"])


        for keyword, response in response_dict.items():
            if keyword.lower() in prompt_str:
                logger.info(f"{self.name} matched keyword '{keyword}' and will return: {str(response)[:100]}...")
                return MockLLMContent(response)

        default_response = f"I'm {self.name} and I don't have a specific answer for that."
        if "json" in prompt_str or "graphoutput" in prompt_str.lower(): # Graph generation for worker
            default_response = self._generate_mock_graph_output()
        elif "classify" in prompt_str: # Router fallback
             default_response = "handle_unknown"

        logger.warning(f"{self.name} did not find a keyword match. Returning default response: {default_response[:100]}...")
        return MockLLMContent(default_response)

def _get_mock_llms() -> Tuple[MockLLM, MockLLM, MockLLM]:
    """Helper function to instantiate and return mock LLMs."""
    logger.info("Instantiating Mock LLMs (Router, Worker, Utility).")
    mock_router_llm = MockLLM(name="MockRouterLLM")
    mock_worker_llm = MockLLM(name="MockWorkerLLM")
    mock_utility_llm = MockLLM(name="MockUtilityLLM", simulate_async_delay=0.05) # Slightly more delay for utility
    return mock_router_llm, mock_worker_llm, mock_utility_llm

def initialize_llms() -> Tuple[Any, Any, Any]:
    """
    Initializes and returns the router, worker, and utility LLMs.
    - If USE_MOCK_LLMS is 'true', uses mock LLMs.
    - Otherwise, tries to use real Google LLMs.
    - Falls back to mock LLMs if real LLM initialization fails or prerequisites are missing.

    Returns:
        A tuple containing (router_llm, worker_llm, utility_llm).
    """
    use_mocks_explicitly = os.getenv(USE_MOCK_LLMS_ENV_VAR, "false").lower() == "true"

    if use_mocks_explicitly:
        logger.info("Explicitly using Mock LLMs as USE_MOCK_LLMS is set to true.")
        return _get_mock_llms()

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

        utility_llm = ChatGoogleGenerativeAI(
            model=UTILITY_LLM_MODEL_NAME,
            temperature=UTILITY_LLM_TEMPERATURE,
            google_api_key=google_api_key,
            convert_system_message_to_human=True,
        )
        logger.info(f"Real Utility LLM initialized successfully with model: {UTILITY_LLM_MODEL_NAME}")

        return router_llm, worker_llm, utility_llm

    except Exception as e:
        logger.error(f"Failed to initialize real Google Generative AI models: {e}. Falling back to Mock LLMs.", exc_info=True)
        return _get_mock_llms()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    # ... (Test scenarios can be updated to check for three LLMs) ...
    logger.info("\n--- Scenario 1: Testing with USE_MOCK_LLMS=true ---")
    os.environ[USE_MOCK_LLMS_ENV_VAR] = "true"
    if GOOGLE_API_KEY_ENV_VAR in os.environ: del os.environ[GOOGLE_API_KEY_ENV_VAR]
    try:
        r_llm, w_llm, u_llm = initialize_llms()
        logger.info(f"Initialized LLMs: Router='{type(r_llm).__name__}', Worker='{type(w_llm).__name__}', Utility='{type(u_llm).__name__}'")
        assert isinstance(r_llm, MockLLM) and isinstance(w_llm, MockLLM) and isinstance(u_llm, MockLLM)
        # Test utility mock
        # cleanup_resp = u_llm.invoke("the following text is an openapi specification")
        # logger.info(f"Mock utility cleanup response: {cleanup_resp.content[:100]}...")
        # summary_resp = u_llm.invoke("summarize the following api specification")
        # logger.info(f"Mock utility summary response: {summary_resp.content[:100]}...")

        logger.info("Scenario 1 PASSED: Explicitly used mock LLMs.")
    except Exception as e_main:
        logger.error(f"Error in main test block (Scenario 1): {e_main}", exc_info=True)
    finally:
        if USE_MOCK_LLMS_ENV_VAR in os.environ: del os.environ[USE_MOCK_LLMS_ENV_VAR]

    logger.info("\nDirect test of llm_config.py completed.")
