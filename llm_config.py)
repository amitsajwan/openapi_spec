# llm_config.py
import os
import logging
from typing import Tuple, Any

# Attempt to import the ChatGoogleGenerativeAI class
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    # Fallback or error handling if the import fails.
    # This is crucial if the environment might not have the package.
    logging.error(
        "Failed to import ChatGoogleGenerativeAI from langchain_google_genai. "
        "Please ensure 'langchain-google-genai' is installed."
    )
    # Define a placeholder or raise an error to prevent the application from starting incorrectly.
    # For now, we'll define a placeholder that will cause issues later if not addressed,
    # forcing the user to ensure the package is installed.
    ChatGoogleGenerativeAI = None

logger = logging.getLogger(__name__)

# --- Environment Variable Names ---
GOOGLE_API_KEY_ENV_VAR = "GOOGLE_API_KEY"

# --- Model Names ---
# Using "latest" is convenient but be mindful of potential breaking changes.
# For production, consider pinning to a specific version, e.g., "gemini-1.5-pro-001"
ROUTER_LLM_MODEL_NAME = os.getenv("ROUTER_LLM_MODEL_NAME", "gemini-1.5-flash-latest") # Faster, good for classification/routing
WORKER_LLM_MODEL_NAME = os.getenv("WORKER_LLM_MODEL_NAME", "gemini-1.5-pro-latest") # More powerful for generation

# --- LLM Configuration Parameters ---
ROUTER_LLM_TEMPERATURE = float(os.getenv("ROUTER_LLM_TEMPERATURE", "0.2")) # Lower for more deterministic routing
WORKER_LLM_TEMPERATURE = float(os.getenv("WORKER_LLM_TEMPERATURE", "0.7")) # Standard for creative/generative tasks

def initialize_llms() -> Tuple[Any, Any]:
    """
    Initializes and returns the router and worker LLMs.

    Retrieves the GOOGLE_API_KEY from environment variables.
    Configures ChatGoogleGenerativeAI models for routing and worker tasks.

    Returns:
        A tuple containing (router_llm, worker_llm).

    Raises:
        ValueError: If the GOOGLE_API_KEY environment variable is not set.
        RuntimeError: If ChatGoogleGenerativeAI could not be imported.
    """
    if ChatGoogleGenerativeAI is None:
        raise RuntimeError(
            "ChatGoogleGenerativeAI class not available. "
            "Please ensure 'langchain-google-genai' is installed correctly."
        )

    google_api_key = os.getenv(GOOGLE_API_KEY_ENV_VAR)
    if not google_api_key:
        logger.error(f"{GOOGLE_API_KEY_ENV_VAR} environment variable not found.")
        raise ValueError(
            f"Missing Google API Key. Please set the {GOOGLE_API_KEY_ENV_VAR} environment variable."
        )

    try:
        # Initialize Router LLM (e.g., Gemini Flash for speed and cost-effectiveness in routing)
        router_llm = ChatGoogleGenerativeAI(
            model=ROUTER_LLM_MODEL_NAME,
            temperature=ROUTER_LLM_TEMPERATURE,
            google_api_key=google_api_key,
            convert_system_message_to_human=True, # Often useful for Gemini models
            # top_p=0.95, # Optional: nucleus sampling
            # top_k=40,   # Optional: top-k sampling
        )
        logger.info(f"Router LLM initialized successfully with model: {ROUTER_LLM_MODEL_NAME}")

        # Initialize Worker LLM (e.g., Gemini Pro for more complex tasks)
        worker_llm = ChatGoogleGenerativeAI(
            model=WORKER_LLM_MODEL_NAME,
            temperature=WORKER_LLM_TEMPERATURE,
            google_api_key=google_api_key,
            convert_system_message_to_human=True,
            # You can add other parameters like top_p, top_k if needed
            # safety_settings={ # Optional: configure safety settings if defaults are not suitable
            #     HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            #     HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            # },
        )
        logger.info(f"Worker LLM initialized successfully with model: {WORKER_LLM_MODEL_NAME}")

        return router_llm, worker_llm

    except Exception as e:
        logger.critical(f"Failed to initialize Google Generative AI models: {e}", exc_info=True)
        # Depending on the application's needs, you might re-raise or return None/placeholders
        # For this setup, re-raising is appropriate as LLMs are critical.
        raise RuntimeError(f"Could not initialize LLMs: {e}")

if __name__ == "__main__":
    # Example of how to use the initializer (for direct testing of this file)
    # Ensure your GOOGLE_API_KEY is set in your environment to run this test.
    logging.basicConfig(level=logging.INFO)
    logger.info("Attempting to initialize LLMs for a direct test...")
    try:
        router, worker = initialize_llms()
        logger.info("LLMs initialized for testing.")
        
        # Test router LLM
        # router_response = router.invoke("Classify this intent: 'Show me the user details API.'")
        # logger.info(f"Router LLM test response: {router_response.content}")

        # Test worker LLM
        # worker_response = worker.invoke("Explain the concept of an API in one sentence.")
        # logger.info(f"Worker LLM test response: {worker_response.content}")

        logger.info("Direct test of llm_config.py completed. (Actual invokes commented out by default)")
    except ValueError as ve:
        logger.error(f"Configuration error during test: {ve}")
    except RuntimeError as rte:
        logger.error(f"Runtime error during test: {rte}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during the test: {e}", exc_info=True)

