# OpenAPI Workflow Agent

## 1. Overview

The OpenAPI Workflow Agent is a web-based application designed to help users understand, plan, and execute sequences of API calls based on an OpenAPI (formerly Swagger) specification. Users can provide an OpenAPI specification, define a high-level goal, and the system will generate a potential workflow (an "execution graph") of API calls to achieve that goal. This graph can then be reviewed, refined, and executed, with support for human-in-the-loop confirmations for sensitive operations.

The agent leverages Large Language Models (LLMs) for understanding user intent, planning API sequences, and generating descriptive content. It uses LangGraph for orchestrating both the planning and execution phases.

## 2. Key Features

* **OpenAPI Specification Processing:** Parses and analyzes user-provided OpenAPI v2/v3 specifications (JSON or YAML).
* **Automated API Workflow Planning (Graph 1):**
    * Generates a summary of the API.
    * Identifies available API operations.
    * Generates example request payloads and response structures.
    * Creates an "Execution Graph" (a plan of API calls) based on a user's natural language goal.
* **Interactive Graph Refinement:** Allows users to provide feedback to refine the generated execution graph.
* **Graph Verification:** Performs basic structural checks on the generated graph.
* **API Workflow Execution (Graph 2):**
    * Executes the planned API calls sequentially.
    * Manages data flow between API calls (mapping outputs of one call to inputs of another).
    * Supports human-in-the-loop for operations requiring user confirmation (e.g., POST, PUT, DELETE).
* **Web-Based User Interface:**
    * Chat interface for interacting with the agent.
    * Visualization of the execution graph (JSON and DAG/diagram view using Mermaid.js).
    * Modal for user confirmations during workflow execution.
* **LLM Integration:** Uses LLMs (configurable, e.g., Google Gemini, with mock fallbacks) for:
    * Routing user intent.
    * Generating API summaries, payload examples, and graph descriptions.
    * Planning and refining API execution graphs.
* **Asynchronous Operations:** Built with FastAPI and `asyncio` for non-blocking backend operations.
* **Real-time Updates:** Uses WebSockets for communication between the frontend and backend.

## 3. Architecture

The system is built around a two-graph architecture, both orchestrated using LangGraph:

1.  **Planning Graph (Graph 1):**
    * Resides in `core_logic.py` and `graph.py`.
    * Takes user input (OpenAPI spec, goals, refinement instructions).
    * Interacts with an LLM (the "worker LLM") to analyze the spec, generate descriptions, and create/refine the execution plan (an instance of `GraphOutput` model).
    * This graph handles the "thinking" and "planning" part of the agent.

2.  **Execution Graph (Graph 2):**
    * Defined by `execution_graph_definition.py` and managed by `execution_manager.py`.
    * Takes the `GraphOutput` plan from Graph 1 as its definition.
    * Each node in this graph corresponds to an API call or a system step.
    * Uses `api_executor.py` to make the actual HTTP requests.
    * Handles data mapping between API calls and manages human-in-the-loop interruptions.

**Communication Flow:**
* **Frontend (UI):** `index.html`, `style.css`, `script.js`
* **Backend API Server:** `main.py` (FastAPI)
* **WebSocket Communication:** `websocket_helpers.py` manages the WebSocket connection and message flow between frontend and backend graphs.
* **LLM Configuration:** `llm_config.py` handles the initialization of LLMs (real or mock).
* **Data Models:** `models.py` defines Pydantic models for state management and graph structures.

## 4. Core Components

* **Frontend (`static/` directory):**
    * `index.html`: Main application page.
    * `style.css`: Styling for the UI.
    * `script.js`: Client-side logic, WebSocket handling, DOM manipulation, Mermaid graph rendering.
* **Backend (Python files):**
    * `main.py`: FastAPI application entry point, WebSocket endpoint, startup/shutdown events.
    * `websocket_helpers.py`: Manages WebSocket connections and message routing.
    * `graph.py`: Defines and builds the main Planning Graph (Graph 1).
    * `core_logic.py`: Contains the core business logic for Graph 1 nodes (parsing specs, generating plans, etc.).
    * `router.py`: Routes user input to appropriate nodes in Graph 1.
    * `execution_graph_definition.py`: Defines how an execution plan (from Graph 1) is translated into a runnable LangGraph (Graph 2).
    * `execution_manager.py`: Manages the lifecycle and execution of Graph 2 instances, including interruptions.
    * `api_executor.py`: Responsible for making the actual HTTP API calls.
    * `llm_config.py`: Configures and initializes LLMs (e.g., Google Gemini or mock LLMs).
    * `models.py`: Pydantic models for application state, graph structures, API definitions, etc.
    * `utils.py`: Utility functions (caching, JSON parsing, etc.).
    * `workflow_executor.py`: (Appears to be a simplified or alternative executor, the primary one seems to be `execution_manager.py`).

## 5. Setup and Installation

### Prerequisites

* Python 3.8+
* `pip` (Python package installer)
* A modern web browser

### Installation Steps

1.  **Clone the Repository (if applicable):**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Create a Virtual Environment:**
    It's highly recommended to use a virtual environment.
    ```bash
    python -m venv venv
    ```
    Activate the environment:
    * Windows: `venv\Scripts\activate`
    * macOS/Linux: `source venv/bin/activate`

3.  **Install Dependencies:**
    Install all required Python packages from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Environment Variables:**
    The application uses environment variables for configuration. Create a `.env` file in the root directory of the project and add the following variables as needed:

    ```env
    # For Google Gemini LLM (required if not using mock LLMs)
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"

    # To force the use of mock LLMs (set to "true" to use mocks, "false" or omit to try real LLMs)
    USE_MOCK_LLMS="true" 

    # Optional: Default base URL for the APIExecutor if your OpenAPI specs use relative paths
    # DEFAULT_API_BASE_URL="[https://api.example.com/v1](https://api.example.com/v1)"

    # Optional: API timeout in seconds
    # API_TIMEOUT="30.0"

    # Optional: LLM Model Names (defaults are provided in llm_config.py)
    # ROUTER_LLM_MODEL_NAME="gemini-1.5-flash-latest"
    # WORKER_LLM_MODEL_NAME="gemini-1.5-pro-latest" 
    # ROUTER_LLM_TEMPERATURE="0.2"
    # WORKER_LLM_TEMPERATURE="0.7"
    ```
    * If `USE_MOCK_LLMS` is not set to `"true"`, you **must** provide a `GOOGLE_API_KEY` for the application to function with real LLMs.
    * The mock LLMs provide predefined responses and are useful for development and testing without API key costs.

## 6. Running the Application

1.  **Ensure your virtual environment is activated.**
2.  **Start the FastAPI Server:**
    From the root directory of the project, run:
    ```bash
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
    ```
    * `--reload`: Enables auto-reloading when code changes (useful for development).
    * `--host 0.0.0.0`: Makes the server accessible on your network.
    * `--port 8000`: Specifies the port.

3.  **Access the UI:**
    Open your web browser and navigate to: `http://localhost:8000` (or `http://<your-machine-ip>:8000` if accessing from another device on your network).

## 7. How It Works (User Flow)

1.  **Provide OpenAPI Spec:** The user pastes an OpenAPI v2 or v3 specification (JSON or YAML format) into the chat input.
2.  **System Parses Spec:** The backend (Graph 1) parses the spec. It identifies API operations, generates a summary, and prepares for planning.
3.  **Define a Goal:** The user types a natural language goal for the API workflow (e.g., "Create a new user and then retrieve their details").
4.  **Generate Execution Graph:** Graph 1, using an LLM, generates an execution graph (a sequence of API calls with input/output mappings) to achieve the goal. This graph is displayed in the UI (JSON and DAG views).
5.  **Review and Refine (Optional):** The user can review the graph. If it's not satisfactory, they can provide feedback (e.g., "Add a notification step after creating the user") to refine the graph. Graph 1 will attempt to update the plan.
6.  **Verify Graph:** The system performs basic verification on the graph structure.
7.  **Run Workflow:** The user clicks the "Run Workflow" button.
8.  **Execute APIs (Graph 2):**
    * The backend initiates Graph 2 based on the current execution graph.
    * Graph 2 executes each API call in the planned sequence.
    * The `APIExecutor` makes the actual HTTP requests.
    * Data from one API call's response is mapped to the next API call's input as defined in the graph.
9.  **Human-in-the-Loop:** If an API call in the graph is marked as `requires_confirmation` (typically for POST, PUT, DELETE operations):
    * Graph 2 pauses.
    * A confirmation modal appears in the UI, showing details of the API call and its payload.
    * The user can review, modify the payload if necessary, and then "Confirm & Proceed" or "Cancel".
    * Graph 2 resumes or stops based on the user's decision.
10. **View Results:** API call responses and workflow status updates are displayed in the chat interface.

## 8. Key Files & Purpose

* `main.py`: FastAPI server, WebSocket endpoint.
* `websocket_helpers.py`: Manages WebSocket communication logic.
* `graph.py`: Defines the LangGraph structure for the Planning Graph (Graph 1).
* `core_logic.py`: Implements the actions performed by the nodes in Graph 1 (e.g., parsing, planning, refining).
* `router.py`: Determines user intent and routes to the appropriate node in Graph 1.
* `execution_graph_definition.py`: Translates the plan from Graph 1 into a runnable LangGraph (Graph 2).
* `execution_manager.py`: Orchestrates the execution of Graph 2, including interruptions and state management.
* `api_executor.py`: Makes the actual HTTP calls to external APIs.
* `llm_config.py`: Handles LLM setup (real or mock).
* `models.py`: Defines Pydantic data structures for state, graph components, etc.
* `utils.py`: Common utility functions.
* `static/index.html`: The main HTML page for the user interface.
* `static/script.js`: Client-side JavaScript for UI interactions, WebSocket handling, and Mermaid rendering.
* `static/style.css`: CSS for styling the user interface.
* `requirements.txt`: Lists Python dependencies.
* `.env` (user-created): Stores environment variables like API keys.

## 9. Potential Future Enhancements

* **Advanced Error Handling & Retries:** Implement more sophisticated error handling and automatic retry mechanisms for API calls in Graph 2.
* **Conditional Logic in Execution Graphs:** Allow the LLM to generate graphs with conditional branching (e.g., "if API call A succeeds, then call B, else call C").
* **Looping/Iteration:** Support for iterating over a list of items and performing an API call for each.
* **Persistent Graph Storage:** Save and load generated execution graphs to/from a database.
* **Enhanced Graph Visualization:** More interactive graph visualization with real-time status updates on nodes.
* **Security Enhancements:** Stricter input validation, output encoding, and considerations for handling sensitive data in API calls.
* **User Authentication & Authorization:** If the agent needs to interact with protected APIs on behalf of users.
* **Tool/Function Calling Abstraction:** Integrate more robustly with LLM function calling capabilities for more reliable graph generation.
* **Testing Framework:** Comprehensive unit and integration tests.

