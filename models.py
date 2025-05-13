# models.py
import logging
from typing import Any, Dict, List, Optional, Literal, Annotated
import operator 
from pydantic import BaseModel, Field, ValidationError, model_validator, field_validator
from datetime import datetime

logger = logging.getLogger(__name__)

# --- Graph Representation Models (Used by Graph 1 to define plan for Graph 2) ---
class InputMapping(BaseModel):
    """Defines how to map data from a source node's output or a shared pool to a target node's input."""
    source_operation_id: str = Field(
        ...,
        description="Effective_id of the source API operation node that (typically) produced the data, or a conceptual source like 'initial_input'. Primarily for context and graph readability."
    )
    source_data_path: str = Field(
        ...,
        description="JSONPath-like string to extract data from the shared 'extracted_ids' data pool (e.g., '$.id', '$.data.items[0].name', '$.some_key_from_output_mapping'). This path directly accesses the shared pool."
    )
    target_parameter_name: str = Field(..., description="Name of the parameter/field in the current node's operation (e.g., 'userId', 'productId', 'full_body_object', 'specific_field_in_body').")
    target_parameter_in: Literal["path", "query", "header", "cookie", "body", "body.fieldName"] = Field(
        ...,
        description=(
            "Location of the target parameter. "
            "'path': for path parameters like /users/{userId}. "
            "'query': for URL query parameters like ?name=value. "
            "'header': for request headers. "
            "'body': to replace the entire request body with the source value. "
            "'body.fieldName': to set a specific field within a JSON request body (e.g., 'body.user.name')."
        )
    )
    transformation: Optional[str] = Field(None, description="Optional instruction for transforming data (e.g., 'format as date string'). Placeholder for future advanced transformations.")

class OutputMapping(BaseModel):
    """Defines how to extract data from a node's response and where to store it in the shared 'extracted_ids' pool."""
    source_data_path: str = Field(..., description="JSONPath-like string to extract data from the node's JSON response body (e.g., '$.id', '$.data.token', '$.items[*]').")
    target_data_key: str = Field(..., description="The key under which the extracted data will be stored in the shared 'extracted_ids' data pool for subsequent nodes (e.g., 'userAuthToken', 'createdItemId').")

class Node(BaseModel):
    """Represents a node (an API call or a system operation) in the execution graph."""
    operationId: str = Field(..., description="Original operationId from the OpenAPI spec, or a system identifier like 'START_NODE', 'END_NODE'.")
    display_name: Optional[str] = Field(None, description="Optional unique name for this node instance if the same operationId is used multiple times in a graph (e.g., 'getUser_firstCall', 'getUser_secondCall'). If None, operationId is used as the effective_id.")
    summary: Optional[str] = Field(None, description="Short summary of the API operation or system node's purpose.")
    description: Optional[str] = Field(None, description="Detailed description of this step's purpose within the workflow context.")
    
    method: Optional[str] = Field(None, description="HTTP method for the API call (e.g., GET, POST). For system nodes like START_NODE/END_NODE, this can be 'SYSTEM' or None.")
    path: Optional[str] = Field(None, description="API path template (e.g., /users/{userId}). For system nodes, this can be a conceptual path like '/start' or '/end'.")
    
    payload: Optional[Dict[str, Any]] = Field(None, description="Payload template for the API request (used by Graph 2 for execution). Graph 1 (planning) might populate this based on its 'payload_description' or generation logic. Should be a dictionary for JSON payloads, or can be other types if handled by APIExecutor.")
    payload_description: Optional[str] = Field(None, description="Natural language description or JSON string template of an example request payload and expected response structure. (Primarily used by Graph 1 for planning and LLM context).")
    
    input_mappings: List[InputMapping] = Field(default_factory=list, description="Defines how data from previous nodes or a shared pool maps to this node's inputs.")
    output_mappings: List[OutputMapping] = Field(default_factory=list, description="Defines how to extract data from this node's response into a shared data pool.")
    
    requires_confirmation: bool = Field(False, description="If true, the workflow should interrupt for user confirmation before executing this node (e.g., for POST, PUT, DELETE operations that modify data).")
    confirmation_prompt: Optional[str] = Field(None, description="Custom prompt to show the user for confirmation if requires_confirmation is true. If None, a default prompt may be generated.")

    @property
    def effective_id(self) -> str:
        """
        Returns the unique identifier for this node instance within the graph.
        Uses display_name if provided, otherwise defaults to operationId.
        """
        return self.display_name if self.display_name else self.operationId

class Edge(BaseModel):
    """Represents a directed edge (dependency or control flow) in the execution graph."""
    from_node: str = Field(..., description="Effective_id of the source node (or 'START' for LangGraph's entry point).") 
    to_node: str = Field(..., description="Effective_id of the target node (or 'END' for LangGraph's exit point).") 
    description: Optional[str] = Field(None, description="Optional natural language description of why this edge exists or the condition it represents.")

    def __hash__(self):
        return hash((self.from_node, self.to_node))

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return NotImplemented
        return self.from_node == other.from_node and self.to_node == other.to_node

class GraphOutput(BaseModel):
    """
    Represents the generated API execution graph/plan.
    This is the output of Graph 1 (Planning) and input to Graph 2 (Execution).
    """
    graph_id: Optional[str] = Field(None, description="Optional unique identifier for this graph plan.")
    description: Optional[str] = Field(None, description="Overall natural language description of the workflow's purpose and flow.")
    nodes: List[Node] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)
    refinement_summary: Optional[str] = Field(None, description="Summary of the last refinement made to this graph by an LLM, or a note from the generation process.")

    @model_validator(mode='after')
    def check_graph_integrity(self) -> 'GraphOutput':
        """
        Validates the structural integrity of the graph:
        - Node effective_ids must be unique.
        - Edge 'from_node' and 'to_node' must refer to existing node effective_ids or 'START'/'END'.
        """
        if not self.nodes: 
            if self.edges: 
                raise ValueError("Graph has edges but no nodes defined.")
            return self 
            
        node_effective_ids = {node.effective_id for node in self.nodes}
        if len(node_effective_ids) != len(self.nodes):
            seen_ids = set()
            duplicates = [node.effective_id for node in self.nodes if node.effective_id in seen_ids or seen_ids.add(node.effective_id)] # type: ignore[func-returns-value]
            raise ValueError(f"Duplicate node effective_ids found: {list(set(duplicates))}. Use 'display_name' to differentiate nodes if using the same operationId multiple times.")

        for edge in self.edges:
            is_from_special_node = edge.from_node.upper() == "START" 
            is_to_special_node = edge.to_node.upper() == "END"     
            
            if not is_from_special_node and edge.from_node not in node_effective_ids:
                raise ValueError(f"Edge source node '{edge.from_node}' not found in graph nodes (and not 'START').")
            if not is_to_special_node and edge.to_node not in node_effective_ids:
                raise ValueError(f"Edge target node '{edge.to_node}' not found in graph nodes (and not 'END').")
        return self

class BotState(BaseModel):
    """Represents the full state of the conversation and processing for Graph 1 (Planning Graph)."""
    session_id: str = Field(..., description="Unique identifier for the current user session.")
    user_input: Optional[str] = Field(None, description="The latest input received from the user.")
    openapi_spec_string: Optional[str] = Field(None, description="Temporary storage for raw OpenAPI spec text received from user input, awaiting parsing.")
    openapi_spec_text: Optional[str] = Field(None, description="The validated and successfully parsed OpenAPI specification text (JSON or YAML as string).")
    openapi_schema: Optional[Dict[str, Any]] = Field(None, description="The parsed OpenAPI schema as a Python dictionary.")
    schema_cache_key: Optional[str] = Field(None, description="Cache key derived from the content of the openapi_spec_text.")
    schema_summary: Optional[str] = Field(None, description="LLM-generated natural language summary of the OpenAPI schema.")
    input_is_spec: bool = Field(False, description="Flag indicating if the last user input was identified as an OpenAPI specification.")
    identified_apis: List[Dict[str, Any]] = Field(default_factory=list, description="List of API operations identified from the schema (includes operationId, method, path, summary, parameters, requestBody details).")
    payload_descriptions: Dict[str, str] = Field(default_factory=dict, description="Maps operationId to LLM-generated example payload and response structure descriptions.")
    execution_graph: Optional[GraphOutput] = Field(None, description="The generated API execution graph/plan (output of Graph 1, input to Graph 2). This defines the sequence of API calls.")
    plan_generation_goal: Optional[str] = Field(None, description="The user's stated goal or objective for the current execution graph generation or refinement.")
    graph_refinement_iterations: int = Field(0, description="Counter for the number of refinement attempts made on the current execution_graph.")
    max_refinement_iterations: int = Field(3, description="Maximum number of allowed refinement iterations before attempting full regeneration or stopping.")
    graph_regeneration_reason: Optional[str] = Field(None, description="Feedback or reason provided for why the graph needs regeneration or refinement (e.g., from user input or verification failure).")
    workflow_execution_status: Literal[
        "idle", "pending_start", "running", "paused_for_confirmation", "completed", "failed"
    ] = Field("idle", description="Current status of the API workflow execution (primarily managed by Graph 2, reflected here).")
    workflow_execution_results: Dict[str, Any] = Field(default_factory=dict, description="Summary or key results from Graph 2's executed nodes, for Graph 1's awareness and potential use in subsequent planning.")
    workflow_extracted_data: Dict[str, Any] = Field(default_factory=dict, description="Data that Graph 1 might want to initialize Graph 2 with, or a summary of data extracted by Graph 2's output mappings for broader use by Graph 1.")
    intent: Optional[str] = Field(None, description="User's high-level intent as determined by Graph 1's router (e.g., 'parse_openapi_spec', 'answer_openapi_query').")
    loop_counter: int = Field(0, description="Counter for detecting potential routing loops within Graph 1.")
    extracted_params: Optional[Dict[str, Any]] = Field(None, description="Parameters extracted by Graph 1's router or other nodes for specific actions (e.g., goal for graph generation).")
    final_response: str = Field("", description="The final, user-facing response generated by Graph 1 for the current turn.")
    response: Optional[str] = Field(None, description="Intermediate response message generated by Graph 1 nodes during processing. This is typically shown to the user before the final_response.")
    next_step: Optional[str] = Field(None, alias="__next__", exclude=True, description="Internal LangGraph field: specifies the next node to execute in Graph 1. Set by nodes to control flow.")
    scratchpad: Dict[str, Any] = Field(default_factory=dict, description="A dictionary for Graph 1 to store temporary data, intermediate results, logs, or any other information useful during a single processing cycle but not meant for long-term state.")

    class Config:
        extra = 'allow' 
        validate_assignment = True 
        populate_by_name = True 

    def update_scratchpad_reason(self, tool_name: str, details: str):
        """Helper method to log reasoning steps into the scratchpad."""
        if not isinstance(self.scratchpad, dict): self.scratchpad = {} 
        reason_log = self.scratchpad.get('reasoning_log', [])
        if not isinstance(reason_log, list): reason_log = [] 
        timestamp = datetime.now().isoformat()
        reason_log.append({"timestamp": timestamp, "tool": tool_name, "details": details})
        self.scratchpad['reasoning_log'] = reason_log[-100:] 
        logger.debug(f"Scratchpad Updated by {tool_name}: {details[:200]}...")

class ExecutionGraphState(BaseModel):
    """
    Defines the runtime state passed between nodes in the Execution LangGraph (Graph 2).
    This state is managed by the LangGraph `StateGraph` for the execution phase.
    """
    api_results: Annotated[Dict[str, Any], operator.ior] = Field( 
        default_factory=dict,
        description="Stores outputs of API calls: {'node_effective_id': result_dict_from_APIExecutor}. Merged using dictionary update."
    )
    extracted_ids: Annotated[Dict[str, Any], operator.ior] = Field( 
        default_factory=dict,
        description="Shared data pool: stores data extracted via OutputMappings from node responses (e.g., {'user_token': 'xyz123'}), used by InputMappings."
    )
    confirmed_data: Annotated[Dict[str, Any], operator.ior] = Field( 
        default_factory=dict,
        description="Stores data confirmed or modified by the user during interrupts for Graph 2 nodes (e.g., {'confirmed_opId_createOrder': True, 'confirmed_opId_createOrder_details': {...user_payload...}})."
    )
    initial_input: Optional[Dict[str, Any]] = Field( 
        None,
        description="Initial input values provided to Graph 2 at the start of its execution, can be used for placeholder resolution."
    )
    # NEW FIELD to hold data for UI confirmation
    pending_confirmation_data: Optional[Dict[str, Any]] = Field(
        None,
        description="If set, indicates that the graph is paused awaiting user confirmation. Contains data for the UI prompt."
    )
    error: Optional[str] = Field( 
        None,
        description="Stores error messages if any step in Graph 2 fails, helping to halt or diagnose issues."
    )

    class Config:
        arbitrary_types_allowed = True
