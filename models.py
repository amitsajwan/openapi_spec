# models.py
import logging
from typing import Any, Dict, List, Optional, Literal, Annotated
import operator # Added for Annotated
from pydantic import BaseModel, Field, ValidationError, model_validator
from datetime import datetime

logger = logging.getLogger(__name__)

# --- Graph Representation Models (Used by Graph 1 to define plan for Graph 2) ---
class InputMapping(BaseModel):
    """Defines how to map data from a source node's output to a target node's input."""
    source_operation_id: str = Field(..., description="Effective_id of the source node providing data (can be a general key if data is in a shared pool).")
    source_data_path: str = Field(..., description="JSONPath-like string to extract data (e.g., '$.id', '$.data.items[0].name', '$.extracted_key_name'). Assumes data is in a shared pool accessible by this path.")
    target_parameter_name: str = Field(..., description="Name of the parameter/field in the current node's operation.")
    target_parameter_in: Literal["path", "query", "header", "cookie", "body", "body.fieldName"] = Field(..., description="Location of the target parameter.")
    transformation: Optional[str] = Field(None, description="Optional instruction for transforming data (e.g., 'format as date string'). Placeholder for future.")

class OutputMapping(BaseModel):
    """Defines how to extract data from a node's response and where to store it."""
    source_data_path: str = Field(..., description="JSONPath-like string to extract data from the node's JSON response body (e.g., '$.id', '$.data.token').")
    target_data_key: str = Field(..., description="The key under which the extracted data will be stored in the shared 'extracted_data' pool for subsequent nodes.")
    # common_model_field: Optional[str] = Field(None, description="Optional: If this output corresponds to a known field in a common data model (e.g., 'user_id', 'product_id').")


class Node(BaseModel):
    """Represents a node (an API call) in the execution graph."""
    operationId: str = Field(..., description="Original operationId from the OpenAPI spec.")
    display_name: Optional[str] = Field(None, description="Unique name for this node instance if operationId is reused (e.g., 'getUser_step1').")
    summary: Optional[str] = Field(None, description="Short summary of the API operation.")
    description: Optional[str] = Field(None, description="Detailed description of this step's purpose in the workflow.")
    
    method: Optional[str] = Field(None, description="HTTP method for the API call (e.g., GET, POST). Populated during graph generation or API identification.")
    path: Optional[str] = Field(None, description="API path template (e.g., /users/{userId}). Populated during graph generation or API identification.")
    
    # This field combines your original 'payload_description' and the need for a payload template for Graph 2.
    # Graph 1 (planning) might populate 'payload_description' with natural language or a JSON string template.
    # Graph 2 (execution) will expect 'payload' to be a dictionary template or actual payload after resolution.
    payload: Optional[Dict[str, Any]] = Field(None, description="Payload template for the API request (used by Graph 2). Graph 1 might populate this based on its 'payload_description' or generation logic.")
    payload_description: Optional[str] = Field(None, description="Natural language description or JSON string template of an example request payload and expected response structure. (Used by Graph 1)")
    
    input_mappings: List[InputMapping] = Field(default_factory=list, description="How data from previous nodes or a shared pool maps to this node's inputs.")
    output_mappings: List[OutputMapping] = Field(default_factory=list, description="How to extract data from this node's response into a shared pool.")
    
    requires_confirmation: bool = Field(False, description="If true, workflow should interrupt for user confirmation before executing this node (e.g., for POST, PUT, DELETE).")
    confirmation_prompt: Optional[str] = Field(None, description="Custom prompt to show user for confirmation if requires_confirmation is true.")


    @property
    def effective_id(self) -> str:
        """Returns the unique identifier for this node instance in the graph."""
        return self.display_name if self.display_name else self.operationId

class Edge(BaseModel):
    """Represents a directed edge (dependency) in the execution graph."""
    from_node: str = Field(..., description="Effective_id of the source node (or 'START' for LangGraph).") 
    to_node: str = Field(..., description="Effective_id of the target node (or 'END' for LangGraph).") 
    description: Optional[str] = Field(None, description="Reason for the dependency.")

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
    refinement_summary: Optional[str] = Field(None, description="Summary of the last refinement made to this graph by an LLM.")

    @model_validator(mode='after')
    def check_graph_integrity(self) -> 'GraphOutput':
        if not self.nodes: 
            return self 
            
        node_effective_ids = {node.effective_id for node in self.nodes}
        if len(node_effective_ids) != len(self.nodes):
            seen_ids = set()
            duplicates = [node.effective_id for node in self.nodes if node.effective_id in seen_ids or seen_ids.add(node.effective_id)] # type: ignore
            raise ValueError(f"Duplicate node effective_ids found: {list(set(duplicates))}. Use 'display_name' for duplicate operationIds.")

        for edge in self.edges:
            is_from_start_node = edge.from_node.upper() == "START" 
            is_to_end_node = edge.to_node.upper() == "END" 
            
            if not is_from_start_node and edge.from_node not in node_effective_ids:
                raise ValueError(f"Edge source node '{edge.from_node}' not found in graph nodes (and not 'START').")
            if not is_to_end_node and edge.to_node not in node_effective_ids:
                raise ValueError(f"Edge target node '{edge.to_node}' not found in graph nodes (and not 'END').")
        return self

# --- State Model for Graph 1 (Planning Graph - Your existing BotState) ---
class BotState(BaseModel):
    """Represents the full state of the conversation and processing for Graph 1."""
    session_id: str = Field(..., description="Unique identifier for the current session.")
    user_input: Optional[str] = Field(None, description="The latest input from the user.")

    openapi_spec_string: Optional[str] = Field(None, description="Temporary storage for raw OpenAPI spec text from user.")
    openapi_spec_text: Optional[str] = Field(None, description="Successfully parsed OpenAPI spec text.")
    openapi_schema: Optional[Dict[str, Any]] = Field(None, description="Parsed OpenAPI schema as a dictionary.")
    schema_cache_key: Optional[str] = Field(None, description="Cache key for the current schema.")
    schema_summary: Optional[str] = Field(None, description="LLM-generated summary of the OpenAPI schema.")
    input_is_spec: bool = Field(False, description="Flag indicating if last input was identified as an OpenAPI spec.")

    identified_apis: List[Dict[str, Any]] = Field(default_factory=list, description="List of APIs identified from spec (operationId, method, path, summary, params, requestBody).")
    payload_descriptions: Dict[str, str] = Field(default_factory=dict, description="Maps operationId to LLM-generated example payload and response descriptions.")

    execution_graph: Optional[GraphOutput] = Field(None, description="The generated API execution graph/plan (output of Graph 1, input to Graph 2).")
    
    plan_generation_goal: Optional[str] = Field(None, description="User's goal for the current execution graph.")
    graph_refinement_iterations: int = Field(0, description="Counter for graph refinement attempts.")
    max_refinement_iterations: int = Field(3, description="Maximum refinement iterations.")
    graph_regeneration_reason: Optional[str] = Field(None, description="Feedback for why graph needs regeneration/refinement.")

    workflow_execution_status: Literal[
        "idle",
        "pending_start", 
        "running",       
        "paused_for_confirmation", 
        "completed",     
        "failed"         
    ] = Field("idle", description="Status of the API workflow execution (managed by Graph 2).")
    
    workflow_execution_results: Dict[str, Any] = Field(default_factory=dict, description="Summary/key results from Graph 2's executed nodes, for Graph 1's awareness.")
    workflow_extracted_data: Dict[str, Any] = Field(default_factory=dict, description="Data that Graph 1 might want to initialize Graph 2 with, or a summary of data extracted by Graph 2.")
    
    intent: Optional[str] = Field(None, description="User's high-level intent from router (for Graph 1).")
    loop_counter: int = Field(0, description="Counter for detecting routing loops in Graph 1.")
    extracted_params: Optional[Dict[str, Any]] = Field(None, description="Parameters extracted by Graph 1's router for specific actions.")

    final_response: str = Field("", description="Final, user-facing response from Graph 1's responder.")
    response: Optional[str] = Field(None, description="Intermediate response message from Graph 1 nodes.")

    next_step: Optional[str] = Field(None, alias="__next__", exclude=True, description="Internal: next LangGraph node for Graph 1.")
    scratchpad: Dict[str, Any] = Field(default_factory=dict, description="Memory for intermediate results, logs, etc., for Graph 1.")

    class Config:
        extra = 'allow' 
        validate_assignment = True
        populate_by_name = True

    def update_scratchpad_reason(self, tool_name: str, details: str):
        if not isinstance(self.scratchpad, dict): self.scratchpad = {}
        reason_log = self.scratchpad.get('reasoning_log', [])
        if not isinstance(reason_log, list): reason_log = []
        timestamp = datetime.now().isoformat()
        reason_log.append({"timestamp": timestamp, "tool": tool_name, "details": details})
        self.scratchpad['reasoning_log'] = reason_log[-100:]
        logger.debug(f"Scratchpad Updated by {tool_name}: {details[:200]}...")

# --- State Definition for Graph 2 (Execution LangGraph) ---
class ExecutionGraphState(BaseModel):
    """
    Defines the runtime state passed between nodes in the Execution LangGraph (Graph 2).
    """
    api_results: Annotated[Dict[str, Any], operator.add] = Field(default_factory=dict, description="Stores outputs of API calls: {'node_effective_id': result_dict}")
    extracted_ids: Annotated[Optional[Dict[str, Any]], operator.ior] = Field(default_factory=dict, description="Shared data pool: stores data extracted via OutputMappings from node responses, used by InputMappings.")
    confirmed_data: Annotated[Optional[Dict[str, Any]], operator.ior] = Field(default_factory=dict, description="Stores data confirmed by user during interrupts for Graph 2 nodes.")
    initial_input: Optional[Dict[str, Any]] = Field(None, description="Initial input values provided to Graph 2, can be used for placeholder resolution.")
    error: Optional[str] = Field(None, description="Stores error messages if any step in Graph 2 fails.")

    class Config:
        arbitrary_types_allowed = True
