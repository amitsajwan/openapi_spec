# workflow_executor.py (Simplified for Debugging RLock & Resume)
import asyncio
import logging
import json
from typing import Any, Callable, Awaitable, Dict, Optional, List

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableLambda

from models import GraphOutput, Node

logger = logging.getLogger(__name__)

class SimplifiedWorkflowState(BaseModel):
    node_visits: List[str] = Field(default_factory=list)
    last_visited_node: Optional[str] = None
    error_message: Optional[str] = None
    # Added field to track if a node is waiting for confirmation
    waiting_for_confirmation_on_node: Optional[str] = None
    last_resume_payload_received: Optional[Dict[str, Any]] = None


    class Config:
        extra = 'allow'


class WorkflowExecutor:
    def __init__(
        self,
        workflow_definition: GraphOutput,
        websocket_callback: Callable[[str, Dict, str], Awaitable[None]],
        session_id: str,
    ):
        if not workflow_definition or not workflow_definition.nodes:
            raise ValueError("Workflow definition is empty or has no nodes.")

        self.workflow_def = workflow_definition
        self.websocket_callback = websocket_callback
        self.session_id = session_id
        self.resume_queue: asyncio.Queue = asyncio.Queue()
        logger.info(f"[{self.session_id}] WorkflowExecutor __init__: Resume queue created: {id(self.resume_queue)}")


        builder = StateGraph(SimplifiedWorkflowState)

        for node_def in self.workflow_def.nodes:
            if node_def.effective_id.upper() not in ["START_NODE", "END_NODE"]:
                node_runner_instance = self._simplified_node_runner_wrapper(node_def)
                builder.add_node(node_def.effective_id, node_runner_instance)

        for edge_def in self.workflow_def.edges:
            if edge_def.from_node.upper() == "START_NODE":
                builder.add_edge(START, edge_def.to_node)
            elif edge_def.to_node.upper() == "END_NODE":
                builder.add_edge(edge_def.from_node, END)
            else:
                builder.add_edge(edge_def.from_node, edge_def.to_node)

        self.compiled_graph = builder.compile(checkpointer=MemorySaver(), debug=True)
        logger.info(f"[{self.session_id}] Simplified WorkflowExecutor: Graph compiled.")
        self.initial_execution_state = SimplifiedWorkflowState()

    def _simplified_node_runner_wrapper(self, node_def: Node) -> Callable[[SimplifiedWorkflowState], Awaitable[SimplifiedWorkflowState]]:
        async def _run_node_instance(current_state: SimplifiedWorkflowState) -> SimplifiedWorkflowState:
            node_id = node_def.effective_id
            logger.info(f"[{self.session_id}] Simplified Node: '{node_id}' (OpID: {node_def.operationId}) execution START.")
            
            new_node_visits = current_state.node_visits + [node_id]
            next_state_dict = {
                "node_visits": new_node_visits,
                "last_visited_node": node_id,
                "error_message": current_state.error_message, # Preserve error
                "waiting_for_confirmation_on_node": None, # Clear previous wait
                "last_resume_payload_received": None
            }

            # Simulate a node that might require confirmation (e.g., if its name contains "confirm")
            # In your real Node model, you have `node_def.requires_confirmation`
            if node_def.requires_confirmation: # Using the actual field from your Node model
                logger.info(f"[{self.session_id}] Node '{node_id}' requires confirmation.")
                next_state_dict["waiting_for_confirmation_on_node"] = node_id
                if self.websocket_callback:
                    await self.websocket_callback(
                        "interrupt_confirmation_required",
                        {"node_id": node_id, "message": f"Node '{node_id}' requires your confirmation to proceed."},
                        self.session_id
                    )
                
                confirmed_payload = None
                try:
                    logger.info(f"[{self.session_id}] Node '{node_id}' awaiting on resume_queue: {id(self.resume_queue)}...")
                    # Set a shorter timeout for testing if needed
                    confirmed_payload = await asyncio.wait_for(self.resume_queue.get(), timeout=600) 
                    logger.info(f"[{self.session_id}] Node '{node_id}' received from queue: {str(confirmed_payload)[:100]}")
                    next_state_dict["last_resume_payload_received"] = confirmed_payload
                    next_state_dict["waiting_for_confirmation_on_node"] = None # Confirmation received
                    if self.websocket_callback:
                         await self.websocket_callback("node_resumed_with_payload", {"node_id": node_id, "payload_received": str(confirmed_payload)[:100]}, self.session_id)
                except asyncio.TimeoutError:
                    error_msg = f"Node '{node_id}' confirmation timed out."
                    logger.error(f"[{self.session_id}] {error_msg}")
                    next_state_dict["error_message"] = error_msg
                    next_state_dict["waiting_for_confirmation_on_node"] = None # Timeout, no longer waiting
                    if self.websocket_callback:
                        await self.websocket_callback("node_execution_failed", {"node_id": node_id, "error": error_msg}, self.session_id)
                    return SimplifiedWorkflowState(**next_state_dict) # End execution for this path
                except asyncio.CancelledError:
                    logger.warning(f"[{self.session_id}] Node '{node_id}' confirmation wait cancelled.")
                    next_state_dict["error_message"] = "Confirmation cancelled"
                    next_state_dict["waiting_for_confirmation_on_node"] = None
                    raise # Re-raise to be handled by streaming loop
                except Exception as e_q:
                    error_msg = f"Error during confirmation wait for {node_id}: {str(e_q)}"
                    logger.error(f"[{self.session_id}] {error_msg}", exc_info=True)
                    next_state_dict["error_message"] = error_msg
                    next_state_dict["waiting_for_confirmation_on_node"] = None
                    if self.websocket_callback:
                        await self.websocket_callback("node_execution_failed", {"node_id": node_id, "error": error_msg}, self.session_id)
                    return SimplifiedWorkflowState(**next_state_dict)
                finally:
                    if confirmed_payload is not None: # Ensure get() was successful
                        self.resume_queue.task_done()
            
            await asyncio.sleep(0.05) # Simulate other work

            if self.websocket_callback:
                await self.websocket_callback(
                    "node_execution_succeeded",
                    {"node_id": node_id, "message": f"Successfully processed {node_id}"},
                    self.session_id
                )
            logger.info(f"[{self.session_id}] Simplified Node: '{node_id}' execution END.")
            return SimplifiedWorkflowState(**next_state_dict)
        return _run_node_instance

    async def run_workflow_streaming(self, thread_config: Dict[str, Any]):
        logger.info(f"[{self.session_id}] Starting SIMPLIFIED workflow streaming. Config: {thread_config}")
        if self.websocket_callback:
            await self.websocket_callback("workflow_execution_started", {"session_id": self.session_id, "message": "Simplified workflow started."}, self.session_id)

        final_state_to_return = self.initial_execution_state
        try:
            async for event in self.compiled_graph.astream_events(self.initial_execution_state, config=thread_config, version="v1"):
                event_type = event["event"]
                event_name = event.get("name", "unknown_event_source")
                event_data = event.get("data", {})
                logger.debug(f"[{self.session_id}] Simplified WF Stream Event: Type='{event_type}', Name='{event_name}'")

                if event_type == "on_chain_end":
                    output_state_candidate = event_data.get("output")
                    if isinstance(output_state_candidate, SimplifiedWorkflowState):
                        final_state_to_return = output_state_candidate
                        logger.info(f"[{self.session_id}] Node '{event_name}' finished. State: last_visited='{final_state_to_return.last_visited_node}', waiting='{final_state_to_return.waiting_for_confirmation_on_node}'")

                elif event_type == "on_graph_end":
                    final_output_data = event_data.get("output")
                    if isinstance(final_output_data, SimplifiedWorkflowState):
                        final_state_to_return = final_output_data
                    logger.info(f"[{self.session_id}] Simplified workflow finished (on_graph_end). Final state: {final_state_to_return.dict() if final_state_to_return else 'N/A'}")
                    if self.websocket_callback:
                        await self.websocket_callback("workflow_execution_completed", {"final_state": final_state_to_return.dict() if final_state_to_return else {}}, self.session_id)
                    return

                elif event_type in ["on_tool_error", "on_chain_error", "on_node_error"]:
                    # ... (error handling as before) ...
                    return
        
        except asyncio.CancelledError:
            logger.warning(f"[{self.session_id}] Simplified workflow streaming task CANCELLED.")
            if self.websocket_callback:
                await self.websocket_callback("workflow_execution_failed", {"error": "Simplified workflow CANCELLED."}, self.session_id)
            return
        except Exception as e_stream:
            # ... (exception handling as before) ...
            return

        logger.info(f"[{self.session_id}] Simplified workflow streaming loop completed (fallback).")
        if self.websocket_callback:
            await self.websocket_callback("workflow_execution_completed", {"message": "Simplified streaming ended (fallback).", "final_state": final_state_to_return.dict() if final_state_to_return else {}}, self.session_id)

    async def submit_interrupt_value(self, value: Dict[str, Any]):
        logger.info(f"[{self.session_id}] WorkflowExecutor submit_interrupt_value: Received value {str(value)[:100]} for queue {id(self.resume_queue)}")
        try:
            json.dumps(value) # Test serializability
            await self.resume_queue.put(value)
            logger.info(f"[{self.session_id}] Value successfully put onto resume_queue. Queue size approx: {self.resume_queue.qsize()}")
        except TypeError as e:
            logger.error(f"[{self.session_id}] Failed to submit non-JSON-serializable interrupt value to queue: {e}. Value: {str(value)[:200]}")
            if self.websocket_callback:
                 await self.websocket_callback("workflow_error", {"error": "Submitted data for resume was not valid (not JSON serializable)."}, self.session_id)
        except Exception as e_put:
            logger.error(f"[{self.session_id}] Error putting value onto resume_queue: {e_put}", exc_info=True)

