import asyncio
import logging
from typing import Any, Callable, Awaitable, Dict, Optional
from collections import defaultdict

from langgraph.graph import CompiledGraph
from langgraph.types import Interrupt # To catch Interrupts raised by nodes

# Assuming models.py (containing ExecutionGraphState) is accessible
# This ExecutionGraphState is the runtime state for Graph 2
from models import ExecutionGraphState 

logger = logging.getLogger(__name__)

class GraphExecutionManager:
    """
    Manages the runtime execution of a compiled LangGraph (Graph 2 - Execution Graph).
    Handles streaming of graph execution, catches interrupts for human-in-the-loop steps,
    communicates with an external system (e.g., UI via WebSockets), and manages resumption.
    """
    def __init__(
        self,
        runnable_graph: CompiledGraph,
        # websocket_callback signature: (event_type: str, data: Dict, thread_id: Optional[str])
        # This callback is provided by main.py to send messages back to the client.
        websocket_callback: Callable[[str, Dict[str, Any], Optional[str]], Awaitable[None]]
    ):
        self.runnable_graph: CompiledGraph = runnable_graph
        self.websocket_callback: Callable[[str, Dict[str, Any], Optional[str]], Awaitable[None]] = websocket_callback
        # One resume queue per main_session_id (Graph 1's thread_id) to handle 
        # resume data for the corresponding Graph 2 instance.
        self.resume_queues: Dict[str, asyncio.Queue[Any]] = defaultdict(asyncio.Queue)
        logger.info("GraphExecutionManager initialized.")

    async def submit_resume_data(self, main_session_id: str, resume_data: Any) -> bool:
        """
        Called by an external system (e.g., main.py's WebSocket handler) to provide
        data and resume a paused Graph 2 execution.
        `main_session_id` is the session ID from Graph 1, used to key the resume queue.
        `resume_data` typically includes information to identify the confirmation
        (e.g., a 'confirmation_key' from the Interrupt) and the user's decision/input.
        """
        if main_session_id in self.resume_queues:
            try:
                await self.resume_queues[main_session_id].put(resume_data)
                logger.info(f"Resume data submitted and queued for main_session_id: {main_session_id} (for its Graph 2). Data: {str(resume_data)[:100]}...")
                return True
            except Exception as e:
                logger.error(f"Error putting resume data onto queue for main_session_id {main_session_id}: {e}")
                return False
        else:
            logger.warning(f"No active resume queue found for main_session_id: {main_session_id}. Cannot submit resume data for its Graph 2.")
            return False

    async def execute_workflow(
        self,
        initial_graph_values: Dict[str, Any], # Initial values for ExecutionGraphState fields
        config: Dict[str, Any], # Must include {"configurable": {"thread_id": "..."}} for Graph 2's LangGraph instance
        main_session_id: str # The session_id from Graph 1, used for queue lookup
    ) -> Optional[Dict[str, Any]]: # Returns final state values as a dictionary, or None on critical error
        
        graph2_thread_id = config.get("configurable", {}).get("thread_id")
        if not graph2_thread_id:
            await self.websocket_callback(
                "execution_error", 
                {"error": "ConfigurationError: thread_id is missing in config for ExecutionGraph (Graph 2)."}, 
                main_session_id 
            )
            raise ValueError("thread_id is missing in config['configurable']['thread_id'] for ExecutionGraph (Graph 2).")
        
        final_state_dict: Optional[Dict[str, Any]] = None
        logger.info(f"--- [Graph 2] Starting Workflow Execution for its ThreadID: {graph2_thread_id} (Main Session: {main_session_id}) ---")
        
        current_input_for_astream: Optional[Dict[str, Any]] = initial_graph_values

        while True: 
            try:
                logger.debug(f"Graph 2 ThreadID '{graph2_thread_id}': Entering astream loop. Input for astream: {str(current_input_for_astream)[:200]}...")
                async for event in self.runnable_graph.astream(current_input_for_astream, config=config):
                    for node_name, node_output_state_update in event.items():
                        if node_name == "__end__": 
                            logger.info(f"Graph 2 ThreadID '{graph2_thread_id}' | Graph execution reached END state.")
                            break 
                        logger.info(f"Graph 2 ThreadID '{graph2_thread_id}' | Node '{node_name}' executed. State update keys: {list(node_output_state_update.keys()) if isinstance(node_output_state_update, dict) else type(node_output_state_update)}")
                    else: 
                        continue 
                    break 
                
                logger.info(f"Graph 2 ThreadID '{graph2_thread_id}': astream loop completed normally (graph finished or END reached).")
                break 

            except Interrupt as e_interrupt:
                interrupt_data_from_node = e_interrupt.args[0] if e_interrupt.args else {}
                interrupted_node_name = interrupt_data_from_node.get("operationId", "UnknownNode") 
                
                logger.info(f"--- [Graph 2] INTERRUPT at node '{interrupted_node_name}' for its ThreadID '{graph2_thread_id}' ---")
                await self.websocket_callback(
                    "human_intervention_required", 
                    {"node_name": interrupted_node_name, "details_for_ui": interrupt_data_from_node},
                    graph2_thread_id 
                )
                logger.info(f"Graph 2 Workflow paused for its ThreadID '{graph2_thread_id}', awaiting resume data via queue for main_session_id '{main_session_id}'.")
                
                try:
                    resume_data = await asyncio.wait_for(self.resume_queues[main_session_id].get(), timeout=600.0) 
                    logger.info(f"Graph 2 Workflow resumed for its ThreadID '{graph2_thread_id}' with data for '{interrupted_node_name}'. Data: {str(resume_data)[:100]}")

                    update_for_state = {"confirmed_data": {}} 
                    if isinstance(resume_data, dict) and "confirmation_key" in resume_data:
                        confirmation_key = resume_data["confirmation_key"]
                        update_for_state["confirmed_data"][confirmation_key] = resume_data.get("decision", True)
                        update_for_state["confirmed_data"][f"{confirmation_key}_details"] = resume_data 
                    else:
                        logger.warning(f"Graph 2 ThreadID '{graph2_thread_id}': Resume data did not follow expected 'confirmation_key' structure. Storing generically.")
                        update_for_state["confirmed_data"]["general_resume_payload"] = resume_data
                    
                    await self.runnable_graph.update_state(config, update_for_state) 
                    current_input_for_astream = None 
                    logger.info(f"State updated for Graph 2 ThreadID '{graph2_thread_id}'. Continuing stream from checkpoint...")
                    continue 

                except asyncio.TimeoutError:
                    error_msg = f"Graph 2 Workflow timed out for its ThreadID '{graph2_thread_id}' waiting for resume data for '{interrupted_node_name}'."
                    logger.error(error_msg)
                    await self.websocket_callback("workflow_timeout", {"message": error_msg, "node_name": interrupted_node_name}, graph2_thread_id)
                    await self.runnable_graph.update_state(config, {"error": "ConfirmationTimeout"}) 
                    error_state_snapshot = await self.runnable_graph.get_state(config)
                    return error_state_snapshot.values if error_state_snapshot else {"error": error_msg}
            
            except Exception as e_stream:
                error_message = f"Unexpected error during ExecutionGraph (Graph 2) stream for its ThreadID '{graph2_thread_id}': {type(e_stream).__name__} - {str(e_stream)}"
                logger.critical(error_message, exc_info=True)
                await self.websocket_callback("execution_error", {"error": error_message}, graph2_thread_id)
                try: 
                    await self.runnable_graph.update_state(config, {"error": error_message})
                except Exception as state_update_error:
                    logger.error(f"Critical: Failed to update Graph 2 state with error after stream exception: {state_update_error}")
                
                error_state_snapshot = await self.runnable_graph.get_state(config) 
                return error_state_snapshot.values if error_state_snapshot else {"error": error_message}
            
        try:
            final_state_snapshot = await self.runnable_graph.get_state(config) 
            if final_state_snapshot:
                final_state_dict = final_state_snapshot.values
                if final_state_dict.get("error"):
                     await self.websocket_callback("execution_failed", {"final_state": final_state_dict}, graph2_thread_id)
                     logger.error(f"--- [Graph 2] Workflow FAILED for its ThreadID: {graph2_thread_id}. Error in final state: {final_state_dict['error']} ---")
                else:
                    await self.websocket_callback("execution_completed", {"final_state": final_state_dict}, graph2_thread_id)
                    logger.info(f"--- [Graph 2] Workflow COMPLETED SUCCESSFULLY for its ThreadID: {graph2_thread_id} ---")
            else:
                await self.websocket_callback("execution_warning", {"message": "Execution finished, but no final state snapshot was retrieved for Graph 2."}, graph2_thread_id)
                logger.warning(f"--- [Graph 2] Workflow FINISHED (No Final State Snapshot) for its ThreadID: {graph2_thread_id} ---")
        except Exception as e_get_final_state:
            error_message = f"Error retrieving final state for Graph 2 ThreadID '{graph2_thread_id}': {type(e_get_final_state).__name__} - {str(e_get_final_state)}"
            logger.error(error_message, exc_info=True)
            await self.websocket_callback("execution_error", {"error": error_message, "detail": "Failed to get final state for Graph 2."}, graph2_thread_id)
            final_state_dict = {"error": error_message} 
            
        return final_state_dict
