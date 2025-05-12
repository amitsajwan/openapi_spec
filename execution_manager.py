import asyncio
import logging
from typing import Any, Callable, Awaitable, Dict, Optional
from collections import defaultdict

# from langgraph.graph import CompiledGraph # Removed CompiledGraph import
from langgraph.types import Interrupt # To catch Interrupts raised by nodes
from langgraph.checkpoint.base import BaseCheckpointSaver # For type hinting

# Assuming models.py (containing ExecutionGraphState) is accessible
# This ExecutionGraphState is the runtime state for Graph 2
from models import BotState, ExecutionGraphState # BotState for type hinting if needed for checkpointer logic

logger = logging.getLogger(__name__)

class GraphExecutionManager:
    """
    Manages the runtime execution of a compiled LangGraph (Graph 2 - Execution Graph).
    Handles streaming of graph execution, catches interrupts for human-in-the-loop steps,
    and manages resumption.
    The provided websocket_callback is now responsible for updating Graph 1's BotState
    upon Graph 2 completion or failure, using the planning_checkpointer.
    """
    def __init__(
        self,
        runnable_graph: Any, # MODIFIED: Type hint to Any
        # websocket_callback signature: (event_type: str, data: Dict, graph2_thread_id: Optional[str])
        # This callback is provided by main.py. It sends messages back to the client
        # AND is now responsible for updating Graph 1's BotState via planning_checkpointer.
        websocket_callback: Callable[[str, Dict[str, Any], Optional[str]], Awaitable[None]],
        planning_checkpointer: BaseCheckpointSaver, # Checkpointer for Graph 1 (BotState)
        main_planning_session_id: str # Session ID for Graph 1, used for context
    ):
        self.runnable_graph: Any = runnable_graph # MODIFIED: Type hint to Any
        self.websocket_callback = websocket_callback
        self.planning_checkpointer = planning_checkpointer # Stored for potential future use if manager needs to update Graph 1 state directly
        self.main_planning_session_id = main_planning_session_id # Stored for context, logging
        
        # Resume queues are keyed by the main_planning_session_id, as one Graph 1 session
        # might spawn one Graph 2 execution at a time that needs resumption.
        self.resume_queues: Dict[str, asyncio.Queue[Any]] = defaultdict(asyncio.Queue)
        logger.info(f"GraphExecutionManager initialized for Main Planning Session ID: {self.main_planning_session_id}")

    async def submit_resume_data(self, main_session_id: str, resume_data: Any) -> bool:
        """
        Called by an external system (e.g., main.py's WebSocket handler) to provide
        data and resume a paused Graph 2 execution.
        `main_session_id` is the session ID from Graph 1 (same as self.main_planning_session_id
        if this manager instance is correctly mapped).
        `resume_data` typically includes information to identify the confirmation
        (e.g., a 'confirmation_key' from the Interrupt) and the user's decision/input.
        """
        # Ensure the queue exists for this main_session_id before trying to put data
        if main_session_id not in self.resume_queues:
             self.resume_queues[main_session_id] = asyncio.Queue() # Initialize if not present
             logger.info(f"Initialized resume queue for Main Session ID: {main_session_id} on first submit.")


        if main_session_id in self.resume_queues: # Check against the provided main_session_id
            try:
                await self.resume_queues[main_session_id].put(resume_data)
                logger.info(f"Resume data submitted and queued for Main Session ID: {main_session_id} (for its Graph 2). Data: {str(resume_data)[:100]}...")
                return True
            except Exception as e:
                logger.error(f"Error putting resume data onto queue for Main Session ID {main_session_id}: {e}")
                return False
        else:
            # This case should be less likely now with the initialization above, but kept for safety.
            logger.warning(f"No active resume queue found for Main Session ID: {main_session_id} despite attempt to initialize. Cannot submit resume data for its Graph 2.")
            return False

    async def execute_workflow(
        self,
        initial_graph_values: Dict[str, Any], # Initial values for ExecutionGraphState fields
        config: Dict[str, Any], # Must include {"configurable": {"thread_id": "..."}} for Graph 2's LangGraph instance
    ) -> Optional[Dict[str, Any]]: # Returns final state values as a dictionary, or None on critical error
        
        graph2_thread_id = config.get("configurable", {}).get("thread_id")
        if not graph2_thread_id:
            # Use self.main_planning_session_id for the callback's session context here
            # The callback needs the Graph 2 thread ID as its third argument for its own context.
            await self.websocket_callback(
                "execution_error", 
                {"error": "ConfigurationError: thread_id is missing in config for ExecutionGraph (Graph 2)."}, 
                graph2_thread_id # Pass graph2_thread_id even if None, callback might handle it
            )
            # This error is critical for Graph 2, so raise it.
            raise ValueError("thread_id is missing in config['configurable']['thread_id'] for ExecutionGraph (Graph 2).")
        
        final_state_dict: Optional[Dict[str, Any]] = None
        logger.info(f"--- [Graph 2] Starting Workflow Execution for its ThreadID: {graph2_thread_id} (Main Session: {self.main_planning_session_id}) ---")
        
        current_input_for_astream: Optional[Dict[str, Any]] = initial_graph_values

        while True: 
            try:
                logger.debug(f"Graph 2 ThreadID '{graph2_thread_id}': Entering astream loop. Input for astream: {str(current_input_for_astream)[:200]}...")
                async for event in self.runnable_graph.astream(current_input_for_astream, config=config):
                    # Log all events for detailed debugging if needed
                    # logger.debug(f"Graph 2 Event (ThreadID: {graph2_thread_id}): {event}")
                    for node_name, node_output_state_update in event.items():
                        if node_name == "__end__": 
                            logger.info(f"Graph 2 ThreadID '{graph2_thread_id}' | Graph execution reached END state.")
                            break  # Break from inner for loop over event items
                        logger.info(f"Graph 2 ThreadID '{graph2_thread_id}' | Node '{node_name}' executed. State update keys: {list(node_output_state_update.keys()) if isinstance(node_output_state_update, dict) else type(node_output_state_update)}")
                    else: # If inner loop didn't break (i.e., no "__end__")
                        continue # Continue astream loop
                    break # Break from astream loop if "__end__" was found
                
                logger.info(f"Graph 2 ThreadID '{graph2_thread_id}': astream loop completed normally (graph finished or END reached).")
                break # Exit the while True loop for streaming

            except Interrupt as e_interrupt:
                interrupt_data_from_node = e_interrupt.args[0] if e_interrupt.args else {}
                interrupted_node_name = interrupt_data_from_node.get("operationId", "UnknownNode") 
                
                logger.info(f"--- [Graph 2] INTERRUPT at node '{interrupted_node_name}' for its ThreadID '{graph2_thread_id}' ---")
                # The callback now uses graph2_thread_id for its third param
                await self.websocket_callback(
                    "human_intervention_required", 
                    {"node_name": interrupted_node_name, "details_for_ui": interrupt_data_from_node},
                    graph2_thread_id # Pass Graph 2's thread_id for this specific message
                )
                logger.info(f"Graph 2 Workflow paused for its ThreadID '{graph2_thread_id}', awaiting resume data via queue for Main Session ID '{self.main_planning_session_id}'.")
                
                try:
                    # Use self.main_planning_session_id to get the correct resume queue
                    # Ensure queue is initialized if this is the first time it's accessed for this session_id
                    if self.main_planning_session_id not in self.resume_queues:
                        self.resume_queues[self.main_planning_session_id] = asyncio.Queue()

                    resume_data = await asyncio.wait_for(self.resume_queues[self.main_planning_session_id].get(), timeout=600.0) # 10 min timeout
                    logger.info(f"Graph 2 Workflow resumed for its ThreadID '{graph2_thread_id}' with data for '{interrupted_node_name}'. Data: {str(resume_data)[:100]}")

                    # Prepare state update for LangGraph based on resume_data
                    update_for_state = {"confirmed_data": {}} 
                    if isinstance(resume_data, dict) and "confirmation_key" in resume_data:
                        confirmation_key = resume_data["confirmation_key"]
                        # Store the decision (True/False) and the full details for the node to process
                        update_for_state["confirmed_data"][confirmation_key] = resume_data.get("decision", True) # Default to True if decision not specified
                        update_for_state["confirmed_data"][f"{confirmation_key}_details"] = resume_data 
                    else:
                        logger.warning(f"Graph 2 ThreadID '{graph2_thread_id}': Resume data did not follow expected 'confirmation_key' structure. Storing generically.")
                        # Store generically if structure is unexpected, node might need to handle this
                        update_for_state["confirmed_data"]["general_resume_payload"] = resume_data
                    
                    # Update Graph 2's state. This tells LangGraph the interrupt is resolved.
                    await self.runnable_graph.update_state(config, update_for_state) 
                    current_input_for_astream = None # LangGraph will continue from checkpoint
                    logger.info(f"State updated for Graph 2 ThreadID '{graph2_thread_id}'. Continuing stream from checkpoint...")
                    continue # Continue the while True loop for streaming

                except asyncio.TimeoutError:
                    error_msg = f"Graph 2 Workflow timed out for its ThreadID '{graph2_thread_id}' waiting for resume data for '{interrupted_node_name}'."
                    logger.error(error_msg)
                    # The callback will also update Graph 1's state
                    await self.websocket_callback(
                        "workflow_timeout", 
                        {"message": error_msg, "node_name": interrupted_node_name},
                        graph2_thread_id
                    )
                    # Update Graph 2's state with the timeout error
                    try:
                        await self.runnable_graph.update_state(config, {"error": "ConfirmationTimeout"})
                    except Exception as e_update_timeout:
                        logger.error(f"Failed to update Graph 2 state with ConfirmationTimeout: {e_update_timeout}")
                    
                    error_state_snapshot = await self.runnable_graph.get_state(config)
                    return error_state_snapshot.values if error_state_snapshot else {"error": error_msg} # End execution
            
            except Exception as e_stream: # Other errors during astream
                error_message = f"Unexpected error during ExecutionGraph (Graph 2) stream for its ThreadID '{graph2_thread_id}': {type(e_stream).__name__} - {str(e_stream)}"
                logger.critical(error_message, exc_info=True)
                # The callback will also update Graph 1's state
                await self.websocket_callback(
                    "execution_error", 
                    {"error": error_message}, 
                    graph2_thread_id
                )
                try: # Attempt to update Graph 2's state with the error
                    await self.runnable_graph.update_state(config, {"error": error_message})
                except Exception as state_update_error:
                    logger.error(f"Critical: Failed to update Graph 2 state with error after stream exception: {state_update_error}")
                
                error_state_snapshot = await self.runnable_graph.get_state(config) 
                return error_state_snapshot.values if error_state_snapshot else {"error": error_message} # End execution
            
        # After the while loop (streaming finished)
        try:
            final_state_snapshot = await self.runnable_graph.get_state(config) 
            if final_state_snapshot:
                final_state_dict = final_state_snapshot.values
                if final_state_dict.get("error"):
                     # Callback handles Graph 1 state update
                     await self.websocket_callback(
                         "execution_failed", 
                         {"final_state": final_state_dict}, 
                         graph2_thread_id
                     )
                     logger.error(f"--- [Graph 2] Workflow FAILED for its ThreadID: {graph2_thread_id}. Error in final state: {final_state_dict['error']} ---")
                else:
                    # Callback handles Graph 1 state update
                    await self.websocket_callback(
                        "execution_completed", 
                        {"final_state": final_state_dict}, 
                        graph2_thread_id
                    )
                    logger.info(f"--- [Graph 2] Workflow COMPLETED SUCCESSFULLY for its ThreadID: {graph2_thread_id} ---")
            else: # Should ideally not happen if graph ran
                # Callback handles Graph 1 state update (as a warning/potential issue)
                await self.websocket_callback(
                    "execution_warning", 
                    {"message": "Execution finished, but no final state snapshot was retrieved for Graph 2."}, 
                    graph2_thread_id
                )
                logger.warning(f"--- [Graph 2] Workflow FINISHED (No Final State Snapshot) for its ThreadID: {graph2_thread_id} ---")
        except Exception as e_get_final_state:
            error_message = f"Error retrieving final state for Graph 2 ThreadID '{graph2_thread_id}': {type(e_get_final_state).__name__} - {str(e_get_final_state)}"
            logger.error(error_message, exc_info=True)
            # Callback handles Graph 1 state update
            await self.websocket_callback(
                "execution_error", 
                {"error": error_message, "detail": "Failed to get final state for Graph 2."}, 
                graph2_thread_id
            )
            final_state_dict = {"error": error_message} # Return an error dict
            
        return final_state_dict
