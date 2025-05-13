import asyncio
import logging
from typing import Any, Callable, Awaitable, Dict, Optional
from collections import defaultdict

# Import Interrupt directly, it's used for raising in execution_graph_definition.
# We will not try to catch it here in the manager's astream loop as it's not a standard exception.
from langgraph.types import Interrupt as LangGraphInterrupt
from langgraph.checkpoint.base import BaseCheckpointSaver

from models import BotState, ExecutionGraphState

logger = logging.getLogger(__name__)

class GraphExecutionManager:
    """
    Manages the runtime execution of a compiled LangGraph (Graph 2 - Execution Graph).
    Handles streaming of graph execution, and manages resumption for human-in-the-loop steps.
    The LangGraph Pregel engine handles the Interrupts internally.
    """
    def __init__(
        self,
        runnable_graph: Any,
        websocket_callback: Callable[[str, Dict[str, Any], Optional[str]], Awaitable[None]],
        planning_checkpointer: Optional[BaseCheckpointSaver], # Made Optional
        main_planning_session_id: str
    ):
        self.runnable_graph: Any = runnable_graph
        self.websocket_callback = websocket_callback
        self.planning_checkpointer = planning_checkpointer
        self.main_planning_session_id = main_planning_session_id
        self.resume_queues: Dict[str, asyncio.Queue[Any]] = defaultdict(asyncio.Queue)
        logger.info(f"GraphExecutionManager initialized for Main Planning Session ID: {self.main_planning_session_id}")

    async def submit_resume_data(self, main_session_id: str, resume_data: Any) -> bool:
        """
        Called by an external system to provide data and resume a paused Graph 2 execution.
        """
        if main_session_id not in self.resume_queues:
             self.resume_queues[main_session_id] = asyncio.Queue()
             logger.info(f"Initialized resume queue for Main Session ID: {main_session_id} on first submit.")

        if main_session_id in self.resume_queues:
            try:
                await self.resume_queues[main_session_id].put(resume_data)
                logger.info(f"Resume data submitted and queued for Main Session ID: {main_session_id}. Data: {str(resume_data)[:100]}...")
                return True
            except Exception as e:
                logger.error(f"Error putting resume data onto queue for Main Session ID {main_session_id}: {e}")
                return False
        else:
            logger.warning(f"No active resume queue found for Main Session ID: {main_session_id}. Cannot submit resume data.")
            return False

    async def execute_workflow(
        self,
        initial_graph_values: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        graph2_thread_id = config.get("configurable", {}).get("thread_id")
        if not graph2_thread_id:
            await self.websocket_callback(
                "execution_error",
                {"error": "ConfigurationError: thread_id is missing in config for ExecutionGraph (Graph 2)."},
                graph2_thread_id
            )
            raise ValueError("thread_id is missing in config['configurable']['thread_id'] for ExecutionGraph (Graph 2).")

        final_state_dict: Optional[Dict[str, Any]] = None
        logger.info(f"--- [Graph 2] Starting Workflow Execution for its ThreadID: {graph2_thread_id} (Main Session: {self.main_planning_session_id}) ---")

        # Log MRO for LangGraphInterrupt for debugging if needed, but we won't catch it here.
        # if hasattr(LangGraphInterrupt, '__mro__'):
        #     logger.debug(f"MRO of LangGraphInterrupt: {LangGraphInterrupt.__mro__}")

        current_input_for_astream: Optional[Dict[str, Any]] = initial_graph_values
        is_resuming_after_interrupt = False
        last_event_was_end = False

        while not last_event_was_end: # Loop until the graph explicitly ends
            try:
                if is_resuming_after_interrupt:
                    logger.debug(f"Graph 2 ThreadID '{graph2_thread_id}': Resuming astream after interrupt. Input is None (LangGraph uses checkpoint).")
                    current_input_for_astream = None # LangGraph continues from checkpoint
                    is_resuming_after_interrupt = False # Reset flag
                else:
                    logger.debug(f"Graph 2 ThreadID '{graph2_thread_id}': Starting/continuing astream. Input: {str(current_input_for_astream)[:200]}...")

                # The astream() method itself will handle pausing on Interrupt internally.
                # It will finish its current stream of events when it pauses or completes.
                async for event in self.runnable_graph.astream(current_input_for_astream, config=config):
                    for node_name, node_output_state_update in event.items():
                        if node_name == "__end__":
                            logger.info(f"Graph 2 ThreadID '{graph2_thread_id}' | Graph execution reached END state.")
                            last_event_was_end = True # Signal to exit the while loop
                            break # Break from inner for loop
                        logger.info(f"Graph 2 ThreadID '{graph2_thread_id}' | Node '{node_name}' executed. Update keys: {list(node_output_state_update.keys()) if isinstance(node_output_state_update, dict) else type(node_output_state_update)}")
                    if last_event_was_end:
                        break # Break from astream event loop if "__end__" was found
                
                # If astream finished and last_event_was_end is False, it means Pregel paused due to an Interrupt
                # (or the graph structure is such that it ended without __end__, which is less common for explicit flows).
                # The human_intervention_required message should have been sent by the node raising Interrupt.
                if not last_event_was_end:
                    logger.info(f"Graph 2 ThreadID '{graph2_thread_id}': astream completed an iteration, but graph not at __end__. Assuming paused for interrupt, attempting to get resume data.")
                    try:
                        if self.main_planning_session_id not in self.resume_queues:
                            self.resume_queues[self.main_planning_session_id] = asyncio.Queue()

                        logger.info(f"Graph 2 ThreadID '{graph2_thread_id}': Waiting for resume data from queue for Main Session ID '{self.main_planning_session_id}'.")
                        resume_data = await asyncio.wait_for(self.resume_queues[self.main_planning_session_id].get(), timeout=600.0) # 10 min timeout
                        logger.info(f"Graph 2 Workflow resumed for its ThreadID '{graph2_thread_id}' with data. Data: {str(resume_data)[:100]}")

                        update_for_state = {"confirmed_data": {}} # Ensure this key exists for merging
                        if isinstance(resume_data, dict) and "confirmation_key" in resume_data:
                            confirmation_key = resume_data["confirmation_key"]
                            # Store the decision (True/False) and the full details for the node to process
                            update_for_state["confirmed_data"][confirmation_key] = resume_data.get("decision", True) # Default to True
                            update_for_state["confirmed_data"][f"{confirmation_key}_details"] = resume_data
                        else:
                            logger.warning(f"Graph 2 ThreadID '{graph2_thread_id}': Resume data did not follow expected 'confirmation_key' structure. Storing generically.")
                            update_for_state["confirmed_data"]["general_resume_payload"] = resume_data
                        
                        await self.runnable_graph.update_state(config, update_for_state)
                        is_resuming_after_interrupt = True # Signal to set current_input_for_astream to None for next iteration
                        logger.info(f"State updated for Graph 2 ThreadID '{graph2_thread_id}'. Will continue stream from checkpoint.")
                        # Loop will continue to call astream

                    except asyncio.TimeoutError:
                        error_msg = f"Graph 2 Workflow timed out for its ThreadID '{graph2_thread_id}' waiting for resume data."
                        logger.error(error_msg)
                        await self.websocket_callback(
                            "workflow_timeout",
                            {"message": error_msg, "node_name": "Unknown (timeout waiting for any node)"},
                            graph2_thread_id
                        )
                        try:
                            await self.runnable_graph.update_state(config, {"error": "ConfirmationTimeout"})
                        except Exception as e_update_timeout:
                            logger.error(f"Failed to update Graph 2 state with ConfirmationTimeout: {e_update_timeout}")
                        
                        error_state_snapshot = await self.runnable_graph.get_state(config)
                        return error_state_snapshot.values if error_state_snapshot else {"error": error_msg} # End execution
                    except Exception as e_resume_proc:
                        error_msg = f"Error processing resume data for Graph 2 ThreadID '{graph2_thread_id}': {type(e_resume_proc).__name__} - {e_resume_proc}"
                        logger.error(error_msg, exc_info=True)
                        await self.websocket_callback("execution_error", {"error": error_msg}, graph2_thread_id)
                        return {"error": error_msg} # End execution

            except Exception as e_stream_loop: # Catch other unexpected errors from astream or the loop logic
                error_message = f"Critical error during ExecutionGraph (Graph 2) stream loop for ThreadID '{graph2_thread_id}': {type(e_stream_loop).__name__} - {str(e_stream_loop)}"
                logger.critical(error_message, exc_info=True)
                await self.websocket_callback(
                    "execution_error",
                    {"error": error_message},
                    graph2_thread_id
                )
                try:
                    await self.runnable_graph.update_state(config, {"error": error_message})
                except Exception as state_update_error:
                    logger.error(f"Critical: Failed to update Graph 2 state with error after stream exception: {state_update_error}")
                error_state_snapshot = await self.runnable_graph.get_state(config)
                return error_state_snapshot.values if error_state_snapshot else {"error": error_message} # End execution
            
        # After the while loop (graph execution finished because last_event_was_end is True)
        try:
            final_state_snapshot = await self.runnable_graph.get_state(config)
            if final_state_snapshot:
                final_state_dict = final_state_snapshot.values
                if final_state_dict.get("error"): # Check if an error was set in the final state
                     await self.websocket_callback(
                         "execution_failed",
                         {"final_state": final_state_dict},
                         graph2_thread_id
                     )
                     logger.error(f"--- [Graph 2] Workflow FAILED for its ThreadID: {graph2_thread_id}. Error in final state: {final_state_dict['error']} ---")
                else:
                    await self.websocket_callback(
                        "execution_completed",
                        {"final_state": final_state_dict},
                        graph2_thread_id
                    )
                    logger.info(f"--- [Graph 2] Workflow COMPLETED SUCCESSFULLY for its ThreadID: {graph2_thread_id} ---")
            else: # Should ideally not happen if graph ran to __end__
                await self.websocket_callback(
                    "execution_warning",
                    {"message": "Execution finished, but no final state snapshot was retrieved for Graph 2."},
                    graph2_thread_id
                )
                logger.warning(f"--- [Graph 2] Workflow FINISHED (No Final State Snapshot) for its ThreadID: {graph2_thread_id} ---")
        except Exception as e_get_final_state:
            error_message = f"Error retrieving final state for Graph 2 ThreadID '{graph2_thread_id}': {type(e_get_final_state).__name__} - {str(e_get_final_state)}"
            logger.error(error_message, exc_info=True)
            await self.websocket_callback(
                "execution_error",
                {"error": error_message, "detail": "Failed to get final state for Graph 2."},
                graph2_thread_id
            )
            final_state_dict = {"error": error_message} # Return an error dict
            
        return final_state_dict
