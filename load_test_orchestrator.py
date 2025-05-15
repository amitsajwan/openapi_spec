# load_test_orchestrator.py
import asyncio
import logging
import time
import uuid
from typing import Any, Callable, Awaitable, Dict, Optional

from fastapi import WebSocket # For type hinting if WebSocket object is passed directly

# Assuming these are importable from their respective locations
# Ensure PlanSchema is correctly aliased or imported if it's GraphOutput
from models import GraphOutput as PlanSchema, ExecutionGraphState, APIExecutor
from execution_graph_definition import ExecutionGraphDefinition
from execution_manager import GraphExecutionManager

logger = logging.getLogger(__name__)

async def execute_load_test(
    websocket: WebSocket,
    main_session_id: str,
    session_store: Dict[str, Any],
    api_executor_instance: APIExecutor,
    graph2_instance_callback: Callable[[str, Dict[str, Any], Optional[str]], Awaitable[None]],
    send_overall_status_callback: Callable[[WebSocket, str, Dict[str, Any], str, str, Optional[str]], Awaitable[None]]
):
    """
    Orchestrates the execution of a load test by running multiple concurrent
    Graph 2 instances.

    Args:
        websocket: The WebSocket connection for sending overall status.
        main_session_id: The G1 session ID for context.
        session_store: The session store containing the execution plan and load test config.
        api_executor_instance: Instance of APIExecutor.
        graph2_instance_callback: Callback for messages from individual G2 worker instances.
        send_overall_status_callback: Callback to send overall load test status messages.
                                      (Expected signature: websocket, type, content, main_session_id, source, g2_thread_id)
    """
    load_test_config = session_store.get("scratchpad", {}).get("load_test_config")
    if not load_test_config:
        logger.error(f"[{main_session_id}] Load test config not found in scratchpad. Cannot start load test.")
        await send_overall_status_callback(
            websocket, "error", {"error": "Load test configuration missing."},
            main_session_id, "system_error", None
        )
        return

    num_concurrent_users = load_test_config.get("num_concurrent_users", 1)
    # duration_minutes = load_test_config.get("duration_minutes") # Placeholder for future implementation
    disable_prompts = load_test_config.get("disable_confirmation_prompts", True)

    exec_plan_dict = session_store.get("execution_graph")
    if not isinstance(exec_plan_dict, dict):
        logger.error(f"[{main_session_id}] No valid execution plan dictionary found for load test.")
        await send_overall_status_callback(
            websocket, "error", {"error": "Execution plan missing or invalid for load test."},
            main_session_id, "system_error", None
        )
        return

    try:
        exec_plan_model = PlanSchema.model_validate(exec_plan_dict)
    except Exception as e_val:
        logger.error(f"[{main_session_id}] Failed to validate execution_graph for load test: {e_val}")
        await send_overall_status_callback(
            websocket, "error", {"error": f"Execution plan invalid: {str(e_val)}"},
            main_session_id, "system_error", None
        )
        return

    await send_overall_status_callback(
        websocket, "info",
        {"message": f"Starting load test with {num_concurrent_users} concurrent virtual users. Confirmations disabled: {disable_prompts}."},
        main_session_id, "system", None
    )

    start_time = time.time()
    test_tasks = []

    async def run_single_virtual_user_flow(user_idx: int):
        g2_thread_id = f"loadtest_{main_session_id}_u{user_idx}_{uuid.uuid4().hex[:6]}"
        logger.info(f"[{main_session_id}] Load Test: Starting virtual user {user_idx} (G2 Thread ID: {g2_thread_id})")

        try:
            exec_graph_def = ExecutionGraphDefinition(
                graph_execution_plan=exec_plan_model,
                api_executor=api_executor_instance,
                disable_confirmation_prompts=disable_prompts
            )
            runnable_exec_graph = exec_graph_def.get_runnable_graph()
            exec_manager = GraphExecutionManager(
                runnable_graph=runnable_exec_graph,
                graph_definition=exec_graph_def,
                websocket_callback=graph2_instance_callback,
                planning_checkpointer=None,
                main_planning_session_id=main_session_id
            )
            initial_exec_state_values = ExecutionGraphState(
                initial_input=session_store.get("workflow_extracted_data", {}),
            ).model_dump(exclude_none=True)
            exec_graph_config = {"configurable": {"thread_id": g2_thread_id}}

            await exec_manager.execute_workflow(initial_exec_state_values, exec_graph_config)
            logger.info(f"[{main_session_id}] Load Test: Virtual user {user_idx} (G2 Thread ID: {g2_thread_id}) completed flow.")
            # Optionally send a per-user completion message (can be noisy)
            # await send_overall_status_callback(websocket, "info", {"message": f"Load test user {user_idx} finished flow."}, main_session_id, "load_test_runner", g2_thread_id)
            return {"status": "success", "user_idx": user_idx, "g2_thread_id": g2_thread_id}
        except Exception as e_worker:
            logger.error(f"[{main_session_id}] Load Test: Error in virtual user {user_idx} (G2 Thread ID: {g2_thread_id}): {e_worker}", exc_info=True)
            await send_overall_status_callback(
                websocket, "error",
                {"error": f"Load test virtual user {user_idx} encountered an error: {str(e_worker)[:100]}", "g2_thread_id_failed": g2_thread_id},
                main_session_id, "load_test_runner_error", g2_thread_id # Pass g2_thread_id here
            )
            return {"status": "failure", "user_idx": user_idx, "g2_thread_id": g2_thread_id, "error": str(e_worker)}

    for i in range(num_concurrent_users):
        test_tasks.append(run_single_virtual_user_flow(i + 1))

    if test_tasks:
        results = await asyncio.gather(*test_tasks, return_exceptions=False) # Errors are handled within run_single_virtual_user_flow
        successful_runs = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
        failed_runs = len(results) - successful_runs
        logger.info(f"[{main_session_id}] Load test: All virtual user tasks launched. Successful iterations: {successful_runs}, Failed iterations: {failed_runs}")

    end_time = time.time()
    total_duration_secs = end_time - start_time
    logger.info(f"[{main_session_id}] Load test with {num_concurrent_users} virtual users completed in {total_duration_secs:.2f} seconds.")
    await send_overall_status_callback(
        websocket, "info",
        {"message": f"Load test finished. {num_concurrent_users} virtual user flows initiated. Total processing duration: {total_duration_secs:.2f}s. Successful iterations: {successful_runs}, Failed: {failed_runs}."},
        main_session_id, "system", None # Final overall message
    )
    # The calling function in websocket_helpers will reset the session_store's workflow_execution_status
