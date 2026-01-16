"""
Defines a generic infrastructure to manage fire-and-forget background tasks.

This module provides a mechanism to schedule asynchronous tasks that should run
in the background without blocking the main execution flow. These tasks are
tracked and automatically awaiting during application shutdown.

Example:
    ```python
    import asyncio
    from lmm_education.background_task_manager import schedule_task

    async def background_job(data: str) -> None:
        await asyncio.sleep(1)
        print(f"Processed: {data}")

    # Schedule the coroutine execution
    schedule_task(background_job("some data"))
    ```
"""

import asyncio
import atexit
from collections.abc import Coroutine
from typing import Any

# Module-level collection of task objects.
active_tasks: set[asyncio.Task[None]] = set()


def _shutdown_sync() -> None:
    """Synchronous atexit handler for cleanup."""
    if not active_tasks:
        return

    try:
        asyncio.get_running_loop()
        print("Warning: Event loop still running at shutdown")
    except RuntimeError:
        # No running loop - use asyncio.run()
        asyncio.run(_shutdown())


async def _shutdown() -> None:
    """Await all pending tasks."""
    if active_tasks:
        print(f"Awaiting {len(active_tasks)} pending background tasks...")
        # Create snapshot to avoid set modification during iteration
        tasks: list[asyncio.Task[None]] = list(active_tasks)
        results: list[BaseException | None] = await asyncio.gather(
            *tasks, return_exceptions=True
        )

        completed = sum(1 for r in results if not isinstance(r, Exception))
        cancelled = sum(
            1 for r in results if isinstance(r, asyncio.CancelledError)
        )
        failed = sum(
            1
            for r in results
            if isinstance(r, Exception)
            and not isinstance(r, asyncio.CancelledError)
        )

        if cancelled or failed:
            print(
                f"Background Tasks Results: {completed} completed, "
                f"{cancelled} cancelled, {failed} failed"
            )
        for result in results:
            if isinstance(result, Exception):
                print(f"Background task failed: {result}")


# Register cleanup handler at module load
atexit.register(_shutdown_sync)


def schedule_task(coro: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
    """
    Schedules a fire-and-forget background task.

    The task is added to a module-level set to be awaited at application exit.

    Args:
        coro: The coroutine to schedule.

    Returns:
        The scheduled asyncio Task.
    """
    task: asyncio.Task[None] = asyncio.create_task(coro)
    active_tasks.add(task)
    task.add_done_callback(active_tasks.discard)
    return task
