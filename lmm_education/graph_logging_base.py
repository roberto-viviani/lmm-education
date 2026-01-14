"""
Defines the infrastructure to implement logging to a database a
LangGraph stream based on its state and context.

This module provides a factory function that creates fire-and-forget
async logging functions, avoiding blocking streaming when logging
interactions to a database. This function wraps a logging co-routine
that is provided as an argument to the factory, thus transforming it
into a non-blocking function immediately returning to the caller. The
actual logging is coded in this co-routine.

How to use: first, define a co-routine that handles the logging
to a stream or a list of streams (for multiple tables), a state, and
a context (you will have defined a graph state - from TypedDict - and
a context object - from BaseModel - when defining the graph itself):

```python
async def logging(
    streams: TextIOBase,
    state: GraphState,
    context: Context,
    interaction_type: str,
    timestamp: datetime,
    record_id: str) -> None:
    ... implementation goes here
```

Then create a logger using the factory function:

```python
with open("logger.csv", 'a', encoding='utf-8') as f:
    log = create_graph_logger(f, context, logging)
    record_id: str = log(state, "MESSAGE")
```

Note: The caller is responsible for keeping the stream open while
logging tasks are pending. Pending tasks are automatically awaited
at application exit via an atexit handler.

"""

from typing import TypeVar, Any
from collections.abc import Mapping, Coroutine, Callable
from datetime import datetime
from io import TextIOBase
import asyncio
import atexit

from pydantic import BaseModel

from lmm.utils.hash import generate_random_string

# pyright: reportMissingTypeStubs=false

# TypedDict instances are structurally Mappings/dicts.
# We bind to Mapping[str, Any] to accommodate any TypedDict.
StateType = TypeVar("StateType", bound=Mapping[str, Any])

# Context must be or derive from Pydantic's BaseModel
ContextType = TypeVar("ContextType", bound=BaseModel)

# Type Alias for the coroutine for better readability
LogCoroutine = Callable[
    [
        TextIOBase | list[TextIOBase],
        StateType,
        ContextType,
        datetime,
        str,
    ],
    Coroutine[Any, Any, None],
]

# Type alias for the function returned by

# Module-level collection of task objects.
active_logs: set[asyncio.Task[None]] = set()


def _shutdown_sync() -> None:
    """Synchronous atexit handler for cleanup."""
    if not active_logs:
        return

    try:
        asyncio.get_running_loop()
        print("Warning: Event loop still running at shutdown")
    except RuntimeError:
        # No running loop - use asyncio.run()
        asyncio.run(_shutdown())


async def _shutdown() -> None:
    """Await all pending log tasks."""
    if active_logs:
        print(f"Awaiting {len(active_logs)} pending logging tasks...")
        # Create snapshot to avoid set modification during iteration
        tasks: list[asyncio.Task[None]] = list(active_logs)
        results: list[BaseException | None] = await asyncio.gather(
            *tasks, return_exceptions=True
        )

        # The only exceptions raised by code are those raised by
        # logging itself, because the coroutines are written to
        # handle all exceptions and write them to the log. But we
        # do not know if Gradio is canceling anything. Diagnostics
        completed = sum(
            1 for r in results if not isinstance(r, Exception)
        )
        cancelled = sum(
            1
            for r in results
            if isinstance(r, asyncio.CancelledError)
        )
        failed = sum(
            1
            for r in results
            if isinstance(r, Exception)
            and not isinstance(r, asyncio.CancelledError)
        )

        # Don't use logging since it may be failing
        if cancelled or failed:
            print(
                f"Results: {completed} completed, {cancelled} "
                f"cancelled, {failed} failed"
            )
        for result in results:
            if isinstance(result, Exception):
                print(f"Logging failed: {result}")


# Register cleanup handler at module load
atexit.register(_shutdown_sync)


def create_graph_logger(
    streams: TextIOBase | list[TextIOBase],
    context: ContextType,
    log_coro: LogCoroutine[StateType, ContextType],
) -> Callable[[StateType, datetime | None, str | None], str]:
    """
    Factory that returns a fire-and-forget log function.

    The returned function creates async tasks for logging operations,
    adding them to the module-level active_logs set for tracking.
    All pending tasks are automatically awaited at application exit.

    Args:
        stream: The text streams to write logs to. The caller is
            responsible for keeping these streams open while logging
            tasks are pending (one stream per table)
        context: a dependency injection object used by log_coro
        log_coro: The async coroutine that performs the actual
            logging.

    Returns:
        A logging function with signature:
        (state, datetime | None, str | None) -> str
        The logging function returns the record_id used for the
        log entry.

    Example:
        ```python
        with open("logger.csv", 'a', encoding='utf-8') as f:
            log = create_graph_logger(f, context, my_logging_coroutine)
            record_id = log(state)
        ```

    Note:
        An asyncio loop must be running to use the returned function.
    """

    def log(
        state: StateType,
        timestamp: datetime | None = None,
        record_id: str | None = None,
    ) -> str:
        """
        Log an interaction to the database asynchronously.

        Args:
            state: The state (TypedDict) to log
            context: The dependency injection object (Pydantic Model)
            interaction_type: The interaction type (default: "MESSAGE")
            timestamp: Timestamp of the interaction. Defaults to
                current time if None.
            record_id: Unique identifier for this record. Generated
                if not provided.

        Returns:
            The record_id used for the record.
        """
        if timestamp is None:
            timestamp = datetime.now()

        if not record_id:
            record_id = generate_random_string(8)

        task: asyncio.Task[None] = asyncio.create_task(
            log_coro(
                streams,
                state,
                context,
                timestamp,
                record_id,
            )
        )
        active_logs.add(task)
        task.add_done_callback(lambda t: active_logs.discard(t))

        return record_id

    return log


def create_null_logger() -> (
    Callable[[Any, datetime | None, str | None], str]
):
    """
    Factory that returns a no-op logger function.

    Useful for testing or when logging is disabled.

    Returns:
        A function that accepts the same arguments as a regular
        logger but does nothing and returns an empty string.

    Example:
        ```python
        log = create_null_logger()
        log(state, context, "MESSAGE")  # Does nothing
        ```
    """

    def null_log(
        state: Any,
        timestamp: datetime | None = None,
        record_id: str | None = None,
    ) -> str:
        return ""

    return null_log
