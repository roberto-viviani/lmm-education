from typing import TypeVar, Generic, Mapping, Any, override
from datetime import datetime
from io import TextIOBase
import asyncio

from pydantic import BaseModel

from lmm.utils.hash import generate_random_string

# pyright: reportMissingTypeStubs=false

# TypedDict instances are structurally Mappings/dicts.
# We bind to Mapping[str, Any] to accommodate any TypedDict.
StateType = TypeVar("StateType", bound=Mapping[str, Any])

# Context must be or derive from Pydantic's BaseModel
ContextType = TypeVar("ContextType", bound=BaseModel)

class DatabaseLoggerBase(Generic[StateType, ContextType]):
    """Definition of the database log interface. This
    class definition returns a functional object that does
    nothing. 
    """

    def log(
        self,
        state: StateType,
        context: ContextType,
        interaction_type: str = "MESSAGE",
        timestamp: datetime | None = None,
        record_id: str = "",
    ) -> str:
        """
        Logging function for CSV files.

        When the interaction_type is "MESSAGE", also logs the
        context in the context table.

        Args:
            state: the state (TypedDict) to log
            context: the dependency injection object (Pydantic Model)
            interaction_type: the interaction type
            timestamp: Timestamp of the interaction. Defaults to 
                current time if None.
            record_id: Unique identifier for this record

        Returns:
            the record_id used for the record.
        """
        # handle mutable default argument safety
        if timestamp is None:
            timestamp = datetime.now()

        return ""

    def close(self) -> None:
        """Flushes logs buffer, if required by the implementation,
        and releases resources."""
        pass


# Type Alias for the coroutine for better readability
from collections.abc import Callable, Awaitable
LogCoroutine = Callable[
    [TextIOBase, StateType, ContextType, str, datetime, str], 
    Awaitable[None]
]

# module-level collection of task objects.
active_logs: set[asyncio.Task[None]] = set()
async def _shutdown() -> None:
    if active_logs:
        print(
            f"App exited without loop cleanup? Awaiting "
            f"{len(active_logs)} logging tasks..."
        )
        # Create snapshot to avoid set modification during iteration
        tasks: list[asyncio.Task[None]] = list(active_logs)
        results: list[BaseException | None] = (
            await asyncio.gather(*tasks, return_exceptions=True)
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


class GraphLogger(DatabaseLoggerBase[StateType, ContextType]):
    """Implementation of the DatabaseLogger interface. We wrap here
    an async log function to implement a 'fire and forget' strategy 
    to avoid blocking responses when streaming the graph.

    Note: an asyncio loop must be running to use this object."""

    def __init__(
        self, 
        stream: TextIOBase, 
        log_coro: LogCoroutine[StateType, ContextType]
    ):
        self._stream = stream
        self._log_coro = log_coro

    @override
    def log(
        self,
        state: StateType,
        context: ContextType,
        interaction_type: str = "MESSAGE",
        timestamp: datetime | None = None,
        record_id: str = "",
    ) -> str:
        if timestamp is None:
            timestamp = datetime.now()

        if not record_id:
            record_id = generate_random_string(8)

        logtask: asyncio.Task[None] = asyncio.create_task(
            self._log_coro(
                self._stream,
                state,
                context,
                interaction_type,
                timestamp,
                record_id
            )  
        )
        active_logs.add(logtask)
        logtask.add_done_callback(active_logs.discard)

        return record_id

    @override
    def close(self) -> None:
        # Check if there's a running loop
        try:
            loop: asyncio.AbstractEventLoop = (
                asyncio.get_running_loop()
            )
        except RuntimeError:
            # No running loop - just use asyncio.run()
            asyncio.run(_shutdown())
        else:
            # Loop is running - use it directly
            if loop.is_running():
                import nest_asyncio

                nest_asyncio.apply()  # type: ignore
                loop.run_until_complete(
                    _shutdown()
                )
            else:
                asyncio.run(_shutdown())

    def __del__(self):
        self.close()
