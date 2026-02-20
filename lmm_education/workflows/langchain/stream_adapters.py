"""
Stream adapters for processing LangGraph response streams.

This module provides composable stream adapters that can wrap and
transform async iterators of compiled LangGraph streams.

Architecture:
- Generic components work with any LangGraph workflow
- Three-tier adapter system:
  * Tier 1: Multi-mode adapters (operate on (mode, event) tuples)
  * Tier 2: Message-only adapters (operate on (BaseMessageChunk,
    metadata) tuples)
  * Tier 3: string adapters (operate on str)

Stream modes for tier 1:
- "messages": (BaseMessageChunk, metadata) tuples for text display
- "values": Complete state snapshot after each node execution
- "updates": State changes {node_name: {field: value, ...}}

Stream modes for tier 2:
- "messages": (BaseMessageChunk, metadata) tuples for text display

Stream mode for tier 3:
- streams are limited to strings (any other information is lost).

Tier 1 adapters work on graphs streamed via stream_mode =
["messages", "values"] or ["messages", "values", "updates"]. Any
combination or two or three of these modes is valid, but the
combination will usually include "messages" for streaming. Tier 2
adapters work on streams from stream_mode = "messages". Tier 3 streams
arise from tier 1 or tier 2 streams, and stream simple strings. A
stream may be filtered from tier 1 to tier 2 or tier 3, but not in the
opposite direction, as information is lost during the conversion.

A use case for tier 3 streams is to extract information from messages,
metadata, or state and convey it into the string stream. A tier 3
stream will always be produced at some stage to stream content to the
output.

The stream output types are captured by type aliases:

- `tier_1_iterator`: For tier 1 streams (multimodal)
- `tier_2_iterator`: For tier 2 streams (messages)
- `tier_3_iterator`: For tier 3 streams (strings)

### Stream setup

Tier 1 stream setup:

- `stream_graph_state`: Entry point with "messages" and "values" modes
- `stream_graph_updates`: Entry point with "messages", "values",
  and "updates" modes for change-reactive patterns

Tier 1 streams may also be obtained by calling the graph .astream()
function directly with the appropriate stream_modes (see the body of
these functions for reference). Once initialized with the appropriate
stream_mode, the stream type is preserved across all tier 1 adapters.

Tier 2 stream setup:

- `stream_graph_messages`: Entry point with "messages" streams.

### Adapters

Tier 1 adapters:

- `field_change_tier_1_adapter`: Reacts to specific field changes via
    "updates"
- `terminal_tier_1_adapter`: De-multiplexes multi-mode stream to
    messages and calls callback with terminal state. The callback is
    given the final state as an argument.

Tier 1 to tier 2 adapters:

- `tier_1_to_2_adapter`

Tier 2 to tier 3 adapters:

- `tier_2_to_3_adapter`

Tier 1 to tier 3 adapters:

- `field_change_terminal_adapter`: Reacts to specific field changes
    via "updates", and calls a callback to insert into the string
    stream the result of the callback. It otherwise streams messages
    by extracting their content. It also takes a terminal state
    callback called after streaming, which is given the final state.
- `tier_1_to_3_adapter`

"""

# rev c 1.25

import copy
import inspect
from typing import Protocol, TypeVar, Any, Hashable, Literal
from collections.abc import (
    AsyncIterator,
    Callable,
    Awaitable,
    Mapping,
    Sequence,
)

from pydantic import BaseModel
from langchain_core.messages import BaseMessageChunk
from langgraph.types import Command

from lmm.utils.logging import LoggerBase, ConsoleLogger

from lmm_education.background_task_manager import schedule_task


# ===================================================================
# Generic Type Variables and Protocols
# ===================================================================

# StateType for generic adapters - any TypedDict (which is a Mapping)
StateT = TypeVar("StateT", bound=Mapping[str, Any])

# ContextType for generic adapters - any Pydantic model
ContextT = TypeVar("ContextT", bound=BaseModel)

# InputStateT for contravariant protocol input positions
InputStateT = TypeVar("InputStateT", contravariant=True)

# InputContextT for contravariant protocol context parameter
InputContextT = TypeVar(
    "InputContextT", bound=BaseModel, contravariant=True
)

# Hashable type parameter for Command type
N = TypeVar("N", bound=Hashable)

# StreamMode type matching LangGraph's signature
StreamMode = Literal[
    "messages", "values", "updates", "debug", "custom"
]

# Type for "updates" mode events: {node_name: {field_name: value}}
UpdatesEvent = dict[str, dict[str, Any]]

# ===================================================================
# Protocols for generics
# ===================================================================


class StreamableGraph(Protocol[InputStateT, InputContextT]):
    """
    Minimal protocol for LangGraph compiled graphs.

    This protocol only specifies the astream() method that we
    actually use, avoiding tight coupling to LangGraph internals.

    The InputStateT and InputContextT are contravariant because they
    appear only in input positions (parameters), allowing a graph
    that accepts specific types (like ChatState, ChatWorkflowContext)
    to be compatible with a protocol expecting broader types (like
    Mapping[str, Any], BaseModel).
    """

    def astream(
        self,
        input: InputStateT | Command[Any] | None,
        *,
        context: InputContextT | None = None,
        stream_mode: StreamMode | Sequence[StreamMode] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Stream graph execution with multiple output modes."""
        ...


# Iterator type for API consumers

# The tier_1_iterator returns tuples arising from calling langgraph's
# astream() method with a multimodal stream_mode, i.e. stream_mode =
# ['messages', 'values'], or stream_mode = ['messages', 'values',
# 'updates']. Any combination of these modes will give rise to a tier
# 1 iterator, but a combination including 'messages' is required for
# streaming.
tier_1_iterator = AsyncIterator[tuple[str, Any]]


# Tier 2 iterators arise when calling astream with stream_mode =
# 'messages'. They do not contain state but provide info on the
# calling node in the metadata member of the tuple. The 'messages'
# stream contains the chunks generated by streaming the language
# model inside nodes.
tier_2_iterator = AsyncIterator[
    tuple[BaseMessageChunk, dict[str, Any]]
]

# Tier 3 iterators arise from adapting iterators from other tiers.
# They only stream strings.
tier_3_iterator = AsyncIterator[str]


# ===================================================================
# Entry Points - Generic
# ===================================================================

# The entry points below document how a stream may be generated from
# a LangGraph workflow to obtain tier 1 or tier 2 streams.


def stream_graph_state(
    graph: StreamableGraph[Any, Any],  # types checked at other args
    initial_state: Mapping[str, Any],
    context: BaseModel,
) -> tier_1_iterator:
    """
    Entry point for streaming a LangGraph workflow with multi-mode
    output.

    This function configures the graph to stream both messages and
    state updates, enabling downstream adapters to:
    - Access and modify state (Tier 1 adapters)
    - Capture terminal state for logging
    - Extract messages for display

    Args:
        graph: Compiled LangGraph workflow
        initial_state: Initial state to start execution
        context: Dependency injection context for the workflow

    Returns:
        AsyncIterator yielding (mode, event) tuples (tier 1 iterator)
        where:
        - mode is "messages" or "values"
        - event is (BaseMessageChunk, metadata) for messages
        - event is StateT for values (a typed dictionary)

    Example:
        ```python
        raw_stream = stream_graph_state(workflow, initial_state,
                                        context)

        # Apply adapters
        validated_stream = stateful_validation_adapter(raw_stream,
            ...)
        terminal_stream = terminal_tier_1_adapter(validated_stream,
            on_terminal_state=log_fn)
        text_stream = tier_1_to_3_adapter(terminal_stream)

        # Consume for display
        async for chunk in text_stream:
            print(chunk.text, end="")
        ```
    """
    return graph.astream(
        initial_state,
        stream_mode=["messages", "values"],
        context=context,
    )


def stream_graph_updates(
    graph: StreamableGraph[Any, Any],  # types checked at other args
    initial_state: Mapping[str, Any],
    context: BaseModel,
) -> tier_1_iterator:
    """
    Entry point for streaming a LangGraph workflow with state updates.

    This function configures the graph to stream messages, full state
    values, and differential state updates. This enables:
    - Tier 1 adapters to access and modify state
    - Change-reactive adapters to respond to specific field updates
    - Terminal state capture for logging
    - Message extraction for display

    The "updates" stream mode provides differential updates in the
    format:
        ("updates", {node_name: {changed_field: new_value, ...}})

    This is more granular than "values" (which provides the complete
    state) and enables building adapters that react to specific state
    changes.

    Args:
        graph: Compiled LangGraph workflow
        initial_state: Initial state to start execution
        context: Dependency injection context for the workflow

    Returns:
        AsyncIterator yielding (mode, event) tuples (tier 1 iterator)
        where:
        - mode is "messages", "values", or "updates"
        - event is (BaseMessageChunk, metadata) for messages
        - event is full StateT for values
        - event is UpdatesEvent for updates

    Example:
        ```python
        raw_stream = stream_graph_updates(workflow, initial_state,
            context)

        # Apply change-reactive adapters
        reactive_stream = field_change_tier_1_adapter(
            raw_stream,
            on_field_change={
                "status": handle_status_change,
                "context": handle_context_ready,
            }
        )

        # Apply other adapters
        terminal_stream = terminal_tier_1_adapter(reactive_stream,
            on_terminal_state=log_fn)
        text_stream = tier_1_to_3_adapter(terminal_stream)

        # Consume for display
        async for chunk in text_stream:
            print(chunk, end="")
        ```
    """
    return graph.astream(
        initial_state,
        stream_mode=["messages", "values", "updates"],
        context=context,
    )


def stream_graph_messages(
    graph: StreamableGraph[Any, Any],  # types checked at other args
    initial_state: Mapping[str, Any],
    context: BaseModel,
) -> tier_2_iterator:
    """
    Entry point for streaming a LangGraph workflow with messages
    output.

    This function configures the graph to stream only messages,
    enabling downstream adapters to:

    - Extract messages for display

    Args:
        graph: Compiled LangGraph workflow
        initial_state: Initial state to start execution
        context: Dependency injection context for the workflow

    Returns:
        AsyncIterator yielding 'event' tuples of (BaseMessageChunk,
        dict[str, Any]) for messages (tier 2 iterator)

    Example:
        ```python
        raw_stream = stream_graph_messages(workflow, initial_state,
            context)

        # Apply adapters
        ...

        # Consume for display
        async for chunk, _ in raw_stream:
            print(chunk.text, end="")
        ```
    """
    return graph.astream(
        initial_state,
        stream_mode="messages",
        context=context,
    )


# ===================================================================
# Tier 1 Adapters - Multi-Mode (State-Aware)
# ===================================================================


async def field_change_tier_1_adapter(
    multi_mode_stream: tier_1_iterator,
    *,
    on_field_change: (
        dict[str, Callable[[Any], Awaitable[str | None]]]
        | dict[str, Callable[[Any], str | None]]
        | None
    ) = None,
    logger: LoggerBase = ConsoleLogger(),
) -> tier_1_iterator:
    """
    Tier 1 adapter that reacts to specific field changes via "updates"
    events.

    This adapter uses the "updates" stream mode to detect when
    specific state fields change, and invokes registered callbacks.

    This enables building reactive adapters that respond to specific
    state changes, such as:
    - React when "status" changes to "valid"
    - React when "context" becomes available

    Args:
        multi_mode_stream: Source stream with (mode, event) tuples
        on_field_change: Dict mapping field names to async callbacks.
            Each callback receives the updated field value as its
            argument. If the callback returns a value, a deep copy of
            the event is created with the field value replaced by the
            callback's return value. If the callback returns None, the
            event is passed through unchanged.
            Callbacks are invoked when those fields are updated.
        logger: Logger to use for error logging

    Yields:
        All (mode, event) tuples from the stream. Update events are
        deep-copied if any registered callback returns a non-None value.

    Example:
        ```python
        async def on_context_ready(context: str):
            logger.info(f"Context retrieved: {len(context)} chars")

        async def on_status_changed(status: str):
            logger.info(f"Status changed to: {status}")

        reactive_stream = field_change_tier_1_adapter(
            multi_mode_stream,
            on_field_change={
                "context": on_context_ready,
                "status": on_status_changed,
            }
        )

        async for mode, event in reactive_stream:
            # Process events as usual
            ...
        ```
    """
    if on_field_change is None:
        async for mode, event in multi_mode_stream:
            yield (mode, event)
        return

    async for mode, event in multi_mode_stream:
        # React to field changes via "updates" events
        if mode == "updates" and isinstance(event, dict):
            # Deep copy to avoid modifying the original event
            event_copy = copy.deepcopy(event)  # type: ignore
            modified = False

            # event format: {node_name: {field_name: value, ...}}
            for node_name, changes in event_copy.items():  # type: ignore[union-attr]
                if isinstance(changes, dict):
                    for field, value in changes.items():  # type: ignore[union-attr]
                        if field in on_field_change:
                            callback_fun = on_field_change[field]
                            try:
                                content: str | None = None
                                if inspect.iscoroutinefunction(
                                    callback_fun
                                ):
                                    content = await callback_fun(
                                        value
                                    )
                                else:
                                    content = callback_fun(value)  # type: ignore
                                if content is not None:
                                    # Actually modify the field in the copy
                                    changes[field] = content
                                    modified = True
                            except Exception as e:
                                # Log the error but don't stop stream
                                logger.error(
                                    f"Error in field_change_adapter: "
                                    f"{e}"
                                )
                                pass

            # Yield the modified copy if we made changes, otherwise the original
            yield (mode, event_copy if modified else event)
        else:
            # For non-update events, yield unchanged
            yield (mode, event)


async def terminal_tier1_adapter(
    multi_mode_stream: tier_1_iterator,
    *,
    on_terminal_state: (
        Callable[[StateT], Any]
        | Callable[[StateT], Awaitable[Any]]
        | None
    ) = None,
    logger: LoggerBase = ConsoleLogger(),
) -> tier_1_iterator:
    """
    Tier 1 terminal adapter that calls a callback with the terminal
    state. The tier 1 iterator must have been created with the
    "values" stream mode for the on_terminal_state callback to be
    called.

    This adapter:
    - Captures the terminal state (last "values" event)
    - Invokes an optional callback with the terminal state

    Args:
        multi_mode_stream: Source stream with (mode, event) tuples
        on_terminal_state: Optional callback for terminal state.
            The callback receives Mapping[str, Any] type.

    Yields:
        (mode, event) tuples from the stream

    Example:
        ```python
        messages = terminal_tier1_adapter(
            multi_mode_stream,
            on_terminal_state=lambda s: logger.log(s),
        )
        async for mode, event in messages:
            if mode == 'messages':
                ...
        ```
    """

    final_state: StateT | None = None
    async for mode, event in multi_mode_stream:
        if mode == "values":
            final_state = event

        yield (mode, event)

    if on_terminal_state and final_state:
        if inspect.iscoroutinefunction(on_terminal_state):
            schedule_task(
                on_terminal_state(final_state),
                error_callback=lambda e: logger.error(
                    f"Error in on_terminal_state: {e}"
                ),
            )
        else:
            try:
                on_terminal_state(final_state)
            except Exception as e:
                # log error but do not stop stream
                logger.error(f"Error in on_terminal_state: {e}")


async def tier_1_filter_messages_adapter(
    stream: tier_1_iterator, exclude_nodes: list[str]
) -> tier_1_iterator:
    """
    Filter out message chunks from the stream that come
    from the excluded nodes (for example, to exclude tool messages).

    Args:
        stream: a multimodal tier 1 iterator
        exclude_nodes: a string list with the names of the nodes
            that should be excluded from the stream (only messages
            stream)
    """
    async for mode, event in stream:
        if mode == "messages":
            _, metadata = event
            if metadata.get("langgraph_node") in exclude_nodes:
                continue

        yield (mode, event)


# ===================================================================
# Tier 1 to tier 2 adapters
# ===================================================================


async def tier_1_to_2_adapter(
    multi_mode_stream: tier_1_iterator,
    source_nodes: list[str] | None = None,
    *,
    logger: LoggerBase = ConsoleLogger(),
) -> tier_2_iterator:
    """
    Terminal adapter: filters multi-mode stream to messages only.
    Non-message events are discarded.

    Args:
        multi_mode_stream: Source stream with (mode, event) tuples
        source_nodes: the source nodes one wants to stream. If
        omitted or None, all nodes will be streamed
        logger: logger to use for error logging

    Yields:
        BaseMessageChunk, metadata tuples (tier 2 iterator)

    Example:
        ```python
        messages = tier_1_to_2_adapter(multi_mode_stream)

        async for chunk, _ in messages:
            yield chunk.text
        ```
    """
    if source_nodes:
        async for mode, event in multi_mode_stream:
            if mode == "messages":
                try:
                    _, metadata = event
                    if metadata["langgraph_node"] in source_nodes:
                        yield event
                except Exception as e:
                    logger.error(
                        f"Could not retrieve langgraph_node property"
                        f" in tier_1_to_2_adapter:\n{e}"
                    )
                    yield event
            else:
                pass
    else:
        async for mode, event in multi_mode_stream:
            if mode == "messages":
                yield event
            else:
                pass


# ===================================================================
# Tier 2 Adapter
# ===================================================================


async def tier_2_filter_messages_adapter(
    stream: tier_2_iterator, exclude_nodes: list[str]
) -> tier_2_iterator:
    """
    Filter out message chunks from the stream that come
    from the excluded nodes (for example, to exclude tool messages).

    Args:
        stream: a chunk, metadata tier 2 iterator
        exclude_nodes: a string list with the names of the nodes
            that should be excluded from the stream
    """
    async for chunk, metadata in stream:
        if metadata.get("langgraph_node") in exclude_nodes:
            continue
        yield chunk, metadata


# ===================================================================
# Tier 2 to Tier 3 Adapter
# ===================================================================


async def tier_2_to_3_adapter(
    messages_stream: tier_2_iterator,
    source_nodes: list[str] | None = None,
    exclude_nodes: list[str] | None = None,
    *,
    logger: LoggerBase = ConsoleLogger(),
) -> tier_3_iterator:
    """
    Transforms a tier 2 iterator into a tier 3 iterator. 'metadata'
    information from the tier 2 iterator is discarded.

    Args:
        messages_stream: Source stream with (chunk, metadata) tuples
        source_nodes: the source nodes one wants to stream. If
            omitted or None, all nodes will be streamed.
        exclude_nodes: the source nodes one wants to exclude from
            streaming. If specified, overrides source_nodes.
        logger: Logger to use for logging

    Yields:
        strings (tier 3 iterator)

    Example:
        ```python
        messages = tier_2_to_3_adapter(messages_stream)

        async for chunk in messages:
            yield chunk
    """
    async for chunk, metadata in messages_stream:
        if exclude_nodes:
            try:
                if metadata["langgraph_node"] in exclude_nodes:
                    continue
            except Exception as e:
                logger.error(
                    f"Could not retrieve langgraph_node property"
                    f" in tier_2_to_3_adapter:\n{e}"
                )
                continue
        elif source_nodes:
            try:
                if metadata["langgraph_node"] in source_nodes:
                    yield chunk.text
            except Exception as e:
                logger.error(
                    f"Could not retrieve langgraph_node property"
                    f" in tier_2_to_3_adapter:\n{e}"
                )
                yield chunk.text
        else:
            yield chunk.text


# ===================================================================
# Tier 1 to tier 3 adapters
# ===================================================================


async def terminal_field_change_adapter(
    multi_mode_stream: tier_1_iterator,
    source_nodes: list[str] | None = None,
    exclude_nodes: list[str] | None = None,
    *,
    on_field_change: (
        dict[str, Callable[[Any], Awaitable[str | None]]]
        | dict[str, Callable[[Any], str | None]]
        | None
    ) = None,
    on_terminal_state: (
        Callable[[StateT], Any]
        | Callable[[StateT], Awaitable[Any]]
        | None
    ) = None,
    logger: LoggerBase = ConsoleLogger(),
) -> tier_3_iterator:
    """
    This adapter captures the 'messages' stream and converts it to a
    tier 3 iterator (a stream of strings). It also captures the
    'updates' stream and invokes callbacks registered in
    on_field_change, if provided. These callbacks return strings
    that are injected into the tier 3 stream.

    The callback given to on_terminal_state is called after
    streaming (may be used, for example, for logging the exchange).
    The output of this callback is not injected into the tier 3
    stream. The stream must have been created with the 'values' mode
    for this callback to be called.

    This adapter:
    - Extracts and yields only the text of messages chunks and
        the return of the on_field_change callback. Any other
        information from the 'updates' stream is discarded.
    - Captures the terminal state (last "values" event), and
        invokes an optional callback with the terminal state.
    - State information is discarded in the output stream.

    Args:
        multi_mode_stream: Source stream with (mode, event) tuples
        source_nodes: the source nodes one wants to stream. If
            omitted or None, all nodes will be streamed (only
            concerns the 'messages' stream).
        exclude_nodes: the source nodes one wants to exclude from
            streaming. If specified, overrides source_nodes.
        on_field_change: Dict mapping field names to async callbacks.
            Each callback receives the updated field value as its
            argument, and returns a string that is injected into the
            tier 3 stream. If the callback returns None, no string is
            injected.
            Callbacks are invoked when those fields are updated.
        on_terminal_state: Optional callback for terminal state.
            The callback receives Mapping[str, Any] type.
        logger: Logger to use for error logging

    Yields:
        strings (tier 3 iterator)

    Example:
        ```python
        async def on_context_ready(context: str) -> str:
            return f"Context retrieved: {len(context)} chars"

        async def on_status_changed(status: str) -> str:
            return f"Status changed to: {status}"

        texts = terminal_field_change_adapter(
            multi_mode_stream,
            on_field_change={
                "context": on_context_ready,
                "status": on_status_changed,
            },
            on_terminal_state=lambda s: logger.log(s),
        )

        async for text in texts:
            yield text
        ```
    """

    final_state: StateT | None = None
    async for mode, event in multi_mode_stream:
        if mode == "messages":
            if exclude_nodes:
                chunk, metadata = event  # Extract chunk and metadata
                try:
                    if metadata["langgraph_node"] in exclude_nodes:
                        continue
                except Exception as e:
                    logger.error(
                        f"Could not retrieve langgraph_node property"
                        f" in terminal_field_change_adapter:\n{e}"
                    )
                    yield chunk.text
            elif source_nodes:
                chunk, metadata = event  # Extract chunk and metadata
                try:
                    if metadata["langgraph_node"] in source_nodes:
                        yield chunk.text
                except Exception as e:
                    logger.error(
                        f"Could not retrieve langgraph_node property"
                        f" in terminal_field_change_adapter:\n{e}"
                    )
                    yield chunk.text
            else:
                chunk, _ = event
                yield chunk.text
        elif (
            mode == "updates"
            and isinstance(event, dict)
            and on_field_change
        ):
            # event format: {node_name: {field_name: value, ...}}
            for node_name, changes in event.items():  # type: ignore[union-attr]
                if isinstance(changes, dict):
                    for field, value in changes.items():  # type: ignore[union-attr]
                        if field in on_field_change:
                            callback_fun = on_field_change[field]
                            try:
                                content: str | None = None
                                if inspect.iscoroutinefunction(
                                    callback_fun
                                ):
                                    content = await callback_fun(
                                        value
                                    )
                                else:
                                    content = callback_fun(value)  # type: ignore
                                # Only yield if callback returned a value
                                if content is not None:
                                    yield content
                            except Exception as e:
                                # Log the error but don't stop stream
                                logger.error(
                                    f"Error in field_change_adapter: "
                                    f"{e}"
                                )
                                pass
        elif mode == "values":
            final_state = event  # type: ignore
        else:
            pass

    if on_terminal_state and final_state:
        if inspect.iscoroutinefunction(on_terminal_state):
            schedule_task(
                on_terminal_state(final_state),
                error_callback=lambda e: logger.error(
                    f"Error in on_terminal_state: {e}"
                ),
            )
        else:
            try:
                on_terminal_state(final_state)
            except Exception as e:
                # log error but do not stop stream
                logger.error(f"Error in on_terminal_state: {e}")


async def tier_1_to_3_adapter(
    multi_mode_stream: tier_1_iterator,
    source_nodes: list[str] | None = None,
    exclude_nodes: list[str] | None = None,
    *,
    logger: LoggerBase = ConsoleLogger(),
) -> tier_3_iterator:
    """
    Transforms a tier 1 iterator into a tier 3 iterator. All
    information from the 'updates' and 'values' streams is discarded.

    Args:
        multi_mode_stream: Source stream with (mode, event) tuples
        source_nodes: the source nodes one wants to stream. If
            omitted or None, all nodes will be streamed (only concerns
            the 'messages' stream).
        exclude_nodes: the source nodes one wants to exclude from
            streaming. If specified, overrides source_nodes.
        logger: Logger to use for error logging

    Yields:
        strings (tier 3 iterator)

    Example:
        ```python
        messages = tier_1_to_3_adapter(multi_mode_stream)

        async for chunk in messages:
            yield chunk
        ```
    """
    async for mode, event in multi_mode_stream:
        if mode == "messages":
            if exclude_nodes:
                chunk, metadata = event  # Extract chunk and metadata
                try:
                    if metadata["langgraph_node"] in exclude_nodes:
                        continue
                except Exception as e:
                    logger.error(
                        f"Could not retrieve langgraph_node property"
                        f" in terminal_field_change_adapter:\n{e}"
                    )
                    yield chunk.text
            elif source_nodes:
                chunk, metadata = event  # Extract chunk and metadata
                try:
                    if metadata["langgraph_node"] in source_nodes:
                        yield chunk.text
                except Exception as e:
                    logger.error(
                        f"Could not retrieve langgraph_node property"
                        f" in terminal_field_change_adapter:\n{e}"
                    )
                    yield chunk.text
            else:
                chunk, _ = event
                yield chunk.text
        else:
            pass
