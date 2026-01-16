"""
Stream adapters for processing LangGraph response streams.

This module provides composable stream adapters that can wrap and
transform async iterators of BaseMessageChunk objects. For example,
one use case is content validation, which buffers initial response
chunks, validates them using a separate LLM, and either passes through
or rejects the stream.

Architecture:
- Generic components work with any LangGraph workflow
- Domain-specific components (e.g., validation) use ChatState
    explicitly
- Three-tier adapter system:
  * Tier 1: Multi-mode adapters (operate on (mode, event) tuples)
  * Tier 2: Message-only adapters (operate on BaseMessageChunk)
  * Tier 3: string adapters (operate on str)

Stream Modes for tier 1 and tier 2:
- "messages": (BaseMessageChunk, metadata) tuples for text display
- "values": Complete state snapshot after each node execution
- "updates": Differential state changes {node_name: {field: value, ...}}

Stream mode for tier 3:
- streams are limited to strings (any other information is lost).

Tier 1 adapters work on graphs streamed via stream_mode =
["messages", "values"] or ["messages", "values", "updates"]. Tier 2
adapters work on streams from stream_mode = "messages". Tier 3 streams
arise from tier 1 or tier 2 streams, and stream simple strings. A
stream may be filtered from tier 1 to tier 2 or tier 3, but not in the
opposite direction, as state is lost from the stream.

A use case for tier 3 streams is to extract information from messages,
metadata, or state and convey it into the string stream. A tier 3
stream will always be produced at some stage to stream content to the
chatbot.

The stream output types are captures by type aliases:

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

- `stateful_validation_adapter`: Validates content and modifies state
- `field_change_adapter`: Reacts to specific field changes via
    "updates"

Tier 1 to tier 2 adapters:

- `terminal_demux_adapter`: De-multiplexes multi-mode stream to
    messages and calls callback with terminal state. The callback is
    given the final state as an argument.
- `demux_adapter`: De-multiplexes multi-mode stream to messages only

Tier 1 to tier 3 adapters:

- `field_change_terminal_adapter`: Reacts to specific field changes
    via "updates", and calls a callback to insert into the string
    stream the result of the callback. It otherwise streams messages
    by extracting their content. It also takes a terminal state
    callback called after streaming, which is given the final state.

"""

import asyncio
from typing import Protocol, TypeVar, Any, Hashable, Literal
from collections.abc import (
    AsyncIterator,
    Callable,
    Awaitable,
    Mapping,
    Sequence,
)

from pydantic import BaseModel
from langchain_core.messages import BaseMessageChunk, AIMessageChunk
from langgraph.types import Command

from lmm.utils.logging import LoggerBase, ConsoleLogger
from lmm.language_models.langchain.runnables import RunnableType

# Import ChatState for domain-specific adapters
from lmm_education.chat_graph import ChatState


# ============================================================================
# Generic Type Variables and Protocols
# ============================================================================

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

# ============================================================================
# Protocols for generics
# ============================================================================


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


class StreamableGraphMessages(Protocol[InputStateT, InputContextT]):
    """
    Minimal protocol for LangGraph compiled graphs.

    This protocol only specifies the astream() method that we
    actually use, avoiding tight coupling to LangGraph internals.
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


# define iterator type for API consumers, The tier_1_iterator returns
# tuples arising from calling langgraph's astream with stream_mode =
# "values" or "updates".
tier_1_iterator = AsyncIterator[tuple[str, Any]]


# Tier 2 iterator arise when calling astream with stream_mode =
# "messages". They do not contain state but provide info on the
# calling node in the metadata member of the tuple.
tier_2_iterator = AsyncIterator[
    tuple[BaseMessageChunk, dict[str, Any]]
]

# Tier 3 iterators arise from adapting iterators from other tiers.
# They only stream strings.
tier_3_iterator = AsyncIterator[str]


# ============================================================================
# Entry Point - Generic
# ============================================================================


def stream_graph_state(
    graph: StreamableGraph[Any, Any],
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
        - event is StateT for values

    Example:
        ```python
        raw_stream = stream_graph_state(workflow, initial_state,
                                        context)

        # Apply adapters
        validated_stream = stateful_validation_adapter(raw_stream, ...)
        message_stream = terminal_demux_adapter(validated_stream,
                                    on_terminal_state=log_fn)

        # Consume for display
        async for chunk in message_stream:
            print(chunk.text, end="")
        ```
    """
    return graph.astream(
        initial_state,
        stream_mode=["messages", "values"],
        context=context,
    )


def stream_graph_updates(
    graph: StreamableGraph[Any, Any],
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
        raw_stream = astream_graph_updates(workflow, initial_state,
            context)

        # Apply change-reactive adapters
        reactive_stream = field_change_adapter(
            raw_stream,
            on_field_change={
                "status": handle_status_change,
                "context": handle_context_ready,
            }
        )

        # Apply other adapters
        message_stream = terminal_demux_adapter(reactive_stream,
            on_terminal_state=log_fn)

        # Consume for display
        async for chunk in message_stream:
            print(chunk.text, end="")
        ```
    """
    return graph.astream(
        initial_state,
        stream_mode=["messages", "values", "updates"],
        context=context,
    )


def astream_graph_messages(
    graph: StreamableGraphMessages[Any, Any],
    initial_state: Mapping[str, Any],
    context: BaseModel,
) -> tier_2_iterator:
    """
    Entry point for streaming a LangGraph workflow with messagess
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
        metadata) for messages (tier 2 iterator)

    Example:
        ```python
        raw_stream = astream_graph_messages(workflow, initial_state,
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


# ============================================================================
# Tier 1 Adapters - Multi-Mode (State-Aware)
# ============================================================================


async def stateful_validation_adapter(
    multi_mode_stream: tier_1_iterator,
    *,
    validator_model: RunnableType,
    allowed_content: list[str],
    buffer_size: int = 320,
    error_message: str = "Content not allowed",
    max_retries: int = 2,
    logger: LoggerBase = ConsoleLogger(),
) -> tier_1_iterator:
    """
    Multi-mode adapter that validates message content and modifies
    state.

    This is a Tier 1 (state-aware) adapter that:
    - Buffers message chunks until sufficient content is collected
    - Validates content using a separate LLM
    - If validation fails:
      * Yields a rejection message
      * Modifies ChatState.status to "rejected"
      * Stops streaming
    - If validation passes:
      * Releases buffered content
      * Continues streaming normally

    NOTE: This adapter is domain-specific - it uses ChatState directly
    and understands the semantics of the "status" field.

    Args:
        multi_mode_stream: Source stream yielding (mode, event) tuples
        query_text: Original user query (for validation context)
        validator_model: Runnable for content classification
        allowed_content: List of allowed content categories
        buffer_size: Characters to buffer before validation
        error_message: Message to yield if validation fails
        max_retries: Validation retry attempts
        logger: Logger for diagnostics

    Yields:
        (mode, event) tuples with potentially modified state
    """
    buffer_chunks: list[tuple[str, Any]] = []
    buffer_text: str = ""
    validation_complete: bool = False
    captured_state: ChatState | None = None
    query_text: str = (
        captured_state.query_text if captured_state else ""
    )

    async def _validate_content(content: str) -> tuple[bool, str]:
        """Validate content with retry logic."""
        for attempt in range(max_retries + 1):
            try:
                classification: str = await validator_model.ainvoke(
                    {"text": content}
                )
                logger.info(
                    "Model content classification: "
                    + classification.replace("\n", " ")
                )

                if classification in allowed_content + [
                    "apology",
                    "human interaction",
                ]:
                    return True, classification
                else:
                    return False, classification

            except Exception as e:
                logger.warning(
                    f"Content check attempt {attempt + 1}/"
                    f"{max_retries + 1} failed: {e}"
                )

                if attempt == max_retries:
                    logger.error(
                        f"Content checker failed after "
                        f"{max_retries + 1} attempts: {e}"
                    )
                    return True, "validation_unavailable"

                await asyncio.sleep(0.5 * (attempt + 1))

        return True, "validation_error"

    async for mode, event in multi_mode_stream:
        # Track latest state for potential modification
        if mode == "values":
            captured_state = event

        # Validation logic only applies to messages
        if mode == "messages" and not validation_complete:
            chunk, _ = event
            buffer_chunks.append((mode, event))
            buffer_text += chunk.text

            if len(buffer_text) >= buffer_size:
                # Validate buffered content
                is_valid, classification = await _validate_content(
                    query_text + "\n\n" + buffer_text + "..."
                )
                validation_complete = True

                if not is_valid:
                    # Validation failed - yield rejection and
                    # modified state
                    logger.info(
                        f"Content rejected with classification: "
                        f"{classification}"
                    )

                    # Yield rejection message
                    yield (
                        "messages",
                        (AIMessageChunk(content=error_message), {}),
                    )

                    # Modify state to reflect rejection
                    if captured_state:
                        modified_state: ChatState = {
                            **captured_state,
                            "query_classification": classification,
                            "status": "rejected",
                        }
                        yield ("values", modified_state)

                    return  # Stop streaming

                # Validation passed - release buffer
                if classification == "validation_unavailable":
                    logger.warning(
                        "LLM exchange without content check "
                        "(validation unavailable)"
                    )
                    if captured_state:
                        modified_state: ChatState = {
                            **captured_state,
                            "query_classification": "NA",
                        }
                        yield ("values", modified_state)
                else:
                    # record classification
                    if captured_state:
                        modified_state: ChatState = {
                            **captured_state,
                            "query_classification": classification,
                        }
                        yield ("values", modified_state)

                for buffered_mode, buffered_event in buffer_chunks:
                    yield (buffered_mode, buffered_event)
        else:
            # Pass through non-message events or post-validation
            # messages
            yield (mode, event)

    # Handle case where stream ended before buffer was full
    if not validation_complete and buffer_text:
        is_valid, classification = await _validate_content(
            query_text + "\n\n" + buffer_text
        )

        if not is_valid:
            logger.info(
                f"Content rejected with classification: "
                f"{classification}"
            )
            yield (
                "messages",
                (AIMessageChunk(content=error_message), {}),
            )

            if captured_state:
                modified_state: ChatState = {
                    **captured_state,
                    "status": "rejected",  # type: ignore
                }
                yield ("values", modified_state)
            return

        if classification == "validation_unavailable":
            logger.warning(
                f"LLM exchange without content check: "
                f"{buffer_text[:100]}..."
            )

        # Yield all buffered chunks
        for buffered_mode, buffered_event in buffer_chunks:
            yield (buffered_mode, buffered_event)


async def field_change_adapter(
    multi_mode_stream: tier_1_iterator,
    *,
    on_field_change: (
        dict[str, Callable[[Any], Awaitable[None]]] | None
    ) = None,
) -> tier_1_iterator:
    """
    Adapter that reacts to specific field changes via "updates"
    events.

    This adapter uses the "updates" stream mode to detect when
    specific state fields change, and invokes registered callbacks.
    It passes through all stream events unchanged.

    This enables building reactive adapters that respond to specific
    state changes, such as:
    - React when "status" changes to "valid"
    - React when "context" becomes available
    - React when "query_classification" is set

    Args:
        multi_mode_stream: Source stream with (mode, event) tuples
        on_field_change: Dict mapping field names to async callbacks.
            Each callback receives the new field value as its
            argument.
            Callbacks are invoked when those fields are updated.

    Yields:
        All (mode, event) tuples from the stream, unchanged

    Example:
        ```python
        async def on_context_ready(context: str):
            logger.info(f"Context retrieved: {len(context)} chars")

        async def on_status_changed(status: str):
            logger.info(f"Status changed to: {status}")

        reactive_stream = field_change_adapter(
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
        on_field_change = {}

    async for mode, event in multi_mode_stream:
        # React to field changes via "updates" events
        if mode == "updates" and isinstance(event, dict):
            # event format: {node_name: {field_name: value, ...}}
            for node_name, changes in event.items():  # type: ignore[union-attr]
                if isinstance(changes, dict):
                    for field, value in changes.items():  # type: ignore[union-attr]
                        if field in on_field_change:
                            try:
                                await on_field_change[field](value)
                            except Exception as e:
                                # Log the error but don't stop stream
                                logger = ConsoleLogger()
                                logger.error(
                                    f"Error in field_change_adapter: "
                                    f"{e}"
                                )
                                pass

        # Always yield the event unchanged
        yield (mode, event)


# ============================================================================
# Tier 1 to tier 2 adapters
# ============================================================================


async def terminal_demux_adapter(
    multi_mode_stream: tier_1_iterator,
    *,
    on_terminal_state: Callable[[StateT], Any] | None = None,
) -> tier_2_iterator:
    """
    Terminal adapter: de-multiplexes multi-mode stream to messages
    only. The callback given to on_terminal_state is called after
    streaming (may be used, for example, for logging the exchange).

    This adapter:
    - Extracts and yields only message chunks and metadata
    - Captures the terminal state (last "values" event)
    - Invokes an optional callback with the terminal state

    Args:
        multi_mode_stream: Source stream with (mode, event) tuples
        on_terminal_state: Optional callback for terminal state.
            The callback receives Mapping[str, Any] type.

    Yields:
        BaseMessageChunk, metadata tuples (tier 2 iterator)

    Example:
        ```python
        messages = terminal_demux_adapter(
            multi_mode_stream,
            on_terminal_state=lambda s: logger.log(s),
        )

        async for chunk, _ in messages:
            yield chunk.text  # To Gradio
        ```
    """

    final_state: StateT | None = None
    async for mode, event in multi_mode_stream:
        if mode == "messages":
            chunk, metadata = event  # Extract chunk and metadata
            yield chunk, metadata
        else:
            final_state = event

    if on_terminal_state and final_state:
        on_terminal_state(final_state)


async def demux_adapter(
    multi_mode_stream: tier_1_iterator,
) -> tier_2_iterator:
    """
    Terminal adapter: de-multiplexes multi-mode stream to messages
    only.

    This is the final adapter in the Tier 1 chain. It:
    - Extracts and yields only message chunks and metadata

    This adapter is generic - it works with any state type.

    Args:
        multi_mode_stream: Source stream with (mode, event) tuples

    Yields:
        BaseMessageChunk, metadata tuples (tier 2 iterator)

    Example:
        ```python
        messages = demux_adapter(multi_mode_stream)

        async for chunk, _ in messages:
            yield chunk.text  # To Gradio
        ```
    """
    async for mode, event in multi_mode_stream:
        if mode == "messages":
            chunk, metadata = event  # Extract chunk and metadata
            yield chunk, metadata
        else:
            pass


# ============================================================================
# Tier 1 to tier 3 adapters
# ============================================================================


async def terminal_field_change_adapter(
    multi_mode_stream: tier_1_iterator,
    *,
    on_field_change: dict[str, Callable[[Any], str]] | None = None,
    on_terminal_state: Callable[[StateT], Any] | None = None,
) -> tier_3_iterator:
    """
    Terminal adapter: de-multiplexes multi-mode stream to strings
    only. Uses the "updates" stream mode to detect specific state
    fields changes, invoking callbacks registered in on_field_change.
    The callback given to on_terminal_state is called after
    streaming (may be used, for example, for logging the exchange).

    This adapter:
    - Extracts and yields only the text of messages chunks and
        the return of the on_field_change callback
    - Captures the terminal state (last "values" event), and
        invokes an optional callback with the terminal state

    Args:
        multi_mode_stream: Source stream with (mode, event) tuples
        on_field_change: Dict mapping field names to async callbacks.
            Each callback receives the new field value as its
            argument.
            Callbacks are invoked when those fields are updated.
        on_terminal_state: Optional callback for terminal state.
            The callback receives Mapping[str, Any] type.

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
            }
            on_terminal_state=lambda s: logger.log(s),
        )

        async for text in texts:
            yield text
        ```
    """

    final_state: StateT | None = None
    async for mode, event in multi_mode_stream:
        if mode == "messages":
            chunk, _ = event  # Extract chunk and metadata
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
                            try:
                                yield on_field_change[field](value)
                            except Exception as e:
                                # Log the error but don't interrupt
                                # the stream
                                logger = ConsoleLogger()
                                logger.error(
                                    f"Error in "
                                    f"terminal_field_change_adapter: "
                                    f"{e}"
                                )
                                pass
        elif mode == "values":
            final_state = event  # type: ignore
        else:
            pass

    if on_terminal_state and final_state:
        on_terminal_state(final_state)


async def tier_3_adapter(
    multi_mode_stream: tier_1_iterator,
) -> tier_3_iterator:
    """
    Terminal adapter: de-multiplexes multi-mode stream to messages
    only.

    This is the final adapter in the Tier 1 chain. It:
    - Extracts and yields only message chunks and metadata

    This adapter is generic - it works with any state type.

    Args:
        multi_mode_stream: Source stream with (mode, event) tuples

    Yields:
        BaseMessageChunk, metadata tuples (tier 2 iterator)

    Example:
        ```python
        messages = demux_adapter(multi_mode_stream)

        async for chunk, _ in messages:
            yield chunk.text  # To Gradio
        ```
    """
    async for mode, event in multi_mode_stream:
        if mode == "messages":
            chunk, _ = event  # Extract chunk and metadata
            yield chunk.text
        else:
            pass


# ============================================================================
# Tier 2 Adapters - Message-Only
# ============================================================================
