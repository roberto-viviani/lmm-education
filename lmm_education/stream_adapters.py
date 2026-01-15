"""
Stream adapters for processing LangGraph response streams.

This module provides composable stream adapters that can wrap and transform
async iterators of BaseMessageChunk objects. The primary use case is
content validation, which buffers initial response chunks, validates them
using a separate LLM, and either passes through or rejects the stream.

Architecture:
- Generic components work with any LangGraph workflow
- Domain-specific components (e.g., validation) use ChatState explicitly
- Two-tier adapter system:
  * Tier 1: Multi-mode adapters (operate on (mode, event) tuples)
  * Tier 2: Message-only adapters (operate on BaseMessageChunk)

Stream Modes:
- "messages": (BaseMessageChunk, metadata) tuples for text display
- "values": Complete state snapshot after each node execution
- "updates": Differential state changes {node_name: {field: value, ...}}

Tier 1 adapters work on graphs streamed via stream_mode =
["messages", "values"] or ["messages", "values", "updates"]. Tier 2
adapters work on streams from stream_mode = "messages". A stream may be
filtered from tier 1 to tier 2, but not in the opposite direction, as
state is lost from the stream.

Adapters:
- `astream_graph`: Entry point with "messages" and "values" modes
- `astream_graph_with_updates`: Entry point with "messages", "values",
  and "updates" modes for change-reactive patterns
- `stateful_validation_adapter`: Validates content and modifies state
- `field_change_adapter`: Reacts to specific field changes via "updates"
- `terminal_demux_adapter`: De-multiplexes multi-mode stream to messages
  and calls callback with terminal state
- `demux_adapter`: De-multiplexes multi-mode stream to messages only

The callback is given the final state as an argument.
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


class StreamableGraph(Protocol[InputStateT, InputContextT]):
    """
    Minimal protocol for LangGraph compiled graphs.

    This protocol only specifies the astream() method that we actually use,
    avoiding tight coupling to LangGraph internals.

    The InputStateT and InputContextT are contravariant because they appear
    only in input positions (parameters), allowing a graph that accepts
    specific types (like ChatState, ChatWorkflowContext) to be compatible
    with a protocol expecting broader types (like Mapping[str, Any], BaseModel).
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

    This protocol only specifies the astream() method that we actually use,
    avoiding tight coupling to LangGraph internals.
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


# ============================================================================
# Entry Point - Generic
# ============================================================================


def astream_graph(
    graph: StreamableGraph[Any, Any],
    initial_state: Mapping[str, Any],
    context: BaseModel,
) -> AsyncIterator[tuple[str, Any]]:
    """
    Entry point for streaming a LangGraph workflow with multi-mode output.

    This function configures the graph to stream both messages and state
    updates, enabling downstream adapters to:
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
        raw_stream = astream_graph(workflow, initial_state, context)

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


def astream_graph_with_updates(
    graph: StreamableGraph[Any, Any],
    initial_state: Mapping[str, Any],
    context: BaseModel,
) -> AsyncIterator[tuple[str, Any]]:
    """
    Entry point for streaming a LangGraph workflow with state updates.

    This function configures the graph to stream messages, full state values,
    and differential state updates. This enables:
    - Tier 1 adapters to access and modify state
    - Change-reactive adapters to respond to specific field updates
    - Terminal state capture for logging
    - Message extraction for display

    The "updates" stream mode provides differential updates in the format:
        ("updates", {node_name: {changed_field: new_value, ...}})

    This is more granular than "values" (which provides the complete state)
    and enables building adapters that react to specific state changes.

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
        raw_stream = astream_graph_with_updates(workflow, initial_state,
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
) -> AsyncIterator[tuple[BaseMessageChunk, dict[str, Any]]]:
    """
    Entry point for streaming a LangGraph workflow with messagess output.

    This function configures the graph to stream only messages, enabling
    downstream adapters to:
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


# define iterator type for API consumers, The tier_1_iterator returns tuples
# arising from calling langgraph's astream with stream_mode = "values" or
# "updates".
tier_1_iterator = AsyncIterator[tuple[str, Any]]


# Tier 2 iterator arise when calling astream with stream_mode = "messages".
# They do not contain state but provide info on the calling node in the
# metadata member of the tuple.
tier_2_iterator = AsyncIterator[
    tuple[BaseMessageChunk, dict[str, Any]]
]

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
    Multi-mode adapter that validates message content and modifies state.

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
                    f"Content check attempt {attempt + 1}/{max_retries + 1} "
                    f"failed: {e}"
                )

                if attempt == max_retries:
                    logger.error(
                        f"Content checker failed after {max_retries + 1} "
                        f"attempts: {e}"
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
                    # Validation failed - yield rejection and modified state
                    logger.info(
                        f"Content rejected with classification: {classification}"
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
                        "LLM exchange without content check (validation unavailable)"
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
            # Pass through non-message events or post-validation messages
            yield (mode, event)

    # Handle case where stream ended before buffer was full
    if not validation_complete and buffer_text:
        is_valid, classification = await _validate_content(
            query_text + "\n\n" + buffer_text
        )

        if not is_valid:
            logger.info(
                f"Content rejected with classification: {classification}"
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
                f"LLM exchange without content check: {buffer_text[:100]}..."
            )

        # Yield all buffered chunks
        for buffered_mode, buffered_event in buffer_chunks:
            yield (buffered_mode, buffered_event)


async def terminal_demux_adapter(
    multi_mode_stream: tier_1_iterator,
    *,
    on_terminal_state: Callable[[StateT], Any] | None = None,
) -> tier_2_iterator:
    """
    Terminal adapter: de-multiplexes multi-mode stream to messages only.

    This is a convenience wrapper that uses the base Mapping type for the
    callback. For type-safe usage with specific state types (like ChatState),
    use create_terminal_demux_adapter() instead.

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
    Terminal adapter: de-multiplexes multi-mode stream to messages only.

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


async def field_change_adapter(
    multi_mode_stream: tier_1_iterator,
    *,
    on_field_change: (
        dict[str, Callable[[Any], Awaitable[None]]] | None
    ) = None,
) -> tier_1_iterator:
    """
    Adapter that reacts to specific field changes via "updates" events.

    This adapter uses the "updates" stream mode to detect when specific state
    fields change, and invokes registered callbacks. It passes through all
    stream events unchanged.

    This enables building reactive adapters that respond to specific state
    changes, such as:
    - React when "status" changes to "valid"
    - React when "context" becomes available
    - React when "query_classification" is set

    Args:
        multi_mode_stream: Source stream with (mode, event) tuples
        on_field_change: Dict mapping field names to async callbacks.
            Each callback receives the new field value as its argument.
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
                            except Exception:
                                # Log the error but don't interrupt the stream
                                # Implement appropriate error handling as needed
                                pass

        # Always yield the event unchanged
        yield (mode, event)


# ============================================================================
# Tier 2 Adapters - Message-Only (Existing Adapters)
# ============================================================================


async def validation_stream_adapter(
    stream: AsyncIterator[BaseMessageChunk],
    query_text: str,
    *,
    validator_model: RunnableType,
    allowed_content: list[str],
    buffer_size: int = 320,
    error_message: str = "Content not allowed",
    max_retries: int = 2,
    logger: LoggerBase = ConsoleLogger(),
) -> AsyncIterator[BaseMessageChunk]:
    """
    Wraps an LLM stream with content validation.

    This adapter buffers initial chunks from the response stream, validates
    the buffered content using a separate LLM call, and then either:
    - Yields the buffered chunks + remaining stream if validation passes
    - Yields an error message and stops if validation fails

    The validation LLM call is independent of the main response graph,
    as discussed in the design.

    Args:
        stream: The LLM response stream to wrap
        query_text: The original user query (included in validation context)
        validator_model: Runnable for content classification
        allowed_content: List of allowed content categories
        buffer_size: Number of characters to buffer before validation
        error_message: Message to yield if validation fails
        max_retries: Number of retry attempts for validation LLM call
        logger: Logger for warnings and errors

    Yields:
        BaseMessageChunk objects - either the original stream or error

    Example:
        ```python
        stream = llm.astream(messages)
        validated_stream = validation_stream_adapter(
            stream,
            query_text,
            validator_model=validator,
            allowed_content=["statistics", "mathematics"],
        )
        async for chunk in validated_stream:
            print(chunk.text, end="")
        ```
    """
    buffer_chunks: list[BaseMessageChunk] = []
    buffer_text: str = ""
    validation_complete: bool = False

    async def _validate_content(content: str) -> tuple[bool, str]:
        """
        Validate content with retry logic.
        Returns (is_valid, classification_result).
        """
        for attempt in range(max_retries + 1):
            try:
                classification: str = await validator_model.ainvoke(
                    {"text": content}
                )
                logger.info(
                    "Model content classification: "
                    + classification.replace("\n", " ")
                )

                # Check against allowed content plus common acceptable responses
                # if check_allowed_content(
                #     classification,
                #     allowed_content
                #     + ["apology", "human interaction"],
                # ):
                if classification in allowed_content + [
                    "apology",
                    "human interaction",
                ]:
                    return True, classification
                else:
                    return False, classification

            except Exception as e:
                logger.warning(
                    f"Content check attempt {attempt + 1}/{max_retries + 1} "
                    f"failed: {e}"
                )

                if attempt == max_retries:
                    # All retries exhausted - fail-open strategy
                    logger.error(
                        f"Content checker failed after {max_retries + 1} "
                        f"attempts: {e}"
                    )
                    return True, "validation_unavailable"

                # Exponential backoff before retry
                await asyncio.sleep(0.5 * (attempt + 1))

        # Should never reach here
        return True, "validation_error"

    async for chunk in stream:
        if not validation_complete:
            # Buffering phase
            buffer_chunks.append(chunk)
            buffer_text += chunk.text

            if len(buffer_text) >= buffer_size:
                # Validate buffered content
                is_valid, classification = await _validate_content(
                    query_text + "\n\n" + buffer_text + "..."
                )

                if not is_valid:
                    logger.info(
                        f"Content rejected with classification: {classification}"
                    )
                    yield AIMessageChunk(content=error_message)
                    return  # Stop streaming

                # Validation passed - yield buffered chunks
                validation_complete = True
                if classification == "validation_unavailable":
                    logger.warning(
                        "LLM exchange without content check (validation unavailable)"
                    )

                for buffered in buffer_chunks:
                    yield buffered
        else:
            # Streaming phase - validation already passed
            yield chunk

    # Handle case where stream ended before buffer was full
    if not validation_complete:
        if buffer_text:  # Only validate if there's content
            is_valid, classification = await _validate_content(
                query_text + "\n\n" + buffer_text
            )

            if not is_valid:
                logger.info(
                    f"Content rejected with classification: {classification}"
                )
                yield AIMessageChunk(content=error_message)
                return

            if classification == "validation_unavailable":
                logger.warning(
                    f"LLM exchange without content check: {buffer_text[:100]}..."
                )

        # Yield all buffered chunks
        for buffered in buffer_chunks:
            yield buffered


async def logging_stream_adapter(
    stream: AsyncIterator[BaseMessageChunk],
    *,
    on_complete: Callable[[str], Awaitable[None]] | None = None,
    on_chunk: Callable[[str, str], Awaitable[None]] | None = None,
) -> AsyncIterator[BaseMessageChunk]:
    """
    Wraps a stream to enable logging callbacks.

    This adapter passes through all chunks unchanged while optionally
    invoking callbacks for logging or monitoring purposes.

    Args:
        stream: The stream to wrap
        on_complete: Async callback invoked with complete response text
        on_chunk: Async callback invoked with (chunk_text, buffer_so_far)

    Yields:
        All chunks from the original stream unchanged
    """
    buffer: str = ""

    async for chunk in stream:
        chunk_text = chunk.text
        buffer += chunk_text

        if on_chunk:
            await on_chunk(chunk_text, buffer)

        yield chunk

    if on_complete:
        await on_complete(buffer)


def compose_adapters(
    *adapters: Callable[
        [AsyncIterator[BaseMessageChunk]],
        AsyncIterator[BaseMessageChunk],
    ],
) -> Callable[
    [AsyncIterator[BaseMessageChunk]], AsyncIterator[BaseMessageChunk]
]:
    """
    Compose multiple stream adapters into a single adapter.

    Adapters are applied left-to-right (first adapter wraps the stream,
    second adapter wraps the result, etc.).

    Args:
        *adapters: Stream adapter functions to compose

    Returns:
        A single adapter that applies all adapters in sequence

    Example:
        ```python
        composed = compose_adapters(
            lambda s: validation_stream_adapter(s, query, ...),
            lambda s: logging_stream_adapter(s, on_complete=log_fn),
        )
        result_stream = composed(llm_stream)
        ```
    """

    def composed(
        stream: AsyncIterator[BaseMessageChunk],
    ) -> AsyncIterator[BaseMessageChunk]:
        result: AsyncIterator[BaseMessageChunk] = stream
        for adapter in adapters:
            result = adapter(result)
        return result

    return composed
