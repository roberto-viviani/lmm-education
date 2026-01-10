"""
Stream adapters for processing LangGraph response streams.

This module provides composable stream adapters that can wrap and transform
async iterators of BaseMessageChunk objects. The primary use case is
content validation, which buffers initial response chunks, validates them
using a separate LLM, and either passes through or rejects the stream.

Architecture:
- Generic components work with any LangGraph workflow
- Domain-specific components (e.g., validation) use ChatState
    explicitly
- Two-tier adapter system:
  * Tier 1: Multi-mode adapters (operate on (mode, event) tuples)
  * Tier 2: Message-only adapters (operate on BaseMessageChunk)

Tier 1 adapters work on graphs that are streamed via stream_mode =
["messages", "state"]. Tier 2 adapters work on streams from
stream_mode = "messages". A stream may be filtered from tier 1 to tier
2, but not in the opposite direction, as state is lost from the
stream. The `terminal_demux_adapter` converts a tier 1 to tier 2
stream, while also optionally calling a callback with the state at the
end of streaming.

The callback is given the final state as an argument.
"""

import asyncio
from typing import Protocol, TypeVar, Any
from collections.abc import (
    AsyncIterator,
    Callable,
    Awaitable,
    Mapping,
)

from pydantic import BaseModel
from langchain_core.messages import BaseMessageChunk, AIMessageChunk

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


class StreamableGraph(Protocol):
    """
    Minimal protocol for LangGraph compiled graphs.

    This protocol only specifies the astream() method that we actually use,
    avoiding tight coupling to LangGraph internals.
    """

    def astream(
        self,
        input: Mapping[str, Any] | None,
        *,
        context: BaseModel | None = None,
        stream_mode: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[tuple[str, Any]]:
        """Stream graph execution with multiple output modes."""
        ...


# ============================================================================
# Entry Point - Generic
# ============================================================================


def stream_graph(
    graph: StreamableGraph,
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
        AsyncIterator yielding (mode, event) tuples where:
        - mode is "messages" or "values"
        - event is (BaseMessageChunk, metadata) for messages
        - event is StateT for values

    Example:
        ```python
        raw_stream = stream_graph(workflow, initial_state, context)

        # Apply adapters
        validated = stateful_validation_adapter(raw_stream, ...)
        messages = terminal_demux_adapter(validated, on_terminal_state=log_fn)

        # Consume for display
        async for chunk in messages:
            print(chunk.text, end="")
        ```
    """
    return graph.astream(
        initial_state,
        stream_mode=["messages", "values"],
        context=context,
    )


# ============================================================================
# Tier 1 Adapters - Multi-Mode (State-Aware)
# ============================================================================


async def stateful_validation_adapter(
    multi_mode_stream: AsyncIterator[tuple[str, Any]],
    query_text: str,
    *,
    validator_model: RunnableType,
    allowed_content: list[str],
    buffer_size: int = 320,
    error_message: str = "Content not allowed",
    max_retries: int = 2,
    logger: LoggerBase = ConsoleLogger(),
) -> AsyncIterator[tuple[str, Any]]:
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
    multi_mode_stream: AsyncIterator[tuple[str, Any]],
    *,
    on_terminal_state: Callable[[StateT], None] | None = None,
) -> AsyncIterator[BaseMessageChunk]:
    """
    Terminal adapter: de-multiplexes multi-mode stream to messages only.

    This is the final adapter in the Tier 1 chain. It:
    - Extracts and yields only message chunks (for display/Gradio)
    - Captures the terminal state (last "values" event)
    - Invokes an optional callback with the terminal state

    This adapter is generic - it works with any state type.

    Args:
        multi_mode_stream: Source stream with (mode, event) tuples
        on_terminal_state: Optional callback for terminal state
            (e.g., for logging). The function takes one StateT
            argument, which will contain the terminal state.

    Yields:
        BaseMessageChunk objects

    Example:
        ```python
        messages = terminal_demux_adapter(
            multi_mode_stream,
            on_terminal_state=lambda s: logger.log(s, "MESSAGE"),
        )

        async for chunk in messages:
            yield chunk.text  # To Gradio
        ```
    """
    final_state: StateT | None = None

    async for mode, event in multi_mode_stream:
        if mode == "messages":
            chunk, _ = event  # Extract chunk, ignore metadata
            yield chunk
        elif mode == "values":
            final_state = event

    # After stream exhaustion, invoke callback if provided
    if final_state is not None and on_terminal_state is not None:
        on_terminal_state(final_state)


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
