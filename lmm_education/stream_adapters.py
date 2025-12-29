"""
Stream adapters for processing LLM response streams.

This module provides composable stream adapters that can wrap and transform
async iterators of BaseMessageChunk objects. The primary use case is
content validation, which buffers initial response chunks, validates them
using a separate LLM, and either passes through or rejects the stream.
"""

import asyncio
from collections.abc import AsyncIterator, Callable, Awaitable

from langchain_core.messages import BaseMessageChunk, AIMessageChunk

from lmm.utils.logging import LoggerBase, ConsoleLogger
from lmm.language_models.langchain.runnables import RunnableType


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
