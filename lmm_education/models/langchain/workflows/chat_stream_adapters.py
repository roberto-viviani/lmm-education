"""
Stream adapter for validating a chat_graph stream.
This module contains domain-specific adapters - it uses ChatState
directly and understands the semantics of the "status" field.
"""

import asyncio
from typing import Any, TypedDict

from langchain_core.messages import AIMessageChunk

from lmm.utils.logging import LoggerBase, ConsoleLogger
from lmm.language_models.langchain.runnables import RunnableType

from ..stream_adapters import tier_1_iterator

# Import ChatState for domain-specific adapters
from .chat_graph import ChatState, create_initial_state


class EmptyState(TypedDict):
    pass


async def stateful_validation_adapter(
    multi_mode_stream: tier_1_iterator,
    *,
    validator_model: RunnableType,
    allowed_content: list[str],
    source_nodes: list[str] | None = None,
    buffer_size: int = 320,
    error_message: str = "Content not allowed",
    max_retries: int = 2,
    logger: LoggerBase = ConsoleLogger(),
) -> tier_1_iterator:
    """
    Multi-mode adapter that validates message content.

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

    If validation is not available (for example, network failure or
    server error), the adapter will log a diagnostic message to the
    logger and continue streaming normally.

    Args:
        multi_mode_stream: Source stream yielding (mode, event) tuples
        validator_model: Runnable for content classification
        allowed_content: List of allowed content categories
        source_nodes: List of graph node names whose messages should be
            validated. If None, all message streams are validated. Use
            this to exclude error messages from nodes like 'validate_query'
            and only validate LLM-generated content from nodes like 'generate'.
        buffer_size: Characters to buffer before validation
        error_message: Message to yield if validation fails
        max_retries: Validation retry attempts
        logger: Logger for diagnostics

    Yields:
        (mode, event) tuples with potentially modified state

    Note:
        This adapter may be used only on graphs that use the ChatState
        state, or of derived types. The graph must emit a 'values'
        chunk before streaming from the language model or in the
        'messages' stream.
    """
    buffer_chunks: list[tuple[str, Any]] = []
    buffer_text: str = ""
    validation_complete: bool = False
    # captured in 'message' stream
    metadata: dict[str, Any] = {}
    # captured in 'values' stream (state from graph)
    captured_state: ChatState = create_initial_state("")
    # state values changes by validation
    modified_state: dict[str, str] = {}

    class ValidationResult(TypedDict):
        is_valid: bool
        classification: str

    async def _validate_content(content: str) -> ValidationResult:
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
                    return ValidationResult(
                        is_valid=True, classification=classification
                    )
                else:
                    return ValidationResult(
                        is_valid=False, classification=classification
                    )

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
                    return ValidationResult(
                        is_valid=False,
                        classification="validation_unavailable",
                    )

                await asyncio.sleep(0.5 * (attempt + 1))

        return ValidationResult(
            is_valid=False, classification="validation_error"
        )

    async for mode, event in multi_mode_stream:
        match mode:
            case "values":
                # Track latest state for potential modification
                captured_state = event
                yield (mode, event)

            case "messages" if not validation_complete:
                # Validation logic for streamed data in the messages mode
                chunk, metadata = event

                # Filter by source node if specified
                if source_nodes is not None:
                    node_name = metadata.get("langgraph_node", "")
                    if node_name not in source_nodes:
                        # Pass through without validation
                        yield (mode, event)
                        continue

                buffer_chunks.append((mode, event))
                buffer_text += chunk.text

                if len(buffer_text) >= buffer_size:
                    # Validate buffered content
                    query: str = (
                        captured_state['query'] if captured_state else ""
                    )
                    validation_result = await _validate_content(
                        query + "\n\n" + buffer_text + "..."
                    )
                    validation_complete = True

                    if not validation_result["is_valid"]:
                        # Validation failed - yield rejection and
                        # modified state
                        logger.info(
                            f"Content rejected with classification: "
                            f"{validation_result['classification']}"
                        )

                        # Yield rejection message
                        yield (
                            "messages",
                            (
                                AIMessageChunk(content=error_message),
                                metadata,
                            ),
                        )

                        # Save change state values to reflect rejection
                        modified_state = {
                            "response": error_message,
                            "query_classification": validation_result[
                                "classification"
                            ],
                            "status": "rejected",
                        }
                        yield (
                            "values",
                            {**captured_state, **modified_state},
                        )

                        return  # Stop streaming

                    # Validation passed - release buffer
                    if (
                        validation_result["classification"]
                        == "validation_unavailable"
                    ):
                        logger.warning(
                            "LLM exchange without content check "
                            "(validation unavailable)"
                        )
                        modified_state = {
                            "query_classification": "<validation unavailable>",
                        }
                    else:
                        # record classification
                        modified_state = {
                            "query_classification": validation_result[
                                "classification"
                            ],
                        }

                    # Yield buffered stuff
                    for buffered_mode, buffered_event in buffer_chunks:
                        yield (buffered_mode, buffered_event)

            case _:
                # Pass through non-message events or post-validation messages
                yield (mode, event)
    # end async for mode, event

    # Handle case where stream ended before buffer was full
    if not validation_complete and buffer_text:
        query_text: str = (
            captured_state['query'] if captured_state else ""
        )
        validation_result = await _validate_content(
            query_text + "\n\n" + buffer_text
        )

        if not validation_result["is_valid"]:
            logger.info(
                f"Content rejected with classification: "
                f"{validation_result['classification']}"
            )
            # the metadata will be the last captured
            if not metadata:
                # this is unrecheable because buffer_text was tested
                # as not empty, and buffer_text can fill only if chunks
                # are read in with the metadata
                raise ValueError(
                    "Unreacheable code reached: empty metadata "
                    "in stateful_validation_adapter"
                )
            yield (
                "messages",
                (AIMessageChunk(content=error_message), metadata),
            )

            modified_state = {
                "status": "rejected",
                "response": error_message,
                "query_classification": validation_result["classification"],
            }
            yield ("values", {**captured_state, **modified_state})
            return

        else:
            modified_state = {
                "query_classification": validation_result["classification"],
            }

        if validation_result["classification"] == "validation_unavailable":
            logger.warning(
                f"LLM exchange without content check: "
                f"{buffer_text[:100]}..."
            )

        # Yield all buffered chunks
        for buffered_mode, buffered_event in buffer_chunks:
            yield (buffered_mode, buffered_event)

    # Yield final state to override previous "values"
    if modified_state:
        yield ("values", {**captured_state, **modified_state})
    else:
        logger.warning("validation adapter: modified state not set")
