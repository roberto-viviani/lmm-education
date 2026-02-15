"""
Stream adapter for validating a chat_graph stream.
This module contains domain-specific adapters - it uses ChatState
directly and understands the semantics of the "status" field.
"""

# rev c 1.25

import asyncio
from typing import Any, Literal, TypedDict

from langchain_core.messages import AIMessageChunk

from lmm.utils.logging import LoggerBase, ConsoleLogger
from lmm.models.langchain.runnables import RunnableType

from .stream_adapters import tier_1_iterator

# Import ChatState for domain-specific adapters
from .base import ChatState, create_initial_state


class EmptyState(TypedDict):
    pass


ValidationStatus = Literal["valid", "invalid", "unavailable", "error"]


async def stateful_validation_adapter(
    multi_mode_stream: tier_1_iterator,
    *,
    validator_model: RunnableType,
    allowed_content: list[str],
    source_nodes: list[str] | None = None,
    buffer_size: int = 320,
    error_message: str = "Content not allowed",
    max_retries: int = 2,
    continue_on_fail: bool = False,
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
    - If validation is unavailable (network/service failure):
      * Behavior depends on continue_on_fail parameter
      * If continue_on_fail=True: allows content through with warning (fail-open)
      * If continue_on_fail=False: rejects content for safety (fail-closed)

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
        max_retries: Validation retry attempts before marking unavailable
        continue_on_fail: If True, content is allowed when validation fails
            (fail-open). If False, content is rejected when validation fails
            (fail-closed). Defaults to False for safety.
        logger: Logger for diagnostics

    Yields:
        (mode, event) tuples with potentially modified state

    Note:
        This adapter works with streams that emit 'messages' events only,
        or combined 'messages' + 'values' events. If no 'values' events are
        emitted, the adapter will yield a final state containing validation
        metadata merged with an empty initial state.
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
        status: ValidationStatus
        classification: str
        reason: str | None

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
                        status="valid",
                        classification=classification,
                        reason=None,
                    )
                else:
                    return ValidationResult(
                        status="invalid",
                        classification=classification,
                        reason=f"Classification '{classification}' not in allowed list",
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
                        status="unavailable",
                        classification="",
                        reason=str(e),
                    )

                await asyncio.sleep(0.5 * (attempt + 1))

        return ValidationResult(
            status="error",
            classification="",
            reason="Unexpected: exhausted retries without returning",
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
                        captured_state['query']
                        if captured_state
                        else ""
                    )
                    validation_result = await _validate_content(
                        query + "\n\n" + buffer_text + "..."
                    )
                    validation_complete = True

                    match validation_result["status"]:
                        case "valid":
                            # Validation passed - release buffer
                            modified_state = {
                                "query_classification": validation_result[
                                    "classification"
                                ],
                            }
                            # Yield buffered content
                            for (
                                buffered_mode,
                                buffered_event,
                            ) in buffer_chunks:
                                yield (buffered_mode, buffered_event)

                        case "invalid":
                            # Validation failed - reject and stop
                            logger.info(
                                f"Content rejected with classification: "
                                f"{validation_result['classification']}"
                            )

                            # Yield rejection message
                            yield (
                                "messages",
                                (
                                    AIMessageChunk(
                                        content=error_message
                                    ),
                                    metadata,
                                ),
                            )

                            # Save changed state values to reflect rejection
                            modified_state = {
                                "model_identification": validator_model.get_name()
                                or "unknown validator",
                                "response": buffer_text,
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

                        case "unavailable" | "error":
                            # Validator failed
                            if continue_on_fail:
                                # Fail-open: allow content through
                                logger.warning(
                                    f"Validation {validation_result['status']}: "
                                    f"{validation_result['reason']}. "
                                    f"Continuing without validation (continue_on_fail=True)"
                                )
                                modified_state = {
                                    "model_identification": validator_model.get_name()
                                    + (" (unavailable)"),
                                    "query_classification": f"<validation {validation_result['status']}>",
                                }
                                # Yield buffered content
                                for (
                                    buffered_mode,
                                    buffered_event,
                                ) in buffer_chunks:
                                    yield (
                                        buffered_mode,
                                        buffered_event,
                                    )
                            else:
                                # Fail-closed: reject for safety
                                logger.error(
                                    f"Validation {validation_result['status']}: "
                                    f"{validation_result['reason']}. "
                                    f"Rejecting content (continue_on_fail=False)"
                                )
                                # Yield rejection message
                                yield (
                                    "messages",
                                    (
                                        AIMessageChunk(
                                            content=error_message
                                        ),
                                        metadata,
                                    ),
                                )
                                # Save changed state
                                modified_state = {
                                    "model_identification": validator_model.get_name()
                                    + " (failure)",
                                    "response": buffer_text,
                                    "query_classification": f"<validation {validation_result['status']}>",
                                    "status": "rejected",
                                }
                                yield (
                                    "values",
                                    {
                                        **captured_state,
                                        **modified_state,
                                    },
                                )
                                return  # Stop streaming

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

        match validation_result["status"]:
            case "valid":
                # Validation passed
                modified_state = {
                    "query_classification": validation_result[
                        "classification"
                    ],
                }
                # Yield buffered chunks
                for buffered_mode, buffered_event in buffer_chunks:
                    yield (buffered_mode, buffered_event)

            case "invalid":
                # Validation failed
                logger.info(
                    f"Content rejected with classification: "
                    f"{validation_result['classification']}"
                )
                # the metadata will be the last captured
                if not metadata:
                    # this is unreachable because buffer_text was tested
                    # as not empty, and buffer_text can fill only if chunks
                    # are read in with the metadata
                    raise ValueError(
                        "Unreachable code reached: empty metadata "
                        "in stateful_validation_adapter"
                    )
                yield (
                    "messages",
                    (AIMessageChunk(content=error_message), metadata),
                )

                modified_state = {
                    "model_identification": validator_model.get_name(),
                    "status": "rejected",
                    "response": buffer_text,
                    "query_classification": validation_result[
                        "classification"
                    ],
                }
                yield ("values", {**captured_state, **modified_state})
                return

            case "unavailable" | "error":
                # Validator failed
                if continue_on_fail:
                    # Fail-open
                    logger.warning(
                        f"Validation {validation_result['status']}: "
                        f"{validation_result['reason']}. "
                        f"Continuing without validation (continue_on_fail=True)"
                    )
                    modified_state = {
                        "model_identification": validator_model.get_name()
                        + " (unavailable)",
                        "query_classification": f"<validation {validation_result['status']}>",
                    }
                    # Yield buffered chunks
                    for (
                        buffered_mode,
                        buffered_event,
                    ) in buffer_chunks:
                        yield (buffered_mode, buffered_event)
                else:
                    # Fail-closed
                    logger.error(
                        f"Validation {validation_result['status']}: "
                        f"{validation_result['reason']}. "
                        f"Rejecting content (continue_on_fail=False)"
                    )
                    if not metadata:
                        raise ValueError(
                            "Unreachable code reached: empty metadata "
                            "in stateful_validation_adapter"
                        )
                    yield (
                        "messages",
                        (
                            AIMessageChunk(content=error_message),
                            metadata,
                        ),
                    )
                    modified_state = {
                        "model_identification": validator_model.get_name()
                        + (" (failure)"),
                        "status": "rejected",
                        "response": buffer_text,
                        "query_classification": f"<validation {validation_result['status']}>",
                    }
                    yield (
                        "values",
                        {**captured_state, **modified_state},
                    )
                    return

    # Yield final state to override previous "values"
    if modified_state:
        yield ("values", {**captured_state, **modified_state})
    else:
        logger.warning("validation adapter: modified state not set")
