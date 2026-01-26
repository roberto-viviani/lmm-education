"""
Stream adapter for validating a chat_graph stream.
This module contains domain-specific adapters - it uses ChatState
directly and understands the semantics of the "status" field.
"""

import asyncio
from typing import Any

from langchain_core.messages import AIMessageChunk

from lmm.utils.logging import LoggerBase, ConsoleLogger
from lmm.language_models.langchain.runnables import RunnableType

from ..stream_adapters import tier_1_iterator

# Import ChatState for domain-specific adapters
from .chat_graph import ChatState


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

    Args:
        multi_mode_stream: Source stream yielding (mode, event) tuples
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
    captured_state: ChatState | None = None  # state from graph
    modified_state: ChatState | None = None  # state to yield
    # by default, we set the metadata to the 'generate' node, which
    # is the node that is streaming the content, so that when
    # source_node is set, it will lead to streaming all this content.
    metadata: dict[str, Any] = {'langgraph_node': "generate"}

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
            # The 'response' field update will arrive after the
            # llm output has finished streaming. Copy it into
            # the captured_state modified by streaming.
            if (
                modified_state
                and captured_state
                and captured_state['response']
            ):
                modified_state['response'] = captured_state[
                    'response'
                ]
            yield (mode, event)

        # Validation logic only applies to messages
        if mode == "messages" and not validation_complete:
            chunk, metadata = event
            buffer_chunks.append((mode, event))
            buffer_text += chunk.text

            if len(buffer_text) >= buffer_size:
                # Validate buffered content
                query: str = (
                    captured_state['query'] if captured_state else ""
                )
                is_valid, classification = await _validate_content(
                    query + "\n\n" + buffer_text + "..."
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
                        (
                            AIMessageChunk(content=error_message),
                            metadata,
                        ),
                    )

                    # Modify state to reflect rejection
                    if captured_state:
                        modified_state = {
                            **captured_state,
                            "response": error_message,
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
                        modified_state = {
                            **captured_state,
                            "query_classification": "NA",
                        }
                        # do not yield state here, wait to end
                        # updates not really valid (not in full state)
                        # yield ("updates", {"query_classification": "NA"})
                else:
                    # record classification
                    if captured_state:
                        modified_state = {
                            **captured_state,
                            "query_classification": classification,
                        }
                        # do not yield state here, wait to end
                        # updates not really valid (not in full state)
                        # yield ("updates", {"query_classification":
                        #                       classification})

                for buffered_mode, buffered_event in buffer_chunks:
                    yield (buffered_mode, buffered_event)

        else:  # if mode == "messages" and ...
            # Pass through non-message events or post-validation
            # messages
            yield (mode, event)
    # end async for mode, event

    # Handle case where stream ended before buffer was full
    if not validation_complete and buffer_text:
        query_text: str = (
            captured_state['query'] if captured_state else ""
        )
        is_valid, classification = await _validate_content(
            query_text + "\n\n" + buffer_text
        )

        if not is_valid:
            logger.info(
                f"Content rejected with classification: "
                f"{classification}"
            )
            # the metadata will be the last captured, or a default
            # from the "generate" node to allow streaming
            yield (
                "messages",
                (AIMessageChunk(content=error_message), metadata),
            )

            if captured_state:
                modified_state = {
                    **captured_state,
                    "status": "rejected",
                    "response": error_message,
                    "query_classification": classification,
                }
                yield ("values", modified_state)
            return

        else:
            if captured_state:
                modified_state = {
                    **captured_state,
                    "query_classification": classification,
                }

        if classification == "validation_unavailable":
            logger.warning(
                f"LLM exchange without content check: "
                f"{buffer_text[:100]}..."
            )

        # Yield all buffered chunks
        for buffered_mode, buffered_event in buffer_chunks:
            yield (buffered_mode, buffered_event)

    # Yield final state to override previous "values"
    if modified_state:
        yield ("values", modified_state)
    else:
        logger.warning("validation adapter: modified state not set")
