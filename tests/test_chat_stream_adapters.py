"""
Comprehensive unit tests for chat_stream_adapters module.

Tests the stateful_validation_adapter which validates streaming LLM
responses and can reject content mid-stream.
"""

# pyright: reportUnknownParameterType=false
# pyright: reportMissingParameterType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false

# pyright: reportArgumentType=false
# pyright: reportOptionalSubscript=false
# pyright: reportUnusedVariable=false

import unittest

from typing import Any

from langchain_core.messages import AIMessageChunk
from lmm.utils.logging import LoglistLogger

from lmm_education.models.langchain.workflows.chat_stream_adapters import (
    stateful_validation_adapter,
)
from lmm_education.models.langchain.workflows.chat_graph import (
    ChatState,
)
from lmm_education.models.langchain.stream_adapters import (
    tier_1_iterator,
)


# ================================================================
# Mock Stream Generators
# ================================================================


async def mock_chat_stream(
    messages: list[str],
    node_name: str = "generate",
    query: str = "Test query",
    include_values: bool = True,
) -> tier_1_iterator:
    """
    Generate a mock tier 1 chat stream with ChatState values events.

    Args:
        messages: List of message text chunks to stream
        node_name: Name of the node emitting messages
        query: Query text to include in state
        include_values: Whether to include values events

    Yields:
        (mode, event) tuples simulating a chat workflow stream
    """
    # Yield initial state
    if include_values:
        initial_state: ChatState = {
            "messages": [],
            "status": "valid",
            "query": query,
            "query_prompt": "",
            "query_classification": "",
            "context": "",
            "response": "",
        }
        yield ("values", initial_state)

    # Yield message chunks
    for text in messages:
        chunk = AIMessageChunk(content=text)
        metadata = {"langgraph_node": node_name}
        yield ("messages", (chunk, metadata))

    # Yield final state
    if include_values:
        final_state: ChatState = {
            "messages": [],
            "status": "valid",
            "query": query,
            "query_prompt": "",
            "query_classification": "",
            "context": "",
            "response": "".join(messages),
        }
        yield ("values", final_state)


async def mock_multi_node_stream(
    node_messages: dict[str, list[str]],
    query: str = "Test query",
) -> tier_1_iterator:
    """
    Generate a stream with messages from multiple nodes.

    Args:
        node_messages: Dict mapping node names to their message chunks
        query: Query text for state

    Yields:
        (mode, event) tuples with messages from different nodes
    """
    # Initial state
    initial_state: ChatState = {
        "messages": [],
        "status": "valid",
        "query": query,
        "query_prompt": "",
        "query_classification": "",
        "context": "",
        "response": "",
    }
    yield ("values", initial_state)

    # Yield messages from each node
    for node_name, messages in node_messages.items():
        for text in messages:
            chunk = AIMessageChunk(content=text)
            metadata = {"langgraph_node": node_name}
            yield ("messages", (chunk, metadata))

    # Final state
    all_text = "".join(
        text
        for messages in node_messages.values()
        for text in messages
    )
    final_state: ChatState = {
        **initial_state,
        "response": all_text,
    }
    yield ("values", final_state)


# ================================================================
# Mock Validator
# ================================================================


class MockValidator:
    """Mock validator for testing."""

    def __init__(
        self,
        classification: str = "statistics",
        should_fail: bool = False,
        fail_count: int = 0,
    ):
        """
        Initialize mock validator.

        Args:
            classification: Classification to return
            should_fail: If True, always raise exceptions
            fail_count: Number of times to fail before succeeding
        """
        self.classification = classification
        self.should_fail = should_fail
        self.fail_count = fail_count
        self.call_count = 0
        self.invocations: list[dict[str, str]] = []

    async def ainvoke(self, input_dict: dict[str, str]) -> str:
        """Mock ainvoke method."""
        self.call_count += 1
        self.invocations.append(input_dict)

        # Fail if configured
        if self.should_fail:
            raise ValueError("Mock validator failure")

        # Fail for first N calls
        if self.call_count <= self.fail_count:
            raise ValueError(f"Temporary failure {self.call_count}")

        return self.classification


# ================================================================
# Helper Functions
# ================================================================


async def consume_stream(stream) -> list[tuple[str, Any]]:
    """Consume an async iterator and return all items as a list."""
    items = []
    async for item in stream:
        items.append(item)
    return items


def get_messages_from_events(
    events: list[tuple[str, Any]],
) -> list[str]:
    """Extract message text from stream events."""
    messages = []
    for mode, event in events:
        if mode == "messages":
            chunk, _ = event
            messages.append(chunk.content)
    return messages


def get_final_state(
    events: list[tuple[str, Any]],
) -> ChatState | None:
    """Get the last state from stream events."""
    for mode, event in reversed(events):
        if mode == "values":
            return event
    return None


# ================================================================
# Test Classes
# ================================================================


class TestValidationFlow(unittest.IsolatedAsyncioTestCase):
    """Tests for core validation pass/fail flow."""

    async def test_validation_passes_releases_buffer(self):
        """Test that buffered messages are released when validation passes."""
        # Create stream with messages that will pass validation
        source = mock_chat_stream(
            messages=["Hello", " ", "world", "!"],
            query="What is statistics?",
        )

        # Validator that approves
        validator = MockValidator(classification="statistics")

        adapted = stateful_validation_adapter(
            source,
            validator_model=validator,
            allowed_content=["statistics"],
            buffer_size=10,  # Low threshold to trigger quickly
        )

        events = await consume_stream(adapted)
        messages = get_messages_from_events(events)
        final_state = get_final_state(events)

        # All messages should be released
        self.assertEqual(messages, ["Hello", " ", "world", "!"])

        # Final state should have classification
        self.assertIsNotNone(final_state)
        self.assertEqual(
            final_state["query_classification"], "statistics"
        )

        # Validator should have been called
        self.assertEqual(validator.call_count, 1)

    async def test_validation_fails_stops_stream(self):
        """Test that stream stops and yields rejection when validation fails."""
        # Create stream
        source = mock_chat_stream(
            messages=["The", " answer", " is", " 42"],
            query="Tell me about physics",
        )

        # Validator that rejects
        validator = MockValidator(classification="physics")

        adapted = stateful_validation_adapter(
            source,
            validator_model=validator,
            allowed_content=["statistics"],
            buffer_size=10,
            error_message="Content not allowed",
        )

        events = await consume_stream(adapted)
        messages = get_messages_from_events(events)
        final_state = get_final_state(events)

        # Should get rejection message, NOT buffered messages
        self.assertEqual(messages, ["Content not allowed"])

        # Final state should show rejection
        self.assertIsNotNone(final_state)
        self.assertEqual(final_state["status"], "rejected")
        self.assertEqual(
            final_state["response"], "Content not allowed"
        )
        self.assertEqual(
            final_state["query_classification"], "physics"
        )

    async def test_classification_in_allowed_list(self):
        """Test that validation passes for any allowed classification."""
        source = mock_chat_stream(
            messages=["Statistics", " content"],
            query="Test",
        )

        validator = MockValidator(classification="mathematics")

        adapted = stateful_validation_adapter(
            source,
            validator_model=validator,
            allowed_content=[
                "statistics",
                "mathematics",
                "data_science",
            ],
            buffer_size=5,
        )

        events = await consume_stream(adapted)
        final_state = get_final_state(events)

        # Should pass validation
        self.assertIsNotNone(final_state)
        self.assertEqual(final_state["status"], "valid")
        self.assertEqual(
            final_state["query_classification"], "mathematics"
        )


class TestSourceNodesFiltering(unittest.IsolatedAsyncioTestCase):
    """Tests for source_nodes filtering functionality."""

    async def test_source_nodes_validates_only_target(self):
        """Test that only messages from specified nodes are validated."""
        # Stream with messages from different nodes
        source = mock_multi_node_stream(
            node_messages={
                "validate_query": ["Query", " valid"],
                "generate": ["Statistics", " answer"],
            }
        )

        validator = MockValidator(classification="statistics")

        adapted = stateful_validation_adapter(
            source,
            validator_model=validator,
            allowed_content=["statistics"],
            source_nodes=["generate"],  # Only validate generate node
            buffer_size=5,
        )

        events = await consume_stream(adapted)
        messages = get_messages_from_events(events)

        # All messages should appear
        self.assertEqual(
            messages, ["Query", " valid", "Statistics", " answer"]
        )

        # Validator should only be called for 'generate' node messages
        self.assertEqual(validator.call_count, 1)
        # Check that validation was called with 'generate' node content
        self.assertIn("Statistics", validator.invocations[0]["text"])

    async def test_source_nodes_passes_through_others(self):
        """Test that messages from non-target nodes pass through immediately."""
        source = mock_multi_node_stream(
            node_messages={
                "validate_query": [
                    "Error:",
                    " query",
                    " too",
                    " long",
                ],
                "generate": ["OK"],
            }
        )

        validator = MockValidator(classification="statistics")
        logger = LoglistLogger()

        adapted = stateful_validation_adapter(
            source,
            validator_model=validator,
            allowed_content=["statistics"],
            source_nodes=["generate"],
            buffer_size=100,  # High threshold
            logger=logger,
        )

        events = await consume_stream(adapted)
        messages = get_messages_from_events(events)

        # validate_query messages should pass through without buffering
        # generate message should be buffered
        self.assertEqual(
            messages[:4],  # First 4 are from validate_query
            ["Error:", " query", " too", " long"],
        )

    async def test_source_nodes_none_validates_all(self):
        """Test that source_nodes=None validates all nodes."""
        source = mock_multi_node_stream(
            node_messages={
                "node1": ["Part", " one"],
                "node2": [" part", " two"],
            }
        )

        # Validator that rejects
        validator = MockValidator(classification="physics")

        adapted = stateful_validation_adapter(
            source,
            validator_model=validator,
            allowed_content=["statistics"],
            source_nodes=None,  # Validate all nodes
            buffer_size=5,
        )

        events = await consume_stream(adapted)
        final_state = get_final_state(events)

        # Should reject because all messages are validated
        self.assertIsNotNone(final_state)
        self.assertEqual(final_state["status"], "rejected")


class TestEdgeCases(unittest.IsolatedAsyncioTestCase):
    """Tests for edge cases and boundary conditions."""

    async def test_stream_ends_before_buffer_full(self):
        """Test that validation still occurs if stream ends before buffer fills."""
        # Short message stream
        source = mock_chat_stream(
            messages=["Hi"],
            query="Short query",
        )

        validator = MockValidator(classification="statistics")

        adapted = stateful_validation_adapter(
            source,
            validator_model=validator,
            allowed_content=["statistics"],
            buffer_size=100,  # Buffer won't fill
        )

        events = await consume_stream(adapted)
        messages = get_messages_from_events(events)
        final_state = get_final_state(events)

        # Message should still appear (validated at end)
        self.assertEqual(messages, ["Hi"])

        # Validator should have been called
        self.assertEqual(validator.call_count, 1)

        # Final state should have classification
        self.assertIsNotNone(final_state)
        self.assertEqual(
            final_state["query_classification"], "statistics"
        )

    async def test_stream_ends_before_buffer_full_rejection(self):
        """Test rejection when stream ends before buffer is full."""
        source = mock_chat_stream(
            messages=["Bad"],
            query="Test",
        )

        validator = MockValidator(classification="physics")

        adapted = stateful_validation_adapter(
            source,
            validator_model=validator,
            allowed_content=["statistics"],
            buffer_size=1000,
            error_message="Rejected",
        )

        events = await consume_stream(adapted)
        messages = get_messages_from_events(events)
        final_state = get_final_state(events)

        # Should see rejection message
        self.assertEqual(messages, ["Rejected"])

        # Should be rejected
        self.assertIsNotNone(final_state)
        self.assertEqual(final_state["status"], "rejected")

    async def test_empty_stream(self):
        """Test that adapter handles empty streams gracefully."""

        async def empty_stream() -> tier_1_iterator:
            return
            yield  # Make it a generator

        validator = MockValidator()

        adapted = stateful_validation_adapter(
            empty_stream(),
            validator_model=validator,
            allowed_content=["statistics"],
        )

        events = await consume_stream(adapted)

        # Should not crash, just return empty
        self.assertEqual(events, [])

    async def test_no_values_events(self):
        """Test handling of streams without state updates."""

        async def messages_only_stream() -> tier_1_iterator:
            for text in ["Hello", " world"]:
                chunk = AIMessageChunk(content=text)
                metadata = {"langgraph_node": "test"}
                yield ("messages", (chunk, metadata))

        validator = MockValidator(classification="statistics")
        logger = LoglistLogger()

        adapted = stateful_validation_adapter(
            messages_only_stream(),
            validator_model=validator,
            allowed_content=["statistics"],
            buffer_size=5,
            logger=logger,
        )

        events = await consume_stream(adapted)
        messages = get_messages_from_events(events)
        final_state = get_final_state(events)

        # Should work correctly without values events
        self.assertEqual(messages, ["Hello", " world"])

        # Should yield a final state with validation metadata
        self.assertIsNotNone(final_state)
        self.assertEqual(
            final_state["query_classification"], "statistics"
        )


class TestErrorHandling(unittest.IsolatedAsyncioTestCase):
    """Tests for error handling and fail-open behavior."""

    async def test_validator_exception_fail_open(self):
        """Test that content is allowed when validator fails with continue_on_fail=True."""
        source = mock_chat_stream(
            messages=["Test", " content"],
            query="Test",
        )

        # Validator that always fails
        validator = MockValidator(should_fail=True)
        logger = LoglistLogger()

        adapted = stateful_validation_adapter(
            source,
            validator_model=validator,
            allowed_content=["statistics"],
            buffer_size=5,
            max_retries=1,
            continue_on_fail=True,  # Fail-open behavior
            logger=logger,
        )

        events = await consume_stream(adapted)
        messages = get_messages_from_events(events)
        final_state = get_final_state(events)

        # Content should pass through (fail-open)
        self.assertEqual(messages, ["Test", " content"])

        # Should be marked as validation unavailable
        self.assertIsNotNone(final_state)
        self.assertEqual(
            final_state["query_classification"],
            "<validation unavailable>",
        )

        # Should have logged warning about continuing without validation
        logs = logger.get_logs()
        self.assertTrue(
            any(
                "Continuing without validation" in log for log in logs
            )
        )

    async def test_validator_retry_logic(self):
        """Test that retries work with exponential backoff."""
        source = mock_chat_stream(
            messages=["Retry", " test"],
            query="Test",
        )

        # Fail twice, then succeed
        validator = MockValidator(
            classification="statistics",
            fail_count=2,
        )
        logger = LoglistLogger()

        adapted = stateful_validation_adapter(
            source,
            validator_model=validator,
            allowed_content=["statistics"],
            buffer_size=5,
            max_retries=3,
            logger=logger,
        )

        events = await consume_stream(adapted)
        final_state = get_final_state(events)

        # Should eventually succeed
        self.assertIsNotNone(final_state)
        self.assertEqual(
            final_state["query_classification"], "statistics"
        )

        # Should have called validator 3 times
        self.assertEqual(validator.call_count, 3)

        # Should have logged warnings
        logs = logger.get_logs()
        warning_logs = [log for log in logs if "WARNING" in log]
        self.assertEqual(len(warning_logs), 2)  # 2 failures

    async def test_max_retries_respected(self):
        """Test that retries stop after max attempts."""
        source = mock_chat_stream(
            messages=["Test"],
            query="Test",
        )

        validator = MockValidator(should_fail=True)

        adapted = stateful_validation_adapter(
            source,
            validator_model=validator,
            allowed_content=["statistics"],
            buffer_size=2,
            max_retries=2,  # Fail 3 times total
        )

        events = await consume_stream(adapted)  # noqa

        # Should have tried 3 times (initial + 2 retries)
        self.assertEqual(validator.call_count, 3)

    async def test_validation_unavailable_marked_correctly(self):
        """Test fail-closed behavior when validation is unavailable."""
        source = mock_chat_stream(
            messages=["Content"],
            query="Test",
        )

        validator = MockValidator(should_fail=True)
        logger = LoglistLogger()

        adapted = stateful_validation_adapter(
            source,
            validator_model=validator,
            allowed_content=["statistics"],
            buffer_size=5,
            max_retries=0,
            continue_on_fail=False,  # Fail-closed (default)
            logger=logger,
        )

        events = await consume_stream(adapted)
        messages = get_messages_from_events(events)
        final_state = get_final_state(events)

        # With fail-closed, content should be rejected
        self.assertEqual(messages, ["Content not allowed"])

        # Should have special classification
        self.assertIsNotNone(final_state)
        self.assertEqual(final_state["status"], "rejected")
        self.assertEqual(
            final_state["query_classification"],
            "<validation unavailable>",
        )

        # Should log error about rejecting content
        logs = logger.get_logs()
        self.assertTrue(
            any(
                "Rejecting content" in log
                and "continue_on_fail=False" in log
                for log in logs
            )
        )


class TestStateSync(unittest.IsolatedAsyncioTestCase):
    """Tests for state synchronization."""

    async def test_final_state_merge(self):
        """Test that final state properly merges captured and modified state."""
        source = mock_chat_stream(
            messages=["Test"],
            query="Original query",
        )

        validator = MockValidator(classification="statistics")

        adapted = stateful_validation_adapter(
            source,
            validator_model=validator,
            allowed_content=["statistics"],
            buffer_size=2,
        )

        events = await consume_stream(adapted)
        final_state = get_final_state(events)

        # Final state should have both original fields and classification
        self.assertIsNotNone(final_state)
        self.assertEqual(final_state["query"], "Original query")
        self.assertEqual(
            final_state["query_classification"], "statistics"
        )
        self.assertEqual(final_state["response"], "Test")

    async def test_intermediate_values_passed_through(self):
        """Test that intermediate values events are yielded unchanged."""

        async def multi_state_stream() -> tier_1_iterator:
            # First state
            state1: ChatState = {
                "messages": [],
                "status": "valid",
                "query": "Q1",
                "query_prompt": "",
                "query_classification": "",
                "context": "",
                "response": "",
            }
            yield ("values", state1)

            # Message that won't trigger validation (too short)
            chunk = AIMessageChunk(content="Hi")
            metadata = {"langgraph_node": "generate"}
            yield ("messages", (chunk, metadata))

            # Second state
            state2: ChatState = {
                **state1,
                "context": "Some context",
            }
            yield ("values", state2)

        validator = MockValidator()

        adapted = stateful_validation_adapter(
            multi_state_stream(),
            validator_model=validator,
            allowed_content=["statistics"],
            buffer_size=1000,  # Won't trigger
        )

        events = await consume_stream(adapted)

        # Should have two original values events plus final merged one
        values_events = [e for mode, e in events if mode == "values"]
        self.assertGreaterEqual(len(values_events), 2)


class TestBufferBehavior(unittest.IsolatedAsyncioTestCase):
    """Tests for message buffering logic."""

    async def test_buffer_size_threshold(self):
        """Test that validation triggers exactly at buffer_size."""
        source = mock_chat_stream(
            messages=["12345", "67890", "ABC"],  # Total: 13 chars
            query="Test",
        )

        validator = MockValidator(classification="statistics")

        # Buffer size of 10 should trigger after first two chunks
        adapted = stateful_validation_adapter(
            source,
            validator_model=validator,
            allowed_content=["statistics"],
            buffer_size=10,
        )

        events = await consume_stream(adapted)  # noqa

        # Validator should be called once buffer reaches 10
        self.assertEqual(validator.call_count, 1)

        # Check that validation included the query and buffered content
        invocation = validator.invocations[0]["text"]
        self.assertIn("Test", invocation)  # Query
        self.assertIn("1234567890", invocation)  # Buffered text

    async def test_buffer_order_preserved(self):
        """Test that buffered chunks are released in original order."""
        source = mock_chat_stream(
            messages=["A", "B", "C", "D", "E"],
            query="Test",
        )

        validator = MockValidator(classification="statistics")

        adapted = stateful_validation_adapter(
            source,
            validator_model=validator,
            allowed_content=["statistics"],
            buffer_size=3,
        )

        events = await consume_stream(adapted)
        messages = get_messages_from_events(events)

        # Messages should be in original order
        self.assertEqual(messages, ["A", "B", "C", "D", "E"])


class TestValidationResultStructure(unittest.IsolatedAsyncioTestCase):
    """Tests for ValidationResult TypedDict."""

    async def test_apology_allowed(self):
        """Test that 'apology' classification is automatically allowed."""
        source = mock_chat_stream(
            messages=["I", " apologize"],
            query="Test",
        )

        validator = MockValidator(classification="apology")

        adapted = stateful_validation_adapter(
            source,
            validator_model=validator,
            allowed_content=["statistics"],  # apology not in list
            buffer_size=5,
        )

        events = await consume_stream(adapted)
        final_state = get_final_state(events)

        # Should pass validation
        self.assertIsNotNone(final_state)
        self.assertEqual(final_state["status"], "valid")
        self.assertEqual(
            final_state["query_classification"], "apology"
        )

    async def test_human_interaction_allowed(self):
        """Test that 'human interaction' classification is automatically allowed."""
        source = mock_chat_stream(
            messages=["Hello"],
            query="Test",
        )

        validator = MockValidator(classification="human interaction")

        adapted = stateful_validation_adapter(
            source,
            validator_model=validator,
            allowed_content=["statistics"],
            buffer_size=3,
        )

        events = await consume_stream(adapted)
        final_state = get_final_state(events)

        # Should pass validation
        self.assertIsNotNone(final_state)
        self.assertEqual(final_state["status"], "valid")
        self.assertEqual(
            final_state["query_classification"], "human interaction"
        )


if __name__ == "__main__":
    unittest.main()
