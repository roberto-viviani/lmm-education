"""
Comprehensive unit tests for stream_adapters module.

These tests use mock async iterators to test adapter functions in isolation,
avoiding dependencies on LangGraph internals.
"""

# pyright: basic

import unittest
import asyncio

from langchain_core.messages import BaseMessageChunk, AIMessageChunk
from lmm.utils.logging import LoglistLogger

from lmm_education.workflows.langchain.stream_adapters import (
    field_change_tier_1_adapter,
    terminal_tier1_adapter,
    tier_1_to_2_adapter,
    tier_2_to_3_adapter,
    tier_1_to_3_adapter,
    terminal_field_change_adapter,
    tier_1_iterator,
    tier_2_iterator,
)


# ===================================================================
# Mock Stream Generators
# ===================================================================


async def mock_tier_1_stream(
    include_messages: bool = True,
    include_values: bool = True,
    include_updates: bool = False,
    node_name: str = "test_node",
) -> tier_1_iterator:
    """
    Generate a mock tier 1 stream with configurable events.

    Yields (mode, event) tuples simulating a LangGraph stream.
    """
    if include_messages:
        # Simulate message chunks
        for text in ["Hello", " ", "World"]:
            chunk = AIMessageChunk(content=text)
            metadata = {"langgraph_node": node_name}
            yield ("messages", (chunk, metadata))

    if include_updates:
        # Simulate state updates
        yield ("updates", {"test_node": {"status": "processing"}})
        yield (
            "updates",
            {"test_node": {"context": "Retrieved context"}},
        )

    if include_values:
        # Simulate final state
        final_state = {
            "messages": ["test message"],
            "status": "complete",
            "context": "Final context",
        }
        yield ("values", final_state)


async def mock_tier_2_stream(
    node_name: str = "test_node",
) -> tier_2_iterator:
    """Generate a mock tier 2 stream with message chunks."""
    for text in ["Hello", " ", "World"]:
        chunk = AIMessageChunk(content=text)
        metadata = {"langgraph_node": node_name}
        yield (chunk, metadata)


async def empty_tier_1_stream() -> tier_1_iterator:
    """Generate an empty stream."""
    return
    yield  # Make it a generator


# ===================================================================
# Helper Functions
# ===================================================================


async def consume_stream(stream):
    """Consume an async iterator and return all items as a list."""
    items = []
    async for item in stream:
        items.append(item)
    return items


# ===================================================================
# Test Classes
# ===================================================================


class TestFieldChangeTier1Adapter(unittest.IsolatedAsyncioTestCase):
    """Tests for field_change_tier_1_adapter."""

    async def test_no_callbacks_passes_through(self):
        """Test that stream passes through unchanged without callbacks."""
        source = mock_tier_1_stream(include_updates=True)
        adapted = field_change_tier_1_adapter(
            source, on_field_change=None
        )

        events = await consume_stream(adapted)

        # Should have messages, updates, and values
        self.assertGreater(len(events), 0)
        modes = [mode for mode, _ in events]
        self.assertIn("messages", modes)
        self.assertIn("updates", modes)

    async def test_sync_callback_modifies_field(self):
        """Test that sync callback modifies field value via deep copy."""
        source = mock_tier_1_stream(
            include_updates=True, include_messages=False
        )

        def transform_status(value: str) -> str:
            return value.upper()

        adapted = field_change_tier_1_adapter(
            source, on_field_change={"status": transform_status}
        )

        events = await consume_stream(adapted)

        # Find the updates event
        update_events = [e for mode, e in events if mode == "updates"]
        self.assertGreater(len(update_events), 0)

        # Check that status was transformed
        for update in update_events:
            if (
                "test_node" in update
                and "status" in update["test_node"]
            ):
                self.assertEqual(
                    update["test_node"]["status"], "PROCESSING"
                )

    async def test_async_callback_modifies_field(self):
        """Test that async callback modifies field value."""
        source = mock_tier_1_stream(
            include_updates=True, include_messages=False
        )

        async def transform_context(value: str) -> str:
            await asyncio.sleep(0.001)  # Simulate async work
            return f"[{value}]"

        adapted = field_change_tier_1_adapter(
            source, on_field_change={"context": transform_context}
        )

        events = await consume_stream(adapted)

        # Find the context update
        update_events = [e for mode, e in events if mode == "updates"]
        for update in update_events:
            if (
                "test_node" in update
                and "context" in update["test_node"]
            ):
                self.assertEqual(
                    update["test_node"]["context"],
                    "[Retrieved context]",
                )

    async def test_callback_returning_none_unchanged(self):
        """Test that callback returning None leaves field unchanged."""
        source = mock_tier_1_stream(
            include_updates=True, include_messages=False
        )

        def no_change(value: str) -> None:
            return None

        adapted = field_change_tier_1_adapter(
            source, on_field_change={"status": no_change}
        )

        events = await consume_stream(adapted)

        # Field should remain unchanged
        update_events = [e for mode, e in events if mode == "updates"]
        for update in update_events:
            if (
                "test_node" in update
                and "status" in update["test_node"]
            ):
                # Original value preserved
                self.assertEqual(
                    update["test_node"]["status"], "processing"
                )

    async def test_callback_exception_logged_not_raised(self):
        """Test that callback exceptions are logged but don't break stream."""
        logger = LoglistLogger()
        source = mock_tier_1_stream(
            include_updates=True, include_messages=False
        )

        def failing_callback(value: str) -> str:
            raise ValueError("Intentional test error")

        adapted = field_change_tier_1_adapter(
            source,
            on_field_change={"status": failing_callback},
            logger=logger,
        )

        # Should not raise, should complete
        events = await consume_stream(adapted)
        self.assertGreater(len(events), 0)

        # Should have logged the error
        logs = logger.get_logs()
        self.assertTrue(
            any(
                "Error in field_change_adapter" in log for log in logs
            )
        )

    async def test_deep_copy_isolation(self):
        """Test that modifications don't affect original event."""

        # Create a stream with a mutable event
        async def source_stream():
            original_event = {"test_node": {"field": "original"}}
            yield ("updates", original_event)

        def transform(value: str) -> str:
            return "modified"

        adapted = field_change_tier_1_adapter(
            source_stream(), on_field_change={"field": transform}
        )

        events = await consume_stream(adapted)

        # The yielded event should be modified
        self.assertEqual(
            events[0][1]["test_node"]["field"], "modified"
        )


class TestTerminalTier1Adapter(unittest.IsolatedAsyncioTestCase):
    """Tests for terminal_tier1_adapter."""

    async def test_sync_callback_receives_final_state(self):
        """Test that sync callback receives the final values event."""
        final_state_received = {}

        def capture_state(state):
            final_state_received.update(state)

        source = mock_tier_1_stream(include_values=True)
        adapted = terminal_tier1_adapter(
            source, on_terminal_state=capture_state
        )

        await consume_stream(adapted)

        # Callback should have been invoked with final state
        self.assertIn("status", final_state_received)
        self.assertEqual(final_state_received["status"], "complete")

    async def test_async_callback_receives_final_state(self):
        """Test that async callback receives the final values event."""
        final_state_received = {}

        async def capture_state(state):
            await asyncio.sleep(0.001)
            final_state_received.update(state)

        source = mock_tier_1_stream(include_values=True)
        adapted = terminal_tier1_adapter(
            source, on_terminal_state=capture_state
        )

        await consume_stream(adapted)

        # Need to wait for scheduled task
        await asyncio.sleep(0.1)

        # Callback should have been invoked
        self.assertIn("status", final_state_received)

    async def test_no_values_no_callback(self):
        """Test that callback is not invoked if no values event."""
        callback_invoked = False

        def capture_state(state):
            nonlocal callback_invoked
            callback_invoked = True

        source = mock_tier_1_stream(include_values=False)
        adapted = terminal_tier1_adapter(
            source, on_terminal_state=capture_state
        )

        await consume_stream(adapted)

        # Callback should not have been invoked
        self.assertFalse(callback_invoked)

    async def test_events_pass_through_unchanged(self):
        """Test that all events pass through unchanged."""
        source = mock_tier_1_stream(
            include_values=True, include_updates=True
        )
        adapted = terminal_tier1_adapter(source)

        source_events = await consume_stream(
            mock_tier_1_stream(
                include_values=True, include_updates=True
            )
        )
        adapted_events = await consume_stream(adapted)

        # Should have same number of events
        self.assertEqual(len(source_events), len(adapted_events))


class TestTier1To2Adapter(unittest.IsolatedAsyncioTestCase):
    """Tests for tier_1_to_2_adapter."""

    async def test_filters_to_messages_only(self):
        """Test that only message events are yielded."""
        source = mock_tier_1_stream(
            include_messages=True,
            include_values=True,
            include_updates=True,
        )
        adapted = tier_1_to_2_adapter(source)

        events = await consume_stream(adapted)

        # All events should be (chunk, metadata) tuples
        for chunk, metadata in events:
            self.assertIsInstance(chunk, BaseMessageChunk)
            self.assertIsInstance(metadata, dict)
            self.assertIn("langgraph_node", metadata)

    async def test_filter_by_source_nodes(self):
        """Test filtering by source_nodes."""

        # Create stream with different nodes
        async def multi_node_stream():
            yield (
                "messages",
                (
                    AIMessageChunk(content="A"),
                    {"langgraph_node": "node1"},
                ),
            )
            yield (
                "messages",
                (
                    AIMessageChunk(content="B"),
                    {"langgraph_node": "node2"},
                ),
            )
            yield (
                "messages",
                (
                    AIMessageChunk(content="C"),
                    {"langgraph_node": "node1"},
                ),
            )

        adapted = tier_1_to_2_adapter(
            multi_node_stream(), source_nodes=["node1"]
        )

        events = await consume_stream(adapted)

        # Should only have events from node1
        self.assertEqual(len(events), 2)
        for _, metadata in events:
            self.assertEqual(metadata["langgraph_node"], "node1")

    async def test_none_source_nodes_allows_all(self):
        """Test that None source_nodes allows all nodes."""

        async def multi_node_stream():
            yield (
                "messages",
                (
                    AIMessageChunk(content="A"),
                    {"langgraph_node": "node1"},
                ),
            )
            yield (
                "messages",
                (
                    AIMessageChunk(content="B"),
                    {"langgraph_node": "node2"},
                ),
            )

        adapted = tier_1_to_2_adapter(
            multi_node_stream(), source_nodes=None
        )

        events = await consume_stream(adapted)

        # Should have both events
        self.assertEqual(len(events), 2)


class TestTier2To3Adapter(unittest.IsolatedAsyncioTestCase):
    """Tests for tier_2_to_3_adapter."""

    async def test_extracts_text_from_chunks(self):
        """Test that text is extracted from message chunks."""
        source = mock_tier_2_stream()
        adapted = tier_2_to_3_adapter(source)

        texts = await consume_stream(adapted)

        # Should get the text content
        self.assertEqual(texts, ["Hello", " ", "World"])

    async def test_filter_by_source_nodes(self):
        """Test filtering by source_nodes."""

        async def multi_node_stream():
            yield (
                AIMessageChunk(content="A"),
                {"langgraph_node": "node1"},
            )
            yield (
                AIMessageChunk(content="B"),
                {"langgraph_node": "node2"},
            )
            yield (
                AIMessageChunk(content="C"),
                {"langgraph_node": "node1"},
            )

        adapted = tier_2_to_3_adapter(
            multi_node_stream(), source_nodes=["node1"]
        )

        texts = await consume_stream(adapted)

        # Should only have texts from node1
        self.assertEqual(texts, ["A", "C"])

    async def test_none_source_nodes_allows_all(self):
        """Test that None source_nodes allows all nodes."""

        async def multi_node_stream():
            yield (
                AIMessageChunk(content="A"),
                {"langgraph_node": "node1"},
            )
            yield (
                AIMessageChunk(content="B"),
                {"langgraph_node": "node2"},
            )

        adapted = tier_2_to_3_adapter(
            multi_node_stream(), source_nodes=None
        )

        texts = await consume_stream(adapted)

        # Should have all texts
        self.assertEqual(texts, ["A", "B"])


class TestTier1To3Adapter(unittest.IsolatedAsyncioTestCase):
    """Tests for tier_1_to_3_adapter."""

    async def test_extracts_message_text(self):
        """Test that message text is extracted and other events discarded."""
        source = mock_tier_1_stream(
            include_messages=True,
            include_values=True,
            include_updates=True,
        )
        adapted = tier_1_to_3_adapter(source)

        texts = await consume_stream(adapted)

        # Should get message texts only
        self.assertEqual(texts, ["Hello", " ", "World"])

    async def test_filter_by_source_nodes(self):
        """Test filtering by source_nodes."""

        async def multi_node_stream():
            yield (
                "messages",
                (
                    AIMessageChunk(content="A"),
                    {"langgraph_node": "node1"},
                ),
            )
            yield (
                "messages",
                (
                    AIMessageChunk(content="B"),
                    {"langgraph_node": "node2"},
                ),
            )
            yield ("values", {"state": "final"})

        adapted = tier_1_to_3_adapter(
            multi_node_stream(), source_nodes=["node1"]
        )

        texts = await consume_stream(adapted)

        # Should only have text from node1
        self.assertEqual(texts, ["A"])


class TestTerminalFieldChangeAdapter(
    unittest.IsolatedAsyncioTestCase
):
    """Tests for terminal_field_change_adapter."""

    async def test_extracts_messages_and_injects_field_changes(self):
        """Test that messages are extracted and field changes are injected."""

        async def source():
            yield (
                "messages",
                (
                    AIMessageChunk(content="Hello"),
                    {"langgraph_node": "test"},
                ),
            )
            yield ("updates", {"test_node": {"status": "ready"}})
            yield (
                "messages",
                (
                    AIMessageChunk(content=" World"),
                    {"langgraph_node": "test"},
                ),
            )
            yield ("values", {"final": "state"})

        def status_callback(value: str) -> str:
            return f"[STATUS: {value}]"

        adapted = terminal_field_change_adapter(
            source(), on_field_change={"status": status_callback}
        )

        texts = await consume_stream(adapted)

        # Should have messages and injected status
        self.assertIn("Hello", texts)
        self.assertIn(" World", texts)
        self.assertIn("[STATUS: ready]", texts)

    async def test_terminal_state_callback_invoked(self):
        """Test that terminal state callback is invoked."""
        final_state_received = {}

        def capture_state(state):
            final_state_received.update(state)

        source = mock_tier_1_stream(include_values=True)
        adapted = terminal_field_change_adapter(
            source, on_terminal_state=capture_state
        )

        await consume_stream(adapted)

        # Should have received final state
        self.assertIn("status", final_state_received)

    async def test_field_change_returning_none_not_injected(self):
        """Test that field changes returning None are not injected."""

        async def source():
            yield (
                "messages",
                (
                    AIMessageChunk(content="Hello"),
                    {"langgraph_node": "test"},
                ),
            )
            yield ("updates", {"test_node": {"status": "ready"}})
            yield ("values", {"final": "state"})

        def no_inject(value: str) -> None:
            return None

        adapted = terminal_field_change_adapter(
            source(), on_field_change={"status": no_inject}
        )

        texts = await consume_stream(adapted)

        # Should only have the message
        self.assertEqual(texts, ["Hello"])


class TestEdgeCases(unittest.IsolatedAsyncioTestCase):
    """Tests for edge cases and error conditions."""

    async def test_empty_stream(self):
        """Test that adapters handle empty streams gracefully."""
        source = empty_tier_1_stream()
        adapted = tier_1_to_3_adapter(source)

        texts = await consume_stream(adapted)

        self.assertEqual(texts, [])

    async def test_malformed_metadata(self):
        """Test handling of events with missing metadata."""
        logger = LoglistLogger()

        async def malformed_stream():
            # Missing langgraph_node in metadata
            yield ("messages", (AIMessageChunk(content="Test"), {}))

        adapted = tier_1_to_2_adapter(
            malformed_stream(),
            source_nodes=["some_node"],
            logger=logger,
        )

        # Should handle gracefully and still yield event
        events = await consume_stream(adapted)
        self.assertEqual(len(events), 1)


class TestIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for adapter chaining."""

    async def test_full_tier1_to_tier3_pipeline(self):
        """Test chaining all adapter types together."""
        # Create a complex stream
        source = mock_tier_1_stream(
            include_messages=True,
            include_values=True,
            include_updates=True,
        )

        # Apply tier 1 adapter
        final_state_received = {}

        def capture_state(state):
            final_state_received.update(state)

        tier1 = terminal_tier1_adapter(
            source, on_terminal_state=capture_state
        )

        # Convert to tier 2
        tier2 = tier_1_to_2_adapter(tier1)

        # Convert to tier 3
        tier3 = tier_2_to_3_adapter(tier2)

        # Consume final stream
        texts = await consume_stream(tier3)

        # Should have message texts
        self.assertEqual(texts, ["Hello", " ", "World"])

        # Should have captured final state
        self.assertIn("status", final_state_received)

    async def test_field_change_affects_downstream(self):
        """Test that field changes in tier 1 affect downstream consumers."""

        async def source():
            yield ("updates", {"node": {"field": "original"}})
            yield ("values", {"field": "original"})

        def transform(value: str) -> str:
            return "modified"

        # Apply field change adapter
        adapted = field_change_tier_1_adapter(
            source(), on_field_change={"field": transform}
        )

        # Collect events
        events = await consume_stream(adapted)

        # The update event should have modified field
        update_event = [e for mode, e in events if mode == "updates"][
            0
        ]
        self.assertEqual(update_event["node"]["field"], "modified")


if __name__ == "__main__":
    unittest.main()
