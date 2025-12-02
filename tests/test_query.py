"""Test the refactored chat_function_with_validation."""

import unittest
import logging

from lmm_education.query import (
    chat_function_with_validation,
    consume_chat_stream,
    chat_function,
    _error_message_iterator,
)
from lmm_education.config.config import (
    ConfigSettings,
    export_settings,
)
from lmm_education.stores.langchain.vector_store_qdrant_langchain import (
    AsyncQdrantVectorStoreRetriever as AsyncQdrantRetriever,
)

original_settings = ConfigSettings()


def setUpModule():
    settings = ConfigSettings(
        major={'model': "Debug/debug"},
        minor={'model': "Debug/debug"},
        aux={'model': "Debug/debug"},
        embeddings={
            'dense_model': "SentenceTransformers/all-MiniLM-L6-v2",
            'sparse_model': "Qdrant/bm25",
        },
    )
    export_settings(settings)


def tearDownModule():
    export_settings(original_settings)


class TestQuery(unittest.IsolatedAsyncioTestCase):

    async def test_empty_query(self):
        """Test that empty query returns error iterator."""
        print("Test 1: Empty query")
        iterator = await chat_function("", [])
        result = await consume_chat_stream(iterator)
        print(f"Result: {result}")
        self.assertIn("If you have questions", result)
        print("✓ Passed\n")

    async def test_long_query(self):
        """Test that overly long query returns error iterator."""
        print("Test 2: Long query")
        long_query = " ".join(["word"] * 200)
        iterator = await chat_function(long_query, [])
        result = await consume_chat_stream(iterator)
        print(f"Result: {result}")
        self.assertIn("too long", result)
        print("✓ Passed\n")

    async def test_normal_query(self):
        """Test a normal query (if LLM is available)."""
        print("Test 3: Normal query")
        try:
            iterator = await chat_function("What is a linear model?")
            result = await consume_chat_stream(iterator)
            print(f"Result length: {len(result)} characters")
            print(f"First 100 chars: {result[:100]}...")
            self.assertTrue(len(result) > 0)
            print("✓ Passed\n")
        except Exception as e:
            print(f"⚠ Skipped (LLM not available): {e}\n")
            raise Exception(
                "Error in test_validation_normal_query"
            ) from e

    async def test_repeated_query(self):
        """Test a repeated query (if LLM is available)."""
        print("Test 4: Repeated query")
        try:
            # explicit retirever, that may be re-used
            retriever = AsyncQdrantRetriever.from_config_settings()
            iterator = await chat_function(
                "What is a linear model?", retriever=retriever
            )
            result = await consume_chat_stream(iterator)
            print(f"Result length: {len(result)} characters")
            print(f"First 100 chars: {result[:100]}...")
            self.assertTrue(len(result) > 0)
            iterator = await chat_function(
                "What is a logistic regression?", retriever=retriever
            )
            result = await consume_chat_stream(iterator)
            print(f"Result length: {len(result)} characters")
            print(f"First 100 chars: {result[:100]}...")
            # must be closed down
            await retriever.close_client()  # type: ignore
            self.assertTrue(len(result) > 0)

            print("✓ Passed\n")
        except Exception as e:
            print(f"⚠ Skipped (LLM not available): {e}\n")
            raise Exception(
                "Error in test_validation_normal_query"
            ) from e


from lmm.utils.logging import (
    LoglistLogger,
    LoggerBase,
)


class TestErrorMessageIterator(unittest.IsolatedAsyncioTestCase):
    """Test the _error_message_iterator helper function."""

    async def test_error_message_iterator(self):
        """Test that _error_message_iterator creates an async iterator that streams the error message."""
        print("Test: Error message iterator")

        # Test message
        test_message = "This is a test error message"

        # Create the iterator by calling _error_message_iterator
        iterator = _error_message_iterator(test_message)

        # Consume the iterator
        result = await consume_chat_stream(iterator)

        # Verify that the streamed content matches the intended message
        print(f"Expected: {test_message}")
        print(f"Result: {result}")
        self.assertEqual(result, test_message)
        print("✓ Passed\n")

    async def test_error_message_iterator_with_special_chars(self):
        """Test that _error_message_iterator handles special characters correctly."""
        print("Test: Error message iterator with special characters")

        # Test message with special characters
        test_message = (
            "Error: Invalid input! Please try again.\n(Code: 400)"
        )

        # Create the iterator
        iterator = _error_message_iterator(test_message)

        # Consume the iterator
        result = await consume_chat_stream(iterator)

        # Verify the content
        print(f"Expected: {repr(test_message)}")
        print(f"Result: {repr(result)}")
        self.assertEqual(result, test_message)
        print("✓ Passed\n")


class TestQueryValidated(unittest.IsolatedAsyncioTestCase):

    async def test_validation_empty_query(self):
        """Test that empty query returns error iterator."""
        print("Test 1: Empty query with validation")
        iterator = await chat_function_with_validation(
            "", [], allowed_content=["statistics"]
        )
        result = await consume_chat_stream(iterator)
        print(f"Result: {result}")
        self.assertIn("If you have questions", result)
        print("✓ Passed\n")

    async def test_validation_long_query(self):
        """Test that overly long query returns error iterator."""
        print("Test 2: Long query with validation")
        logger: LoggerBase = LoglistLogger()
        long_query = " ".join(["word"] * 200)
        iterator = await chat_function_with_validation(
            long_query,
            [],
            allowed_content=["statistics"],
            logger=logger,
        )
        if logger.count_logs(level=logging.WARNING) > 0:
            print("\n".join(logger.get_logs()))
        self.assertEqual(logger.count_logs(level=logging.WARNING), 0)
        result = await consume_chat_stream(iterator)
        print(f"Result: {result}")
        self.assertIn("too long", result)
        print("✓ Passed\n")

    async def test_validation_normal_query(self):
        """Test a normal query with validation (if LLM is available)."""
        print("Test 3: Normal query with validation")
        try:
            iterator = await chat_function_with_validation(
                "What is a linear model?",
                [],
                initial_buffer_size=50,  # Use smaller buffer for testing
                allowed_content=["statistics"],
            )
            result = await consume_chat_stream(iterator)
            print(f"Result length: {len(result)} characters")
            print(f"First 200 chars: {result[:200]}...")
            self.assertTrue(len(result) > 0)
            print("✓ Passed\n")
        except Exception as e:
            print(
                f"⚠ Skipped (validation model not available): {e}\n"
            )
            raise Exception(
                "Error in test_validation_normal_query"
            ) from e


if __name__ == "__main__":
    unittest.main()
