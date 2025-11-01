"""Test the refactored chat_function_with_validation."""

import unittest

from lmm_education.query import (
    chat_function_with_validation,
    consume_chat_stream,
    chat_function,
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
        self.assertIn("Please ask a question", result)
        print("✓ Passed\n")

    async def test_long_query(self):
        """Test that overly long query returns error iterator."""
        print("Test 2: Long query")
        long_query = " ".join(["word"] * 100)
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


class TestQueryValidated(unittest.IsolatedAsyncioTestCase):

    async def test_validation_empty_query(self):
        """Test that empty query returns error iterator."""
        print("Test 1: Empty query with validation")
        iterator = await chat_function_with_validation("", [])
        result = await consume_chat_stream(iterator)
        print(f"Result: {result}")
        self.assertIn("Please ask a question", result)
        print("✓ Passed\n")

    async def test_validation_long_query(self):
        """Test that overly long query returns error iterator."""
        print("Test 2: Long query with validation")
        long_query = " ".join(["word"] * 100)
        iterator = await chat_function_with_validation(long_query, [])
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
