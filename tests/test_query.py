"""Test the refactored chat_function_with_validation."""

# pyright: strict

import unittest
import asyncio
import io
import logging
from collections.abc import AsyncIterator

from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from lmm.language_models.langchain.models import (
    create_model_from_settings,
)
from lmm_education.query import (
    chat_stream,
    ChatWorkflowContext,
)
from lmm_education.config.config import (
    ConfigSettings,
    export_settings,
)
from lmm_education.config.appchat import ChatSettings, CheckResponse
from lmm_education.stores.langchain.vector_store_qdrant_langchain import (
    AsyncQdrantVectorStoreRetriever as AsyncQdrantRetriever,
)


# An embedding engine object is created here just to load the engine.
# This avoids the first query to take too long. The object is cached
# internally, so we do not actually use the embedding object here.
from langchain_core.embeddings import Embeddings
from lmm.language_models.langchain.runnables import create_embeddings
from requests import ConnectionError

try:
    settings: ConfigSettings = ConfigSettings()
    embedding: Embeddings = create_embeddings()
    if "SentenceTransformer" not in settings.embeddings.dense_model:
        embedding.embed_query("Test data")
except ConnectionError as e:
    print("Could not connect to the model provider -- no internet?")
    print(f"Error message:\n{e}")
    exit()
except Exception as e:
    print(
        "Could not connect to the model provider due to a system error."
    )
    print(f"Error message:\n{e}")
    exit()


original_settings = ConfigSettings()


async def consume_chat_stream(
    iterator: AsyncIterator[str],
) -> str:
    """
    Consumes an async iterator of BaseMessageChunk objects and returns
    the complete response as a string.

    This function is designed to work with the iterator returned by
    chat_stream. It accumulates the text content from each chunk
    and returns the final result.

    Args:
        iterator: AsyncIterator yielding BaseMessageChunk objects

    Returns:
        str: The complete accumulated response text
    """
    buffer: str = ""
    async for chunk in iterator:
        buffer += chunk
    return buffer


def setUpModule():
    settings = ConfigSettings(
        major={'model': "Debug/debug"},
        minor={'model': "Debug/debug"},
        aux={'model': "Debug/debug"},
        # the current database may not use this encoding model
        # embeddings={
        #     'dense_model': "SentenceTransformers/all-MiniLM-L6-v2",
        #     'sparse_model': "Qdrant/bm25",
        # },
    )
    export_settings(settings)


def tearDownModule():
    export_settings(original_settings)


class TestQuery(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.llm: BaseChatModel = create_model_from_settings(
            ConfigSettings().major
        )
        self.retriever: BaseRetriever = (
            AsyncQdrantRetriever.from_config_settings()
        )

    def get_workflow_context(
        self,
        chat_settings: ChatSettings = ChatSettings(
            check_response=CheckResponse(check_response=False)
        ),
    ) -> ChatWorkflowContext:

        return ChatWorkflowContext(
            llm=self.llm,
            retriever=self.retriever,
            chat_settings=chat_settings,
        )

    async def test_empty_query(self):
        """Test that empty query returns error iterator."""
        print("Test 1: Empty query")

        iterator = chat_stream(
            "",
            [],
            self.get_workflow_context(),
        )
        result = await consume_chat_stream(iterator)
        print(f"Result: {result}")
        self.assertIn("If you have questions", result)
        print("✓ Passed\n")

    async def test_long_query(self):
        """Test that overly long query returns error iterator."""
        print("Test 2: Long query")
        long_query = " ".join(["word"] * 200)

        iterator = chat_stream(
            long_query,
            [],
            self.get_workflow_context(),
        )
        result = await consume_chat_stream(iterator)
        print(f"Result: {result}")
        self.assertIn("too long", result)
        print("✓ Passed\n")

    async def test_normal_query(self):
        """Test a normal query (if LLM is available)."""
        print("Test 3: Normal query")
        try:

            iterator = chat_stream(
                "What is a linear model?",
                None,
                self.get_workflow_context(),
            )
            result = await consume_chat_stream(iterator)
        except Exception as e:
            print(f"⚠ Skipped (LLM not available): {e}\n")
            raise Exception(
                "Error in test_validation_normal_query"
            ) from e

        print(f"Result length: {len(result)} characters")
        print(f"First 100 chars: {result[:100]}...")
        self.assertTrue(len(result) > 0)
        print("✓ Passed\n")

    async def test_repeated_query(self):
        """Test a repeated query (if LLM is available)."""
        print("Test 4: Repeated query")
        try:
            # explicit retriever, that may be re-used
            context: ChatWorkflowContext = self.get_workflow_context()
            context.retriever = (
                AsyncQdrantRetriever.from_config_settings()
            )

            iterator = chat_stream(
                "What is a linear model?",
                None,
                context,
            )
            result = await consume_chat_stream(iterator)
            print(f"Result length: {len(result)} characters")
            print(f"First 100 chars: {result[:100]}...")
            self.assertTrue(len(result) > 0)

            iterator = chat_stream(
                "What is a logistic regression?",
                [],
                context,
            )
            result = await consume_chat_stream(iterator)
            print(f"Result length: {len(result)} characters")
            print(f"First 100 chars: {result[:100]}...")
            # must be closed down
            await context.retriever.close_client()  # type: ignore
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


from lmm_education.config.appchat import CheckResponse


class TestQueryValidated(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.llm: BaseChatModel = create_model_from_settings(
            ConfigSettings().major
        )
        self.retriever: BaseRetriever = (
            AsyncQdrantRetriever.from_config_settings()
        )

    def get_workflow_context(
        self, chat_settings: ChatSettings = ChatSettings()
    ) -> ChatWorkflowContext:

        return ChatWorkflowContext(
            llm=self.llm,
            retriever=self.retriever,
            chat_settings=chat_settings,
        )

    async def test_validation_empty_query(self):
        """Test that empty query returns error iterator."""
        print("Test 1 (validation): Empty query with validation")
        chat_settings = ChatSettings(
            check_response=CheckResponse(
                check_response=True,
                allowed_content=['statistics'],
            )
        )
        context = self.get_workflow_context(chat_settings)
        # chat_function_with_validation is async generator - call directly without await
        iterator = chat_stream(
            "",
            [],
            context,
            # validate defaults to None, which uses context
        )
        result = await consume_chat_stream(iterator)
        print(f"Result: {result}")
        self.assertIn("If you have questions", result)
        print("✓ Passed\n")

    async def test_validation_long_query(self):
        """Test that overly long query returns error iterator."""
        print("Test 2 (validation): Long query with validation")

        logger: LoggerBase = LoglistLogger()
        long_query = " ".join(["word"] * 200)
        chat_settings = ChatSettings(
            check_response=CheckResponse(
                check_response=True, allowed_content=['statistics']
            )
        )
        context = self.get_workflow_context(chat_settings)
        # chat_function_with_validation is async generator - call directly without await
        iterator = chat_stream(
            long_query,
            [],
            context,
            validate=None,  # use context
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
        print("Test 3 (validation): Normal query with validation")
        try:
            chat_settings = ChatSettings(
                check_response=CheckResponse(
                    check_response=True,
                    allowed_content=['statistics'],
                    initial_buffer_size=120,
                )
            )
            # chat_function_with_validation is async generator - call directly without await
            context = self.get_workflow_context(chat_settings)
            iterator = chat_stream(
                "What is a linear model?",
                [],
                context,
                validate=chat_settings.check_response,
            )
            result = await consume_chat_stream(iterator)
        except Exception as e:
            print(
                f"⚠ Skipped (validation model not available): {e}\n"
            )
            raise Exception(
                "Error in test_validation_normal_query"
            ) from e

        print(f"Result length: {len(result)} characters")
        print(f"First 200 chars: {result[:200]}...")
        self.assertTrue(len(result) > 0)
        print("✓ Passed\n")


class TestQueryDatabaseLog(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.llm: BaseChatModel = create_model_from_settings(
            ConfigSettings().major
        )
        self.retriever: BaseRetriever = (
            AsyncQdrantRetriever.from_config_settings()
        )

    def get_workflow_context(
        self,
        chat_settings: ChatSettings = ChatSettings(
            check_response=CheckResponse(check_response=False)
        ),
    ) -> ChatWorkflowContext:

        return ChatWorkflowContext(
            llm=self.llm,
            retriever=self.retriever,
            chat_settings=chat_settings,
        )

    async def test_empty_query(self):
        """Test that empty query returns error iterator."""
        print("Test 1 (log): Empty query")

        stream = io.StringIO()
        stream_context = io.StringIO()

        iterator = chat_stream(
            "",
            [],
            self.get_workflow_context(),
            database_streams=[stream, stream_context],
            database_log=True,
        )
        result = await consume_chat_stream(iterator)
        self.assertIn("If you have questions", result)

        await asyncio.sleep(0.1)

        logtext: str = stream.getvalue()
        self.assertGreater(len(logtext), 0)
        self.assertIn("EMPTYQUERY", logtext)
        self.assertEqual(len(stream_context.getvalue()), 0)

        print("✓ Passed\n")

    async def test_long_query(self):
        """Test that overly long query returns error iterator."""
        print("Test 2 (log): Long query")
        long_query = " ".join(["word"] * 200)

        stream = io.StringIO()
        stream_context = io.StringIO()

        iterator = chat_stream(
            long_query,
            [],
            self.get_workflow_context(),
            database_streams=[stream, stream_context],
            database_log=True,
        )
        result = await consume_chat_stream(iterator)
        self.assertIn("too long", result)

        await asyncio.sleep(0.1)

        logtext: str = stream.getvalue()
        self.assertGreater(len(logtext), 0)
        self.assertIn("LONGQUERY", logtext)
        self.assertEqual(len(stream_context.getvalue()), 0)
        print("✓ Passed\n")

    async def test_normal_query(self):
        """Test a normal query (if LLM is available)."""
        print("Test 3 (log): Normal query")

        stream = io.StringIO()
        stream_context = io.StringIO()

        try:

            iterator = chat_stream(
                "What is a linear model?",
                None,
                self.get_workflow_context(),
                database_streams=[stream, stream_context],
                database_log=True,
            )
            result = await consume_chat_stream(iterator)
        except Exception as e:
            print(f"⚠ Skipped (LLM not available): {e}\n")
            raise Exception(
                "Error in test_validation_normal_query"
            ) from e

        self.assertTrue(len(result) > 0)

        await asyncio.sleep(0.1)

        logtext: str = stream.getvalue()
        self.assertGreater(len(logtext), 0)
        self.assertIn("MESSAGE", logtext)
        logcontext: str = stream_context.getvalue()
        self.assertGreater(len(logcontext), 0)

        print("✓ Passed\n")

    async def test_normal_query_nolog(self):
        """Test a normal query (if LLM is available)."""
        print("Test 3 (log): Normal query")

        stream = io.StringIO()
        stream_context = io.StringIO()

        try:

            iterator = chat_stream(
                "What is a linear model?",
                None,
                self.get_workflow_context(),
            )
            result = await consume_chat_stream(iterator)

        except Exception as e:
            print(f"⚠ Skipped (LLM not available): {e}\n")
            raise Exception(
                "Error in test_validation_normal_query"
            ) from e

        self.assertTrue(len(result) > 0)

        await asyncio.sleep(0.1)

        logtext: str = stream.getvalue()
        self.assertEqual(len(logtext), 0)
        logcontext: str = stream_context.getvalue()
        self.assertEqual(len(logcontext), 0)

        print("✓ Passed\n")


class TestQueryPrintContext(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.llm: BaseChatModel = create_model_from_settings(
            ConfigSettings().major
        )
        self.retriever: BaseRetriever = (
            AsyncQdrantRetriever.from_config_settings()
        )

    def get_workflow_context(
        self,
        chat_settings: ChatSettings = ChatSettings(
            check_response=CheckResponse(check_response=False)
        ),
    ) -> ChatWorkflowContext:

        return ChatWorkflowContext(
            llm=self.llm,
            retriever=self.retriever,
            chat_settings=chat_settings,
        )

    async def test_empty_query(self):
        """Test that empty query returns error iterator."""
        print("Test 1 (context): Empty query")

        iterator = chat_stream(
            "",
            [],
            self.get_workflow_context(),
            print_context=True,
        )
        result = await consume_chat_stream(iterator)
        self.assertFalse(result.startswith("CONTEXT"))
        self.assertNotIn("END CONTEXT--", result)
        self.assertIn("If you have questions", result)
        print("✓ Passed\n")

    async def test_normal_query(self):
        """Test a normal query (if LLM is available)."""
        print("Test 3 (context): Normal query")

        stream = io.StringIO()
        stream_context = io.StringIO()

        try:
            iterator = chat_stream(
                "What is a linear model?",
                None,
                self.get_workflow_context(),
                print_context=True,
                database_streams=[stream, stream_context],
                database_log=True,
            )
            result = await consume_chat_stream(iterator)

        except Exception as e:
            print(f"⚠ Skipped (LLM not available): {e}\n")
            raise Exception(
                "Error in test_validation_normal_query"
            ) from e

        self.assertTrue(len(result) > 0)
        self.assertTrue(result.startswith("CONTEXT"))
        self.assertIn("END CONTEXT--", result)

        await asyncio.sleep(0.1)

        logtext: str = stream.getvalue()
        self.assertGreater(len(logtext), 0)
        self.assertIn("MESSAGE", logtext)
        logcontext: str = stream_context.getvalue()
        self.assertGreater(len(logcontext), 0)

        print("✓ Passed\n")


if __name__ == "__main__":
    unittest.main()
