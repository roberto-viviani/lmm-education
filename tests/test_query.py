"""Test the refactored chat_function_with_validation."""

# ruff: noqa: E402

import unittest
import asyncio
import io
import logging
from collections.abc import AsyncIterator

from langchain_core.retrievers import BaseRetriever
from lmm_education.query import (
    create_chat_stringstream,
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

# Control whether to run tests that call real LLMs (which incur API costs)
# Set environment variable RUN_EXPENSIVE_TESTS=1 to enable these tests
# Example: RUN_EXPENSIVE_TESTS=1 poetry run pytest tests/test_chatagent.py
import os

RUN_EXPENSIVE_TESTS = os.getenv("RUN_EXPENSIVE_TESTS", "0") == "1"

# Decorator to skip tests that call real LLMs unless explicitly enabled
skip_if_expensive = unittest.skipUnless(
    RUN_EXPENSIVE_TESTS,
    "Skipping expensive test that calls real LLM API. "
    "Set RUN_EXPENSIVE_TESTS=1 to run.",
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

import atexit

original_settings = ConfigSettings()
atexit.register(export_settings, original_settings)


def _print_state(state: dict[str, str]) -> None:  # type: ignore (not accessed)
    for fld in state.keys():
        print(
            f"{fld}: {state[fld][:22]}{'...' if len(state[fld]) > 21 else ''}"
        )


async def consume_create_chat_stream(
    iterator: AsyncIterator[str],
) -> str:
    """
    Consumes an async iterator of BaseMessageChunk objects and returns
    the complete response as a string.

    This function is designed to work with the iterator returned by
    create_chat_stringstream. It accumulates the text content from each chunk
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
        major={"model": "Debug/debug"},
        minor={"model": "Debug/debug"},
        aux={"model": "Debug/debug"},
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
            retriever=self.retriever,
            chat_settings=chat_settings,
        )

    async def test_empty_query(self):
        """Test that empty query returns error iterator."""
        print("Test 1: Empty query")

        iterator = create_chat_stringstream(
            "",
            [],
            self.get_workflow_context(),
        )
        result = await consume_create_chat_stream(iterator)
        print(f"Result: {result}")
        self.assertIn("If you have questions", result)
        print("✓ Passed\n")

    async def test_long_query(self):
        """Test that overly long query returns error iterator."""
        print("Test 2: Long query")
        long_query = " ".join(["word"] * 200)

        iterator = create_chat_stringstream(
            long_query,
            [],
            self.get_workflow_context(),
        )
        result = await consume_create_chat_stream(iterator)
        print(f"Result: {result}")
        self.assertIn("too long", result)
        print("✓ Passed\n")

    async def test_normal_query(self):
        """Test a normal query (if LLM is available)."""
        print("Test 3: Normal query")
        try:
            iterator = create_chat_stringstream(
                "What is a linear model?",
                None,
                self.get_workflow_context(),
            )
            result = await consume_create_chat_stream(iterator)
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

            iterator = create_chat_stringstream(
                "What is a linear model?",
                None,
                context,
            )
            result = await consume_create_chat_stream(iterator)
            print(f"Result length: {len(result)} characters")
            print(f"First 100 chars: {result[:100]}...")
            self.assertTrue(len(result) > 0)

            iterator = create_chat_stringstream(
                "What is a logistic regression?",
                [],
                context,
            )
            result = await consume_create_chat_stream(iterator)
            print(f"Result length: {len(result)} characters")
            print(f"First 100 chars: {result[:100]}...")
            # please close
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


class TestQueryMalformed(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.retriever: BaseRetriever = (
            AsyncQdrantRetriever.from_config_settings()
        )

    def get_workflow_context(
        self, chat_settings: ChatSettings = ChatSettings()
    ) -> ChatWorkflowContext:
        return ChatWorkflowContext(
            retriever=self.retriever,
            chat_settings=chat_settings,
        )

    async def test_validation_empty_query(self):
        """Test that empty query returns error iterator."""
        print("Test 1 (validation): Empty query with validation")
        chat_settings = ChatSettings(
            check_response=CheckResponse(
                check_response=True,
                allowed_content=["statistics"],
            )
        )
        context = self.get_workflow_context(chat_settings)
        # chat_function_with_validation is async generator - call directly without await
        iterator = create_chat_stringstream(
            "",
            [],
            context,
            # validate defaults to None, which uses context
        )
        result = await consume_create_chat_stream(iterator)
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
                check_response=True, allowed_content=["statistics"]
            )
        )
        context = self.get_workflow_context(chat_settings)
        # chat_function_with_validation is async generator - call directly without await
        iterator = create_chat_stringstream(
            long_query,
            [],
            context,
            validate=None,  # use context
            logger=logger,
        )
        if logger.count_logs(level=logging.WARNING) > 0:
            print("\n".join(logger.get_logs()))
        self.assertEqual(logger.count_logs(level=logging.WARNING), 0)
        result = await consume_create_chat_stream(iterator)
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
                    allowed_content=["statistics"],
                    initial_buffer_size=120,
                )
            )
            # chat_function_with_validation is async generator - call directly without await
            context = self.get_workflow_context(chat_settings)
            iterator = create_chat_stringstream(
                "What is a linear model?",
                [],
                context,
                validate=chat_settings.check_response,
            )
            result = await consume_create_chat_stream(iterator)
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
            retriever=self.retriever,
            chat_settings=chat_settings,
        )

    async def test_empty_query(self):
        """Test that empty query returns error iterator."""
        print("Test 1 (log): Empty query")

        stream = io.StringIO()
        stream_context = io.StringIO()

        iterator = create_chat_stringstream(
            "",
            [],
            self.get_workflow_context(),
            database_log=(stream, stream_context),
        )
        result = await consume_create_chat_stream(iterator)
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

        iterator = create_chat_stringstream(
            long_query,
            [],
            self.get_workflow_context(),
            database_log=(stream, stream_context),
        )
        result = await consume_create_chat_stream(iterator)
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
            iterator = create_chat_stringstream(
                "What is a linear model?",
                None,
                self.get_workflow_context(),
                database_log=(stream, stream_context),
            )
            result = await consume_create_chat_stream(iterator)
        except Exception as e:
            print(f"⚠ Skipped (LLM not available): {e}\n")
            raise Exception(
                "Error in test_validation_normal_query"
            ) from e

        self.assertTrue(len(result) > 0)

        await asyncio.sleep(0.1)

        logtext: str = stream.getvalue()
        print(f"LOGTEXT\n{logtext[:120]}...")
        self.assertGreater(len(logtext), 0)
        self.assertIn("MESSAGE", logtext)
        logcontext: str = stream_context.getvalue()
        print(f"LOGCONTEXT\n{logcontext[:120]}...")
        self.assertGreater(len(logcontext), 0)

        print("✓ Passed\n")

    async def test_normal_query_nolog(self):
        """Test a normal query (if LLM is available)."""
        print("Test 3 (log): Normal no log")

        stream = io.StringIO()
        stream_context = io.StringIO()

        try:
            iterator = create_chat_stringstream(
                "What is a linear model?",
                None,
                self.get_workflow_context(),
            )
            result = await consume_create_chat_stream(iterator)

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

    async def test_normal_appchat_query(self):
        """Test a normal query using the appChat stream
        configuration"""
        print("Test 3 (log): Appchat query")
        from lmm_education.query import create_chat_stream
        from lmm_education.workflows.langchain.stream_adapters import (
            tier_1_iterator,
            tier_3_iterator,
            terminal_field_change_adapter,
        )
        from lmm_education.logging_db import (
            ChatDatabaseInterface,
            CsvChatDatabase,
        )
        from functools import partial
        from lmm_education.workflows.langchain.base import (
            graph_logger,
        )

        stream = io.StringIO()
        stream_context = io.StringIO()
        database: ChatDatabaseInterface = CsvChatDatabase(
            stream, stream_context
        )
        log_function = partial(
            graph_logger,
            database=database,
            context=self.get_workflow_context(),
        )

        try:
            stream_raw: tier_1_iterator = create_chat_stream(
                "What is a linear model?",
                None,
                self.get_workflow_context(),
                database_log=(stream, stream_context),
            )

            final_stream: tier_3_iterator = (
                terminal_field_change_adapter(
                    stream_raw,
                    on_terminal_state=partial(
                        log_function,
                        client_host="unknown",
                        session_hash="none",
                        timestamp=None,  # will be set at time of msg
                        record_id=None,  # handled by logger
                    ),
                )
            )
            result = await consume_create_chat_stream(final_stream)
        except Exception as e:
            print(f"⚠ Skipped (LLM not available): {e}\n")
            raise Exception(
                "Error in test_validation_normal_query"
            ) from e

        self.assertTrue(len(result) > 0)

        await asyncio.sleep(0.1)

        logtext: str = stream.getvalue()
        print(f"LOGTEXT\n{logtext[:120]}...")
        self.assertIn("MESSAGE", logtext)
        logcontext: str = stream_context.getvalue()
        print(f"LOGCONTEXT\n{logcontext[:120]}...")
        self.assertGreater(len(logcontext), 0)

        print("✓ Passed\n")


class TestQueryRejection(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
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
            retriever=self.retriever,
            chat_settings=chat_settings,
        )

    @skip_if_expensive
    async def test_valid_query(self):
        """Test a normal query (if LLM is available) with
        validation."""
        print("Test query validation: Valid query")

        from lmm_education.workflows.langchain.stream_adapters import (
            tier_1_iterator,
        )
        from lmm_education.workflows.langchain.chat_stream_adapters import (
            stateful_validation_adapter,
        )
        from lmm_education.query import (
            create_chat_stream,
            create_initial_state,
        )
        from lmm_education.workflows.langchain.chat_graph import (
            ChatState,
        )
        from lmm.language_models.langchain.runnables import (
            create_runnable,
        )

        stream = io.StringIO()
        stream_context = io.StringIO()

        try:
            iterator_raw: tier_1_iterator = create_chat_stream(
                "What is a linear model?",
                None,
                self.get_workflow_context(),
                database_log=(stream, stream_context),
            )
            iterator: tier_1_iterator = stateful_validation_adapter(
                iterator_raw,
                validator_model=create_runnable(
                    "allowed_content_validator",
                    user_settings={"model": "OpenAI/gpt-4.1-nano"},
                    system_prompt="You are a helpful assistant",
                    allowed_content=["statistics", "R programming"],
                ),
                allowed_content=["statistics", "R programming"],
            )

            state: ChatState = create_initial_state(
                "What is a linear model?"
            )
            counter = 0
            async for mode, event in iterator:
                if mode == "values":
                    counter += 1
                    state = event

            print("\n=========================")
            print(f"status: {state["status"]}")
            print(f"classification: {state["query_classification"]}")
            print("--------------------------\n")

        except Exception as e:
            print(
                f"Error in TestQueryRejection.test_valid_query: {e}\n"
            )
            return

        self.assertGreater(counter, 0)
        self.assertEqual(state["status"], "valid")

        await asyncio.sleep(0.1)

        logtext: str = stream.getvalue()
        self.assertGreater(len(logtext), 0)
        self.assertIn("MESSAGE", logtext)
        logcontext: str = stream_context.getvalue()
        self.assertGreater(len(logcontext), 0)

        print("✓ Passed\n")

    @skip_if_expensive
    async def test_invalid_query(self):
        """Test an invalid query (if LLM is available) with
        rejection"""
        print("Test query validation: Invalid query")

        from lmm_education.workflows.langchain.stream_adapters import (
            tier_1_iterator,
        )
        from lmm_education.workflows.langchain.chat_stream_adapters import (
            stateful_validation_adapter,
        )
        from lmm_education.query import (
            create_chat_stream,
            create_initial_state,
        )
        from lmm_education.workflows.langchain.chat_graph import (
            ChatState,
        )
        from lmm.language_models.langchain.runnables import (
            create_runnable,
        )

        stream = io.StringIO()
        stream_context = io.StringIO()

        try:
            iterator_raw: tier_1_iterator = create_chat_stream(
                "Why is the sky blue?",
                None,
                self.get_workflow_context(),
                database_log=(stream, stream_context),
            )
            iterator: tier_1_iterator = stateful_validation_adapter(
                iterator_raw,
                validator_model=create_runnable(
                    "allowed_content_validator",
                    user_settings={"model": "OpenAI/gpt-4.1-nano"},
                    system_prompt="You are a helpful assistant",
                    allowed_content=["statistics", "R programming"],
                ),
                allowed_content=["statistics", "R programming"],
                error_message="This content cannot be streamed",
            )

            state: ChatState = create_initial_state(
                "What is a linear model?"
            )
            counter = 0
            async for mode, event in iterator:
                if mode == "values":
                    counter += counter
                    state = event

            print("\n=========================")
            print(f"status: {state["status"]}")
            print(f"classification: {state["query_classification"]}")
            print("--------------------------\n")

        except Exception as e:
            print(
                f"Error in TestQueryRejection.test_invalid_query: {e}\n"
            )
            return

        self.assertEqual(state["status"], "rejected")

        # note that here the streams do not contain the rejection,
        # because the validation stream has been applied after the
        # logging stream. You need a new model with a real aux model

        print("✓ Passed\n")


class TestQueryPrintContext(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
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
            retriever=self.retriever,
            chat_settings=chat_settings,
        )

    async def test_empty_query(self):
        """Test that empty query returns error iterator."""
        print("Test 1 (context): Empty query")

        iterator = create_chat_stringstream(
            "",
            [],
            self.get_workflow_context(),
            print_context=True,
        )
        result = await consume_create_chat_stream(iterator)
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
            iterator = create_chat_stringstream(
                "What is a linear model?",
                None,
                self.get_workflow_context(),
                print_context=True,
                database_log=(stream, stream_context),
            )
            result = await consume_create_chat_stream(iterator)

        except Exception as e:
            print(f"⚠ Skipped (LLM not available): {e}\n")
            raise Exception(
                "Error in test_validation_normal_query"
            ) from e

        self.assertTrue(len(result) > 0)
        print(f"CONTEXT\n{result[:120]}...")
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
