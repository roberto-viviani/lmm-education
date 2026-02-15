import unittest
import io
import asyncio
import gradio as gr
from functools import partial

from lmm_education.config.config import (
    ConfigSettings,
    export_settings,
)
from lmm_education.config.appchat import ChatSettings, CheckResponse
from lmm_education.workflows.langchain.base import (
    graph_logger,
)
from lmm_education.workflows.langchain.chat_graph import (
    ChatWorkflowContext,
)
import atexit

original_settings = ConfigSettings()
atexit.register(export_settings, original_settings)


def setUpModule():
    settings = ConfigSettings(
        major={'model': "Debug/debug"},
        minor={'model': "Debug/debug"},
        aux={'model': "Debug/debug"},
    )
    export_settings(settings)

    # An embedding engine object is created here just to load the engine.
    # This avoids the first query to take too long. The object is cached
    # internally, so we do not actually use the embedding object here.
    from lmm.models.langchain.runnables import (
        create_embeddings,
    )
    from requests import ConnectionError

    try:
        create_embeddings()
    except ConnectionError as e:
        print(
            "Could not connect to the model provider -- no internet?"
        )
        print(f"Error message:\n{e}")
        raise Exception from e
    except Exception as e:
        print(
            "Could not connect to the model provider due to a system error."
        )
        print(f"Error message:\n{e}")
        raise Exception from e


def tearDownModule():
    export_settings(original_settings)


class TestGradioCallback(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        from langchain_core.retrievers import BaseRetriever
        from lmm_education.stores.langchain.vector_store_qdrant_langchain import (
            AsyncQdrantVectorStoreRetriever as AsyncQdrantRetriever,
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
            retriever=self.retriever,
            chat_settings=chat_settings,
        )

    async def test_garden_path(self):
        # import after setUpModule
        from appChat import gradio_callback_fn, AsyncLogfunType
        from lmm_education.logging_db import CsvChatDatabase

        stream = io.StringIO()
        stream_context = io.StringIO()
        db_logger = CsvChatDatabase(
            message_stream=stream, context_stream=stream_context
        )
        logfun: AsyncLogfunType = partial(
            graph_logger,
            database=db_logger,
            context=self.get_workflow_context(),
        )

        buffer: str = ""
        query: str = (
            """In the field of linear models and statistics, what is logistic regression?"""
        )
        async for chunk in gradio_callback_fn(
            query,
            [],
            gr.Request,  # type: ignore
            logfun,
        ):
            buffer += chunk

        self.assertGreater(len(buffer), 0)

        await asyncio.sleep(0.1)
        msg_log: str = stream.getvalue()
        ctx_log: str = stream_context.getvalue()

        self.assertIn("MESSAGES", msg_log)
        self.assertIn(query, msg_log)
        self.assertNotEqual(len(ctx_log), 0)
        self.assertEqual(msg_log.split(",")[0], ctx_log.split(",")[0])

    # This only testable after prompting a real model
    # async def test_rejection(self):
    #     # import after setUpModule
    #     from appChat import gradio_callback_fn, AsyncLogfunType
    #     from lmm_education.logging_db import CsvChatDatabase

    #     stream = io.StringIO()
    #     stream_context = io.StringIO()
    #     db_logger = CsvChatDatabase(
    #         message_stream=stream, context_stream=stream_context
    #     )
    #     logfun: AsyncLogfunType = partial(
    #         graph_logger,
    #         database=db_logger,
    #         context=self.get_workflow_context(),
    #     )

    #     settings: ChatSettings = ChatSettings()

    #     buffer: str = ""
    #     async for chunk in gradio_callback_fn(
    #         "Why is the sky blue?",
    #         [],
    #         gr.Request,  # type: ignore
    #         logfun,
    #     ):
    #         buffer += chunk

    #     self.assertGreater(len(buffer), 0)
    #     self.assertEqual(buffer, settings.MSG_WRONG_CONTENT)

    #     await asyncio.sleep(0.1)
    #     msg_log: str = stream.getvalue()
    #     ctx_log: str = stream_context.getvalue()

    #     self.assertIn("REJECTION", msg_log)
    #     self.assertNotEqual(len(ctx_log), 0)
    #     self.assertEqual(msg_log.split(",")[0], ctx_log.split(",")[0])

    async def test_empty_message(self):
        # import after setUpModule
        from appChat import gradio_callback_fn, AsyncLogfunType
        from lmm_education.logging_db import CsvChatDatabase

        chat_settings = ChatSettings()

        stream = io.StringIO()
        stream_context = io.StringIO()
        db_logger = CsvChatDatabase(
            message_stream=stream, context_stream=stream_context
        )
        logfun: AsyncLogfunType = partial(
            graph_logger,
            database=db_logger,
            context=self.get_workflow_context(),
        )

        buffer: str = ""
        async for chunk in gradio_callback_fn(
            "",
            [],
            gr.Request,  # type: ignore
            logfun,
        ):
            buffer += chunk

        self.assertGreater(len(buffer), 0)
        self.assertEqual(buffer, chat_settings.MSG_EMPTY_QUERY)

        await asyncio.sleep(0.1)
        msg_log: str = stream.getvalue()
        ctx_log: str = stream_context.getvalue()

        self.assertIn("EMPTYQUERY", msg_log)
        self.assertEqual(len(ctx_log), 0)

    async def test_long_query(self):
        # import after setUpModule
        from appChat import gradio_callback_fn, AsyncLogfunType
        from lmm_education.logging_db import CsvChatDatabase

        chat_settings = ChatSettings()

        stream = io.StringIO()
        stream_context = io.StringIO()
        db_logger = CsvChatDatabase(
            message_stream=stream, context_stream=stream_context
        )
        logfun: AsyncLogfunType = partial(
            graph_logger,
            database=db_logger,
            context=self.get_workflow_context(),
        )

        buffer: str = ""
        async for chunk in gradio_callback_fn(
            "This is a long query " * 200,
            [],
            gr.Request,  # type: ignore
            logfun,
        ):
            buffer += chunk

        self.assertGreater(len(buffer), 0)
        self.assertEqual(buffer, chat_settings.MSG_LONG_QUERY)

        await asyncio.sleep(0.1)
        msg_log: str = stream.getvalue()
        ctx_log: str = stream_context.getvalue()

        print(msg_log)

        self.assertIn("LONGQUERY", msg_log)
        self.assertEqual(len(ctx_log), 0)

    async def test_vote(self):
        from appChat import vote
        from lmm_education.logging_db import CsvChatDatabase

        stream = io.StringIO()
        stream_context = io.StringIO()
        db_logger = CsvChatDatabase(
            message_stream=stream, context_stream=stream_context
        )

        data: gr.LikeData = gr.LikeData(
            None, {'index': 0, 'value': "liked"}
        )
        await vote(data, gr.Request, db_logger)  # type: ignore

        await asyncio.sleep(0.1)
        msg_log: str = stream.getvalue()
        ctx_log: str = stream_context.getvalue()

        self.assertIn("USER REACTION", msg_log)
        self.assertEqual(len(ctx_log), 0)

    async def test_postcomment(self):
        from appChat import postcomment
        from lmm_education.logging_db import CsvChatDatabase

        stream = io.StringIO()
        stream_context = io.StringIO()
        db_logger = CsvChatDatabase(
            message_stream=stream, context_stream=stream_context
        )

        comment = "This was great"
        request = gr.Request
        await postcomment(comment, request, db_logger)  # type: ignore

        await asyncio.sleep(0.1)
        msg_log: str = stream.getvalue()
        ctx_log: str = stream_context.getvalue()

        self.assertIn("USER COMMENT", msg_log)
        self.assertIn(comment, msg_log)
        self.assertEqual(len(ctx_log), 0)


if __name__ == "__main__":
    unittest.main()
