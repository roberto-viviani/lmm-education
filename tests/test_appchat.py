import unittest
import io
import asyncio
import gradio as gr
from lmm_education.apputils import async_log_factory, AsyncLogfuncType
from lmm_education.config.config import (
    ConfigSettings,
    export_settings,
)
from lmm_education.config.appchat import ChatSettings
from lmm.utils.logging import ConsoleLogger  #

original_settings = ConfigSettings()


def setUpModule():
    settings = ConfigSettings(
        major={'model': "Debug/debug"},
        minor={'model': "Debug/debug"},
        aux={'model': "Debug/debug"},
    )
    export_settings(settings)


def tearDownModule():
    export_settings(original_settings)


class TestGradioCallback(unittest.IsolatedAsyncioTestCase):

    async def test_garden_path(self):
        # import after setUpModule
        from appChat import gradio_callback_fn

        stream = io.StringIO()
        stream_context = io.StringIO()
        logfun: AsyncLogfuncType = async_log_factory(
            stream, stream_context, ConsoleLogger()
        )

        buffer: str = ""
        async for chunk in gradio_callback_fn(
            "In the field of linear models and statistics, what is logistic regression?",
            [],
            gr.Request,
            logfun,
        ):
            buffer += chunk

        self.assertGreater(len(buffer), 0)
        print(buffer)

        await asyncio.sleep(0.1)
        msg_log: str = stream.getvalue()
        ctx_log: str = stream_context.getvalue()

        self.assertIn("MESSAGE", msg_log)
        self.assertNotEqual(len(ctx_log), 0)
        self.assertEqual(msg_log.split(",")[0], ctx_log.split(",")[0])

    # This only testable after prompting a real model
    # async def test_rejection(self):
    #     # import after setUpModule
    #     from appChat import gradio_callback_fn

    #     chat_settings = ChatSettings()

    #     stream = io.StringIO()
    #     stream_context = io.StringIO()
    #     logfun: AsyncLogfuncType = async_log_factory(
    #         stream, stream_context, ConsoleLogger()
    #     )

    #     buffer: str = ""
    #     async for chunk in gradio_callback_fn(
    #         "Why is the sky blue?",
    #         [],
    #         gr.Request,
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
        from appChat import gradio_callback_fn

        chat_settings = ChatSettings()

        stream = io.StringIO()
        stream_context = io.StringIO()
        logfun: AsyncLogfuncType = async_log_factory(
            stream, stream_context, ConsoleLogger()
        )

        buffer: str = ""
        async for chunk in gradio_callback_fn(
            "",
            [],
            gr.Request,
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
        from appChat import gradio_callback_fn

        chat_settings = ChatSettings()

        stream = io.StringIO()
        stream_context = io.StringIO()
        logfun: AsyncLogfuncType = async_log_factory(
            stream, stream_context, ConsoleLogger()
        )

        buffer: str = ""
        async for chunk in gradio_callback_fn(
            "This is a long query " * 200,
            [],
            gr.Request,
            logfun,
        ):
            buffer += chunk

        self.assertGreater(len(buffer), 0)
        self.assertEqual(buffer, chat_settings.MSG_LONG_QUERY)

        await asyncio.sleep(0.1)
        msg_log: str = stream.getvalue()
        ctx_log: str = stream_context.getvalue()

        self.assertIn("LONGQUERY", msg_log)
        self.assertEqual(len(ctx_log), 0)

    async def test_vote(self):
        from appChat import vote

        stream = io.StringIO()
        stream_context = io.StringIO()
        logfun: AsyncLogfuncType = async_log_factory(
            stream, stream_context, ConsoleLogger()
        )

        data: gr.LikeData = gr.LikeData(
            None, {'index': 0, 'value': "liked"}
        )
        await vote(data, gr.Request, logfun)

        await asyncio.sleep(0.1)
        msg_log: str = stream.getvalue()
        ctx_log: str = stream_context.getvalue()

        self.assertIn("USER REACTION", msg_log)
        self.assertEqual(len(ctx_log), 0)

    async def test_postcomment(self):
        from appChat import postcomment

        stream = io.StringIO()
        stream_context = io.StringIO()
        logfun: AsyncLogfuncType = async_log_factory(
            stream, stream_context, ConsoleLogger()
        )

        comment = "This was great"
        await postcomment(comment, gr.Request, logfun)

        await asyncio.sleep(0.1)
        msg_log: str = stream.getvalue()
        ctx_log: str = stream_context.getvalue()

        self.assertIn("USER COMMENT", msg_log)
        self.assertIn(comment, msg_log)
        self.assertEqual(len(ctx_log), 0)


if __name__ == "__main__":
    unittest.main()
