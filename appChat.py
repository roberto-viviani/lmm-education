"""
Entry point for the RAG model chat application.
"""

# ruff: noqa: E402

from datetime import datetime
from collections.abc import (
    AsyncGenerator,
    Callable,
)

# Libraries
import gradio as gr
from langchain_core.language_models import BaseChatModel

# lmm_education and lmm
from lmm_education.config.appchat import (
    ChatSettings,
)
from lmm_education.models.langchain.workflows.chat_graph import (
    ChatWorkflowContext,
)
from lmm.language_models.langchain.models import (
    create_model_from_settings,
)
from lmm.utils.hash import generate_random_string

# Logging of info and errors. Set up first to allow
# logging errors suring the rest of the setup.
import logging
from lmm.utils.logging import FileConsoleLogger  # fmt: skip

logger = FileConsoleLogger(
    "LM Markdown for Education",
    "appChat.log",
    console_level=logging.INFO,
    file_level=logging.ERROR,
)

# The appbase module centralizes set up of all app-modules such as
# the present one.
try:
    from lmm_education.appbase import appbase as base
except Exception as e:
    logger.error(f"Could not start application. {e}")
    exit()

# This is displayed on the chatbot, as specified in the chat
# config file.
title: str = base.chat_settings.title
description: str = base.chat_settings.description

# We now set up the major language model used for chatting.
# By default, we use the 'major' category set up in config.toml,
# but this may be changed manually here.
try:
    llm: BaseChatModel = create_model_from_settings(
        base.settings.major
    )
except Exception as e:
    logger.error(f"Could not create LLM: {e}")
    exit()

# Create dependency injection object. The graph uses a dependency
# injection object, which we load with the objects created at setup.
context = ChatWorkflowContext(
    llm=llm,
    retriever=base.retriever,
    chat_settings=base.chat_settings,
    logger=logger,
)

# -Database------------------------------------------------------
# Logging of exchange in database. The exchange may be graph-
# -specific: we load graph_logger from the same graph definition
# we will be using later.
from lmm_education.logging_db import ChatDatabaseInterface
from lmm_education.models.langchain.workflows.chat_graph import (
    graph_logger,
)
from typing import Any
from collections.abc import Coroutine
from functools import partial
import atexit

# We use the database as specified in config.toml.
logging_database: ChatDatabaseInterface
logging_database = ChatDatabaseInterface.from_config()

# Register cleanup handler to ensure files are closed
# on exit. The database will already be closed if the
# app is run through main.
atexit.register(logging_database.close)

AsyncLogfunType = partial[Coroutine[Any, Any, None]]
async_log_partial: AsyncLogfunType = partial(
    graph_logger, database=logging_database, context=context
)


# Chat functions to use in gradio callback------------------
from lmm_education.query import create_chat_stream
from lmm_education.models.langchain.workflows.stream_adapters import (
    tier_1_iterator,
    tier_3_iterator,
    terminal_field_change_adapter,
)

# centralize used latex style, perhaps depending
# on model.
from lmm_education.apputils import preproc_markdown_factory

_preproc_for_markdown: Callable[[str], str] = (
    preproc_markdown_factory(base.settings.major)
)


# Callback for Gradio to call when a chat message is sent.
async def gradio_callback_fn(
    querytext: str,
    history: list[dict[str, str]],
    request: gr.Request,
    async_log: AsyncLogfunType | None = None,
) -> AsyncGenerator[str, None]:
    """
    This function is called by the gradio framework each time the
    user posts a new message in the chatbot. The user message is the
    querytext, the history the list of previous exchanges.
    request is a wrapper around FastAPI information, used to write a
    session ID in the logs. Note we collect IP address.

    The content of the response is streamed using the appropriate
    chat_function from lmm_education.query.
    """

    # this only to override closure in testing
    if async_log is None:
        async_log = async_log_partial

    # Safely extract client host and session hash
    client_host: str = getattr(
        getattr(request, "client", None), "host", "[in-process]"
    )
    session_hash: str = getattr(request, "session_hash", "[none]")

    # Create stream
    buffer: str = ""
    try:
        stream_raw: tier_1_iterator = create_chat_stream(
            querytext=querytext,
            history=history or None,
            context=context,
            validate=context.chat_settings.check_response,
            database_log=False,  # do downstream in on_terminal_state
            logger=logger,
        )

        stream: tier_3_iterator = terminal_field_change_adapter(
            stream_raw,
            on_terminal_state=partial(
                async_log,
                client_host=client_host,
                session_hash=session_hash,
                timestamp=None,  # will be set at time of msg
                record_id=None,  # handled by logger
            ),
        )
    except Exception as e:
        logger.error(f"Could not create stream: {e}")
        yield context.chat_settings.MSG_ERROR_QUERY
        return

    try:
        # Stream and yield for Gradio
        async for item in stream:
            buffer += _preproc_for_markdown(item)
            yield buffer

    except Exception as e:
        logger.error(
            f"{client_host}: {e}\nOFFENDING QUERY:\n{querytext}\n\n"
        )
        buffer = str(e)
        yield context.chat_settings.MSG_ERROR_QUERY
        return

    return


async def vote(
    data: gr.LikeData,
    request: gr.Request,
    logging_db: ChatDatabaseInterface | None = None,
):
    """
    Async function to log user reactions (like/dislike) to messages.
    """
    # this to override closure for testing purposes
    if logging_db is None:
        logging_db = logging_database

    record_id = generate_random_string()
    reaction = "approved" if data.liked else "disapproved"

    # Safely extract client host and session hash
    client_host: str = getattr(
        getattr(request, "client", None), "host", "unknown"
    )
    session_hash: str = getattr(request, "session_hash", "unknown")

    logging_db.schedule_message(
        record_id=record_id,
        client_host=client_host,
        session_hash=session_hash,
        timestamp=datetime.now(),
        message_count=0,
        model_name="",
        interaction_type="USER REACTION",
        query=reaction,
        response="",
    )


def clearchat() -> None:
    pass


async def postcomment(
    comment: object,
    request: gr.Request,
    logging_db: ChatDatabaseInterface | None = None,
):
    """
    Async function to log user comments.
    """
    # this to override closure for testing purposes
    if logging_db is None:
        logging_db = logging_database

    record_id = generate_random_string()

    # Safely extract client host and session hash
    client_host: str = getattr(
        getattr(request, "client", None), "host", "unknown"
    )
    session_hash: str = getattr(request, "session_hash", "unknown")

    logging_db.schedule_message(
        record_id=record_id,
        client_host=client_host,
        session_hash=session_hash,
        timestamp=datetime.now(),
        message_count=0,
        model_name="",
        interaction_type="USER COMMENT",
        query=str(comment),
        response="",
    )


# Gradio app definition ---------------------------------------------
# latex delimeters for gradio.
ldelims: list[dict[str, str | bool]] = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "$", "right": "$", "display": False},
    {"left": r"\[", "right": r"\]", "display": True},
    {"left": r"\(", "right": r"\)", "display": False},
]

with gr.Blocks() as app:
    gr.Markdown("# " + title)
    gr.Markdown(description)
    chatbot = gr.Chatbot(
        type="messages",
        latex_delimiters=ldelims,
        show_copy_all_button=True,
        layout="panel",
    )
    chatbot.like(vote, None, None)
    chatbot.clear(clearchat)
    gr.ChatInterface(
        gradio_callback_fn,
        type="messages",
        theme="default",
        api_name=False,
        show_api=False,
        save_history=True,
        chatbot=chatbot,
    )
    gr.Markdown(base.chat_settings.comment)
    comment = gr.Textbox(label="Comment:", submit_btn="Post comment")
    comment.submit(fn=postcomment, inputs=comment, outputs=comment)


if __name__ == "__main__":
    # run the app

    settings: ChatSettings = base.chat_settings
    try:
        if settings.server.mode == "local":
            app.launch(
                server_port=settings.server.port,
                show_api=False,
                auth=("accesstoken", "hackerbrücke"),
            )
        else:
            # allow public access on internet computer
            app.launch(
                server_name="85.124.80.91",  # keep this
                server_port=settings.server.port,
                show_api=False,
                auth=("accesstoken", "hackerbrücke"),
            )
    except Exception as e:
        logger.error(f"Could not run the app: {e}")
    finally:
        logging_database.close()
