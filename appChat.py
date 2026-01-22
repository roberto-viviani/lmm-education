"""
Entry point for the RAG model chat application.
"""

# ruff: noqa: E402

from datetime import datetime
import os
from collections.abc import (
    AsyncGenerator,
    Callable,
)


import gradio as gr
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

# lmm_education and lmm
from lmm_education.config.config import (
    ConfigSettings,
    load_settings,
    create_default_config_file,
    DEFAULT_CONFIG_FILE,
)
from lmm_education.config.appchat import (
    ChatSettings,
    load_settings as load_chat_settings,
    create_default_config_file as create_default_chat_config_file,
    CHAT_CONFIG_FILE,
)
from lmm_education.models.langchain.workflows.chat_graph import (
    ChatWorkflowContext,
)
from lmm.language_models.langchain.models import (
    create_model_from_settings,
)
from lmm.utils.hash import generate_random_string

# logs
import logging
from lmm.utils.logging import FileConsoleLogger  # fmt: skip

logger = FileConsoleLogger(
    "LM Markdown for Education",
    "appChat.log",
    console_level=logging.INFO,
    file_level=logging.ERROR,
)

# Config files. If config.toml does not exist, create it
if not os.path.exists(DEFAULT_CONFIG_FILE):
    create_default_config_file(DEFAULT_CONFIG_FILE)
    logger.info(
        f"{DEFAULT_CONFIG_FILE} created in app folder, change as appropriate"
    )

settings: ConfigSettings | None = load_settings()
if settings is None:
    logger.error("Could not load settings")
    exit()

if not os.path.exists(CHAT_CONFIG_FILE):
    create_default_chat_config_file(CHAT_CONFIG_FILE)
    logger.info(
        f"{CHAT_CONFIG_FILE} created in app folder, change as appropriate"
    )

chat_settings: ChatSettings | None = load_chat_settings()
if chat_settings is None:
    logger.error("Could not load chat settings")
    exit()

# This is displayed on the chatbot. Change it as appropriate
title: str = chat_settings.title
description: str = chat_settings.description

#  create retriever
from langchain_core.retrievers import BaseRetriever
from lmm_education.stores.langchain.vector_store_qdrant_langchain import (
    AsyncQdrantVectorStoreRetriever as QdrantRetriever,
)

# will return grouped retriever if appropriate
try:
    retriever: BaseRetriever = QdrantRetriever.from_config_settings()
except Exception as e:
    logger.error(f"Could not create retriever: {e}")
    exit()

try:
    llm: BaseChatModel = create_model_from_settings(settings.major)
except Exception as e:
    logger.error(f"Could not create LLM: {e}")
    exit()

# An embedding engine object is created here just to load the engine.
# This avoids the first query to take too long. The object is cached
# internally, so we do not actually use the embedding object here.
from lmm.language_models.langchain.runnables import create_embeddings
from requests import ConnectionError

try:
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

# Create dependency injection object
context = ChatWorkflowContext(
    llm=llm,
    retriever=retriever,
    chat_settings=chat_settings,
    logger=logger,
)

# Logging.
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
    preproc_markdown_factory(settings.major)
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

    # chat_settings is captured by closure, but type checker...
    if chat_settings is None:
        raise ValueError(
            "Unreachable code reached: in gradio_callback_fn"
        )

    # Create stream
    buffer: str = ""
    try:
        stream_raw: tier_1_iterator = create_chat_stream(
            querytext=querytext,
            history=history or None,
            context=context,
            validate=chat_settings.check_response,
            database_log=False,  # do downstream in on_terminal_state
            logger=logger,
        )

        stream: tier_3_iterator = terminal_field_change_adapter(
            stream_raw,
            source_nodes=["generate", "validate_query"],
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
        yield chat_settings.MSG_ERROR_QUERY
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
        yield chat_settings.MSG_ERROR_QUERY
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


# create the app-------------------------------------
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
    gr.Markdown(chat_settings.comment)
    comment = gr.Textbox(label="Comment:", submit_btn="Post comment")
    comment.submit(fn=postcomment, inputs=comment, outputs=comment)


if __name__ == "__main__":
    # run the app

    try:
        chat_settings = ChatSettings()
    except Exception as e:
        logger.error("Could not load chat settings:\n" + str(e))
        exit()

    try:
        if chat_settings.server.mode == "local":
            app.launch(
                server_port=chat_settings.server.port,
                show_api=False,
                auth=("accesstoken", "hackerbrücke"),
            )
        else:
            # allow public access on internet computer
            app.launch(
                server_name="85.124.80.91",  # keep this
                server_port=chat_settings.server.port,
                show_api=False,
                auth=("accesstoken", "hackerbrücke"),
            )
    except Exception as e:
        logger.error(f"Could not run the app: {e}")
    finally:
        logging_database.close()
