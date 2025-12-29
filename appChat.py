"""
Entry point for the RAG model chat application.
"""

# ruff: noqa: E402

from datetime import datetime
import os
from collections.abc import AsyncGenerator, AsyncIterator, Callable
import asyncio


import gradio as gr
from langchain_core.messages import BaseMessageChunk
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
from lmm_education.apputils import async_log_factory, AsyncLogfuncType
from lmm.utils.hash import generate_random_string

# logs
import logging
from lmm.utils.logging import FileConsoleLogger # fmt: skip
logger = FileConsoleLogger(
    "LM Markdown for Education",
    "appChat.log",
    console_level=logging.INFO,
    file_level=logging.ERROR,
)
DATABASE_FILE = "messages.csv"
CONTEXT_DATABASE_FILE = "queries.csv"

# Initialize CSV database files with headers if they don't exist
if not os.path.exists(DATABASE_FILE):
    with open(DATABASE_FILE, "w", encoding='utf-8') as f:
        f.write(
            "record_id,client_host,session_hash,timestamp,history_length,model_name,interaction_type,query,response\n"
        )

if not os.path.exists(CONTEXT_DATABASE_FILE):
    with open(CONTEXT_DATABASE_FILE, "w", encoding='utf-8') as f:
        f.write("record_id,evaluation,context,classification\n")

# Set up a global container of active tasks (used by logging)
active_logs: set[asyncio.Task[None]] = set()
async_log: AsyncLogfuncType = async_log_factory(
    DATABASE_FILE, CONTEXT_DATABASE_FILE, logger
)

# Config files. If config.toml does not exist, create it
if not os.path.exists(DEFAULT_CONFIG_FILE):
    create_default_config_file(DEFAULT_CONFIG_FILE)
    print(
        f"{DEFAULT_CONFIG_FILE} created in app folder, change as appropriate"
    )

settings: ConfigSettings | None = load_settings()
if settings is None:
    exit()

if not os.path.exists(CHAT_CONFIG_FILE):
    create_default_chat_config_file(CHAT_CONFIG_FILE)
    print(
        f"{CHAT_CONFIG_FILE} created in app folder, change as appropriate"
    )

chat_settings: ChatSettings | None = load_chat_settings()
if chat_settings is None:
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
retriever: BaseRetriever = QdrantRetriever.from_config_settings()

# Create chat engine.
from langchain_core.prompts import PromptTemplate # fmt: skip
prompt: PromptTemplate = PromptTemplate.from_template(
    chat_settings.PROMPT_TEMPLATE
)

from lmm.language_models.langchain.models import (
    create_model_from_settings,
)

llm: BaseChatModel = create_model_from_settings(settings.major)

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


# Chat functions to use in gradio callback
from lmm_education.query import (
    chat_function,
    chat_function_with_validation,
)
from typing import Any
from collections.abc import Coroutine

AsyncChatfuncType = Callable[
    ..., Coroutine[Any, Any, AsyncIterator[BaseMessageChunk]]
]
_chat_function: AsyncChatfuncType

if chat_settings.check_response.check_response:
    _chat_function = chat_function_with_validation
else:
    _chat_function = chat_function

# centralize used latex style, perhaps depending
# on model.
from lmm_education.apputils import preproc_markdown_factory

_preproc_for_markdown: Callable[[str], str] = (
    preproc_markdown_factory(settings.major)
)


# Callback for Gradio to call when a chat message is sent.
async def gradio_callback_fn(
    querytext: str, history: list[dict[str, str]], request: gr.Request,
    async_log: AsyncLogfuncType=async_log,
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

    # Get iterator from refactored chat_function
    buffer: str = ""
    if chat_settings is None:
        raise ValueError("Unreachable code reached")
    try:
        response_iter: AsyncIterator[BaseMessageChunk] = (
            await _chat_function(
                querytext=querytext,
                history=history,
                retriever=retriever,
                llm=llm,
                chat_settings=chat_settings,
                system_msg=chat_settings.SYSTEM_MESSAGE,
                logger=logger,
            )
        )

        # Stream and yield for Gradio
        async for item in response_iter:
            buffer += _preproc_for_markdown(item.text())
            yield buffer

        # Non-blocking logging hook - fires after streaming completes
        record_id: str = generate_random_string(8)
        model_name: str = settings.major.get_model_name()  # type: ignore
        logtask: asyncio.Task[None] = asyncio.create_task(  # type: ignore (pyright confused)
            async_log(
                record_id=record_id,
                client_host=request.client.host,  # type: ignore
                session_hash=request.session_hash or 'unknown',
                timestamp=datetime.now(),
                interaction_type="MESSAGE",
                history=history,
                query=querytext,
                response=buffer,
                model_name=model_name,
            )
        )
        active_logs.add(logtask)
        logtask.add_done_callback(active_logs.discard)

    except Exception as e:
        logger.error(
            f"{request.client.host}: "  # type: ignore (dynamic properties)
            f"{e}\nOFFENDING QUERY:\n{querytext}\n\n"
        )
        buffer = str(e)
        yield chat_settings.MSG_ERROR_QUERY
        return

    return


async def vote(data: gr.LikeData, request: gr.Request):
    """
    Async function to log user reactions (like/dislike) to messages.
    """
    record_id = generate_random_string(8)
    reaction = "approved" if data.liked else "disapproved"

    task: asyncio.Task[None] = asyncio.create_task(  # type: ignore (pyright confused)
        async_log(
            record_id=record_id,
            client_host=request.client.host,  # type: ignore
            session_hash=request.session_hash or 'unknown',
            timestamp=datetime.now(),
            interaction_type="USER REACTION",
            history=[],
            query=reaction,
            response="",
            model_name="",
        )
    )
    active_logs.add(task)
    task.add_done_callback(active_logs.discard)


def clearchat() -> None:
    pass


async def postcomment(comment: object, request: gr.Request):
    """
    Async function to log user comments.
    """
    record_id = generate_random_string(8)

    task: asyncio.Task[None] = asyncio.create_task(  # type: ignore (pyright confused)
        async_log(
            record_id=record_id,
            client_host=request.client.host,  # type: ignore
            session_hash=request.session_hash or 'unknown',
            timestamp=datetime.now(),
            interaction_type="USER COMMENT",
            history=[],
            query=str(comment),
            response="",
            model_name="",
        )
    )
    active_logs.add(task)
    task.add_done_callback(active_logs.discard)


# create the app
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

    if chat_settings.server.mode == "local":
        app.launch(
            server_port=chat_settings.server.port,
            show_api=False,
            auth=('accesstoken', 'hackerbrücke'),
        )
    else:
        # allow public access on internet computer
        app.launch(
            server_name='85.124.80.91',  # keep this
            server_port=chat_settings.server.port,
            show_api=False,
            auth=('accesstoken', 'hackerbrücke'),
        )

    # cleanup - asyncio diagnostics after Gradio closes
    from lmm_education.apputils import shutdown

    # Check if there's a running loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop - just use asyncio.run()
        asyncio.run(shutdown(active_logs))
    else:
        # Loop is running - use it directly
        if loop.is_running():
            import nest_asyncio  # type: ignore[reportMissingTypeStubs]

            nest_asyncio.apply()  # type: ignore
            loop.run_until_complete(shutdown(active_logs))
        else:
            asyncio.run(shutdown(active_logs))
