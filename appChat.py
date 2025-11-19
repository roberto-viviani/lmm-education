"""
Entry point for the RAG model chat application.
"""

# ruff: noqa: E402

from datetime import datetime
import os
from typing import Literal
from collections.abc import AsyncGenerator

import gradio as gr
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

# settings. If config.toml does not exist, create it
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

# Initialize CSV files with headers if they don't exist
if not os.path.exists(DATABASE_FILE):
    with open(DATABASE_FILE, "w", encoding='utf-8') as f:
        f.write(
            "record_id,client_host,session_hash,timestamp,history_length,model_name,query,response\n"
        )

if not os.path.exists(CONTEXT_DATABASE_FILE):
    with open(CONTEXT_DATABASE_FILE, "w", encoding='utf-8') as f:
        f.write("record_id,context\n")

# Config files.
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


from collections.abc import AsyncIterator
from langchain_core.messages import BaseMessageChunk
import asyncio

# Import refactored chat functions
from lmm_education.query import (
    chat_function,
    chat_function_with_validation,
)


# Non-blocking async logging function
async def async_log_interaction(
    record_id: str,
    query: str,
    response: str,
    history: list[dict[str, str]],
    client_host: str,
    session_hash: str,
    timestamp: datetime,
    model_name: str,
) -> None:
    """
    Non-blocking logging function for query-response interactions.
    Logs to CSV files without blocking the main async flow.
    """

    def _fmat(text: str) -> str:
        # Replace double quotation marks with single quotation marks
        modified_text = text.replace('"', "'")

        # 2. Replace newline characters with " | "
        modified_text = modified_text.replace('\n', ' | ')
        return modified_text

    try:
        # Log main interaction to messages.csv
        with open(DATABASE_FILE, "a", encoding='utf-8') as f:
            f.write(
                f'{record_id},{client_host},{session_hash},'
                f'{timestamp},{len(history)},'
                f'{model_name},"{_fmat(query)}","{_fmat(response)}"\n'
            )

        # Log context if available (from developer role in history)
        if history and history[-1]['role'] == "developer":
            with open(
                CONTEXT_DATABASE_FILE, "a", encoding='utf-8'
            ) as f:
                f.write(
                    f'{record_id},"{_fmat(history[-1]['content'])}"\n'
                )

    except Exception as e:
        logger.error(f"Background logging failed: {e}")


LatexStyle = Literal["dollar", "backslash", "default", "raw"]


def _get_latex_style() -> LatexStyle:
    # centralize used latex style, perhaps depending
    # on model. Now we just use the one in tendency in
    # OpenAI, Mistral.
    return "backslash"


def _preproc_for_markdown(response: str) -> str:
    # This function allows centrailizing preprocessing
    # of strings using a latex style.
    from lmm.markdown.ioutils import (
        convert_dollar_latex_delimiters,
        convert_backslash_latex_delimiters,
    )

    match _get_latex_style():
        case "backslash":
            return convert_dollar_latex_delimiters(response)
        case "dollar":
            return convert_backslash_latex_delimiters(response)
        case _:
            return response


# Callback for Gradio to call when a chat message is sent.
async def fn(
    querytext: str, history: list[dict[str, str]], request: gr.Request
) -> AsyncGenerator[str, None]:
    """
    This function is called by the gradio framework each time the
    user posts a new message in the chatbot. The user message is the
    querytext, the history the list of previous exchanges.
    request is a wrapper around FastAPI information, used to write a
    session ID in the logs. Note we collect IP address.

    This version streams the content of the response using the refactored
    chat_function from lmm_education.query.
    """

    # Get iterator from refactored chat_function
    buffer: str = ""
    if chat_settings is None:
        raise ValueError("Unreachable code reached")
    try:
        response_iter: AsyncIterator[BaseMessageChunk] = (
            await chat_function(
                querytext=querytext,
                history=history,
                retriever=retriever,
                llm=llm,
                system_msg=chat_settings.SYSTEM_MESSAGE,
                prompt=prompt,
                logger=logger,
            )
        )

        # Stream and yield for Gradio
        async for item in response_iter:
            buffer += _preproc_for_markdown(item.text())
            yield buffer

        # Non-blocking logging hook - fires after streaming completes
        record_id = generate_random_string(8)
        model_name = getattr(llm, 'model_name', 'unknown')
        asyncio.create_task(
            async_log_interaction(
                record_id,
                querytext,
                buffer,
                history,
                request.client.host,  # type: ignore
                request.session_hash or 'unknown',
                datetime.now(),
                model_name,
            )
        )

    except Exception as e:
        logger.error(
            f"{request.client.host}: "  # type: ignore (dynamic properties)
            f"{e}\nOFFENDING QUERY:\n{querytext}\n\n"
        )
        buffer = str(e)
        yield chat_settings.MSG_ERROR_QUERY
        return

    return


async def fn_checked(
    querytext: str, history: list[dict[str, str]], request: gr.Request
) -> AsyncGenerator[str, None]:
    """
    This function is called by the gradio framework each time the
    user posts a new message in the chatbot. The user message is the
    querytext, the history the list of previous exchanges.
    request is a wrapper around FastAPI information, used to write a
    session ID in the logs. Note we collect IP address.

    This version checks the content of the stream before releasing it
    using the refactored chat_function_with_validation from lmm_education.query.
    """

    # Get validated iterator from refactored chat_function_with_validation
    buffer: str = ""
    if chat_settings is None:
        raise ValueError("Unreachable code reached")
    try:
        if settings is None:  # for the type checker
            raise ValueError("Unreacheable code reached.")
        response_iter: AsyncIterator[BaseMessageChunk] = (
            await chat_function_with_validation(
                querytext=querytext,
                history=history,
                retriever=retriever,
                llm=llm,
                system_msg=chat_settings.SYSTEM_MESSAGE,
                prompt=prompt,
                allowed_content=settings.check_response.allowed_content,
                initial_buffer_size=settings.check_response.initial_buffer_size,
                max_retries=2,
                logger=logger,
            )
        )

        # Stream and yield for Gradio
        async for item in response_iter:
            buffer += _preproc_for_markdown(item.text())
            yield buffer

        # Non-blocking logging hook - fires after streaming completes
        record_id = generate_random_string(8)
        model_name = getattr(llm, 'model_name', 'unknown')
        asyncio.create_task(
            async_log_interaction(
                record_id,
                querytext,
                buffer,
                history,
                request.client.host,  # type: ignore
                request.session_hash or 'unknown',
                datetime.now(),
                model_name,
            )
        )

    except Exception as e:
        logger.error(
            f"{request.client.host}: "  # type: ignore (dynamic properties)
            f"{e}\nOFFENDING QUERY:\n{querytext}\n\n"
        )
        buffer = str(e)
        yield chat_settings.MSG_ERROR_QUERY
        return

    return


def vote(data: gr.LikeData, request: gr.Request):
    with open(DATABASE_FILE, "a", encoding='utf-8') as f:
        if data.liked:
            f.write(
                f"USER REACTION,,{request.client.host},"  # type: ignore
                + f"{request.session_hash},approved,\n"
            )
        else:
            f.write(
                f"USER REACTION,,{request.client.host},"  # type: ignore
                + f"{request.session_hash},disapproved,\n"
            )


def clearchat() -> None:
    pass


def postcomment(comment: object, request: gr.Request):
    with open(DATABASE_FILE, "a", encoding="utf-8") as f:
        f.write(
            f'USER COMMENT,,{request.client.host},'  # type: ignore
            + f'{request.session_hash},"{comment}",\n'
        )


# create the app
ldelims: list[dict[str, str | bool]] = (
    # latex delimeters for gradio. It seems that the last
    # option should work regardless, but the exact behaviour
    # is not specified in the docs.
    [
        {"left": r"\[", "right": r"\]", "display": True},
        {"left": r"\(", "right": r"\)", "display": False},
    ]
    if _get_latex_style() == "backslash"
    else (
        [
            {"left": "$$", "right": "$$", "display": True},
            {"left": "$", "right": "$", "display": False},
        ]
        if _get_latex_style() == "dollar"
        else (
            []
            if _get_latex_style() == "raw"
            else [
                {"left": "$$", "right": "$$", "display": True},
                {"left": "$", "right": "$", "display": False},
                {"left": r"\[", "right": r"\]", "display": True},
                {"left": r"\(", "right": r"\)", "display": False},
            ]
        )
    )
)
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
        fn_checked if settings.check_response.check_response else fn,
        type="messages",
        theme="default",
        api_name=False,
        show_api=False,
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
            # server_name='85.124.80.91',  # probably not necessary
            server_port=chat_settings.server.port,
            show_api=False,
            auth=('accesstoken', 'hackerbrücke'),
        )
