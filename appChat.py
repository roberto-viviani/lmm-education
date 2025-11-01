"""
Entry point for the RAG model chat application.
"""

from datetime import datetime
import re
from typing import AsyncGenerator

import gradio as gr

# This is displayed on the chatbot. Change it as appropriate
title = "VU720201 Study Assistant"
description = """
Study assistant chatbot for VU Specific Scientific Methods 
in Psychology: Data analysis with linear models in R. 
Ask a question about the course, and the assistant will provide a 
response based on it. 
Example: "How can I fit a model with kid_score as outcome and mom_iq as predictor?" 
"""

# internationalization
MSG_EMPTY_QUERY = "Please ask a question about the course."
MSG_WRONG_CONTENT = "I can only answer questions about the course."
MSG_LONG_QUERY = (
    "Your question is too long. Please ask a shorter question."
)
MSG_ERROR_QUERY = (
    "I am sorry, I cannot answer this question. Please retry."
)

# settings. If config.toml does not exist, create it
from lmm_education.config.config import (
    load_settings,
    create_default_config_file,
)
import os # fmt: skip
if not os.path.exists("config.toml"):
    create_default_config_file("config.toml")
    print("config.toml created in app folder, change as appropriate")

# this reads the settings from config.toml
settings = load_settings()
if settings is None:
    exit()

# logs
from lmm.utils.logging import FileConsoleLogger # fmt: skip
logger = FileConsoleLogger(title, "appChat.log")
DATABASE_FILE = "messages.csv"

#  create retriever
from langchain_core.retrievers import BaseRetriever
from lmm_education.stores.langchain.vector_store_qdrant_langchain import (
    AsyncQdrantVectorStoreRetriever as QdrantRetriever,
)

# will return grouped retriever if appropriate
retriever: BaseRetriever = QdrantRetriever.from_config_settings()

# Create chat engine. Modify system and user prompts as appropriate.
SYSTEM_MESSAGE = """
You are a university tutor teaching undergraduates in a statistics course 
that uses R to fit models, explaining background and guiding understanding. 
Please assist students by responding to their QUERY by using the provided CONTEXT.
If the CONTEXT does not provide information for your answer, integrate the CONTEXT
only for the use and syntax of R. Otherwise, reply that you do not have information 
to answer the query.
"""

PROMPT_TEMPLATE = """
Please answer my QUERY by using the provided CONTEXT. 
Please answer in the language of the QUERY.
---
CONTEXT: "{context}"

---
QUERY: "{query}"

---
RESPONSE:

"""

from langchain_core.prompts import PromptTemplate # fmt: skip
prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

from lmm.language_models.langchain.models import (
    create_model_from_settings,
)

llm = create_model_from_settings(settings.major)

# An embedding engine object is created here just to load the engine.
# This avoids the first query to take too long. The object is cached
# internally, so we do not actually use the embedding object here.
# This will also fail if there is no internet (SentenceTransformer
# engines fail immediately).
from lmm.language_models.langchain.runnables import create_embeddings
from requests import ConnectionError

try:
    embedding = create_embeddings()
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

# Import refactored chat functions
from lmm_education.query import (
    chat_function,
    chat_function_with_validation,
)


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

    def _preproc_for_markdown(response: str) -> str:
        # replace square brackets containing the character '\' to one
        # that is enclosed between '$$' for rendering in markdown
        response = re.sub(r"\\\[|\\\]", "$$", response)
        response = re.sub(r"\\\(|\\\)", "$", response)
        return response

    # Get iterator from refactored chat_function
    buffer: str = ""
    try:
        response_iter: AsyncIterator[BaseMessageChunk] = (
            await chat_function(
                querytext=querytext,
                history=history,
                retriever=retriever,
                llm=llm,
                system_msg=SYSTEM_MESSAGE,
                prompt=prompt,
                logger=logger,
            )
        )

        # Stream and yield for Gradio
        async for item in response_iter:
            buffer += _preproc_for_markdown(item.text())
            yield buffer

    except Exception as e:
        now = datetime.now()
        logger.error(
            f"{request.client.host}: "  # type: ignore (dynamic properties)
            f"{e}\nOFFENDING QUERY:\n{querytext}\n\n"
        )
        buffer = str(e)
        yield MSG_ERROR_QUERY
        return

    finally:  # Log
        now = datetime.now()
        try:
            with open(DATABASE_FILE, "a", encoding='utf-8') as f:
                f.write(
                    f'MESSAGES,{now},{request.client.host},'  # type:ignore
                    + f'{request.session_hash},'
                    + f'"{querytext}","RESPONSE: {buffer}"\n'
                )
        except Exception:
            logger.error(
                f"{now}: Could not write to database {DATABASE_FILE}"
            )

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

    def _preproc_for_markdown(response: str) -> str:
        # replace square brackets containing the character '\' to one
        # that is enclosed between '$$' for rendering in markdown
        response = re.sub(r"\\\[|\\\]", "$$", response)
        response = re.sub(r"\\\(|\\\)", "$", response)
        return response

    # Get validated iterator from refactored chat_function_with_validation
    buffer: str = ""
    try:
        response_iter: AsyncIterator[BaseMessageChunk] = (
            await chat_function_with_validation(
                querytext=querytext,
                history=history,
                retriever=retriever,
                llm=llm,
                system_msg=SYSTEM_MESSAGE,
                prompt=prompt,
                validation_config="check_content",
                initial_buffer_size=150,
                max_retries=2,
                logger=logger,
            )
        )

        # Stream and yield for Gradio
        async for item in response_iter:
            buffer += _preproc_for_markdown(item.text())
            yield buffer

    except Exception as e:
        now = datetime.now()
        logger.error(
            f"{request.client.host}: "  # type: ignore (dynamic properties)
            f"{e}\nOFFENDING QUERY:\n{querytext}\n\n"
        )
        buffer = str(e)
        yield MSG_ERROR_QUERY
        return

    finally:  # Log
        now = datetime.now()
        try:
            with open(DATABASE_FILE, "a", encoding='utf-8') as f:
                f.write(
                    f'MESSAGES,{now},{request.client.host},'  # type:ignore
                    + f'{request.session_hash},'
                    + f'"{querytext}","RESPONSE: {buffer}"\n'
                )
        except Exception:
            logger.error(
                f"{now}: Could not write to database {DATABASE_FILE}"
            )

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


def clearchat():
    pass


def postcomment(comment: object, request: gr.Request):
    with open(DATABASE_FILE, "a", encoding="utf-8") as f:
        f.write(
            f'USER COMMENT,,{request.client.host},'  # type: ignore
            + f'{request.session_hash},"{comment}",\n'
        )


# create the app
ldelims = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "$", "right": "$", "display": False},
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
        fn_checked,
        type="messages",
        theme="default",
        api_name=False,
        show_api=False,
        chatbot=chatbot,
    )
    gr.Markdown(
        "Please leave a comment on the response of the chatbot here"
    )
    comment = gr.Textbox(label="Comment:", submit_btn="Post comment")
    comment.submit(fn=postcomment, inputs=comment, outputs=comment)


if __name__ == "__main__":
    # run the app

    config_settings = load_settings()
    if config_settings is None:
        exit()

    if config_settings.server.mode == "local":
        app.launch(
            show_api=False, auth=('accesstoken', 'hackerbrücke')
        )
    else:
        # allow public access on internet computer
        app.launch(
            # server_name='85.124.80.91',  # probably not necessary
            server_port=config_settings.server.port,
            show_api=False,
            auth=('accesstoken', 'hackerbrücke'),
        )
