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
MSG_LONG_QUERY = (
    "Your question is too long. Please ask a shorter question."
)
MSG_ERROR_QUERY = (
    "I am sorry, I cannot answer this question. Please retry."
)

# settings. If config.toml does not exist, create it
from lmm.config.config import Settings, create_default_config_file # fmt: skip
import os # fmt: skip
if not os.path.exists("config.toml"):
    create_default_config_file("config.toml")
    print("config.toml created in app folder, change as appropriate")

settings = Settings()

# logs
from lmm.utils.logging import FileConsoleLogger # fmt: skip
logger = FileConsoleLogger(title, "appChat.log")
DATABASE_FILE = "messages.csv"

#  create retriever
from lmm_education.stores.vector_store_qdrant import (
    QdrantEmbeddingModel,
)
from lmm_education.stores.langchain.vector_store_qdrant_langchain import (
    AsyncQdrantVectorStoreRetrieverGrouped as QdrantRetriever,
)
from qdrant_client import AsyncQdrantClient

PERSIST_DIR = "./storage"
COLLECTION_CHUNKS = "chunks"
COLLECTION_DOCUMENTS = "documents"
GROUP_FOREIGN_KEY = "doc_id"
LIMIT_GROUPS = 4
LIMIT_CHUNKS = 1
retriever = QdrantRetriever(
    AsyncQdrantClient(path=PERSIST_DIR),
    COLLECTION_CHUNKS,
    COLLECTION_DOCUMENTS,
    GROUP_FOREIGN_KEY,
    LIMIT_GROUPS,
    QdrantEmbeddingModel.MULTIVECTOR,
)

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


# Callback for Gradio to call when a chat message is sent.
def _prepare_messages(
    query: str,
    history: list[dict[str, str]],
    system_message: str = "",
) -> list[tuple[str, str]]:
    """
    Customized message history for stateless interaction. The first
    element is always the system message. There follow the first
    request and its response (the first request contains the context),
    and the last query and response.
    """

    messages: list[tuple[str, str]] = []
    if system_message:
        messages.append(('system', system_message))

    # first two messages are user's and assistant's
    if len(history) > 1:
        for m in history[:2]:
            messages.append((m['role'], m['content']))

    # let us resend the last two messages' exchange
    if len(history) > 3:
        for m in history[len(history) - 2 :]:
            messages.append((m['role'], m['content']))
    elif len(history) > 2:  # not clear when this would occur
        for m in history[-1:]:
            messages.append((m['role'], m['content']))
    else:
        pass

    messages.append(('user', query))

    return messages


from collections.abc import AsyncIterator
from langchain_core.messages import BaseMessageChunk
from langchain_core.documents import Document


async def fn(
    querytext: str, history: list[dict[str, str]], request: gr.Request
) -> AsyncGenerator[str, None]:
    """This function is called by the gradio framework each time the
    user posts a new message in the chatbot. The user message is the
    querytext, the history the list of previous exchanges.
    request is a wrapper around FastAPI information, used to write a
    session ID in the logs. Note we collect IP address."""

    # the max allowed number of words in the user's query
    MAX_QUERY_LENGTH: int = 60

    def _preproc_for_markdown(response: str) -> str:
        # replace square brackets containing the character '\' to one
        # that is enclosed between '$$' for rendering in markdown
        response = re.sub(r"\\\[|\\\]", "$$", response)
        response = re.sub(r"\\\(|\\\)", "$", response)
        return response

    # checks
    if not querytext:
        yield MSG_EMPTY_QUERY
        return

    if len(querytext.split()) > MAX_QUERY_LENGTH:
        yield MSG_LONG_QUERY
        return

    querytext = querytext.replace(
        "the textbook", "the context provided"
    )

    # The gradio framework will deliver an empty history list
    # if this is the first message in the exchange. Context is
    # retrieved in this case
    if not history:
        try:
            documents: list[Document] = await retriever.ainvoke(
                querytext
            )
        except Exception as e:
            logger.error(
                f"Error retrieving from vector database:\n{e}"
            )
            yield MSG_ERROR_QUERY
            return
        context: str = "\n\n".join(
            [d.page_content for d in documents]
        )
        querytext = prompt.format(context=context, query=querytext)

    # Query
    query = _prepare_messages(querytext, history, SYSTEM_MESSAGE)
    buffer: str = ""
    try:
        response_iter: AsyncIterator[BaseMessageChunk] = llm.astream(
            query
        )
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
                    + f'"{querytext}","ERROR: {buffer}"\n'
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
        fn,
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
    from lmm_education.config.config import ConfigSettings

    config_settings = ConfigSettings()
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
