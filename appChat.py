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
    ConfigSettings,
    create_default_config_file,
)
import os # fmt: skip
if not os.path.exists("config.toml"):
    create_default_config_file("config.toml")
    print("config.toml created in app folder, change as appropriate")

# this reads the settings from config.toml
settings = ConfigSettings()

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
from lmm.language_models.langchain.runnables import create_embeddings

embedding = create_embeddings()


# Internal facility to format messages in chat exchanges.
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

    This version streams the content of the response.
    """

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
                querytext,
                limit=6,
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

    This version checks the content of the stream before releasing it.
    """

    from lmm.language_models.langchain.runnables import (
        create_runnable,
    )

    try:
        query_model = create_runnable(
            "check_content"
        )  # uses config.toml
    except Exception as e:
        logger.error(f"Could not initialize query_model: {e}")
        yield MSG_ERROR_QUERY
        return

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
                querytext,
                limit=6,
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

    # check function querying model for content of response with retry logic
    async def _check_content(
        response: str, max_retries: int = 2
    ) -> tuple[bool, str]:
        """
        Check content with retry logic and proper error handling.
        Returns (is_valid, error_message) tuple.
        """
        import asyncio

        for attempt in range(max_retries + 1):
            try:
                # Use ainvoke for truly async, non-blocking operation
                check: str = await query_model.ainvoke(
                    {'text': response}
                )
                if (
                    "statistics" in check
                    or "human interaction" in check
                ):
                    return True, ""
                else:
                    return (
                        False,
                        MSG_WRONG_CONTENT,
                    )

            except Exception as e:
                logger.warning(
                    f"Content check attempt {attempt + 1}/{max_retries + 1} failed: {e}"
                )

                if attempt == max_retries:
                    # All retries exhausted - fail-open strategy
                    logger.error(
                        f"Content checker failed after {max_retries + 1} attempts: {e}"
                    )
                    return (
                        True,
                        "Content validation temporarily unavailable",
                    )

                # Wait before retry with exponential backoff
                await asyncio.sleep(0.5 * (attempt + 1))

        # Should never reach here, but for safety
        return (
            True,
            "Content validation system error",
        )

    # local iterator wrapping up the language model stream
    # imolementing check and conditional release
    async def _check_and_relay(
        response_iter: AsyncIterator[BaseMessageChunk],
    ) -> AsyncGenerator[str, None]:
        """
        Buffers initial chunks from the LLM, checks its content by
        making a second request to a LLM, and then either yields an
        error or starts relaying the stream.
        """
        buffer: str = ""
        initial_buffer_size: int = 150  # Check after 150 characters
        check_complete: bool = False

        async for item in response_iter:
            chunk = _preproc_for_markdown(item.text())
            buffer += chunk

            # --- Check Logic ---
            if (
                not check_complete
                and len(buffer) >= initial_buffer_size
            ):
                # Check the initial 'N' characters for sensitive content
                flag, error_message = await _check_content(
                    buffer + "..."
                )
                if not flag:
                    # If the check fails, yield an error message and stop
                    yield error_message
                    return

                # If the check passes, mark as complete and yield the buffered content
                check_complete = True
                if (
                    error_message
                ):  # This should be empty, unless LMM did not respond
                    logger.warning(
                        "LMM exchange without check: " + buffer
                    )
                yield buffer
                continue  # Go to the next iteration

            # --- Relay Logic ---
            if check_complete:
                # Once the check is complete, yield the stream *chunk by chunk*
                yield buffer  # Yield the *accumulated* buffer after this new chunk

            # Note: If check_complete is False and len(buffer) < initial_buffer_size,
            # it just keeps buffering and doesn't yield anything yet.

        # If the stream finishes before the buffer size is reached,
        # perform a final check and yield the result
        if not check_complete:
            flag, error_message = await _check_content(buffer)
            if not flag:
                # If the check fails, yield an error message and stop
                yield error_message
            else:
                if (
                    error_message
                ):  # This should be empty, unless LMM did not respond
                    logger.warning(
                        "LMM exchange without check: " + buffer
                    )
                yield buffer  # Yield the full response if it passed the check

    # Query
    query = _prepare_messages(querytext, history, SYSTEM_MESSAGE)
    buffer: str = ""
    try:
        response_iter: AsyncIterator[BaseMessageChunk] = llm.astream(
            query
        )

        # Use the wrapper generator here
        async for stream_output in _check_and_relay(response_iter):
            buffer = stream_output  # Keep track of the full output for logging/exceptions
            yield stream_output  # Release the checked/relayed output

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
