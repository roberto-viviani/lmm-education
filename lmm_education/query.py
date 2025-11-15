"""
This module allows interactive query with a language model to
test use of material ingested in the RAG database.

A database must have been created (for example, with the ingest
module). In the following examples, we assume the existence of
a database on basic statistical modelling.

Examples:

```python
# from the python REPL

python -m lmm_education.query 'What is logistic regression?'
```

```python
# from python code
from lmm_education.query import query

respons = query('what is logistic regression?')
```

Because ingest replaces the content of the database when documents
are edited, you can set up an ingest-evaluate loop:

```python
# from the python REPL

# append True to ingest the file 'RaggedDocument.md'
python -m lmm_education.ingest RaggedDocument.md True
python -m lmm_education.query 'what is logistic regression?'
"""

import asyncio
from lmm.language_models.langchain.runnables import RunnableType
from pydantic import validate_call
from collections.abc import AsyncIterator

# Langchain
from langchain_core.messages import BaseMessageChunk
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models.chat_models import BaseChatModel

# LM markdown
from lmm.utils.logging import ConsoleLogger, LoggerBase
from lmm.config.config import LanguageModelSettings
from lmm.language_models.langchain.models import (
    create_model_from_settings,
)

# LM markdown for education
from .config.config import ConfigSettings
from .stores.langchain.vector_store_qdrant_langchain import (
    AsyncQdrantVectorStoreRetriever as AsyncQdrantRetriever,
)

# messages
MSG_EMPTY_QUERY = "Please ask a question about the course."
MSG_WRONG_CONTENT = "I can only answer questions about the course."
MSG_LONG_QUERY = (
    "Your question is too long. Please ask a shorter question."
)
MSG_ERROR_QUERY = (
    "I am sorry, I cannot answer this question. Please retry."
)

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
default_prompt: PromptTemplate = PromptTemplate.from_template(
    PROMPT_TEMPLATE
)


# Helper function to create error message iterators
async def _error_message_iterator(
    message: str,
) -> AsyncIterator[BaseMessageChunk]:
    """
    Creates an async iterator that yields a single error message as a BaseMessageChunk.
    This allows error messages to be returned in the same format as LLM responses.
    """
    from langchain_core.messages import AIMessageChunk

    yield AIMessageChunk(content=message)


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


async def chat_function(
    querytext: str,
    history: list[dict[str, str]] = [],
    retriever: BaseRetriever | None = None,
    llm: BaseChatModel | None = None,
    *,
    system_msg: str = "You are a helpful assistant",
    prompt: PromptTemplate = default_prompt,
    logger: LoggerBase = ConsoleLogger(),
) -> AsyncIterator[BaseMessageChunk]:
    """
    Returns an async iterator that yields BaseMessageChunk objects.

    This function processes a user query and returns an iterator that streams
    the LLM response. In case of validation errors or retrieval failures,
    it returns an iterator that yields an appropriate error message.

    Args:
        querytext: The user's query text
        history: List of previous message exchanges
        retriever: Optional document retriever for RAG
        llm: Optional language model to use
        system_msg: System message for the conversation
        prompt: Prompt template for formatting context and query
        logger: Logger instance for error reporting

    Returns:
        AsyncIterator[BaseMessageChunk]: Iterator yielding response chunks
    """

    if llm is None:
        from lmm_education.config.config import ConfigSettings # fmt: skip
        llm = create_model_from_settings(ConfigSettings().major)

    # the max allowed number of words in the user's query
    MAX_QUERY_LENGTH: int = 60

    # checks
    if not querytext:
        return _error_message_iterator(MSG_EMPTY_QUERY)

    if len(querytext.split()) > MAX_QUERY_LENGTH:
        return _error_message_iterator(MSG_LONG_QUERY)

    querytext = querytext.replace(
        "the textbook", "the context provided"
    )

    # if the history is empty, retrieve the context. The context
    # will be contained in the history thereafter.
    import typing

    if not history:
        create_flag: bool = False
        if retriever is None:
            retriever = AsyncQdrantRetriever.from_config_settings()
            create_flag = True

        try:
            documents: list[Document] = await retriever.ainvoke(
                querytext,
                limit=6,
            )
        except Exception as e:
            logger.error(
                f"Error retrieving from vector database:\n{e}"
            )
            return _error_message_iterator(MSG_ERROR_QUERY)
        context: str = "\n\n".join(
            [d.page_content for d in documents]
        )
        querytext = prompt.format(context=context, query=querytext)

        if create_flag:
            retriever = typing.cast(AsyncQdrantRetriever, retriever)
            await retriever.close_client()

    # Query language model
    query: list[tuple[str, str]] = _prepare_messages(
        querytext, history, system_msg
    )
    return llm.astream(query)


async def chat_function_with_validation(
    querytext: str,
    history: list[dict[str, str]],
    retriever: BaseRetriever | None = None,
    llm: BaseChatModel | None = None,
    *,
    system_msg: str = "You are a helpful assistant",
    prompt: PromptTemplate = default_prompt,
    logger: LoggerBase = ConsoleLogger(),
    validation_config: str = "check_content",
    initial_buffer_size: int = 180,
    max_retries: int = 2,
) -> AsyncIterator[BaseMessageChunk]:
    """
    Returns an async iterator that yields BaseMessageChunk objects with
    content validation.

    This function extends chat_function by adding content validation. It
    buffers the initial response, validates it using a separate LLM, and
    only streams the content if validation passes. This is useful for
    ensuring responses meet certain content criteria before being shown
    to users.

    Args:
        querytext: The user's query text
        history: List of previous message exchanges
        retriever: Optional document retriever for RAG
        llm: Optional language model to use
        system_msg: System message for the conversation
        prompt: Prompt template for formatting context and query
        logger: Logger instance for error reporting
        validation_config: Config name for the validation runnable
        initial_buffer_size: Number of characters to buffer before validation
        max_retries: Maximum number of validation retry attempts

    Returns:
        AsyncIterator[BaseMessageChunk]: Iterator yielding validated response chunks
    """
    from lmm.language_models.langchain.runnables import (
        create_runnable,
    )

    # Initialize the validation model
    try:
        query_model: RunnableType = create_runnable(validation_config)
    except Exception as e:
        logger.error(f"Could not initialize validation model: {e}")
        return _error_message_iterator(MSG_ERROR_QUERY)

    # Get the base chat iterator
    base_iterator: AsyncIterator[BaseMessageChunk] = (
        await chat_function(
            querytext=querytext,
            history=history,
            retriever=retriever,
            llm=llm,
            system_msg=system_msg,
            prompt=prompt,
            logger=logger,
        )
    )

    # Content validation function with retry logic
    async def _check_content(response: str) -> tuple[bool, str]:
        """
        Check content with retry logic and proper error handling.
        Returns (is_valid, error_message) tuple.
        """
        for attempt in range(max_retries + 1):
            try:
                check: str = await query_model.ainvoke(
                    {'text': response}
                )
                if (
                    "statistics" in check
                    or "human interaction" in check
                ):
                    return True, ""
                else:
                    return False, MSG_WRONG_CONTENT

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
        return True, "Content validation system error"

    # Wrapper generator that implements validation logic
    async def _validated_iterator() -> (
        AsyncIterator[BaseMessageChunk]
    ):
        """
        Buffers initial chunks, validates content, then streams the rest.
        """
        from langchain_core.messages import AIMessageChunk

        buffer_chunks: list[BaseMessageChunk] = []
        buffer_text: str = ""
        check_complete: bool = False

        async for chunk in base_iterator:
            chunk_text = chunk.text()

            # Buffering phase
            if not check_complete:
                buffer_chunks.append(chunk)
                buffer_text += chunk_text

                # Check if we've buffered enough
                if len(buffer_text) >= initial_buffer_size:
                    flag, error_message = await _check_content(
                        buffer_text + "..."
                    )

                    if not flag:
                        # Validation failed - yield error and stop
                        yield AIMessageChunk(content=error_message)
                        return

                    # Validation passed - mark complete and yield buffered content
                    check_complete = True
                    if error_message:
                        logger.warning(
                            f"LLM exchange without check: {buffer_text}"
                        )

                    # Yield all buffered chunks
                    for buffered_chunk in buffer_chunks:
                        yield buffered_chunk

            else:
                # Streaming phase - validation already passed
                yield chunk

        # If stream ended before buffer was full, validate what we have
        if not check_complete:
            flag, error_message = await _check_content(buffer_text)

            if not flag:
                # Validation failed - yield error
                yield AIMessageChunk(content=error_message)
            else:
                if error_message:
                    logger.warning(
                        f"LLM exchange without check: {buffer_text}"
                    )
                # Validation passed - yield all buffered chunks
                for buffered_chunk in buffer_chunks:
                    yield buffered_chunk

    return _validated_iterator()


async def consume_chat_stream(
    iterator: AsyncIterator[BaseMessageChunk],
) -> str:
    """
    Consumes an async iterator of BaseMessageChunk objects and returns
    the complete response as a string.

    This function is designed to work with the iterator returned by
    chat_function. It accumulates the text content from each chunk
    and returns the final result.

    Args:
        iterator: AsyncIterator yielding BaseMessageChunk objects

    Returns:
        str: The complete accumulated response text
    """
    buffer: str = ""
    async for chunk in iterator:
        buffer += chunk.text()
    return buffer


@validate_call(config={'arbitrary_types_allowed': True})
def query(
    querystr: str,
    settings: LanguageModelSettings | str | None = None,
    *,
    console_print: bool = False,
    validate_content: bool = False,
    logger: LoggerBase = ConsoleLogger(),
) -> str:

    if not querystr:
        return ""

    if len(querystr.split()) < 3:
        print("Please enter a complete query.")
        return ""

    if settings is None:
        config = ConfigSettings()
        settings = config.major
    elif isinstance(settings, str):
        config = ConfigSettings()
        if settings == "major":
            settings = config.major
        elif settings == "minor":
            settings = config.minor
        elif settings == "aux":
            settings = config.aux
        else:
            logger.error(
                f"Invalid language model settings: {settings}\nShould"
                " be one of 'major', 'minor', 'aux'"
            )
            return ""

    llm = create_model_from_settings(settings)

    # Get the iterator and consume it
    async def run_query():
        if validate_content:
            iterator: AsyncIterator[BaseMessageChunk] = (
                await chat_function_with_validation(
                    querystr, [], llm=llm, logger=logger
                )
            )
        else:
            iterator = await chat_function(
                querystr, [], llm=llm, logger=logger
            )
        result: str = ""
        async for chunk in iterator:
            content: str = chunk.text()
            if console_print:
                print(content, end="", flush=True)
            result += content
        print()  # New line after completion
        return result

    return asyncio.run(run_query())


if __name__ == "__main__":
    import sys
    from requests import ConnectionError

    if len(sys.argv) == 2:
        try:
            query(
                sys.argv[1],
                console_print=True,
                validate_content=True,
            )
        except ConnectionError as e:
            print("Cannot form embeddings due a connection error")
            print(e)
            print("Check the internet connection.")
        except Exception as e:
            print(f"ERROR: {e}")
    else:
        print("Usage: call querydb followed by your query.")
        print("Example: querydb 'what is logistic regression?'")
