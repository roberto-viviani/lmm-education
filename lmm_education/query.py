"""
This module allows interactively querying a language model, such that
it can be tested using material ingested in the RAG database.

A RAG database must have been previously created (for example, with the ingest
module). In the following examples, we assume the existence of
a database on basic statistical modelling.

Examples:

```bash
# Python called from the console

python -m lmm_education.query 'What is logistic regression?'
```

```python
# from python code
from lmm_education.query import query

response = query('what is logistic regression?')
print(response)
```

Because ingest replaces the content of the database when documents
are edited, you can set up an ingest-evaluate loop:

```bash
# Python called from the console

# append True to ingest the file 'LogisticRegression.md'
python -m lmm_education.ingest LogisticRegression.md True
python -m lmm_education.query 'what is logistic regression?'
```
"""

import asyncio
from lmm.language_models.langchain.runnables import RunnableType
from collections.abc import AsyncIterator, Iterator

# Langchain
from langchain_core.messages import BaseMessageChunk, AIMessageChunk
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models.chat_models import BaseChatModel

# LM markdown
from lmm.utils.logging import ConsoleLogger, LoggerBase
from lmm.utils.ioutils import check_allowed_content
from lmm.config.config import LanguageModelSettings
from lmm.language_models.langchain.models import (
    create_model_from_settings,
)
from lmm.markdown.ioutils import convert_backslash_latex_delimiters
from qdrant_client import AsyncQdrantClient

# LM markdown for education
from .config.config import load_settings
from .config.config import ConfigSettings, DEFAULT_CONFIG_FILE
from .stores.langchain.vector_store_qdrant_langchain import (
    AsyncQdrantVectorStoreRetriever as AsyncQdrantRetriever,
)
from .config.appchat import (
    ChatSettings,
    load_settings as load_chat_settings,
)

from langchain_core.prompts import PromptTemplate # fmt: skip


# Helper function to create error message iterators
async def _error_message_iterator(
    message: str,
) -> AsyncIterator[BaseMessageChunk]:
    """
    Creates an async iterator that yields a single error message as a BaseMessageChunk.
    This allows error messages to be returned in the same format as LLM responses.
    """

    yield AIMessageChunk(content=message)


# Internal facility to format messages in chat exchanges.
def _prepare_messages(
    query: str,
    history: list[dict[str, str]],
    system_message: str = "",
) -> list[tuple[str, str]]:
    """
    Customized message history for stateless interaction. The first
    element is always the system message. We append to this the last
    four interactions (i.e. user and assistant messages).
    """
    # Note that the context is not contained in the history. The
    # Gradio chat app keeps a list of what the user typed in the web
    # app as a query, and what the app displayed as a response. In
    # this system, the first message if modified by code to include
    # context, but the following messages are not.

    messages: list[tuple[str, str]] = []
    if system_message:
        messages.append(('system', system_message))

    if history:
        for m in history[-4:]:
            if m['role'] in ("context", "message", "rejection"):
                continue
            messages.append((m['role'], m['content']))

    messages.append(('user', query))

    return messages


async def chat_function(
    querytext: str,
    history: list[dict[str, str]] = [],
    retriever: BaseRetriever | None = None,
    *,
    llm: BaseChatModel,
    chat_settings: ChatSettings,
    system_msg: str = "You are a helpful assistant",
    context_print: bool = False,
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
        llm: Optional Langchain language model object
        system_msg: System message for the conversation
        prompt: Prompt template for formatting context and query
        context_print: Prints the results of the query to the logger
        logger: Logger instance for info and error reporting

    Returns:
        AsyncIterator[BaseMessageChunk]: Iterator yielding response chunks
    """
    # Implementation note: history is passed by the gradio framework
    # with the responses of the language model and what the user
    # tipped into the interface -- it is not what the model got.
    # Here history is used also to store the context that was sent
    # to the model and information about other conditions. This
    # is used upstream to log these interactions (subject to revision)

    config_settings: ConfigSettings | None = load_settings(
        logger=logger
    )
    if config_settings is None:
        logger.error("Could not load settings")
        return _error_message_iterator("Could not load settings")

    # prompt and the max allowed number of words in the user's
    # query are in chat settings.
    prompt: PromptTemplate = PromptTemplate.from_template(
        chat_settings.PROMPT_TEMPLATE
    )
    MAX_QUERY_LENGTH: int = chat_settings.max_query_word_count

    # checks
    if not querytext:
        history.append({'role': 'message', 'content': "EMPTYQUERY"})
        return _error_message_iterator(chat_settings.MSG_EMPTY_QUERY)

    if len(querytext.split()) > MAX_QUERY_LENGTH:
        history.append({'role': 'message', 'content': "LONGQUERY"})
        return _error_message_iterator(chat_settings.MSG_LONG_QUERY)

    querytext = querytext.replace(
        "the textbook", "the context provided"
    )

    # if the history is empty, retrieve the context. The context
    # will be contained in the history thereafter.
    if not history:
        if retriever is None:
            try:
                retriever = AsyncQdrantRetriever.from_config_settings(
                    config_settings  # type: ignore (checked above)
                )
            except Exception as e:
                logger.error(f"Could not open vector database: {e}")
                return _error_message_iterator(
                    "System currently not available."
                )

        try:
            documents: list[Document] = await retriever.ainvoke(
                querytext,
                # limit=6,
            )
        except Exception as e:
            logger.error(
                f"Error retrieving from vector database:\n{e}"
            )
            return _error_message_iterator(
                chat_settings.MSG_ERROR_QUERY
            )
        context: str = "\n-----\n".join(
            [d.page_content for d in documents]
        )

        history.append({'role': 'context', 'content': context})

        context = convert_backslash_latex_delimiters(context)
        if context_print:
            logger.info(
                "CONTEXT:\n" + context + "\nEND CONTEXT------\n\n"
            )
        querytext = prompt.format(context=context, query=querytext)

    # Query language model
    query: list[tuple[str, str]] = _prepare_messages(
        querytext, history, system_msg
    )
    return llm.astream(query)


async def chat_function_with_validation(
    querytext: str,
    history: list[dict[str, str]],
    retriever: BaseRetriever | None = None,
    *,
    llm: BaseChatModel,
    chat_settings: ChatSettings,
    system_msg: str = "You are a helpful assistant",
    context_print: bool = False,
    logger: LoggerBase = ConsoleLogger(),
    initial_buffer_size: int = 320,
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
        context_print: Prints to logger the results of the query
        logger: Logger instance for error reporting
        validation_config: Config name for the validation runnable
        initial_buffer_size: Number of characters to buffer before
            validation
        max_retries: Maximum number of validation retry attempts

    Returns:
        AsyncIterator[BaseMessageChunk]: Iterator yielding validated
            response chunks
    """
    # Implementation note: history is passed by the gradio framework
    # with the responses of the language model and what the user
    # tipped into the interface -- it is not what the model got.
    # Here history is used also to store the context that was sent
    # to the model and information about other conditions. This
    # is used upstream to log these interactions (subject to revision)

    from lmm.language_models.langchain.runnables import (
        create_runnable,
    )

    # Perform basic validation checks before calling chat_function
    # This ensures error iterators are returned directly without wrapping
    MAX_QUERY_LENGTH: int = chat_settings.max_query_word_count

    # The allowed content is read from chat settings
    allowed_content: list[str] = (
        chat_settings.check_response.allowed_content
    )

    # Check for empty query
    if not querytext:
        history.append({'role': 'message', 'content': "EMPTYQUERY"})
        return _error_message_iterator(chat_settings.MSG_EMPTY_QUERY)

    # Check for overly long query
    if len(querytext.split()) > MAX_QUERY_LENGTH:
        history.append({'role': 'message', 'content': "LONGQUERY"})
        return _error_message_iterator(chat_settings.MSG_LONG_QUERY)

    # Initialize the validation model
    try:
        query_model: RunnableType = create_runnable(
            "allowed_content_validator", allowed_content=allowed_content  # type: ignore
        )
    except Exception as e:
        logger.error(f"Could not initialize validation model: {e}")
        return _error_message_iterator(chat_settings.MSG_ERROR_QUERY)

    # Get the base chat iterator
    base_iterator: AsyncIterator[BaseMessageChunk] = (
        await chat_function(
            querytext=querytext,
            history=history,
            retriever=retriever,
            llm=llm,
            chat_settings=chat_settings,
            system_msg=system_msg,
            context_print=context_print,
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
                logger.info(
                    "Model content classification: "
                    + check.replace("\n", " ")
                    + "\n"
                )

                if check_allowed_content(
                    check,
                    allowed_content
                    + ['apology', 'human interaction'],
                ):
                    return True, ""
                else:
                    history.append(
                        {'role': 'rejection', 'content': check}
                    )
                    return False, chat_settings.MSG_WRONG_CONTENT

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
            chunk_text: str = chunk.text()

            # Buffering phase
            if not check_complete:
                buffer_chunks.append(chunk)
                buffer_text += chunk_text

                # Check if we've buffered enough
                if len(buffer_text) >= initial_buffer_size:
                    flag, error_message = await _check_content(
                        querytext + "\n\n" + buffer_text + "..."
                    )

                    if not flag:
                        # Validation failed - yield error and stop
                        yield AIMessageChunk(content=error_message)
                        return

                    # Validation passed - mark complete and yield buffered content
                    check_complete = True
                    if error_message:
                        logger.warning(
                            "LLM exchange without content check (aux failure)"
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


def query(
    querystr: str,
    *,
    model_settings: LanguageModelSettings | str | None = None,
    chat_settings: ChatSettings | None = None,
    context_print: bool = False,
    validate_content: bool = False,
    allowed_content: list[str] = [],
    logger: LoggerBase = ConsoleLogger(),
) -> Iterator[str]:
    """
    Synchronous generator that yields text strings from the query() async function.

    This is a convenience wrapper that allows synchronous code to consume
    the streaming response from the language model. It creates an event loop
    internally to bridge the async/sync boundary.

    Args:
        querystr: The query text to send to the language model
        model_settings: Language model settings (or 'major', 'minor', 'aux')
        chat_settings: Chat settings for the query
        context_print: If True, print the RAG context to the logger
        validate_content: If True, validate response content
        allowed_content: List of allowed content types for validation
        logger: Logger instance for error reporting

    Yields:
        str: Text chunks from the language model response

    Example:
        for text in query_sync("What is logistic regression?"):
            print(text, end="", flush=True)
        print()
    """
    from .asyncutils import async_gen_to_sync_iter

    # Create the async generator object
    async_gen = aquery(
        querystr,
        model_settings=model_settings,
        chat_settings=chat_settings,
        context_print=context_print,
        validate_content=validate_content,
        allowed_content=allowed_content,
        logger=logger,
    )

    # Iterate synchronously and yield text from each chunk
    for chunk in async_gen_to_sync_iter(async_gen):
        yield chunk


async def aquery(
    querystr: str,
    *,
    model_settings: LanguageModelSettings | str | None = None,
    chat_settings: ChatSettings | None = None,
    context_print: bool = False,
    validate_content: bool = False,
    allowed_content: list[str] = [],
    client: AsyncQdrantClient | None = None,
    logger: LoggerBase = ConsoleLogger(),
) -> AsyncIterator[str]:
    """
    Asynchronous generator that yields text strings from the query() async function.

    Args:
        querystr: The query text to send to the language model
        model_settings: Language model settings (or 'major', 'minor', 'aux')
        chat_settings: Chat settings for the query
        context_print: If True, print the RAG context to the logger
        validate_content: If True, validate response content
        allowed_content: List of allowed content types for validation
        logger: Logger instance for error reporting

    Yields:
        str: Text chunks from the language model response

    Example:
        async for text in query("What is logistic regression?"):
            print(text, end="", flush=True)
        print()
    """

    config_settings: ConfigSettings | None = load_settings(
        logger=logger
    )
    if config_settings is None:
        logger.error("Could not load settings.")
        await consume_chat_stream(_error_message_iterator(""))
        return
    if model_settings is None:
        model_settings = config_settings.major
    elif isinstance(model_settings, str):
        config = ConfigSettings()
        if model_settings == "major":
            model_settings = config.major
        elif model_settings == "minor":
            model_settings = config.minor
        elif model_settings == "aux":
            model_settings = config.aux
        else:
            errmsg: str = (
                f"Invalid language model settings: {model_settings}\nShould"
                " be one of 'major', 'minor', 'aux'"
            )
            logger.error(errmsg)
            yield errmsg
            return

    llm: BaseChatModel = create_model_from_settings(model_settings)

    if chat_settings is None:
        chat_settings = load_chat_settings(logger=logger)
        if chat_settings is None:
            logger.error("Could not load chat settings")
            return

    if validate_content and not allowed_content:
        allowed_content = chat_settings.check_response.allowed_content
        if not allowed_content:
            errmsg = (
                "A request to validate content was made, but there is"
                " no allowed content value in the configuration file."
                "\nAdd a list of allowed contents in the "
                "[check_response] section of " + DEFAULT_CONFIG_FILE
            )
            logger.error(errmsg)
            yield errmsg
            return

    try:
        retriever = AsyncQdrantRetriever.from_config_settings(
            client=client
        )
    except Exception as e:
        logger.error(f"Could not load retriever: {e}")
        return

    # Get the iterator and consume it
    if validate_content:
        iterator: AsyncIterator[BaseMessageChunk] = (
            await chat_function_with_validation(
                querystr,
                [],
                retriever=retriever,
                llm=llm,
                chat_settings=chat_settings,
                context_print=context_print,
                logger=logger,
            )
        )
    else:
        iterator = await chat_function(
            querystr,
            [],
            llm=llm,
            chat_settings=chat_settings,
            context_print=context_print,
            logger=logger,
        )
    async for chunk in iterator:
        yield chunk.text()


if __name__ == "__main__":
    import sys
    from requests import ConnectionError

    if len(sys.argv) == 2:
        try:
            for text in query(
                sys.argv[1],
                validate_content=True,
            ):
                print(text, end="", flush=True)
            print()  # New line at end
        except ConnectionError as e:
            print("Cannot form embeddings due a connection error")
            print(e)
            print("Check the internet connection.")
        except Exception as e:
            print(f"ERROR: {e}")
    else:
        print("Usage: call querydb followed by your query.")
        print("Example: querydb 'what is logistic regression?'")
