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

from collections.abc import AsyncIterator, Iterator, AsyncGenerator
from typing import Any

# Langchain
from langchain_core.messages import (
    BaseMessageChunk,
    BaseMessage,
    HumanMessage,
    AIMessage,
    AIMessageChunk,
)
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models.chat_models import BaseChatModel

# LM markdown
from lmm.utils.logging import ConsoleLogger, LoggerBase
from lmm.config.config import LanguageModelSettings
from lmm.language_models.langchain.models import (
    create_model_from_settings,
)
from lmm.language_models.langchain.runnables import (
    create_runnable,
    RunnableType,
)
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

# New imports for refactored architecture
from .chat_graph import (
    ChatStateGraphType,
    ChatState,
    ChatWorkflowContext,
    workflow_library,
)
from .stream_adapters import validation_stream_adapter


def history_to_messages(
    history: list[dict[str, str]],
) -> list[HumanMessage | AIMessage | BaseMessage]:
    """
    Convert Gradio history format to LangChain messages.

    Filters out non-message entries (context, message, rejection) that
    were previously stuffed into history for metadata transport.

    Args:
        history: Gradio-format history list

    Returns:
        List of LangChain message objects
    """
    messages: list[HumanMessage | AIMessage | BaseMessage] = []
    for m in history:
        role = m.get("role", "")
        content = m.get("content", "")
        # Skip metadata entries that were misused in history
        if role in ("context", "message", "rejection"):
            continue
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def create_initial_state(
    querytext: str,
    history: list[dict[str, str]] | None = None,
) -> ChatState:
    """
    Create initial ChatState from query and optional history.

    Args:
        querytext: The user's query text
        history: Optional Gradio-format conversation history

    Returns:
        ChatState initialized for the workflow
    """
    messages: list[HumanMessage | AIMessage | BaseMessage] = (
        history_to_messages(history) if history else []
    )

    return ChatState(
        messages=messages,
        query_text=querytext,
        original_query=querytext,
        query_status="valid",  # Will be validated by workflow
        context="",
        documents=[],
        error_message="",
        log_data={},
    )


async def chat_function(
    querytext: str,
    history: list[dict[str, str]] | None = None,
    retriever: BaseRetriever | None = None,
    *,
    llm: BaseChatModel,
    chat_settings: ChatSettings,
    system_msg: str = "You are a helpful assistant",
    print_context: bool = False,
    logger: LoggerBase = ConsoleLogger(),
) -> AsyncGenerator[BaseMessageChunk, None]:
    """
    Async generator that yields BaseMessageChunk objects.

    This function uses LangGraph's native streaming to process a user query.
    The workflow handles query validation, context retrieval, query formatting,
    and LLM response generation in a single streaming call.

    Note: This is an async generator function that may be called drectly:
        stream = chat_function(...)
        async for chunk in stream:
            ...

    Args:
        querytext: The user's query text
        history: List of previous message exchanges (Gradio format)
        retriever: Optional document retriever for RAG
        llm: Langchain language model object
        chat_settings: Chat configuration settings
        system_msg: System message for the conversation
        print_context: Prints the results of the query to the logger
        logger: Logger instance for info and error reporting

    Yields:
        BaseMessageChunk objects from the LLM stream
    """

    # Utility function to process error messages
    def _error_chunk(message: str) -> AIMessageChunk:
        logger.error(message)
        return AIMessageChunk(content="Error: " + message)

    config_settings: ConfigSettings | None = load_settings(
        logger=logger
    )
    if config_settings is None:
        yield _error_chunk("Could not load settings")
        return

    # Set up retriever if not provided
    if retriever is None:
        try:
            retriever = AsyncQdrantRetriever.from_config_settings(
                config_settings
            )
        except Exception as e:
            yield _error_chunk(
                f"Could not open vector database: {e}. "
                "The system is not available."
            )
            return

    # Create workflow configuration
    workflow_config = ChatWorkflowContext(
        llm=llm,
        retriever=retriever,
        system_message=system_msg,
        chat_settings=chat_settings,
        print_context=print_context,
        logger=logger,
    )

    # Create and compile workflow
    workflow: ChatStateGraphType = workflow_library['query']

    # Create initial state
    initial_state: ChatState = create_initial_state(
        querytext,
        history if history else None,
    )

    # Stream from workflow using LangGraph's native streaming
    # stream_mode="messages" emits (chunk, metadata) tuples
    event: tuple[AIMessageChunk, dict[str, Any]]
    try:
        async for event in workflow.astream(  # type: ignore
            initial_state,
            stream_mode="messages",
            context=workflow_config,
        ):
            # event is a tuple: (chunk, metadata)
            if not isinstance(event, tuple) and len(event) == 2:  # type: ignore
                raise ValueError(
                    "Unreachable code reached, invalid "
                    "event in LangGraph stream"
                )

            chunk, metadata = event
            node_name: str = metadata.get("langgraph_node", "")

            # Filter chunks from required nodes
            if node_name in {"validate_query", "generate"}:
                yield chunk

            if print_context and node_name == "retrieve_context":
                yield chunk

    except Exception as e:
        yield _error_chunk(f"Workflow streaming failed: {e}")
        return


async def chat_function_with_validation(
    querytext: str,
    history: list[dict[str, str]] | None = None,
    retriever: BaseRetriever | None = None,
    *,
    llm: BaseChatModel,
    chat_settings: ChatSettings,
    system_msg: str = "You are a helpful assistant",
    print_context: bool = False,
    logger: LoggerBase = ConsoleLogger(),
    initial_buffer_size: int = 320,
    max_retries: int = 2,
) -> AsyncGenerator[BaseMessageChunk, None]:
    """
    Returns an async iterator that yields BaseMessageChunk objects with
    content validation.

    This function extends chat_function by wrapping the output stream
    with a validation adapter. The adapter buffers initial response
    content, validates it using a separate LLM, and only streams
    the content if validation passes.

    Args:
        querytext: The user's query text
        history: List of previous message exchanges
        retriever: Optional document retriever for RAG
        llm: Language model to use
        chat_settings: Chat configuration settings
        system_msg: System message for the conversation
        print_context: Prints to logger the results of the query
        logger: Logger instance for error reporting
        initial_buffer_size: Number of characters to buffer before
            validation
        max_retries: Maximum number of validation retry attempts

    Returns:
        AsyncIterator[BaseMessageChunk]: Iterator yielding validated
            response chunks
    """

    # Utility function to process error messages
    def _error_chunk(message: str) -> AIMessageChunk:
        logger.error(message)
        return AIMessageChunk(content="Error: " + message)

    # Get allowed content from settings
    allowed_content: list[str] = (
        chat_settings.check_response.allowed_content
    )

    # Initialize the validation model
    try:
        validator_model: RunnableType = create_runnable(
            "allowed_content_validator",
            allowed_content=allowed_content,  # type: ignore
        )
    except Exception as e:
        err_msg = f"Could not initialize validation model: {e}"
        yield _error_chunk(err_msg)
        return

    # Get the base chat stream
    base_stream: AsyncIterator[BaseMessageChunk] = chat_function(
        querytext=querytext,
        history=history,
        retriever=retriever,
        llm=llm,
        chat_settings=chat_settings,
        system_msg=system_msg,
        print_context=print_context,
        logger=logger,
    )

    # Wrap with validation adapter
    validated_stream: AsyncIterator[BaseMessageChunk] = (
        validation_stream_adapter(
            base_stream,
            querytext,
            validator_model=validator_model,
            allowed_content=allowed_content,
            buffer_size=initial_buffer_size,
            error_message=chat_settings.MSG_WRONG_CONTENT,
            max_retries=max_retries,
            logger=logger,
        )
    )

    # Yield from validated stream
    async for chunk in validated_stream:
        yield chunk


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
        buffer += chunk.text
    return buffer


def query(
    querystr: str,
    *,
    model_settings: LanguageModelSettings | str | None = None,
    chat_settings: ChatSettings | None = None,
    print_context: bool = False,
    validate_content: bool = False,
    allowed_content: list[str] | None = None,
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
        print_context: If True, print the RAG context to the logger
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
    async_gen: AsyncIterator[str] = aquery(
        querystr,
        model_settings=model_settings,
        chat_settings=chat_settings,
        print_context=print_context,
        validate_content=validate_content,
        allowed_content=allowed_content or [],
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
    print_context: bool = False,
    validate_content: bool = False,
    allowed_content: list[str] | None = None,
    client: AsyncQdrantClient | None = None,
    logger: LoggerBase = ConsoleLogger(),
) -> AsyncIterator[str]:
    """
    Asynchronous generator that yields text strings from the query() async function.

    Args:
        querystr: The query text to send to the language model
        model_settings: Language model settings (or 'major', 'minor', 'aux')
        chat_settings: Chat settings for the query
        print_context: If True, print the RAG context to the logger
        validate_content: If True, validate response content
        allowed_content: List of allowed content types for validation
        client: Optional pre-configured Qdrant client
        logger: Logger instance for error reporting

    Yields:
        str: Text chunks from the language model response

    Example:
        async for text in query("What is logistic regression?"):
            print(text, end="", flush=True)
        print()
    """
    if allowed_content is None:
        allowed_content = []

    config_settings: ConfigSettings | None = load_settings(
        logger=logger
    )
    if config_settings is None:
        logger.error("Could not load settings.")
        yield "Could not load settings."
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
            yield "Could not load chat settings."
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
        retriever: BaseRetriever = (
            AsyncQdrantRetriever.from_config_settings(client=client)
        )
    except Exception as e:
        logger.error(f"Could not load retriever: {e}")
        yield f"Could not load retriever: {e}"
        return

    # Get the iterator and consume it
    if validate_content:
        iterator: AsyncIterator[BaseMessageChunk] = (
            chat_function_with_validation(
                querystr,
                None,  # No history for standalone queries
                retriever=retriever,
                llm=llm,
                chat_settings=chat_settings,
                print_context=print_context,
                logger=logger,
            )
        )
    else:
        iterator = chat_function(
            querystr,
            None,  # No history for standalone queries
            llm=llm,
            chat_settings=chat_settings,
            print_context=print_context,
            logger=logger,
        )

    async for chunk in iterator:
        yield chunk.text


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
