"""
This module allows interactively querying a language model, such that
it can be tested using material ingested in the RAG database.

A RAG database must have been previously created (for example, with the
ingest module). In the following examples, we assume the existence of
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

from collections.abc import (
    AsyncIterator,
    Iterator,
    AsyncGenerator,
    Callable,
)
from typing import Any, Literal
from io import TextIOBase
from datetime import datetime

# Langchain
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
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
from lmm_education.config.appchat import CheckResponse
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
    logging,
)
from .stream_adapters import (
    tier_1_iterator,
    tier_2_iterator,
    astream_graph,
    stateful_validation_adapter,
    terminal_demux_adapter,
    demux_adapter,
)
from .graph_logging_base import create_graph_logger, LogCoroutine


def history_to_messages(
    history: list[dict[str, str]],
) -> list[HumanMessage | AIMessage | BaseMessage]:
    """
    Convert Gradio history format to LangChain messages.

    Args:
        history: Gradio-format history list

    Returns:
        List of LangChain message objects
    """
    messages: list[HumanMessage | AIMessage | BaseMessage] = []
    for m in history:
        role = m.get("role", "")
        content = m.get("content", "")
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
        status="valid",  # Will be validated by workflow
        query_text=querytext,
        query_classification="",
        context="",
    )


async def chat_function(
    querytext: str,
    history: list[dict[str, str]] | None,
    context: ChatWorkflowContext,
    *,
    print_context: bool = False,
    validate: CheckResponse | Literal[False] | None = None,
    database_streams: list[TextIOBase] = [],
    database_log: (
        LogCoroutine[ChatState, ChatWorkflowContext] | bool
    ) = False,
    logger: LoggerBase = ConsoleLogger(),
) -> AsyncGenerator[str, None]:
    """
    Async generator that yields BaseMessageChunk objects.

    This function retrieves a compiled LangGraph's graph, creates the
    initial state, and returns a stream to process a user query.

    Note: This is an async generator function that may be called directly:
        stream = chat_function(...)
        async for chunk in stream:
            ...

    Args:
        querytext: The user's query text
        history: List of previous message exchanges (Gradio format)
        context: a ChatWorkflowContext object for dependencies
        print_context: Prints the results of the query to the logger
        validate: if None, validates response using settings from
            appchat.toml. If False, carries out no validation. If a
            CheckReponse object, overrides the settings from
            appchat.toml.
        database_log: if False (default), carries out no database
            logging. If True, carries out database logging with the
            default function defined with the graph. If a logger
            function is provided, it uses that function.
        logger: Logger instance for info and error reporting

    Yields:
        strings from the LLM stream
    """

    # Utility function to process error messages. The chunks here are
    # shown in the chat output, the logger is for internal use.
    def _error_chunk(message: str) -> str:
        logger.error(message)
        return "Error: " + message

    config_settings: ConfigSettings | None = load_settings(
        logger=logger
    )
    if config_settings is None:
        yield _error_chunk("Could not load settings")
        return

    # Settings for print_context
    context.print_context = print_context

    # Retrieve settings for validation.
    response_settings: CheckResponse
    match validate:
        case None:
            response_settings = context.chat_settings.check_response
        case False:
            response_settings = CheckResponse(check_response=False)
        case CheckResponse():
            response_settings = validate

    # Settings for logging.
    database_logger: (
        Callable[[ChatState, datetime | None, str | None], str] | None
    ) = None
    match database_log:
        case False:
            database_logger = None
        case True:
            database_logger = create_graph_logger(
                database_streams, context, logging
            )
        case _:
            database_logger = create_graph_logger(
                database_streams, context, database_log
            )
    if database_logger and not database_streams:
        yield _error_chunk(
            "chat_function: database_logger used without streams"
        )
        return

    # Fetch graph from workflow library
    wfname = "query"
    try:
        workflow: ChatStateGraphType = workflow_library[wfname]
    except Exception as e:
        yield _error_chunk(
            f"Could not create workflow {wfname}:\n{e}"
        )
        return

    # Create initial state
    initial_state: ChatState = create_initial_state(
        querytext,
        history if history else None,
    )

    # Set up the stream
    raw_stream: tier_1_iterator = astream_graph(
        workflow, initial_state, context
    )

    # Configure stream for validation requests
    tier_1_stream: tier_1_iterator = raw_stream
    if response_settings.check_response:
        # Initialize the validation model
        try:
            validator_model: RunnableType = create_runnable(
                "allowed_content_validator",
                allowed_content=response_settings.allowed_content,  # type: ignore
            )
        except Exception as e:
            err_msg = f"Could not initialize validation model: {e}"
            yield _error_chunk(err_msg)
            return

        tier_1_stream = stateful_validation_adapter(
            raw_stream,
            validator_model=validator_model,
            allowed_content=response_settings.allowed_content,
            buffer_size=response_settings.initial_buffer_size,
            logger=logger,
        )

    message_stream: tier_2_iterator
    if database_logger:
        dblogger: Callable[[ChatState], Any] = (
            lambda state: database_logger(state, None, None)
        )
        message_stream = terminal_demux_adapter(
            tier_1_stream,
            on_terminal_state=dblogger,
        )
    else:
        message_stream = demux_adapter(tier_1_stream)

    try:
        async for chunk, _ in message_stream:
            yield str(chunk.text)  # type: ignore

    except Exception as e:
        yield _error_chunk(f"Workflow streaming ex failed: {e}")
        return


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
        print_context: If True, print the RAG context in the output
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

    # Create dependency injection object
    context = ChatWorkflowContext(
        llm=llm,
        retriever=retriever,
        chat_settings=chat_settings,
        print_context=print_context,
        logger=logger,
    )

    # Get the iterator and consume it
    response_settings = CheckResponse(
        check_response=validate_content,
        allowed_content=allowed_content,
    )
    iterator: AsyncGenerator[str, None] = chat_function(
        querystr,
        None,
        context,
        print_context=print_context,
        validate=response_settings,
        logger=logger,
    )

    async for chunk in iterator:
        yield chunk


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
