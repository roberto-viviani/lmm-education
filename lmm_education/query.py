"""
This module allows interactively querying a language model, such that
it can be tested using material ingested in the RAG database.

A RAG database must have been previously created (for example, with
the ingest module). In the following examples, we assume the
existence of a database on basic statistical modelling.

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

Main functions:

- `create_chat_stream` and `create_chat_stringstream`: these functions
set up streams or LangGraph 'states' and of text, respectively, to
stream from the graph. They take as arguments the input to the graph
(i.e., the query text), a history of previous messages, a Workflow
context object for the graph, and optional parameters to configure the
function of the graph (validation of query) or the stream (logging
the exchange to a database).
- `query` and `aquery`: they take the query text and execute the
query, streaming the result. The provide an interface to config.toml,
but the settings contained there can be overridden by arguments.
Internally, they create a chat stream with a Workflow context
initialized from the config.toml parameters.

These two layers of functions handle construction of a streamable
graph to execute a query by taking care of the two layers of possible
specifications, i.e. from config.toml and for the components of the
graph.
```
"""

from collections.abc import (
    AsyncIterator,
    Iterator,
    Callable,
    Awaitable,
)
from typing import Literal
from io import TextIOBase

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

from lmm_education.config.appchat import ChatDatabase

# LM markdown for education
from .config.config import load_settings
from .config.config import ConfigSettings, DEFAULT_CONFIG_FILE
from qdrant_client import AsyncQdrantClient
from .stores.langchain.vector_store_qdrant_langchain import (
    AsyncQdrantVectorStoreRetriever as AsyncQdrantRetriever,
)
from .config.appchat import (
    ChatSettings,
    CheckResponse,
    load_settings as load_chat_settings,
)
from lmm_education.models.langchain.workflows.chat_graph import (
    ChatStateGraphType,
    ChatState,
    ChatWorkflowContext,
    workflow_library,
    graph_logger,
)
from lmm_education.models.langchain.workflows.stream_adapters import (
    tier_1_iterator,
    tier_3_iterator,
    stream_graph_state,
    stream_graph_updates,
    stateful_validation_adapter,
    tier_3_adapter,
    terminal_tier1_adapter,
    terminal_field_change_adapter,
)
from .logging_db import (
    ChatDatabaseInterface,
    CsvChatDatabase,
    CsvFileChatDatabase,
)


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
        query=querytext,
        query_prompt="",
        query_classification="",
        context="",
        response="",
    )


def create_chat_stream(
    querytext: str,
    history: list[dict[str, str]] | None,
    context: ChatWorkflowContext,
    *,
    stream_updates: bool = False,
    validate: CheckResponse | Literal[False] | None = None,
    database_log: (
        bool | tuple[TextIOBase, TextIOBase] | tuple[str, str]
    ) = False,
    logger: LoggerBase = ConsoleLogger(),
) -> tier_1_iterator:
    """
    Creates and configures the chat stream. Returns tuples of
    (mode, event) items as returned from the graph.

    This function retrieves a compiled LangGraph's graph, creates the
    initial state, and returns a configured stream to process a user
    query.

    Note:
        The tier_1_iterator contains all information from the graph
        stream, and may be combined with other stream adaptors to
        customize behaviour.
        Use create_chat_stringstream to get a stream of strings.

    Example:

        ```python
        try:
            stream = create_chat_stream(...)
            async for mode, event in stream:
                ...
        except Exception as e:
            ...
        ```

        ```python
        try:
            stream_raw = create_chat_stream(...)
            stream = tier_3_adapter(stream_raw)
            async for txt in stream:
                ...
        except Exception as e:
            ...
        ```
    Args:
        querytext: The user's query text
        history: List of previous message exchanges (Gradio format)
        context: a ChatWorkflowContext object for dependencies to be
            injected into the graph
        stream_updates: streams an updates channel (default to False)
        validate: if None, validates response using settings from
            context object. If False, carries out no validation. If a
            CheckReponse object, overrides the settings from
            the context object.
        database_log: if False (default), carries out no database
            logging of the exchanges. If True, carries out database
            logging with the settings defined in the context object,
            i.e. the files where the database is located. If a tuple
            of streams or file paths is provided, it uses those streams
            for logging.
        logger: Logger instance for info and error reporting

    Yields:
        (mode, event) tuples from the LLM stream

    Behaviour:
        This function does not stream the LLM response, it only sets
        up the stream and returns it. Exceptions will be raised in
        case the stream fails to initialize (usually, due to invalid
        settings or failure to acquire resources).
    """

    # Load settings.
    config_settings: ConfigSettings | None = load_settings(
        logger=logger
    )
    if config_settings is None:
        raise ValueError("Could not load settings")

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
    chat_database: ChatDatabaseInterface | None = None
    match database_log:
        case False:
            chat_database = None
        case True:
            # create logger using config info.
            db_settings: ChatDatabase = (
                context.chat_settings.chat_database
            )
            chat_database = CsvFileChatDatabase(
                db_settings.messages_database_file,
                db_settings.context_database_file,
            )
        case (TextIOBase(), TextIOBase()):
            # create logger using provided streams.
            chat_database = CsvChatDatabase(
                database_log[0], database_log[1]
            )
        case (str(), str()):
            # create logger using provided file paths.
            chat_database = CsvFileChatDatabase(
                database_log[0], database_log[1]
            )
        case _:
            raise ValueError(
                "chat_function: database_log "
                "must be a boolean, tuple of streams, or "
                "tuple of file paths"
            )

    # map dblogger to typed lambda (required for typing)
    dblogger: Callable[[ChatState], Awaitable[None]] | None = None
    if chat_database:

        async def _log_state(state: ChatState) -> None:
            # chat_database is captured from closure, but we verified it is not None
            if chat_database:
                await graph_logger(
                    database=chat_database,
                    state=state,
                    context=context,
                    client_host=context.client_host,
                    session_hash=context.session_hash,
                )

        dblogger = _log_state

    # Fetch graph from workflow library
    wfname = "query"
    try:
        workflow: ChatStateGraphType = workflow_library[wfname]
    except Exception as e:
        raise ValueError(
            f"Could not create workflow {wfname}:\n{e}"
        ) from e

    # Create initial state
    initial_state: ChatState = create_initial_state(
        querytext,
        history if history else None,
    )

    # Set up the stream
    raw_stream: tier_1_iterator = (
        stream_graph_updates(workflow, initial_state, context)
        if stream_updates
        else stream_graph_state(workflow, initial_state, context)
    )

    # Configure stream for validation requests
    tier_1_stream: tier_1_iterator = raw_stream
    if response_settings.check_response:
        # Initialize the validation model
        allowed_content = response_settings.allowed_content
        try:
            validator_model: RunnableType = create_runnable(
                "allowed_content_validator",
                allowed_content=allowed_content,  # type: ignore
            )
        except Exception as e:
            raise ValueError(
                f"Could not initialize validation model: {e}"
            ) from e

        tier_1_stream = stateful_validation_adapter(
            raw_stream,
            validator_model=validator_model,
            allowed_content=response_settings.allowed_content,
            buffer_size=response_settings.initial_buffer_size,
            error_message=context.chat_settings.MSG_WRONG_CONTENT,
            logger=logger,
        )

    if dblogger:
        tier_1_stream = terminal_tier1_adapter(
            tier_1_stream, on_terminal_state=dblogger
        )

    return tier_1_stream


def create_chat_stringstream(
    querytext: str,
    history: list[dict[str, str]] | None,
    context: ChatWorkflowContext,
    *,
    print_context: bool = False,
    validate: CheckResponse | Literal[False] | None = None,
    database_log: (
        bool | tuple[TextIOBase, TextIOBase] | tuple[str, str]
    ) = False,
    logger: LoggerBase = ConsoleLogger(),
) -> tier_3_iterator:
    """
    Creates and configures the chat stream.

    This function retrieves a compiled LangGraph's graph, creates the
    initial state, and returns a configured stream to process a user
    query.

    Example:

        ```python
        try:
            stream = create_chat_stream(...)
            async for chunk in stream:
                ...
        except Exception as e:
            ...
        ```

    Args:
        querytext: The user's query text
        history: List of previous message exchanges (Gradio format)
        context: a ChatWorkflowContext object for dependencies to be
            injected into the graph
        print_content: streams the retrieved context (default to False)
        validate: if None, validates response using settings from
            context object. If False, carries out no validation. If a
            CheckReponse object, overrides the settings from
            the context object.
        database_log: if False (default), carries out no database
            logging of the exchanges. If True, carries out database
            logging with the settings defined in the context object,
            i.e. the files where the database is located. If a tuple
            of streams or file paths is provided, it uses those streams
            for logging.
        logger: Logger instance for info and error reporting

    Yields:
        strings from the LLM stream

    Behaviour:
        This function does not stream the LLM response, it only sets
        up the stream and returns it. Exceptions will be raised in
        case the stream fails to initialize (usually, due to invalid
        settings or failure to acquire resources).
    """

    stream: tier_1_iterator = create_chat_stream(
        querytext=querytext,
        history=history,
        context=context,
        stream_updates=print_context,
        validate=validate,
        database_log=database_log,
        logger=logger,
    )

    if print_context:
        return terminal_field_change_adapter(
            stream,
            on_field_change={
                "context": lambda c: "CONTEXT:\n"
                + c
                + "\nEND CONTEXT------\n\n"
            },
        )
    else:
        return tier_3_adapter(
            stream,
        )


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
    Synchronous generator that yields a text stream from the graph
    coding the RAG chatting workflow.

    This is a convenience wrapper that allows synchronous code to
    consume the streaming response from the language model. It
    creates an event loop internally to bridge the async/sync
    boundary.

    Args:
        querystr: The query text to send to the language model
        model_settings: Language model settings (or 'major', 'minor',
            'aux'). If None (default) take settings from config.toml
        chat_settings: Chat settings for the query
        print_context: If True, streams the RAG context
        validate_content: If True, validate response content
        allowed_content: List of allowed content types for validation
        logger: Logger instance for error reporting

    Yields:
        str: Text chunks from the language model response

    Behaviour:
        On successful execution, it will yield the text chunks from
        the language model response. If exceptions are raised, it
        will yield an error message. Errors may also be propagated
        through the logger. This behaviour is consistent with its
        use in a CLI.

    Example:
        ```python
        # assumes a RAG vector database has been created
        for text in query("What is logistic regression?"):
            print(text, end="", flush=True)
        print()
        ```
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
    Asynchronous generator that yields a text stream from the graph
    coding the RAG chatting workflow.

    Args:
        querystr: The query text to send to the language model
        model_settings: Language model settings (or 'major', 'minor',
            'aux'). If None (default) take settings from config.toml
        chat_settings: Chat settings for the query
        print_context: If True, streams the RAG context
        validate_content: If True, validate response content
        allowed_content: List of allowed content types for validation
        client: Optional pre-configured Qdrant client
        logger: Logger instance for error reporting

    Yields:
        str: Text chunks from the language model response

    Behaviour:
        On successful execution, it will yield the text chunks from
        the language model response. If exceptions are raised, it
        will yield an error message. Errors may also be propagated
        through the logger. This behaviour is consistent with its
        use in a CLI.

    Example:
        ```python
        # assumes vector database is present
        async for text in aquery("What is logistic regression?"):
            print(text, end="", flush=True)
        print()
        ```
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
                f"Invalid language model settings: "
                f"{model_settings}\nShould"
                " be one of 'major', 'minor', 'aux'"
            )
            logger.error(errmsg)
            yield errmsg
            return

    try:
        llm: BaseChatModel = create_model_from_settings(
            model_settings
        )
    except Exception as e:
        logger.error(f"Could not load language model: {e}")
        yield f"Could not load language model: {e}"
        return

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
    response_settings = CheckResponse(
        check_response=validate_content,
        allowed_content=allowed_content,
    )  # this for override config settings
    context = ChatWorkflowContext(
        llm=llm,
        retriever=retriever,
        chat_settings=chat_settings.from_instance(
            check_response=response_settings
        ),
        logger=logger,
    )

    # Get the iterator and consume it
    try:
        iterator: tier_3_iterator = create_chat_stringstream(
            querystr,
            None,
            context,
            print_context=print_context,
            validate=response_settings,
            logger=logger,
        )
    except Exception as e:
        logger.error(f"Could not load chat stream: {e}")
        yield f"Could not load chat stream: {e}"
        return

    try:
        async for chunk in iterator:
            yield chunk
    except Exception as e:
        errmsg: str = f"Workflow streaming failed: {e}"
        logger.error(errmsg)
        yield errmsg
        return


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
