"""
LangGraph-based chat workflow for RAG-enabled language model
interactions.

This module provides a state-based graph architecture for processing
chat queries with retrieval-augmented generation (RAG). The workflow
handles:
- Query validation
- Context retrieval from vector store
- Query formatting with retrieved context
- LLM response generation

The graph is parametrized by dependencies injected through a
ChatWorkflowContext object. This object includes:

llm: BaseChatModel, an initialized LangChain object wrapping the model
retriever: BaseRetriever, an object wrapping the vector database
system_message: str, a system message overriding defaults
chat_settings: ChatSettings, an objects that loads the appchat.toml
    settings automatically, or allows them to be overridden when given
logger: LoggerBase, a logger object for errors printing

The remaining settings are used by streams to log the interaction to
a database. Except for .database, they are not set here.

database: ChatDatabaseInterface, the names of the files of the database
client_host: str = "<unknown>", filled in internally
session_hash: str = "<unknown>", filled in internally

To create a dependency injection using the config.toml settings, call

```python
context = ChatWorkflowContext.from_default_config()
```

The graph is designed for being used as a stream. The graph is created
as follows,

```python
workflow = create_chat_workflow()
state: ChatState = create_initial_state("What is logistic regression?")
dependencies = ChatWorkflowContext.from_default_config()
stream = workflow.astream(state, stream_mode="messages",
                context=dependencies)
```

after which the stream may be consumed directly. Note that this is a
LangGraph stream, returning tuples of (chunk, metadata):

```python
async for chunk, metadata in stream:
    ...
    print(chunk.text, sep="", flush=True)
```

Use stream adapters from the stream_adapters module to set up and
transform the stream. For example, to emit strings:

```python
from lmm_education.models.langchain.workflows.stream_adapters import (
    tier_2_to_3_adapter
)
text_stream = tier_2_to_3_adapter(stream)
async for txt in text_stream:
    print(txt, sep="", flush=True)
```

The function graph_logger supports writing the user/llm interaction to
a database, and is meant to be used by a streaming object.
"""

# LangGraph missing type stubs
# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownMemberType=false

from typing import TypedDict, Literal, Annotated
from collections.abc import Callable
from math import ceil
from datetime import datetime

from pydantic import BaseModel, Field
from pydantic.config import ConfigDict

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate

from langgraph.constants import TAG_NOSTREAM
from langgraph.types import RetryPolicy
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from langgraph.graph.state import CompiledStateGraph

from lmm.language_models.langchain.runnables import (
    RunnableType,
    create_runnable,
)
from lmm.utils.hash import generate_random_string
from lmm.utils.logging import LoggerBase, ConsoleLogger
from lmm.markdown.ioutils import convert_backslash_latex_delimiters

from lmm_education.config.config import ConfigSettings
from lmm_education.config.appchat import ChatSettings
from lmm_education.logging_db import ChatDatabaseInterface


class ChatState(TypedDict):
    """
    State object for the chat workflow.

    This typed dictionary replaces the misuse of the history parameter
    for carrying metadata. Each field has a clear purpose:

    Attributes:
        messages: Conversation history as LangChain messages
        status: status of the graph
        query: The current user query
        refined_query: Refined query integrating history
        query_classification: Semantic categorization of query
        context: Retrieved context from vector store
        response: Collects response for logging to database
    """

    # Conversation messages (uses LangGraph's message reducer)
    messages: Annotated[list[BaseMessage], add_messages]
    status: Literal[
        "valid", "empty_query", "long_query", "rejected", "error"
    ]

    # Query processing
    query: str
    refined_query: str
    query_classification: str  # (set by stream object)

    # context from vector database
    context: str

    # response. We need to collect this separately because
    # the streaming will include lmm streams, but these will
    # not be in the 'messages' field.
    response: str


def create_initial_state(query: str) -> ChatState:
    """Creates a default initial state, set to a user query."""

    return ChatState(
        messages=[],
        status="valid",
        query=query,
        refined_query="",
        query_classification="",
        context="",
        response="",
    )


# (inherit from BaseModel as dataclass cannot be used for context)
class ChatWorkflowContext(BaseModel):
    """
    Configuration for the chat workflow.

    Encapsulates all dependencies and settings needed by the workflow,
    avoiding global state and making the workflow testable.
    """

    llm: BaseChatModel
    retriever: BaseRetriever
    system_message: str = "You are a helpful assistant"  # maybe empty
    chat_settings: ChatSettings = Field(default_factory=ChatSettings)
    logger: LoggerBase = Field(default_factory=ConsoleLogger)

    # fields set dynamically at construction
    client_host: str = Field(default="<unknown>", min_length=6)
    session_hash: str = Field(default="<unknown>", min_length=6)

    # database log of interactions
    database: ChatDatabaseInterface | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_default_config(cls) -> 'ChatWorkflowContext':
        from lmm_education.config.config import ConfigSettings
        from lmm.language_models.langchain.models import (
            create_model_from_settings,
        )
        from lmm_education.stores.langchain.vector_store_qdrant_langchain import (
            AsyncQdrantVectorStoreRetriever as AsyncRetriever,
        )

        config = ConfigSettings()
        model = create_model_from_settings(config.major)

        return ChatWorkflowContext(
            llm=model,
            retriever=AsyncRetriever.from_config_settings(),
        )


# graph alias type
ChatStateGraphType = CompiledStateGraph[
    ChatState, ChatWorkflowContext, ChatState, ChatState
]


def create_chat_workflow() -> ChatStateGraphType:
    """
    Create the chat workflow graph with native streaming support.

    The graph implements the following flow:

    START               ---> validate query
    validate query      -.-> END [empty query]
                        -.-> END [long query]
                        -.-> integrate history
    integrate history   -.-> END [error in retrieval]
                        -.-> retrieve context
    retrieve context    ---> format query
    format query        ---> generate
    generate            ---> END

    Returns:
        Compiled StateGraph ready for streaming
    """

    def validate_query(
        state: ChatState, runtime: Runtime[ChatWorkflowContext]
    ) -> dict[str, str | AIMessage]:
        """Validate the user's query for length and content."""

        query: str = state.get("query", "")
        settings: ChatSettings = runtime.context.chat_settings

        if not query or not query.strip():
            return {
                "status": "empty_query",
                "response": settings.MSG_EMPTY_QUERY,
                "messages": AIMessage(
                    content=settings.MSG_EMPTY_QUERY
                ),
            }

        if len(query.split()) > settings.max_query_word_count:
            return {
                "status": "long_query",
                "response": settings.MSG_LONG_QUERY,
                "messages": AIMessage(
                    content=settings.MSG_LONG_QUERY
                ),
            }

        return {'status': "valid"}  # no change, everything ok.

    async def integrate_history(
        state: ChatState, runtime: Runtime[ChatWorkflowContext]
    ) -> dict[str, str | AIMessage]:
        """Integrate query with history"""
        # not for streaming

        query: str = state.get("query", "")
        context: ChatWorkflowContext = runtime.context

        messages: list[str] = [
            str(m.content)
            for m in state['messages']
            if isinstance(m.content, str)
        ]
        if not messages:
            return {
                "refined_query": query,
            }

        try:
            match context.chat_settings.history_integration:
                case 'none':
                    pass
                case 'summary':
                    # TODO: configure this through context
                    config = ConfigSettings()
                    model = create_runnable("summarizer", config.aux)
                    summary: str = await model.ainvoke(
                        {
                            'text': "\n---\n".join(messages),
                        },
                        config={'tags': [TAG_NOSTREAM]},
                    )
                    summary = summary.replace('text', 'chat')

                    # re-weight summary and query
                    weight: int = ceil(
                        len(summary.split())
                        / (len(query.split()) + 1)
                    )
                    query = " ".join([query] * weight)

                    # join summary and query
                    query = f"{summary}\n\nQuery: {query}"
                case 'context_extraction':
                    model = create_runnable("chat_summarizer")
                    summary: str = await model.ainvoke(
                        {
                            'text': "\n---\n".join(messages),
                            'query': query,
                        },
                        config={'tags': [TAG_NOSTREAM]},
                    )
                    summary = summary.replace('text', 'chat')

                    # re-weight summary and query. We repeat query
                    # to increase its wieght in the embedding.
                    weight: int = ceil(
                        len(summary.split())
                        / (len(query.split()) + 1)
                    )
                    query = " ".join([query] * weight)

                    # join summary and query
                    query = f"{summary}\n\nQuery: {query}"
                case 'rewrite':
                    model = create_runnable("rewrite_query")
                    query: str = await model.ainvoke(
                        {
                            'text': "\n---\n".join(messages),
                            'query': query,
                        },
                        config={'tags': [TAG_NOSTREAM]},
                    )

        except Exception as e:
            context.logger.error(f"Error integrating history: {e}")
            pass

        return {
            "refined_query": query,
        }

    async def retrieve_context(
        state: ChatState, runtime: Runtime[ChatWorkflowContext]
    ) -> dict[str, str | AIMessage]:
        """Retrieve relevant documents from vector store."""

        query: str = state["refined_query"]
        config: ChatWorkflowContext = runtime.context

        try:
            documents: list[Document] = (
                await config.retriever.ainvoke(query)
            )
            context: str = "\n-----\n".join(
                [d.page_content for d in documents]
            )

            # TODO: include doc id in logging (revise db schema)
            # Store document metadata for logging
            # doc_metadata: list[dict[str, Any]] = [
            #     {
            #         "content": (
            #             d.page_content[:200] + "..."
            #             if len(d.page_content) > 200
            #             else d.page_content
            #         ),
            #         "metadata": d.metadata,
            #     }
            #     for d in documents
            # ]

            return {"context": context}

        except Exception as e:
            config.logger.error(
                f"Error retrieving from vector database:\n{e}"
            )
            settings: ChatSettings = config.chat_settings
            return {
                "status": "error",
                "response": settings.MSG_ERROR_QUERY,
                "messages": AIMessage(
                    content=settings.MSG_ERROR_QUERY
                ),
            }

    def format_query(
        state: ChatState, runtime: Runtime[ChatWorkflowContext]
    ) -> dict[str, str | AIMessage]:
        """Format the query with retrieved context using prompt
        template."""

        context: str = state.get("context", "")
        query: str = state["query"]

        settings: ChatSettings = runtime.context.chat_settings

        # Convert LaTeX delimiters for display
        formatted_context = convert_backslash_latex_delimiters(
            context
        )

        # Format with prompt template
        template = PromptTemplate.from_template(
            settings.PROMPT_TEMPLATE
        )
        formatted_query: str = template.format(
            context=formatted_context,
            query=query,
        )

        return {"refined_query": formatted_query}

    async def generate(
        state: ChatState, runtime: Runtime[ChatWorkflowContext]
    ) -> dict[str, str | AIMessage]:
        """
        Generate LLM response using streaming.

        This node invokes the LLM with the formatted query and
        conversation history. The actual streaming is handled by
        LangGraph's astream() with stream_mode="messages".

        Note: This node stores the final complete response in state,
            but when using astream() with stream_mode="messages",
            you'll receive chunks as they're generated.
        """
        # for streaming

        config: ChatWorkflowContext = runtime.context
        messages = prepare_messages_for_llm(
            state, config, config.system_message
        )

        # Stream the response - chunks will be emitted by
        # LangGraph's astream, even if not 'yielded'
        response_chunks: list[str] = []
        try:
            async for chunk in config.llm.astream(messages):
                # Extract text from AIMessageChunk
                # LLM.astream() returns AIMessageChunk objects with .content attribute
                if hasattr(chunk, "content"):
                    content: str = str(chunk.content)  # type: ignore
                    response_chunks.append(content)
                else:
                    # Fallback for unexpected chunk types
                    response_chunks.append(str(chunk))
        except Exception as e:
            context: ChatWorkflowContext = runtime.context
            context.logger.error(
                f"Error while streaming in 'generate' node: {e}"
            )
            return {
                'status': "error",
                'response': context.chat_settings.MSG_ERROR_QUERY,
                'messages': AIMessage(
                    content=context.chat_settings.MSG_ERROR_QUERY
                ),
            }

        # Store complete response in state (for logging)
        complete_response = "".join(response_chunks)

        return {
            "response": complete_response,
        }

    # utility functions for add_conditional_edges
    def continue_if_valid(
        next_node: str,
    ) -> Callable[[ChatState], str]:
        continuation: Callable[[ChatState], str] = lambda state: (
            next_node
            if state.get("status", "error") == "valid"
            else END
        )
        return continuation

    def continue_if_no_error(
        next_node: str,
    ) -> Callable[[ChatState], str]:
        continuation: Callable[[ChatState], str] = lambda state: (
            END
            if state.get("status", "error") == "error"
            else next_node
        )
        return continuation

    # Build the graph------------------------------------------------
    workflow: StateGraph[
        ChatState, ChatWorkflowContext, ChatState, ChatState
    ] = StateGraph(ChatState, ChatWorkflowContext)

    # Add nodes
    workflow.add_node("validate_query", validate_query)
    workflow.add_node("integrate_history", integrate_history)
    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("format_query", format_query)
    workflow.add_node(
        "generate",
        generate,
        # automatically retry network issues and rate limits
        retry_policy=RetryPolicy(
            max_attempts=3, initial_interval=1.0
        ),
    )

    # Add edges
    workflow.add_edge(START, "validate_query")
    workflow.add_conditional_edges(
        "validate_query",
        continue_if_valid("integrate_history"),
    )
    workflow.add_conditional_edges(
        "integrate_history",
        continue_if_no_error("retrieve_context"),
    )
    workflow.add_conditional_edges(
        "retrieve_context", 
        continue_if_no_error("format_query"))
    workflow.add_edge("format_query", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


def workflow_factory(workflow_name: str) -> ChatStateGraphType:
    """
    Factory function to retrieve compiled workflow graphs.
    """

    # At present, we only have one graph, so we put this function
    # here, but it will be moved to a factory module when we have
    # more workflows.
    match workflow_name:
        case "query":  # only query at first chat
            return create_chat_workflow()
        case _:
            raise ValueError(f"Invalid workflow: {workflow_name}")


def prepare_messages_for_llm(
    state: ChatState,
    context: ChatWorkflowContext,
    system_message: str = "",
    history_window: int | None = None,
) -> list[tuple[str, str]]:
    """
    Prepare messages for LLM invocation from state.

    This replaces the _prepare_messages function, using state instead
    of the history list.

    Args:
        state: Current chat state
        system_message: Optional system message to prepend
        history_window: Number of recent messages to include
            (defaults to 4)

    Returns:
        List of (role, content) tuples for LLM invocation
    """
    if history_window is None:
        history_window = context.chat_settings.history_length

    messages: list[tuple[str, str]] = []

    if system_message:
        messages.append(("system", system_message))

    # Add recent conversation history
    state_messages: list[BaseMessage] = state.get("messages", [])
    for msg in state_messages[-history_window:]:
        role: str = (
            "user" if isinstance(msg, HumanMessage) else "assistant"
        )
        content: str = ""
        try:
            content = str(msg.content)  # type: ignore (dict type)
        except Exception:
            pass
        messages.append((role, content))

    # Add the current formatted query
    messages.append(("user", state["refined_query"]))

    return messages


# -----------------------------------------------------------------
# Logging for this graph, using its specific information


async def graph_logger(
    state: ChatState,
    database: ChatDatabaseInterface,
    context: ChatWorkflowContext,
    client_host: str = "<unknown>",
    session_hash: str = "<unknown>",
    timestamp: datetime | None = None,
    record_id: str | None = None,
) -> None:
    """
    Log to database function.

    Extracts information from the state and context and delegates to
    the database interface. Adds context validation information to the
    log.
    """

    if timestamp is None:
        timestamp = datetime.now()

    if record_id is None:
        record_id = generate_random_string()

    logger: LoggerBase = context.logger

    # info from state and context
    model_name: str | None
    model_name = context.llm.name
    if not model_name:
        if hasattr(context.llm, "model_name"):
            model_name = str(context.llm.model_name)  # type: ignore
    if not model_name:
        model_name = "<unknown>"

    messages: list[BaseMessage] = state.get("messages", [])
    status = state.get("status", "error")

    # Extract info from state
    response: str = state.get("response", "")
    query: str = state.get("query", "")
    classification: str = (
        state.get("query_classification", "") or "NA"
    )
    message_count = len(messages)

    try:
        match status:
            case "valid":
                # Check for context
                context_text: str = state.get("context", "")

                if context_text:
                    # Evaluate consistency of context prior to saving
                    validation: str = "<unknown>"
                    try:
                        lmm_validator: RunnableType = create_runnable(
                            "context_validator"  # will be a lookup
                        )
                        validation_res: str = (
                            await lmm_validator.ainvoke(
                                {
                                    "query": f"{query}. {response}",
                                    "context": context_text,
                                }
                            )
                        )
                        validation = str(
                            validation_res.strip().upper()
                        )
                    except Exception as e:
                        logger.error(
                            f"Could not connect to aux model to validate context: {e}"
                        )
                        validation = "<failed>"

                    await database.log_message_with_context(
                        record_id=record_id,
                        client_host=client_host,
                        session_hash=session_hash,
                        timestamp=timestamp,
                        message_count=message_count,
                        model_name=model_name,
                        interaction_type="MESSAGES",
                        query=query,
                        response=response,
                        validation=validation,
                        context=context_text,
                        classification=classification,
                    )
                else:
                    await database.log_message(
                        record_id=record_id,
                        client_host=client_host,
                        session_hash=session_hash,
                        timestamp=timestamp,
                        message_count=message_count,
                        model_name=model_name,
                        interaction_type="MESSAGES",
                        query=query,
                        response=response,
                    )

            case "empty_query":
                await database.log_message(
                    record_id=record_id,
                    client_host=client_host,
                    session_hash=session_hash,
                    timestamp=timestamp,
                    message_count=message_count,
                    model_name="",
                    interaction_type="EMPTYQUERY",
                    query="",
                    response=response,
                )

            case "long_query":
                await database.log_message(
                    record_id=record_id,
                    client_host=client_host,
                    session_hash=session_hash,
                    timestamp=timestamp,
                    message_count=message_count,
                    model_name="",
                    interaction_type="LONGQUERY",
                    query=(
                        query[:1500] + "..."
                        if len(query) > 1500
                        else query
                    ),
                    response=response,
                )

            case "rejected":
                # Check for context
                context_text: str = state.get("context", "")

                await database.log_message_with_context(
                    record_id=record_id,
                    client_host=client_host,
                    session_hash=session_hash,
                    timestamp=timestamp,
                    message_count=message_count,
                    model_name=model_name,
                    interaction_type="REJECTED",
                    query=query,
                    response=response,
                    validation="NA",
                    context=context_text,
                    classification=classification,
                )

            case _:  # ignore all others
                pass

    except Exception as e:
        logger.error(f"Async logging failed: {e}")
