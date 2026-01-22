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

The state object (ChatState) cleanly separates concerns that were
previously conflated in the history parameter.
"""

# LangGraph missing type stubs
# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownMemberType=false

# ruff: noqa: E402

from typing import TypedDict, Literal, Annotated
from math import ceil

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

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime

from langgraph.graph.state import CompiledStateGraph
from lmm.utils.logging import LoggerBase, ConsoleLogger
from lmm.markdown.ioutils import convert_backslash_latex_delimiters
from lmm_education.config.config import ConfigSettings
from lmm_education.config.appchat import ChatSettings
from lmm_education.logging_db import ChatDatabaseInterface


# Type aliases
ChatStatus = Literal[
    "valid", "empty_query", "long_query", "rejected", "error"
]


class ChatState(TypedDict):
    """
    State object for the chat workflow.

    This typed dictionary replaces the misuse of the history parameter
    for carrying metadata. Each field has a clear purpose:

    Attributes:
        messages: Conversation history as LangChain messages
        status: status of the graph
        query_text: The current user query (possibly formatted with
            context)
        context: Retrieved context from vector store
    """

    # Conversation messages (uses LangGraph's message reducer)
    messages: Annotated[list[BaseMessage], add_messages]
    status: ChatStatus

    # Query processing - required field
    query: str
    query_prompt: str
    query_classification: str

    # RAG context
    context: str

    # response
    response: str


def create_initial_state(query: str) -> ChatState:
    """Creates a default initial state, set to a user query."""

    return ChatState(
        messages=[],
        status="valid",
        query=query,
        query_prompt="",
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
    system_message: str = "You are a helpful assistant"
    chat_settings: ChatSettings = Field(default_factory=ChatSettings)
    client_host: str = "<unknown>"
    session_hash: str = "<unknown>"
    logger: LoggerBase = Field(default_factory=ConsoleLogger)
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

    START → validate_query → [conditional]
                                ↓ valid
                            retrieve_context → format_query →
                                                    generate → END
                                ↓ invalid
                            END

    The generate node uses llm.astream() to produce streaming output.
    Dependency injection handled at the level of streaming call
    through context argument. Use
    ```python
    workflow.astream(state, stream_mode="messages",
                    context=workflow_context)
    ```
    to consume the stream.

    Returns:
        Compiled StateGraph ready for streaming
    """

    def validate_query(
        state: ChatState, runtime: Runtime[ChatWorkflowContext]
    ) -> dict[str, str | AIMessage]:
        """Validate the user's query for length and content."""
        # note: separate node hier, as this needs be streamed.

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

        if state['messages']:
            messages: list[str] = [
                str(m.content)
                for m in state['messages']
                if isinstance(m.content, str)
            ]
            match runtime.context.chat_settings.history_integration:
                case 'none':
                    pass
                case 'summary':
                    # TODO: configure this through context
                    config = ConfigSettings()
                    model = create_runnable("summarizer", config.aux)
                    summary: str = await model.ainvoke(
                        {
                            'text': "\n---\n".join(messages),
                        }
                    )
                    summary = summary.replace('text', 'chat')

                    # re-weight summary and query
                    weight: int = ceil(
                        len(summary.split()) / len(query.split())
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
                        }
                    )
                    summary = summary.replace('text', 'chat')

                    # re-weight summary and query
                    weight: int = ceil(
                        len(summary.split()) / len(query.split())
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
                        }
                    )

        return {
            "query_prompt": query,
        }

    async def retrieve_context(
        state: ChatState, runtime: Runtime[ChatWorkflowContext]
    ) -> dict[str, str | AIMessage]:
        """Retrieve relevant documents from vector store."""

        query: str = state["query_prompt"]
        config: ChatWorkflowContext = runtime.context

        try:
            documents: list[Document] = (
                await config.retriever.ainvoke(query)
            )
            context: str = "\n-----\n".join(
                [d.page_content for d in documents]
            )

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

        print("\n???????????????")
        print(formatted_query)

        return {"query_prompt": formatted_query}

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
            state, config.system_message
        )

        # Stream the response - chunks will be emitted by astream()
        response_chunks: list[str] = []
        async for chunk in config.llm.astream(messages):
            # Extract text from chunk safely
            if hasattr(chunk, "text") and callable(chunk.text):
                response_chunks.append(chunk.text)
            elif hasattr(chunk, "content"):
                content: str = str(chunk.content)  # type: ignore
                response_chunks.append(content)
            else:
                response_chunks.append(str(chunk))

        # Store complete response in state (for non-streaming access)
        complete_response = "".join(response_chunks)

        return {
            "response": complete_response,
        }

    def should_retrieve(state: ChatState) -> str:
        """Conditional edge: check if query is valid."""
        if state.get("status") == "valid":
            return "retrieve"
        return "error"

    # Build the graph------------------------------------------------
    workflow: StateGraph[
        ChatState, ChatWorkflowContext, ChatState, ChatState
    ] = StateGraph(ChatState, ChatWorkflowContext)

    # Add nodes
    workflow.add_node("validate_query", validate_query)
    workflow.add_node("integrate_history", integrate_history)
    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("format_query", format_query)
    workflow.add_node("generate", generate)

    # Add edges
    workflow.add_edge(START, "validate_query")
    workflow.add_conditional_edges(
        "validate_query",
        should_retrieve,
        {
            "retrieve": "integrate_history",
            "error": END,
        },
    )
    workflow.add_edge("integrate_history", "retrieve_context")
    workflow.add_edge("retrieve_context", "format_query")
    workflow.add_edge("format_query", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


def _workflow_factory(workflow_name: str) -> ChatStateGraphType:
    """
    Factory function to store chat agents in global library using
    a LazyLoadingDict.
    """

    match workflow_name:
        case "query":  # only query at first chat
            return create_chat_workflow()
        case _:
            raise ValueError(f"Invalid workflow: {workflow_name}")


# At present, we put the dict here, but will be moved to a setup
# file when we have more workflows.
from lmm.language_models.lazy_dict import LazyLoadingDict

workflow_library: LazyLoadingDict[str, ChatStateGraphType] = (
    LazyLoadingDict(_workflow_factory)
)


def prepare_messages_for_llm(
    state: ChatState,
    system_message: str = "",
    history_window: int = 4,
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
    messages.append(("user", state["query_prompt"]))

    return messages


# -----------------------------------------------------------------
# Logging for this graph, using its specific information
from datetime import datetime
from lmm.language_models.langchain.runnables import (
    RunnableType,
    create_runnable,
)
from lmm.utils.hash import generate_random_string


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
    status: ChatStatus = state.get("status", "error")

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
                        validation = validation_res.upper()
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
