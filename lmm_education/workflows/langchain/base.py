"""
workflows.langchain.base module. Definitions and
utility functions for LangChain graph models.

Important definitions:

- ChatState and ChatWorkflContext: state and context for graphs
- create_initial_state: default initialized state
- graph_logger: a function to log ChatState to a .csv database
"""

# LangGraph missing type stubs
# pyright: reportMissingTypeStubs=false

from typing import TypedDict, Literal, Annotated
from datetime import datetime

from pydantic import BaseModel, Field
from pydantic.config import ConfigDict

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
)
from langchain_core.retrievers import BaseRetriever

from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph

from lmm.utils.logging import LoggerBase, ConsoleLogger
from lmm_education.config.appchat import ChatSettings
from .graph_logging import ChatDatabaseInterface


class ChatState(TypedDict):
    """
    State object for the chat workflow.

    This typed dictionary replaces the misuse of the history parameter
    for carrying metadata. Each field has a clear purpose:

    Attributes:
        messages: Conversation history as LangChain messages
        status: status of the graph
        model_identification: optional info on model used in node
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
    model_identification: str

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

    # system performance evaluation
    timestamp: datetime
    time_to_context: float | None
    time_to_FB: float | None
    time_to_response: float | None


def create_initial_state(
    querytext: str,
    *,
    messages: (
        list[HumanMessage | AIMessage | BaseMessage] | None
    ) = None,
    history: list[dict[str, str]] | None = None,
    timestamp: datetime | None = None,
) -> ChatState:
    """
    Create initial ChatState from query and optional history.

    Args:
        querytext: The user's query text
        history: optional Gradio-format conversation history
        timestamp: optional timestamp (will be provided if
            omitted)

    Returns:
        ChatState initialized for the workflow
    """

    return ChatState(
        messages=messages or [],
        status="valid",  # Will be validated by workflow
        model_identification="<unknown>",
        query=querytext,
        refined_query="",
        query_classification="",
        context="",
        response="",
        timestamp=timestamp or datetime.now(),
        time_to_context=None,
        time_to_FB=None,
        time_to_response=None,
    )


# (inherit from BaseModel as dataclass cannot be used for context)
class ChatWorkflowContext(BaseModel):
    """
    Configuration for the chat workflow.

    Encapsulates all dependencies and settings needed by the workflow,
    avoiding global state and making the workflow testable.
    """

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
        from lmm_education.stores.langchain.vector_store_qdrant_langchain import (
            AsyncQdrantVectorStoreRetriever as AsyncRetriever,
        )

        return ChatWorkflowContext(
            retriever=AsyncRetriever.from_config_settings(),
        )


# graph alias type
ChatStateGraphType = CompiledStateGraph[
    ChatState, ChatWorkflowContext, ChatState, ChatState
]


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
# Logging for this state, using its specific information
# DO NOT MOVE THIS FUNCTION -- for example, to graph_logging.
# The move creates nearby intractable circular import errors.

# ruff: noqa: E402
from .base import ChatState, ChatWorkflowContext
from lmm.utils.logging import LoggerBase
from langchain_core.messages import BaseMessage
from lmm.utils.hash import generate_random_string
from lmm.models.langchain.runnables import (
    RunnableType,
    create_runnable,
)


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
    # Deferred imports to avoid circular import with base.py
    # from .base import ChatState, ChatWorkflowContext

    if timestamp is None:
        timestamp = datetime.now()

    if record_id is None:
        record_id = generate_random_string()

    logger: LoggerBase = context.logger

    # info from state and context
    model_name: str = state.get("model_identification") or "<unknown>"

    messages: list[BaseMessage] = state.get("messages", [])
    status = state.get("status", "error")

    # Extract info from state
    response: str = state.get("response", "")
    query: str = state.get("query", "")
    classification: str = (
        state.get("query_classification", "") or "NA"
    )
    message_count = len(messages)

    # audit system performance
    time_to_context: float | None = state.get("time_to_context", None)
    time_to_FB: float | None = state.get("time_to_FB", None)
    time_to_response: float | None = state.get(
        "time_to_response", None
    )

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
                        time_to_context=time_to_context,
                        time_to_FB=time_to_FB,
                        time_to_response=time_to_response,
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
                        time_to_FB=time_to_FB,
                        time_to_response=time_to_response,
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
