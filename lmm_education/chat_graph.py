"""
LangGraph-based chat workflow for RAG-enabled language model interactions.

This module provides a state-based graph architecture for processing chat
queries with retrieval-augmented generation (RAG). The workflow handles:
- Query validation
- Context retrieval from vector store
- Query formatting with retrieved context
- LLM response generation

The state object (ChatState) cleanly separates concerns that were previously
conflated in the history parameter.
"""

# pyright: reportMissingTypeStubs=false
# pyright: reportMissingTypeArgument=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false

from typing import TypedDict, Literal, Annotated

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
from lmm_education.config.appchat import ChatSettings


# Type aliases
ChatStatus = Literal["valid", "empty_query", "long_query", "error"]
ValidationStatus = Literal["pending", "passed", "failed", "skipped"]


class ChatState(TypedDict):
    """
    State object for the chat workflow.

    This typed dictionary replaces the misuse of the history parameter
    for carrying metadata. Each field has a clear purpose:

    Attributes:
        messages: Conversation history as LangChain messages
        status: status of the graph
        query_text: The current user query (possibly formatted with context)
        context: Retrieved context from vector store
    """

    # Conversation messages (uses LangGraph's message reducer)
    messages: Annotated[list[BaseMessage], add_messages]
    status: ChatStatus

    # Query processing - required field
    query_text: str

    # RAG context
    context: str  # TODO: check this is not a list


def create_initial_state(query: str) -> ChatState:
    """Creates a default initial state, set to a user query."""

    return ChatState(
        messages=[], status="valid", query_text=query, context=""
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
    print_context: bool = False
    client_host: str = "<unknown>"
    session_hash: str = "<unknown>"
    logger: LoggerBase = Field(default_factory=ConsoleLogger)

    model_config = ConfigDict(arbitrary_types_allowed=True)


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
                              retrieve_context → format_query → generate → END
                                    ↓ invalid
                              END

    The generate node uses llm.astream() to produce streaming output.
    Dependency injection handled at the level of streaming call
    through context argument. Use
    workflow.astream(state, stream_mode="messages", context=workflow_context)
    to consume the stream.

    Returns:
        Compiled StateGraph ready for streaming
    """

    def validate_query(
        state: ChatState, runtime: Runtime[ChatWorkflowContext]
    ) -> ChatState:
        """Validate the user's query for length and content."""
        query: str = state.get("query_text", "")
        settings: ChatSettings = runtime.context.chat_settings

        if not query or not query.strip():
            return {
                **state,
                "status": "empty_query",
                "messages": [
                    AIMessage(content=settings.MSG_EMPTY_QUERY)
                ],
            }

        if len(query.split()) > settings.max_query_word_count:
            return {
                **state,
                "status": "long_query",
                "messages": [
                    AIMessage(content=settings.MSG_LONG_QUERY)
                ],
            }

        # Normalize query text
        normalized_query = query.replace(
            "the textbook", "the context provided"
        )

        return {
            **state,
            "query_text": normalized_query,
            "status": "valid",
        }

    async def retrieve_context(
        state: ChatState, runtime: Runtime[ChatWorkflowContext]
    ) -> ChatState:
        """Retrieve relevant documents from vector store."""
        query: str = state["query_text"]
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

            new_state: ChatState = {
                **state,
                "status": "valid",
                "context": context,
            }
            # this copies context to messages channel so that it will
            # be streamed to output. This has the downside that the
            # message list is messed up. The alternative is to stream
            # the state and filter it, moving the logic to the caller.
            if config.print_context:
                message: list[BaseMessage] = [
                    AIMessage(
                        content="CONTEXT:\n"
                        + context
                        + "\nEND CONTEXT------\n\n"
                    )
                ]
                new_state['messages'] = message

            return new_state

        except Exception as e:
            config.logger.error(
                f"Error retrieving from vector database:\n{e}"
            )
            settings: ChatSettings = config.chat_settings
            return {
                **state,
                "status": "error",
                "messages": [
                    AIMessage(content=settings.MSG_ERROR_QUERY)
                ],
                "context": "",
            }

    def format_query(
        state: ChatState, runtime: Runtime[ChatWorkflowContext]
    ) -> ChatState:
        """Format the query with retrieved context using prompt template."""
        context: str = state.get("context", "")
        query: str = state["query_text"]
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

        return {
            **state,
            "query_text": formatted_query,
        }

    async def generate(
        state: ChatState, runtime: Runtime[ChatWorkflowContext]
    ) -> ChatState:
        """
        Generate LLM response using streaming.

        This node invokes the LLM with the formatted query and conversation
        history. The actual streaming is handled by LangGraph's astream()
        with stream_mode="messages".

        Note: This node stores the final complete response in state, but
        when using astream() with stream_mode="messages", you'll receive
        chunks as they're generated.
        """
        config: ChatWorkflowContext = runtime.context
        messages = prepare_messages_for_llm(
            state, config.system_message
        )

        # Stream the response - chunks will be emitted by astream()
        response_chunks: list[str] = []
        async for chunk in config.llm.astream(messages):
            # Extract text from chunk safely
            if hasattr(chunk, 'text') and callable(chunk.text):
                response_chunks.append(chunk.text)
            elif hasattr(chunk, 'content'):
                content = chunk.content  # type: ignore
                if isinstance(content, str):
                    response_chunks.append(content)
                else:
                    response_chunks.append(str(content))
            else:
                response_chunks.append(str(chunk))

        # Store complete response in state (for non-streaming access)
        complete_response = "".join(response_chunks)

        return {
            **state,
            "messages": state.get("messages", [])
            + [AIMessage(content=complete_response)],
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
    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("format_query", format_query)
    workflow.add_node("generate", generate)

    # Add edges
    workflow.add_edge(START, "validate_query")
    workflow.add_conditional_edges(
        "validate_query",
        should_retrieve,
        {
            "retrieve": "retrieve_context",
            "error": END,
        },
    )
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
        history_window: Number of recent messages to include (default 4)

    Returns:
        List of (role, content) tuples for LLM invocation
    """
    messages: list[tuple[str, str]] = []

    if system_message:
        messages.append(("system", system_message))

    # Add recent conversation history
    state_messages = state.get("messages", [])
    for msg in state_messages[-history_window:]:
        if isinstance(msg, HumanMessage):
            messages.append(("user", str(msg.content)))
        elif isinstance(msg, AIMessage):
            messages.append(("assistant", str(msg.content)))

    # Add the current formatted query
    messages.append(("user", state["query_text"]))

    return messages
