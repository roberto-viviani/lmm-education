"""
Shared graph node functions for LangGraph chat workflows.

This module contains node functions used by both the simple workflow
(`chat_graph.py`) and the agentic workflow (`chat_agent.py`).

Nodes defined here:
- `validate_query`: validates the user's query for length and content
- `create_integrate_history_node`: factory that returns an
  `integrate_history` node function. The factory captures the
  ConfigSettings to access settings.aux for the summarizer model.

Nodes unique to each graph remain in their respective modules.
"""

# LangGraph missing type stubs
# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownMemberType=false

from collections.abc import Callable
from math import ceil
from datetime import datetime
from types import CoroutineType

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from langgraph.constants import TAG_NOSTREAM
from langgraph.runtime import Runtime

from lmm.models.langchain.models import (
    create_model_from_settings,
)
from lmm.models.langchain.runnables import (
    RunnableType,
    create_runnable,
)

from lmm_education.config.config import ConfigSettings
from lmm_education.config.appchat import ChatSettings

from .base import (
    ChatState,
    ChatWorkflowContext,
    prepare_messages_for_llm,
)

# Node return type — a partial state update
NodeReturn = dict[str, str | AIMessage | float]

# Type for nodes created by factories
FactoryNode = Callable[
    ..., CoroutineType[any, any, dict[str, str | AIMessage | float]]
]


def validate_query(
    state: ChatState, runtime: Runtime[ChatWorkflowContext]
) -> NodeReturn:
    """Validate the user's query for length and content."""

    query: str = state.get("query", "")
    settings: ChatSettings = runtime.context.chat_settings

    if not query or not query.strip():
        return {
            "status": "empty_query",
            "response": settings.MSG_EMPTY_QUERY,
            "messages": AIMessage(content=settings.MSG_EMPTY_QUERY),
            "time_to_response": (
                datetime.now() - state['timestamp']
            ).total_seconds(),
        }

    if len(query.split()) > settings.max_query_word_count:
        return {
            "status": "long_query",
            "response": settings.MSG_LONG_QUERY,
            "messages": AIMessage(content=settings.MSG_LONG_QUERY),
            "time_to_response": (
                datetime.now() - state['timestamp']
            ).total_seconds(),
        }

    return {'status': "valid"}  # no change, everything ok.


def create_integrate_history_node(
    settings: ConfigSettings,
) -> FactoryNode:
    """Factory that creates an integrate_history node function.

    The factory captures ConfigSettings so the node can access
    `settings.aux` for the summarizer model, preserving the
    closure pattern used by LangGraph nested node definitions.

    Args:
        settings: ConfigSettings providing the aux model config

    Returns:
        An async node function with the LangGraph-compatible
        signature ``(ChatState, Runtime[ChatWorkflowContext])``
    """

    async def integrate_history(
        state: ChatState, runtime: Runtime[ChatWorkflowContext]
    ) -> NodeReturn:
        """Integrate query with history."""
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

        model: RunnableType | None = None
        try:
            match context.chat_settings.history_integration:
                case 'none':
                    pass
                case 'summary':
                    model = create_runnable(
                        "summarizer", settings.aux
                    )
                    summary: str = await model.ainvoke(
                        {
                            'text': "\n---\n".join(messages),
                        },
                        config={'tags': [TAG_NOSTREAM]},
                    )

                    # re-weight summary and query. We repeat query
                    # to increase its weight in the embedding.
                    summary_len = len(summary.split())
                    weight: int = (
                        1
                        if summary_len < 15
                        else ceil(summary_len / len(query.split()))
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

                    # re-weight summary and query. We repeat query
                    # to increase its weight in the embedding.
                    summary_len = len(summary.split())
                    weight: int = (
                        1
                        if summary_len < 15
                        else ceil(summary_len / len(query.split()))
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
            "model_identification": (
                model.get_name() if model else "<unknown>"
            ),
            "refined_query": query,
        }

    return integrate_history


def create_generate_node(
    settings: ConfigSettings,
    llm_major: BaseChatModel | None = None,
) -> FactoryNode:
    """Factory that creates a generic generate node function.

    This node invokes a tool-less LLM with the formatted query and
    conversation history. The streaming is handled by
    LangGraph's astream() with stream_mode="messages".

    Args:
        settings: a ConfigSettings object (for settings.major)
        llm_major: an override of settings.major (used to inject
            a model for testing purposes)

    Returns:
        An async node function returning dict[str, str | AIMessage | float]
    """

    async def generate(
        state: ChatState, runtime: Runtime[ChatWorkflowContext]
    ) -> NodeReturn:

        config: ChatWorkflowContext = runtime.context
        messages = prepare_messages_for_llm(
            state, config, config.system_message
        )

        llm: BaseChatModel = llm_major or create_model_from_settings(
            settings.major
        )

        response_chunks: list[str] = []
        first_chunk: datetime | None = None
        try:
            async for chunk in llm.astream(messages):
                if first_chunk is None:
                    first_chunk = datetime.now()
                if hasattr(chunk, "text"):
                    content: str = chunk.text
                    response_chunks.append(content)
                else:
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

        latency_resp: float = (
            datetime.now() - state['timestamp']
        ).total_seconds()
        latency_FB: float = (
            -1
            if first_chunk is None
            else (first_chunk - state['timestamp']).total_seconds()
        )
        return {
            'model_identification': (
                llm.get_name() if llm_major else settings.major.model
            ),
            'response': "".join(response_chunks),
            'time_to_FB': latency_FB,
            'time_to_response': latency_resp,
        }

    return generate
