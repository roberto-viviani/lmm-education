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

from langchain_core.messages import AIMessage

from langgraph.constants import TAG_NOSTREAM
from langgraph.runtime import Runtime

from lmm.models.langchain.runnables import (
    RunnableType,
    create_runnable,
)

from lmm_education.config.config import ConfigSettings
from lmm_education.config.appchat import ChatSettings

from .base import (
    ChatState,
    ChatWorkflowContext,
)

# Node return type — a partial state update
NodeReturn = dict[str, str | AIMessage | float]


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


# Type alias for the integrate_history node function signature


IntegratedHistoryNode = Callable[
    ..., CoroutineType[any, any, dict[str, str | AIMessage | float]]
]


def create_integrate_history_node(
    settings: ConfigSettings,
) -> IntegratedHistoryNode:
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
