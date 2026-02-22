"""
Routing helper functions for LangGraph conditional edges.

These functions create routing callbacks for use with
`StateGraph.add_conditional_edges()`. They check the `status` field
of ChatState to determine whether to continue or end the graph.
"""

from collections.abc import Callable

from langgraph.graph import END  # type: ignore (missing stubs)
from langgraph.prebuilt import tools_condition

from .base import ChatState


def continue_if_valid(
    next_node: str,
) -> Callable[[ChatState], str]:
    """Route to next_node if status is 'valid', otherwise END."""
    continuation: Callable[[ChatState], str] = lambda state: (
        next_node if state.get("status", "error") == "valid" else END
    )
    return continuation


def continue_if_no_error(
    next_node: str,
) -> Callable[[ChatState], str]:
    """Route to END if status is 'error', otherwise next_node."""
    continuation: Callable[[ChatState], str] = lambda state: (
        END if state.get("status", "error") == "error" else next_node
    )
    return continuation


def continue_after_tool_call(
    tool_node: str,
    next_node: str,
) -> Callable[[ChatState], str]:
    """Route after generate: check errors, then tool calls.

    Note: This function replaces the `tool_condition` utility
    provided by LangGraph because the graph may return an empty
    message list when it does not call any tool. The graph
    outputs to the `results` key, not `messages`. However,
    the `tool_condition` function raises an error when the
    message list is empty."""

    def _continuation(state: ChatState) -> str:
        # No messages generated
        if not state.get('messages', []):
            return next_node

        # Error occurred in generate node
        if state.get('status') == "error":
            return END

        # Check if tools were called
        nextval = tools_condition(state)  # type: ignore
        return tool_node if nextval == 'tools' else next_node

    return _continuation


def continue_to_generate_or_fallback(
    generate_node: str,
    fallback_node: str,
    max_tool_calls: int = 3,
) -> Callable[[ChatState], str]:
    """Route after check_tool_result to either generate or fallback.
    
    If the max tool call count is reached, forces generation using
    the fallback node. Otherwise, returns to the core generate agent.
    """

    def _continuation(state: ChatState) -> str:
        # Stop everything on error
        if state.get("status", "error") == "error":
            return END
            
        tool_call_count: int = state.get("tool_call_count", 0)
        
        # fallback path if loop gets out of control
        if tool_call_count >= max_tool_calls:
            return fallback_node
            
        return generate_node

    return _continuation
