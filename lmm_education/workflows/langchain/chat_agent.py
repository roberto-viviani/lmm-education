"""
Agentic workflow for handling RAG interactions. The agent decides
how to use the vector database to retrieve information to respond
to the user.

The workflow handles:
- Query validation, including responding to confused queries
- Integration of previous messages
- Context retrieval from vector store
- LLM response generation

The graph workflow is created with the `create_chat_agent` function
taking a ConfigSettings argument to specify major, minor, and aux
models:

```python
settings = ConfigSettings()
workflow = create_chat_agent(settings)
state: ChatState = create_initial_state("What is logistic regression?")
dependencies = ChatWorkflowContext.from_default_config()
stream = workflow.astream(state, stream_mode="messages",
                context=dependencies)
```

Please note that the major model must support tool use. When using
an OpenAI model, be sure that config.toml specifies the responses API:

```ini
[major.provider_params]
use_responses_api = true
```

After creation, the stream may be consumed directly. Note that this is a
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

Note:
    to support streaming, the graph does not provide the output
    in the `messages`, but in the `response` channel instead. When
    used with `.ainvoke()`, extract the response from this key in
    the returned state. When used with `.astream()`, consume the
    stream.
"""

# rev c 1.25

# pyright: standard
# pyright: reportAttributeAccessIssue=false

from collections.abc import Sequence
from datetime import datetime, timedelta

from .base import (
    ChatState,
    ChatWorkflowContext,
    ChatStateGraphType,
    prepare_messages_for_llm,
)
from .nodes import validate_query, create_integrate_history_node
from .graph_routing import (
    continue_if_valid,
    continue_if_no_error,
    continue_after_tool_call,
)

from langchain_core.runnables import Runnable
from langchain_core.tools.base import BaseTool
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    AIMessageChunk,
)
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool, ToolRuntime

from langgraph.types import RetryPolicy
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime
from langgraph.prebuilt import ToolNode

from lmm.models.langchain.models import (
    create_model_from_settings,
)

from lmm_education.config.config import ConfigSettings


def create_chat_agent(
    settings: ConfigSettings,
    llm_major: BaseChatModel | None = None,
) -> ChatStateGraphType:
    """
    Create the agentic chat workflow graph with native streaming
    support.

    The graph implements the following flow:

    START               ---> validate query
    validate query      -.-> END [empty query]
                        -.-> END [long query]
                        -.-> integrate history
    integrate history   -.-> END [error in retrieval]
                        -.-> format query
    format query        ---> generate
    generate            -x-> tool_caller -x-> check_tool_result
                            check_tool_result -x-> generate
                        -x-> END

    Args:
        settings: a ConfigSettings object (for settings.major,
            settings.minor, settings.aux)
        llm_major: an override of settings.major (used to inject
            a model for testing purposes)

    Returns:
        Compiled StateGraph ready for streaming
    """

    # Shared nodes from nodes.py
    integrate_history = create_integrate_history_node(settings)

    def format_query(state: ChatState) -> dict[str, str | AIMessage]:
        """Format the query with retrieved context using prompt
        template."""

        query: str = state["query"]

        # Format with prompt template
        template = PromptTemplate.from_template(
            """Please assist students by responding to their QUERY by 
using the context obtained by searching the database. If such context 
does not provide information for your answer, integrate the context 
only for the use and syntax of R. Otherwise, reply that you do not 
have information to answer the query, as the course focuses on linear 
models and their use in R.

####
QUERY: "{query}"
"""
        )
        formatted_query: str = template.format(
            query=query,
        )

        return {"refined_query": formatted_query}

    @tool(
        "search_database",
        description=(
            "Searches the vector database to retrieve context "
            "information to answer the user's question.\n"
            "Args:\n"
            "   query: str. A query text to be matched in the "
            "vector database."
        ),
    )
    async def retrieve_context(
        query: str,
        runtime: ToolRuntime[ChatWorkflowContext, ChatState],
    ) -> str:
        config: ChatWorkflowContext = runtime.context

        try:
            documents: list[Document] = (
                await config.retriever.ainvoke(query)
            )
            context: str = "\n-----\n".join(
                [d.page_content for d in documents]
            )

            return context

        except Exception as e:
            config.logger.error(
                f"Error retrieving from vector database:\n{e}"
            )
            raise RuntimeError(
                "Error retrieving from vector database"
            ) from e

    tools: list[BaseTool] = [retrieve_context]

    # the type of the model bound to tools is complex
    base_model: BaseChatModel = (
        llm_major or create_model_from_settings(settings.major)
    )
    model_with_tools: Runnable[
        Sequence[BaseMessage],
        AIMessage,
    ] = base_model.bind_tools(tools)

    def check_tool_result(
        state: ChatState, runtime: Runtime[ChatWorkflowContext]
    ) -> dict[str, str | AIMessage | float]:
        """Check if the tool execution resulted in an error.

        When handle_tool_errors=True, ToolNode catches exceptions
        and creates a ToolMessage with the error text in .content
        starting with 'Error:'.
        """
        from langchain_core.messages import ToolMessage

        messages = state.get("messages", [])
        if not messages:
            return {}

        last_message: BaseMessage = messages[-1]

        # Check if last message is a ToolMessage with error content
        if isinstance(last_message, ToolMessage):
            # ToolNode with handle_tool_errors=True creates error messages
            # with content starting with "Error:"
            if (
                last_message.content
                and last_message.content.startswith("Error")
            ):
                settings = runtime.context.chat_settings
                runtime.context.logger.error(
                    f"Tool execution failed: {last_message.content}"
                )
                return {
                    'status': "error",
                    'response': settings.MSG_ERROR_QUERY,
                    'messages': AIMessage(
                        content=settings.MSG_ERROR_QUERY
                    ),
                }

        latency: timedelta = datetime.now() - state['timestamp']
        return {
            'status': "valid",
            'context': str(last_message.content),
            'time_to_context': latency.total_seconds(),
        }  # Tool succeeded

    async def generate(
        state: ChatState, runtime: Runtime[ChatWorkflowContext]
    ) -> dict[str, str | AIMessage | float]:
        # Call the tool calling model

        system_message: str = (
            runtime.context.chat_settings.SYSTEM_MESSAGE
            + "\n"
            + "You use a searchable database to obtain the context "
            "on which you base your responses. To respond to users,"
            "base your actions on the following steps:\n"
            "1. determine if the QUERY of the user makes sense and is"
            "intelligible. If so, go to the next step. If not, respond "
            "immediately inviting the user to clarify their intended"
            " meaning.\n"
            "2. Search the vector database using the user's QUERY as "
            "argument to obtain material to answer it. If the QUERY "
            "includes different concepts or entities, you may "
            "optionally search the database by rewriting the query "
            "and splitting it into multiple queries, to be used in "
            "separate search database calls.\n"
            "3. Answer the query as specified in the user message."
        )
        messages = prepare_messages_for_llm(
            state, runtime.context, system_message=system_message
        )

        response_chunks: list[str] = []
        response_message = AIMessageChunk(
            content="", type="AIMessageChunk"
        )
        try:
            # While declared as AIMessage, astream may return
            # AIMessageChunk. At any rate, the .text member is
            # inherited from AIMessage, but __add__ gives
            # AIMessageChunk's, not ChatPromptTemplate objects.
            chunk: AIMessageChunk
            async for chunk in model_with_tools.astream(messages):  # type: ignore
                # Extract text from AIMessageChunk (from AIMessage).
                # https://docs.langchain.com/oss/python/langchain/messages#attributes
                if hasattr(chunk, "text") and chunk.text:
                    response_chunks.append(chunk.text)

                response_message = response_message + chunk

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

        latency: timedelta = datetime.now() - state['timestamp']
        if response_message.tool_calls:
            return {'messages': response_message}
        else:
            # Store complete response in state
            return {
                'model_identification': settings.major.model,
                'response': "".join(response_chunks),
                'time_to_response': latency.total_seconds(),
            }

    # Build the graph------------------------------------------------
    workflow: StateGraph[
        ChatState, ChatWorkflowContext, ChatState, ChatState
    ] = StateGraph(ChatState, ChatWorkflowContext)

    # Add nodes
    workflow.add_node("validate_query", validate_query)
    workflow.add_node("integrate_history", integrate_history)
    workflow.add_node("format_query", format_query)
    workflow.add_node(
        "tool_caller",
        ToolNode(tools, name="tool_caller", handle_tool_errors=True),
    )
    workflow.add_node("check_tool_result", check_tool_result)
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
        continue_if_no_error("format_query"),
    )
    workflow.add_edge("format_query", "generate")
    workflow.add_conditional_edges(
        "generate", continue_after_tool_call("tool_caller", END)
    )
    workflow.add_edge("tool_caller", "check_tool_result")
    workflow.add_conditional_edges(
        "check_tool_result", continue_if_no_error("generate")
    )

    return workflow.compile()
