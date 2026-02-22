"""
Default workflow for handling RAG interactions.

The workflow handles:
- Query validation
- Integration of previous messages
- Context retrieval from vector store
- LLM response generation

The graph workflow is created with the `create_chat_workflow` function
taking a ConfigSettings argument to specify major, minor, and aux
models:

```python
settings = ConfigSettings()
workflow = create_chat_workflow(settings)
state: ChatState = create_initial_state("What is logistic regression?")
dependencies = ChatWorkflowContext.from_default_config()
stream = workflow.astream(state, stream_mode="messages",
                context=dependencies)
```

After this, the stream may be consumed directly. Note that this is a
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

# LangGraph missing type stubs
# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownMemberType=false

from datetime import datetime, timedelta

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
)
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from langgraph.types import RetryPolicy
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime

from lmm.markdown.ioutils import convert_backslash_latex_delimiters

from lmm_education.config.config import ConfigSettings
from lmm_education.config.appchat import ChatSettings

from .base import (
    ChatState,
    ChatStateGraphType,
    ChatWorkflowContext,
)
from .nodes import (
    validate_query,
    create_integrate_history_node,
    create_generate_node,
)
from .graph_routing import continue_if_valid, continue_if_no_error


def create_chat_workflow(
    settings: ConfigSettings,
    llm_major: BaseChatModel | None = None,
) -> ChatStateGraphType:
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

    async def retrieve_context(
        state: ChatState, runtime: Runtime[ChatWorkflowContext]
    ) -> dict[str, str | AIMessage | float]:
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

            latency: timedelta = datetime.now() - state['timestamp']
            return {
                "context": context,
                "time_to_context": latency.total_seconds(),
            }

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

    # Extract the shared generate factory
    generate = create_generate_node(settings, llm_major)

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
        "retrieve_context", continue_if_no_error("format_query")
    )
    workflow.add_edge("format_query", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()
