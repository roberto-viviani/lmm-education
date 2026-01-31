"""
Default workflow for handling RAG interactions.

The workflow handles:
- Query validation
- Integration previous messages
- Context retrieval from vector store
- LLM response generation

The graph workflow is created with the `create_chat_workflow` function
taking a CongiSettings argument to specify major, minor, and aux
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

# LangGraph missing type stubs
# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownMemberType=false

from collections.abc import Callable
from math import ceil

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
)
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from langgraph.constants import TAG_NOSTREAM
from langgraph.types import RetryPolicy
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime

from lmm.language_models.langchain.models import (
    create_model_from_settings,
)
from lmm.language_models.langchain.runnables import (
    RunnableType,
    create_runnable,
)
from lmm.markdown.ioutils import convert_backslash_latex_delimiters

from lmm_education.config.config import ConfigSettings
from lmm_education.config.appchat import ChatSettings

from .base import (
    ChatState,
    ChatStateGraphType,
    ChatWorkflowContext,
    prepare_messages_for_llm,
)


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
            "model_identification": (
                model.get_name() if model else "<unknown>"
            ),
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

        llm: BaseChatModel = llm_major or create_model_from_settings(
            settings.major
        )

        # Stream the response - chunks will be emitted by
        # LangGraph's astream, even if not 'yielded'
        response_chunks: list[str] = []
        try:
            async for chunk in llm.astream(messages):
                # Extract text from AIMessageChunk
                # LLM.astream() returns AIMessageChunk objects. We
                # only stream text here. Please note that `content`
                # contains anything streamed by the model. See
                # https://docs.langchain.com/oss/python/langchain/messages#attributes
                if hasattr(chunk, "text"):
                    content: str = chunk.text
                    response_chunks.append(content)
                else:
                    # somewhat theoretical case, direct model stream
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

        return {
            "model_identification": (
                llm.get_name() if llm_major else settings.major.model
            ),
            "response": "".join(response_chunks),
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
        "retrieve_context", continue_if_no_error("format_query")
    )
    workflow.add_edge("format_query", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()
