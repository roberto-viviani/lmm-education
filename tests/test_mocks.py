"""
Mock helpers for testing chat_graph.py workflow nodes.

This module provides mock implementations of LangChain models and
runnables used in testing, particularly for testing the integrate_history
node which uses various auxiliary models for history integration.
"""

# pyright: basic

from typing import Any
from collections.abc import AsyncIterator

from pydantic import ConfigDict

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage


class MockRunnable:
    """
    Mock implementation of a LangChain Runnable for testing.

    This mock can be configured to return specific responses or raise
    exceptions, allowing comprehensive testing of error handling paths.
    """

    def __init__(
        self,
        response: str | None = None,
        exception: Exception | None = None,
    ):
        """
        Initialize the mock runnable.

        Args:
            response: String to return from ainvoke/astream calls
            exception: Exception to raise instead of returning response
        """
        self.response = response or "Mock response"
        self.exception = exception
        self.call_count = 0
        self.last_input: dict[str, Any] = {}

    async def ainvoke(
        self,
        input_dict: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> str:
        """Mock ainvoke implementation."""
        self.call_count += 1
        self.last_input = input_dict

        if self.exception:
            raise self.exception

        return self.response

    async def astream(
        self,
        input_dict: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """Mock astream implementation."""
        self.call_count += 1
        self.last_input = input_dict

        if self.exception:
            raise self.exception

        # Yield response in chunks to simulate streaming
        chunk_size = 10
        for i in range(0, len(self.response), chunk_size):
            yield self.response[i : i + chunk_size]

    def get_name(self) -> str:
        return "mock model"


class MockSummarizerRunnable(MockRunnable):
    """Mock summarizer model that returns a summary of text."""

    def __init__(self, summary_prefix: str = "Summary:"):
        super().__init__()
        self.summary_prefix = summary_prefix

    async def ainvoke(
        self,
        input_dict: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> str:
        """Return a mock summary of the input text."""
        self.call_count += 1
        self.last_input = input_dict

        text = input_dict.get("text", "")
        # Simulate summarization by taking first few words
        words = text.split()[:5]
        return f"{self.summary_prefix} {' '.join(words)}..."


class MockChatSummarizerRunnable(MockRunnable):
    """Mock chat summarizer that extracts context from chat history."""

    async def ainvoke(
        self,
        input_dict: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> str:
        """Return a mock context extraction."""
        self.call_count += 1
        self.last_input = input_dict

        text = input_dict.get("text", "")
        query = input_dict.get("query", "")
        # Simulate context extraction
        return f"Context about '{query}' from chat: {text[:50]}..."


class MockRewriteQueryRunnable(MockRunnable):
    """Mock query rewriter that rewrites queries based on history."""

    async def ainvoke(
        self,
        input_dict: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> str:
        """Return a rewritten query."""
        self.call_count += 1
        self.last_input = input_dict

        query = input_dict.get("query", "")
        # Simulate query rewriting
        return f"Rewritten: {query}"


class MockRetriever(BaseRetriever):
    """
    Mock implementation of a LangChain BaseRetriever for testing.

    This mock can return documents or raise exceptions to test
    error handling in the retrieve_context node.
    """

    # Pydantic model configuration to allow arbitrary attributes
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Declare fields that are not part of BaseRetriever
    documents: list[Document] = []
    exception: Exception | None = None
    call_count: int = 0
    last_query: str = ""

    def __init__(
        self,
        documents: list[Document] | None = None,
        exception: Exception | None = None,
        **kwargs,
    ):
        """
        Initialize the mock retriever.

        Args:
            documents: List of Document objects to return from _aget_relevant_documents
            exception: Exception to raise instead of returning documents
        """
        # Initialize BaseRetriever
        super().__init__(**kwargs)

        if documents is None:
            # Create default mock document
            documents = [
                Document(
                    page_content="Mock document content about linear models.",
                    metadata={"source": "mock_doc_1"},
                )
            ]

        # Set attributes directly on the instance
        object.__setattr__(self, 'documents', documents)
        object.__setattr__(self, 'exception', exception)
        object.__setattr__(self, 'call_count', 0)
        object.__setattr__(self, 'last_query', "")

    def _get_relevant_documents(self, query: str) -> list[Document]:
        """Sync implementation (required by BaseRetriever)."""
        raise NotImplementedError("Use ainvoke for async operations")

    async def _aget_relevant_documents(
        self, query: str
    ) -> list[Document]:
        """Async implementation."""
        object.__setattr__(self, 'call_count', self.call_count + 1)
        object.__setattr__(self, 'last_query', query)

        if self.exception:
            raise self.exception

        return self.documents


class MockLLM(BaseChatModel):
    """
    Mock LLM for testing the generate node.

    This mock can stream responses or raise exceptions to test
    error handling in the generate node.
    """

    # Pydantic model configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Declare fields
    responses: list[str] = []
    exception: Exception | None = None
    chunk_content_attr: str = "content"
    call_count: int = 0
    last_messages: Any = None

    def __init__(
        self,
        responses: list[str] | None = None,
        exception: Exception | None = None,
        chunk_content_attr: str = "content",
        **kwargs,
    ):
        """
        Initialize the mock LLM.

        Args:
            responses: List of response strings to stream as chunks
            exception: Exception to raise during streaming
            chunk_content_attr: Attribute name for content ("content" or "text")
        """
        super().__init__(**kwargs)

        if responses is None:
            responses = ["This ", "is ", "a ", "mock ", "response."]

        object.__setattr__(self, 'responses', responses)
        object.__setattr__(self, 'exception', exception)
        object.__setattr__(
            self, 'chunk_content_attr', chunk_content_attr
        )
        object.__setattr__(self, 'call_count', 0)
        object.__setattr__(self, 'last_messages', None)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Sync generation (not used in our tests)."""
        raise NotImplementedError("Use astream for async operations")

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Async generation (not used - we use astream)."""
        raise NotImplementedError("Use astream for streaming")

    async def astream(
        self, messages: list[tuple[str, str]]
    ) -> AsyncIterator[Any]:
        """Mock astream implementation that yields response chunks."""
        object.__setattr__(self, 'call_count', self.call_count + 1)
        object.__setattr__(self, 'last_messages', messages)

        if self.exception:
            # Optionally yield some chunks before raising
            for i, response in enumerate(
                self.responses[:2]
            ):  # Yield first 2
                yield self._create_chunk(response)
            raise self.exception

        # Yield all response chunks
        for response in self.responses:
            yield self._create_chunk(response)

    def _create_chunk(self, content: str) -> Any:
        """Create a mock chunk with the specified content attribute."""
        from unittest.mock import Mock
        from langchain_core.messages import AIMessageChunk

        chunk = Mock(AIMessageChunk)
        object.__setattr__(chunk, 'additional_kwargs', {})
        object.__setattr__(chunk, 'response_metadata', {})
        object.__setattr__(chunk, 'tool_call_chunks', [])
        object.__setattr__(chunk, 'usage_metadata', {})
        object.__setattr__(chunk, 'id', 0)
        object.__setattr__(chunk, 'chunk_position', 0)

        if self.chunk_content_attr == "content":
            chunk.content = content
            if hasattr(chunk, "text"):
                delattr(chunk, "text")  # Ensure text doesn't exist
        elif self.chunk_content_attr == "text":
            # The generate node checks callable(chunk.text) and then appends chunk.text
            # NOT chunk.text(), so we need to return the value directly
            # but still make it callable to pass the check
            chunk.text = content  # Direct value, not lambda
            if hasattr(chunk, "content"):
                delattr(
                    chunk, "content"
                )  # Ensure content doesn't exist

        return chunk

    def bind_tools(self, tools: list[str]) -> 'MockLLM':
        return self

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "mock_llm"


def create_mock_runnable_factory(
    mock_instances: dict[str, MockRunnable],
):
    """
    Create a factory function that returns mock runnables.

    This can be used to patch the create_runnable function from
    lmm.language_models.langchain.runnables.

    Args:
        mock_instances: Dict mapping runnable names to mock instances

    Returns:
        Factory function compatible with create_runnable signature

    Example:
        ```python
        from unittest.mock import patch

        mocks = {
            "summarizer": MockSummarizerRunnable(),
            "chat_summarizer": MockChatSummarizerRunnable(),
            "rewrite_query": MockRewriteQueryRunnable(),
        }
        factory = create_mock_runnable_factory(mocks)

        with patch("path.to.create_runnable", factory):
            # Your test code here
        ```
    """

    def factory(
        runnable_name: str,
        user_settings: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> MockRunnable:
        """Factory function that returns configured mocks."""
        if runnable_name not in mock_instances:
            raise ValueError(
                f"No mock configured for runnable: {runnable_name}"
            )
        return mock_instances[runnable_name]

    return factory
