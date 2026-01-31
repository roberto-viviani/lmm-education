import unittest
import os
from typing import Any

from lmm_education.config.config import ConfigSettings
from lmm_education.config.appchat import ChatSettings, CheckResponse

from lmm_education.workflows.langchain.base import (
    ChatWorkflowContext,
    ChatState,
    create_initial_state,
)

from lmm_education.workflows.langchain.chat_agent import (
    create_chat_agent,
)

# pyright: basic
# pyright: reportArgumentType=false

# Control whether to run tests that call real LLMs (which incur API costs)
# Set environment variable RUN_EXPENSIVE_TESTS=1 to enable these tests
# Example: RUN_EXPENSIVE_TESTS=1 poetry run pytest tests/test_chatagent.py
RUN_EXPENSIVE_TESTS = os.getenv("RUN_EXPENSIVE_TESTS", "0") == "1"

# Decorator to skip tests that call real LLMs unless explicitly enabled
skip_if_expensive = unittest.skipUnless(
    RUN_EXPENSIVE_TESTS,
    "Skipping expensive test that calls real LLM API. "
    "Set RUN_EXPENSIVE_TESTS=1 to run.",
)

print_messages: bool = False
print_response: bool = True


class TestGraph(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        from langchain_core.retrievers import BaseRetriever
        from lmm_education.stores.langchain.vector_store_qdrant_langchain import (
            AsyncQdrantVectorStoreRetriever as AsyncQdrantRetriever,
        )

        self.retriever: BaseRetriever = (
            AsyncQdrantRetriever.from_config_settings()
        )

    def get_workflow_context(
        self,
        chat_settings: ChatSettings = ChatSettings(
            check_response=CheckResponse(check_response=False)
        ),
    ) -> ChatWorkflowContext:
        return ChatWorkflowContext(
            retriever=self.retriever,
            chat_settings=chat_settings,
        )

    @skip_if_expensive
    async def test_invoke(self):

        context: ChatWorkflowContext = self.get_workflow_context()
        initial_state: ChatState = create_initial_state(
            "When should I log-transform the output variable of a regression?"
        )

        config: ConfigSettings = ConfigSettings()
        workflow = create_chat_agent(config)

        end_state: ChatState = await workflow.ainvoke(
            initial_state, context=context
        )  # type: ignore

        if print_messages:
            print("===================================")
            msgs = end_state["messages"]
            print(
                f"There are {len(msgs)} messages in the end state\n"
            )
            counter = 0
            for m in msgs:
                counter += 1
                print(f"MESSAGE {counter}:")
                print(m)
                print("------\n")

        if print_response:
            resp = end_state["response"]
            if resp:
                print(f"Response: {resp}")
            else:
                print("No response")

        self.assertGreater(len(end_state["response"]), 0)

    @skip_if_expensive
    async def test_invoke_with_garbage(self):

        context: ChatWorkflowContext = self.get_workflow_context()
        initial_state: ChatState = create_initial_state(
            "When yes I wes print"
        )

        config: ConfigSettings = ConfigSettings()
        workflow = create_chat_agent(config)

        end_state: ChatState = await workflow.ainvoke(
            initial_state, context=context
        )  # type: ignore

        if print_messages:
            print("===================================")
            msgs = end_state["messages"]
            print(
                f"There are {len(msgs)} messages in the end state\n"
            )
            counter = 0
            for m in msgs:
                counter += 1
                print(f"MESSAGE {counter}:")
                print(m)
                print("------\n")

        if print_response:
            resp = end_state["response"]
            if resp:
                print(f"Response: {resp}")
            else:
                print("No response")

        self.assertGreater(len(end_state["response"]), 0)

    @skip_if_expensive
    async def test_stream_messages(self):

        from lmm_education.workflows.langchain.stream_adapters import (
            tier_2_filter_messages_adapter,
        )

        context: ChatWorkflowContext = self.get_workflow_context()
        initial_state: ChatState = create_initial_state(
            "Can you help me fit a logistic regression?"
        )

        config: ConfigSettings = ConfigSettings()
        workflow = create_chat_agent(config)

        text: str = ""
        counter = 0
        # The tier_2_filter_messages_adapter gets rid of the tool output
        async for chunk, meta in tier_2_filter_messages_adapter(
            workflow.astream(
                initial_state, context=context, stream_mode="messages"
            ),
            "tool_caller",
        ):
            counter += 1
            if print_messages:
                print(
                    f"message {counter} from "
                    f"{meta['langgraph_node']} node: "  # type: ignore
                    f"{chunk.text}"  # type: ignore
                )
            text += chunk.text  # type: ignore

        if print_messages:
            print("===================================")
            print(f"There were {counter} chunks:\n")

        if print_response:
            print(text)

        self.assertGreater(len(text), 0)

    @skip_if_expensive
    async def test_stream_state(self):
        """Test streaming with stream_mode='values' to get complete state."""
        from typing import Any

        context: ChatWorkflowContext = self.get_workflow_context()
        initial_state: ChatState = create_initial_state(
            "What is logistic regression?"
        )

        config: ConfigSettings = ConfigSettings()
        workflow = create_chat_agent(config)

        end_state: Any = initial_state
        counter = 0
        async for event in workflow.astream(
            initial_state, context=context, stream_mode="values"
        ):
            counter += 1
            end_state = event  # type: ignore

        if print_messages:
            print("===================================")
            print(f"There were {counter} chunks:\n")
            msgs = end_state["messages"]
            print(
                f"There are {len(msgs)} messages in the end state\n"
            )
            resp = end_state["response"]
            if resp:
                print(f"Response: {resp}")
            else:
                print("No response")

        self.assertTrue(end_state['response'])

    @skip_if_expensive
    async def test_stream_updates(self):
        """Test streaming with stream_mode='updates' to track field changes."""
        context: ChatWorkflowContext = self.get_workflow_context()
        initial_state: ChatState = create_initial_state(
            "What is logistic regression?"
        )

        config: ConfigSettings = ConfigSettings()
        workflow = create_chat_agent(config)

        counter = 0
        async for event in workflow.astream(
            initial_state, context=context, stream_mode="updates"
        ):
            counter += 1
            if print_messages:
                print("======================")
                print(f"Chunk event {counter}:\n")
                for k in event.keys():
                    print(f"{k}: {event[k]}\n")

        if print_messages:
            print("===================================")
            print(f"There were {counter} chunks")

        self.assertGreater(counter, 0)

    @skip_if_expensive
    async def test_stream_multimodal(self):
        """Test streaming with multiple modes: messages + values."""
        from typing import Any

        context: ChatWorkflowContext = self.get_workflow_context()
        initial_state: ChatState = create_initial_state(
            "What is logistic regression?"
        )

        config: ConfigSettings = ConfigSettings()
        workflow = create_chat_agent(config)

        end_state: Any = initial_state
        text: str = ""
        counter = 0
        counter_msg = 0
        counter_val = 0

        async for mode, event in workflow.astream(
            initial_state,
            context=context,
            stream_mode=["messages", "values"],
        ):
            counter += 1
            if mode == "messages":
                counter_msg += 1
                chunk, meta = event
                # Filter out tool_caller messages
                if meta.get("langgraph_node") != "tool_caller":  # type: ignore
                    if print_messages:
                        print(
                            f"message {counter_msg} from "
                            f"{meta['langgraph_node']} node: "  # type: ignore
                            f"{chunk.text}"  # type: ignore
                        )
                    text = text + str(chunk.text)  # type: ignore
            elif mode == "values":
                counter_val += 1
                end_state = event

        if print_messages:
            print("===================================")
            print(
                f"There were {counter} chunks ({counter_msg} "
                f"messages and {counter_val} values):\n"
            )
            print(text)
            msgs = end_state["messages"]
            print("---------")
            print(
                f"\nThere are {len(msgs)} messages in the end state\n"
            )

        if print_messages and end_state['response']:
            resp = end_state['response']
            print(f"RESPONSE:\n{resp}")

        self.assertTrue(end_state['response'])


class TestIntegrateHistory(unittest.IsolatedAsyncioTestCase):
    """Test the integrate_history node with different history integration modes."""

    def setUp(self):
        from langchain_core.retrievers import BaseRetriever
        from lmm_education.stores.langchain.vector_store_qdrant_langchain import (
            AsyncQdrantVectorStoreRetriever as AsyncQdrantRetriever,
        )

        self.retriever: BaseRetriever = (
            AsyncQdrantRetriever.from_config_settings()
        )

    def get_workflow_context(
        self,
        chat_settings: ChatSettings = ChatSettings(
            check_response=CheckResponse(check_response=False)
        ),
    ) -> ChatWorkflowContext:
        return ChatWorkflowContext(
            retriever=self.retriever,
            chat_settings=chat_settings,
        )

    async def test_no_history(self):
        """Test integrate_history with no messages in state."""
        print("Test: integrate_history with no history")

        # Create state with no message history
        initial_state = create_initial_state(
            "What is a linear model?"
        )
        # Ensure messages is empty
        initial_state["messages"] = []

        from tests.test_mocks import MockLLM

        mock_llm = MockLLM(
            responses=["Something", "to fit", "to data"]
        )

        context = self.get_workflow_context()
        workflow = create_chat_agent(ConfigSettings(), mock_llm)

        # Run workflow
        end_state = await workflow.ainvoke(
            initial_state, context=context
        )

        # With no history, refined_query should equal the original query
        self.assertEqual(end_state["status"], "valid")
        self.assertEqual(
            end_state["query"], "What is a linear model?"
        )
        print("✓ Passed: No history handled correctly\n")

    @skip_if_expensive
    async def test_history_integration_none(self):
        """Test integrate_history with history_integration='none'."""
        from langchain_core.messages import HumanMessage, AIMessage

        print("Test: integrate_history mode='none'")

        # Create state with message history
        initial_state = create_initial_state(
            "Tell me more about logistic regression"
        )
        initial_state["messages"] = [
            HumanMessage(content="What is logistic regression?"),
            AIMessage(
                content="Logistic regression is a statistical method."
            ),
        ]

        # Configure chat settings with history_integration='none'
        chat_settings = ChatSettings(
            check_response=CheckResponse(check_response=False),
            history_integration='none',
        )
        context = self.get_workflow_context(chat_settings)
        workflow = create_chat_agent(ConfigSettings())

        # Run workflow
        end_state = await workflow.ainvoke(
            initial_state, context=context
        )

        # With 'none' mode, the query should pass through unchanged
        self.assertEqual(end_state["status"], "valid")
        self.assertEqual(
            end_state["query"],
            "Tell me more about logistic regression",
        )
        self.assertGreater(len(end_state["response"]), 0)
        print("✓ Passed: 'none' mode handled correctly\n")

    @skip_if_expensive
    async def test_history_integration_summary(self):
        """Test integrate_history with history_integration='summary'."""
        from unittest.mock import patch
        from langchain_core.messages import HumanMessage, AIMessage
        from tests.test_mocks import (
            MockSummarizerRunnable,
            create_mock_runnable_factory,
        )

        print("Test: integrate_history mode='summary'")

        # Create mock summarizer
        mock_summarizer = MockSummarizerRunnable(
            summary_prefix="Summary of chat:"
        )
        mocks = {"summarizer": mock_summarizer}
        factory = create_mock_runnable_factory(mocks)

        # Create state with message history
        initial_state = create_initial_state("Tell me more")
        initial_state["messages"] = [
            HumanMessage(content="What is logistic regression?"),
            AIMessage(
                content="Logistic regression is a statistical method for binary classification."
            ),
        ]

        # Configure chat settings with history_integration='summary'
        chat_settings = ChatSettings(
            check_response=CheckResponse(check_response=False),
            history_integration='summary',
        )
        context = self.get_workflow_context(chat_settings)
        workflow = create_chat_agent(ConfigSettings())

        # Patch create_runnable to return our mock
        with patch(
            "lmm_education.workflows.langchain.chat_agent.create_runnable",
            factory,
        ):
            # Run workflow
            end_state = await workflow.ainvoke(
                initial_state, context=context
            )

        # Verify the mock was called
        self.assertEqual(mock_summarizer.call_count, 1)
        self.assertIn("text", mock_summarizer.last_input)

        # Verify workflow completed successfully
        self.assertEqual(end_state["status"], "valid")
        self.assertGreater(len(end_state["response"]), 0)
        print("✓ Passed: 'summary' mode with mock summarizer\n")

    @skip_if_expensive
    async def test_history_integration_context_extraction(self):
        """Test integrate_history with history_integration='context_extraction'."""
        from unittest.mock import patch
        from langchain_core.messages import HumanMessage, AIMessage
        from tests.test_mocks import (
            MockChatSummarizerRunnable,
            create_mock_runnable_factory,
        )

        print("Test: integrate_history mode='context_extraction'")

        # Create mock chat summarizer
        mock_chat_summarizer = MockChatSummarizerRunnable()
        mocks = {"chat_summarizer": mock_chat_summarizer}
        factory = create_mock_runnable_factory(mocks)

        # Create state with message history
        initial_state = create_initial_state(
            "What about regularization?"
        )
        initial_state["messages"] = [
            HumanMessage(content="Explain linear regression"),
            AIMessage(
                content="Linear regression is a method for modeling relationships."
            ),
        ]

        # Configure chat settings
        chat_settings = ChatSettings(
            check_response=CheckResponse(check_response=False),
            history_integration='context_extraction',
        )
        context = self.get_workflow_context(chat_settings)
        workflow = create_chat_agent(ConfigSettings())

        # Patch create_runnable to return our mock
        with patch(
            "lmm_education.workflows.langchain.chat_agent.create_runnable",
            factory,
        ):
            # Run workflow
            end_state = await workflow.ainvoke(
                initial_state, context=context
            )

        # Verify the mock was called with both text and query
        self.assertEqual(mock_chat_summarizer.call_count, 1)
        self.assertIn("text", mock_chat_summarizer.last_input)
        self.assertIn("query", mock_chat_summarizer.last_input)

        # Verify workflow completed successfully
        self.assertEqual(end_state["status"], "valid")
        self.assertGreater(len(end_state["response"]), 0)
        print("✓ Passed: 'context_extraction' mode with mock\n")

    @skip_if_expensive
    async def test_history_integration_rewrite(self):
        """Test integrate_history with history_integration='rewrite'."""
        from unittest.mock import patch
        from langchain_core.messages import HumanMessage, AIMessage
        from tests.test_mocks import (
            MockRewriteQueryRunnable,
            create_mock_runnable_factory,
        )

        print("Test: integrate_history mode='rewrite'")

        # Create mock query rewriter
        mock_rewriter = MockRewriteQueryRunnable()
        mocks = {"rewrite_query": mock_rewriter}
        factory = create_mock_runnable_factory(mocks)

        # Create state with message history
        initial_state = create_initial_state("How does it work?")
        initial_state["messages"] = [
            HumanMessage(content="Tell me about neural networks"),
            AIMessage(
                content="Neural networks are computational models."
            ),
        ]

        # Configure chat settings
        chat_settings = ChatSettings(
            check_response=CheckResponse(check_response=False),
            history_integration='rewrite',
        )
        context = self.get_workflow_context(chat_settings)
        workflow = create_chat_agent(ConfigSettings())

        # Patch create_runnable to return our mock
        with patch(
            "lmm_education.workflows.langchain.chat_agent.create_runnable",
            factory,
        ):
            # Run workflow
            end_state = await workflow.ainvoke(
                initial_state, context=context
            )

        # Verify the mock was called
        self.assertEqual(mock_rewriter.call_count, 1)
        self.assertIn("text", mock_rewriter.last_input)
        self.assertIn("query", mock_rewriter.last_input)

        # Verify workflow completed successfully
        self.assertEqual(end_state["status"], "valid")
        self.assertGreater(len(end_state["response"]), 0)
        print("✓ Passed: 'rewrite' mode with mock\n")

    async def test_integration_error_handling(self):
        """Test that errors in integrate_history are caught and logged."""
        from unittest.mock import patch
        from langchain_core.messages import HumanMessage, AIMessage
        from tests.test_mocks import MockRunnable
        from lmm.utils.logging import LoglistLogger

        print("Test: integrate_history error handling")

        from tests.test_mocks import MockLLM

        mock_llm = MockLLM(
            responses=["Something", "to fit", "to data"]
        )

        # Create a mock that raises an exception
        error_message = "Simulated model failure"
        mock_failing = MockRunnable(
            exception=RuntimeError(error_message)
        )
        mocks = {"summarizer": mock_failing}

        def failing_factory(runnable_name: str, *args, **kwargs):  # type: ignore
            if runnable_name in mocks:
                return mocks[runnable_name]
            raise ValueError(f"No mock for {runnable_name}")

        # Create state with message history
        initial_state = create_initial_state("Tell me more")
        initial_state["messages"] = [
            HumanMessage(content="What is machine learning?"),
            AIMessage(content="Machine learning is a subset of AI."),
        ]

        # Use a logger that captures logs
        logger = LoglistLogger()

        # Configure chat settings with history_integration='summary'
        chat_settings = ChatSettings(
            check_response=CheckResponse(check_response=False),
            history_integration='summary',
        )
        context = self.get_workflow_context(chat_settings)
        context.logger = logger
        workflow = create_chat_agent(ConfigSettings(), mock_llm)

        # Patch create_runnable to return failing mock
        with patch(
            "lmm_education.workflows.langchain.chat_agent.create_runnable",
            failing_factory,  # type: ignore
        ):
            # Run workflow - should NOT fail, error should be caught
            end_state = await workflow.ainvoke(
                initial_state, context=context
            )

        # Verify error was logged
        logs = logger.get_logs()
        error_logged = any(
            "Error integrating history" in log for log in logs
        )
        self.assertTrue(
            error_logged,
            f"Expected error log not found. Logs: {logs}",
        )

        # Workflow should continue despite the error
        self.assertEqual(end_state["status"], "valid")
        print("✓ Passed: Error handling works correctly\n")


class TestGenerateNode(unittest.IsolatedAsyncioTestCase):
    """Test the generate node with different LLM behaviors."""

    def setUp(self):
        """Set up test fixtures."""
        from tests.test_mocks import MockRetriever

        self.retriever = MockRetriever()

    def get_workflow_context(
        self,
        system_message: str = "You are a helpful assistant",
    ) -> ChatWorkflowContext:
        """Helper to create workflow context."""
        chat_settings = ChatSettings(
            check_response=CheckResponse(check_response=False)
        )
        return ChatWorkflowContext(
            retriever=self.retriever,
            chat_settings=chat_settings,
            system_message=system_message,
        )

    async def test_llm_streaming_exception(self):
        """Test error handling when LLM raises exception during streaming."""
        from tests.test_mocks import MockLLM
        from lmm.utils.logging import LoglistLogger

        print("Test: generate node with LLM streaming exception")

        # Create mock LLM that raises exception
        streaming_error = RuntimeError(
            "LLM connection lost during streaming"
        )
        mock_llm = MockLLM(
            responses=["Partial ", "response..."],
            exception=streaming_error,
        )

        initial_state = create_initial_state("Test query")

        # Use logger that captures logs
        logger = LoglistLogger()
        context = self.get_workflow_context()
        context.logger = logger
        workflow = create_chat_agent(ConfigSettings(), mock_llm)

        # Run workflow
        end_state = await workflow.ainvoke(
            initial_state, context=context
        )

        # Verify error was logged
        logs = logger.get_logs()
        error_logged = any(
            "Error while streaming in 'generate' node" in log
            for log in logs
        )
        self.assertTrue(
            error_logged,
            f"Expected streaming error log not found. Logs: {logs}",
        )

        # Verify workflow set error status
        self.assertEqual(
            end_state["status"],
            "error",
            "Workflow should set status to 'error' when LLM streaming fails",
        )

        # Verify error message was returned
        self.assertEqual(
            end_state["response"],
            context.chat_settings.MSG_ERROR_QUERY,
        )

        print("✓ Passed: LLM streaming exception handled correctly\n")

    async def test_empty_llm_response(self):
        """Test handling of empty LLM response."""
        from tests.test_mocks import MockLLM

        print("Test: generate node with empty LLM response")

        # Create mock LLM that returns no chunks
        mock_llm = MockLLM(responses=[])

        initial_state = create_initial_state("Query")

        context = self.get_workflow_context()
        workflow = create_chat_agent(ConfigSettings(), mock_llm)

        end_state = await workflow.ainvoke(
            initial_state, context=context
        )

        # Should complete without error, but response is empty
        self.assertEqual(end_state["status"], "valid")
        self.assertEqual(end_state["response"], "")

        print("✓ Passed: Empty LLM response handled correctly\n")

    async def test_chunk_without_content_or_text(self):
        """Test handling of chunks with neither content nor text attributes.
        This violates LangGraph's API and is handled with error message.
        """
        from tests.test_mocks import MockLLM

        print("Test: generate node with non-standard chunk format")

        # Create a custom mock LLM with strange chunk format
        class StrangeLLM(MockLLM):
            def _create_chunk(self, content: str):
                # Return a chunk that only has __str__
                return content  # Just a string, no attributes

        mock_llm = StrangeLLM(
            responses=["String ", "chunks ", "only."]
        )

        initial_state = create_initial_state("Query")

        context = self.get_workflow_context()
        workflow = create_chat_agent(ConfigSettings(), mock_llm)

        end_state = await workflow.ainvoke(
            initial_state, context=context
        )

        # Should use str(chunk) fallback
        self.assertEqual(end_state["status"], "error")
        self.assertEqual(
            end_state["response"], ChatSettings().MSG_ERROR_QUERY
        )

        print(
            "✓ Passed: Non-standard chunk format handled correctly\n"
        )


class TestToolBehavior(unittest.IsolatedAsyncioTestCase):
    """Test the search_database tool behavior and error handling."""

    def get_workflow_context(
        self,
        retriever: Any,
        system_message: str = "You are a helpful assistant",
    ) -> ChatWorkflowContext:
        """Helper to create workflow context with custom retriever."""
        chat_settings = ChatSettings(
            check_response=CheckResponse(check_response=False)
        )
        return ChatWorkflowContext(
            retriever=retriever,
            chat_settings=chat_settings,
            system_message=system_message,
        )

    @skip_if_expensive
    async def test_tool_successful_retrieval(self):
        """Test that the agent successfully calls the retrieval tool and uses results."""
        from tests.test_mocks import MockRetriever
        from langchain_core.documents import Document

        print("Test: Agent successfully retrieves context via tool")

        # Create mock retriever with specific documents
        mock_docs = [
            Document(
                page_content="Logistic regression is used for binary classification.",
                metadata={"source": "doc1"},
            ),
            Document(
                page_content="It uses a sigmoid activation function.",
                metadata={"source": "doc2"},
            ),
        ]
        mock_retriever = MockRetriever(documents=mock_docs)

        # Create initial state with a query that should trigger search
        initial_state = create_initial_state(
            "What is logistic regression used for?"
        )

        context = self.get_workflow_context(retriever=mock_retriever)
        workflow = create_chat_agent(ConfigSettings())

        # Run workflow
        end_state = await workflow.ainvoke(
            initial_state, context=context
        )

        # Verify retriever was called (implicitly via tool)
        self.assertEqual(
            mock_retriever.call_count,
            1,
            "Retriever should be called once by the agent's tool",
        )
        self.assertGreater(
            len(mock_retriever.last_query),
            0,
            "Query should be passed to retriever",
        )

        # Verify workflow completed successfully
        self.assertEqual(end_state["status"], "valid")
        self.assertGreater(
            len(end_state["response"]),
            0,
            "Agent should generate a response using retrieved context",
        )

        print(
            "✓ Passed: Tool successfully retrieved and used context\n"
        )

    @skip_if_expensive
    async def test_tool_retriever_exception(self):
        """Test that retriever exceptions are caught by handle_tool_errors=True."""
        from tests.test_mocks import MockRetriever
        from lmm.utils.logging import LoglistLogger
        from langchain_core.messages import ToolMessage

        print("Test: Tool errors are handled gracefully")

        # Create mock retriever that raises exception
        retrieval_error = RuntimeError("Database connection failed")
        mock_retriever = MockRetriever(exception=retrieval_error)

        # Create initial state
        initial_state = create_initial_state(
            "What is linear regression?"
        )

        # Use logger that captures logs
        logger = LoglistLogger()
        context = self.get_workflow_context(retriever=mock_retriever)
        context.logger = logger
        workflow = create_chat_agent(ConfigSettings())

        # Run workflow - with handle_tool_errors=True, exceptions are caught
        end_state = await workflow.ainvoke(
            initial_state, context=context
        )

        # Verify retriever was called
        self.assertEqual(
            mock_retriever.call_count,
            1,
            "Retriever should be called by the tool",
        )

        # Verify ToolMessage with error was created in messages
        messages = end_state.get("messages", [])
        tool_messages = [
            msg for msg in messages if isinstance(msg, ToolMessage)
        ]
        self.assertGreater(
            len(tool_messages),
            0,
            "Should have ToolMessage in messages",
        )

        # Verify at least one ToolMessage has error content
        has_error_content = any(
            msg.content[-1] and msg.content.startswith("Error")  # type: ignore
            for msg in tool_messages
        )
        self.assertTrue(
            has_error_content,
            "Should have ToolMessage with error content starting with 'Error'",
        )

        # Verify check_tool_result caught the error and set error status
        self.assertEqual(
            end_state["status"],
            "error",
            "Workflow should set status to 'error' when tool fails",
        )

        # Verify error message was returned to user
        self.assertEqual(
            end_state["response"],
            context.chat_settings.MSG_ERROR_QUERY,
            "Should return generic error message to user",
        )

        # Verify error was logged by check_tool_result
        logs = logger.get_logs()
        error_logged = any(
            "Tool execution failed" in log for log in logs
        )
        self.assertTrue(
            error_logged,
            f"Expected check_tool_result error log. Logs: {logs}",
        )

        print("✓ Passed: Tool exception handled correctly\n")

    @skip_if_expensive
    async def test_workflow_stops_on_retrieval_error(self):
        """Test that workflow stops at check_tool_result when tool errors."""
        from tests.test_mocks import MockRetriever

        print("Test: Workflow stops after retrieval error")

        # Create failing retriever
        mock_retriever = MockRetriever(
            exception=RuntimeError("Retrieval failed")
        )

        initial_state = create_initial_state("Test query")

        context = self.get_workflow_context(retriever=mock_retriever)
        workflow = create_chat_agent(ConfigSettings())

        # Run workflow - should complete (not raise) with error status
        end_state = await workflow.ainvoke(
            initial_state, context=context
        )

        # Verify error status set by check_tool_result
        self.assertEqual(
            end_state["status"],
            "error",
            "check_tool_result should set error status",
        )

        # Verify retriever was called
        self.assertEqual(mock_retriever.call_count, 1)

        # Verify error response
        self.assertEqual(
            end_state["response"],
            context.chat_settings.MSG_ERROR_QUERY,
        )

        print(
            "✓ Passed: Workflow correctly stopped on retrieval error\n"
        )

    @skip_if_expensive
    async def test_tool_error_creates_tool_message(self):
        """Test that handle_tool_errors=True creates ToolMessage with error."""
        from tests.test_mocks import MockRetriever
        from langchain_core.messages import ToolMessage

        print(
            "Test: Tool errors create ToolMessage with error content"
        )

        # Create failing retriever
        mock_retriever = MockRetriever(
            exception=RuntimeError("Tool execution failed")
        )

        initial_state = create_initial_state(
            "What is machine learning?"
        )

        context = self.get_workflow_context(retriever=mock_retriever)
        workflow = create_chat_agent(ConfigSettings())

        # Run workflow
        end_state = await workflow.ainvoke(
            initial_state, context=context
        )

        # Verify the state contains a ToolMessage with error
        messages = end_state.get("messages", [])
        tool_messages = [
            msg for msg in messages if isinstance(msg, ToolMessage)
        ]

        self.assertGreater(
            len(tool_messages),
            0,
            "Should have at least one ToolMessage in final state",
        )

        # Check that at least one ToolMessage has error content
        error_tool_msg = None
        for msg in tool_messages:
            if msg.content[-1] and msg.content.startswith("Error"):  # type: ignore
                error_tool_msg = msg
                break

        self.assertIsNotNone(
            error_tool_msg,
            "Should have a ToolMessage with content starting with 'Error'",
        )

        # Verify the error content contains useful information
        self.assertIn(
            "Error",
            error_tool_msg.content,  # type: ignore (checked above)
            "ToolMessage should contain error information",
        )

        # Verify workflow ended with error status
        self.assertEqual(
            end_state["status"],
            "error",
            "check_tool_result should set error status",
        )

        print("✓ Passed: Tool error creates ToolMessage correctly\n")

    @skip_if_expensive
    async def test_tool_error_propagation(self):
        """Test that tool errors propagate as exceptions."""
        from tests.test_mocks import MockRetriever

        print("Test: Tool errors propagate as exceptions")

        # Create failing retriever
        mock_retriever = MockRetriever(
            exception=RuntimeError("Tool execution failed")
        )

        initial_state = create_initial_state(
            "What is machine learning?"
        )

        context = self.get_workflow_context(retriever=mock_retriever)
        workflow = create_chat_agent(ConfigSettings())

        # Run workflow - expect RuntimeError to be raised
        with self.assertRaises(RuntimeError) as cm:
            await workflow.ainvoke(initial_state, context=context)

        # Verify the exception message
        self.assertIn(
            "Error retrieving from vector database",
            str(cm.exception),
            "Should raise the retrieval error",
        )

        # Verify retriever was called
        self.assertEqual(
            mock_retriever.call_count,
            1,
            "Retriever should be called once",
        )

        print("✓ Passed: Tool error routing works correctly\n")


class TestAgentBehavior(unittest.IsolatedAsyncioTestCase):
    """Test agent-specific decision-making and complex behaviors."""

    def get_workflow_context(
        self,
        retriever: Any = None,
        system_message: str = "You are a helpful assistant",
    ) -> ChatWorkflowContext:
        """Helper to create workflow context."""
        from tests.test_mocks import MockRetriever
        from langchain_core.documents import Document

        chat_settings = ChatSettings(
            check_response=CheckResponse(check_response=False)
        )

        # Use provided retriever or create default mock
        if retriever is None:
            retriever = MockRetriever(
                documents=[
                    Document(
                        page_content="Default content",
                        metadata={"source": "default"},
                    )
                ]
            )

        return ChatWorkflowContext(
            retriever=retriever,
            chat_settings=chat_settings,
            system_message=system_message,
        )

    @skip_if_expensive
    async def test_agent_decides_not_to_search(self):
        """Test that agent can answer directly without using search tool."""
        from tests.test_mocks import MockRetriever

        print("Test: Agent decides not to use search tool")
        # Create retriever to track if it's called
        mock_retriever = MockRetriever(documents=[])
        # Ask a question that doesn't require database search
        initial_state = create_initial_state("What is 2 + 2?")
        context = self.get_workflow_context(retriever=mock_retriever)
        workflow = create_chat_agent(ConfigSettings())
        # Run workflow
        end_state = await workflow.ainvoke(
            initial_state, context=context
        )
        # Verify retriever was NOT called (agent didn't use tool)
        self.assertEqual(
            mock_retriever.call_count,
            0,
            "Agent should not call retriever for simple math question",
        )
        # Verify workflow completed successfully
        self.assertEqual(end_state["status"], "valid")
        self.assertGreater(
            len(end_state["response"]),
            0,
            "Agent should generate a direct response",
        )
        print(
            "✓ Passed: Agent correctly avoided unnecessary tool use\n"
        )

    @skip_if_expensive
    async def test_agent_multiple_searches(self):
        """Test that agent can make multiple tool calls if needed."""
        from tests.test_mocks import MockRetriever
        from langchain_core.documents import Document

        print("Test: Agent makes multiple tool calls")
        # Create retriever that tracks calls
        mock_retriever = MockRetriever(
            documents=[
                Document(
                    page_content="Logistic regression is a classification algorithm.",
                    metadata={"source": "doc1"},
                ),
            ]
        )
        # Ask a complex question that might trigger multiple searches
        initial_state = create_initial_state(
            "What is the difference between logistic regression and linear regression?"
        )
        context = self.get_workflow_context(retriever=mock_retriever)
        workflow = create_chat_agent(ConfigSettings())
        # Run workflow
        end_state = await workflow.ainvoke(
            initial_state, context=context
        )
        # Verify retriever was called at least once (agent used tool)
        self.assertGreaterEqual(
            mock_retriever.call_count,
            1,
            "Agent should call retriever at least once",
        )
        # Verify workflow completed successfully
        self.assertEqual(end_state["status"], "valid")
        self.assertGreater(
            len(end_state["response"]),
            0,
            "Agent should generate a response",
        )
        # Check that messages contain tool interactions
        from langchain_core.messages import ToolMessage

        messages = end_state.get("messages", [])
        tool_messages = [
            msg for msg in messages if isinstance(msg, ToolMessage)
        ]

        self.assertGreater(
            len(tool_messages),
            0,
            "Should have ToolMessages in conversation",
        )
        print(
            f"✓ Passed: Agent made {mock_retriever.call_count} tool call(s)\n"
        )

    async def test_generate_error_routing(self):
        """Test error handling when generate node fails."""
        from tests.test_mocks import MockLLM
        from lmm.utils.logging import LoglistLogger

        print("Test: Generate node error handling")
        # Create LLM that raises exception during streaming
        mock_llm = MockLLM(exception=RuntimeError("LLM API error"))
        initial_state = create_initial_state("Test query")
        # Use logger that captures logs
        logger = LoglistLogger()
        context = self.get_workflow_context()
        context.logger = logger
        workflow = create_chat_agent(
            ConfigSettings(), llm_major=mock_llm
        )
        # Run workflow - expect it to handle generate node errors
        try:
            end_state = await workflow.ainvoke(
                initial_state, context=context
            )

            # If workflow completed, verify error status
            self.assertEqual(
                end_state.get("status"),
                "error",
                "Should set error status when generate fails",
            )

            # Verify error was logged
            logs = logger.get_logs()
            error_logged = any(
                "Error while streaming" in log
                or "LLM API error" in log
                for log in logs
            )
            self.assertTrue(
                error_logged,
                f"Expected error log from generate node. Logs: {logs}",
            )

        except Exception as e:
            # If exception propagated, verify it's from the generate node
            self.assertIn(
                "LLM API error",
                str(e),
                "Exception should be from LLM failure",
            )

            # Verify error was logged before exception
            logs = logger.get_logs()
            error_logged = any(
                "Error while streaming" in log for log in logs
            )
            self.assertTrue(
                error_logged,
                f"Expected error log before exception. Logs: {logs}",
            )
        print("✓ Passed: Generate node error handled correctly\n")


if __name__ == "__main__":
    unittest.main()
