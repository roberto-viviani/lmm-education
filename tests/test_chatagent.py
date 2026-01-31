import unittest

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

print_messages: bool = True
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


if __name__ == "__main__":
    unittest.main()
