"""Tests to inspect graph output"""

import unittest

from typing import Any

from lmm_education.config.config import (
    ConfigSettings,
    export_settings,
)
from lmm_education.config.appchat import ChatSettings, CheckResponse
from lmm_education.chat_graph import (
    ChatWorkflowContext,
    create_initial_state,
    create_chat_workflow,
)
import atexit

# pyright: reportUnknownMemberType=false

original_settings = ConfigSettings()
atexit.register(export_settings, original_settings)


def setUpModule():
    settings = ConfigSettings(
        major={'model': "Debug/debug"},
        minor={'model': "Debug/debug"},
        aux={'model': "Debug/debug"},
    )
    export_settings(settings)

    # An embedding engine object is created here just to load the engine.
    # This avoids the first query to take too long. The object is cached
    # internally, so we do not actually use the embedding object here.
    from lmm.language_models.langchain.runnables import (
        create_embeddings,
    )
    from requests import ConnectionError

    try:
        create_embeddings()
    except ConnectionError as e:
        print(
            "Could not connect to the model provider -- no internet?"
        )
        print(f"Error message:\n{e}")
        raise Exception from e
    except Exception as e:
        print(
            "Could not connect to the model provider due to a system error."
        )
        print(f"Error message:\n{e}")
        raise Exception from e


def tearDownModule():
    export_settings(original_settings)


class TestGraph(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        from langchain_core.language_models import BaseChatModel
        from langchain_core.retrievers import BaseRetriever
        from lmm.language_models.langchain.models import (
            create_model_from_settings,
        )
        from lmm_education.stores.langchain.vector_store_qdrant_langchain import (
            AsyncQdrantVectorStoreRetriever as AsyncQdrantRetriever,
        )

        self.llm: BaseChatModel = create_model_from_settings(
            ConfigSettings().major
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
            llm=self.llm,
            retriever=self.retriever,
            chat_settings=chat_settings,
        )

    async def test_invoke(self):

        context = self.get_workflow_context()
        initial_state = create_initial_state(
            "What is logistic regression?"
        )

        workflow = create_chat_workflow()

        end_state = await workflow.ainvoke(
            initial_state, context=context
        )

        print("===================================")
        print(
            f"There are {len(end_state["messages"])} messages in the end state\n"
        )
        messages = end_state["messages"]
        counter = 0
        for m in messages:
            counter += 1
            print(f"MESSAGE {counter}:")
            print(m)
            print("------\n")

        self.assertGreater(len(end_state["messages"]), 0)

    async def test_stream_messages(self):

        context = self.get_workflow_context()
        initial_state = create_initial_state(
            "What is logistic regression?"
        )

        workflow = create_chat_workflow()

        text: str = ""
        counter = 0
        async for chunk, meta in workflow.astream(
            initial_state, context=context, stream_mode="messages"
        ):
            counter += 1
            print(
                f"message {counter} from "
                f"{meta['langgraph_node']} node: "  # type: ignore
                f"{chunk.text}"  # type: ignore
            )
            text = text + str(chunk.text)  # type: ignore

        print("===================================")
        print(f"There were {counter} chunks:\n")
        print(text)

        self.assertGreater(len(text), 0)

    async def test_stream_state(self):

        context = self.get_workflow_context()
        initial_state = create_initial_state(
            "What is logistic regression?"
        )

        workflow = create_chat_workflow()

        end_state: Any = initial_state
        counter = 0
        async for event in workflow.astream(
            initial_state, context=context, stream_mode="values"
        ):
            counter += 1
            end_state = event  # type: ignore

        print("===================================")
        print(f"There were {counter} chunks:\n")
        print(
            f"There are {len(end_state["messages"])} messages in the end state\n"
        )
        messages = end_state["messages"]
        counter = 0
        for m in messages:
            counter += 1
            print(f"MESSAGE {counter}:")
            print(m)
            print("------\n")

        self.assertGreater(len(messages), 0)

    async def test_stream_updates(self):

        context = self.get_workflow_context()
        initial_state = create_initial_state(
            "What is logistic regression?"
        )

        workflow = create_chat_workflow()

        counter = 0
        async for event in workflow.astream(
            initial_state, context=context, stream_mode="updates"
        ):
            counter += 1
            print("======================")
            print(f"Chunk event {counter}:\n")
            for k in event.keys():
                print(f"{k}: {event[k]}\n")

        print("===================================")
        print(f"There were {counter} chunks")

        self.assertGreater(counter, 0)

    async def test_stream_multimodal(self):

        context = self.get_workflow_context()
        initial_state = create_initial_state(
            "What is logistic regression?"
        )

        workflow = create_chat_workflow()

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
                print(
                    f"message {counter_msg} from "
                    f"{meta['langgraph_node']} node: "  # type: ignore
                    f"{chunk.text}"  # type: ignore
                )
                text = text + str(chunk.text)  # type: ignore
            elif mode == "values":
                counter_val += 1
                end_state = event

        print("===================================")
        print(
            f"There were {counter} chunks ({counter_msg} "
            f"messages and {counter_val} values):\n"
        )
        print(text)
        print(
            f"\nThere are {len(end_state["messages"])} messages in the end state\n"
        )
        messages = end_state["messages"]
        counter = 0
        for m in messages:
            counter += 1
            print(f"MESSAGE {counter}:")
            print(m)
            print("------\n")

        self.assertGreater(len(messages), 0)


if __name__ == "__main__":
    unittest.main()
