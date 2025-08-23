"""A langchain interface to a Qdrant retriever.
Note: only the query functions are supported."""

from typing import Coroutine, Any
from typing_extensions import override
import asyncio

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)

from qdrant_client import QdrantClient, AsyncQdrantClient
from .. import vector_store_qdrant as vsq

from pydantic import Field, ConfigDict


# TODO: implement langchain approach to providing extra args
# to invoke()
class QdrantVectorStoreRetriever(BaseRetriever):
    client: QdrantClient = Field(
        default=QdrantClient(':memory:'), init=False
    )
    collection_name: str = Field(default="", init=False)
    embedding_model: vsq.QdrantEmbeddingModel = Field(
        default=vsq.QdrantEmbeddingModel.DENSE, init=False
    )

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embedding_model: vsq.QdrantEmbeddingModel,
    ):
        flag: bool = vsq.initialize_collection(
            client, collection_name, embedding_model
        )
        if not flag:
            raise RuntimeError("Could not initialize the collection")

        super().__init__(
            metadata={'embedding_model': embedding_model.value}
        )

        # Now these can be set normally
        self.client = client
        self.collection_name = collection_name
        self.embedding_model = embedding_model

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _points_to_documents(
        self, points: list[vsq.ScoredPoint]
    ) -> list[Document]:
        docs: list[Document] = []
        for p in points:
            payload = p.payload if p.payload is not None else {}
            docs.append(
                Document(
                    page_content=payload.pop('page_content', ""),
                    metadata=payload.pop('metadata', {}),
                )
            )

        return docs

    @override
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        limit: int = 12,
        payload: list[str] = ['page_content'],
    ) -> list[Document]:

        points: list[vsq.ScoredPoint] = vsq.query(
            self.client,
            self.collection_name,
            self.embedding_model,
            query,
            limit,
            payload,
        )

        return self._points_to_documents(points)


class AsyncQdrantVectorStoreRetriever(BaseRetriever):
    client: AsyncQdrantClient = Field(
        default=AsyncQdrantClient(':memory:'), init=False
    )
    collection_name: str = Field(default="", init=False)
    embedding_model: vsq.QdrantEmbeddingModel = Field(
        default=vsq.QdrantEmbeddingModel.DENSE, init=False
    )

    def __init__(
        self,
        client: AsyncQdrantClient,
        collection_name: str,
        embedding_model: vsq.QdrantEmbeddingModel,
    ):
        # TODO: verify the collection asynchronously
        # flag: bool = vsq.initialize_collection(
        #     client, collection_name, embedding_model
        # )
        # if not flag:
        #   raise RuntimeError("Could not initialize the collection")

        super().__init__(
            metadata={'embedding_model': embedding_model.value}
        )

        # Now these can be set normally
        self.client = client
        self.collection_name = collection_name
        self.embedding_model = embedding_model

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _points_to_documents(
        self, points: list[vsq.ScoredPoint]
    ) -> list[Document]:
        docs: list[Document] = []
        for p in points:
            payload = p.payload if p.payload is not None else {}
            docs.append(
                Document(
                    page_content=payload.pop('page_content', ""),
                    metadata=payload.pop('metadata', {}),
                )
            )

        return docs

    @override
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        limit: int = 12,
        payload: list[str] = ['page_content'],
    ) -> list[Document]:
        # this is a required override, so we need to await the async
        import nest_asyncio  # type: ignore[stubFileNotFound]
        from lmm.utils.logging import ConsoleLogger

        logger = ConsoleLogger(__name__)
        logger.warning(
            "Sync function in vector store called in async"
            + " context. Are you using .invoke instad of "
            + ".ainvoke?"
        )
        nest_asyncio.apply()  # type: ignore

        cpoints: Coroutine[Any, Any, list[vsq.ScoredPoint]] = (
            vsq.aquery(
                self.client,
                self.collection_name,
                self.embedding_model,
                query,
                limit,
                payload,
            )
        )

        loop = asyncio.get_event_loop()
        task = loop.create_task(cpoints)
        points: list[vsq.ScoredPoint] = loop.run_until_complete(task)

        return self._points_to_documents(points)

    @override
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        limit: int = 12,
        payload: list[str] = ['page_content'],
    ) -> list[Document]:

        points: list[vsq.ScoredPoint] = await vsq.aquery(
            self.client,
            self.collection_name,
            self.embedding_model,
            query,
            limit,
            payload,
        )

        return self._points_to_documents(points)


class QdrantVectorStoreRetrieverGrouped(BaseRetriever):
    client: QdrantClient = Field(
        default=QdrantClient(':memory:'), init=False
    )
    collection_name: str = Field(default="", init=False)
    group_collection: str = Field(default="", init=False)
    group_field: str = Field(default="", init=False)
    limitgroups: int = Field(default=4, init=False)
    embedding_model: vsq.QdrantEmbeddingModel = Field(
        default=vsq.QdrantEmbeddingModel.DENSE, init=False
    )
    limit: int = Field(default=1, init=False)

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        group_collection: str,
        group_field: str,
        limitgroups: int,
        embedding_model: vsq.QdrantEmbeddingModel,
        limit: int,
    ):
        flag: bool = vsq.initialize_collection(
            client, collection_name, embedding_model
        )
        if not flag:
            raise RuntimeError("Could not initialize the collection")

        super().__init__(
            metadata={'embedding_model': embedding_model.value}
        )

        # Now these can be set normally
        self.client = client
        self.collection_name = collection_name
        self.group_collection = group_collection
        self.group_field = group_field
        self.limitgroups = limitgroups
        self.embedding_model = embedding_model
        self.limit = limit

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _results_to_documents(
        self, results: vsq.GroupsResult
    ) -> list[Document]:
        docs: list[Document] = []
        result_points: list[vsq.ScoredPoint] = vsq.groups_to_points(
            results
        )
        for p in result_points:
            payload = p.payload if p.payload is not None else {}
            docs.append(
                Document(
                    page_content=payload.pop('page_content', ""),
                    metadata=payload.pop('metadata', {}),
                )
            )

        return docs

    @override
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        payload: list[str] = ['page_content'],
    ) -> list[Document]:

        results: vsq.GroupsResult = vsq.query_grouped(
            self.client,
            self.collection_name,
            self.group_collection,
            self.group_field,
            self.limitgroups,
            self.embedding_model,
            query,
            self.limit,
            payload,
        )

        return self._results_to_documents(results)


class AsyncQdrantVectorStoreRetrieverGrouped(BaseRetriever):
    client: AsyncQdrantClient = Field(
        default=AsyncQdrantClient(':memory:'), init=False
    )
    collection_name: str = Field(default="", init=False)
    group_collection: str = Field(default="", init=False)
    group_field: str = Field(default="", init=False)
    limitgroups: int = Field(default=4, init=False)
    embedding_model: vsq.QdrantEmbeddingModel = Field(
        default=vsq.QdrantEmbeddingModel.DENSE, init=False
    )
    limit: int = Field(default=1, init=False)

    def __init__(
        self,
        client: AsyncQdrantClient,
        collection_name: str,
        group_collection: str,
        group_field: str,
        limitgroups: int,
        embedding_model: vsq.QdrantEmbeddingModel,
        limit: int,
    ):
        # TODO: verify the collection asynchronously
        # flag: bool = vsq.initialize_collection(
        #     client, collection_name, embedding_model
        # )
        # if not flag:
        #   raise RuntimeError("Could not initialize the collection")

        super().__init__(
            metadata={'embedding_model': embedding_model.value}
        )

        # Now these can be set normally
        self.client = client
        self.collection_name = collection_name
        self.group_collection = group_collection
        self.group_field = group_field
        self.limitgroups = limitgroups
        self.embedding_model = embedding_model
        self.limit = limit

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _results_to_documents(
        self, results: vsq.GroupsResult
    ) -> list[Document]:
        docs: list[Document] = []
        result_points: list[vsq.ScoredPoint] = vsq.groups_to_points(
            results
        )
        for p in result_points:
            payload = p.payload if p.payload is not None else {}
            docs.append(
                Document(
                    page_content=payload.pop('page_content', ""),
                    metadata=payload.pop('metadata', {}),
                )
            )

        return docs

    @override
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        payload: list[str] = ['page_content'],
    ) -> list[Document]:
        # this is a required override, so we need to await the async
        import nest_asyncio  # type: ignore[stubFileNotFound]
        from lmm.utils.logging import ConsoleLogger

        logger = ConsoleLogger(__name__)
        logger.warning(
            "Sync function in vector store called in async"
            + " context. Are you using .invoke instad of "
            + ".ainvoke?"
        )

        nest_asyncio.apply()  # type: ignore

        gresults: Coroutine[Any, Any, vsq.GroupsResult] = (
            vsq.aquery_grouped(
                self.client,
                self.collection_name,
                self.group_collection,
                self.group_field,
                self.limitgroups,
                self.embedding_model,
                query,
                self.limit,
                payload,
            )
        )

        loop = asyncio.get_event_loop()
        task = loop.create_task(gresults)
        results: vsq.GroupsResult = loop.run_until_complete(task)

        return self._results_to_documents(results)

    @override
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        payload: list[str] = ['page_content'],
    ) -> list[Document]:

        results: vsq.GroupsResult = await vsq.aquery_grouped(
            self.client,
            self.collection_name,
            self.group_collection,
            self.group_field,
            self.limitgroups,
            self.embedding_model,
            query,
            self.limit,
            payload,
        )

        return self._results_to_documents(results)
