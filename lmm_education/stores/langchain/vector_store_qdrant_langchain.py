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

from lmm.scan.scan_keys import GROUP_UUID_KEY
from lmm.utils.logging import ExceptionConsoleLogger
from lmm_education.config.config import ConfigSettings
from lmm_education.stores.vector_store_qdrant import (
    client_from_config,
    async_client_from_config,
    encoding_to_qdrantembedding_model,
)


# TODO: implement langchain approach to providing extra args
# to invoke()
class QdrantVectorStoreRetriever(BaseRetriever):
    """
    Langchain retriever interface to the Qdrant vector store.
    """

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

    @staticmethod
    def from_config_settings(
        opts: ConfigSettings | None = None,
    ) -> BaseRetriever:
        """
        Initializes a QdrantVectorStoreRetriever from a ConfigSettings
        object, or from the config.toml file.

        Args:
            opts: a ConfigSettings object, or none to read settings
                from the configutation file

        Returns:
            A QdrantVectorStoreRetriever object
        """

        if opts is None:
            opts = ConfigSettings()

        if bool(opts.companion_collection):
            return QdrantVectorStoreRetrieverGrouped.from_config_settings(
                opts
            )

        logger = ExceptionConsoleLogger()
        client: QdrantClient | None = client_from_config(
            opts=opts, logger=logger
        )
        if client is None:
            raise ValueError("Could not initialize client")
        return QdrantVectorStoreRetriever(
            client,
            opts.collection_name,
            encoding_to_qdrantembedding_model(opts.encoding_model),
        )

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

        logger = ExceptionConsoleLogger()
        points: list[vsq.ScoredPoint] = vsq.query(
            self.client,
            self.collection_name,
            self.embedding_model,
            query,
            limit=limit,
            payload=payload,
            logger=logger,
        )

        return self._points_to_documents(points)


class AsyncQdrantVectorStoreRetriever(BaseRetriever):
    """
    Langchain asynchronous retriever interface to the Qdrant
    vector store.
    """

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

    @staticmethod
    def from_config_settings(
        opts: ConfigSettings | None = None,
    ) -> BaseRetriever:
        """
        Initializes a ansynchronous QdrantVectorStoreRetriever from a
        ConfigSettings object, or from the config.toml file.

        Args:
            opts: a ConfigSettings object, or none to read settings
                from the configutation file

        Returns:
            An AsyncQdrantVectorStoreRetriever object
        """
        if opts is None:
            opts = ConfigSettings()

        if bool(opts.companion_collection):
            return AsyncQdrantVectorStoreRetrieverGrouped.from_config_settings(
                opts
            )

        logger = ExceptionConsoleLogger()
        client: AsyncQdrantClient | None = async_client_from_config(
            opts, logger
        )
        if client is None:
            raise ValueError("Could not initialize client")
        return AsyncQdrantVectorStoreRetriever(
            client,
            opts.collection_name,
            encoding_to_qdrantembedding_model(opts.encoding_model),
        )

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
                limit=limit,
                payload=payload,
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
            limit=limit,
            payload=payload,
        )

        return self._points_to_documents(points)


class QdrantVectorStoreRetrieverGrouped(BaseRetriever):
    """
    Langchain retriever interface to the Qdrant vector store
    for grouped queries.
    """

    client: QdrantClient = Field(
        default=QdrantClient(':memory:'), init=False
    )
    collection_name: str = Field(default="", init=False)
    group_collection: str = Field(default="", init=False)
    group_field: str = Field(default=GROUP_UUID_KEY, init=False)
    limitgroups: int = Field(default=4, init=False)
    embedding_model: vsq.QdrantEmbeddingModel = Field(
        default=vsq.QdrantEmbeddingModel.DENSE, init=False
    )

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        group_collection: str,
        group_field: str,
        limitgroups: int,
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
        self.group_collection = group_collection
        self.group_field = group_field
        self.limitgroups = limitgroups
        self.embedding_model = embedding_model

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @staticmethod
    def from_config_settings(
        opts: ConfigSettings | None = None,
    ) -> BaseRetriever:
        """
        Initializes a QdrantVectorStoreRetrieverGrouped from a
        ConfigSettings object, or from the config.toml file.

        Args:
            opts: a ConfigSettings object, or none to read settings
                from the configutation file

        Returns:
            A QdrantVectorStoreRetrieverGrouped object
        """
        from lmm.scan.scan_keys import GROUP_UUID_KEY

        if opts is None:
            opts = ConfigSettings()

        if not bool(opts.companion_collection):
            return QdrantVectorStoreRetriever.from_config_settings(
                opts
            )

        logger = ExceptionConsoleLogger()
        client: QdrantClient | None = client_from_config(
            opts=opts, logger=logger
        )
        if client is None:
            raise ValueError("Could not initialize client")
        return QdrantVectorStoreRetrieverGrouped(
            client,
            opts.collection_name,
            opts.companion_collection,
            GROUP_UUID_KEY,
            4,
            encoding_to_qdrantembedding_model(opts.encoding_model),
        )

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
        limit: int = 4,
        payload: list[str] = ['page_content'],
    ) -> list[Document]:

        results: vsq.GroupsResult = vsq.query_grouped(
            self.client,
            self.collection_name,
            self.group_collection,
            self.embedding_model,
            query,
            group_field=self.group_field,
            group_size=self.limitgroups,
            limit=limit,
            payload=payload,
        )

        return self._results_to_documents(results)


class AsyncQdrantVectorStoreRetrieverGrouped(BaseRetriever):
    """
    Langchain asynchronous retriever interface to the Qdrant vector
    store for grouped queries.
    """

    client: AsyncQdrantClient = Field(
        default=AsyncQdrantClient(':memory:'), init=False
    )
    collection_name: str = Field(default="", init=False)
    group_collection: str = Field(default="", init=False)
    group_field: str = Field(default=GROUP_UUID_KEY, init=False)
    limitgroups: int = Field(default=4, init=False)
    embedding_model: vsq.QdrantEmbeddingModel = Field(
        default=vsq.QdrantEmbeddingModel.DENSE, init=False
    )

    def __init__(
        self,
        client: AsyncQdrantClient,
        collection_name: str,
        group_collection: str,
        group_field: str,
        limitgroups: int,
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
        self.group_collection = group_collection
        self.group_field = group_field
        self.limitgroups = limitgroups
        self.embedding_model = embedding_model

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @staticmethod
    def from_config_settings(
        opts: ConfigSettings | None = None,
    ) -> BaseRetriever:
        """
        Initializes a ansynchronous QdrantVectorStoreRetrieverGrouped
        instance from a ConfigSettings object, or from the
        config.toml file.

        Args:
            opts: a ConfigSettings object, or none to read settings
                from the configutation file

        Returns:
            An AsyncQdrantVectorStoreRetrieverGrouped object
        """
        from lmm.scan.scan_keys import GROUP_UUID_KEY

        if opts is None:
            opts = ConfigSettings()

        if not bool(opts.companion_collection):
            return (
                AsyncQdrantVectorStoreRetriever.from_config_settings(
                    opts
                )
            )

        logger = ExceptionConsoleLogger()
        client: AsyncQdrantClient | None = async_client_from_config(
            opts, logger
        )
        if client is None:
            raise ValueError("Could not initialize client")
        return AsyncQdrantVectorStoreRetrieverGrouped(
            client,
            opts.collection_name,
            opts.companion_collection,
            GROUP_UUID_KEY,
            4,
            encoding_to_qdrantembedding_model(opts.encoding_model),
        )

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
        limit: int = 4,
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
                self.embedding_model,
                query,
                group_field=self.group_field,
                limit=limit,
                group_size=self.limitgroups,
                payload=payload,
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
        limit: int = 4,
        payload: list[str] = ['page_content'],
    ) -> list[Document]:

        results: vsq.GroupsResult = await vsq.aquery_grouped(
            self.client,
            self.collection_name,
            self.group_collection,
            self.embedding_model,
            query,
            limit=limit,
            group_field=self.group_field,
            group_size=self.limitgroups,
            payload=payload,
        )

        return self._results_to_documents(results)
