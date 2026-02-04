"""
A langchain interface to Qdrant vector store retrievers.

This module provides multiple retriever implementations for querying
Qdrant vector stores through the Langchain framework:

- QdrantVectorStoreRetriever: Synchronous retriever for basic queries
- AsyncQdrantVectorStoreRetriever: Asynchronous retriever for basic
    queries
- QdrantVectorStoreRetrieverGrouped: Synchronous retriever for grouped
    queries
- AsyncQdrantVectorStoreRetrieverGrouped: Asynchronous retriever for
    grouped queries

Note: only the query functions are supported (no document insertion).

Examples:

    Basic usage with configuration file:

    ```python
    retriever = QdrantVectorStoreRetriever.from_config_settings()
    results: list[Document] = retriever.invoke(
        "What are the main uses of logistic regression?"
    )
    ```

    Manual initialization with all required parameters:

    ```python
    from qdrant_client import QdrantClient
    from lmm_education.stores.vector_store_qdrant import QdrantEmbeddingModel
    from lmm_education.config.config import ConfigSettings

    client = QdrantClient("./storage")
    config = ConfigSettings()

    retriever = QdrantVectorStoreRetriever(
        client,
        collection_name="documents",
        qdrant_embedding=QdrantEmbeddingModel.DENSE,
        embedding_settings=config.embeddings,
    )
    results: list[Document] = retriever.invoke(
        "What are the main uses of logistic regression?"
    )
    ```

    Using the asynchronous retriever:

    ```python
    from qdrant_client import AsyncQdrantClient

    async_retriever = AsyncQdrantVectorStoreRetriever.from_config_settings()
    results: list[Document] = await async_retriever.ainvoke(
        "What are the main uses of logistic regression?"
    )
    ```

    Using grouped queries for document organization:

    ```python
    grouped_retriever = QdrantVectorStoreRetrieverGrouped.from_config_settings()
    results: list[Document] = grouped_retriever.invoke(
        "What are the main uses of logistic regression?"
    )
    ```
"""

from typing import Any
from collections.abc import Coroutine
from typing_extensions import override
import asyncio

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)

from qdrant_client import QdrantClient, AsyncQdrantClient

from lmm_education.config.config import DatabaseSettings
from .. import vector_store_qdrant as vsq

from pydantic import Field, ConfigDict

from lmm.scan.scan_keys import GROUP_UUID_KEY
from lmm.utils.logging import ExceptionConsoleLogger
from lmm.config.config import EmbeddingSettings
from ...config.config import ConfigSettings, load_settings
from ..vector_store_qdrant import (
    encoding_to_qdrantembedding_model,
    QdrantEmbeddingModel,
)
from ..vector_store_qdrant_context import (
    global_async_client_from_config,
    global_client_from_config,
)


# TODO: implement langchain approach to providing extra args
# to invoke()
class QdrantVectorStoreRetriever(BaseRetriever):
    """
    Langchain retriever interface to the Qdrant vector store.
    """

    # Implementation note: the default QdrantClient is needed
    # by the Pydantic constructor, and is closed immediately.
    # The real client object is passed by the constructor.
    # Do not close it in a destructor, as it is passed in.

    client: QdrantClient = Field(
        # Pydantic's BaseModel requires an object here
        default=QdrantClient(":memory:"),
        init=False,
    )
    collection_name: str = Field(default="", init=False)
    qdrant_embedding: QdrantEmbeddingModel = Field(
        default=encoding_to_qdrantembedding_model(
            ConfigSettings().RAG.encoding_model
        ),
        init=False,
    )
    embedding_settings: EmbeddingSettings = Field(
        default=ConfigSettings().embeddings, init=False
    )

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        qdrant_embedding: QdrantEmbeddingModel,
        embedding_settings: EmbeddingSettings,
    ):
        flag: bool = vsq.initialize_collection(
            client,
            collection_name,
            qdrant_embedding,
            embedding_settings,
        )
        if not flag:
            raise RuntimeError("Could not initialize the collection")

        super().__init__(
            metadata={'embedding_model': qdrant_embedding.value}
        )

        # Now these can be set normally, except that self.client
        # may have already been initialized
        try:
            self.client.close()
        except Exception:
            pass
        self.client = client
        self.collection_name = collection_name
        self.qdrant_embedding = qdrant_embedding
        self.embedding_settings = embedding_settings

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def close_client(self) -> None:
        """Close client. It is a no-op since we delegate
        closing to the global dict repo."""
        pass

    @staticmethod
    def from_config_settings(
        opts: ConfigSettings | None = None,
        *,
        client: QdrantClient | None = None,
    ) -> BaseRetriever:
        """
        Initializes a QdrantVectorStoreRetriever from a ConfigSettings
        object, or from the config.toml file.

        Args:
            opts: a ConfigSettings object, or none to read settings
                from the configutation file
            client: a QdrantClient object (optional). If provided,
                overrides the settings in ConfigSettings to locate
                the database.

        Returns:
            A QdrantVectorStoreRetriever object
        """

        logger = ExceptionConsoleLogger()
        opts = opts or load_settings(logger=logger)
        if opts is None:
            raise ValueError(
                "Could not initialize retriever due to "
                + "invalid config settings"
            )

        if bool(opts.RAG.retrieve_docs):
            return QdrantVectorStoreRetrieverGrouped.from_config_settings(
                opts, client=client
            )

        if client is None:
            # will raise error if not creatable
            client = global_client_from_config(opts.storage)
        return QdrantVectorStoreRetriever(
            client,
            opts.database.collection_name,
            encoding_to_qdrantembedding_model(
                opts.RAG.encoding_model
            ),
            opts.embeddings,
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
            self.qdrant_embedding,
            self.embedding_settings,
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

    # Implementation note: the default QdrantClient is needed
    # by the Pydantic constructor, and is just left to garbage
    # collection. It cannot be await-closed in init.
    # The real client object is passed by the constructor.
    # No need to close it in a destructor, as it is passed in.

    client: AsyncQdrantClient = Field(
        # Pydantic's BaseModel requires an object here
        default=AsyncQdrantClient(":memory:"),
        init=False,
    )
    collection_name: str = Field(default="", init=False)
    qdrant_embedding: QdrantEmbeddingModel = Field(
        default=encoding_to_qdrantembedding_model(
            ConfigSettings().RAG.encoding_model
        ),
        init=False,
    )
    embedding_settings: EmbeddingSettings = Field(
        default=ConfigSettings().embeddings, init=False
    )

    def __init__(
        self,
        client: AsyncQdrantClient,
        collection_name: str,
        qdrant_embedding: QdrantEmbeddingModel,
        embedding_settings: EmbeddingSettings,
    ):
        # TODO: verify the collection asynchronously
        # flag: bool = vsq.initialize_collection(
        #     client, collection_name, embedding_model
        # )
        # if not flag:
        #   raise RuntimeError("Could not initialize the collection")

        super().__init__(
            metadata={'embedding_model': qdrant_embedding.value}
        )

        # Now these can be set normally. self.client may already
        # be initialized, but we can't await close() here.
        self.client = client
        self.collection_name = collection_name
        self.qdrant_embedding = qdrant_embedding
        self.embedding_settings = embedding_settings

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def close_client(self) -> None:
        """Close client. It is a no-op since we delegate
        closing to the global dict repo."""
        pass

    @staticmethod
    def from_config_settings(
        opts: ConfigSettings | None = None,
        *,
        client: AsyncQdrantClient | None = None,
    ) -> BaseRetriever:
        """
        Initializes a ansynchronous QdrantVectorStoreRetriever from a
        ConfigSettings object, or from the config.toml file.

        Args:
            opts: a ConfigSettings object, or none to read settings
                from the configutation file
            client: an Async QdrantClient object (optional). If
                provided, overrides the settings in ConfigSettings
                to locate the database.


        Returns:
            An AsyncQdrantVectorStoreRetriever object
        """
        logger = ExceptionConsoleLogger()
        opts = opts or load_settings(logger=logger)
        if opts is None:
            raise ValueError(
                "Could not initialize retriever due to "
                + "invalid config settings"
            )

        if bool(opts.RAG.retrieve_docs):
            return AsyncQdrantVectorStoreRetrieverGrouped.from_config_settings(
                opts, client=client
            )

        if client is None:
            # will raise error if not creatable
            client = global_async_client_from_config(opts.storage)
        return AsyncQdrantVectorStoreRetriever(
            client,
            opts.database.collection_name,
            encoding_to_qdrantembedding_model(
                opts.RAG.encoding_model
            ),
            embedding_settings=opts.embeddings,
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
                self.qdrant_embedding,
                self.embedding_settings,
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
            self.qdrant_embedding,
            self.embedding_settings,
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

    # Implementation note: the default QdrantClient is needed
    # by the Pydantic constructor, and is closed immediately.
    # The real client object is passed by the constructor.
    # Do not close it in a destructor, as it is passed in.

    client: QdrantClient = Field(
        # Pydantic's BaseModel requires an object here
        default=QdrantClient(":memory:"),
        init=False,
    )
    collection_name: str = Field(default="", init=False)
    group_collection: str = Field(default="", init=False)
    group_field: str = Field(default=GROUP_UUID_KEY, init=False)
    limitgroups: int = Field(default=4, init=False)
    qdrant_embedding: QdrantEmbeddingModel = Field(
        default=encoding_to_qdrantembedding_model(
            ConfigSettings().RAG.encoding_model
        ),
        init=False,
    )
    embedding_settings: EmbeddingSettings = Field(
        default=ConfigSettings().embeddings, init=False
    )

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        group_collection: str,
        group_field: str,
        limitgroups: int,
        qdrant_embedding: QdrantEmbeddingModel,
        embedding_settings: EmbeddingSettings,
    ):
        flag: bool = vsq.initialize_collection(
            client,
            collection_name,
            qdrant_embedding,
            embedding_settings,
        )
        if not flag:
            raise RuntimeError("Could not initialize the collection")

        super().__init__(
            metadata={'embedding_model': qdrant_embedding.value}
        )

        # Now these can be set normally, except that self.client
        # may have already been initialized
        try:
            self.client.close()
        except Exception:
            pass
        self.client = client
        self.collection_name = collection_name
        self.group_collection = group_collection
        self.group_field = group_field
        self.limitgroups = limitgroups
        self.qdrant_embedding = qdrant_embedding
        self.embedding_settings = embedding_settings

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def close_client(self) -> None:
        """Close client. It is a no-op since we delegate
        closing to the global dict repo."""
        pass

    @staticmethod
    def from_config_settings(
        opts: ConfigSettings | None = None,
        *,
        client: QdrantClient | None = None,
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

        logger = ExceptionConsoleLogger()
        opts = opts or load_settings(logger=logger)
        if opts is None:
            raise ValueError(
                "Could not initialize retriever due to "
                + "invalid config settings"
            )
        dbOpts: DatabaseSettings = opts.database
        retrieve_docs = opts.RAG.retrieve_docs

        if retrieve_docs and not bool(dbOpts.companion_collection):
            logger.warning(
                "Retrieve docs directive ignored, no companion collection"
            )
            retrieve_docs = False

        if not retrieve_docs:
            return QdrantVectorStoreRetriever.from_config_settings(
                opts, client=client
            )

        if client is None:
            # will raise error if not creatable
            client = global_client_from_config(opts.storage)
        return QdrantVectorStoreRetrieverGrouped(
            client,
            dbOpts.collection_name,
            dbOpts.companion_collection,  # type: ignore (checked above)
            GROUP_UUID_KEY,
            4,
            encoding_to_qdrantembedding_model(
                opts.RAG.encoding_model
            ),
            opts.embeddings,
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
            self.qdrant_embedding,
            self.embedding_settings,
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

    # Implementation note: the default QdrantClient is needed
    # by the Pydantic constructor, and is just left to garbage
    # collection. It cannot be await-closed in init.
    # The real client object is passed by the constructor.
    # No need to close it in a destructor, as it is passed in.

    client: AsyncQdrantClient = Field(
        # Pydantic's BaseModel requires an object here
        default=AsyncQdrantClient(":memory:"),
        init=False,
    )
    collection_name: str = Field(default="", init=False)
    group_collection: str = Field(default="", init=False)
    group_field: str = Field(default=GROUP_UUID_KEY, init=False)
    limitgroups: int = Field(default=4, init=False)
    qdrant_embedding: QdrantEmbeddingModel = Field(
        default=encoding_to_qdrantembedding_model(
            ConfigSettings().RAG.encoding_model
        ),
        init=False,
    )
    embedding_settings: EmbeddingSettings = Field(
        default=ConfigSettings().embeddings, init=False
    )

    def __init__(
        self,
        client: AsyncQdrantClient,
        collection_name: str,
        group_collection: str,
        group_field: str,
        limitgroups: int,
        qdrant_embedding: vsq.QdrantEmbeddingModel,
        embedding_settings: EmbeddingSettings,
    ):
        # TODO: verify the collection asynchronously
        # flag: bool = vsq.initialize_collection(
        #     client, collection_name, embedding_model
        # )
        # if not flag:
        #   raise RuntimeError("Could not initialize the collection")

        super().__init__(
            metadata={'embedding_model': qdrant_embedding.value}
        )

        # Now these can be set normally. self.client may already
        # be initialized, but we can't await close() here.
        self.client = client
        self.collection_name = collection_name
        self.group_collection = group_collection
        self.group_field = group_field
        self.limitgroups = limitgroups
        self.qdrant_embedding = qdrant_embedding
        self.embedding_settings = embedding_settings

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def close_client(self) -> None:
        """Close client. It is a no-op since we delegate
        closing to the global dict repo."""
        pass

    @staticmethod
    def from_config_settings(
        opts: ConfigSettings | None = None,
        *,
        client: AsyncQdrantClient | None = None,
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

        logger = ExceptionConsoleLogger()
        opts = opts or load_settings(logger=logger)
        if opts is None:
            raise ValueError(
                "Could not initialize retriever due to "
                + "invalid config settings"
            )
        dbOpts: DatabaseSettings = opts.database
        retrieve_docs = opts.RAG.retrieve_docs

        if retrieve_docs and not bool(dbOpts.companion_collection):
            logger.warning(
                "Retrieve docs directive ignored, no companion collection"
            )
            retrieve_docs = False

        if not retrieve_docs:
            return (
                AsyncQdrantVectorStoreRetriever.from_config_settings(
                    opts, client=client
                )
            )

        if client is None:
            # will raise error if not creatable
            client = global_async_client_from_config(opts.storage)
        return AsyncQdrantVectorStoreRetrieverGrouped(
            client,
            dbOpts.collection_name,
            dbOpts.companion_collection,  # type: ignore (checked above)
            GROUP_UUID_KEY,
            4,
            encoding_to_qdrantembedding_model(
                opts.RAG.encoding_model
            ),
            opts.embeddings,
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
                self.qdrant_embedding,
                self.embedding_settings,
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
            self.qdrant_embedding,
            self.embedding_settings,
            query,
            limit=limit,
            group_field=self.group_field,
            group_size=self.limitgroups,
            payload=payload,
        )

        return self._results_to_documents(results)
