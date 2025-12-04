"""
Provides a global context to share a qdrant connection globally.

Global QdrantClient objects may be obtained by calling the functions
qdrant_client_from_config and qdrant_async_client_from_config. Both
accept a DatabaseSource value to identify the database. If no
argument is provided, the database storage in the configuation
file is used.

Database source values example values are: ":memory:",
LocalStorage(folder="path_to_storage"),
RemoteStorage(url=127.0.0.1,12324) (see config.py)

"""

import asyncio

from lmm.language_models.lazy_dict import LazyLoadingDict
from lmm.utils.logging import ConsoleLogger
from ..config.config import load_settings
from .vector_store_qdrant import (
    client_from_config,
    async_client_from_config,
    ConfigSettings,
    DatabaseSource,
)
from qdrant_client import QdrantClient, AsyncQdrantClient

logger = ConsoleLogger()


def _client_constructor(
    dbsource: DatabaseSource,
) -> QdrantClient:
    """The factory function of a QdrantClient"""
    client: QdrantClient | None = client_from_config(dbsource, logger)
    if client is None:
        raise ValueError("Could not create global client.")
    return client


def _async_client_constructor(
    dbsource: DatabaseSource,
) -> AsyncQdrantClient:
    """The factory function of an AsyncQdrantClient"""
    client: AsyncQdrantClient | None = async_client_from_config(
        dbsource, logger
    )
    if client is None:
        raise ValueError("Could not create global client.")
    return client


def _async_client_destructor(client: AsyncQdrantClient):
    asyncio.create_task(client.close())


qdrant_clients: LazyLoadingDict[DatabaseSource, QdrantClient] = (
    LazyLoadingDict(_client_constructor)
)

qdrant_async_clients: LazyLoadingDict[
    DatabaseSource, AsyncQdrantClient
] = LazyLoadingDict(
    _async_client_constructor, _async_client_destructor
)


def global_client_from_config(
    dbsource: DatabaseSource | None = None,
) -> QdrantClient:
    """Override of vector_store_qdrant homonymous function.
    This version caches a unique link to the database source.

    Args:
        dbsource: a DatabaseSource type or None. If None or missing,
            returns the database as specified in config.toml.

    Returns:
        a QdrantClient object.
    """

    if dbsource is None:
        opts: ConfigSettings | None = load_settings(logger=logger)
        if opts is None:
            raise ValueError(
                "Could not read settings to create global client."
            )
        dbsource = opts.storage

    return qdrant_clients[dbsource]


def global_async_client_from_config(
    dbsource: DatabaseSource | None = None,
) -> AsyncQdrantClient:
    """Override of vector_store_qdrant homonymous function.
    This version caches a unique link to the database source.

    Args:
        dbsource: a DatabaseSource type or None. If None or missing,
            returns the database as specified in config.toml.

    Returns:
        an AsyncQdrantClient object.
    """

    if dbsource is None:
        opts: ConfigSettings | None = load_settings(logger=logger)
        if opts is None:
            raise ValueError(
                "Could not read settings to create global client."
            )
        dbsource = opts.storage

    return qdrant_async_clients[dbsource]
