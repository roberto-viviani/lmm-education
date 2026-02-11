"""
Provides a global context to share a qdrant connection globally.
Provides singleton client objects and handles destruction
automatically.

Global QdrantClient objects may be obtained by calling the functions
global_client_from_config and global_async_client_from_config. Both
accept a DatabaseSource value to identify the database. If no
argument is provided, the database storage in the configuration
file config.toml is used.

Database source possible example values are: ":memory:",
LocalStorage(folder="path_to_storage"),
RemoteStorage(url=127.0.0.1,12324) (see config.py):

```python
from lmm_education.stores.vector_store_qdrant_context import (
    global_async_client_from_config,
)

# get an async client using default config from config.toml
client = global_async_client_from_config()
```

Important! Because destruction closes the client connection, which
is stored globally, DO NOT close the client obtained rhough this
module manually, i.e. by calling client.close().

Closing clients is handled automatically, but if you need to close
them, use global_clients_close() or global_async_clients_close().

Behavior:
    Raises ValueError if creation of Client object fails.

"""
# code rev: g+

# Automatic construction and destruction of singleton objects
# implemented with LazyLoadingDict

import asyncio

from lmm.language_models.lazy_dict import LazyLoadingDict
from lmm.utils.logging import ConsoleLogger, LoggerBase
from ..config.config import load_settings
from .vector_store_qdrant import (
    client_from_config,
    async_client_from_config,
    ConfigSettings,
    DatabaseSource,
)
from .vector_store_qdrant_utils import database_name
from qdrant_client import QdrantClient, AsyncQdrantClient

_logger = ConsoleLogger()


def _client_constructor(
    dbsource: DatabaseSource,
    logger: LoggerBase = _logger,
) -> QdrantClient:
    """The factory function of a QdrantClient uniquely defined by
    the dbsource argument"""
    client: QdrantClient | None = client_from_config(dbsource, logger)
    if client is None:
        # provide feedback on client type, as trying to access the
        # database with the wrong (synchronous) type of client is the
        # most common source of problems
        raise ValueError(
            "Could not create synchronous global client."
        )
    return client


def _async_client_constructor(
    dbsource: DatabaseSource,
    logger: LoggerBase = _logger,
) -> AsyncQdrantClient:
    """The factory function of an AsyncQdrantClient, uniquely
    define by the dbsoure argument"""
    client: AsyncQdrantClient | None = async_client_from_config(
        dbsource, logger
    )
    if client is None:
        # provide feedback on client type, as trying to access the
        # database with the wrong (asynchronous) type of client is the
        # most common source of problems
        raise ValueError(
            "Could not create asynchronous global client."
        )
    return client


def _client_destructor(
    client: QdrantClient, logger: LoggerBase = _logger
) -> None:
    """Destructor: wrapper of client close coroutine"""
    try:
        # these functions should run ok even if client closed.
        # However, a likely bug in qdrant_client.py tries to close
        # when client is garbage-collected, raising an error when
        # Python is shutting down.
        source: str = database_name(client)
        logger.info(f"Closing client {source}")
        client.close(grpc_grace=0.3)
    except Exception:
        # this may still fail if python is closing down
        pass


def _async_client_destructor(
    client: AsyncQdrantClient, logger: LoggerBase = _logger
) -> None:
    """Destructor: sync wrapper of async close coroutine"""
    try:
        # these functions should run ok even if client closed
        source: str = database_name(client)
        logger.info(f"Closing client {source}")
    except Exception:
        pass
    try:
        loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(client.close())
        finally:
            loop.close()
    except Exception:
        # During shutdown, even this might fail - suppress errors
        # (if Python is sutting down, sys.meta_path is None,
        # with the result that the loop fails and the file is not
        # closed).
        pass


qdrant_clients: LazyLoadingDict[DatabaseSource, QdrantClient] = (
    LazyLoadingDict(_client_constructor, _client_destructor)
)

qdrant_async_clients: LazyLoadingDict[
    DatabaseSource, AsyncQdrantClient
] = LazyLoadingDict(
    _async_client_constructor, _async_client_destructor
)


def global_client_from_config(
    dbsource: DatabaseSource | None = None,
    logger: LoggerBase = _logger,
) -> QdrantClient:
    """Override of vector_store_qdrant homonymous function.
    This version caches a unique link to the database source.

    Args:
        dbsource: a DatabaseSource type or None. If None or missing,
            returns the database as specified in config.toml.
        logger: a logger object for error reporting.

    Returns:
        a QdrantClient object.

    Raises:
        ValueError: If settings cannot be loaded from config.toml when
            dbsource is None, or if the underlying client_from_config()
            function fails to create a synchronous client.
    """

    if dbsource is None:
        opts: ConfigSettings | None = load_settings(logger=logger)
        if opts is None:
            raise ValueError(
                "Could not read settings to create global client."
            )
        dbsource = opts.storage

    return qdrant_clients[dbsource]


def global_clients_close() -> None:
    """Close all synchronous global clients.

    This function clears the global client cache, triggering
    the destructor for each cached client connection.

    Args:
        None

    Returns:
        None
    """
    qdrant_clients.clear()


def global_async_client_from_config(
    dbsource: DatabaseSource | None = None,
    logger: LoggerBase = _logger,
) -> AsyncQdrantClient:
    """Override of vector_store_qdrant homonymous function.
    This version caches a unique link to the database source.

    Args:
        dbsource: a DatabaseSource type or None. If None or missing,
            returns the database as specified in config.toml.
        logger: a logger object for error reporting.

    Returns:
        an AsyncQdrantClient object.

    Raises:
        ValueError: If settings cannot be loaded from config.toml when
            dbsource is None, or if the underlying async_client_from_config()
            function fails to create an asynchronous client.
    """

    if dbsource is None:
        opts: ConfigSettings | None = load_settings(logger=logger)
        if opts is None:
            raise ValueError(
                "Could not read settings to create global client."
            )
        dbsource = opts.storage

    return qdrant_async_clients[dbsource]


def global_async_clients_close() -> None:
    """Close all asynchronous global clients.

    This function clears the global client cache, triggering
    the destructor for each cached client connection.

    Args:
        None

    Returns:
        None
    """
    qdrant_async_clients.clear()
