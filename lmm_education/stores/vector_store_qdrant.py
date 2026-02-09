"""
Computes the embeddings and handles uploading and saving the data
to the vector database.

This is a low-level module that provides the interface to the qdrant
database, while also bringing it together with the settings informa-
tion in config.toml and an embeddings provider. Exceptions are relayed
to a logger object for use in a REPL interface. A connection to the
database is representend by a `QdrantClient` object of the Qdrant API,
which may be initialized directly through the constructor, or through
the `client_from_config` function which reads the database options
from config.toml:

```python
    from lmm_education.stores.vector_store_qdrant import client_from_config

    client: QdrantClient | None = client_from_config()
    # check client is not None
```

It is also possible to intialize a config object explicitly for the
properties that override config.toml.

```python
    from lmm_education.stores.vector_store_qdrant import client_from_config
    from lmm_education.config.config import ConfigSettings

    # read settings from config.toml, but override 'storage':
    settings = ConfigSettings(storage=":memory:")
    client: QdrantClient | None = client_from_config(settings)
```

(Please also see the vector_store_qdrant_context module to create a
global client object that is automatically closed).

The functions in this module use a logger to communicate errors, so
that the way exceptions are handled depends on the logger type. If no
logger is specified, an error message is printed on the console.

```python
    from lmm_education.stores.vector_store_qdrant import client_from_config
    from lmm.utils.logging import LogfileLogger

    logger = LogfileLogger()
    client: QdrantClient | None = client_from_config(None, logger)
    if client is None:
        # read causes from logger
```

The role of client_from_config is to bind the creation of a
QdrantClient object with the relevant settings in config.toml, and
channel possible exceptions through the logger. Note, however, that
the QdrantClient can also be created with the qdrant API directly.
Note that in both cases the client needs be closed before exiting.

A slightly more higher-lever alternative is obtaining the client
through a context manager by calling `qdrant_client_context`:

```python
try:
    with qdrant_client_context() as client:
        result_docs = upload(
            client, "documents", model, settings, doc_chunks
        )
        result_imgs = upload(
            client, "images", model, settings, img_chunks
        )
except Exception as e:
    .... error handling
```

Please note that at present you have to capture errors when using
the context manager.

The remaining functions of the module take the client object to read
and write to the database. All calls go through initialize_collection,
which takes the name of the collection and an embedding model to
specify how the data should be embedded (what type of dense and sparse
vector, or any hybrid combination of those, should be used):

```python
    # ... client creation not shown
    from lmm_education.stores.vector_store_qdrant import (
        initialize_collection,
        QdrantEmbeddingModel,
    )

    embedding_model = QdrantEmbeddingModel.DENSE
    flag: bool = initialize_collection(
        client,
        "documents",
        embedding_model,
        ConfigSettings().embeddings,
        logger=logger)
```

```python
embedding_model = QdrantEmbeddingModel.DENSE
opts: DatabaseSource = ":memory:"
try:
    with qdrant_client_context(opts) as client:
        result = initialize_collection(
            client,
            "Main",
            embedding_model,
            ConfigSettings().embeddings,
        )
except Exception as e:
    .... handle exceptions
```

In every call to the functions of the module, the client, the
collection name, the embedding model, and the embedding provider
settings are given as arguments.

Data are ingested in the database in the form of lists of `Chunk`
objects (see the `lmm_education.stores.chunks` module).

```python
points: list[Point] = upload(
    client,
    "documents",
    embedding_model,
    ConfigSettings().embeddings,
    chunks,
    logger=logger,
)
```

The `Point` objects are the representations of records used by Qdrant.
This function converts each `Chunk` object to a `Point` object prior
to ingesting. This conversion includes the formation of dense and
sparse vectors, as specified by the embedding model. The points are
returned if the upload is successful.

The database may then be queried:

```python
points: list[ScoredPoint] = query(
    client,
    "documents",
    embedding_model,
    ConfigSettings().embeddings,
    "What are the main uses of logistic regression?",
    limit=12,  # max number retrieved points
    payload=True,  # all payload fields
    logger=logger,
)

# retrieve text
for pt in points:
    print(f"{pt.score}\n{pt.payload['page_content']}\n\n")
```

`ScoredPoint` is the Qdrant class to return the payload of the
retrieved points (which includes the text).

Responsibilities:
    definition of Qdrant embedding model
    compute embeddings
    upload data and query

Main functions:
    client_from_config / async_client_from_config
    initialize_collection / ainitialize_collection
    initialize_collection_from_config / ainitialize_collection_from_config
    upload / aupload
    query / aquery
    query_grouped / aquery_grouped

rev a 26.10
"""

from enum import Enum
from typing import Any
from contextlib import contextmanager, asynccontextmanager
from collections.abc import Callable, Generator, AsyncGenerator

# langchain
from langchain_core.embeddings import Embeddings

# qdrant
from qdrant_client import QdrantClient, models
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    PointStruct as Point,
    ScoredPoint,
    Record,
    GroupsResult,
    PointGroup,
)
from qdrant_client.http.models import QueryResponse as QdrantResponse
from qdrant_client.http.exceptions import (
    ApiException,
    UnexpectedResponse,
)
from fastembed import SparseEmbedding, SparseTextEmbedding


# lmmarkdown
from lmm.scan.scan_keys import GROUP_UUID_KEY
from lmm.config.config import EmbeddingSettings
from lmm.scan.chunks import (
    EncodingModel,
    Chunk,
)
from lmm.markdown.parse_markdown import (
    Block,
    TextBlock,
    MetadataBlock,
)

# lmm markdown for education
from ..config.config import (
    DatabaseSource,
    ConfigSettings,
    load_settings,
)

# utils
from .vector_store_qdrant_utils import (
    check_schema,
    acheck_schema,
)

# Set up logger
from lmm.utils.logging import LoggerBase, get_logger

default_logger: LoggerBase = get_logger(__name__)


class QdrantEmbeddingModel(Enum):
    """Enum for embedding strategies"""

    DENSE = "dense"
    MULTIVECTOR = "multivector"
    SPARSE = "sparse"
    HYBRID_DENSE = "hybrid_dense"
    HYBRID_MULTIVECTOR = "hybrid_multivector"
    UUID = "UUID"  # no embedding


def encoding_to_qdrantembedding_model(
    em: EncodingModel,
) -> QdrantEmbeddingModel:
    """Embedding model required for encoding"""
    match em:
        case EncodingModel.NONE:
            return QdrantEmbeddingModel.UUID
        case EncodingModel.CONTENT | EncodingModel.MERGED:
            return QdrantEmbeddingModel.DENSE
        case EncodingModel.MULTIVECTOR:
            return QdrantEmbeddingModel.MULTIVECTOR
        case EncodingModel.SPARSE:
            return QdrantEmbeddingModel.SPARSE
        case (
            EncodingModel.SPARSE_CONTENT | EncodingModel.SPARSE_MERGED
        ):
            return QdrantEmbeddingModel.HYBRID_DENSE
        case EncodingModel.SPARSE_MULTIVECTOR:
            return QdrantEmbeddingModel.HYBRID_MULTIVECTOR
        case _:
            raise Exception(
                "Unreachable code reached due to "
                + "invalid encoding model: "
                + str(em)
            )


DENSE_VECTOR_NAME: str = 'content'
SPARSE_VECTOR_NAME: str = 'annotations'

# Memoized sparse model instance
_sparse_model_cache: SparseTextEmbedding | None = None
SPARSE_MODEL_NAME: str = "Qdrant/bm25"


def _get_sparse_model(
    embedding_settings: EmbeddingSettings | None = None,
) -> SparseTextEmbedding:
    """Get memoized SparseTextEmbedding model instance."""
    global _sparse_model_cache
    if _sparse_model_cache is None:
        if embedding_settings is None:
            embedding_settings = ConfigSettings().embeddings
        _sparse_model_cache = SparseTextEmbedding(
            model_name=str(embedding_settings.sparse_model)
        )
    return _sparse_model_cache


def client_from_config(
    opts: DatabaseSource | ConfigSettings | None = None,
    logger: LoggerBase = default_logger,
) -> QdrantClient | None:
    """ "
    Create a qdrant clients from config settings. Reads from config
    toml file settings if none given.

    Please note that a client should be closed before exiting.

    Args:
        opts: the config settings
        logger: a logger object

    Returns:
        a QdrantClient object
    """
    from lmm_education.config.config import (
        LocalStorage,
        RemoteSource,
    )

    opts = opts or load_settings(logger=logger)
    if opts is None:
        logger.error(
            "Could not initialize client due to invalid settings."
        )
        return None

    try:
        opts_storage: DatabaseSource = (
            opts.storage if isinstance(opts, ConfigSettings) else opts
        )
        client: QdrantClient
        match opts_storage:
            case ':memory:':
                client = QdrantClient(':memory:')
            case LocalStorage(folder=folder):
                client = QdrantClient(path=folder)
            case RemoteSource(url=url, port=port):
                client = QdrantClient(url=str(url), port=port)
            case _:
                logger.error("Invalid database source")
                return None
    except ConnectionError:
        logger.error(
            "Could not connect to the qdrant server, which may be down."
        )
        return None
    except UnexpectedResponse as e:
        logger.error(f"Could not initialize qdrant client: {e}")
        return None
    except ApiException as e:
        logger.error(
            f"Could not initialize qdrant client due to API error: {e}"
        )
        return None
    except RuntimeError as e:
        # This often caused by accessing qdrant with sync client after
        # prior acces with async client
        if "aready accessed" in str(e):
            logger.error(
                f"Could not initialize qdrant client due to "
                f"previous async initialization?\n{e}"
            )
        else:
            logger.error(
                f"Could not initialize qdrant client due to "
                f"runtime error:\n{e}"
            )
        return None
    except Exception as e:
        logger.error(f"Could not initialize qdrant client:\n{e}")
        return None

    return client


def async_client_from_config(
    opts: DatabaseSource | ConfigSettings | None = None,
    logger: LoggerBase = default_logger,
) -> AsyncQdrantClient | None:
    """ "
    Create a qdrant clients from config settings. Reads from config
    toml file settings if none gievn.

    Args:
        opts: the config settings
        logger: a logger object

    Returns:
        a QdrantClient object
    """
    from lmm_education.config.config import (
        LocalStorage,
        RemoteSource,
    )

    opts = opts or load_settings(logger=logger)
    if opts is None:
        logger.error(
            "Could not initialize client due to invalid settings."
        )
        return None

    try:
        opts_storage: DatabaseSource = (
            opts.storage if isinstance(opts, ConfigSettings) else opts
        )

        client: AsyncQdrantClient
        match opts_storage:
            case ':memory:':
                client = AsyncQdrantClient(':memory:')
            case LocalStorage(folder=folder):
                client = AsyncQdrantClient(path=folder)
            case RemoteSource(url=url, port=port):
                client = AsyncQdrantClient(url=str(url), port=port)
            case _:
                raise ValueError("Invalid database source")
    except ConnectionError:
        logger.error(
            "Could not connect to the qdrant server, which may be down."
        )
        return None
    except UnexpectedResponse as e:
        logger.error(f"Could not initialize qdrant client: {e}")
        return None
    except ApiException as e:
        logger.error(
            f"Could not initialize qdrant client due to API error: {e}"
        )
        return None
    except RuntimeError as e:
        # This often caused by accessing qdrant with async client after
        # prior acces with sync client
        if "aready accessed" in str(e):
            logger.error(
                f"Could not initialize qdrant client due to "
                f"previous sync initialization?\n{e}"
            )
        else:
            logger.error(
                f"Could not initialize qdrant client due to "
                f"runtime error:\n{e}"
            )
        return None
    except Exception as e:
        print(type(e))
        logger.error(f"Could not initialize qdrant client:\n{e}")
        return None

    return client


@contextmanager
def qdrant_client_context(
    config: DatabaseSource | ConfigSettings | None = None,
    logger: LoggerBase = default_logger,
) -> Generator[QdrantClient]:
    client = None
    try:
        client = client_from_config(config, logger)
        if client is None:
            logger.error("Failed to create Qdrant client")
            raise ConnectionError("Failed to create Qdrant client")
        yield client
    finally:
        if client is not None:
            try:
                client.close()
            except Exception as e:
                logger.warning(f"Error closing client: {e}")


@asynccontextmanager
async def async_qdrant_client_context(
    config: DatabaseSource | ConfigSettings | None = None,
    logger: LoggerBase = default_logger,
) -> AsyncGenerator[AsyncQdrantClient]:
    client = None
    try:
        client = async_client_from_config(config, logger)
        if client is None:
            logger.error("Failed to create Qdrant client")
            raise ConnectionError("Failed to create Qdrant client")
        yield client
    finally:
        if client is not None:
            try:
                await client.close()
            except Exception as e:
                logger.warning(f"Error closing client: {e}")


def initialize_collection(
    client: QdrantClient,
    collection_name: str,
    qdrant_model: QdrantEmbeddingModel,
    embedding_settings: EmbeddingSettings,
    *,
    logger: LoggerBase = default_logger,
) -> bool:
    """
    Check that the collection supports the embedding model, if
    already in the database. If not, create a collection supporting
    the embedding model

    Args:
        client: QdrantClient object encapsulating the conn to the db
        collection_name: the collection.
        qdrant_model: the qdrant embedding model
        embedding_settings: the embedding settings

    Returns:
        a boolean flag indicating that the client may be used with
            these parameters.

    Note:
        the embedding_settings are required to create an embedding to
            obtain the dense embedding vector length, and record in
            the schema their dimension.
    """

    from requests.exceptions import ConnectionError

    try:
        from lmm.language_models.langchain.runnables import (
            create_embeddings,
        )

        encoder: Embeddings = create_embeddings(embedding_settings)
    except ConnectionError:
        logger.error(
            "Could not connect to the language model.\n"
            + "Check the internet connection."
        )
        return False
    except ImportError:
        logger.error(
            "Could not create langchain kernel. Check that langchain "
            "is installed, and an internet connection is available."
        )
        return False
    except Exception as e:
        logger.error(
            "Could not initialize language model engine:\n" + str(e)
        )
        return False

    try:
        if client.collection_exists(collection_name):
            info = client.get_collection(collection_name)
            params = info.config.params  # noqa  #type: ignore

            # checks that the opts and the collection
            # are compatible
            return check_schema(
                client,
                collection_name,
                qdrant_model,
                embedding_settings,
                logger=logger,
            )

        # determine embedding size
        if not (
            qdrant_model == QdrantEmbeddingModel.UUID
            or qdrant_model == QdrantEmbeddingModel.SPARSE
        ):
            try:
                data: list[float] = encoder.embed_query("Test query")
            except Exception as e:
                logger.error(
                    "Could not determine embedding vector size, "
                    + f"cannot embed.\n{e}"
                )
                return False

            embedding_size: int = len(data)
        else:
            embedding_size: int = 0

        match qdrant_model:
            case QdrantEmbeddingModel.UUID:
                client.create_collection(
                    collection_name=collection_name, vectors_config={}
                )
            case QdrantEmbeddingModel.DENSE:
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        DENSE_VECTOR_NAME: models.VectorParams(
                            size=embedding_size,
                            distance=models.Distance.COSINE,
                        )
                    },
                )
            case QdrantEmbeddingModel.MULTIVECTOR:
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        DENSE_VECTOR_NAME: models.VectorParams(
                            size=embedding_size,
                            distance=models.Distance.COSINE,
                            multivector_config=models.MultiVectorConfig(
                                comparator=models.MultiVectorComparator.MAX_SIM
                            ),
                        )
                    },
                )
            case QdrantEmbeddingModel.SPARSE:
                # TO DO: indices
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config={},
                    sparse_vectors_config={
                        SPARSE_VECTOR_NAME: models.SparseVectorParams()
                    },
                )
            case QdrantEmbeddingModel.HYBRID_DENSE:
                # TO DO: indices
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        DENSE_VECTOR_NAME: models.VectorParams(
                            size=embedding_size,
                            distance=models.Distance.COSINE,
                        )
                    },
                    sparse_vectors_config={
                        SPARSE_VECTOR_NAME: models.SparseVectorParams()
                    },
                )
            case QdrantEmbeddingModel.HYBRID_MULTIVECTOR:
                # TO DO: indices
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        DENSE_VECTOR_NAME: models.VectorParams(
                            size=embedding_size,
                            distance=models.Distance.COSINE,
                            multivector_config=models.MultiVectorConfig(
                                comparator=models.MultiVectorComparator.MAX_SIM
                            ),
                        )
                    },
                    sparse_vectors_config={
                        SPARSE_VECTOR_NAME: models.SparseVectorParams()
                    },
                )
            case _:
                raise RuntimeError(
                    "Unreachable code reached due to invalid "
                    + "embedding model: "
                    + str(qdrant_model)
                )
    except ConnectionError:
        logger.error(
            "Could not connect to the qdrant server, which may be down."
        )
        return False
    except UnexpectedResponse as e:
        logger.error(f"Could not initialize {collection_name}: {e}")
        return False
    except ApiException as e:
        logger.error(
            f"Could not initialize {collection_name} due to API error: {e}"
        )
        return False
    except Exception as e:
        logger.error(f"Could not initialize {collection_name}: {e}")
        return False

    # register the schema
    try:
        check_schema(
            client,
            collection_name,
            qdrant_model,
            embedding_settings,
            logger=logger,
        )
    except Exception as e:
        logger.error(
            f"Unepected error: {collection_name} was"
            "initialized, but the schema could not be"
            "registered.\nCall initialize_collection again to"
            f"attempt to record schema. Reason for the failure:\n{e}"
        )
        # return True nevertheless

    return True


def initialize_collection_from_config(
    client: QdrantClient,
    collection_name: str | None = None,
    opts: ConfigSettings | None = None,
    *,
    logger: LoggerBase = default_logger,
) -> bool:
    """
    See initialize_collection. If collection_name is provided, it
    will override the collection_name in the config settings object.
    """
    opts = opts or load_settings(logger=logger)
    if opts is None:
        logger.error(
            "Could not initialize client due to invalid " "settings."
        )
        return False
    if not collection_name:
        collection_name = opts.database.collection_name

    return initialize_collection(
        client,
        collection_name,
        encoding_to_qdrantembedding_model(opts.RAG.encoding_model),
        opts.embeddings,
        logger=logger,
    )


async def ainitialize_collection(
    client: AsyncQdrantClient,
    collection_name: str,
    qdrant_model: QdrantEmbeddingModel,
    embedding_settings: EmbeddingSettings,
    *,
    logger: LoggerBase = default_logger,
) -> bool:
    """
    Check that the collection supports the embedding model, if
    already in the database. If not, create a collection supporting
    the embedding model

    Args:
        client: QdrantClient object encapsulating the conn to the db
        collection_name: the collection
        qdrant_model: the qdrant embedding model
        embedding_settings: the embedding settings

    Returns:
        a boolean signaling successful initialization and that the
            client may be used with these parameters.

    Note:
        the embedding_settings are required to create an embedding to
            obtain the dense embedding vector length, and record the
            embedding vector length in the schema.
    """

    from requests.exceptions import ConnectionError

    try:
        from lmm.language_models.langchain.runnables import (
            create_embeddings,
        )

        encoder: Embeddings = create_embeddings(embedding_settings)
    except ConnectionError:
        logger.error(
            "Could not connect to language models.\n"
            + "Check the internet connection."
        )
        return False
    except ImportError:
        logger.error(
            "Could not create langchain kernel. Check that langchain "
            "is installed, and an internet connection is available."
        )
        return False
    except Exception as e:
        logger.error(
            "Could not initialize language model engine:\n" + str(e)
        )
        return False

    try:
        if await client.collection_exists(collection_name):
            info = await client.get_collection(collection_name)
            params = info.config.params  # noqa  # type: ignore

            # checks that the opts and the collection
            # are compatible
            return await acheck_schema(
                client,
                collection_name,
                qdrant_model,
                embedding_settings,
            )

        # determine embedding size
        if not (
            qdrant_model == QdrantEmbeddingModel.UUID
            or qdrant_model == QdrantEmbeddingModel.SPARSE
        ):
            data: list[float] = await encoder.aembed_query(
                "Test query"
            )
            embedding_size: int = len(data)
        else:
            embedding_size: int = 0

        match qdrant_model:
            case QdrantEmbeddingModel.UUID:
                await client.create_collection(
                    collection_name=collection_name, vectors_config={}
                )
            case QdrantEmbeddingModel.DENSE:
                await client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        DENSE_VECTOR_NAME: models.VectorParams(
                            size=embedding_size,
                            distance=models.Distance.COSINE,
                        )
                    },
                )
            case QdrantEmbeddingModel.MULTIVECTOR:
                await client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        DENSE_VECTOR_NAME: models.VectorParams(
                            size=embedding_size,
                            distance=models.Distance.COSINE,
                            multivector_config=models.MultiVectorConfig(
                                comparator=models.MultiVectorComparator.MAX_SIM
                            ),
                        )
                    },
                )
            case QdrantEmbeddingModel.SPARSE:
                # TO DO: indices
                await client.create_collection(
                    collection_name=collection_name,
                    vectors_config={},
                    sparse_vectors_config={
                        SPARSE_VECTOR_NAME: models.SparseVectorParams()
                    },
                )
            case QdrantEmbeddingModel.HYBRID_DENSE:
                # TO DO: indices
                await client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        DENSE_VECTOR_NAME: models.VectorParams(
                            size=embedding_size,
                            distance=models.Distance.COSINE,
                        )
                    },
                    sparse_vectors_config={
                        SPARSE_VECTOR_NAME: models.SparseVectorParams()
                    },
                )
            case QdrantEmbeddingModel.HYBRID_MULTIVECTOR:
                # TO DO: indices
                await client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        DENSE_VECTOR_NAME: models.VectorParams(
                            size=embedding_size,
                            distance=models.Distance.COSINE,
                            multivector_config=models.MultiVectorConfig(
                                comparator=models.MultiVectorComparator.MAX_SIM
                            ),
                        )
                    },
                    sparse_vectors_config={
                        SPARSE_VECTOR_NAME: models.SparseVectorParams()
                    },
                )

            case _:
                raise RuntimeError(
                    "Unreachable code reached due to invalid "
                    + "embedding model: "
                    + str(qdrant_model)
                )

    except ConnectionError:
        logger.error(
            "Could not connect to the qdrant server, which may be down."
        )
        return False
    except UnexpectedResponse as e:
        logger.error(f"Could not initialize {collection_name}: {e}")
        return False
    except ApiException as e:
        logger.error(
            f"Could not initialize {collection_name} due to API error: {e}"
        )
        return False
    except Exception as e:
        logger.error(f"Could not initialize {collection_name}: {e}")
        return False

    # register the schema
    try:
        await acheck_schema(
            client,
            collection_name,
            qdrant_model,
            embedding_settings,
            logger=logger,
        )
    except Exception as e:
        logger.error(
            f"Unepected error: {collection_name} was"
            "initialized, but the schema could not be"
            "registered.\nCall initialize_collection again to"
            f"attempt to record schema. Reason for the failure:\n{e}"
        )
        # return True nevertheless

    return True


async def ainitialize_collection_from_config(
    client: AsyncQdrantClient,
    collection_name: str | None = None,
    opts: ConfigSettings | None = None,
    *,
    logger: LoggerBase = default_logger,
) -> bool:
    """See ainitialize_collection"""
    opts = opts or load_settings(logger=logger)
    if opts is None:
        logger.error(
            "Could not initialize client due to invalid " "settings."
        )
        return False
    if not collection_name:
        collection_name = opts.database.collection_name

    return await ainitialize_collection(
        client,
        collection_name,
        encoding_to_qdrantembedding_model(opts.RAG.encoding_model),
        opts.embeddings,
        logger=logger,
    )


def _chunk_to_payload_langchain(x: 'Chunk') -> dict[str, Any]:
    """
    Map content of Chunk object to payload (langchain schama)

    Args:
        x: a chunk object

    Returns:
        a dict object mapping a string key to a value
    """
    # One issue here is that llm frameworks like langchain store
    # the text from which the embeddings were computed in a main
    # key, putting all the rest in a metadata key. This is not
    # the same model as the vector databases, which may store
    # many things in the payload, with the text being one among
    # many. This is important for filtering, etc.
    # One can always filter through 'metadata.keyname', while this
    # is not most natural. Alas, to maintain compatibility with
    # langchain, we adopt its schema in the data saved into the
    # vector database. This function does precisely this: maps
    # the content for saving from the Chunk object into the
    # scheme recognised by langchain.
    # The exception is the GROUP_UUID_KEY for 'look_up' queries.
    if GROUP_UUID_KEY in x.metadata:
        # Here, we bring out GROUP_UUID_KEY from the metadata
        # so that qdrant can use it in 'look_up' queries.
        return {
            'page_content': x.content,
            GROUP_UUID_KEY: x.metadata.pop(GROUP_UUID_KEY, ""),
            'metadata': x.metadata,
        }
    else:
        # The standard langchain schema.
        return {'page_content': x.content, 'metadata': x.metadata}


# TODO: async version
def chunks_to_points(
    chunks: list[Chunk],
    qdrant_model: QdrantEmbeddingModel,
    embedding_settings: EmbeddingSettings,
    *,
    logger: LoggerBase = default_logger,
    chunk_to_payload: Callable[
        [Chunk], dict[str, Any]
    ] = _chunk_to_payload_langchain,
) -> list[Point]:
    """
    Converts a list of chunks into a list of the record objects
    understood by the Qdrant database, using an embedding model
    for the conversion.

    Args:
        chunks: the chunks list
        qdrant_model: the embedding model, or a ConfigSettings object from
            which the model may be deduced
        embedding_settings: the embedding settings
        logger: a logger object
        chunk_to_payload: a function to map chunks to the Langchain
            representation (internal use)

    Returns:
        a list of Point objects (PointStruct)
    """

    if not chunks:
        return []

    # load embedding model
    from requests.exceptions import ConnectionError

    try:
        from lmm.language_models.langchain.runnables import (
            create_embeddings,
        )

        # if embeddings_model is None, read from config.toml
        encoder: Embeddings = create_embeddings(embedding_settings)
    except ConnectionError:
        logger.error(
            "Could not connect to language models.\n"
            + "Check the internet connection.",
        )
        return []
    except ImportError:
        logger.error(
            "Could not create langchain kernel. Check that langchain "
            "is installed, and an internet connection is available."
        )
        return []
    except Exception as e:
        logger.error(
            "Could not initialize language model engine:\n" + str(e)
        )
        return []

    # the payload saved in the database is given by chunk_to_payload,
    # the emmbedding by the embedding model
    points: list[Point] = []
    match qdrant_model:
        case QdrantEmbeddingModel.UUID:
            points = [
                Point(
                    id=d.uuid, vector={}, payload=chunk_to_payload(d)
                )
                for d in chunks
            ]

        case QdrantEmbeddingModel.DENSE:
            try:
                vect = encoder.embed_documents(
                    [t.dense_encoding for t in chunks]
                )
            except Exception:
                logger.error("Could not create encoding")
                return []
            points = [
                Point(
                    id=d.uuid,
                    vector={DENSE_VECTOR_NAME: v},
                    payload=chunk_to_payload(d),
                )
                for d, v in zip(chunks, vect)
            ]

        case QdrantEmbeddingModel.MULTIVECTOR:
            try:
                vect = [
                    encoder.embed_documents(
                        [t.annotations for t in chunks]
                    ),
                    encoder.embed_documents(
                        [t.dense_encoding for t in chunks]
                    ),
                ]
            except Exception:
                logger.error("Could not create encoding")
                return []
            points = [
                Point(
                    id=d.uuid,
                    vector={DENSE_VECTOR_NAME: [v1, v2]},
                    payload=chunk_to_payload(d),
                )
                for d, v1, v2 in zip(chunks, vect[0], vect[1])
            ]

        case QdrantEmbeddingModel.SPARSE:
            try:
                sparse_model = _get_sparse_model(embedding_settings)
                sparse_embeddings = list(
                    sparse_model.embed(
                        [d.sparse_encoding for d in chunks]
                    )
                )
            except Exception:
                logger.error("Could not create encoding")
                return []
            points = [
                Point(
                    id=d.uuid,
                    vector={
                        SPARSE_VECTOR_NAME: models.SparseVector(
                            indices=s.indices.tolist(),
                            values=s.values.tolist(),
                        )
                    },
                    payload=chunk_to_payload(d),
                )
                for d, s in zip(chunks, sparse_embeddings)
            ]

        case QdrantEmbeddingModel.HYBRID_DENSE:
            try:
                vect = encoder.embed_documents(
                    [t.dense_encoding for t in chunks]
                )
                sparse_model = _get_sparse_model(embedding_settings)
                sparse_embeddings = list(
                    sparse_model.embed(
                        [d.sparse_encoding for d in chunks]
                    )
                )
            except Exception:
                logger.error("Could not create encoding")
                return []
            points = [
                Point(
                    id=d.uuid,
                    vector={
                        DENSE_VECTOR_NAME: v,
                        SPARSE_VECTOR_NAME: models.SparseVector(
                            indices=s.indices.tolist(),
                            values=s.values.tolist(),
                        ),
                    },
                    payload=chunk_to_payload(d),
                )
                for d, v, s in zip(chunks, vect, sparse_embeddings)
            ]

        case QdrantEmbeddingModel.HYBRID_MULTIVECTOR:
            try:
                vect = [
                    encoder.embed_documents(
                        [t.annotations for t in chunks]
                    ),
                    encoder.embed_documents(
                        [t.dense_encoding for t in chunks]
                    ),
                ]
                sparse_model = _get_sparse_model(embedding_settings)
                sparse_embeddings = list(
                    sparse_model.embed(
                        [d.sparse_encoding for d in chunks]
                    )
                )
            except Exception:
                logger.error("Could not create encoding")
                return []
            points = [
                Point(
                    id=d.uuid,
                    vector={
                        DENSE_VECTOR_NAME: [v1, v2],
                        SPARSE_VECTOR_NAME: models.SparseVector(
                            indices=s.indices.tolist(),
                            values=s.values.tolist(),
                        ),
                    },
                    payload=chunk_to_payload(d),
                )
                for d, v1, v2, s in zip(
                    chunks, vect[0], vect[1], sparse_embeddings
                )
            ]

        case _:
            raise RuntimeError(
                "Unreachable code reached due to invalid "
                + "embedding model: "
                + str(qdrant_model)
            )

    # This should not happen, but it is important to check
    # as there will be no way to detect the origin of problems
    # at query time.
    if not len(points) == len(chunks):
        raise SystemError(
            f"Unexpected failure in generating "
            f"Qdrant points: {len(points)} points for "
            f"{len(chunks)} chunks "
            f"(qdrant embedding model: {qdrant_model})"
        )

    return points


def upload(
    client: QdrantClient,
    collection_name: str,
    qdrant_model: QdrantEmbeddingModel,
    embedding_settings: EmbeddingSettings,
    chunks: list[Chunk],
    *,
    logger: LoggerBase = default_logger,
) -> list[Point]:
    """
    Upload a list of chunks into the Qdrant database

    Args:
        client: the connection to the database
        collection_name: the collection
        qdrant_model: the qdrant embedding model
        embedding_settings: the embedding settings
        chunks: the chunk list

    Returns:
        a list of Point objects
    """

    if not initialize_collection(
        client,
        collection_name,
        qdrant_model,
        embedding_settings,
        logger=logger,
    ):
        logger.error("Could not initialize collection.")
        return []

    points: list[Point] = chunks_to_points(
        chunks, qdrant_model, embedding_settings, logger=logger
    )

    if not points:
        return []

    try:
        client.upsert(collection_name=collection_name, points=points)
    except ConnectionError:
        logger.error(
            "Could not connect to the qdrant server, which may be down."
        )
        return []
    except UnexpectedResponse as e:
        logger.error(f"Could not write to vector database: {e}")
        return []
    except ApiException as e:
        logger.error(
            f"Could not write to database due to API error: {e}"
        )
        return []
    except Exception as e:
        logger.error(f"Could not upload chunks to database:\n{e}")
        return []

    return points


async def aupload(
    client: AsyncQdrantClient,
    collection_name: str,
    qdrant_model: QdrantEmbeddingModel,
    embedding_settings: EmbeddingSettings,
    chunks: list[Chunk],
    *,
    logger: LoggerBase = default_logger,
) -> list[Point]:
    """
    Upload a list of chunks into the Qdrant database

    Args:
        client: the connection to the database
        collection_name: the collection
        qdrant_model: the qdrant embedding model
        embedding_settings: the embedding settings
        chunks: the chunk list

    Returns:
        a list of Point objects
    """

    if not await ainitialize_collection(
        client,
        collection_name,
        qdrant_model,
        embedding_settings,
        logger=logger,
    ):
        logger.error("Could not initialize collection")
        return []

    points: list[Point] = chunks_to_points(
        chunks, qdrant_model, embedding_settings, logger=logger
    )
    if not points:
        return []

    try:
        await client.upsert(
            collection_name=collection_name, points=points
        )
    except ConnectionError:
        logger.error(
            "Could not connect to the qdrant server, which may be down."
        )
        return []
    except UnexpectedResponse as e:
        logger.error(f"Could not write to vector database: {e}")
        return []
    except ApiException as e:
        logger.error(
            f"Could not write to database due to API error: {e}"
        )
        return []
    except Exception:
        logger.error("Could not upload chunks to database")
        return []

    return points


def query(
    client: QdrantClient,
    collection_name: str,
    qdrant_model: QdrantEmbeddingModel,
    embedding_settings: EmbeddingSettings,
    querytext: str,
    *,
    limit: int = 12,
    payload: list[str] | bool = ['page_content'],
    logger: LoggerBase = default_logger,
) -> list[ScoredPoint]:
    """
    Executes a query on the client.

    Args:
        client: a QdrantClient object
        collection_name: the collection to query
        qdrant_model: the qdrant embedding model
        embedding_settings: the embedding settings
        querytext: the target text
        limit: max number of chunks retrieved
        payload: what properties to be retrieved; defaults to the text
            retrieved for similarity to the querytext
        logger: a logger object

    Returns:
        a list of ScoredPoint objects.
    """

    # load language model
    from requests.exceptions import ConnectionError

    try:
        from lmm.language_models.langchain.runnables import (
            create_embeddings,
        )

        encoder: Embeddings = create_embeddings(embedding_settings)
    except ConnectionError:
        logger.error(
            "Could not connect to language models.\n"
            + "Check the internet connection.",
        )
        return []
    except ImportError:
        logger.error(
            "Could not create langchain kernel. Check that langchain "
            "is installed, and an internet connection is available."
        )
        return []
    except Exception as e:
        logger.error(
            "Could not initialize language model engine:\n" + str(e)
        )
        return []

    # TODO: check querytext is UUID in UUID model

    response: QdrantResponse = QdrantResponse(points=[])
    try:
        match qdrant_model:
            case QdrantEmbeddingModel.UUID:
                records: list[Record] = client.retrieve(
                    collection_name=collection_name,
                    ids=[querytext],
                    with_payload=payload,
                )
                points: list[ScoredPoint] = [
                    ScoredPoint(
                        id=r.id, version=1, payload=r.payload, score=1
                    )
                    for r in records
                ]
                response = QdrantResponse(points=points)

            case QdrantEmbeddingModel.DENSE:
                vect: list[float] = encoder.embed_query(querytext)
                query_dict: dict[str, Any] = {
                    'collection_name': collection_name,
                    'query': vect,
                    'using': DENSE_VECTOR_NAME,
                    'with_payload': payload,
                    'limit': limit,
                }
                response = client.query_points(**query_dict)

            case QdrantEmbeddingModel.MULTIVECTOR:
                vect: list[float] = encoder.embed_query(querytext)
                query_dict: dict[str, Any] = {
                    'collection_name': collection_name,
                    'query': [vect, vect],
                    'using': DENSE_VECTOR_NAME,
                    'with_payload': payload,
                    'limit': limit,
                }
                response = client.query_points(**query_dict)

            case QdrantEmbeddingModel.SPARSE:
                sparse_model = _get_sparse_model(embedding_settings)
                sparse_embeddings = list(
                    sparse_model.embed(querytext)
                )
                query_dict: dict[str, Any] = {
                    'collection_name': collection_name,
                    'query': models.SparseVector(
                        indices=list(sparse_embeddings[0].indices),
                        values=list(sparse_embeddings[0].values),
                    ),
                    'using': SPARSE_VECTOR_NAME,
                    'with_payload': payload,
                    'limit': limit,
                }
                response = client.query_points(**query_dict)

            case QdrantEmbeddingModel.HYBRID_DENSE:
                try:
                    vect: list[float] = encoder.embed_query(querytext)
                    sparse_model = _get_sparse_model(
                        embedding_settings
                    )
                    sparse_embeddings = list(
                        sparse_model.embed(querytext)
                    )
                except Exception:
                    logger.error("Could not create encoding")
                    return []
                dense_dict: dict[str, Any] = {
                    'query': vect,
                    'using': DENSE_VECTOR_NAME,
                    'limit': 25,
                }
                sparse_dict: dict[str, Any] = {
                    'query': models.SparseVector(
                        indices=list(sparse_embeddings[0].indices),
                        values=list(sparse_embeddings[0].values),
                    ),
                    'using': SPARSE_VECTOR_NAME,
                    'limit': limit,
                }
                query_dict: dict[str, Any] = {
                    'collection_name': collection_name,
                    'prefetch': [
                        models.Prefetch(**dense_dict),
                        models.Prefetch(**sparse_dict),
                    ],
                    'query': models.FusionQuery(
                        fusion=models.Fusion.RRF
                    ),
                    'with_payload': payload,
                }
                response = client.query_points(**query_dict)

            case QdrantEmbeddingModel.HYBRID_MULTIVECTOR:
                try:
                    vect: list[float] = encoder.embed_query(querytext)
                    sparse_model = _get_sparse_model(
                        embedding_settings
                    )
                    sparse_embeddings = list(
                        sparse_model.embed(querytext)
                    )
                except Exception:
                    logger.error("Could not create encoding")
                    return []
                dense_dict: dict[str, Any] = {
                    'query': [vect, vect],
                    'using': DENSE_VECTOR_NAME,
                    'limit': 25,
                }
                sparse_dict: dict[str, Any] = {
                    'query': models.SparseVector(
                        indices=list(sparse_embeddings[0].indices),
                        values=list(sparse_embeddings[0].values),
                    ),
                    'using': SPARSE_VECTOR_NAME,
                    'limit': limit,
                }
                query_dict: dict[str, Any] = {
                    'collection_name': collection_name,
                    'prefetch': [
                        models.Prefetch(**dense_dict),
                        models.Prefetch(**sparse_dict),
                    ],
                    'query': models.FusionQuery(
                        fusion=models.Fusion.RRF
                    ),
                    'with_payload': payload,
                }
                response = client.query_points(**query_dict)

            case _:
                raise RuntimeError(
                    "Unreachable code reached due to invalid "
                    + "embedding model: "
                    + str(qdrant_model)
                )

    except ConnectionError:
        logger.error(
            "Could not connect to the qdrant server, which may be down."
        )
        return []
    except UnexpectedResponse as e:
        logger.error(f"Could not read from the vector database: {e}")
        return []
    except ApiException as e:
        logger.error(
            f"Could not read from the database due to API error: {e}"
        )
        return []
    except Exception as e:
        logger.error(f"Could not read from the database: {e}")
        return []

    return response.points


async def aquery(
    client: AsyncQdrantClient,
    collection_name: str,
    qdrant_model: QdrantEmbeddingModel,
    embedding_settings: EmbeddingSettings,
    querytext: str,
    *,
    limit: int = 12,
    payload: list[str] | bool = ['page_content'],
    logger: LoggerBase = default_logger,
) -> list[ScoredPoint]:
    """
    Executes a query on the client asynchronously.

    Args:
        client: a QdrantClient object
        collection_name: the collection to query
        qdrant_model: the qdrant embedding model
        embedding_settings: the embedding settings
        querytext: the target text
        limit: max number of chunks retrieved
        payload: what properties to be retrieved; defaults to the text
            retrieved for similarity to the querytext
        logger: a logger object

    Returns:
        a list of ScoredPoint objects.
    """

    # load language model
    from requests.exceptions import ConnectionError

    try:
        from lmm.language_models.langchain.runnables import (
            create_embeddings,
        )

        encoder: Embeddings = create_embeddings(embedding_settings)
    except ConnectionError:
        logger.error(
            "Could not connect to language models.\n"
            + "Check the internet connection.",
        )
        return []
    except ImportError:
        logger.error(
            "Could not create langchain kernel. Check that langchain "
            "is installed, and an internet connection is available."
        )
        return []
    except Exception as e:
        logger.error(
            "Could not initialize language model engine:\n" + str(e)
        )
        return []

    # TODO: check querytext is UUID in UUID model

    response: QdrantResponse = QdrantResponse(points=[])
    try:
        match qdrant_model:
            case QdrantEmbeddingModel.UUID:
                records: list[Record] = await client.retrieve(
                    collection_name=collection_name,
                    ids=[querytext],
                    with_payload=payload,
                )
                points: list[ScoredPoint] = [
                    ScoredPoint(
                        id=r.id, version=1, payload=r.payload, score=1
                    )
                    for r in records
                ]
                response = QdrantResponse(points=points)

            case QdrantEmbeddingModel.DENSE:
                vect: list[float] = await encoder.aembed_query(
                    querytext
                )
                query_dict: dict[str, Any] = {
                    'collection_name': collection_name,
                    'query': vect,
                    'using': DENSE_VECTOR_NAME,
                    'with_payload': payload,
                    'limit': limit,
                }
                response = await client.query_points(**query_dict)

            case QdrantEmbeddingModel.MULTIVECTOR:
                vect: list[float] = await encoder.aembed_query(
                    querytext
                )
                multi_vect: list[list[float]] = [vect, vect]
                query_dict: dict[str, Any] = {
                    'collection_name': collection_name,
                    'query': multi_vect,
                    'using': DENSE_VECTOR_NAME,
                    'with_payload': payload,
                    'limit': limit,
                }
                response = await client.query_points(**query_dict)

            case QdrantEmbeddingModel.SPARSE:
                sparse_model = _get_sparse_model(embedding_settings)
                sparse_embeddings = list(
                    sparse_model.embed(querytext)
                )
                query_dict: dict[str, Any] = {
                    'collection_name': collection_name,
                    'query': models.SparseVector(
                        indices=list(sparse_embeddings[0].indices),
                        values=list(sparse_embeddings[0].values),
                    ),
                    'using': SPARSE_VECTOR_NAME,
                    'with_payload': payload,
                    'limit': limit,
                }
                response = await client.query_points(**query_dict)

            case QdrantEmbeddingModel.HYBRID_DENSE:
                try:
                    vect: list[float] = await encoder.aembed_query(
                        querytext
                    )
                    sparse_model = _get_sparse_model(
                        embedding_settings
                    )
                    sparse_embeddings = list(
                        sparse_model.embed(querytext)
                    )
                except Exception:
                    logger.error("Could not create encoding")
                    return []
                dense_dict: dict[str, Any] = {
                    'query': vect,
                    'using': DENSE_VECTOR_NAME,
                    'limit': 25,
                }
                sparse_dict: dict[str, Any] = {
                    'query': models.SparseVector(
                        indices=list(sparse_embeddings[0].indices),
                        values=list(sparse_embeddings[0].values),
                    ),
                    'using': SPARSE_VECTOR_NAME,
                    'limit': limit,
                }
                query_dict: dict[str, Any] = {
                    'collection_name': collection_name,
                    'prefetch': [
                        models.Prefetch(**dense_dict),
                        models.Prefetch(**sparse_dict),
                    ],
                    'query': models.FusionQuery(
                        fusion=models.Fusion.RRF
                    ),
                    'with_payload': payload,
                }
                response = await client.query_points(**query_dict)

            case QdrantEmbeddingModel.HYBRID_MULTIVECTOR:
                try:
                    vect: list[float] = await encoder.aembed_query(
                        querytext
                    )
                    sparse_model = _get_sparse_model(
                        embedding_settings
                    )
                    sparse_embeddings = list(
                        sparse_model.embed(querytext)
                    )
                except Exception:
                    logger.error("Could not create encoding")
                    return []
                dense_dict: dict[str, Any] = {
                    'query': [vect, vect],
                    'using': DENSE_VECTOR_NAME,
                    'limit': 25,
                }
                sparse_dict: dict[str, Any] = {
                    'query': models.SparseVector(
                        indices=list(sparse_embeddings[0].indices),
                        values=list(sparse_embeddings[0].values),
                    ),
                    'using': SPARSE_VECTOR_NAME,
                    'limit': limit,
                }
                query_dict: dict[str, Any] = {
                    'collection_name': collection_name,
                    'prefetch': [
                        models.Prefetch(**dense_dict),
                        models.Prefetch(**sparse_dict),
                    ],
                    'query': models.FusionQuery(
                        fusion=models.Fusion.RRF
                    ),
                    'with_payload': payload,
                }
                response = await client.query_points(**query_dict)

            case _:
                raise RuntimeError(
                    "Unreachable code reached due to invalid "
                    + "embedding model: "
                    + str(qdrant_model)
                )

    except ConnectionError:
        logger.error(
            "Could not connect to the qdrant server, which may be down."
        )
        return []
    except UnexpectedResponse as e:
        logger.error(f"Could not read from the vector database: {e}")
        return []
    except ApiException as e:
        logger.error(
            f"Could not read from the database due to API error: {e}"
        )
        return []
    except Exception as e:
        logger.error(f"Could not read from the database: {e}")
        return []

    return response.points


def query_grouped(
    client: QdrantClient,
    collection_name: str,
    group_collection: str,
    qdrant_model: QdrantEmbeddingModel,
    embedding_settings: EmbeddingSettings,
    querytext: str,
    *,
    limit: int = 4,
    payload: list[str] | bool = True,
    group_size: int = 1,
    group_field: str = GROUP_UUID_KEY,
    logger: LoggerBase = default_logger,
) -> GroupsResult:
    """
    Executes a grouped query on the client.

    Args:
        client: a QdrantClient object
        collection_name: the collection to query
        group_collection: the companion collection that will provide
            the output of the query
        qdrant_model: the qdrant embedding model
        embedding_settings: the embedding settings
        querytext: the target text
        limit: max number of chunks retrieved
        payload: what properties to be retrieved; defaults to all
            metadata propoerties. Note that the text is 'page_content',
            and this will be included unless payload = False.
            If payload = True, all propertes are included.
        group_size: max retrieved output from the group collection
        group_field: the filed to group on
        logger: a logger object

    Returns:
        a list of ScoredPoint objects.
    """

    # make sure if payload is given, it contains `page_content`
    if isinstance(payload, list):
        if 'page_content' not in payload:
            payload.append('page_content')

    # Essentially, the qdrant API declares different
    # types for any search API.
    NullResult: GroupsResult = GroupsResult(groups=[])
    # load language model
    from requests.exceptions import ConnectionError

    try:
        from lmm.language_models.langchain.runnables import (
            create_embeddings,
        )

        encoder: Embeddings = create_embeddings(embedding_settings)
    except ConnectionError:
        logger.error(
            "Could not connect to language models.\n"
            + "Check the internet connection.",
        )
        return NullResult
    except ImportError:
        logger.error(
            "Could not create langchain kernel. Check that langchain "
            "is installed, and an internet connection is available."
        )
        return NullResult
    except Exception as e:
        logger.error(
            "Could not initialize language model engine:\n" + str(e)
        )
        return NullResult

    # TODO: check querytext is UUID in UUID model

    response: GroupsResult = NullResult
    try:
        match qdrant_model:
            case QdrantEmbeddingModel.UUID:
                # ignore grouping
                hits: list[ScoredPoint] = query(
                    client,
                    collection_name,
                    qdrant_model,
                    embedding_settings,
                    querytext,
                    limit=limit,
                    payload=payload,
                    logger=logger,
                )
                return GroupsResult(
                    groups=[PointGroup(hits=hits, id=querytext)]
                )

            case QdrantEmbeddingModel.DENSE:
                vect = encoder.embed_query(querytext)
                query_dict: dict[str, Any] = {
                    'collection_name': collection_name,
                    'query': vect,
                    'using': DENSE_VECTOR_NAME,
                    'group_by': group_field,
                    'limit': limit,
                    'group_size': group_size,
                    'with_lookup': models.WithLookup(
                        collection=group_collection,
                        with_payload=payload,
                        with_vectors=False,
                    ),
                }
                response = client.query_points_groups(**query_dict)

            case QdrantEmbeddingModel.MULTIVECTOR:
                vect = encoder.embed_query(querytext)
                query_dict: dict[str, Any] = {
                    'collection_name': collection_name,
                    'query': [vect, vect],
                    'using': DENSE_VECTOR_NAME,
                    'group_by': group_field,
                    'limit': limit,
                    'group_size': group_size,
                    'with_lookup': models.WithLookup(
                        collection=group_collection,
                        with_payload=payload,
                        with_vectors=False,
                    ),
                }
                response = client.query_points_groups(**query_dict)

            case QdrantEmbeddingModel.SPARSE:
                sparse_model: SparseTextEmbedding = _get_sparse_model(
                    embedding_settings
                )
                sparse_embeddings: list[SparseEmbedding] = list(
                    sparse_model.embed(querytext)
                )
                query_dict: dict[str, Any] = {
                    'collection_name': collection_name,
                    'query': models.SparseVector(
                        indices=list(sparse_embeddings[0].indices),
                        values=list(sparse_embeddings[0].values),
                    ),
                    'using': SPARSE_VECTOR_NAME,
                    'group_by': group_field,
                    'limit': limit,
                    'group_size': group_size,
                    'with_lookup': models.WithLookup(
                        collection=group_collection,
                        with_payload=payload,
                        with_vectors=False,
                    ),
                }
                response = client.query_points_groups(**query_dict)

            case QdrantEmbeddingModel.HYBRID_DENSE:
                try:
                    vect: list[float] = encoder.embed_query(querytext)
                    sparse_model: SparseTextEmbedding = (
                        _get_sparse_model(embedding_settings)
                    )
                    sparse_embeddings: list[SparseEmbedding] = list(
                        sparse_model.embed(querytext)
                    )
                except Exception:
                    logger.error("Could not create encoding")
                    return NullResult
                dense_dict: dict[str, Any] = {
                    'query': vect,
                    'using': DENSE_VECTOR_NAME,
                    'limit': 25,
                }
                sparse_dict: dict[str, Any] = {
                    'query': models.SparseVector(
                        indices=list(sparse_embeddings[0].indices),
                        values=list(sparse_embeddings[0].values),
                    ),
                    'using': SPARSE_VECTOR_NAME,
                    'limit': limit,
                }
                query_dict: dict[str, Any] = {
                    'collection_name': collection_name,
                    'prefetch': [
                        models.Prefetch(**dense_dict),
                        models.Prefetch(**sparse_dict),
                    ],
                    'query': models.FusionQuery(
                        fusion=models.Fusion.RRF
                    ),
                    'group_by': group_field,
                    'group_size': group_size,
                    'with_lookup': models.WithLookup(
                        collection=group_collection,
                        with_payload=payload,
                        with_vectors=False,
                    ),
                }
                response = client.query_points_groups(**query_dict)

            case QdrantEmbeddingModel.HYBRID_MULTIVECTOR:
                try:
                    vect: list[float] = encoder.embed_query(querytext)
                    sparse_model: SparseTextEmbedding = (
                        _get_sparse_model(embedding_settings)
                    )
                    sparse_embeddings: list[SparseEmbedding] = list(
                        sparse_model.embed(querytext)
                    )
                except Exception:
                    logger.error("Could not create encoding")
                    return NullResult
                dense_dict: dict[str, Any] = {
                    'query': [vect, vect],
                    'using': DENSE_VECTOR_NAME,
                    'limit': 25,
                }
                sparse_dict: dict[str, Any] = {
                    'query': models.SparseVector(
                        indices=list(sparse_embeddings[0].indices),
                        values=list(sparse_embeddings[0].values),
                    ),
                    'using': SPARSE_VECTOR_NAME,
                    'limit': limit,
                }
                query_dict: dict[str, Any] = {
                    'collection_name': collection_name,
                    'prefetch': [
                        models.Prefetch(**dense_dict),
                        models.Prefetch(**sparse_dict),
                    ],
                    'query': models.FusionQuery(
                        fusion=models.Fusion.RRF
                    ),
                    'group_by': group_field,
                    'group_size': group_size,
                    'with_lookup': models.WithLookup(
                        collection=group_collection,
                        with_payload=payload,
                        with_vectors=False,
                    ),
                }
                response = client.query_points_groups(**query_dict)

            case _:
                raise RuntimeError(
                    "Unreachable code reached due to invalid "
                    + "embedding model: "
                    + str(qdrant_model)
                )

    except ConnectionError:
        logger.error(
            "Could not connect to the qdrant server, which may be down."
        )
        return NullResult
    except UnexpectedResponse as e:
        logger.error(f"Could not read from the vector database: {e}")
        return NullResult
    except ApiException as e:
        logger.error(
            f"Could not read from the database due to API error: {e}"
        )
        return NullResult
    except Exception as e:
        logger.error(f"Could not read from the database: {e}")
        return NullResult

    return response


async def aquery_grouped(
    client: AsyncQdrantClient,
    collection_name: str,
    group_collection: str,
    qdrant_model: QdrantEmbeddingModel,
    embedding_settings: EmbeddingSettings,
    querytext: str,
    *,
    limit: int = 4,
    payload: list[str] | bool = True,
    group_size: int = 1,
    group_field: str = GROUP_UUID_KEY,
    logger: LoggerBase = default_logger,
) -> GroupsResult:
    """
    Executes a grouped query on the client asynchronously.

    Args:
        client: a QdrantClient object
        collection_name: the collection to query
        group_collection: the companion collection that will provide
            the output of the query
        qdrant_model: the qdrant embedding model
        embedding_settings: the embedding settings
        group_field: the filed to group on
        querytext: the target text
        limit: max number of chunks retrieved
        payload: what properties to be retrieved; defaults to all
            metadata properties. Note that text is 'page_content', and
            this will be included unless payload = False
            If payload = True, all propertes are included.
        group_size: max retrieved output from the group collection
        group_field: the filed to group on
        logger: a logger object

    Returns:
        a list of ScoredPoint objects.
    """

    # make sure if payload is given, it contains `page_content`
    if isinstance(payload, list):
        if 'page_content' not in payload:
            payload.append('page_content')

    # Essentially, the qdrant API declares different
    # types for any search API.
    NullResult: GroupsResult = GroupsResult(groups=[])
    # load language model
    from requests.exceptions import ConnectionError

    try:
        from lmm.language_models.langchain.runnables import (
            create_embeddings,
        )

        encoder: Embeddings = create_embeddings(embedding_settings)
    except ConnectionError:
        logger.error(
            "Could not connect to language models.\n"
            + "Check the internet connection.",
        )
        return NullResult
    except ImportError:
        logger.error(
            "Could not create langchain kernel. Check that langchain "
            "is installed, and an internet connection is available."
        )
        return NullResult
    except Exception as e:
        logger.error(
            "Could not initialize language model engine:\n" + str(e)
        )
        return NullResult

    # TODO: check querytext is UUID in UUID model

    response: GroupsResult = NullResult
    try:
        match qdrant_model:
            case QdrantEmbeddingModel.UUID:
                # ignore grouping
                hits: list[ScoredPoint] = await aquery(
                    client,
                    collection_name,
                    qdrant_model,
                    embedding_settings,
                    querytext,
                    limit=limit,
                    payload=payload,
                    logger=logger,
                )
                return GroupsResult(
                    groups=[PointGroup(hits=hits, id=querytext)]
                )

            case QdrantEmbeddingModel.DENSE:
                vect = await encoder.aembed_query(querytext)
                query_dict: dict[str, Any] = {
                    'collection_name': collection_name,
                    'query': vect,
                    'using': DENSE_VECTOR_NAME,
                    'group_by': group_field,
                    'limit': limit,
                    'group_size': group_size,
                    'with_lookup': models.WithLookup(
                        collection=group_collection,
                        with_payload=payload,
                        with_vectors=False,
                    ),
                }
                response = await client.query_points_groups(
                    **query_dict
                )

            case QdrantEmbeddingModel.MULTIVECTOR:
                vect = await encoder.aembed_query(querytext)
                query_dict: dict[str, Any] = {
                    'collection_name': collection_name,
                    'query': [vect, vect],
                    'using': DENSE_VECTOR_NAME,
                    'group_by': group_field,
                    'limit': limit,
                    'group_size': group_size,
                    'with_lookup': models.WithLookup(
                        collection=group_collection,
                        with_payload=payload,
                        with_vectors=False,
                    ),
                }
                response = await client.query_points_groups(
                    **query_dict
                )

            case QdrantEmbeddingModel.SPARSE:
                sparse_model: SparseTextEmbedding = _get_sparse_model(
                    embedding_settings
                )
                sparse_embeddings: list[SparseEmbedding] = list(
                    sparse_model.embed(querytext)
                )
                query_dict: dict[str, Any] = {
                    'collection_name': collection_name,
                    'query': models.SparseVector(
                        indices=list(sparse_embeddings[0].indices),
                        values=list(sparse_embeddings[0].values),
                    ),
                    'using': SPARSE_VECTOR_NAME,
                    'group_by': group_field,
                    'limit': limit,
                    'group_size': group_size,
                    'with_lookup': models.WithLookup(
                        collection=group_collection,
                        with_payload=payload,
                        with_vectors=False,
                    ),
                }
                response = await client.query_points_groups(
                    **query_dict
                )

            case QdrantEmbeddingModel.HYBRID_DENSE:
                try:
                    vect: list[float] = await encoder.aembed_query(
                        querytext
                    )
                    sparse_model: SparseTextEmbedding = (
                        _get_sparse_model(embedding_settings)
                    )
                    sparse_embeddings: list[SparseEmbedding] = list(
                        sparse_model.embed(querytext)
                    )
                except Exception:
                    logger.error("Could not create encoding")
                    return NullResult
                dense_dict: dict[str, Any] = {
                    'query': vect,
                    'using': DENSE_VECTOR_NAME,
                    'limit': 25,
                }
                sparse_dict: dict[str, Any] = {
                    'query': models.SparseVector(
                        indices=list(sparse_embeddings[0].indices),
                        values=list(sparse_embeddings[0].values),
                    ),
                    'using': SPARSE_VECTOR_NAME,
                    'limit': limit,
                }
                query_dict: dict[str, Any] = {
                    'collection_name': collection_name,
                    'prefetch': [
                        models.Prefetch(**dense_dict),
                        models.Prefetch(**sparse_dict),
                    ],
                    'query': models.FusionQuery(
                        fusion=models.Fusion.RRF
                    ),
                    'group_by': group_field,
                    'group_size': group_size,
                    'with_lookup': models.WithLookup(
                        collection=group_collection,
                        with_payload=payload,
                        with_vectors=False,
                    ),
                }
                response = await client.query_points_groups(
                    **query_dict
                )

            case QdrantEmbeddingModel.HYBRID_MULTIVECTOR:
                try:
                    vect: list[float] = await encoder.aembed_query(
                        querytext
                    )
                    sparse_model: SparseTextEmbedding = (
                        _get_sparse_model(embedding_settings)
                    )
                    sparse_embeddings: list[SparseEmbedding] = list(
                        sparse_model.embed(querytext)
                    )
                except Exception:
                    logger.error("Could not create encoding")
                    return NullResult
                dense_dict: dict[str, Any] = {
                    'query': [vect, vect],
                    'using': DENSE_VECTOR_NAME,
                    'limit': 25,
                }
                sparse_dict: dict[str, Any] = {
                    'query': models.SparseVector(
                        indices=list(sparse_embeddings[0].indices),
                        values=list(sparse_embeddings[0].values),
                    ),
                    'using': SPARSE_VECTOR_NAME,
                    'limit': limit,
                }
                query_dict: dict[str, Any] = {
                    'collection_name': collection_name,
                    'prefetch': [
                        models.Prefetch(**dense_dict),
                        models.Prefetch(**sparse_dict),
                    ],
                    'query': models.FusionQuery(
                        fusion=models.Fusion.RRF
                    ),
                    'group_by': group_field,
                    'group_size': group_size,
                    'with_lookup': models.WithLookup(
                        collection=group_collection,
                        with_payload=payload,
                        with_vectors=False,
                    ),
                }
                response = await client.query_points_groups(
                    **query_dict
                )

            case _:
                raise RuntimeError(
                    "Unreachable code reached due to invalid "
                    + "embedding model: "
                    + str(qdrant_model)
                )

    except ConnectionError:
        logger.error(
            "Could not connect to the qdrant server, which may be down."
        )
        return NullResult
    except UnexpectedResponse as e:
        logger.error(f"Could not read from the vector database: {e}")
        return NullResult
    except ApiException as e:
        logger.error(
            f"Could not read from the database due to API error: {e}"
        )
        return NullResult
    except Exception as e:
        logger.error(f"Could not read from the database: {e}")
        return NullResult

    return response


def groups_to_points(groups: GroupsResult) -> list[ScoredPoint]:
    """transform a GroupsResult object into a list of ScoredPoint
    objects of the group lookup list"""
    # assumes groups is structured as it should
    records: list[Record] = [
        g.lookup for g in groups.groups if g.lookup
    ]
    _max: Callable[[list[ScoredPoint]], float] = lambda hits: max(
        [h.score for h in hits]
    )  # noqa: E731
    scores: list[float] = [
        _max(g.hits) for g in groups.groups if g.lookup
    ]
    points: list[ScoredPoint] = [
        ScoredPoint(id=r.id, version=1, payload=r.payload, score=s)
        for r, s in zip(records, scores)
    ]
    return points


def points_to_ids(
    points: list[Point] | list[ScoredPoint],
) -> list[str]:
    """transform a list of points into a list of their id's"""
    return [str(p.id) for p in points]


def points_to_text(
    points: list[Point] | list[ScoredPoint],
) -> list[str]:
    """transform a list of points into a list of their textual
    content"""
    return [
        str(p.payload['page_content'])
        for p in points
        if p.payload and 'page_content' in p.payload
    ]


def points_to_payload(
    points: list[Point] | list[ScoredPoint],
    payload_key: str | None = None,
) -> list[dict[str, Any]]:
    """transform a list of points into a list of their payload"""
    if payload_key is None:
        return [p.payload for p in points if p.payload]
    else:
        return [
            p.payload.get(payload_key, "")
            for p in points
            if p.payload
        ]


def points_to_metadata(
    points: list[Point] | list[ScoredPoint],
    metadata_key: str | None = None,
) -> list[Any]:
    metas = points_to_payload(points, 'metadata')
    if metadata_key is None:
        return metas
    else:
        return [m.get(metadata_key, "") for m in metas]


def points_to_blocks(points: list[Point]) -> list[Block]:
    """transform a list of ingestion points into a block list (to
    visualize what one ingested)"""
    blocks: list[Block] = []
    for p in points:
        blocks.append(
            MetadataBlock(
                content=(
                    p.payload['metadata']
                    if p.payload and 'metadata' in p.payload
                    else {}
                )
            )
        )
        blocks.append(
            TextBlock(
                content=(
                    p.payload['page_content']
                    if p.payload and 'page_content' in p.payload
                    else ""
                )
            )
        )
    return blocks
