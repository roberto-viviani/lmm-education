"""
Computes the embeddings and handles uploading and saving the data
to the vector database.

Provides the interface to the qdrant database. A connection to the
database is representend by a `QdrantClient` object of the Qdrant API,
which may be initialized directly through the constructor, or through
the `client_from_config` function which reads the database options
from config.toml:

```python
    from lmm_education.stores.vector_store_qdrant import client_from_config

    client: QdrantCient | None = client_from_config()
    # check client is not None
```

It is also possible to intialize a config object explicitly for the
properties that override config.toml.

```python
    from lmm_education.stores.vector_store_qdrant import client_from_config
    from lmm_education.config.config import ConfigSettings

    # read settings from config.toml, but override 'storage':
    settings = ConfigSettings(storage=":memory:")
    client: QdrantCient | None = client_from_config(settings)
```

The functions in this module use a logger to communicate errors, so
that the way exceptions are handled depends on the logger type. If no
lgger is specified, an error message is printed on the console.

```python
    from lmm_education.stores.vector_store_qdrant import client_from_config
    from lmm.utils.logging import LogfileLogger

    logger = LogfileLogger()
    client: QdrantCient | None = client_from_config(None, logger)
    if client is None:
        # read causes from logger
```

The remaining functions of the module take the client object to read
and write to the database. All calls go through initialize_connection,
which takes the name of the collection and an embedding model to
specify how the data should be embedded (what type of dense and sparse
vector, or any hybrid combination of those, should be used):

```python
    # ... client creation not shown
    from lmm_education.stores.vector_store_qdrant import (
        initialize_connection,
        QdrantEmbeddingModel,
    )

    embedding_model = QdrantEmbeddingModel
    flag: bool = initialize_connection(
        client,
        "documents",
        embedding_model,
        logger)
```

In every call to the functions of the model, the client, the
collection name, and the embedding model are given as arguments.

Data are ingested in the database in the form of lists of `Chunk`
objects (see the `lmm_education.stores.chunks` module).

```python
points: list[Point] = upload(
    client,
    "documents",
    embedding_model,
    chunks,
    logger,
)
```

The `point` objects are the representations of records used by Qdrant.
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
    "What are the main uses of logistic regression?",
    limit = 12,  # max number retrieved points
    payload = True,  # all payload fields
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
    initialize_connection / ainitialize_connection
    upload / aupload
    query / aquery
    query_grouped / aquery_grouped
"""

from enum import Enum
from typing import Callable, Any

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

# lmmarkdown
from lmm.scan.scan_keys import GROUP_UUID_KEY
from lmm.config.config import (
    Settings as LmmSettings,
    EmbeddingSettings,
)
from lmm_education.stores.chunks import EncodingModel, Chunk
from lmm.markdown.parse_markdown import (
    Block,
    TextBlock,
    MetadataBlock,
)

# lmm markdown for education
from lmm_education.config.config import ConfigSettings

# Set up logger
from lmm.utils.logging import LoggerBase, get_logger

default_logger: LoggerBase = get_logger(__name__)


from fastembed import SparseEmbedding, SparseTextEmbedding


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


def _get_sparse_model() -> SparseTextEmbedding:
    """Get memoized SparseTextEmbedding model instance."""
    global _sparse_model_cache
    if _sparse_model_cache is None:
        esets: EmbeddingSettings = LmmSettings().embeddings
        _sparse_model_cache = SparseTextEmbedding(
            model_name=str(esets.sparse_model)
        )
    return _sparse_model_cache


def client_from_config(
    opts: ConfigSettings | None = None,
    logger: LoggerBase = default_logger,
) -> QdrantClient | None:
    """ "
    Create a qdrant clients from config settings. Reads from config
    toml file settings if none gievn.

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

    try:
        if opts is None:
            opts = ConfigSettings()
        client: QdrantClient
        match opts.storage:
            case ':memory:':
                client = QdrantClient(':memory:')
            case LocalStorage(folder=folder):
                client = QdrantClient(path=folder)
            case RemoteSource(url=url, port=port):
                client = QdrantClient(url=str(url), port=port)
            case _:
                logger.error("Invalid database source")
                return None
    except Exception as e:
        logger.error(f"Could not initialize qdrant client:\n{e}")
        return None

    return client


def initialize_collection(
    client: QdrantClient,
    collection_name: str,
    embedding_model: QdrantEmbeddingModel,
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
        embedding_model: the embedding model

    Returns:
        a boolean flag indicating that the client may be used with
            these parameters.
    """

    from requests.exceptions import ConnectionError

    try:
        from lmm.language_models.langchain.runnables import (
            create_embeddings,
        )

        encoder: Embeddings = create_embeddings()
    except ConnectionError:
        logger.error(
            "Could not connect to the language model.\n"
            + "Check the internet connection."
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
            # TODO: add checks that the opts and the collection
            # are compatible
            return True

        # determine embedding size
        if not (
            embedding_model == QdrantEmbeddingModel.UUID
            or embedding_model == QdrantEmbeddingModel.SPARSE
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

        match embedding_model:
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
                    + str(embedding_model)
                )
    except Exception:
        logger.error("Could not initialize vector database")
        return False

    return True


def async_client_from_config(
    opts: ConfigSettings | None,
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

    try:
        if opts is None:
            opts = ConfigSettings()
        client: AsyncQdrantClient
        match opts.storage:
            case ':memory:':
                client = AsyncQdrantClient(':memory:')
            case LocalStorage(folder=folder):
                client = AsyncQdrantClient(path=folder)
            case RemoteSource(url=url, port=port):
                client = AsyncQdrantClient(url=str(url), port=port)
            case _:
                raise ValueError("Invalid database source")
    except Exception as e:
        logger.error(f"Could not initialize qdrant client:\n{e}")
        return None

    return client


async def ainitialize_collection(
    client: AsyncQdrantClient,
    collection_name: str,
    embedding_model: QdrantEmbeddingModel,
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
        embedding_model: the embedding model

    Returns:
        a boolean
    """

    from requests.exceptions import ConnectionError
    from lmm.language_models.langchain.runnables import (
        create_embeddings,
    )

    try:
        encoder: Embeddings = create_embeddings()
    except ConnectionError:
        logger.error(
            "Could not connect to language models.\n"
            + "Check the internet connection."
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
            # TODO: add checks that the opts and the collection
            # are compatible
            return True

        # determine embedding size
        if not (
            embedding_model == QdrantEmbeddingModel.UUID
            or embedding_model == QdrantEmbeddingModel.SPARSE
        ):
            data: list[float] = encoder.embed_query("Test query")
            embedding_size: int = len(data)
        else:
            embedding_size: int = 0

        match embedding_model:
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
                    + str(embedding_model)
                )
    except Exception:
        logger.error("Could not initialize vector database")
        return False

    return True


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
    # the content for saveing from the Chunk object into the
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


def chunks_to_points(
    chunks: list[Chunk],
    model: QdrantEmbeddingModel,
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
        model: the embedding model

    Returns:
        a list of Point objects (PointStruct)
    """

    if not chunks:
        return []

    # load embedding model
    from requests.exceptions import ConnectionError
    from lmm.language_models.langchain.runnables import (
        create_embeddings,
    )

    try:
        encoder: Embeddings = create_embeddings()
    except ConnectionError:
        logger.error(
            "Could not connect to language models.\n"
            + "Check the internet connection.",
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
    match model:
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
                    vector={DENSE_VECTOR_NAME: v},
                    payload=chunk_to_payload(d),
                )
                for d, v in zip(chunks, vect)
            ]

        case QdrantEmbeddingModel.SPARSE:
            try:
                sparse_model = _get_sparse_model()
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
                sparse_model = _get_sparse_model()
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
                sparse_model = _get_sparse_model()
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
        case _:
            raise RuntimeError(
                "Unreachable code reached due to invalid "
                + "embedding model: "
                + str(model)
            )

    return points


def upload(
    client: QdrantClient,
    collection_name: str,
    model: QdrantEmbeddingModel,
    chunks: list[Chunk],
    *,
    logger: LoggerBase = default_logger,
) -> list[Point]:
    """
    Upload a list of chunks into the Qdrant database

    Args:
        client: the connection to the database
        collection_name: the collection
        model: the embedding moel
        chunks: the chunk list

    Returns:
        a list of Point objects
    """

    if not initialize_collection(
        client, collection_name, model, logger=logger
    ):
        logger.error("Could not initialize collection.")
        return []

    points: list[Point] = chunks_to_points(
        chunks, model, logger=logger
    )
    if not points:
        return []

    try:
        client.upsert(collection_name=collection_name, points=points)
    except Exception as e:
        logger.error(f"Could not upload chunks to database:\n{e}")
        return []

    return points


async def aupload(
    client: AsyncQdrantClient,
    collection_name: str,
    model: QdrantEmbeddingModel,
    chunks: list[Chunk],
    *,
    logger: LoggerBase = default_logger,
) -> list[Point]:
    """
    Upload a list of chunks into the Qdrant database

    Args:
        client: the connection to the database
        collection_name: the collection
        model: the embedding moel
        chunks: the chunk list

    Returns:
        a list of Point objects
    """

    if not await ainitialize_collection(
        client, collection_name, model, logger=logger
    ):
        logger.error("Could not initialize collection")
        return []

    points: list[Point] = chunks_to_points(
        chunks, model, logger=logger
    )
    if not points:
        return []

    try:
        await client.upsert(
            collection_name=collection_name, points=points
        )
    except Exception:
        logger.error("Could not upload chunks to database")
        return []

    return points


def query(
    client: QdrantClient,
    collection_name: str,
    model: QdrantEmbeddingModel,
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
        model: the embedding model
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
    from lmm.language_models.langchain.runnables import (
        create_embeddings,
    )

    try:
        encoder: Embeddings = create_embeddings()
    except ConnectionError:
        logger.error(
            "Could not connect to language models.\n"
            + "Check the internet connection.",
        )
        return []
    except Exception as e:
        logger.error(
            "Could not initialize language model engine:\n" + str(e)
        )
        return []

    # TODO: check querytext is UUID in UUID model

    response: QdrantResponse = QdrantResponse(points=[])
    match model:
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

        case (
            QdrantEmbeddingModel.DENSE
            | QdrantEmbeddingModel.MULTIVECTOR
        ):
            vect = encoder.embed_query(querytext)
            query_dict: dict[str, Any] = {
                'collection_name': collection_name,
                'query': vect,
                'using': DENSE_VECTOR_NAME,
                'with_payload': payload,
                'limit': limit,
            }
            response = client.query_points(**query_dict)

        case QdrantEmbeddingModel.SPARSE:
            sparse_model = _get_sparse_model()
            sparse_embeddings = list(sparse_model.embed(querytext))
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

        case (
            QdrantEmbeddingModel.HYBRID_DENSE
            | QdrantEmbeddingModel.HYBRID_MULTIVECTOR
        ):
            try:
                vect: list[float] = encoder.embed_query(querytext)
                sparse_model = _get_sparse_model()
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
                'query': models.FusionQuery(fusion=models.Fusion.RRF),
                'with_payload': payload,
            }
            response = client.query_points(**query_dict)
        case _:
            raise RuntimeError(
                "Unreachable code reached due to invalid "
                + "embedding model: "
                + str(model)
            )

    return response.points


async def aquery(
    client: AsyncQdrantClient,
    collection_name: str,
    model: QdrantEmbeddingModel,
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
        model: the embedding model
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
    from lmm.language_models.langchain.runnables import (
        create_embeddings,
    )

    try:
        encoder: Embeddings = create_embeddings()
    except ConnectionError:
        logger.error(
            "Could not connect to language models.\n"
            + "Check the internet connection.",
        )
        return []
    except Exception as e:
        logger.error(
            "Could not initialize language model engine:\n" + str(e)
        )
        return []

    # TODO: check querytext is UUID in UUID model

    response: QdrantResponse = QdrantResponse(points=[])
    match model:
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

        case (
            QdrantEmbeddingModel.DENSE
            | QdrantEmbeddingModel.MULTIVECTOR
        ):
            vect = encoder.embed_query(querytext)
            query_dict: dict[str, Any] = {
                'collection_name': collection_name,
                'query': vect,
                'using': DENSE_VECTOR_NAME,
                'with_payload': payload,
                'limit': limit,
            }
            response = await client.query_points(**query_dict)

        case QdrantEmbeddingModel.SPARSE:
            sparse_model = _get_sparse_model()
            sparse_embeddings = list(sparse_model.embed(querytext))
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

        case (
            QdrantEmbeddingModel.HYBRID_DENSE
            | QdrantEmbeddingModel.HYBRID_MULTIVECTOR
        ):
            try:
                vect: list[float] = encoder.embed_query(querytext)
                sparse_model = _get_sparse_model()
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
                'query': models.FusionQuery(fusion=models.Fusion.RRF),
                'with_payload': payload,
            }
            response = await client.query_points(**query_dict)
        case _:
            raise RuntimeError(
                "Unreachable code reached due to invalid "
                + "embedding model: "
                + str(model)
            )

    return response.points


def query_grouped(
    client: QdrantClient,
    collection_name: str,
    group_collection: str,
    model: QdrantEmbeddingModel,
    querytext: str,
    *,
    limit: int = 4,
    payload: list[str] = ['page_content'],
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
        group_field: the filed to group on
        limitgroups: max retrieved output from the group collection
        model: the embedding model
        querytext: the target text
        limit: max number of chunks retrieved
        payload: what properties to be retrieved; defaults to the text
            retrieved for similarity to the querytext
        logger: a logger object

    Returns:
        a list of ScoredPoint objects.
    """

    # Essentially, the qdrant API declares different
    # types for any search API.
    NullResult: GroupsResult = GroupsResult(groups=[])
    # load language model
    from requests.exceptions import ConnectionError
    from lmm.language_models.langchain.runnables import (
        create_embeddings,
    )

    try:
        encoder: Embeddings = create_embeddings()
    except ConnectionError:
        logger.error(
            "Could not connect to language models.\n"
            + "Check the internet connection.",
        )
        return NullResult
    except Exception as e:
        logger.error(
            "Could not initialize language model engine:\n" + str(e)
        )
        return NullResult

    # TODO: check querytext is UUID in UUID model

    response: GroupsResult = NullResult
    match model:
        case QdrantEmbeddingModel.UUID:
            # ignore grouping
            hits: list[ScoredPoint] = query(
                client,
                collection_name,
                model,
                querytext,
                limit=limit,
                payload=payload,
                logger=logger,
            )
            return GroupsResult(
                groups=[PointGroup(hits=hits, id=querytext)]
            )

        case (
            QdrantEmbeddingModel.DENSE
            | QdrantEmbeddingModel.MULTIVECTOR
        ):
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

        case QdrantEmbeddingModel.SPARSE:
            sparse_model: SparseTextEmbedding = _get_sparse_model()
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

        case (
            QdrantEmbeddingModel.HYBRID_DENSE
            | QdrantEmbeddingModel.HYBRID_MULTIVECTOR
        ):
            try:
                vect: list[float] = encoder.embed_query(querytext)
                sparse_model: SparseTextEmbedding = (
                    _get_sparse_model()
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
                'query': models.FusionQuery(fusion=models.Fusion.RRF),
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
                + str(model)
            )

    return response


async def aquery_grouped(
    client: AsyncQdrantClient,
    collection_name: str,
    group_collection: str,
    model: QdrantEmbeddingModel,
    querytext: str,
    *,
    limit: int = 4,
    payload: list[str] = ['page_content'],
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
        group_field: the filed to group on
        limitgroups: max retrieved output from the group collection
        model: the embedding model
        querytext: the target text
        limit: max number of chunks retrieved
        payload: what properties to be retrieved; defaults to the text
            retrieved for similarity to the querytext
        logger: a logger object

    Returns:
        a list of ScoredPoint objects.
    """

    # Essentially, the qdrant API declares different
    # types for any search API.
    NullResult: GroupsResult = GroupsResult(groups=[])
    # load language model
    from requests.exceptions import ConnectionError
    from lmm.language_models.langchain.runnables import (
        create_embeddings,
    )

    try:
        encoder: Embeddings = create_embeddings()
    except ConnectionError:
        logger.error(
            "Could not connect to language models.\n"
            + "Check the internet connection.",
        )
        return NullResult
    except Exception as e:
        logger.error(
            "Could not initialize language model engine:\n" + str(e)
        )
        return NullResult

    # TODO: check querytext is UUID in UUID model

    response: GroupsResult = NullResult
    match model:
        case QdrantEmbeddingModel.UUID:
            # ignore grouping
            hits: list[ScoredPoint] = await aquery(
                client,
                collection_name,
                model,
                querytext,
                limit=limit,
                payload=payload,
                logger=logger,
            )
            return GroupsResult(
                groups=[PointGroup(hits=hits, id=querytext)]
            )

        case (
            QdrantEmbeddingModel.DENSE
            | QdrantEmbeddingModel.MULTIVECTOR
        ):
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
            response = await client.query_points_groups(**query_dict)

        case QdrantEmbeddingModel.SPARSE:
            sparse_model: SparseTextEmbedding = _get_sparse_model()
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
            response = await client.query_points_groups(**query_dict)

        case (
            QdrantEmbeddingModel.HYBRID_DENSE
            | QdrantEmbeddingModel.HYBRID_MULTIVECTOR
        ):
            try:
                vect: list[float] = encoder.embed_query(querytext)
                sparse_model: SparseTextEmbedding = (
                    _get_sparse_model()
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
                'query': models.FusionQuery(fusion=models.Fusion.RRF),
                'group_by': group_field,
                'group_size': group_size,
                'with_lookup': models.WithLookup(
                    collection=group_collection,
                    with_payload=payload,
                    with_vectors=False,
                ),
            }
            response = await client.query_points_groups(**query_dict)
        case _:
            raise RuntimeError(
                "Unreachable code reached due to invalid "
                + "embedding model: "
                + str(model)
            )

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
    )
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
) -> list[dict[str, Any]]:
    """transform a list of points into a list of their textual
    content"""
    return [p.payload for p in points if p.payload]


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
