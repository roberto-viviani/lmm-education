"""
This module allows retrieval of material from the vector database.

Since retrieval is performed by the RAG code automatically (for
example, in appChat), this module is meant to be used for test
purposes.

A database must have been created (for example, with the ingest
module).

Examples:

```python
# from the python REPL

python -m lmm_education.querydb 'What is logistic regression?'
```

```python
# from python code
from lmm_education.querydb import querydb

response = querydb('what is logistic regression?')
print(response)
```

Because ingest replaces the content of the database when documents
are edited, you can set up an ingest-evaluate loop:

```python
# from the python REPL

# append True to ingest the file 'RaggedDocument.md'
python -m lmm_education.ingest RaggedDocument.md True
python -m lmm_education.querydb 'what is logistic regression?'
"""

from pydantic import validate_call
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint
from lmm.utils.logging import LoggerBase, ConsoleLogger
from lmm_education.stores.vector_store_qdrant import (
    client_from_config,
    query,
    query_grouped,
    encoding_to_qdrantembedding_model,
    groups_to_points,
    points_to_text,
)
from lmm_education.config.config import load_settings


@validate_call(config={'arbitrary_types_allowed': True})
def querydb(
    query_text: str,
    *,
    client: QdrantClient | None = None,
    logger: LoggerBase = ConsoleLogger(),
) -> str:
    """
    Execute a query on the database using the settings in
    config.toml.

    Args:
        query_text: a text to use as query.

    Returns:
        a string concatenating the results of the query.
    """
    if not query_text:
        logger.info("No query text provided")
        return ""

    if len(query_text.split()) < 3:
        logger.info("Invalid query? " + query_text)
        return ""

    settings = load_settings()
    if not settings:
        return ""

    create_flag: bool = False
    if client is None:
        client = client_from_config(settings)
        if client is None:
            logger.error("Failed to initialize qdrant client")
            return ""
        else:
            create_flag = True

    points: list[ScoredPoint] = []
    if settings.database.companion_collection:
        results = query_grouped(
            client,
            settings.database.collection_name,
            settings.database.companion_collection,
            encoding_to_qdrantembedding_model(
                settings.RAG.encoding_model
            ),
            settings.embeddings,
            query_text,
            logger=logger,
        )
        points = groups_to_points(results)
    else:
        points = query(
            client,
            settings.database.collection_name,
            encoding_to_qdrantembedding_model(
                settings.RAG.encoding_model
            ),
            settings.embeddings,
            query_text,
            logger=logger,
        )
    if create_flag:
        client.close()

    if not points:
        return "No results, please check connection/database."

    return "\n-------\n".join(points_to_text(points))


if __name__ == "__main__":
    import sys
    from requests import ConnectionError

    if len(sys.argv) == 2:
        try:
            print(querydb(sys.argv[1]))
        except ConnectionError as e:
            print("Cannot form embeddings due a connection error")
            print(e)
            print("Check the internet connection.")
        except Exception as e:
            print(f"ERROR: {e}")
    else:
        print("Usage: call querydb followed by your query.")
        print("Example: querydb 'what is logistic regression?'")
