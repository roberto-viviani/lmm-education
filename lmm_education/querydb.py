"""
This module allows retrieval of material from the vector database.

Since retrieval is performed by the RAG code automatically (for
example, in appChat), this module is meant to be used for test
purposes.

A database must have been created (for example, with the ingest
module).

Examples:

```bash
# Python from the command line

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

```bash
# Python called from the command line

# append True to ingest the file 'LogisticRegression.md'
python -m lmm_education.ingest LogisticRegression.md True
python -m lmm_education.querydb 'what is logistic regression?'
```

Important: this module is only provided for use from an
interactive envirnoment, such as the Python REPL. It opens the
database exclusively.
"""

from pydantic import validate_call

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import GroupsResult, ScoredPoint

from lmm.utils.logging import LoggerBase, ConsoleLogger
from lmm.scan.scan_keys import CTXT_SUMMARY_KEY

from lmm_education.config.config import ConfigSettings

from .stores.vector_store_qdrant import (
    query,
    query_grouped,
    aquery,
    aquery_grouped,
    encoding_to_qdrantembedding_model,
    groups_to_points,
    points_to_text,
    points_to_metadata,
)
from .stores.vector_store_qdrant_context import (
    global_client_from_config,
    global_async_client_from_config,
)
from .config.config import load_settings


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
        client: a QdrantClient object
        logger: a logger object. Defaults to the console.

    Returns:
        a string concatenating the results of the query.
    """
    if not query_text:
        logger.info("No query text provided")
        return ""

    if len(query_text.split()) < 3:
        logger.info("Invalid query? " + query_text)
        return ""

    settings: ConfigSettings | None = load_settings(logger=logger)
    if not settings:
        logger.error("Could not read settings")
        return ""

    if client is None:
        try:
            client = global_client_from_config(settings.storage)
        except Exception as e:
            logger.error(f"Could not create client: {e}")
            return ""

    points: list[ScoredPoint] = []
    retrieve_docs: bool = settings.RAG.retrieve_docs or False
    if retrieve_docs and not settings.database.companion_collection:
        logger.warning(
            "Retrieve docs directive ignores, no companion collection"
        )
        retrieve_docs = False
    if retrieve_docs:
        results: GroupsResult = query_grouped(
            client,
            settings.database.collection_name,
            settings.database.companion_collection,  # type: ignore
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

    if not points:
        return "No results, please check connection/database."

    if retrieve_docs:
        summaries = points_to_metadata(points, CTXT_SUMMARY_KEY)
        texts = points_to_text(points)
        whole = [
            str(sm) + "\n\n" + txt
            for sm, txt in zip(summaries, texts)
        ]
        return "\n-------\n".join(whole)
    else:
        return "\n-------\n".join(points_to_text(points))


async def aquerydb(
    query_text: str,
    *,
    client: AsyncQdrantClient | None = None,
    logger: LoggerBase = ConsoleLogger(),
) -> str:
    """
    Execute a query on the database using the settings in
    config.toml.

    Args:
        query_text: a text to use as query.
        client: a QdrantClient object
        logger: a logger object. Defaults to the console.

    Returns:
        a string concatenating the results of the query.
    """
    if not query_text:
        logger.info("No query text provided")
        return ""

    if len(query_text.split()) < 3:
        logger.info("Invalid query? " + query_text)
        return ""

    settings: ConfigSettings | None = load_settings(logger=logger)
    if not settings:
        logger.error("Could not read settings")
        return ""

    if client is None:
        try:
            client = global_async_client_from_config(settings.storage)
        except Exception as e:
            logger.error(f"Could not create client: {e}")
            return ""

    points: list[ScoredPoint] = []
    retrieve_docs: bool = settings.RAG.retrieve_docs or False
    if retrieve_docs and not settings.database.companion_collection:
        retrieve_docs = False
        logger.warning(
            "Retrieve docs directive ignored, no companion collection"
        )
    if retrieve_docs:
        results: GroupsResult = await aquery_grouped(
            client,
            settings.database.collection_name,
            settings.database.companion_collection,  # type: ignore
            encoding_to_qdrantembedding_model(
                settings.RAG.encoding_model
            ),
            settings.embeddings,
            query_text,
            logger=logger,
        )
        points = groups_to_points(results)
    else:
        points = await aquery(
            client,
            settings.database.collection_name,
            encoding_to_qdrantembedding_model(
                settings.RAG.encoding_model
            ),
            settings.embeddings,
            query_text,
            logger=logger,
        )

    if not points:
        return "No results, please check connection/database."

    if retrieve_docs:
        summaries = points_to_metadata(points, CTXT_SUMMARY_KEY)
        texts = points_to_text(points)
        whole = [
            str(sm) + "\n\n" + txt
            for sm, txt in zip(summaries, texts)
        ]
        return "\n-------\n".join(whole)
    else:
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
