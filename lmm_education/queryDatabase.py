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

python -m lmm_education.queryDatabase 'what is logistic regression?'
```

```python
# from python code
from lmm_education.queryDatabase import do_query

response = do_query('what is logistic regression?')
print(response)
```

Because ingest replaces the content of the database when documents
are edited, you can set up an ingest-evaluate loop with queryDatabase:

```python
# from the python REPL

# append True to ingest the file 'RaggedDocument.md'
python -m lmm_education.ingest RaggedDocument.md True
python -m lmm_education.queryDatabase 'what is logistic regression?'
"""

from pydantic import validate_call
from qdrant_client.models import ScoredPoint
from lmm_education.stores.vector_store_qdrant import (
    client_from_config,
    query,
    query_grouped,
    encoding_to_qdrantembedding_model,
    groups_to_points,
    points_to_text,
)
from lmm_education.config.config import ConfigSettings


@validate_call
def do_query(query_text: str) -> str:
    """
    Execute a query on the database using the settings in
    config.toml.

    Args:
        query_text: a text to use as query.

    Returns:
        a string concatenating the results of the query.
    """
    if not query_text:
        print("No query text provided")
        return ""

    settings = ConfigSettings()
    client = client_from_config(settings)
    if client is None:
        print("Failed to initialize qdrant client")
        return ""

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
        )
    client.close()

    return "\n---\n".join(points_to_text(points))


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 2:
        print(do_query(sys.argv[1]))
    else:
        print("Usage: call queryDatabase followed by your query.")
        print("Example: queryDatabase 'what is logistic regression?'")
