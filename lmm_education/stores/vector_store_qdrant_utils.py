"""
Support functions for the vector_store_qdrant module.

Main functions:
    check_schema/acheck_schema: add or check the schema
        of a collection
    get_schema/aget_schema: a utility to inspect the schema
        of a collection.

A schema consists of the qdrant embedding model enum selection
and of the embedding settings. Schemas are collection-specific.
"""

# pyright: reportUnusedImport=true


from enum import Enum
from typing import Any

# qdrant
from qdrant_client import QdrantClient
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    PointStruct as Point,
    Record,
    UpdateResult,
)
from qdrant_client.http.exceptions import (
    ApiException,
    UnexpectedResponse,
)

# lmmarkdown
from lmm.config.config import EmbeddingSettings
from lmm.utils.hash import generate_uuid


# lmm markdown for education
from lmm_education.config.utils import (
    find_dictionary_differences,
    format_difference_report,
)

# Set up logger
from lmm.utils.logging import LoggerBase, get_logger

default_logger: LoggerBase = get_logger(__name__)


SCHEMA_COLLECTION_NAME = "__SCHEMA__"


def check_schema(
    client: QdrantClient,
    collection_name: str,
    qdrant_model: Enum,
    embedding_settings: EmbeddingSettings,
    *,
    logger: LoggerBase = default_logger,
) -> bool:
    """
    Performs a check that the collection has been initialized
    with the correct qdrant model and embedding settings.

    If no schema exists for the collection, it will add the
    schema. Hence, this function may be called after creating
    a collection to add the schema to the database.

    Args:
        client: the qdrant client
        collection_name: the collection to check
        qdrant_model: the model of the collection
        embedding_settings: the settings of the collection
        logger: a logger object

    Returns:
        a boolean flag

    Note:
        This is a low-level function for internal use.
        Call within a try block to catch connetion errors to
        the database.
    """

    if not client.collection_exists(SCHEMA_COLLECTION_NAME):
        flag: bool = client.create_collection(
            collection_name=SCHEMA_COLLECTION_NAME,
            vectors_config={},
        )
        if flag:
            logger.info("Schema collection created.")
        else:
            logger.error("Schema collection could not be created.")
            return True

    uuid: str = generate_uuid(collection_name)
    payload = {
        'qdrant_embedding_model': qdrant_model.value,
        'embeddings': embedding_settings.model_dump(mode="json"),
    }

    records: list[Record] = client.retrieve(
        collection_name=SCHEMA_COLLECTION_NAME,
        ids=[uuid],
        with_payload=True,
    )

    if not records:
        pt = Point(id=uuid, vector={}, payload=payload)
        result: UpdateResult = client.upsert(
            SCHEMA_COLLECTION_NAME, [pt]
        )
        if result.status == "completed":
            logger.info(f"Schema added for {collection_name}")
        else:
            logger.info(
                f"Attempted schema registation {collection_name}"
            )
    else:
        if not records[0].payload:
            logger.error(
                "System error. The internal schema"
                f" for collection {collection_name}"
                " is corrupted. Repeat call to recreate"
                " schema with current settings."
            )
            client.delete(
                SCHEMA_COLLECTION_NAME,
                [uuid],
            )
            return False
        delta = find_dictionary_differences(
            payload, records[0].payload
        )
        if delta:
            logger.error(
                format_difference_report(
                    delta,
                    collection_name,
                )
            )
            return False

    return True


async def acheck_schema(
    client: AsyncQdrantClient,
    collection_name: str,
    qdrant_model: Enum,
    embedding_settings: EmbeddingSettings,
    *,
    logger: LoggerBase = default_logger,
) -> bool:
    """
    Performs a check that the collection has been initialized
    with the correct qdrant model and embedding settings.

    If no schema exists for the collection, it will add the
    schema. Hence, this function may be called after creating
    a collection to add the schema to the database.

    Args:
        client: the qdrant client
        collection_name: the collection to check
        qdrant_model: the model of the collection
        embedding_settings: the settings of the collection
        logger: a logger object

    Returns:
        a boolean flag

    Note:
        This is a low-level function for internal use.
        Call within a try block to catch connection errors to
        the database.
    """

    if not await client.collection_exists(SCHEMA_COLLECTION_NAME):
        await client.create_collection(
            collection_name=SCHEMA_COLLECTION_NAME,
            vectors_config={},
        )

    uuid: str = generate_uuid(collection_name)
    payload = {
        'qdrant_embedding_model': qdrant_model.value,
        'embeddings': embedding_settings.model_dump(mode="json"),
    }

    records: list[Record] = await client.retrieve(
        collection_name=SCHEMA_COLLECTION_NAME,
        ids=[uuid],
        with_payload=True,
    )

    if not records:
        pt = Point(id=uuid, vector={}, payload=payload)
        result = await client.upsert(SCHEMA_COLLECTION_NAME, [pt])
        if result.status == "completed":
            logger.info(f"Schema added for {collection_name}")
        else:
            logger.info(
                f"Attempted schema registation {collection_name}"
            )
    else:
        if records[0].payload is None:
            logger.error(
                "System error. The internal schema"
                f" for collection {collection_name}"
                " is corrupted. Repeat call to recreate"
                " schema with current settings."
            )
            await client.delete(
                SCHEMA_COLLECTION_NAME,
                [uuid],
            )
            return False
        check = find_dictionary_differences(
            payload, records[0].payload
        )
        if check:
            logger.error(
                format_difference_report(
                    check,
                    collection_name,
                )
            )
            return False

    return True


def get_schema(
    client: QdrantClient,
    collection_name: str,
    *,
    logger: LoggerBase = default_logger,
) -> dict[str, Any] | None:
    """
    Retrieves the schema for a collection.

    Args:
        client: the qdrant client
        collection_name: the name of the collection
        logger: a logger object

    Returns:
        a dictionary with the qdrant embedding model and
        vector embedding settings, or None if the collection
        does not exist, or errors are raised.
    """

    try:
        if not client.collection_exists(collection_name):
            logger.info(f"{collection_name} is not in the database.")
            return None

        if not client.collection_exists(SCHEMA_COLLECTION_NAME):
            logger.error(
                "System error. The internal schema"
                " is corrupted. Call initialize_collection on"
                " a collection to reinitialize."
            )
            return None

        uuid: str = generate_uuid(collection_name)

        records: list[Record] = client.retrieve(
            collection_name=SCHEMA_COLLECTION_NAME,
            ids=[uuid],
            with_payload=True,
        )

        if (not records) or (records[0].payload is None):
            logger.error(
                "System error. The internal schema"
                f" for collection {collection_name}"
                " is corrupted or missing. Call initialize_collection"
                f" on {collection_name} to restore schema with current"
                " settings."
            )
            return None
    except ConnectionError:
        logger.error(
            "Could not connect to the qdrant server, which may be down."
        )
        return None
    except UnexpectedResponse as e:
        logger.error(f"Could not initialize vector database: {e}")
        return None
    except ApiException as e:
        logger.error(
            f"Could not initialize vector database due to API error: {e}"
        )
        return None
    except Exception as e:
        logger.error(f"Could not initialize vector database: {e}")
        return None

    return records[0].payload


async def aget_schema(
    client: AsyncQdrantClient,
    collection_name: str,
    *,
    logger: LoggerBase = default_logger,
) -> dict[str, Any] | None:
    """
    Retrieves the schema for a collection.

    Args:
        client: the qdrant client
        collection_name: the name of the collection
        logger: a logger object

    Returns:
        a dictionary with the qdrant embedding model and
        vector embedding settings, or None if the collection
        does not exist, or errors are raised.
    """

    try:
        if not await client.collection_exists(collection_name):
            logger.info(f"{collection_name} is not in the database.")
            return None

        if not await client.collection_exists(SCHEMA_COLLECTION_NAME):
            logger.error(
                "System error. The internal schema"
                " is corrupted. Call initialize_collection on"
                " a collection to reinitialize."
            )
            return None

        uuid: str = generate_uuid(collection_name)

        records: list[Record] = await client.retrieve(
            collection_name=SCHEMA_COLLECTION_NAME,
            ids=[uuid],
            with_payload=True,
        )

        if (not records) or (records[0].payload is None):
            logger.error(
                "System error. The internal schema"
                f" for collection {collection_name}"
                " is corrupted. Repeat call to recreate"
                " schema with current settings."
            )
            return None
    except ConnectionError:
        logger.error(
            "Could not connect to the qdrant server, which may be down."
        )
        return None
    except UnexpectedResponse as e:
        logger.error(f"Could not initialize vector database: {e}")
        return None
    except ApiException as e:
        logger.error(
            f"Could not initialize vector database due to API error: {e}"
        )
        return None
    except Exception as e:
        logger.error(f"Could not initialize vector database: {e}")
        return None

    return records[0].payload


def database_info(
    client: QdrantClient | None = None,
) -> dict[str, str]:
    from lmm_education.config.config import ConfigSettings

    create_flag: bool = False
    if client is None:
        from lmm_education.stores.vector_store_qdrant import (
            client_from_config,
        )

        client = client_from_config()
        if client is None:
            return {}
        else:
            create_flag = True

    try:
        config = ConfigSettings()
        main_collection: str = config.database.collection_name
        comp_collection: str | None = (
            config.database.companion_collection
        )
        if client.collection_exists(SCHEMA_COLLECTION_NAME):
            return {
                'schema_collection': "present",
                'main_collection': main_collection
                + " || "
                + str(get_schema(client, main_collection)),
                'companion_collection': (
                    comp_collection
                    + " || "
                    + str(get_schema(client, comp_collection))
                    if comp_collection
                    else "none"
                ),
            }
        else:
            return {
                'schema_collection': "none",
                'main_collection': (
                    "present"
                    if client.collection_exists(main_collection)
                    else "none"
                ),
                'companion_collection': (
                    "present"
                    if comp_collection
                    and client.collection_exists(comp_collection)
                    else "none"
                ),
            }
    except Exception as e:
        return {'ERROR': str(e)}
    finally:
        if create_flag:
            client.close()


async def adatabase_info(
    client: AsyncQdrantClient | None = None,
) -> dict[str, str]:
    from lmm_education.config.config import ConfigSettings

    create_flag: bool = False
    if client is None:
        from lmm_education.stores.vector_store_qdrant import (
            async_client_from_config,
        )

        client = await async_client_from_config()
        if client is None:
            return {}
        else:
            create_flag = True

    try:
        config = ConfigSettings()
        main_collection: str = config.database.collection_name
        comp_collection: str | None = (
            config.database.companion_collection
        )
        if await client.collection_exists(SCHEMA_COLLECTION_NAME):
            return {
                'schema_collection': "present",
                'main_collection': main_collection
                + " || "
                + str(await aget_schema(client, main_collection)),
                'companion_collection': (
                    comp_collection
                    + " || "
                    + str(await aget_schema(client, comp_collection))
                    if comp_collection
                    else "none"
                ),
            }
        else:
            return {
                'schema_collection': "none",
                'main_collection': (
                    "present"
                    if await client.collection_exists(main_collection)
                    else "none"
                ),
                'companion_collection': (
                    "present"
                    if comp_collection
                    and await client.collection_exists(
                        comp_collection
                    )
                    else "none"
                ),
            }
    except Exception as e:
        return {'ERROR': str(e)}
    finally:
        if create_flag:
            await client.close()
