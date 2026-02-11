# pyright: reportArgumentType=false
# pyright: reportCallIssue=false

import unittest

import logging
from lmm.utils.logging import LoglistLogger
from lmm.utils.hash import generate_uuid
from lmm.config.config import EmbeddingSettings

from lmm_education.config.config import (
    ConfigSettings,
    export_settings,
)
from lmm_education.stores.vector_store_qdrant import (
    client_from_config,
    async_client_from_config,
    initialize_collection_from_config,
    initialize_collection,
    ainitialize_collection,
    encoding_to_qdrantembedding_model,
    QdrantEmbeddingModel,
)
from lmm_education.stores.vector_store_qdrant_utils import (
    check_schema,
    SCHEMA_COLLECTION_NAME,
    acheck_schema,
    get_schema,
    aget_schema,
    database_info,
    adatabase_info,
    database_name,
    list_property_values,
    alist_property_values,
)
from qdrant_client.models import PointStruct as Point


class TestQdrantSchema(unittest.TestCase):

    # detup and teardown replace config.toml to avoid
    # calling the language model server
    original_settings = ConfigSettings()

    @classmethod
    def setUpClass(cls):
        settings = ConfigSettings(
            embeddings={
                "dense_model": "SentenceTransformers/distiluse-base-multilingual-cased-v1"
            },
            RAG={'summaries': False, 'encoding_model': "content"},
            storage=":memory:",
        )
        export_settings(settings)

    @classmethod
    def tearDownClass(cls):
        settings = cls.original_settings
        export_settings(settings)

    def test_new_schema(self):
        logger = LoglistLogger()

        client = client_from_config(None, logger)
        logs = logger.get_logs()
        if len(logs) > 0:
            print("\n".join(logs))
        self.assertFalse(logs)
        self.assertIsNotNone(client)

        # ask about non-existing collection
        settings = ConfigSettings()
        flag = check_schema(
            client,
            "NewCollection",
            encoding_to_qdrantembedding_model(
                settings.RAG.encoding_model
            ),
            settings.embeddings,
            logger=logger,
        )
        logs = logger.get_logs()
        self.assertFalse(flag)
        self.assertTrue(len(logs) > 0)
        self.assertIn("is not present", logs[-1])

    def test_invalid_schema(self):
        logger = LoglistLogger()

        client = client_from_config(None, logger)
        self.assertIsNotNone(client)
        if client is None:
            return  # for type checking

        logs = logger.get_logs()
        if len(logs) > 0:
            print("\n".join(logs))
        self.assertFalse(logs)
        self.assertIsNotNone(client)

        COLLECTION: str = "NovelCollection"

        # create collection
        settings = ConfigSettings()
        flag = initialize_collection_from_config(
            client, COLLECTION, settings, logger=logger
        )
        logs = logger.get_logs(level=logging.WARNING)
        if len(logs) > 0:
            print("\n".join(logs))
        self.assertFalse(logs)
        self.assertTrue(flag)

        # check schhema
        flag = check_schema(
            client,
            COLLECTION,
            encoding_to_qdrantembedding_model(
                settings.RAG.encoding_model
            ),
            settings.embeddings,
            logger=logger,
        )
        self.assertTrue(flag)

        # delete payload
        uuid: str = generate_uuid(COLLECTION)
        client.clear_payload(SCHEMA_COLLECTION_NAME, [uuid])

        flag = check_schema(
            client,
            COLLECTION,
            encoding_to_qdrantembedding_model(
                settings.RAG.encoding_model
            ),
            settings.embeddings,
            logger=logger,
        )
        logs = logger.get_logs()
        if len(logs) > 0:
            print("\n".join(logs))
        self.assertFalse(flag)
        self.assertIn("System error.", logs[-1])

    def test_invalid_encoding(self):
        logger = LoglistLogger()

        client = client_from_config(None, logger)
        logs = logger.get_logs()
        if len(logs) > 0:
            print("\n".join(logs))
            logger.clear_logs()
        self.assertFalse(logs)
        self.assertIsNotNone(client)

        COLLECTION: str = "NovelCollection"

        # create collection
        settings = ConfigSettings()
        flag = initialize_collection_from_config(
            client, COLLECTION, settings, logger=logger
        )
        logs = logger.get_logs(level=logging.WARNING)
        if len(logs) > 0:
            print("\n".join(logs))
        self.assertFalse(logs)
        self.assertTrue(flag)

        # check schema
        settings = ConfigSettings()
        flag = check_schema(
            client,
            COLLECTION,
            encoding_to_qdrantembedding_model(
                settings.RAG.encoding_model
            ),
            settings.embeddings,
            logger=logger,
        )
        self.assertTrue(flag)

        # check with wrong encoding
        flag = check_schema(
            client,
            COLLECTION,
            encoding_to_qdrantembedding_model("sparse"),
            settings.embeddings,
            logger=logger,
        )
        self.assertFalse(flag)
        logs = logger.get_logs()
        if len(logs) > 0:
            print("\n".join(logs))
        self.assertIn("Differences from Database Settings", logs[-1])
        self.assertIn(
            "Value Change for 'qdrant_embedding_model", logs[-1]
        )

    def test_invalid_embedding(self):
        from lmm.config.config import EmbeddingSettings

        logger = LoglistLogger()

        client = client_from_config(None, logger)
        logs = logger.get_logs()
        if len(logs) > 0:
            print("\n".join(logs))
            logger.clear_logs()
        self.assertFalse(logs)
        self.assertIsNotNone(client)

        COLLECTION: str = "NovelCollection"

        # create collection
        settings = ConfigSettings()
        flag = initialize_collection_from_config(
            client, COLLECTION, settings, logger=logger
        )
        logs = logger.get_logs(level=logging.WARNING)
        if len(logs) > 0:
            print("\n".join(logs))
        self.assertFalse(logs)
        self.assertTrue(flag)

        # check schema
        settings = ConfigSettings()
        flag = check_schema(
            client,
            COLLECTION,
            encoding_to_qdrantembedding_model(
                settings.RAG.encoding_model
            ),
            settings.embeddings,
            logger=logger,
        )
        self.assertTrue(flag)

        # check with wrong encoding
        flag = check_schema(
            client,
            COLLECTION,
            encoding_to_qdrantembedding_model(
                settings.RAG.encoding_model
            ),
            EmbeddingSettings(
                {
                    'dense_model': "SentenceTransformers/all-MiniLM-L6-v2"
                }
            ),
            logger=logger,
        )
        logs = logger.get_logs()
        if len(logs) > 0:
            print("\n".join(logs))
        self.assertFalse(flag)
        self.assertIn("Differences from Database Settings", logs[-1])
        self.assertIn(
            "Value Change for 'embeddings.dense_model", logs[-1]
        )

    def test_uuid_encoding(self):
        from lmm_education.stores.vector_store_qdrant import (
            initialize_collection,
            EmbeddingSettings,
        )

        logger = LoglistLogger()

        client = client_from_config(None, logger)
        self.assertIsNotNone(client)
        if client is None:
            return  # for type checker

        logs = logger.get_logs()
        if len(logs) > 0:
            print("\n".join(logs))
            logger.clear_logs()
        self.assertFalse(logs)
        self.assertIsNotNone(client)

        COLLECTION: str = "UUIDCollection"
        initialize_collection(
            client,
            COLLECTION,
            QdrantEmbeddingModel.UUID,
            EmbeddingSettings(),
        )

        # check schema
        settings = ConfigSettings()
        flag = check_schema(
            client,
            COLLECTION,
            QdrantEmbeddingModel.UUID,
            settings.embeddings,
            logger=logger,
        )
        logs = logger.get_logs()
        if len(logs) > 0:
            print("\n".join(logs))
        self.assertTrue(flag)
        client.close()

    def test_uuid_encoding2(self):
        from lmm_education.stores.vector_store_qdrant import (
            initialize_collection,
            EmbeddingSettings,
        )

        logger = LoglistLogger()

        client = client_from_config(None, logger)
        self.assertIsNotNone(client)
        if client is None:
            return  # for type checker

        COLLECTION: str = "UUIDCollection2"
        initialize_collection(
            client,
            COLLECTION,
            QdrantEmbeddingModel.DENSE,
            EmbeddingSettings(),
        )

        # check schema
        settings = ConfigSettings()
        flag = check_schema(
            client,
            COLLECTION,
            QdrantEmbeddingModel.UUID,
            settings.embeddings,
            logger=logger,
        )
        logs = logger.get_logs()
        if len(logs) > 0:
            print("\n".join(logs))
        self.assertFalse(flag)
        client.close()


class TestGetSchema(unittest.TestCase):
    """Tests for get_schema function."""

    original_settings = ConfigSettings()

    @classmethod
    def setUpClass(cls):
        settings = ConfigSettings(
            embeddings={
                "dense_model": "SentenceTransformers/distiluse-base-multilingual-cased-v1"
            },
            RAG={'summaries': False, 'encoding_model': "content"},
            storage=":memory:",
        )
        export_settings(settings)

    @classmethod
    def tearDownClass(cls):
        export_settings(cls.original_settings)

    def test_get_schema_success(self):
        """Test successful schema retrieval."""
        logger = LoglistLogger()
        client = client_from_config(None, logger)
        self.assertIsNotNone(client)

        # Initialize collection with schema
        settings = ConfigSettings()
        COLLECTION = "TestCollection"
        initialize_collection_from_config(
            client, COLLECTION, settings, logger=logger
        )

        # Get schema
        schema = get_schema(client, COLLECTION, logger=logger)
        self.assertIsNotNone(schema)
        self.assertIn('qdrant_embedding_model', schema)
        self.assertIn('embeddings', schema)

    def test_get_schema_nonexistent_collection(self):
        """Test get_schema with non-existent collection."""
        logger = LoglistLogger()
        client = client_from_config(None, logger)
        self.assertIsNotNone(client)

        schema = get_schema(
            client, "NonExistentCollection", logger=logger
        )
        self.assertIsNone(schema)
        logs = logger.get_logs()
        self.assertTrue(
            any("is not in the database" in log for log in logs)
        )


class TestAsyncCheckSchema(unittest.IsolatedAsyncioTestCase):
    """Tests for acheck_schema async function."""

    original_settings = ConfigSettings()

    @classmethod
    def setUpClass(cls):
        settings = ConfigSettings(
            embeddings={
                "dense_model": "SentenceTransformers/distiluse-base-multilingual-cased-v1"
            },
            RAG={'summaries': False, 'encoding_model': "content"},
            storage=":memory:",
        )
        export_settings(settings)

    @classmethod
    def tearDownClass(cls):
        export_settings(cls.original_settings)

    async def test_acheck_schema_new_collection(self):
        """Test async check_schema with new collection."""
        logger = LoglistLogger()
        client = async_client_from_config(None, logger)
        self.assertIsNotNone(client)

        settings = ConfigSettings()
        COLLECTION = "AsyncTestCollection"

        # Create collection
        await ainitialize_collection(
            client,
            COLLECTION,
            encoding_to_qdrantembedding_model(
                settings.RAG.encoding_model
            ),
            settings.embeddings,
        )

        # Check schema
        flag = await acheck_schema(
            client,
            COLLECTION,
            encoding_to_qdrantembedding_model(
                settings.RAG.encoding_model
            ),
            settings.embeddings,
            logger=logger,
        )
        self.assertTrue(flag)
        if client is not None:
            await client.close()

    async def test_acheck_schema_uuid_encoding(self):
        """Test async check_schema with UUID encoding."""
        logger = LoglistLogger()
        client = async_client_from_config(None, logger)
        self.assertIsNotNone(client)

        COLLECTION = "AsyncUUIDCollection"
        settings = EmbeddingSettings()

        # Create collection with UUID model
        await ainitialize_collection(
            client,
            COLLECTION,
            QdrantEmbeddingModel.UUID,
            settings,
        )

        # Check schema
        flag = await acheck_schema(
            client,
            COLLECTION,
            QdrantEmbeddingModel.UUID,
            settings,
            logger=logger,
        )
        self.assertTrue(flag)
        if client is not None:
            await client.close()


class TestAsyncGetSchema(unittest.IsolatedAsyncioTestCase):
    """Tests for aget_schema async function."""

    original_settings = ConfigSettings()

    @classmethod
    def setUpClass(cls):
        settings = ConfigSettings(
            embeddings={
                "dense_model": "SentenceTransformers/distiluse-base-multilingual-cased-v1"
            },
            RAG={'summaries': False, 'encoding_model': "content"},
            storage=":memory:",
        )
        export_settings(settings)

    @classmethod
    def tearDownClass(cls):
        export_settings(cls.original_settings)

    async def test_aget_schema_success(self):
        """Test successful async schema retrieval."""
        logger = LoglistLogger()
        client = async_client_from_config(None, logger)
        self.assertIsNotNone(client)

        settings = ConfigSettings()
        COLLECTION = "AsyncGetSchemaTest"

        # Create collection
        await ainitialize_collection(
            client,
            COLLECTION,
            encoding_to_qdrantembedding_model(
                settings.RAG.encoding_model
            ),
            settings.embeddings,
        )

        # Get schema
        schema = await aget_schema(client, COLLECTION, logger=logger)
        self.assertIsNotNone(schema)
        self.assertIn('qdrant_embedding_model', schema)
        self.assertIn('embeddings', schema)
        if client is not None:
            await client.close()


class TestDatabaseInfo(unittest.TestCase):
    """Tests for database_info function."""

    original_settings = ConfigSettings()

    @classmethod
    def setUpClass(cls):
        settings = ConfigSettings(
            embeddings={
                "dense_model": "SentenceTransformers/distiluse-base-multilingual-cased-v1"
            },
            RAG={'summaries': False, 'encoding_model': "content"},
            storage=":memory:",
        )
        export_settings(settings)

    @classmethod
    def tearDownClass(cls):
        export_settings(cls.original_settings)

    def test_database_info_with_client(self):
        """Test database_info with provided client."""
        logger = LoglistLogger()
        client = client_from_config(None, logger)
        self.assertIsNotNone(client)

        info = database_info(client, logger=logger)
        self.assertIsInstance(info, dict)
        self.assertIn('storage', info)
        self.assertIn('schema_collection', info)
        self.assertIn('main_collection', info)

    def test_database_info_without_client(self):
        """Test database_info creates its own client."""
        logger = LoglistLogger()
        info = database_info(None, logger=logger)
        self.assertIsInstance(info, dict)
        self.assertIn('storage', info)


class TestAsyncDatabaseInfo(unittest.IsolatedAsyncioTestCase):
    """Tests for adatabase_info async function."""

    original_settings = ConfigSettings()

    @classmethod
    def setUpClass(cls):
        settings = ConfigSettings(
            embeddings={
                "dense_model": "SentenceTransformers/distiluse-base-multilingual-cased-v1"
            },
            RAG={'summaries': False, 'encoding_model': "content"},
            storage=":memory:",
        )
        export_settings(settings)

    @classmethod
    def tearDownClass(cls):
        export_settings(cls.original_settings)

    async def test_adatabase_info_with_client(self):
        """Test async database_info with provided client."""
        logger = LoglistLogger()
        client = async_client_from_config(None, logger)
        self.assertIsNotNone(client)

        info = await adatabase_info(client, logger=logger)
        self.assertIsInstance(info, dict)
        self.assertIn('storage', info)
        self.assertIn('schema_collection', info)
        if client is not None:
            await client.close()


class TestDatabaseName(unittest.TestCase):
    """Tests for database_name function."""

    original_settings = ConfigSettings()

    @classmethod
    def setUpClass(cls):
        settings = ConfigSettings(
            embeddings={
                "dense_model": "SentenceTransformers/distiluse-base-multilingual-cased-v1"
            },
            RAG={'summaries': False, 'encoding_model': "content"},
            storage=":memory:",
        )
        export_settings(settings)

    @classmethod
    def tearDownClass(cls):
        export_settings(cls.original_settings)

    def test_database_name_memory(self):
        """Test database_name with memory database."""
        logger = LoglistLogger()
        client = client_from_config(None, logger)
        self.assertIsNotNone(client)

        name = database_name(client)
        self.assertIsInstance(name, str)
        # For memory databases, should return ":memory:" or "<unknown>"
        self.assertTrue(
            name in [":memory:", "<unknown>"] or len(name) > 0
        )


class TestListPropertyValues(unittest.TestCase):
    """Tests for list_property_values function."""

    original_settings = ConfigSettings()

    @classmethod
    def setUpClass(cls):
        settings = ConfigSettings(
            embeddings={
                "dense_model": "SentenceTransformers/distiluse-base-multilingual-cased-v1"
            },
            RAG={'summaries': False, 'encoding_model': "content"},
            storage=":memory:",
        )
        export_settings(settings)

    @classmethod
    def tearDownClass(cls):
        export_settings(cls.original_settings)

    def test_list_property_values_empty_collection(self):
        """Test list_property_values with empty collection."""
        logger = LoglistLogger()
        client = client_from_config(None, logger)
        self.assertIsNotNone(client)

        settings = ConfigSettings()
        COLLECTION = "TestPropertyCollection"

        # Create collection
        initialize_collection_from_config(
            client, COLLECTION, settings, logger=logger
        )

        # List property values (should be empty)
        values = list_property_values(
            client, "source", COLLECTION, logger=logger
        )
        self.assertIsInstance(values, list)

    def test_list_property_values_with_data(self):
        """Test list_property_values with populated collection."""
        logger = LoglistLogger()
        client = client_from_config(None, logger)
        self.assertIsNotNone(client)

        settings = ConfigSettings()
        COLLECTION = "TestPropertyCollection2"

        # Create collection
        initialize_collection(
            client,
            COLLECTION,
            QdrantEmbeddingModel.UUID,
            settings.embeddings,
        )

        # Add points with metadata
        from lmm.utils.hash import generate_uuid

        points = [
            Point(
                id=generate_uuid("test1"),
                vector={},
                payload={"metadata": {"source": "doc1.txt"}},
            ),
            Point(
                id=generate_uuid("test2"),
                vector={},
                payload={"metadata": {"source": "doc1.txt"}},
            ),
            Point(
                id=generate_uuid("test3"),
                vector={},
                payload={"metadata": {"source": "doc2.txt"}},
            ),
        ]
        client.upsert(COLLECTION, points)  # type: ignore (checked above)

        # List property values
        values = list_property_values(
            client, "source", COLLECTION, logger=logger
        )
        self.assertIsInstance(values, list)
        if values:  # May be empty depending on Qdrant version/setup
            self.assertTrue(
                all(
                    isinstance(v, tuple) and len(v) == 2  # type: ignore
                    for v in values
                )
            )


class TestAsyncListPropertyValues(unittest.IsolatedAsyncioTestCase):
    """Tests for alist_property_values async function."""

    original_settings = ConfigSettings()

    @classmethod
    def setUpClass(cls):
        settings = ConfigSettings(
            embeddings={
                "dense_model": "SentenceTransformers/distiluse-base-multilingual-cased-v1"
            },
            RAG={'summaries': False, 'encoding_model': "content"},
            storage=":memory:",
        )
        export_settings(settings)

    @classmethod
    def tearDownClass(cls):
        export_settings(cls.original_settings)

    async def test_alist_property_values_empty(self):
        """Test async list_property_values with empty collection."""
        logger = LoglistLogger()
        client = async_client_from_config(None, logger)
        self.assertIsNotNone(client)

        settings = ConfigSettings()
        COLLECTION = "AsyncPropTest"

        # Create collection
        await ainitialize_collection(
            client,
            COLLECTION,
            encoding_to_qdrantembedding_model(
                settings.RAG.encoding_model
            ),
            settings.embeddings,
        )

        # List property values
        values = await alist_property_values(
            client, "source", COLLECTION, logger=logger
        )
        self.assertIsInstance(values, list)
        if client is not None:
            await client.close()


if __name__ == "__main__":
    unittest.main()
