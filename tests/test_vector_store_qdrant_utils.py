import unittest

from lmm.utils.logging import LoglistLogger
from lmm.utils.hash import generate_uuid

from lmm_education.config.config import (
    ConfigSettings,
    export_settings,
)
from lmm_education.stores.vector_store_qdrant import (
    client_from_config,
    initialize_collection_from_config,
    encoding_to_qdrantembedding_model,
    QdrantEmbeddingModel,
)
from lmm_education.stores.vector_store_qdrant_utils import (
    check_schema,
    SCHEMA_COLLECTION_NAME,
)


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
        logs = logger.get_logs(level=1)
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
        logs = logger.get_logs(level=1)
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
        logs = logger.get_logs(level=1)
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


if __name__ == "__main__":
    unittest.main()
