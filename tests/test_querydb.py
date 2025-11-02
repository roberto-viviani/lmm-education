"""
Tests for querydb module.
NOTE: most functionality is tested in the test_vector_store... modules.

IMORTANT: A wokring database must be present for tests to function.
"""

import unittest

from lmm.utils.logging import LoglistLogger
from lmm_education.config.config import (
    ConfigSettings,
    export_settings,
)

from lmm_education.querydb import querydb


class TestQueryDb(unittest.TestCase):
    # detup and teardown replace config.toml
    original_settings = ConfigSettings()

    @classmethod
    def setUpClass(cls):
        settings = ConfigSettings(
            embeddings={
                "dense_model": "SentenceTransformers/distiluse-base-multilingual-cased-v1"
            },
            storage=":memory:",
            RAG={'encoding_model': "content", 'questions': True},
        )
        export_settings(settings)

    @classmethod
    def tearDownClass(cls):
        settings = cls.original_settings
        export_settings(settings)

    def test_empty(self):
        logger = LoglistLogger()
        result = querydb("", logger=logger)
        self.assertFalse(result)
        self.assertIn("No query text", logger.get_logs()[-1])

    def test_short(self):
        logger = LoglistLogger()
        result = querydb("ba", logger=logger)
        self.assertFalse(result)
        self.assertIn("Invalid query", logger.get_logs()[-1])

    def test_empty_database(self):
        logger = LoglistLogger()
        result = querydb(
            "What is logistic regression?", logger=logger
        )
        logs = logger.get_logs()
        self.assertIn("Could not read from the database", logs[-1])
        self.assertIn("please check connection/database", result)

    def test_query(self):
        from lmm_education.ingest import markdown_upload
        from qdrant_client import QdrantClient

        markdown = """---\ntitle: document\nfrozen: True\n---\n
        ---\nquestions: what is logistic regression?\n---\n
        This is logistic regression.\n"""

        import tempfile
        import os

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as temp_file:
            temp_file.write(markdown)
            temp_file_path = temp_file.name

        client = QdrantClient(":memory:")
        points = markdown_upload([temp_file_path], client=client)
        self.assertTrue(points)
        print(points)
        from lmm_education.stores.vector_store_qdrant_utils import (
            database_info,
        )

        print(database_info(client))

        logger = LoglistLogger()
        try:
            result = querydb(
                "What is logistic regression?",
                client=client,
                logger=logger,
            )
        except Exception as e:
            logger.error(f"{e}")
        finally:
            client.close()

        print(logger.get_logs())
        self.assertIn("This is logistic regression.", result)

        # Clean up the temporary file
        os.remove(temp_file_path)


if __name__ == "__main__":
    unittest.main()
