"""
Tests for querydb module.
NOTE: most functionality is tested in the test_vector_store... modules.

IMORTANT: A wokring database must be present for tests to function.
"""

# pyright: reportArgumentType=false
# pyright: reportAssignmentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportUnknownVariableType=false
# pyright: reportGeneralTypeIssues=false

import unittest
import logging

from lmm.utils.logging import LoglistLogger
from qdrant_client.fastembed_common import QueryResponse
from lmm_education.config.config import (
    ConfigSettings,
    export_settings,
)

from lmm_education.querydb import querydb


class TestQueryDbSparse(unittest.TestCase):
    # detup and teardown replace config.toml
    original_settings = ConfigSettings()

    @classmethod
    def setUpClass(cls):
        settings = ConfigSettings(
            embeddings={
                "dense_model": "SentenceTransformers/distiluse-base-multilingual-cased-v1"
            },
            storage=":memory:",
            RAG={
                'encoding_model': "sparse",
                'questions': True,
            },
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

    def test_query_companion(self):
        from lmm_education.ingest import markdown_upload
        from qdrant_client import QdrantClient

        logger = LoglistLogger()

        markdown = """
---\ntitle: the story\ndocid: tte4\nfrozen: True\n---\n
---\nquestions: what is logistic regression?\n---\n
\nThis is logistic regression.\n"""

        import tempfile
        import os

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as temp_file:
            temp_file.write(markdown)
            temp_file_path = temp_file.name

        client = QdrantClient(":memory:")
        points = markdown_upload(
            [temp_file_path], client=client, logger=logger
        )
        if logger.count_logs() > 0:
            print("\n".join(logger.get_logs()))
            logger.clear_logs()
        self.assertTrue(client.collection_exists("chunks"))
        self.assertTrue(client.collection_exists("documents"))
        self.assertTrue(points)

        # get the points in the chunks collection
        response: QueryResponse = client.query_points(
            "chunks",
            None,
        )
        self.assertFalse(len(response.points) == 0)
        chunkID: str = response.points[0].id
        groupID: str = response.points[0].payload['group_UUID']
        self.assertNotEqual(chunkID, groupID)
        self.assertEqual(chunkID, points[0][0])
        self.assertEqual(groupID, points[0][1])

        # verify we have the data in the documents collection
        response: QueryResponse = client.query_points(
            "documents",
            None,
        )
        self.assertEqual(response.points[0].id, groupID)

        from lmm_education.stores.vector_store_qdrant import (
            query,
            QdrantEmbeddingModel,
        )

        data = query(
            client,
            "documents",
            QdrantEmbeddingModel.UUID,
            {},
            points[0][1],
            logger=logger,
        )
        if logger.count_logs() > 0:
            print("\n".join(logger.get_logs()))
        self.assertEqual(logger.count_logs(level=logging.WARNING), 0)
        self.assertEqual(len(data), 1)
        logger.clear_logs()
        docID: str = data[0].id
        self.assertEqual(docID, groupID)

        # now the query with querydb
        try:
            result = querydb(
                "What is logistic regression?",
                client=client,
                logger=logger,
            )
        except Exception as e:
            logger.error(f"{e}")
            self.assertFalse(f"querydb error: {e}")
            return
        finally:
            client.close()

        if logger.count_logs() > 0:
            print(logger.get_logs())
        self.assertIn("This is logistic regression.", result)

        # Clean up the temporary file
        os.remove(temp_file_path)


class TestQueryDbMerged(unittest.TestCase):
    # detup and teardown replace config.toml
    original_settings = ConfigSettings()

    @classmethod
    def setUpClass(cls):
        settings = ConfigSettings(
            embeddings={
                "dense_model": "SentenceTransformers/distiluse-base-multilingual-cased-v1"
            },
            storage=":memory:",
            RAG={
                'encoding_model': "merged",
                'questions': True,
            },
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

    def test_query_companion(self):
        from lmm_education.ingest import markdown_upload
        from qdrant_client import QdrantClient

        logger = LoglistLogger()

        markdown = """
---\ntitle: the story\ndocid: tte4\nfrozen: True\n---\n
---\nquestions: what is logistic regression?\n---\n
\nThis is logistic regression.\n"""

        import tempfile
        import os

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as temp_file:
            temp_file.write(markdown)
            temp_file_path = temp_file.name

        client = QdrantClient(":memory:")
        points = markdown_upload(
            [temp_file_path], client=client, logger=logger
        )
        if logger.count_logs() > 0:
            print("\n".join(logger.get_logs()))
            logger.clear_logs()
        self.assertTrue(client.collection_exists("chunks"))
        self.assertTrue(client.collection_exists("documents"))
        self.assertTrue(points)

        # get the points in the chunks collection
        response: QueryResponse = client.query_points(
            "chunks",
            None,
        )
        self.assertFalse(len(response.points) == 0)
        chunkID: str = response.points[0].id
        groupID: str = response.points[0].payload['group_UUID']
        self.assertNotEqual(chunkID, groupID)
        self.assertEqual(chunkID, points[0][0])
        self.assertEqual(groupID, points[0][1])

        # verify we have the data in the documents collection
        response: QueryResponse = client.query_points(
            "documents",
            None,
        )
        self.assertEqual(response.points[0].id, groupID)

        from lmm_education.stores.vector_store_qdrant import (
            query,
            QdrantEmbeddingModel,
        )

        data = query(
            client,
            "documents",
            QdrantEmbeddingModel.UUID,
            {},
            points[0][1],
            logger=logger,
        )
        if logger.count_logs() > 0:
            print("\n".join(logger.get_logs()))
        self.assertEqual(logger.count_logs(level=logging.WARNING), 0)
        self.assertEqual(len(data), 1)
        logger.clear_logs()
        docID: str = data[0].id
        self.assertEqual(docID, groupID)

        # now the query with querydb
        try:
            result = querydb(
                "What is logistic regression?",
                client=client,
                logger=logger,
            )
        except Exception as e:
            logger.error(f"{e}")
            self.assertFalse(f"querydb error: {e}")
            return
        finally:
            client.close()

        if logger.count_logs() > 0:
            print(logger.get_logs())
        self.assertIn("This is logistic regression.", result)

        # Clean up the temporary file
        os.remove(temp_file_path)


if __name__ == "__main__":
    unittest.main()
