"""Tests mirroring the docstring examples in vector_store_qdrant module.

These tests ensure that the code examples in the module docstring work
exactly as documented.
"""

import unittest
import tempfile

from lmm.scan.chunks import Chunk, EncodingModel
from lmm.utils.logging import LoglistLogger
from lmm.utils.hash import generate_uuid
from lmm_education.config.config import (
    ConfigSettings,
    DatabaseSource,
    LocalStorage,
)
from lmm_education.stores.vector_store_qdrant import (
    client_from_config,
    qdrant_client_context,
    initialize_collection,
    upload,
    query,
    QdrantClient,
    QdrantEmbeddingModel,
    encoding_to_qdrantembedding_model,
)


class TestDocstringExamples(unittest.TestCase):
    """Test that all docstring examples work as documented."""

    def test_example_1_client_from_config(self):
        """Example 1: Client creation from config (lines 14-19)."""
        # Example code from docstring:
        # from lmm_education.stores.vector_store_qdrant import client_from_config
        #
        # client: QdrantClient | None = client_from_config()
        # # check client is not None

        with tempfile.TemporaryDirectory() as temp_dir:
            settings = ConfigSettings(
                storage=LocalStorage(folder=temp_dir)
            )
            client: QdrantClient | None = client_from_config(settings)

            # Check client is not None
            self.assertIsNotNone(client)

            if client is not None:
                client.close()

    def test_example_2_client_with_override(self):
        """Example 2: Client with config override (lines 24-31)."""
        # Example code from docstring:
        # from lmm_education.stores.vector_store_qdrant import client_from_config
        # from lmm_education.config.config import ConfigSettings
        #
        # # read settings from config.toml, but override 'storage':
        # settings = ConfigSettings(storage=":memory:")
        # client: QdrantClient | None = client_from_config(settings)

        # Read settings from config.toml, but override 'storage':
        settings = ConfigSettings(storage=":memory:")
        client: QdrantClient | None = client_from_config(settings)

        self.assertIsNotNone(client)

        if client is not None:
            client.close()

    def test_example_3_context_manager(self):
        """Example 3: Context manager usage (lines 59-70)."""
        # Example code from docstring (simplified for testing):
        # try:
        #     with qdrant_client_context() as client:
        #         result_docs = upload(
        #             client, "documents", model, settings, doc_chunks
        #         )
        #         result_imgs = upload(
        #             client, "images", model, settings, img_chunks
        #         )
        # except Exception as e:
        #     .... error handling

        logger = LoglistLogger()
        encoding_model = EncodingModel.CONTENT
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )

        doc_chunks = [
            Chunk(
                content="Document content",
                metadata={},
                uuid=generate_uuid("doc-uuid-1"),
            )
        ]
        img_chunks = [
            Chunk(
                content="Image description",
                metadata={},
                uuid=generate_uuid("img-uuid-1"),
            )
        ]

        try:
            with qdrant_client_context(
                ConfigSettings(storage=":memory:")
            ) as client:
                # Initialize collections
                initialize_collection(
                    client,
                    "documents",
                    embedding_model,
                    ConfigSettings().embeddings,
                    logger=logger,
                )
                initialize_collection(
                    client,
                    "images",
                    embedding_model,
                    ConfigSettings().embeddings,
                    logger=logger,
                )

                result_docs = upload(
                    client,
                    "documents",
                    embedding_model,
                    ConfigSettings().embeddings,
                    doc_chunks,
                    logger=logger,
                )
                result_imgs = upload(
                    client,
                    "images",
                    embedding_model,
                    ConfigSettings().embeddings,
                    img_chunks,
                    logger=logger,
                )

                self.assertEqual(len(result_docs), len(doc_chunks))
                self.assertEqual(len(result_imgs), len(img_chunks))
        except Exception as e:
            self.fail(f"Context manager example failed: {e}")

    def test_example_4_initialize_collection(self):
        """Example 4: Collection initialization (lines 81-95)."""
        # Example code from docstring:
        # # ... client creation not shown
        # from lmm_education.stores.vector_store_qdrant import (
        #     initialize_collection,
        #     QdrantEmbeddingModel,
        # )
        #
        # embedding_model = QdrantEmbeddingModel.DENSE
        # flag: bool = initialize_collection(
        #     client,
        #     "documents",
        #     embedding_model,
        #     ConfigSettings().embeddings,
        #     logger=logger)

        logger = LoglistLogger()

        with tempfile.TemporaryDirectory() as temp_dir:
            client = QdrantClient(path=temp_dir)

            embedding_model = QdrantEmbeddingModel.DENSE
            flag: bool = initialize_collection(
                client,
                "documents",
                embedding_model,
                ConfigSettings().embeddings,
                logger=logger,
            )

            self.assertTrue(flag)

            client.close()

    def test_example_5_context_with_database_source(self):
        """Example 5: Context manager with DatabaseSource (lines 97-110)."""
        # Example code from docstring:
        # embedding_model = QdrantEmbeddingModel.DENSE
        # opts: DatabaseSource = ":memory:"
        # try:
        #     with qdrant_client_context(opts) as client:
        #         result = initialize_collection(
        #             client,
        #             "Main",
        #             embedding_model,
        #             ConfigSettings().embeddings,
        #         )
        # except Exception as e:
        #     .... handle exceptions

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
                self.assertTrue(result)
        except Exception as e:
            self.fail(
                f"Context manager with DatabaseSource failed: {e}"
            )

    def test_example_6_upload(self):
        """Example 6: Upload (lines 119-128)."""
        # Example code from docstring:
        # points: list[Point] = upload(
        #     client,
        #     "documents",
        #     embedding_model,
        #     ConfigSettings().embeddings,
        #     chunks,
        #     logger=logger,
        # )

        logger = LoglistLogger()
        encoding_model = EncodingModel.CONTENT
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )

        chunks = [
            Chunk(
                content="Test content",
                metadata={},
                uuid=generate_uuid("test-uuid-1"),
            )
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            client = QdrantClient(path=temp_dir)

            # Initialize collection first
            initialize_collection(
                client,
                "documents",
                embedding_model,
                ConfigSettings().embeddings,
                logger=logger,
            )

            # Upload as in example
            points = upload(
                client,
                "documents",
                embedding_model,
                ConfigSettings().embeddings,
                chunks,
                logger=logger,
            )

            self.assertEqual(len(points), len(chunks))

            client.close()

    def test_example_7_query(self):
        """Example 7: Query (lines 138-153)."""
        # Example code from docstring:
        # points: list[ScoredPoint] = query(
        #     client,
        #     "documents",
        #     embedding_model,
        #     ConfigSettings().embeddings,
        #     "What are the main uses of logistic regression?",
        #     limit=12,  # max number retrieved points
        #     payload=True,  # all payload fields
        #     logger=logger,
        # )
        #
        # # retrieve text
        # for pt in points:
        #     print(f"{pt.score}\n{pt.payload['page_content']}\n\n")

        logger = LoglistLogger()
        encoding_model = EncodingModel.CONTENT
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )

        chunks = [
            Chunk(
                content="Logistic regression is used for classification tasks.",
                metadata={},
                uuid=generate_uuid("test-uuid-1"),
            ),
            Chunk(
                content="Main uses include binary classification and probability estimation.",
                metadata={},
                uuid=generate_uuid("test-uuid-2"),
            ),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            client = QdrantClient(path=temp_dir)

            # Initialize and upload
            initialize_collection(
                client,
                "documents",
                embedding_model,
                ConfigSettings().embeddings,
                logger=logger,
            )
            upload(
                client,
                "documents",
                embedding_model,
                ConfigSettings().embeddings,
                chunks,
                logger=logger,
            )

            # Query as in example
            points = query(
                client,
                "documents",
                embedding_model,
                ConfigSettings().embeddings,
                "What are the main uses of logistic regression?",
                limit=12,  # max number retrieved points
                payload=True,  # all payload fields
                logger=logger,
            )

            # Should retrieve points
            self.assertGreater(len(points), 0)

            # Retrieve text as in example
            for pt in points:
                self.assertIsNotNone(pt.payload)
                self.assertIn('page_content', pt.payload)  # type: ignore
                # Verify we can access content (don't print in tests)
                content = pt.payload['page_content']  # type: ignore
                self.assertIsInstance(content, str)  # type: ignore
                self.assertGreater(len(content), 0)  # type: ignore

            client.close()


if __name__ == "__main__":
    unittest.main()
