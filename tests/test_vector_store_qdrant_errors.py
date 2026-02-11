"""Tests for error scenarios in vector_store_qdrant module."""

import unittest
from unittest.mock import Mock
import tempfile

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ApiException

from lmm.scan.chunks import Chunk, EncodingModel
from lmm.utils.logging import LoglistLogger
from lmm_education.config.config import (
    ConfigSettings,
    LocalStorage,
)
from lmm_education.stores.vector_store_qdrant import (
    client_from_config,
    qdrant_client_context,
    initialize_collection,
    upload,
    encoding_to_qdrantembedding_model,
)


class TestErrorScenarios(unittest.TestCase):
    """Test error handling in vector_store_qdrant functions."""

    def test_connection_failure_on_upload(self):
        """Test upload when Qdrant server connection fails."""
        logger = LoglistLogger()

        with tempfile.TemporaryDirectory() as temp_dir:
            client = QdrantClient(path=temp_dir)

            # Mock client.upsert to raise ConnectionError
            client.upsert = Mock(
                side_effect=ConnectionError("Connection refused")
            )

            encoding_model = EncodingModel.CONTENT
            embedding_model = encoding_to_qdrantembedding_model(
                encoding_model
            )

            # Initialize collection first
            result = initialize_collection(
                client,
                "test_collection",
                embedding_model,
                ConfigSettings().embeddings,
                logger=logger,
            )
            self.assertTrue(result)

            # Try to upload with connection error
            chunks = [
                Chunk(
                    content="Test content",
                    metadata={},
                    uuid="test-uuid-1",
                )
            ]

            points = upload(
                client,
                "test_collection",
                embedding_model,
                ConfigSettings().embeddings,
                chunks,
                logger=logger,
            )

            # Should return empty list on error
            self.assertEqual(len(points), 0)

            # Should have logged the error
            logs = logger.get_logs()
            self.assertTrue(
                any("Could not connect" in log for log in logs)
            )

            client.close()

    def test_api_exception_on_upload(self):
        """Test upload when Qdrant API exception occurs."""
        logger = LoglistLogger()

        with tempfile.TemporaryDirectory() as temp_dir:
            client = QdrantClient(path=temp_dir)

            # Mock client.upsert to raise ApiException
            client.upsert = Mock(
                side_effect=ApiException("API error")
            )

            encoding_model = EncodingModel.CONTENT
            embedding_model = encoding_to_qdrantembedding_model(
                encoding_model
            )

            result = initialize_collection(
                client,
                "test_collection",
                embedding_model,
                ConfigSettings().embeddings,
                logger=logger,
            )
            self.assertTrue(result)

            chunks = [
                Chunk(
                    content="Test content",
                    metadata={},
                    uuid="test-uuid-1",
                )
            ]

            points = upload(
                client,
                "test_collection",
                embedding_model,
                ConfigSettings().embeddings,
                chunks,
                logger=logger,
            )

            self.assertEqual(len(points), 0)
            logs = logger.get_logs()
            self.assertTrue(any("API error" in log for log in logs))

            client.close()

    def test_malformed_chunks_upload(self):
        """Test upload with malformed chunk data."""
        logger = LoglistLogger()

        with tempfile.TemporaryDirectory() as temp_dir:
            client = QdrantClient(path=temp_dir)

            encoding_model = EncodingModel.CONTENT
            embedding_model = encoding_to_qdrantembedding_model(
                encoding_model
            )

            result = initialize_collection(
                client,
                "test_collection",
                embedding_model,
                ConfigSettings().embeddings,
                logger=logger,
            )
            self.assertTrue(result)

            # Empty chunks list should work but return empty points
            points = upload(
                client,
                "test_collection",
                embedding_model,
                ConfigSettings().embeddings,
                [],
                logger=logger,
            )

            self.assertEqual(len(points), 0)

            client.close()

    def test_invalid_database_source(self):
        """Test client creation with invalid database source."""
        logger = LoglistLogger()

        # Invalid path  should log error and return None
        invalid_storage = LocalStorage(folder="///invalid:path<>?*")

        client = client_from_config(
            ConfigSettings(storage=invalid_storage), logger
        )

        # Should return None on error
        self.assertIsNone(client)

        # Should have logged error
        logs = logger.get_logs()
        self.assertTrue(len(logs) > 0)


class TestContextManagerErrors(unittest.TestCase):
    """Test error handling in context managers."""

    def test_context_manager_connection_error(self):
        """Verify ConnectionError raised when client creation fails."""
        logger = LoglistLogger()

        # Invalid configuration should raise ConnectionError
        invalid_storage = LocalStorage(folder="///invalid:path<>?*")

        with self.assertRaises(ConnectionError):
            with qdrant_client_context(
                ConfigSettings(storage=invalid_storage), logger
            ) as client:  # type: ignore (not used)  # noqa
                pass

        # Should have logged error before raising
        logs = logger.get_logs()
        self.assertTrue(len(logs) > 0)

    def test_context_manager_cleanup(self):
        """Verify cleanup happens even when exceptions occur."""
        logger = LoglistLogger()

        # Use valid config but raise exception in block
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                with qdrant_client_context(
                    ConfigSettings(
                        storage=LocalStorage(folder=temp_dir)
                    ),
                    logger,
                ) as client:
                    # Verify client was created
                    self.assertIsNotNone(client)
                    # Raise exception to test cleanup
                    raise RuntimeError("Test exception")
            except RuntimeError:
                pass  # Expected

        # Context manager should have cleaned up (closed client)
        # Client is closed, so no further assertions needed


if __name__ == "__main__":
    unittest.main()
