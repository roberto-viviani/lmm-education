import unittest

from qdrant_client import AsyncQdrantClient, QdrantClient

from lmm_education.stores.vector_store_qdrant_context import (
    global_client_from_config,
    global_async_client_from_config,
    qdrant_clients,
    qdrant_async_clients,
    global_async_clients_close,
    global_clients_close,
)

# pyright: basic

class TestQdrantGlobal(unittest.TestCase):

    def test_sync(self):
        qdrant_clients.clear()
        self.assertEqual(len(qdrant_clients), 0)
        client: QdrantClient = global_client_from_config(":memory:")
        self.assertIsNotNone(client)
        self.assertEqual(len(qdrant_clients), 1)

        # get a second client
        client2: QdrantClient = global_client_from_config(":memory:")
        self.assertEqual(len(qdrant_clients), 1)
        self.assertIs(client, client2)

        # test destruction
        global_clients_close()
        self.assertEqual(len(qdrant_clients), 0)

    def test_async(self):
        qdrant_async_clients.clear()
        self.assertEqual(len(qdrant_async_clients), 0)
        client: AsyncQdrantClient = global_async_client_from_config(
            ":memory:"
        )
        self.assertIsNotNone(client)
        self.assertEqual(len(qdrant_async_clients), 1)

        # get a second client
        client2: AsyncQdrantClient = global_async_client_from_config(
            ":memory:"
        )
        self.assertEqual(len(qdrant_async_clients), 1)
        self.assertIs(client, client2)
        global_async_clients_close()
        self.assertEqual(len(qdrant_async_clients), 0)

    def test_async_config(self):
        qdrant_async_clients.clear()
        self.assertEqual(len(qdrant_async_clients), 0)
        client: AsyncQdrantClient = global_async_client_from_config()
        self.assertIsNotNone(client)
        self.assertEqual(len(qdrant_async_clients), 1)

        # get a second client
        client2: AsyncQdrantClient = global_async_client_from_config()
        self.assertEqual(len(qdrant_async_clients), 1)
        self.assertIs(client, client2)

    def test_sync_config(self):
        """Test sync client with default config from config.toml."""
        qdrant_clients.clear()
        self.assertEqual(len(qdrant_clients), 0)
        client: QdrantClient = global_client_from_config()
        self.assertIsNotNone(client)
        self.assertEqual(len(qdrant_clients), 1)

        # get a second client - should be the same instance
        client2: QdrantClient = global_client_from_config()
        self.assertEqual(len(qdrant_clients), 1)
        self.assertIs(client, client2)

        # cleanup
        global_clients_close()
        self.assertEqual(len(qdrant_clients), 0)

    def test_multiple_sources_sync(self):
        """Test that different database sources are cached separately."""
        from lmm_education.config.config import LocalStorage

        qdrant_clients.clear()
        self.assertEqual(len(qdrant_clients), 0)

        # Create clients for different sources
        client1: QdrantClient = global_client_from_config(":memory:")
        client2: QdrantClient = global_client_from_config(
            LocalStorage(folder="./test_storage")
        )

        # Should be different client instances
        self.assertIsNot(client1, client2)
        # Should have 2 cached clients
        self.assertEqual(len(qdrant_clients), 2)

        # Getting same source again should return cached instance
        client1_again: QdrantClient = global_client_from_config(":memory:")
        self.assertIs(client1, client1_again)
        self.assertEqual(len(qdrant_clients), 2)

        # cleanup
        global_clients_close()
        self.assertEqual(len(qdrant_clients), 0)

    def test_multiple_sources_async(self):
        """Test that different database sources are cached separately for async."""
        from lmm_education.config.config import LocalStorage

        qdrant_async_clients.clear()
        self.assertEqual(len(qdrant_async_clients), 0)

        # Create clients for different sources
        client1: AsyncQdrantClient = global_async_client_from_config(
            ":memory:"
        )
        client2: AsyncQdrantClient = global_async_client_from_config(
            LocalStorage(folder="./test_storage")
        )

        # Should be different client instances
        self.assertIsNot(client1, client2)
        # Should have 2 cached clients
        self.assertEqual(len(qdrant_async_clients), 2)

        # Getting same source again should return cached instance
        client1_again: AsyncQdrantClient = global_async_client_from_config(
            ":memory:"
        )
        self.assertIs(client1, client1_again)
        self.assertEqual(len(qdrant_async_clients), 2)

        # cleanup
        global_async_clients_close()
        self.assertEqual(len(qdrant_async_clients), 0)

    def test_sync_client_creation_failure(self):
        """Test ValueError is raised when sync client creation fails."""
        from unittest.mock import patch
        from lmm.utils.logging import ConsoleLogger

        logger = ConsoleLogger()

        # Mock client_from_config to return None (simulating failure)
        with patch(
            "lmm_education.stores.vector_store_qdrant_context.client_from_config",
            return_value=None,
        ):
            with self.assertRaises(ValueError) as context:
                global_client_from_config(":memory:", logger=logger)

            # Verify the error message mentions synchronous client
            self.assertIn("synchronous", str(context.exception))

    def test_async_client_creation_failure(self):
        """Test ValueError is raised when async client creation fails."""
        from unittest.mock import patch
        from lmm.utils.logging import ConsoleLogger

        logger = ConsoleLogger()

        # Mock async_client_from_config to return None (simulating failure)
        with patch(
            "lmm_education.stores.vector_store_qdrant_context.async_client_from_config",
            return_value=None,
        ):
            with self.assertRaises(ValueError) as context:
                global_async_client_from_config(":memory:", logger=logger)

            # Verify the error message mentions asynchronous client
            self.assertIn("asynchronous", str(context.exception))

    def test_config_load_failure_sync(self):
        """Test ValueError when config cannot be loaded for sync client."""
        from unittest.mock import patch
        from lmm.utils.logging import ConsoleLogger

        logger = ConsoleLogger()

        # Mock load_settings to return None (simulating config load failure)
        with patch(
            "lmm_education.stores.vector_store_qdrant_context.load_settings",
            return_value=None,
        ):
            with self.assertRaises(ValueError) as context:
                global_client_from_config(None, logger=logger)

            # Verify error message
            self.assertIn("Could not read settings", str(context.exception))

    def test_config_load_failure_async(self):
        """Test ValueError when config cannot be loaded for async client."""
        from unittest.mock import patch
        from lmm.utils.logging import ConsoleLogger

        logger = ConsoleLogger()

        # Mock load_settings to return None (simulating config load failure)
        with patch(
            "lmm_education.stores.vector_store_qdrant_context.load_settings",
            return_value=None,
        ):
            with self.assertRaises(ValueError) as context:
                global_async_client_from_config(None, logger=logger)

            # Verify error message
            self.assertIn("Could not read settings", str(context.exception))

    def test_destructors_called_on_clear(self):
        """Test that destructors are called when clients are cleared."""
        from unittest.mock import patch

        qdrant_clients.clear()

        # Create a client
        client: QdrantClient = global_client_from_config(":memory:")
        self.assertIsNotNone(client)

        # Mock the close method to verify it's called
        with patch.object(client, "close") as mock_close:
            # Clear the cache, which should trigger the destructor
            global_clients_close()

            # Verify close was called
            mock_close.assert_called_once()

        self.assertEqual(len(qdrant_clients), 0)


if __name__ == "__main__":
    unittest.main()

