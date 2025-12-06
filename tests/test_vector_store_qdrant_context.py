import unittest

from qdrant_client import AsyncQdrantClient, QdrantClient

from lmm_education.stores.vector_store_qdrant_context import (
    global_client_from_config,
    global_async_client_from_config,
    qdrant_clients,
    qdrant_async_clients,
)


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
        qdrant_async_clients.clear()

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


if __name__ == "__main__":
    unittest.main()
