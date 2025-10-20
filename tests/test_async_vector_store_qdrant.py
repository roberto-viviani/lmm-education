import unittest
import tempfile
import asyncio

from lmm_education.stores.chunks import (
    Chunk,
    blocks_to_chunks,
)
from lmm.markdown.parse_markdown import blocklist_copy, TextBlock
from lmm.config.config import Settings, export_settings
from lmm.scan.scan_keys import TITLES_KEY, QUESTIONS_KEY
from lmm.scan.scan_rag import scan_rag, ScanOpts
from lmm_education.config.config import AnnotationModel
from lmm_education.stores.vector_store_qdrant import (
    AsyncQdrantClient,
    QdrantEmbeddingModel,
    EncodingModel,
    ScoredPoint,
    encoding_to_qdrantembedding_model,
    points_to_ids,
    points_to_text,
    ainitialize_collection,
    aupload,
    aquery,
)
from qdrant_client.models import PointStruct

from .test_vector_store_qdrant import blocks, text, text2


class TestQuery(unittest.IsolatedAsyncioTestCase):

    # detup and teardown replace config.toml to avoid
    # calling the language model server
    original_settings = Settings()

    @classmethod
    def setUpClass(cls):
        settings = Settings(
            major={'model': "Debug/debug"},
            minor={'model': "Debug/debug"},
            aux={'model': "Debug/debug"},
            embeddings={
                'dense_model': "SentenceTransformers/distiluse-base-multilingual-cased-v1"
            },
        )
        export_settings(settings)

    @classmethod
    def tearDownClass(cls):
        settings = cls.original_settings
        export_settings(settings)

    def setUp(self):
        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir_path = self.temp_dir.name

        # The synchronous client will load data before querying
        # self.synch_client = QdrantClient(path=self.test_dir_path)
        self.async_client = AsyncQdrantClient(path=self.test_dir_path)

    def tearDown(self):
        # self.synch_client.close()

        promise = self.async_client.close()
        loop = asyncio.get_event_loop()
        task = loop.create_task(promise)
        loop.run_until_complete(task)

        self.temp_dir.cleanup()

    async def _create_collection(
        self,
        collection_name: str,
        chunks: list[Chunk],
        embedding_model: QdrantEmbeddingModel,
    ) -> list[PointStruct]:
        if not await ainitialize_collection(
            self.async_client, collection_name, embedding_model
        ):
            raise Exception("Could not initialize collection")
        points = await aupload(
            self.async_client,
            collection_name,
            embedding_model,
            chunks,
        )
        return points

    async def test_query_NULL(self):
        encoding_model = EncodingModel.NONE
        chunks = blocks_to_chunks(
            blocks,
            encoding_model,
            AnnotationModel(inherited_properties=[TITLES_KEY]),
        )
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        points = await self._create_collection(
            collection_name, chunks, embedding_model
        )
        self.assertEqual(len(points), len(chunks))
        uuids = points_to_ids(points)
        self.assertEqual(len(uuids), len(chunks))
        results: list[ScoredPoint] = await aquery(
            self.async_client,
            collection_name,
            embedding_model,
            uuids[0],
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].payload['page_content'], text.get_content())  # type: ignore

    async def test_query_CONTENT(self):
        encoding_model = EncodingModel.CONTENT
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        ps = await self._create_collection(
            collection_name, chunks, embedding_model
        )
        results: list[ScoredPoint] = await aquery(
            self.async_client,
            collection_name,
            embedding_model,
            "What follows the heading",
        )
        self.assertEqual(len(results), len(ps))
        self.assertIsNotNone(results[0].payload)
        self.assertEqual(results[0].payload['page_content'], text.get_content())  # type: ignore

    async def test_query_CONTENT2(self):
        encoding_model = EncodingModel.CONTENT
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        ps = await self._create_collection(
            collection_name, chunks, embedding_model
        )
        results: list[ScoredPoint] = await aquery(
            self.async_client,
            collection_name,
            embedding_model,
            "The oygen composition of the air",
        )
        self.assertEqual(len(results), len(ps))
        self.assertEqual(results[0].payload['page_content'], text2.get_content())  # type: ignore

    async def test_query_MERGED(self):
        encoding_model = EncodingModel.MERGED
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        ps = await self._create_collection(
            collection_name, chunks, embedding_model
        )
        results: list[ScoredPoint] = await aquery(
            self.async_client,
            collection_name,
            embedding_model,
            "What follows the heading",
        )
        self.assertEqual(len(results), len(ps))
        self.assertEqual(results[0].payload['page_content'], text.get_content())  # type: ignore

    async def test_query_MERGED2(self):
        encoding_model = EncodingModel.MERGED
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        ps = await self._create_collection(
            collection_name, chunks, embedding_model
        )
        results: list[ScoredPoint] = await aquery(
            self.async_client,
            collection_name,
            embedding_model,
            "What is the ingested text?",
        )
        self.assertEqual(len(results), len(ps))
        self.assertEqual(results[0].payload['page_content'], text.get_content())  # type: ignore

    async def test_query_MERGED3(self):
        encoding_model = EncodingModel.MERGED
        chunks = blocks_to_chunks(
            blocks, encoding_model, [QUESTIONS_KEY]
        )
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        ps = await self._create_collection(
            collection_name, chunks, embedding_model
        )
        results: list[ScoredPoint] = await aquery(
            self.async_client,
            collection_name,
            embedding_model,
            "Why is the sky blue?",
        )
        self.assertEqual(len(results), len(ps))
        self.assertEqual(results[1].payload['page_content'], text.get_content())  # type: ignore

    async def test_query_SPARSE(self):
        encoding_model = EncodingModel.SPARSE
        chunks = blocks_to_chunks(
            blocks, encoding_model, [QUESTIONS_KEY]
        )
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        await self._create_collection(
            collection_name, chunks, embedding_model
        )
        results: list[ScoredPoint] = await aquery(
            self.async_client,
            collection_name,
            embedding_model,
            "What is the ingested text?",
        )
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0].payload['page_content'], text.get_content())  # type: ignore

    async def test_query_SPARSE_CONTENT(self):
        encoding_model = EncodingModel.SPARSE_CONTENT
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        ps = await self._create_collection(
            collection_name, chunks, embedding_model
        )
        results: list[ScoredPoint] = await aquery(
            self.async_client,
            collection_name,
            embedding_model,
            "What follows the heading",
        )
        self.assertEqual(len(results), len(ps))
        self.assertEqual(results[0].payload['page_content'], text.get_content())  # type: ignore

    async def test_query_SPARSE_CONTENT2(self):
        encoding_model = EncodingModel.SPARSE_CONTENT
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        ps = await self._create_collection(
            collection_name, chunks, embedding_model
        )
        results: list[ScoredPoint] = await aquery(
            self.async_client,
            collection_name,
            embedding_model,
            "What is the ingested text?",
        )
        self.assertEqual(len(results), len(ps))
        self.assertEqual(results[0].payload['page_content'], text.get_content())  # type: ignore

    async def test_query_SPARSE_CONTENT3(self):
        encoding_model = EncodingModel.SPARSE_CONTENT
        blocklist = scan_rag(
            blocklist_copy(blocks), ScanOpts(textid=True, UUID=True)
        )
        chunks = blocks_to_chunks(blocklist, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        ps = await self._create_collection(
            collection_name, chunks, embedding_model
        )
        results: list[ScoredPoint] = await aquery(
            self.async_client,
            collection_name,
            embedding_model,
            "Why is the sky blue?",
        )
        self.assertEqual(len(results), len(ps))
        self.assertEqual(results[1].payload['page_content'], text.get_content())  # type: ignore

    async def test_query_SPARSE_MERGED(self):
        encoding_model = EncodingModel.SPARSE_MERGED
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        ps = await self._create_collection(
            collection_name, chunks, embedding_model
        )
        results: list[ScoredPoint] = await aquery(
            self.async_client,
            collection_name,
            embedding_model,
            "What follows the heading",
        )
        self.assertEqual(len(results), len(ps))
        self.assertEqual(results[0].payload['page_content'], text.get_content())  # type: ignore

    async def test_query_SPARSE_MERGED2(self):
        encoding_model = EncodingModel.SPARSE_MERGED
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        ps = await self._create_collection(
            collection_name, chunks, embedding_model
        )
        results: list[ScoredPoint] = await aquery(
            self.async_client,
            collection_name,
            embedding_model,
            "What is the ingested text?",
        )
        self.assertEqual(len(results), len(ps))
        self.assertEqual(results[0].payload['page_content'], text.get_content())  # type: ignore

    async def test_query_SPARSE_MERGED3(self):
        encoding_model = EncodingModel.SPARSE_MERGED
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        ps = await self._create_collection(
            collection_name, chunks, embedding_model
        )
        results: list[ScoredPoint] = await aquery(
            self.async_client,
            collection_name,
            embedding_model,
            "Why is the sky blue?",
        )
        self.assertEqual(len(results), len(ps))
        self.assertEqual(results[1].payload['page_content'], text.get_content())  # type: ignore

    async def test_query_to_uuids(self):
        encoding_model = EncodingModel.SPARSE_MERGED
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        ps = await self._create_collection(
            collection_name, chunks, embedding_model
        )
        self.assertListEqual(
            [c.uuid for c in chunks], [str(p.id) for p in ps]
        )
        results: list[ScoredPoint] = await aquery(
            self.async_client,
            collection_name,
            embedding_model,
            "What follows the heading",
        )
        self.assertEqual(len(results), len(ps))
        self.assertListEqual(
            [c.uuid for c in chunks], points_to_ids(results)
        )

    async def test_query_to_text(self):
        encoding_model = EncodingModel.SPARSE_MERGED
        textlist = [
            b.get_content()
            for b in blocks
            if isinstance(b, TextBlock)
        ]
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        ps = await self._create_collection(
            collection_name, chunks, embedding_model
        )
        self.assertListEqual([c.content for c in chunks], textlist)
        results: list[ScoredPoint] = await aquery(
            self.async_client,
            collection_name,
            embedding_model,
            "What follows the heading",
        )
        self.assertEqual(len(results), len(ps))
        self.assertListEqual(textlist, points_to_text(results))


if __name__ == '__main__':
    unittest.main()
