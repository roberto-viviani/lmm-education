"""tests for vector_store_qdrant_langchain.py

NOTE: initialize collection tested in test_lmm_rag.py
"""

import unittest

from lmm_education.stores.chunks import (
    Chunk,
    blocks_to_chunks,
)
from lmm.markdown.parse_markdown import (
    blocklist_copy,
    TextBlock,
    HeaderBlock,
    MetadataBlock,
    HeadingBlock,
    Block,
)
from lmm.config.config import Settings, export_settings
from lmm.scan.scan_rag import scan_rag, ScanOpts
from lmm.scan.scan_split import scan_split
from lmm.scan.scan_keys import UUID_KEY, GROUP_UUID_KEY, QUESTIONS_KEY
from lmm_education.stores.vector_store_qdrant import (
    AsyncQdrantClient,
    QdrantEmbeddingModel,
    EncodingModel,
    Point,
    encoding_to_qdrantembedding_model,
    points_to_ids,
    points_to_text,
    ainitialize_collection,
    aupload,
)
from lmm_education.stores.langchain.vector_store_qdrant_langchain import (
    AsyncQdrantVectorStoreRetriever as Retriever,
)

from langchain_core.documents import Document

# A global client object (for now)
QDRANT_SOURCE = ":memory:"
client = AsyncQdrantClient(QDRANT_SOURCE)

header = HeaderBlock(content={'title': "Test blocklist"})
metadata = MetadataBlock(
    content={
        'questions': "What is the ingested test?",
        '~chat': "Some discussion",
        'summary': "The summary of the ingested text.",
    }
)
heading = HeadingBlock(level=2, content="Ingested text")
text = TextBlock(content="This is text following the heading.")
metadata2 = MetadataBlock(
    content={
        'questions': "Why is the sky blue?",
        'summary': "The summary of explanations about sky colour.",
    }
)
heading2 = HeadingBlock(level=2, content="Sky colour")
text2 = TextBlock(
    content="The sky is blue because of the oxygen composition of air."
)

blocks: list[Block] = [
    header,
    metadata,
    heading,
    text,
    metadata2,
    heading2,
    text2,
]
blocks = scan_rag(blocks, ScanOpts(textid=True, UUID=True))


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

    # ------ query ----------------------------------------------------

    async def test_query_NULL(self):
        encoding_model = EncodingModel.NONE
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        if not await ainitialize_collection(
            client, collection_name, embedding_model
        ):
            raise Exception("Could not initialize collection")
        points = await aupload(
            client, collection_name, embedding_model, chunks
        )
        uuids = points_to_ids(points)
        self.assertEqual(len(uuids), len(chunks))

        retriever: Retriever = Retriever(
            client, collection_name, embedding_model
        )
        results: list[Document] = await retriever.ainvoke(uuids[0])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].page_content, text.get_content())  # type: ignore

    async def test_query_CONTENT(self):
        encoding_model = EncodingModel.CONTENT
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        if not await ainitialize_collection(
            client, collection_name, embedding_model
        ):
            raise Exception("Could not initialize collection")
        ps = await aupload(
            client, collection_name, embedding_model, chunks
        )

        retriever: Retriever = Retriever(
            client, collection_name, embedding_model
        )
        results: list[Document] = await retriever.ainvoke(
            "What follows the heading"
        )
        self.assertEqual(len(results), len(ps))
        self.assertTrue(results[0].page_content)
        self.assertEqual(results[0].page_content, text.get_content())  # type: ignore

    async def test_query_CONTENT2(self):
        encoding_model = EncodingModel.CONTENT
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        if not await ainitialize_collection(
            client, collection_name, embedding_model
        ):
            raise Exception("Could not initialize collection")
        ps = await aupload(
            client, collection_name, embedding_model, chunks
        )

        retriever: Retriever = Retriever(
            client, collection_name, embedding_model
        )
        results: list[Document] = await retriever.ainvoke(
            "The oygen composition of the air"
        )
        self.assertEqual(len(results), len(ps))
        self.assertTrue(results[0].page_content)
        self.assertEqual(results[0].page_content, text2.get_content())  # type: ignore

    async def test_query_MERGED(self):
        encoding_model = EncodingModel.MERGED
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        if not await ainitialize_collection(
            client, collection_name, embedding_model
        ):
            raise Exception("Could not initialize collection")
        ps = await aupload(
            client, collection_name, embedding_model, chunks
        )

        retriever: Retriever = Retriever(
            client, collection_name, embedding_model
        )
        results: list[Document] = await retriever.ainvoke(
            "What follows the heading"
        )
        self.assertEqual(len(results), len(ps))
        self.assertTrue(results[0].page_content)
        self.assertEqual(results[0].page_content, text.get_content())  # type: ignore

    async def test_query_MERGED2(self):
        encoding_model = EncodingModel.MERGED
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        if not await ainitialize_collection(
            client, collection_name, embedding_model
        ):
            raise Exception("Could not initialize collection")
        ps = await aupload(
            client, collection_name, embedding_model, chunks
        )

        retriever: Retriever = Retriever(
            client, collection_name, embedding_model
        )
        results: list[Document] = await retriever.ainvoke(
            "What is the ingested text?"
        )
        self.assertEqual(len(results), len(ps))
        self.assertTrue(results[0].page_content)
        self.assertEqual(results[0].page_content, text.get_content())  # type: ignore

    async def test_query_MERGED3(self):
        encoding_model = EncodingModel.MERGED
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        if not await ainitialize_collection(
            client, collection_name, embedding_model
        ):
            raise Exception("Could not initialize collection")
        ps = await aupload(
            client, collection_name, embedding_model, chunks
        )

        retriever: Retriever = Retriever(
            client, collection_name, embedding_model
        )
        results: list[Document] = await retriever.ainvoke(
            "Why is the sky blue?"
        )
        self.assertEqual(len(results), len(ps))
        self.assertTrue(results[1].page_content)
        self.assertEqual(results[1].page_content, text.get_content())  # type: ignore

    async def test_query_SPARSE(self):
        encoding_model = EncodingModel.SPARSE
        chunks = blocks_to_chunks(
            blocks, encoding_model, [QUESTIONS_KEY]
        )
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        if not await ainitialize_collection(
            client, collection_name, embedding_model
        ):
            raise Exception("Could not initialize collection")
        await aupload(
            client,
            collection_name,
            embedding_model,  # type: ignore
            chunks,
        )

        retriever: Retriever = Retriever(
            client, collection_name, embedding_model
        )
        results: list[Document] = await retriever.ainvoke(
            "What is the ingested text?"
        )
        self.assertTrue(results[0].page_content)
        self.assertEqual(results[0].page_content, text.get_content())  # type: ignore

    async def test_query_SPARSE_CONTENT(self):
        encoding_model = EncodingModel.SPARSE_CONTENT
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        if not await ainitialize_collection(
            client, collection_name, embedding_model
        ):
            raise Exception("Could not initialize collection")
        ps = await aupload(
            client, collection_name, embedding_model, chunks
        )

        retriever: Retriever = Retriever(
            client, collection_name, embedding_model
        )
        results: list[Document] = await retriever.ainvoke(
            "What follows the heading"
        )
        self.assertEqual(len(results), len(ps))
        self.assertTrue(results[0].page_content)
        self.assertEqual(results[0].page_content, text.get_content())  # type: ignore

    async def test_query_SPARSE_CONTENT2(self):
        encoding_model = EncodingModel.SPARSE_CONTENT
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        if not await ainitialize_collection(
            client, collection_name, embedding_model
        ):
            raise Exception("Could not initialize collection")
        ps = await aupload(
            client, collection_name, embedding_model, chunks
        )
        retriever: Retriever = Retriever(
            client, collection_name, embedding_model
        )
        results: list[Document] = await retriever.ainvoke(
            "What is the ingested text?"
        )
        self.assertEqual(len(results), len(ps))
        self.assertTrue(results[0].page_content)
        self.assertEqual(results[0].page_content, text.get_content())  # type: ignore

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
        if not await ainitialize_collection(
            client, collection_name, embedding_model
        ):
            raise Exception("Could not initialize collection")
        ps = await aupload(
            client, collection_name, embedding_model, chunks
        )

        retriever: Retriever = Retriever(
            client, collection_name, embedding_model
        )
        results: list[Document] = await retriever.ainvoke(
            "Why is the sky blue?"
        )
        self.assertEqual(len(results), len(ps))
        self.assertTrue(results[1].page_content)
        self.assertEqual(results[1].page_content, text.get_content())  # type: ignore

    async def test_query_SPARSE_MERGED(self):
        encoding_model = EncodingModel.SPARSE_MERGED
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        if not await ainitialize_collection(
            client, collection_name, embedding_model
        ):
            raise Exception("Could not initialize collection")
        ps = await aupload(
            client, collection_name, embedding_model, chunks
        )

        retriever: Retriever = Retriever(
            client, collection_name, embedding_model
        )
        results: list[Document] = await retriever.ainvoke(
            "What follows the heading"
        )
        self.assertEqual(len(results), len(ps))
        self.assertTrue(results[0].page_content)
        self.assertEqual(results[0].page_content, text.get_content())  # type: ignore

    async def test_query_SPARSE_MERGED2(self):
        encoding_model = EncodingModel.SPARSE_MERGED
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        if not await ainitialize_collection(
            client, collection_name, embedding_model
        ):
            raise Exception("Could not initialize collection")
        ps = await aupload(
            client, collection_name, embedding_model, chunks
        )

        retriever: Retriever = Retriever(
            client, collection_name, embedding_model
        )
        results: list[Document] = await retriever.ainvoke(
            "What is the ingested text?"
        )
        self.assertEqual(len(results), len(ps))
        self.assertTrue(results[0].page_content)
        self.assertEqual(results[0].page_content, text.get_content())  # type: ignore

    async def test_query_SPARSE_MERGED3(self):
        encoding_model = EncodingModel.SPARSE_MERGED
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        if not await ainitialize_collection(
            client, collection_name, embedding_model
        ):
            raise Exception("Could not initialize collection")
        ps = await aupload(
            client, collection_name, embedding_model, chunks
        )

        retriever: Retriever = Retriever(
            client, collection_name, embedding_model
        )
        results: list[Document] = await retriever.ainvoke(
            "Why is the sky blue?"
        )
        self.assertEqual(len(results), len(ps))
        self.assertTrue(results[1].page_content)
        self.assertEqual(results[1].page_content, text.get_content())  # type: ignore


class TestQueryGrouped(unittest.IsolatedAsyncioTestCase):

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

    async def test_query(self):
        import lmm_education.stores.chunks as chk

        from langchain_text_splitters import (
            RecursiveCharacterTextSplitter,
        )

        # import long document, parsed as block list
        from .test_vector_store_qdrant_groups import blocklist

        from lmm_education.stores.langchain.vector_store_qdrant_langchain import (
            AsyncQdrantVectorStoreRetrieverGrouped,
        )

        COLLECTION_MAIN = "Main"
        COLLECTION_DOCS = "Main_docs"

        # init collections
        encoding_model_main: chk.EncodingModel = (
            chk.EncodingModel.CONTENT
        )
        embedding_model_main: QdrantEmbeddingModel = (
            encoding_to_qdrantembedding_model(encoding_model_main)
        )
        encoding_model_companion: chk.EncodingModel = (
            chk.EncodingModel.CONTENT
        )
        embedding_model_companion: QdrantEmbeddingModel = (
            encoding_to_qdrantembedding_model(encoding_model_main)
        )

        if not await ainitialize_collection(
            client, COLLECTION_MAIN, embedding_model_main
        ):
            raise Exception("Could not initialize main collection")
        if not await ainitialize_collection(
            client, COLLECTION_DOCS, embedding_model_companion
        ):
            raise Exception(
                "Could not initialize companion collection"
            )

        blocks: list[Block] = scan_rag(
            blocklist,
            ScanOpts(titles=True, textid=True, UUID=True),
        )

        # ingest text into companion collection
        companion_chunks: list[Chunk] = chk.blocks_to_chunks(
            blocks, encoding_model_companion
        )
        companion_points: list[Point] = await aupload(
            client,
            COLLECTION_DOCS,
            embedding_model_companion,
            companion_chunks,
        )
        self.assertTrue(len(companion_points) > 0)
        companion_uuids: list[str] = points_to_ids(companion_points)
        companion_texts: list[str] = points_to_text(companion_points)
        self.assertTrue(len(companion_uuids) > 0)
        self.assertTrue(len(companion_texts) > 0)

        # split text
        for b in blocks:
            if isinstance(b, MetadataBlock):
                if UUID_KEY in b.content.keys():
                    b.content[GROUP_UUID_KEY] = b.content.pop(
                        UUID_KEY
                    )
        blocks = scan_split(
            blocks,
            text_splitter=RecursiveCharacterTextSplitter(
                chunk_size=750,
                chunk_overlap=150,
                add_start_index=False,
            ),
        )
        blocks = scan_rag(
            blocks, ScanOpts(titles=True, textid=True, UUID=True)
        )

        # ingest chunks of splitted text
        chunks: list[Chunk] = chk.blocks_to_chunks(
            blocks, encoding_model_main
        )
        points: list[Point] = await aupload(
            client, COLLECTION_MAIN, embedding_model_main, chunks
        )
        self.assertLess(0, len(points))
        texts: list[str] = points_to_text(points)
        self.assertTrue(len(texts) > 0)

        # grouped query
        retriever = AsyncQdrantVectorStoreRetrieverGrouped(
            client,
            COLLECTION_MAIN,
            COLLECTION_DOCS,
            GROUP_UUID_KEY,
            3,
            embedding_model_main,
        )
        results: list[Document] = await retriever.ainvoke(
            "How can I estimate the predicted depressiveness from this model?"
        )

        # tests
        self.assertLess(0, len(results))
        self.assertTrue(results[0].page_content in companion_texts)


if __name__ == '__main__':
    unittest.main()
