"""tests for vector_store_qdrant.py

NOTE: initialize collection tested in test_lmm_rag.py
"""

# flake8: noqa

import unittest

from lmm.markdown.parse_markdown import *
from lmm_education.stores.chunks import *
from lmm_education.stores.vector_store_qdrant import *
from lmm.config.config import Settings, export_settings

# A global client object (for now)
QDRANT_SOURCE = ":memory:"
COLLECTION_MAIN = "Main"
COLLECTION_DOCS = "Main_docs"
client = QdrantClient(QDRANT_SOURCE)

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

sets: Settings = Settings()


def setUpModule() -> None:
    pass


def tearDownModule():
    # export_settings(sets)
    pass


class TestInitialization(unittest.TestCase):

    sets: Settings

    def test_encoding_none(self):
        encoding_model = EncodingModel.NONE
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        result = initialize_collection(
            client, collection_name, embedding_model
        )
        self.assertTrue(
            result,
            "init_collection should return True for encoding model",
        )

    def test_encoding_content(self):
        encoding_model = EncodingModel.CONTENT
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        result = initialize_collection(
            client, collection_name, embedding_model
        )
        self.assertTrue(
            result,
            "init_collection should return True for encoding model",
        )

    def test_encoding_merged(self):
        encoding_model = EncodingModel.MERGED
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        result = initialize_collection(
            client, collection_name, embedding_model
        )
        self.assertTrue(
            result,
            "init_collection should return True for encoding model",
        )

    def test_encoding_multivector(self):
        encoding_model = EncodingModel.MULTIVECTOR
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        result = initialize_collection(
            client, collection_name, embedding_model
        )
        self.assertTrue(
            result,
            "init_collection should return True for encoding model",
        )

    def test_encoding_sparse(self):
        encoding_model = EncodingModel.SPARSE
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        result = initialize_collection(
            client, collection_name, embedding_model
        )
        self.assertTrue(
            result,
            "init_collection should return True for encoding model",
        )

    def test_encoding_sparse_merged(self):
        encoding_model = EncodingModel.SPARSE_MERGED
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        result = initialize_collection(
            client, collection_name, embedding_model
        )
        self.assertTrue(
            result,
            "init_collection should return True for encoding model",
        )

    def test_encoding_sparse_content(self):
        encoding_model = EncodingModel.SPARSE_CONTENT
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        result = initialize_collection(
            client, collection_name, embedding_model
        )
        self.assertTrue(
            result,
            "init_collection should return True for encoding model",
        )

    def test_encoding_sparse_multivector(self):
        encoding_model = EncodingModel.SPARSE_MULTIVECTOR
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        result = initialize_collection(
            client, collection_name, embedding_model
        )
        self.assertTrue(
            result,
            "init_collection should return True for encoding model",
        )


class TestIngestionAndQuery(unittest.TestCase):

    # ------ ingestion ------------------------------------------------
    def test_ingestion_empty(self):
        encoding_model = EncodingModel.NONE
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        points = chunks_to_points([], embedding_model)
        self.assertEqual(len(points), 0)
        flag = initialize_collection(
            client, collection_name, embedding_model
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(client, collection_name, embedding_model, [])
        self.assertEqual(len(ps), 0)

    def test_ingestion_nontext(self):
        # blocklist w/o text blocks
        chunks = blocks_to_chunks(
            [header, metadata, heading], EncodingModel.CONTENT
        )
        self.assertEqual(len(chunks), 0)
        encoding_model = EncodingModel.CONTENT
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        points = chunks_to_points(chunks, embedding_model)
        self.assertEqual(len(chunks), len(points))
        ps: list[Point] = upload(
            client, collection_name, embedding_model, chunks
        )
        self.assertEqual(len(chunks), len(ps))
        for p, u in zip(points, points_to_ids(ps)):
            self.assertEqual(p.id, u)

    def test_ingestion_NULL(self):
        chunks = blocks_to_chunks(blocks, EncodingModel.NONE)
        self.assertEqual(len(chunks), 2)
        chunk = chunks[0]
        self.assertEqual(chunk.dense_encoding, "")
        encoding_model = EncodingModel.NONE
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        points = chunks_to_points(chunks, embedding_model)
        self.assertEqual(len(chunks), len(points))
        flag = initialize_collection(
            client, collection_name, embedding_model
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps: list[Point] = upload(
            client, collection_name, embedding_model, chunks
        )
        self.assertEqual(len(chunks), len(ps))
        for p, u in zip(points, points_to_ids(ps)):
            self.assertEqual(p.id, u)

    def test_ingestion_CONTENT(self):
        # just one text here
        shortblocks = blocks[:5]
        chunks = blocks_to_chunks(shortblocks, EncodingModel.CONTENT)
        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.dense_encoding, text.get_content())
        encoding_model = EncodingModel.CONTENT
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        points = chunks_to_points(chunks, embedding_model)
        self.assertEqual(len(chunks), len(points))
        flag = initialize_collection(
            client, collection_name, embedding_model
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps: list[Point] = upload(
            client, collection_name, embedding_model, chunks
        )
        self.assertEqual(len(chunks), len(ps))
        for p, u in zip(points, points_to_ids(ps)):
            self.assertEqual(p.id, u)

    def test_ingestion_MERGED(self):
        chunks = blocks_to_chunks(blocks, EncodingModel.MERGED)
        self.assertEqual(len(chunks), 2)
        chunk = chunks[0]
        self.assertEqual(
            chunk.dense_encoding,
            chunk.annotations + ": " + text.get_content(),
        )
        encoding_model = EncodingModel.MERGED
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        points = chunks_to_points(chunks, embedding_model)
        self.assertEqual(len(chunks), len(points))
        flag = initialize_collection(
            client, collection_name, embedding_model
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps: list[Point] = upload(
            client, collection_name, embedding_model, chunks
        )
        self.assertEqual(len(chunks), len(ps))
        for p, u in zip(points, points_to_ids(ps)):
            self.assertEqual(p.id, u)

    def test_ingestion_SPARSE(self):
        chunks = blocks_to_chunks(blocks, EncodingModel.SPARSE)
        self.assertEqual(len(chunks), 2)
        chunk = chunks[0]
        self.assertEqual(chunk.sparse_encoding, chunk.annotations)
        encoding_model = EncodingModel.SPARSE
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        points = chunks_to_points(chunks, embedding_model)
        self.assertEqual(len(chunks), len(points))
        flag = initialize_collection(
            client, collection_name, embedding_model
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps: list[Point] = upload(
            client, collection_name, embedding_model, chunks
        )
        self.assertEqual(len(chunks), len(ps))
        for p, u in zip(points, points_to_ids(ps)):
            self.assertEqual(p.id, u)

    def test_ingestion_SPARSE_CONTENT(self):
        chunks = blocks_to_chunks(
            blocks, EncodingModel.SPARSE_CONTENT
        )
        self.assertEqual(len(chunks), 2)
        chunk = chunks[0]
        self.assertEqual(chunk.sparse_encoding, chunk.annotations)
        self.assertEqual(chunk.dense_encoding, text.get_content())
        encoding_model = EncodingModel.SPARSE_CONTENT
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        points = chunks_to_points(chunks, embedding_model)
        self.assertEqual(len(chunks), len(points))
        flag = initialize_collection(
            client, collection_name, embedding_model
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(client, collection_name, embedding_model, chunks)
        self.assertEqual(len(chunks), len(ps))

    def test_ingestion_SPARSE_MERGED(self):
        chunks = blocks_to_chunks(blocks, EncodingModel.SPARSE_MERGED)
        self.assertEqual(len(chunks), 2)
        chunk = chunks[0]
        self.assertEqual(chunk.sparse_encoding, chunk.annotations)
        self.assertEqual(
            chunk.dense_encoding,
            chunk.annotations + ": " + text.get_content(),
        )
        encoding_model = EncodingModel.SPARSE_MERGED
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        points = chunks_to_points(chunks, embedding_model)
        self.assertEqual(len(chunks), len(points))
        flag = initialize_collection(
            client, collection_name, embedding_model
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(client, collection_name, embedding_model, chunks)
        self.assertEqual(len(chunks), len(ps))

    # ------ query ----------------------------------------------------

    def test_query_NULL(self):
        encoding_model = EncodingModel.NONE
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        flag = initialize_collection(
            client, collection_name, embedding_model
        )
        self.assertTrue(flag, "Could not initialize collection")
        points = upload(
            client, collection_name, embedding_model, chunks
        )
        uuids = points_to_ids(points)
        self.assertEqual(len(uuids), len(chunks))
        results: list[ScoredPoint] = query(
            client, collection_name, embedding_model, uuids[0]
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].payload['page_content'], text.get_content())  # type: ignore

    def test_query_CONTENT(self):
        encoding_model = EncodingModel.CONTENT
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        flag = initialize_collection(
            client, collection_name, embedding_model
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(client, collection_name, embedding_model, chunks)
        results: list[ScoredPoint] = query(
            client,
            collection_name,
            embedding_model,
            "What follows the heading",
        )
        self.assertEqual(len(results), len(ps))
        self.assertIsNotNone(results[0].payload)
        self.assertEqual(results[0].payload['page_content'], text.get_content())  # type: ignore

    def test_query_CONTENT2(self):
        encoding_model = EncodingModel.CONTENT
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        flag = initialize_collection(
            client, collection_name, embedding_model
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(client, collection_name, embedding_model, chunks)
        results: list[ScoredPoint] = query(
            client,
            collection_name,
            embedding_model,
            "The oygen composition of the air",
        )
        self.assertEqual(len(results), len(ps))
        self.assertEqual(results[0].payload['page_content'], text2.get_content())  # type: ignore

    def test_query_MERGED(self):
        encoding_model = EncodingModel.MERGED
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        flag = initialize_collection(
            client, collection_name, embedding_model
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(client, collection_name, embedding_model, chunks)
        results: list[ScoredPoint] = query(
            client,
            collection_name,
            embedding_model,
            "What follows the heading",
        )
        self.assertEqual(len(results), len(ps))
        self.assertEqual(results[0].payload['page_content'], text.get_content())  # type: ignore

    def test_query_MERGED2(self):
        encoding_model = EncodingModel.MERGED
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        flag = initialize_collection(
            client, collection_name, embedding_model
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(client, collection_name, embedding_model, chunks)
        results: list[ScoredPoint] = query(
            client,
            collection_name,
            embedding_model,
            "What is the ingested text?",
        )
        self.assertEqual(len(results), len(ps))
        self.assertEqual(results[0].payload['page_content'], text.get_content())  # type: ignore

    def test_query_MERGED3(self):
        encoding_model = EncodingModel.MERGED
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        flag = initialize_collection(
            client, collection_name, embedding_model
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(client, collection_name, embedding_model, chunks)
        results: list[ScoredPoint] = query(
            client,
            collection_name,
            embedding_model,
            "Why is the sky blue?",
        )
        self.assertEqual(len(results), len(ps))
        self.assertEqual(results[1].payload['page_content'], text.get_content())  # type: ignore

    def test_query_SPARSE(self):
        encoding_model = EncodingModel.SPARSE
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        flag = initialize_collection(
            client, collection_name, embedding_model
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(
            client,
            collection_name,
            embedding_model,  # type: ignore
            chunks,
        )
        results: list[ScoredPoint] = query(
            client,
            collection_name,
            embedding_model,
            "What is the ingested text?",
        )
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0].payload['page_content'], text.get_content())  # type: ignore

    def test_query_SPARSE_CONTENT(self):
        encoding_model = EncodingModel.SPARSE_CONTENT
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        flag = initialize_collection(
            client, collection_name, embedding_model
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(client, collection_name, embedding_model, chunks)
        results: list[ScoredPoint] = query(
            client,
            collection_name,
            embedding_model,
            "What follows the heading",
        )
        self.assertEqual(len(results), len(ps))
        self.assertEqual(results[0].payload['page_content'], text.get_content())  # type: ignore

    def test_query_SPARSE_CONTENT2(self):
        encoding_model = EncodingModel.SPARSE_CONTENT
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        flag = initialize_collection(
            client, collection_name, embedding_model
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(client, collection_name, embedding_model, chunks)
        results: list[ScoredPoint] = query(
            client,
            collection_name,
            embedding_model,
            "What is the ingested text?",
        )
        self.assertEqual(len(results), len(ps))
        self.assertEqual(results[0].payload['page_content'], text.get_content())  # type: ignore

    def test_query_SPARSE_CONTENT3(self):
        encoding_model = EncodingModel.SPARSE_CONTENT
        blocklist = scan_rag(
            blocklist_copy(blocks), ScanOpts(textid=True, UUID=True)
        )
        chunks = blocks_to_chunks(blocklist, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        flag = initialize_collection(
            client, collection_name, embedding_model
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(client, collection_name, embedding_model, chunks)
        results: list[ScoredPoint] = query(
            client,
            collection_name,
            embedding_model,
            "Why is the sky blue?",
        )
        self.assertEqual(len(results), len(ps))
        self.assertEqual(results[1].payload['page_content'], text.get_content())  # type: ignore

    def test_query_SPARSE_MERGED(self):
        encoding_model = EncodingModel.SPARSE_MERGED
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        flag = initialize_collection(
            client, collection_name, embedding_model
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(client, collection_name, embedding_model, chunks)
        results: list[ScoredPoint] = query(
            client,
            collection_name,
            embedding_model,
            "What follows the heading",
        )
        self.assertEqual(len(results), len(ps))
        self.assertEqual(results[0].payload['page_content'], text.get_content())  # type: ignore

    def test_query_SPARSE_MERGED2(self):
        encoding_model = EncodingModel.SPARSE_MERGED
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        flag = initialize_collection(
            client, collection_name, embedding_model
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(client, collection_name, embedding_model, chunks)
        results: list[ScoredPoint] = query(
            client,
            collection_name,
            embedding_model,
            "What is the ingested text?",
        )
        self.assertEqual(len(results), len(ps))
        self.assertEqual(results[0].payload['page_content'], text.get_content())  # type: ignore

    def test_query_SPARSE_MERGED3(self):
        encoding_model = EncodingModel.SPARSE_MERGED
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        flag = initialize_collection(
            client, collection_name, embedding_model
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(client, collection_name, embedding_model, chunks)
        results: list[ScoredPoint] = query(
            client,
            collection_name,
            embedding_model,
            "Why is the sky blue?",
        )
        self.assertEqual(len(results), len(ps))
        self.assertEqual(results[1].payload['page_content'], text.get_content())  # type: ignore

    def test_query_to_uuids(self):
        encoding_model = EncodingModel.SPARSE_MERGED
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        ps: list[Point] = upload(
            client, collection_name, embedding_model, chunks
        )
        self.assertListEqual(
            [c.uuid for c in chunks], [str(p.id) for p in ps]
        )
        results: list[ScoredPoint] = query(
            client,
            collection_name,
            embedding_model,
            "What follows the heading",
        )
        self.assertEqual(len(results), len(ps))
        self.assertListEqual(
            [c.uuid for c in chunks], points_to_ids(results)
        )

    def test_query_to_text(self):
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
        ps: list[Point] = upload(
            client, collection_name, embedding_model, chunks
        )
        self.assertListEqual([c.content for c in chunks], textlist)
        results: list[ScoredPoint] = query(
            client,
            collection_name,
            embedding_model,
            "What follows the heading",
        )
        self.assertEqual(len(results), len(ps))
        self.assertListEqual(textlist, points_to_text(results))


if __name__ == '__main__':
    unittest.main()
