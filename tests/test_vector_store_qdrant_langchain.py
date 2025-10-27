"""tests for vector_store_qdrant_langchain.py

NOTE: initialize collection tested in test_lmm_rag.py
"""

import unittest

from lmm.markdown.parse_markdown import (
    Block,
    HeaderBlock,
    MetadataBlock,
    HeadingBlock,
    TextBlock,
    blocklist_copy,
)
from lmm.scan.scan_keys import UUID_KEY, GROUP_UUID_KEY, QUESTIONS_KEY
from lmm.scan.scan_rag import scan_rag, ScanOpts
from lmm.config.config import (
    Settings,
    export_settings,
)
from lmm_education.config.config import (
    ConfigSettings,
    DatabaseSettings,
    RAGSettings,
    AnnotationModel,
)
from lmm_education.stores.chunks import Chunk, blocks_to_chunks
from lmm_education.stores.vector_store_qdrant import (
    QdrantClient,
    EncodingModel,
    QdrantEmbeddingModel,
    Point,
    encoding_to_qdrantembedding_model,
    initialize_collection,
    initialize_collection_from_config,
    upload,
    points_to_ids,
    points_to_text,
)
from lmm_education.stores.langchain.vector_store_qdrant_langchain import (
    QdrantVectorStoreRetriever as Retriever,
)

from langchain_core.documents import Document

# A global client object (for now)
QDRANT_SOURCE = ":memory:"
client = QdrantClient(QDRANT_SOURCE)

header = HeaderBlock(content={"title": "Test blocklist"})
metadata = MetadataBlock(
    content={
        "questions": "What is the ingested test?",
        "~chat": "Some discussion",
        "summary": "The summary of the ingested text.",
    }
)
heading = HeadingBlock(level=2, content="Ingested text")
text = TextBlock(content="This is text following the heading.")
metadata2 = MetadataBlock(
    content={
        "questions": "Why is the sky blue?",
        "summary": "The summary of explanations about sky colour.",
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


class TestQuery(unittest.TestCase):
    # detup and teardown replace config.toml to avoid
    # calling the language model server
    original_settings = Settings()

    @classmethod
    def setUpClass(cls):
        settings = Settings(
            major={"model": "Debug/debug"},
            minor={"model": "Debug/debug"},
            aux={"model": "Debug/debug"},
            embeddings={
                "dense_model": "SentenceTransformers/distiluse-base-multilingual-cased-v1"
            },
        )
        export_settings(settings)

    @classmethod
    def tearDownClass(cls):
        settings = cls.original_settings
        export_settings(settings)

    # ------ query ----------------------------------------------------

    def test_query_NULL(self):
        encoding_model = EncodingModel.NONE
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = encoding_model.value
        if not initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        ):
            raise Exception("Could not initialize collection")
        points = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
        )
        uuids = points_to_ids(points)
        self.assertEqual(len(uuids), len(chunks))

        retriever: Retriever = Retriever(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        )
        results: list[Document] = retriever.invoke(uuids[0])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].page_content, text.get_content())

    def test_query_CONTENT(self):
        encoding_model = EncodingModel.CONTENT
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = encoding_model.value
        if not initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        ):
            raise Exception("Could not initialize collection")
        ps = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
        )

        retriever: Retriever = Retriever(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        )
        results: list[Document] = retriever.invoke(
            "What follows the heading"
        )
        self.assertEqual(len(results), len(ps))
        self.assertTrue(results[0].page_content)
        self.assertEqual(results[0].page_content, text.get_content())  # type: ignore

    def test_query_CONTENT2(self):
        encoding_model = EncodingModel.CONTENT
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = encoding_model.value
        if not initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        ):
            raise Exception("Could not initialize collection")
        ps = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
        )

        retriever: Retriever = Retriever(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        )
        results: list[Document] = retriever.invoke(
            "The oygen composition of the air"
        )
        self.assertEqual(len(results), len(ps))
        self.assertTrue(results[0].page_content)
        self.assertEqual(results[0].page_content, text2.get_content())  # type: ignore

    def test_query_MERGED(self):
        encoding_model = EncodingModel.MERGED
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = encoding_model.value
        if not initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        ):
            raise Exception("Could not initialize collection")
        ps = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
        )

        retriever: Retriever = Retriever(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        )
        results: list[Document] = retriever.invoke(
            "What follows the heading"
        )
        self.assertEqual(len(results), len(ps))
        self.assertTrue(results[0].page_content)
        self.assertEqual(results[0].page_content, text.get_content())  # type: ignore

    def test_query_MERGED2(self):
        encoding_model = EncodingModel.MERGED
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = encoding_model.value
        if not initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        ):
            raise Exception("Could not initialize collection")
        ps = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
        )

        retriever: Retriever = Retriever(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        )
        results: list[Document] = retriever.invoke(
            "What is the ingested text?"
        )
        self.assertEqual(len(results), len(ps))
        self.assertTrue(results[0].page_content)
        self.assertEqual(results[0].page_content, text.get_content())  # type: ignore

    def test_query_MERGED3(self):
        encoding_model = EncodingModel.MERGED
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = encoding_model.value
        if not initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        ):
            raise Exception("Could not initialize collection")
        ps = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
        )

        retriever: Retriever = Retriever(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        )
        results: list[Document] = retriever.invoke(
            "Why is the sky blue?"
        )
        self.assertEqual(len(results), len(ps))
        self.assertTrue(results[1].page_content)
        self.assertEqual(results[1].page_content, text.get_content())  # type: ignore

    def test_query_MULTIVECTOR(self):
        encoding_model = EncodingModel.MULTIVECTOR
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = encoding_model.value
        if not initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        ):
            raise Exception("Could not initialize collection")
        ps = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
        )

        retriever: Retriever = Retriever(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        )
        results: list[Document] = retriever.invoke(
            "What is the ingested text?"
        )
        self.assertEqual(len(results), len(ps))
        self.assertTrue(results[0].page_content)
        self.assertEqual(results[0].page_content, text.get_content())  # type: ignore

    def test_query_SPARSE(self):
        encoding_model = EncodingModel.SPARSE
        chunks = blocks_to_chunks(
            blocks,
            encoding_model,
            AnnotationModel(inherited_properties=[QUESTIONS_KEY]),
        )
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = encoding_model.value
        if not initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        ):
            raise Exception("Could not initialize collection")
        upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
        )

        retriever: Retriever = Retriever(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        )
        results: list[Document] = retriever.invoke(
            "What is the ingested text?"
        )
        self.assertTrue(len(results) > 0)
        self.assertTrue(results[0].page_content)
        self.assertEqual(results[0].page_content, text.get_content())  # type: ignore

    def test_query_SPARSE_CONTENT(self):
        encoding_model = EncodingModel.SPARSE_CONTENT
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = encoding_model.value
        if not initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        ):
            raise Exception("Could not initialize collection")
        ps = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
        )

        retriever: Retriever = Retriever(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        )
        results: list[Document] = retriever.invoke(
            "What follows the heading"
        )
        self.assertEqual(len(results), len(ps))
        self.assertTrue(results[0].page_content)
        self.assertEqual(results[0].page_content, text.get_content())  # type: ignore

    def test_query_SPARSE_CONTENT2(self):
        encoding_model = EncodingModel.SPARSE_CONTENT
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = encoding_model.value
        if not initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        ):
            raise Exception("Could not initialize collection")
        ps = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
        )
        retriever: Retriever = Retriever(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        )
        results: list[Document] = retriever.invoke(
            "What is the ingested text?"
        )
        self.assertEqual(len(results), len(ps))
        self.assertTrue(results[0].page_content)
        self.assertEqual(results[0].page_content, text.get_content())  # type: ignore

    def test_query_SPARSE_CONTENT3(self):
        encoding_model = EncodingModel.SPARSE_CONTENT
        blocklist = scan_rag(
            blocklist_copy(blocks), ScanOpts(textid=True, UUID=True)
        )
        chunks = blocks_to_chunks(blocklist, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = encoding_model.value
        if not initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        ):
            raise Exception("Could not initialize collection")
        ps = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
        )

        retriever: Retriever = Retriever(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        )
        results: list[Document] = retriever.invoke(
            "Why is the sky blue?"
        )
        self.assertEqual(len(results), len(ps))
        self.assertTrue(results[1].page_content)
        self.assertEqual(results[1].page_content, text.get_content())  # type: ignore

    def test_query_SPARSE_MERGED(self):
        encoding_model = EncodingModel.SPARSE_MERGED
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = encoding_model.value
        if not initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        ):
            raise Exception("Could not initialize collection")
        ps = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
        )

        retriever: Retriever = Retriever(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        )
        results: list[Document] = retriever.invoke(
            "What follows the heading"
        )
        self.assertEqual(len(results), len(ps))
        self.assertTrue(results[0].page_content)
        self.assertEqual(results[0].page_content, text.get_content())  # type: ignore

    def test_query_SPARSE_MERGED2(self):
        encoding_model = EncodingModel.SPARSE_MERGED
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = encoding_model.value
        if not initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        ):
            raise Exception("Could not initialize collection")
        ps = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
        )

        retriever: Retriever = Retriever(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        )
        results: list[Document] = retriever.invoke(
            "What is the ingested text?"
        )
        self.assertEqual(len(results), len(ps))
        self.assertTrue(results[0].page_content)
        self.assertEqual(results[0].page_content, text.get_content())  # type: ignore

    def test_query_SPARSE_MERGED3(self):
        encoding_model = EncodingModel.SPARSE_MERGED
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = encoding_model.value
        if not initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        ):
            raise Exception("Could not initialize collection")
        ps = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
        )

        retriever: Retriever = Retriever(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        )
        results: list[Document] = retriever.invoke(
            "Why is the sky blue?"
        )
        self.assertEqual(len(results), len(ps))
        self.assertTrue(results[1].page_content)
        self.assertEqual(results[1].page_content, text.get_content())  # type: ignore

    def test_query_SPARSE_MERGED3_config(self):
        from lmm_education.config.config import ConfigSettings
        from lmm.scan.scan_keys import TITLES_KEY, QUESTIONS_KEY
        from lmm_education.stores.vector_store_qdrant import (
            client_from_config,
        )

        collection_name: str = "adsfosi"
        annotation_model = AnnotationModel(
            inherited_properties=[TITLES_KEY, QUESTIONS_KEY]
        )
        opts = ConfigSettings(
            storage=":memory:",
            database=DatabaseSettings(
                annotation_model=annotation_model,
                collection_name=collection_name,
            ),
            RAG=RAGSettings(
                questions=True,
                encoding_model=EncodingModel.SPARSE_MERGED,
            ),
            embeddings={
                "dense_model": "SentenceTransformers/distiluse-base-multilingual-cased-v1"
            },
        )
        client = client_from_config(opts)
        embedding_model = encoding_to_qdrantembedding_model(
            opts.RAG.encoding_model
        )
        flag = initialize_collection_from_config(
            client, collection_name, opts
        )
        if not flag:
            raise Exception("Could not initialize collection")
        chunks = blocks_to_chunks(
            blocks, opts.RAG.encoding_model, annotation_model
        )
        ps = upload(
            client,
            collection_name,
            embedding_model,
            opts.embeddings,
            chunks,
        )

        retriever: Retriever = Retriever(
            client, collection_name, embedding_model, opts.embeddings
        )
        results: list[Document] = retriever.invoke(
            "Why is the sky blue?"
        )
        self.assertEqual(len(results), len(ps))
        self.assertTrue(results[1].page_content)
        self.assertEqual(results[1].page_content, text.get_content())

    def test_query_SPARSE_MULTIVECTOR(self):
        encoding_model = EncodingModel.SPARSE_MULTIVECTOR
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = encoding_model.value
        if not initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        ):
            raise Exception("Could not initialize collection")
        ps = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
        )

        retriever: Retriever = Retriever(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        )
        results: list[Document] = retriever.invoke(
            "What follows the heading"
        )
        self.assertEqual(len(results), len(ps))
        self.assertTrue(results[0].page_content)
        self.assertEqual(results[0].page_content, text.get_content())  # type: ignore


class TestQueryGrouped(unittest.TestCase):
    # detup and teardown replace config.toml to avoid
    # calling the language model server
    original_settings = Settings()

    @classmethod
    def setUpClass(cls):
        settings = Settings(
            major={"model": "Debug/debug"},
            minor={"model": "Debug/debug"},
            aux={"model": "Debug/debug"},
            embeddings={
                "dense_model": "SentenceTransformers/distiluse-base-multilingual-cased-v1"
            },
        )
        export_settings(settings)

    @classmethod
    def tearDownClass(cls):
        settings = cls.original_settings
        export_settings(settings)

    def test_query(self):
        import lmm_education.stores.chunks as chk
        from lmm.scan.scan_split import scan_split

        from langchain_text_splitters import (
            RecursiveCharacterTextSplitter,
        )

        # import long document, parsed as block list
        from .test_vector_store_qdrant_groups import blocklist

        from lmm_education.stores.langchain.vector_store_qdrant_langchain import (
            QdrantVectorStoreRetrieverGrouped,
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
        embedding_settings = ConfigSettings().embeddings

        if not initialize_collection(
            client,
            COLLECTION_MAIN,
            embedding_model_main,
            embedding_settings,
        ):
            raise Exception("Could not initialize main collection")
        if not initialize_collection(
            client,
            COLLECTION_DOCS,
            embedding_model_companion,
            embedding_settings,
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
        companion_points: list[Point] = upload(
            client,
            COLLECTION_DOCS,
            embedding_model_companion,
            embedding_settings,
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
        points: list[Point] = upload(
            client,
            COLLECTION_MAIN,
            embedding_model_main,
            embedding_settings,
            chunks,
        )
        self.assertLess(0, len(points))
        texts: list[str] = points_to_text(points)
        self.assertTrue(len(texts) > 0)

        # grouped query
        retriever = QdrantVectorStoreRetrieverGrouped(
            client,
            COLLECTION_MAIN,
            COLLECTION_DOCS,
            GROUP_UUID_KEY,
            3,
            embedding_model_main,
            embedding_settings,
        )
        results: list[Document] = retriever.invoke(
            "How can I estimate the predicted depressiveness from this model?"
        )

        # tests
        self.assertLess(0, len(results))
        self.assertTrue(results[0].page_content in companion_texts)


if __name__ == "__main__":
    unittest.main()
