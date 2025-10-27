"""tests for vector_store_qdrant.py

NOTE: initialize collection tested in test_lmm_rag.py
"""

# flake8: noqa

import unittest

from lmm.markdown.parse_markdown import *
from lmm_education.stores.chunks import *
from lmm_education.stores.vector_store_qdrant import *
from lmm.config.config import Settings, export_settings
from lmm.scan.scan_keys import TITLES_KEY, QUESTIONS_KEY
from lmm.utils.logging import ExceptionConsoleLogger

exception_logger = ExceptionConsoleLogger()

# A global client object (for now)
QDRANT_SOURCE = ":memory:"
COLLECTION_MAIN = "Main"
COLLECTION_DOCS = "Main_docs"
client = QdrantClient(QDRANT_SOURCE)

header = HeaderBlock(
    content={"title": "Test blocklist", "frozen": True}
)
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

metadata3 = MetadataBlock(
    content={
        "questions": "Why is the grass green?",
        "summary": "The summary of explanations about grass colour.",
    }
)
heading3 = HeadingBlock(level=2, content="Grass colour")
text3 = TextBlock(
    content="The grass is green because of the presence of chlorophyll in plants."
)

blocks: list[Block] = [
    header,
    metadata,
    heading,
    text,
    metadata2,
    heading2,
    text2,
    metadata3,
    heading3,
    text3,
]
blocks = scan_rag(blocks, ScanOpts(textid=True, UUID=True))

sets: Settings = Settings(
    embeddings={
        "dense_model": "SentenceTransformers/distiluse-base-multilingual-cased-v1"
    }
)

am = AnnotationModel(inherited_properties=[TITLES_KEY, QUESTIONS_KEY])


# write a teardown function for the module where client is closed
def tearDownModule():
    client.close()


class TestInitialization(unittest.TestCase):
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

    def test_encoding_none(self):
        encoding_model = EncodingModel.NONE
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = "TI_" + encoding_model.value
        result = initialize_collection(
            client,
            collection_name,
            embedding_model,
            ConfigSettings().embeddings,
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
        collection_name: str = "TI_" + encoding_model.value
        result = initialize_collection(
            client,
            collection_name,
            embedding_model,
            ConfigSettings().embeddings,
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
        collection_name: str = "TI_" + encoding_model.value
        result = initialize_collection(
            client,
            collection_name,
            embedding_model,
            ConfigSettings().embeddings,
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
        collection_name: str = "TI_" + encoding_model.value
        result = initialize_collection(
            client,
            collection_name,
            embedding_model,
            ConfigSettings().embeddings,
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
        collection_name: str = "TI_" + encoding_model.value
        result = initialize_collection(
            client,
            collection_name,
            embedding_model,
            ConfigSettings().embeddings,
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
        collection_name: str = "TI_" + encoding_model.value
        result = initialize_collection(
            client,
            collection_name,
            embedding_model,
            ConfigSettings().embeddings,
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
        collection_name: str = "TI_" + encoding_model.value
        result = initialize_collection(
            client,
            collection_name,
            embedding_model,
            ConfigSettings().embeddings,
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
        collection_name: str = "TI_" + encoding_model.value
        result = initialize_collection(
            client,
            collection_name,
            embedding_model,
            ConfigSettings().embeddings,
        )
        self.assertTrue(
            result,
            "init_collection should return True for encoding model",
        )


class TestInitializationContext(unittest.TestCase):

    def test_init_context(self):
        from lmm_education.config.config import ConfigSettings
        from lmm_education.stores.vector_store_qdrant import (
            qdrant_client_context,
        )

        encoding_model = EncodingModel.SPARSE_MULTIVECTOR
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        opts = ConfigSettings(storage=":memory:")
        with qdrant_client_context(opts) as client:
            result = initialize_collection(
                client,
                "Main34788",
                embedding_model,
                ConfigSettings().embeddings,
            )

        self.assertTrue(
            result,
            "init_collection should return True for encoding model",
        )

    def test_init_context_datasource(self):
        from lmm_education.config.config import DatabaseSource
        from lmm_education.stores.vector_store_qdrant import (
            qdrant_client_context,
        )

        encoding_model = EncodingModel.SPARSE_MULTIVECTOR
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        opts: DatabaseSource = ":memory:"
        with qdrant_client_context(opts) as client:
            result = initialize_collection(
                client,
                "Main34789",
                embedding_model,
                ConfigSettings().embeddings,
            )

        self.assertTrue(
            result,
            "init_collection should return True for encoding model",
        )


from lmm_education.config.config import RAGSettings


class TestInitializationConfigObject(unittest.TestCase):
    def test_encoding_none(self):
        settings = ConfigSettings(
            major={"model": "Debug/debug"},
            minor={"model": "Debug/debug"},
            aux={"model": "Debug/debug"},
            embeddings={
                "dense_model": "SentenceTransformers/distiluse-base-multilingual-cased-v1"
            },
            storage=":memory:",
            RAG=RAGSettings(encoding_model=EncodingModel.NONE),
        )
        collection_name: str = settings.RAG.encoding_model.value
        result = initialize_collection_from_config(
            client, collection_name, settings
        )
        self.assertTrue(
            result,
            "init_collection should return True for encoding model",
        )

    def test_encoding_content(self):
        settings = ConfigSettings(
            major={"model": "Debug/debug"},
            minor={"model": "Debug/debug"},
            aux={"model": "Debug/debug"},
            embeddings={
                "dense_model": "SentenceTransformers/distiluse-base-multilingual-cased-v1"
            },
            storage=":memory:",
            RAG=RAGSettings(encoding_model=EncodingModel.CONTENT),
        )
        collection_name: str = settings.RAG.encoding_model.value
        result = initialize_collection_from_config(
            client, collection_name, settings
        )
        self.assertTrue(
            result,
            "init_collection should return True for encoding model",
        )

    def test_encoding_merged(self):
        settings = ConfigSettings(
            major={"model": "Debug/debug"},
            minor={"model": "Debug/debug"},
            aux={"model": "Debug/debug"},
            embeddings={
                "dense_model": "SentenceTransformers/distiluse-base-multilingual-cased-v1"
            },
            storage=":memory:",
            RAG=RAGSettings(encoding_model=EncodingModel.MERGED),
        )
        collection_name: str = settings.RAG.encoding_model.value
        result = initialize_collection_from_config(
            client, collection_name, settings
        )
        self.assertTrue(
            result,
            "init_collection should return True for encoding model",
        )

    def test_encoding_multivector(self):
        settings = ConfigSettings(
            major={"model": "Debug/debug"},
            minor={"model": "Debug/debug"},
            aux={"model": "Debug/debug"},
            embeddings={
                "dense_model": "SentenceTransformers/distiluse-base-multilingual-cased-v1"
            },
            storage=":memory:",
            RAG=RAGSettings(encoding_model=EncodingModel.MULTIVECTOR),
        )
        collection_name: str = settings.RAG.encoding_model.value
        result = initialize_collection_from_config(
            client, collection_name, settings
        )
        self.assertTrue(
            result,
            "init_collection should return True for encoding model",
        )

    def test_encoding_sparse(self):
        settings = ConfigSettings(
            major={"model": "Debug/debug"},
            minor={"model": "Debug/debug"},
            aux={"model": "Debug/debug"},
            embeddings={
                "dense_model": "SentenceTransformers/distiluse-base-multilingual-cased-v1"
            },
            storage=":memory:",
            RAG=RAGSettings(
                encoding_model=EncodingModel.SPARSE_MERGED
            ),
        )
        collection_name: str = settings.RAG.encoding_model.value
        result = initialize_collection_from_config(
            client, collection_name, settings
        )
        self.assertTrue(
            result,
            "init_collection should return True for encoding model",
        )

    def test_encoding_sparse_merged(self):
        settings = ConfigSettings(
            major={"model": "Debug/debug"},
            minor={"model": "Debug/debug"},
            aux={"model": "Debug/debug"},
            embeddings={
                "dense_model": "SentenceTransformers/distiluse-base-multilingual-cased-v1"
            },
            storage=":memory:",
            RAG=RAGSettings(
                encoding_model=EncodingModel.SPARSE_MERGED
            ),
        )
        collection_name: str = settings.RAG.encoding_model.value
        result = initialize_collection_from_config(
            client, collection_name, settings
        )
        self.assertTrue(
            result,
            "init_collection should return True for encoding model",
        )

    def test_encoding_sparse_content(self):
        settings = ConfigSettings(
            major={"model": "Debug/debug"},
            minor={"model": "Debug/debug"},
            aux={"model": "Debug/debug"},
            embeddings={
                "dense_model": "SentenceTransformers/distiluse-base-multilingual-cased-v1"
            },
            storage=":memory:",
            RAG=RAGSettings(
                encoding_model=EncodingModel.SPARSE_CONTENT
            ),
        )
        collection_name: str = settings.RAG.encoding_model.value
        result = initialize_collection_from_config(
            client, collection_name, settings
        )
        self.assertTrue(
            result,
            "init_collection should return True for encoding model",
        )

    def test_encoding_sparse_multivector(self):
        settings = ConfigSettings(
            major={"model": "Debug/debug"},
            minor={"model": "Debug/debug"},
            aux={"model": "Debug/debug"},
            embeddings={
                "dense_model": "SentenceTransformers/distiluse-base-multilingual-cased-v1"
            },
            storage=":memory:",
            RAG=RAGSettings(
                encoding_model=EncodingModel.SPARSE_MULTIVECTOR
            ),
        )
        collection_name: str = settings.RAG.encoding_model.value
        result = initialize_collection_from_config(
            client, collection_name, settings
        )
        self.assertTrue(
            result,
            "init_collection should return True for encoding model",
        )


class TestInitializationLocal(unittest.TestCase):
    # detup and teardown replace config.toml to avoid
    # calling the language model server
    original_settings = ConfigSettings()

    @classmethod
    def setUpClass(cls):
        settings = ConfigSettings(
            major={"model": "Debug/debug"},
            minor={"model": "Debug/debug"},
            aux={"model": "Debug/debug"},
            embeddings={
                "dense_model": "SentenceTransformers/distiluse-base-multilingual-cased-v1"
            },
            storage=":memory:",
        )
        export_settings(settings)

    @classmethod
    def tearDownClass(cls):
        settings = cls.original_settings
        export_settings(settings)

    def test_encoding_content(self):
        from lmm.utils.logging import LoglistLogger

        logger = LoglistLogger()

        try:
            local_client = QdrantClient(path="./test_storage")
        except Exception as e:
            print(f"{e}")
            self.assertTrue(False)

        encoding_model = EncodingModel.CONTENT
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = "TIL_" + encoding_model.value
        result = initialize_collection(
            local_client,
            "chunks",
            embedding_model,
            ConfigSettings().embeddings,
            logger=logger,
        )
        if logger.count_logs(level=1):
            print("\n".join(logger.get_logs()))
        self.assertTrue(logger.count_logs(level=1) == 0)
        self.assertTrue(
            result,
            "init_collection should return True for encoding model",
        )
        local_client.close()
        # delete the storage directory and all its contents
        import shutil

        shutil.rmtree("./test_storage")


class TestIngestionAndQuery(unittest.TestCase):
    # detup and teardown replace config.toml to avoid
    # calling the language model server
    original_settings = ConfigSettings()

    @classmethod
    def setUpClass(cls):
        settings = ConfigSettings(
            major={"model": "Debug/debug"},
            minor={"model": "Debug/debug"},
            aux={"model": "Debug/debug"},
            embeddings={
                "dense_model": "SentenceTransformers/distiluse-base-multilingual-cased-v1"
            },
            storage=":memory:",
        )
        export_settings(settings)

    @classmethod
    def tearDownClass(cls):
        settings = cls.original_settings
        export_settings(settings)

    # ------ ingestion ------------------------------------------------
    def test_ingestion_empty(self):
        encoding_model = EncodingModel.NONE
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = "TIQ_" + encoding_model.value
        embedding_settings = ConfigSettings().embeddings
        points = chunks_to_points(
            [], embedding_model, embedding_settings
        )
        self.assertEqual(len(points), 0)
        flag = initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            [],
        )
        self.assertEqual(len(ps), 0)

    def test_ingestion_nontext(self):
        # blocklist w/o text blocks
        chunks = blocks_to_chunks(
            [header, metadata, heading], EncodingModel.CONTENT, am
        )
        self.assertEqual(len(chunks), 0)
        encoding_model = EncodingModel.CONTENT
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = "TIQ_" + encoding_model.value
        points = chunks_to_points(
            chunks, embedding_model, embedding_settings
        )
        self.assertEqual(len(chunks), len(points))
        ps: list[Point] = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
        )
        self.assertEqual(len(chunks), len(ps))
        for p, u in zip(points, points_to_ids(ps)):
            self.assertEqual(p.id, u)

    def test_ingestion_NULL(self):
        chunks = blocks_to_chunks(
            blocks,
            EncodingModel.NONE,
            AnnotationModel(inherited_properties=[TITLES_KEY]),
        )
        self.assertEqual(len(chunks), 3)
        chunk = chunks[0]
        self.assertEqual(chunk.dense_encoding, "")
        encoding_model = EncodingModel.NONE
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = "TIQ_" + encoding_model.value
        points = chunks_to_points(
            chunks, embedding_model, embedding_settings
        )
        self.assertEqual(len(chunks), len(points))
        flag = initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps: list[Point] = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
        )
        self.assertEqual(len(chunks), len(ps))
        for p, u in zip(points, points_to_ids(ps)):
            self.assertEqual(p.id, u)

    def test_ingestion_CONTENT(self):
        # just one text here
        shortblocks = blocks[:5]
        chunks = blocks_to_chunks(
            shortblocks, EncodingModel.CONTENT, am
        )
        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.dense_encoding, text.get_content())
        encoding_model = EncodingModel.CONTENT
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = "TIQ_" + encoding_model.value
        points = chunks_to_points(
            chunks, embedding_model, embedding_settings
        )
        self.assertEqual(len(chunks), len(points))
        flag = initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps: list[Point] = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
        )
        self.assertEqual(len(chunks), len(ps))
        for p, u in zip(points, points_to_ids(ps)):
            self.assertEqual(p.id, u)

    def test_ingestion_MERGED(self):
        chunks = blocks_to_chunks(blocks, EncodingModel.MERGED, am)
        self.assertEqual(len(chunks), 3)
        chunk = chunks[0]
        self.assertEqual(
            chunk.dense_encoding,
            chunk.annotations + ": " + text.get_content(),
        )
        encoding_model = EncodingModel.MERGED
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = "TIQ_" + encoding_model.value
        points = chunks_to_points(
            chunks, embedding_model, embedding_settings
        )
        self.assertEqual(len(chunks), len(points))
        flag = initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps: list[Point] = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
        )
        self.assertEqual(len(chunks), len(ps))
        for p, u in zip(points, points_to_ids(ps)):
            self.assertEqual(p.id, u)

    def test_ingestion_MULTIVECTOR(self):
        chunks = blocks_to_chunks(
            blocks, EncodingModel.MULTIVECTOR, am
        )
        self.assertEqual(len(chunks), 3)
        chunk = chunks[0]
        self.assertEqual(chunk.dense_encoding, text.get_content())
        encoding_model = EncodingModel.MULTIVECTOR
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = "TIQ_" + encoding_model.value
        points = chunks_to_points(
            chunks, embedding_model, embedding_settings
        )
        self.assertEqual(len(chunks), len(points))
        flag = initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps: list[Point] = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
        )
        self.assertEqual(len(chunks), len(ps))
        for p, u in zip(points, points_to_ids(ps)):
            self.assertEqual(p.id, u)

    def test_ingestion_SPARSE(self):
        chunks = blocks_to_chunks(blocks, EncodingModel.SPARSE, am)
        self.assertEqual(len(chunks), 3)
        chunk = chunks[0]
        self.assertEqual(chunk.sparse_encoding, chunk.annotations)
        encoding_model = EncodingModel.SPARSE
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = "TIQ_" + encoding_model.value
        points = chunks_to_points(
            chunks, embedding_model, embedding_settings
        )
        self.assertEqual(len(chunks), len(points))
        flag = initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps: list[Point] = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
        )
        self.assertEqual(len(chunks), len(ps))
        for p, u in zip(points, points_to_ids(ps)):
            self.assertEqual(p.id, u)

    def test_ingestion_SPARSE_CONTENT(self):
        chunks = blocks_to_chunks(
            blocks, EncodingModel.SPARSE_CONTENT, am
        )
        self.assertEqual(len(chunks), 3)
        chunk = chunks[0]
        self.assertEqual(chunk.sparse_encoding, chunk.annotations)
        self.assertEqual(chunk.dense_encoding, text.get_content())
        encoding_model = EncodingModel.SPARSE_CONTENT
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = "TIQ_" + encoding_model.value
        points = chunks_to_points(
            chunks, embedding_model, embedding_settings
        )
        self.assertEqual(len(chunks), len(points))
        flag = initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
        )
        self.assertEqual(len(chunks), len(ps))

    def test_ingestion_SPARSE_MERGED(self):
        chunks = blocks_to_chunks(
            blocks, EncodingModel.SPARSE_MERGED, am
        )
        self.assertEqual(len(chunks), 3)
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
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = "TIQ_" + encoding_model.value
        points = chunks_to_points(
            chunks, embedding_model, embedding_settings
        )
        self.assertEqual(len(chunks), len(points))
        flag = initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
        )
        self.assertEqual(len(chunks), len(ps))

    def test_ingestion_SPARSE_MULTIVECTOR(self):
        chunks = blocks_to_chunks(
            blocks, EncodingModel.SPARSE_MULTIVECTOR, am
        )
        self.assertEqual(len(chunks), 3)
        chunk = chunks[0]
        self.assertEqual(chunk.dense_encoding, text.get_content())
        encoding_model = EncodingModel.SPARSE_MULTIVECTOR
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = "TIQ_" + encoding_model.value
        points = chunks_to_points(
            chunks, embedding_model, embedding_settings
        )
        self.assertEqual(len(chunks), len(points))
        flag = initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps: list[Point] = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
        )
        self.assertEqual(len(chunks), len(ps))
        for p, u in zip(points, points_to_ids(ps)):
            self.assertEqual(p.id, u)

    # ------ query ----------------------------------------------------

    def test_query_NULL(self):
        encoding_model = EncodingModel.NONE
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = "TIQ_" + encoding_model.value
        flag = initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        )
        self.assertTrue(flag, "Could not initialize collection")
        points = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
        )
        uuids = points_to_ids(points)
        self.assertEqual(len(uuids), len(chunks))
        results: list[ScoredPoint] = query(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            uuids[0],
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].payload["page_content"], text.get_content())  # type: ignore

    def test_query_CONTENT(self):
        encoding_model = EncodingModel.CONTENT
        chunks = blocks_to_chunks(blocks, encoding_model, am)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = "TIQ_" + encoding_model.value
        flag = initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
        )
        results: list[ScoredPoint] = query(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            "What follows the heading",
        )
        self.assertEqual(len(results), len(ps))
        self.assertIsNotNone(results[0].payload)
        self.assertEqual(results[0].payload["page_content"], text.get_content())  # type: ignore

    def test_query_CONTENT2(self):
        encoding_model = EncodingModel.CONTENT
        chunks = blocks_to_chunks(blocks, encoding_model, am)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = "TIQ_" + encoding_model.value
        flag = initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
        )
        results: list[ScoredPoint] = query(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            "The oygen composition of the air",
        )
        self.assertEqual(len(results), len(ps))
        self.assertEqual(results[0].payload["page_content"], text2.get_content())  # type: ignore

    def test_query_MERGED(self):
        encoding_model = EncodingModel.MERGED
        chunks = blocks_to_chunks(blocks, encoding_model, am)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = "TIQ_" + encoding_model.value
        flag = initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
        )
        results: list[ScoredPoint] = query(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            "What follows the heading",
        )
        self.assertEqual(len(results), len(ps))
        self.assertEqual(results[0].payload["page_content"], text.get_content())  # type: ignore

    def test_query_MERGED2(self):
        encoding_model = EncodingModel.MERGED
        chunks = blocks_to_chunks(blocks, encoding_model, am)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = "TIQ_" + encoding_model.value
        flag = initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
        )
        results: list[ScoredPoint] = query(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            "What is the ingested text?",
        )
        self.assertEqual(len(results), len(ps))
        self.assertEqual(results[0].payload["page_content"], text.get_content())  # type: ignore

    def test_query_MERGED3(self):
        encoding_model = EncodingModel.MERGED
        chunks = blocks_to_chunks(blocks, encoding_model, am)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = "TIQ_" + encoding_model.value
        flag = initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
        )
        results: list[ScoredPoint] = query(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            "Why is the sky blue?",
        )
        self.assertEqual(len(results), len(ps))
        self.assertEqual(
            str(results[0].payload["page_content"]),
            text2.get_content(),
        )

    def test_query_MULTIVECTOR(self):
        encoding_model = EncodingModel.MULTIVECTOR
        chunks = blocks_to_chunks(blocks, encoding_model, am)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = "TIQ_" + encoding_model.value
        flag = initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
        )
        results: list[ScoredPoint] = query(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            "Why is the sky blue?",
        )
        self.assertEqual(len(results), len(ps))
        self.assertEqual(
            str(results[0].payload["page_content"]),
            text2.get_content(),
        )

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
        collection_name: str = "TIQ_" + encoding_model.value
        flag = initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
        )
        results: list[ScoredPoint] = query(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            "What is the ingested text?",
        )
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0].payload["page_content"], text.get_content())  # type: ignore

    def test_query_SPARSE_CONTENT(self):
        encoding_model = EncodingModel.SPARSE_CONTENT
        chunks = blocks_to_chunks(
            blocks,
            encoding_model,
            AnnotationModel(inherited_properties=[QUESTIONS_KEY]),
        )
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = "TIQ_" + encoding_model.value
        flag = initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
        )
        results: list[ScoredPoint] = query(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            "What follows the heading",
        )
        self.assertEqual(len(results), len(ps))
        self.assertEqual(results[0].payload["page_content"], text.get_content())  # type: ignore

    def test_query_SPARSE_CONTENT2(self):
        encoding_model = EncodingModel.SPARSE_CONTENT
        chunks = blocks_to_chunks(blocks, encoding_model, am)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = "TIQ_" + encoding_model.value
        flag = initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
        )
        results: list[ScoredPoint] = query(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            "What is the ingested text?",
        )
        self.assertEqual(len(results), len(ps))
        self.assertEqual(results[0].payload["page_content"], text.get_content())  # type: ignore

    def test_query_SPARSE_CONTENT3(self):
        encoding_model = EncodingModel.SPARSE_CONTENT
        blocklist = scan_rag(
            blocklist_copy(blocks), ScanOpts(textid=True, UUID=True)
        )
        chunks = blocks_to_chunks(blocklist, encoding_model, am)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = "TIQ_" + encoding_model.value
        flag = initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
        )
        results: list[ScoredPoint] = query(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            "Why is the sky blue?",
        )
        self.assertEqual(len(results), len(ps))
        self.assertEqual(results[0].payload["page_content"], text2.get_content())  # type: ignore

    def test_query_SPARSE_MERGED(self):
        encoding_model = EncodingModel.SPARSE_MERGED
        chunks = blocks_to_chunks(blocks, encoding_model, am)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = "TIQ_" + encoding_model.value
        flag = initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
        )
        results: list[ScoredPoint] = query(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            "What follows the heading",
        )
        self.assertEqual(len(results), len(ps))
        self.assertEqual(results[0].payload["page_content"], text.get_content())  # type: ignore

    def test_query_SPARSE_MERGED2(self):
        encoding_model = EncodingModel.SPARSE_MERGED
        chunks = blocks_to_chunks(blocks, encoding_model, am)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = "TIQ_" + encoding_model.value
        flag = initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
        )
        results: list[ScoredPoint] = query(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            "What is the ingested text?",
        )
        self.assertEqual(len(results), len(ps))
        self.assertEqual(results[0].payload["page_content"], text.get_content())  # type: ignore

    def test_query_SPARSE_MERGED3(self):
        encoding_model = EncodingModel.SPARSE_MERGED
        chunks = blocks_to_chunks(blocks, encoding_model, am)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = "TIQ_" + encoding_model.value
        flag = initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
        )
        results: list[ScoredPoint] = query(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            "Why is the sky blue?",
        )
        self.assertEqual(len(results), len(ps))
        self.assertEqual(results[0].payload["page_content"], text2.get_content())  # type: ignore

    def test_query_SPARSE_MULTIVECTOR(self):
        encoding_model = EncodingModel.SPARSE_MULTIVECTOR
        chunks = blocks_to_chunks(blocks, encoding_model, am)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = "TIQ_" + encoding_model.value
        flag = initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
        )
        results: list[ScoredPoint] = query(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            "What follows the heading",
        )
        self.assertEqual(len(results), len(ps))
        self.assertEqual(results[0].payload["page_content"], text.get_content())  # type: ignore

    def test_query_to_uuids(self):
        # tests uuids we get out are those we put in
        encoding_model = EncodingModel.SPARSE_MERGED
        chunks = blocks_to_chunks(blocks, encoding_model, am)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = "TIQ_" + encoding_model.value + "uuids"
        ps: list[Point] = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
        )
        self.assertEqual(len(chunks), len(ps))
        cuuids = [c.uuid for c in chunks]
        cuuids.sort()
        pids = [str(p.id) for p in ps]
        pids.sort()
        self.assertListEqual(cuuids, pids)
        results: list[ScoredPoint] = query(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            "What follows the heading",
        )
        self.assertEqual(len(results), len(ps))
        uuids = points_to_ids(results)
        uuids.sort()
        self.assertListEqual(cuuids, uuids)

    def test_query_to_text(self):
        # tests equality of text ingested and retrieved
        encoding_model = EncodingModel.SPARSE_MERGED
        textlist = [
            b.get_content()
            for b in blocks
            if isinstance(b, TextBlock)
        ]
        chunks = blocks_to_chunks(blocks, encoding_model, am)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = "TIQ_" + encoding_model.value
        ps: list[Point] = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
        )
        self.assertListEqual([c.content for c in chunks], textlist)
        results: list[ScoredPoint] = query(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            "What follows the heading",
        )
        self.assertEqual(len(results), len(ps))
        textlist.sort()
        resultlist = points_to_text(results)
        resultlist.sort()
        self.assertListEqual(textlist, resultlist)


class TestIngestionMisspecified(unittest.TestCase):

    def test_query_SPARSE(self):
        encoding_model = EncodingModel.SPARSE
        with self.assertRaises(RuntimeError):
            blocks_to_chunks(
                blocks,
                encoding_model,
                # no annotation model given
                logger=exception_logger,
            )

    def test_query_SPARSE_logged(self):
        # use LoglistLogger to get exception otherwise printed
        # to console
        from lmm.utils.logging import LoglistLogger

        logger = LoglistLogger()

        encoding_model = EncodingModel.SPARSE
        blocks_to_chunks(
            blocks,
            encoding_model,
            # no annotation model given
            logger=logger,
        )
        self.assertEqual(logger.count_logs(level=1), 1)

    def test_query_MULTIVECTOR_logged(self):
        # use LoglistLogger to get exception otherwise printed
        # to console
        from lmm.utils.logging import LoglistLogger

        logger = LoglistLogger()

        encoding_model = EncodingModel.MULTIVECTOR
        blocks_to_chunks(
            blocks,
            encoding_model,
            # no annotation model given
            logger=logger,
        )
        self.assertEqual(logger.count_logs(level=1), 1)
        self.assertIn("WARNING", logger.get_logs(level=1)[0])

    def test_query_SPARSE_MERGED_logged(self):
        # use LoglistLogger to get exception otherwise printed
        # to console
        from lmm.utils.logging import LoglistLogger

        logger = LoglistLogger()

        encoding_model = EncodingModel.SPARSE_MERGED
        blocks_to_chunks(
            blocks,
            encoding_model,
            # no annotation model given
            logger=logger,
        )
        self.assertEqual(logger.count_logs(level=1), 1)
        self.assertIn("WARNING", logger.get_logs(level=1)[0])


class TestIngestionLargeText(unittest.TestCase):

    # detup and teardown replace config.toml to avoid
    # calling the language model server
    original_settings = ConfigSettings()

    @classmethod
    def setUpClass(cls):
        settings = ConfigSettings(
            major={"model": "Debug/debug"},
            minor={"model": "Debug/debug"},
            aux={"model": "Debug/debug"},
            embeddings={
                "dense_model": "SentenceTransformers/distiluse-base-multilingual-cased-v1"
            },
            RAG={'summaries': False},
            storage=":memory:",
        )
        export_settings(settings)

    @classmethod
    def tearDownClass(cls):
        settings = cls.original_settings
        export_settings(settings)

    def _get_blocks(self) -> list[Block]:
        from tests.test_integration_ingestion_query import document

        return parse_markdown_text(document)

    def _get_blocks_len(self) -> int:
        blocks = self._get_blocks()
        return len([b for b in blocks if isinstance(b, TextBlock)])

    # ------ ingestion ------------------------------------------------

    def test_ingestion_CONTENT(self):
        blocks = self._get_blocks()
        chunks = blocks_to_chunks(blocks, EncodingModel.CONTENT, am)
        self.assertEqual(len(chunks), self._get_blocks_len())
        encoding_model = EncodingModel.CONTENT
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = "TILT_" + encoding_model.value
        points = chunks_to_points(
            chunks,
            embedding_model,
            embedding_settings,
            logger=exception_logger,
        )
        self.assertEqual(len(chunks), len(points))
        flag = initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            logger=exception_logger,
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps: list[Point] = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
            logger=exception_logger,
        )
        self.assertEqual(len(chunks), len(ps))
        for p, u in zip(points, points_to_ids(ps)):
            self.assertEqual(p.id, u)

    def test_ingestion_MERGED(self):
        blocks = self._get_blocks()
        chunks = blocks_to_chunks(blocks, EncodingModel.MERGED, am)
        self.assertEqual(len(chunks), self._get_blocks_len())
        encoding_model = EncodingModel.MERGED
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = "TILT_" + encoding_model.value
        points = chunks_to_points(
            chunks,
            embedding_model,
            embedding_settings,
            logger=exception_logger,
        )
        self.assertEqual(len(chunks), len(points))
        flag = initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            logger=exception_logger,
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps: list[Point] = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
            logger=exception_logger,
        )
        self.assertEqual(len(chunks), len(ps))
        for p, u in zip(points, points_to_ids(ps)):
            self.assertEqual(p.id, u)

    def test_ingestion_MULTIVECTOR(self):
        blocks = self._get_blocks()
        chunks = blocks_to_chunks(blocks, EncodingModel.MERGED, am)
        self.assertEqual(len(chunks), self._get_blocks_len())
        encoding_model = EncodingModel.MULTIVECTOR
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = "TILT_" + encoding_model.value
        points = chunks_to_points(
            chunks,
            embedding_model,
            embedding_settings,
            logger=exception_logger,
        )
        self.assertEqual(len(chunks), len(points))
        flag = initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            logger=exception_logger,
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps: list[Point] = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
            logger=exception_logger,
        )
        self.assertEqual(len(chunks), len(ps))
        for p, u in zip(points, points_to_ids(ps)):
            self.assertEqual(p.id, u)

    def test_ingestion_SPARSE(self):

        blocks = self._get_blocks()
        chunks = blocks_to_chunks(blocks, EncodingModel.SPARSE, am)
        self.assertEqual(len(chunks), self._get_blocks_len())
        encoding_model = EncodingModel.SPARSE
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = "TILT_" + encoding_model.value
        points = chunks_to_points(
            chunks,
            embedding_model,
            embedding_settings,
            logger=exception_logger,
        )
        self.assertEqual(len(chunks), len(points))
        flag = initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            logger=exception_logger,
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps: list[Point] = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
            logger=exception_logger,
        )
        self.assertEqual(len(chunks), len(ps))
        for p, u in zip(points, points_to_ids(ps)):
            self.assertEqual(p.id, u)

    def test_ingestion_SPARSE_CONTENT(self):

        blocks = self._get_blocks()
        chunks = blocks_to_chunks(
            blocks, EncodingModel.SPARSE_CONTENT, am
        )
        self.assertEqual(len(chunks), self._get_blocks_len())
        encoding_model = EncodingModel.SPARSE_CONTENT
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = "TILT_" + encoding_model.value
        points = chunks_to_points(
            chunks,
            embedding_model,
            embedding_settings,
            logger=exception_logger,
        )
        self.assertEqual(len(chunks), len(points))
        flag = initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            logger=exception_logger,
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
            logger=exception_logger,
        )
        self.assertEqual(len(chunks), len(ps))

    def test_ingestion_SPARSE_MERGED(self):

        blocks = self._get_blocks()
        chunks = blocks_to_chunks(
            blocks, EncodingModel.SPARSE_MERGED, am
        )
        self.assertEqual(len(chunks), self._get_blocks_len())
        encoding_model = EncodingModel.SPARSE_MERGED
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = "TILT_" + encoding_model.value
        points = chunks_to_points(
            chunks,
            embedding_model,
            embedding_settings,
            logger=exception_logger,
        )
        self.assertEqual(len(chunks), len(points))
        flag = initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            logger=exception_logger,
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
            logger=exception_logger,
        )
        self.assertEqual(len(chunks), len(ps))

    def test_ingestion_SPARSE_MULTIVECTOR(self):

        blocks = self._get_blocks()
        chunks = blocks_to_chunks(
            blocks, EncodingModel.SPARSE_MULTIVECTOR, am
        )
        self.assertEqual(len(chunks), self._get_blocks_len())
        encoding_model = EncodingModel.SPARSE_MULTIVECTOR
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = "TILT_" + encoding_model.value
        points = chunks_to_points(
            chunks,
            embedding_model,
            embedding_settings,
            logger=exception_logger,
        )
        self.assertEqual(len(chunks), len(points))
        flag = initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            logger=exception_logger,
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
            logger=exception_logger,
        )
        self.assertEqual(len(chunks), len(ps))


class TestQueryLargeText(unittest.TestCase):

    # detup and teardown replace config.toml to avoid
    # calling the language model server
    original_settings = ConfigSettings()

    @classmethod
    def setUpClass(cls):
        settings = ConfigSettings(
            major={"model": "Debug/debug"},
            minor={"model": "Debug/debug"},
            aux={"model": "Debug/debug"},
            embeddings={
                "dense_model": "SentenceTransformers/distiluse-base-multilingual-cased-v1"
            },
            textSplitter={"splitter": "default", "threshold": 125},
            storage=":memory:",
            RAG={'summaries': False},
        )
        export_settings(settings)

    @classmethod
    def tearDownClass(cls):
        settings = cls.original_settings
        export_settings(settings)

    def _get_blocks(self) -> list[Block]:
        from tests.test_integration_ingestion_query import document

        return parse_markdown_text(document)

    # ------ query ----------------------------------------------------

    def test_query_NULL(self):

        blocks = self._get_blocks()
        encoding_model = EncodingModel.NONE
        chunks = blocks_to_chunks(blocks, encoding_model)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = "QLT_" + encoding_model.value
        flag = initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            logger=exception_logger,
        )
        self.assertTrue(flag, "Could not initialize collection")
        points = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
            logger=exception_logger,
        )
        uuids = points_to_ids(points)
        self.assertEqual(len(uuids), len(chunks))
        results: list[ScoredPoint] = query(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            uuids[0],
            logger=exception_logger,
        )
        self.assertEqual(len(results), 1)

    def test_query_CONTENT(self):

        blocks = self._get_blocks()
        encoding_model = EncodingModel.CONTENT
        chunks = blocks_to_chunks(blocks, encoding_model, am)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = "QLT_" + encoding_model.value
        flag = initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            logger=exception_logger,
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
            logger=exception_logger,
        )
        results: list[ScoredPoint] = query(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            "What follows the heading",
            logger=exception_logger,
        )
        self.assertEqual(len(results), len(ps))
        self.assertIsNotNone(results[0].payload)

    def test_query_CONTENT2(self):

        blocks = self._get_blocks()
        encoding_model = EncodingModel.CONTENT
        chunks = blocks_to_chunks(blocks, encoding_model, am)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = "QLT_" + encoding_model.value
        flag = initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            logger=exception_logger,
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
            logger=exception_logger,
        )
        results: list[ScoredPoint] = query(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            "The oygen composition of the air",
            logger=exception_logger,
        )
        self.assertEqual(len(results), len(ps))

    def test_query_MERGED(self):

        blocks = self._get_blocks()
        encoding_model = EncodingModel.MERGED
        chunks = blocks_to_chunks(blocks, encoding_model, am)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = "QLT_" + encoding_model.value
        flag = initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            logger=exception_logger,
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
            logger=exception_logger,
        )
        results: list[ScoredPoint] = query(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            "What follows the heading",
            logger=exception_logger,
        )
        self.assertEqual(len(results), len(ps))

    def test_query_MERGED2(self):

        blocks = self._get_blocks()
        encoding_model = EncodingModel.MERGED
        chunks = blocks_to_chunks(blocks, encoding_model, am)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = "QLT_" + encoding_model.value
        flag = initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            logger=exception_logger,
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
            logger=exception_logger,
        )
        results: list[ScoredPoint] = query(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            "What is the ingested text?",
            logger=exception_logger,
        )
        self.assertEqual(len(results), len(ps))

    def test_query_MERGED3(self):

        blocks = self._get_blocks()
        encoding_model = EncodingModel.MERGED
        chunks = blocks_to_chunks(blocks, encoding_model, am)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = "QLT_" + encoding_model.value
        flag = initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            logger=exception_logger,
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
            logger=exception_logger,
        )
        results: list[ScoredPoint] = query(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            "Why is the sky blue?",
            logger=exception_logger,
        )
        self.assertEqual(len(results), len(ps))

    def test_query_MULTIVECTOR(self):

        blocks = self._get_blocks()
        encoding_model = EncodingModel.MULTIVECTOR
        chunks = blocks_to_chunks(blocks, encoding_model, am)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = "QLT_" + encoding_model.value
        flag = initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            logger=exception_logger,
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
            logger=exception_logger,
        )
        results: list[ScoredPoint] = query(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            "What follows the heading",
            logger=exception_logger,
        )
        if len(results) > len(ps):
            print("\n-------\n".join(points_to_text(results)))
            print("#####################")
            print("\n-------\n".join(points_to_text(ps)))
            print("#####################")
        self.assertEqual(len(results), len(ps))

    def test_query_SPARSE(self):

        blocks = self._get_blocks()
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
        collection_name: str = "QLT_" + encoding_model.value
        flag = initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            logger=exception_logger,
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
            logger=exception_logger,
        )
        results: list[ScoredPoint] = query(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            "What is the ingested text?",
            logger=exception_logger,
        )
        self.assertTrue(len(results) > 0)

    def test_query_SPARSE_CONTENT(self):

        blocks = self._get_blocks()
        encoding_model = EncodingModel.SPARSE_CONTENT
        chunks = blocks_to_chunks(
            blocks,
            encoding_model,
            AnnotationModel(inherited_properties=[QUESTIONS_KEY]),
            logger=exception_logger,
        )
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = "QLT_" + encoding_model.value
        flag = initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            logger=exception_logger,
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
            logger=exception_logger,
        )
        results: list[ScoredPoint] = query(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            "What follows the heading",
            logger=exception_logger,
        )
        self.assertEqual(len(results), len(ps))

    def test_query_SPARSE_CONTENT2(self):

        blocks = self._get_blocks()
        encoding_model = EncodingModel.SPARSE_CONTENT
        chunks = blocks_to_chunks(blocks, encoding_model, am)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = "QLT_" + encoding_model.value
        flag = initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            logger=exception_logger,
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
            logger=exception_logger,
        )
        results: list[ScoredPoint] = query(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            "What is the ingested text?",
            logger=exception_logger,
        )
        self.assertEqual(len(results), len(ps))

    def test_query_SPARSE_CONTENT3(self):

        blocks = self._get_blocks()
        encoding_model = EncodingModel.SPARSE_CONTENT
        blocklist = scan_rag(
            blocklist_copy(blocks), ScanOpts(textid=True, UUID=True)
        )
        chunks = blocks_to_chunks(blocklist, encoding_model, am)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = "QLT_" + encoding_model.value
        flag = initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            logger=exception_logger,
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
            logger=exception_logger,
        )
        results: list[ScoredPoint] = query(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            "Why is the sky blue?",
            logger=exception_logger,
        )
        self.assertEqual(len(results), len(ps))

    def test_query_SPARSE_MERGED(self):

        blocks = self._get_blocks()
        encoding_model = EncodingModel.SPARSE_MERGED
        chunks = blocks_to_chunks(blocks, encoding_model, am)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = "QLT_" + encoding_model.value
        flag = initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            logger=exception_logger,
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
            logger=exception_logger,
        )
        results: list[ScoredPoint] = query(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            "What follows the heading",
            logger=exception_logger,
        )
        self.assertEqual(len(results), len(ps))

    def test_query_SPARSE_MERGED2(self):

        blocks = self._get_blocks()
        encoding_model = EncodingModel.SPARSE_MERGED
        chunks = blocks_to_chunks(blocks, encoding_model, am)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = "QLT_" + encoding_model.value
        flag = initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            logger=exception_logger,
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
            logger=exception_logger,
        )
        results: list[ScoredPoint] = query(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            "What is the ingested text?",
            logger=exception_logger,
        )
        self.assertEqual(len(results), len(ps))

    def test_query_SPARSE_MERGED3(self):

        blocks = self._get_blocks()
        encoding_model = EncodingModel.SPARSE_MERGED
        chunks = blocks_to_chunks(blocks, encoding_model, am)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = "QLT_" + encoding_model.value
        flag = initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            logger=exception_logger,
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
            logger=exception_logger,
        )
        results: list[ScoredPoint] = query(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            "Why is the sky blue?",
            logger=exception_logger,
        )
        self.assertEqual(len(results), len(ps))

    def test_query_SPARSE_MERGED_MULTIVECTOR(self):

        blocks = self._get_blocks()
        encoding_model = EncodingModel.SPARSE_MULTIVECTOR
        chunks = blocks_to_chunks(blocks, encoding_model, am)
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        embedding_settings = ConfigSettings().embeddings
        collection_name: str = "QLT_" + encoding_model.value
        flag = initialize_collection(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            logger=exception_logger,
        )
        self.assertTrue(flag, "Could not initialize collection")
        ps = upload(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            chunks,
            logger=exception_logger,
        )
        results: list[ScoredPoint] = query(
            client,
            collection_name,
            embedding_model,
            embedding_settings,
            "What is the ingested text?",
            logger=exception_logger,
        )
        self.assertEqual(len(results), len(ps))


if __name__ == "__main__":
    unittest.main()
