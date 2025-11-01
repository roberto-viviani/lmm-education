"""Tests for grouped qdrant queries"""

import unittest
import asyncio

from langchain_text_splitters import RecursiveCharacterTextSplitter

from lmm.scan.chunks import (
    Chunk,
    EncodingModel,
)
from lmm.markdown.parse_markdown import (
    MetadataBlock,
    Block,
    parse_markdown_text,
    blocklist_haserrors,
)
from lmm.config.config import Settings, export_settings
from lmm.scan.scan_rag import scan_rag, ScanOpts
from lmm.scan.scan_split import scan_split
from lmm.scan.scan_keys import UUID_KEY, GROUP_UUID_KEY, TITLES_KEY
from lmm.scan.chunks import blocks_to_chunks
from lmm_education.stores.vector_store_qdrant import (
    AsyncQdrantClient,
    QdrantEmbeddingModel,
    Point,
    ScoredPoint,
    GroupsResult,
    encoding_to_qdrantembedding_model,
    points_to_ids,
    points_to_text,
    groups_to_points,
    ainitialize_collection,
    aupload,
    aquery,
    aquery_grouped,
)
from lmm_education.config.config import ConfigSettings

from lmm.utils.logging import ExceptionConsoleLogger

default_logger = ExceptionConsoleLogger(
    "test_vector_store_qdrant_groups"
)

# A global client object (for now)
QDRANT_SOURCE = ":memory:"
COLLECTION_MAIN = "Main"
COLLECTION_DOCS = "Main_docs"
client = AsyncQdrantClient(QDRANT_SOURCE)

# same document as in the sync version
from tests.test_vector_store_qdrant_groups import document

# prepare blocklist for RAG
blocklist: list[Block] = parse_markdown_text(document)
if blocklist_haserrors(blocklist):
    raise Exception("Invalid markdown")


def tearDownModule():
    asyncio.run(client.close())


class TestEncoding(unittest.IsolatedAsyncioTestCase):
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

    async def test_encode(self):
        # init collections
        encoding_model_main: EncodingModel = EncodingModel.CONTENT
        embedding_model_main: QdrantEmbeddingModel = (
            encoding_to_qdrantembedding_model(encoding_model_main)
        )
        # encoding_model_companion: EncodingModel = EncodingModel.NONE
        # checks loading content independently by adding encoding to
        # companion collection
        encoding_model_companion: EncodingModel = (
            EncodingModel.CONTENT
        )
        embedding_model_companion: QdrantEmbeddingModel = (
            encoding_to_qdrantembedding_model(
                encoding_model_companion
            )
        )
        embedding_settings = ConfigSettings().embeddings

        if not await ainitialize_collection(
            client,
            COLLECTION_MAIN,
            embedding_model_main,
            embedding_settings,
            logger=default_logger,
        ):
            raise Exception("Could not initialize main collection")
        if not await ainitialize_collection(
            client,
            COLLECTION_DOCS,
            embedding_model_companion,
            embedding_settings,
            logger=default_logger,
        ):
            raise Exception(
                "Could not initialize companion collection"
            )

        blocks: list[Block] = scan_rag(
            blocklist,
            ScanOpts(titles=True, textid=True, UUID=True),
        )

        # ingest text into companion collection
        companion_chunks: list[Chunk] = blocks_to_chunks(
            blocks, encoding_model_companion, [TITLES_KEY]
        )
        companion_points: list[Point] = await aupload(
            client,
            COLLECTION_DOCS,
            embedding_model_companion,
            embedding_settings,
            companion_chunks,
            logger=default_logger,
        )
        self.assertTrue(len(companion_points) > 0)
        companion_uuids: list[str] = points_to_ids(companion_points)
        companion_texts: list[str] = points_to_text(companion_points)

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
        chunks: list[Chunk] = blocks_to_chunks(
            blocks, encoding_model_main, []
        )
        points: list[Point] = await aupload(
            client,
            COLLECTION_MAIN,
            embedding_model_main,
            embedding_settings,
            chunks,
            logger=default_logger,
        )
        self.assertLess(0, len(points))
        texts: list[str] = points_to_text(points)

        # query in the main collection (splitted chunks)
        splitres: list[ScoredPoint] = await aquery(
            client,
            collection_name=COLLECTION_MAIN,
            qdrant_model=embedding_model_main,
            embedding_settings=embedding_settings,
            querytext="How can I estimate the predicted depressiveness from this model?",
            limit=2,
            payload=["page_content", GROUP_UUID_KEY],
            logger=default_logger,
        )
        self.assertLess(0, len(splitres))
        self.assertTrue(splitres[0].payload)
        if splitres[0].payload:
            self.assertIn(splitres[0].payload["page_content"], texts)
            self.assertIn(GROUP_UUID_KEY, splitres[0].payload)

        # query in the companion collection
        if (
            not embedding_model_companion == QdrantEmbeddingModel.UUID
        ):  # content query
            splitres: list[ScoredPoint] = await aquery(
                client,
                COLLECTION_DOCS,
                embedding_model_companion,
                embedding_settings,
                "How can I estimate the predicted depressiveness from this model?",
                logger=default_logger,
                limit=2,
            )
            self.assertLess(0, len(splitres))
            self.assertTrue(splitres[0].payload)
            if splitres[0].payload:
                self.assertIn(
                    splitres[0].payload["page_content"],
                    companion_texts,
                )

        # grouped query
        results: GroupsResult = await aquery_grouped(
            client,
            collection_name=COLLECTION_MAIN,
            group_collection=COLLECTION_DOCS,
            qdrant_model=embedding_model_main,
            embedding_settings=embedding_settings,
            querytext="How can I estimate the predicted depressiveness from this model?",
            group_field=GROUP_UUID_KEY,
            limit=1,
            logger=default_logger,
        )
        result_points: list[ScoredPoint] = groups_to_points(results)
        result_text: list[str] = points_to_text(result_points)

        self.assertLess(0, len(result_points))
        self.assertTrue(result_text[0] in companion_texts)
        self.assertTrue(results.groups[0].id in companion_uuids)

    async def _do_test_encode_grouped(
        self, encoding_model: EncodingModel
    ):
        from lmm_education.config.config import AnnotationModel
        from lmm.scan.scan_keys import QUESTIONS_KEY

        # init collections
        encoding_model_main: EncodingModel = encoding_model
        embedding_model_main: QdrantEmbeddingModel = (
            encoding_to_qdrantembedding_model(encoding_model_main)
        )
        # encoding_model_companion: EncodingModel = EncodingModel.NONE
        # checks loading content independently by adding encoding to
        # companion collection
        encoding_model_companion: EncodingModel = (
            EncodingModel.CONTENT
        )
        embedding_model_companion: QdrantEmbeddingModel = (
            encoding_to_qdrantembedding_model(
                encoding_model_companion
            )
        )
        embedding_settings = ConfigSettings().embeddings

        if not await ainitialize_collection(
            client,
            COLLECTION_MAIN + str(encoding_model),
            embedding_model_main,
            embedding_settings,
            logger=default_logger,
        ):
            raise Exception("Could not initialize main collection")
        if not await ainitialize_collection(
            client,
            COLLECTION_DOCS + str(encoding_model),
            embedding_model_companion,
            embedding_settings,
            logger=default_logger,
        ):
            raise Exception(
                "Could not initialize companion collection"
            )

        blocks: list[Block] = scan_rag(
            blocklist,
            ScanOpts(titles=True, textid=True, UUID=True),
        )
        self.assertTrue(len(blocks) > 0)

        # ingest text into companion collection
        companion_chunks: list[Chunk] = blocks_to_chunks(
            blocks,
            encoding_model_companion,
            AnnotationModel(inherited_properties=[QUESTIONS_KEY]),
        )
        companion_points: list[Point] = await aupload(
            client,
            COLLECTION_DOCS + str(encoding_model),
            embedding_model_companion,
            embedding_settings,
            companion_chunks,
            logger=default_logger,
        )
        self.assertTrue(len(companion_points) > 0)
        self.assertEqual(len(companion_points), len(companion_chunks))
        companion_uuids: list[str] = points_to_ids(companion_points)
        companion_texts: list[str] = points_to_text(companion_points)

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
        chunks: list[Chunk] = blocks_to_chunks(
            blocks,
            encoding_model_main,
            annotation_model=AnnotationModel(
                inherited_properties=[QUESTIONS_KEY]
            ),
        )
        points: list[Point] = await aupload(
            client,
            COLLECTION_MAIN + str(encoding_model),
            embedding_model_main,
            embedding_settings,
            chunks,
            logger=default_logger,
        )
        self.assertLess(
            0, len(points), "Failure for " + str(encoding_model)
        )
        self.assertEqual(len(points), len(chunks))
        texts: list[str] = points_to_text(points)

        # query in the main collection (splitted chunks)
        splitres: list[ScoredPoint] = await aquery(
            client,
            collection_name=COLLECTION_MAIN + str(encoding_model),
            qdrant_model=embedding_model_main,
            embedding_settings=embedding_settings,
            querytext="What is the model equation?",
            limit=2,
            payload=['page_content', GROUP_UUID_KEY],
            logger=default_logger,
        )
        self.assertLess(
            0, len(splitres), "Failure for " + str(encoding_model)
        )
        self.assertTrue(splitres[0].payload)
        if splitres[0].payload:
            self.assertIn(splitres[0].payload['page_content'], texts)
            self.assertIn(GROUP_UUID_KEY, splitres[0].payload)

        # query in the companion collection
        if (
            not embedding_model_companion == QdrantEmbeddingModel.UUID
        ):  # content query
            splitres: list[ScoredPoint] = await aquery(
                client,
                COLLECTION_DOCS + str(encoding_model),
                embedding_model_companion,
                embedding_settings,
                "What is the model equation?",
                logger=default_logger,
            )
            self.assertLess(
                0, len(splitres), "Failure for " + str(encoding_model)
            )
            self.assertTrue(
                splitres[0].payload,
                "Failure for " + str(encoding_model),
            )
            if splitres[0].payload:
                self.assertIn(
                    splitres[0].payload['page_content'],
                    companion_texts,
                    "Failure for " + str(encoding_model),
                )

        # grouped query
        results: GroupsResult = await aquery_grouped(
            client,
            collection_name=COLLECTION_MAIN + str(encoding_model),
            group_collection=COLLECTION_DOCS + str(encoding_model),
            qdrant_model=embedding_model_main,
            embedding_settings=embedding_settings,
            querytext="What is the model equation?",
            group_field=GROUP_UUID_KEY,
            limit=1,
            logger=default_logger,
        )
        result_points: list[ScoredPoint] = groups_to_points(results)
        result_text: list[str] = points_to_text(result_points)

        self.assertLess(
            0,
            len(result_points),
            "Failure for " + str(encoding_model),
        )
        self.assertTrue(
            result_text[0] in companion_texts,
            "Failure for " + str(encoding_model),
        )
        self.assertTrue(
            results.groups[0].id in companion_uuids,
            "Failure for " + str(encoding_model),
        )

    async def test_encode_encodings(self):
        # EncodingModel.CONTENT already tested, test rest
        models: list[EncodingModel] = [
            EncodingModel.MERGED,
            EncodingModel.SPARSE,
            EncodingModel.MULTIVECTOR,
            EncodingModel.SPARSE_MERGED,
            EncodingModel.SPARSE_CONTENT,
            EncodingModel.SPARSE_MULTIVECTOR,
        ]
        for encoding_model in models:
            await self._do_test_encode_grouped(encoding_model)


if __name__ == "__main__":
    unittest.main()
