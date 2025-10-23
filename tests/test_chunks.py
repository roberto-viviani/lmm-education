"""tests for chunk.py"""

# pyright: basic
# pyright: reportMissingTypeStubs=false

import unittest

from lmm_education.config.config import AnnotationModel
from lmm_education.stores.chunks import (
    blocks_to_chunks,
    chunks_to_blocks,
    EncodingModel,
)
from lmm.markdown.parse_markdown import (
    HeaderBlock,
    MetadataBlock,
    HeadingBlock,
    TextBlock,
    ErrorBlock,
    Block,
)
from lmm.scan.scan_keys import (
    QUESTIONS_KEY,
    CHAT_KEY,
    SUMMARY_KEY,
    TITLES_KEY,
)

header = HeaderBlock(content={'title': "Test blocklist"})
metadata = MetadataBlock(
    content={
        QUESTIONS_KEY: "What is the nature of the test? - How can we fix it?",
        CHAT_KEY: "Some discussion",
        SUMMARY_KEY: "The summary of the text.",
    }
)
heading = HeadingBlock(level=2, content="First title")
text = TextBlock(content="This is text following the heading")

blocks: list[Block] = [header, metadata, heading, text]
lenblocks: int = len(blocks)


class TestChunkNulls(unittest.TestCase):

    def test_empty_list(self):
        chunks = blocks_to_chunks([], EncodingModel.CONTENT)
        self.assertEqual(len(chunks), 0)

    def test_nontext_list(self):
        chunks = blocks_to_chunks(
            [header, metadata, heading], EncodingModel.CONTENT
        )
        self.assertEqual(len(chunks), 0)

    def test_dangling_metadata(self):
        # this creates an empty chunk list as there is no text
        # content.
        chunks = blocks_to_chunks(
            [header, metadata, heading, metadata],
            EncodingModel.CONTENT,
        )
        self.assertEqual(len(chunks), 0)


class TestBlocklistWithErrors(unittest.TestCase):

    def test_blocklist_errors(self):
        # expected behaviour: empty chunk list
        chunks = blocks_to_chunks(
            blocks + [ErrorBlock(content="Invalid block here")],
            EncodingModel.CONTENT,
        )
        self.assertEqual(len(chunks), 0)


class TestChunkFormation(unittest.TestCase):

    def test_transf_and_inverse(self) -> None:
        chunks = blocks_to_chunks(blocks, EncodingModel.CONTENT)
        reformed_blocks = chunks_to_blocks(chunks, "")
        # expected result: a metadata and a text block
        self.assertEqual(len(reformed_blocks), 2)
        self.assertIsInstance(reformed_blocks[0], MetadataBlock)
        self.assertIsInstance(reformed_blocks[1], TextBlock)


class TestChunkInheritance(unittest.TestCase):

    def test_annotate_questions(self):
        from lmm.scan.scan_rag import scan_rag, ScanOpts
        from lmm.markdown.parse_markdown import blocklist_copy

        self.assertEqual(len(blocks), lenblocks)
        blocklist = scan_rag(
            blocklist_copy(blocks),
            ScanOpts(titles=True, questions=True),
        )
        print(blocklist)
        annotation_model = AnnotationModel(
            inherited_properties=[QUESTIONS_KEY, TITLES_KEY],
        )
        chunks = blocks_to_chunks(
            blocklist, EncodingModel.CONTENT, annotation_model
        )
        self.assertEqual(len(blocklist), lenblocks)
        chunk = chunks[0]
        print(chunk.annotations)
        self.assertTrue(
            str(metadata.content[QUESTIONS_KEY]) in chunk.annotations
        )
        self.assertTrue(
            str("Test blocklist - First title") in chunk.annotations
        )

    def test_inherit_summary(self):
        self.assertEqual(len(blocks), lenblocks)
        chunks = blocks_to_chunks(blocks, EncodingModel.CONTENT)
        self.assertEqual(len(blocks), lenblocks)
        chunk = chunks[0]
        self.assertTrue(QUESTIONS_KEY in chunk.metadata)
        self.assertTrue(SUMMARY_KEY in chunk.metadata)


class TestChunkEncoding(unittest.TestCase):

    def test_encoding_NULL(self):
        chunks = blocks_to_chunks(blocks, EncodingModel.NONE)
        self.assertEqual(len(blocks), lenblocks)
        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.dense_encoding, "")

    def test_encoding_CONTENT(self):
        chunks = blocks_to_chunks(blocks, EncodingModel.CONTENT)
        self.assertEqual(len(blocks), lenblocks)
        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.dense_encoding, text.get_content())

    def test_encoding_MERGED(self):
        chunks = blocks_to_chunks(blocks, EncodingModel.MERGED)
        self.assertEqual(len(blocks), lenblocks)
        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(
            chunk.dense_encoding,
            chunk.annotations + ": " + text.get_content(),
        )

    def test_encoding_SPARSE(self):
        chunks = blocks_to_chunks(blocks, EncodingModel.SPARSE)
        self.assertEqual(len(blocks), lenblocks)
        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.sparse_encoding, chunk.annotations)

    def test_encoding_SPARSE_CONTENT(self):
        chunks = blocks_to_chunks(
            blocks, EncodingModel.SPARSE_CONTENT
        )
        self.assertEqual(len(blocks), lenblocks)
        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.sparse_encoding, chunk.annotations)
        self.assertEqual(chunk.dense_encoding, text.get_content())

    def test_encoding_SPARSE_MERGED(self):
        chunks = blocks_to_chunks(blocks, EncodingModel.SPARSE_MERGED)
        self.assertEqual(len(blocks), lenblocks)
        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.sparse_encoding, chunk.annotations)
        self.assertEqual(
            chunk.dense_encoding,
            chunk.annotations + ": " + text.get_content(),
        )


if __name__ == '__main__':
    unittest.main()
