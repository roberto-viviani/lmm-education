"""tests for vector_store_qdrant.py

NOTE: initialize collection tested in test_lmm_rag.py
"""

# flake8: noqa

import unittest

from lmm.markdown.parse_markdown import *
from lmm_education.stores.chunks import *
from lmm_education.stores.vector_store_qdrant import *

from lmm_education.config.config import ConfigSettings

settings = ConfigSettings()


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
blocks = blocklist_rag(blocks, ScanOpts(textid=True, textUUID=True))


class TestInitialization(unittest.TestCase):

    def test_encoding_none(self):
        encoding_model = EncodingModel.NONE
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        result = initialize_collection(
            client, collection_name, embedding_model, settings
        )
        self.assertFalse(result)

    def test_encoding_content(self):
        encoding_model = EncodingModel.CONTENT
        embedding_model = encoding_to_qdrantembedding_model(
            encoding_model
        )
        collection_name: str = encoding_model.value
        flag = initialize_collection(
            client, collection_name, embedding_model, settings
        )
        self.assertEqual(flag, False)


if __name__ == '__main__':
    unittest.main()
