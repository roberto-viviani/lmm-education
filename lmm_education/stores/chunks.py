"""
Converts a list of markdown blocks into a list of chunk objects. The
Chunk class constitutes a framework-neutral replacement for the
Document class, commonly used by framework in RAG applications. Chunk
objects may carry additional information to direct the interaction
with the vector database.

The Chunk module defines an EncodingModel enum to define the
interaction with the vector database. Note: no embedding is computed
here, but the list of chunks can be further processed to get the
embeddings.

Responsibilities:
    define encoding models
    complement metadata of headings required by encoding and ingestion
        (such as titles)
    implement the encoding model when transforming blocks to chunks
        (collect the adequate information in dense_encoding and
        sparse_encoding)

Main inputs: block list

Main functions:
    blocks_to_chunks: blocklist to list of chunks with splitting
    chunks_to_blocks: the inverse transformation
"""

from pydantic import BaseModel
import copy
from uuid import uuid4

# LM markdown
from lmm.markdown.parse_markdown import (
    Block,
    MetadataBlock,
    TextBlock,
)
from lmm.markdown.parse_markdown import (
    blocklist_haserrors,
    blocklist_copy,
)
from lmm.markdown.parse_yaml import MetadataDict
from lmm.markdown.tree import MarkdownTree, MarkdownNode, TextNode
from lmm.markdown.tree import blocks_to_tree, traverse_tree_nodetype
from lmm.markdown.treeutils import inherit_metadata
from lmm.scan.scan_rag import scan_rag, ScanOpts
from lmm.scan.scan_keys import (
    TXTHASH_KEY,
    CHAT_KEY,
    QUERY_KEY,
    EDIT_KEY,
    MESSAGE_KEY,
    TITLES_KEY,
    QUESTIONS_KEY,
    UUID_KEY,
)


# embedding strategies allowed by the system
from enum import Enum  # fmt: skip
class EncodingModel(Enum):
    """Enum for encoding strategies"""

    # No encoding
    NONE = "none"

    # Encode only textual content in dense vector
    CONTENT = "content"

    # Encode textual content merged with metadata
    # annotations in dense vectors
    MERGED = "merged"

    # Encode content and annotations using multivectors
    MULTIVECTOR = "multivector"

    # Sparse encoding of annotations only
    SPARSE = "sparse"

    # Sparse annotations, dense encoding of content
    SPARSE_CONTENT = "sparse_content"

    # Sparse annotations, dense encoding of merged
    # content and annotations
    SPARSE_MERGED = "sparse_merged"

    # Sparse annotations, multivector encoding of merged
    # content and annotations
    SPARSE_MULTIVECTOR = "sparse_multivector"


class Chunk(BaseModel):
    """Class for storing a piece of text and associated metadata, with
    an additional uuid field. dense_encoding and sparse_encoding
    collect the text that is to be embedded with dense and sparse
    embeddings.

    This replaces the langchain_core.documents.Document class by
    adding an uuid and embedding fields
    """

    content: str
    metadata: MetadataDict = {}
    annotations: str = ""
    dense_encoding: str = ""
    sparse_encoding: str = ""
    uuid: str = ""

    def get_uuid(self) -> str:
        """Return the UUID of the document."""
        return self.uuid if self.uuid else str(uuid4())


def blocks_to_chunks(
    blocklist: list[Block], encoding_model: EncodingModel
) -> list[Chunk]:
    """Transform a blocklist into a list of chunk objects.

    Implements the encoding model by collecting appropriate data
        and metadata.

    Args:   blocklist, a list of markdown blocks

    Returns:    a list of Chunk objects
    """

    if not blocklist:
        return []

    # collect or create required metadata for RAG: uuid, titles
    blocks: list[Block] = scan_rag(
        blocklist_copy(blocklist),
        ScanOpts(textid=True, UUID=True, titles=True),
    )
    if blocklist_haserrors(blocks):
        raise ValueError("blocks_to_chunks called with error blocks")

    root: MarkdownTree = blocks_to_tree(blocks)
    if not root:
        return []

    # integrate text node metadata by collecting metadata from parent,
    # unless the metadata properties are already in the text node
    exclude_set: list[str] = [
        TXTHASH_KEY,
        CHAT_KEY,
        QUERY_KEY,
        EDIT_KEY,
        MESSAGE_KEY,
    ]
    rootnode: MarkdownNode = inherit_metadata(root, exclude_set)

    # map a text node with the inherited metadata to a Chunk object
    def _textnode_to_chunk(n: TextNode) -> Chunk:

        # form content and annotations
        meta: MetadataDict = copy.deepcopy(n.metadata)
        annlist: list[str] = []
        title: str | None = n.get_metadata_string_for_key(TITLES_KEY)
        if title:
            annlist.append(title)
        question: str | None = n.get_metadata_string_for_key(
            QUESTIONS_KEY
        )
        if question:
            annlist.append(question)
        chunk: Chunk = Chunk(
            content=n.get_content(),
            annotations="\n".join(annlist),
            uuid=str(meta.pop(UUID_KEY, "")),
            metadata=meta,
        )

        # determine content to be encoded according to model
        match encoding_model:
            case EncodingModel.NONE:
                # no encoding
                pass

            case EncodingModel.CONTENT | EncodingModel.MULTIVECTOR:
                # encode only the content of the text blocks or
                # encode the content and metadata annotations using
                # multivectors
                chunk.dense_encoding = chunk.content

            case EncodingModel.MERGED:
                # encode the content merged with metadata annotations
                chunk.dense_encoding = (
                    chunk.annotations + ": " + chunk.content
                )

            case EncodingModel.SPARSE:
                # sparse encoding of metadata annotations only
                chunk.sparse_encoding = chunk.annotations

            case (
                EncodingModel.SPARSE_CONTENT
                | EncodingModel.SPARSE_MULTIVECTOR
            ):
                # sparse encoding of metadata annotations, dense
                #   encoding of content or
                # sparse encoding of metadata annotations, multidense
                #   encoding of content
                chunk.sparse_encoding = chunk.annotations
                chunk.dense_encoding = chunk.content

            case EncodingModel.SPARSE_MERGED:
                # sparse encoding of metadata annotations, dense
                # encoding of merged content and annotations
                chunk.sparse_encoding = chunk.annotations
                chunk.dense_encoding = (
                    chunk.annotations + ": " + chunk.content
                )

            # let type checker flag missing case's
            # case _:
            #     raise ValueError(
            #         f"Unsupported encoding model: {encoding_model}"
            #     )
        return chunk

    chunks = traverse_tree_nodetype(
        rootnode, _textnode_to_chunk, TextNode
    )
    return [c for c in chunks if c.content]


def chunks_to_blocks(
    chunks: list[Chunk], sep: str = ""
) -> list[Block]:
    """Transform a list of chunks to a list of blocks.

    Args:   chunks, a list of Chunk objects
            sep (opt), an optional separator to visualize the breaks
            between chunks

    Returns: a list of markdown blocks that can be serialized as
            a Markdown document

    Note: the content of the chunk is splitted into a metadata block
        and a text block, containing the 'content' value of the chunk.
    """

    blocks: list[Block] = []
    for c in chunks:
        if sep:
            blocks.append(TextBlock(content=sep))
        if c.metadata:
            blockmeta = c.metadata.copy()
            meta = {
                'uuid': c.uuid,
                'content': "<block content>",
                'annotations': c.annotations,
                'dense_encoding': c.dense_encoding,
                'sparse_encoding': c.sparse_encoding,
            }
            blockmeta['~chunk'] = meta  # type: ignore (safe here)
            blocks.append(MetadataBlock(content=blockmeta))
        blocks.append(TextBlock(content=c.content))

    return blocks


if __name__ == "__main__":
    """Interactive loop to test module"""
    import sys
    from lmm.utils.ioutils import create_interface
    from lmm.markdown.parse_markdown import load_blocks, save_blocks
    from lmm.utils.logging import ILogger, get_logger

    # Set up logger
    logger: ILogger = get_logger(__name__)

    def interactive_scan(filename: str, target: str) -> list[Block]:
        if filename == target:
            logger.info(
                "Usage: output file in second command line arg must "
                "differ from input file"
            )
            return []
        blocks = load_blocks(filename)
        if not blocks:
            return []
        if blocklist_haserrors(blocks):
            logger.warning("Errors in markdown, fix first.")
            return []

        opts: EncodingModel = EncodingModel.MULTIVECTOR
        chunks = blocks_to_chunks(blocks, opts)
        blocks = chunks_to_blocks(chunks, sep="------")
        if blocks:
            save_blocks(target, blocks)
        return blocks

    create_interface(interactive_scan, sys.argv)
