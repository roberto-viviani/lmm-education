"""
Converts a list of markdown blocks into a list of chunk objects, which
include all the information for being igested into a vector database.
The list of markdown blocks will have been preprocessed as necessary,
i.e. splitted into smaller text blocks, endowed with metadata.

When using a vector database to store information, data may be used to
obtain embeddings (the semantic representation of the content, which
the database uses to identify text from a query based on similarity),
and to select parts of the information that is stored in the database
and retrieved when records are selected. The Chunk class and its
member methods collects and organizes this information. It constitutes
a framework-neutral replacement for the Document class, commonly used
by frameworks in RAG applications.

Embeddings are increaseangly supported in a variety of configurations,
based on two approaches: dense and sparse vector representations. Once
the data are selected for embeddings and storage, the next logical
step is the definition of how dense and sparse vector representations
are used to compute the embeddings. The Chunk module defines an
_encoding model_ to map the data selected for embedding to the
embedding type supported by the database engine.

Note:
    no embedding is computed here; this is done when the chunks are
    transformed into the format that the vector database requires
    for ingestion.

Responsibilities:
    define encoding models
    complement metadata of headings required by encoding and ingestion
        (such as titles)
    implement the encoding model when transforming blocks to chunks
        (collect the adequate information in dense_encoding and
        sparse_encoding)

Main functions:
    blocks_to_chunks: list of blocks to list of chunks
    chunks_to_blocks: the inverse transformation (for inspection and
        verification).
"""

from pydantic import BaseModel, Field
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
    UUID_KEY,
)


# embedding strategies allowed by the system
from enum import StrEnum  # fmt: skip
class EncodingModel(StrEnum):
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
    an additional uuid field.

    The fields content and metadata contain information that is
    stored in the database.

    The field annotations contains concatenated metadata strings that,
    depending on the encoding model, may end up in the sparse or in
    the dense encoding.

    The fields dense_encoding and sparse_encoding the text that is
    used for embedding using the respective approaches.

    The uuid field contains the id of the database record.

    This replaces the langchain_core.documents.Document class by
    adding an uuid and embedding fields
    """

    content: str = Field(
        description="The textual content for storage in the database"
        + "in the content field of the payload"
    )
    metadata: MetadataDict = Field(
        default={},
        description="Metadata of the original text block"
        + "for storage in the database payload as fields",
    )
    annotations: str = Field(
        default="",
        description="Selected parts of the metadata that may be used "
        + "for encoding",
    )
    dense_encoding: str = Field(
        default="",
        description="The content selected for dense encoding",
    )
    sparse_encoding: str = Field(
        default="",
        description="The content selected for sparse encoding",
    )
    uuid: str = Field(
        default="",
        description="Identification of the record in the database",
    )

    def get_uuid(self) -> str:
        """Return the UUID of the document."""
        if not bool(self.uuid):
            self.uuid = str(uuid4())
        return self.uuid


def blocks_to_chunks(
    blocklist: list[Block],
    encoding_model: EncodingModel,
    annotations_model: list[str] = [TITLES_KEY],
) -> list[Chunk]:
    """Transform a blocklist into a list of chunk objects.

    Implements the encoding model by collecting appropriate data
        and metadata.

    Args:
        blocklist, a list of markdown blocks
        encoding_mode: how to allocate information to dense and
            sparse encoding
        annotations_model: the fields from the metadata to use for
            encoding. Titles, if present, are included if there is
            no model. This field is ignored if the encoding model
            makes no use of annotations.

    Returns:
        a list of Chunk objects
    """

    if not blocklist:
        return []

    if not annotations_model:
        annotations_model = [TITLES_KEY]

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
        for key in annotations_model:
            value: str | None = n.get_metadata_string_for_key(key)
            if value:
                annlist.append(value)

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
        chunks = blocks_to_chunks(blocks, opts, [TITLES_KEY])
        blocks = chunks_to_blocks(chunks, sep="------")
        if blocks:
            save_blocks(target, blocks)
        return blocks

    create_interface(interactive_scan, sys.argv)
