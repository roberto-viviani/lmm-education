"""
Converts a list of markdown blocks into a list of `Chunk` objects,
which include all the information for being ingested into a vector
database. The list of markdown blocks will have been preprocessed as
necessary, i.e. splitted into smaller text blocks, endowed with
metadata and an uuid identification code.

When using a vector database to store information, data may be used to
obtain embeddings (the semantic representation of the content, which
the database uses to identify text from a query based on similarity),
and to select parts of the information that is stored in the database
and retrieved when records are selected. These two sets of information
are often the same, but they not need be. The `Chunk` class and its
member methods collects and organizes this information. It constitutes
a framework-neutral replacement for the Document class, commonly used
by frameworks in RAG applications.

Embeddings are increasingly supported in a variety of configurations.
Besides the data selected for storage, portion of data may be selected
to compute the embeddings. This module defines an _encoding model_ to
map the data selected for embedding to the embedding type supported by
the database engine. In what follows, the metadata properties used to
generate embeddings are called 'annotations', to distinguish them from
other properties (among others, metadata properties used for
housekeeping purposes).

The annotation model not only specifies what metadata properties are
included in the embedding, but also whether to look for them in the
ancestors of the markdown text, represented as a hierachical tree
where headings are the nodes in the hierarchy. The _encoding model_
further specifies how the annotations are used in dense and sparse
encodings.

Example:

```python
from lmm.markdown.parse_markdown import (
    blocklist_haserrors,
)
from lmm.scan.scan import markdown_scan
from lmm.scan.scan_rag import scan_rag, ScanOpts
from lmm.scan.scan_keys import TITLES_KEY
from lmm.utils.logging import LoglistLogger
from lmm_education.config.config import (
    AnnotationModel,
    EncodingModel,
)
from lmm_education.stores.chunks import blocks_to_chunks

logger = LoglistLogger()

# the starting point is a list of blocks, such as one originated
# from parsing a markdown file
blocks = markdown_scan("mymarkdown.md")
if blocklist_haserrors(blocks)
    raise ValueError("Errors in  markdown")

# add metadata for annotations (here titles)
blocks = scan_rag(blocks, ScanOpts(titles=True), logger)
if logger.count_logs(level=1) > 0:
    raise ValueError("\n".join(logger.get_logs(level=1)))

# transform to chunks specifying titles for annotations
encoding_model = EncodingModel.SPARSE_CONTENT
chunks = blocks_to_chunks(
    blocks,
    annotation_model=AnnotationModel(
        inherited_properties=[TITLES_KEY]
    ),
    encoding_model=encoding_model,
    logger=logger,
)

# now chunks can be ingested
from lmm_education.stores.vector_store_qdrant import (
    upload,
    client_from_config,
    encoding_to_qdrantembedding_model as to_embedding_model,
)
from lmm_education.config.config import (
    ConfigSettings,
    LocalStorage,
)

settings = ConfigSettings(
    storage=LocalStorage(folder="./test_storage")
)
points = upload(
    client=client_from_config(settings, logger),
    collection_name="documents",
    model=to_embedding_model(encoding_model),
    chunks=chunks,
    logger=logger,
)

if logger.count_logs(level=1) > 0:
    raise ValueError("Could not ingest blocks")
```

Note:
    no embedding is computed here; this is done by the upload function
    in the example above.

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
    UUID_KEY,
)
from lmm.utils.logging import LoggerBase, get_logger

# Set up logger
logger: LoggerBase = get_logger(__name__)

from lmm_education.config.config import AnnotationModel, EncodingModel


class Chunk(BaseModel):
    """
    Class for storing a piece of text and associated metadata, with
    an additional uuid field for its identification in the database.
    Each instanch of this class becomes a record or 'point' in the
    database.

    The fields `content` and `metadata` contain information that will
    be stored in the database. The field `content` is meant to contain
    the text. The field `metadata` contains an associative array. (In
    some databases, there is no difference in the way material is
    stored, i.e. text is one field among many possible others; the
    distinction is present in many frameworks, however).

    The field `annotations` contains concatenated metadata strings
    that, depending on the encoding model, may end up in the sparse
    or in the dense encoding.

    The fields `dense_encoding` and `sparse_encoding` contain the text
    that is used for embedding using the respective approaches.

    The `uuid` field contains the id of the database record.
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
    annotation_model: AnnotationModel | list[str] = AnnotationModel(),
    logger: LoggerBase = logger,
) -> list[Chunk]:
    """
    Transform a blocklist into a list of `Chunk` objects.

    Implements the encoding model by collecting appropriate data
        and metadata.

    Args:
        blocklist: a list of markdown blocks
        encoding_model: how to allocate information to dense and
            sparse encoding
        annotation_model: the fields from the metadata to use for
            encoding. Titles, if present, are included if there is
            no model. This field is ignored if the encoding model
            makes no use of annotations.

    Returns:
        a list of `Chunk` objects

    Note:
        this function only encodes text blocks. Markdown documents
        consisting only of headings and metadata are considered
        empty.
    """

    if not blocklist:
        return []

    if isinstance(annotation_model, list):
        annotation_model = AnnotationModel(
            inherited_properties=annotation_model
        )

    # collect or create required metadata for RAG: uuid, textid
    blocks: list[Block] = scan_rag(
        blocklist_copy(blocklist),
        ScanOpts(textid=True, UUID=True),
        logger,
    )
    if blocklist_haserrors(blocks):
        logger.error("blocks_to_chunks called with error blocks")
        return []

    root: MarkdownTree = blocks_to_tree(blocks, logger)
    if not root:
        return []

    # integrate text node metadata by collecting metadata from parent,
    # unless metadata are already specified in the text node. These
    # metadata will be stored in the database as payload. This will
    # not inherit specific properties from ancestors, only the first
    # metadata block on the ancestor's path. We exclude metadata
    # properties that are used to chat and housekeeping.
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
        # annotations
        annlist: list[str] = []
        value: str | None = None
        for key in annotation_model.inherited_properties:
            value = n.fetch_metadata_string_for_key(key, False)
            if value:
                annlist.append(value)
        for key in annotation_model.own_properties:
            value = n.get_metadata_string_for_key(key)
            if value:
                annlist.append(value)

        # metadata for payload
        meta: MetadataDict = copy.deepcopy(n.metadata)
        for key in exclude_set:
            meta.pop(key, None)
        chunk: Chunk = Chunk(
            content=n.get_content(),
            annotations="\n".join(annlist),
            uuid=str(meta.pop(UUID_KEY, "")),
            metadata=meta,
        )

        # determine content to be encoded according to encoding model
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
    chunks: list[Chunk], sep: str = "", key_chunk: str = "~chunk"
) -> list[Block]:
    """
    Transform a list of `Chunk` objects to a list of blocks.

    Args:
        chunks: a list of `Chunk` objects
        sep: an optional separator to visualize the breaks
            between chunks
        key_chunk: the metadata key where the chunk is copied into

    Returns:
        a list of markdown blocks that can be serialized as
            a Markdown document

    Note:
        the content of the chunk is splitted into a metadata block
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
            blockmeta[key_chunk] = meta  # type: ignore (safe here)
            blocks.append(MetadataBlock(content=blockmeta))
        blocks.append(TextBlock(content=c.content))

    return blocks


def serialize_chunks(
    chunks: list[Chunk], sep: str = "", key_chunk: str = "~chunk"
) -> str:
    """
    Serialize a list of `Chunk`objects for debug/inspection purposes.
    See chunks_to_blocks.
    """

    # lazy load
    from lmm.markdown.parse_markdown import serialize_blocks

    return serialize_blocks(chunks_to_blocks(chunks, sep, key_chunk))


if __name__ == "__main__":
    """Interactive loop to test module"""
    import sys
    from lmm.utils.ioutils import create_interface
    from lmm.markdown.parse_markdown import load_blocks, save_blocks

    def interactive_scan(filename: str, target: str) -> list[Block]:
        if filename == target:
            logger.info(
                "Usage: output file in second command line arg must "
                "differ from input file"
            )
            return []
        blocks = load_blocks(filename, logger)
        if not blocks:
            return []
        if blocklist_haserrors(blocks):
            logger.warning("Errors in markdown, fix first.")
            return []

        opts: EncodingModel = EncodingModel.MULTIVECTOR
        chunks: list[Chunk] = blocks_to_chunks(
            blocks, encoding_model=opts
        )
        blocks = chunks_to_blocks(chunks, sep="------")
        if blocks:
            save_blocks(target, blocks, logger)
        return blocks

    create_interface(interactive_scan, sys.argv)
