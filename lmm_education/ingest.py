"""
This module provides the facilities to ingest markdown files into a
vector database for the LM markdown for education project.

Prior to ingesting, the files are processed according to specifications
stored in an IndexOpts object. The IndexOpts contains the specification of
what annotations to create (questions, summaries) and how to use them
in the encoding of the database. Once initialized, the IndexOpts object
can be passed to the functions of the package as well as in the retrieval
phase.

The information and the stdorage options are the following:

    DatabaseSource: one of
        'memory'
        "./storage" | LocalStorage(folder = "./storage")
        RemoteSource(url = "1.1.1.127", port = 21465)

    IndexOpts:
    collection_name: the collection to use for ingestion
    encoding_model (enum EncodingModel): the model used for encoding
        (what input is used to generate the embedding vectors)
    questions (bool): create questions for the textual content for
        each markdown heading (note: existing questions in the
        metadata before the heading will be used if present). (note:
        running titles will always be added); default False
    summaries (bool): summarize textual content under each heading
        while including the summaries of sub-headings (note: existing
        summaries will be used, if the text was not changed since the
        time of the summary generation); default False
    pool_threshold (int): pool the text under each heading prior to
        chunking. Possible values: 0 (do not pool), -1 (pool all text
        under the heading together, positive number: pool text chunks
        under a heading together unless the numer of words in the
        pooled text crosses the threshold). Note: equation and code
        chunks are pooled with surrounding text irrespective of the
        option chosen here; default 0 (do not pool)
    companion_collection (bool): create a companion collection
        containing the pooled text prior to chunking. The companion
        collection supports group_by queries returning the pooled text
        instead of the text used for embedding; default False
    text_splitter: the splitter class that will be used to split the
        text into chunks (note: chunking takes place after pooling)

Encoding models (.chunks.EncodingModel):

    NONE: no encoding (used to retrieve data via UUID)
    CONTENT: Encode only textual content in dense vector
    MERGED: Encode textual content merged with metadata annotations in
        dense vectors
    MULTIVECTOR: Encode content and annotions using multivectors
    SPARSE: Sparse encoding of annotations only
    SPARSE_CONTENT: Sparse annotations, dense encoding of content
    SPARSE_MERGED: Sparse annotations, dense encoding of merged
        content and annotations
    SPARSE_MULTIVECTOR: Sparse annotations, multivector encoding of
        merged content and annotations

Here, 'annotations' are the titles and questions added to the markdown

Uses:
    scan_rag, scan_split to modify parsed markdown blocks and split
        text chunks
    vector_store_qdrant: to implement encoding model and upload
        material to the vector database

Main functions:
    markdown_upload: ingest markdown files
"""

from pathlib import Path
from pydantic import BaseModel, validate_call, Field, HttpUrl
from typing import Literal

# LM markdown
from lmm.markdown.parse_markdown import (
    Block,
    MetadataBlock,
    ErrorBlock,
    blocklist_haserrors,
    blocklist_errors,
    blocklist_copy,
)
from lmm.markdown.blockutils import (
    merge_code_blocks,
    merge_equation_blocks,
)
from lmm.markdown.tree import (
    blocks_to_tree,
    tree_to_blocks,
)
from lmm.markdown.treeutils import propagate_property
from lmm.markdown.ioutils import save_markdown
from lmm.utils.ioutils import append_postfix_to_filename
from lmm.scan.scan import (
    markdown_scan,
)
from lmm.scan.scan_rag import (
    ScanOpts,
    scan_rag,
)
from lmm.scan.scan_keys import (
    UUID_KEY,
    GROUP_UUID_KEY,
    SUMMARY_KEY,
)
from lmm.scan.scan_split import (
    NullTextSplitter,
    defaultSplitter,
    scan_split,
)

from lmm_education.stores import (
    Chunk,
    EncodingModel,
    blocks_to_chunks,
    chunks_to_blocks,
    QdrantEmbeddingModel as EmbeddingModel,
    encoding_to_qdrantembedding_model as encoding_to_embedding_model,
    initialize_collection,
    upload,
)

# langchain, import text splitters from here
from langchain_text_splitters import TextSplitter

# qdrant
from qdrant_client import QdrantClient

# We import from utils a logger to the console
from lmm.utils import logger


# We set up here the different ways in which we will extract
# properties from our markdown documents, so that we can experiment
# with different options.
# The encoding model specifies the range of encoding options that are
# available in qdrant databases, which include hybrid dense+sparse
# encodings and multivector encodings.
class IndexOpts(BaseModel):
    """This is used to fix a set of arguments to send to
    diverse functions."""

    collection_name: str = Field(
        default="chunks",
        min_length=1,
        description="The name of the collection used in the database "
        + "to store the chunks of text that will be retrieved by "
        + "similarity with the question of the user.",
    )
    encoding_model: EncodingModel = Field(
        default=EncodingModel.CONTENT,
        description="How the chunk propoerties are being encoded",
    )
    questions: bool = Field(
        default=False,
        description="Annotate text with questions to aid retrieval",
    )
    summaries: bool = Field(
        default=False,
        description="Add summaries as chunks to aid retrieval",
    )
    companion_collection: str | None = Field(
        default=None,
        description="Use a companion collection to store the text "
        + "from which the chunks were taken, providing the language "
        + "model with larger, contexdt-rich input, instead of the "
        + "chunks retrieved through the questions fo the user. This"
        + "strategy is a simple form of graph RAG, i.e. the use"
        + "of properties of documents to enrich context. If set to"
        + "None, no companion collection will be used; provide the "
        + "name of the collection to create one, for example "
        + "'documents'.",
    )
    text_splitter: TextSplitter = Field(
        default=NullTextSplitter(),
        description="Provide the text splitter object to split "
        + "text into chunks. The default uses the text blocks of"
        + "the markdown as the chunks.",
    )


# This constrains the way in which the database may be specified.
class LocalStorage(BaseModel):
    folder: str = Field(
        ..., min_length=1, description="Path to the vector database"
    )


class RemoteSource(BaseModel):
    url: HttpUrl = Field(..., description="URL of the remote source.")
    port: int = Field(
        ...,
        gt=0,
        lt=65536,
        description="Port number for the remote source (1-65535).",
    )


DatabaseSource = Literal['memory'] | str | LocalStorage | RemoteSource


# Instead of providing calls such as create_database, open_connection
# etc. we rely on an initialize_client function to create the database
# if it does not exist, initializing the collections according to the
# IndexOpts object. When writing or reading to the database, the
# IndexOpts object will make sure that the data are consistent with
# the schema of the database.
@validate_call(config={'arbitrary_types_allowed': True})
def initialize_client(
    database_source: DatabaseSource,
    opts: IndexOpts,
) -> QdrantClient | None:
    """Initialize QDRANT database. Will open or create the specified
    database and open or create the collection specified in opts.

    Args:
        database_source: e.g. one of
            'memory'
            LocalStorage(folder = "./storage")
            RemoteSource(url = "1.1.1.127", port = 21465)
        opts: IndexOpts

    Returns: a Qdrant client object
    """

    collection_name: str = opts.collection_name
    if not collection_name:
        return None

    # TODO: errors and logs
    client: QdrantClient
    match database_source:
        case 'memory':
            client = QdrantClient(':memory:')
        case str() as folder if len(folder) > 1:
            client = QdrantClient(path=folder)
        case LocalStorage(folder=folder):
            client = QdrantClient(path=folder)
        case RemoteSource(url=url, port=port):
            client = QdrantClient(url=str(url), port=port)
        case _:
            raise ValueError("Invalid database source")

    flag: bool = initialize_collection(
        client,
        collection_name,
        encoding_to_embedding_model(opts.encoding_model),
    )
    if not flag:
        return None

    if bool(opts.companion_collection):
        flag = initialize_collection(
            client,
            opts.companion_collection,
            EmbeddingModel.UUID,
        )
        if not flag:
            return None

    return client


# This is the function to be called to upload a list of markdown files
# specified as a list of path names. It calls initialize_client on the
# specified qdrant database, loads and parses the markdown files, and
# passes them to the upload_blocks function for further processing.
@validate_call(config={'arbitrary_types_allowed': True})
def markdown_upload(
    database_source: DatabaseSource,
    sources: list[str] | list[Path],
    index_opts: IndexOpts,
    *,
    save_files: bool = False,
    ingest: bool = True,
) -> list[list[Chunk]]:
    """Upload a list of markdown files in the vector database.

    Args:
        database_source: one of
            'memory'
            "./storage" | LocalStorage(folder = "./storage")
            RemoteSource(url = "1.1.1.127", port = 21465)
        sources: a list of file names containing the markdown files
        index_opts:
            collection_name: The collection in the database to store
                data in
            questions: Whether to generate questions from the
                content (default: False)
            summaries: Whether to generate summaries for the
                content (default: False)
            pool_threshold: Threshold for pooling text blocks (-1:
                merge under headings, 0: no pooling, >0: pool until
                threshold) (default: 0)
            document_collection: Whether to build a separate document
                collection (default: True)
            text_splitter: The text splitter to use for chunking
                (default: default splitter)
        ingest: Whether to ingest documents into the vector database
            (default: True)
        save_files: Whether to save processed blocks to files
            (default: False)

    Returns:
        A list of Document objects representing the processed content
    """

    # Load files. This is done using markdown_scan, which ensures
    # that there is a header with a title (this is needed to idnetify
    # the documents at ingestion).
    # The markdown parser responds to errors by creating a special
    # type of block, the ErrorBlock. This blocks can be serialized
    # back to document, giving the user a chance to address the error,
    # but here we just stop the process if there are parse errors in
    # any of the documents.
    parsed_documents: list[list[Block]] = []
    error_sources: dict[str, list[ErrorBlock]] = {}
    for s in sources:
        blocks = markdown_scan(s, False)
        if blocklist_haserrors(blocks):
            error_sources[str(s)] = blocklist_errors(blocks)
            continue
        parsed_documents.append(blocks)
    if not parsed_documents:
        return []
    if error_sources:
        logger.error(
            "Problems in markdowns, fix before continuing:\n\t"
            + "\n\t".join(error_sources.keys())
        )
        return []

    # initialize qdrant client. You pass index_opts to this
    # function and it will validate that the database that is
    # being written into has a schema that corresponds to the
    # encoding model applied to the documents. If the database
    # does not exist, it will be created.
    client: QdrantClient | None = initialize_client(
        database_source, index_opts
    )
    if not client:
        return []

    # the encoding strategy may use two collections rather than
    # one: the chunks collection is used to match the query, and
    # the documents collection returns pre-ingested content
    # that exploits the structure of the directed acyclic graph
    # of documents to extract context.
    doclist: list[list[Chunk]] = []
    for docblocks in parsed_documents:
        doclist.extend(
            blocklist_encode_and_upload(
                client, docblocks, index_opts, ingest=ingest
            )
        )

    if not doclist:
        return []

    # this allows users to inspect the annotations that were
    # used to encode the document
    if save_files and len(doclist) > 1:
        newblocks: list[Block] = chunks_to_blocks(doclist[1])
        for s in sources:
            newfile: str = append_postfix_to_filename(
                str(s), "_documents"
            )
            save_markdown(newfile, newblocks)

    if save_files:
        newblocks: list[Block] = chunks_to_blocks(doclist[0])
        for s in sources:
            newfile: str = append_postfix_to_filename(
                str(s), "_chunks"
            )
            save_markdown(newfile, newblocks)

    return doclist


# This is the internal function doing the job of uploading parsed
# markdown files into the database. It implements the encoding
# strategy specified in the IndexOpts object, adding annotations
# using a language model if the strategy requires it. It is the
# key point where an encoding specification and strategies to
# extract further information from mere text is coded.
def blocklist_encode_and_upload(
    client: QdrantClient,
    blocklist: list[Block],
    opts: IndexOpts,
    ingest: bool = True,
) -> list[list[Chunk]]:
    """Upload a list of markdown blocks to the database client.
    Preprocessing will be executed at this stage as specified in the
    IndexOpts object opts.

    Args:
        client: the QdrantClient
        blocklist: the list of markdown blocks
        opts:
            collection_name: The collection in the database to store
                data in (default: "")
            questions: Whether to generate questions from the
                content (default: False)
            summaries: Whether to generate summaries for the
                content (default: False)
            pool_threshold: Threshold for pooling text blocks (-1:
                merge under headings, 0: no pooling, >0: pool until
                threshold) (default: 0)
            document_collection: Whether to build a separate document
                collection (default: False)
            text_splitter: The text splitter to use for chunking
                (default: no splitting)
        ingest: Whether to ingest documents into the vector database
            (default: True)

    Returns:
        A list of two Document object list, representing the processed
        content. If document_collection is True, the list contains a
        second element with the pooled text document list.
    """

    if not blocklist:
        return []
    if blocklist_haserrors(blocklist):
        logger.warning("Problems in markdown, fix before continuing")
        return []

    # preprocessing for RAG. You tell scan_rag what annotations
    # to make and the pooling of text blocks prior to the annotations
    scan_opts = ScanOpts(
        titles=True,
        textid=bool(opts.companion_collection),
        UUID=bool(opts.companion_collection),
        questions=bool(opts.questions),
        summaries=bool(opts.summaries),
    )
    blocks: list[Block] = scan_rag(blocklist, scan_opts)
    if not blocks:
        return []

    # ingest here the large text portions
    if opts.companion_collection:
        doc_coll_name: str = opts.companion_collection

        # create embeddings
        coll_chunks: list[Chunk] = blocks_to_chunks(
            blocks, EncodingModel.NONE
        )
        if not coll_chunks:
            return []

        try:
            if ingest and not upload(
                client,
                doc_coll_name,
                EmbeddingModel.UUID,
                coll_chunks,
            ):
                logger.error(
                    "Could not upload documents to " + doc_coll_name
                )
                return []
        except Exception as e:
            logger.error("Could not upload documents:\n" + str(e))
            return []
    else:
        coll_chunks = []

    # before chunking, copy the UUID in the metadata to a group_UUID
    # key in the metadata of the chunks to allow qdrant GROUP_BY
    # queries
    for b in blocks:
        if isinstance(b, MetadataBlock):
            if UUID_KEY in b.content.keys():
                b.content[GROUP_UUID_KEY] = b.content.pop(UUID_KEY)

    # chunking
    splits: list[Block] = scan_split(blocks, opts.text_splitter)

    # we avoid chunks that isolate or split equations or code to keep
    # those together with textual context to improve the embedding
    splits = merge_code_blocks(splits)
    splits = merge_equation_blocks(splits)

    # summaries. To add summaries as additional chunks, we transform
    # the summary property into a child text block of the heading node
    def _propagate_summaries(blocks: list[Block]) -> list[Block]:
        root = blocks_to_tree(blocklist_copy(blocks))
        if not root:
            return []

        # propagate_property acts on heading nodes moving the content
        # of the metadata from the metadata to a child text node. If
        # add_type_property is set to True, the child text node will
        # have a metadata property marking its content type as summary
        add_type_property: bool = True
        root = propagate_property(
            root, SUMMARY_KEY, add_type_property
        )
        return tree_to_blocks(root)

    if bool(opts.summaries):
        splits = _propagate_summaries(splits)

    # the blocks_to_chunks function transforms the chunks into the
    # format recognized by qdrant for ingestion, also adding the
    # embedding specified by the encoding model.
    chunks: list[Chunk] = blocks_to_chunks(
        splits, opts.encoding_model
    )
    if not chunks:
        return []

    # ingestion
    try:
        model: EmbeddingModel = encoding_to_embedding_model(
            opts.encoding_model
        )
        if ingest and not upload(
            client, opts.collection_name, model, chunks
        ):
            logger.error("Could not upload documents")
            return []
    except Exception as e:
        logger.error("Could not upload documents:\n" + str(e))
        return []

    return [chunks, coll_chunks]


if __name__ == "__main__":
    """Interactive loop to test module"""
    import sys

    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        logger.info("Usage: first command line arg is source file")
        filename = ""

    if filename:
        opts = IndexOpts(
            collection_name="chunk_collection",
            companion_collection="document_collection",
            questions=True,
            summaries=True,
            encoding_model=EncodingModel.MERGED,
            text_splitter=defaultSplitter,
        )
        markdown_upload(
            'memory',
            [filename],
            opts,
            save_files=True,
            ingest=False,
        )
