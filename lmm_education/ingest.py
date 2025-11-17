"""
This module provides the facilities to ingest markdown files into a
vector database for the LM markdown for education project.

Prior to ingesting, the files are processed according to specifications
read from the config.toml file. They include what annotations
to create (questions, summaries) and how to use them in the encoding of
the data. Because these specifications determine the schema of the
database, they should not be changed after the database is created!

Please see config/config.py for more information on the configuration
options. They contain the name of the file or source where the
database is located, the name of the collection or collections used to
store the data, and the information to extract from text prior to
storage.

The LMM for education package allows two different models of creating
storage for a RAG application. In the standard approach, the text is
chunked, possibly with overlapping segments, combined with additional
text (such as questions answered by the text) and ingested. In the
hierarchical graph RAG approach, the chunks provide the embeddings for
retrieving larger parts of text, determined by text blocks under
headings.

The two approaches intermingle somewhat as the hierarchical structure
is exploited to extract information from the text in both cases.

The module may be called directly; the main function reads the
configuration options and reads in all markdown filed in the input
folder and ingests them into the database.

Main functions:
    markdown_upload: ingest markdown files
    __main__: uploads and ingests markdown files using default
        configuration settings from a default input folder.
"""

# pyright: reportUnusedImport=false
# flake8: noqa

import io
from pathlib import Path

from pydantic import validate_call

# LM markdown
from lmm.markdown.parse_markdown import (
    Block,
    ErrorBlock,
    HeaderBlock,
    HeadingBlock,
    MetadataBlock,
    TextBlock,
    blocklist_haserrors,
    blocklist_errors,
    blocklist_copy,
    save_blocks,
)
from lmm.markdown.blockutils import (
    merge_code_blocks,
    merge_equation_blocks,
    merge_textblocks,
    clear_metadata_properties,
)
from lmm.markdown.treeutils import (
    inherit_metadata,
)
from lmm.markdown.tree import (
    blocks_to_tree,
    tree_to_blocks,
    extract_content,
    HeadingNode,
    MarkdownTree,
    MarkdownNode,
    TextNode,
    post_order_traversal,
)
from lmm.markdown.treeutils import (
    propagate_property,
    bequeath_properties,
)
from lmm.markdown.ioutils import save_markdown, report_error_blocks
from lmm.utils.ioutils import append_postfix_to_filename
from lmm.scan.scan import (
    markdown_scan,
)
from lmm.scan.scan_rag import (
    ScanOpts,
    blocklist_rag,
)
from lmm.scan.scan_keys import (
    UUID_KEY,
    GROUP_UUID_KEY,
    SUMMARY_KEY,
    TXTHASH_KEY,
    CHAT_KEY,
)
from lmm.scan.scan_split import (
    NullTextSplitter,
    defaultSplitter,
    scan_split,
)
from lmm.scan.chunks import (
    Chunk,
    blocks_to_chunks,
    chunks_to_blocks,
    AnnotationModel,
)
from qdrant_client.http.models.models import PointStruct

from lmm_education.config.config import DatabaseSettings

# LMM for education
from .config.config import (
    create_default_config_file,
    load_settings,
    ConfigSettings,
    LocalStorage,
    RemoteSource,
    EncodingModel,
)
from .stores.vector_store_qdrant import (
    QdrantEmbeddingModel,
    encoding_to_qdrantembedding_model as encoding_to_embedding_model,
    client_from_config,
    initialize_collection_from_config,
    initialize_collection,
    upload,
    chunks_to_points,
)

# langchain, import text splitters from here
from langchain_text_splitters import (
    TextSplitter,
    RecursiveCharacterTextSplitter,
)


# qdrant, vector database implementation
from qdrant_client import QdrantClient

# We import from utils a logger to the console
from lmm.utils import logger
from lmm.utils.logging import LoggerBase


# The configurations settings are contained in a config file.
from .config.config import DEFAULT_CONFIG_FILE

# Create a default config.toml file, if there is none. This
# file becomes the default location from where the configuration
# settings are loaded.
if not Path(DEFAULT_CONFIG_FILE).exists():
    create_default_config_file(DEFAULT_CONFIG_FILE)


# The interaction with the vector database takes place through
# functions in the vector_store_qdrant module. This module contains
# an `initialize_collection` function that may be used to create
# a collection with the required properties in the database. The
# initialize_client function reads the required configuration from
# config.toml and calls this function to set up collections.
# Instead of providing calls such as create_database, open_connection
# etc. we rely on initialize_client to create the database if it does
# not exist, initializing the collections according to the
# configuration settings. When writing or reading to the database, the
# settings object will make sure that the data are consistent with the
# schema of the database. This function returns a QdrantClient object
# that may be called directly in further code, if required.
@validate_call(config={'arbitrary_types_allowed': True})
def initialize_client(
    opts: ConfigSettings | None = None,
    logger: LoggerBase = logger,
) -> QdrantClient | None:
    """
    Initialize QDRANT database. Will open or create the specified
    database and open or create the collection specified in opts,
    a settings object that reads the specifications in the
    configuration file.

    Args:
        opts: a ConfigurationSettings object. If None, an object will
            be created from config.toml.

    Returns:
        a Qdrant client object
    """

    # Load and checks configuration settings.
    if opts is None:
        opts = load_settings(logger=logger)
        if opts is None:
            return None

    dbOpts = opts.database
    collection_name: str = dbOpts.collection_name
    if not collection_name:
        return None

    # obtain a QdrantClient object using the config file settings
    client: QdrantClient | None = client_from_config(
        opts=opts, logger=logger
    )
    if client is None:
        return None

    flag: bool = initialize_collection_from_config(
        client,
        collection_name,
        opts,
        logger=logger,
    )
    if not flag:
        return None

    if bool(dbOpts.companion_collection):
        flag = initialize_collection(
            client,
            dbOpts.companion_collection,
            QdrantEmbeddingModel.UUID,
            opts.embeddings,
            logger=logger,
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
    sources: list[str] | list[Path] | str,
    *,
    config_opts: ConfigSettings | None = None,
    save_files: bool | io.TextIOBase = False,
    ingest: bool = True,
    client: QdrantClient | None = None,
    logger: LoggerBase = logger,
) -> list[tuple[str] | tuple[str, str]]:
    """
    Upload a list of markdown files in the vector database.

    Args:
        sources: a list of file names containing the markdown files
        config_opts: a ConfigSettings object declaring the schema;
            if None, will be read from config.toml
        ingest: Whether to ingest documents into the vector database
            (default: True)
        save_files: Whether to save processed blocks to files
            (default: False)
        client: if None (default) a QdrantClient will be initialized
            from the config_opts spec. Note that for the :memory:
            client, the client must be created once and passed around,
            otherwise any initialization of the client will create
            a new database instance.
        logger: a logger object

    Returns:
        A list of Document objects representing the processed content
    """

    # Check the config file is ok. The ConfigSettings constructor
    # will read the config.toml file if no arguments are provided
    if config_opts is None:
        config_opts = load_settings(logger=logger)
        if config_opts is None:
            return []

    # initialize qdrant client. The config_opts object will be used
    # to validate the database schema, i.e that it has a schema that
    # corresponds to the encoding model applied to the documents. If
    # the database does not exist, it will be created.
    local_client_flag: bool = False
    if client is None:
        client = initialize_client(config_opts, logger)
        local_client_flag = True
    if not client:
        logger.warning("Database could not be initialized.")
        return []

    # The markdown parser responds to errors by creating a special
    # type of block, the ErrorBlock. This blocks can be serialized
    # back to document, giving the user a chance to address the error,
    # but here we just stop the process if there are parse errors in
    # any of the documents. Hence, we do a priminary scan of all
    # documents to check for errors.
    if not bool(sources):
        logger.warning("No documents for ingestion in the database.")
        if local_client_flag:
            client.close()
        return []
    if isinstance(sources, str):
        sources = [sources]
    error_sources: dict[str, list[ErrorBlock]] = {}
    logger_level: int = logger.get_level()
    for source in sources:
        blocks: list[Block] = markdown_scan(
            source, False, logger=logger
        )
        if not bool(blocks):
            error_sources[str(source)] = [
                ErrorBlock(
                    content=f"Empty or nonexistent file: {source}"
                )
            ]
        elif blocklist_haserrors(blocks):
            error_sources[str(source)] = blocklist_errors(blocks)
    if error_sources:
        logger.error(
            "Problems in markdowns, fix before continuing:\n\t"
            + "\n\t".join(error_sources.keys())
        )
        if local_client_flag:
            client.close()
        return []
    logger.set_level(logger_level)

    # the key loop. Each file is encoded and uploaded. We go through
    # the files again to avoid loading all files into memory at
    # once.
    ids: list[tuple[str] | tuple[str, str]] = []
    for source in sources:
        # Markdown documents are loaded from files with markdown_scan
        # beacause this function can initialize default properties
        # such as 'title' from the file name.
        blocks = markdown_scan(source, False, logger=logger)
        # Process the markdown documents prior to ingesting.
        chunks, comp_chunks = blocklist_encode(
            blocks, config_opts, logger
        )
        if not bool(chunks):
            logger.warning(f"{source} could not be encoded.")
            continue
        # Ingestion.
        idss = blocklist_upload(
            client,
            chunks,
            comp_chunks,
            config_opts,
            ingest,
            logger,
        )
        if bool(idss):
            if ingest:
                storage_location: str
                match config_opts.storage:
                    case LocalStorage():
                        storage_location = config_opts.storage.folder
                    case RemoteSource():
                        storage_location = str(
                            config_opts.storage.url
                        )
                    case _:
                        storage_location = "memory database"
                logger.info(f"{source} added to {storage_location}.")
            ids.extend(idss)
        else:
            logger.warning(f"{source} could not be ingested.")
            continue

        # this allows users to inspect the annotations that were
        # used to encode the document
        if bool(save_files) and bool(idss):
            chunk_blocks = chunks_to_blocks(chunks, '+++++')
            comp_blocks = chunks_to_blocks(comp_chunks, ".....")
            if isinstance(save_files, bool):
                out_file: str = append_postfix_to_filename(
                    str(source), "_chunks"
                )
                save_markdown(out_file, chunk_blocks, logger)
                if bool(comp_blocks):
                    out_file = append_postfix_to_filename(
                        str(source), "_documents"
                    )
                    save_markdown(out_file, comp_blocks, logger)
            else:  # it's a stream
                from lmm.markdown.parse_markdown import TextBlock

                if bool(comp_blocks):
                    chunk_blocks.append(
                        TextBlock(
                            content="-----------------------------------------"
                        )
                    )
                save_markdown(
                    save_files, chunk_blocks + comp_blocks, logger
                )

    if local_client_flag:
        client.close()

    return ids


# This is the function that does the actual processing of the
# markdown blocks. It implements the annotation strategy specified in
# the ConfigSettings object: adds annotations using a language model
# if the settings require it. It is the key point where a strategy
# to extract further information from mere text is executed. Returns
# the chunks for ingestion in the database.
def blocklist_encode(
    blocklist: list[Block],
    opts: ConfigSettings,
    logger: LoggerBase = logger,
) -> tuple[list[Chunk], list[Chunk]]:
    """
    Encode a list of markdown blocks. Preprocessing will be
    executed at this stage as specified in the ConfigSettings opts
    object.

    Args:
        blocklist: the list of markdown blocks
        opts: the ConfigSettings object
        logger: a logger object

    Returns:
        A tuple of two lists of chunks. The first list contains the
        chunks to be ingested into the main collection, the second
        list contains the chunks to be ingested into the companion
        collection. If no companion collection is specified in opts,
        the second list will be empty.
    """

    if not blocklist:
        return [], []
    if blocklist_haserrors(blocklist):
        report_error_blocks(blocklist, logger)
        logger.error("Problems in markdown, fix before continuing")
        return [], []

    # we get rid of all existing UUIDs preliminarly
    blocklist = clear_metadata_properties(blocklist, [UUID_KEY])

    # this contains the info if form a companion collection
    dbOpts: DatabaseSettings = opts.database

    # preprocessing for RAG. You tell scan_rag what annotations
    # to make.
    # * titles and questions are used in annotations
    # * summaries are additional text blocks encoded on their own (see
    #       raptor paper)
    # * headingUUID are required to form groupUUID's
    scan_opts = ScanOpts(
        titles=bool(opts.RAG.titles),
        questions=bool(opts.RAG.questions),
        summaries=bool(opts.RAG.summaries),
        # if computing a companion collection (group queries)
        # we need the id's to link records between collections
        headingid=bool(dbOpts.companion_collection),
        headingUUID=bool(dbOpts.companion_collection),
        language_model_settings=opts,
    )
    blocks: list[Block] = blocklist_rag(blocklist, scan_opts, logger)
    if not blocks:
        return [], []

    # ingest here the whole text under headings in the companion coll.
    if dbOpts.companion_collection:
        coll_blocks: list[Block] = blocklist_copy(blocks)
        # in the companion collection, we merge textblocks
        # under headings together prior to saving them in the
        # documents collection with the UUID of the headings
        coll_blocks = merge_textblocks(coll_blocks)
        coll_root: MarkdownNode | None = blocks_to_tree(coll_blocks)
        if coll_root is None:
            return [], []

        # this will also inherit the UUID
        coll_root = inherit_metadata(
            coll_root,
            exclude=[TXTHASH_KEY, CHAT_KEY, SUMMARY_KEY],
            inherit=True,
            include_header=True,
        )
        coll_blocks = tree_to_blocks(coll_root)

        # collect text and annotations into chunk objects
        coll_chunks: list[Chunk] = blocks_to_chunks(
            coll_blocks,
            encoding_model=EncodingModel.NONE,
            annotation_model=AnnotationModel(),  # no annotations here
            logger=logger,
        )
        if not coll_chunks:
            return [], []

    else:
        coll_chunks = []

    # chunking
    splitter: TextSplitter = defaultSplitter
    match opts.textSplitter.splitter:
        case 'none':
            splitter = NullTextSplitter()
        case 'default':
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=200,
                add_start_index=False,
            )
        case _:
            raise ValueError(
                f"Invalid splitter: {opts.textSplitter.splitter}"
            )

    splits: list[Block] = scan_split(blocks, splitter)

    # now copy the UUID in the heading into the metadata of the
    # splits under the name GROUP_UUID_KEY
    root: MarkdownTree = blocks_to_tree(splits)
    if not root:
        return [], []

    root = bequeath_properties(root, [UUID_KEY], [GROUP_UUID_KEY])
    splits = tree_to_blocks(root)

    # we avoid chunks that isolate or split equations or code to keep
    # those together with textual context to improve the embedding
    # TODO: handle text overlap
    # splits = merge_code_blocks(splits)
    # splits = merge_equation_blocks(splits)

    # summaries. To add summaries as additional chunks, we transform
    # the summary property into a child text block of the heading node
    def _propagate_summaries(blocks: list[Block]) -> list[Block]:
        root: MarkdownTree = blocks_to_tree(
            blocklist_copy(blocks), logger
        )
        if not root:
            return []

        # propagate_property acts on heading nodes moving the content
        # of the metadata from the metadata to a child text node. If
        # add_type_property is set to True, the child text node will
        # have a metadata property marking its content type as summary
        add_type_property: bool = True
        root = propagate_property(
            root, SUMMARY_KEY, add_key_info=add_type_property
        )
        return tree_to_blocks(root)

    if bool(opts.RAG.summaries):
        splits = _propagate_summaries(splits)

    # the blocks_to_chunks function transforms the chunks into the
    # format recognized by qdrant for ingestion, also collecting the
    # annotations specified by the encoding model.
    chunks: list[Chunk] = blocks_to_chunks(
        splits,
        encoding_model=opts.RAG.encoding_model,
        annotation_model=opts.get_annotation_model(),
        logger=logger,
    )
    logger.info(f"Created {len(chunks)} chunks for ingestion.")
    if not chunks:
        return [], []

    return chunks, coll_chunks


# This is the internal function doing the job of uploading the
# chunks of text into the database. It is the point where
# annotations and text are used to compute an embedding, dense
# or sparse, or both, prior to sending the chunks to the
# database.
def blocklist_upload(
    client: QdrantClient,
    chunks: list[Chunk],
    companion_chunks: list[Chunk],
    opts: ConfigSettings,
    ingest: bool = True,
    logger: LoggerBase = logger,
) -> list[tuple[str] | tuple[str, str]]:
    """
    Upload a list of preprocessed chunks (including document chunks)
    to the database. Uses the encoding strategy specified in the
    ConfigSettings object opts.

    Args:
        client: the QdrantClient
        chunks: list of chunks to be uploaded
        companion_chunks: list of chunks for the companion collection
        opts: the ConfigSettings object
        ingest: Whether to ingest documents into the vector database
            (default: True)
        logger: a logger object

    Returns:
        A list of tuples containing the id's of the ingested objects,
        If document_collection is True, the second element of the
        tuples contains the id's of pooled text documents. The first
        element always contains the id of the chunk.
    """

    # ingestion of the chunks
    dbOpts: DatabaseSettings = opts.database
    model: QdrantEmbeddingModel = encoding_to_embedding_model(
        opts.RAG.encoding_model
    )

    if ingest:
        points: list[PointStruct] = upload(
            client,
            collection_name=dbOpts.collection_name,
            qdrant_model=model,
            embedding_settings=opts.embeddings,
            chunks=chunks,
            logger=logger,
        )
    else:
        points = chunks_to_points(chunks, model, opts.embeddings)
    if not bool(points):
        logger.error("Could not upload documents")
        return []
    else:
        if ingest:
            logger.info(
                f"Ingested {len(points)} chunks in main collection."
            )

    # ingestion of the document collection (if specified)
    if bool(dbOpts.companion_collection):
        if not (bool(companion_chunks)):
            logger.error(
                "Companion collection specified, but no "
                + "document chunks generated."
            )
            return []
        doc_coll: str = dbOpts.companion_collection
        if ingest:
            logger.info(
                f"Ingesting companion collection ({len(companion_chunks)} chunks.)"
            )
            docpoints: list[PointStruct] = upload(
                client,
                collection_name=doc_coll,
                qdrant_model=QdrantEmbeddingModel.UUID,
                embedding_settings=opts.embeddings,
                chunks=companion_chunks,
                logger=logger,
            )
        else:
            docpoints = chunks_to_points(
                chunks, QdrantEmbeddingModel.UUID, opts.embeddings
            )
        if not bool(docpoints):
            logger.error("Could not upload documents to " + doc_coll)
            return []

        return [
            (
                str(p.id),
                (
                    p.payload.get(GROUP_UUID_KEY, "")
                    if p.payload
                    else ""
                ),
            )
            for p in points
        ]

    else:
        return [(str(p.id),) for p in points]


def _list_files_by_extension(
    path: str, extensions: list[str]
) -> list[str]:
    """
    Determines if a given path is a file or a folder.

    If it's a file, it returns a list with that file as member.
    If it's a folder, it returns a list of files within that folder
    that match a list of given extensions.

    Args:
        path (str): The path to a file or folder.
        extensions (list[str]): A list of file extensions to
            filter by (e.g., ['.txt', '.log']).

    Returns:
        list: A list of matching filenames if the path is a folder,
              or a single-item list with the file if the path is a
              file. Returns an empty list if the path does not exist
              or there are no files.
    """
    import os

    if not os.path.exists(path):
        raise ValueError(f"Error: The path '{path}' does not exist.")

    if os.path.isfile(path):
        return [path]

    if os.path.isdir(path):
        matching_files: list[str] = []
        for file_name in os.listdir(path):
            (
                file_name_without_extension,  # type: ignore
                file_extension,
            ) = os.path.splitext(file_name)

            if file_extension.lower() in [
                ext.lower() for ext in extensions
            ]:
                full_path: str = os.path.join(path, file_name)
                matching_files.append(full_path)
        return matching_files
    return []


if __name__ == "__main__":
    """CLI interface to ingest files"""
    import sys
    from lmm.utils.logging import ConsoleLogger
    from requests import ConnectionError

    if len(sys.argv) > 1:
        try:
            filenames: list[str] = _list_files_by_extension(
                sys.argv[1], ["md", "Rmd"]
            )
        except Exception as e:
            print(e)
            exit()
    else:
        print("Usage: first command line arg is source file")
        print(
            "       second command line argument is ingest? (default True)"
        )
        print(
            "       third command line argument is save files? (default False)"
        )
        exit()

    ingest: bool = (
        bool(eval(sys.argv[2])) if len(sys.argv) > 2 else True
    )
    save_files: bool = (
        bool(eval(sys.argv[3])) if len(sys.argv) > 3 else False
    )

    # error logged to console, and exit if required
    logger = ConsoleLogger()
    if filenames:
        opts: ConfigSettings | None = load_settings(logger=logger)
        if opts is None:
            exit()

        # this will go through the encoding and chunking, and
        # save the chunked documents for inspection in the input folder
        try:
            markdown_upload(
                filenames,
                config_opts=opts,
                save_files=save_files,
                ingest=ingest,
                logger=logger,
            )
        except ConnectionError as e:
            print("Cannot form embeddings due a connection error")
            print(e)
            print("Check the internet connection.")
        except Exception as e:
            print(f"ERROR: {e}")
