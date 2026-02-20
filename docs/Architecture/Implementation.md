# Implementation notes

LM markdown for education leverages the code from LM markdown to implement facilities to provide lectures online, using language models for the interaction with students. Here, we describe briefly the architecture of the parts that are specific to LM markdown for education.

## Configuration (config module)

This module defines a `ConfigSettings` object to read from `config.toml` the settings of the project. `ConfigSettings` inherit from the `Settings` class in LM markdown, so that the `config.toml` file includes both the settings that configure LM markdown and LM markdown for education. The `Settings` class inherits from pydantic's `BaseSettings`, which handles reading the settings from `config.toml` and validating them.

A convenient aspect of these objects is that they may be initialized in code for the parts that need override the settings in config.toml, with the others being initialized from the values there. When looking in a file like `config.toml`, one can see the keys that are available for initialization. Keys that are under headings can by initialized via a dictionary.

Example:
```
# content of config.toml
[RAG]
questions = false
summaries = false

[embeddings]
dense_model = "OpenAI/text-embedding-3-small"
```

These settings can be overriden in code as follows:

```python
config = ConfigSettings(
    RAG = {'questions': True},
    embeddings = {'dense_model': "OpenAI/text-embedding-3-large"}
)
```

The resulting `config` object will be initialized with `questions = true`, and all other values overridden as in code. (This coding will be flagged a type-invalid by the type checker, because Pydantic validates the content, also enforcing coertion, behind the hood).

## Error handling

The code distinguishes between three types of exceptions: genuine coding errors (for example, a `match` statement with a missing case), validation errors, and errors that arise from normal adversity (like invalid yaml headers from the users, missing internet connections, etc.).

The first type of errors are designed to crash the application while delivering an error trace that allows fixing them. Of note, the code is written using the most strict setting for type checking available, which means that most of this type of errors should be flagged by the type checker already. This kind of errors should be truly exceptional.

The second type of errors arise when data for operating the framework is validated. For example, if an invalid setting is entered in `config.toml`, the code will crash as soon as the setting is read, leaving an error trace in the console that is informative of the problem. Also functions that are meant to be entered in Python's REPL are validated by pydantic, crashing immediately on invalid inputs instead of creating some problem down the line. This behaviour is justified by the fact that these invalid inputs are not compatible with the program's functioning. Note that function calls that are not validated by Pydantic are supposed to be used in code, not in a REPL, and in code strict type checking is followed in the whole framework.

The third and final kind of "errors" are not cosidered exceptions, but are modelled in code by a `Logger` class (provided by the LM markdown framework). The term 'modelled' refers to the fact that this class is designed so that these situations may be handled like any other situation that code is supposed to handle, contrary to what the terms 'exceptions' or 'errors' imply. An object of the `Logger` class is passed to functions that follow this convention. Because different implementations of this class exist, the error handling may be customized according to the context in which the code is called, or the way in which these situation are modelled:

- providing a console trace. This is useful when calling code manually through the REPL, and is the default behaviour of the logger.
- collecting the error into a list of error messages. The list needs be checked manually. This is used in the code mostly in the tests.
- throwing a Python exception. This is useful when interfacing with code that behaves traditionally by rasing exceptions, as in the Langchain and in Qdrant APIs, and do not model the exception as it is done elsewehere in the LM markdow for education framework.
- logging a console message and throwing an exception. Similar use as above.

The functions that adopt this convention are easily recognizable by the fact that they take a `Logger` object as a last argument. Many functions do not do this; they are function that may be viewed as "pure": they are not expected to raise exceptions, unless a genuine coding error intervenes.

Functions that adopt the logger convention usually return a value that can be `None`. Because the type checker may be used to enforce checking for `None` return values, the programmer can consider this type of error handling explicit, as long as the type checker discipline is followed.

Some functions from the LM markdown module do not return `None`, but things such as an empty list where a list is expected. This behaviour is intentional: several errors result in messages displayed in the markdown for the user to fix; from the perspective of the code, the program should keep working. This behaviour is documented in LM markown.

## Qdrant vector store

LM markdown for education uses Qdrant to store the RAG data. The package is organized as follows:

| subpackage | module | purpose |
| --- | --- | --- |
| . | vector_store_qdrant | interface to qdrant code |
| lmm | chunks | main code to tranform text into chunks |
| langchain | vector_store_qdrant_langchain | langchain interface to the vector store (for retrieval) |

### vector_store_qdrant module

The Qdrant API uses A `QdrantClient` object as an argument to all functions that write and read to/from the database. The functions in this module also take this object as an argument; the object may be initialized directly through a call to the Qdrant API, or through the function `client_from_config`. This function takes a `ConfigSettings` object, reading from it the `storage` value that is relevant to initialize the database:

- `":memory:"`: initializes the database in memory
- `LocalStorage`: a class representing a local folder
- `RemoteSource`: a class representing a remote Qdrant server.

Note that if you obtain a client from the `client_from_config` function, you'll have to handle closing the connection to the database yourself. A better option is to use the function `global_client_from_config` from the vector_store_qdrant_context module. This function centralizes creation and destruction (including closing connections) of Qdrant objects. Only one connection is created and is closed automatically.

Within the database represented by the `QdrantClient` object, there are collections (the equivalent of tables in a traditional database) that are initialized through a call to `initialize_collection`. 

The modalities in which data are stored in the database are captured by two elements: the `QdrantEmbeddingModel` and `Chunk` objects. The `QdrantEmbeddingModel`, which is declared in the call to `initialize_collection`, specifies how data are embedded in the database. These are the embedding models that are supported by Qdrant, as supported by the framework:

| code | config value | description |
| --- | --- | --- |
DENSE | "dense" | text embedded with dense vector |
MULTIVECTOR | "multivector" | text and annotations embedded with dense multivector |
SPARSE | "sparse" | annotations embedding only, sparse |
HYBRID_DENSE | "hybrid_dense" | sparse annotations, dense text |
HYBRID_MULTIVECTOR | "hybrid_multivector" | sparse annotations, dense text and annotations |
UUID | "UUID"  | no embedding (chunks retrieved by UUID) |

Dense embedding uses a dense vector to capture text; dense encoding is also used by Multivector, but in this case more than input is given to the encdoing model, resulting in two distinct embedding vectors (one from text, one from annotations). Sparse only embeds annotations, and the other combine dense and sparse encoding. The UUID encoding only uses an ID.

These settings are used in conjunction with an annotation model to define the encoding strategy in the database. The annotation model defines what metadata properties are used in the embedding. The `encoding_to_qdrantembedding_model` function is used internally to translate an `EncodingModel` object into the embedding model understood by Qdrant.

Instead of providing a `QdrantEmbeddingModel`, one can  provide a `ConfigSettings` object from which the `QdrantEmbeddingModel` may be deduced. A `ConfigSettings` object also overrides the embedding model specified in `config.toml`, which are used when a `QdrantEmbeddingModel` is specified.

The `Chunk` objects are the elementary units that are stored in the database, i.e. the equivalent of records (Qdrant calls them "points"). In frameworks such as Langchain or Llamaindex, these objects are called "Document". Chunk objects emerge from chunking up large portions of text, however (the coduments proper), and also contain the metadata used for embedding them. The `chunk` module, detailed below, handles the conversion from markdown document to chunks prior to ingesting. The `vector_store_qdrant` module only handles ingesting and retrieving chunks:

| function | purpose |
| --- | --- |
| upload | upload chunks into database |
| query | retrieves chunks based on query text |
| query_grouped | retrieves text stored in a companion collection |

All these functions are also present in an asynchronous version, and take a logger object to model errors.

The `upload` function takes a list of chunks and ingests them into the database. Importantly, the embedding takes place here: chunk contain text and annotations that define the emebdding, and the `upload` function takes care of calling the language model and the sparse embedding library to create the embeddings (this work is handled internally by the `chunk_to_points` function). The specifications are loaded from `config.toml`.

Query functions return list of `ScoredPoint` object, which is what Qdrant returns. These data may be transformed into markdown blocks by the `points_to_blocks` function.

Here is a schematic example of the code that ingests and retrieves chunks/records into/from the database.

```python
# get a client as specified in config.toml
client = client_from_config()
if client is None
    ... #(error handling)

points: list[Point] = upload(
    client,      # the Qdrant client object 
    collection_name = "documents", # the collection to ingest into
    model = QdrantEmbeddingModel.DENSE, 
    chunks = chunks,      # the chunks we want to ingest
)

# Retrieval code with query text
points: list[ScoredPoints] = query(
    client, 
    collection_name = "documents",
    model = QdrantEmbeddingModel.DENSE,
    querytext = "What are the main uses of logistic regression?",
    limit = 6,        # max 6 ScoredPoints
    payload = True,   # all payload fields
)

# Retrieve text
for pt in points:
    print(f"{pf.score} - {pt.payload['page_content']}\n")

```

The `ScoredPoint` object is the Qdrant object in which the data from the database are returned. The 'page_content' field contains the text that was chunked, while other fields contain other parts of the metadata that were ingested in the database (based on the annotation model).

### vector_store_qdrant_langchain module

Intead of using the functions from the vector_store_qdrant module to retrieve data from the vector database, it is possible to initialize a Langchain retriever. Instead of creating a QdrantClient object from a config settings, the Langchain retriever is initialized by a call to `from_config_settings':

```python
retriever = QdrantVectorStoreRetriever.from_config_settings()
results: list[Document] = retriever.invoke(
    "What are the main uses of logistic regression?"
)
```

Here, `from_config_settings` retrieves the settings from `config.toml`, but a partially initialized `ConfigSettings` object may be passed as an argument to override these settings. `AsynQdrantVectorStoreRetriever` provides asynchronous calls. In alternative, the retriever object may be instantiated explicitly:

```python
client = QdrantClient("./storage")
retriever = QdrantVectorStoreRetriever(
    client, 
    collection_name = "documents',
    embedding_model = QdrantEmbeddingModel.DENSE,
)
```
Note that the embedding_model must match that used to embed the chunks originally (this is an argument in favour of creating a `config_settings` object and using only this object to interface to all Qdrant functions).

The query returns here a list of `Document` object, the record representation in Langchain. These objects contain two fields: `page_content`, the text, and `metadata`, a dictionary with key/value pairs.

## chunks module

This is a key module in the vector storage implementation, as it handles the conversion from a parsed markdown file (in the form of a list of blocks) into a list of chunks.

The chunks module defines an _encoding model_ to map properties of the markdown to the embeddings used to make parts of the markdown retrievable. The encoding model includes, on the one hand, the text that is stored in the database, and on the other hand metadata propoerties stored in the markdown that may be used in the embedding. Such metadata properties are referred to as _annotations_, to distinguish them from other metadata that are not suitable for embedding (for example, properties used for housekeeping purposes).

This is the specification of the encoding model.

| model | description |
| --- | --- |
| NONE | no encoding (chunks identified by their UUID) |
| CONTENT | encode only textual content in dense vector |
| MERGED | encode textual content merged with metadata |
| MULTIVECTOR | encode content and annotations using multivectors |
| SPARSE | sparse encoding of annotations only |
| SPARSE_CONTENT | sparse encoding of content only |
| SPARSE_MERGED | sparse encoding of merged content and annotations |
| SPARSE_MULTIVECTOR | sparse encoding of merged content and annotations using multivectors |

This is the specification of the annotation model, i.e. what counts as an annotation.

| model | description |
| --- | --- |
| inherited properties | properties own and inherited by parent nodes |
| own properties | properties owned in the node metadata |
| filters | (reserved for future use) |

The central function in the module is `blocks_to_chunks`, which takes a list of parsed metadata blocks and returns a list of chunks (which are given to `upload` in the `vector_store_qdrant` module).

```python
def blocks_to_chunks(
    blocklist: list[Block],
    encoding_model: EncodingModel,
    annotation_model: AnnotationModel | list[str] = AnnotationModel(),
    logger: LoggerBase = logger,
) -> list[Chunk]:
```

Internally, this function does the following. It uses `scan_rag` to generate text id's and UUIDs for the text blocks, if they are missing (`scan_rag` is idempotent). It then produces the tree representation of the blocks, and collected the metadata properties (inheriting them if appropriate) as specified by the annotation model. It also collects other metadata for possible storage in the payload of the database, even if these metadata properties are not used in the embedding. 

Note that `scan_rag` is not used here to produce annotations with a language model. This step should take place previously and while using an explicit call.

The explicit call may be done so:

```python
from lmm.scan.scan_rag import ScanOpts, markdown_rag

markdown_rag("MyMarkdown.md", ScanOpts(questions=True, titles=True))
```

This saves MyMarkdown.md with the annotations.


### Implementation encoding

The following tyble summarizes the flow of directives and their effects from encodings to embeddings specifications.

| encoding | effect | qdrant emb. | effect |
| --- | --- | --- | --- |
| NONE | (no effect) | UUID | uuid -> id (also in all others) |
| CONTENT | content -> dense_encoding | DENSE | dense_encoding -> vector |
| MULTIVECTOR | content -> dense_encoding | MULTIVECTOR | annotations, dense_encoding -> [vectors] |
| MERGED | annotations+content -> dense_encoding | DENSE | dense_encoding -> vector |
| SPARSE | annotations -> sparse_encoding | SPARSE | sparse_encoding -> sparse vector |
| SPARSE_CONTENT | annotations -> sparse_encoding | HYBRID_DENSE | sparse_encoding -> sparse vector |
|   | content -> dense_encoding |   | dense_encoding -> vector |
| SPARSE_MULTIVECTOR | annotations -> sparse_encoding | HYBRID_MULTIVECTOR |  sparse_encoding -> sparse vector |
|   | content -> dense_encoding |   | annotations, dense_encoding -> [vectors] |
| SPARSE_MERGED | annotations -> sparse_encoding | HYBRID_DENSE | sparse_encoding -> sparse vector |
|   | annotations+content -> dense_encoding |   | dense_encoding -> vector |

Effect of encoding: in blocks_to_chunks, chunks module; qdrant emb: in encoding_to_qdrant_embedding, vector_store_qdrant module; effect of qdrant embedding: chunks_to_points, vector_store_qdrant module.

## Identification of portions of markdown files

Markdown files can be repeatedly ingested into a vector database without giving rise to duplicates. This relies on the value of a `docid` property in the header, which is set automatically by `scan_rag` at each call, unless this property is already present. It can be set manually to an intelligible string, as long as it is unique across markdown documents.

The steps to identify parts of the text are the following.

1. A markdown document is created and edited. Optionally, a `docid` property is added manually.
2. Metadata can be created if desired and edited.
3. Missing metadata are created automatically with the help of a language model. Since metadata are not recomputed if the text did not change, there can be several cycles of steps 2 and 3.
4. The markdown document is split by `scan_split`.
5. After splitting, the document is sent to `blocks_to_chunks` in the chunks module. Internally, this function calls `scan_rag`, which will create a `docid` if is is missing, and a sequential `textid` property for each chunk (i.e., shorter text block). The value of this property is {docid}.{sequential number}. A UUID field will also be created based on the textid value, which will identify the chunk. As long as the document grows, it can be repeatedly ingested without creating duplications.
6. The chunks are sent to `upload` in the vector_store_qdrant module. Internally, this function calls the `chunks_to_points` function which creates embedding for each chunk, based on its text and annotations, as directed by the encoding model. 

All these steps are handled by `markdown_upload` (see next section).

## Ingestion (ingest module)

This module contains the function `markdown_upload`, which takes a list of file names or the markdown documents, and a optional `ConfigSettings` object. This object specifies how the files are processed and ingested in the vector database, based on the settings specified in `config.toml`, or by settings that override these when an object is specified explicitly.

Each markdown file is processed through a series of steps.

First, it is read in and parsed by `markdown_scan`, a function from LM markdown scan module. This function checks that the markdown is well-formed and there are no problems. 

Second, it is processed by `blocklist_encode`, another function in the module. This function implements the specific RAG strategy used in the package and specified in the configuration options in `config.toml`. Internally, it calls `scan_rag` to use the language model to create annotations and summaries, if required, and `scan_split` to split the text into chunks. The chunks are created with the `blocks_to_chunks` function from the chunks module.

Finally, the chunks are passed to `blocklist_upload`, another function in the module that is tasked with implementing the configuration options when calling the vector store qdrant functions and save the data in the vector database.

(There is no equivalent functions for retrieval. One can just instantiate the Langchain retriever pointing at the database and call the `invoke` function with the query text.)
