# Encoding and embeddings

The basis of retrieval-augmented generation (RAG) is the selection of textual material from a database. This material is given to the large language model to provide an answer to the user's query. The retrieval of correct material is therefore crucial for the performance of the RAG system. The material of a RAG system is stored in a database which is accessed during the generation of responses.

Vector database are special because they allow retrieving data based on a search query that contains natural text, such as a question (_semantic search_). This differs from standard relational databases, where the search is based on matching the content of a field or searching part of text for keywords. The advantage of semantic search, relative to keyword search (such as the one used to search the web) is the capacity to disambiguate a word using its context. For example, the word 'black' has very different meanings in "dressed in black" and "black friday". Large language model provide representations of the meaning of whole sentences (_embeddings_), which contain the context required for this disambiguation. But another big advantage of semantic search is that the query is given by the question directly: there is no need to transform it into a set of keywords or into the specialized format of relational database queries. An embedding is created for the question itself using the same procedure for creating embeddings of text. The text in the database is then retrieved with the embeddings that are most similar to the embedding of the question.

Initial typical implementations of RAG made extensive use of semantic search and vector databases. The text is chunked into pieces, and these pieces are inserted into the database together with their embeddings. There are two issues with using semantic search directly to feed the language model, however. First, while the chunks may be retrieved correctly, they may lack the overall context. Chunk-based retrieval has a tendency to provide disorganized fragments of text. The second is simply failure to retrieve relevant fragments.

There are two ways to address these shortcomings.

1. Instead of retrieving the chunks, the system may retrieve the surrounding text as well, or the paragraphs, or the chapters of text, where the chunks are located. LM Markdown for Education adopts the approach of optionally retrieving whole chapters, which in markdown are defined by headings, instead of chunks.
2. Instead of creating embeddings of the chunk content, the text is preliminary processed to extract information that is relevant for retrieval: what we call here an _encoding_. This information may then be used for embeddings or in a traditional keyword-based search. We refer here to the specification of a strategy to extract or define this information as to the _encoding model_. More generally, the encoding model may refer to the content of the chunks or of any property extracted from text, or any mixture of those. The original text is the _content_ of the chunk, while other properties are referred to as _metadata_ or _annotations_.

There are two types of embeddings: dense and sparse. Dense embeddings are provided by language models or other models trained from large corpora of text in the same way. Sparse embeddings are more similar to keywords, but are capable to match semantically related words even if the words are not identical. Depending on the kind of encoding, either or both may be used to represent content in the database. Here, we refer to the combination chosen to represent an encoding as the _embedding model_: dense, sparse, or a combination of both.

While the encoding model specifies how and what parts of the text are used to represent meaning, the embedding model specifies if the representation consists of dense or of sparse embeddings. In general, dense embeddings are used for the chunks of text that represent the text stored in the RAG database or from complex textual annotations (for example, summaries). In constrast, sparse embeddings, and possibly dense embeddings, are usually formed from keyword-like annotations. Both types of annotations enrich the text so as to increase the efficiency of its retrieval. One common strategy is to couple text with the questions that the text answers, or other metadata such as keywords, the title of the heading, etc. These data can then be combined with the text of the dense embeddings, or be kept separate for an additional embedding, usually of the sparse variety.

Most often, annotations are produced by a language model prior to ingestion, but the framework allows one to add or review them manually.

## Preparing text for ingestion

The encoding model determines how the text is prepared for ingestion:

annotation model          -->   encoding model  -->   embedding model
(and its implementation)

Note that the implementation of the annotation model is not a monolithic step, as it depends on whether the annotations should be created by code, a language model, or manually. When the annotations are created by code or a language model, they are automatically added to the annotation model. However, it is also possible to annotate parts of text manually and add the relevant information to the annotation model. In short, annotations may be those that are added automatically when code or language models are directed to process text, or those that are added by explictly listing them in the annotation model.

Annotations are written in the markdown text in metadata blocks, i.e. special markdown blocks that refer to the text block that follows. When they precede a heading, they refer to all text blocks under that heading, unless overridden by metadata preceding sub-headings. Therefore, the process to which annotiation contribute to the semantic encoding of text is transparent. Properties in metadata are not automatically considered annotations; they must be added to the annotation model.

## Language model directives or code that create annotations

Markdown files may be reviewed by a large language model, and part of this review is the creation of annotations. When annotations are produced in this way, they are automatically added to the annotation model.

The following directives create annotations: `questions = True`, `titles = True`. The directive `summaries = True` creates additional text, not annotations. These directives are added to the [RAG] section of config.toml:

```
[RAG]
questions = True
titles = True
```

This has the following effect on the markdown:

```markdown
---
~txthash: S/ec1NIX0xV3NhFFuHY6bQ
titles: Chapter 1 - What are linear models?
questions: What are the two main ways linear models are used in practice? - How do linear models relate predictors to an outcome variable? - Why is understanding the output of linear models important for their correct application?  - How do generalizations of linear models enable capturing more complex associations? - In what way does the purpose of a linear model (prediction vs. inference) affect the consideration of confounding factors?
---
## What are linear models?

Linear models and their generalizations constitute the majority of the statistical models used in practice. Here, we will look at linear models from a practical perspective, emphasizing the issues in correctly applying them and in understanding their output.

(further text omitted) ....

```

As the example shows, setting the `questions` and the `titles` in the configuration file has the effect of creating headiers before all headings in a markdown files with two metadata properties containing the annotations (the example only shows one of such headings). The metadata block may contain any other property, but some property keys (such as `questions`, `titles` and `summary`) are reserved for LM Markdown for Education. Note also the `~txthash` property. This is an automatically generated hash that checks that the text has not changed since the last time annotations were computed. Annotations are not recomputed if the text to which they refer was not changed.

When an encoding model is specified in the `[RAG]` section of the configuration file, the annotations are automatically generated with a language model prior to ingesting the text in the vector database. If a metadata property containing an annotation is removed, or the whole metdata block is deleted, it will be recreated prior to ingesting.

## Manual annotations

Manual annotations may be added, if desired, directly in the markdown file by the human RAG author. An annotation is simply a metadata property. To tell the framework that this property is an annotation, you can specify it in config.toml.

For example, here you introduce a manual annotation called 'topic':

```markdown
---
topic: 'review'
---

We provide here a reviwe of logistic regression... (further text omitted)

```

Then, in config.toml:

```
[RAG.annotation_model]
own_properties = ["topic"]
```

Manual annotations can be "own" or "inherited". Inherited annotations are taken from ancestor nodes, i.e. headings higher in the hierarchy.

All annotations that are included in the annotation model are concatenated together prior to be embedded.

## Reviewing annotations

LM Markdown for Education is designed to allow the RAG author to review all annotations manually prior to ingesting the documents. In the command window, start the LM markdown terminal (this may take some time and require an internet connection, depending on the embedding library in use):

```bash
lmme terminal
```

This command prints some information on the emebdding configuration of the system and prepares a prompt waiting for commands. These commands work on a file (generally, the file will be open in a markdown editor to view the effects of commands).

To direct the system to create a specific annotation, use the `scan_rag` command with the annotation that you need to produce. For example, to produce questions for the my_markdown.md document, type

```bash
> scan_rag --questions my_markdown.md
```

and look in the editor for its effects (type `scan_rag --help` to get an overview of all subcommands). You can edit the question generated by the language model by typing directly in the markdown editor.

While the `ingest` command is used to send the file to the vector database, it may also be used to create the annotations specified by the annotation model without ingesting the document. To this end, use it as follows:

```bash
> ingest --save_files=true --skip_ingest=true my_markdown.md
```

This will prepare the markdown for ingestion without ingesting it, and save it instead to disk for review and editing.

As mentioned, the annotations may be edited manually prior to ingestion. At the time of ingestion, annotations will not be recomputed if the text to which they refer does not change. However, they will after any change to the text to which they refer (that is, under the heading that the metadata block precedes). To prevent annotations from being recomputed when the text changes, add the property `frozen` to the metadata:

```markdown
---
~txthash: S/ec1NIX0xV3NhFFuHY6bQ
titles: Chapter 1 - What are linear model
questions: What are the two main ways linear models are used in practice? - How do linear models relate predictors to an outcome variable? - Why is understanding the output of linear models important for their correct application?  - How do generalizations of linear models enable capturing more complex associations? - In what way does the purpose of a linear model (prediction vs. inference) affect the consideration of confounding factors?
frozen = True
---

Changed text...

```

## Encoding model

The encoding model may be specified in the RAG section of config.toml:

```
[RAG]
encoding_model = "content"
```

The possible values of the encoding model are the following.

| encoding model | description |
| --- | --- |
| none | no encoding (chunks identified by their UUID) |
| content | encode only textual content in dense vector |
| merged | encode textual content merged with metadata in the dense vector|
| multivector | encode content and annotations using multiple dense vectors |
| sparse | sparse encoding of annotations only (textual content ignored) |
| sparse_content | sparse encoding of annotations, dense encoding of content |
| sparse_merged | sparse encoding of annotations, dense encoding ofmerged content and annotations |
| sparse_multivector | sparse encoding of annotations, content and annotations embedded using multivectors |


