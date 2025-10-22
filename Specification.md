---
title: LM markdown for education
author: Roberto Viviani
date: 21.05.2025
---

# LM markdown for education

This adds special provisions for educational use of the markdown. This includes:

- providing summaries to headings that are indexed and retrievable by the student user
- providing means to enable the student user to retrieve a table of content/study plan
- providing means to enable the student user to produce study cards
- providing means to mark out exercises

Retrieval here is keyword-based, leveraging hybrid retrieval techniques.

It also uses a retrieval technique whereby text of headings up to a level are retrieved, leaving to the LMM the task of fishing out the content to be served to the student user. Here as well hybrid retrieval techniques make sure that relevant chapters are retrieved also based on hints that the rag author may insert for this purpose in appropriate metadata fields (~questions).

## Special text blocks for the RAG database

It will be possible to specify special types of text blocks to be ingested in the RAG database with a specific denomination. For example, it will be possible to add text marked as "exercise", or "study topics" that may be prepared for the exam. Also definitions, explanations not included in the main text but available in the non-linear interaction format allowed by RAG. These special text blocks will not be included in the "knitted" lecture notes, but will be available for the RAG interaction.

## Transformation of the document for ingestion

LM markdown uses the tree structure of the document to annotate the document with additional information and, if required, prepare separate lists of "document" objects for search and for retrieval. This intermediate steps can be exported as a markdown to visualize the material that is being ingested in the index. Remember that text, not headings or metadata, are ingested.

The transformation proceeds in three steps. We will here refer to the list of markdown blocks as to the internal representation of the markdown, to "documents" as the format understood by the language model framework (for example, langchain), and to "points" as the format understood by the vector database (if it is used directly instead of being intermediated by the framework).

In a first step, a preliminary evaluation of the text is made. This includes pooling text blocks that contain equations only and very short text blocks, so that context is not lost. A policy could alos be implemented for code blocks.

A second optional step identifies the larger documents that will be retrieved. These could be -- depending on the structure of the headings -- the text under each headings, or the text splitted across text blocks if a given size threshold has been reached. Usually, this step will consist of establishing an adequate document size that will be returned to the language model at retrieval. A final part of this step specifies the annotations that will be computed at this level of granularity on the documents.

A third and final step computes the "chunks" of text by splitting the text from the second step and compute the embeddings (it only makes sense if the level of granularity of the text is smaller than in the previous step). It is always possible to use a no-op chunking, but this is not foreseeen as a normal strategy. To do this, the list of markdown blocks must be transformed into a list of "documents" as understood by the language model framework. Note, however, that before doing this transformation it will be necessary to collect all metadata that go into the embedding, or compute the relevant annotations.

Apart from the first step, the operations of the transformations are customizable. The user can specify the transformation options in the header. The intermediate output is a list of markdown blocks that may be saved as a markdown file for inspections. The final output is a list of "points" that the vector database can ingest (or of "documents" if a language model framework is used as intermediary).

Note that the transformation of a markdown list block will result in one list of "documents" only if the second step is missing. If not, the fransformation returns two lists: the list of the larger documents for retrieval, and the smaller documents for embedding and search.

In addition, some annotations may be transformed into text to be appended to the document lists. This is the case for summary annotations.

From an implementational point of view, this scheme corresponds to the "strategy" pattern. In a functional style, a record of functions or options (i.e., an associative array) may be enough.

## RAG with QDRANT

Markdown documents provide information of two kinds: text and metadata. Headings (the titles in the markdown) are available as metadata of the text under the heading. Metadata are given by the information in the header, and any other information stored in the metadata blocks of the markdown document.

When storing the markdown file with qudrant in a _collection_ (the equivalent of the database table), the data it contains is used in two ways:

- data are used to form the _embedding_ of the data point, i.e. dense or sparse vector (or the combination of both) that the vector database uses to retrieve data based on similarity. Which of these emebdding is used is specified by an _embdding model_. Ordinarily, the text of the markdown is embedded in the dense vector. There are, however, many variations on this strategy. Note that the records in the the database may always also be retrieved based on their id, which is here an uuid string.

- data are stored as the _payload_ of the records (_points_ in qdrant terminology), itself containing one or more fields. When stored as payload, the data may be retrieved from an embedding (obtained from embedding the text of a query) based on its similarity with the embeddings of the records in the database. The fields that should be retrieved from the payload are specified in the query. If no field is specified, the query retrieves the id's of the points.

There is no constraint about data being included in the one or the other group. However, text will normally be used in the dense embeddings, and at some stage also be retrieved to provide context for a language model. More generaly, a storage schema implies selecting the data (text and metadata fields) that go into the embedding on the one hand, and into the payload on the other hand.

ML markdown for education will include in the payload all metadata properties with the exception of those generated during the interaction with user (chat property) and block id's.

The document organization for ML markdown for education includes the following properties:

- a list of questions that the text answers
- titles
- summary of text under headings

These properties are retrieved before the text is chunked. The document and chunks, therefore, can be indexed using different strategies. 

### Questions and titles:

- none present: only text is available for embedding.
- either or both present:
  - merged with text and indexed as a whole block
  - merged together and indexed, sent as a "multivector" to the database together with the separate index of the text
  - added as a sparse vector (for users of the same language)
  - added as a sparse vector (for users of same language), with text as a dense vector for hybrid search
  - added as a sparse vector (for users of same lanugage), with the merged variants listed above for hybrid search

For example, a strategy would be a hybrid query on a multivector to represent questions and titles in boch sparse and dense representations.

### Summaries

Indexed as autonomous blocks (see raptor paper)

### Pooling of text

The document organization allows to retrieve text prior to chunking. Specifically, it is possible to merge the text under a heading, or to merge the text under a heading under a word count limit (see step 2 of document transformation). This text may be stored in a separate collection and retrieved using the 'group_by' and 'lookup' query options. This is a departure from classic RAG strategies, but may be advantageous since now there are hardly limits in the context memory, and the model may be tasked to fish out the relevant information better than just embedding. That is, embedding is used only to make sure to retrieve all relevant texts. 

As a result, there are two options to store the documents:

- based on overlapped chunking (standard strategy)
- form pooled text that is retrieved when a chunk is selected

### Types of material

In addition to text, the following type of text may be explicitly marked by a descriptor in the payload.

- summaries (arising from previous step)
- exercises
- topics included in exams
- whatever else the instructor may want to include

This special material is marked by a value in a 'type' keyword in the payload, and indexed in the main collection.

The user is told that this material may be selectively retrieved by using a keyword in the query. For example, *summary*, or *exercise*. If this keyword is detected, then the software computes a prefetch with a large limit, and applies boosting on the type keyword indicated by the user. If the keyword is found, the prompt to the model can be modified to indicate this fact.

