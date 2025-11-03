# Encoding and embeddings

Vector database allows retrieving data based on a search query that contains natural text, such as a question. This differs from standard relational databases, where the search is based on matching the content of a field or searching part of text for keywords.

To retrieve data from the vector database, the text that is ingested in the databased is coupled with _embeddings_ that represent its content semantically. There are two types of embeddings: dense and sparse. Dense embeddings are provided by language model or other models trained from large corpora of text in the same way. Sparse embeddings are more similar to keywords, but are capable to match semantically related words even if the words are not identical.

The embedding model is dependent on the vector database implementation, and it is not directly exposed to the user. Internally, the framework specifies an _embedding model_ to define what types of embeddings should be used in the vector database: dense, sparse, or a combination of both. The mebdding model does not know where the input for the embeddings come from.

Instead, the user controls the embedding strategy by setting up an _encoding model_. The encoding modes specifies how and what parts of the text are used for the dense and what are used for the sparse embeddings. In general, dense embeddings are used for the chunks of text that represent the text stored in the RAG database. In constrast, sparse embeddings, and possibly dense embeddings, are usually formed after enriching the text so as to increase the efficiency of its retrieval.

One strategy is to couple text with the questions that the text answers, or other metadata such as keywords, the title of the heading, etc. These data can then be combined with the text of the dense embeddings, or be kept separate for an additional embedding, usually of the sparse variety. These properties are called _annotations_ and are stored in the markdown in metadata.

Most often, annotations are produced by a language model prior to ingestion, but the framework allows one to add or review them manually.

In short, this is the general strategy for preparing text for ingestion.

annotation model          -->   encoding model  -->   embedding model
(and its implementation)

Note that the implementation of the annotation model is not a monolithic step, as it depends on whether the annotations should be created by code, a language model, or manually. When the annotations are created by code or a language model, they are automatically added to the annotation model. However, it is also possible to annotate parts of text manually and add the relevant information to the annotation model. In short, annotations may be those that are added automatically when code or language models are directed to process text, or those that are added by explictly listing them in the annotation model.

Annotations are invariably properties of metadata, and metadata are blocks in the markdown that refer to the block that follows. When they precede a heading, they refer to all text blocks under that heading, unless overridden by metadata preceding sub-headings. Properties in metadata are not automatically considered annotations; they must be added to the annotation model.

## Language model directives or code that create annotations

Markdown files may be reviewed by a large language model, and part of this review is the creation of annotations. As mentioned, these annotations are automatically added to the annotation model.

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

This results in a metadata block being added prior to headings (the example only shows one of such headings). The titles and questions are now two properties of the metadata. Note also the `~txthash` property. This is an automatically generated hash that checks that the text has not changed since the last time annotations were computed. Annotations are not recomputed if the text to which they refer was not changed.

## Manual annotations

Manual annotations may be added, if desired, directly in the markdown file. An annotation is simply a metadata property. To tell the framework that this property is an annotation, you can specify it in config.toml.

For example, here you introduce a manual annotation called 'topic':

```markdown
---
topic: 'review'
---

Explain when it is appropriate to use a logistic regression.

```

Then, in config.toml:

```
[RAG.annotation_model]
own_properties = ["topic"]
```

Manual annotations can be "own" or "inherited". Inherited annotations are taken from ancestor nodes, i.e. headings higher in the hierarchy.

Annotations are concatenated for sparse or dense embeddings, depending on the encoding model.

## Reviewing annotations

LM markdown for education is designed to allow the RAG author to review all annotations manually prior to ingesting the documents. In the python REPL,

```python
from lmm_education import ingest

ingest("MyMarkdown.md", save_files=True, ingest=False)
```

This will prepare the markdown for ingestion without ingesting it, and save it instead to disk for review and editing.

When editing prepared markdown, annotations will not be recomputed if the text to which they refer does not change. To prevent annotations from being recomputed when the text changes, add the property 'frozen' to the metadata:

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


