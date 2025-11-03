# RAG authoring

When creating a vector database for RAG, it is not necessary to curate the content: LM markdown for education will ingest all markdown documents in the input directory and add the relevant information automatically. However, it is also possible to curate this document. Here, we detail how this may be done in an interactive interactive process.

## Step 1: preparing content

Creation of a RAG application starts with the documents the application will be serving. You can prepare all documents at once, or incrementally.

The language model may assist the RAG author in the creation of content. You can ask the language model to comment on your content for clarity and conciseness, to suggest improvements, to create questions that your text answers to verify the points that your text addresses.

The interaction with the model takes place through metadata blocks. You insert the property `query:` in the block, and send the markdown to the language model for response. For example, in the following markdown, a query property was manually added to the markdown text:

``` markdown
---
query: evaluate the text for clarity and conciseness.
---

## What are linear models?

Linear models and their generalizations constitute the majority of the statistical models used in practice. Here, we will look at linear models from a practical perspective, emphasizing the issues in correctly applying them and in understanding their output.

Linear models capture the association between a set of variables, the *predictors* (also known as *independent variables*), and an *outcome* variable (or *dependent variable*). In the simplest setting, this association may be approximated and displayed by a line relating a predictor and the outcome. However, the generalizations of linear model allow capturing much more complex associations.

## The generalization of linear models

Linear models can be generalized to cover situations where the outcome is not continuous, and therefore linearity cannot apply. Further text omitted...
```

Here, part of the message that is sent to the language model is "evaluate the text for clarity and conciseness". This message part is integrated with the text to which the metadata refer, which is here the text under "What are linear models?", but not the text under "The generalization of linear models". If there had been subheadings after "What are linear models", the text of the subheadings would also be sent to the language model. The term 'text' here is key: the language model knows that your question refers to the parts of the markdown that are sent over, because these are qualified as "text" in the message.

To send the message to the model, open a Python REPL, and send the following command:

```python
from lmm_education import scan_messages

scan_messages("Lecture01.md")
```

where Lecture01.md is the markdown file in questions. The response of the language model appears in the editor under the property `~chat`:

``` markdown
---
query: evaluate the text for clarity and conciseness.
---

## What are linear models?

Linear models and their generalizations constitute the majority of the statistical models used in practice. Here, we will look at linear models from a practical perspective, emphasizing the issues in correctly applying them and in understanding their output.

Linear models capture the association between a set of variables, the *predictors* (also known as *independent variables*), and an *outcome* variable (or *dependent variable*). In the simplest setting, this association may be approximated and displayed by a line relating a predictor and the outcome. However, the generalizations of linear model allow capturing much more complex associations.

## The generalization of linear models

Linear models can be generalized to cover situations where the outcome is not continuous, and therefore linearity cannot apply. Further text omitted...
```

You can use the interaction with the language model to have it criticize your text, and adopt the improvements that the model suggests if they do improve the quality of your writing. If you ask the model to criticize text or suggest improvements, it will always find something to criticize and improve. Hence, you can keep improving until you think that the suggestions of the language model are superfluous.

A common approach to interact with the language model is to create provisional sub-sections to stake our the parts of the text that the language model should work on. After the interaction is completed, the sub-sections and the chat content is deleted.

## Step 2: generation of annotations

Annotations are metadata properties that facilitate the retrieval of text from the vector database. Consider the following text.

``` markdown
### Observational studies

These are studies with models that contain predictors that were observed in the field, and could not or were not the result of an experimental manipulation of the predictor value. For example, a model of depressive symptoms in adults as predicted by childhood trauma would be such a model.
```

Here, the text that is sent to the vector database never contains the word "observational". Hence, the database will have a hard time figuring out that this text is relevant to answer the question "What are observational models?". The only common point between the question and text here is the word "model", but there will be a lot of text in the database with this word. Worse, the question "What are observational studies?" has no points of contact with the text.

Furthermore, other text in the database may contain the term "observational model" even if it does not explain at all what observational models are. Consider the following example.

``` markdown
### Somewhere in text

When we look at the significance of the association between predictors and outcomes, it is important to distinguish between two different settings. In the first, we have variables that we have observed in the field. For example, early traumas may exert an influence on the predisposition to psychopathology in adult age. We do not have any way to change the occurrence of traumas in the past with an experiment, so we look at their consequences in a sample from the population. Studies of this type are called *observational*.

### Elsewhere in text

Linear models are not oracles that can divine aspects of reality: all that they see is just numbers. Therefore, an important task when estimating linear models is our capacity to relate these numbers to measurements and observations in the real world, and interpret the output of the model in the light of this knowledge. The distinction between the interpretation of the associations in observational and experimental studies is one example of this task.
```

Here, questions on "observational studies" may retrieve the second chunk, even if the chunk that explains what observational studies are is the first. This is because the fragment "observational and experimental studies" is closer semantically to a questions on observational studies than the fragment "studies of this type are called observational." This example shows that a mechanism is needed to characterize the semantics of pieces of text to facilitate their retrieval.

One think to keep in mind is that the performance of sentence embeddings to capture semantic similarity varies across providers and embeddings size. Hence, the efficiency of retrieval may improve by adopting better sentence embeddings. However, a downside of this approach is that one embedding type must be used for the whole database, so it is not possible to improve the embedding selectively for parts of text that do not perform. To update the embedding, the whole vector database must be re-ingested anew.

Annotations are the best mechanism to improve the capacity of the vector database to retrieve the right parts of text, because they more precisely enhence the capacity to represent specific content. Annotations constitute additional information that is used in the embedding. Annotations that may improve embedding performance are:

-   the title or the concatenated title of the heading over the text
-   questions that the text answers
-   keywords (forthcoming)

LM markdown for education uses the language model to compute the annotations, but they can always be integrated or edited in an interactive loop. What annotations should be generated is specified in the "RAG" section of config.toml:

```         
[RAG]
questions=true
titles=true
keywords=false
```

'true' and 'false' switch the automatic generation of annotations on and off.

Note: it is strongly recommended to use the same annotation model in the whole project.

To have LM markdown to generate the annotations without ingesting the document, use Python from the console:

``` bash
python -m lmm.scan.scan_rag.markdown_rag("Lecture01.md")
```

or from the Python REPL:

```python
from lmm_education import scan_rag

scan_rag("Lecture01.md")
```

After this, any modern editor that has Lecture01.md open will display the annotations added by the langauge model:

``` markdown
---
~txthash: 7JdFs+GLpXTfvmONOnyc5g
titles: Chapter 1 - What are linear models? - Observational studies
source: Ch1
questions: What are observational studies? - How does confounding affect the interpretation of associations in observational studies?
---
### Observational studies

We will first discuss issues arising from assessing the significance of associations.

When we look at the significance of the association between predictors and outcomes, it is important to distinguish between two different settings. In the first, we have variables that we have observed in the field. For example, early traumas may exert an influence on the predisposition to psychopathology in adult age. We do not have any way to change the occurrence of traumas in the past with an experiment, so we look at their consequences in a sample from the population. Studies of this type are called *observational*. An important issue in observational studies is that predictors may be *confounded* by other factors affecting the outcome. Confounding occurs when the relationship between the predictor and the outcome is distorted by the presence of a third variable. For example, traumas may occur more frequently in adverse socioeconomic conditions, and these conditions may in turn adversely affect the predisposition to psychopathology. When we assess the association between traumas and psychopathology, the association we find may inadvertently include the effects of socioeconomic conditions, as traumatized individuals are often those who are most disadvantaged socioeconomically. Importantly, the existence of an assocatio between an observational variable and an outcome cannot be drectly interpreted as providing evidence of a _causal_ relationship between the variables.
```

To edit or add questions, just add them in the metadata block in the `questions` property.

### Manual annotations

The properties `questions` and `keywords` are special, because LM markdown for education uses language models to fill them. However, you can also add you own property. For example, you might want to add a property `concepts` or `topic`, where you manually insert the values of the property in the metadata block:

``` markdown
---
~txthash: 7JdFs+GLpXTfvmONOnyc5g
titles: Chapter 1 - What are linear models? - Observational studies
source: Ch1
questions: What are observational studies? - How does confounding affect the interpretation of associations in observational studies?
concepts: observational studies - confounders - confounding
---
Text follows...
```

To tell LM markdown that these metadata properties are meant for annotations, include them in the `RAG.annotation_model` section of config.toml:

``` toml
[RAG.annotation_model]
inherited_properties = []
own_properties = [concepts]
```

There are two entries here. The entry `inherited_property` means that an annotation of a heading will be propagated to all subheadings. This does not happen if the annotation is listed in `own_properties`.

You can list more than one property as an annotation. In this case, separate the properties with a comma: `own_propertes = [concepts, topics]`.

It is not necessary that a manual annotation, when listed in config.toml, is present in all headings of the markdown. However, if the encoding model uses annotations to compute the semantic representation of text (see next section), then some annotation must be present. Annotations are added to each other when computing the semantic representation. In the example above, the annotations used for this computations are `titles`, `questions`, and `concepts`, after adding this latter to the annotation model in config.toml. So, if `concepts` is missing, the other two annotations are still available to compute the semantic of the text.

If a property is added to a header without listing it in config.toml, then it is not used to encode the semantics of the text. For example, you could add a `comments` property, or a `TODO` property for your own use.

### Structuring the markdown text

LM markdown allows for inserting manual annotations for each text block individually, while automatic annotations are created on a per-heading basis. Keep this in mind when structuring the text. As a rule of thumb, it is better to insert sub-headings to cluster text that goes together and put the related annotation in the metadata of the subheading, rather than inserting metadata blocks for individual text blocks. However, this latter strategy allows for differentiating retrieval of individual blocks. It is not necessary to create manual annotations for that - one could insert a `questions` property in a metadata block before a text block. However, keep in mind that this annotation completely replaces the annotation of the heading.

When responding to a query, the vector database retrieves parts of text (called *chunks*). How text is split into chunks depends on the *text splitter* used to prepare chunks (see the section on text splitting below). Therefore, the structuring of the text in text blocks and headings should be made keeping in mind what the language model may obtain when formulating a response to the question of the student. This encourages text organization styles that group content into relatively well contained blocks or headings.

In LM markdown for education, you can direct the software to retrieve the whole text of a subheading where the chunk is located, instead of the chunk. This decouples the splitting of text for the purposes of establishing its semantics and relevance for the query, and the coherence of the material the language model can use to formulate the response. In this case, keep in mind that the retrieved text is that of the subheading. Text blocks can be annotated liberally as this will increase the precision of their semantic encoding, without having an impact on the retrieval of background text.

### Encoding models and embeddings

Vector databases retrieve data based on embeddings. These are vectors (ordered sets of numbers) with the property that vectors with similar numbers represent similar meanings. Therefore, vector databases are queried by providing some text (for example, the question of the user). The embedding of the query text is computed, and the chunks of text in the database are retrieved with the most similar meaning, according to the representation afforded by the embedding.

Without annotations, embeddings are created from the text that is stored in the database. With annotations, there are a number of ways to create embeddings that better represent the content of the text.

| encoding type | description |
|------------------------------------|------------------------------------|
| merged | the content of the annotations and the content of text are concatenated prior to forming the embedding |
| multivector | two separate vectors are used as embeddings, one from the annotations and one from the text |
| sparse | annotations are treated as keywords (as in web search) and are the only ones that are encoded |
| sparse_content | annotations are embedded as keywords, and text through a vector |
| sparse_merged | annotations are both embedded as keywords and concatenated to text prior to create the vector embeddings |
| sparse_multivector | annotations are embedded as keywords, separate vector embed annotations and text |
| content | the case when there are no annotations (only text embedded as vector) |

These options trade off accurate encoding of annotations on the one hand and cost on the other. In the *merged* encoding option, there is little increase in cost but more accurate representation of annotation semantics in the embeddings. The *multivector* and *sparse_content* encoding options provide separate encoding of annotations, so that they automatically receive more weight when the text is large. However, two encodings are computed and stored for each data point. the *sparse_multivector* is the most expensive and the option that puts most weight to the annotations relative to the text.

You can customize the encoding model by entering the option in config.toml in the RAG section:

```         
[RAG]
encoding_model=sparse_merged
```

Remember that the same encoding model must be used in all interactions with the database.

The encoding model influences how the text is encoded, but the retrieved text is not necessarily the same as the one from which the encoding was computed. This is because the retrieved text must contain context for its use in generating a response well. LM markdown for education can be directed to retrieve the whole text of the subheading where the chunk is located. This directive is in the `database` section:

```         
[database]
collection_name = "chunks"
companion_collection = "documents"
```

If `companion_collection` is not empty, then the retrieved text is the one of the subheading.

The following table summarizes the models used in RAG.

| model | explanation |
|------------------------------------|------------------------------------|
| annotation model | what properties of metadata, if any, are used to compute the encoding |
| encoding model | how to combine annotations and main text to create embeddings (a technical question about how to structure the database) |

### Skipping parts of the markdown

You can annotate text blocks or entire headings and their underlying material such as to exclude them from the RAG. To this end, use the annotation `skip: True`:

``` markdown
---
title: chapter 1
---

# Introduction

This text is sent to the RAG database.

---
skip: True
---
This text is not sent.

This text is sent again.

---
skip: True
---
## More introductory words

Everything that lies under this heading is not included in the database.
---
```

## Step 3: test retrieval

When forming the database, or after adding new text, you may want that the right text is retrieved by queries. LM markdown for education is also designed for the database to grow with the experience of the questions asked by students and that the system did not address. You can write a new markdown document with the material to answer these new question, and add it to the database.

When doing this, you may want to be sure that the original question or similar questions retrieve this material if asked again. This is what test retrieval is for.

You start with ingesting the document with Python, and then you test that the material is retrieved when you ask the question:

``` bash
python -m lmm_education.ingest("AddedMaterial.md")

python -m lmm_education.querydb("What is the reason to add a family parameter to glm call?")
```

or from the Python REPL:

```python
from lmm_education import ingest
from lmm_education import querydb

ingest("AddedMaterial.md")
response = querydb("What is the reason to add a family parameter to a glm call?")
print(response)
```

If the material is not retrieved as you expected, go back to the document and change the annotations, trying to be specific. If even this does not work, it may help changing the text to be more explicit about what the text explains (for example, by mentioning the textual fragment "family parameter to glm call"). The interaction with the model may be repeated, as long as the `docid` property in AddedMaterial.md is not changed. Keeping this property to the same value ensures that the edited markdown replaces the old in the database.

After you are satisfied with the way the material is being retrieved from the database in response to your queries, you can test the response from the language model with the `query` function:

``` bash
python -m lmm_education.query "What are observational studies?"
```

Using the Python REPL:

``` python
from lmm_education import query

response = query("What are observational studies?")
print(response)
```

The configuration file contains three model specifications: `major`, `minor`, and `aux`. By default, the end-user interactions with the language model use the major model. You can experiment with other models by specifying them as a second argument to the query:

``` bash
python -m lmm_education.query "What are observational studies?" minor
```

When using the Python REPL or from code, you can specify a model to which you have access directly using the following syntax.

``` python
from lmm_education import query

response = query("What are observational studies?", {'model': "OpenAI/gpt-4-mini", 'temperature': 0.3})
print(response)
```

### Forming a database of questions

It may be a good a idea to keep a note of the questions that created problems during the course, or of questions that are asked frequently. If the encoding model is changed and the database formed anew, or if the RAG material is subsequently substantially revised, these questions may be used to create a battery to evaluate the performance of retrieval.
