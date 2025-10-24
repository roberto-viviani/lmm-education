# Working with markdown

Markdown is a convenient format to create textual material that may be then converted into a large variety of formats. In LM markdown for education, markdown is used to harness large language model to increase the capability of the author and to produce AI-supported material for teaching, such as a system to ask questions and review the lecture content.

Markdown may be seen to contain three general types of material: metadata, headings, and everthing else (largely including the text). A simple rule is that these markdown elements are separated by a blank line. There are exceptions to this, determined by the markdown specification, but putting a blank line between elements keeps things simple. Multiple blank lines are treated as a single blank line.

## Metadata

Metadata are very commonly used in markdown tools to create a header with information such as title, author, date, etc. This is how this may look like:

```markdown
---
title: Working with markdown
author: John Sammelwater
date: 23.10.2025
---

Text follows here...
```

In LM markdown, a rule is that the header should contain a title property. If you create a markdown without a header with this property, LM markdown will try and create one from the name of the file. (If this name is not available, it will create a title property with value 'title', but it is best to avoid this when the markdown is used for RAG.)

LM markdown also differs from other markdown systems in that it uses metadata throughout the whole document. Metadata are marked at the beginning and at the end by three dashes starting at the start of the line,

```
---
```

These dashes delimit _metadata blocks_. In LM markdown, metadata blocks are interpreted as referring to whatever follows them. Thus, metadata blocks immediately preceding a heading refer to the heading and all content within it; metadata blocks preceding a text block refer and therefor annotate the text block that follows.

## Headings

Headings are the markdown elements that mark chapters, giving them a title. Headings are defined by one or more hash symbols `#`, followed by a blank space, followed by the title. The number of hashes gives the level of the heading, which can be up to 6.

LM markdown uses headings to view a markdown file as organized in a tree. The headings constitute the branches of the tree, and the leaves are text elements under the headings. Metadata preceding a heading are then used to annotate with properties the whole portion of the markdown that falls within the heading (and its subheadings). The metadata properties are inheritable. This means that the metadata properties defined in the header refer to the whole markdown file, unless overridden by the same property somewhere in the file.

Metadata preceding a text block only refer and annotate the text block that immediately follows.

## Text blocks

For our purposes, text blocks is everything else. Text blocks may contain text, of course, but other common content is code, equations, links to images. All these elements are subtypes of text blocks, and form blocks whenever they are separated by blank lines. As noted, whenever you form a block by delimiting it with blank lines, you can also annotate it with a metadata block if you wish. Equation blocks start and end with `$$`. Code blocks start and end with `\`\`\``, optionally followed by the language in which the code is written.

Here is an example of parts of a markdown. It starts with a header, has a heading, and two blocks of text under the heading, one being an equation.

```markdown
---
title: Working with markdown
author: John Sammelwater
date: 23.10.2025
---

This text introduces users to the markdown format.

---
note: the following is a title at level 1
---
# What can markdown contain? 

Markdown can contain text, equations, or code. Below, you see an equation.

---
note: the following is an example of an equation, and this is the meatadata block that annotates it.
---

$$ y ~ \mu + x + \epsilon $$

```

## Identification of content in vector database

The data from the markdown are chunked prior to ingestion in the database. Each chunk generates an embedding, so that when a query is made, the correct chunks are retrieved from the database (see the [encoding](EncodingEmbedding.md) page for details).

Because the material may be changed and updated, the question arises as to how new material is integrated with the old. LM markdown for education uses properties from metadata to identify the chunks so that, if the text is changed, it replaces old text in the vector database.

In the header, the property `docid` is created automatically by LM markdown to identify a document, and initialized to a random string. You can, however, initialize to an intelligible string, as long as it remains unique across documents:

```markdown
---
title: Working with markdown
author: John Sammelwater
date: 23.10.2025
docid: workmd
---

Based on this property, LM markdown adds a `textid` property to all text blocks in the document. The texdid property has the format {docid}.{sequential_number}, where docid is the value of the docid property in the header, and the second element is a sequential number.

As long as these textid elements are present in a document, they will be re-ingested in lieu of the old text in the vector database. Note that the text can become longer, as new docid's will add new material in the database. If the document has not changed, ingesting the document in the database results in the very same content as it had before. Is the database is damaged or missing, it is reintegrated with the old material. Note that embeddings are always recomputed when ingesting documents, but there will be no difference with the old embedding if the content hasn't changed. Because embeddings are recomputed, the order of the material in the markdown file can change without affecting the outcome.

If there are parts of the document that have changed, or new parts, they replace or add the material in the vector database. (To remove parts of the database, it is at present necessary to delete the old database and ingest all documents again).

For legibility, it is a good idea to set the docid property in the header before ingesting a document, and never change it after that. If docid is changed, LM markdown will consider the document a new separate document and will ingest duplicate material in the database.

