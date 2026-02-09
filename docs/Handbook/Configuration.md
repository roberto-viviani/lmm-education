# Configuration

The files `config.toml` and `appchat.toml` in the directory of the project contains the instructions to configure its function. Essentially all user-configurable options are specified in these files.

# config.toml

## Language models

The language models used in LM markdown for education are specified through the sections `major`, `minor`, and `aux`. These sections constitute three tiers of models, grading cost and latency depending on the task. 

| model | use (examples) |
| --- | --- |
| major | in direct interactions with the end user |
| minor | to form summaries, generate annotations |
| aux | to classify text, agent routing |

The language models are specified by separating the model provider and the model name by '/':

    OpenAI/gpt-4.1-mini
    OpenAI/gpt-4.1-nano
    Mistral/mistral-small-latest

It is also possible to specify other parameters of the model, such as temperature. Some parameters are model-specific, and need to be set following the specific instructions of the model vendor.

A special language model specification is Debug/debug, which avoids using a real language model.

## Embeddings

The embedding model is specified in the section `embeddings`. Embeddings are used to store and retrieve text from vector databases. The embeddings that are used by the project are dense embeddings (classically used by RAG applications) and sparse embeddings.

Specification of dense embeddings contain an embedding provider and a model name, separated by '/':

    OpenAI/text-embedding-3-small
    SentenceTransformers/all-MiniLM-L6-v2
    SentenceTransformers/distiluse-base-multilingual-cased-v1"

Two sparse embeddings are supported at present.

    Qdrant/bm25
    prithivida/Splade_PP_en_v1

The default sparse embedding is Qdrant/bm25 as it is a multilingual embeddings. distiluse-base-multilingual-cased-v1 is a multilingual open-source embedding model.
 
## Storage

This section configures where the vector database is located. Specify a folder to save the vector database locally, or a valid internet address/port number for a Qdrant server.

The `database` section contains the names of the collections used to store the data. These names can be left as they are. However, the existence of a non-empty `companion_collection` property means that whole text sections, not the chunks used to generate embeddings, will be stored in the database (more details on this are in [RAG authoring](RAGauthoring.md)). To switch retrieval mode, use the setting `retrieve_companion_docs` in the `[RAG]` section of config.toml.   

## RAG section: annotation model

This section is used to activate a language model to analyze the content of the markdown file and produce additional content. This content may either be a summary of the text under the heading (option `summaries`), or metadata describing the text that will be using to encode the semantic content ("annotations"). These metadata properties can also be filled in or changed manually. At present, you can specify `titles` and `questions` as metadata for endoding. 

The [RAG.annotation_model] section allows specifying user-defined metadata properties to be used in the encoding. The annotation model is specified as a list of keys under `inherited_properties` and `own_properties`. These two types of specifications refer to how annotations are sought. When specified as inherited, the property is sought in the node and in all its ancenstor, until it is found (if present anywhere in the ancestor tree). When specified as own, only the node is sought.

When one specifies a predefined annotation, for example `questions = true`, this metadata property will be automatically added to the annotation model -- there is no need to repeat it in the RAG.annotation_model section.

'Encoding' specifies how annotations are used to generate an embedding. The options here are described in detail in the [encoding and embedding](EncodingEmbedding.md) part of the manual.

The specification `filters` in the annotation model is reserved for future use. Setting this specification has no effect on the working of the program at present.

The specification `retrieve_companion_docs` retrieves whole documents instead of chunks. For this directive to have effect, a companion collection must be specified at ingestion.

## Text splitting

Choose here the text splitter to generate the chunks of the embeddings and their size. Default options are generally ok.

# Application servers

The application servers start listening for connection at the specified port. They may be configured using their specific settings. The system settings of config.toml apply to all application servers.

## appChat

This configuration file controls the text that appears on the web application, the type of content that is allowed in the chat, and the port where the server is listening.

### Server configuration

The server section allows specifying how the web server will be running. The `mode = local` is useful for testing and debugging, running a local server. The settings `mode = remote` sets up a web server listening at the port `port`.

### Text appearing in the web application

The following settings control the text displayed in the web app.

- `title`, `description`, `comment`: the name of the lecture and descriptive information. See the documentation of the [chat application](ChatApp.md) for a more detailed description of the text that is displayed on the web page.

- `MSG_EMPTY_QUERY`, `MSG_WRONG_CONTENT`, `MSG_LONG_QUERY`, `MSG_ERROR_QUERY`: text of the responses displayed in several user
error conditions. If the English language is acceptable, these settings can be left as they are.

- `SYSTEM_MESSAGE`: Customize this message to specify the personality of the chatbot and -- importantly -- to specify constraints on the content the chatbot may deliver.

- `PROMPT_TEMPLATE`: used in a simple chat workflow to prompt the model with the query and the context from the vector database.

### Chat settings

- `max_query_word_count`: the maximal number of words contained in a query. Larger query texts will be rejected. The application may use  MSG_LONG_QUERY as a response to the user.

- `history_integration`: the approach used to integrate past chat history when retrieving context. Possible values: 'none', 'summary', 'content_extraction', and 'rewrite'. 'none' attaches a fixed number of past messages in the exchange. It is fast and the most economical choice, but tends to reproduce the original context in every new chat turn. The most performant options are 'content_extraction' and 'rewrite'.
    
- `history_length`: the number of past messages sent to the language model when chatting.

- `workflow`: the agent used for chatting. Possible options are 'workflow' and 'agent'. 'workflow' provides a sequential series of steps to process the query. This is the most efficient option when chatting, while delivering good results to reasonably behaved users. 'agent' is a more sophisticated strategy that autonomously rewrites the query and retrieves context from the vector database. It can respond meaningfully in a number of situations when 'workflow' performs poorly, but is slower and more expensive.

### Content validation

Content validation can be implemented at two levels. The first is the prompt, instructing the language model only to reply to queries about a certain content, or that are represented in the retrieved context. The second is to send query and the first part of the response to a LMM to classify the content.

The default chat prompt includes text for the first level of validation. To switch on the second, you set the `check_response` property in config.toml to true:

```ini
[check_response]
check_response = true
allowed_content = ["statistics", "R programming"]
initial_buffer_size = 320
```

If you do that, you must also provide one or more allowed contents in the `allowed_content` list. The `initial_buffer_size` controls how much of the initial response of the model is sent for content classification (in characters). If the chatbot is replying that it cannot respond to legitimate queries due to misclassification, try and increase the buffer size. There is a tredeoff between the accuracy of content assessment and the latency with which the response starts streaming in the chatbot.

In summary, the any strategies to control content can be activie at two levels:

* introduce a content limitation in the system prompt, instead of leaving to something generic like "You are a helpful assistant". The system prompt is present in all chat interactions, irrespective of how prolonged.
* switch on content validation, which checks each response of the language model.

### Exchange database logging

The exchanged messages and the retrieved context are saved to a database as specified in the `chat_database`setting. At present, the application only supports .csv files. Consists of the following fields:

- `messages_database_file`: the file saving queries and responses.

- `context_database_file`: saves the retrieved context.

## appWebcast

This application displays videos _and_ a chat over the content of the video. The same settings as in appChat apply. In addition, the following settings may be set here.

- `SOURCE_DIR`: the folder containing video/audio/image files for presentations.

- `OPENAI_VOICE`: the OpenAI voice selected for rendering. Defaults to 'nova'.

- `OPENAI_VOICE_INSTRUCTION`: the prompt for the OpenAI voice. 

