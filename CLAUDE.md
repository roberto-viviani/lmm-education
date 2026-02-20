# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LM Markdown for Education is a RAG (Retrieval-Augmented Generation) system that enables educators to create AI-powered interactive lectures and chatbots from markdown documents. The system ingests markdown files into a vector database (Qdrant), processes them with language models, and serves them through chat interfaces or interactive webcasts.

## Development Setup

```bash
# Install dependencies
poetry install

# Create default configuration files
poetry run lmme create-default-config-file

# Activate virtual environment
poetry shell

# Run tests
poetry run pytest

# Run a specific test
poetry run pytest tests/test_integration.py -v

# Run tests matching a pattern
poetry run pytest -k "test_config"

# Serve documentation locally
mkdocs serve
```

## Core Commands

### CLI Interface

The `lmme` command provides the main CLI. For interactive use with faster response times:

```bash
# Start interactive terminal (preloads Python and libraries)
lmme terminal

# Then use commands without 'lmme' prefix:
> scan_rag MyLecture.md
> ingest MyLecture.md
> query "What are linear models?"
```

### RAG Workflow

```bash
# 1. Process markdown with language model annotations (summaries, questions)
lmme scan-rag MyLecture.md

# 2. Ingest processed markdown into vector database
lmme ingest MyLecture.md

# 3. Query the RAG system
lmme query "your question here"

# Query the database directly (without LLM synthesis)
lmme querydb "your question here"

# Ingest entire folder
lmme ingest-folder ./lectures --extensions ".md;.Rmd"
```

### Content Authoring

```bash
# Interact with LLM for content improvement (add 'query:' metadata in markdown)
lmme scan-messages MyLecture.md

# Clear LLM interaction messages from markdown
lmme scan-clear-messages MyLecture.md

# List titles that have changed since last scan
lmme scan-changed-titles MyLecture.md
```

### Running Applications

```bash
# Start chat web application (Gradio)
appChat

# Start webcast application (video + chat)
appWebcast

# or via python
python -m lmm_education.appChat
python -m lmm_education.appWebcast
```

## Architecture

### Directory Structure

- **lmm_education/** - Main package
  - **config/** - Configuration management (`config.toml`, `appchat.toml`)
  - **stores/** - Vector database interfaces
    - `vector_store_qdrant.py` - Core Qdrant operations
    - `langchain/vector_store_qdrant_langchain.py` - LangChain integration
  - **workflows/** - LangGraph workflows for chat and query processing
    - `langchain/chat_graph.py` - Simple workflow for chat
    - `langchain/chat_agent.py` - Advanced agentic workflow
    - `langchain/stream_adapters.py` - Stream processing adapters
  - `ingest.py` - Markdown ingestion pipeline
  - `query.py` - RAG query functions
  - `lmme.py` - CLI interface (Typer)
  - **appbase/** - Shared setup for Gradio applications
- **tests/** - Pytest test suite
- **docs/** - MkDocs documentation
  - `appChat.py`, `appWebcast.py` - Gradio web applications (CLI entry points: `appChat`, `appWebcast`)

### Key Architectural Concepts

**Markdown Processing:**
- Markdown documents are parsed into hierarchical trees based on heading levels
- Metadata blocks (YAML) can annotate sections (before headings) or individual blocks
- Text is chunked for embedding, but whole sections can optionally be retrieved
- The `lmm` package (dependency) provides markdown parsing and tree utilities

**Vector Database (Qdrant):**
- Multiple encoding strategies: `dense`, `sparse`, `multivector`, `hybrid_dense`, `hybrid_multivector`, `UUID`
- "Annotations" are metadata properties used for encoding (e.g., `questions`, `summaries`, `titles`)
- Companion collections enable retrieving whole document sections instead of chunks
- Collections are configured in `config.toml` under `[database]`

**Workflows:**
- `workflow`: Simple sequential workflow (fast, economical, recommended default)
- `agent`: Sophisticated agentic workflow with autonomous query rewriting (slower, more capable)
- Workflows use LangGraph with dependency injection and state management
- Stream adapters provide functionality like logging, content validation, and context streaming

**Configuration:**
- `config.toml`: System-wide settings (models, embeddings, storage, RAG parameters)
- `appchat.toml`: Chat application settings (prompts, validation, logging)
- Settings use Pydantic validation and can be overridden programmatically
- Three model tiers: `major` (user-facing), `minor` (summaries), `aux` (classification)

### Encoding Models

The encoding model determines how text and annotations are embedded in the vector database. Configured in `config.toml`:

```toml
[RAG]
summaries = true          # Generate summaries with LLM
questions = true          # Generate questions with LLM
titles = true             # Use titles in encoding

[RAG.encoding]
encoding = "hybrid_dense"  # Options: dense, sparse, multivector, hybrid_dense, hybrid_multivector

[RAG.annotation_model]
inherited_properties = ["title"]  # Search up ancestor tree
own_properties = ["questions"]    # Only at current node
```

### Companion Collections

When `companion_collection` is set in config, whole document sections are stored separately from chunks:
- Chunks provide embeddings for retrieval
- Retrieved sections are coherent, complete text under a heading
- Enable in config: `retrieve_companion_docs = true`
- Control retrieval count: `max_companion_docs = 5`

## Error Handling Philosophy

The codebase uses a custom `Logger` class (from `lmm` package) instead of traditional exception handling:

1. **Coding errors**: Crash with stack trace (type-checked to prevent)
2. **Validation errors**: Pydantic validation crashes immediately with informative messages
3. **Expected errors**: Handled via Logger passed as last function parameter
   - Returns `None` or empty results instead of raising exceptions
   - Type checker enforces checking return values
   - Logger implementations: console trace, error collection, exception throwing

Functions using this pattern are recognizable by their `logger` parameter.

## Testing

- Tests use pytest with strict type checking
- Integration tests in `test_integration.py` and `test_integration_ingestion_query.py`
- Mock objects in `test_mocks.py` for testing without real LLMs/databases
- Many tests verify configuration, vector store operations, and workflow streams
- Tests prefixed with `xtest_` are disabled (e.g., `xtest_vector_store_qdrant_nointernet.py`)

## Important Notes

### Language Models
- Supports multiple providers via LangChain: OpenAI, Anthropic, Mistral, etc.
- Format: `Provider/model-name` (e.g., `OpenAI/gpt-4o-mini`)
- Special provider: `Debug/debug` for testing without real LLM calls
- Temperature and other parameters configurable in config.toml

### Embeddings
- Dense embeddings: `OpenAI/text-embedding-3-small`, `SentenceTransformers/all-MiniLM-L6-v2`
- Sparse embeddings: `Qdrant/bm25` (multilingual), `prithivida/Splade_PP_en_v1`
- Cannot change embeddings after database creation without re-ingesting

### Storage
- Local: Specify folder path in `config.toml` `[storage]`
- Remote: Qdrant server address and port
- Memory: `:memory:` for testing

### Content Validation
Two levels of content validation for chat applications:
1. System prompt instructions (always active)
2. LLM-based classification of response content (enable with `check_response = true`)

### Dependencies
- Depends on `lmm` package (private GitLab repository)
- `lmm` provides markdown parsing, tree utilities, scanning, and base configuration
- See `.venv/src/lmm/` for lmm package source (installed in editable mode)

## Common Patterns

### Initializing Configuration
```python
from lmm_education.config.config import ConfigSettings

# Load from config.toml
config = ConfigSettings()

# Override specific settings
config = ConfigSettings(
    RAG={'questions': True, 'summaries': False},
    embeddings={'dense_model': 'OpenAI/text-embedding-3-large'}
)
```

### Working with Qdrant
```python
from lmm_education.stores.vector_store_qdrant_context import global_client_from_config

# Always use global_client_from_config (handles connection lifecycle)
client = global_client_from_config(config)

# Client is automatically closed when no longer needed
```

### Processing Markdown
```python
from lmm_education.ingest import markdown_upload
from lmm_education.config.config import ConfigSettings

config = ConfigSettings()
markdown_upload(
    "MyLecture.md",
    config_opts=config,
    save_files=True,  # Save intermediate files
    ingest=True,      # Upload to database
    logger=logger
)
```

## Type Checking

The codebase uses strict type checking. When working with Pydantic models, type checkers may flag dictionary-based initialization, but this is intentional and validated at runtime.

## Poetry CLI Entry Points for Gradio Apps

- Gradio apps need a `main()` function as the entry point (`:main`), not the `gr.Blocks` object (`:app`)
- Calling `app()` on a `gr.Blocks` object re-enters its context manager — it does NOT launch the server
- After updating `pyproject.toml` entry points, run `poetry install` to register changes