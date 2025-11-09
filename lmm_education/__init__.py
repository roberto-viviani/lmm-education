"""
LMM Education - A RAG-based educational content management system.

This package provides a stable API for working with markdown-based educational
content, vector databases, and RAG (Retrieval-Augmented Generation) workflows.

Public API
----------
Configuration:
    create_default_config_file: Create or reset the configuration file

Content Processing:
    scan: Scan and parse markdown content
    scan_messages: Extract messages from markdown
    scan_remove_messages: Remove messages from markdown
    markdown_rag: Apply RAG preprocessing to markdown

Database Operations:
    ingest: Upload markdown content to vector database
    querydb: Query the vector database
    query: Perform semantic queries
    database_info: Get database information and statistics

Example
-------
    >>> import lmm_education as lme
    >>> lme.create_default_config_file()
    >>> lme.ingest("path/to/markdown.md")
    >>> results = lme.query("What is the main topic?")
"""

# Import configuration first to ensure proper initialization
from .config.config import create_default_config_file

# Import from external lmm package (scanning and preprocessing)
from lmm.scan.scan import markdown_scan as scan
from lmm.scan.scan_messages import (
    markdown_messages as scan_messages,
    markdown_remove_messages as scan_remove_messages,
)

# Import from local lmm_education package (core functionality)
from .ingest import markdown_upload as ingest
from .querydb import querydb
from .query import query
from .stores.vector_store_qdrant_utils import (
    database_info,
)
from .scan_rag import rag as scan_rag

# Define public API
__all__ = [
    # Configuration
    "create_default_config_file",
    # Content processing
    "scan",
    "scan_messages",
    "scan_remove_messages",
    "scan_rag",
    # Database operations
    "ingest",
    "querydb",
    "query",
    "database_info",
]

# Version information
__version__ = "0.1.0"
