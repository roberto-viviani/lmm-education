"""This is a module that centralizes set up for all app modules when
the app start. All app modules are possible entry points of a specific
application. By importing this module, set up takes place in a unified
way across all apps.

At import, this module does the following:

- It reads the name of the config files defined in config.config and
config.appchat, and checks that they exist. If not, default config
files are created and saved to disk.
- It reads in the config files. Validation errors are raised at this
point if any entry in the config files is malformed or illegal. The
config objects are exported as module variables `settings` and
`chat_settings`.
- It creates a BaseRetriever object to read and write from the
vector database. This object is exported as the module-level variable
`retriever`, but the underlying database stream is managed globally.
- It creates an embedding object. This object is not used by the
clients of the module, but the creation loads the embedding model,
which is lengthy and takes place at best before the app starts serving
possible clients.

Example:
    ```python
    try:
        from lmm_education.appbase import appbase as base
    except Exception as e:
        logger.error(f"Could not set up app: {e}")
        exit()

    retriever = base.retriever
    ```
"""

import os

from langchain_core.embeddings import Embeddings

# lmm_education and lmm
from lmm_education.config.config import (
    ConfigSettings,
    load_settings,
    create_default_config_file,
    DEFAULT_CONFIG_FILE,
)
from lmm_education.config.appchat import (
    ChatSettings,
    load_settings as load_chat_settings,
    create_default_config_file as create_default_chat_config_file,
    CHAT_CONFIG_FILE,
)

# Logging of info and errors. Set up first to allow
# logging errors suring the rest of the setup.
import logging
from lmm.utils.logging import FileConsoleLogger  # fmt: skip

logger = FileConsoleLogger(
    "LM Markdown for Education",
    "application setup",
    console_level=logging.INFO,
    file_level=logging.ERROR,
)

# Config files. If config.toml does not exist, create it
if not os.path.exists(DEFAULT_CONFIG_FILE):
    create_default_config_file(DEFAULT_CONFIG_FILE)
    logger.info(
        f"{DEFAULT_CONFIG_FILE} created in app folder, change as appropriate"
    )

if not os.path.exists(CHAT_CONFIG_FILE):
    create_default_chat_config_file(CHAT_CONFIG_FILE)
    logger.info(
        f"{CHAT_CONFIG_FILE} created in app folder, change as appropriate"
    )

# Read settings. Errors are logged prior to raising exception.
_settings: ConfigSettings | None = load_settings(logger=logger)
if _settings is None:
    raise RuntimeError(f"Could not load {DEFAULT_CONFIG_FILE}")
settings: ConfigSettings = _settings

_chat_settings: ChatSettings | None = load_chat_settings(
    logger=logger
)
if _chat_settings is None:
    raise RuntimeError(f"Could not load {CHAT_CONFIG_FILE}")
chat_settings: ChatSettings = _chat_settings

#  create retriever
from langchain_core.retrievers import BaseRetriever
from lmm_education.stores.langchain.vector_store_qdrant_langchain import (
    AsyncQdrantVectorStoreRetriever as QdrantRetriever,
)

# will return grouped retriever if appropriate
try:
    retriever: BaseRetriever = QdrantRetriever.from_config_settings()
except Exception as e:
    raise RuntimeError(f"Could not create retriever: {e}") from e

# An embedding engine object is created here just to load the engine.
# This avoids the first query to take too long. The object is cached
# internally, so we do not actually use the embedding object here.
from lmm.models.langchain.runnables import create_embeddings
from requests import ConnectionError

try:
    embedding: Embeddings = create_embeddings()
    if "SentenceTransformer" not in settings.embeddings.dense_model:
        embedding.embed_query("Test data")
except ConnectionError as e:
    logger.error(
        "Could not connect to the model provider -- no internet?"
    )
    logger.error(f"Error message:\n{e}")

except Exception as e:
    logger.error(
        "Could not connect to the model provider due to a system error."
    )
    logger.error(f"Error message:\n{e}")
    raise RuntimeError(
        f"System error while attempting to "
        f"test embedding model: {e}"
    )
