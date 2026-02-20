"""
This module allows interactively scanning a markdown file to create
a RAG database.

Examples:

```bash
# Python called from the console

python -m lmm_education.scan_rag LogisticRegression.md
```

```python
# from python code
from lmm_education.scan_rag import markdown_rag

markdown_rag('LogisticRegression.md')
```

These functions allow interactively scanning a markdown file to create
a RAG database.

Main functions:

- `markdown_rag` and `amarkdown_rag`: these functions allow interactively
scanning a markdown file to create a RAG database.
"""

from pathlib import Path

from lmm.scan.scan_rag import (
    markdown_rag as scan_markdown_rag,
    ScanOpts,
)
from lmm.utils.logging import ConsoleLogger, LoggerBase
from .config.config import ConfigSettings, load_settings

logger = ConsoleLogger()


def markdown_rag(
    sourcefile: str | Path,
    *,
    config: ConfigSettings | None = None,
    logger: LoggerBase = logger,
) -> None:
    """Call scan_markdown_rag with default config settings."""

    if config is None:
        config = load_settings(logger=logger)
    if config is None:
        logger.error("Could not load settings.")
        return

    scan_markdown_rag(
        sourcefile,
        ScanOpts(
            titles=config.RAG.titles,
            questions=config.RAG.questions,
            questions_threshold=15,
            summaries=config.RAG.summaries,
            summary_threshold=50,
            language_model_settings=config.major,
        ),
        save=True,
        logger=logger,
    )


async def amarkdown_rag(
    sourcefile: str | Path,
    *,
    config: ConfigSettings | None = None,
    logger: LoggerBase = logger,
) -> None:
    """Call scan_markdown_rag with default config settings."""

    return markdown_rag(
        sourcefile=sourcefile,
        config=config,
        logger=logger,
    )
