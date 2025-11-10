"""Interface for lmm.scan.markdown_scan."""

from pathlib import Path

from lmm.scan.scan_rag import (
    markdown_rag as scan_markdown_rag,
    ScanOpts,
)
from lmm.utils.logging import ConsoleLogger, LoggerBase
from .config.config import ConfigSettings

logger = ConsoleLogger()


def markdown_rag(
    sourcefile: str | Path,
    *,
    config: ConfigSettings | None = None,
    logger: LoggerBase = logger,
) -> None:

    if config is None:
        config = ConfigSettings()

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
