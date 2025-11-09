"""Interface for lmm.scan.markdown_scan."""

from pathlib import Path

from lmm.scan.scan_rag import markdown_rag, ScanOpts
from lmm.utils.logging import ConsoleLogger, LoggerBase
from .config.config import ConfigSettings

logger = ConsoleLogger()


def scan_rag(
    sourcefile: str | Path,
    *,
    questions: bool = False,
    titles: bool = False,
    summaries: bool = False,
    logger: LoggerBase = logger,
) -> None:

    markdown_rag(
        sourcefile,
        ScanOpts(
            questions=questions, titles=titles, summaries=summaries
        ),
        save=True,
        logger=logger,
    )


def rag(
    sourcefile: str | Path,
    *,
    config: ConfigSettings | None = None,
    logger: LoggerBase = logger,
) -> None:

    if config is None:
        config = ConfigSettings()

    markdown_rag(
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
