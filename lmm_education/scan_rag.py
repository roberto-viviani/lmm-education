"""Interface for lmm.scan.markdown_scan."""

from pathlib import Path

from lmm.scan.scan_rag import markdown_rag, ScanOpts
from lmm.utils.logging import ConsoleLogger, LoggerBase

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
        logger=logger,
    )
