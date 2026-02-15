"""Utility functions for app modules"""

# from datetime import datetime
from collections.abc import Callable  # Coroutine

# from typing import TextIO, Any
# from io import IOBase
import asyncio

# from lmm.utils.logging import LoggerBase
from lmm.config.config import LanguageModelSettings


def preproc_markdown_factory(
    model_settings: LanguageModelSettings,
) -> Callable[[str], str]:
    """This function allows centralizing preprocessing
    of strings using a latex style."""
    from lmm.markdown.ioutils import (
        convert_dollar_latex_delimiters,
        convert_backslash_latex_delimiters,
    )

    model: str = model_settings.get_model_source()
    return (
        convert_backslash_latex_delimiters
        if model == "Mistral"
        else convert_dollar_latex_delimiters
    )


async def shutdown(active_logs: set[asyncio.Task[None]]) -> None:
    if active_logs:
        print(
            f"Gradio exited without loop cleanup? Awaiting "
            f"{len(active_logs)} logging tasks..."
        )
        # Create snapshot to avoid set modification during iteration
        tasks = list(active_logs)
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # The only exceptions raised by code are those raised by
        # logging itself, because the coroutines are written to
        # handle all exceptions and write them to the log. But we
        # do not know if Gradio is canceling anything. Diagnostics
        completed = sum(
            1 for r in results if not isinstance(r, Exception)
        )
        cancelled = sum(
            1
            for r in results
            if isinstance(r, asyncio.CancelledError)
        )
        failed = sum(
            1
            for r in results
            if isinstance(r, Exception)
            and not isinstance(r, asyncio.CancelledError)
        )

        # Don't use logging since it may be failing
        if cancelled or failed:
            print(
                f"Results: {completed} completed, {cancelled} "
                f"cancelled, {failed} failed"
            )
        for result in results:
            if isinstance(result, Exception):
                print(f"Logging failed: {result}")
