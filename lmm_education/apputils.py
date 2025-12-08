"""Utility functions for app modules"""

from datetime import datetime
from collections.abc import Callable, Awaitable

from lmm.utils.logging import LoggerBase
from lmm.config.config import LanguageModelSettings
from lmm.language_models.langchain.runnables import (
    create_runnable,
    RunnableType,
)

# a type alias helper with factories of coroutines
AsyncLogfuncType = Callable[..., Awaitable[None]]


def async_log_factory(
    DATABASE_FILE: str, CONTEXT_DATABASE_FILE: str, logger: LoggerBase
) -> AsyncLogfuncType:
    """Create a coroutine that logs to DATABASE_FILE and
    CONTEXT_DATABASE_FILE.
    See appChat.py for an example of use of factory and coroutine."""

    def fmat_for_csv(text: str) -> str:
        """Format text for CSV storage by escaping quotes and newlines."""

        # Replace double quotation marks with single quotation marks
        modified_text = text.replace('"', "'")
        # Replace newline characters with " | "
        modified_text = modified_text.replace('\n', ' | ')
        return modified_text

    # Unified non-blocking async logging function
    async def async_log(
        record_id: str,
        client_host: str,
        session_hash: str,
        timestamp: datetime,
        interaction_type: str,
        history: list[dict[str, str]],
        query: str = "",
        response: str = "",
        model_name: str = "",
    ) -> None:
        """
        Unified non-blocking logging function for all interaction types.
        Logs to CSV files without blocking the main async flow.

        Args:
            record_id: Unique identifier for this record
            client_host: Client IP address
            session_hash: Session identifier
            timestamp: Timestamp of the interaction
            interaction_type: Type of interaction ("MESSAGE", "USER REACTION", "USER COMMENT")
            query: Query text (or action for reactions, comment for comments)
            response: Response text (empty for reactions and comments)
            history: conversation history (empty for reactions and comments)
            model_name: Name of the model used (empty for reactions and comments)
        """
        try:
            context: list[str] = [
                b['content']
                for b in history
                if b['role'] == "context"
            ]
            message: list[str] = [
                b['content']
                for b in history
                if b['role'] == "message"
            ]
            rejection: list[str] = [
                b['content']
                for b in history
                if b['role'] == "rejection"
            ]

            # Log main interaction to messages.csv
            if message:
                interaction_type = message[0]
            if rejection:
                interaction_type = "REJECTION"
                response = rejection[0]
            with open(DATABASE_FILE, "a", encoding='utf-8') as f:
                f.write(
                    f'{record_id},{client_host},{session_hash},'
                    f'{timestamp},{len(history)},'
                    f'{model_name},{interaction_type},'
                    f'"{fmat_for_csv(query)}","{fmat_for_csv(response)}"\n'
                )

            # Log context if available (from context role in history). We also
            # record relevance of context for further monitoring.
            if context:
                # we evaluate consistency of context prior to saving
                try:
                    lmm_validator: RunnableType = create_runnable(
                        'context_validator'  # this will be a dict lookup
                    )
                    validation: str = await lmm_validator.ainvoke(
                        {
                            'query': f"{query}. {response}",
                            'context': context[0],
                        }
                    )
                    validation = validation.upper()
                except Exception as e:
                    logger.error(
                        f'Could not connect to aux model to validate context: {e}'
                    )
                    validation = "<failed>"

                with open(
                    CONTEXT_DATABASE_FILE, "a", encoding='utf-8'
                ) as f:
                    f.write(
                        f'{record_id},{validation},'
                        '"{fmat_for_csv(context[0])},NA"\n'
                    )

        except Exception as e:
            logger.error(f"Async logging failed: {e}")

    return async_log


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
