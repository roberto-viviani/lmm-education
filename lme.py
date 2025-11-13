"""CLI interface for LM mearkdown for education"""

import typer

from lmm.utils.logging import ConsoleLogger

logger = ConsoleLogger()
app = typer.Typer()


@app.command()
def create_default_config_file() -> None:
    """
    Create a default configuration file, or reset the configuration
    file to default values.
    """
    from lmm_education.config.config import create_default_config_file

    try:
        create_default_config_file()
    except Exception as e:
        logger.error(str(e))
        raise typer.Exit(1)


@app.command()
def scan_messages(
    sourcefile: str = typer.Argument(
        ..., help="Markdown file to process"
    ),
    max_size_mb: float = typer.Option(
        50,
        "--max-size-mb",
        "-ms",
        help="Maximum size of the markdown file in MB",
    ),
    warn_size_mb: float = typer.Option(
        10,
        "--warn-size-mb",
        "-ws",
        help="Warning size of the markdown file in MB",
    ),
) -> None:
    """
    Scans the markdown document for messages to send to the language
    model, and carries out the interaction.
    """
    from lmm.scan.scan_messages import markdown_messages

    try:
        SAVE_FILE = True
        markdown_messages(
            sourcefile,
            SAVE_FILE,
            max_size_mb=max_size_mb,
            warn_size_mb=warn_size_mb,
            logger=logger,
        )
    except Exception as e:
        logger.error(str(e))
        raise typer.Exit(1)


@app.command()
def scan_clear_messages(
    sourcefile: str = typer.Argument(
        ..., help="Markdown file to process"
    ),
    key: str = typer.Option(
        None,
        "--key",
        "-k",
        help="Key to clear in the markdown file",
    ),
) -> None:
    """
    Clears all messages created during an interaction with a language
    model, such as a chat or the outcome of a query, or properties
    specified by a key
    """
    from lmm.scan.scan_messages import markdown_clear_messages

    SAVE_FILE = True
    try:
        if key is None:  # type: ignore
            markdown_clear_messages(
                sourcefile, None, SAVE_FILE, logger=logger
            )
        else:
            if key:
                markdown_clear_messages(
                    sourcefile, [key], SAVE_FILE, logger=logger
                )
            else:
                logger.error("Invalid or empty key")

    except Exception as e:
        logger.error(str(e))
        raise typer.Exit(1)


@app.command()
def scan_rag(
    sourcefile: str = typer.Argument(
        ..., help="Markdown file to process"
    ),
    titles: bool = typer.Option(
        None, "--titles", "-t", help="Add titles to the markdown file"
    ),
    questions: bool = typer.Option(
        None,
        "--questions",
        "-q",
        help="Add questions to the markdown file",
    ),
    questions_threshold: int = typer.Option(
        None,
        "--questions-threshold",
        "-qt",
        help="Threshold for questions",
    ),
    summaries: bool = typer.Option(
        None,
        "--summaries",
        "-s",
        help="Add summaries to the markdown file",
    ),
    summary_threshold: int = typer.Option(
        None,
        "--summary-threshold",
        "-st",
        help="Threshold for summaries",
    ),
    remove_messages: bool = typer.Option(
        None,
        "--remove-messages",
        "-rm",
        help="Remove messages from the markdown file",
    ),
    max_size_mb: float = typer.Option(
        50,
        "--max-size-mb",
        "-ms",
        help="Maximum size of the markdown file in MB",
    ),
    warn_size_mb: float = typer.Option(
        10,
        "--warn-size-mb",
        "-ws",
        help="Warning size of the markdown file in MB",
    ),
) -> None:
    """
    Scans the markdown file and adds information required for the
    ingestion in the vector database.
    """
    from lmm.scan.scan_rag import ScanOpts, markdown_rag

    config_dict: dict[str, bool | int | float] = {}
    if titles is not None:  # type: ignore
        config_dict['titles'] = titles
    if questions is not None:  # type: ignore
        config_dict['questions'] = questions
    if questions_threshold is not None:  # type: ignore
        config_dict['questions_threshold'] = questions_threshold
    if summaries is not None:  # type: ignore
        config_dict['summaries'] = summaries
    if summary_threshold is not None:  # type: ignore
        config_dict['summary_threshold'] = summary_threshold
    if remove_messages is not None:  # type: ignore
        config_dict['remove_messages'] = remove_messages
    scan_opts = ScanOpts(**config_dict)  # type: ignore

    try:
        SAVE_FILE = True
        markdown_rag(
            sourcefile,
            scan_opts,
            SAVE_FILE,
            max_size_mb=max_size_mb,
            warn_size_mb=warn_size_mb,
            logger=logger,
        )
    except Exception as e:
        logger.error(str(e))
        raise typer.Exit(1)


@app.command()
def ingest(
    sourcefile: str = typer.Argument(
        ..., help="Markdown file to process"
    ),
    save_files: bool = typer.Option(
        False,
        "--save-files",
        "-s",
        help="Save processed blocks to files",
    ),
    skip_ingest: bool = typer.Option(
        False,
        "--skip-ingest",
        "-si",
        help="Ingest documents into the vector database",
    ),
) -> None:
    """
    Ingests a markdown file into the vector database, after processing
     it as required by the encoding model.
    """
    from lmm_education.ingest import markdown_upload
    from lmm_education.config.config import ConfigSettings

    try:
        settings = ConfigSettings()
        markdown_upload(
            sourcefile,
            config_opts=settings,
            save_files=save_files,
            ingest=not skip_ingest,
            logger=logger,
        )
    except Exception as e:
        logger.error(str(e))
        raise typer.Exit(1)


@app.command()
def ingest_folder(
    folder: str = typer.Argument(..., help="Folder to process"),
    extensions: str = typer.Option(
        ".md;.Rmd",
        "--extensions",
        "-e",
        help="Extensions to process",
    ),
) -> None:
    """
    Ingests a folder into the vector database, after
    processing them as required by the encoding model.
    """
    from lmm.utils.ioutils import list_files_with_extensions
    from lmm_education.ingest import markdown_upload
    from lmm_education.config.config import ConfigSettings

    try:
        settings = ConfigSettings()
        save_files = True
        skip_ingest = False
        files: list[str] = list_files_with_extensions(
            folder, extensions
        )
        markdown_upload(
            files,
            config_opts=settings,
            save_files=save_files,
            ingest=not skip_ingest,
            logger=logger,
        )
    except Exception as e:
        logger.error(str(e))
        raise typer.Exit(1)


@app.command()
def querydb(
    query_text: str = typer.Argument(..., help="Query text"),
) -> None:
    """
    Carries out a query to the RAG vector database.
    """
    from lmm_education.querydb import querydb

    try:
        from rich import print
    except Exception:
        pass

    try:
        response: str = querydb(query_text, logger=logger)
        print(response)  # type: ignore
    except Exception as e:
        logger.error(str(e))
        raise typer.Exit(1)


@app.command()
def query(
    query_text: str = typer.Argument(..., help="Query text"),
    validate_content: bool = typer.Option(
        False, "--validate-content", "-vc", help="Validate content"
    ),
) -> None:
    """
    Carries out a RAG query with the language model.
    """
    from lmm_education.query import query

    try:
        # when console_print is true, query will stream the output
        # to the console itself.
        query(
            query_text,
            console_print=True,
            validate_content=validate_content,
            logger=logger,
        )
    except Exception as e:
        logger.error(str(e))
        raise typer.Exit(1)


@app.command()
def database_info() -> None:
    """
    Returns information on the database schema.
    """
    from lmm_education.stores.vector_store_qdrant_utils import (
        database_info,
    )

    try:
        from rich import print
    except Exception:
        pass

    try:
        info: dict[str, object] = database_info()
        print(info)  # type: ignore
    except Exception as e:
        logger.error(str(e))
        raise typer.Exit(1)


@app.command()
def config_info() -> None:
    """
    Returns information on the active configuration.
    """
    from lmm_education.config.config import (
        load_settings,
        serialize_settings,
        ConfigSettings,
    )

    try:
        settings: ConfigSettings | None = load_settings()
        if settings is None:
            print("ERROR: config.toml could not be loaded")  # type: ignore
            raise typer.Exit(1)
        info: str = serialize_settings(settings)
        print(info)  # type: ignore
    except Exception as e:
        logger.error(str(e))
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
