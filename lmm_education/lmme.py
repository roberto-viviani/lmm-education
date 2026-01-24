"""CLI interface for LM mearkdown for education"""

from qdrant_client import QdrantClient
import typer
import sys

from pydantic import ValidationError

from lmm.utils.logging import ConsoleLogger
from lmm.config.config import format_pydantic_error_message

logger = ConsoleLogger()
app = typer.Typer()


@app.command()
def terminal() -> None:
    """Opens a lmme terminal."""

    from .lme import main
    import asyncio

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Catch the very final exit if needed
        pass


@app.command()
def create_default_config_file() -> None:
    """
    Create default configuration files, or reset the configuration
    files to default values.
    """
    from lmm_education.config.config import (
        create_default_config_file as create_config_file,
    )
    from lmm_education.config.appchat import (
        create_default_config_file as create_appchat_file,
    )

    try:
        create_config_file()
        create_appchat_file()
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
    from lmm_education.config.config import (
        ConfigSettings,
        RAGSettings,
    )

    # read existing settings from config.toml
    try:
        config = ConfigSettings()
    except Exception as e:
        logger.error(f"Could not load config: {e}")
        raise typer.Exit(1)

    # override RAG settings with those given
    try:
        rag_settings: RAGSettings = config.RAG.from_instance(
            titles, questions, summaries
        )
        # override the rest
        config_dict: dict[str, bool | int | float] = {}
        config_dict['titles'] = rag_settings.titles
        config_dict['questions'] = rag_settings.questions
        if questions_threshold is not None:  # type: ignore
            config_dict['questions_threshold'] = questions_threshold
        config_dict['summaries'] = rag_settings.summaries
        if summary_threshold is not None:  # type: ignore
            config_dict['summary_threshold'] = summary_threshold
        if remove_messages is not None:  # type: ignore
            config_dict['remove_messages'] = remove_messages
        scan_opts = ScanOpts(**config_dict)  # type: ignore
    except ValidationError as e:
        errmsg: str = format_pydantic_error_message(str(e))
        errmsg = errmsg.replace("ScanOpts", "setting")
        logger.error("Validation error: " + errmsg)
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Could not load config: {e}")
        raise typer.Exit(1)

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
def scan_changed_titles(
    sourcefile: str = typer.Argument(
        ..., help="Markdown file to process"
    ),
) -> None:
    """
    List titles whose text has changed and would be processed at next scan.
    """
    from lmm.utils.logging import ConsoleLogger
    from lmm.markdown.parse_markdown import blocklist_haserrors, Block
    from lmm.scan.scan import markdown_scan
    from lmm.scan.scan_rag import get_changed_titles

    logger = ConsoleLogger()

    blocks: list[Block] = markdown_scan(sourcefile, logger=logger)
    if blocklist_haserrors(blocks):
        print("Errors in markdown. Fix before continuing")
        raise typer.Exit(0)

    titles: list[str] = get_changed_titles(blocks, logger)
    if not titles:
        print("No text changes.")
    else:
        for title in titles:
            print(title)


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

    # this is a workaround for the qdrant bug when clients
    # are closed during garbage collection. We close all
    # clients explicitly prior to exiting when this routine
    # is called from the CLI, i.e. here Python is exiting.
    if hasattr(sys, 'ps1'):
        return
    else:
        from lmm_education.stores.vector_store_qdrant_context import (
            qdrant_clients,
        )

        qdrant_clients.clear()


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
    processing the documents as required by the encoding model.
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

    # this is a workaround for the qdrant bug when clients
    # are closed during garbage collection. We close all
    # clients explicitly prior to exiting when this routine
    # is called from the CLI, i.e. here Python is exiting.
    if hasattr(sys, 'ps1'):
        return
    else:
        from lmm_education.stores.vector_store_qdrant_context import (
            qdrant_clients,
        )

        qdrant_clients.clear()


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

    # this is a workaround for the qdrant bug when clients
    # are closed during garbage collection. We close all
    # clients explicitly prior to exiting when this routine
    # is called from the CLI, i.e. here Python is exiting.
    if hasattr(sys, 'ps1'):
        return
    else:
        from lmm_education.stores.vector_store_qdrant_context import (
            qdrant_clients,
        )

        qdrant_clients.clear()


@app.command()
def query(
    query_text: str = typer.Argument(..., help="Query text"),
    model: str = typer.Option(
        None,
        "--model",
        "-m",
        help="Language model (major, minor, aux, or provider/modelname)",
    ),
    temperature: float = typer.Option(
        None,
        "--temperature",
        "-t",
        help="Controls randomness in model responses (0.0-2.0)",
    ),
    max_tokens: int = typer.Option(
        None,
        "--max-tokens",
        "-mt",
        help="Maximum number of tokens to generate",
    ),
    max_retries: int = typer.Option(
        None,
        "--max-retries",
        "-mr",
        help="Maximum number of retry attempts",
    ),
    timeout: float = typer.Option(
        None, "--timeout", "-to", help="Request timeout in seconds"
    ),
    validate_content: bool = typer.Option(
        False, "--validate-content", "-vc", help="Validate content"
    ),
    print_context: bool = typer.Option(
        False,
        "--print-context",
        "-pc",
        help="Print context given to model",
    ),
) -> None:
    """
    Carries out a RAG query with the language model.
    """
    from lmm_education.query import query
    from lmm_education.config.config import ConfigSettings
    from lmm_education.config.appchat import ChatSettings
    from lmm.config.config import LanguageModelSettings

    try:
        settings = ConfigSettings()
    except Exception as e:
        logger.error(f"Could not load settings: {e}")
        raise typer.Exit(1)

    try:
        chat_settings = ChatSettings()
    except Exception as e:
        logger.error(f"Could not load settings: {e}")
        raise typer.Exit(1)

    model_settings: LanguageModelSettings | None
    opts: dict[str, str | float | int] = {}
    if temperature is not None:  # type: ignore
        opts['temperature'] = temperature
    if max_tokens is not None:  # type: ignore
        opts['max_tokens'] = max_tokens
    if max_retries is not None:  # type: ignore
        opts['max_retries'] = max_retries
    if timeout is not None:  # type: ignore
        opts['timeout'] = timeout

    try:
        if model is not None:  # type: ignore
            match model:
                case 'major':
                    model_settings = (
                        settings.major.from_instance(**opts)  # type: ignore
                        if opts
                        else settings.major
                    )
                case 'minor':
                    model_settings = (
                        settings.minor.from_instance(**opts)  # type: ignore
                        if opts
                        else settings.minor
                    )
                case 'aux':
                    model_settings = (
                        settings.aux.from_instance(**opts)  # type: ignore
                        if opts
                        else settings.aux
                    )
                case _:
                    # try to parse, will give error if not supported
                    opts['model'] = model
                    model_settings = LanguageModelSettings(**opts)  # type: ignore

        else:
            if model is None and not opts:
                model_settings = (
                    None  # code reacheable, typer can give None
                )
            else:
                model_settings = settings.major.from_instance(**opts)
    except ValidationError as e:
        errmsg: str = format_pydantic_error_message(str(e))
        errmsg = errmsg.replace(
            "LanguageModelSettings", "language model setting"
        )
        logger.error("Validation error: " + errmsg)
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Invalid model settings: {model}\n{e}")
        raise typer.Exit(1)

    try:
        for text in query(
            query_text,
            model_settings=model_settings,
            chat_settings=chat_settings,
            validate_content=validate_content,
            print_context=print_context,
            logger=logger,
        ):
            print(text, end="", flush=True)
        print()
    except Exception as e:
        logger.error(str(e))
        raise typer.Exit(1)

    # this is a workaround for the qdrant issue when clients
    # are closed during garbage collection. We close all
    # clients explicitly prior to exiting when this routine
    # is called from the CLI, i.e. here Python is exiting.
    if hasattr(sys, 'ps1'):
        return
    else:
        from lmm_education.stores.vector_store_qdrant_context import (
            qdrant_clients,
        )

        qdrant_clients.clear()


@app.command()
def property_values(
    property: str = typer.Argument(..., help="Property to query"),
    collection: str | None = typer.Argument(
        default=None, help="Collection to query"
    ),
) -> None:
    """
    Displays the values of a property and their count in
    a collection.
    """
    from lmm_education.stores.vector_store_qdrant_utils import (
        list_property_values,
    )
    from lmm_education.stores.vector_store_qdrant_context import (
        global_client_from_config,
    )
    from lmm_education.config.config import ConfigSettings

    settings = ConfigSettings()
    if collection is None:
        try:
            if settings.database.companion_collection:
                collection = settings.database.companion_collection
            else:
                collection = settings.database.collection_name
        except Exception as e:
            logger.error(f"Could not load settings: {e}")
            raise typer.Exit(1)

    client: QdrantClient
    try:
        client = global_client_from_config(settings.storage)
    except Exception as e:
        logger.error(f"Could not load database: {e}")
        raise typer.Exit(1)
    values: list[tuple[str, int]] = list_property_values(
        client, property, collection, logger=logger
    )
    for item in values:
        print(f"{item[0]}\t{item[1]}")

    # this is a workaround for the qdrant bug when clients
    # are closed during garbage collection. We close all
    # clients explicitly prior to exiting when this routine
    # is called from the CLI, i.e. here Python is exiting.
    if hasattr(sys, 'ps1'):
        return
    else:
        from lmm_education.stores.vector_store_qdrant_context import (
            qdrant_clients,
        )

        qdrant_clients.clear()


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
