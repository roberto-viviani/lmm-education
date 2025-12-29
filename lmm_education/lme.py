import asyncio
import sys
import argparse
import shlex
from langchain_core.embeddings import Embeddings
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.shortcuts import set_title

from lmm.utils.logging import LoggerBase, ConsoleLogger
from qdrant_client import AsyncQdrantClient
from lmm_education.config.config import ConfigSettings, load_settings
from lmm_education.stores.vector_store_qdrant_context import (
    global_async_client_from_config,
    global_async_clients_close,
)


# pyright: reportUnknownVariableType=false

# -- Engine --


class LMMEngine:
    """
    Manages the application state and resources for the LMM Education CLI.
    """

    def __init__(self):
        self.logger: LoggerBase = ConsoleLogger()
        config: ConfigSettings | None = load_settings(
            logger=self.logger
        )
        if config is None:
            print("Could not initialize CLI.")
        self.config: ConfigSettings | None = config

    async def initialize(self) -> bool:
        """
        Initializes the engine, loading configuration and setting up logging.
        """

        # load an embedding to load transformers library. the
        # embeddings variable is cached in the global dictionary
        # and the local reference is not used here.
        from lmm.language_models.langchain.runnables import (
            create_embeddings,
        )

        embeddings: Embeddings = create_embeddings()  # type: ignore # noqa

        print(" [Engine] Configuration loaded.")
        return True

    async def shutdown(self):
        """
        Cleans up resources, specifically closing database connections.
        """
        if self.logger:
            self.logger.info("Shutting down engine...")

        # Close all async qdrant clients
        try:
            global_async_clients_close()
            print(" [Engine] Resources released.")
        except Exception as e:
            print(f"Could not close Qdrant clients: {e}")
            return


# -- Argument Parsing --


class ArgumentError(Exception):
    def __init__(self, message: str):
        self.message = message


class ArgumentHelp(Exception):
    pass


class SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str):
        raise ArgumentError(message)

    def exit(self, status: int = 0, message: str | None = None):
        if message:
            print(message)
        if status == 0:
            raise ArgumentHelp()
        else:
            raise ArgumentError(f"Exit status {status}")


# -- Command Handlers --


async def cmd_scan_messages(engine: LMMEngine, args: list[str]):
    parser = SafeArgumentParser(
        prog="scan_messages",
        description="Scans markdown for messages.",
    )
    parser.add_argument("sourcefile", help="Markdown file to process")
    parser.add_argument(
        "--max_size_mb",
        type=float,
        default=50,
        help="Maximum size in MB",
    )
    parser.add_argument(
        "--warn_size_mb",
        type=float,
        default=10,
        help="Warning size in MB",
    )

    try:
        parsed = parser.parse_args(args)
    except (ArgumentError, ArgumentHelp):
        return

    from lmm.scan.scan_messages import markdown_messages

    print(f"Scanning messages in: {parsed.sourcefile}")

    def _run():
        try:
            markdown_messages(
                parsed.sourcefile,
                True,  # SAVE_FILE
                max_size_mb=parsed.max_size_mb,
                warn_size_mb=parsed.warn_size_mb,
                logger=engine.logger,
            )
        except Exception as e:
            engine.logger.error(str(e))

    await asyncio.to_thread(_run)


async def cmd_scan_clear_messages(engine: LMMEngine, args: list[str]):
    parser = SafeArgumentParser(
        prog="scan_clear_messages",
        description="Clears messages from markdown.",
    )
    parser.add_argument("sourcefile", help="Markdown file to process")
    parser.add_argument("--key", help="Key to clear", default=None)

    try:
        parsed = parser.parse_args(args)
    except (ArgumentError, ArgumentHelp):
        return

    from lmm.scan.scan_messages import markdown_clear_messages

    print(f"Clearing messages in: {parsed.sourcefile}")

    def _run():
        try:
            keys = [parsed.key] if parsed.key else None
            markdown_clear_messages(
                parsed.sourcefile, keys, True, logger=engine.logger
            )
        except Exception as e:
            engine.logger.error(str(e))

    await asyncio.to_thread(_run)


async def cmd_scan_rag(engine: LMMEngine, args: list[str]):
    parser = SafeArgumentParser(
        prog="scan_rag",
        description="Scans markdown for RAG ingestion.",
    )
    parser.add_argument("sourcefile", help="Markdown file to process")
    parser.add_argument(
        "--titles", action="store_true", help="Add titles"
    )
    parser.add_argument(
        "--questions", action="store_true", help="Add questions"
    )
    parser.add_argument(
        "--summaries", action="store_true", help="Add summaries"
    )
    parser.add_argument(
        "--questions_threshold", type=int, help="Questions threshold"
    )
    parser.add_argument(
        "--summary_threshold", type=int, help="Summary threshold"
    )
    parser.add_argument(
        "--remove_messages",
        action="store_true",
        help="Remove messages",
    )
    parser.add_argument(
        "--max_size_mb", type=float, default=50, help="Max size MB"
    )
    parser.add_argument(
        "--warn_size_mb", type=float, default=10, help="Warn size MB"
    )

    try:
        parsed = parser.parse_args(args)
    except (ArgumentError, ArgumentHelp):
        return

    from lmm.scan.scan_rag import ScanOpts, markdown_rag
    from pydantic import ValidationError
    from lmm.config.config import format_pydantic_error_message

    print(f"Scanning RAG in: {parsed.sourcefile}")

    def _run():
        try:
            if engine.config is None:
                return
            # Override RAG settings
            rag_settings = engine.config.RAG.from_instance(
                parsed.titles, parsed.questions, parsed.summaries
            )
            config_dict = {}
            config_dict['titles'] = rag_settings.titles
            config_dict['questions'] = rag_settings.questions
            if parsed.questions_threshold is not None:
                config_dict['questions_threshold'] = (
                    parsed.questions_threshold
                )
            config_dict['summaries'] = rag_settings.summaries
            if parsed.summary_threshold is not None:
                config_dict['summary_threshold'] = (
                    parsed.summary_threshold
                )
            if parsed.remove_messages:
                config_dict['remove_messages'] = (
                    parsed.remove_messages
                )

            scan_opts = ScanOpts(**config_dict)  # type: ignore

            markdown_rag(
                parsed.sourcefile,
                scan_opts,
                True,  # SAVE_FILE
                max_size_mb=parsed.max_size_mb,
                warn_size_mb=parsed.warn_size_mb,
                logger=engine.logger,
            )

        except ValidationError as e:
            errmsg = format_pydantic_error_message(str(e))
            engine.logger.error("Validation error: " + errmsg)
        except Exception as e:
            engine.logger.error(str(e))

    await asyncio.to_thread(_run)


async def cmd_scan_changed_titles(engine: LMMEngine, args: list[str]):
    parser = SafeArgumentParser(
        prog="scan_changed_titles", description="List titles changed."
    )
    parser.add_argument("sourcefile", help="Markdown file to process")

    try:
        parsed = parser.parse_args(args)
    except (ArgumentError, ArgumentHelp):
        return

    from lmm.scan.scan_rag import get_changed_titles
    from lmm.scan.scan import markdown_scan
    from lmm.markdown.parse_markdown import blocklist_haserrors

    def _run():
        blocks = markdown_scan(
            parsed.sourcefile, logger=engine.logger
        )
        if blocklist_haserrors(blocks):
            print("Errors in markdown. Fix before continuing")
            return

        titles = get_changed_titles(blocks, engine.logger)
        if not titles:
            print("No text changes.")
        else:
            for title in titles:
                print(title)

    await asyncio.to_thread(_run)


# -- Application Logic --


async def cmd_ingest(engine: LMMEngine, args: list[str]):
    parser = SafeArgumentParser(
        prog="ingest", description="Ingests markdown."
    )
    parser.add_argument("sourcefile", help="Markdown file")
    parser.add_argument(
        "--save-files",
        "-s",
        action="store_true",
        help="Save processed blocks",
    )
    parser.add_argument(
        "--skip-ingest",
        "-si",
        action="store_true",
        help="Skip ingestion",
    )

    try:
        parsed = parser.parse_args(args)
    except (ArgumentError, ArgumentHelp):
        return

    from lmm_education.ingest import amarkdown_upload

    print(f"Ingesting: {parsed.sourcefile}")
    try:
        await amarkdown_upload(
            parsed.sourcefile,
            config_opts=engine.config,
            save_files=parsed.save_files,
            ingest=not parsed.skip_ingest,
            client=global_async_client_from_config(),
            logger=engine.logger,
        )
    except Exception as e:
        engine.logger.error(str(e))


async def cmd_ingest_folder(engine: LMMEngine, args: list[str]):
    parser = SafeArgumentParser(
        prog="ingest_folder", description="Ingests folder."
    )
    parser.add_argument("folder", help="Folder to process")
    parser.add_argument(
        "--extensions", "-e", default=".md;.Rmd", help="Extensions"
    )

    try:
        parsed = parser.parse_args(args)
    except (ArgumentError, ArgumentHelp):
        return

    from lmm.utils.ioutils import list_files_with_extensions
    from lmm_education.ingest import amarkdown_upload

    try:
        files = list_files_with_extensions(
            parsed.folder, parsed.extensions
        )
        await amarkdown_upload(
            files,
            config_opts=engine.config,
            save_files=True,
            ingest=True,
            client=global_async_client_from_config(),
            logger=engine.logger,
        )
    except Exception as e:
        engine.logger.error(str(e))


async def cmd_query(engine: LMMEngine, args: list[str]):
    parser = SafeArgumentParser(
        prog="query", description="RAG Query."
    )
    parser.add_argument("query_text", nargs="+", help="Query text")
    parser.add_argument("--model", "-m", help="Language model")
    parser.add_argument(
        "--temperature", "-t", type=float, help="Temperature"
    )
    parser.add_argument(
        "--max_tokens", "-mt", type=int, help="Max tokens"
    )
    parser.add_argument(
        "--max_retries", "-mr", type=int, help="Max retries"
    )
    parser.add_argument(
        "--timeout", "-to", type=float, help="Timeout"
    )
    parser.add_argument(
        "--validate-content",
        "-vc",
        action="store_true",
        help="Validate content",
    )
    parser.add_argument(
        "--print-context",
        "-pc",
        action="store_true",
        help="Print context",
    )

    try:
        parsed = parser.parse_args(args)
    except (ArgumentError, ArgumentHelp):
        return

    query_text = " ".join(parsed.query_text)

    from lmm_education.query import aquery
    from lmm_education.config.appchat import ChatSettings
    from lmm.config.config import LanguageModelSettings
    from pydantic import ValidationError
    from lmm.config.config import format_pydantic_error_message

    try:
        chat_settings = ChatSettings()
    except Exception as e:
        engine.logger.error(f"Could not load chat settings: {e}")
        return

    model_settings = None
    opts = {}
    if parsed.temperature is not None:
        opts['temperature'] = parsed.temperature
    if parsed.max_tokens is not None:
        opts['max_tokens'] = parsed.max_tokens
    if parsed.max_retries is not None:
        opts['max_retries'] = parsed.max_retries
    if parsed.timeout is not None:
        opts['timeout'] = parsed.timeout

    try:
        if engine.config is None:
            return
        if parsed.model:
            match parsed.model:
                case 'major':
                    model_settings = (
                        engine.config.major.from_instance(**opts)  # type: ignore
                        if opts
                        else engine.config.major
                    )
                case 'minor':
                    model_settings = (
                        engine.config.minor.from_instance(**opts)  # type: ignore
                        if opts
                        else engine.config.minor
                    )
                case 'aux':
                    model_settings = (
                        engine.config.aux.from_instance(**opts)  # type: ignore
                        if opts
                        else engine.config.aux
                    )
                case _:
                    opts['model'] = parsed.model
                    model_settings = LanguageModelSettings(**opts)  # type: ignore
        elif opts:
            model_settings = engine.config.major.from_instance(**opts)  # type: ignore

    except ValidationError as e:
        engine.logger.error(
            "Validation error: "
            + format_pydantic_error_message(str(e))
        )
        return
    except Exception as e:
        engine.logger.error(f"Invalid model settings: {e}")
        return

    print("Thinking...")
    try:
        async for chunk in aquery(
            query_text,
            model_settings=model_settings,
            chat_settings=chat_settings,
            print_context=parsed.print_context,
            validate_content=parsed.validate_content,
            client=global_async_client_from_config(),
            logger=engine.logger,
        ):
            print(chunk, end="", flush=True)
        print()
    except Exception as e:
        engine.logger.error(str(e))


async def cmd_querydb(engine: LMMEngine, args: list[str]):
    parser = SafeArgumentParser(
        prog="querydb", description="Search vector database."
    )
    parser.add_argument("query_text", nargs="+", help="Query text")

    try:
        parsed = parser.parse_args(args)
    except (ArgumentError, ArgumentHelp):
        return

    query_text = " ".join(parsed.query_text)
    from lmm_education.querydb import aquerydb

    try:
        response: str = await aquerydb(
            query_text,
            client=global_async_client_from_config(),
            logger=engine.logger,
        )
        print(response)
    except Exception as e:
        engine.logger.error(str(e))


async def cmd_property_values(engine: LMMEngine, args: list[str]):
    parser = SafeArgumentParser(
        prog="property_values", description="Get property values."
    )
    parser.add_argument("property", help="Property name")
    parser.add_argument(
        "collection", nargs="?", default=None, help="Collection name"
    )

    try:
        parsed = parser.parse_args(args)
    except (ArgumentError, ArgumentHelp):
        return

    from lmm_education.stores.vector_store_qdrant_utils import (
        alist_property_values,
    )
    from lmm_education.stores.vector_store_qdrant_context import (
        global_async_client_from_config,
    )

    try:
        if engine.config is None:
            return
        collection = parsed.collection
        if collection is None:
            if engine.config.database.companion_collection:
                collection = (
                    engine.config.database.companion_collection
                )
            else:
                collection = engine.config.database.collection_name

        # We use global_client_from_config which caches the sync client too
        # This is safe because LMMEngine manages async clients, but
        # vector_store_qdrant_context manages both via LazyLoadingDict.
        client: AsyncQdrantClient = global_async_client_from_config(
            engine.config.storage
        )

        values: list[tuple[str, int]] = await alist_property_values(
            client,
            parsed.property,
            collection,
            logger=engine.logger,
        )
        for item in values:
            print(f"{item[0]}\t{item[1]}")
    except Exception as e:
        engine.logger.error(str(e))


async def cmd_database_info(engine: LMMEngine, args: list[str]):
    from lmm_education.stores.vector_store_qdrant_utils import (
        adatabase_info,
    )

    try:
        info: dict[str, object] = await adatabase_info(
            global_async_client_from_config(), logger=engine.logger
        )
        print(info)
    except Exception:
        return


async def cmd_config_info(engine: LMMEngine, args: list[str]):
    from lmm_education.config.config import serialize_settings

    try:
        if engine.config is None:
            print("ERROR: config could not be loaded")
            return
        info = serialize_settings(engine.config)
        print(info)
    except Exception as e:
        engine.logger.error(str(e))


async def cmd_create_default_config_file(
    engine: LMMEngine, args: list[str]
):
    from lmm_education.config.config import create_default_config_file
    from lmm_education.config.appchat import (
        create_default_config_file as create_appchat_file,
    )

    def _run():
        try:
            create_default_config_file()
            create_appchat_file()
            print("Created default config files.")
        except Exception as e:
            engine.logger.error(f"Error creating config: {e}")

    await asyncio.to_thread(_run)


async def cmd_help(engine: LMMEngine, args: list[str]):
    print("Available commands:")
    print("  scan_messages <file> [options]")
    print("  scan_clear_messages <file> [options]")
    print("  scan_rag <file> [options]")
    print("  scan_changed_titles <file>")
    print("  ingest <file> [options]")
    print("  ingest_folder <folder> [options]")
    print("  query <text> [options]")
    print("  querydb <text>")
    print("  property_values <property> [collection]")
    print("  database_info")
    print("  config_info")
    print("  create_default_config_file")
    print("  exit")
    print("\nType '<command> --help' for specific command usage.")


async def handle_command(engine: LMMEngine, user_input: str):
    """
    Handles the user command. This function is designed to be cancellable.
    """
    user_input = user_input.strip()
    if not user_input:
        return

    try:
        parts = shlex.split(user_input)
    except ValueError as e:
        print(f"Error parsing input: {e}")
        return

    if not parts:
        return

    command = parts[0]
    args = parts[1:]

    if command == "exit":
        print("Exiting...")
        sys.exit(0)

    # Registry of commands
    commands = {
        "scan_messages": cmd_scan_messages,
        "scan_clear_messages": cmd_scan_clear_messages,
        "scan_rag": cmd_scan_rag,
        "scan_changed_titles": cmd_scan_changed_titles,
        "ingest": cmd_ingest,
        "ingest_folder": cmd_ingest_folder,
        "query": cmd_query,
        "querydb": cmd_querydb,
        "property_values": cmd_property_values,
        "database_info": cmd_database_info,
        "config_info": cmd_config_info,
        "create_default_config_file": cmd_create_default_config_file,
        "help": cmd_help,
    }

    if command in commands:
        await commands[command](engine, args)
    else:
        print(f"Unknown command: '{command}'")
        print("Type 'help' for a list of available commands.")


async def main():
    set_title("Async Transformer REPL")

    # 1. Initialize Heavy Resources (Once)
    engine = LMMEngine()
    if not await engine.initialize():
        print("Failed to initialize engine. Exiting.")
        return

    # 2. Setup Loop
    session = PromptSession()

    print(
        "\nREPL Ready. Type 'help' for commands. Ctrl-C to cancel a task. 'exit' to quit.\n"
    )

    while True:
        try:
            # We use patch_stdout so that if background threads print,
            # it doesn't mess up the prompt line.
            text: str
            with patch_stdout():
                # specific prompt await
                text = await session.prompt_async("> ")

            # 3. Create a task for the command
            # We wrap it in a task so we can cancel it specifically
            task = asyncio.create_task(handle_command(engine, text))  # type: ignore

            try:
                # Wait for the task to complete
                await task
            except KeyboardInterrupt:
                # 4. Handle Interruption
                # This acts as the "Stop Generation" signal
                print("\n [Ctrl-C detected] Stopping task...")
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass  # Task successfully cancelled
                except Exception as e:
                    print(f"Error during cancellation: {e}")

        except KeyboardInterrupt:
            # Handle Ctrl-C at the prompt level (optional: clear line or exit)
            print("\n(Ctrl-C at prompt. Type 'exit' to quit)")
            continue
        except EOFError:
            # Ctrl-D to exit
            print("Exiting...")
            await engine.shutdown()
            break


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Catch the very final exit if needed
        pass
