"""
Database interface and CSV implementation for chat logging.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from io import TextIOBase
import asyncio
import os
from typing import Self

from lmm_education.background_task_manager import schedule_task


class ChatDatabaseInterface(ABC):
    """Abstract base class for chat logging databases."""

    @abstractmethod
    async def log_message(
        self,
        record_id: str,
        client_host: str,
        session_hash: str,
        timestamp: datetime,
        message_count: int,
        model_name: str,
        interaction_type: str,
        query: str,
        response: str,
    ) -> None:
        """Log a basic chat interaction/message."""
        pass

    @abstractmethod
    async def log_message_with_context(
        self,
        record_id: str,
        client_host: str,
        session_hash: str,
        timestamp: datetime,
        message_count: int,
        model_name: str,
        interaction_type: str,
        query: str,
        response: str,
        validation: str,
        context: str,
        classification: str,
    ) -> None:
        """Log a chat interaction including retrieval context details."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close database and release resources."""
        pass

    def __enter__(self) -> Self:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore
        """Exit context manager and cleanup resources."""
        self.close()

    def schedule_message(
        self,
        record_id: str,
        client_host: str,
        session_hash: str,
        timestamp: datetime,
        message_count: int,
        model_name: str,
        interaction_type: str,
        query: str,
        response: str,
    ) -> None:
        """Log a basic chat interaction/message (async)."""
        schedule_task(
            self.log_message(
                record_id=record_id,
                client_host=client_host,
                session_hash=session_hash,
                timestamp=timestamp,
                message_count=message_count,
                model_name=model_name,
                interaction_type=interaction_type,
                query=query,
                response=response,
            )
        )

    def schedule_message_with_context(
        self,
        record_id: str,
        client_host: str,
        session_hash: str,
        timestamp: datetime,
        message_count: int,
        model_name: str,
        interaction_type: str,
        query: str,
        response: str,
        validation: str,
        context: str,
        classification: str,
    ) -> None:
        """Log a basic chat interaction/message (async)."""
        schedule_task(
            self.log_message_with_context(
                record_id=record_id,
                client_host=client_host,
                session_hash=session_hash,
                timestamp=timestamp,
                message_count=message_count,
                model_name=model_name,
                interaction_type=interaction_type,
                query=query,
                response=response,
                validation=validation,
                context=context,
                classification=classification,
            )
        )

    @classmethod
    def from_config(cls) -> 'CsvFileChatDatabase':
        from lmm_education.config.appchat import (
            ChatSettings,
            load_settings,
            ChatDatabase,
        )

        chat_settings: ChatSettings | None = load_settings()
        if chat_settings is None:
            raise ValueError("Could not load chat settings")

        db_settings: ChatDatabase = chat_settings.chat_database
        return CsvFileChatDatabase(
            db_settings.messages_database_file,
            db_settings.context_database_file,
        )


class CsvChatDatabase(ChatDatabaseInterface):
    """
    CSV implementation of the chat database interface.
    Writes to provided text streams.
    """

    def __init__(
        self,
        message_stream: TextIOBase,
        context_stream: TextIOBase | None = None,
    ):
        self.message_stream = message_stream
        self.context_stream = context_stream

    def _fmat_for_csv(self, text: str) -> str:
        """Format text for CSV storage by escaping quotes and newlines."""
        if not text:
            return ""
        # Replace double quotation marks with single quotation marks
        modified_text = text.replace('"', "'")
        # Replace newline characters with " | "
        modified_text = modified_text.replace("\n", " | ")
        return modified_text

    def _write_message_sync(
        self,
        record_id: str,
        client_host: str,
        session_hash: str,
        timestamp: datetime,
        message_count: int,
        model_name: str,
        interaction_type: str,
        query: str,
        response: str,
    ) -> None:
        """Synchronous implementation of message writing."""
        self.message_stream.write(
            f"{record_id},{client_host},{session_hash},"
            f"{timestamp},{message_count},"
            f"{model_name},{interaction_type},"
            f'"{self._fmat_for_csv(query)}",'
            f'"{self._fmat_for_csv(response)}"\n'
        )
        self.message_stream.flush()

    def _write_context_sync(
        self,
        record_id: str,
        validation: str,
        context: str,
        classification: str,
    ) -> None:
        """Synchronous implementation of context writing."""
        if self.context_stream:
            self.context_stream.write(
                f"{record_id},{validation},"
                f'"{self._fmat_for_csv(context)}",'
                f"{classification}\n"
            )
            self.context_stream.flush()

    async def log_message(
        self,
        record_id: str,
        client_host: str,
        session_hash: str,
        timestamp: datetime,
        message_count: int,
        model_name: str,
        interaction_type: str,
        query: str,
        response: str,
    ) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            self._write_message_sync,
            record_id,
            client_host,
            session_hash,
            timestamp,
            message_count,
            model_name,
            interaction_type,
            query,
            response,
        )

    async def log_message_with_context(
        self,
        record_id: str,
        client_host: str,
        session_hash: str,
        timestamp: datetime,
        message_count: int,
        model_name: str,
        interaction_type: str,
        query: str,
        response: str,
        validation: str,
        context: str,
        classification: str,
    ) -> None:
        loop = asyncio.get_running_loop()

        def _log_both() -> None:
            self._write_message_sync(
                record_id,
                client_host,
                session_hash,
                timestamp,
                message_count,
                model_name,
                interaction_type,
                query,
                response,
            )
            self._write_context_sync(
                record_id,
                validation,
                context,
                classification,
            )

        await loop.run_in_executor(None, _log_both)

    def close(self) -> None:
        """Close database resources. No-op for stream-based implementation."""
        # This implementation doesn't own the streams, so it doesn't close them.
        # The caller is responsible for managing stream lifecycle.
        pass


class CsvFileChatDatabase(CsvChatDatabase):
    """
    File-based CSV chat database that manages file lifecycle.
    Handles file creation, header initialization, and proper cleanup.
    Can be used as a context manager or with explicit close() calls.
    """

    def __init__(
        self,
        database_file: str,
        database_context_file: str | None = None,
    ):
        self.database_file = database_file
        self.database_context_file = database_context_file
        
        # Initialize headers immediately
        self._ensure_headers()
        
        # Open files immediately for use
        self._message_file = open(database_file, "a", encoding="utf-8")
        self._context_file = (
            open(database_context_file, "a", encoding="utf-8")
            if database_context_file
            else None
        )
        
        # Initialize parent class with opened streams
        super().__init__(self._message_file, self._context_file)

    def _ensure_headers(self) -> None:
        """Creates the database files with the correct headers if they don't exist."""
        if not os.path.exists(self.database_file):
            with open(self.database_file, "w", encoding="utf-8") as f:
                f.write(
                    "record_id,client_host,session_hash,timestamp,"
                    "history_length,model_name,interaction_type,"
                    "query,response\n"
                )

        if self.database_context_file and not os.path.exists(
            self.database_context_file
        ):
            with open(
                self.database_context_file, "w", encoding="utf-8"
            ) as f:
                f.write(
                    "record_id,evaluation,context,classification\n"
                )

    def close(self) -> None:
        """Explicitly close database files and release resources."""
        if hasattr(self, '_message_file'):
            try:
                self._message_file.close()
            except Exception as e:
                print(f"Error closing message log file: {e}")
        
        if hasattr(self, '_context_file') and self._context_file:
            try:
                self._context_file.close()
            except Exception as e:
                print(f"Error closing context log file: {e}")

    def __enter__(self) -> Self:
        """Enter context manager. Files are already open, just return self."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore
        """Exit context manager and cleanup resources."""
        self.close()

    def __del__(self) -> None:
        """Fallback cleanup - not guaranteed to be called."""
        try:
            self.close()
        except Exception:
            # Suppress errors during cleanup in destructor
            pass
