from pathlib import Path
from typing import Any, Self, Literal
from pydantic import Field, BaseModel, model_validator
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    PydanticBaseSettingsSource,
    TomlConfigSettingsSource,
)

from lmm.utils.logging import LoggerBase, ExceptionConsoleLogger

from .config import ServerSettings

CHAT_CONFIG_FILE: str = "appchat.toml"


class CheckResponse(BaseModel):
    """
    Settings to check appropriateness of chat.
    """

    check_response: bool = Field(default=False)
    allowed_content: list[str] = Field(default=[])
    initial_buffer_size: int = Field(
        default=320,
        ge=120,
        le=12000,
        description="buffer size to send to model to check content",
    )

    @model_validator(mode="after")
    def validate_allowed_content(self) -> Self:
        """Validate that allowed_content is not empty when check_response is True."""
        if self.check_response and not self.allowed_content:
            raise ValueError(
                "allowed_content must not be empty when check_response is True"
            )
        if "general knowledge" in self.allowed_content:
            raise ValueError(
                "'general knowledge' cannot be included in allowed content"
            )
        return self


class ChatDatabase(BaseModel):
    """Names of databases to log queries and responses."""

    messages_database_file: str = Field(
        default="messages.csv",
        min_length=1,
        description="Database of message exchanges",
    )
    context_database_file: str = Field(
        default="queries.csv",
        min_length=1,
        description="Database for retrieved context",
    )


# Settings for the integration of history.
HistoryIntegration = Literal[
    'none', 'summary', 'context_extraction', 'rewrite'
]


class ChatSettings(BaseSettings):
    model_config = SettingsConfigDict(
        toml_file=CHAT_CONFIG_FILE,
        env_prefix="LMMEDU_",  # Uppercase for environment variables
        frozen=True,
        validate_assignment=True,
        extra="forbid",  # Prevent unexpected fields
    )

    # This is displayed on the chatbot. Change it as appropriate
    title: str = Field(default="VU Study Assistant")
    description: str = Field(
        default="""
Study assistant chatbot for VU Scientific Methods in Psychology:
Data analysis with linear models in R. 
Ask a question about the course, and the assistant will provide a 
response based on it. 
Example: "How can I fit a model with kid_score as outcome and mom_iq as predictor?" 
"""
    )
    comment: str = Field(
        default="Please leave a comment on the response of the chatbot here"
    )

    # messages
    MSG_EMPTY_QUERY: str = Field(
        default=(
            "It seems you didn't type anything in the input box... "
            "If you have questions related to "
            "linear models, their interpretation, or how to "
            "implement them in R, I am happy to help."
        )
    )
    MSG_WRONG_CONTENT: str = Field(
        default=(
            "I do not have information to answer this query"
            " as the course focuses on linear models and "
            "their use in R. If you have questions related to "
            "linear models, their interpretation, or how to "
            "implement them in R, I would be happy to help!"
        )
    )
    MSG_LONG_QUERY: str = Field(
        default="Hmm, your question is too long... Can you think"
        " of a way to make it shorter?"
    )
    MSG_ERROR_QUERY: str = Field(
        default=(
            "I am sorry, due to an error I cannot answer "
            "this question. The failure is being recorded by the system."
        )
    )

    SYSTEM_MESSAGE: str = Field(
        default=(
            "You are a university tutor teaching "
            "undergraduates in a statistics course that uses R"
            " to fit models, explaining background and guiding "
            "understanding. Limit your responses in the chat to "
            "the field of statistics and the syntax and use of "
            "the R programming language."
        )
    )

    PROMPT_TEMPLATE: str = Field(
        default="""
Please assist students by responding to their QUERY by using the provided CONTEXT.
If the CONTEXT does not provide information for your answer, integrate the CONTEXT
only for the use and syntax of R. Otherwise, reply that you do not have information 
to answer the query, as the course focuses on linear models and their use in R.

####
CONTEXT: "{context}"

####
QUERY: "{query}"


"""
    )

    max_query_word_count: int = Field(
        default=120, ge=0, description="Max word count in query"
    )

    # Technique to include history
    history_integration: HistoryIntegration = Field(
        default='context_extraction',
        description="Technique for the integration of history",
    )
    history_length: int = Field(
        default=2,
        ge=0,
        description="Number of exchanges to include in history",
    )

    # thematic control of interaction
    check_response: CheckResponse = Field(
        default_factory=CheckResponse,
        description="Check thematic appropriateness of chat",
    )

    server: ServerSettings = Field(
        default_factory=ServerSettings,
        description="Server configuration",
    )

    # database of exchanges
    chat_database: ChatDatabase = Field(
        default_factory=ChatDatabase,
        description="Database of chat exchanges",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Customize the order of settings sources to include TOML file."""
        return (
            init_settings,
            TomlConfigSettingsSource(settings_cls),
            env_settings,
        )

    def from_instance(
        self,
        *,
        title: str | None = None,
        description: str | None = None,
        comment: str | None = None,
        MSG_EMPTY_QUERY: str | None = None,
        MSG_WRONG_CONTENT: str | None = None,
        MSG_LONG_QUERY: str | None = None,
        MSG_ERROR_QUERY: str | None = None,
        SYSTEM_MESSAGE: str | None = None,
        PROMPT_TEMPLATE: str | None = None,
        max_query_word_count: int | None = None,
        check_response: CheckResponse | None = None,
        server: ServerSettings | None = None,
        chat_database: ChatDatabase | None = None,
    ) -> 'ChatSettings':
        return ChatSettings(
            title=title or self.title,
            description=description or self.description,
            comment=comment or self.comment,
            MSG_EMPTY_QUERY=MSG_EMPTY_QUERY or self.MSG_EMPTY_QUERY,
            MSG_WRONG_CONTENT=MSG_WRONG_CONTENT
            or self.MSG_WRONG_CONTENT,
            MSG_LONG_QUERY=MSG_LONG_QUERY or self.MSG_LONG_QUERY,
            MSG_ERROR_QUERY=MSG_ERROR_QUERY or self.MSG_ERROR_QUERY,
            SYSTEM_MESSAGE=SYSTEM_MESSAGE or self.SYSTEM_MESSAGE,
            PROMPT_TEMPLATE=PROMPT_TEMPLATE or self.PROMPT_TEMPLATE,
            max_query_word_count=(
                max_query_word_count or self.max_query_word_count
            ),
            check_response=check_response or self.check_response,
            server=server or self.server,
            chat_database=chat_database or self.chat_database,
        )

    def __init__(self, **data: Any) -> None:
        """
        Initialize ChatSettings with file existence verification.
        Prints a warning message on the console if the file does not exist.
        """
        config_path = Path(CHAT_CONFIG_FILE)
        if not config_path.exists():
            print(
                f"Configuration file not found: {config_path.absolute()}\n"
                "Creating a default configuration object."
            )
        super().__init__(**data)


def create_default_config_file(
    file_path: str | Path | None = None,
) -> None:
    """Create a default settings file.

    Args:
        file_path: config file (defaults to config.toml)

    Raises:
        ImportError: If tomlkit is not available
        OSError: If file cannot be written
        ValueError: If settings cannot be serialized
        ValidationError: if config.toml is invalid
        tomllib.TOMLDecodeError: if config.toml is invalid

    Example:
        ```python
        # Creates appachat.toml in base folder with default values
        from lmm_education.config.appchat import create_default_config_file
        create_default_config_file()

        # Creates custom config file
        create_default_config_file(file_path="custom_config.toml")
        ```
    """
    from lmm.config.config import create_default_config_file as _cdcf

    if file_path is None:
        file_path = CHAT_CONFIG_FILE

    file_path = Path(file_path)

    _cdcf(file_path, ChatSettings)


def load_settings(
    *,
    file_name: str | Path | None = None,
    logger: LoggerBase = ExceptionConsoleLogger(),
) -> ChatSettings | None:
    """Load and return a ChatSettings object from the specified file.

    Args:
        file_name: Path to settings file (defaults to config.toml)
        logger: logger to use. Defaults to a exception-raising logger.
        This centralizes exception handling, instead of writing
        the except clauses for each instantiation of ConfigSettings().

    Returns:
        ChatSettings: The loaded configuration settings object, or
        None if exception raised.

    Expected behaviour:
        Exceptions handled through logger, but raises exceptions in
        the default logger.

    Example:
        ```python
        # Load settings from a custom config file
        settings = load_settings("my_config.toml")

        # Load settings using Path object
        from pathlib import Path
        settings = load_settings(Path("configs/custom.toml"))
        ```

    Note:
        Use of an ExceptionConsoleLogger still requires to check that
        return value is not None to satisfy a type checker.

        ```python
        logger = ExceptionConsoleLogger()
        settings = load_settings(logger=logger)
        if settings is None:
            raise ValueError("Unreacheable code reached")
        ```

        Here, the type checker is told that settings is not None, but
        the condition is always satisfied because load_settings will
        raise an exception whenever it would be returning None.

        Contrast with the following:

        ```python
        logger = ConsoleLogger()
        settings = load_settings(logger=logger)
        if settings is None:
            raise ValueError("Could not read config.toml")
        ```
    """
    from lmm.config.config import load_settings as _load_settings

    if file_name is None:
        file_name = CHAT_CONFIG_FILE

    file_path = Path(file_name)

    return _load_settings(
        file_name=file_path,
        logger=logger,
        settings_class=ChatSettings,
    )
