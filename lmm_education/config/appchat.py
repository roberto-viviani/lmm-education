from pathlib import Path
from typing import Any
from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    PydanticBaseSettingsSource,
    TomlConfigSettingsSource,
)

from .config import ServerSettings

CHAT_CONFIG_FILE: str = "appchat.toml"


class ChatSettings(BaseSettings):

    model_config = SettingsConfigDict(
        toml_file=CHAT_CONFIG_FILE,
        env_prefix="LMMEDU_",  # Uppercase for environment variables
        frozen=True,
        validate_assignment=True,
        extra='forbid',  # Prevent unexpected fields
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
        default="Please ask a question about the course."
    )
    MSG_WRONG_CONTENT: str = Field(
        default="I can only answer questions about the course."
    )
    MSG_LONG_QUERY: str = Field(
        default="Your question is too long. Please ask a shorter question."
    )
    MSG_ERROR_QUERY: str = Field(
        default="I am sorry, due to an error I cannot answer this question. Please report the error."
    )

    SYSTEM_MESSAGE: str = Field(
        default="""
You are a university tutor teaching undergraduates in a statistics course 
that uses R to fit models, explaining background and guiding understanding. 
"""
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

    server: ServerSettings = Field(
        default_factory=ServerSettings,
        description="Server configuration",
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

    def __init__(self, **data: Any) -> None:
        """
        Initialize ChatSettings with file existence verification.
        Prints a warning message on the console if the file does not exist.
        """
        config_path = Path(CHAT_CONFIG_FILE)
        if not config_path.exists():
            print(
                f"Configuration file not found: {config_path.absolute()}\n"
                "Returning a default configuration object."
            )
        super().__init__(**data)


def create_default_config_file(
    file_path: str | Path | None = None,
    settings: BaseSettings | None = None,
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
    from lmm.config.config import export_settings

    if file_path is None:
        file_path = CHAT_CONFIG_FILE

    file_path = Path(file_path)

    if file_path.exists():
        # otherwise, it will be read in
        print("Deleting old configuration file...")
        file_path.unlink()

    if settings is None:
        settings = ChatSettings()

    print("Saving new config file: " + str(file_path))
    export_settings(settings, file_path)
