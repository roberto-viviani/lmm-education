"""
This module provides the configuration of the project, i.e. the
configurable ways in which we will extract properties from our markdown
documents, so that one can experiment with different options.

The configuration options are the following:

    storage: one of
        ':memory:'
        LocalStorage(folder = "./storage")  (or another folder name)
        RemoteSource(url = "1.1.1.127", port = 21465)  (or others)
    collection_name: the collection to use for ingestion
    encoding_model (enum EncodingModel): the model used for encoding
        i.e. the use of dense and sparse vectors, or hybrid models
    questions (bool): create questions for the textual content for
        each markdown heading (note: existing questions in the
        metadata before the heading will be used if present). (note:
        running titles will always be added); default False
    summaries (bool): summarize textual content under each heading
        while including the summaries of sub-headings (note: existing
        summaries will be used, if the text was not changed since the
        time of the summary generation); default False
    companion_collection (str|None): create a companion collection
        containing the pooled text prior to chunking. The companion
        collection supports group_by queries returning the pooled text
        instead of the text used for embedding; default False
    text_splitter: the splitter class that will be used to split the
        text into chunks

Encoding models (.chunks.EncodingModel):

    NONE: no encoding (used to retrieve data via UUID)
    CONTENT: Encode only textual content in dense vector
    MERGED: Encode textual content merged with metadata annotations in
        dense vectors
    MULTIVECTOR: Encode content and annotions using multivectors
    SPARSE: Sparse encoding of annotations only
    SPARSE_CONTENT: Sparse annotations, dense encoding of content
    SPARSE_MERGED: Sparse annotations, dense encoding of merged
        content and annotations
    SPARSE_MULTIVECTOR: Sparse annotations, multivector encoding of
        merged content and annotations

Here, 'annotations' are the titles and questions added to the markdown,
or other metadata properties, as established in an `AnnotationModel`
object.

"""

from pathlib import Path
from pydantic import (
    BaseModel,
    Field,
    HttpUrl,
    model_validator,
    field_validator,
    ValidationError,
)
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from typing import Literal, Self

# LM markdown
from lmm.config.config import (
    Settings as LMMSettings,
    export_settings,
    serialize_settings,
    format_pydantic_error_message,
)
from lmm.scan.chunks import EncodingModel, AnnotationModel
from lmm.utils.logging import LoggerBase, ExceptionConsoleLogger
from lmm.scan.scan_keys import QUESTIONS_KEY, TITLES_KEY

from tomllib import TOMLDecodeError

# Module-level constants
DEFAULT_CONFIG_FILE = "config.toml"
DEFAULT_PORT_RANGE = (
    1024,
    65535,
)  # Valid port range excluding system ports


# web server for chat application
class ServerSettings(BaseSettings):
    """
    Server configuration settings.

    Attributes:
        mode: one of 'local' or 'remote'
        port: port number (only if mode is 'remote')
        host: server host address (defaults to 'localhost')
    """

    mode: Literal["local", "remote"] = Field(
        default="local", description="Server deployment mode"
    )
    port: int = Field(
        default=61543,
        ge=0,
        le=65535,
        description="Server port (0 for auto-assignment)",
    )
    host: str = Field(
        default="localhost", description="Server host address"
    )

    model_config = SettingsConfigDict(frozen=True, extra='forbid')

    @field_validator('port')
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate port number is in acceptable range."""
        if v != 0 and not (
            DEFAULT_PORT_RANGE[0] <= v <= DEFAULT_PORT_RANGE[1]
        ):
            raise ValueError(
                f"Port must be 0 (auto-assign) or between "
                f"{DEFAULT_PORT_RANGE[0]} and {DEFAULT_PORT_RANGE[1]}"
            )
        return v


# In the qdrant implementation used here, the database may be located
# in local or in a remote storage. The following models help making
# sure that the specification is correct.
class LocalStorage(BaseModel):
    folder: str = Field(
        ..., min_length=1, description="Path to the vector database"
    )


class RemoteSource(BaseModel):
    url: HttpUrl = Field(..., description="URL of the remote source.")
    port: int = Field(
        ...,
        gt=0,
        lt=65536,
        description="Port number for the remote source (1-65535).",
    )


DatabaseSource = Literal[':memory:'] | LocalStorage | RemoteSource


# Here the text splitters available in the application are
# defined. The 'none' text splitter just leaves blocks as they
# are.
Splitter = Literal['none', 'default']  # fmt: skip
class TextSplitters(BaseModel):
    splitter: Splitter = Field(
        default='default',
        description="Text splitter name",
    )
    threshold: int = Field(
        default=125,
        gt=0,
        description="Threshold for text splitter."
        + " Ignored if the splitter value is 'none'",
    )


# vector database settings. Location in sturage field
class DatabaseSettings(BaseModel):

    collection_name: str = Field(
        default="chunks",
        min_length=1,
        description="The name of the collection used in the database "
        + "to store the chunks of text that will be retrieved by "
        + "similarity with the question of the user.",
    )
    companion_collection: str | None = Field(
        default='documents',
        description="Use a companion collection to store the text "
        + "from which the chunks were taken, providing the language "
        + "model with larger, context-rich input, instead of the "
        + "chunks retrieved through the questions of the user. This"
        + "strategy is a simple form of graph RAG, i.e. the use"
        + "of properties of documents to enrich context. If set to"
        + "None, no companion collection will be used; provide the "
        + "name of the collection to create one, for example "
        + "'documents'.",
    )


# RAG settings
class RAGSettings(BaseModel):

    titles: bool = Field(
        default=True,
        description="Annotate text blocks with titles to aid retrieval",
    )
    questions: bool = Field(
        default=False,
        description="Annotate text with questions to aid retrieval",
    )
    summaries: bool = Field(
        default=False,
        description="Add summaries as chunks to aid retrieval",
    )
    encoding_model: EncodingModel = Field(
        default=EncodingModel.CONTENT,
        description="How the chunk propoerties are being encoded. "
        + "Encoding options that are availble include hybrid dense+"
        + "sparse embeddings and multivector embeddings.",
    )
    annotation_model: AnnotationModel = Field(
        default=AnnotationModel(),
        description="Model to select metadata for annotations and "
        + "filtering",
    )

    def get_annotation_model(self) -> AnnotationModel:
        """Returns an annotation model that is consistent with the
        RAG settings"""
        model: AnnotationModel = self.annotation_model.model_copy()

        if self.titles:
            if not model.has_property(TITLES_KEY):
                model.add_own_properties(TITLES_KEY)
        if self.questions:
            if not model.has_property(QUESTIONS_KEY):
                model.add_inherited_properties(QUESTIONS_KEY)
        return model

    def from_instance(
        self,
        titles: bool | None = None,
        questions: bool | None = None,
        summaries: bool | None = None,
        encoding_model: EncodingModel | None = None,
        annotation_model: AnnotationModel | None = None,
    ) -> 'RAGSettings':
        """Create a new RAGSettings object with modified properties"""
        return RAGSettings(
            titles=titles if titles else self.titles,
            questions=questions if questions else self.questions,
            summaries=summaries if summaries else self.summaries,
            encoding_model=(
                encoding_model
                if encoding_model is not None
                else self.encoding_model
            ),
            annotation_model=(
                annotation_model
                if annotation_model is not None
                else self.annotation_model
            ),
        )


# By inheriting from LMMSettings, we add the functionality to read
# these specifications from a config file.
class ConfigSettings(LMMSettings):
    """
    This object reads and writes to file the configuration options.

    Attributes:
        server: the server specification for the web chat
        storage: where the database is located
        database: database settings
        RAG: generation of properties such as questions, summaries
        textSplitter: the text plitter to use to form chunks

    Note:
        The annotation model will usually want to include in the
        annotation information such as questions, that were generated
        as specified by the encoding model in the settings. To obtain
        an annotation model that is consistent with the encoding
        model, get the model through the member function
        `get_annotation_model`. Additional properties given to this
        function will include the properties as inherited properties.
    """

    server: ServerSettings = Field(
        default_factory=ServerSettings,
        description="Server configuration",
    )

    storage: DatabaseSource = Field(
        default=LocalStorage(folder="./storage"),
        description="Vector database local or remote source",
    )

    database: DatabaseSettings = Field(
        default_factory=DatabaseSettings,
        description="Vector database settings",
    )

    RAG: RAGSettings = Field(
        default_factory=RAGSettings,
        description="RAG settings",
    )

    textSplitter: TextSplitters = Field(
        default=TextSplitters(),
        description="Provide the text splitter name to split "
        + "text into chunks. The default uses the text blocks of"
        + "the markdown as the chunks (no additional splitting).",
    )

    def get_annotation_model(
        self, keys: list[str] = []
    ) -> AnnotationModel:
        """
        Returns an annotation model that is consistent with other
        ConfigSettings options. The annotation model will add to
        the model saved in the config file the appropriate keys.

        Returns:
            an annotation model object

        Note:
            use this function to add manual annotations to the
                annotation model
        """
        annotation_model: AnnotationModel = (
            self.RAG.get_annotation_model()
        )
        annotation_model.add_inherited_properties(keys)
        return annotation_model

    @model_validator(mode='after')
    def validate_comp_coll_name(self) -> Self:
        if self.textSplitter.splitter not in Splitter.__args__:
            raise ValueError(
                f"Invalid splitter: {self.textSplitter.splitter}\n"
                + " must be one of {Splitter.__args__}"
            )
        return self

    model_config = SettingsConfigDict(
        toml_file=DEFAULT_CONFIG_FILE,
        env_prefix="LMMEDU_",  # Uppercase for environment variables
        frozen=True,
        validate_assignment=True,
        extra='forbid',  # Prevent unexpected fields
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
        """Customize the order of settings sources."""
        return (
            init_settings,
            TomlConfigSettingsSource(settings_cls),
            env_settings,
        )

    def __str__(self) -> str:
        return serialize_settings(self)


# The config settings are read from the config file, if it is present.
# If not, a config file is created with default values.
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
        # Creates config.toml in base folder with default values
        create_default_config_file()

        # Creates custom config file
        create_default_config_file(file_path="custom_config.toml")
        ```
    """
    if file_path is None:
        file_path = DEFAULT_CONFIG_FILE

    file_path = Path(file_path)

    if file_path.exists():
        # otherwise, it will be read in
        file_path.unlink()

    settings = ConfigSettings()

    export_settings(settings, file_path)


def load_settings(
    *,
    file_name: str | Path | None = None,
    logger: LoggerBase = ExceptionConsoleLogger(),
) -> ConfigSettings | None:
    """Load and return a ConfigSettings object from the specified file.

    Args:
        file_name: Path to settings file (defaults to config.toml)
        logger: logger to use. Defaults to a exception-raising logger.
        This centralizes exception handling, instead of writing
        the except clauses for each instantiation of ConfigSettings().

    Returns:
        ConfigSettings: The loaded configuration settings object, or
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
    if file_name is None:
        file_name = DEFAULT_CONFIG_FILE

    file_path = Path(file_name)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {file_path}"
        )

    try:
        # Create a temporary ConfigSettings class that uses the specified file
        class TempConfigSettings(ConfigSettings):
            # Provide a default for the required field
            storage: DatabaseSource = Field(
                default=":memory:",
                description="The vector database local or remote source",
            )

            model_config = SettingsConfigDict(
                toml_file=str(file_path),
                env_prefix="LMMEDU_",
                frozen=True,
                validate_assignment=True,
                extra='forbid',
            )

        # Load and return the settings from the specified file
        return TempConfigSettings()

    except TOMLDecodeError:
        logger.error(
            "An invalid value was found in the config file "
            "(often, 'None').\nCheck that all values are numbers "
            "or strings.\n"
            "Express None as an empty string or as 'None'."
        )
        return None
    except ValidationError as e:
        logger.error(
            format_pydantic_error_message(f"Invalid settings:\n{e}")
        )
        return None
    except ValueError as e:
        logger.error(f"Invalid settings:\n{e}")
        return None
    except Exception as e:
        logger.error(f"Could not load config settings:\n{e}")
        return None


# Create a default config.toml file, if there is none.
if not Path(DEFAULT_CONFIG_FILE).exists():
    create_default_config_file()
else:
    # complement possible config.toml written by lmm
    settings_ = ConfigSettings()
    export_settings(settings_)
