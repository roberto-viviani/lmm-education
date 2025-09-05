"""
This module provides the configuration of the project, i.e. the
configuable ways in which we will extract properties from our markdown
documents, so that one can experiment with different options.

The configuation options are the following:

    storage: one of
        ':memory:'
        LocalStorage(folder = "./storage")
        RemoteSource(url = "1.1.1.127", port = 21465)
    collection_name: the collection to use for ingestion
    encoding_model (enum EncodingModel): the model used for encoding
        (what input is used to generate the embedding vectors)
    questions (bool): create questions for the textual content for
        each markdown heading (note: existing questions in the
        metadata before the heading will be used if present). (note:
        running titles will always be added); default False
    summaries (bool): summarize textual content under each heading
        while including the summaries of sub-headings (note: existing
        summaries will be used, if the text was not changed since the
        time of the summary generation); default False
    pool_threshold (int): pool the text under each heading prior to
        chunking. Possible values: 0 (do not pool), -1 (pool all text
        under the heading together, positive number: pool text chunks
        under a heading together unless the numer of words in the
        pooled text crosses the threshold). Note: equation and code
        chunks are pooled with surrounding text irrespective of the
        option chosen here; default 0 (do not pool)
    companion_collection (bool): create a companion collection
        containing the pooled text prior to chunking. The companion
        collection supports group_by queries returning the pooled text
        instead of the text used for embedding; default False
    companion_collection_name (string): the name of the companion
        collection
    text_splitter: the splitter class that will be used to split the
        text into chunks (note: chunking takes place after pooling)

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

Here, 'annotations' are the titles and questions added to the markdown

"""

from pathlib import Path
from pydantic import (
    BaseModel,
    Field,
    HttpUrl,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from typing import Literal, Self

# LM markdown
from lmm.config.config import export_settings
from lmm_education.stores import EncodingModel


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
        default='none', description="Text splitter name"
    )
    threshold: int = Field(
        default=125,
        gt=0,
        description="Threshold for text splitter."
        + " Ignored if the splitter value is 'none'",
    )


# The configurations settings are specified here, read from a
# config file.
DEFAULT_CONFIG_FILE = "ConfigEducation.toml" # fmt: skip


# By inheriting from BaseSettings, we add the functionality to read
# these specifications from a config file.
class ConfigSettings(BaseSettings):
    storage: DatabaseSource = Field(
        ..., description="The vector database local or remote source"
    )
    collection_name: str = Field(
        default="chunks",
        min_length=1,
        description="The name of the collection used in the database "
        + "to store the chunks of text that will be retrieved by "
        + "similarity with the question of the user.",
    )
    encoding_model: EncodingModel = Field(
        default=EncodingModel.CONTENT,
        description="How the chunk propoerties are being encoded. "
        + "Encoding options that are availble include hybrid dense+"
        + "sparse embeddings and multivector embeddings.",
    )
    questions: bool = Field(
        default=True,
        description="Annotate text with questions to aid retrieval",
    )
    summaries: bool = Field(
        default=True,
        description="Add summaries as chunks to aid retrieval",
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
    text_splitter: TextSplitters = Field(
        default=TextSplitters(),
        description="Provide the text splitter name to split "
        + "text into chunks. The default uses the text blocks of"
        + "the markdown as the chunks (no additional splitting).",
    )

    @model_validator(mode='after')
    def validate_comp_coll_name(self) -> Self:
        if self.text_splitter.splitter not in Splitter.__args__:
            raise ValueError(
                f"Invalid splitter: {self.text_splitter.splitter}\n"
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


# The config settings are read from the config file, if it is present.
# If not, a config file is created with default values.
def create_default_config_file(
    file_path: str | Path | None = None,
) -> None:
    """Create a default settings file.

    Args:
        settings: Custom settings object (defaults to Settings())
        file_path: Target file path (defaults to config.toml)

    Raises:
        ImportError: If tomlkit is not available
        OSError: If file cannot be written
        ValueError: If settings cannot be serialized

    Example:
        ```python
        # Creates config.toml in base folder with default values
        create_default_settings_file()

        # Creates custom config file
        create_default_settings_file(file_path="custom_config.toml")
        ```
    """
    if file_path is None:
        file_path = DEFAULT_CONFIG_FILE

    file_path = Path(file_path)

    if file_path.exists():
        # otherwise, it will be read in
        file_path.unlink()

    settings = ConfigSettings(storage=":memory:")

    export_settings(settings, file_path)


def load_settings(file_name: str | Path) -> ConfigSettings:
    """Load and return a ConfigSettings object from the specified file.

    Args:
        file_name: Path to the configuration file (TOML format)

    Returns:
        ConfigSettings: The loaded configuration settings object

    Raises:
        FileNotFoundError: If the specified file does not exist
        ValueError: If the file contains invalid configuration data
        Exception: For other file reading or parsing errors

    Example:
        ```python
        # Load settings from a custom config file
        settings = upload_settings("my_config.toml")

        # Load settings using Path object
        from pathlib import Path
        settings = upload_settings(Path("configs/custom.toml"))
        ```
    """
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

    except Exception as e:
        raise ValueError(
            f"Error loading configuration from {file_path}: {str(e)}"
        ) from e
