import unittest
import tempfile
import os
from pydantic import ValidationError

# pyright: reportArgumentType=false
# pyright: reportOptionalMemberAccess=false

from lmm_education.config.config import (
    load_settings,
    ConfigSettings,
    LocalStorage,
    RemoteSource,
    TextSplitters,
    RAGSettings,
    DatabaseSettings,
)
from lmm.scan.chunks import EncodingModel

# pyright: basic

# Create a test TOML file
test_content = '''
storage = ':memory:'

[database]
collection_name = 'test_chunks'

[RAG]
questions = true
summaries = false
'''

temp_file = {'temp_file': ""}


def setUpModule():
    # Write to a temporary file
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.toml', delete=False
    ) as f:
        f.write(test_content)
        temp_file['temp_file'] = f.name


def tearDownModule() -> None:
    os.unlink(temp_file['temp_file'])


class TestReadConfig(unittest.TestCase):

    def test_load_settings(self):
        # Test the load_settings function
        settings = load_settings(file_name=temp_file['temp_file'])
        self.assertTrue(bool(settings))
        self.assertEqual(settings.storage, ":memory:")
        self.assertEqual(
            settings.database.collection_name, "test_chunks"
        )
        self.assertTrue(settings.RAG.questions)
        self.assertFalse(settings.RAG.summaries)
        print('Successfully loaded settings:')
        print(f'  storage: {settings.storage}')
        print(
            f'  collection_name: {settings.database.collection_name}'
        )
        print(f'  questions: {settings.RAG.questions}')
        print(f'  summaries: {settings.RAG.summaries}')
        print('✓ load_settings function works correctly!')


class TestLocalStorageValidation(unittest.TestCase):
    """Test validation errors for LocalStorage model."""

    def test_empty_folder_raises_error(self):
        """Test that empty folder string raises ValidationError."""
        with self.assertRaises(ValidationError) as context:
            LocalStorage(folder="")

        error = context.exception
        self.assertIn(
            "String should have at least 1 character", str(error)
        )

    def test_missing_folder_raises_error(self):
        """Test that missing folder field raises ValidationError."""
        with self.assertRaises(ValidationError) as context:
            LocalStorage()

        error = context.exception
        self.assertIn("Field required", str(error))

    def test_valid_folder_succeeds(self):
        """Test that valid folder path succeeds."""
        storage = LocalStorage(folder="./test_storage")
        self.assertEqual(storage.folder, "./test_storage")


class TestRemoteSourceValidation(unittest.TestCase):
    """Test validation errors for RemoteSource model."""

    def test_invalid_url_raises_error(self):
        """Test that invalid URL raises ValidationError."""
        with self.assertRaises(ValidationError) as context:
            RemoteSource(url="not-a-url", port=8080)

        error = context.exception
        self.assertIn("URL", str(error))

    def test_missing_url_raises_error(self):
        """Test that missing URL raises ValidationError."""
        with self.assertRaises(ValidationError) as context:
            RemoteSource(port=8080)

        error = context.exception
        self.assertIn("Field required", str(error))

    def test_port_zero_raises_error(self):
        """Test that port 0 raises ValidationError."""
        with self.assertRaises(ValidationError) as context:
            RemoteSource(url="http://example.com", port=0)

        error = context.exception
        self.assertIn("Input should be greater than 0", str(error))

    def test_port_too_high_raises_error(self):
        """Test that port > 65535 raises ValidationError."""
        with self.assertRaises(ValidationError) as context:
            RemoteSource(url="http://example.com", port=65536)

        error = context.exception
        self.assertIn("Input should be less than 65536", str(error))

    def test_negative_port_raises_error(self):
        """Test that negative port raises ValidationError."""
        with self.assertRaises(ValidationError) as context:
            RemoteSource(url="http://example.com", port=-1)

        error = context.exception
        self.assertIn("Input should be greater than 0", str(error))

    def test_missing_port_raises_error(self):
        """Test that missing port raises ValidationError."""
        with self.assertRaises(ValidationError) as context:
            RemoteSource(url="http://example.com")

        error = context.exception
        self.assertIn("Field required", str(error))

    def test_valid_remote_source_succeeds(self):
        """Test that valid RemoteSource succeeds."""
        remote = RemoteSource(url="http://example.com", port=8080)
        self.assertEqual(str(remote.url), "http://example.com/")
        self.assertEqual(remote.port, 8080)


class TestTextSplittersValidation(unittest.TestCase):
    """Test validation errors for TextSplitters model."""

    def test_zero_threshold_raises_error(self):
        """Test that threshold 0 raises ValidationError."""
        with self.assertRaises(ValidationError) as context:
            TextSplitters(threshold=0)

        error = context.exception
        self.assertIn("Input should be greater than 0", str(error))

    def test_negative_threshold_raises_error(self):
        """Test that negative threshold raises ValidationError."""
        with self.assertRaises(ValidationError) as context:
            TextSplitters(threshold=-1)

        error = context.exception
        self.assertIn("Input should be greater than 0", str(error))

    def test_invalid_splitter_raises_error(self):
        """Test that invalid splitter raises ValidationError."""
        with self.assertRaises(ValidationError) as context:
            TextSplitters(splitter="invalid_splitter")

        error = context.exception
        self.assertIn(
            "Input should be 'none' or 'default'", str(error)
        )

    def test_valid_text_splitters_succeed(self):
        """Test that valid TextSplitters succeed."""
        splitter1 = TextSplitters(splitter="none", threshold=100)
        self.assertEqual(splitter1.splitter, "none")
        self.assertEqual(splitter1.threshold, 100)

        splitter2 = TextSplitters(splitter="default", threshold=200)
        self.assertEqual(splitter2.splitter, "default")
        self.assertEqual(splitter2.threshold, 200)

    def test_default_values(self):
        """Test that default values are set correctly."""
        splitter = TextSplitters()
        self.assertEqual(splitter.splitter, "default")
        self.assertEqual(splitter.threshold, 125)


class TestConfigSettingsValidation(unittest.TestCase):
    """Test validation errors for ConfigSettings model."""

    def test_empty_collection_name_raises_error(self):
        """Test that empty collection_name raises ValidationError."""
        with self.assertRaises(ValidationError) as context:
            ConfigSettings(
                storage=":memory:", database={'collection_name': ""}
            )

        error = context.exception
        self.assertIn(
            "String should have at least 1 character", str(error)
        )

    def test_invalid_encoding_model_raises_error(self):
        """Test that invalid encoding_model raises ValidationError."""
        with self.assertRaises(ValidationError) as context:
            ConfigSettings(
                storage=":memory:",
                RAG={'encoding_model': "INVALID_MODEL"},
            )

        error = context.exception
        self.assertIn("Input should be", str(error))

    def test_invalid_storage_type_raises_error(self):
        """Test that invalid storage type raises ValidationError."""
        with self.assertRaises(ValidationError) as context:
            ConfigSettings(storage="invalid_storage")

        error = context.exception
        # Should fail because it's not a valid DatabaseSource
        self.assertTrue(len(str(error)) > 0)

    def test_extra_field_raises_error(self):
        """Test that extra fields raise ValidationError due to extra='forbid'."""
        with self.assertRaises(ValidationError) as context:
            ConfigSettings(
                storage=":memory:",
                collection_name="test",
                invalid_extra_field="should_fail",
            )

        error = context.exception
        self.assertIn("Extra inputs are not permitted", str(error))

    def test_invalid_text_splitter_in_custom_validator(self):
        """Test that custom validator catches invalid splitter."""
        # This should be caught by the custom validator validate_comp_coll_name
        # However, the Pydantic validation will catch it first
        with self.assertRaises(ValidationError) as context:
            ConfigSettings(
                storage=":memory:",
                text_splitter=TextSplitters(splitter="invalid"),
            )

        error = context.exception
        self.assertIn(
            "Input should be 'none' or 'default'", str(error)
        )

    def test_valid_config_settings_succeed(self):
        """Test that valid ConfigSettings succeed."""
        # Test with memory storage
        config1 = ConfigSettings(storage=":memory:")
        self.assertEqual(config1.storage, ":memory:")
        self.assertEqual(config1.database.collection_name, "chunks")

        # Test with LocalStorage
        config2 = ConfigSettings(
            storage=LocalStorage(folder="./test"),
            database=DatabaseSettings(
                collection_name="test_chunks",
                companion_collection=None,
            ),
            RAG=RAGSettings(
                questions=False,
                summaries=False,
                encoding_model=EncodingModel.MERGED,
            ),
            textSplitter=TextSplitters(
                splitter="default", threshold=200
            ),
        )
        self.assertIsInstance(config2.storage, LocalStorage)
        self.assertEqual(
            config2.database.collection_name, "test_chunks"
        )
        self.assertEqual(
            config2.RAG.encoding_model, EncodingModel.MERGED
        )
        self.assertFalse(config2.RAG.questions)
        self.assertFalse(config2.RAG.summaries)
        self.assertIsNone(config2.database.companion_collection)

        # Test with RemoteSource
        config3 = ConfigSettings(
            storage=RemoteSource(url="http://localhost", port=6333),
            database=DatabaseSettings(
                collection_name="remote_chunks"
            ),
        )
        self.assertIsInstance(config3.storage, RemoteSource)
        self.assertEqual(
            config3.database.collection_name, "remote_chunks"
        )

    # currently, ConfigSettings has a default storage argument.
    # def test_missing_required_storage_raises_error(self):
    #     """Test that missing required storage field raises ValidationError."""

    #     import os
    #     from lmm_education.config.config import DEFAULT_CONFIG_FILE

    #     # if a config file exists, the validation will not raise an error
    #     if not os.path.exists(DEFAULT_CONFIG_FILE):
    #         with self.assertRaises(ValidationError):
    #             ConfigSettings(collection_name="TestCollection")


if __name__ == "__main__":
    unittest.main()
