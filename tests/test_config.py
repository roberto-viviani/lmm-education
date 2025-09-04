import unittest

from lmm_education.config.config import upload_settings
import tempfile
import os

# Create a test TOML file
test_content = '''
database_source = ':memory:'
collection_name = 'test_chunks'
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


class test_read_config(unittest.TestCase):

    def test_upload_settings(self):
        # Test the upload_settings function
        settings = upload_settings(temp_file['temp_file'])
        self.assertTrue(bool(settings))
        print('Successfully loaded settings:')
        print(f'  database_source: {settings.database_source}')
        print(f'  collection_name: {settings.collection_name}')
        print(f'  questions: {settings.questions}')
        print(f'  summaries: {settings.summaries}')
        print('✓ upload_settings function works correctly!')


if __name__ == "__main__":
    unittest.main()
