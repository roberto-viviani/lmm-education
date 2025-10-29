"""
Unit tests for the find_dictionary_differences function in lmm_education/config/utils.py
"""

import json
import unittest
from lmm_education.config.utils import (
    find_dictionary_differences,
    format_difference_report,
)


class TestFindDictionaryDifferences(unittest.TestCase):
    """Test cases for find_dictionary_differences function"""

    def test_identical_dictionaries(self):
        """Test that identical dictionaries return no differences"""
        dict_a = {"key1": "value1", "key2": 42}
        json_b = json.dumps({"key1": "value1", "key2": 42})

        result = find_dictionary_differences(dict_a, json_b)

        self.assertEqual(result, {})

    def test_simple_value_difference(self):
        """Test detection of simple value differences at top level"""
        dict_a = {"key1": "value1", "key2": 42}
        json_b = json.dumps({"key1": "value1", "key2": 100})

        result = find_dictionary_differences(dict_a, json_b)

        self.assertIn("key2", result)
        self.assertEqual(result["key2"]["dict_a"], 42)
        self.assertEqual(result["key2"]["dict_b"], 100)

    def test_missing_key_in_dict_b(self):
        """Test detection of keys present in dict_a but missing in dict_b"""
        dict_a = {"key1": "value1", "key2": 42, "extra_key": "extra"}
        json_b = json.dumps({"key1": "value1", "key2": 42})

        result = find_dictionary_differences(dict_a, json_b)

        self.assertIn("extra_key", result)
        self.assertEqual(result["extra_key"]["dict_a"], "extra")
        self.assertIsNone(result["extra_key"]["dict_b"])

    def test_missing_key_in_dict_a(self):
        """Test detection of keys missing in dict_a but present in dict_b"""
        dict_a = {"key1": "value1"}
        json_b = json.dumps(
            {"key1": "value1", "new_key": "new_value"}
        )

        result = find_dictionary_differences(dict_a, json_b)

        self.assertIn("new_key", result)
        self.assertIsNone(result["new_key"]["dict_a"])
        self.assertEqual(result["new_key"]["dict_b"], "new_value")

    def test_nested_dictionary_differences(self):
        """Test detection of differences in nested dictionaries"""
        dict_a = {"settings": {"theme": "dark", "font_size": 12}}
        json_b = json.dumps(
            {"settings": {"theme": "light", "font_size": 12}}
        )

        result = find_dictionary_differences(dict_a, json_b)

        self.assertIn("settings.theme", result)
        self.assertEqual(result["settings.theme"]["dict_a"], "dark")
        self.assertEqual(result["settings.theme"]["dict_b"], "light")
        self.assertNotIn("settings.font_size", result)

    def test_deeply_nested_differences(self):
        """Test detection of differences in deeply nested structures"""
        dict_a = {"level1": {"level2": {"level3": {"value": "old"}}}}
        json_b = json.dumps(
            {"level1": {"level2": {"level3": {"value": "new"}}}}
        )

        result = find_dictionary_differences(dict_a, json_b)

        self.assertIn("level1.level2.level3.value", result)
        self.assertEqual(
            result["level1.level2.level3.value"]["dict_a"], "old"
        )
        self.assertEqual(
            result["level1.level2.level3.value"]["dict_b"], "new"
        )

    def test_nested_missing_keys(self):
        """Test detection of missing keys in nested structures"""
        dict_a = {
            "settings": {
                "theme": "dark",
                "extra_field": "extra_value",
            }
        }
        json_b = json.dumps(
            {
                "settings": {
                    "theme": "dark",
                    "new_setting": "new_value",
                }
            }
        )

        result = find_dictionary_differences(dict_a, json_b)

        self.assertIn("settings.extra_field", result)
        self.assertEqual(
            result["settings.extra_field"]["dict_a"], "extra_value"
        )
        self.assertIsNone(result["settings.extra_field"]["dict_b"])

        self.assertIn("settings.new_setting", result)
        self.assertIsNone(result["settings.new_setting"]["dict_a"])
        self.assertEqual(
            result["settings.new_setting"]["dict_b"], "new_value"
        )

    def test_invalid_json(self):
        """Test handling of invalid JSON input"""
        dict_a = {"key": "value"}
        invalid_json = '{"key": "value",'  # Missing closing brace

        result = find_dictionary_differences(dict_a, invalid_json)

        self.assertIn("error", result)
        self.assertIn("Invalid JSON", result["error"])

    def test_empty_dictionaries(self):
        """Test comparison of empty dictionaries"""
        dict_a = {}
        json_b = json.dumps({})

        result = find_dictionary_differences(dict_a, json_b)

        self.assertEqual(result, {})

    def test_empty_vs_non_empty(self):
        """Test comparison of empty dict with non-empty dict"""
        dict_a = {}
        json_b = json.dumps({"key": "value"})

        result = find_dictionary_differences(dict_a, json_b)

        self.assertIn("key", result)
        self.assertIsNone(result["key"]["dict_a"])
        self.assertEqual(result["key"]["dict_b"], "value")

    def test_different_types_same_key(self):
        """Test when same key has different types in both dicts"""
        dict_a = {"settings": "string_value"}
        json_b = json.dumps({"settings": {"nested": "dict"}})

        result = find_dictionary_differences(dict_a, json_b)

        self.assertIn("settings", result)
        self.assertEqual(result["settings"]["dict_a"], "string_value")
        self.assertEqual(
            result["settings"]["dict_b"], {"nested": "dict"}
        )

    def test_dict_becomes_non_dict(self):
        """Test when a nested dict in dict_a becomes a simple value in dict_b"""
        dict_a = {"config": {"nested": "value"}}
        json_b = json.dumps({"config": "simple_string"})

        result = find_dictionary_differences(dict_a, json_b)

        self.assertIn("config", result)
        self.assertEqual(
            result["config"]["dict_a"], {"nested": "value"}
        )
        self.assertEqual(result["config"]["dict_b"], "simple_string")

    def test_list_values(self):
        """Test comparison of list values"""
        dict_a = {"ports": [80, 443], "items": [1, 2, 3]}
        json_b = json.dumps({"ports": [80, 443], "items": [1, 2, 4]})

        result = find_dictionary_differences(dict_a, json_b)

        # Lists with different values should be detected
        self.assertIn("items", result)
        self.assertEqual(result["items"]["dict_a"], [1, 2, 3])
        self.assertEqual(result["items"]["dict_b"], [1, 2, 4])

        # Identical lists should not appear in differences
        self.assertNotIn("ports", result)

    def test_boolean_values(self):
        """Test comparison of boolean values"""
        dict_a = {"enabled": True, "debug": False}
        json_b = json.dumps({"enabled": False, "debug": False})

        result = find_dictionary_differences(dict_a, json_b)

        self.assertIn("enabled", result)
        self.assertEqual(result["enabled"]["dict_a"], True)
        self.assertEqual(result["enabled"]["dict_b"], False)
        self.assertNotIn("debug", result)

    def test_null_values(self):
        """Test comparison involving null/None values"""
        dict_a = {"field1": None, "field2": "value"}
        json_b = json.dumps({"field1": "not_null", "field2": None})

        result = find_dictionary_differences(dict_a, json_b)

        self.assertIn("field1", result)
        self.assertIsNone(result["field1"]["dict_a"])
        self.assertEqual(result["field1"]["dict_b"], "not_null")

        self.assertIn("field2", result)
        self.assertEqual(result["field2"]["dict_a"], "value")
        self.assertIsNone(result["field2"]["dict_b"])

    def test_complex_example_from_code(self):
        """Test with the complex example from the source code"""
        config_a = {
            "id": 100,
            "settings": {
                "theme": "dark",
                "font_size": 12,
                "features": {"logging": True, "caching": False},
            },
            "metadata": "initial",
            "ports": [80, 443],
        }

        config_b_json = """
        {
            "id": 100,
            "settings": {
                "theme": "light",             
                "features": {"logging": true, "caching": true, "monitoring": false},
                "compression": true
            },
            "ports": [80, 443],
            "new_field": "test"
        }
        """

        result = find_dictionary_differences(config_a, config_b_json)

        # Check expected differences
        self.assertIn("settings.theme", result)
        self.assertEqual(result["settings.theme"]["dict_a"], "dark")
        self.assertEqual(result["settings.theme"]["dict_b"], "light")

        self.assertIn("settings.font_size", result)
        self.assertEqual(result["settings.font_size"]["dict_a"], 12)
        self.assertIsNone(result["settings.font_size"]["dict_b"])

        self.assertIn("settings.features.caching", result)
        self.assertEqual(
            result["settings.features.caching"]["dict_a"], False
        )
        self.assertEqual(
            result["settings.features.caching"]["dict_b"], True
        )

        self.assertIn("settings.features.monitoring", result)
        self.assertIsNone(
            result["settings.features.monitoring"]["dict_a"]
        )
        self.assertEqual(
            result["settings.features.monitoring"]["dict_b"], False
        )

        self.assertIn("metadata", result)
        self.assertEqual(result["metadata"]["dict_a"], "initial")
        self.assertIsNone(result["metadata"]["dict_b"])

        self.assertIn("new_field", result)
        self.assertIsNone(result["new_field"]["dict_a"])
        self.assertEqual(result["new_field"]["dict_b"], "test")


class TestFormatDifferenceReport(unittest.TestCase):
    """Test cases for format_difference_report function"""

    def test_format_no_differences(self):
        """Test formatting when there are no differences"""
        differences = {}
        report = format_difference_report(differences)

        self.assertIn("identical", report.lower())

    def test_format_value_change(self):
        """Test formatting of value changes"""
        differences = {"key1": {"dict_a": "old", "dict_b": "new"}}
        report = format_difference_report(differences)

        self.assertIn("Value Change", report)
        self.assertIn("key1", report)
        self.assertIn("old", report)
        self.assertIn("new", report)

    def test_format_extra_field(self):
        """Test formatting of extra fields (in dict_a, not in dict_b)"""
        differences = {"extra": {"dict_a": "value", "dict_b": None}}
        report = format_difference_report(differences)

        self.assertIn("Extra Field", report)
        self.assertIn("extra", report)

    def test_format_missing_field(self):
        """Test formatting of missing fields (not in dict_a, in dict_b)"""
        differences = {"missing": {"dict_a": None, "dict_b": "value"}}
        report = format_difference_report(differences)

        self.assertIn("Missing Field", report)
        self.assertIn("missing", report)

    def test_format_error(self):
        """Test formatting of error results"""
        error_result = {"error": "Invalid JSON string"}
        report = format_difference_report(error_result)

        self.assertIn("Error", report)
        self.assertIn("Invalid JSON", report)

    def test_format_multiple_differences(self):
        """Test formatting with multiple differences"""
        differences = {
            "key1": {"dict_a": "old", "dict_b": "new"},
            "key2": {"dict_a": "extra", "dict_b": None},
            "key3": {"dict_a": None, "dict_b": "missing"},
        }
        report = format_difference_report(differences)

        self.assertIn("key1", report)
        self.assertIn("key2", report)
        self.assertIn("key3", report)
        self.assertIn("Value Change", report)
        self.assertIn("Extra Field", report)
        self.assertIn("Missing Field", report)


if __name__ == '__main__':
    unittest.main()
