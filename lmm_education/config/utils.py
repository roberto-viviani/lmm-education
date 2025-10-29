"""
A utility to check for differences between dictionaries,
and report them in a user-friendly message.
"""

# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

import json
from typing import Any, Dict, Union

# Define a specific type for the output of the recursive function to
# improve clarity
DiffResult = Dict[str, Dict[str, Any]]
ErrorResult = Dict[str, str]


def _recursive_diff(
    dict_a: Dict[str, Any], dict_b: Dict[str, Any], path: str = ""
) -> DiffResult:
    """
    Recursively compares two dictionaries and finds differences.

    Differences are reported using dot-notation for nested keys (e.g., 'parent.child').
    """
    differences: DiffResult = {}

    # Get the union of all keys from both dictionaries at the current level
    all_keys = set(dict_a.keys()) | set(dict_b.keys())

    for key in all_keys:
        full_key = f"{path}.{key}" if path else key
        value_a = dict_a.get(key)
        value_b = dict_b.get(key)

        is_in_a = key in dict_a
        is_in_b = key in dict_b

        if is_in_a and is_in_b:
            # Case 1: Key exists in both.
            is_dict_a = isinstance(value_a, dict)
            is_dict_b = isinstance(value_b, dict)

            if is_dict_a and is_dict_b:
                # If both are dictionaries, recurse
                nested_diffs = _recursive_diff(
                    value_a, value_b, full_key  #
                )
                differences.update(nested_diffs)
            elif value_a != value_b:
                # If values differ (and are not both dicts), report difference
                differences[full_key] = {
                    'dict_a': value_a,
                    'dict_b': value_b,
                }
            # Note: If values are equal (and not dicts), we do nothing.

        elif is_in_a and not is_in_b:
            # Case 2: Key only in dict_a (missing in dict_b)
            differences[full_key] = {
                'dict_a': value_a,
                'dict_b': None,
            }
        elif not is_in_a and is_in_b:
            # Case 3: Key only in dict_b (missing in dict_a)
            differences[full_key] = {
                'dict_a': None,
                'dict_b': value_b,
            }

    return differences


def find_dictionary_differences(
    dict_a: Dict[str, Any], json_b: str | Dict[str, Any]
) -> Union[DiffResult, ErrorResult]:
    """
    Compares two dictionaries (one provided as a dict, one as a JSON string)
    and returns a dictionary detailing the differences, including nested differences.
    """
    # 1. Parse the JSON string into a dictionary (dict_b)
    if isinstance(json_b, str):
        try:
            dict_b: Dict[str, Any] = json.loads(json_b)
        except json.JSONDecodeError as e:
            # Handle case where the JSON input is invalid
            return {
                "error": f"Invalid JSON string provided for second dictionary: {e}"
            }
    else:
        dict_b = json_b

    # 2. Start the recursive comparison
    return _recursive_diff(dict_a, dict_b)


def format_difference_report(
    differences: Union[DiffResult, ErrorResult],
) -> str:
    """
    Takes the output of find_dictionary_differences and formats it into
    a human-readable string report, referring to dict_b as the 'reference dictionary'.

    Args:
        differences: The output dictionary from find_dictionary_differences.

    Returns:
        A multiline string summarizing the differences.
    """
    # Check for an error result from JSON parsing
    if 'error' in differences:
        return f"Error encountered: {differences['error']}"

    # Cast the result to the expected type for iteration
    diff_result: DiffResult = differences  # type: ignore

    if not diff_result:
        return ""

    report_lines = ["--- Differences from Database Settings ---"]

    for key, data in diff_result.items():
        value_a = data.get('dict_a')
        value_b = data.get('dict_b')

        if value_a is not None and value_b is not None:
            # Value Change
            report_lines.append(
                f"Value Change for '{key}': Current setting value is '{value_a}', which differs from the database value '{value_b}'."
            )
        elif value_a is not None and value_b is None:
            # Missing in Reference (Extra field in current dict_a)
            report_lines.append(
                f"Extra Field: The setting '{key}' was given but is missing in the database settings."
            )
        elif value_a is None and value_b is not None:
            # Missing in Current (Field missing in dict_a but present in reference)
            report_lines.append(
                f"Missing Field: The setting '{key}' is missing in the current settings, but the reference databased contains '{value_b}'."
            )

    report_lines.append("------------------------------------------")
    return "\n".join(report_lines)


from lmm_education.config.config import ConfigSettings


def vector_store_settings(settings: ConfigSettings) -> dict[str, Any]:
    """Extract from a ConfigSettings object the data members that
    are relevant for the integrity of a vector store, and return
    a dictionary representation of them"""
    buffer: dict[str, Any] = settings.model_dump(mode="json")
    dd: dict[str, Any] = {
        'database': buffer['database'],
        'RAG': buffer['RAG'],
    }
    return dd


# --- Example Usage ---

# # Dictionary A (Python Dict)
# config_a = {
#     "id": 100,
#     "settings": {
#         "theme": "dark",
#         "font_size": 12,
#         "features": {"logging": True, "caching": False},
#     },
#     "metadata": "initial",
#     "ports": [80, 443],
# }

# # Dictionary B (JSON String) - This is the "Reference Dictionary"
# config_b_json = """
# {
#     "id": 100,
#     "settings": {
#         "theme": "light",
#         "features": {"logging": True, "caching": true, "monitoring": false},
#         "compression": true
#     },
#     "ports": [80, 443],
#     "new_field": "test"
# }
# """

# print("--- Step 1: Comparing Dictionaries (Raw JSON Output) ---")
# differences = find_dictionary_differences(config_a, config_b_json)
# print(json.dumps(differences, indent=2))

# print("\n--- Step 2: Generating Human-Readable Report ---")
# report = format_difference_report(differences)
# print(report)

# print("\n--- Testing with Invalid JSON (Error Report) ---")
# invalid_json = '{"key": "value",'
# error_result = find_dictionary_differences(config_a, invalid_json)
# error_report = format_difference_report(error_result)
# print(error_report)
