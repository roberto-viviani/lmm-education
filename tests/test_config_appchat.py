"""Tests for CheckResponse validator in appchat.py configuration."""

import unittest
from pydantic import ValidationError

from lmm_education.config.appchat import CheckResponse


class TestCheckResponseValidator(unittest.TestCase):
    """Test cases for CheckResponse allowed_content field validator."""

    def test_check_response_false_with_empty_allowed_content(self):
        """Test that empty allowed_content is valid when check_response is False."""
        check = CheckResponse(
            check_response=False, allowed_content=[]
        )
        self.assertFalse(check.check_response)
        self.assertEqual(check.allowed_content, [])

    def test_check_response_false_with_non_empty_allowed_content(
        self,
    ):
        """Test that non-empty allowed_content is valid when check_response is False."""
        check = CheckResponse(
            check_response=False, allowed_content=["course"]
        )
        self.assertFalse(check.check_response)
        self.assertEqual(check.allowed_content, ["course"])

    def test_check_response_true_with_empty_allowed_content_raises_error(
        self,
    ):
        """Test that empty allowed_content raises ValueError when check_response is True."""
        with self.assertRaises(ValidationError) as context:
            CheckResponse(check_response=True, allowed_content=[])

        # Check that the error message contains the expected validation error
        error_msg = str(context.exception)
        self.assertIn("allowed_content must not be empty", error_msg)

    def test_check_response_true_with_non_empty_allowed_content(self):
        """Test that non-empty allowed_content is valid when check_response is True."""
        check = CheckResponse(
            check_response=True,
            allowed_content=["course", "homework"],
        )
        self.assertTrue(check.check_response)
        self.assertEqual(
            check.allowed_content, ["course", "homework"]
        )

    def test_default_values(self):
        """Test that default values are valid (check_response=False, allowed_content=[])."""
        check = CheckResponse()
        self.assertFalse(check.check_response)
        self.assertEqual(check.allowed_content, [])


if __name__ == "__main__":
    unittest.main()
