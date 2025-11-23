"""
Video file validation utilities for videocast application.
Provides runtime validation of video files during playback.
"""

from pathlib import Path
from typing import Literal

from lmm_education.webcast.error_handling import ErrorType


VideoLoadResult = Literal[
    "success", "not_found", "corrupted", "permission", "unknown"
]


def validate_video_file(
    video_path: str | Path,
) -> tuple[VideoLoadResult, str]:
    """
    Validate that a video file exists and is accessible.

    Args:
        video_path: Path to the video file

    Returns:
        Tuple of (result_status, error_message)
    """
    video_path = Path(video_path)

    # Check if file exists
    if not video_path.exists():
        return (
            "not_found",
            f"Video file not found: {video_path.name}",
        )

    # Check if it's a file (not directory)
    if not video_path.is_file():
        return ("corrupted", f"Path is not a file: {video_path.name}")

    # Check file size
    try:
        file_size = video_path.stat().st_size
        if file_size == 0:
            return (
                "corrupted",
                f"Video file is empty: {video_path.name}",
            )

        # Check if file is too small to be valid video (less than 1KB)
        if file_size < 1024:
            return (
                "corrupted",
                f"Video file too small: {video_path.name}",
            )

    except PermissionError:
        return ("permission", f"Permission denied: {video_path.name}")
    except Exception as e:
        return ("unknown", f"Error accessing file: {str(e)}")

    # Check file extension
    valid_extensions = {'.mp4', '.webm', '.ogg'}
    if video_path.suffix.lower() not in valid_extensions:
        return (
            "corrupted",
            f"Unsupported video format: {video_path.suffix}",
        )

    # Check if file is readable
    try:
        with open(video_path, 'rb') as f:
            # Try to read first few bytes
            header = f.read(12)
            if not header:
                return (
                    "corrupted",
                    f"Cannot read video file: {video_path.name}",
                )
    except PermissionError:
        return (
            "permission",
            f"Cannot read video file: {video_path.name}",
        )
    except Exception as e:
        return ("unknown", f"Error reading file: {str(e)}")

    return ("success", "")


def map_validation_to_error_type(
    result: VideoLoadResult,
) -> ErrorType:
    """
    Map video validation result to error type for placeholder generation.

    Args:
        result: Validation result status

    Returns:
        ErrorType for placeholder generation
    """
    mapping = {
        "not_found": "missing",
        "corrupted": "corrupted",
        "permission": "permission",
        "unknown": "unknown",
    }
    return mapping.get(result, "unknown")  # type: ignore
