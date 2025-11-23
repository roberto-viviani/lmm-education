"""
Error handling utilities for videocast application.
Provides error placeholder video generation and caching.
"""

import tempfile
import hashlib
from pathlib import Path
from typing import Literal

# Try to import imageio for video generation
try:
    import imageio

    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False  # type: ignore (constant redef)


ErrorType = Literal[
    "missing", "corrupted", "permission", "codec", "unknown"
]


class ErrorPlaceholderCache:
    """
    Manages creation and caching of error placeholder videos.
    """

    def __init__(self, cache_dir: str | None = None):
        """
        Initialize the error placeholder cache.

        Args:
            cache_dir: Directory for caching error videos.
                      If None, uses system temp directory.
        """
        if cache_dir is None:
            self.cache_dir = (
                Path(tempfile.gettempdir()) / "videocast_errors"
            )
        else:
            self.cache_dir = Path(cache_dir)

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_placeholder_video(
        self, error_type: ErrorType, error_message: str
    ) -> str:
        """
        Get or create an error placeholder video.

        Args:
            error_type: Type of error (affects color scheme)
            error_message: Error message to display

        Returns:
            Path to the error placeholder video file
        """
        # Create cache key from error type and message
        cache_key = self._generate_cache_key(
            error_type, error_message
        )
        cache_file = self.cache_dir / f"{cache_key}.mp4"

        # Return cached video if it exists
        if cache_file.exists():
            return str(cache_file)

        # Generate new placeholder video
        return self._create_placeholder_video(
            cache_file, error_type, error_message
        )

    def _generate_cache_key(
        self, error_type: ErrorType, message: str
    ) -> str:
        """Generate a cache key from error type and message."""
        content = f"{error_type}:{message}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _create_placeholder_video(
        self,
        output_path: Path,
        error_type: ErrorType,
        error_message: str,
    ) -> str:
        """
        Create an error placeholder video using imageio-ffmpeg.

        Args:
            output_path: Where to save the video
            error_type: Type of error (affects color)
            error_message: Message to display

        Returns:
            Path to created video file
        """
        if not IMAGEIO_AVAILABLE:
            # Fallback: create a minimal valid video file
            return self._create_fallback_placeholder(output_path)

        # Video parameters
        width, height = 854, 480  # 480p resolution
        fps = 30
        duration = 3  # seconds
        num_frames = fps * duration

        # Color scheme based on error type
        bg_color, text_color = self._get_error_colors(error_type)

        try:
            # Create video writer
            writer = imageio.get_writer(  # type: ignore (checked above)
                str(output_path),
                fps=fps,
                codec='libx264',
                pixelformat='yuv420p',
                quality=8,
            )

            # Generate frames
            for _ in range(num_frames):
                frame = self._create_error_frame(
                    width,
                    height,
                    bg_color,
                    text_color,
                    error_type,
                    error_message,
                )
                writer.append_data(frame)

            writer.close()
            return str(output_path)

        except Exception as e:
            # If video generation fails, create fallback
            print(f"Error creating placeholder video: {e}")
            return self._create_fallback_placeholder(output_path)

    def _get_error_colors(
        self, error_type: ErrorType
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """
        Get background and text colors for error type.

        Returns:
            (bg_color, text_color) as RGB tuples
        """
        color_schemes = {
            "missing": ((20, 20, 20), (255, 100, 100)),  # Dark red bg
            "corrupted": (
                (60, 40, 20),
                (255, 220, 180),
            ),  # Dark orange bg
            "permission": (
                (40, 20, 60),
                (220, 180, 255),
            ),  # Dark purple bg
            "codec": ((20, 40, 60), (180, 220, 255)),  # Dark blue bg
            "unknown": (
                (40, 40, 40),
                (220, 220, 220),
            ),  # Dark gray bg
        }
        return color_schemes.get(error_type, color_schemes["unknown"])

    def _create_error_frame(
        self,
        width: int,
        height: int,
        bg_color: tuple[int, int, int],
        text_color: tuple[int, int, int],
        error_type: ErrorType,
        error_message: str,
    ):
        """
        Create a single frame with error message rendered as text.
        Uses PIL/Pillow for text rendering if available.
        """
        import numpy as np

        # Create frame with background color
        frame = np.full((height, width, 3), bg_color, dtype=np.uint8)

        # Add a simple border
        border_width = 10
        frame[:border_width, :] = text_color  # Top
        frame[-border_width:, :] = text_color  # Bottom
        frame[:, :border_width] = text_color  # Left
        frame[:, -border_width:] = text_color  # Right

        # Try to render text with PIL/Pillow
        try:
            from PIL import Image, ImageDraw, ImageFont

            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(frame)
            draw = ImageDraw.Draw(pil_image)

            # Try to use a reasonable font size
            font_size = 32
            try:
                # Try to use a system font
                font = ImageFont.truetype("arial.ttf", font_size)
            except Exception:
                try:
                    # Try alternative font names
                    font = ImageFont.truetype("Arial.ttf", font_size)
                except Exception:
                    # Fall back to default font
                    font = ImageFont.load_default()

            # Prepare text
            title = "VIDEO ERROR"

            # Word wrap the error message to fit width
            max_chars_per_line = 50
            words = error_message.split()
            lines: list[str] = []
            current_line = []
            current_length = 0

            for word in words:
                word_length = len(word) + 1  # +1 for space
                if current_length + word_length <= max_chars_per_line:
                    current_line.append(word)
                    current_length += word_length
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                    current_line = [word]
                    current_length = word_length

            if current_line:
                lines.append(' '.join(current_line))

            # Limit to 3 lines
            lines = lines[:3]
            if len(error_message.split()) > len(
                ' '.join(lines).split()
            ):
                lines[-1] = lines[-1][:47] + "..."

            # Calculate text positioning
            title_bbox = draw.textbbox((0, 0), title, font=font)
            title_width = title_bbox[2] - title_bbox[0]
            title_height = title_bbox[3] - title_bbox[1]

            # Draw title centered
            title_x = (width - title_width) // 2
            title_y = height // 2 - 60
            draw.text(
                (title_x, title_y),
                title,
                fill=text_color,
                font=font,
            )

            # Draw error message lines
            line_height = title_height + 10
            start_y = title_y + title_height + 20

            for i, line in enumerate(lines):
                line_bbox = draw.textbbox((0, 0), line, font=font)
                line_width = line_bbox[2] - line_bbox[0]
                line_x = (width - line_width) // 2
                line_y = start_y + (i * line_height)

                draw.text(
                    (line_x, line_y),
                    line,
                    fill=text_color,
                    font=font,
                )

            # Convert back to numpy array
            frame = np.array(pil_image)

        except Exception as e:
            # If PIL is not available or text rendering fails,
            # fall back to simple geometric shapes
            print(f"Text rendering failed: {e}")

            # Add centered rectangle for "video error" indication
            center_y, center_x = height // 2, width // 2
            rect_height, rect_width = 100, 400
            y1 = center_y - rect_height // 2
            y2 = center_y + rect_height // 2
            x1 = center_x - rect_width // 2
            x2 = center_x + rect_width // 2

            frame[y1:y2, x1:x2] = text_color

            # Inner rectangle (creates hollow effect)
            inner_border = 5
            frame[
                y1 + inner_border : y2 - inner_border,
                x1 + inner_border : x2 - inner_border,
            ] = bg_color

        return frame

    def _create_fallback_placeholder(self, output_path: Path) -> str:
        """
        Create a minimal fallback placeholder when imageio is unavailable.
        Creates a very simple black video.
        """
        import numpy as np

        try:
            # Try with minimal imageio
            import imageio

            width, height = 640, 360
            fps = 30
            duration = 3
            num_frames = fps * duration

            writer = imageio.get_writer(  # type: ignore (imageio API)
                str(output_path), fps=fps, codec='libx264'
            )

            # Create simple black frames
            black_frame = np.zeros((height, width, 3), dtype=np.uint8)

            for _ in range(num_frames):
                writer.append_data(black_frame)

            writer.close()
            return str(output_path)

        except Exception:
            # Last resort: return a non-existent path
            # The calling code should handle this gracefully
            return str(output_path)

    def clear_cache(self):
        """Remove all cached error videos."""
        if self.cache_dir.exists():
            for video_file in self.cache_dir.glob("*.mp4"):
                try:
                    video_file.unlink()
                except Exception as e:
                    print(
                        f"Error removing cache file {video_file}: {e}"
                    )


# Global cache instance
_global_cache: ErrorPlaceholderCache | None = None


def get_error_placeholder(
    error_type: ErrorType, error_message: str
) -> str:
    """
    Get an error placeholder video (uses global cache).

    Args:
        error_type: Type of error
        error_message: Error message to display

    Returns:
        Path to error placeholder video file
    """
    global _global_cache

    if _global_cache is None:
        _global_cache = ErrorPlaceholderCache()

    return _global_cache.get_placeholder_video(
        error_type, error_message
    )


def cleanup_error_cache():
    """Clear the global error placeholder cache."""
    global _global_cache

    if _global_cache is not None:
        _global_cache.clear_cache()
