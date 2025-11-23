import os
import json
import time
from collections.abc import Callable

import gradio as gr

from lmm_education.config.appwebcast import SLIDE_GAP

SOURCE_DIR = "./webcast_sources/"

if not os.path.exists(f'{SOURCE_DIR}lecture_list.json'):
    print("lecture_list.json not found in Sources directory.")
    raise ValueError("Lecture sources not found.")

# Load the lecture list from the json file defining the script

with open(f"{SOURCE_DIR}lecture_list.json", "r") as file:
    lecture_list = json.load(file)


def video_generator_factory(
    srcdir: str = SOURCE_DIR,
) -> Callable[[], str | dict[str, str] | None]:
    """
    Factory function to create a video generator for videocast presentations.
    Returns a generator function that yields video file paths sequentially.

    :param srcdir: Source directory containing lecture_list.json and video files
    :return: Generator function that returns video file paths
    """
    # check if there is a file 'lecture_list.json' in the Sources directory. If not, exit.
    if not os.path.exists(f'{srcdir}lecture_list.json'):
        print("lecture_list.json not found in Sources directory.")
        raise ValueError("Lecture sources not found.")

    # Load the lecture list from the json file defining the script
    with open(f"{srcdir}lecture_list.json", "r") as file:
        lecture_list = json.load(file)

    # Validate that all lectures have videofile field
    for idx, lecture in enumerate(lecture_list):
        if 'videofile' not in lecture:
            print(f"Warning: Lecture {idx} missing 'videofile' field")

    # a co-routine based on closures
    keeper = {"counter": -1}

    def generate_video() -> str | dict[str, str] | None:
        keeper["counter"] += 1
        if keeper["counter"] < len(lecture_list):
            if SLIDE_GAP > 0.01:
                time.sleep(SLIDE_GAP)

            lecture = lecture_list[keeper["counter"]]
            if 'videofile' not in lecture:
                print(
                    f"Error: Lecture {keeper['counter']} missing 'videofile' field"
                )
                return None

            videofile = srcdir + lecture['videofile']
            print(f"Returning video: {videofile}")

            if not os.path.exists(videofile):
                print(f"Warning: Video file not found: {videofile}")
                return None

            return videofile
        else:
            return None

    return generate_video


def slide_generator_factory(
    srcdir: str = SOURCE_DIR,
) -> Callable[[], list[dict[str, object] | None]]:
    # check if there is a file 'lecture_list.json' in the Sources directory. If not, exit.
    if not os.path.exists(f'{srcdir}lecture_list.json'):
        print("lecture_list.json not found in Sources directory.")
        exit()

    # Load the lecture list from the json file defining the script
    with open(f"{srcdir}lecture_list.json", "r") as file:
        lecture_list = json.load(file)

    # a co-routine based on closures (do not use yield, as gradio will not
    # handle it correctly, speeding up the images). View this as a list of
    # events obtained by transforming lecture_list. It'll start from the
    # second slide, since the first slide is already loaded in the interface.
    # You could also write an iterator class for this, but this is simpler.
    keeper = {"counter": -1}

    def generate_slide() -> list[dict[str, object] | None]:
        keeper["counter"] += 1
        if keeper["counter"] < len(lecture_list):
            if SLIDE_GAP > 0.01:
                time.sleep(SLIDE_GAP)
            print("Returning audio:")
            print(
                srcdir + lecture_list[keeper["counter"]]['audiofile']
            )
            print(
                os.path.exists(
                    srcdir
                    + lecture_list[keeper["counter"]]['audiofile']
                )
            )
            return [
                srcdir + lecture_list[keeper["counter"]]['imagefile'],
                srcdir + lecture_list[keeper["counter"]]['audiofile'],
            ]
        else:
            return [gr.skip(), None]  # type: ignore (incomplete gradio type)

    return generate_slide


def create_video_from_pairs(
    jpg_list: list[str],
    mp3_list: list[str],
    output_directory: str = "output_videos",
) -> None:
    """
    Converts paired JPG and MP3 files into MP4 video files.

    :param jpg_list: List of full paths to JPG image files.
    :param mp3_list: List of full paths to MP3 audio files.
    :param output_directory: The directory where the resulting MP4 files will be saved.
    """
    import os
    from moviepy import ImageClip, AudioFileClip  # type: ignore (stub not found)

    if len(jpg_list) != len(mp3_list):
        print(
            "Error: The number of JPG and MP3 files must be the same."
        )
        return

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    print(f"Starting video creation for {len(jpg_list)} pairs...")

    # Process each pair of files
    for i, (jpg_file, mp3_file) in enumerate(zip(jpg_list, mp3_list)):
        try:
            # 1. Load the audio file
            audio_clip = AudioFileClip(mp3_file)

            # 2. Create the image clip and set its duration to the audio's duration
            image_clip = ImageClip(
                jpg_file, duration=audio_clip.duration  # type: ignore
            )

            # 3. Set the image clip's audio track
            video_clip = image_clip.with_audio(audio_clip)

            # 4. Define the output file name
            # Use the base name of the MP3 file for the video name
            base_name = os.path.splitext(os.path.basename(mp3_file))[
                0
            ]
            output_filename = os.path.join(
                output_directory, f"{base_name}.mp4"
            )

            # 5. Write the final video file
            print(
                f"Processing pair {i + 1}/{len(jpg_list)}: {jpg_file} + {mp3_file} -> {output_filename}"
            )

            video_clip.write_videofile(
                output_filename,
                fps=24,  # Frames per second for the still image (standard value)
                codec='libx264',  # Standard H.264 codec for MP4
                audio_codec='aac',  # Standard AAC audio codec for MP4
                logger=None,  # Suppress MoviePy log messages for cleaner output
            )

            # Close clips to free up resources
            audio_clip.close()
            image_clip.close()
            video_clip.close()

        except Exception as e:
            print(
                f"An error occurred while processing pair {i + 1} ({jpg_file}, {mp3_file}): {e}"
            )

    print("\n✅ All processing complete!")


def convert_source_folder_to_videos(
    srcdir: str = SOURCE_DIR, output_directory: str | None = None
) -> None:
    import glob
    import os

    # Define the directory where your files are located
    FILES_DIR = srcdir
    if output_directory is None:
        output_directory = FILES_DIR

    # Use glob to find all files of a specific type
    jpg_files = sorted(glob.glob(os.path.join(FILES_DIR, "*.jpg")))
    mp3_files = sorted(glob.glob(os.path.join(FILES_DIR, "*.mp3")))

    # Assuming the files are named such that they sort correctly (e.g., 01.jpg, 01.mp3)
    # The `sorted()` function ensures the lists are ordered identically for correct pairing.

    # Now call your function
    create_video_from_pairs(jpg_files, mp3_files, output_directory)
