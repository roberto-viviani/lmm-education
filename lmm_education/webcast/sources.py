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
