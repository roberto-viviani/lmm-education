import base64
from io import BytesIO
from openai import OpenAI
import pygame

client = OpenAI()

# from config import OPENAI_VOICE
OPEANAI_VOICE: str = "nova"


def text_audio_chat_completion(text: str) -> tuple[bool, str]:
    """Get chat completion from OpenAI API and play audio directly
    to loudspeaker. Input in form of text. A text transcript is
    returned.

    Args:
        text: The text prompt to send to the API

    Returns:
        A tuple of (success_flag, transcript) where:
        - success_flag: True if audio was played successfully
        - transcript: The text transcript of the audio response
    """
    completion = client.chat.completions.create(
        model="gpt-4o-audio-preview",
        modalities=["text", "audio"],
        audio={"voice": OPEANAI_VOICE, "format": "wav"},
        messages=[{"role": "user", "content": text}],
    )

    # Decode the base64 audio data
    wav_bytes = base64.b64decode(
        completion.choices[0].message.audio.data  # type: ignore
    )

    # Initialize pygame mixer if not already initialized
    if not pygame.mixer.get_init():
        pygame.mixer.init()

    # Load and play audio directly from bytes
    audio_buffer = BytesIO(wav_bytes)
    sound = pygame.mixer.Sound(audio_buffer)
    sound.play()

    # Wait for the audio to finish playing
    while pygame.mixer.get_busy():
        pygame.time.Clock().tick(10)

    return (
        True,
        completion.choices[0].message.audio.transcript,  # type: ignore
    )


from lmm_education.config.appwebcast import SLIDE_GAP
import os
import json
import time
import gradio as gr
from collections.abc import Callable


def slide_generator_factory(
    srcdir: str,
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
