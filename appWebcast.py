# basic imports
import gradio as gr
import numpy as np

# configuration imports
from lmm_education.config.appwebcast import SOURCE_DIR

# RAGS imports
from openai import OpenAI

aiclient = OpenAI()

# Whisper imports
import io
import soundfile as sf

import whisper

model = whisper.load_model('tiny.en')
SAMPLE_RATE = whisper.audio.SAMPLE_RATE


# check if there is a file 'lecture_list.json' in the Sources directory. If not, exit.
import os

if not os.path.exists(f'{SOURCE_DIR}lecture_list.json'):
    print("lecture_list.json not found in Sources directory.")
    exit()

# Load the lecture list from the json file defining the script
import json

with open(f"{SOURCE_DIR}lecture_list.json", "r") as file:
    lecture_list = json.load(file)

import time
from lmm_education.config.appwebcast import SLIDE_GAP


def flog(eventType: str) -> str:
    print(f"Event trggered: {eventType}")
    return eventType


AUDIO_FORMAT = "wav"  # "mp3" or "wav"


def transcribe(audiodata: tuple[float, np.ndarray]) -> str:
    if audiodata is None:
        print("No audio data received.")
        return ""
    sr, data = audiodata

    # Convert to mono if stereo
    if data.ndim > 1:
        data = data.mean(axis=1)

    data = data.astype(np.float32)
    data /= np.max(np.abs(data))  # Normalize the audio data

    # transform data, np.ndarray, to a file object
    audio_file = 'C:/temp/slidetext.' + AUDIO_FORMAT
    with io.FileIO(audio_file, 'wb') as f:
        sf.write(f, data, sr, format=AUDIO_FORMAT)

    transcription = model.transcribe(
        audio_file, language='en', fp16=False
    )
    return transcription['text']


import tempfile
import base64
from openai import OpenAI

client = OpenAI()
from lmm_education.config.appwebcast import OPENAI_VOICE


def chat_completion(text: str) -> tuple[str, str]:
    """Get chat completion from OpenAI API."""
    completion = client.chat.completions.create(
        model="gpt-4o-audio-preview",
        modalities=["text", "audio"],
        audio={"voice": OPENAI_VOICE, "format": "wav"},
        messages=[{"role": "user", "content": text}],
    )

    # at present, with go through writing to a temp file
    # return (completion.choices[0].message.audio.data, completion.choices[0].message.audio.transcript)
    wav_bytes = base64.b64decode(
        completion.choices[0].message.audio.data
    )

    # write to temp file
    f = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    f.write(wav_bytes)

    return (f.name, completion.choices[0].message.audio.transcript)


# imports to query index
# import lm_llamaindex as lmlib
import datetime

# lmlib.lmprint()  # feedback about what models we are using.
PERSIST_DIR = "./storage_openai"
# load retriever from index
# retriever = lmlib.load_retriever(persist_folder=PERSIST_DIR)


def queryfn(querytext: str) -> str:
    # history is not used as it is cached and rehashed in the chat engine object
    # but is required by the gradio interface

    # check if the querytext is empty
    if querytext == "":
        return "Please ask a question about the textbook."

    # check if the querytext is too long
    if len(querytext) > 3000:
        return "Your question is too long. Please ask a shorter question."

    # Replacements probably Mistral-specific
    querytext = querytext.replace(
        "the textbook", "the context provided"
    )
    try:
        docs = []  # retriever.retrieve(querytext)
        context = ''
        for doc in docs:
            context += doc.text + r'\n\n'
    except Exception as e:
        print(e)
        with open('./excpetions.log', "a", encoding='utf-8') as f:
            now = datetime.now()
            f.write(
                "\n\n*********************************************************\n"
            )
            f.write(f"{now}: {querytext}\n\nOFFENDING QUERY\n")
        return ''
    return context


def synthetise_query(question: str) -> str:
    prompt = """Please assist the user by answering the user's question with the provided context.\\n
             --------------------------------------\\nQuestion:\\n"""
    prompt += question
    prompt += "\\n--------------------------------------\\n"
    prompt += "Context:\\n"
    context = queryfn(question)
    if context == "":
        return ""
    prompt += context
    return prompt


LOG_EVALUATION = False
LOGFILE = "access_rags_statistics.log"


def chat_function(question: str) -> tuple[str, str]:
    if question == "":
        return (
            './Resources/ErrorEmptyQuestion.mp3',
            'No question recorded. Please feel free to ask your question.',
        )

    if len(question) > 2000:
        return (
            './Resources/ErrorQuestionTooLong.mp3',
            'Your question is too long.',
        )

    prompt = synthetise_query(question)
    if prompt == "":
        return (
            './Resources/ErrorWrongQuestion.mp3',
            'I cannot find information about your question.',
        )

    # for debug, for now
    print('PROMPT: ' + prompt)

    # ask openai to give a reponse with audio
    (audio, response) = chat_completion(prompt)

    # check pertinence
    # (flag, content) = lmlib.check_content(response)
    # if False == flag:
    #     if LOG_EVALUATION:
    #         lmlib.log_evaluation_chat_response(question, response, True)
    #     print(content)
    #     return ('./Resources/ErrorWongQuestion.mp3',
    #             'I am sorry, I do not have this information.')

    # log evaluation
    if LOG_EVALUATION:
        lmlib.log_evaluation_chat_response(question, response, False)

    # log querytext and accesstoken in LOGFILE
    from datetime import datetime

    with open(LOGFILE, "a", encoding='utf-8') as f:
        now = datetime.now()
        f.write(
            "\n\n---------------------------------------------------------------------\n"
        )
        f.write(f"{now}: {question}\n\n{response}\n")

    return (audio, response)


from lmm_education.webcast.sources import slide_generator_factory

# generate_slide = slide_generator_factory()

# a co-routine based on closures (do not use yield, as gradio will not
# handle it correctly, speeding up the images). View this as a list of
# events obtained by transforming lecture_list. It'll start from the
# second slide, since the first slide is already loaded in the interface.
# You could also write an iterator class for this, but this is simpler.
# keeper = {"counter": -1}


# def generate_slide():
#     keeper["counter"] += 1
#     print(f"Slide {keeper["counter"]} of {len(lecture_list)}")
#     if keeper["counter"] < len(lecture_list):
#         print("Loading slide...")
#         if SLIDE_GAP > 0.01:
#             time.sleep(SLIDE_GAP)
#         if not os.path.exists(
#             lecture_list[keeper["counter"]]['imagefile']
#         ):
#             print(
#                 lecture_list[keeper["counter"]]['imagefile']
#                 + " does not exist"
#             )
#         if not os.path.exists(
#             lecture_list[keeper["counter"]]['audiofile']
#         ):
#             print(
#                 lecture_list[keeper["counter"]]['audiofile']
#                 + " does not exist"
#             )
#         print(
#             'Image file: '
#             + lecture_list[keeper["counter"]]['imagefile']
#         )
#         print(
#             'Audio file: '
#             + lecture_list[keeper["counter"]]['audiofile']
#         )
#         return [
#             lecture_list[keeper["counter"]]['imagefile'],
#             lecture_list[keeper["counter"]]['audiofile'],
#         ]
#     else:
#         return [gr.skip(), None]


# Create a Gradio interface to play the slides
with gr.Blocks() as webcast:
    generate_slide = slide_generator_factory()
    img = gr.Image(
        type='filepath',
        show_download_button=False,
        label='Slides',
        show_share_button=False,
    )
    with gr.Row():
        audio = gr.Audio(
            type='filepath',
            show_download_button=False,
            show_share_button=False,
            autoplay=True,
            label='Play lecture',
            waveform_options={'show_recording_waveform': False},
        )
        chatinput = gr.Audio(
            sources="microphone",
            type="numpy",
            show_download_button=False,
            show_share_button=False,
            label='Ask a question',
            waveform_options={
                'show_recording_waveform': False,
                'sample_rate': SAMPLE_RATE,
            },
        )
    txt = gr.Textbox(interactive=False, label='Questions...')
    chatoutput = gr.Audio(
        type='filepath', autoplay=True, visible=False
    )

    # loads the first slide
    webcast.load(
        fn=generate_slide,
        outputs=[img, audio],
    )
    # loads the next slide when the audio has been played
    audio.stop(fn=generate_slide, outputs=[img, audio])

    # you can chain transcribe and chat_completion together, but it gives more
    # feedback if the question is displayed immediately. Here, I suppress the
    # text output of the chat completion, but you can display it if you want.
    chatinput.stop_recording(lambda x: transcribe(x), chatinput, txt)
    txt.change(
        lambda x: (
            gr.Audio(value=None, interactive=False),
            chat_function(x)[0],
        ),
        txt,
        [chatinput, chatoutput],
    )  # suppress text

    # this resets the chat audio input when the output has been read
    chatoutput.stop(
        lambda: gr.Audio(value=None, interactive=True), [], chatinput
    )

# launch the chatcast server
# webcast.launch(server_port=61543)

if __name__ == "__main__":
    # run the app
    from lmm_education.config.appchat import ChatSettings
    from lmm.utils.logging import ConsoleLogger

    logger = ConsoleLogger()

    try:
        chat_settings = ChatSettings()
    except Exception as e:
        logger.error("Could not load chat settings:\n" + str(e))
        exit()

    if chat_settings.server.mode == "local":
        webcast.launch(
            server_port=chat_settings.server.port,
            show_api=False,
            # auth=('accesstoken', 'hackerbrücke'),
        )
    else:
        # allow public access on internet computer
        webcast.launch(
            # server_name='85.124.80.91',  # probably not necessary
            server_port=chat_settings.server.port,
            show_api=False,
            # auth=('accesstoken', 'hackerbrücke'),
        )
