# basic imports
import os
import io
import time
import json

import whisper  # type: ignore (stub file not found)
import soundfile as sf  # type: ignore (stub file etc.)
import gradio as gr
import numpy as np

# configuration imports
from lmm_education.config.appwebcast import (
    SOURCE_DIR,
    SLIDE_GAP,
    OPENAI_VOICE,
)

# logs
import logging
from lmm.utils.logging import FileConsoleLogger # fmt: skip
logger = FileConsoleLogger(
    "LM Markdown for Education",
    "appWebcast.log",
    console_level=logging.INFO,
    file_level=logging.ERROR,
)

# settings
from lmm_education.config.appchat import ChatSettings, load_settings

chat_settings: ChatSettings | None = load_settings(logger=logger)
if chat_settings is None:
    exit()

# RAGS imports
from openai import OpenAI

aiclient = OpenAI()
model = whisper.load_model('tiny.en')
SAMPLE_RATE = whisper.audio.SAMPLE_RATE
AUDIO_FORMAT = "wav"  # "mp3" or "wav"


# check if there is a file 'lecture_list.json' in the Sources directory. If not, exit.
if not os.path.exists(f'{SOURCE_DIR}lecture_list.json'):
    print("lecture_list.json not found in Sources directory.")
    exit()

# Load the lecture list from the json file defining the script
with open(f"{SOURCE_DIR}lecture_list.json", "r") as file:
    lecture_list = json.load(file)


def transcribe(audiodata: tuple[float, np.ndarray] | None) -> str:
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
        sf.write(f, data, sr, format=AUDIO_FORMAT)  # type: ignore

    transcription = model.transcribe(  # type: ignore
        audio_file, language='en', fp16=False
    )
    return str(transcription['text'])  # type: ignore


def chat_completion(text: str) -> tuple[str, str]:
    """Get chat completion from OpenAI API."""
    import tempfile
    import base64

    completion = aiclient.chat.completions.create(
        model="gpt-4o-audio-preview",
        modalities=["text", "audio"],
        audio={"voice": OPENAI_VOICE, "format": "wav"},
        messages=[{"role": "user", "content": text}],
    )

    # at present, with go through writing to a temp file
    # return (completion.choices[0].message.audio.data, completion.choices[0].message.audio.transcript)
    wav_bytes = base64.b64decode(
        completion.choices[0].message.audio.data  # type: ignore
    )

    # write to temp file
    f = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    f.write(wav_bytes)

    return (f.name, completion.choices[0].message.audio.transcript)  # type: ignore


# load retriever
from lmm_education.stores.langchain.vector_store_qdrant_langchain import (
    QdrantVectorStoreRetriever as Retriever,
)
from langchain_core.retrievers import BaseRetriever

# TODO: revise from_config_settings with logger and None result
retriever: BaseRetriever = Retriever.from_config_settings()


def queryfn(querytext: str) -> str:
    # required by type checker
    if chat_settings is None:
        raise ValueError("Unreacheable code reached")

    # check if the querytext is empty
    if querytext == "":
        return chat_settings.MSG_EMPTY_QUERY

    # check if the querytext is too long
    if len(querytext) > 3000:
        return chat_settings.MSG_LONG_QUERY

    # query database
    context: str = ""
    try:
        docs = retriever.invoke(querytext)
        for doc in docs:
            context += doc.page_content + "\n------\n"
    except Exception as e:
        logger.error(f"Error while retrieving context: {e}")
        return context

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
    print("RESPONSE from model: ------------------")
    print(response)

    return (audio, response)


# Create a Gradio interface to play the slides
with gr.Blocks() as webcast:
    chatbot = gr.Chatbot(type="messages", visible=False)

    # Start button - visible initially
    with gr.Row():
        start_btn = gr.Button(
            "🎬 Start Webcast", variant="primary", size="lg", scale=1
        )

    # Welcome message
    welcome_msg = gr.Markdown(
        """
        ### Welcome to the Webcast
        
        Click the **Start Webcast** button above to begin the presentation.
        
        Once started, you can:
        - 📊 View lecture slides with synchronized audio
        - 🎤 Ask questions using your microphone
        - ⏯️ Control playback with the audio controls
        
        **Note:** Audio will autoplay after you start the webcast.
        """,
        visible=True,
    )

    # Main content - hidden initially
    img = gr.Image(
        type='filepath',
        show_download_button=False,
        label='Slides',
        show_share_button=False,
        visible=False,
    )
    with gr.Row(visible=False) as audio_row:
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
    txt = gr.Textbox(
        interactive=False, label='Questions...', visible=False
    )
    chatoutput = gr.Audio(
        type='filepath', autoplay=True, visible=False
    )

    def _get_lecture(
        history: list[gr.ChatMessage],
    ) -> tuple[
        str | dict[str, str], str | None, list[gr.ChatMessage]
    ]:
        # returns image and audio file, updates history
        counter: int = len(history)
        if counter < len(lecture_list):
            imagefile = lecture_list[counter]['imagefile']
            audiofile = lecture_list[counter]['audiofile']
            if not os.path.exists(imagefile):
                print('Image not found')
            if not os.path.exists(audiofile):
                print('Audio not found')
            history.append(
                gr.ChatMessage(
                    role="assistant",
                    content=imagefile,
                )
            )
            time.sleep(SLIDE_GAP)
            return (imagefile, audiofile, history)
        else:
            return (gr.skip(), None, history)  # type: ignore (gradio type)

    def _start_webcast(
        history: list[gr.ChatMessage],
    ) -> tuple[
        gr.Button,
        gr.Markdown,
        gr.Image,
        gr.Row,
        gr.Textbox,
        str | dict[str, str],
        str | None,
        list[gr.ChatMessage],
    ]:
        """Handle start button click - shows UI and loads first slide"""
        # Get the first lecture slide
        imagefile, audiofile, updated_history = _get_lecture(history)

        # Return updates for all components:
        # Hide start button and welcome message, show main content
        return (
            gr.Button(visible=False),  # start_btn
            gr.Markdown(visible=False),  # welcome_msg
            gr.Image(visible=True),  # img
            gr.Row(visible=True),  # audio_row
            gr.Textbox(visible=True),  # txt
            imagefile,  # img value
            audiofile,  # audio value
            updated_history,  # chatbot
        )

    # Connect start button to show UI and load first slide
    start_btn.click(
        fn=_start_webcast,
        inputs=[chatbot],
        outputs=[
            start_btn,
            welcome_msg,
            img,
            audio_row,
            txt,
            img,
            audio,
            chatbot,
        ],
    )
    # loads the next slide when the audio has been played
    audio.stop(
        fn=_get_lecture,
        inputs=[chatbot],
        outputs=[img, audio, chatbot],
    )

    # you can chain transcribe and chat_completion together, but it gives more
    # feedback if the question is displayed immediately. Here, I suppress the
    # text output of the chat completion, but you can display it if you want.
    chatinput.stop_recording(lambda x: transcribe(x), chatinput, txt)  # type: ignore
    txt.change(
        lambda x: (  # type: ignore
            gr.Audio(value=None, interactive=False),
            chat_function(x)[0],  # type: ignore
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
            server_name='85.124.80.91',  # probably not necessary
            server_port=chat_settings.server.port,
            show_api=False,
            # auth=('accesstoken', 'hackerbrücke'),
        )
    # shut down db client
    if retriever is not None:  # type: ignore
        retriever.close_client()  # type: ignore
