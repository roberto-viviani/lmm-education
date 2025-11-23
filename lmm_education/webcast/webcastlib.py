import io
import numpy as np
import soundfile as sf  # type: ignore (stub file etc.)
import whisper  # type: ignore (stub file not found)

whisper_model = whisper.load_model('tiny.en')

SAMPLE_RATE: int = whisper.audio.SAMPLE_RATE

from langchain_core.retrievers import BaseRetriever

from lmm.utils.logging import LoggerBase
from lmm_education.config.appchat import ChatSettings

AUDIO_FORMAT = "wav"


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

    transcription = whisper_model.transcribe(  # type: ignore
        audio_file, language='en', fp16=False
    )
    return str(transcription['text'])  # type: ignore


def queryfn(
    querytext: str,
    retriever: BaseRetriever,
    chat_settings: ChatSettings,
    logger: LoggerBase,
) -> str:
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


def synthetise_query(
    question: str,
    retriever: BaseRetriever,
    settings: ChatSettings,
    logger: LoggerBase,
) -> str:
    prompt = """Please assist the user by answering the user's question with the provided context.\\n
             --------------------------------------\\nQuestion:\\n"""
    prompt += question
    prompt += "\\n--------------------------------------\\n"
    prompt += "Context:\\n"
    context = queryfn(question, retriever, settings, logger)
    if context == "":
        return ""
    prompt += context
    return prompt
