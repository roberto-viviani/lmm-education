import base64
from openai import OpenAI

client = OpenAI()

# from config import OPENAI_VOICE
OPEANAI_VOICE: str = "nova"


def chat_completion(text: str) -> tuple[str, str]:
    """Get chat completion from OpenAI API."""
    completion = client.chat.completions.create(
        model="gpt-4o-audio-preview",
        modalities=["text", "audio"],
        audio={"voice": OPEANAI_VOICE, "format": "wav"},
        messages=[{"role": "user", "content": text}],
    )

    # at present, with go through writing to a temp file
    # return (completion.choices[0].message.audio.data, completion.choices[0].message.audio.transcript)
    wav_bytes = base64.b64decode(
        completion.choices[0].message.audio.data  # type: ignore
    )

    # write to temp file
    # f = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    # f.write(wav_bytes)
    with open("nova.wav", "wb") as f:
        f.write(wav_bytes)

    return (
        "nova.wav",
        completion.choices[0].message.audio.transcript,  # type: ignore
    )
