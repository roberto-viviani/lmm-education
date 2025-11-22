"""
This file contains the configuration for the Webcast source files.
"""

# Settings for directories where the source files are located
SOURCE_DIR = './Sources/'  # Files in the sources folder are used to create scripts

# Settings for the OpenAI client
OPENAI_VOICE = 'nova'  # The voice to use for the audio files
OPENAI_VOICE_INSTRUCTION = (
    "Accent/Affect: Warm, refined, and gently "
    "instructive, reminiscent of a friendly art instructor.\n\nTone: "
    "Calm, encouraging, and articulate, clearly describing each step "
    "with patience.\n\nPacing: Slow and deliberate, pausing often to "
    "allow the listener to follow instructions comfortably.\n\n"
    "Emotion: Cheerful, supportive, and pleasantly enthusiastic; "
    "convey genuine enjoyment and appreciation of scientific insight."
    "\n\nPronunciation: Clearly articulate statistical and coding "
    "terminology (e.g., 'logistic', 'link function', 'model "
    "equation') with gentle emphasis.\n\nPersonality Affect: Friendly "
    "and approachable with a hint of sophistication; speak confidently "
    "and reassuringly, guiding users through each coding step "
    "patiently and warmly."
)

SLIDE_GAP = 0.35  # The time to wait between slides
