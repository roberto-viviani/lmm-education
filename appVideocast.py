# basic imports
import os
import time
import json
from typing import TypedDict, NamedTuple

import gradio as gr
from lmm_education.webcast.webcastlib import (
    synthetise_query,
    transcribe,
    SAMPLE_RATE,
)

# logs
import logging
from lmm.utils.logging import FileConsoleLogger # fmt: skip
logger = FileConsoleLogger(
    "LM Markdown for Education - Videocast",
    "appVideocast.log",
    console_level=logging.INFO,
    file_level=logging.ERROR,
)

# configuration imports
from lmm_education.config.appchat import ChatSettings, load_settings
from lmm_education.config.appwebcast import (
    SOURCE_DIR,
    SLIDE_GAP,
    OPENAI_VOICE,
)

chat_settings: ChatSettings | None = load_settings(logger=logger)
if chat_settings is None:
    exit()


# RAGS imports
from openai import OpenAI

aiclient = OpenAI()


# check if there is a file 'lecture_list.json' in the Sources directory. If not, exit.
if not os.path.exists(f'{SOURCE_DIR}lecture_list.json'):
    logger.error(
        f"lecture_list.json not found in {SOURCE_DIR} directory."
    )
    exit()

# Load the lecture list from the json file defining the script
with open(f"{SOURCE_DIR}lecture_list.json", "r") as file:
    lecture_list = json.load(file)

# Validate lecture list structure and check for video files
video_file_missing = False
for idx, lecture in enumerate(lecture_list):
    if 'videofile' not in lecture:
        logger.warning(
            f"Lecture {idx} missing 'videofile' field, skipping validation"
        )
        continue

    video_path = os.path.join(SOURCE_DIR, lecture['videofile'])
    if not os.path.exists(video_path):
        video_file_missing = True
        logger.error(f"Video file not found: {video_path}")
    else:
        # Check file size
        file_size = os.path.getsize(video_path)
        if file_size == 0:
            logger.warning(f"Video file is empty: {video_path}")
        elif file_size > 100 * 1024 * 1024:  # 100MB warning threshold
            logger.warning(
                f"Large video file ({file_size / (1024 * 1024):.1f}MB): {video_path}"
            )

if video_file_missing:
    logger.error(
        "Video files missing. Replace video files before starting app."
    )
    exit(1)


def chat_function(
    question: str,
) -> tuple[str | bytes, str]:
    """
    Process a chat question and return audio response with transcript.
    Returns (audio_data, transcript) where audio_data can be file path or bytes.
    """
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

    if chat_settings is None:  # type checker
        raise ValueError("Unreacheable code reached.")

    prompt = synthetise_query(
        question, retriever, chat_settings, logger
    )
    if prompt == "":
        return (
            './Resources/ErrorWrongQuestion.mp3',
            'I cannot find information about your question.',
        )

    # ask openai to give a reponse with audio
    (audio, response) = chat_completion(prompt)
    # TODO: logging

    return (audio, response)


def chat_completion(text: str) -> tuple[bytes, str]:
    """
    Get chat completion from OpenAI API.
    Returns audio data as bytes and transcript text.
    """
    import base64

    completion = aiclient.chat.completions.create(
        model="gpt-4o-audio-preview",
        modalities=["text", "audio"],
        audio={"voice": OPENAI_VOICE, "format": "wav"},
        messages=[{"role": "user", "content": text}],
    )

    # Decode audio data from base64 to bytes
    wav_bytes = base64.b64decode(
        completion.choices[0].message.audio.data  # type: ignore
    )

    # Return bytes directly - no temporary file needed!
    return (wav_bytes, completion.choices[0].message.audio.transcript)  # type: ignore


# load retriever
from lmm_education.stores.langchain.vector_store_qdrant_langchain import (
    QdrantVectorStoreRetriever as Retriever,
)
from langchain_core.retrievers import BaseRetriever

# TODO: revise from_config_settings with logger and None result
retriever: BaseRetriever = Retriever.from_config_settings()


# ========== STATE MANAGEMENT FUNCTIONS ==========


class SessionStateDict(TypedDict):
    current_index: int
    viewed: list[int]
    navigation_mode: str
    video_file: str
    error_state: (
        str | None
    )  # "error" if current video has error, None otherwise
    failed_videos: list[
        int
    ]  # List of video indices that failed to load
    retry_count: int  # Number of retries for current video


class NavigationResult(NamedTuple):
    """Result of video navigation operations."""

    video_source: str | dict[str, str]  # Video file path or gr.skip()
    progress_text: str  # Progress indicator text
    video_index: int  # Current video position (0-based)
    chat_history: list[gr.ChatMessage]  # Updated chat history


def _get_video_title(index: int) -> str:
    """Get a display title for a video."""
    if 0 <= index < len(lecture_list):
        lecture = lecture_list[index]
        # Try to extract a meaningful title from text field
        text = lecture.get('text', '')
        if text:
            # Get first sentence or first 50 chars
            first_sentence = text.split('.')[0][:50]
            return f"Video {index + 1}: {first_sentence}..."
        return f"Video {index + 1}"
    return "Unknown"


def _get_user_state(
    history: list[gr.ChatMessage] | list[gr.MessageDict],
) -> SessionStateDict:
    """
    Extract current user state from chatbot history.
    Returns state dict with current_index, viewed list, and navigation_mode.
    """
    if not history:
        return {
            "current_index": 0,
            "viewed": [],
            "navigation_mode": "auto",
            "video_file": "",
            "error_state": None,
            "failed_videos": [],
            "retry_count": 0,
        }

    # convert input
    msg_history: list[gr.ChatMessage] = [
        gr.ChatMessage(**h) if isinstance(h, dict) else h
        for h in history
    ]
    try:
        # Try to parse the last entry as JSON state
        last_content = msg_history[-1].content  # type: ignore (gradio type)
        if isinstance(last_content, str) and last_content.startswith(
            '{"current_index"'
        ):
            state = json.loads(last_content)
            # Validate required fields
            if all(
                key in state
                for key in [
                    "current_index",
                    "viewed",
                    "navigation_mode",
                ]
            ):
                return state
    except (json.JSONDecodeError, KeyError, AttributeError):
        pass

    # Fallback: treat history length as current index (backward compatibility)
    # Count non-state entries (video file paths)
    video_count = sum(
        1
        for msg in msg_history
        if isinstance(msg.content, str)  # type: ignore (gradio type)
        and not msg.content.startswith('{"current_index"')
    )
    return {
        "current_index": max(0, video_count - 1),
        "viewed": list(range(video_count)),
        "navigation_mode": "auto",
        "video_file": "",
        "error_state": None,
        "failed_videos": [],
        "retry_count": 0,
    }


def _create_state_entry(
    index: int,
    mode: str,
    viewed_list: list[int],
    error_state: str | None = None,
    failed_videos: list[int] | None = None,
    retry_count: int = 0,
) -> SessionStateDict:
    """Create a state data structure for storing in history."""
    if 0 <= index < len(lecture_list):
        lecture = lecture_list[index]
        videofile = lecture.get('videofile', '')
    else:
        videofile = ''

    return {
        "current_index": index,
        "video_file": videofile,
        "viewed": sorted(list(set(viewed_list + [index]))),
        "navigation_mode": mode,
        "error_state": error_state,
        "failed_videos": (
            failed_videos if failed_videos is not None else []
        ),
        "retry_count": retry_count,
    }


def _get_progress_text(state: SessionStateDict) -> str:
    """Generate progress indicator text from state."""
    current = state.get("current_index", 0)
    total = len(lecture_list)
    viewed = state.get("viewed", [])
    mode = state.get("navigation_mode", "auto")

    title = _get_video_title(current)  # type: ignore (current_index)
    mode_emoji = "🔄" if mode == "auto" else "⏸️"

    progress = f"{mode_emoji} **{title}** ({current + 1} of {total})"
    if len(viewed) > 1:
        progress += f" | Viewed: {len(viewed)}/{total}"

    return progress


def _navigate_to_video(
    target_index: int,
    history: list[gr.ChatMessage] | list[gr.MessageDict],
    mode: str | None = None,
) -> NavigationResult:
    """
    Navigate to a specific video index with state management and error handling.
    Includes 2-attempt retry logic with 2-second delays.
    Returns NavigationResult with video source, progress text, index, and history.
    """
    from lmm_education.webcast.video_validation import (
        validate_video_file,
        map_validation_to_error_type,
    )
    from lmm_education.webcast.error_handling import (
        get_error_placeholder,
    )

    # convert input
    msg_history: list[gr.ChatMessage] = [
        gr.ChatMessage(**h) if isinstance(h, dict) else h
        for h in history
    ]

    # Get current state
    current_state = _get_user_state(msg_history)

    # Validate target index
    if not (0 <= target_index < len(lecture_list)):
        logger.warning(f"Invalid video index: {target_index}")
        return NavigationResult(
            video_source=gr.skip(),  # type: ignore (gradio type)
            progress_text=_get_progress_text(current_state),
            video_index=0,
            chat_history=msg_history,
        )

    # Get lecture info
    lecture = lecture_list[target_index]

    if 'videofile' not in lecture:
        logger.error(
            f"Lecture {target_index} missing 'videofile' field"
        )
        return NavigationResult(
            video_source=gr.skip(),  # type: ignore (gradio type)
            progress_text=_get_progress_text(current_state),
            video_index=0,
            chat_history=msg_history,
        )

    videofile = os.path.join(SOURCE_DIR, lecture['videofile'])

    # Check if this video has already failed
    failed_videos = current_state.get("failed_videos", [])
    if target_index in failed_videos:
        # Video previously failed, use error placeholder immediately
        logger.warning(
            f"Video {target_index} previously failed, showing error placeholder"
        )
        error_type = map_validation_to_error_type("corrupted")
        placeholder = get_error_placeholder(
            error_type, f"Video {target_index + 1} unavailable"
        )

        error_state = _create_state_entry(
            target_index,
            "manual",  # Force manual mode on error
            current_state.get("viewed", []),
            error_state="error",
            failed_videos=failed_videos,
            retry_count=0,
        )

        msg_history.append(
            gr.ChatMessage(
                role="assistant", content=json.dumps(error_state)
            )
        )

        error_progress = f"⚠️ **Video Error** ({target_index + 1} of {len(lecture_list)})"

        return NavigationResult(
            video_source=placeholder,
            progress_text=error_progress,
            video_index=target_index,
            chat_history=msg_history,
        )

    # Retry logic: Try to load video with up to 2 attempts
    MAX_RETRIES = 2
    RETRY_DELAY = 2.0  # seconds

    for attempt in range(MAX_RETRIES):
        # Validate video file
        validation_result, error_message = validate_video_file(
            videofile
        )

        if validation_result == "success":
            # Video is valid, proceed with normal loading
            navigation_mode = (
                mode
                if mode is not None
                else current_state.get("navigation_mode", "auto")
            )

            # Create new state (clear any error state)
            new_state = _create_state_entry(
                target_index,
                navigation_mode,
                current_state.get("viewed", []),
                error_state=None,
                failed_videos=failed_videos,
                retry_count=0,
            )

            # Add state to history
            msg_history.append(
                gr.ChatMessage(
                    role="assistant", content=json.dumps(new_state)
                )
            )

            # Add slide gap delay if configured
            if SLIDE_GAP > 0.01:
                time.sleep(SLIDE_GAP)

            logger.info(
                f"Successfully navigated to video {target_index + 1}/{len(lecture_list)}: {videofile}"
            )

            return NavigationResult(
                video_source=videofile,
                progress_text=_get_progress_text(new_state),
                video_index=target_index,
                chat_history=msg_history,
            )

        # Video validation failed
        if attempt < MAX_RETRIES - 1:
            # Not the last attempt, retry after delay
            logger.warning(
                f"Video validation failed (attempt {attempt + 1}/{MAX_RETRIES}): {error_message}"
            )
            time.sleep(RETRY_DELAY)
        else:
            # Final attempt failed, show error placeholder
            logger.error(
                f"Video validation failed after {MAX_RETRIES} attempts: {error_message}"
            )

            # Add to failed videos list
            failed_videos = list(set(failed_videos + [target_index]))

            # Generate error placeholder
            error_type = map_validation_to_error_type(
                validation_result
            )
            placeholder = get_error_placeholder(
                error_type, error_message
            )

            # Create error state (force manual mode to pause auto-advance)
            error_state = _create_state_entry(
                target_index,
                "manual",  # Force manual mode when error occurs
                current_state.get("viewed", []),
                error_state="error",
                failed_videos=failed_videos,
                retry_count=MAX_RETRIES,
            )

            msg_history.append(
                gr.ChatMessage(
                    role="assistant", content=json.dumps(error_state)
                )
            )

            # Create error progress message
            error_progress = f"⚠️ **Video Error: {error_message}** ({target_index + 1} of {len(lecture_list)})"

            return NavigationResult(
                video_source=placeholder,
                progress_text=error_progress,
                video_index=target_index,
                chat_history=msg_history,
            )

    # Should never reach here, but handle it just in case
    return NavigationResult(
        video_source=gr.skip(),  # type: ignore (gradio type)
        progress_text=_get_progress_text(current_state),
        video_index=target_index,
        chat_history=msg_history,
    )


def _previous_video(
    history: list[gr.ChatMessage] | list[gr.MessageDict],
) -> NavigationResult:
    """Navigate to previous video."""
    state = _get_user_state(history)
    target = max(0, state["current_index"] - 1)
    return _navigate_to_video(target, history)


def _next_video(
    history: list[gr.ChatMessage] | list[gr.MessageDict],
) -> NavigationResult:
    """Navigate to next video."""
    state = _get_user_state(history)
    target = min(len(lecture_list) - 1, state["current_index"] + 1)
    return _navigate_to_video(target, history)


def _toggle_autoplay(
    current_mode: str, history: list[gr.ChatMessage]
) -> tuple[str, str, list[gr.ChatMessage]]:
    """Toggle between auto-advance and manual mode."""
    state = _get_user_state(history)
    new_mode = "manual" if current_mode == "auto" else "auto"

    # Update state with new mode
    new_state = _create_state_entry(
        state["current_index"], new_mode, state.get("viewed", [])
    )

    history.append(
        gr.ChatMessage(
            role="assistant", content=json.dumps(new_state)
        )
    )

    button_text = (
        "🔄 Auto-Advance: ON"
        if new_mode == "auto"
        else "⏸️ Auto-Advance: OFF"
    )
    logger.info(f"Playback mode changed to: {new_mode}")

    return (button_text, new_mode, history)


# Create a Gradio interface to play the video slides
with gr.Blocks() as videocast:
    chatbot = gr.Chatbot(type="messages", visible=False)
    # Hidden state for tracking navigation mode
    nav_mode_state = gr.State("auto")

    # Start button - visible initially
    with gr.Row():
        start_btn = gr.Button(
            "🎬 Start Videocast",
            variant="primary",
            size="lg",
            scale=1,
        )

    # Welcome message
    welcome_msg = gr.Markdown(
        """
        ### Welcome to the Videocast
        
        Click the **Start Videocast** button above to begin the presentation.
        
        Once started, you can:
        - 🎥 View lecture videos with synchronized visuals and audio
        - 🎤 Ask questions using your microphone
        - ⏯️ Control playback with the video controls
        - ⬅️➡️ Navigate manually between videos
        
        **Note:** Videos will autoplay by default. Use the Auto-Advance toggle to control progression.
        """,
        visible=True,
    )

    # Progress indicator - hidden initially
    progress_info = gr.Markdown(
        "🔄 **Video 1** (1 of 7)",
        visible=False,
    )

    # Navigation controls - hidden initially
    with gr.Row(visible=False) as nav_row:
        prev_btn = gr.Button("⬅️ Previous", size="sm", scale=1)
        autoplay_btn = gr.Button(
            "🔄 Auto-Advance: ON", size="sm", scale=1
        )
        next_btn = gr.Button("Next ➡️", size="sm", scale=1)

    # Video selector dropdown - hidden initially
    with gr.Accordion(
        "📋 Jump to Video", open=False, visible=False
    ) as video_selector_accordion:
        video_choices = [
            (_get_video_title(i), i) for i in range(len(lecture_list))
        ]
        video_selector = gr.Dropdown(
            choices=video_choices,
            label="Select Video",
            value=0,
        )

    # Main content - hidden initially
    video = gr.Video(
        show_download_button=False,
        label='Lecture Video',
        show_share_button=False,
        autoplay=True,
        visible=False,
        height="30vw",
    )

    with gr.Row(visible=False) as chat_row:
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

    def _start_videocast(
        history: list[gr.ChatMessage],
    ) -> tuple[
        gr.Button,
        gr.Markdown,
        gr.Markdown,
        gr.Row,
        gr.Accordion,
        gr.Video,
        gr.Row,
        gr.Textbox,
        str | dict[str, str],
        str,
        int,
        list[gr.ChatMessage],
    ]:
        """Handle start button click - shows UI and loads first video"""
        # Navigate to first video using state management
        videofile, progress_text, video_index, updated_history = (
            _navigate_to_video(0, history, mode="auto")
        )

        # Return updates for all components:
        # Hide start button and welcome message, show main content
        return (
            gr.Button(visible=False),  # start_btn
            gr.Markdown(visible=False),  # welcome_msg
            gr.Markdown(visible=True),  # progress_info
            gr.Row(visible=True),  # nav_row
            gr.Accordion(visible=True),  # video_selector_accordion
            gr.Video(visible=True),  # video
            gr.Row(visible=True),  # chat_row
            gr.Textbox(visible=True),  # txt
            videofile,  # video value
            progress_text,  # progress_info value
            video_index,  # video_selector value
            updated_history,  # chatbot
        )

    def _auto_advance_video(
        history: list[gr.ChatMessage],
    ) -> tuple[str | dict[str, str], str, int, list[gr.ChatMessage]]:
        """
        Auto-advance to next video when current video ends.
        Only advances if in auto mode.
        """
        state = _get_user_state(history)

        # Check if we're in auto mode
        if state.get("navigation_mode") != "auto":
            # In manual mode, don't auto-advance
            current_index = state.get("current_index", 0)
            return (gr.skip(), _get_progress_text(state), current_index, history)  # type: ignore

        # Auto-advance to next video
        current_index = state.get("current_index", 0)
        next_index = current_index + 1

        # Check if there's a next video
        if next_index >= len(lecture_list):
            logger.info("Reached end of lecture series")
            return (gr.skip(), _get_progress_text(state), current_index, history)  # type: ignore

        # Navigate to next video
        return _navigate_to_video(next_index, history)

    # Connect start button to show UI and load first video
    start_btn.click(
        fn=_start_videocast,
        inputs=[chatbot],
        outputs=[
            start_btn,
            welcome_msg,
            progress_info,
            nav_row,
            video_selector_accordion,
            video,
            chat_row,
            txt,
            video,
            progress_info,
            video_selector,
            chatbot,
        ],
    )

    # Auto-advance when video ends (only if in auto mode)
    video.stop(
        fn=_auto_advance_video,
        inputs=[chatbot],
        outputs=[video, progress_info, video_selector, chatbot],
    )

    # Navigation button handlers
    prev_btn.click(
        fn=_previous_video,
        inputs=[chatbot],
        outputs=[video, progress_info, video_selector, chatbot],
    )

    next_btn.click(
        fn=_next_video,
        inputs=[chatbot],
        outputs=[video, progress_info, video_selector, chatbot],
    )

    # Auto-advance toggle handler
    autoplay_btn.click(
        fn=_toggle_autoplay,
        inputs=[nav_mode_state, chatbot],
        outputs=[autoplay_btn, nav_mode_state, chatbot],
    )

    # Video selector dropdown handler
    video_selector.change(
        fn=lambda idx, history: _navigate_to_video(idx, history),  # type: ignore
        inputs=[video_selector, chatbot],
        outputs=[video, progress_info, video_selector, chatbot],
    )

    # Chain transcribe and chat_completion for Q&A functionality
    chatinput.stop_recording(
        lambda x: transcribe(x), chatinput, txt  # type: ignore
    ).then(
        lambda x: (  # type: ignore
            gr.Audio(value=None, interactive=False),
            chat_function(x)[0],  # type: ignore
        ),
        txt,
        [chatinput, chatoutput],
    )

    # Reset the chat audio input when the output has been played
    chatoutput.stop(
        lambda: gr.Audio(value=None, interactive=True), [], chatinput
    )

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
        videocast.launch(
            server_port=chat_settings.server.port,
            show_api=False,
            # auth=('accesstoken', 'hackerbrücke'),
        )
    else:
        # allow public access on internet computer
        videocast.launch(
            server_name='85.124.80.91',  # probably not necessary
            server_port=chat_settings.server.port,
            show_api=False,
            # auth=('accesstoken', 'hackerbrücke'),
        )

    # shut down db client
    if retriever is not None:  # type: ignore
        retriever.close_client()  # type: ignore
