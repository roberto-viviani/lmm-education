"""
Webcast app. Displays a video and allows to chat with a language
model about the content of the video.
"""

# errors line initiates with blank for strings
# flake8: noqa

# basic imports
import os
import time
import json
from typing import TypedDict, NamedTuple
from collections.abc import Callable

import gradio as gr

# logs
import logging
from lmm.utils.logging import FileConsoleLogger # fmt: skip
logger = FileConsoleLogger(
    "LM Markdown for Education - Videocast",
    "appVideocast.log",
    console_level=logging.INFO,
    file_level=logging.ERROR,
)

# settings.
from lmm_education.apputils import async_log_factory, AsyncLogfuncType
from lmm_education.config.config import (
    ConfigSettings,
    load_settings,
)
from lmm_education.config.appchat import (
    ChatSettings,
    load_settings as load_chat_settings,
)
from lmm_education.config.appwebcast import (
    SOURCE_DIR,
    SLIDE_GAP,
)

settings: ConfigSettings | None = load_settings()
if settings is None:
    exit()

chat_settings: ChatSettings | None = load_chat_settings(logger=logger)
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


# load retriever
from lmm_education.stores.langchain.vector_store_qdrant_langchain import (
    AsyncQdrantVectorStoreRetriever as Retriever,
)
from langchain_core.retrievers import BaseRetriever

# TODO: revise from_config_settings with logger and None result
try:
    retriever: BaseRetriever = Retriever.from_config_settings()
except Exception as e:
    logger.error(f"Could not open database: {e}")
    exit(1)


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


# ========== CHATBOT FUNCTIONS ===================

# We need to define fn and fn_checked here, rather than
# importing them, as we need to get the closure with the
# present vector database, not that in appChat.
# The signature of these functions is determined by
# Gradio so it is not clear if it makes sense to pass
# in the vector database.
# Furthermore, just importing from appChat will result
# in a new model-level instantiation of the retriever
# from the same database. Hence, we need to copy all relevant
# functions instead of importing them.

from datetime import datetime
import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from langchain_core.messages import BaseMessageChunk
from langchain_core.language_models import BaseChatModel
from lmm.utils.hash import generate_random_string
from lmm_education.query import (
    chat_function,
    chat_function_with_validation,
)

# logs
from lmm.utils.logging import FileConsoleLogger # fmt: skip
logger = FileConsoleLogger(
    "LM Markdown for Education",
    "appChat.log",
    console_level=logging.INFO,
    file_level=logging.ERROR,
)
DATABASE_FILE = "messages.csv"
CONTEXT_DATABASE_FILE = "queries.csv"

# Initialize CSV database files with headers if they don't exist
if not os.path.exists(DATABASE_FILE):
    with open(DATABASE_FILE, "w", encoding='utf-8') as f:
        f.write(
            "record_id,client_host,session_hash,timestamp,history_length,model_name,interaction_type,query,response\n"
        )

if not os.path.exists(CONTEXT_DATABASE_FILE):
    with open(CONTEXT_DATABASE_FILE, "w", encoding='utf-8') as f:
        f.write("record_id,evaluation,context,classification\n")

# Set up a global container of active tasks (used by logging)
active_logs: set[asyncio.Task[None]] = set()
async_log: AsyncLogfuncType = async_log_factory(
    DATABASE_FILE, CONTEXT_DATABASE_FILE, logger
)


# Create chat engine.
from langchain_core.prompts import PromptTemplate # fmt: skip
prompt: PromptTemplate = PromptTemplate.from_template(
    chat_settings.PROMPT_TEMPLATE
)

from lmm.language_models.langchain.models import (
    create_model_from_settings,
)

llm: BaseChatModel = create_model_from_settings(settings.major)


# Callback for Gradio to call when a chat message is sent.
# TODO: prepare a factory for the chat function, as fn and fn_checked
# are essentially the same, and reused in app modules
async def fn(
    querytext: str, history: list[dict[str, str]], request: gr.Request
) -> AsyncGenerator[str, None]:
    """
    This function is called by the gradio framework each time the
    user posts a new message in the chatbot. The user message is the
    querytext, the history the list of previous exchanges.
    request is a wrapper around FastAPI information, used to write a
    session ID in the logs. Note we collect IP address.

    This version streams the content of the response using the refactored
    chat_function from lmm_education.query.
    """

    # Get iterator from refactored chat_function
    buffer: str = ""
    if chat_settings is None:
        raise ValueError("Unreachable code reached")
    try:
        response_iter: AsyncIterator[BaseMessageChunk] = (
            await chat_function(
                querytext=querytext,
                history=history,
                retriever=retriever,
                llm=llm,
                chat_settings=chat_settings,
                system_msg=chat_settings.SYSTEM_MESSAGE,
                logger=logger,
            )
        )

        # Stream and yield for Gradio
        async for item in response_iter:
            buffer += _preproc_for_markdown(item.text())
            yield buffer

        # Non-blocking logging hook - fires after streaming completes
        record_id: str = generate_random_string(8)
        model_name: str = settings.major.get_model_name()  # type: ignore
        logtask: asyncio.Task[None] = asyncio.create_task(  # type: ignore (pyright confused)
            async_log(
                record_id=record_id,
                client_host=request.client.host,  # type: ignore
                session_hash=request.session_hash or 'unknown',
                timestamp=datetime.now(),
                interaction_type="MESSAGE",
                history=history,
                query=querytext,
                response=buffer,
                model_name=model_name,
            )
        )
        active_logs.add(logtask)
        logtask.add_done_callback(active_logs.discard)

    except Exception as e:
        logger.error(
            f"{request.client.host}: "  # type: ignore (dynamic properties)
            f"{e}\nOFFENDING QUERY:\n{querytext}\n\n"
        )
        buffer = str(e)
        yield chat_settings.MSG_ERROR_QUERY
        return

    return


async def fn_checked(
    querytext: str, history: list[dict[str, str]], request: gr.Request
) -> AsyncGenerator[str, None]:
    """
    This function is called by the gradio framework each time the
    user posts a new message in the chatbot. The user message is the
    querytext, the history the list of previous exchanges.
    request is a wrapper around FastAPI information, used to write a
    session ID in the logs. Note we collect IP address.

    This version checks the content of the stream before releasing it
    using the refactored chat_function_with_validation from lmm_education.query.
    """

    # Get validated iterator from refactored chat_function_with_validation
    buffer: str = ""
    if chat_settings is None:
        raise ValueError("Unreachable code reached")
    try:
        if settings is None:  # for the type checker
            raise ValueError("Unreacheable code reached.")
        response_iter: AsyncIterator[BaseMessageChunk] = (
            await chat_function_with_validation(
                querytext=querytext,
                history=history,
                retriever=retriever,
                llm=llm,
                chat_settings=chat_settings,
                system_msg=chat_settings.SYSTEM_MESSAGE,
                allowed_content=settings.check_response.allowed_content,
                initial_buffer_size=settings.check_response.initial_buffer_size,
                max_retries=2,
                logger=logger,
            )
        )

        # Stream and yield for Gradio
        async for item in response_iter:
            buffer += _preproc_for_markdown(item.text())
            yield buffer

        # Non-blocking logging hook - fires after streaming completes
        record_id: str = generate_random_string(8)
        model_name: str = settings.major.get_model_name()  # type: ignore
        logtask: asyncio.Task[None] = asyncio.create_task(  # type: ignore (pyright confused)
            async_log(
                record_id=record_id,
                client_host=request.client.host,  # type: ignore
                session_hash=request.session_hash or 'unknown',
                timestamp=datetime.now(),
                interaction_type="MESSAGE",
                history=history,
                query=querytext,
                response=buffer,
                model_name=model_name,
            )
        )
        active_logs.add(logtask)
        logtask.add_done_callback(active_logs.discard)

    except Exception as e:
        logger.error(
            f"{request.client.host}: "  # type: ignore (dynamic properties)
            f"{e}\nOFFENDING QUERY:\n{querytext}\n\n"
        )
        buffer = str(e)
        yield chat_settings.MSG_ERROR_QUERY
        return

    return


# centralize used latex style, perhaps depending
# on model.
from lmm_education.apputils import preproc_markdown_factory

_preproc_for_markdown: Callable[[str], str] = (
    preproc_markdown_factory(settings.major)
)


async def vote(data: gr.LikeData, request: gr.Request):
    """
    Async function to log user reactions (like/dislike) to messages.
    """
    record_id = generate_random_string(8)
    reaction = "approved" if data.liked else "disapproved"

    task: asyncio.Task[None] = asyncio.create_task(  # type: ignore (confused)
        async_log(
            record_id=record_id,
            client_host=request.client.host,  # type: ignore
            session_hash=request.session_hash or 'unknown',
            timestamp=datetime.now(),
            interaction_type="USER REACTION",
            history=[],
            query=reaction,
            response="",
            model_name="",
        )
    )
    active_logs.add(task)
    task.add_done_callback(active_logs.discard)


async def postcomment(comment: object, request: gr.Request):
    """
    Async function to log user comments.
    """
    record_id = generate_random_string(8)

    task: asyncio.Task[None] = asyncio.create_task(  # type: ignore (pyright confused)
        async_log(
            record_id=record_id,
            client_host=request.client.host,  # type: ignore
            session_hash=request.session_hash or 'unknown',
            timestamp=datetime.now(),
            interaction_type="USER COMMENT",
            history=[],
            query=str(comment),
            response="",
            model_name="",
        )
    )
    active_logs.add(task)
    task.add_done_callback(active_logs.discard)


ldelims: list[dict[str, str | bool]] = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "$", "right": "$", "display": False},
    {"left": r"\[", "right": r"\]", "display": True},
    {"left": r"\(", "right": r"\)", "display": False},
]


# ========== INTERFACE ===========================

# Create a Gradio interface to play the video slides
with gr.Blocks() as videocast:
    # chatbot for series
    chatbot = gr.Chatbot(type="messages", visible=False)
    # Hidden state for tracking navigation mode
    nav_mode_state = gr.State("auto")

    # Start button - visible initially
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
    # TODO: fix total video number
    progress_info = gr.Markdown(
        "🔄 **Video 1** (1 of 7)",
        visible=False,
    )

    with gr.Row():
        with gr.Column():

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
                    (_get_video_title(i), i)
                    for i in range(len(lecture_list))
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

        with gr.Column():
            chat = gr.Chatbot(
                type="messages",
                latex_delimiters=ldelims,
                show_copy_all_button=True,
                layout="panel",
                visible=False,
                height="24vw",
            )
            chat.like(vote, None, None)
            gr.ChatInterface(
                (
                    fn_checked
                    if settings.check_response.check_response
                    else fn
                ),
                type="messages",
                theme="default",
                api_name=False,
                show_api=False,
                chatbot=chat,
            )
            comment_label = gr.Markdown(
                chat_settings.comment, visible=False
            )
            comment = gr.Textbox(
                label="Comment:",
                submit_btn="Post comment",
                visible=False,
            )
            comment.submit(
                fn=postcomment, inputs=comment, outputs=comment
            )

    def _start_videocast() -> tuple[
        gr.Button,
        gr.Markdown,
        gr.Markdown,
        gr.Row,
        gr.Accordion,
        gr.Video,
        gr.Chatbot,
        gr.Markdown,
        gr.Textbox,
        str | dict[str, str],
        str,
        int,
        list[gr.ChatMessage],
    ]:
        """Handle start button click - shows UI and loads first video"""
        # Navigate to first video using state management
        videofile, progress_text, video_index, updated_history = (
            _navigate_to_video(0, [], mode="auto")
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
            gr.Chatbot(visible=True, type="messages"),  # chatbot
            gr.Markdown(visible=True),  # label for textbox
            gr.Textbox(visible=True),  # comment textbox
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
        outputs=[
            start_btn,
            welcome_msg,
            progress_info,
            nav_row,
            video_selector_accordion,
            video,
            chat,
            comment_label,
            comment,
            video,  # video value
            progress_info,  # progress_info value
            video_selector,  # video_selector value
            chatbot,  # history for chatbot
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
            # decomment to create login
            # auth=('accesstoken', 'hackerbrücke'),
        )

    # cleanup - asyncio diagnostics after Gradio closes
    from lmm_education.apputils import shutdown

    # Check if there's a running loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop - just use asyncio.run()
        asyncio.run(shutdown(active_logs))
    else:
        # Loop is running - use it directly
        if loop.is_running():
            import nest_asyncio  # type: ignore[reportMissingTypeStubs]

            nest_asyncio.apply()  # type: ignore
            loop.run_until_complete(shutdown(active_logs))
        else:
            asyncio.run(shutdown(active_logs))
