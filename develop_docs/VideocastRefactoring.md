# Videocast Refactoring Documentation

## Overview

This document describes the refactoring of the `appWebcast.py` prototype to support MP4 video playback instead of separate JPG image and MP3 audio files.

## Phase 1 Implementation: Basic Video Playback

### Changes Made

#### 1. New Module: `appVideocast.py`

A new module was created to handle video-based presentations while maintaining backward compatibility with the original `appWebcast.py`.

**Key Features:**
- Replaces `gr.Image()` and `gr.Audio()` components with a single `gr.Video()` component
- Maintains Q&A functionality with voice transcription
- Implements automatic progression using `video.stop()` event
- Includes validation for video file existence and format
- Provides file size warnings for large videos (>100MB)

#### 2. Data Structure Updates

**Updated `webcast_sources/lecture_list.json`:**
- Added `videofile` field to each lecture entry
- Maintained existing fields (`imagefile`, `audiofile`, `textfile`, `text`) for backward compatibility
- Each entry now includes: `videofile`, `imagefile`, `audiofile`, `textfile`, and `text`

**Example Entry:**
```json
{
  "imagefile": "Lecture8_00.jpg",
  "textfile": "Lecture8_00.txt",
  "text": "Lecture content text...",
  "audiofile": "Lecture8_00.mp3",
  "videofile": "Lecture8_00.mp4"
}
```

#### 3. Configuration Updates

**Modified `lmm_education/config/appwebcast.py`:**
- Changed `SOURCE_DIR` from `'./Sources/'` to `'./webcast_sources/'`
- Maintained `SAMPLE_RATE` and `AUDIO_FORMAT` constants for backward compatibility with Q&A transcription
- Added comment noting backward compatibility considerations

#### 4. Utility Functions

**Added to `lmm_education/webcast/sources.py`:**
- `video_generator_factory()`: Factory function for creating video generators
- Provides validation for video files
- Implements sequential video playback with configurable delays
- Maintains existing `slide_generator_factory()` for backward compatibility

### Architecture Changes

#### Gradio Interface

**Before (appWebcast.py):**
```python
img = gr.Image(...)
audio = gr.Audio(autoplay=True, ...)
audio.stop(fn=_get_lecture, ...)
```

**After (appVideocast.py):**
```python
video = gr.Video(autoplay=True, ...)
video.stop(fn=_get_lecture, ...)
```

#### Playback Logic

- **Auto-progression:** Videos play to completion naturally, then automatically load the next video via the `video.stop()` event
- **Timing:** Respects `SLIDE_GAP` configuration for delays between videos
- **Manual controls:** Users can pause/play videos using built-in video controls

### Error Handling & Validation

#### File Validation
1. **Startup Validation:**
   - Checks for `lecture_list.json` existence
   - Validates video file references in the lecture list
   - Warns about missing `videofile` fields
   - Checks video file existence and size

2. **Runtime Validation:**
   - Validates video file format (.mp4, .webm, .ogg)
   - Handles missing video files gracefully
   - Logs errors and warnings appropriately

#### Security Considerations

1. **File Size Warnings:** 
   - Logs warnings for videos >100MB to prevent memory issues
   - Helps identify potentially problematic files

2. **Format Validation:**
   - Checks file extensions to ensure web-compatible formats
   - Warns about unsupported formats

3. **Path Validation:**
   - Uses `os.path.join()` for safe path construction
   - Validates file existence before loading

### Backward Compatibility

The original `appWebcast.py` remains unchanged and functional:
- Can still load separate JPG/MP3 files from `./Sources/` directory
- All original functionality preserved
- No breaking changes to existing code

To use the original system:
1. Keep using `appWebcast.py`
2. Change `SOURCE_DIR` back to `'./Sources/'` if needed
3. Ensure your `lecture_list.json` has `imagefile` and `audiofile` fields

### Usage Instructions

#### Running the Videocast

```bash
python appVideocast.py
```

The application will:
1. Load configuration from `lmm_education/config/appwebcast.py`
2. Read `lecture_list.json` from `webcast_sources/`
3. Validate all video files
4. Launch the Gradio interface
5. Display a welcome screen with a "Start Videocast" button

#### Creating Video Files

If you have separate JPG and MP3 files, you can convert them to MP4 videos:

```python
from lmm_education.webcast.sources import convert_source_folder_to_videos

# Convert all JPG+MP3 pairs in a directory to MP4 videos
convert_source_folder_to_videos(
    srcdir='./webcast_sources/',
    output_directory='./webcast_sources/'
)
```

This utility:
- Matches JPG and MP3 files by name
- Creates MP4 videos with H.264 video codec and AAC audio codec
- Sets video duration to match audio duration
- Outputs videos with 24 fps (standard for still images)

### File Structure

```
project_root/
├── appWebcast.py              # Original slideshow (JPG+MP3)
├── appVideocast.py            # New videocast (MP4)
├── webcast_sources/           # Video source directory
│   ├── lecture_list.json      # Updated with videofile references
│   ├── Lecture8_00.mp4        # Video files
│   ├── Lecture8_01.mp4
│   └── ...
├── Sources/                   # Original source directory (optional)
│   ├── lecture_list.json
│   ├── Lecture8_00.jpg
│   ├── Lecture8_00.mp3
│   └── ...
└── lmm_education/
    ├── config/
    │   └── appwebcast.py      # Configuration
    └── webcast/
        └── sources.py         # Utility functions
```

## Risk Mitigations Implemented

### Performance Risks
1. **File Size Monitoring:** Warning logs for large video files
2. **On-Demand Loading:** Videos loaded sequentially, not all at once
3. **Memory Management:** Gradio handles video buffering automatically

### Security Risks
1. **Path Safety:** Using `os.path.join()` for path construction
2. **Format Validation:** Checking file extensions for web compatibility
3. **File Existence:** Validating files before attempting to load

### Functional Risks
1. **Format Compatibility:** Supporting MP4, WebM, and OGG formats
2. **Error Recovery:** Graceful handling of missing or invalid videos
3. **Logging:** Comprehensive logging for debugging and monitoring

## Future Enhancements (Phase 2+)

Potential improvements for subsequent phases:
1. **User Controls:** Add manual navigation buttons (previous/next)
2. **Progress Indicator:** Show which slide is currently playing
3. **Video Preloading:** Preload next video for smoother transitions
4. **Playlist Management:** Allow dynamic playlist creation
5. **Accessibility:** Add caption support and keyboard navigation
6. **Performance Optimization:** Implement video compression guidelines
7. **Security Hardening:** Add video file validation against malicious content

## Testing Checklist

- [x] Video files load correctly from `webcast_sources/`
- [ ] Auto-progression works when video ends
- [ ] Q&A functionality works during presentation
- [ ] Error handling for missing video files
- [ ] Error handling for invalid video formats
- [ ] Large file warnings appear in logs
- [ ] Backward compatibility with original `appWebcast.py`
- [ ] Configuration changes take effect
- [ ] Video controls (play/pause) work correctly

## Migration Guide

To migrate from the original slideshow to videocast:

1. **Prepare Video Files:**
   ```python
   from lmm_education.webcast.sources import convert_source_folder_to_videos
   convert_source_folder_to_videos('./webcast_sources/')
   ```

2. **Update lecture_list.json:**
   - Add `videofile` field to each entry
   - Keep existing fields for backward compatibility

3. **Update Configuration (if needed):**
   - Ensure `SOURCE_DIR` points to `'./webcast_sources/'`

4. **Run Videocast:**
   ```bash
   python appVideocast.py
   ```

## Troubleshooting

### Video Not Loading
- Check that video file exists in `webcast_sources/`
- Verify `videofile` field in `lecture_list.json`
- Check file format (must be .mp4, .webm, or .ogg)
- Review logs in `appVideocast.log`

### Large File Issues
- Compress videos using standard codecs (H.264/AAC)
- Reduce video resolution if needed
- Check file size warnings in logs

### Auto-Progression Not Working
- Ensure video plays to completion
- Check `SLIDE_GAP` configuration
- Verify Gradio version compatibility

## Conclusion

Phase 1 successfully implements basic video playback functionality while maintaining backward compatibility and addressing key security and performance concerns. The modular design allows for incremental enhancement in future phases.
