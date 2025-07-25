# app.py - ClipMaker (Ultra-Light MoviePy Approach)
import os
import json
import tempfile
import traceback
import re
import streamlit as st
from moviepy.video.io.VideoFileClip import VideoFileClip
import gdown
import time
from openai import OpenAI
import gc

# ----------
# Helper Functions
# ----------

def get_api_key() -> str:
    """Retrieve the OpenAI API key from Streamlit secrets or environment."""
    if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
        return st.secrets["openai"]["api_key"]
    for key in ("OPENAI_API_KEY", "api_key"):
        if key in st.secrets:
            return st.secrets[key]
    return os.getenv("OPENAI_API_KEY", "")


def get_system_prompt(platform: str, selected_parameters: list, video_duration: float = None) -> str:
    duration_info = f"\n\nIMPORTANT: The video is {video_duration:.1f} seconds ({video_duration/60:.1f} minutes) long. All timestamps MUST be within 0 to {video_duration:.1f} seconds." if video_duration else ""
    
    # Build parameter descriptions
    parameter_descriptions = []
    for param in selected_parameters:
        if param == "Educational Value":
            parameter_descriptions.append("🧠 Educational Value: Clear insights, tips, or new perspectives")
        elif param == "Surprise Factor":
            parameter_descriptions.append("😲 Surprise Factor: Plot twists, myth-busting, or unexpected revelations")
        elif param == "Emotional Impact":
            parameter_descriptions.append("😍 Emotional Impact: Inspiration, humor, shock, or relatability")
        elif param == "Replayability":
            parameter_descriptions.append("🔁 Replayability: Content viewers want to watch multiple times")
        elif param == "Speaker Energy":
            parameter_descriptions.append("🎤 Speaker Energy: Passionate delivery, voice modulation")
        elif param == "Relatability":
            parameter_descriptions.append("🎯 Relatability: Reflects common struggles or experiences")
        elif param == "Contrarian Takes":
            parameter_descriptions.append("🔥 Contrarian Takes: Challenges popular beliefs")
        elif param == "Storytelling":
            parameter_descriptions.append("📖 Storytelling: Personal anecdotes, case studies")
    
    parameters_text = "\n".join(parameter_descriptions) if parameter_descriptions else "🎯 General viral potential"
    
    return f"""You are a content strategist for {platform}. Identify 20-59 second segments that will perform well as short-form content.

REQUIREMENTS:
🎯 Duration: 20-59 seconds each
🪝 Hook: Compelling 0-3 second opening that stops scrolling
🎬 Flow: Complete narrative arc with smooth ending
📱 Context: Must make sense without prior video context

FOCUS PARAMETERS:
{parameters_text}

{duration_info}

Return TOP 3 best segments as JSON array with these keys:
- start: "HH:MM:SS"
- end: "HH:MM:SS" 
- hook: "exact opening transcript text (first 0-3 seconds)"
- flow: "narrative arc summary"
- reason: "why this will go viral"
- score: integer (0-100)
- caption: "social media caption with emojis/hashtags"

Example:
[
  {{
    "start": "00:02:15",
    "end": "00:03:02",
    "hook": "This will change how you think about money forever",
    "flow": "Hook → reveals misconception → explains truth → gives actionable tip",
    "reason": "Myth-busting with strong educational value and immediate utility",
    "score": 88,
    "caption": "This money myth is costing you thousands 😱💰 #MoneyTips #Viral #FinanceTips"
  }}
]"""


def time_to_seconds(time_str: str) -> float:
    """Convert HH:MM:SS to seconds."""
    try:
        parts = time_str.split(':')
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        elif len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
        else:
            return float(parts[0])
    except:
        st.error(f"Could not parse time: {time_str}")
        return 0


def seconds_to_time(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def extract_audio_minimal(video_path: str) -> str:
    """Minimal audio extraction."""
    video = None
    audio = None
    try:
        st.info("🎵 Extracting audio...")
        
        # Create temp file
        audio_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        audio_temp.close()
        
        # Load and extract with minimal settings
        video = VideoFileClip(video_path)
        if video.audio is None:
            raise Exception("No audio track found")
        
        audio = video.audio
        audio.write_audiofile(audio_temp.name)
        
        return audio_temp.name
        
    finally:
        if audio:
            audio.close()
        if video:
            video.close()
        del audio, video
        gc.collect()


def transcribe_audio(video_path: str, client: OpenAI) -> str:
    """Simple transcription."""
    try:
        audio_path = extract_audio_minimal(video_path)
        
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        st.info(f"Audio file: {file_size_mb:.1f}MB")
        
        if file_size_mb > 25:
            st.warning(f"Large audio file ({file_size_mb:.1f}MB). May fail - try shorter video if needed.")
        
        with open(audio_path, "rb") as f:
            resp = client.audio.transcriptions.create(
                model="whisper-1", 
                file=f,
                response_format="text"
            )
        
        try:
            os.unlink(audio_path)
        except:
            pass
        
        return resp.strip()
        
    except Exception as e:
        st.error(f"Transcription failed: {str(e)}")
        raise


def analyze_transcript(transcript: str, platform: str, selected_parameters: list, client: OpenAI, video_duration: float = None) -> str:
    """Get segment suggestions."""
    messages = [
        {"role": "system", "content": get_system_prompt(platform, selected_parameters, video_duration)},
        {"role": "user", "content": f"""Analyze this transcript for {platform} segments based on: {', '.join(selected_parameters)}

CRITICAL: All timestamps must be within 0 to {video_duration:.1f} seconds.

Transcript:
{transcript}"""}
    ]
    
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=2000
        )
        return resp.choices[0].message.content
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        raise


def parse_segments(text: str, video_duration: float = None) -> list:
    """Parse and validate segments."""
    try:
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        segments = json.loads(text)
        
        valid_segments = []
        for i, seg in enumerate(segments):
            required_keys = ["start", "end", "hook", "flow", "reason", "score", "caption"]
            if all(key in seg for key in required_keys):
                try:
                    start_seconds = time_to_seconds(seg["start"])
                    end_seconds = time_to_seconds(seg["end"])
                    
                    if video_duration:
                        if start_seconds >= video_duration:
                            continue
                        if end_seconds > video_duration:
                            end_seconds = video_duration
                            seg["end"] = seconds_to_time(end_seconds)
                    
                    if start_seconds >= end_seconds:
                        continue
                    
                    duration = end_seconds - start_seconds
                    if duration < 20:
                        continue
                    
                    if duration > 59:
                        new_end = start_seconds + 59
                        if video_duration and new_end > video_duration:
                            new_end = video_duration
                        seg["end"] = seconds_to_time(new_end)
                    
                    valid_segments.append(seg)
                    
                except:
                    continue
        
        return valid_segments
    except json.JSONDecodeError as e:
        st.error(f"JSON parse error: {e}")
        return []


def create_single_clip_imageio_only(video_path: str, start_time: float, end_time: float, output_path: str) -> bool:
    """Create clip using only imageio and MoviePy frame extraction."""
    video = None
    clip = None
    
    try:
        st.info(f"Creating clip using imageio method...")
        
        # Load video with MoviePy (just for frame extraction)
        video = VideoFileClip(video_path)
        clip = video.subclipped(start_time, end_time)
        
        # Get video properties
        fps = clip.fps
        duration = clip.duration
        
        st.info(f"Extracting {duration:.1f}s at {fps} fps...")
        
        # Extract all frames at once
        frames = list(clip.iter_frames())
        
        if len(frames) == 0:
            st.error("No frames extracted")
            return False
        
        st.info(f"Extracted {len(frames)} frames, creating video...")
        
        # Use imageio to create video (more stable than MoviePy write)
        try:
            import imageio
            with imageio.get_writer(output_path, fps=fps) as writer:
                for frame in frames:
                    writer.append_data(frame)
            
            st.info(f"Video written with imageio")
            
        except ImportError:
            st.error("imageio not available, trying basic approach...")
            return create_single_clip_basic_moviepy(video_path, start_time, end_time, output_path)
        
        return os.path.isfile(output_path) and os.path.getsize(output_path) > 0
        
    except Exception as e:
        st.error(f"imageio method failed: {str(e)}")
        return create_single_clip_basic_moviepy(video_path, start_time, end_time, output_path)
        
    finally:
        if clip:
            try:
                clip.close()
            except:
                pass
        if video:
            try:
                video.close()
            except:
                pass
        del clip, video
        gc.collect()


def create_single_clip_basic_moviepy(video_path: str, start_time: float, end_time: float, output_path: str) -> bool:
    """Most basic MoviePy approach possible."""
    video = None
    clip = None
    
    try:
        st.info("Using most basic MoviePy approach...")
        
        # Load and create subclip
        video = VideoFileClip(video_path)
        clip = video.subclipped(start_time, end_time)
        
        # Write with absolutely minimal settings
        clip.write_videofile(output_path)
        
        return os.path.isfile(output_path) and os.path.getsize(output_path) > 0
        
    except Exception as e:
        st.error(f"Basic MoviePy failed: {str(e)}")
        return False
        
    finally:
        if clip:
            try:
                clip.close()
            except:
                pass
        if video:
            try:
                video.close()
            except:
                pass
        del clip, video
        gc.collect()


def create_clips_one_by_one(video_path: str, segments: list) -> list:
    """Create clips one by one with maximum isolation."""
    clips = []
    
    st.info("🎬 Creating clips with ultra-light processing...")
    
    for i, seg in enumerate(segments, start=1):
        try:
            start_time = time_to_seconds(seg.get("start", "0"))
            end_time = time_to_seconds(seg.get("end", "0"))
            
            st.info(f"Processing clip {i}: {start_time:.1f}s - {end_time:.1f}s")
            
            # Validate
            duration = end_time - start_time
            if duration < 1:
                st.warning(f"Skipping clip {i}: Too short")
                continue
            
            # Create temp file
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, 
                suffix=".mp4",
                prefix=f"clip_{i}_"
            )
            temp_file_path = temp_file.name
            temp_file.close()
            
            # Create clip with imageio method (no MoviePy write)
            success = create_single_clip_imageio_only(
                video_path, 
                start_time, 
                end_time, 
                temp_file_path
            )
            
            if success:
                clips.append({
                    "path": temp_file_path,
                    "caption": seg.get("caption", f"clip_{i}"),
                    "score": seg.get("score", 0),
                    "reason": seg.get("reason", ""),
                    "hook": seg.get("hook", ""),
                    "flow": seg.get("flow", ""),
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                    "duration": f"{duration:.1f}s",
                    "format": "imageio-generated Video",
                    "index": i
                })
                st.success(f"✅ Created clip {i} (imageio method)")
            else:
                st.error(f"❌ Failed to create clip {i}")
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
            
            # Force cleanup between clips
            gc.collect()
            time.sleep(1)
            
        except Exception as e:
            st.error(f"Error with clip {i}: {str(e)}")
            continue
    
    return clips


def download_drive_file(drive_url: str, out_path: str) -> str:
    """Download Google Drive file."""
    try:
        file_id = None
        if "id=" in drive_url:
            file_id = drive_url.split("id=")[1].split("&")[0]
        else:
            patterns = [
                r"/d/([a-zA-Z0-9_-]+)",
                r"id=([a-zA-Z0-9_-]+)",
                r"/file/d/([a-zA-Z0-9_-]+)"
            ]
            for pattern in patterns:
                m = re.search(pattern, drive_url)
                if m:
                    file_id = m.group(1)
                    break
        
        if not file_id:
            raise ValueError("Could not extract file ID")
        
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        result = gdown.download(download_url, out_path, quiet=True)
        
        if result and os.path.isfile(result) and os.path.getsize(result) > 0:
            return result
        else:
            raise Exception("Download failed")
        
    except Exception as e:
        raise Exception(str(e))


# ----------
# Streamlit App
# ----------

def main():
    st.set_page_config(page_title="ClipMaker", layout="wide")
    
    st.title("🎬 Long‑form to Short‑form ClipMaker")
    st.markdown("Transform your long-form content into viral short-form clips (20-59s) with compelling hooks!")

    st.warning("⚠️ **Optimized Processing**: Uses imageio for video generation to prevent MoviePy crashes.")
    st.info("✅ This approach extracts frames and rebuilds video files more reliably.")

    # Initialize session state
    for key in ['clips_generated', 'all_clips', 'processing_complete']:
        if key not in st.session_state:
            st.session_state[key] = False if 'complete' in key or 'generated' in key else []

    # API Key validation
    API_KEY = get_api_key()
    if not API_KEY:
        st.error("❌ OpenAI API key not found. Add it to Streamlit secrets or env var OPENAI_API_KEY.")
        return
    
    try:
        client = OpenAI(api_key=API_KEY)
    except Exception as e:
        st.error(f"❌ Error initializing OpenAI client: {str(e)}")
        return

    # Sidebar settings
    st.sidebar.header("⚙️ Settings")
    platform = st.sidebar.selectbox(
        "Target Platform", 
        ["YouTube Shorts", "Instagram Reels", "TikTok"],
        help="Choose platform to optimize for"
    )
    
    # Content focus
    st.sidebar.subheader("🎯 Content Focus")
    st.sidebar.caption("Select content types to prioritize")
    
    available_parameters = [
        "Educational Value", "Surprise Factor", "Emotional Impact",
        "Replayability", "Speaker Energy", "Relatability", 
        "Contrarian Takes", "Storytelling"
    ]
    
    selected_parameters = []
    for param in available_parameters:
        if st.sidebar.checkbox(param, key=f"param_{param}"):
            selected_parameters.append(param)
    
    if not selected_parameters:
        st.sidebar.warning("⚠️ Select at least one parameter")
    else:
        st.sidebar.success(f"✅ {len(selected_parameters)} selected")

    # Video source
    st.sidebar.subheader("📹 Video Source")
    uploaded = st.sidebar.file_uploader(
        "📁 Upload video file", 
        type=["mp4", "mov", "mkv", "avi", "webm"],
        help="Direct upload recommended"
    )
    
    video_path = None
    
    if uploaded:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
        tmp.write(uploaded.read())
        tmp.close()
        video_path = tmp.name
        
        st.session_state['video_path'] = video_path
        st.session_state['video_size'] = len(uploaded.getvalue()) / (1024 * 1024)
        
        # Reset processing state
        st.session_state.clips_generated = False
        st.session_state.all_clips = []
        st.session_state.processing_complete = False
        
        st.success(f"✅ Uploaded {st.session_state['video_size']:.1f}MB successfully!")
        
        if st.session_state['video_size'] <= 200:
            st.video(video_path)
        else:
            st.info("Large file uploaded. Skipping preview to save memory.")
    
    # Google Drive option
    if not uploaded:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Alternative: Google Drive**")
        
        drive_url = st.sidebar.text_input(
            "Google Drive URL", 
            placeholder="https://drive.google.com/file/d/...",
            help="May hit rate limits"
        )
        
        if drive_url and st.sidebar.button("📥 Download from Drive"):
            with st.spinner("Downloading from Google Drive..."):
                try:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    result = download_drive_file(drive_url, tmp.name)
                    
                    if result and os.path.isfile(result):
                        size_mb = os.path.getsize(result) / (1024 * 1024)
                        st.success(f"✅ Downloaded {size_mb:.2f} MB")
                        video_path = result
                        st.session_state['video_path'] = video_path
                        st.session_state['video_size'] = size_mb
                        
                        st.session_state.clips_generated = False
                        st.session_state.all_clips = []
                        st.session_state.processing_complete = False
                        
                        if size_mb <= 200:
                            st.video(video_path)
                except Exception as e:
                    st.error(f"❌ Download failed: {str(e)}")

    # Use video from session state
    if not video_path and 'video_path' in st.session_state:
        video_path = st.session_state['video_path']
        if os.path.isfile(video_path):
            st.info(f"Using previously loaded video ({st.session_state.get('video_size', 0):.1f} MB)")
            if st.session_state.get('video_size', 0) <= 200:
                st.video(video_path)
        else:
            st.warning("Previously loaded video no longer available. Please reload.")
            del st.session_state['video_path']
            video_path = None

    if not video_path:
        st.info("🎯 Upload a video file to begin.")
        return

    if not selected_parameters:
        st.warning("⚠️ Please select at least one content focus parameter.")
        return

    # Show generated clips if they exist
    if st.session_state.clips_generated and st.session_state.all_clips:
        st.markdown("---")
        st.header("🎬 Generated Clips")
        
        clips = st.session_state.all_clips
        if clips:
            st.success(f"🎉 Successfully generated {len(clips)} clips!")
            st.info("💡 **Note**: Clips generated without audio for stability. Add audio using any video editor.")
            
            # Summary stats
            total_duration = sum(float(c.get('duration', '0').replace('s', '')) for c in clips)
            total_size_mb = sum(os.path.getsize(c.get('path', '')) / (1024 * 1024) for c in clips if c.get('path') and os.path.isfile(c.get('path', '')))
            avg_score = sum(c.get('score', 0) for c in clips) / len(clips)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Generated", len(clips))
            col2.metric("Avg Score", f"{avg_score:.1f}/100")
            col3.metric("Total Duration", f"{total_duration:.1f}s")
            col4.metric("Total Size", f"{total_size_mb:.1f}MB")
        
        # Display clips
        for clip_info in clips:
            st.markdown(f"### 🎬 Clip {clip_info['index']} - Score: {clip_info['score']}/100")
            
            video_col, details_col = st.columns([1, 1.5])
            
            with video_col:
                if os.path.isfile(clip_info["path"]):
                    st.video(clip_info["path"])
                    st.caption("🎬 Generated with imageio")
                else:
                    st.error("❌ Clip file no longer available")
            
            with details_col:
                st.markdown("**📊 Clip Details**")
                detail_info = f"""
                ⏱️ **Duration:** {clip_info['duration']}  
                🕐 **Time:** {clip_info['start']} - {clip_info['end']}  
                🎯 **Score:** {clip_info['score']}/100  
                📱 **Format:** {clip_info['format']}
                """
                st.markdown(detail_info)
                
                with st.expander("📝 Suggested Caption", expanded=False):
                    st.code(clip_info["caption"], language="text")
                
                with st.expander("🪝 Hook (Exact Transcript)", expanded=False):
                    st.write(f"**Starting words:** {clip_info['hook']}")
                
                with st.expander("🎬 Content Flow", expanded=False):
                    st.write(clip_info['flow'])
                
                with st.expander("💡 Viral Potential", expanded=False):
                    st.write(clip_info['reason'])
                
                st.markdown("---")
                if os.path.isfile(clip_info["path"]):
                    with open(clip_info["path"], "rb") as file:
                        download_key = f"download_{clip_info['index']}_{hash(clip_info['path'])}"
                        st.download_button(
                            label="⬇️ Download Video Clip",
                            data=file,
                            file_name=f"clip_{clip_info['index']}_{platform.replace(' ', '_').lower()}_no_audio.mp4",
                            mime="video/mp4",
                            use_container_width=True,
                            type="primary",
                            key=download_key
                        )
                else:
                    st.error("❌ File no longer available")
            
            st.markdown("---")

        # Instructions
        st.markdown("---")
        st.header("🔧 Adding Audio Back")
        st.info("""
        **To add audio back to your clips:**
        
        1. **Download the video clips** above (visual content only)
        2. **Use any video editor** (CapCut, DaVinci Resolve, Adobe Premiere, etc.)
        3. **Import both** the original video and the downloaded clip
        4. **Extract audio** from the original video at the same timestamps
        5. **Sync the audio** with the video clip
        6. **Export** your final clip with audio
        
        **Timestamps are provided** in each clip's details for easy reference!
        """)

        # Alternative: Provide audio extraction timestamps
        st.markdown("### 📋 Audio Extraction Reference")
        for i, clip in enumerate(clips, 1):
            st.code(f"Clip {i}: Extract audio from {clip['start']} to {clip['end']}", language="text")
        # Reset button
        if st.button("🔄 Clear All Clips & Start Over", type="secondary"):
            for clip in st.session_state.all_clips:
                try:
                    if clip.get("path") and os.path.isfile(clip["path"]):
                        os.unlink(clip["path"])
                except:
                    pass
            
            st.session_state.clips_generated = False
            st.session_state.all_clips = []
            st.session_state.processing_complete = False
            st.rerun()

        return

    # Main processing
    if not st.session_state.processing_complete:
        if st.button("🚀 Generate Clips", type="primary"):
            if not video_path or not os.path.isfile(video_path):
                st.error("Video file not found. Please reload.")
                return
                
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Get video info
                status_text.text("📹 Analyzing video...")
                progress_bar.progress(10)
                
                temp_video = VideoFileClip(video_path)
                video_duration = temp_video.duration
                video_width = temp_video.w
                video_height = temp_video.h
                temp_video.close()
                del temp_video
                gc.collect()
                
                st.success(f"✅ Video: {video_width}x{video_height}, {video_duration:.1f}s")
                
                # Step 2: Transcription
                status_text.text("🎤 Transcribing audio...")
                progress_bar.progress(25)
                
                transcript = transcribe_audio(video_path, client)
                st.success("✅ Transcription complete")
                
                progress_bar.progress(50)
                with st.expander("📄 Transcript Preview", expanded=False):
                    st.text_area("Full Transcript", transcript, height=200, disabled=True)

                # Step 3: AI analysis
                status_text.text("🤖 Analyzing for viral segments...")
                progress_bar.progress(75)
                
                ai_json = analyze_transcript(transcript, platform, selected_parameters, client, video_duration)
                st.success("✅ Analysis complete")

                with st.expander("🔍 AI Analysis Output", expanded=False):
                    st.code(ai_json, language="json")

                # Step 4: Parse segments
                status_text.text("📊 Processing segments...")
                progress_bar.progress(90)
                
                segments = parse_segments(ai_json, video_duration)
                if not segments:
                    st.warning("⚠️ No valid segments found.")
                    return
                    
                segments_sorted = sorted(segments, key=lambda x: x.get('score', 0), reverse=True)
                
                # Step 5: Generate clips
                status_text.text("✂️ Creating video clips...")
                progress_bar.progress(95)
                
                st.success(f"🎯 Found {len(segments_sorted)} segments! Generating clips...")

                st.markdown("---")
                st.header("🎬 Creating Clips")
                st.info("🚀 Generating clips using imageio method")
                
                all_clips = create_clips_one_by_one(video_path, segments_sorted)
                
                if all_clips:
                    st.session_state.all_clips = all_clips
                    st.session_state.clips_generated = True
                    st.session_state.processing_complete = True
                    
                    progress_bar.progress(100)
                    status_text.text("✅ All clips generated successfully!")
                    
                    st.success(f"🎉 Generated {len(all_clips)} video clips!")
                    
                    gc.collect()
                    st.rerun()
                else:
                    st.warning("❌ No clips were generated successfully.")
                    return
                    
            except Exception as e:
                st.error("❌ Processing failed")
                st.error(f"Error: {str(e)}")
                
                with st.expander("🔍 Error Details", expanded=False):
                    st.code(traceback.format_exc())
                
                return

    # Sidebar reset
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Start Over", help="Clear all data"):
        if 'all_clips' in st.session_state:
            for clip in st.session_state.all_clips:
                try:
                    if clip.get("path") and os.path.isfile(clip["path"]):
                        os.unlink(clip["path"])
                except:
                    pass
        
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        gc.collect()
        st.rerun()


if __name__ == "__main__":
    main()
