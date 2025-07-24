# app.py - ClipMaker (Cloud-Compatible Version - No ffmpeg Required)
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

Return TOP 5 best segments as JSON array with these keys:
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


def extract_audio_moviepy(video_path: str) -> str:
    """Extract audio using MoviePy with minimal settings."""
    try:
        st.info("🎵 Extracting audio from video...")
        
        # Create temporary audio file
        audio_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        audio_temp.close()
        
        # Load video and extract audio with minimal settings
        video = VideoFileClip(video_path)
        audio = video.audio
        
        # Write with basic settings only
        audio.write_audiofile(audio_temp.name)
        
        # Clean up immediately
        audio.close()
        video.close()
        del video, audio
        gc.collect()
        
        return audio_temp.name
    except Exception as e:
        st.error(f"Audio extraction failed: {str(e)}")
        raise


def split_audio_moviepy(audio_path: str, chunk_minutes: int = 10) -> list:
    """Split audio using MoviePy if file is too large."""
    try:
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        
        if file_size_mb <= 20:
            return [audio_path]
        
        st.info(f"Audio file is {file_size_mb:.1f}MB. Splitting into chunks...")
        
        from moviepy.audio.io.AudioFileClip import AudioFileClip
        audio_clip = AudioFileClip(audio_path)
        duration = audio_clip.duration
        
        chunk_duration = chunk_minutes * 60
        chunks = []
        
        start_time = 0
        chunk_num = 1
        
        while start_time < duration:
            end_time = min(start_time + chunk_duration, duration)
            
            chunk_temp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_chunk_{chunk_num}.mp3")
            chunk_temp.close()
            
            chunk_audio = audio_clip.subclip(start_time, end_time)
            chunk_audio.write_audiofile(chunk_temp.name)
            
            chunks.append(chunk_temp.name)
            chunk_audio.close()
            
            start_time = end_time
            chunk_num += 1
        
        audio_clip.close()
        del audio_clip
        gc.collect()
        
        st.success(f"Split audio into {len(chunks)} chunks")
        return chunks
        
    except Exception as e:
        st.error(f"Audio splitting failed: {str(e)}")
        raise


def transcribe_audio(video_path: str, client: OpenAI) -> str:
    """Transcribe audio using MoviePy extraction + OpenAI Whisper."""
    try:
        # Extract audio
        audio_path = extract_audio_moviepy(video_path)
        
        # Check file size and split if necessary
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        st.info(f"Audio file size: {file_size_mb:.1f}MB")
        
        # Split if too large
        audio_chunks = split_audio_moviepy(audio_path) if file_size_mb > 20 else [audio_path]
        
        # Transcribe chunks
        full_transcript = ""
        
        if len(audio_chunks) > 1:
            st.info(f"Transcribing {len(audio_chunks)} audio chunks...")
            progress_bar = st.progress(0)
            
            for i, chunk_path in enumerate(audio_chunks):
                st.info(f"Transcribing chunk {i+1}/{len(audio_chunks)}...")
                
                try:
                    with open(chunk_path, "rb") as f:
                        resp = client.audio.transcriptions.create(
                            model="whisper-1", 
                            file=f,
                            response_format="text"
                        )
                    
                    chunk_transcript = resp.strip()
                    if chunk_transcript:
                        full_transcript += chunk_transcript + " "
                    
                    progress_bar.progress((i + 1) / len(audio_chunks))
                    
                except Exception as e:
                    st.warning(f"Failed to transcribe chunk {i+1}: {str(e)}")
                
                # Clean up chunk
                try:
                    os.unlink(chunk_path)
                except:
                    pass
        else:
            # Single file transcription
            with open(audio_chunks[0], "rb") as f:
                resp = client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=f,
                    response_format="text"
                )
            full_transcript = resp
        
        # Clean up main audio file
        try:
            os.unlink(audio_path)
        except:
            pass
        
        return full_transcript.strip()
        
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        raise


def analyze_transcript(transcript: str, platform: str, selected_parameters: list, client: OpenAI, video_duration: float = None) -> str:
    """Get segment suggestions via ChatCompletion."""
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
    """Parse JSON segments and validate."""
    try:
        # Clean text
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        segments = json.loads(text)
        
        # Validate segments
        valid_segments = []
        for i, seg in enumerate(segments):
            required_keys = ["start", "end", "hook", "flow", "reason", "score", "caption"]
            if all(key in seg for key in required_keys):
                try:
                    start_seconds = time_to_seconds(seg["start"])
                    end_seconds = time_to_seconds(seg["end"])
                    
                    # Validate against video duration
                    if video_duration:
                        if start_seconds >= video_duration:
                            st.warning(f"Segment {i+1}: Start time exceeds video duration. Skipping.")
                            continue
                        if end_seconds > video_duration:
                            st.warning(f"Segment {i+1}: Adjusting end time to video duration.")
                            end_seconds = video_duration
                            seg["end"] = seconds_to_time(end_seconds)
                    
                    # Validate duration
                    if start_seconds >= end_seconds:
                        st.warning(f"Segment {i+1}: Invalid time range. Skipping.")
                        continue
                    
                    duration = end_seconds - start_seconds
                    if duration < 20:
                        st.warning(f"Segment {i+1}: Duration too short ({duration:.1f}s). Skipping.")
                        continue
                    
                    if duration > 59:
                        st.warning(f"Segment {i+1}: Adjusting duration to 59 seconds.")
                        new_end_seconds = start_seconds + 59
                        if video_duration and new_end_seconds > video_duration:
                            new_end_seconds = video_duration
                        seg["end"] = seconds_to_time(new_end_seconds)
                    
                    valid_segments.append(seg)
                    
                except Exception:
                    st.warning(f"Segment {i+1}: Invalid timestamp format. Skipping.")
                    continue
            else:
                st.warning(f"Segment {i+1}: Missing required fields. Skipping.")
        
        return valid_segments
    except json.JSONDecodeError as e:
        st.error(f"JSON parse error: {e}")
        st.error(f"Raw text: {text[:500]}...")
        return []


def download_drive_file(drive_url: str, out_path: str) -> str:
    """Download Google Drive file."""
    try:
        # Extract file ID
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
        
        # Download
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

    # Show mode info
    st.info("🎯 **Cloud Mode**: Provides viral segment analysis + timestamps for manual clip creation")
    st.warning("💡 **Note**: Automatic clip generation requires ffmpeg (not available in cloud). You'll get precise timestamps to create clips manually.")

    # Initialize session state
    for key in ['clips_analyzed', 'segments_data', 'processing_complete']:
        if key not in st.session_state:
            st.session_state[key] = False if 'complete' in key or 'analyzed' in key else []

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
        # Save uploaded file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
        tmp.write(uploaded.read())
        tmp.close()
        video_path = tmp.name
        
        st.session_state['video_path'] = video_path
        st.session_state['video_size'] = len(uploaded.getvalue()) / (1024 * 1024)
        
        # Reset processing state
        st.session_state.clips_analyzed = False
        st.session_state.segments_data = []
        st.session_state.processing_complete = False
        
        st.success(f"✅ Uploaded {st.session_state['video_size']:.1f}MB successfully!")
        
        # Show preview for smaller files
        if st.session_state['video_size'] <= 100:
            st.video(video_path)
        else:
            st.info("Large file uploaded. Skipping preview to conserve resources.")
    
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
                        
                        # Reset processing
                        st.session_state.clips_analyzed = False
                        st.session_state.segments_data = []
                        st.session_state.processing_complete = False
                        
                        if size_mb <= 100:
                            st.video(video_path)
                except Exception as e:
                    st.error(f"❌ Download failed: {str(e)}")

    # Use video from session state
    if not video_path and 'video_path' in st.session_state:
        video_path = st.session_state['video_path']
        if os.path.isfile(video_path):
            st.info(f"Using previously loaded video ({st.session_state.get('video_size', 0):.1f} MB)")
            if st.session_state.get('video_size', 0) <= 100:
                st.video(video_path)
        else:
            st.warning("Previously loaded video no longer available. Please reload.")
            del st.session_state['video_path']
            video_path = None

    if not video_path:
        st.info("🎯 Upload a video file to begin analysis.")
        return

    if not selected_parameters:
        st.warning("⚠️ Please select at least one content focus parameter.")
        return

    # Show analyzed segments if they exist
    if st.session_state.clips_analyzed and st.session_state.segments_data:
        st.markdown("---")
        st.header("🎯 Viral Segment Analysis")
        
        segments = st.session_state.segments_data
        if segments:
            st.success(f"🎉 Found {len(segments)} viral segments!")
            
            # Summary stats
            avg_score = sum(s.get('score', 0) for s in segments) / len(segments)
            total_duration = sum(time_to_seconds(s.get('end', '0')) - time_to_seconds(s.get('start', '0')) for s in segments)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Segments Found", len(segments))
            col2.metric("Avg Score", f"{avg_score:.1f}/100")
            col3.metric("Total Duration", f"{total_duration:.1f}s")
        
        # Display segments
        for i, segment in enumerate(segments, 1):
            start_seconds = time_to_seconds(segment.get('start', '0'))
            end_seconds = time_to_seconds(segment.get('end', '0'))
            duration = end_seconds - start_seconds
            
            st.markdown(f"### 🎬 Segment {i} - Score: {segment.get('score', 0)}/100")
            
            # Create columns
            timing_col, details_col = st.columns([1, 2])
            
            with timing_col:
                st.markdown("**⏱️ Timing**")
                st.code(f"""
Start: {segment.get('start', 'N/A')}
End: {segment.get('end', 'N/A')}
Duration: {duration:.1f}s
                """)
                
                # Quick copy buttons
                st.markdown("**📋 Quick Copy**")
                col_start, col_end = st.columns(2)
                with col_start:
                    st.code(segment.get('start', ''), language="text")
                with col_end:
                    st.code(segment.get('end', ''), language="text")
            
            with details_col:
                # Expandable sections
                with st.expander("🪝 Hook (Opening)", expanded=True):
                    st.write(f"**First 3 seconds:** {segment.get('hook', 'N/A')}")
                
                with st.expander("🎬 Content Flow", expanded=False):
                    st.write(segment.get('flow', 'N/A'))
                
                with st.expander("💡 Why This Will Go Viral", expanded=False):
                    st.write(segment.get('reason', 'N/A'))
                
                with st.expander("📝 Suggested Caption", expanded=False):
                    st.code(segment.get('caption', 'N/A'), language="text")
            
            st.markdown("---")

        # Instructions for manual clip creation
        st.markdown("---")
        st.header("📝 How to Create Clips")
        
        st.info("""
        **Manual Clip Creation Instructions:**
        
        1. **Use any video editor** (CapCut, DaVinci Resolve, Adobe Premiere, etc.)
        2. **Import your video** into the editor
        3. **Use the timestamps above** to cut each segment
        4. **Start with the hook** - ensure the opening 3 seconds are compelling
        5. **Export as MP4** in your desired format (9:16 for mobile, 16:9 for YouTube)
        
        **Pro Tips:**
        - Add captions/subtitles for better engagement
        - Use the suggested social media captions
        - Test different thumbnails for each clip
        - Post at optimal times for your audience
        """)

        # Reset button
        if st.button("🔄 Analyze New Video", type="secondary"):
            # Reset session state
            st.session_state.clips_analyzed = False
            st.session_state.segments_data = []
            st.session_state.processing_complete = False
            st.rerun()

        return

    # Main processing
    if not st.session_state.processing_complete:
        if st.button("🚀 Analyze Video for Viral Segments", type="primary"):
            if not video_path or not os.path.isfile(video_path):
                st.error("Video file not found. Please reload.")
                return
                
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Get video info
                status_text.text("📹 Analyzing video...")
                progress_bar.progress(10)
                
                video = VideoFileClip(video_path)
                video_duration = video.duration
                video_width = video.w
                video_height = video.h
                video.close()
                del video
                gc.collect()
                
                st.success(f"✅ Video: {video_width}x{video_height}, {video_duration:.1f}s")
                
                # Step 2: Transcription
                status_text.text("🎤 Transcribing audio...")
                progress_bar.progress(25)
                
                transcript = transcribe_audio(video_path, client)
                st.success("✅ Transcription complete")
                
                # Show transcript
                progress_bar.progress(50)
                with st.expander("📄 Transcript Preview", expanded=False):
                    st.text_area("Full Transcript", transcript, height=200, disabled=True)

                # Step 3: AI analysis
                status_text.text("🤖 Analyzing for viral segments...")
                progress_bar.progress(75)
                
                ai_json = analyze_transcript(transcript, platform, selected_parameters, client, video_duration)
                st.success("✅ Analysis complete")

                # Show AI output
                with st.expander("🔍 AI Analysis Output", expanded=False):
                    st.code(ai_json, language="json")

                # Step 4: Parse segments
                status_text.text("📊 Processing segments...")
                progress_bar.progress(90)
                
                segments = parse_segments(ai_json, video_duration)
                if not segments:
                    st.warning("⚠️ No valid segments found.")
                    progress_bar.progress(100)
                    status_text.text("❌ No segments found")
                    return
                    
                # Sort by score
                segments_sorted = sorted(segments, key=lambda x: x.get('score', 0), reverse=True)
                
                # Store results
                st.session_state.segments_data = segments_sorted
                st.session_state.clips_analyzed = True
                st.session_state.processing_complete = True
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                
                st.success(f"🎉 Found {len(segments_sorted)} viral segments!")
                
                # Trigger rerun to show results
                st.rerun()
                    
            except Exception as e:
                st.error("❌ Analysis failed")
                st.error(f"Error: {str(e)}")
                
                with st.expander("🔍 Error Details", expanded=False):
                    st.code(traceback.format_exc())
                
                progress_bar.progress(0)
                status_text.text("❌ Analysis failed")
                return

    # Sidebar reset
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Start Over", help="Clear all data"):
        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        gc.collect()
        st.rerun()


if __name__ == "__main__":
    main()
