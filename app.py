# app.py - MoviePy ClipMaker (Ultra-Stable Version)
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
import subprocess
import shutil

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
    duration_info = f"\n\nIMPORTANT: The video is {video_duration:.1f} seconds ({video_duration/60:.1f} minutes) long. All timestamps MUST be within 0 to {video_duration:.1f} seconds. Do not generate any timestamps beyond this range." if video_duration else ""
    
    # Build parameter descriptions based on user selection
    parameter_descriptions = []
    for param in selected_parameters:
        if param == "Educational Value":
            parameter_descriptions.append("🧠 Educational Value: Clear insights, tips, or new perspectives delivered quickly")
        elif param == "Surprise Factor":
            parameter_descriptions.append("😲 Surprise Factor: Plot twists, myth-busting, or unexpected revelations")
        elif param == "Emotional Impact":
            parameter_descriptions.append("😍 Emotional Impact: Inspiration, humor, shock, or relatability that drives engagement")
        elif param == "Replayability":
            parameter_descriptions.append("🔁 Replayability: Content viewers want to watch multiple times or share")
        elif param == "Speaker Energy":
            parameter_descriptions.append("🎤 Speaker Energy: Passionate delivery, voice modulation, natural pauses")
        elif param == "Relatability":
            parameter_descriptions.append("🎯 Relatability: Reflects common struggles, desires, or experiences")
        elif param == "Contrarian Takes":
            parameter_descriptions.append("🔥 Contrarian Takes: Challenges popular beliefs or conventional wisdom")
        elif param == "Storytelling":
            parameter_descriptions.append("📖 Storytelling: Personal anecdotes, case studies, or narrative elements")
    
    parameters_text = "\n".join(parameter_descriptions) if parameter_descriptions else "🎯 General viral potential focusing on engagement and shareability"
    
    return f"""You are a content strategist and social media editor trained to analyze long-form video/podcast transcripts. Your task is to identify 20-59 second segments that are highly likely to perform well as short-form content on {platform}.

CRITICAL REQUIREMENTS:
🎯 Duration: Each clip must be 20-59 seconds (no shorter, no longer)
🪝 Hook: Every clip MUST start with a compelling 0-3 second hook that stops scrolling
🎬 Flow: Complete narrative arc with smooth, non-abrupt ending
📱 Context: Must make sense without prior video context

SELECTED FOCUS PARAMETERS:
{parameters_text}

HOOK REQUIREMENTS (First 0-3 seconds):
The hook is the exact transcript text where the clip should START. It must be:
- A shocking statement, question, bold claim, or attention-grabbing phrase
- The actual words spoken at the beginning of the segment
- Something that makes viewers stop scrolling immediately

HOOK EXAMPLES:
- "This will change how you think about..."
- "Nobody talks about this, but..."
- "I made a $10,000 mistake so you don't have to..."
- "The thing they don't tell you is..."
- "Here's what 99% of people get wrong..."

{duration_info}

CRITICAL: Since you don't have access to actual timestamps, estimate reasonable time intervals based on content flow and speech patterns. Assume average speaking pace of 150-200 words per minute.

For each recommended cut, provide:
1. Start and end timestamps (HH:MM:SS format) - MUST be within video duration
2. Hook text (the exact transcript words that should start the clip)
3. Content flow (brief summary of 20-59 second narrative)
4. Reason for virality (based on selected parameters above)
5. Predicted engagement score (0–100) — confidence in performance
6. Suggested caption for social media with emojis/hashtags

IMPORTANT: Only return the TOP 3 best segments to minimize processing load.

Output ONLY valid JSON as an array of objects with these exact keys:
- start: "HH:MM:SS"
- end: "HH:MM:SS" 
- hook: "exact transcript text that starts the clip (first 0-3 seconds)"
- flow: "brief narrative arc of the full 20-59 second clip"
- reason: "why this will go viral focusing on selected parameters"
- score: integer (0-100)
- caption: "social media caption with emojis and hashtags"

Example format:
[
  {{
    "start": "00:02:15",
    "end": "00:03:02",
    "hook": "This will change how you think about credit scores forever",
    "flow": "Hook → explains common misconception → reveals truth → actionable tip → strong finish",
    "reason": "Myth-busting content with strong hook, educational value, and shareable insight that challenges assumptions",
    "score": 88,
    "caption": "This credit score myth is costing you money 😱 #MoneyMyths #FinanceTips #Viral"
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


def check_ffmpeg_available() -> bool:
    """Check if ffmpeg is available on the system."""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except:
        return False


def extract_audio_with_ffmpeg(video_path: str) -> str:
    """Extract audio using ffmpeg - more stable than MoviePy."""
    try:
        # Create temporary audio file
        audio_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        audio_temp.close()
        
        # Use ffmpeg to extract audio
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn',  # No video
            '-acodec', 'mp3',
            '-ab', '64k',  # Low bitrate
            '-y',  # Overwrite output
            audio_temp.name
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0 and os.path.isfile(audio_temp.name):
            return audio_temp.name
        else:
            raise Exception(f"ffmpeg failed: {result.stderr}")
            
    except Exception as e:
        st.error(f"FFmpeg audio extraction failed: {str(e)}")
        raise


def extract_audio_from_video(video_path: str) -> str:
    """Extract audio from video - try ffmpeg first, fallback to MoviePy."""
    # Try ffmpeg first (more stable)
    if check_ffmpeg_available():
        try:
            st.info("Using ffmpeg for audio extraction...")
            return extract_audio_with_ffmpeg(video_path)
        except Exception as e:
            st.warning(f"ffmpeg failed, falling back to MoviePy: {str(e)}")
    
    # Fallback to MoviePy (less stable but available)
    try:
        st.info("Using MoviePy for audio extraction...")
        # Create temporary audio file
        audio_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        audio_temp.close()
        
        # Load video and extract audio with minimal settings
        video = VideoFileClip(video_path)
        audio = video.audio
        
        # Write with minimal options
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


def split_audio_file(audio_path: str, chunk_duration_minutes: int = 10) -> list:
    """Split audio file into smaller chunks if it's too large."""
    try:
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        
        # If file is small enough, return as is
        if file_size_mb <= 20:
            return [audio_path]
        
        st.info(f"Audio file is {file_size_mb:.1f}MB. Splitting into chunks...")
        
        # Try ffmpeg first
        if check_ffmpeg_available():
            return split_audio_with_ffmpeg(audio_path, chunk_duration_minutes)
        
        # Fallback to MoviePy
        from moviepy.audio.io.AudioFileClip import AudioFileClip
        audio_clip = AudioFileClip(audio_path)
        duration = audio_clip.duration
        
        chunk_duration_seconds = chunk_duration_minutes * 60
        chunks = []
        
        start_time = 0
        chunk_num = 1
        
        while start_time < duration:
            end_time = min(start_time + chunk_duration_seconds, duration)
            
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


def split_audio_with_ffmpeg(audio_path: str, chunk_duration_minutes: int) -> list:
    """Split audio using ffmpeg."""
    try:
        # Get audio duration first
        cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', 
               '-of', 'csv=p=0', audio_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = float(result.stdout.strip())
        
        chunk_duration_seconds = chunk_duration_minutes * 60
        chunks = []
        
        start_time = 0
        chunk_num = 1
        
        while start_time < duration:
            end_time = min(start_time + chunk_duration_seconds, duration)
            chunk_duration = end_time - start_time
            
            chunk_temp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_chunk_{chunk_num}.mp3")
            chunk_temp.close()
            
            cmd = [
                'ffmpeg', '-ss', str(start_time), '-i', audio_path,
                '-t', str(chunk_duration), '-acodec', 'copy',
                '-y', chunk_temp.name
            ]
            
            subprocess.run(cmd, capture_output=True)
            chunks.append(chunk_temp.name)
            
            start_time = end_time
            chunk_num += 1
        
        return chunks
    except Exception as e:
        raise Exception(f"FFmpeg audio splitting failed: {str(e)}")


def transcribe_audio(path: str, client: OpenAI) -> str:
    """Transcribe audio via Whisper-1, handling large files by chunking."""
    try:
        # Extract and compress audio from video
        st.info("🎵 Extracting audio from video...")
        audio_path = extract_audio_from_video(path)
        
        # Check file size and split if necessary
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        st.info(f"Audio file size: {file_size_mb:.1f}MB")
        
        # Split audio if too large
        audio_chunks = split_audio_file(audio_path)
        
        # Transcribe each chunk
        full_transcript = ""
        
        if len(audio_chunks) > 1:
            st.info(f"Transcribing {len(audio_chunks)} audio chunks...")
            progress_bar = st.progress(0)
            
            for i, chunk_path in enumerate(audio_chunks):
                st.info(f"Transcribing chunk {i+1}/{len(audio_chunks)}...")
                
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
                
                # Clean up chunk file immediately
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
        
        # Clean up audio file
        try:
            os.unlink(audio_path)
        except:
            pass
        
        return full_transcript.strip()
        
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        
        # Provide helpful error message based on error type
        error_str = str(e).lower()
        if "413" in error_str or "too large" in error_str:
            st.error("💡 The audio file is too large. Try using a shorter video or check if the file downloaded correctly.")
        elif "401" in error_str or "unauthorized" in error_str:
            st.error("💡 Check your OpenAI API key permissions and credits.")
        elif "429" in error_str or "rate limit" in error_str:
            st.error("💡 Rate limit exceeded. Please wait a moment and try again.")
        
        raise


def analyze_transcript(transcript: str, platform: str, selected_parameters: list, client: OpenAI, video_duration: float = None) -> str:
    """Get segment suggestions via ChatCompletion."""
    # Calculate rough transcript sections for better timestamp estimation
    transcript_length = len(transcript.split())
    words_per_section = max(1, transcript_length // 5)  # Prevent division by zero
    
    transcript_context = ""
    if video_duration and transcript_length > 0:
        transcript_context = f"""
TRANSCRIPT TIMING CONTEXT:
- Total transcript words: {transcript_length}
- Video duration: {video_duration:.1f} seconds ({int(video_duration//60)}:{int(video_duration%60):02d})
- Speaking rate: ~{transcript_length/(video_duration/60):.0f} words per minute

SECTION TIMING GUIDE:
- Words 1-{words_per_section}: Early timestamps (00:00:30 - {int((video_duration*0.2)//60):02d}:{int((video_duration*0.2)%60):02d})
- Words {words_per_section+1}-{words_per_section*2}: Early-mid timestamps ({int((video_duration*0.2)//60):02d}:{int((video_duration*0.2)%60):02d} - {int((video_duration*0.4)//60):02d}:{int((video_duration*0.4)%60):02d})
- Words {words_per_section*2+1}-{words_per_section*3}: Mid timestamps ({int((video_duration*0.4)//60):02d}:{int((video_duration*0.4)%60):02d} - {int((video_duration*0.6)//60):02d}:{int((video_duration*0.6)%60):02d})
- Words {words_per_section*3+1}-{words_per_section*4}: Late-mid timestamps ({int((video_duration*0.6)//60):02d}:{int((video_duration*0.6)%60):02d} - {int((video_duration*0.8)//60):02d}:{int((video_duration*0.8)%60):02d})
- Words {words_per_section*4+1}-{transcript_length}: Late timestamps ({int((video_duration*0.8)//60):02d}:{int((video_duration*0.8)%60):02d} - {int(max(0, video_duration-60)//60):02d}:{int(max(0, video_duration-60)%60):02d})

Use this guide to estimate where content appears in the video timeline.
"""
    
    messages = [
        {"role": "system", "content": get_system_prompt(platform, selected_parameters, video_duration)},
        {"role": "user", "content": f"""Analyze this transcript and identify the best segments for {platform} based on the selected parameters: {', '.join(selected_parameters)}. 

{transcript_context}

Focus on segments with powerful hooks in the first 3 seconds and proper endings. 

CRITICAL: All timestamps must be within 0 to {video_duration:.1f} seconds. Use the section timing guide above to estimate realistic timestamps based on where content appears in the transcript.

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
    """Parse JSON text into a list of segments and validate timestamps."""
    try:
        # Clean the text in case there's extra content
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
                # Additional timestamp validation
                try:
                    start_seconds = time_to_seconds(seg["start"])
                    end_seconds = time_to_seconds(seg["end"])
                    
                    # Check if timestamps are within video duration
                    if video_duration:
                        if start_seconds >= video_duration:
                            st.warning(f"Segment {i+1}: Start time ({seg['start']}) exceeds video duration ({video_duration:.1f}s). Skipping.")
                            continue
                        if end_seconds > video_duration:
                            st.warning(f"Segment {i+1}: End time ({seg['end']}) exceeds video duration. Adjusting to video end.")
                            # Adjust end time to video duration
                            end_minutes = int(video_duration // 60)
                            end_secs = int(video_duration % 60)
                            seg["end"] = f"{end_minutes//60:02d}:{end_minutes%60:02d}:{end_secs:02d}"
                            end_seconds = video_duration
                    
                    # Check if start < end and minimum duration
                    if start_seconds >= end_seconds:
                        st.warning(f"Segment {i+1}: Invalid time range ({seg['start']} >= {seg['end']}). Skipping.")
                        continue
                    
                    # Check duration is 20-59 seconds
                    duration = end_seconds - start_seconds
                    if duration < 20:
                        st.warning(f"Segment {i+1}: Duration too short ({duration:.1f}s < 20s). Skipping.")
                        continue
                    
                    if duration > 59:
                        st.warning(f"Segment {i+1}: Duration too long ({duration:.1f}s > 59s). Adjusting.")
                        # Adjust end time to 59 seconds from start
                        new_end_seconds = start_seconds + 59
                        if video_duration and new_end_seconds > video_duration:
                            new_end_seconds = video_duration
                        end_minutes = int(new_end_seconds // 60)
                        end_secs = int(new_end_seconds % 60)
                        seg["end"] = f"{end_minutes//60:02d}:{end_minutes%60:02d}:{end_secs:02d}"
                    
                    valid_segments.append(seg)
                    
                except Exception as time_error:
                    st.warning(f"Segment {i+1}: Invalid timestamp format. Skipping. Error: {time_error}")
                    continue
            else:
                st.warning(f"Skipping invalid segment {i+1}: Missing required fields")
        
        return valid_segments
    except json.JSONDecodeError as e:
        st.error(f"JSON parse error: {e}")
        st.error(f"Raw text received: {text[:500]}...")
        return []


def generate_clips_with_ffmpeg(video_path: str, segments: list) -> list:
    """Generate clips using ffmpeg - much more stable than MoviePy."""
    clips = []
    
    if not check_ffmpeg_available():
        raise Exception("ffmpeg not available - cannot generate clips")
    
    try:
        for i, seg in enumerate(segments, start=1):
            try:
                start_time = time_to_seconds(seg.get("start", "0"))
                end_time = time_to_seconds(seg.get("end", "0"))
                caption = seg.get("caption", f"clip_{i}")
                score = seg.get("score", 0)
                reason = seg.get("reason", "")
                hook = seg.get("hook", "")
                flow = seg.get("flow", "")
                
                st.info(f"Processing clip {i} with ffmpeg: {start_time:.1f}s - {end_time:.1f}s")
                
                # Validate times
                if start_time >= end_time:
                    st.warning(f"Skipping segment {i}: Invalid time range")
                    continue
                
                duration = end_time - start_time
                if duration < 1:
                    st.warning(f"Skipping segment {i}: Clip too short")
                    continue
                
                # Create temporary file
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False, 
                    suffix=".mp4",
                    prefix=f"clip_{i}_"
                )
                temp_file.close()
                
                # Use ffmpeg to extract clip
                cmd = [
                    'ffmpeg', '-ss', str(start_time), '-i', video_path,
                    '-t', str(duration),
                    '-c', 'copy',  # Copy streams without re-encoding (much faster)
                    '-avoid_negative_ts', 'make_zero',
                    '-y',  # Overwrite output
                    temp_file.name
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0 and os.path.isfile(temp_file.name) and os.path.getsize(temp_file.name) > 0:
                    clips.append({
                        "path": temp_file.name, 
                        "caption": caption,
                        "score": score,
                        "reason": reason,
                        "hook": hook,
                        "flow": flow,
                        "start": seg.get("start"),
                        "end": seg.get("end"),
                        "duration": f"{duration:.1f}s",
                        "format": "Original (ffmpeg)",
                        "index": i
                    })
                    st.success(f"✅ Created clip {i}")
                else:
                    st.error(f"Failed to create clip {i}: ffmpeg error")
                    st.error(f"ffmpeg stderr: {result.stderr}")
                
            except Exception as e:
                st.error(f"Error creating clip {i}: {str(e)}")
                continue
                
    except Exception as e:
        st.error(f"Error in ffmpeg clip generation: {str(e)}")
        raise
    
    return clips


def generate_clips_with_moviepy(video_path: str, segments: list) -> list:
    """Fallback: Generate clips using MoviePy with ultra-conservative settings."""
    clips = []
    main_video = None
    
    try:
        st.warning("Using MoviePy fallback - this may be less stable")
        
        # Load video once
        main_video = VideoFileClip(video_path)
        total_duration = main_video.duration
        
        # Process only the first segment to minimize crash risk
        for i, seg in enumerate(segments[:1], start=1):  # Only process first segment
            clip = None
            try:
                start_time = time_to_seconds(seg.get("start", "0"))
                end_time = time_to_seconds(seg.get("end", "0"))
                caption = seg.get("caption", f"clip_{i}")
                score = seg.get("score", 0)
                reason = seg.get("reason", "")
                hook = seg.get("hook", "")
                flow = seg.get("flow", "")
                
                st.info(f"Processing clip {i} with MoviePy: {start_time:.1f}s - {end_time:.1f}s")
                
                # Validate times
                if start_time >= end_time or start_time >= total_duration:
                    continue
                    
                if end_time > total_duration:
                    end_time = total_duration
                
                duration = end_time - start_time
                if duration < 1:
                    continue
                
                # Create clip
                clip = main_video.subclipped(start_time, end_time)
                
                # Create temporary file
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False, 
                    suffix=".mp4",
                    prefix=f"clip_{i}_"
                )
                temp_file.close()
                
                # Write with minimal settings
                clip.write_videofile(temp_file.name)
                
                if os.path.isfile(temp_file.name) and os.path.getsize(temp_file.name) > 0:
                    clips.append({
                        "path": temp_file.name, 
                        "caption": caption,
                        "score": score,
                        "reason": reason,
                        "hook": hook,
                        "flow": flow,
                        "start": seg.get("start"),
                        "end": seg.get("end"),
                        "duration": f"{duration:.1f}s",
                        "format": "Original (MoviePy)",
                        "index": i
                    })
                    st.success(f"✅ Created clip {i}")
                
            except Exception as e:
                st.error(f"Error creating clip {i}: {str(e)}")
                continue
            finally:
                if clip is not None:
                    try:
                        clip.close()
                    except:
                        pass
                gc.collect()
                time.sleep(1)  # Give system time to recover
        
    except Exception as e:
        st.error(f"Error processing video with MoviePy: {str(e)}")
        raise
    finally:
        if main_video is not None:
            try:
                main_video.close()
            except:
                pass
        gc.collect()
    
    return clips


def generate_clips(video_path: str, segments: list, make_vertical: bool = False) -> list:
    """Generate clips - try ffmpeg first, fallback to MoviePy."""
    try:
        # Try ffmpeg first (much more stable)
        if check_ffmpeg_available():
            st.info("🚀 Using ffmpeg for clip generation (recommended)")
            return generate_clips_with_ffmpeg(video_path, segments)
        else:
            st.warning("⚠️ ffmpeg not available, using MoviePy fallback")
            return generate_clips_with_moviepy(video_path, segments)
            
    except Exception as e:
        st.error(f"Clip generation failed: {str(e)}")
        # If ffmpeg fails, try MoviePy as last resort
        if check_ffmpeg_available():
            st.info("🔄 ffmpeg failed, trying MoviePy fallback...")
            try:
                return generate_clips_with_moviepy(video_path, segments)
            except Exception as e2:
                st.error(f"MoviePy fallback also failed: {str(e2)}")
                raise
        else:
            raise


def download_drive_file(drive_url: str, out_path: str) -> str:
    """Download a Google Drive file given its share URL to out_path."""
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
            raise ValueError("Could not extract file ID from URL")
        
        # Try gdown with simplified error handling
        try:
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            result = gdown.download(download_url, out_path, quiet=True)
            
            if result and os.path.isfile(result) and os.path.getsize(result) > 0:
                return result
        except Exception as gdown_error:
            error_msg = str(gdown_error).lower()
            if "too many users" in error_msg or "rate limit" in error_msg or "quota" in error_msg:
                raise Exception("Google Drive rate limit - please use direct upload")
        
        raise Exception("Download failed - please use direct upload instead")
        
    except Exception as e:
        raise Exception(str(e))


# ----------
# Streamlit App
# ----------

def main():
    st.set_page_config(page_title="ClipMaker", layout="wide")
    
    st.title("🎬 Long‑form to Short‑form ClipMaker")
    st.markdown("Transform your long-form content into viral short-form clips (20-59s) with compelling hooks!")

    # System compatibility check
    ffmpeg_available = check_ffmpeg_available()
    if ffmpeg_available:
        st.success("✅ ffmpeg detected - using stable clip generation")
    else:
        st.warning("⚠️ ffmpeg not available - using MoviePy fallback (less stable)")

    # Add helpful notice
    if st.session_state.get('show_drive_warning', True):
        with st.container():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.info("💡 **Tip:** Use direct file upload for best reliability. Limited to TOP 3 clips for stability.")
            with col2:
                if st.button("✕", help="Dismiss", key="dismiss_warning"):
                    st.session_state['show_drive_warning'] = False
                    st.rerun()

    # Initialize session state
    if 'app_initialized' not in st.session_state:
        st.session_state.app_initialized = True
        st.session_state.current_step = 'upload'
        
    if 'clips_generated' not in st.session_state:
        st.session_state.clips_generated = False
    if 'all_clips' not in st.session_state:
        st.session_state.all_clips = []
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False

    # Load & validate API Key
    API_KEY = get_api_key()
    if not API_KEY:
        st.error("❌ OpenAI API key not found. Add it to Streamlit secrets or env var OPENAI_API_KEY.")
        st.info("💡 Add your OpenAI API key in the Streamlit secrets or as an environment variable.")
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
        help="Choose the platform to optimize clips for"
    )
    
    # Content focus parameters selection
    st.sidebar.subheader("🎯 Content Focus")
    st.sidebar.caption("Select content types to prioritize (multiple selections allowed)")
    
    available_parameters = [
        "Educational Value",
        "Surprise Factor", 
        "Emotional Impact",
        "Replayability",
        "Speaker Energy",
        "Relatability",
        "Contrarian Takes",
        "Storytelling"
    ]
    
    selected_parameters = []
    for param in available_parameters:
        if st.sidebar.checkbox(param, key=f"param_{param}"):
            selected_parameters.append(param)
    
    if not selected_parameters:
        st.sidebar.warning("⚠️ Select at least one content focus parameter")
    else:
        st.sidebar.success(f"✅ {len(selected_parameters)} parameters selected")
    
    # Info about clip limits
    st.sidebar.markdown("---")
    st.sidebar.info("🎯 **Stability Mode:** Generates TOP 3 clips only to prevent system crashes")
    
    # Store in session state
    st.session_state['selected_parameters'] = selected_parameters

    # Video source
    st.sidebar.subheader("📹 Video Source")
    
    # Direct upload (recommended)
    st.sidebar.markdown("**🚀 Recommended: Direct Upload**")
    uploaded = st.sidebar.file_uploader(
        "📁 Upload video file", 
        type=["mp4", "mov", "mkv", "avi", "webm"],
        help="Most reliable method - no rate limits"
    )
    
    video_path = None
    
    if uploaded:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
        tmp.write(uploaded.read())
        video_path = tmp.name
        
        st.session_state['video_path'] = video_path
        st.session_state['video_size'] = len(uploaded.getvalue()) / (1024 * 1024)
        
        # Reset processing state when new video is loaded
        st.session_state.clips_generated = False
        st.session_state.all_clips = []
        st.session_state.processing_complete = False
        
        st.success(f"✅ Uploaded {st.session_state['video_size']:.1f}MB successfully!")
        
        # Show preview for smaller files only
        if st.session_state['video_size'] <= 200:  # Reduced threshold
            st.video(video_path)
        else:
            st.info("Large file uploaded. Skipping preview to conserve memory.")
    
    # Google Drive option (if no file uploaded)
    if not uploaded:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**⚠️ Alternative: Google Drive (Limited)**")
        st.sidebar.caption("⚠️ May hit rate limits - direct upload recommended")
        
        drive_url = st.sidebar.text_input(
            "Google Drive URL", 
            placeholder="https://drive.google.com/file/d/...",
            help="Often hits rate limits - use at your own risk"
        )
        
        if drive_url:
            if st.sidebar.button("📥 Try Download from Drive"):
                with st.spinner("Attempting Google Drive download..."):
                    try:
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                        result = download_drive_file(drive_url, tmp.name)
                        
                        if result and os.path.isfile(result):
                            size_mb = os.path.getsize(result) / (1024 * 1024)
                            st.success(f"✅ Downloaded {size_mb:.2f} MB from Drive")
                            video_path = result
                            st.session_state['video_path'] = video_path
                            st.session_state['video_size'] = size_mb
                            
                            # Reset processing state
                            st.session_state.clips_generated = False
                            st.session_state.all_clips = []
                            st.session_state.processing_complete = False
                            
                            if size_mb <= 200:
                                st.video(video_path)
                    except Exception as e:
                        st.error("❌ Google Drive download failed (rate limited)")
                        st.info("💡 **Solution:** Use the direct upload option above instead")
                        return

    # Use video from session state if available
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
        st.info("🎯 Upload a video file or provide a Drive link to begin.")
        return

    # Check if parameters are selected
    if not selected_parameters:
        st.warning("⚠️ Please select at least one content focus parameter in the sidebar to continue.")
        return

    # Show generated clips if they exist
    if st.session_state.clips_generated and st.session_state.all_clips:
        st.markdown("---")
        st.header("🎬 Generated Clips")
        
        # Summary stats
        successful_clips = len(st.session_state.all_clips)
        if successful_clips > 0:
            st.success(f"🎉 Successfully generated {successful_clips} clips!")
            
            # Summary stats
            st.subheader("📈 Generation Summary")
            total_duration = sum(float(c.get('duration', '0').replace('s', '')) for c in st.session_state.all_clips)
            total_size_mb = sum(os.path.getsize(c.get('path', '')) / (1024 * 1024) for c in st.session_state.all_clips if c.get('path') and os.path.isfile(c.get('path', '')))
            avg_score = sum(c.get('score', 0) for c in st.session_state.all_clips) / successful_clips
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Generated", successful_clips)
            col2.metric("Avg Score", f"{avg_score:.1f}/100")
            col3.metric("Total Duration", f"{total_duration:.1f}s")
            col4.metric("Total Size", f"{total_size_mb:.1f}MB")
        
        # Display clips
        for clip_info in st.session_state.all_clips:
            st.markdown(f"### 🎬 Clip {clip_info['index']} - Score: {clip_info['score']}/100")
            
            # Create columns
            video_col, details_col = st.columns([1, 1.5])
            
            with video_col:
                if os.path.isfile(clip_info["path"]):
                    st.video(clip_info["path"])
                else:
                    st.error("❌ Clip file no longer available")
            
            with details_col:
                # Clip details
                st.markdown("**📊 Clip Details**")
                detail_info = f"""
                ⏱️ **Duration:** {clip_info['duration']}  
                🕐 **Time:** {clip_info['start']} - {clip_info['end']}  
                🎯 **Score:** {clip_info['score']}/100  
                📱 **Format:** {clip_info['format']}
                """
                st.markdown(detail_info)
                
                # Expandable sections
                with st.expander("📝 Suggested Caption", expanded=False):
                    st.code(clip_info["caption"], language="text")
                
                with st.expander("🪝 Hook (Exact Transcript)", expanded=False):
                    st.write(f"**Starting words:** {clip_info['hook']}")
                
                with st.expander("🎬 Content Flow", expanded=False):
                    st.write(clip_info['flow'])
                
                with st.expander("💡 Viral Potential", expanded=False):
                    st.write(clip_info['reason'])
                
                # Download button
                st.markdown("---")
                if os.path.isfile(clip_info["path"]):
                    with open(clip_info["path"], "rb") as file:
                        download_key = f"download_{clip_info['index']}_{hash(clip_info['path'])}"
                        st.download_button(
                            label="⬇️ Download Clip",
                            data=file,
                            file_name=f"clip_{clip_info['index']}_{platform.replace(' ', '_').lower()}.mp4",
                            mime="video/mp4",
                            use_container_width=True,
                            type="primary",
                            key=download_key
                        )
                else:
                    st.error("❌ File no longer available")
            
            st.markdown("---")

        # Reset button
        if st.button("🔄 Clear All Clips & Start Over", type="secondary"):
            # Clean up clip files
            for clip in st.session_state.all_clips:
                try:
                    if clip.get("path") and os.path.isfile(clip["path"]):
                        os.unlink(clip["path"])
                except:
                    pass
            
            # Reset session state
            st.session_state.clips_generated = False
            st.session_state.all_clips = []
            st.session_state.processing_complete = False
            st.rerun()

        return

    # Main processing
    if not st.session_state.processing_complete:
        if st.button("🚀 Generate Clips", type="primary"):
            if not video_path or not os.path.isfile(video_path):
                st.error("Video file not found. Please reload your video.")
                return
                
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Analyze video
                status_text.text("📹 Analyzing video...")
                progress_bar.progress(10)
                
                # Use ffmpeg for video info if available
                if check_ffmpeg_available():
                    try:
                        cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', 
                               '-of', 'csv=p=0', video_path]
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                        video_duration = float(result.stdout.strip())
                        
                        cmd = ['ffprobe', '-v', 'quiet', '-select_streams', 'v:0', 
                               '-show_entries', 'stream=width,height', '-of', 'csv=p=0', video_path]
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                        dimensions = result.stdout.strip().split(',')
                        video_width, video_height = int(dimensions[0]), int(dimensions[1])
                    except:
                        # Fallback to MoviePy
                        temp_video = VideoFileClip(video_path)
                        video_duration = temp_video.duration
                        video_width = temp_video.w
                        video_height = temp_video.h
                        temp_video.close()
                        del temp_video
                        gc.collect()
                else:
                    # Use MoviePy
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
                
                # Show transcript
                progress_bar.progress(50)
                with st.expander("📄 Transcript Preview", expanded=False):
                    st.text_area("Full Transcript", transcript, height=200, disabled=True)

                # Step 3: AI analysis
                status_text.text(f"🤖 Analyzing for viral segments...")
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
                    st.warning("⚠️ No valid segments found in AI response.")
                    return
                    
                # Sort by score
                segments_sorted = sorted(segments, key=lambda x: x.get('score', 0), reverse=True)
                
                # Step 5: Generate clips
                status_text.text("✂️ Generating video clips...")
                progress_bar.progress(95)
                
                st.success(f"🎯 Found {len(segments_sorted)} segments! Generating clips...")

                st.markdown("---")
                st.header("🎬 Generating Clips")
                st.info(f"🚀 Processing {len(segments_sorted)} clips with stability optimizations")
                
                # Generate clips
                all_clips = generate_clips(video_path, segments_sorted, False)
                
                if all_clips:
                    # Store in session state
                    st.session_state.all_clips = all_clips
                    st.session_state.clips_generated = True
                    st.session_state.processing_complete = True
                    
                    progress_bar.progress(100)
                    status_text.text("✅ All clips generated successfully!")
                    
                    st.success(f"🎉 Generated {len(all_clips)} clips!")
                    st.rerun()
                else:
                    st.warning("No clips were generated.")
                    return
                    
            except Exception as e:
                st.error("❌ Processing failed")
                st.error(f"Error: {str(e)}")
                st.error(f"Traceback: {traceback.format_exc()}")
                return
    else:
        st.info("🎯 Click 'Generate Clips' to start processing your video.")

    # Reset button
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Start Over", help="Clear all data and start fresh"):
        # Clean up clip files
        if 'all_clips' in st.session_state:
            for clip in st.session_state.all_clips:
                try:
                    if clip.get("path") and os.path.isfile(clip["path"]):
                        os.unlink(clip["path"])
                except:
                    pass
        
        # Clear session state
        for key in list(st.session_state.keys()):
            if key not in ['app_initialized', 'show_drive_warning']:
                del st.session_state[key]
        st.session_state['current_step'] = 'upload'
        gc.collect()
        st.rerun()


if __name__ == "__main__":
    main()
