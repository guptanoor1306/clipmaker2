# app.py - FFmpeg-only ClipMaker (User-selected parameters + Auto-generation)
import os
import json
import tempfile
import traceback
import re
import subprocess
import streamlit as st
import gdown
import time
from openai import OpenAI

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


def get_video_info(video_path: str, ffmpeg_path: str = 'ffmpeg') -> dict:
    """Get video information using multiple fallback methods."""
    try:
        # Method 1: Try imageio-ffmpeg's ffmpeg with shorter timeout and specific flags
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            
            # Use ffmpeg with minimal probing - just get basic info quickly
            cmd = [ffmpeg_exe, "-hide_banner", "-i", video_path, "-t", "0.1", "-f", "null", "-"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            # Parse from stderr even if command "fails" (it will because of -t 0.1)
            import re
            duration_match = re.search(r'Duration: (\d+):(\d+):(\d+\.\d+)', result.stderr)
            size_match = re.search(r'(\d+)x(\d+)', result.stderr)
            
            if duration_match and size_match:
                h, m, s = duration_match.groups()
                duration = int(h) * 3600 + int(m) * 60 + float(s)
                width, height = size_match.groups()
                
                st.info(f"✅ Video info: {width}x{height}, {duration:.1f}s")
                return {
                    'duration': duration,
                    'width': int(width),
                    'height': int(height)
                }
            else:
                raise Exception("Could not parse ffmpeg output")
                
        except Exception as ffmpeg_error:
            st.warning(f"imageio-ffmpeg failed: {str(ffmpeg_error)[:100]}...")
            
            # Method 2: Try system ffmpeg with very short timeout
            try:
                cmd = ['ffmpeg', "-hide_banner", "-i", video_path, "-t", "0.1", "-f", "null", "-"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
                
                # Parse from stderr
                import re
                duration_match = re.search(r'Duration: (\d+):(\d+):(\d+\.\d+)', result.stderr)
                size_match = re.search(r'(\d+)x(\d+)', result.stderr)
                
                if duration_match and size_match:
                    h, m, s = duration_match.groups()
                    duration = int(h) * 3600 + int(m) * 60 + float(s)
                    width, height = size_match.groups()
                    
                    st.info(f"✅ Video info (system): {width}x{height}, {duration:.1f}s")
                    return {
                        'duration': duration,
                        'width': int(width),
                        'height': int(height)
                    }
                else:
                    raise Exception("Could not parse system ffmpeg output")
                    
            except Exception as system_error:
                st.warning(f"System ffmpeg also failed: {str(system_error)[:100]}...")
                
                # Method 3: Use file-based estimation
                try:
                    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
                    
                    # Rough estimation based on file size
                    # Average: ~1MB per minute for compressed video
                    estimated_duration = max(60.0, file_size_mb * 60.0)  # At least 1 minute
                    estimated_duration = min(estimated_duration, 7200.0)  # Max 2 hours
                    
                    st.warning(f"Using file size estimation: {file_size_mb:.1f}MB → ~{estimated_duration/60:.1f} minutes")
                    
                    return {
                        'duration': estimated_duration,
                        'width': 1920,  # Common default
                        'height': 1080
                    }
                    
                except Exception as file_error:
                    st.warning(f"File size estimation failed: {file_error}")
                    
                    # Method 4: Ask user for duration as absolute fallback
                    st.error("🚨 Could not analyze video automatically")
                    
                    user_duration = st.number_input(
                        "Please enter video duration in minutes:",
                        min_value=1.0,
                        max_value=180.0,
                        value=30.0,
                        step=1.0,
                        help="Estimate the length of your video in minutes"
                    )
                    
                    if user_duration:
                        duration_seconds = user_duration * 60
                        st.info(f"Using user-provided duration: {duration_seconds:.1f}s")
                        
                        return {
                            'duration': duration_seconds,
                            'width': 1920,
                            'height': 1080
                        }
                    else:
                        # Final fallback
                        return {
                            'duration': 1800.0,  # 30 minutes
                            'width': 1920,
                            'height': 1080
                        }
                
    except Exception as e:
        st.error(f"Complete video analysis failure: {str(e)}")
        return {
            'duration': 1800.0,  # 30 minutes default
            'width': 1920,
            'height': 1080
        }


def transcribe_audio_ffmpeg(video_path: str, client: OpenAI, ffmpeg_path: str = 'ffmpeg') -> str:
    """Extract audio using FFmpeg and transcribe with OpenAI Whisper."""
    try:
        # Quick file size check first
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        st.info(f"📹 Video file size: {file_size_mb:.1f}MB")
        
        # For very large files, show estimated time but proceed automatically
        if file_size_mb > 100:
            estimated_time = (file_size_mb / 100) * 10  # Rough estimate: 10 min per 100MB
            st.warning(f"⚠️ Large video detected ({file_size_mb:.1f}MB). Transcription estimated time: {estimated_time:.1f} minutes.")
            st.info("🚀 Starting full transcription automatically...")
        
        # Extract audio to temporary file using FFmpeg
        audio_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        audio_temp.close()
        
        st.info("🎵 Extracting audio with FFmpeg...")
        
        # Get FFmpeg executable
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except:
            ffmpeg_exe = ffmpeg_path
        
        # FFmpeg command to extract audio
        ffmpeg_cmd = [
            ffmpeg_exe, '-y', '-i', video_path, 
            '-vn',  # No video
            '-acodec', 'mp3', 
            '-ab', '64k',  # Lower bitrate to reduce file size
            '-ar', '22050',  # Lower sample rate for smaller files
            audio_temp.name
        ]
        
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            raise Exception(f"Audio extraction failed: {result.stderr}")
        
        # Check file size and split if needed
        audio_file_size_mb = os.path.getsize(audio_temp.name) / (1024 * 1024)
        st.info(f"Audio file size: {audio_file_size_mb:.1f}MB")
        
        if audio_file_size_mb > 20:  # Split if too large
            st.info("Audio too large, splitting into chunks...")
            # Split into 10-minute chunks
            chunks = []
            chunk_duration = 600  # 10 minutes
            
            # Get audio duration - use a simpler method
            try:
                # Quick duration check with shorter timeout
                probe_cmd = [ffmpeg_exe, '-i', audio_temp.name, '-f', 'null', '-', '-t', '0.1']
                probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
                
                # Parse duration from stderr
                duration_match = re.search(r'Duration: (\d+):(\d+):(\d+\.\d+)', probe_result.stderr)
                if duration_match:
                    h, m, s = duration_match.groups()
                    total_duration = int(h) * 3600 + int(m) * 60 + float(s)
                else:
                    # Estimate from file size (rough: 1MB ≈ 8 minutes of 64k audio)
                    total_duration = audio_file_size_mb * 8 * 60
                    st.info(f"Estimated audio duration: {total_duration/60:.1f} minutes")
            except Exception:
                # Final fallback
                total_duration = 3600  # 1 hour default
                st.warning("Using default duration estimate")
            
            chunk_num = 0
            start_time = 0
            
            while start_time < total_duration:
                chunk_temp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_chunk_{chunk_num}.mp3")
                chunk_temp.close()
                
                duration = min(chunk_duration, total_duration - start_time)
                
                chunk_cmd = [
                    ffmpeg_exe, '-y', '-i', audio_temp.name,
                    '-ss', str(start_time), '-t', str(duration),
                    '-acodec', 'copy',
                    chunk_temp.name
                ]
                
                try:
                    subprocess.run(chunk_cmd, capture_output=True, timeout=120)
                    if os.path.exists(chunk_temp.name) and os.path.getsize(chunk_temp.name) > 1000:
                        chunks.append(chunk_temp.name)
                    else:
                        st.warning(f"Chunk {chunk_num} failed or empty, skipping")
                except subprocess.TimeoutExpired:
                    st.warning(f"Chunk {chunk_num} timed out, skipping")
                
                start_time += chunk_duration
                chunk_num += 1
            
            if not chunks:
                raise Exception("No valid audio chunks created")
            
            # Transcribe chunks
            full_transcript = ""
            for i, chunk_path in enumerate(chunks):
                st.info(f"Transcribing chunk {i+1}/{len(chunks)}...")
                try:
                    with open(chunk_path, "rb") as f:
                        resp = client.audio.transcriptions.create(model="whisper-1", file=f)
                    full_transcript += resp.text + " "
                except Exception as chunk_error:
                    st.warning(f"Chunk {i+1} transcription failed: {chunk_error}")
                finally:
                    try:
                        os.unlink(chunk_path)  # Clean up chunk
                    except:
                        pass
                
        else:
            # Transcribe single file
            st.info("🎤 Transcribing audio with OpenAI Whisper...")
            with open(audio_temp.name, "rb") as f:
                resp = client.audio.transcriptions.create(model="whisper-1", file=f)
            full_transcript = resp.text
        
        # Clean up audio file
        try:
            os.unlink(audio_temp.name)
        except:
            pass
            
        if not full_transcript.strip():
            raise Exception("Transcription returned empty result")
            
        return full_transcript.strip()
        
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        # Clean up on error
        try:
            if 'audio_temp' in locals():
                os.unlink(audio_temp.name)
        except:
            pass
        raise


def generate_clip_ffmpeg(video_path: str, start_time: float, end_time: float, make_vertical: bool = True, ffmpeg_path: str = 'ffmpeg') -> str:
    """Generate a clip using FFmpeg with simplified and reliable approach."""
    try:
        # Create output file
        temp_clip = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_clip.close()
        
        duration = end_time - start_time
        
        # Get FFmpeg executable
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except:
            ffmpeg_exe = ffmpeg_path
        
        # Progress tracking setup
        progress_placeholder = st.empty()
        start_generation_time = time.time()
        
        if make_vertical:
            # Get video info for cropping calculation
            video_info = get_video_info(video_path, ffmpeg_exe)
            original_width = video_info['width']
            original_height = video_info['height']
            
            progress_placeholder.info(f"🎬 Converting {original_width}x{original_height} to shorts format...")
            
            # Target dimensions for shorts
            target_width = 1080
            target_height = 1920
            target_ratio = target_width / target_height  # 0.5625
            original_ratio = original_width / original_height
            
            if original_ratio > target_ratio:
                # Horizontal video - crop from center and scale
                # Simplified approach: crop to 9:16 ratio then scale
                crop_height = original_height
                crop_width = int(crop_height * target_ratio)
                crop_x = (original_width - crop_width) // 2
                crop_y = 0
                
                # Simplified FFmpeg command
                cmd = [
                    ffmpeg_exe, '-y', 
                    '-i', video_path,
                    '-ss', str(start_time),
                    '-t', str(duration),
                    '-vf', f'crop={crop_width}:{crop_height}:{crop_x}:{crop_y},scale={target_width}:{target_height}',
                    '-c:a', 'aac', '-b:a', '128k',
                    '-c:v', 'libx264',
                    '-preset', 'veryfast',  # Even faster than ultrafast but more stable
                    '-crf', '28',
                    '-movflags', '+faststart',  # For better web compatibility
                    '-avoid_negative_ts', 'make_zero',  # Fix timestamp issues
                    temp_clip.name
                ]
                progress_placeholder.info(f"📐 Cropping center {crop_width}x{crop_height} and scaling to {target_width}x{target_height}")
                
            else:
                # Already vertical or square - just scale
                cmd = [
                    ffmpeg_exe, '-y',
                    '-i', video_path,
                    '-ss', str(start_time),
                    '-t', str(duration),
                    '-vf', f'scale={target_width}:{target_height}',
                    '-c:a', 'aac', '-b:a', '128k',
                    '-c:v', 'libx264',
                    '-preset', 'veryfast',
                    '-crf', '28',
                    '-movflags', '+faststart',
                    '-avoid_negative_ts', 'make_zero',
                    temp_clip.name
                ]
                progress_placeholder.info(f"📱 Scaling to {target_width}x{target_height} shorts format")
        else:
            # Horizontal clip - keep original aspect ratio
            cmd = [
                ffmpeg_exe, '-y',
                '-i', video_path,
                '-ss', str(start_time),
                '-t', str(duration),
                '-c:a', 'aac', '-b:a', '128k',
                '-c:v', 'libx264',
                '-preset', 'veryfast',
                '-crf', '28',
                '-movflags', '+faststart',
                '-avoid_negative_ts', 'make_zero',
                temp_clip.name
            ]
            progress_placeholder.info(f"📺 Creating horizontal clip...")
        
        # Reduce timeout for better user experience
        timeout_seconds = max(45, min(180, duration * 3))  # 3x duration, max 3 minutes
        
        progress_placeholder.info(f"⚙️ Processing {duration:.1f}s clip (timeout: {timeout_seconds}s)...")
        
        # Execute FFmpeg with better error handling
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)
            
            elapsed = time.time() - start_generation_time
            
            if result.returncode != 0:
                progress_placeholder.error(f"❌ FFmpeg failed after {elapsed:.1f}s")
                # Show more detailed error for debugging
                error_details = result.stderr[-800:] if result.stderr else "No error details"
                st.error(f"FFmpeg error details: {error_details}")
                raise Exception(f"FFmpeg failed with return code {result.returncode}")
            
            if not os.path.isfile(temp_clip.name):
                raise Exception("Output file was not created")
            
            file_size = os.path.getsize(temp_clip.name)
            if file_size < 1000:  # Less than 1KB
                raise Exception(f"Output file too small ({file_size} bytes)")
            
            file_size_mb = file_size / (1024 * 1024)
            
            # Verify the output dimensions if possible
            try:
                verify_info = get_video_info(temp_clip.name, ffmpeg_exe)
                actual_width = verify_info.get('width', 0)
                actual_height = verify_info.get('height', 0)
                
                if make_vertical and actual_width > 0 and actual_height > 0:
                    if actual_width == 1080 and actual_height == 1920:
                        progress_placeholder.success(f"✅ Perfect shorts format: {actual_width}x{actual_height} in {elapsed:.1f}s ({file_size_mb:.1f}MB)")
                    else:
                        progress_placeholder.success(f"✅ Clip created: {actual_width}x{actual_height} in {elapsed:.1f}s ({file_size_mb:.1f}MB)")
                else:
                    progress_placeholder.success(f"✅ Clip generated in {elapsed:.1f}s ({file_size_mb:.1f}MB)")
            except:
                progress_placeholder.success(f"✅ Clip generated in {elapsed:.1f}s ({file_size_mb:.1f}MB)")
            
            return temp_clip.name
            
        except subprocess.TimeoutExpired:
            progress_placeholder.error(f"⏰ FFmpeg timed out after {timeout_seconds}s")
            raise Exception(f"Processing timed out after {timeout_seconds}s - try shorter clips or use instructions mode")
        
    except Exception as e:
        # Clean up on failure
        try:
            if os.path.exists(temp_clip.name):
                os.unlink(temp_clip.name)
        except:
            pass
        raise Exception(f"Clip generation failed: {str(e)}")


def analyze_transcript(transcript: str, platform: str, selected_parameters: list, client: OpenAI, video_duration: float = None) -> str:
    """Get segment suggestions via ChatCompletion."""
    messages = [
        {"role": "system", "content": get_system_prompt(platform, selected_parameters, video_duration)},
        {"role": "user", "content": f"Analyze this transcript and identify the best 20-59 second segments for {platform} based on the selected parameters: {', '.join(selected_parameters)}. Each must start with a compelling hook (exact transcript text) in the first 3 seconds and end smoothly. Remember the video is {video_duration:.1f} seconds long, so all timestamps must be within this range:\n\n{transcript}"}
    ]
    
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=3000
        )
        return resp.choices[0].message.content
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        raise


def parse_segments(text: str) -> list:
    """Parse JSON text into a list of segments."""
    try:
        # Clean the text in case there's extra content
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        segments = json.loads(text)
        
        # Validate segments and check duration
        valid_segments = []
        for seg in segments:
            required_keys = ["start", "end", "hook", "flow", "reason", "score", "caption"]
            if all(key in seg for key in required_keys):
                # Check duration is 20-59 seconds
                start_time = time_to_seconds(seg.get("start", "0"))
                end_time = time_to_seconds(seg.get("end", "0"))
                duration = end_time - start_time
                
                if 20 <= duration <= 59:
                    valid_segments.append(seg)
                else:
                    st.warning(f"Skipping segment {seg.get('start')}-{seg.get('end')} (duration: {duration:.1f}s, should be 20-59s)")
            else:
                st.warning(f"Skipping invalid segment: missing required fields")
        
        return valid_segments
    except json.JSONDecodeError as e:
        st.error(f"JSON parse error: {e}")
        st.error(f"Raw text received: {text[:500]}...")
        return []


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

    # Add helpful notice about Google Drive issues
    if st.session_state.get('show_drive_warning', True):
        with st.container():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.info("💡 **Tip:** Use direct file upload in the sidebar for best reliability. Google Drive often hits rate limits.")
            with col2:
                if st.button("✕", help="Dismiss", key="dismiss_warning"):
                    st.session_state['show_drive_warning'] = False
                    st.rerun()

    # Initialize session state for better state management
    if 'app_initialized' not in st.session_state:
        st.session_state.app_initialized = True
        st.session_state.current_step = 'upload'  # upload, analyzing, results, generating
    
    # Preserve state across downloads
    if 'preserve_state' not in st.session_state:
        st.session_state.preserve_state = True

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
    st.sidebar.caption("Select the types of content you want to prioritize (multiple selections allowed)")
    
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
    
    # Quick mode toggle
    quick_mode = st.sidebar.checkbox(
        "⚡ Quick Mode", 
        value=True,
        help="Faster processing with lower quality encoding"
    )
    
    # Store in session state for use in functions
    st.session_state['quick_mode'] = quick_mode
    st.session_state['selected_parameters'] = selected_parameters
    
    # Check FFmpeg availability using imageio-ffmpeg
    ffmpeg_available = False
    ffmpeg_path = None
    
    try:
        # Try imageio-ffmpeg first (works on Streamlit Cloud)
        import imageio_ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        
        # Test if it works
        result = subprocess.run([ffmpeg_path, '-version'], capture_output=True, timeout=5)
        if result.returncode == 0:
            ffmpeg_available = True
        else:
            raise Exception("FFmpeg test failed")
            
    except Exception as e:
        # Fallback to system FFmpeg
        try:
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
            if result.returncode == 0:
                ffmpeg_available = True
                ffmpeg_path = 'ffmpeg'
        except:
            pass
            
    if not ffmpeg_available:
        st.sidebar.markdown("""
        **📝 Note:** FFmpeg not available in this environment.
        
        **Options:**
        1. **Get instructions** for manual processing
        2. **Use online tools** for conversion
        3. **Install imageio-ffmpeg** for full features
        """)
        
        # Provide instructions instead of clips
        make_vertical = st.sidebar.checkbox(
            "Generate Vertical Instructions", 
            value=True,
            help="Get FFmpeg commands for vertical conversion"
        )
        st.sidebar.info("📋 Will provide FFmpeg commands")
    else:
        make_vertical = st.sidebar.checkbox(
            "Create Vertical Clips (9:16)", 
            value=True,
            help="Convert to vertical format perfect for Instagram Reels"
        )

    # Video source
    st.sidebar.subheader("📹 Video Source")
    
    # Make direct upload more prominent
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
        
        st.success(f"✅ Uploaded {st.session_state['video_size']:.1f}MB successfully!")
        
        if st.session_state['video_size'] <= 500:
            st.video(video_path)
        else:
            st.info("Large file uploaded. Skipping preview to save memory.")
    
    # Only show Google Drive if no file uploaded
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
                            
                            if size_mb <= 500:
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
        else:
            st.warning("Previously loaded video no longer available. Please reload.")
            del st.session_state['video_path']
            video_path = None

    if not video_path:
        st.info("🎯 Upload a video file or provide a Drive link to begin.")
        return

    # Only show generate button if parameters are selected
    if not selected_parameters:
        st.warning("⚠️ Please select at least one content focus parameter in the sidebar to continue.")
        return

    # Main processing
    if st.button("🚀 Generate Clips", type="primary"):
        if not video_path or not os.path.isfile(video_path):
            st.error("Video file not found. Please reload your video.")
            return
        
        # Clear any old transcription state
        if 'transcription_choice' in st.session_state:
            del st.session_state['transcription_choice']
            
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Get video info
        status_text.text("📹 Analyzing video...")
        progress_bar.progress(10)
        
        try:
            video_info = get_video_info(video_path, ffmpeg_path if ffmpeg_available else 'ffmpeg')
            st.success(f"✅ Video: {video_info['width']}x{video_info['height']}, {video_info['duration']:.1f}s")
        except Exception as e:
            st.error(f"Video analysis failed: {str(e)}")
            return
        
        # Step 2: Transcription
        status_text.text("🎤 Transcribing audio...")
        progress_bar.progress(25)
        
        try:
            if ffmpeg_available:
                transcript = transcribe_audio_ffmpeg(video_path, client, ffmpeg_path)
            else:
                # Simple fallback transcription
                st.warning("Limited transcription without FFmpeg")
                transcript = "Transcript not available without FFmpeg. Please install imageio-ffmpeg for full functionality."
            st.success("✅ Transcription complete")
        except Exception as e:
            st.error(f"Transcription failed: {str(e)}")
            return

        # Show transcript
        progress_bar.progress(50)
        with st.expander("📄 Transcript Preview", expanded=False):
            st.text_area("Full Transcript", transcript, height=200, disabled=True)

        # Step 3: AI analysis
        status_text.text(f"🤖 Analyzing transcript for viral segments based on: {', '.join(selected_parameters)}...")
        progress_bar.progress(75)
        
        try:
            ai_json = analyze_transcript(transcript, platform, selected_parameters, client, video_info['duration'])
            st.success("✅ Analysis complete")
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            return

        # Show AI output
        with st.expander("🔍 AI Analysis Output", expanded=False):
            st.code(ai_json, language="json")

        # Parse segments
        segments = parse_segments(ai_json)
        if not segments:
            st.warning("⚠️ No valid segments found in AI response.")
            return
        
        # Store segments in session state
        st.session_state['ai_segments'] = segments
        st.session_state['video_analyzed'] = True
        st.session_state['current_step'] = 'results'
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete! Generating all clips automatically...")
        
        st.success(f"🎯 Found {len(segments)} potential clips! Generating all clips automatically...")

        # Auto-generate all clips
        if ffmpeg_available and segments:
            st.markdown("---")
            st.header("🎬 Auto-Generating All Clips")
            st.info(f"🚀 Generating all {len(segments)} clips automatically based on your selected parameters")
            
            # Clear any previous clips
            if 'generated_clips' in st.session_state:
                st.session_state.generated_clips.clear()
            else:
                st.session_state.generated_clips = []
            
            # Generate clips one by one
            generation_progress = st.progress(0)
            generation_status = st.empty()
            
            for i, segment in enumerate(segments, 1):
                generation_status.text(f"🎬 Generating clip {i}/{len(segments)}...")
                
                try:
                    start_time = time_to_seconds(segment.get("start", "0"))
                    end_time = time_to_seconds(segment.get("end", "0"))
                    duration = end_time - start_time
                    
                    # Generate clip with FFmpeg
                    clip_path = generate_clip_ffmpeg(video_path, start_time, end_time, make_vertical, ffmpeg_path)
                    
                    file_size_mb = os.path.getsize(clip_path) / (1024 * 1024)
                    
                    clip_info = {
                        "path": clip_path,
                        "caption": segment.get("caption", ""),
                        "score": segment.get("score", 0),
                        "hook": segment.get("hook", "Strong opening hook"),
                        "flow": segment.get("flow", "Complete narrative arc"),
                        "reason": segment.get("reason", ""),
                        "start": segment.get("start"),
                        "end": segment.get("end"),
                        "duration": f"{duration:.1f}s",
                        "format": "Vertical 9:16" if make_vertical else "Original",
                        "file_size": f"{file_size_mb:.1f}MB",
                        "index": i
                    }
                    
                    st.session_state.generated_clips.append(clip_info)
                    
                except Exception as e:
                    st.error(f"❌ Failed to generate clip {i}: {str(e)}")
                    # Continue with next clip instead of stopping
                
                generation_progress.progress(i / len(segments))
            
            generation_status.text("✅ All clips generated!")
            st.session_state['current_step'] = 'completed'

    # Display generated clips if they exist
    if 'generated_clips' in st.session_state and st.session_state.generated_clips:
        st.markdown("---")
        st.header("🎬 Generated Clips")
        
        # Summary stats
        successful_clips = len(st.session_state.generated_clips)
        if successful_clips > 0:
            st.success(f"🎉 Successfully generated {successful_clips} clips!")
            
            # Summary stats
            st.subheader("📈 Generation Summary")
            total_duration = sum(float(c.get('duration', '0').replace('s', '')) for c in st.session_state.generated_clips)
            total_size = sum(float(c.get('file_size', '0').replace('MB', '')) for c in st.session_state.generated_clips)
            avg_score = sum(c.get('score', 0) for c in st.session_state.generated_clips) / successful_clips
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Generated", successful_clips)
            col2.metric("Avg Score", f"{avg_score:.1f}/100")
            col3.metric("Total Duration", f"{total_duration:.1f}s")
            col4.metric("Total Size", f"{total_size:.1f}MB")
            col5.metric("Format", "9:16 Vertical" if make_vertical else "Original")
        
        for clip_info in st.session_state.generated_clips:
            # Display each clip
            st.markdown(f"### 🎬 Clip {clip_info['index']} - Score: {clip_info['score']}/100")
            
            # Create two columns with proper proportions
            video_col, details_col = st.columns([1, 1.5])
            
            with video_col:
                if os.path.isfile(clip_info["path"]):
                    st.video(clip_info["path"], start_time=0)
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
                💾 **Size:** {clip_info['file_size']}
                """
                st.markdown(detail_info)
                
                # Caption section
                with st.expander("📝 Suggested Caption", expanded=False):
                    st.code(clip_info["caption"], language="text")
                
                # Hook section (exact transcript text)
                with st.expander("🪝 Hook (Exact Transcript)", expanded=False):
                    st.write(f"**Starting words:** {clip_info['hook']}")
                
                # Flow section
                with st.expander("🎬 Content Flow", expanded=False):
                    st.write(clip_info['flow'])
                
                # Why it works section
                with st.expander("💡 Viral Potential", expanded=False):
                    st.write(clip_info['reason'])
                
                # Download button
                st.markdown("---")
                if os.path.isfile(clip_info["path"]):
                    with open(clip_info["path"], "rb") as file:
                        file_extension = "vertical" if make_vertical else "horizontal"
                        # Use unique key to prevent state issues
                        download_key = f"download_{clip_info['index']}_{hash(clip_info['path'])}"
                        st.download_button(
                            label="⬇️ Download Clip",
                            data=file,
                            file_name=f"clip_{clip_info['index']}_{platform.replace(' ', '_').lower()}_{file_extension}.mp4",
                            mime="video/mp4",
                            use_container_width=True,
                            type="primary",
                            key=download_key,
                            on_click=None
                        )
                else:
                    st.error("❌ File no longer available")
            
            st.markdown("---")

    # Reset button with confirmation
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Start Over", help="Clear all data and start fresh"):
        # Clear all session state including any old transcription choices
        for key in list(st.session_state.keys()):
            if key not in ['app_initialized', 'show_drive_warning']:  # Keep only essential app state
                del st.session_state[key]
        st.session_state['current_step'] = 'upload'
        st.rerun()


if __name__ == "__main__":
    main()
