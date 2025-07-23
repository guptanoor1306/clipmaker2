# app.py - FFmpeg-only ClipMaker (Fixed ffprobe issue + improved parameters)
import os
import json
import tempfile
import traceback
import re
import subprocess
import streamlit as st
import gdown
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


def get_system_prompt(platform: str, video_duration: float = None) -> str:
    duration_info = f"\n\nIMPORTANT: The video is {video_duration:.1f} seconds ({video_duration/60:.1f} minutes) long. All timestamps MUST be within 0 to {video_duration:.1f} seconds. Do not generate any timestamps beyond this range." if video_duration else ""
    
    return f"""You are a content strategist and social media editor trained to analyze long-form video/podcast transcripts. Your task is to identify 20-59 second segments that are highly likely to perform well as short-form content on {platform}.

CRITICAL REQUIREMENTS:
🎯 Duration: Each clip must be 20-59 seconds (no shorter, no longer)
🪝 Hook: Every clip MUST start with a compelling 0-3 second hook that stops scrolling
🎬 Flow: Complete narrative arc with smooth, non-abrupt ending
📱 Context: Must make sense without prior video context

Key Parameters for Cut Selection:
🪝 HOOK STRENGTH (0-3 seconds): Shocking statement, question, bold claim, or visual grab
🧠 Educational Value: Clear insight, tip, or new perspective delivered quickly  
😲 Surprise Factor: Plot twist, myth-busting, or unexpected revelation
😍 Emotional Impact: Inspiration, humor, shock, or relatability that drives engagement
🔁 Replayability: Content viewers want to watch multiple times or share
📉 Retention: Strong momentum throughout - no dead moments or filler
🎯 Relatability: Reflects common struggles, desires, or experiences
🎤 Speaker Energy: Passionate delivery, voice modulation, natural pauses
🏁 Clean Ending: Natural conclusion, not mid-sentence or abrupt cutoff
📱 Platform Fit: Perfect for {platform} audience and algorithm

HOOK EXAMPLES (First 0-3 seconds):
- "This will change how you think about..."
- "Nobody talks about this, but..."
- "I made a $10,000 mistake so you don't have to..."
- "The thing they don't tell you is..."
- Visual: Dramatic pause, gesture, or surprising action

{duration_info}

CRITICAL: Since you don't have access to actual timestamps, estimate reasonable time intervals based on content flow and speech patterns. Assume average speaking pace of 150-200 words per minute.

For each recommended cut, provide:
1. Start and end timestamps (HH:MM:SS format) - MUST be within video duration
2. Hook description (what happens in first 3 seconds)
3. Content flow (brief summary of 20-59 second narrative)
4. Reason for virality (based on parameters above)
5. Predicted engagement score (0–100) — confidence in performance
6. Suggested caption for social media with emojis/hashtags

Output ONLY valid JSON as an array of objects with these exact keys:
- start: "HH:MM:SS"
- end: "HH:MM:SS" 
- hook: "description of compelling opening 0-3 seconds"
- flow: "brief narrative arc of the full 20-59 second clip"
- reason: "why this will go viral focusing on engagement factors"
- score: integer (0-100)
- caption: "social media caption with emojis and hashtags"

Example format:
[
  {{
    "start": "00:02:15",
    "end": "00:03:02",
    "hook": "Opens with shocking statistic that 90% of people get wrong",
    "flow": "Hook → explains common misconception → reveals truth → actionable tip → strong finish",
    "reason": "Myth-busting content with strong hook, educational value, and shareable insight that challenges assumptions",
    "score": 88,
    "caption": "90% of people believe this financial myth 😱 The truth will shock you! #MoneyMyths #FinanceTips #Viral"
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


def transcribe_audio_ffmpeg(video_path: str, client: OpenAI, ffmpeg_path: str = 'ffmpeg') -> str:
    """Extract audio using FFmpeg and transcribe with OpenAI Whisper."""
    try:
        # Extract audio to temporary file using FFmpeg
        audio_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        audio_temp.close()
        
        st.info("🎵 Extracting audio with FFmpeg...")
        
        # FFmpeg command to extract audio
        ffmpeg_cmd = [
            ffmpeg_path, '-y', '-i', video_path, 
            '-vn',  # No video
            '-acodec', 'mp3', 
            '-ab', '64k',  # Lower bitrate to reduce file size
            audio_temp.name
        ]
        
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            raise Exception(f"Audio extraction failed: {result.stderr}")
        
        # Check file size and split if needed
        file_size_mb = os.path.getsize(audio_temp.name) / (1024 * 1024)
        st.info(f"Audio file size: {file_size_mb:.1f}MB")
        
        if file_size_mb > 20:  # Split if too large
            st.info("Audio too large, splitting into chunks...")
            # Split into 10-minute chunks
            chunks = []
            chunk_duration = 600  # 10 minutes
            
            # Get audio duration using ffprobe or ffmpeg
            try:
                # Try imageio-ffmpeg's ffprobe first
                import imageio_ffmpeg
                ffprobe_path = imageio_ffmpeg.get_ffprobe_exe()
                probe_cmd = [ffprobe_path, '-v', 'quiet', '-print_format', 'json', '-show_format', audio_temp.name]
                probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
                if probe_result.returncode == 0:
                    info = json.loads(probe_result.stdout)
                    total_duration = float(info['format']['duration'])
                else:
                    raise Exception("ffprobe failed")
            except:
                # Fallback: use ffmpeg to get duration
                probe_cmd = [ffmpeg_path, '-i', audio_temp.name, '-f', 'null', '-']
                probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
                
                # Parse duration from stderr
                duration_match = re.search(r'Duration: (\d+):(\d+):(\d+\.\d+)', probe_result.stderr)
                if duration_match:
                    h, m, s = duration_match.groups()
                    total_duration = int(h) * 3600 + int(m) * 60 + float(s)
                else:
                    total_duration = 3600  # Default 1 hour
            
            chunk_num = 0
            start_time = 0
            
            while start_time < total_duration:
                chunk_temp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_chunk_{chunk_num}.mp3")
                chunk_temp.close()
                
                duration = min(chunk_duration, total_duration - start_time)
                
                chunk_cmd = [
                    ffmpeg_path, '-y', '-i', audio_temp.name,
                    '-ss', str(start_time), '-t', str(duration),
                    '-acodec', 'copy',
                    chunk_temp.name
                ]
                
                subprocess.run(chunk_cmd, capture_output=True, timeout=120)
                chunks.append(chunk_temp.name)
                
                start_time += chunk_duration
                chunk_num += 1
            
            # Transcribe chunks
            full_transcript = ""
            for i, chunk_path in enumerate(chunks):
                st.info(f"Transcribing chunk {i+1}/{len(chunks)}...")
                with open(chunk_path, "rb") as f:
                    resp = client.audio.transcriptions.create(model="whisper-1", file=f)
                full_transcript += resp.text + " "
                os.unlink(chunk_path)  # Clean up chunk
                
        else:
            # Transcribe single file
            with open(audio_temp.name, "rb") as f:
                resp = client.audio.transcriptions.create(model="whisper-1", file=f)
            full_transcript = resp.text
        
        # Clean up audio file
        os.unlink(audio_temp.name)
        return full_transcript.strip()
        
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        raise


def generate_clip_commands(video_path: str, start_time: float, end_time: float, make_vertical: bool = False) -> dict:
    """Generate FFmpeg commands for clip creation when direct processing isn't available."""
    try:
        duration = end_time - start_time
        
        # Base FFmpeg command for horizontal clip
        base_cmd = f'ffmpeg -i "{os.path.basename(video_path)}" -ss {start_time} -t {duration} -c:v libx264 -c:a aac -preset fast -crf 23'
        
        commands = {}
        
        if make_vertical:
            # Vertical clip command (assumes 3840x2160 input)
            crop_width = 1215  # For 9:16 ratio from 2160 height
            crop_x = 1312      # Center position
            
            vertical_cmd = f'{base_cmd} -vf "crop={crop_width}:2160:{crop_x}:0,scale=1080:1920" "clip_vertical.mp4"'
            commands['vertical'] = {
                'command': vertical_cmd,
                'description': 'Creates 1080x1920 vertical clip perfect for Instagram Reels',
                'filename': 'clip_vertical.mp4'
            }
        
        # Horizontal clip command
        horizontal_cmd = f'{base_cmd} "clip_horizontal.mp4"'
        commands['horizontal'] = {
            'command': horizontal_cmd,
            'description': 'Creates horizontal clip maintaining original quality',
            'filename': 'clip_horizontal.mp4'
        }
        
        return commands
        
    except Exception as e:
        raise Exception(f"Command generation failed: {str(e)}")


def create_instructions_file(selected_segments: list, video_filename: str, make_vertical: bool) -> str:
    """Create a downloadable instructions file with all FFmpeg commands."""
    try:
        instructions = f"""# ClipMaker - Video Clip Generation Instructions

## Original Video File: {video_filename}

### Prerequisites:
1. Install FFmpeg: https://ffmpeg.org/download.html
2. Place this file and your video in the same folder
3. Open terminal/command prompt in that folder

### Generated Clips (20-59 seconds each):

"""
        
        for i, segment in enumerate(selected_segments, 1):
            start_time = time_to_seconds(segment.get("start", "0"))
            end_time = time_to_seconds(segment.get("end", "0"))
            duration = end_time - start_time
            
            instructions += f"""
## Clip {i} - Score: {segment.get('score', 0)}/100
**Time:** {segment.get('start')} - {segment.get('end')} ({duration:.1f}s)
**Hook:** {segment.get('hook', 'Strong opening hook')}
**Flow:** {segment.get('flow', 'Complete narrative arc')}
**Caption:** {segment.get('caption', '')}
**Why it works:** {segment.get('reason', '')}

"""
            
            commands = generate_clip_commands(video_filename, start_time, end_time, make_vertical)
            
            if make_vertical and 'vertical' in commands:
                instructions += f"""**Instagram/TikTok (Vertical 9:16):**
```bash
{commands['vertical']['command'].replace(f'"{video_filename}"', f'"{video_filename}"')}
```

"""
            
            if 'horizontal' in commands:
                instructions += f"""**YouTube/General (Horizontal):**
```bash
{commands['horizontal']['command'].replace(f'"{video_filename}"', f'"{video_filename}"')}
```

"""
            
            instructions += "---\n"
        
        instructions += f"""
### Batch Processing (All Clips at Once):

Create a batch file to generate all clips automatically:

**Windows (create run_clips.bat):**
```batch
@echo off
"""
        
        for i, segment in enumerate(selected_segments, 1):
            start_time = time_to_seconds(segment.get("start", "0"))
            end_time = time_to_seconds(segment.get("end", "0"))
            commands = generate_clip_commands(video_filename, start_time, end_time, make_vertical)
            
            if make_vertical and 'vertical' in commands:
                cmd = commands['vertical']['command'].replace('"clip_vertical.mp4"', f'"clip_{i}_vertical.mp4"')
                instructions += f'{cmd}\n'
        
        instructions += """
pause
```

**Mac/Linux (create run_clips.sh):**
```bash
#!/bin/bash
"""
        
        for i, segment in enumerate(selected_segments, 1):
            start_time = time_to_seconds(segment.get("start", "0"))
            end_time = time_to_seconds(segment.get("end", "0"))
            commands = generate_clip_commands(video_filename, start_time, end_time, make_vertical)
            
            if make_vertical and 'vertical' in commands:
                cmd = commands['vertical']['command'].replace('"clip_vertical.mp4"', f'"clip_{i}_vertical.mp4"')
                instructions += f'{cmd}\n'
        
        instructions += """
echo "All clips generated!"
```

### Notes:
- All clips are 20-59 seconds with strong hooks and smooth endings
- Replace the video filename in commands if your file has a different name
- Adjust crop parameters if your video has different dimensions
- All clips will be high quality with H.264 encoding
- Vertical clips are optimized for Instagram Reels (1080x1920)

### Online Alternatives:
If you prefer not to use command line:
1. **Kapwing.com** - Upload and crop to vertical
2. **Canva.com** - Video editor with crop tools  
3. **ClipChamp.com** - Microsoft's online video editor
4. **InShot** (mobile app) - Easy vertical conversion

Generated by ClipMaker - AI-Powered Video Analysis
"""
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.md', encoding='utf-8')
        temp_file.write(instructions)
        temp_file.close()
        
        return temp_file.name
        
    except Exception as e:
        raise Exception(f"Instructions file creation failed: {str(e)}")


def get_video_info(video_path: str, ffmpeg_path: str = 'ffmpeg') -> dict:
    """Get video information using FFmpeg with better error handling."""
    try:
        # Try imageio-ffmpeg's ffprobe first if available
        try:
            import imageio_ffmpeg
            ffprobe_path = imageio_ffmpeg.get_ffprobe_exe()
            
            # Use ffprobe (comes with ffmpeg) to get video info
            cmd = [ffprobe_path, "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", video_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                info = json.loads(result.stdout)
                
                # Find video stream
                video_stream = None
                for stream in info['streams']:
                    if stream['codec_type'] == 'video':
                        video_stream = stream
                        break
                
                if not video_stream:
                    raise Exception("No video stream found")
                
                return {
                    'duration': float(info['format']['duration']),
                    'width': int(video_stream['width']),
                    'height': int(video_stream['height'])
                }
            else:
                raise Exception("ffprobe failed")
                
        except Exception as ffprobe_error:
            st.warning(f"ffprobe unavailable ({ffprobe_error}), trying ffmpeg fallback...")
            
            # Fallback: use ffmpeg to get duration
            cmd = [ffmpeg_path, "-i", video_path, "-f", "null", "-"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            # Parse from stderr
            import re
            duration_match = re.search(r'Duration: (\d+):(\d+):(\d+\.\d+)', result.stderr)
            size_match = re.search(r'(\d+)x(\d+)', result.stderr)
            
            if duration_match and size_match:
                h, m, s = duration_match.groups()
                duration = int(h) * 3600 + int(m) * 60 + float(s)
                width, height = size_match.groups()
                
                return {
                    'duration': duration,
                    'width': int(width),
                    'height': int(height)
                }
            else:
                # Last resort: return reasonable defaults
                st.warning("Could not parse video info, using defaults")
                return {
                    'duration': 1800.0,  # 30 minutes default
                    'width': 1920,
                    'height': 1080
                }
                
    except Exception as e:
        st.warning(f"Video analysis had issues: {str(e)}, using defaults")
        return {
            'duration': 1800.0,  # 30 minutes default
            'width': 1920,
            'height': 1080
        }


def generate_clip_ffmpeg(video_path: str, start_time: float, end_time: float, make_vertical: bool = True, ffmpeg_path: str = 'ffmpeg') -> str:
    """Generate a clip using FFmpeg only."""
    try:
        # Create output file
        temp_clip = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_clip.close()
        
        if make_vertical:
            # Get video info for cropping calculation
            video_info = get_video_info(video_path, ffmpeg_path)
            original_width = video_info['width']
            original_height = video_info['height']
            
            # Calculate center crop for 9:16
            target_width = 1080
            target_height = 1920
            target_ratio = target_width / target_height
            original_ratio = original_width / original_height
            
            if original_ratio > target_ratio:
                # Horizontal video - crop from center
                crop_width = int(original_height * target_ratio)
                crop_x = (original_width - crop_width) // 2
                
                cmd = [
                    ffmpeg_path, '-y', 
                    '-i', video_path,
                    '-ss', str(start_time),
                    '-t', str(end_time - start_time),
                    '-vf', f'crop={crop_width}:{original_height}:{crop_x}:0,scale={target_width}:{target_height}',
                    '-c:a', 'aac',
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-crf', '23',
                    temp_clip.name
                ]
            else:
                # Already vertical - just scale
                cmd = [
                    ffmpeg_path, '-y',
                    '-i', video_path,
                    '-ss', str(start_time),
                    '-t', str(end_time - start_time),
                    '-vf', f'scale={target_width}:{target_height}',
                    '-c:a', 'aac',
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-crf', '23',
                    temp_clip.name
                ]
        else:
            # Horizontal clip
            cmd = [
                ffmpeg_path, '-y',
                '-i', video_path,
                '-ss', str(start_time),
                '-t', str(end_time - start_time),
                '-c:a', 'aac',
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                temp_clip.name
            ]
        
        # Execute FFmpeg
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            raise Exception(f"FFmpeg failed: {result.stderr}")
        
        if not os.path.isfile(temp_clip.name) or os.path.getsize(temp_clip.name) < 1000:
            raise Exception("Output file not created or too small")
        
        return temp_clip.name
        
    except Exception as e:
        # Clean up on failure
        try:
            if os.path.exists(temp_clip.name):
                os.unlink(temp_clip.name)
        except:
            pass
        raise Exception(f"Clip generation failed: {str(e)}")


def analyze_transcript(transcript: str, platform: str, client: OpenAI, video_duration: float = None) -> str:
    """Get segment suggestions via ChatCompletion."""
    messages = [
        {"role": "system", "content": get_system_prompt(platform, video_duration)},
        {"role": "user", "content": f"Analyze this transcript and identify the best 20-59 second segments for {platform}. Each must start with a compelling hook in the first 3 seconds and end smoothly. Remember the video is {video_duration:.1f} seconds long, so all timestamps must be within this range:\n\n{transcript}"}
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


def display_segment_selector(segments: list) -> list:
    """Display segments and let user select which ones to generate."""
    st.subheader("🎯 Select Clips to Generate")
    st.info("💡 Choose up to 5 clips to generate (to manage processing time)")
    
    # Initialize selection state if not exists
    if 'selected_clips' not in st.session_state:
        st.session_state.selected_clips = set()
    
    selected_segments = []
    
    # Sort segments by score
    sorted_segments = sorted(segments, key=lambda x: x.get('score', 0), reverse=True)
    
    for i, segment in enumerate(sorted_segments[:10], 1):  # Show top 10
        start_time = time_to_seconds(segment.get("start", "0"))
        end_time = time_to_seconds(segment.get("end", "0"))
        duration = end_time - start_time
        
        with st.expander(f"🎬 Clip {i} - Score: {segment.get('score', 0)}/100 ({duration:.1f}s)", expanded=i <= 3):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**⏱️ Time:** {segment.get('start')} - {segment.get('end')} ({duration:.1f}s)")
                st.write(f"**🪝 Hook:** {segment.get('hook', 'Strong opening hook')}")
                st.write(f"**🎬 Flow:** {segment.get('flow', 'Complete narrative')}")
                st.write(f"**📝 Caption:** {segment.get('caption', 'No caption')}")
                st.write(f"**💡 Why this will work:** {segment.get('reason', 'No reason provided')}")
            
            with col2:
                # Check if this clip was previously selected
                is_selected = i in st.session_state.selected_clips
                
                # Disable if we have 5 selections and this isn't one of them
                is_disabled = len(st.session_state.selected_clips) >= 5 and not is_selected
                
                selected = st.checkbox(
                    f"Generate", 
                    key=f"select_{i}",
                    value=is_selected,
                    disabled=is_disabled
                )
                
                # Update session state based on checkbox
                if selected and i not in st.session_state.selected_clips:
                    if len(st.session_state.selected_clips) < 5:
                        st.session_state.selected_clips.add(i)
                elif not selected and i in st.session_state.selected_clips:
                    st.session_state.selected_clips.remove(i)
                
                if selected and len(st.session_state.selected_clips) > 5:
                    st.warning("Maximum 5 clips allowed")
    
    # Build selected segments list based on session state
    for i in st.session_state.selected_clips:
        if i <= len(sorted_segments):
            segment = sorted_segments[i-1]  # Convert to 0-based index
            selected_segments.append({**segment, "display_index": i})
    
    st.write(f"**Selected:** {len(selected_segments)}/5 clips")
    
    return selected_segments


def download_drive_file(drive_url: str, out_path: str) -> str:
    """Download a Google Drive file given its share URL to out_path."""
    import requests
    import time
    
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
        
        st.info(f"📥 Attempting to download file ID: {file_id}")
        
        # Try gdown first
        try:
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            result = gdown.download(download_url, out_path, quiet=False)
            
            if result and os.path.isfile(result) and os.path.getsize(result) > 0:
                return result
        except Exception as gdown_error:
            error_msg = str(gdown_error).lower()
            if "too many users" in error_msg or "rate limit" in error_msg:
                st.error("🚫 Google Drive rate limit reached!")
                st.markdown("""
                **Solutions:**
                1. **Wait 1-24 hours** and try again
                2. **Use direct upload** in the sidebar instead
                3. **Manual download** then upload the file
                """)
                raise Exception("Google Drive rate limit - please use direct upload")
        
        raise Exception("Download failed - please use direct file upload instead")
        
    except Exception as e:
        st.error(f"Download error: {str(e)}")
        raise


# ----------
# Streamlit App
# ----------

def main():
    st.set_page_config(page_title="ClipMaker", layout="wide")
    
    st.title("🎬 Long‑form to Short‑form ClipMaker")
    st.markdown("Transform your long-form content into viral short-form clips (20-59s) with compelling hooks!")

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
            st.sidebar.success("✅ FFmpeg available (imageio-ffmpeg)")
        else:
            raise Exception("FFmpeg test failed")
            
    except Exception as e:
        st.sidebar.warning("⚠️ imageio-ffmpeg not available, trying system FFmpeg")
        
        # Fallback to system FFmpeg
        try:
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
            if result.returncode == 0:
                ffmpeg_available = True
                ffmpeg_path = 'ffmpeg'
                st.sidebar.success("✅ System FFmpeg available")
        except:
            st.sidebar.error("❌ No FFmpeg found")
            
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
        
        if make_vertical:
            st.sidebar.success("📱 Perfect 1080x1920 Instagram format")
        else:
            st.sidebar.info("📺 Original horizontal format")

    # Video source
    st.sidebar.subheader("📹 Video Source")
    st.sidebar.info("💡 **Direct upload recommended** to avoid rate limits!")
    
    # Direct upload (primary option)
    uploaded = st.sidebar.file_uploader(
        "📁 Upload video file", 
        type=["mp4", "mov", "mkv", "avi", "webm"],
        help="Direct upload is most reliable"
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
    
    # Google Drive option (secondary)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔗 Google Drive (Alternative)")
    drive_url = st.sidebar.text_input(
        "Google Drive URL", 
        placeholder="https://drive.google.com/file/d/...",
        help="May hit rate limits - direct upload recommended"
    )
    
    if drive_url and not uploaded:
        if st.sidebar.button("📥 Download from Drive"):
            with st.spinner("Downloading from Google Drive…"):
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
                    st.error(f"Drive download failed: {str(e)}")
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

    # Main processing
    if st.button("🚀 Generate Clips", type="primary"):
        if not video_path or not os.path.isfile(video_path):
            st.error("Video file not found. Please reload your video.")
            return
            
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
        status_text.text("🤖 Analyzing transcript for viral 20-59s segments with hooks...")
        progress_bar.progress(75)
        
        try:
            ai_json = analyze_transcript(transcript, platform, client, video_info['duration'])
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
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete! Select clips below.")
        
        st.success(f"🎯 Found {len(segments)} potential clips (20-59s each with hooks)!")

    # Check if we have analyzed segments to display
    if 'ai_segments' in st.session_state and st.session_state.get('video_analyzed', False):
        segments = st.session_state['ai_segments']
        
        # Let user select which clips to generate
        selected_segments = display_segment_selector(segments)
        
        if not selected_segments:
            st.info("👆 Please select clips to generate using the checkboxes above.")
        else:
            # Check if we can generate actual clips or just instructions
            if ffmpeg_available:
                # Generate actual clips
                if st.button(f"🚀 Generate {len(selected_segments)} Selected Clips", type="primary"):
                    st.markdown("---")
                    st.header("🎬 Generating Selected Clips")
                    
                    # Clear any previous clips
                    if 'generated_clips' in st.session_state:
                        st.session_state.generated_clips.clear()
                    else:
                        st.session_state.generated_clips = []
                    
                    # Generate clips one by one
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, segment in enumerate(selected_segments, 1):
                        status_text.text(f"🎬 Generating clip {i}/{len(selected_segments)}...")
                        
                        try:
                            start_time = time_to_seconds(segment.get("start", "0"))
                            end_time = time_to_seconds(segment.get("end", "0"))
                            duration = end_time - start_time
                            
                            st.info(f"Creating clip {segment.get('display_index', i)}: {start_time:.1f}s - {end_time:.1f}s ({duration:.1f}s)")
                            
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
                                "index": segment.get('display_index', i)
                            }
                            
                            st.session_state.generated_clips.append(clip_info)
                            
                            # Display the clip immediately
                            st.markdown(f"### 🎬 Clip {segment.get('display_index', i)} (Score: {clip_info['score']}/100)")
                            
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.video(clip_info["path"])
                                st.markdown("**📝 Suggested Caption:**")
                                st.code(clip_info["caption"], language="text")
                            
                            with col2:
                                st.markdown("**📊 Details:**")
                                st.write(f"⏱️ **Duration:** {clip_info['duration']}")
                                st.write(f"🕐 **Time:** {clip_info['start']} - {clip_info['end']}")
                                st.write(f"🎯 **Score:** {clip_info['score']}/100")
                                st.write(f"📱 **Format:** {clip_info['format']}")
                                st.write(f"💾 **Size:** {clip_info['file_size']}")
                                
                                st.markdown("**🪝 Hook:**")
                                st.write(clip_info['hook'])
                                
                                st.markdown("**🎬 Flow:**")
                                st.write(clip_info['flow'])
                                
                                st.markdown("**💡 Why this will work:**")
                                st.write(clip_info['reason'])
                                
                                # Download button
                                with open(clip_info["path"], "rb") as file:
                                    file_extension = "vertical" if make_vertical else "horizontal"
                                    st.download_button(
                                        label="⬇️ Download Clip",
                                        data=file,
                                        file_name=f"clip_{clip_info['index']}_{platform.replace(' ', '_').lower()}_{file_extension}.mp4",
                                        mime="video/mp4",
                                        use_container_width=True,
                                        key=f"download_{clip_info['index']}"
                                    )
                            
                            st.markdown("---")
                            st.success(f"✅ Clip {i} generated successfully!")
                            
                        except Exception as e:
                            st.error(f"❌ Failed to generate clip {i}: {str(e)}")
                        
                        progress_bar.progress(i / len(selected_segments))
                    
                    status_text.text("✅ All selected clips generated!")
                    
                    # Summary
                    successful_clips = len(st.session_state.generated_clips)
                    if successful_clips > 0:
                        st.success(f"🎉 Successfully generated {successful_clips}/{len(selected_segments)} clips!")
                        
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
                    else:
                        st.error("❌ No clips were successfully generated.")
            else:
                # Generate instructions instead
                if st.button(f"🚀 Generate Instructions for {len(selected_segments)} Selected Clips", type="primary"):
                    st.markdown("---")
                    st.header("📋 Clip Generation Instructions")
                    
                    st.info("Since video processing libraries aren't available in this environment, we'll provide you with ready-to-use FFmpeg commands!")
                    
                    try:
                        # Get the original video filename
                        video_filename = os.path.basename(video_path)
                        if 'video_path' in st.session_state:
                            # Try to get original uploaded filename
                            video_filename = "your_video.mp4"  # Generic name
                        
                        # Create instructions file
                        instructions_file = create_instructions_file(selected_segments, video_filename, make_vertical)
                        
                        # Display preview of selected clips
                        st.subheader("📝 Selected Clips Summary (20-59s with Hooks)")
                        
                        for i, segment in enumerate(selected_segments, 1):
                            start_time = time_to_seconds(segment.get("start", "0"))
                            end_time = time_to_seconds(segment.get("end", "0"))
                            duration = end_time - start_time
                            
                            with st.expander(f"🎬 Clip {i} - Score: {segment.get('score', 0)}/100 ({duration:.1f}s)", expanded=False):
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    st.write(f"**⏱️ Time:** {segment.get('start')} - {segment.get('end')} ({duration:.1f}s)")
                                    st.write(f"**🪝 Hook:** {segment.get('hook', 'Strong opening hook')}")
                                    st.write(f"**🎬 Flow:** {segment.get('flow', 'Complete narrative arc')}")
                                    st.write(f"**📝 Caption:** {segment.get('caption', '')}")
                                    st.write(f"**💡 Why this works:** {segment.get('reason', '')}")
                                
                                with col2:
                                    # Show sample command
                                    if make_vertical:
                                        sample_cmd = f'ffmpeg -i "your_video.mp4" -ss {start_time} -t {duration} -vf "crop=1215:2160:1312:0,scale=1080:1920" -c:v libx264 -c:a aac "clip_{i}_vertical.mp4"'
                                    else:
                                        sample_cmd = f'ffmpeg -i "your_video.mp4" -ss {start_time} -t {duration} -c:v libx264 -c:a aac "clip_{i}.mp4"'
                                    
                                    st.code(sample_cmd, language="bash")
                        
                        # Download instructions
                        st.subheader("📥 Download Complete Instructions")
                        
                        with open(instructions_file, 'r', encoding='utf-8') as f:
                            instructions_content = f.read()
                        
                        st.download_button(
                            label="📋 Download FFmpeg Instructions (.md)",
                            data=instructions_content,
                            file_name="clipmaker_instructions.md",
                            mime="text/markdown",
                            help="Complete instructions with all FFmpeg commands for your selected clips"
                        )
                        
                        # Quick start guide
                        st.subheader("🚀 Quick Start Guide")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("""
                            **For Beginners:**
                            1. 📥 Download the instructions file above
                            2. 🌐 Use **Kapwing.com** or **Canva.com**
                            3. 📤 Upload your video and crop to vertical
                            4. ⏰ Set the time ranges from our analysis
                            5. 🪝 Ensure strong hooks in first 3 seconds
                            """)
                        
                        with col2:
                            st.markdown("""
                            **For Advanced Users:**
                            1. 📥 Download the instructions file
                            2. 💻 Install FFmpeg on your computer
                            3. 🎬 Run the provided commands
                            4. 📱 Get perfect Instagram-ready clips!
                            5. 🪝 Each clip has optimized hooks and flow
                            """)
                        
                        # Online alternatives
                        st.subheader("🌐 Online Video Editors (No Installation Needed)")
                        
                        st.markdown("""
                        | Tool | Best For | Link |
                        |------|----------|------|
                        | **Kapwing** | Easy vertical conversion + hooks | kapwing.com |
                        | **Canva** | Templates + video editing | canva.com |
                        | **ClipChamp** | Microsoft's editor | clipchamp.com |
                        | **InShot** | Mobile app with hook features | Download from app store |
                        """)
                        
                        st.success("🎉 Instructions generated! Use the FFmpeg commands or online tools to create your viral clips with compelling hooks!")
                        
                        # Clean up
                        try:
                            os.unlink(instructions_file)
                        except:
                            pass
                            
                    except Exception as e:
                        st.error(f"Failed to generate instructions: {str(e)}")
    
    # Display previously generated clips if they exist
    if 'generated_clips' in st.session_state and st.session_state.generated_clips:
        st.markdown("---")
        st.header("📁 Previously Generated Clips")
        
        for clip_info in st.session_state.generated_clips:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"### 🎬 Clip {clip_info['index']} (Score: {clip_info['score']}/100)")
                if os.path.isfile(clip_info["path"]):
                    st.video(clip_info["path"])
                else:
                    st.error("Clip file no longer available")
                st.markdown("**📝 Suggested Caption:**")
                st.code(clip_info["caption"], language="text")
            
            with col2:
                st.markdown("**📊 Details:**")
                st.write(f"⏱️ **Duration:** {clip_info['duration']}")
                st.write(f"🕐 **Time:** {clip_info['start']} - {clip_info['end']}")
                st.write(f"🎯 **Score:** {clip_info['score']}/100")
                st.write(f"📱 **Format:** {clip_info['format']}")
                st.write(f"💾 **Size:** {clip_info['file_size']}")
                
                st.markdown("**🪝 Hook:**")
                st.write(clip_info.get('hook', 'Strong opening hook'))
                
                st.markdown("**🎬 Flow:**")
                st.write(clip_info.get('flow', 'Complete narrative arc'))
                
                st.markdown("**💡 Why this will work:**")
                st.write(clip_info['reason'])
                
                # Download button if file still exists
                if os.path.isfile(clip_info["path"]):
                    with open(clip_info["path"], "rb") as file:
                        file_extension = "vertical" if make_vertical else "horizontal"
                        st.download_button(
                            label="⬇️ Download Clip",
                            data=file,
                            file_name=f"clip_{clip_info['index']}_{platform.replace(' ', '_').lower()}_{file_extension}.mp4",
                            mime="video/mp4",
                            use_container_width=True,
                            key=f"persistent_download_{clip_info['index']}"
                        )
                else:
                    st.error("File no longer available")
            
            st.markdown("---")

    # Reset button
    if st.sidebar.button("🔄 Start Over"):
        # Clear all session state
        for key in list(st.session_state.keys()):
            if key.startswith(('ai_segments', 'video_analyzed', 'generated_clips', 'selected_clips')):
                del st.session_state[key]
        st.rerun()


if __name__ == "__main__":
    main()
