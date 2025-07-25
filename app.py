# app.py - Working ClipMaker with Parameter Selection
import os
import json
import tempfile
import traceback
import re
import streamlit as st
from moviepy.video.io.VideoFileClip import VideoFileClip
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


def get_system_prompt(platform: str, selected_parameters: list, video_duration: float = None) -> str:
    if video_duration:
        duration_minutes = int(video_duration // 60)
        duration_seconds = int(video_duration % 60)
        max_start_time = max(0, video_duration - 60)  # Latest possible start for a 60s clip
        duration_info = f"""
CRITICAL VIDEO CONSTRAINTS:
- Video duration: {video_duration:.1f} seconds ({duration_minutes}:{duration_seconds:02d})
- ALL timestamps MUST be between 00:00:00 and {duration_minutes//60:02d}:{(duration_minutes%60):02d}:{duration_seconds:02d}
- Maximum start time for any clip: {int(max_start_time//60):02d}:{int(max_start_time%60):02d}:{int(max_start_time%60):02d}
- DO NOT generate timestamps beyond the video duration
- Estimate timestamps based on transcript position (beginning = early timestamps, end = later timestamps)
"""
    else:
        duration_info = ""
    
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
    
    return f"""You are a content strategist and social media editor trained to analyze long-form video/podcast transcripts. Your task is to identify 15–60 second segments that are highly likely to perform well as short-form content on {platform}.

{duration_info}

CRITICAL REQUIREMENTS FOR VIRAL SUCCESS:
🎯 ZERO-SECOND HOOK: The first 1-3 seconds MUST grab attention immediately - no slow intros or context setting. Start with the most compelling moment, question, or statement.
🔚 PROPER ENDING: Clips must have a satisfying conclusion - avoid abrupt cuts mid-sentence. End with a complete thought, punchline, or call-to-action.  
⏱️ DURATION: Clips must be between 15-60 seconds. Shorter clips often perform better due to higher completion rates.

SELECTED FOCUS PARAMETERS:
{parameters_text}

PRIORITIZE CONTENT THAT MATCHES THE SELECTED PARAMETERS ABOVE. Focus your analysis on finding segments that excel in these specific areas.

Key Parameters for Cut Selection:
🧠 Educational Value: Does it provide insight, tip, or new perspective?
😲 Surprise Factor: Is there a twist, unexpected truth, or myth-busting idea?
😍 Emotional Impact: Does it make you feel something? (inspiration, laughter, shock)
🔁 Replayability: Would a viewer want to watch this again?
📉 Drop-off Resistance: Does it have a POWERFUL hook in the first 3 seconds?
🧠 Relatability: Does it reflect common struggles, habits, or aspirations?
🎤 Speaker Delivery: Energy, voice modulation, pauses
🪞 Format Fit: Complete thought that makes sense out of context
🎬 Hook Quality: Opens with intrigue, question, bold statement, or surprising fact

TIMESTAMP GENERATION RULES:
- Estimate timestamps based on transcript position (early content = early times, later content = later times)
- Use average speaking pace of 150-200 words per minute to estimate timing
- Content at the beginning of transcript should have timestamps like 00:01:30, 00:03:45
- Content in the middle should have timestamps around the middle of the video duration
- Content at the end should have timestamps near (but not exceeding) the video end time
- Always ensure end timestamp is no more than 60 seconds after start timestamp
- Double-check that ALL timestamps are within the video duration limits

For each recommended cut, provide:
1. Start and end timestamps (HH:MM:SS format) - MUST be within video duration
2. Reason why this segment will work (focusing on hook strength, selected parameters, and complete ending)
3. Predicted engagement score (0–100) — your confidence in performance
4. Suggested caption for social media with emojis/hashtags

Output ONLY valid JSON as an array of objects with these exact keys:
- start: "HH:MM:SS"
- end: "HH:MM:SS" 
- reason: "brief rationale focusing on hook strength, selected parameters, virality factors, and proper ending"
- score: integer (0-100)
- caption: "social media caption with emojis and hashtags"

Example format:
[
  {{
    "start": "00:02:15",
    "end": "00:02:45",
    "reason": "Opens with shocking statistic that hooks immediately, myth-busts credit score beliefs (Educational Value + Surprise Factor), ends with complete actionable advice",
    "score": 88,
    "caption": "Wait... credit scores don't work how you think?! 😱 #MoneyMyths #FinanceTips #CreditScore"
  }}
]"""


def extract_audio_from_video(video_path: str) -> str:
    """Extract audio from video and compress if needed."""
    try:
        # Create temporary audio file
        audio_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        
        # Load video and extract audio
        video = VideoFileClip(video_path)
        audio = video.audio
        
        # Write audio to temp file with compression
        audio.write_audiofile(
            audio_temp.name,
            codec='mp3',
            bitrate='64k'  # Lower bitrate to reduce file size
        )
        
        # Clean up
        audio.close()
        video.close()
        
        return audio_temp.name
    except Exception as e:
        st.error(f"Audio extraction failed: {str(e)}")
        raise


def split_audio_file(audio_path: str, chunk_duration_minutes: int = 10) -> list:
    """Split audio file into smaller chunks if it's too large."""
    try:
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        
        # If file is small enough, return as is
        if file_size_mb <= 20:  # Leave some margin below 25MB limit
            return [audio_path]
        
        st.info(f"Audio file is {file_size_mb:.1f}MB. Splitting into chunks...")
        
        # Load audio clip
        from moviepy.audio.io.AudioFileClip import AudioFileClip
        audio_clip = AudioFileClip(audio_path)
        duration = audio_clip.duration
        
        chunk_duration_seconds = chunk_duration_minutes * 60
        chunks = []
        
        # Split into chunks
        start_time = 0
        chunk_num = 1
        
        while start_time < duration:
            end_time = min(start_time + chunk_duration_seconds, duration)
            
            # Create chunk
            chunk_temp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_chunk_{chunk_num}.mp3")
            chunk_audio = audio_clip.subclipped(start_time, end_time)
            
            chunk_audio.write_audiofile(
                chunk_temp.name,
                codec='mp3',
                bitrate='64k'
            )
            
            chunks.append(chunk_temp.name)
            chunk_audio.close()
            
            start_time = end_time
            chunk_num += 1
        
        audio_clip.close()
        st.success(f"Split audio into {len(chunks)} chunks")
        return chunks
        
    except Exception as e:
        st.error(f"Audio splitting failed: {str(e)}")
        raise


def transcribe_audio(path: str, client: OpenAI) -> str:
    """Transcribe audio via Whisper-1, handling large files by chunking."""
    try:
        # First extract and compress audio from video
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
                
                # Clean up chunk file
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
    words_per_section = transcript_length // 5  # Divide into 5 sections
    
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

Focus on segments with powerful hooks in the first 3 seconds and proper endings. PRIORITIZE content that matches the selected parameters: {', '.join(selected_parameters)}.

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
            if all(key in seg for key in ["start", "end", "reason", "score", "caption"]):
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
                    
                    if end_seconds - start_seconds < 15:
                        st.warning(f"Segment {i+1}: Duration too short ({end_seconds - start_seconds:.1f}s < 15s). Skipping.")
                        continue
                    
                    if end_seconds - start_seconds > 60:
                        st.warning(f"Segment {i+1}: Duration too long ({end_seconds - start_seconds:.1f}s > 60s). Adjusting.")
                        # Adjust end time to 60 seconds from start
                        new_end_seconds = start_seconds + 60
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


def generate_clips(video_path: str, segments: list) -> list:
    """Use moviepy to cut video segments."""
    clips = []
    
    try:
        # Load video once
        video = VideoFileClip(video_path)
        total_duration = video.duration
        
        st.info(f"Video duration: {total_duration:.1f} seconds")
        
        for i, seg in enumerate(segments, start=1):
            try:
                start_time = time_to_seconds(seg.get("start", "0"))
                end_time = time_to_seconds(seg.get("end", "0"))
                caption = seg.get("caption", f"clip_{i}")
                score = seg.get("score", 0)
                reason = seg.get("reason", "")
                
                st.info(f"Processing clip {i}: {start_time:.1f}s - {end_time:.1f}s")
                
                # Validate times
                if start_time >= end_time:
                    st.warning(f"Skipping segment {i}: Invalid time range ({start_time:.1f}s >= {end_time:.1f}s)")
                    continue
                    
                if start_time >= total_duration:
                    st.warning(f"Skipping segment {i}: Start time ({start_time:.1f}s) beyond video duration ({total_duration:.1f}s)")
                    continue
                    
                if end_time > total_duration:
                    st.warning(f"Adjusting segment {i}: End time beyond video duration")
                    end_time = total_duration
                
                # Ensure minimum clip duration
                if end_time - start_time < 1:
                    st.warning(f"Skipping segment {i}: Clip too short ({end_time - start_time:.1f}s)")
                    continue
                
                # Create clip using the correct method
                try:
                    # Try different methods for creating subclips
                    if hasattr(video, 'subclipped'):
                        clip = video.subclipped(start_time, end_time)
                    elif hasattr(video, 'subclip'):
                        clip = video.subclip(start_time, end_time)
                    else:
                        # Fallback method
                        clip = video.cutout(0, start_time).cutout(end_time - start_time, video.duration)
                        
                except AttributeError as attr_error:
                    st.error(f"MoviePy method error for clip {i}: {str(attr_error)}")
                    # Try alternative approach
                    from moviepy.video.fx import subclip
                    clip = subclip(video, start_time, end_time)
                
                # Create temporary file with better naming
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False, 
                    suffix=".mp4",
                    prefix=f"clip_{i}_"
                )
                
                st.info(f"Writing clip {i} to file...")
                
                # Write video file with error handling
                try:
                    clip.write_videofile(
                        temp_file.name, 
                        codec="libx264", 
                        audio_codec="aac",
                        temp_audiofile_path=tempfile.gettempdir(),
                        preset='ultrafast',  # Faster encoding
                        fps=24  # Standard fps
                    )
                except Exception as write_error:
                    st.error(f"Error writing clip {i}: {str(write_error)}")
                    # Try simpler encoding
                    clip.write_videofile(
                        temp_file.name,
                        preset='ultrafast'
                    )
                
                # Verify file was created
                if os.path.isfile(temp_file.name) and os.path.getsize(temp_file.name) > 0:
                    clips.append({
                        "path": temp_file.name, 
                        "caption": caption,
                        "score": score,
                        "reason": reason,
                        "start": seg.get("start"),
                        "end": seg.get("end"),
                        "duration": f"{end_time - start_time:.1f}s"
                    })
                    st.success(f"✅ Created clip {i}")
                else:
                    st.error(f"Failed to create clip {i}: File not generated")
                
                # Close clip to free memory
                clip.close()
                
            except Exception as e:
                st.error(f"Error creating clip {i}: {str(e)}")
                st.error(f"Segment data: {seg}")
                continue
        
        # Close video to free memory
        video.close()
        
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        raise
    
    return clips


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
            # Try different patterns
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
        
        # Method 1: Try gdown first (fastest when it works)
        try:
            st.info("🔄 Trying gdown method...")
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            result = gdown.download(download_url, out_path, quiet=False)
            
            if result and os.path.isfile(result) and os.path.getsize(result) > 0:
                return result
            else:
                st.warning("gdown failed, trying alternative method...")
        except Exception as gdown_error:
            st.warning(f"gdown failed: {str(gdown_error)}")
        
        # Method 2: Direct requests with session (handles larger files and auth issues)
        try:
            st.info("🔄 Trying direct download method...")
            session = requests.Session()
            
            # First, get the file info page
            file_url = f"https://drive.google.com/file/d/{file_id}/view"
            response = session.get(file_url)
            
            if response.status_code != 200:
                raise Exception(f"Cannot access file: HTTP {response.status_code}")
            
            # Try direct download URL
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            
            # Make request with stream=True for large files
            with session.get(download_url, stream=True) as response:
                if response.status_code == 200:
                    # Check if it's the actual file or a download warning page
                    content_type = response.headers.get('content-type', '')
                    
                    if 'text/html' in content_type:
                        # This means we got a download warning page, need to extract the real download link
                        html_content = response.text
                        
                        # Look for download confirmation link
                        confirm_patterns = [
                            r'action="([^"]*)"[^>]*>.*?download',
                            r'href="(/uc\?export=download[^"]*)"',
                            r'"downloadUrl":"([^"]*)"'
                        ]
                        
                        real_download_url = None
                        for pattern in confirm_patterns:
                            match = re.search(pattern, html_content, re.IGNORECASE | re.DOTALL)
                            if match:
                                real_download_url = match.group(1)
                                if real_download_url.startswith('/'):
                                    real_download_url = 'https://drive.google.com' + real_download_url
                                break
                        
                        if real_download_url:
                            st.info("🔄 Found confirmation link, downloading...")
                            with session.get(real_download_url, stream=True) as final_response:
                                if final_response.status_code == 200:
                                    with open(out_path, 'wb') as f:
                                        for chunk in final_response.iter_content(chunk_size=8192):
                                            if chunk:
                                                f.write(chunk)
                                else:
                                    raise Exception(f"Final download failed: HTTP {final_response.status_code}")
                        else:
                            raise Exception("Could not find download confirmation link")
                    else:
                        # Direct file download
                        with open(out_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                else:
                    raise Exception(f"Download request failed: HTTP {response.status_code}")
            
            # Verify file was downloaded
            if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
                return out_path
            else:
                raise Exception("File download completed but file is empty or missing")
                
        except Exception as requests_error:
            st.warning(f"Direct download failed: {str(requests_error)}")
        
        # Method 3: Try gdown with fuzzy matching as last resort
        try:
            st.info("🔄 Trying gdown with fuzzy matching...")
            result = gdown.download(
                f"https://drive.google.com/file/d/{file_id}/view", 
                out_path, 
                quiet=False, 
                fuzzy=True
            )
            
            if result and os.path.isfile(result) and os.path.getsize(result) > 0:
                return result
        except Exception as fuzzy_error:
            st.warning(f"Fuzzy gdown failed: {str(fuzzy_error)}")
        
        # If all methods fail
        raise Exception(
            f"All download methods failed. Please ensure:\n"
            f"1. The file is shared with 'Anyone with the link can view'\n"
            f"2. The file is not too large (Google Drive has download limits)\n"
            f"3. The link is correct and accessible\n"
            f"4. Try downloading manually first to test: "
            f"https://drive.google.com/uc?export=download&id={file_id}"
        )
        
    except Exception as e:
        st.error(f"Download error: {str(e)}")
        raise


def display_clips(clips: list, platform: str, start_index: int = 0, max_clips: int = 5):
    """Display clips with download buttons, maintaining state between downloads."""
    clips_to_show = clips[start_index:start_index + max_clips]
    
    for i, clip in enumerate(clips_to_show, start=start_index + 1):
        # Use unique keys to maintain state
        clip_key = f"clip_{start_index}_{i}"
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"Clip #{i} (Score: {clip.get('score', 0)}/100)")
            
            # Check if video file exists before trying to display
            video_path = clip.get("path")
            if video_path and os.path.isfile(video_path):
                try:
                    # Don't use key parameter for st.video - it's not supported in all versions
                    st.video(video_path)
                except Exception as video_error:
                    st.error(f"Error displaying video for clip {i}: {str(video_error)}")
                    st.info("Video file may have been corrupted. Try regenerating clips.")
            else:
                st.error(f"Video file not found for clip {i}. Please regenerate clips.")
                st.info(f"Expected path: {video_path}")
            
            # Caption with copy button
            st.markdown("**📝 Suggested Caption:**")
            st.code(clip.get("caption", "No caption available"), language="text")
            
        with col2:
            st.markdown("**📊 Details:**")
            st.write(f"⏱️ **Duration:** {clip.get('duration', 'N/A')}")
            st.write(f"🕐 **Time:** {clip.get('start', 'N/A')} - {clip.get('end', 'N/A')}")
            st.write(f"🎯 **Score:** {clip.get('score', 0)}/100")
            
            st.markdown("**💡 Why this will work:**")
            st.write(clip.get('reason', 'No reason provided'))
            
            # Download button with file existence check
            video_path = clip.get("path")
            if video_path and os.path.isfile(video_path):
                try:
                    with open(video_path, "rb") as file:
                        file_data = file.read()
                        st.download_button(
                            label="⬇️ Download Clip",
                            data=file_data,
                            file_name=f"clip_{i}_{platform.replace(' ', '_').lower()}.mp4",
                            mime="video/mp4",
                            use_container_width=True,
                            key=f"download_{clip_key}"
                        )
                except Exception as download_error:
                    st.error(f"Error preparing download for clip {i}: {str(download_error)}")
            else:
                st.error("❌ File not available for download")
                if st.button(f"🔄 Regenerate Clip {i}", key=f"regen_{clip_key}"):
                    st.info("Please use 'Generate Clips' to recreate all clips.")
        
        st.markdown("---")
