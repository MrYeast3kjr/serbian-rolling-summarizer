import os
import sys
import subprocess
import re
import time
import json
from datetime import datetime
import argparse
import requests
import threading
import queue
try:
    from faster_whisper import WhisperModel
except ImportError:
    print("Error: 'faster-whisper' package is not installed in the python environment.")
    print("Please install it using: pip install faster-whisper")
    sys.exit(1)

def sanitize_filename(name):
    clean = re.sub(r'[\\/*?:"<>|]', "", name)
    clean = re.sub(r'\s+', ' ', clean)
    return clean.strip()

def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"

def download_audio(youtube_url, download_dir):
    print(f"\n--- Downloading Audio from: {youtube_url} ---")
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
        print(f"Created directory: {download_dir}")

    print("Fetching video title...")
    title_cmd = ["yt-dlp", "--get-title", youtube_url]
    try:
        title_result = subprocess.run(title_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        video_title = None
        for enc in ['utf-8', 'cp1250', 'cp852']:
            try:
                video_title = title_result.stdout.decode(enc).strip()
                if video_title:
                    break
            except UnicodeDecodeError:
                continue
        if not video_title:
            video_title = title_result.stdout.decode('utf-8', errors='replace').strip()
        print(f"Video Title: {video_title}")
    except Exception as e:
        print(f"Warning: Could not fetch video title automatically. Error: {e}")
        video_title = "youtube_audio"

    print("Downloading audio track...")
    video_id = "youtube_audio"
    match = re.search(r'(?:v=|\/)([a-zA-Z0-9_-]{11})(?:\?|&|$)', youtube_url)
    if match:
        video_id = match.group(1)
    else:
        id_cmd = ["yt-dlp", "--get-id", youtube_url]
        try:
            id_result = subprocess.run(id_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            video_id = id_result.stdout.decode('utf-8', errors='replace').strip()
        except Exception:
            pass

    temp_filename = f"yt_{video_id}"
    audio_path_temp = os.path.join(download_dir, f"{temp_filename}.mp3")
    
    if os.path.exists(audio_path_temp):
        print(f"Local audio file '{audio_path_temp}' already exists. Reusing existing file to save bandwidth and skip download!")
        return audio_path_temp, video_title

    dl_cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "mp3",
        "--audio-quality", "0",
        "-o", os.path.join(download_dir, f"{temp_filename}.%(ext)s"),
        youtube_url
    ]
    
    try:
        subprocess.run(dl_cmd, check=True)
        if os.path.exists(audio_path_temp):
            print("Audio downloaded and converted successfully.")
            return audio_path_temp, video_title
        else:
            files = [f for f in os.listdir(download_dir) if f.startswith(temp_filename) and f.endswith(".mp3")]
            if files:
                return os.path.join(download_dir, files[0]), video_title
            raise FileNotFoundError("Could not find downloaded audio file.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing yt-dlp: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Failed to download audio: {e}")
        sys.exit(1)


def transcribe_audio(audio_path, video_title, youtube_url, download_dir, model_name_or_path, language):
    print(f"\n--- Transcribing: {video_title} ---")
    
    # Resolve local path if short name is passed
    if model_name_or_path == "base":
        model_path = r"c:\Users\fdjok\gemini\models\whisper-base"
    elif model_name_or_path == "large-v3":
        model_path = r"c:\Users\fdjok\gemini\models\whisper-large-v3"
    else:
        model_path = model_name_or_path

    if not os.path.exists(model_path):
        print(f"Warning: Local model path not found at '{model_path}'. Falling back to downloading/caching model '{model_name_or_path}' from Hugging Face...")
        model_path = model_name_or_path
        
    print(f"Using Whisper model path: {model_path}")
    
    # Load model
    start_load = time.time()
    try:
        print("Initializing Whisper model on CUDA (GPU) with float16...")
        model = WhisperModel(model_path, device="cuda", compute_type="float16")
        print("Model loaded successfully on CUDA!")
    except Exception as e:
        print(f"CUDA initialization failed: {e}. Falling back to CPU...")
        model = WhisperModel(model_path, device="cpu", compute_type="float32")
    end_load = time.time()
    print(f"Model loaded in {end_load - start_load:.2f} seconds.")

    sanitized_title = sanitize_filename(video_title)
    if not sanitized_title:
        sanitized_title = "transcript"
        
    output_txt_path = os.path.join(download_dir, f"{sanitized_title}_full_transcript.txt")

    print(f"Starting transcription (beam_size=5, language={language})...")
    
    # Fix for Windows console encoding
    sys.stdout.reconfigure(encoding='utf-8')
    
    start_transcribe = time.time()
    segments, info = model.transcribe(audio_path, beam_size=5, language=language, vad_filter=True)
    
    audio_duration = info.duration

    print("\n--- Processing Transcription Segments ---")
    segment_list = []
    
    for segment in segments:
        segment_list.append((segment.start, segment.end, segment.text))
        timestamp_str = f"[{format_timestamp(segment.start)} -> {format_timestamp(segment.end)}]"
        print(f"{timestamp_str} {segment.text}")
        
    end_transcribe = time.time()
    
    transcribe_time = end_transcribe - start_transcribe
    speed_ratio = audio_duration / transcribe_time if transcribe_time > 0 else 0
    
    print("\n--- Speed Metrics ---")
    print(f"Audio Duration: {audio_duration:.2f} seconds ({format_timestamp(audio_duration)})")
    print(f"Transcription Time: {transcribe_time:.2f} seconds")
    print(f"Transcription Speed: {speed_ratio:.2f}x Real-time")
    
    # Write full transcript to plain text file
    with open(output_txt_path, "w", encoding="utf-8") as txt:
        txt.write(f"Transcript: {video_title}\n")
        if youtube_url:
            txt.write(f"URL: {youtube_url}\n")
        else:
            txt.write(f"Source: Local Audio File ({audio_path})\n")
        txt.write(f"Language: {language}\n")
        txt.write("="*50 + "\n\n")
        for start, end, text in segment_list:
            txt.write(f"[{format_timestamp(start)} -> {format_timestamp(end)}] {text.strip()}\n")
            
    print(f"\nTranscription completed!")
    print(f"Saved Full Transcript to: {output_txt_path}")
            
    return output_txt_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio using Whisper with rolling LLM summaries.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url", help="YouTube video URL")
    group.add_argument("--audio", help="Path to local audio file")
    
    parser.add_argument("--model", default="base", help="Model size ('base' or 'large-v3')")
    parser.add_argument("--language", default="en", help="Language of transcription (default: 'en')")
    parser.add_argument("--output-dir", default=None, help="Directory to save the transcripts and summaries")
    
    args = parser.parse_args()
    
    # Default transcription language
    transcribe_lang = args.language
    
    if args.url:
        # YouTube workflow
        download_dir = args.output_dir if args.output_dir else r"C:\Users\fdjok\gemini\transcripts\yt_downloads"
        audio_path, video_title = download_audio(args.url, download_dir)
        transcribe_audio(
            audio_path=audio_path,
            video_title=video_title,
            youtube_url=args.url,
            download_dir=download_dir,
            model_name_or_path=args.model,
            language=transcribe_lang
        )
    else:
        # Local audio workflow
        audio_path = os.path.abspath(args.audio)
        if not os.path.exists(audio_path):
            print(f"Error: Local audio file not found at '{audio_path}'")
            sys.exit(1)
            
        video_title = os.path.splitext(os.path.basename(audio_path))[0]
        download_dir = args.output_dir if args.output_dir else os.path.dirname(audio_path)
        
        transcribe_audio(
            audio_path=audio_path,
            video_title=video_title,
            youtube_url=None,
            download_dir=download_dir,
            model_name_or_path=args.model,
            language=transcribe_lang
        )
