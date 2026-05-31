# Serbian Rolling Summarizer 🇷🇸📝

An advanced transcription and summarization utility suite tailored for Serbian and Balkan language audio/video sources. It leverages a local high-performance Whisper model for offline transcription combined with Google Gemini or OpenAI APIs to produce continuous, real-time English summaries.

## 🚀 Features
* **Offline GPU-Accelerated Transcription**: Uses OpenAI's Whisper model (highly optimized for `large-v3` or `base` models) locally to transcribe Balkan languages.
* **Rolling Summarization Window**: Automatically processes incoming transcribed text in continuous blocks (e.g., every 10 minutes) and yields detailed English summaries without interrupting the local transcriber.
* **Intelligent Synthesis**: Translates, maps idioms/cultural context, and summarizes critical details in a structured output format.

## 🛠️ Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/MrYeast3kjr/serbian-rolling-summarizer.git
   cd serbian-rolling-summarizer
   ```

2. **Install Python Dependencies**:
   ```bash
   pip install openai-whisper google-generativeai openai tqdm yt-dlp
   ```

3. **FFmpeg**: Required for audio processing. Install on Windows via Winget:
   ```powershell
   winget install Gyan.FFmpeg
   ```

## 📖 Usage
To start the rolling transcriber and summarizer:
```bash
python transcribe_rolling_summary.py --url "<YOUTUBE_URL_OR_LOCAL_FILE>" --language sr --model large-v3
```

Parameters:
* `--url`: YouTube link or path to a local audio/video file.
* `--language`: Whisper language code (e.g. `sr` for Serbian, `hr` for Croatian).
* `--model`: Whisper model size (`large-v3` recommended for highest accuracy, `base` for speed).
