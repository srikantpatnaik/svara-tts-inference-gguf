# Svara TTS - Indic Text-to-Speech

GGUF-based TTS API with integrated llama-server. 36 Indic language voices.

## Setup

```bash
cd src && pip install -r requirements.txt && cd ..
# Edit config.yaml with llama-server path and model path
python api_server.py
```

## Features

- 36 voices (18 languages, male/female)
- Emotion tags: `<giggle>`, `<laugh>`, `<sigh>`, etc.
- Web UI at `http://localhost:8000`
- Sync/Stream endpoints

## API

```bash
# Sync (returns JSON with base64 audio)
curl -s -X POST "http://localhost:8000/tts/sync" -F "text=Hello" -F "voice=hi_male" -o response.json

# Stream (returns WAV directly)
curl -s -X POST "http://localhost:8000/tts/stream" -F "text=Hello" -F "voice=hi_male" -o speech.wav

# List voices
curl -s http://localhost:8000/voices

# Health
curl -s http://localhost:8000/health
```

## Available Voices

Hindi, Bengali, Marathi, Telugu, Kannada, Bhojpuri, Magahi, Chhattisgarhi, Maithili, Assamese, Bodo, Dogri, Gujarati, Malayalam, Punjabi, Tamil, English (Indian), Nepali, Sanskrit - each with male/female variants.

Default: `hi_male`
