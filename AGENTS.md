# Agent Guidelines for svara-tts-inference-gguf

## Project Overview

Python-based TTS inference using GGUF models with integrated llama-server. Supports 36 Indic language voices.

## Project Structure

```
.
├── api_server.py     # FastAPI server (entry point)
├── config.yaml       # All configuration
├── src/
│   ├── worker.py     # TTS worker pool
│   ├── decoder.py    # Audio decoder
│   ├── gguf_svara.py # CLI client
│   └── requirements.txt
```

## Setup

```bash
cd src && pip install -r requirements.txt
```

## Running

```bash
# API Server (config via config.yaml)
python api_server.py

# CLI
python src/gguf_svara.py --text "Hello" --voice hi_male --output output.wav
python src/gguf_svara.py --list-voices
```

## Code Style

- Black (line length 100), isort, flake8
- Type hints required
- Naming: snake_case (vars), PascalCase (classes), UPPER_SNAKE (constants)
- Use FormData in JavaScript (NOT URLSearchParams) for multipart/form-data

## Architecture

- **api_server.py**: FastAPI + integrated llama-server management
- **src/worker.py**: Async workers with SNAC model (use asyncio.Lock for decode)
- **config.yaml**: All settings (llama-server, model, audio)

## Key Implementation Notes

### Concurrent Request Handling
- SNAC model is shared (`TTSWorkerPool._snac_model`)
- Always use `asyncio.Lock` around SNAC decode to prevent race conditions:
```python
async with TTSWorkerPool._snac_lock:
    # decode with SNAC
```

### Web UI JavaScript
- Use `FormData` for POST requests - NOT `URLSearchParams`:
```javascript
const formData = new FormData();
formData.append('text', text);
formData.append('voice', voice);
fetch('/tts/sync', { method: 'POST', body: formData });
```
- Never set `Content-Type` header manually - browser sets it for FormData

### API Endpoints
- `POST /tts/sync` - Returns JSON with base64 audio (use Form)
- `POST /tts/stream` - Returns WAV directly (use Form)
- `GET /voices` - List voices
- `GET /health` - Health check

## config.yaml

```yaml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 2

llama_server:
  bin: "/path/to/llama-server"
  model_path: "/path/to/model.gguf"
  args:
    - "-c"
    - "4096"
    - "-ngl"
    - "99"

model:
  max_tokens: 1200
  temperature: 0.6

audio:
  sample_rate: 24000
```
