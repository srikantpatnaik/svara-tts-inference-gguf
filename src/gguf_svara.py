#!/usr/bin/env python3
"""
Svara TTS - Indic Text-to-Speech using llama.cpp server.

This script generates speech from text using the svara-tts-v1 GGUF model
via llama.cpp's OpenAI-compatible API.
"""

import os
import sys
import json
import time
import wave
import argparse
import threading
import queue
import asyncio
from typing import Optional, Generator

try:
    import requests
except ImportError:
    print("Error: 'requests' package is required. Install with: pip install requests")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("Error: 'numpy' package is required. Install with: pip install numpy")
    sys.exit(1)

try:
    import sounddevice as sd
except ImportError:
    print("Error: 'sounddevice' package is required for audio playback. Install with: pip install sounddevice")
    sd = None

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_API_URL = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}/v1/completions"

API_HEADERS = {
    "Content-Type": "application/json"
}

MAX_TOKENS = 1200
TEMPERATURE = 0.6
TOP_P = 0.9
REPETITION_PENALTY = 1.1
SAMPLE_RATE = 24000

AVAILABLE_VOICES = [
    "hi_male", "hi_female",
    "bn_male", "bn_female",
    "mr_male", "mr_female",
    "te_male", "te_female",
    "kn_male", "kn_female",
    "bh_male", "bh_female",
    "mag_male", "mag_female",
    "hne_male", "hne_female",
    "mai_male", "mai_female",
    "as_male", "as_female",
    "brx_male", "brx_female",
    "doi_male", "doi_female",
    "gu_male", "gu_female",
    "ml_male", "ml_female",
    "pa_male", "pa_female",
    "ta_male", "ta_female",
    "en_male", "en_female",
    "ne_male", "ne_female",
    "sa_male", "sa_female",
]
DEFAULT_VOICE = "hi_male"

CUSTOM_TOKEN_PREFIX = "<custom_token_"


def check_dependencies():
    """Check if all required dependencies are installed."""
    missing = []
    
    try:
        import requests
    except ImportError:
        missing.append("requests")
    
    try:
        import numpy as np
    except ImportError:
        missing.append("numpy")
    
    try:
        import sounddevice as sd
    except ImportError:
        missing.append("sounddevice")
    
    try:
        import torch
    except ImportError:
        missing.append("torch")
    
    try:
        from snac import SNAC
    except ImportError:
        missing.append("snac")
    
    if missing:
        print("Error: Missing required dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall all dependencies with: pip install -r requirements.txt")
        return False
    
    return True


def check_llama_server(url: str, timeout: int = 5) -> bool:
    """Check if llama.cpp server is running and accessible."""
    try:
        response = requests.get(url.replace("/v1/completions", "/health"), 
                              timeout=timeout)
        return response.status_code == 200
    except requests.RequestException:
        pass
    
    try:
        response = requests.get(url.replace("/v1/completions", "/v1/models"),
                              timeout=timeout)
        return response.status_code == 200
    except requests.RequestException:
        pass
    
    return False


def format_prompt(prompt: str, voice: str = DEFAULT_VOICE) -> str:
    """Format prompt for svara-tts-v1 model with voice prefix and special tokens."""
    if voice not in AVAILABLE_VOICES:
        print(f"Warning: Voice '{voice}' not recognized. Using '{DEFAULT_VOICE}' instead.")
        voice = DEFAULT_VOICE
    
    formatted_prompt = f"{voice}: {prompt}"
    special_start = "<|audio|>"
    special_end = "<|eot_id|>"
    
    return f"{special_start}{formatted_prompt}{special_end}"


def generate_tokens_from_api(
    prompt: str,
    api_url: str,
    voice: str = DEFAULT_VOICE,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    max_tokens: int = MAX_TOKENS,
    repetition_penalty: float = REPETITION_PENALTY
) -> Generator[str, None, None]:
    """Generate tokens from text using llama.cpp server API."""
    formatted_prompt = format_prompt(prompt, voice)
    print(f"Generating speech for: {formatted_prompt}")
    
    payload = {
        "model": "svara-tts-v1",
        "prompt": formatted_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repeat_penalty": repetition_penalty,
        "stream": True
    }
    
    try:
        response = requests.post(api_url, headers=API_HEADERS, json=payload, stream=True)
    except requests.ConnectionError:
        print(f"Error: Could not connect to llama.cpp server at {api_url}")
        print("Make sure the server is running. Example:")
        print("  ./server -m models/svara-tts-v1.gguf --host 127.0.0.1 --port 8080")
        return
    except requests.RequestException as e:
        print(f"Error: API request failed: {e}")
        return
    
    if response.status_code != 200:
        print(f"Error: API request failed with status code {response.status_code}")
        print(f"Error details: {response.text}")
        return
    
    token_counter = 0
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                data_str = line[6:]
                if data_str.strip() == '[DONE]':
                    break
                    
                try:
                    data = json.loads(data_str)
                    if 'choices' in data and len(data['choices']) > 0:
                        token_text = data['choices'][0].get('text', '')
                        token_counter += 1
                        if token_text:
                            yield token_text
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                    continue
    
    print("Token generation complete")


def turn_token_into_id(token_string: str, index: int) -> Optional[int]:
    """Convert token string to numeric ID for audio processing."""
    token_string = token_string.strip()
    last_token_start = token_string.rfind(CUSTOM_TOKEN_PREFIX)
    
    if last_token_start == -1:
        return None
    
    last_token = token_string[last_token_start:]
    
    if last_token.startswith(CUSTOM_TOKEN_PREFIX) and last_token.endswith(">"):
        try:
            number_str = last_token[14:-1]
            token_id = int(number_str) - 10 - ((index % 7) * 4096)
            return token_id
        except ValueError:
            return None
    return None


def convert_to_audio(multiframe, count):
    """Convert token frames to audio."""
    try:
        from decoder import convert_to_audio as svara_convert_to_audio
        return svara_convert_to_audio(multiframe, count)
    except ImportError as e:
        print(f"Error: Could not import decoder: {e}")
        print("Make sure the decoder module is available.")
        return None


async def tokens_decoder(token_gen):
    """Asynchronous token decoder that converts token stream to audio stream."""
    buffer = []
    count = 0
    async for token_text in token_gen:
        token = turn_token_into_id(token_text, count)
        if token is not None and token > 0:
            buffer.append(token)
            count += 1
            
            if count % 7 == 0 and count > 27:
                buffer_to_proc = buffer[-28:]
                audio_samples = convert_to_audio(buffer_to_proc, count)
                if audio_samples is not None:
                    yield audio_samples


def tokens_decoder_sync(syn_token_gen, output_file: Optional[str] = None):
    """Synchronous wrapper for the asynchronous token decoder."""
    audio_queue = queue.Queue()
    audio_segments = []
    
    wav_file = None
    if output_file:
        output_dir = os.path.dirname(os.path.abspath(output_file))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        wav_file = wave.open(output_file, "wb")
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
    
    async def async_token_gen():
        for token in syn_token_gen:
            yield token

    async def async_producer():
        async for audio_chunk in tokens_decoder(async_token_gen()):
            audio_queue.put(audio_chunk)
        audio_queue.put(None)

    def run_async():
        asyncio.run(async_producer())

    thread = threading.Thread(target=run_async)
    thread.start()

    while True:
        audio = audio_queue.get()
        if audio is None:
            break
        
        audio_segments.append(audio)
        
        if wav_file:
            wav_file.writeframes(audio)
    
    if wav_file:
        wav_file.close()
    
    thread.join()
    
    duration = sum([len(segment) // 2 for segment in audio_segments]) / SAMPLE_RATE
    print(f"Generated {len(audio_segments)} audio segments")
    print(f"Generated {duration:.2f} seconds of audio")
    
    return audio_segments


def stream_audio(audio_buffer):
    """Stream audio buffer to output device."""
    if audio_buffer is None or len(audio_buffer) == 0:
        return
    
    if sd is None:
        print("Warning: sounddevice not available, skipping playback")
        return
    
    audio_data = np.frombuffer(audio_buffer, dtype=np.int16)
    audio_float = audio_data.astype(np.float32) / 32767.0
    
    sd.play(audio_float, SAMPLE_RATE)
    sd.wait()


def generate_speech_from_api(
    prompt: str,
    api_url: str,
    voice: str = DEFAULT_VOICE,
    output_file: Optional[str] = None,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    max_tokens: int = MAX_TOKENS,
    repetition_penalty: float = REPETITION_PENALTY
):
    """Generate speech from text using svara-tts-v1 model via llama.cpp API."""
    return tokens_decoder_sync(
        generate_tokens_from_api(
            prompt=prompt, 
            api_url=api_url,
            voice=voice,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty
        ),
        output_file=output_file
    )


def list_available_voices():
    """List all available Indic language voices with the default one marked."""
    voice_names = {
        "hi_male": "Hindi (Male)", "hi_female": "Hindi (Female)",
        "bn_male": "Bengali (Male)", "bn_female": "Bengali (Female)",
        "mr_male": "Marathi (Male)", "mr_female": "Marathi (Female)",
        "te_male": "Telugu (Male)", "te_female": "Telugu (Female)",
        "kn_male": "Kannada (Male)", "kn_female": "Kannada (Female)",
        "bh_male": "Bhojpuri (Male)", "bh_female": "Bhojpuri (Female)",
        "mag_male": "Magahi (Male)", "mag_female": "Magahi (Female)",
        "hne_male": "Chhattisgarhi (Male)", "hne_female": "Chhattisgarhi (Female)",
        "mai_male": "Maithili (Male)", "mai_female": "Maithili (Female)",
        "as_male": "Assamese (Male)", "as_female": "Assamese (Female)",
        "brx_male": "Bodo (Male)", "brx_female": "Bodo (Female)",
        "doi_male": "Dogri (Male)", "doi_female": "Dogri (Female)",
        "gu_male": "Gujarati (Male)", "gu_female": "Gujarati (Female)",
        "ml_male": "Malayalam (Male)", "ml_female": "Malayalam (Female)",
        "pa_male": "Punjabi (Male)", "pa_female": "Punjabi (Female)",
        "ta_male": "Tamil (Male)", "ta_female": "Tamil (Female)",
        "en_male": "English Indian (Male)", "en_female": "English Indian (Female)",
        "ne_male": "Nepali (Male)", "ne_female": "Nepali (Female)",
        "sa_male": "Sanskrit (Male)", "sa_female": "Sanskrit (Female)",
    }
    
    print("Available Indic voices (36 languages):")
    for voice in AVAILABLE_VOICES:
        marker = "★" if voice == DEFAULT_VOICE else " "
        name = voice_names.get(voice, voice)
        print(f"{marker} {voice} - {name}")
    print(f"\nDefault voice: {DEFAULT_VOICE} (Hindi Male)")


def main():
    parser = argparse.ArgumentParser(
        description="Svara TTS - Indic Text-to-Speech using llama.cpp server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --text "नमस्ते" --voice hi_male
  %(prog)s --text "Hello" --voice en_male --output speech.wav
  %(prog)s --host 192.168.1.100 --port 8080 --text "Test"
  %(prog)s --url http://localhost:8080/v1/completions --list-voices

Start llama.cpp server:
  ./server -m models/svara-tts-v1.gguf --host 127.0.0.1 --port 8080
        """
    )
    
    parser.add_argument("--text", type=str, help="Text to convert to speech")
    parser.add_argument("--voice", type=str, default=DEFAULT_VOICE,
                       help=f"Voice to use (default: {DEFAULT_VOICE})")
    parser.add_argument("--output", "-o", type=str, help="Output WAV file path")
    parser.add_argument("--list-voices", action="store_true", help="List available voices")
    
    parser.add_argument("--host", type=str, default=DEFAULT_HOST,
                       help=f"llama.cpp server host (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                       help=f"llama.cpp server port (default: {DEFAULT_PORT})")
    parser.add_argument("--url", type=str,
                       help="Full API URL (overrides host/port)")
    parser.add_argument("--check-server", action="store_true",
                       help="Check if llama.cpp server is accessible")
    
    parser.add_argument("--temperature", type=float, default=TEMPERATURE,
                       help="Temperature for generation (default: 0.6)")
    parser.add_argument("--top_p", type=float, default=TOP_P,
                       help="Top-p sampling parameter (default: 0.9)")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                       help=f"Maximum tokens to generate (default: {MAX_TOKENS})")
    parser.add_argument("--repetition-penalty", type=float, default=REPETITION_PENALTY,
                       help="Repetition penalty (default: 1.1)")
    
    parser.add_argument("--play", action="store_true",
                       help="Play audio after generation (requires sounddevice)")
    
    args = parser.parse_args()
    
    if not check_dependencies():
        sys.exit(1)
    
    if args.list_voices:
        list_available_voices()
        return
    
    api_url = args.url if args.url else f"http://{args.host}:{args.port}/v1/completions"
    
    if args.check_server:
        print(f"Checking llama.cpp server at {api_url}...")
        if check_llama_server(api_url):
            print("Server is accessible!")
        else:
            print("Server is not accessible. Make sure llama.cpp server is running.")
            print(f"Start server with: ./server -m <model.gguf> --host {args.host} --port {args.port}")
        return
    
    prompt = args.text
    if not prompt:
        prompt = input("Enter text to synthesize: ")
        if not prompt:
            prompt = "Hello, this is a test of the Svara TTS system for Indic languages."
    
    output_file = args.output
    if not output_file:
        os.makedirs("outputs", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f"outputs/{args.voice}_{timestamp}.wav"
        print(f"No output file specified. Saving to {output_file}")
    
    print(f"Connecting to llama.cpp server at {api_url}")
    
    start_time = time.time()
    audio_segments = generate_speech_from_api(
        prompt=prompt,
        api_url=api_url,
        voice=args.voice,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
        output_file=output_file
    )
    end_time = time.time()
    
    print(f"Speech generation completed in {end_time - start_time:.2f} seconds")
    print(f"Audio saved to {output_file}")
    
    if args.play and audio_segments:
        print("Playing audio...")
        for segment in audio_segments:
            stream_audio(segment)


if __name__ == "__main__":
    main()
