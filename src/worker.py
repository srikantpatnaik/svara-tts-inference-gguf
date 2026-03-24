#!/usr/bin/env python3
"""Background worker for processing TTS requests."""

import asyncio
import base64
import io
import json
import time
import uuid
import wave
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

import numpy as np
import requests
import yaml


def get_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


_config = get_config()


def update_llama_url(host: str, port: int):
    """Update the llama-server URL in config."""
    global _config
    _config["llama_server"]["host"] = host
    _config["llama_server"]["port"] = port
    _config["llama_server"]["url"] = f"http://{host}:{port}/v1/completions"


class TaskStatus(Enum):
    """Task status enum."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TTSRequest:
    """TTS request data."""

    request_id: str
    text: str
    voice: str
    temperature: float
    top_p: float
    max_tokens: int
    repetition_penalty: float
    return_format: str  # "json" or "binary"
    timestamp: float


@dataclass
class TTSResult:
    """TTS result data."""

    request_id: str
    status: TaskStatus
    audio_base64: Optional[str] = None
    audio_duration: Optional[float] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None


class TTSWorkerPool:
    """Worker pool for processing TTS requests."""

    _snac_model = None
    _snac_device = None
    _snac_lock = None

    def __init__(self, cfg: dict = None):
        if cfg is None:
            cfg = _config
        self._config = cfg
        server_cfg = cfg.get("server", {})
        llama_cfg = cfg.get("llama_server", {})
        model_cfg = cfg.get("model", {})
        audio_cfg = cfg.get("audio", {})

        self.workers_count = server_cfg.get("workers", 2)
        self.max_queue_size = server_cfg.get("max_queue_size", 100)
        self.llama_url = llama_cfg.get(
            "url",
            f"http://{llama_cfg.get('host', '127.0.0.1')}:{llama_cfg.get('port', 8080)}/v1/completions",
        )
        self.request_timeout = server_cfg.get("request_timeout", 300)
        self.sample_rate = audio_cfg.get("sample_rate", 24000)
        self.max_tokens = model_cfg.get("max_tokens", 1200)
        self.temperature = model_cfg.get("temperature", 0.6)
        self.top_p = model_cfg.get("top_p", 0.9)
        self.repetition_penalty = model_cfg.get("repetition_penalty", 1.1)

        self.queue: asyncio.Queue = asyncio.Queue(maxsize=self.max_queue_size)
        self.workers: List[asyncio.Task] = []
        self.results: Dict[str, TTSResult] = {}
        self._shutdown = False

        # Voice configuration
        self.available_voices = [
            "hi_male",
            "hi_female",
            "bn_male",
            "bn_female",
            "mr_male",
            "mr_female",
            "te_male",
            "te_female",
            "kn_male",
            "kn_female",
            "bh_male",
            "bh_female",
            "mag_male",
            "mag_female",
            "hne_male",
            "hne_female",
            "mai_male",
            "mai_female",
            "as_male",
            "as_female",
            "brx_male",
            "brx_female",
            "doi_male",
            "doi_female",
            "gu_male",
            "gu_female",
            "ml_male",
            "ml_female",
            "pa_male",
            "pa_female",
            "ta_male",
            "ta_female",
            "en_male",
            "en_female",
            "ne_male",
            "ne_female",
            "sa_male",
            "sa_female",
        ]
        self.default_voice = "hi_male"

        # Custom token prefix for svara-tts
        self.token_prefix = "<custom_token_"

    async def start(self):
        """Start the worker pool."""
        for i in range(self.workers_count):
            task = asyncio.create_task(self._worker(i))
            self.workers.append(task)

    async def stop(self):
        """Stop the worker pool."""
        self._shutdown = True
        # Cancel pending tasks
        for task in self.workers:
            task.cancel()
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()

    async def submit(self, request: TTSRequest) -> TTSResult:
        """Submit a request to the queue."""
        # Validate voice
        if request.voice not in self.available_voices:
            request.voice = self.default_voice

        # Create pending result
        result = TTSResult(
            request_id=request.request_id,
            status=TaskStatus.QUEUED,
        )
        self.results[request.request_id] = result

        # Submit to queue
        try:
            self.queue.put_nowait(request)
        except asyncio.QueueFull:
            result.status = TaskStatus.FAILED
            result.error = "Queue is full, please try again later"

        return result

    def get_result(self, request_id: str) -> Optional[TTSResult]:
        """Get result for a request."""
        return self.results.get(request_id)

    async def _worker(self, worker_id: int):
        """Worker coroutine."""
        print(f"Worker {worker_id} started")

        while not self._shutdown:
            try:
                # Get request from queue with timeout
                try:
                    request = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                # Process request
                result = await self._process_request(request)
                self.results[request.request_id] = result

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Worker {worker_id} error: {e}")

        print(f"Worker {worker_id} stopped")

    async def _process_request(self, request: TTSRequest) -> TTSResult:
        """Process a single TTS request."""
        start_time = time.time()

        result = TTSResult(
            request_id=request.request_id,
            status=TaskStatus.PROCESSING,
        )

        try:
            # Generate tokens from llama.cpp
            tokens = await self._generate_tokens(request)

            if not tokens:
                raise Exception("No tokens generated")

            # Decode tokens to audio
            audio_segments = await self._decode_audio(tokens)

            if not audio_segments:
                raise Exception("Audio decoding failed")

            # Combine audio segments
            audio_data = self._combine_audio(audio_segments)

            # Encode to base64 if needed
            if request.return_format == "json":
                result.audio_base64 = base64.b64encode(audio_data).decode("utf-8")
            else:
                # For binary, we'll store raw bytes and handle in endpoint
                result.audio_base64 = base64.b64encode(audio_data).decode("utf-8")

            # Calculate duration
            result.audio_duration = len(audio_data) / (2 * self.sample_rate)
            result.status = TaskStatus.COMPLETED

        except Exception as e:
            result.status = TaskStatus.FAILED
            result.error = str(e)

        result.processing_time = time.time() - start_time
        return result

    async def _generate_tokens(self, request: TTSRequest) -> List[str]:
        """Generate tokens from llama.cpp server."""
        # Format prompt
        prompt = self._format_prompt(request.text, request.voice)

        payload = {
            "model": "svara-tts-v1",
            "prompt": prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "repeat_penalty": request.repetition_penalty,
            "stream": True,
        }

        tokens = []

        try:
            response = requests.post(
                self.llama_url,
                json=payload,
                stream=True,
                timeout=self.request_timeout,
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            if "choices" in data and data["choices"]:
                                token_text = data["choices"][0].get("text", "")
                                if token_text:
                                    tokens.append(token_text)
                        except json.JSONDecodeError:
                            continue

        except requests.RequestException as e:
            raise Exception(f"Failed to connect to llama.cpp: {e}")

        return tokens

    def _format_prompt(self, text: str, voice: str) -> str:
        """Format prompt with voice and special tokens."""
        return f"<|audio|>{voice}: {text}<|eot_id|>"

    def _token_to_id(self, token_text: str, index: int) -> Optional[int]:
        """Convert token text to numeric ID."""
        token_text = token_text.strip()
        start = token_text.rfind(self.token_prefix)

        if start == -1:
            return None

        token = token_text[start:]

        if token.startswith(self.token_prefix) and token.endswith(">"):
            try:
                num = int(token[14:-1])
                return num - 10 - ((index % 7) * 4096)
            except ValueError:
                return None
        return None

    async def _decode_audio(self, tokens: List[str]) -> List[bytes]:
        """Decode tokens to audio using SNAC."""
        import asyncio

        try:
            from snac import SNAC
            import torch
        except ImportError as e:
            raise Exception(f"Missing dependency: {e}")

        # Initialize lock if not already
        if TTSWorkerPool._snac_lock is None:
            TTSWorkerPool._snac_lock = asyncio.Lock()

        async with TTSWorkerPool._snac_lock:
            # Preload model if not already loaded
            if (
                not hasattr(TTSWorkerPool, "_snac_model")
                or TTSWorkerPool._snac_model is None
            ):
                print("Loading SNAC model...")
                TTSWorkerPool._snac_model = SNAC.from_pretrained(
                    "hubertsiuzdak/snac_24khz"
                ).eval()
                device = (
                    "cuda"
                    if torch.cuda.is_available()
                    else "mps"
                    if torch.backends.mps.is_available()
                    else "cpu"
                )
                print(f"Using device: {device}")
                TTSWorkerPool._snac_model = TTSWorkerPool._snac_model.to(device)
                TTSWorkerPool._snac_device = device

            model = TTSWorkerPool._snac_model
            device = TTSWorkerPool._snac_device

            # Buffer tokens and decode
            buffer = []
            count = 0
            audio_segments = []

            for token_text in tokens:
                token_id = self._token_to_id(token_text, count)

                if token_id is not None and token_id > 0:
                    buffer.append(token_id)
                    count += 1

                    # Process every 7 tokens
                    if count % 7 == 0 and count > 27:
                        buffer_to_proc = buffer[-28:]
                        audio = self._decode_tokens(model, device, buffer_to_proc)
                        if audio:
                            audio_segments.append(audio)

        return audio_segments

    def _decode_tokens(self, model, device: str, tokens: List[int]) -> Optional[bytes]:
        """Decode a batch of tokens to audio."""
        import torch

        if len(tokens) < 7:
            return None

        codes_0 = torch.tensor([], device=device, dtype=torch.int32)
        codes_1 = torch.tensor([], device=device, dtype=torch.int32)
        codes_2 = torch.tensor([], device=device, dtype=torch.int32)

        num_frames = len(tokens) // 7
        frame = tokens[: num_frames * 7]

        for j in range(num_frames):
            i = 7 * j

            # codes_0
            codes_0 = torch.cat(
                [codes_0, torch.tensor([frame[i]], device=device, dtype=torch.int32)]
            )

            # codes_1
            codes_1 = torch.cat(
                [
                    codes_1,
                    torch.tensor([frame[i + 1]], device=device, dtype=torch.int32),
                    torch.tensor([frame[i + 4]], device=device, dtype=torch.int32),
                ]
            )

            # codes_2
            codes_2 = torch.cat(
                [
                    codes_2,
                    torch.tensor([frame[i + 2]], device=device, dtype=torch.int32),
                    torch.tensor([frame[i + 3]], device=device, dtype=torch.int32),
                    torch.tensor([frame[i + 5]], device=device, dtype=torch.int32),
                    torch.tensor([frame[i + 6]], device=device, dtype=torch.int32),
                ]
            )

        codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]

        # Validate token range
        if (
            torch.any(codes[0] < 0)
            or torch.any(codes[0] > 4096)
            or torch.any(codes[1] < 0)
            or torch.any(codes[1] > 4096)
            or torch.any(codes[2] < 0)
            or torch.any(codes[2] > 4096)
        ):
            return None

        with torch.inference_mode():
            audio_hat = model.decode(codes)

        audio_slice = audio_hat[:, :, 2048:4096]
        audio_np = audio_slice.detach().cpu().numpy()
        audio_int16 = (audio_np * 32767).astype(np.int16)

        return audio_int16.tobytes()

    async def generate_streaming(
        self,
        text: str,
        voice: str,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_tokens: int = 1200,
        repetition_penalty: float = 1.1,
    ):
        """Generate audio with streaming - yields chunks as they're generated."""
        start_time = time.time()

        # Validate voice
        if voice not in self.available_voices:
            voice = self.default_voice

        # Import SNAC
        try:
            from snac import SNAC
            import torch
        except ImportError as e:
            yield {"error": f"Missing dependency: {e}"}
            return

        # Lazy load model
        if not hasattr(self, "_snac_model"):
            print("Loading SNAC model...")
            self._snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
            print(f"Using device: {device}")
            self._snac_model = self._snac_model.to(device)
            self._device = device
            self._torch = torch

        model = self._snac_model
        device = self._device

        # Format prompt
        prompt = self._format_prompt(text, voice)

        payload = {
            "model": "svara-tts-v1",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repeat_penalty": repetition_penalty,
            "stream": True,
        }

        # Buffer for tokens
        buffer = []
        count = 0
        token_count = 0
        audio_segments = []
        audio_start_time = time.time()

        try:
            response = requests.post(
                self.llama_url,
                json=payload,
                stream=True,
                timeout=self.request_timeout,
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    line_str = line.decode("utf-8")
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            if "choices" in data and data["choices"]:
                                token_text = data["choices"][0].get("text", "")
                                if token_text:
                                    token_count += 1
                                    token_id = self._token_to_id(token_text, count)

                                    if token_id is not None and token_id > 0:
                                        buffer.append(token_id)
                                        count += 1

                                        # Process every 7 tokens
                                        if count % 7 == 0 and count > 27:
                                            buffer_to_proc = buffer[-28:]
                                            audio = self._decode_tokens(
                                                model, device, buffer_to_proc
                                            )
                                            if audio:
                                                audio_segments.append(audio)

                                                # Yield progress update
                                                elapsed = time.time() - start_time
                                                yield {
                                                    "type": "progress",
                                                    "tokens": token_count,
                                                    "audio_chunks": len(audio_segments),
                                                    "elapsed": elapsed,
                                                    "tokens_per_second": token_count
                                                    / elapsed
                                                    if elapsed > 0
                                                    else 0,
                                                }

                        except json.JSONDecodeError:
                            continue

        except requests.RequestException as e:
            yield {"error": f"Failed to connect to llama.cpp: {e}"}
            return

        # Combine and yield final audio
        if audio_segments:
            combined_audio = self._combine_audio(audio_segments)
            audio_base64 = base64.b64encode(combined_audio).decode("utf-8")

            total_time = time.time() - start_time
            audio_duration = len(combined_audio) / (2 * self.sample_rate)

            # Get GPU memory if available
            gpu_info = {}
            if hasattr(self, "_torch") and self._torch.cuda.is_available():
                gpu_info["gpu_memory_allocated"] = (
                    self._torch.cuda.memory_allocated() / 1024**3
                )
                gpu_info["gpu_memory_reserved"] = (
                    self._torch.cuda.memory_reserved() / 1024**3
                )

            yield {
                "type": "complete",
                "audio": audio_base64,
                "duration": audio_duration,
                "total_time": total_time,
                "tokens": token_count,
                "tokens_per_second": token_count / total_time if total_time > 0 else 0,
                "audio_chunks": len(audio_segments),
                "voice": voice,
                "text": text,
                **gpu_info,
            }
        else:
            yield {"error": "No audio generated"}

    def _combine_audio(self, segments: List[bytes]) -> bytes:
        """Combine audio segments into a single WAV file."""
        if not segments:
            return b""

        # Combine raw audio
        combined = b"".join(segments)

        # Create WAV in memory
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(combined)

        return buffer.getvalue()


# Global worker pool instance
worker_pool: Optional[TTSWorkerPool] = None


async def get_worker_pool(cfg: dict = None) -> TTSWorkerPool:
    """Get the global worker pool instance."""
    global worker_pool
    if cfg is None:
        cfg = _config
    if worker_pool is None:
        worker_pool = TTSWorkerPool(cfg)
        await worker_pool.start()
    else:
        # Update URL from latest config
        worker_pool.llama_url = cfg.get("llama_server", {}).get(
            "url", f"http://127.0.0.1:8080/v1/completions"
        )
    return worker_pool
