#!/usr/bin/env python3
"""FastAPI server for Svara TTS API with integrated llama-server management."""

import asyncio
import base64
import os
import platform
import subprocess
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Depends, Header, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import yaml

from src.worker import (
    TTSWorkerPool,
    TTSRequest,
    TaskStatus,
    get_worker_pool,
    update_llama_url,
    worker_pool,
)


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def find_available_port(
    host: str = "127.0.0.1", start_port: int = 8080, max_attempts: int = 100
) -> int:
    """Find an available port on the given host."""
    import socket

    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                sock.bind((host, port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find available port on {host}")


config_data = load_config()

llama_manager: Optional["LlamaServerManager"] = None


class LlamaServerManager:
    """Manages llama-server lifecycle."""

    def __init__(self, cfg: dict):
        self.config = cfg
        self.process: Optional[subprocess.Popen] = None
        self.log_file = None

    def find_binary(self) -> str:
        """Find llama-server binary."""
        bin_path = self.config.get("bin")
        if bin_path:
            path = Path(bin_path)
            if path.exists() and path.is_file():
                if platform.system() != "Windows":
                    os.chmod(str(path), 0o755)
                return str(path.absolute())
        raise RuntimeError(f"llama-server binary not found: {bin_path}")

    def start(self) -> bool:
        """Start llama-server."""
        binary = self.find_binary()

        model_path = Path(self.config["model_path"])
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        port = self.config.get("port", 0)
        if port == 0:
            port = find_available_port(self.config["host"])

        host = self.config.get("host", "127.0.0.1")

        cmd = [
            binary,
            "-m",
            str(model_path.absolute()),
            "--host",
            host,
            "--port",
            str(port),
        ]
        args_str = self.config.get("args", "")
        if args_str:
            cmd.extend(args_str.split())

        env = os.environ.copy()
        if "CUDA_VISIBLE_DEVICES" not in env:
            env["CUDA_VISIBLE_DEVICES"] = "0"

        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        self.log_file = open(log_dir / f"llama-server_{port}.log", "w")

        print(f"Starting llama-server: {' '.join(cmd)}")
        self.process = subprocess.Popen(
            cmd,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )

        import time

        time.sleep(2)

        if self.process.poll() is not None:
            raise RuntimeError("llama-server failed to start")

        self.config["port"] = port
        print(f"llama-server running at http://{host}:{port}")
        return True

    def stop(self):
        """Stop llama-server."""
        if self.process:
            self.process.terminate()
            self.process.wait()
        if self.log_file:
            self.log_file.close()

    def is_running(self) -> bool:
        """Check if llama-server is running."""
        return self.process is not None and self.process.poll() is None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global llama_manager

    llama_cfg = config_data.get("llama_server", {})
    if llama_cfg.get("bin"):
        try:
            llama_manager = LlamaServerManager(llama_cfg)
            llama_manager.start()
            # Update URL with actual port
            if llama_manager.config.get("port"):
                actual_port = llama_manager.config["port"]
                actual_host = llama_cfg.get("host", "127.0.0.1")
                update_llama_url(actual_host, actual_port)
                config_data["llama_server"]["port"] = actual_port
                config_data["llama_server"]["url"] = (
                    f"http://{actual_host}:{actual_port}/v1/completions"
                )
                # Update existing worker pool URL
                if worker_pool:
                    worker_pool.llama_url = config_data["llama_server"]["url"]
        except Exception as e:
            print(f"Warning: Could not start llama-server: {e}")

    worker = await get_worker_pool(config_data)
    print(f"TTS Worker pool started with {config_data['server']['workers']} workers")
    yield
    if worker:
        await worker.stop()
        print("TTS Worker pool stopped")
    if llama_manager:
        llama_manager.stop()


app = FastAPI(
    title="Svara TTS API",
    version="1.0.0",
    lifespan=lifespan,
)

server_cfg = config_data["server"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=server_cfg.get("cors_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


WEB_UI_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Svara TTS</title>
    <link rel="icon" type="image/x-icon" href="/favicon.ico">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            --bg-primary: #1a1a1a;
            --bg-secondary: #2d2d2d;
            --bg-tertiary: #3d3d3d;
            --text-primary: #ffffff;
            --text-secondary: #a0a0a0;
            --accent: #3b82f6;
            --accent-hover: #2563eb;
            --border: #404040;
            --success: #10b981;
            --error: #ef4444;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            height: 100vh;
            display: flex;
            overflow: hidden;
        }

        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 6px;
        }
        ::-webkit-scrollbar-track {
            background: transparent;
        }
        ::-webkit-scrollbar-thumb {
            background: var(--bg-tertiary);
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: var(--border);
        }

        /* Sidebar */
        .sidebar {
            width: 280px;
            background: var(--bg-secondary);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            transition: width 0.3s ease;
            flex-shrink: 0;
        }

        .sidebar.collapsed {
            width: 0;
            border-right: none;
        }

        .sidebar-header {
            padding: 16px;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .logo {
            width: 32px;
            height: 32px;
            background: var(--accent);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 18px;
            flex-shrink: 0;
        }

        .sidebar-title {
            font-size: 16px;
            font-weight: 600;
            white-space: nowrap;
        }

        .sidebar-toggle {
            margin-left: auto;
            background: none;
            border: none;
            color: var(--text-secondary);
            cursor: pointer;
            padding: 4px;
            font-size: 18px;
        }

        .new-chat-btn {
            margin: 16px;
            padding: 12px 16px;
            background: var(--accent);
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }

        .new-chat-btn:hover {
            background: var(--accent-hover);
        }

        /* Voice list */
        .voice-list {
            flex: 1;
            overflow-y: auto;
            padding: 8px;
        }

        .voice-section {
            margin-bottom: 16px;
        }

        .voice-section-title {
            font-size: 11px;
            font-weight: 600;
            color: var(--text-secondary);
            padding: 8px 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .voice-item {
            padding: 10px 12px;
            border-radius: 6px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 10px;
            transition: background 0.15s;
        }

        .voice-item:hover {
            background: var(--bg-tertiary);
        }

        .voice-item.active {
            background: var(--accent);
        }

        .voice-icon {
            font-size: 16px;
        }

        .voice-name {
            font-size: 13px;
        }

        /* Main content */
        .main {
            flex: 1;
            display: flex;
            flex-direction: column;
            min-width: 0;
        }

        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        .message {
            display: flex;
            gap: 16px;
            max-width: 800px;
            margin: 0 auto;
            width: 100%;
        }

        .message.user {
            flex-direction: row-reverse;
        }

        .message-avatar {
            width: 32px;
            height: 32px;
            border-radius: 8px;
            background: var(--bg-tertiary);
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            font-size: 14px;
        }

        .message.user .message-avatar {
            background: var(--accent);
        }

        .message-content {
            background: var(--bg-secondary);
            padding: 16px;
            border-radius: 12px;
            font-size: 14px;
            line-height: 1.5;
            white-space: pre-wrap;
            word-break: break-word;
        }

        .message.user .message-content {
            background: var(--bg-tertiary);
        }

        .audio-player {
            margin-top: 12px;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .audio-player audio {
            height: 32px;
            flex: 1;
        }

        .download-btn {
            padding: 8px 12px;
            background: var(--accent);
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }

        .download-btn:hover {
            background: var(--accent-hover);
        }

        /* Input area */
        .input-container {
            padding: 24px;
            background: var(--bg-primary);
            border-top: 1px solid var(--border);
        }

        .input-wrapper {
            max-width: 800px;
            margin: 0 auto;
            display: flex;
            gap: 12px;
            align-items: flex-end;
        }

        .input-box {
            flex: 1;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 14px 16px;
            color: var(--text-primary);
            font-size: 14px;
            font-family: inherit;
            resize: none;
            min-height: 52px;
            max-height: 200px;
            line-height: 1.5;
        }

        .input-box:focus {
            outline: none;
            border-color: var(--accent);
        }

        .input-box::placeholder {
            color: var(--text-secondary);
        }

        .send-btn {
            width: 52px;
            height: 52px;
            background: var(--accent);
            border: none;
            border-radius: 12px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 20px;
            transition: background 0.15s;
        }

        .send-btn:hover {
            background: var(--accent-hover);
        }

        .send-btn:disabled {
            background: var(--bg-tertiary);
            cursor: not-allowed;
        }

        /* Loading */
        .loading {
            display: flex;
            align-items: center;
            gap: 8px;
            color: var(--text-secondary);
            font-size: 14px;
            padding: 8px 0;
        }

        .spinner {
            width: 20px;
            height: 20px;
            border: 2px solid var(--border);
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Settings panel */
        .settings-toggle {
            position: fixed;
            bottom: 24px;
            left: 24px;
            width: 40px;
            height: 40px;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 10px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--text-secondary);
            font-size: 18px;
            z-index: 100;
        }

        .settings-toggle:hover {
            background: var(--bg-tertiary);
        }

        .settings-panel {
            position: fixed;
            bottom: 80px;
            left: 24px;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
            display: none;
            z-index: 100;
            min-width: 280px;
        }

        .settings-panel.visible {
            display: block;
        }

        .setting-item {
            margin-bottom: 16px;
        }

        .setting-item:last-child {
            margin-bottom: 0;
        }

        .setting-label {
            font-size: 12px;
            color: var(--text-secondary);
            margin-bottom: 6px;
            display: block;
        }

        .setting-input {
            width: 100%;
            padding: 8px 12px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text-primary);
            font-size: 13px;
        }

        .setting-input:focus {
            outline: none;
            border-color: var(--accent);
        }
    </style>
</head>
<body>
    <div class="sidebar" id="sidebar">
        <div class="sidebar-header">
            <div class="logo">s</div>
            <span class="sidebar-title">Svara TTS</span>
            <button class="sidebar-toggle" onclick="toggleSidebar()">‹</button>
        </div>
        <button class="new-chat-btn" onclick="clearChat()">+ New Chat</button>
        <div class="voice-list">
            <div class="voice-section">
                <div class="voice-section-title">Indian Languages</div>
                <div class="voice-item active" data-voice="hi_male" onclick="selectVoice(this)">
                    <span class="voice-icon">👨</span>
                    <span class="voice-name">Hindi (Male)</span>
                </div>
                <div class="voice-item" data-voice="hi_female" onclick="selectVoice(this)">
                    <span class="voice-icon">👩</span>
                    <span class="voice-name">Hindi (Female)</span>
                </div>
                <div class="voice-item" data-voice="bn_male" onclick="selectVoice(this)">
                    <span class="voice-icon">👨</span>
                    <span class="voice-name">Bengali (Male)</span>
                </div>
                <div class="voice-item" data-voice="bn_female" onclick="selectVoice(this)">
                    <span class="voice-icon">👩</span>
                    <span class="voice-name">Bengali (Female)</span>
                </div>
                <div class="voice-item" data-voice="ta_male" onclick="selectVoice(this)">
                    <span class="voice-icon">👨</span>
                    <span class="voice-name">Tamil (Male)</span>
                </div>
                <div class="voice-item" data-voice="ta_female" onclick="selectVoice(this)">
                    <span class="voice-icon">👩</span>
                    <span class="voice-name">Tamil (Female)</span>
                </div>
                <div class="voice-item" data-voice="te_male" onclick="selectVoice(this)">
                    <span class="voice-icon">👨</span>
                    <span class="voice-name">Telugu (Male)</span>
                </div>
                <div class="voice-item" data-voice="te_female" onclick="selectVoice(this)">
                    <span class="voice-icon">👩</span>
                    <span class="voice-name">Telugu (Female)</span>
                </div>
                <div class="voice-item" data-voice="ml_male" onclick="selectVoice(this)">
                    <span class="voice-icon">👨</span>
                    <span class="voice-name">Malayalam (Male)</span>
                </div>
                <div class="voice-item" data-voice="ml_female" onclick="selectVoice(this)">
                    <span class="voice-icon">👩</span>
                    <span class="voice-name">Malayalam (Female)</span>
                </div>
                <div class="voice-item" data-voice="kn_male" onclick="selectVoice(this)">
                    <span class="voice-icon">👨</span>
                    <span class="voice-name">Kannada (Male)</span>
                </div>
                <div class="voice-item" data-voice="kn_female" onclick="selectVoice(this)">
                    <span class="voice-icon">👩</span>
                    <span class="voice-name">Kannada (Female)</span>
                </div>
                <div class="voice-item" data-voice="gu_male" onclick="selectVoice(this)">
                    <span class="voice-icon">👨</span>
                    <span class="voice-name">Gujarati (Male)</span>
                </div>
                <div class="voice-item" data-voice="gu_female" onclick="selectVoice(this)">
                    <span class="voice-icon">👩</span>
                    <span class="voice-name">Gujarati (Female)</span>
                </div>
                <div class="voice-item" data-voice="mr_male" onclick="selectVoice(this)">
                    <span class="voice-icon">👨</span>
                    <span class="voice-name">Marathi (Male)</span>
                </div>
                <div class="voice-item" data-voice="mr_female" onclick="selectVoice(this)">
                    <span class="voice-icon">👩</span>
                    <span class="voice-name">Marathi (Female)</span>
                </div>
                <div class="voice-item" data-voice="pa_male" onclick="selectVoice(this)">
                    <span class="voice-icon">👨</span>
                    <span class="voice-name">Punjabi (Male)</span>
                </div>
                <div class="voice-item" data-voice="pa_female" onclick="selectVoice(this)">
                    <span class="voice-icon">👩</span>
                    <span class="voice-name">Punjabi (Female)</span>
                </div>
            </div>
            <div class="voice-section">
                <div class="voice-section-title">Other Languages</div>
                <div class="voice-item" data-voice="en_male" onclick="selectVoice(this)">
                    <span class="voice-icon">👨</span>
                    <span class="voice-name">English (Male)</span>
                </div>
                <div class="voice-item" data-voice="en_female" onclick="selectVoice(this)">
                    <span class="voice-icon">👩</span>
                    <span class="voice-name">English (Female)</span>
                </div>
                <div class="voice-item" data-voice="ne_male" onclick="selectVoice(this)">
                    <span class="voice-icon">👨</span>
                    <span class="voice-name">Nepali (Male)</span>
                </div>
                <div class="voice-item" data-voice="ne_female" onclick="selectVoice(this)">
                    <span class="voice-icon">👩</span>
                    <span class="voice-name">Nepali (Female)</span>
                </div>
            </div>
        </div>
    </div>

    <div class="main">
        <div class="chat-container" id="chatContainer">
            <div class="message">
                <div class="message-avatar">s</div>
                <div class="message-content">Hi! I'm Svara TTS. Enter text and I'll convert it to speech in your selected voice.</div>
            </div>
        </div>
        <div class="input-container">
            <div class="input-wrapper">
                <textarea class="input-box" id="inputBox" placeholder="Enter text to convert to speech..." rows="1"></textarea>
                <button class="send-btn" id="sendBtn" onclick="sendMessage()">➤</button>
            </div>
        </div>
    </div>

    <button class="settings-toggle" onclick="toggleSettings()">⚙</button>
    <div class="settings-panel" id="settingsPanel">
        <div class="setting-item">
            <label class="setting-label">Temperature</label>
            <input type="number" class="setting-input" id="temperature" value="0.6" step="0.1" min="0" max="2">
        </div>
        <div class="setting-item">
            <label class="setting-label">Top P</label>
            <input type="number" class="setting-input" id="topP" value="0.9" step="0.1" min="0" max="1">
        </div>
        <div class="setting-item">
            <label class="setting-label">Max Tokens</label>
            <input type="number" class="setting-input" id="maxTokens" value="1200" step="100" min="1">
        </div>
    </div>

    <script>
        let currentVoice = 'hi_male';
        let isGenerating = false;
        let messages = [];

        // Auto-resize textarea
        const inputBox = document.getElementById('inputBox');
        inputBox.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 200) + 'px';
        });

        // Enter to send
        inputBox.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        function toggleSidebar() {
            const sidebar = document.getElementById('sidebar');
            const toggle = document.querySelector('.sidebar-toggle');
            sidebar.classList.toggle('collapsed');
            toggle.textContent = sidebar.classList.contains('collapsed') ? '›' : '‹';
        }

        function selectVoice(element) {
            document.querySelectorAll('.voice-item').forEach(item => item.classList.remove('active'));
            element.classList.add('active');
            currentVoice = element.dataset.voice;
        }

        function toggleSettings() {
            const panel = document.getElementById('settingsPanel');
            panel.classList.toggle('visible');
        }

        function clearChat() {
            const container = document.getElementById('chatContainer');
            container.innerHTML = `
                <div class="message">
                    <div class="message-avatar">s</div>
                    <div class="message-content">Hi! I'm Svara TTS. Enter text and I'll convert it to speech in your selected voice.</div>
                </div>
            `;
            messages = [];
        }

        async function sendMessage() {
            const text = inputBox.value.trim();
            if (!text || isGenerating) return;

            isGenerating = true;
            document.getElementById('sendBtn').disabled = true;

            // Add user message
            addMessage('user', text);
            inputBox.value = '';
            inputBox.style.height = 'auto';

            // Add loading message
            const loadingId = addLoadingMessage();

            try {
                const temperature = parseFloat(document.getElementById('temperature').value);
                const topP = parseFloat(document.getElementById('topP').value);
                const maxTokens = parseInt(document.getElementById('maxTokens').value);

                const formData = new FormData();
                formData.append('text', text);
                formData.append('voice', currentVoice);
                formData.append('temperature', temperature);
                formData.append('top_p', topP);
                formData.append('max_tokens', maxTokens);
                formData.append('repetition_penalty', 1.1);

                const response = await fetch('/tts/sync', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                // Remove loading
                document.getElementById(loadingId)?.remove();

                if (data.success && data.audio) {
                    addMessage('assistant', text, data.audio);
                } else {
                    addMessage('assistant', 'Error: ' + (data.detail || 'Failed to generate speech'));
                }
            } catch (error) {
                document.getElementById(loadingId)?.remove();
                addMessage('assistant', 'Error: ' + error.message);
            }

            isGenerating = false;
            document.getElementById('sendBtn').disabled = false;
        }

        function addMessage(role, text, audioBase64 = null) {
            const container = document.getElementById('chatContainer');
            const id = 'msg-' + Date.now();
            messages.push({ role, text, audio: audioBase64, id });

            let audioHtml = '';
            if (audioBase64) {
                const audioBlob = base64ToBlob(audioBase64, 'audio/wav');
                const audioUrl = URL.createObjectURL(audioBlob);
                audioHtml = `
                    <div class="audio-player">
                        <audio controls autoplay src="${audioUrl}"></audio>
                        <a href="${audioUrl}" download="speech.wav" class="download-btn">⬇ Download</a>
                    </div>
                `;
            }

            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;
            messageDiv.id = id;
            messageDiv.innerHTML = `
                <div class="message-avatar">${role === 'user' ? 'You' : 's'}</div>
                <div class="message-content">${escapeHtml(text)}${audioHtml}</div>
            `;
            container.appendChild(messageDiv);
            container.scrollTop = container.scrollHeight;
        }

        function addLoadingMessage() {
            const container = document.getElementById('chatContainer');
            const id = 'loading-' + Date.now();
            const div = document.createElement('div');
            div.className = 'message assistant';
            div.id = id;
            div.innerHTML = `
                <div class="message-avatar">s</div>
                <div class="message-content">
                    <div class="loading">
                        <div class="spinner"></div>
                        Generating speech...
                    </div>
                </div>
            `;
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
            return id;
        }

        function base64ToBlob(base64, mimeType) {
            const byteCharacters = atob(base64);
            const byteNumbers = new Array(byteCharacters.length);
            for (let i = 0; i < byteCharacters.length; i++) {
                byteNumbers[i] = byteCharacters.charCodeAt(i);
            }
            const byteArray = new Uint8Array(byteNumbers);
            return new Blob([byteArray], { type: mimeType });
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // Collapse sidebar by default
        document.getElementById('sidebar').classList.add('collapsed');
    </script>
</body>
</html>
"""


def verify_api_key(x_api_key: str = Header(None)):
    """Verify API key if configured."""
    if server_cfg.get("api_key"):
        if x_api_key != server_cfg["api_key"]:
            raise HTTPException(status_code=401, detail="Invalid API key")


def get_llama_url() -> str:
    """Get llama-server URL from config."""
    llama_cfg = config_data.get("llama_server", {})
    host = llama_cfg.get("host", "127.0.0.1")
    port = llama_cfg.get("port", 8080)
    return f"http://{host}:{port}/v1/completions"


async def process_tts_request(
    text: str,
    voice: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    repetition_penalty: float,
):
    """Process TTS request through worker pool."""
    worker = await get_worker_pool(config_data)

    request = TTSRequest(
        request_id=str(uuid.uuid4()),
        text=text,
        voice=voice,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
        return_format="json",
        timestamp=0,
    )

    result = await worker.submit(request)

    # Wait for completion - poll for status
    max_wait = 60  # 60 seconds timeout
    waited = 0
    while (
        result.status in (TaskStatus.QUEUED, TaskStatus.PROCESSING)
        and waited < max_wait
    ):
        await asyncio.sleep(0.5)
        waited += 0.5
        # Get fresh result from worker
        result = worker.get_result(request.request_id)
        if result is None:
            break

    if result.status == TaskStatus.FAILED:
        raise HTTPException(status_code=500, detail=result.error)

    return result


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    llama_status = "unavailable"
    if llama_manager and llama_manager.is_running():
        llama_status = "available"

    worker = await get_worker_pool(config_data)
    return {
        "status": "healthy" if llama_status == "available" else "degraded",
        "llama_server": llama_status,
        "llama_url": get_llama_url(),
    }


@app.get("/voices")
async def list_voices():
    """List all available voices."""
    voices = [
        {
            "voice_id": "hi_male",
            "name": "Hindi (Male)",
            "languages": ["hi"],
            "gender": "male",
        },
        {
            "voice_id": "hi_female",
            "name": "Hindi (Female)",
            "languages": ["hi"],
            "gender": "female",
        },
        {
            "voice_id": "bn_male",
            "name": "Bengali (Male)",
            "languages": ["bn"],
            "gender": "male",
        },
        {
            "voice_id": "bn_female",
            "name": "Bengali (Female)",
            "languages": ["bn"],
            "gender": "female",
        },
    ]
    return {"voices": voices}


@app.post("/tts/sync")
async def generate_speech_sync(
    text: str = Form(..., description="Text to convert to speech"),
    voice: str = Form("hi_male", description="Voice ID"),
    temperature: float = Form(0.6, description="Generation temperature"),
    top_p: float = Form(0.9, description="Top-p sampling"),
    max_tokens: int = Form(1200, description="Max tokens"),
    repetition_penalty: float = Form(1.1, description="Repetition penalty"),
    api_key_verified: bool = Depends(verify_api_key),
):
    """Generate speech from text (synchronous)."""
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    result = await process_tts_request(
        text, voice, temperature, top_p, max_tokens, repetition_penalty
    )

    if result.status == TaskStatus.FAILED:
        raise HTTPException(status_code=500, detail=result.error)

    return {
        "success": True,
        "request_id": result.request_id,
        "audio": result.audio_base64,
        "format": "wav",
        "duration": result.audio_duration,
    }


@app.post("/tts/stream")
async def generate_speech_stream(
    text: str = Form(..., description="Text to convert to speech"),
    voice: str = Form("hi_male", description="Voice ID"),
    temperature: float = Form(0.6, description="Generation temperature"),
    top_p: float = Form(0.9, description="Top-p sampling"),
    max_tokens: int = Form(1200, description="Max tokens"),
    repetition_penalty: float = Form(1.1, description="Repetition penalty"),
    api_key_verified: bool = Depends(verify_api_key),
):
    """Generate speech and stream audio directly."""
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    result = await process_tts_request(
        text, voice, temperature, top_p, max_tokens, repetition_penalty
    )

    if result.status == TaskStatus.FAILED:
        raise HTTPException(status_code=500, detail=result.error)

    audio_bytes = base64.b64decode(result.audio_base64)

    from fastapi.responses import Response

    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={
            "Content-Disposition": f"attachment; filename=speech_{result.request_id[:8]}.wav",
            "X-Request-ID": result.request_id,
        },
    )


@app.get("/tts/status/{request_id}")
async def get_request_status(
    request_id: str,
    api_key_verified: bool = Depends(verify_api_key),
):
    """Get status of a TTS request."""
    worker = await get_worker_pool(config_data)
    result = worker.get_result(request_id)

    if not result:
        raise HTTPException(status_code=404, detail="Request not found")

    return {
        "request_id": result.request_id,
        "status": result.status.value,
        "audio": result.audio_base64 if result.audio_base64 else None,
        "error": result.error,
    }


@app.get("/")
async def root():
    """Root endpoint - API docs with examples."""
    from fastapi.responses import HTMLResponse

    return HTMLResponse(content=DOCS_HTML)


@app.get("/favicon.ico")
async def favicon():
    """Serve favicon."""
    from fastapi.responses import Response
    from pathlib import Path

    favicon_path = Path(__file__).parent / "favicon.ico"
    if favicon_path.exists():
        return Response(content=favicon_path.read_bytes(), media_type="image/x-icon")
    return Response(status_code=404)


DOCS_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Svara TTS API</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        :root {
            --bg: #0a0a0f;
            --bg-secondary: #12121a;
            --bg-tertiary: #1a1a24;
            --border: #2a2a3a;
            --border-hover: #3a3a4a;
            --text: #e4e4e7;
            --text-muted: #71717a;
            --accent: #6366f1;
            --accent-hover: #818cf8;
            --success: #22c55e;
            --error: #ef4444;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, var(--bg) 0%, #0f0f18 100%);
            color: var(--text);
            line-height: 1.6;
            padding: 24px;
            min-height: 100vh;
        }
        .container { max-width: 900px; margin: 0 auto; }
        h1 { 
            color: var(--text); 
            margin-bottom: 8px; 
            font-size: 32px; 
            font-weight: 700;
            background: linear-gradient(135deg, #fff 0%, var(--accent) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .subtitle { color: var(--text-muted); margin-bottom: 32px; font-size: 15px; }
        
        .section {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }
        h2 { color: var(--text); font-size: 18px; margin-bottom: 16px; font-weight: 600; }
        h3 { color: var(--text); font-size: 14px; margin-bottom: 12px; }
        
        .endpoint {
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 16px;
            margin-bottom: 12px;
        }
        .method {
            display: inline-block;
            background: var(--accent);
            color: var(--bg);
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
            margin-right: 8px;
        }
        .method.post { background: var(--success); }
        .path { color: var(--text); font-family: monospace; font-size: 14px; }
        
        .example {
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 16px;
            margin: 12px 0;
        }
        .example-label {
            font-size: 12px;
            color: var(--text-muted);
            margin-bottom: 8px;
        }
        .example-row {
            display: flex;
            gap: 12px;
            margin-bottom: 12px;
        }
        .example-row input, .example-row select {
            flex: 1;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            color: var(--text);
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 14px;
        }
        .example-row input:focus, .example-row select:focus {
            outline: none;
            border-color: var(--accent);
        }
        
        .btn {
            background: linear-gradient(135deg, var(--accent) 0%, #4f46e5 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 10px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.2s ease;
            box-shadow: 0 2px 8px rgba(99, 102, 241, 0.3);
        }
        .btn:hover { 
            background: linear-gradient(135deg, var(--accent-hover) 0%, #6366f1 100%);
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
        }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
        
        .result {
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px;
            font-family: monospace;
            font-size: 12px;
            max-height: 200px;
            overflow: auto;
            display: none;
        }
        .result.show { display: block; }
        
        .audio-result {
            margin-top: 16px;
            padding: 16px;
            background: var(--bg-tertiary);
            border-radius: 12px;
            border: 1px solid #4a4a5a;
        }
        .audio-result.show { display: block; }
        .audio-result audio {
            width: 100%;
            height: 40px;
            border-radius: 8px;
            background: var(--bg-tertiary);
            border: 1px solid #4a4a5a;
        }
        .audio-result audio::-webkit-media-controls-panel {
            background: var(--bg-tertiary);
        }
        .audio-result a {
            color: #a5a5b5;
            font-size: 13px;
            display: inline-block;
            margin-top: 8px;
        }
        
        .api-call {
            margin-top: 12px;
            padding: 12px;
            background: var(--bg);
            border-radius: 8px;
            border: 1px solid var(--border);
            font-family: monospace;
            font-size: 12px;
            color: var(--text-muted);
            word-break: break-all;
            overflow-x: auto;
        }
        .api-call span {
            color: var(--accent);
        }
        
        .voices-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 8px;
        }
        .voice-chip {
            background: var(--bg);
            border: 1px solid var(--border);
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 13px;
            cursor: pointer;
            transition: border-color 0.2s;
        }
        .voice-chip:hover { border-color: var(--accent); }
        
        .nav { margin-bottom: 24px; }
        .nav a { color: var(--accent); text-decoration: none; margin-right: 16px; }
        .nav a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="container">
        <div class="nav">
            <a href="/">Home</a>
            <a href="/docs">Swagger</a>
        </div>
        
        <h1>Svara TTS API</h1>
        <p class="subtitle">Indic Text-to-Speech with 36+ voices</p>
        
        <div class="section">
            <h2>Language & Emotion Examples</h2>
            <p style="color: var(--text-muted); font-size: 13px; margin-bottom: 16px;">
                Select a language, choose an example sentence (or type your own), and add emotions with tags.
            </p>
            
            <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 20px; flex-wrap: wrap;">
                <span style="color: var(--text-muted); font-size: 14px;">API:</span>
                <label style="display: flex; align-items: center; gap: 8px; cursor: pointer;">
                    <input type="radio" name="apiType" value="sync" checked style="accent-color: var(--accent);">
                    <span style="color: var(--text); font-size: 14px;">Sync</span>
                </label>
                <label style="display: flex; align-items: center; gap: 8px; cursor: pointer;">
                    <input type="radio" name="apiType" value="stream" style="accent-color: var(--accent);">
                    <span style="color: var(--text); font-size: 14px;">Stream</span>
                </label>
            </div>
            
            <div style="display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap;">
                <select id="langSelect" onchange="updateExamples()" style="padding: 12px 16px; background: var(--bg-tertiary); border: 1px solid var(--border); color: var(--text); border-radius: 10px; font-size: 14px; min-width: 180px;">
                    <option value="hi_male">Hindi</option>
                    <option value="hi_female">Hindi (Female)</option>
                    <option value="bn_male">Bengali</option>
                    <option value="bn_female">Bengali (Female)</option>
                    <option value="ta_male">Tamil</option>
                    <option value="ta_female">Tamil (Female)</option>
                    <option value="te_male">Telugu</option>
                    <option value="te_female">Telugu (Female)</option>
                    <option value="ml_male">Malayalam</option>
                    <option value="ml_female">Malayalam (Female)</option>
                    <option value="kn_male">Kannada</option>
                    <option value="kn_female">Kannada (Female)</option>
                    <option value="mr_male">Marathi</option>
                    <option value="mr_female">Marathi (Female)</option>
                    <option value="gu_male">Gujarati</option>
                    <option value="gu_female">Gujarati (Female)</option>
                    <option value="pa_male">Punjabi</option>
                    <option value="pa_female">Punjabi (Female)</option>
                    <option value="en_male">English</option>
                    <option value="en_female">English (Female)</option>
                </select>
                
                <select id="exampleSelect" onchange="loadExample()" style="padding: 12px 16px; background: var(--bg-tertiary); border: 1px solid var(--border); color: var(--text); border-radius: 10px; font-size: 14px; min-width: 320px;">
                </select>
            </div>
            
            <div style="margin-bottom: 12px;">
                <button class="btn" onclick="addEmotion('<giggle>')" style="font-size: 12px; padding: 4px 8px; margin-right: 4px; margin-bottom: 4px;">&lt;giggle&gt;</button>
                <button class="btn" onclick="addEmotion('<laugh>')" style="font-size: 12px; padding: 4px 8px; margin-right: 4px; margin-bottom: 4px;">&lt;laugh&gt;</button>
                <button class="btn" onclick="addEmotion('<chuckle>')" style="font-size: 12px; padding: 4px 8px; margin-right: 4px; margin-bottom: 4px;">&lt;chuckle&gt;</button>
                <button class="btn" onclick="addEmotion('<sigh>')" style="font-size: 12px; padding: 4px 8px; margin-right: 4px; margin-bottom: 4px;">&lt;sigh&gt;</button>
                <button class="btn" onclick="addEmotion('<cough>')" style="font-size: 12px; padding: 4px 8px; margin-right: 4px; margin-bottom: 4px;">&lt;cough&gt;</button>
                <button class="btn" onclick="addEmotion('<sniffle>')" style="font-size: 12px; padding: 4px 8px; margin-right: 4px; margin-bottom: 4px;">&lt;sniffle&gt;</button>
                <button class="btn" onclick="addEmotion('<groan>')" style="font-size: 12px; padding: 4px 8px; margin-right: 4px; margin-bottom: 4px;">&lt;groan&gt;</button>
                <button class="btn" onclick="addEmotion('<yawn>')" style="font-size: 12px; padding: 4px 8px; margin-right: 4px; margin-bottom: 4px;">&lt;yawn&gt;</button>
                <button class="btn" onclick="addEmotion('<gasp>')" style="font-size: 12px; padding: 4px 8px; margin-right: 4px; margin-bottom: 4px;">&lt;gasp&gt;</button>
            </div>
            
            <textarea id="customText" autocomplete="off" autocorrect="off" autocapitalize="off" spellcheck="false" placeholder="Or type/paste your own text here (supports large text)..." style="width: 100%; min-height: 120px; background: var(--bg-tertiary); border: 1px solid var(--border); color: var(--text); padding: 16px; border-radius: 12px; font-size: 15px; resize: vertical; font-family: inherit;"></textarea>
            
            <div style="margin-top: 12px;">
                <button class="btn" onclick="generateFromExample()">Generate Speech</button>
            </div>
            
            <div class="result" id="exampleResult"></div>
            <div class="audio-result" id="exampleAudio">
                <audio controls autoplay style="width: 100%;"></audio>
                <br><a href="#" download="speech.wav">Download WAV</a>
                <div class="api-call" id="apiCall"></div>
            </div>
        </div>
        
        <div class="section">
            <h2>API Endpoints</h2>
            
            <div class="endpoint">
                <span class="method post">POST</span>
                <span class="path">/tts/sync</span>
                <p style="color: var(--text-muted); font-size: 13px; margin-top: 8px;">Generate speech, returns JSON with base64 audio</p>
            </div>
            
            <div class="endpoint">
                <span class="method post">POST</span>
                <span class="path">/tts/stream</span>
                <p style="color: var(--text-muted); font-size: 13px; margin-top: 8px;">Generate speech, returns WAV audio directly</p>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span>
                <span class="path">/voices</span>
                <p style="color: var(--text-muted); font-size: 13px; margin-top: 8px;">List all available voices</p>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span>
                <span class="path">/health</span>
                <p style="color: var(--text-muted); font-size: 13px; margin-top: 8px;">Check server status</p>
            </div>
        </div>
    </div>
    
    <script>
        const examples = {
            hi_male: [
                { text: "नमस्ते, आप कैसे हैं?", label: "Greeting - How are you?" },
                { text: "आज का दिन बहुत अच्छा है।", label: "Today is a good day" },
                { text: "मैं बाजार जा रहा हूँ।", label: "I'm going to the market" },
                { text: "क्या आप मदद कर सकते हैं?", label: "Can you help?" },
                { text: "यह पुस्तक बहुत दिलचस्प है।", label: "This book is interesting" },
                { text: "मुझे खाने में कुछ चाहिए।", label: "I need something to eat" },
                { text: "कल मौसम बदलेगा।", label: "Weather will change tomorrow" },
            ],
            hi_female: [
                { text: "नमस्ते, आप कैसी हैं?", label: "Greeting - How are you?" },
                { text: "आज बहुत अच्छा दिन है।", label: "Today is a good day" },
                { text: "मैं घर जा रही हूँ।", label: "I'm going home" },
                { text: "क्या आप बता सकती हैं?", label: "Can you tell me?" },
                { text: "यह फिल्म बहुत अच्छी है।", label: "This movie is great" },
            ],
            bn_male: [
                { text: "নমস্কার, আপনি কেমন আছেন?", label: "Greeting - How are you?" },
                { text: "আজকের দিনটা ভালো।", label: "Today is a good day" },
                { text: "আমি বাজারে যাচ্ছি।", label: "I'm going to market" },
                { text: "এই বইটি খুব মজার।", label: "This book is very funny" },
            ],
            ta_male: [
                { text: "வணக்கம், நீங்க எப்படி இருக்கீங்க?", label: "Greeting - How are you?" },
                { text: "இன்று நல்ல நாள்.", label: "Today is a good day" },
                { text: "நான் வீடு போகிறேன்.", label: "I'm going home" },
                { text: "இந்த படம் சுவாரசியமானது.", label: "This movie is interesting" },
            ],
            te_male: [
                { text: "namaste, miru ela undara?", label: "Greeting - How are you?" },
                { text: "ippudu ela feel avuthundi?", label: "How are you feeling now?" },
                { text: "naaku tiffin istha", label: "I'll get lunch for me" },
            ],
            ml_male: [
                { text: "നമസ്കാരം, നിങ്ങള്‍ എങ്ങനെയാണ്?", label: "Greeting - How are you?" },
                { text: "ഇന്ന് നല്ല ദിവസമാണ്.", label: "Today is a good day" },
            ],
            kn_male: [
                { text: "ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಾ?", label: "Greeting - How are you?" },
                { text: "ಇಂದು ಚೆನ್ನಾಗಿದೆ.", label: "Today is good" },
            ],
            mr_male: [
                { text: "नमस्कार, तुम्ही कसे आहात?", label: "Greeting - How are you?" },
                { text: "आजचा दिवस चांगला आहे.", label: "Today is good" },
            ],
            gu_male: [
                { text: "નમસ્તે, તમે કેવા છો?", label: "Greeting - How are you?" },
                { text: "આજનો દિવસ સારો છે.", label: "Today is good" },
            ],
            pa_male: [
                { text: "ਨਮਸਤੇ, ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ?", label: "Greeting - How are you?" },
                { text: "ਅੱਜ ਦਿਨ ਚੰਗਾ ਹੈ।", label: "Today is good" },
            ],
            en_male: [
                { text: "Hello, how are you?", label: "Greeting - How are you?" },
                { text: "The weather is beautiful today.", label: "Weather comment" },
                { text: "I'm going to the store.", label: "Going out" },
                { text: "This is a very interesting book.", label: "Book comment" },
                { text: "Can you help me with this?", label: "Request help" },
                { text: "I would like some water please.", label: "Request drink" },
                { text: "The movie was fantastic!", label: "Movie review" },
            ],
            en_female: [
                { text: "Hello, how are you doing?", label: "Greeting - How are you?" },
                { text: "It's a lovely day today.", label: "Weather comment" },
                { text: "I'm heading to the market.", label: "Going out" },
            ],
        };
        
        // Add _female variants
        Object.keys(examples).forEach(k => {
            if (k.includes('_female') === false) {
                examples[k.replace('_male', '_female')] = examples[k];
            }
        });
        
        function updateExamples() {
            const voice = document.getElementById('langSelect').value;
            const select = document.getElementById('exampleSelect');
            select.innerHTML = '';
            
            const items = examples[voice] || [];
            items.forEach((item, i) => {
                const opt = document.createElement('option');
                opt.value = i;
                opt.textContent = item.label + ' - ' + (item.text.substring(0, 40) + (item.text.length > 40 ? '...' : ''));
                select.appendChild(opt);
            });
        }
        
        function loadExample() {
            const voice = document.getElementById('langSelect').value;
            const idx = document.getElementById('exampleSelect').value;
            const items = examples[voice] || [];
            if (items[idx]) {
                document.getElementById('customText').value = items[idx].text;
            }
        }
        
        function addEmotion(tag) {
            const textArea = document.getElementById('customText');
            const start = textArea.selectionStart;
            const end = textArea.selectionEnd;
            const text = textArea.value;
            const newText = text.substring(0, start) + tag + text.substring(end);
            textArea.value = newText;
            textArea.focus();
        }
        
        async function generateFromExample() {
            const textInput = document.getElementById('customText');
            const text = textInput.value;
            const voice = document.getElementById('langSelect').value;
            const result = document.getElementById('exampleResult');
            const audioDiv = document.getElementById('exampleAudio');
            
            console.log('generateFromExample - text:', text, 'voice:', voice, 'text length:', text.length);
            
            // Get selected API type
            const apiType = document.querySelector('input[name="apiType"]:checked').value;
            const endpoint = '/tts/' + apiType;
            
            if (!text || !text.trim()) {
                result.textContent = 'Please enter some text';
                result.classList.add('show');
                return;
            }
            
            result.textContent = 'Generating...';
            result.classList.add('show');
            audioDiv.classList.remove('show');
            
            try {
                const formData = new FormData();
                formData.append('text', text);
                formData.append('voice', voice);
                
                if (apiType === 'stream') {
                    // Stream endpoint - returns audio directly
                    const resp = await fetch(endpoint, {
                        method: 'POST',
                        body: formData
                    });
                    
                    const apiCall = 'curl -s -X POST "<span>http://' + window.location.host + '/tts/stream</span>" -F "text=' + encodeURIComponent(text) + '" -F "voice=' + voice + '" -o speech.wav';
                    audioDiv.querySelector('#apiCall').innerHTML = apiCall;
                    
                    if (resp.ok) {
                        const blob = await resp.blob();
                        const audioUrl = URL.createObjectURL(blob);
                        audioDiv.querySelector('audio').src = audioUrl;
                        audioDiv.querySelector('audio').style.display = 'block';
                        audioDiv.querySelector('a').href = audioUrl;
                        audioDiv.querySelector('a').download = 'speech.wav';
                        
                        audioDiv.classList.add('show');
                        result.textContent = 'Generated! Size: ' + (blob.size/1024).toFixed(1) + ' KB';
                    } else {
                        const data = await resp.json();
                        audioDiv.classList.add('show');
                        result.textContent = 'Error: ' + (data.detail || 'Generation failed');
                    }
                } else {
                    // Sync endpoint - returns JSON with base64 audio
                    const resp = await fetch(endpoint, {
                        method: 'POST',
                        body: formData
                    });
                    const data = await resp.json();
                    
                    const apiCall = 'curl -s -X POST "<span>http://' + window.location.host + '/tts/sync</span>" -F "text=' + encodeURIComponent(text) + '" -F "voice=' + voice + '" -o speech.wav';
                    audioDiv.querySelector('#apiCall').innerHTML = apiCall;
                    
                    if (data.audio) {
                        const audioBlob = base64ToBlob(data.audio, 'audio/wav');
                        const audioUrl = URL.createObjectURL(audioBlob);
                        audioDiv.querySelector('audio').src = audioUrl;
                        audioDiv.querySelector('audio').style.display = 'block';
                        audioDiv.querySelector('a').href = audioUrl;
                        
                        audioDiv.classList.add('show');
                        result.textContent = 'Generated! Duration: ' + data.duration?.toFixed(2) + 's';
                    } else {
                        audioDiv.classList.add('show');
                        result.textContent = 'Error: ' + (data.detail || data.error || 'No audio');
                    }
                }
            } catch(e) {
                result.textContent = 'Error: ' + e.message;
            }
        }
        
        function base64ToBlob(base64, mimeType) {
            const byteCharacters = atob(base64);
            const byteNumbers = new Array(byteCharacters.length);
            for (let i = 0; i < byteCharacters.length; i++) {
                byteNumbers[i] = byteCharacters.charCodeAt(i);
            }
            const byteArray = new Uint8Array(byteNumbers);
            return new Blob([byteArray], { type: mimeType });
        }
        
        // Initialize
        updateExamples();
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Svara TTS API Server")
    parser.add_argument("--kill", action="store_true", help="Kill any running instance")
    args = parser.parse_args()

    if args.kill:
        import signal
        import psutil

        killed = []
        target_patterns = ["svara-tts", "api_server", "llama-server"]

        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = proc.info.get("cmdline", []) or []
                cmd_str = " ".join(cmdline).lower()

                if any(p in cmd_str for p in target_patterns):
                    print(f"Killing PID {proc.info['pid']}: {cmd_str[:80]}...")
                    proc.send_signal(signal.SIGTERM)
                    killed.append(proc.info["pid"])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        if killed:
            print(f"Killed {len(killed)} process(es)")
            import time

            time.sleep(1)
        else:
            print("No running instances found")
        exit(0)

    print(f"Starting Svara TTS API Server on {server_cfg['host']}:{server_cfg['port']}")
    print(f"Workers: {server_cfg['workers']}")

    uvicorn.run(
        app,
        host=server_cfg["host"],
        port=server_cfg["port"],
    )
