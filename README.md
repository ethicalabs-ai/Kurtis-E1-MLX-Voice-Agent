# 🧠 Kurtis-EON1-Voice-Agent

A privacy-focused, **offline voice assistant for macOS & Linux**, powered by:

- 🧠 **Local LLM Inference**: Powered by local engines via `mlx-lm` or any OpenAI-compatible API endpoint (Ollama, LM Studio, vLLM).
- 🎤 **Triple-Backend Speech-to-Text**: Real-time transcription via `mlx-whisper`, `whisper-cpp-python` (GGML format), or the **OpenAI-compatible API** (GPU-accelerated on your proxy server).
- 🌍 **Offline Translations**: On-device multilingual translations powered by [ethicalabs/Tower-Plus-2B-mlx](https://huggingface.co/ethicalabs/Tower-Plus-2B-mlx).
- 🗣️ **Emotive Multilingual TTS**: High-quality voice synthesis powered by XTTS v2.
- 🖥️ **Interactive Visualizer**: Responsive wxPython GUI featuring a real-time hue-shifting gradient visualizer.

This project supports **both Apple Silicon Macs (via MLX) and Linux systems (including native AMD ROCm and CPU execution)**.

It prioritizes simplicity, speed, and on-device privacy for empathetic mental health conversations.

---

## 🛠️ Requirements

To run this project, you'll need:

- Python >=3.11
- Open-AI compatible API endpoint (Ollama, LM Studio, vLLM...)

Default Open-AI API endpoint (mlx-lm) is set as default and already pre-installed:

```
$ mlx_lm.server
2025-10-19 20:24:49,928 - INFO - Starting httpd at 127.0.0.1 on port 8080
```

For LM Studio you can set the following environment variables:

```bash
# Remember to start LM Studio server.
export OPENAI_API_URL=http://localhost:1234/v1
export OPENAI_API_KEY=lmstudio
```

For Lemonade Server:

```bash
# Start Lemonade Server (runs on default port 13305)
export OPENAI_API_URL=http://localhost:13305/v1
export OPENAI_API_KEY=lemonade
```


## 🚀 Quick Usage

We recommend using [`uv`](https://github.com/astral-sh/uv) as the Python runner.

Depending on your backend setup and hardware, you can run the agent in different configurations:

### 1. Graphical UI Mode (Highly Recommended)
To run the interactive voice assistant with the responsive, hue-shifting visualizer GUI:
```bash
# For AMD GPU / ROCm (uses whisper.cpp and local Lemonade model)
# Note: `--max-tokens 1024` is required for reasoning models (e.g., Qwen 3.5/3.6 Instruct GGUFs)
# as they output system-thinking tokens before the actual response.
uv run --extra rocm python3 -m kurtis_mlx --ui --whisper-backend whisper_cpp --ggml-model-path models/ggml-tiny.bin --llm-model Qwen3.5-4B-GGUF --max-tokens 1024

# For macOS / Apple Silicon (uses MLX-Whisper and mlx-lm server)
uv run python3 -m kurtis_mlx --ui

# GPU-accelerated via OpenAI-compatible proxy (e.g. Lemonade, vLLM)
# Transcription runs on the proxy's GPU; LLM runs on the same endpoint.
uv run --extra rocm python3 -m kurtis_mlx --ui --whisper-backend openai --llm-model Qwen3.5-4B-GGUF --max-tokens 4096
```

### 2. Headless CLI Mode
To run the voice assistant directly in your terminal without the GUI window:
```bash
# Headless run using local CPU/GPU whisper.cpp
uv run --extra rocm python3 -m kurtis_mlx --whisper-backend whisper_cpp --ggml-model-path models/ggml-tiny.bin --llm-model Qwen3.5-4B-GGUF --max-tokens 1024
```

### CLI Customizations
You can fully customize the agent's behavior with the following options:

- `--ui`: Enable the interactive graphical GUI.
- `--language`: Select between `english`, `italian`, etc. (defaults to `english`).
- `--speaker`: Change the default TTS speaker.
- `--translate`: Translate your native language spoken into English for English-only LLMs.
- `--llm-model`: Specify the LLM model identifier.
- `--whisper-backend`: Select between `mlx` (macOS native), `whisper_cpp` (Linux/Cross-platform, local GGML model), or `openai` (GPU-accelerated via your OpenAI-compatible proxy).
- `--ggml-model-path`: Specify the path to a GGML-format Whisper model (required for `whisper_cpp` backend, e.g., `models/ggml-tiny.bin`).
- `--whisper-model-openai`: Whisper model name for the `openai` backend. Defaults to `Whisper-Large-v3-Turbo1`.

---

## 🔄 Goals

- ✅ Faster startup and playback (TTS runs in background worker)
- 🔐 100% offline: STT, LLMs and TTS run locally
- ☁️ Optional offline translation (only when `--translate` is enabled)

---

## 🤝 Contributing

### Development Prerequisites (Linux)

To compile and install all development and graphical package dependencies (such as `pyaudio` for audio capturing and `wxpython` for the GUI), install the required system libraries first:

```bash
# Install PortAudio (for audio streams) and GTK3/OpenGL (for the wxPython GUI)
sudo apt-get update && sudo apt-get install -y \
    portaudio19-dev \
    libgtk-3-dev \
    libglu1-mesa-dev
```

### Setup & Installation

Once the system packages are installed, set up your development environment using [`uv`](https://github.com/astral-sh/uv).

Depending on your hardware backend, you can sync with either the AMD ROCm GPU extra or the CPU-only extra:

#### For AMD ROCm GPU:
```bash
CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5" uv sync --extra rocm
```

#### For CPU-only:
```bash
CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5" uv sync --extra cpu
```

*(Note: Passing `CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"` ensures that `whisper-cpp-python` successfully compiles with modern versions of CMake).*

### Linting and Formatting

We enforce code quality standards using `pre-commit` hooks (managing `black` and `ruff`). To run validation checks:

```bash
uvx pre-commit run --all
```
