# 🧠 Kurtis-E1-MLX-Voice-Agent

A privacy-focused, **offline voice assistant for macOS**, powered by:

- 🧠 Local LLM inference via **Ollama** (soon replaceable with [LM Studio](https://lmstudio.ai) for MLX backend)
- 🎤 Speech-to-text via [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper)
- 🌍 Offline translations via [Unbabel/TowerInstruct-13B-v0.1](https://huggingface.co/Unbabel/TowerInstruct-13B-v0.1)
- 🗣️ High-quality multilingual TTS (currently using XTTS v2)

This project is designed **specifically for Apple Silicon Macs**.

It prioritizes simplicity, speed, and on-device privacy for empathetic mental health conversations.

---

## 🚀 Quick Usage

We recommend using [`uv`](https://github.com/astral-sh/uv) as the Python runner:

```bash
uv run python3 -m kurtis_mlx
```

You can customize:

- `--language`: Select between `english`, `italian`, etc.
- `--translate`: Use your native language while chatting with an English-only LLM
- `--llm-model`: Defaults to Kurtis-E1 via Ollama
- `--tts-model`: Use a different voice model (e.g., XTTS v2)
- `--whisper-model`: Switch out Whisper variants

---

## 🔄 Goals

- ✅ Replace Ollama with **LM Studio's MLX endpoints**
- ✅ Faster startup and playback (TTS runs in background worker)
- 🔐 100% offline: STT, LLMs and TTS run locally
- ☁️ Optional offline translation (only when `--translate` is enabled)
