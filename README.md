# 🧠 Kurtis-E1-MLX-Voice-Agent

A privacy-focused, **offline voice assistant for macOS**, powered by:

- 🧠 Local LLM inference via [Ollama](https://ollama.com/) (replaceable with [LM Studio](https://lmstudio.ai) or any OpenAI-compatible API endpoint)
- 🎤 Speech-to-text via [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper)
- 🌍 Offline translations via [Unbabel/TowerInstruct-13B-v0.1](https://huggingface.co/Unbabel/TowerInstruct-13B-v0.1)
- 🗣️ High-quality multilingual TTS (currently using XTTS v2)

This project is designed **specifically for Apple Silicon Macs**.

It prioritizes simplicity, speed, and on-device privacy for empathetic mental health conversations.

---

## 🛠️ Requirements

To run this project, you'll need:

- Python >=3.11
- Open-AI compatible API endpoint (Ollama, LM Studio, vLLM...)

Default Open-AI API endpoint (Ollama) is set as default. For LM Studio you can set the following environment variables:

```
# Remember to start LM Studio server.
export OPENAI_API_URL=http://localhost:1234/v1
export OPENAI_API_KEY=lmstudio
```

Remember to pull your model from Ollama (or LM Studio) before running the experiment.

```
# Default Ollama GGUF models

ollama pull hf.co/ethicalabs/Kurtis-E1-SmolLM2-1.7B-Instruct-Q4_K_M-GGUF
ollama pull hf.co/mradermacher/TowerInstruct-7B-v0.2-GGUF:Q4_K_S
```


## 🚀 Quick Usage

We recommend using [`uv`](https://github.com/astral-sh/uv) as the Python runner:

```bash
uv run python3 -m kurtis_mlx
```

You can customize:

- `--language`: Select between `english`, `italian`, etc.
- `--speaker`: Change default speaker.
- `--translate`: Use your native language while chatting with an English-only LLM
- `--llm-model`: Defaults to Kurtis-E1 via Ollama
- `--tts-model`: Use a different voice model (e.g., XTTS v2)
- `--whisper-model`: Switch out Whisper variants

---

## 🔄 Goals

- ✅ Faster startup and playback (TTS runs in background worker)
- 🔐 100% offline: STT, LLMs and TTS run locally
- ☁️ Optional offline translation (only when `--translate` is enabled)
