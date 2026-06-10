import io
import librosa
import numpy as np
import mlx_whisper

try:
    from whisper_cpp_python import Whisper

    HAS_WHISPER_CPP = True
except ImportError:
    HAS_WHISPER_CPP = False


TARGET_SAMPLE_RATE = 16000


_whisper_cpp_instance = None


def _encode_to_wav(audio_np: np.ndarray, sr: int) -> io.BytesIO:
    """Encode float32 audio array as a WAV file in memory."""
    buf = io.BytesIO()
    import soundfile as sf

    sf.write(buf, audio_np, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf


def transcribe(
    audio_np,
    stt_model_name,
    sample_rate=TARGET_SAMPLE_RATE,
    use_whisper_cpp=False,
    use_openai=False,
    openai_model="whisper-1",
    openai_client=None,
    ggml_model_path=None,
    language=None,
):
    """
    Transcribes audio to text using mlx-whisper, whisper-cpp-python,
    or the OpenAI-compatible API (proxy).
    """
    global _whisper_cpp_instance

    # Whisper expects audio at 16kHz. Resample if needed.
    if sample_rate != TARGET_SAMPLE_RATE:
        audio_resampled = librosa.resample(
            np.asarray(audio_np, dtype=np.float32),
            orig_sr=sample_rate,
            target_sr=TARGET_SAMPLE_RATE,
            res_type="soxr_vhq",
        ).astype(np.float32)
    else:
        audio_resampled = audio_np

    # ── OpenAI-compatible API (proxy with GPU-accelerated Whisper) ──
    if use_openai:
        if openai_client is None:
            raise ValueError(
                "openai_client must be provided when use_openai=True."
            )
        wav = _encode_to_wav(
            audio_resampled.astype(np.float32) / 32768.0, TARGET_SAMPLE_RATE
        )
        result = openai_client.audio.transcriptions.create(
            model=openai_model,
            file=wav,
            language=language,
            response_format="verbose_json",
        )
        return {
            "text": result.text,
            "segments": [],
        }

    # ── whisper-cpp-python (local GGML model, CPU by default) ──
    if use_whisper_cpp:
        if not HAS_WHISPER_CPP:
            raise ImportError("whisper-cpp-python is not installed.")
        if not ggml_model_path:
            raise ValueError(
                "ggml_model_path must be provided when using whisper-cpp-python."
            )

        if _whisper_cpp_instance is None:
            _whisper_cpp_instance = Whisper(model_path=ggml_model_path)

        audio_float32 = audio_resampled.astype(np.float32) / 32768.0

        import tempfile
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, audio_float32, TARGET_SAMPLE_RATE)
            output = _whisper_cpp_instance.transcribe(open(tmp.name, "rb"))
        return output

    # ── mlx-whisper (local) ──
    return mlx_whisper.transcribe(
        audio_resampled.astype(np.float32) / 32768.0,
        fp16=False,
        path_or_hf_repo=stt_model_name,
    )
