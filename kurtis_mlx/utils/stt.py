import librosa
import numpy as np
import mlx_whisper


TARGET_SAMPLE_RATE = 16000


def transcribe(audio_np, stt_model_name, sample_rate=TARGET_SAMPLE_RATE):
    """
    Transcribes audio to text using mlx-whisper.
    The sample rate of the audio must be provided.
    """
    # Whisper expects audio at 16kHz. We need to resample if it's different.
    if sample_rate != TARGET_SAMPLE_RATE:
        audio_resampled = librosa.resample(
            np.asarray(audio_np, dtype=np.float32),
            orig_sr=sample_rate,
            target_sr=TARGET_SAMPLE_RATE,
            res_type="soxr_vhq",  # Use a high-quality resampler
        ).astype(np.float32)
    else:
        audio_resampled = audio_np

    # This will now correctly normalize:
    # 1. The new 16kHz resampled float array (from 8kHz)
    # 2. Or the original 16kHz int16 array (in non-SIP mode)
    return mlx_whisper.transcribe(
        audio_resampled.astype(np.float32) / 32768.0,
        fp16=False,
        path_or_hf_repo=stt_model_name,
    )
