import numpy as np
import mlx_whisper
from scipy.signal import resample


def transcribe(audio_np, stt_model_name, sample_rate=16000):
    """
    Transcribes audio to text using mlx-whisper.
    The sample rate of the audio must be provided.
    """
    # Whisper expects audio at 16kHz. We need to resample if it's different.
    if sample_rate != 16000:
        resampling_factor = 16000 / sample_rate
        num_samples = int(len(audio_np) * resampling_factor)

        # Do NOT cast to int16. resample() outputs a float.
        audio_resampled = resample(audio_np, num_samples)
    else:
        audio_resampled = audio_np

    # This will now correctly normalize:
    # 1. The new 16kHz resampled float array (from 8kHz)
    # 2. Or the original 16kHz int16 array (in non-SIP mode)
    return mlx_whisper.transcribe(
        audio_resampled.astype(np.float32) / 32768.0,
        fp16=False,
        path_or_hf_repo=stt_model_name,
    )["text"]
