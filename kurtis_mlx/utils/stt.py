import numpy as np
import mlx_whisper


def transcribe(audio_np, stt_model_name):
    return mlx_whisper.transcribe(
        audio_np.astype(np.float32) / 32768.0,
        fp16=False,
        path_or_hf_repo=stt_model_name,
    )["text"]
