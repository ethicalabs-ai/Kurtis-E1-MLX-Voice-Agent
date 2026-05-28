import librosa

import numpy as np
from TTS.api import TTS
from rich.console import Console

console = Console()
_tts_model = None


def text_to_speech(model_name, lang_code, speaker, orig_sr, target_sr, text):
    global _tts_model
    if not _tts_model:
        _tts_model = TTS(model_name=model_name, progress_bar=False, gpu=False)
    waveform_list = _tts_model.tts(text, language=lang_code, speaker=speaker)
    waveform_np = np.asarray(waveform_list, dtype=np.float32)
    if orig_sr != target_sr:
        console.print(f"[Audio] Resampling audio to {target_sr}Hz...")
        waveform_resampled = librosa.resample(
            waveform_np,
            orig_sr=orig_sr,
            target_sr=target_sr,
            res_type="soxr_vhq",  # Use a high-quality resampler
        ).astype(np.float32)
    else:
        # No resampling needed, use the original audio
        waveform_resampled = waveform_np
    return waveform_resampled
