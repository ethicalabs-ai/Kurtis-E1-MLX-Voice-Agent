import nltk
import numpy as np
import librosa
from TTS.api import TTS
from rich.console import Console

console = Console()

# The native sample rate of the XTTSv2 model
SOURCE_SAMPLE_RATE = 24000


def clean_text(text):
    clean_text = text.strip()
    clean_text = [s.strip().rstrip(".") for s in nltk.sent_tokenize(clean_text)]
    return clean_text


def tts_worker(text_queue, sound_queue, tts_model, samplerate, lang_code, speaker):
    TARGET_SAMPLE_RATE = samplerate

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        console.print("Downloading punkt tokenizer...")
        nltk.download("punkt_tab")

    tts = TTS(model_name=tts_model, progress_bar=False, gpu=False)

    while True:
        try:
            text = text_queue.get()
        except KeyboardInterrupt:
            break
        else:
            if text is None:
                break
        try:
            console.print("[blue]Speaking: ...")
            text = clean_text(text)
            for t in text:
                # 1. Generate audio at its native 24kHz
                waveform_list = tts.tts(t, language=lang_code, speaker=speaker)
                waveform_np = np.asarray(waveform_list, dtype=np.float32)

                # 2. Resample directly to the target sample rate if needed.
                #    All complex processing has been removed to ensure intelligibility.
                if SOURCE_SAMPLE_RATE != TARGET_SAMPLE_RATE:
                    console.print(
                        f"[Audio] Resampling audio to {TARGET_SAMPLE_RATE}Hz..."
                    )
                    waveform_resampled = librosa.resample(
                        waveform_np,
                        orig_sr=SOURCE_SAMPLE_RATE,
                        target_sr=TARGET_SAMPLE_RATE,
                        res_type="soxr_vhq",  # Use a high-quality resampler
                    ).astype(np.float32)
                else:
                    # No resampling needed, use the original audio
                    waveform_resampled = waveform_np

                # 3. Put the final, clean audio on the queue
                sound_queue.put(waveform_resampled.tolist())

        except Exception as e:
            print(f"[TTS Error]: {e}")
