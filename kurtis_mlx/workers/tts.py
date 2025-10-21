import nltk
import numpy as np  # <-- ADD
from scipy.signal import resample  # <-- ADD
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
    # samplerate here is 8000, passed from __main__.py
    TARGET_SAMPLE_RATE = samplerate

    try:
        nltk.data.find("tokenizers/punkt")
        console.print("Punkt tokenizer is already downloaded.")
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

                # --- START MODIFICATION ---

                # 2. Convert to numpy array for processing
                waveform_np = np.asarray(waveform_list, dtype=np.float32)

                # 3. Resample from 24kHz down to the 8kHz target
                if SOURCE_SAMPLE_RATE != TARGET_SAMPLE_RATE:
                    resampling_factor = TARGET_SAMPLE_RATE / SOURCE_SAMPLE_RATE
                    num_samples = int(len(waveform_np) * resampling_factor)
                    waveform_resampled = resample(waveform_np, num_samples)
                else:
                    waveform_resampled = waveform_np

                # 4. Put the 8kHz, 16-bit float audio on the queue
                sound_queue.put(waveform_resampled.tolist())
                # --- END MODIFICATION ---

        except Exception as e:
            print(f"[TTS Error]: {e}")
