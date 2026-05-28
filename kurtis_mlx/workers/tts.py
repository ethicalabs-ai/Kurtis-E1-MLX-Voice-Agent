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


def tts_worker(text_queue, sound_queue, tts_model, samplerate, lang_code, speaker, interrupt_event):
    TARGET_SAMPLE_RATE = samplerate

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        console.print("Downloading punkt tokenizer...")
        nltk.download("punkt_tab")

    try:
        tts = TTS(model_name=tts_model, progress_bar=False, gpu=False)
    except Exception as e:
        console.print(f"[bold red][TTS Error] Failed to load TTS model: {e}[/bold red]")
        return

    while True:
        try:
            # Check for interruption before getting new text
            if interrupt_event.is_set():
                # Clear text queue
                while not text_queue.empty():
                    try:
                        text_queue.get_nowait()
                    except:
                        break
                # We don't clear the event here, sound_worker or handlers might do it, 
                # or we can clear it if we are the ones handling it.
                # But usually sound_worker clears it after stopping playback.
                # If we clear it here, sound_worker might not see it?
                # Actually, interrupt_event is set by handlers.
                # sound_worker sees it and stops playback and clears it.
                # If tts_worker sees it, it should also stop.
                # Race condition: who clears it?
                # Maybe we don't clear it here, just skip.
                # But if sound_worker clears it fast, we might miss it?
                # It's better if handlers set it, and everyone checks it.
                # But clearing is tricky.
                # Let's assume sound_worker is the primary consumer for "stopping playback".
                # tts_worker is for "stopping generation".
                # If sound_worker clears it, tts_worker might continue generating the rest of the sentence.
                # That's okay, sound_worker will just discard it or play it (if it cleared it).
                # Wait, if sound_worker clears it, then plays next chunk...
                # We need a robust way.
                # Maybe tts_worker should check if it's set, and if so, discard everything.
                pass

            text = text_queue.get(timeout=0.1)
        except:
            continue
            
        if text is None:
            break

        # Check interruption again after getting text
        if interrupt_event.is_set():
             console.print("[yellow]TTS generation skipped due to interruption.")
             continue

        sentences = clean_text(text.strip())

        for sentence in sentences:
            # Check interruption before each sentence
            if interrupt_event.is_set():
                console.print("[yellow]TTS generation interrupted.")
                break

            try:
                waveform_list = tts.tts(sentence, language=lang_code, speaker=speaker)
                
                # Check interruption after generation (before sending)
                if interrupt_event.is_set():
                    console.print("[yellow]TTS generation interrupted (discarding).")
                    break

                waveform_np = np.asarray(waveform_list, dtype=np.float32)
                if SOURCE_SAMPLE_RATE != TARGET_SAMPLE_RATE:
                    console.print(f"[Audio] Resampling audio to {TARGET_SAMPLE_RATE}Hz...")
                    waveform_resampled = librosa.resample(
                        waveform_np,
                        orig_sr=SOURCE_SAMPLE_RATE,
                        target_sr=TARGET_SAMPLE_RATE,
                        res_type="soxr_vhq",  # Use a high-quality resampler
                    ).astype(np.float32)
                else:
                    # No resampling needed, use the original audio
                    waveform_resampled = waveform_np
                sound_queue.put(waveform_resampled.tolist())
            except Exception as e:
                console.print(f"[bold red][TTS Error] Failed to process sentence: {e}[/bold red]")
                # Put an empty list to avoid blocking
                sound_queue.put([])
