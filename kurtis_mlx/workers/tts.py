import nltk
from TTS.api import TTS
from rich.console import Console

console = Console()


def clean_text(text):
    clean_text = text.strip()
    clean_text = [s.strip().rstrip(".") for s in nltk.sent_tokenize(clean_text)]
    return clean_text


def tts_worker(text_queue, sound_queue, tts_model, samplerate, lang_code, speaker):
    nltk.download("punkt_tab")
    tts = TTS(model_name=tts_model, progress_bar=False, gpu=False)

    while True:
        text = text_queue.get()
        if text is None:
            break
        try:
            console.print("[blue]Speaking: ...")
            text = clean_text(text)
            for t in text:
                waveform = tts.tts(t, language=lang_code, speaker=speaker)
                sound_queue.put(waveform)
        except Exception as e:
            print(f"[TTS Error]: {e}")
