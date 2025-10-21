import nltk
import numpy as np
import librosa
from TTS.api import TTS
from rich.console import Console
from pydub import AudioSegment, effects

console = Console()

# The native sample rate of the XTTSv2 model
SOURCE_SAMPLE_RATE = 24000


def clean_text(text):
    clean_text = text.strip()
    clean_text = [s.strip().rstrip(".") for s in nltk.sent_tokenize(clean_text)]
    return clean_text


def convert_np_to_pydub(audio_np, sample_rate):
    """Converts a float32 numpy array to a pydub AudioSegment."""
    audio_int16 = (audio_np * 32767).astype(np.int16)
    raw_audio = audio_int16.tobytes()
    return AudioSegment(
        data=raw_audio,
        sample_width=2,
        frame_rate=sample_rate,
        channels=1,
    )


def convert_pydub_to_np(audio_segment):
    """Converts a pydub AudioSegment back to a float32 numpy array."""
    raw_audio = audio_segment.raw_data
    audio_int16 = np.frombuffer(raw_audio, dtype=np.int16)
    return audio_int16.astype(np.float32) / 32767.0


def apply_noise_gate(audio_segment, threshold=-50.0, chunk_size=10):
    """
    A simple noise gate.
    Removes audio chunks quieter than the threshold.
    - threshold: The volume in dBFS. Chunks quieter than this will be silenced.
    - chunk_size: The size of chunks to analyze in milliseconds.
    """
    console.print(f"[Audio] Applying noise gate with threshold: {threshold} dBFS")
    # Split the audio into small chunks
    chunks = [
        audio_segment[i : i + chunk_size]
        for i in range(0, len(audio_segment), chunk_size)
    ]
    # Filter out chunks that are quieter than the threshold
    filtered_chunks = [chunk for chunk in chunks if chunk.dBFS > threshold]

    # If all chunks are removed, return a small silence to avoid errors
    if not filtered_chunks:
        return AudioSegment.silent(duration=10)

    # Combine the remaining chunks
    return sum(filtered_chunks)


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

                if SOURCE_SAMPLE_RATE != TARGET_SAMPLE_RATE:
                    console.print(
                        f"[Audio] Pre-processing audio for {TARGET_SAMPLE_RATE}Hz..."
                    )
                    # Convert to pydub format to use its advanced functions
                    audio_segment = convert_np_to_pydub(waveform_np, SOURCE_SAMPLE_RATE)

                    # APPLY NOISE GATE to remove background hiss from silence
                    gated_segment = apply_noise_gate(audio_segment, threshold=-50.0)

                    # Convert back to numpy for pre-emphasis
                    waveform_processed = convert_pydub_to_np(gated_segment)

                    # PRE-EMPHASIS to boost high frequencies for clarity
                    waveform_processed = librosa.effects.preemphasis(
                        waveform_processed, coef=0.97
                    )

                    # Convert back to pydub for compression
                    audio_segment = convert_np_to_pydub(
                        waveform_processed, SOURCE_SAMPLE_RATE
                    )

                    # DYNAMIC RANGE COMPRESSION to lift voice out of the 8-bit noise floor
                    compressed_segment = effects.compress_dynamic_range(
                        audio_segment,
                        threshold=-18.0,
                        ratio=4.0,
                        attack=5.0,
                        release=100.0,
                    )

                    # NORMALIZATION to boost the compressed audio to the full 0dB range
                    normalized_segment = effects.normalize(compressed_segment)
                    waveform_processed = convert_pydub_to_np(normalized_segment)

                    # 5. Resample from 24kHz down to the 8kHz target
                    waveform_resampled = librosa.resample(
                        waveform_processed,
                        orig_sr=SOURCE_SAMPLE_RATE,
                        target_sr=TARGET_SAMPLE_RATE,
                        res_type="soxr_vhq",
                    ).astype(np.float32)
                else:
                    # No resampling needed, use the original audio
                    waveform_resampled = waveform_np

                # 6. Put the final audio on the queue
                sound_queue.put(waveform_resampled.tolist())

        except Exception as e:
            print(f"[TTS Error]: {e}")
