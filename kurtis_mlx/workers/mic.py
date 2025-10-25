import time
import sounddevice as sd
from rich.console import Console

from kurtis_mlx.utils.vad import VADCollector
from kurtis_mlx import config

console = Console()

# 16 khz for local recording is supported by webrtcvad and it's whisper's target sample rate.
TARGET_SAMPLE_RATE = 16000
VAD_FRAME_MS = config.VAD_FRAME_MS  # 30ms
VAD_BLOCK_SAMPLES = int(TARGET_SAMPLE_RATE * (VAD_FRAME_MS / 1000.0))
# 16000 * 0.030 = 480 samples per frame


def mic_worker(transcription_queue, is_busy_event):
    """
    Listens to the microphone, applies VAD, and puts
    speech utterances into the transcription_queue.
    """
    try:
        vad_collector = VADCollector(
            sample_rate=TARGET_SAMPLE_RATE,
            aggressiveness=config.VAD_AGGRESSIVENESS,  # from config
            frame_ms=VAD_FRAME_MS,  # from config
            silence_ms=config.SILENCE_FRAMES_THRESHOLD
            * VAD_FRAME_MS,  # e.g. 30 * 30 = 900ms
            min_speech_ms=2000,
            debug=False,
        )

        console.print("[mic_worker] Listening for speech (16kHz)...")
        with sd.InputStream(
            samplerate=TARGET_SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=VAD_BLOCK_SAMPLES,
            latency="low",
        ) as stream:
            while True:
                if is_busy_event.is_set():
                    # If audio is playing, discard audio from the stream
                    # to prevent a backlog, and skip processing.
                    stream.read(VAD_BLOCK_SAMPLES)
                    time.sleep(0.01)  # Yield CPU
                    continue

                # Read a block (frame) of audio
                block, _ = stream.read(VAD_BLOCK_SAMPLES)

                # Get the raw bytes for the VAD
                audio_bytes = block[:, 0].tobytes()

                # Process with VAD. This will yield full utterances
                for utterance in vad_collector.process_audio(audio_bytes):
                    if utterance is not None:
                        console.print(
                            f"[VAD] Queuing {len(utterance)} audio samples for transcription."
                        )
                        # The queue expects the np.ndarray
                        transcription_queue.put(utterance)

    except KeyboardInterrupt:
        console.print("\n[mic_worker] Interrupted.")
    except Exception as e:
        console.print(f"[bold red][mic_worker Error] {e}[/bold red]")
    finally:
        transcription_queue.put(None)  # Signal shutdown
        console.print("[mic_worker] Process finished.")
