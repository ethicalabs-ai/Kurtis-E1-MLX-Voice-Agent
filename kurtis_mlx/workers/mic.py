import time
import numpy as np
import sounddevice as sd
import scipy.signal
from rich.console import Console

from kurtis_mlx.utils.vad import VADCollector
from kurtis_mlx import config

console = Console()

# 16 khz for local recording is supported by webrtcvad and it's whisper's target sample rate.
TARGET_SAMPLE_RATE = 16000
VAD_FRAME_MS = config.VAD_FRAME_MS  # 30ms
VAD_BLOCK_SAMPLES = int(TARGET_SAMPLE_RATE * (VAD_FRAME_MS / 1000.0))
# 16000 * 0.030 = 480 samples per frame


def mic_worker(
    transcription_queue,
    is_busy_event,
    pause_event,
    ui_queue=None,
    control_queue=None,
    debug=False,
):
    """
    Listens to the microphone, applies VAD, and puts
    speech utterances into the transcription_queue.
    """
    try:
        # Try to get available devices
        devices = sd.query_devices()
        console.print(f"[mic_worker] Available audio devices: {len(devices)}")

        # Find default input device
        # Find default input device
        default_input = sd.default.device[0]
        if default_input is None:
            # Find first input device
            for i, device in enumerate(devices):
                if device["max_input_channels"] > 0:
                    default_input = i
                    break

        if default_input is not None:
            console.print(f"[mic_worker] Initial device index: {default_input}")
        else:
            console.print("[mic_worker] No input device found.")
            return

        vad_collector = VADCollector(
            sample_rate=TARGET_SAMPLE_RATE,
            aggressiveness=config.VAD_AGGRESSIVENESS,  # from config
            frame_ms=VAD_FRAME_MS,  # from config
            silence_ms=config.SILENCE_FRAMES_THRESHOLD
            * VAD_FRAME_MS,  # e.g. 30 * 30 = 900ms
            min_speech_ms=250,
            debug=False,
        )

        console.print(f"[mic_worker] Listening for speech at {TARGET_SAMPLE_RATE}Hz...")

        # Main loop to allow restarting stream with new device
        while True:
            stream = None
            current_samplerate = TARGET_SAMPLE_RATE
            current_blocksize = VAD_BLOCK_SAMPLES
            selected_channel = 0

            try:
                # Try to open stream with different configurations
                try:
                    # Attempt 1: Preferred settings (16kHz, 1 channel, low latency)
                    stream = sd.InputStream(
                        device=default_input,
                        samplerate=TARGET_SAMPLE_RATE,
                        channels=1,
                        dtype="float32",  # Use float32 for easier resampling if needed
                        blocksize=VAD_BLOCK_SAMPLES,
                        latency="low",
                    )
                    stream.start()
                    current_samplerate = TARGET_SAMPLE_RATE
                    current_blocksize = VAD_BLOCK_SAMPLES
                except Exception:
                    # console.print(f"[mic_worker] Failed to open with preferred settings: {e1}")
                    try:
                        # Attempt 2: Device default sample rate
                        device_info = sd.query_devices(default_input, "input")
                        default_sr = int(device_info["default_samplerate"])
                        # console.print(f"[mic_worker] Trying device default sample rate: {default_sr}")

                        # Calculate new blocksize for 30ms at this rate
                        # 30ms = 0.03s
                        current_blocksize = int(default_sr * VAD_FRAME_MS / 1000)

                        stream = sd.InputStream(
                            device=default_input,
                            samplerate=default_sr,
                            channels=1,
                            dtype="float32",
                            blocksize=current_blocksize,
                        )
                        stream.start()
                        current_samplerate = default_sr
                    except Exception:
                        # console.print(f"[mic_worker] Failed to open with default rate: {e2}")
                        # Attempt 3: Default everything (let SD decide)
                        stream = sd.InputStream(
                            device=default_input,
                            channels=1,
                            dtype="float32",
                        )
                        stream.start()
                        current_samplerate = int(stream.samplerate)
                        current_blocksize = stream.blocksize
                        console.print(
                            f"[mic_worker] Fallback: Rate={current_samplerate}, Block={current_blocksize}"
                        )

                console.print(
                    f"[mic_worker] Stream started on device {default_input} at {current_samplerate}Hz"
                )

                while True:
                    # Check for control messages
                    if control_queue and not control_queue.empty():
                        try:
                            msg = control_queue.get_nowait()
                            if msg.get("action") == "set_device":
                                new_device = msg.get("device_index")
                                console.print(
                                    f"[mic_worker] Switching to device: {new_device}"
                                )
                                default_input = new_device
                                selected_channel = 0  # Reset channel on device change
                                break  # Break inner loop to restart stream
                            elif msg.get("action") == "set_channel":
                                selected_channel = msg.get("channel_index", 0)
                                console.print(
                                    f"[mic_worker] Switching to channel: {selected_channel}"
                                )
                            elif msg.get("action") == "toggle_pause":
                                if pause_event.is_set():
                                    pause_event.clear()
                                    console.print("[mic_worker] Resumed.")
                                else:
                                    pause_event.set()
                                    console.print("[mic_worker] Paused.")
                        except Exception:
                            pass

                    # Handle Pause
                    if pause_event.is_set():
                        # Read and discard to keep stream alive but not process
                        stream.read(current_blocksize)
                        time.sleep(0.1)
                        continue

                    # Read a block (frame) of audio
                    block, overflow = stream.read(current_blocksize)
                    if overflow:
                        pass
                        # console.print("[mic_worker] Audio overflow")

                    # Handle multi-channel input: take the selected channel
                    if block.ndim > 1 and block.shape[1] > selected_channel:
                        block = block[:, selected_channel]
                    elif block.ndim > 1:
                        block = block[:, 0]  # Fallback to first channel
                    elif block.ndim > 1:
                        block = block.flatten()

                    # Resample if necessary
                    if current_samplerate != TARGET_SAMPLE_RATE:
                        # Resample to 16kHz
                        # We need 480 samples at 16kHz
                        # scipy.signal.resample is faster than librosa for fixed size
                        num_samples = VAD_BLOCK_SAMPLES
                        block = scipy.signal.resample(block, num_samples)

                    # Ensure we have exactly VAD_BLOCK_SAMPLES (480)
                    if len(block) != VAD_BLOCK_SAMPLES:
                        # Pad or trim
                        if len(block) > VAD_BLOCK_SAMPLES:
                            block = block[:VAD_BLOCK_SAMPLES]
                        else:
                            block = np.pad(block, (0, VAD_BLOCK_SAMPLES - len(block)))

                    # Calculate RMS for debug or UI
                    # block is float32
                    rms = np.sqrt(np.mean(block**2))
                    normalized_rms = rms

                    if debug and np.random.rand() < 0.05:  # Log 5% of packets
                        console.print(f"[mic_worker] RMS: {normalized_rms:.4f}")

                    if ui_queue is not None:
                        try:
                            ui_queue.put_nowait(
                                {"type": "rms", "value": normalized_rms}
                            )
                        except Exception:
                            pass  # Queue might be full, ignore

                    # Convert to int16 for VAD
                    audio_int16 = (block * 32767).astype(np.int16)
                    audio_bytes = audio_int16.tobytes()

                    # Determine silence threshold
                    # If agent is speaking (is_busy), we want to interrupt faster, so we use a shorter silence threshold.
                    # Normal: 50 frames (~1.5s)
                    # Interruption: 10 frames (~300ms)
                    silence_threshold = None
                    if is_busy_event.is_set():
                        silence_threshold = 10  # 300ms

                    # Process audio with VAD
                    for speech_chunk in vad_collector.process_audio(
                        audio_bytes, silence_threshold_override=silence_threshold
                    ):
                        if speech_chunk is not None:
                            # We have a complete utterance
                            if debug:
                                console.print(
                                    f"[VAD] Queuing {len(speech_chunk)} audio samples for transcription."
                                )
                            # The queue expects the np.ndarray
                            transcription_queue.put(speech_chunk)

            except Exception as e:
                console.print(f"[bold red][mic_worker Stream Error] {e}[/bold red]")
                time.sleep(2)  # Wait a bit before retrying
            finally:
                if stream:
                    stream.stop()
                    stream.close()

    except KeyboardInterrupt:
        console.print("\n[mic_worker] Interrupted.")
    except Exception as e:
        console.print(f"[bold red][mic_worker Error] {e}[/bold red]")
    finally:
        transcription_queue.put(None)  # Signal shutdown
        console.print("[mic_worker] Process finished.")
