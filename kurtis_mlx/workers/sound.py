import numpy as np
import sounddevice as sd
import time
from rich.console import Console

console = Console()


def sd_worker(
    sound_queue, samplerate, is_busy_event, interrupt_event, pause_event, ui_queue=None
):
    while True:
        try:
            au = sound_queue.get()
        except KeyboardInterrupt:
            break
        else:
            if au is None:
                break

        # Check for pause before starting
        if pause_event.is_set():
            # If paused, we might want to discard or wait?
            # For now, let's discard to avoid backlog when resuming,
            # or we could wait. Discarding seems safer for a "live" feel.
            # But if it's a long response, maybe we want to hear it?
            # Let's wait a bit and check again, or just skip.
            # Given "pause", skipping seems appropriate for new audio.
            continue

        try:
            console.print("[purple]Playing Audio: ...")
            is_busy_event.set()
            if ui_queue:
                try:
                    ui_queue.put_nowait({"type": "status", "state": "speaking"})
                except Exception:
                    pass

            au_np = np.asarray(au, dtype=np.float32)

            # Play audio
            with sd.OutputStream(
                samplerate=samplerate, channels=1, dtype="float32"
            ) as stream:
                # We need to feed the stream in chunks to send RMS updates in real-time
                # otherwise the UI will only get one RMS value (or none) for the whole clip
                chunk_size = 1024
                for i in range(0, len(au_np), chunk_size):
                    # Check for interruption
                    if interrupt_event.is_set():
                        console.print("[yellow]Playback interrupted.")
                        # Do not clear interrupt_event here. It is cleared by handlers.py before new playback.
                        # interrupt_event.clear()
                        # Clear the queue to stop pending sentences
                        while not sound_queue.empty():
                            try:
                                sound_queue.get_nowait()
                            except Exception:
                                break
                        break

                    # Check for pause during playback
                    if pause_event.is_set():
                        console.print("[yellow]Playback paused.")
                        stream.stop()
                        # Wait until resumed or interrupted
                        while pause_event.is_set():
                            if interrupt_event.is_set():
                                # Do not clear interrupt_event here.
                                # interrupt_event.clear()
                                # Clear the queue to stop pending sentences
                                while not sound_queue.empty():
                                    try:
                                        sound_queue.get_nowait()
                                    except Exception:
                                        break
                                break  # Break inner wait, then will break outer loop
                            time.sleep(0.1)

                        if (
                            interrupt_event.is_set()
                        ):  # Double check if we broke due to interrupt
                            break
                        stream.start()

                    chunk = au_np[i : i + chunk_size]

                    # Calculate RMS for this chunk
                    if ui_queue:
                        rms = np.sqrt(np.mean(chunk**2))
                        try:
                            ui_queue.put_nowait({"type": "rms", "value": float(rms)})
                        except Exception:
                            pass

                    stream.write(chunk)

                # stream.stop() is called automatically by context manager exit or we can let it drain
                # But we might want to ensure it's finished
                # stream.stop()
        except Exception as e:
            console.print(f"[bold red][Audio Error]: {e}[/bold red]")
        finally:
            is_busy_event.clear()
            if ui_queue:
                try:
                    ui_queue.put_nowait({"type": "status", "state": "listening"})
                except Exception:
                    pass
