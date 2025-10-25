import numpy as np
import sounddevice as sd
from rich.console import Console

console = Console()


def sd_worker(sound_queue, samplerate, is_busy_event):
    while True:
        try:
            au = sound_queue.get()
        except KeyboardInterrupt:
            break
        else:
            if au is None:
                break
        try:
            console.print("[purple]Playing Audio: ...")
            is_busy_event.set()
            au_np = np.asarray(au, dtype=np.float32)
            with sd.OutputStream(
                samplerate=samplerate, channels=1, dtype="float32"
            ) as stream:
                stream.write(au_np)
                stream.stop()
        except Exception as e:
            print(f"[Audio Error]: {e}")
        finally:
            is_busy_event.clear()
