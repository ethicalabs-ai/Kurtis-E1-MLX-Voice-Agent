import numpy as np
import sounddevice as sd
from rich.console import Console

console = Console()


def sd_worker(sound_queue, samplerate):
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
            au_np = np.asarray(au, dtype=np.float32)
            with sd.OutputStream(
                samplerate=samplerate, channels=1, dtype="float32"
            ) as stream:
                stream.write(au_np)
                stream.stop()
        except Exception as e:
            print(f"[Audio Error]: {e}")
