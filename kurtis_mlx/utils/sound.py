import threading
import numpy as np
import sounddevice as sd
from rich.console import Console


console = Console()


def record_until_enter(samplerate):
    blocksize = 1024
    audio_chunks = []
    console.print("[green]Recording... Press Enter to stop.")

    stop_flag = threading.Event()

    def wait_for_enter():
        input()
        stop_flag.set()

    threading.Thread(target=wait_for_enter, daemon=True).start()

    with sd.InputStream(
        samplerate=samplerate,
        channels=1,
        dtype="int16",
        blocksize=blocksize,
        latency="low",
    ) as stream:
        while not stop_flag.is_set():
            block, _ = stream.read(blocksize)
            chunk = block[:, 0]
            audio_chunks.append(chunk)
    if audio_chunks:
        return np.concatenate(audio_chunks).astype(np.int16)
    else:
        return np.array(audio_chunks)
