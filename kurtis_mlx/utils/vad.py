import webrtcvad
import collections
import numpy as np
from rich.console import Console

console = Console()


class VADCollector:
    """
    Collects audio frames and yields complete speech utterances using WebRTCVAD.

    This class is stateful and processes audio in 16-bit signed PCM chunks.
    """

    def __init__(
        self,
        sample_rate: int,
        aggressiveness: int,
        frame_ms: int = 30,
        silence_ms: int = 900,
        min_speech_ms: int = 2000,
        debug: bool = False,
    ):
        """
        Initializes the VADCollector.

        Args:
            sample_rate (int): The audio sample rate (e.g., 8000 for SIP, 22050 for local).
                               Must be one of 8000, 16000, 32000, 48000.
            aggressiveness (int): VAD aggressiveness (0 to 3).
            frame_ms (int): Duration of each VAD frame in ms (10, 20, or 30).
            silence_ms (int): How long to wait for silence before ending an utterance.
            min_speech_ms (int): Minimum duration of speech to be considered valid.
            debug (bool): Print debug messages.
        """
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.debug = debug

        if sample_rate not in [8000, 16000, 32000, 48000]:
            console.print(
                f"[VAD Warning] Invalid sample rate {sample_rate}. VAD may not function correctly."
            )

        self.vad = webrtcvad.Vad(aggressiveness)

        # Calculate frame sizes based on config
        self.frame_samples = int(sample_rate * (frame_ms / 1000.0))
        self.frame_bytes = self.frame_samples * 2  # 16-bit PCM

        # Calculate thresholds
        self.silence_frames_threshold = int(
            silence_ms / frame_ms
        )  # e.g., 900ms / 30ms = 30 frames
        self.min_speech_samples = int(
            sample_rate * (min_speech_ms / 1000.0)
        )  # e.g., 8000 * 2.0 = 16000 samples

        if self.debug:
            console.print(f"[VAD Init] Sample Rate: {self.sample_rate}Hz")
            console.print(
                f"[VAD Init] Frame Size: {self.frame_samples} samples ({self.frame_bytes} bytes)"
            )
            console.print(
                f"[VAD Init] Silence Threshold: {self.silence_frames_threshold} frames (~{silence_ms}ms)"
            )
            console.print(
                f"[VAD Init] Min Speech: {self.min_speech_samples} samples (~{min_speech_ms}ms)"
            )

        # State variables, as seen in sip_client.py
        self.audio_buffer = bytearray()
        self.speech_frames = collections.deque()
        self.triggered = False
        self.silence_frames = 0

    def reset(self):
        """Resets the internal state of the VAD."""
        if self.debug:
            console.print("[VAD] State reset.")
        self.audio_buffer.clear()
        self.speech_frames.clear()
        self.triggered = False
        self.silence_frames = 0

    def process_audio(self, pcm_16_signed_bytes: bytes):
        """
        Processes a chunk of 16-bit signed PCM audio bytes and yields
        complete speech utterances as np.ndarray (dtype=np.int16).

        This is a generator function.
        """
        self.audio_buffer.extend(pcm_16_signed_bytes)

        # Process audio in VAD-required frame sizes
        while len(self.audio_buffer) >= self.frame_bytes:
            frame = self.audio_buffer[: self.frame_bytes]
            del self.audio_buffer[: self.frame_bytes]

            try:
                is_speech = self.vad.is_speech(frame, self.sample_rate)
            except Exception as e:
                console.print(f"[VAD Error] {e} - skipping frame.")
                continue

            if self.triggered:
                # We are in a speech segment
                self.speech_frames.append(frame)
                if not is_speech:
                    self.silence_frames += 1
                    if self.silence_frames > self.silence_frames_threshold:
                        # End of speech detected
                        if self.debug:
                            console.print("[VAD] End of speech detected (silence).")

                        # Yield the utterance
                        utterance = self._yield_utterance()
                        if utterance is not None:
                            yield utterance

                        # Reset for next utterance
                        self.reset()
                else:
                    # Still speech, reset silence counter
                    self.silence_frames = 0
            else:
                # We are not in a speech segment
                if is_speech:
                    # Start of speech detected
                    if self.debug:
                        console.print("[VAD] Start of speech detected.")
                    self.triggered = True
                    self.speech_frames.append(frame)
                    self.silence_frames = 0

    def flush(self):
        """
        Flushes any remaining audio in the buffer as a final utterance,
        regardless of silence.

        Returns:
            np.ndarray (dtype=np.int16) or None: The final utterance, or None if invalid.
        """
        if not self.speech_frames:
            self.reset()
            return None

        if self.debug:
            console.print("[VAD] Flushing remaining audio.")

        utterance = self._yield_utterance()
        self.reset()
        return utterance

    def _yield_utterance(self):
        """Helper to package and check the utterance length."""
        complete_speech_bytes = b"".join(self.speech_frames)
        pcm_data = np.frombuffer(complete_speech_bytes, dtype=np.int16)

        if len(pcm_data) > self.min_speech_samples:
            if self.debug:
                console.print(f"[VAD] Yielding {len(pcm_data)} audio samples.")
            return pcm_data
        else:
            if self.debug:
                console.print(
                    f"[VAD] Discarding short audio segment ({len(pcm_data)} samples)."
                )
            return None
