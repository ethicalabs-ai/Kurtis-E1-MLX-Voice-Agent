import time
import socket
import threading
import numpy as np
import audioop
from rich.console import Console
from pyVoIP.VoIP import VoIPPhone, InvalidStateError, CallState

from kurtis_mlx import config
from kurtis_mlx.utils.vad import VADCollector


TARGET_SAMPLE_RATE = 8000  # G.711 uses 8kHz.
VAD_FRAME_MS = config.VAD_FRAME_MS  # 30ms
VAD_BLOCK_SAMPLES = int(
    TARGET_SAMPLE_RATE * (VAD_FRAME_MS / 1000.0)
)  # 8000 * 0.030 = 240 samples


console = Console()


def get_local_ip():
    """Gets the local IP address of the machine."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't even have to be reachable
        s.connect(("10.255.255.255", 1))
        IP = s.getsockname()[0]
    except Exception:
        print("Using 127.0.0.1 as local ip.")
        IP = "127.0.0.1"
    finally:
        s.close()
    return IP


class SipClient:
    """
    Main SIP client using pyVoIP.
    Handles registration, incoming calls, and media bridging using I/O threads.
    """

    def __init__(
        self,
        server,
        user,
        password,
        port,
        queues,
        interrupt_event,
        assistant_prompt_au=None,
        debug=False,
    ):
        self.queues = queues
        self.interrupt_event = interrupt_event
        self.active_call = None
        self.phone = None
        self.reading_thread = None
        self.writing_thread = None
        self.monitor_thread = None
        self.assistant_prompt_au = assistant_prompt_au
        self.debug = debug

        # Store connection details to initialize the phone in the run method
        self._server = server
        self._port = port
        self._user = user
        self._password = password
        # self.playback_timestamps = collections.deque() # Removed for interruption support
        # self.playback_lock = threading.Lock() # Removed
        # self.EXCLUSION_WINDOW = 2.0  # Removed

    def handle_incoming_call(self, call):
        if self.active_call:
            console.print("[SIP] Busy: Rejecting incoming call.")
            try:
                call.hangup()
            except InvalidStateError:
                pass
            return

        from_header = call.request.headers.get("From", "Unknown Caller")
        console.print(f"[SIP] Incoming call from: {from_header}")
        self.active_call = call

        try:
            call.answer()
            console.print("[SIP] Call answered.")

            # Start I/O threads and state monitor
            self.reading_thread = threading.Thread(target=self._read_loop, args=(call,))
            self.writing_thread = threading.Thread(
                target=self._write_loop, args=(call,)
            )
            self.monitor_thread = threading.Thread(
                target=self._monitor_call_state, args=(call,)
            )
            self.reading_thread.daemon = True
            self.writing_thread.daemon = True
            self.monitor_thread.daemon = True
            self.reading_thread.start()
            self.writing_thread.start()
            self.monitor_thread.start()

            # Play initial message
            if self.assistant_prompt_au is not None:
                self.queues["playback"].put(self.assistant_prompt_au)

        except InvalidStateError as e:
            console.print(f"[bold red][SIP] Error answering call: {e}[/bold red]")
            self.active_call = None

    def _monitor_call_state(self, call):
        """Monitors the call state in a separate thread and handles cleanup."""
        while self.active_call == call:
            if call.state == CallState.ENDED:
                console.print("[SIP] Call terminated.")
                self.active_call = None
                # Signal the worker that the call has ended by putting None in the queue.
                # The I/O threads will see active_call is None and terminate.
                self.queues["transcription"].put(None)
                self.queues["playback"].put(None)
                break
            time.sleep(0.5)

    def _read_loop(self, call):
        """
        Reads 8-bit unsigned PCM audio, converts it to 16-bit signed PCM,
        and puts complete 16-bit utterances into the queue using VAD.
        """
        # Initialize the VADCollector
        vad_collector = VADCollector(
            sample_rate=config.SIP_SAMPLE_RATE,  # 8000
            aggressiveness=config.VAD_AGGRESSIVENESS,  # Use config value (default 3)
            frame_ms=VAD_FRAME_MS,  # from config
            silence_ms=config.SILENCE_FRAMES_THRESHOLD
            * VAD_FRAME_MS,  # e.g. 30 * 30 = 900ms
            min_speech_ms=2000,  # 2 seconds, matches old logic
            debug=self.debug,
        )
        console.print("[VAD] Listening for speech...")

        while self.active_call == call:
            try:
                # Removed exclusion window logic to allow interruption

                # Normal audio processing
                pcm_8_unsigned_bytes = call.read_audio()
                if not pcm_8_unsigned_bytes:
                    continue

                # Log when we're actually processing audio
                if self.debug:
                    console.print("[DEBUG] Processing audio")

                # Convert 8-bit unsigned (0 to 255) to 8-bit signed (-128 to 127)
                # '1' is the width (8-bit)
                pcm_8_signed_bytes = audioop.bias(pcm_8_unsigned_bytes, 1, -128)

                # Convert 8-bit signed to 16-bit signed
                # '1' is input width, '2' is output width
                pcm_16_signed_bytes = audioop.lin2lin(pcm_8_signed_bytes, 1, 2)

                for utterance in vad_collector.process_audio(pcm_16_signed_bytes):
                    if utterance is not None:
                        console.print(
                            f"[VAD] Queuing {len(utterance)} audio samples for transcription."
                        )
                        self.queues["transcription"].put(utterance)

            except InvalidStateError:
                console.print("[SIP] Read loop ending, call state invalid.")
                break
            except Exception as e:
                console.print(f"[bold red][SIP Read Error] {e}[/bold red]")
                break

    def _write_loop(self, call):
        """
        Gets audio from the TTS, resamples it to 8kHz using librosa for high
        quality, converts it to 8-bit unsigned linear PCM, and writes it to the call.
        """
        while self.active_call == call:
            try:
                # Check for interruption
                if self.interrupt_event.is_set():
                    # If interrupted, we should probably clear the queue or just skip current playback
                    # But here we are waiting for get().
                    # If we are already playing, we need to stop.
                    # Since we process chunk by chunk (actually whole clips here),
                    # we can check before processing.
                    # To support immediate interruption of a long clip, we'd need to chunk it.
                    # For now, let's just check before processing.
                    # Ideally, we should chunk the playback like in sd_worker.

                    # Clear the queue
                    while not self.queues["playback"].empty():
                        try:
                            self.queues["playback"].get_nowait()
                        except Exception:
                            break
                    pass

                # 1. Get the float audio list from TTS worker
                try:
                    audio_list = self.queues["playback"].get(timeout=0.1)
                except Exception:  # Empty
                    continue

                if audio_list is None:
                    continue

                # Check interrupt again
                if self.interrupt_event.is_set():
                    self.interrupt_event.clear()
                    console.print("[SIP] Playback skipped due to interruption.")
                    continue

                # 2. Convert to a numpy float32 array.
                audio_np_float = np.asarray(audio_list, dtype=np.float32)

                # 3. Clip the audio to the valid [-1.0, 1.0] range to prevent distortion.
                np.clip(audio_np_float, -1.0, 1.0, out=audio_np_float)

                # 4. Scale and shift the float signal directly to the 8-bit unsigned range [0, 255].
                audio_np_uint8 = ((audio_np_float * 127.5) + 127.5).astype(np.uint8)

                # 5. Convert the numpy array to raw bytes.
                pcm_8_unsigned_bytes = audio_np_uint8.tobytes()

                console.print(
                    f"[SIP] Streaming {len(pcm_8_unsigned_bytes)} bytes of audio..."
                )

                # Chunking for interruption support
                CHUNK_SIZE = (
                    160 * 5
                )  # 100ms chunks (8000Hz * 0.1s * 1 byte) = 800 bytes
                # Actually 160 samples is 20ms.

                for i in range(0, len(pcm_8_unsigned_bytes), CHUNK_SIZE):
                    if self.interrupt_event.is_set():
                        console.print("[SIP] Playback interrupted.")
                        self.interrupt_event.clear()
                        break

                    chunk = pcm_8_unsigned_bytes[i : i + CHUNK_SIZE]
                    call.write_audio(chunk)
                    # Small sleep to match timing? write_audio might be blocking or fast.
                    # pyVoIP write_audio usually puts into a buffer.
                    # If we write too fast, we fill the buffer.
                    # We should pace it. 8000 bytes per second.
                    # If chunk is 800 bytes, it's 0.1s.
                    time.sleep(len(chunk) / 8000.0)

                console.print("[SIP] Finished streaming audio.")

            except InvalidStateError:
                console.print("[SIP] Write loop ending, call state invalid.")
                break
            except Exception as e:
                console.print(f"[bold red][SIP Write Error] {e}[/bold red]")
                break

    def run(self):
        """Initializes and starts the VoIP phone client."""
        local_ip = get_local_ip()
        self.phone = VoIPPhone(
            self._server,
            self._port,
            self._user,
            self._password,
            callCallback=self.handle_incoming_call,
            myIP=local_ip,
        )
        try:
            console.print("[SIP] Starting SIP client...")
            self.phone.start()
            console.print("[SIP] SIP client running. Press Ctrl+C to exit.")
            # Keep the main thread alive while the phone's threads run
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            console.print("[SIP] Stopping SIP client...")
        except Exception as e:
            console.print(
                f"[bold red][SIP Client Error] An unexpected error occurred: {e}[/bold red]"
            )
        finally:
            if self.phone:
                self.phone.stop()
