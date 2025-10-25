import time
import socket
import threading
import numpy as np
import collections
import audioop
from rich.console import Console
from pyVoIP.VoIP import VoIPPhone, InvalidStateError, CallState

from kurtis_mlx import config
from kurtis_mlx.utils.vad import VADCollector


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

    def __init__(self, server, user, password, port, queues, debug=False):
        self.queues = queues
        self.active_call = None
        self.phone = None
        self.reading_thread = None
        self.writing_thread = None
        self.monitor_thread = None
        self.debug = debug

        # Store connection details to initialize the phone in the run method
        self._server = server
        self._port = port
        self._user = user
        self._password = password
        self.playback_timestamps = collections.deque()
        self.playback_lock = threading.Lock()
        self.EXCLUSION_WINDOW = 2.0  # 2-second exclusion window

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
            sample_rate=config.SIP_SAMPLE_RATE,
            aggressiveness=config.VAD_AGGRESSIVENESS,
            frame_ms=config.VAD_FRAME_MS,
            silence_ms=config.SILENCE_FRAMES_THRESHOLD
            * config.VAD_FRAME_MS,  # e.g. 30 * 30 = 900ms
            min_speech_ms=2000,  # 2s
            debug=self.debug,
        )
        console.print("[VAD] Listening for speech...")

        while self.active_call == call:
            try:
                current_time = time.time()
                with self.playback_lock:
                    # Remove old timestamps
                    while (
                        self.playback_timestamps
                        and current_time - self.playback_timestamps[0]
                        > self.EXCLUSION_WINDOW
                    ):
                        old_ts = self.playback_timestamps.popleft()
                        if self.debug:
                            console.print(
                                f"[DEBUG] Removed old timestamp: {current_time - old_ts:.2f}s ago"
                            )

                    # Check if we're in exclusion window
                    is_excluded = bool(self.playback_timestamps)
                    if is_excluded:
                        latest_playback = self.playback_timestamps[-1]
                        time_since_playback = current_time - latest_playback
                        if self.debug:
                            console.print(
                                f"[DEBUG] Exclusion check: {time_since_playback:.2f}s since playback, threshold: {self.EXCLUSION_WINDOW}s"
                            )

                if is_excluded:
                    # Read and discard audio to keep buffer clear
                    discarded_audio = call.read_audio()
                    if discarded_audio:
                        self.debug_counter += 1
                        if (
                            self.debug_counter % 50 == 0 and self.debug
                        ):  # Log every 50 discards
                            console.print(
                                f"[DEBUG] Discarding audio during exclusion (count: {self.debug_counter})"
                            )
                    time.sleep(0.01)
                    continue
                else:
                    self.debug_counter = 0  # Reset counter when not excluding

                # Normal audio processing
                pcm_8_unsigned_bytes = call.read_audio()
                if not pcm_8_unsigned_bytes:
                    continue

                # Log when we're actually processing audio
                if self.debug:
                    console.print("[DEBUG] Processing audio (not in exclusion window)")

                # Convert 8-bit unsigned (0 to 255) to 8-bit signed (-128 to 127)
                # '1' is the width (8-bit)
                pcm_8_signed_bytes = audioop.bias(pcm_8_unsigned_bytes, 1, -128)

                # Convert 8-bit signed to 16-bit signed
                # '1' is input width, '2' is output width
                pcm_16_signed_bytes = audioop.lin2lin(pcm_8_signed_bytes, 1, 2)

                for utterance in vad_collector.process_audio(pcm_16_signed_bytes):
                    if utterance is not None:
                        # This is the same logic as [cite: 70, 71]
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
                # 1. Get the float audio list from TTS worker
                audio_list = self.queues["playback"].get()
                if audio_list is not None:
                    playback_start = time.time()  # Record playback start time
                    with self.playback_lock:
                        self.playback_timestamps.append(playback_start)
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
                    call.write_audio(pcm_8_unsigned_bytes)
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
