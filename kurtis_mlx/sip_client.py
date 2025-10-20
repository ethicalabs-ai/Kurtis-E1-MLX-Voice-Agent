import time
import socket
import threading
import numpy as np
from rich.console import Console
from pyVoIP.VoIP import VoIPPhone, InvalidStateError, CallState

console = Console()

SAMPLE_RATE = 8000  # G.711 uses an 8kHz sample rate


def get_local_ip():
    """Gets the local IP address of the machine."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't even have to be reachable
        s.connect(("10.255.255.255", 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = "127.0.0.1"
    finally:
        s.close()
    return IP


class SipClient:
    """
    Main SIP client using pyVoIP.
    Handles registration, incoming calls, and media bridging using I/O threads.
    """

    def __init__(self, server, user, password, port, queues):
        self.queues = queues
        self.active_call = None
        self.phone = None
        self.reading_thread = None
        self.writing_thread = None
        self.monitor_thread = None

        # Store connection details to initialize the phone in the run method
        self._server = server
        self._port = port
        self._user = user
        self._password = password

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
            self.writing_thread = threading.Thread(target=self._write_loop, args=(call,))
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
                break
            time.sleep(0.5)

    def _read_loop(self, call):
        """Reads audio from the call and puts it into the transcription queue."""
        while self.active_call == call:
            try:
                # read_audio() provides 16-bit linear PCM audio as bytes
                audio_data = call.read_audio()
                if audio_data:
                    # Convert raw bytes to a numpy array for the STT worker
                    pcm_data = np.frombuffer(audio_data, dtype=np.int16)
                    self.queues["transcription"].put(pcm_data)
            except InvalidStateError:
                console.print("[SIP] Read loop ending, call state invalid.")
                break
            except Exception as e:
                console.print(f"[bold red][SIP Read Error] {e}[/bold red]")
                break

    def _write_loop(self, call):
        """Gets audio from the playback queue and writes it to the call."""
        while self.active_call == call:
            try:
                # The TTS worker produces a list of floats
                audio_list = self.queues["playback"].get()
                if audio_list is not None:
                    # Convert float audio to 16-bit PCM for the library
                    audio_np_float = np.asarray(audio_list, dtype=np.float32)
                    audio_np_int16 = (audio_np_float * 32767).astype(np.int16)
                    call.write_audio(audio_np_int16.tobytes())
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
            console.print(f"[bold red][SIP Client Error] An unexpected error occurred: {e}[/bold red]")
        finally:
            if self.phone:
                self.phone.stop()
