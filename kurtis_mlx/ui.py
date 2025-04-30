import pystray
from PIL import Image
import sounddevice as sd
import numpy as np
import threading
import os
import wx
import matplotlib

matplotlib.use("WXAgg")  # Use the WXAgg backend for wxPython integration
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas

SAMPLE_RATE = 44100
RECORD_DURATION = 5  # seconds
AUDIO_FILE = "recorded_audio.wav"

recording = False
recorded_data = None
main_frame = None
app = None  # Global wx.App instance
tray_icon = None


def record_audio():
    global recording, recorded_data, main_frame
    recording = True
    print("Recording started...")
    if main_frame:
        wx.CallAfter(main_frame.update_recording_status, "Recording...")
    try:
        recorded_data = sd.rec(
            int(RECORD_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1
        )
        sd.wait()  # Wait until recording is finished
        print("Recording finished.")
        recording = False
        if main_frame:
            wx.CallAfter(main_frame.update_recording_status, "Idle")
            wx.CallAfter(main_frame.update_spectrogram, recorded_data)  # update
            wx.CallAfter(main_frame.update_waveform, recorded_data)
    except sd.PortAudioError as e:
        print(f"Error during recording: {e}")
        if main_frame:
            wx.CallAfter(main_frame.update_recording_status, f"Error: {e}")


def play_audio():
    global recorded_data, main_frame
    if recorded_data is not None:
        print("Playing recorded audio...")
        if main_frame:
            wx.CallAfter(main_frame.update_playback_status, "Playing...")
        try:
            sd.play(recorded_data, SAMPLE_RATE)
            sd.wait()  # Wait until playback is finished
            print("Playback finished.")
            if main_frame:
                wx.CallAfter(main_frame.update_playback_status, "Idle")
                wx.CallAfter(main_frame.update_spectrogram, recorded_data)  # update
                wx.CallAfter(main_frame.update_waveform, recorded_data)
        except sd.PortAudioError as e:
            print(f"Error during playback: {e}")
            if main_frame:
                wx.CallAfter(main_frame.update_playback_status, "Idle")
    else:
        if main_frame:
            wx.CallAfter(main_frame.update_playback_status, "No audio to play")


def on_show_ui(icon=None, item=None):
    global main_frame
    if main_frame:

        def show_actions():
            if not main_frame.IsShown():
                main_frame.Show()
            main_frame.MakeAlwaysOnTop(True)
            main_frame.Raise()

        wx.CallAfter(show_actions)


def on_hide_ui(icon=None, item=None):
    global main_frame
    if main_frame:
        wx.CallAfter(main_frame.Hide)


def on_quit_tray(icon=None, item=None):
    global main_frame, app, tray_icon
    if tray_icon:
        tray_icon.stop()
    if main_frame:
        wx.CallAfter(main_frame.Destroy)
    if app:
        wx.CallAfter(app.ExitMainLoop)


def on_record_tray(icon, item):
    if not recording:
        threading.Thread(target=record_audio).start()
    else:
        print("Already recording.")


def on_play_tray(icon, item):
    threading.Thread(target=play_audio).start()


# Create the menu *after* defining the functions.
if not os.path.exists("audio_icon.png"):
    image = Image.new("RGB", (64, 64), color="black")
    image.save("audio_icon.png")
else:
    image = Image.open("audio_icon.png")

menu = pystray.Menu(
    pystray.MenuItem("Show UI", on_show_ui),
    pystray.MenuItem("Hide UI", on_hide_ui),
    pystray.MenuItem("Record", on_record_tray),
    pystray.MenuItem("Play", on_play_tray),
    pystray.MenuItem("Quit", on_quit_tray),
)


def create_ui():
    global main_frame, app
    if not app:
        app = wx.App(redirect=False)
    main_frame = MainFrame(None, "Audio Recorder/Player")
    return main_frame


class MainFrame(wx.Frame):
    def __init__(self, parent, title):
        super().__init__(parent, title=title, style=wx.DEFAULT_FRAME_STYLE)
        self.panel = wx.Panel(self)
        self.sizer = wx.BoxSizer(wx.VERTICAL)

        self.record_button = wx.Button(self.panel, label="Record")
        self.play_button = wx.Button(self.panel, label="Play")
        self.hide_button = wx.Button(self.panel, label="Hide UI")

        self.recording_status_label = wx.StaticText(
            self.panel, label="Recording Status: Idle"
        )
        self.playback_status_label = wx.StaticText(
            self.panel, label="Playback Status: Idle"
        )

        # Spectrogram setup
        self.figure = Figure(figsize=(6, 4), dpi=80)
        self.canvas = FigureCanvas(self.panel, -1, self.figure)
        self.ax_spectrogram = self.figure.add_subplot(
            211
        )  # Top subplot for spectrogram
        self.ax_spectrogram.set_xlabel("Time (s)")
        self.ax_spectrogram.set_ylabel("Frequency (Hz)")
        self.spectrogram_data = None  # To store the spectrogram data
        self.spectrogram_plot = None

        # Waveform setup
        self.ax_waveform = self.figure.add_subplot(212)  # Bottom subplot for waveform
        self.ax_waveform.set_xlabel("Time (s)")
        self.ax_waveform.set_ylabel("Amplitude")
        self.waveform_data = None
        self.waveform_plot = None

        self.sizer.Add(self.record_button, 0, wx.ALL | wx.EXPAND, 5)
        self.sizer.Add(self.play_button, 0, wx.ALL | wx.EXPAND, 5)
        self.sizer.Add(self.recording_status_label, 0, wx.ALL, 5)
        self.sizer.Add(self.playback_status_label, 0, wx.ALL, 5)
        self.sizer.Add(
            self.canvas, 1, wx.EXPAND | wx.ALL, 5
        )  # Add the canvas to the sizer
        self.sizer.Add(self.hide_button, 0, wx.ALL | wx.EXPAND, 5)

        self.panel.SetSizer(self.sizer)
        self.sizer.Fit(self)

        self.record_button.Bind(wx.EVT_BUTTON, self.on_record_button)
        self.play_button.Bind(wx.EVT_BUTTON, self.on_play_button)
        self.hide_button.Bind(wx.EVT_BUTTON, self.on_hide_button)
        self.Bind(wx.EVT_CLOSE, self.on_close)
        self.is_always_on_top = False

    def on_record_button(self, event):
        if not recording:
            threading.Thread(target=record_audio).start()
        else:
            wx.MessageBox(
                "Already recording.", "Warning", wx.OK | wx.ICON_WARNING, self
            )

    def on_play_button(self, event):
        threading.Thread(target=play_audio).start()

    def on_hide_button(self, event):
        self.Hide()

    def on_close(self, event):
        # Override the close event.
        self.Hide()  # Just hide.
        event.Skip()

    def update_recording_status(self, status_text):
        self.recording_status_label.SetLabel(f"Recording Status: {status_text}")

    def update_playback_status(self, status_text):
        self.playback_status_label.SetLabel(f"Playback Status: {status_text}")

    def MakeAlwaysOnTop(self, alwaysOnTop):
        self.SetWindowStyleFlag(
            wx.STAY_ON_TOP if alwaysOnTop else wx.DEFAULT_FRAME_STYLE
        )
        self.is_always_on_top = alwaysOnTop
        self.Refresh()

    def update_spectrogram(self, audio_data):
        if audio_data is not None:
            # Calculate spectrogram
            NFFT = 256  # Length of the FFT window
            Fs = SAMPLE_RATE
            noverlap = 128
            try:
                self.ax_spectrogram.clear()
                Pxx, freqs, bins, im = self.ax_spectrogram.specgram(
                    audio_data.flatten(), Fs=Fs, NFFT=NFFT, noverlap=noverlap
                )
                self.ax_spectrogram.set_xlabel("Time (s)")
                self.ax_spectrogram.set_ylabel("Frequency (Hz)")
                self.canvas.draw()
            except Exception as e:
                print(f"Error updating spectrogram: {e}")
        else:
            self.ax_spectrogram.clear()
            self.ax_spectrogram.set_xlabel("Time (s)")
            self.ax_spectrogram.set_ylabel("Frequency (Hz)")
            self.canvas.draw()

    def update_waveform(self, audio_data):
        if audio_data is not None:
            time_axis = np.arange(0, len(audio_data)) / SAMPLE_RATE
            self.ax_waveform.clear()
            self.ax_waveform.plot(time_axis, audio_data)
            self.ax_waveform.set_xlabel("Time (s)")
            self.ax_waveform.set_ylabel("Amplitude")
            self.canvas.draw()
        else:
            self.ax_waveform.clear()
            self.ax_waveform.set_xlabel("Time (s)")
            self.ax_waveform.set_ylabel("Amplitude")
            self.canvas.draw()


if __name__ == "__main__":
    app = wx.App(redirect=False)
    main_frame = create_ui()
    main_frame.Show()

    icon = pystray.Icon("Audio App", image, "Simple Audio Recorder/Player", menu)
    tray_icon = icon

    # Run the tray icon in a separate thread
    def run_tray():
        icon.run()

    tray_thread = threading.Thread(target=run_tray, daemon=True)
    tray_thread.start()

    # Run the wx main loop
    app.MainLoop()
