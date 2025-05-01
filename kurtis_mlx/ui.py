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

# from kurtis_mlx import config
# from kurtis_mlx.workers.tts import tts_worker
# from kurtis_mlx.workers.sound import sd_worker


SAMPLE_RATE = 44100
RECORD_DURATION = 5  # seconds
AUDIO_FILE = "recorded_audio.wav"

recording = False
recorded_data = None
main_frame = None
app = None  # Global wx.App instance
tray_icon = None
show_ui = True  # Global variable to track ui state
show_developer_mode = False  # Global variable to track developer mode state
developer_mode_menu_item = None  # Global variable for the developer mode menu item
show_ui_menu_item = None  # Global variable for the show ui menu item
developer_mode_text = "Show Developer Mode"
show_ui_text = "Hide UI"


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
            wx.CallAfter(play_audio)  # Auto play after recording
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
        except sd.PortAudioError as e:
            print(f"Error during playback: {e}")
            if main_frame:
                wx.CallAfter(main_frame.update_playback_status, "Idle")
    else:
        if main_frame:
            wx.CallAfter(main_frame.update_playback_status, "No audio to play")


def on_show_ui_toggle(icon=None, item=None):
    global show_ui, main_frame, show_ui_menu_item, show_ui_text  # noqa
    show_ui = not show_ui
    if main_frame:
        wx.CallAfter(main_frame.toggle_ui_visibility, show_ui)
    if show_ui:
        show_ui_text = "Hide UI"  # Change menu item text
    else:
        show_ui_text = "Show UI"
    # Update the menu item in the tray
    tray_icon.update_menu()


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


def on_developer_mode_toggle(icon, item):
    global show_developer_mode, main_frame, developer_mode_menu_item, developer_mode_text  # noqa
    show_developer_mode = not show_developer_mode
    if main_frame:
        wx.CallAfter(main_frame.toggle_developer_mode_visibility, show_developer_mode)
    if show_developer_mode:
        developer_mode_text = "Hide Developer Mode"  # Change menu item text
    else:
        developer_mode_text = "Show Developer Mode"
    # Update the menu item in the tray
    tray_icon.update_menu()


# Create the menu *after* defining the functions.
if not os.path.exists("audio_icon.png"):
    image = Image.new("RGB", (64, 64), color="black")
    image.save("audio_icon.png")
else:
    image = Image.open("audio_icon.png")

developer_mode_menu_item = pystray.MenuItem(
    lambda text: developer_mode_text, on_developer_mode_toggle
)
show_ui_menu_item = pystray.MenuItem(lambda text: show_ui_text, on_show_ui_toggle)
menu = pystray.Menu(
    show_ui_menu_item,
    pystray.MenuItem("Ask Kurtis", on_record_tray),
    developer_mode_menu_item,
    pystray.MenuItem("Quit", on_quit_tray),
)


def create_ui():
    global main_frame, app
    if not app:
        app = wx.App(redirect=False)
    main_frame = MainFrame(None, "Kurtis E1")
    return main_frame


class MainFrame(wx.Frame):
    def __init__(self, parent, title):
        super().__init__(parent, title=title, style=wx.DEFAULT_FRAME_STYLE)
        # Removed the creation of self.panel
        self.sizer = wx.BoxSizer(wx.VERTICAL)

        self.record_button = wx.Button(self, label="Ask Kurtis")  # Parent is now self

        self.recording_status_label = wx.StaticText(
            self,
            label="Recording Status: Idle",  # Parent is now self
        )
        self.playback_status_label = wx.StaticText(
            self,
            label="Playback Status: Idle",  # Parent is now self
        )

        # Developer mode section
        self.developer_mode_section = wx.Panel(self)  # Parent is now self
        self.developer_mode_sizer = wx.BoxSizer(wx.VERTICAL)  # Use a new sizer
        self.developer_mode_section.SetSizer(
            self.developer_mode_sizer
        )  # Set sizer for the panel
        self.developer_mode_section.Show(
            show_developer_mode
        )  # Initially hide/show based on global state

        self.spectrogram_label = wx.StaticText(
            self.developer_mode_section, label="Spectrogram and Waveform"
        )  # Label for spectrogram

        # Spectrogram and Waveform setup
        self.figure = Figure(figsize=(6, 4), dpi=80)
        self.canvas = FigureCanvas(
            self.developer_mode_section, -1, self.figure
        )  # Use developer_mode_section as parent
        self.ax_spectrogram = self.figure.add_subplot(
            211
        )  # Top subplot for spectrogram
        self.ax_spectrogram.set_xlabel("Time (s)")
        self.ax_spectrogram.set_ylabel("Frequency (Hz)")
        self.spectrogram_data = None  # To store the spectrogram data
        self.spectrogram_plot = None

        self.ax_waveform = self.figure.add_subplot(212)  # Bottom subplot for waveform
        self.ax_waveform.set_xlabel("Time (s)")
        self.ax_waveform.set_ylabel("Amplitude")
        self.waveform_data = None
        self.waveform_plot = None

        self.developer_mode_sizer.Add(self.spectrogram_label, 0, wx.ALL, 5)  # Add label
        self.developer_mode_sizer.Add(
            self.canvas, 1, wx.EXPAND | wx.ALL, 5
        )  # Add the canvas to thesizer

        self.sizer.Add(self.record_button, 0, wx.ALL | wx.EXPAND, 5)
        self.sizer.Add(self.recording_status_label, 0, wx.ALL, 5)
        self.sizer.Add(self.playback_status_label, 0, wx.ALL, 5)
        self.sizer.Add(
            self.developer_mode_section, 1, wx.EXPAND | wx.ALL, 5
        )  # Add dev mode section

        # self.panel.SetSizer(self.sizer)  # Removed
        self.SetSizer(self.sizer)  # Set the sizer for the frame.
        self.sizer.Fit(self)
        self.SetInitialSize((640, -1))  # Set fixed width, and initial height
        self.Layout()  # Add this line

        self.record_button.Bind(wx.EVT_BUTTON, self.on_record_button)
        self.Bind(wx.EVT_CLOSE, self.on_close)
        self.is_always_on_top = False

    def on_record_button(self, event):
        if not recording:
            threading.Thread(target=record_audio).start()
        else:
            wx.MessageBox(
                "Already recording.", "Warning", wx.OK | wx.ICON_WARNING, self
            )

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

    def toggle_ui_visibility(self, show):
        def show_actions():
            if not main_frame.IsShown():
                main_frame.Show()
            main_frame.MakeAlwaysOnTop(True)
            main_frame.Raise()

        if show:
            wx.CallAfter(show_actions)
        else:
            wx.CallAfter(main_frame.Hide)

    def toggle_developer_mode_visibility(self, show):
        self.developer_mode_section.Show(show)
        if show:
            self.sizer.Show(self.developer_mode_section)
        else:
            self.sizer.Hide(self.developer_mode_section)
        self.sizer.Layout()
        self.Fit()  # tell the sizer to fit
        self.Layout()
        self.Fit()


if __name__ == "__main__":
    app = wx.App(redirect=False)
    main_frame = create_ui()
    main_frame.Show()

    icon = pystray.Icon("Audio App", image, "Kurtis E1", menu)
    tray_icon = icon

    # Run the tray icon in a separate thread
    def run_tray():
        icon.run()

    tray_thread = threading.Thread(target=run_tray, daemon=True)
    tray_thread.start()

    # Run the wx main loop
    app.MainLoop()
