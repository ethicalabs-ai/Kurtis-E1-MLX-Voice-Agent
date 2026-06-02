"""
A minimal, standalone wxPython GUI for the Kurtis MLX Voice Agent.

This application provides a visual representation of microphone input through a
pulsating circle, inspired by modern voice assistant interfaces. It runs as a
separate process and does not integrate with the main Kurtis MLX agent logic,
using a placeholder `KurtisAgent` to simulate background activity.

The UI is designed to be minimal, responsive, and to demonstrate real-time
audio data processing and visualization without freezing the main application loop.
"""
import threading
import time
import queue
import colorsys

import numpy as np
import pyaudio
import sounddevice as sd
import wx

from kurtis_mlx.utils.settings import SettingsManager

# --- Constants ---
# Audio settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100  # Default rate, will be overridden by system

# UI settings
FRAME_SIZE = (400, 600)
BACKGROUND_COLOR = "#212121"
CIRCLE_COLOR = "#42a5f5"  # Light Blue
FPS = 60

# Circle animation parameters
BASE_RADIUS = 50
SCALING_FACTOR = 100  # Adjust this to control sensitivity to mic volume


class PulsatingCirclePanel(wx.Panel):
    """
    A custom wx.Panel that draws a circle whose radius pulsates based on
    an amplitude value. It uses a wx.GraphicsContext for anti-aliased drawing.
    """

    STATE_LISTENING = "listening"
    STATE_SPEAKING = "speaking"

    def __init__(self, parent):
        super().__init__(parent)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.SetBackgroundColour(BACKGROUND_COLOR)
        self.amplitude = 0.0
        self.base_radius = BASE_RADIUS
        self.scaling_factor = SCALING_FACTOR
        self.state = self.STATE_LISTENING
        self.hue_offset = 0.0

        self.Bind(wx.EVT_PAINT, self.on_paint)

    def set_amplitude(self, value):
        """Thread-safe method to update the audio amplitude."""
        self.amplitude = value

    def set_state(self, state):
        """Thread-safe method to update the agent state."""
        self.state = state

    def shift_color(self, color, offset):
        """Shifts the hue of a wx.Colour by a given offset (0.0 - 1.0)."""
        r, g, b = color.Red() / 255.0, color.Green() / 255.0, color.Blue() / 255.0
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        h = (h + offset) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return wx.Colour(int(r * 255), int(g * 255), int(b * 255))

    def on_paint(self, event):
        """Handles the paint event to draw the pulsating circle."""
        dc = wx.AutoBufferedPaintDC(self)
        dc.SetBackground(wx.Brush(BACKGROUND_COLOR))
        dc.Clear()

        gc = wx.GraphicsContext.Create(dc)
        if gc:
            width, height = self.GetSize()
            center_x, center_y = width // 2, height // 2

            # Update animation state
            self.hue_offset = (self.hue_offset + 0.005) % 1.0

            # Dynamic radius
            radius = self.base_radius + (self.amplitude * self.scaling_factor)

            # Determine base colors based on state
            if self.state == self.STATE_SPEAKING:
                # Speaking: Purple/Pink base
                base_c1 = wx.Colour(186, 104, 200)  # Medium Purple
                base_c2 = wx.Colour(255, 64, 129)  # Pink Accent
            else:
                # Listening: Cyan/Blue base
                base_c1 = wx.Colour(66, 165, 245)  # Blue 400
                base_c2 = wx.Colour(38, 198, 218)  # Cyan 400

            # Apply hue shift
            color1 = self.shift_color(base_c1, self.hue_offset)
            color2 = self.shift_color(base_c2, self.hue_offset)

            # Create radial gradient
            brush = gc.CreateRadialGradientBrush(
                center_x, center_y, center_x, center_y, radius, color1, color2
            )

            gc.SetBrush(brush)
            gc.SetPen(wx.TRANSPARENT_PEN)

            # Draw the circle
            gc.DrawEllipse(center_x - radius, center_y - radius, 2 * radius, 2 * radius)


class SettingsDialog(wx.Dialog):
    """
    A dialog for configuring application settings.
    """

    def __init__(
        self,
        parent,
        current_device_index,
        current_channel_index,
        current_api_url,
        current_api_key,
        current_llm_model,
        device_list,
        device_indices,
    ):
        super().__init__(parent, title="Settings", size=(400, 600))

        self.device_indices = device_indices
        self.device_list = device_list
        self.selected_device_index = current_device_index
        self.selected_channel_index = current_channel_index
        self.api_url = current_api_url
        self.api_key = current_api_key
        self.llm_model = current_llm_model

        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        # Audio Device Selection
        lbl_device = wx.StaticText(panel, label="Audio Input Device:")
        vbox.Add(lbl_device, flag=wx.LEFT | wx.TOP, border=10)

        self.device_choice = wx.Choice(panel, choices=self.device_list)
        # Find selection index based on device index
        try:
            selection = self.device_indices.index(self.selected_device_index)
            self.device_choice.SetSelection(selection)
        except ValueError:
            if self.device_list:
                self.device_choice.SetSelection(0)

        self.device_choice.Bind(wx.EVT_CHOICE, self.on_device_change)
        vbox.Add(self.device_choice, flag=wx.EXPAND | wx.LEFT | wx.RIGHT, border=10)

        # Audio Channel Selection
        lbl_channel = wx.StaticText(panel, label="Audio Channel:")
        vbox.Add(lbl_channel, flag=wx.LEFT | wx.TOP, border=10)

        self.channel_choice = wx.Choice(panel, choices=[])
        self.populate_channels()
        if self.selected_channel_index is not None:
            # Ensure index is within bounds
            if self.selected_channel_index < self.channel_choice.GetCount():
                self.channel_choice.SetSelection(self.selected_channel_index)
            else:
                self.channel_choice.SetSelection(0)

        vbox.Add(self.channel_choice, flag=wx.EXPAND | wx.LEFT | wx.RIGHT, border=10)

        # OpenAI API URL
        lbl_api = wx.StaticText(panel, label="OpenAI API URL:")
        vbox.Add(lbl_api, flag=wx.LEFT | wx.TOP, border=10)

        self.txt_api_url = wx.TextCtrl(panel, value=self.api_url)
        vbox.Add(self.txt_api_url, flag=wx.EXPAND | wx.LEFT | wx.RIGHT, border=10)

        # OpenAI API Key
        lbl_key = wx.StaticText(panel, label="OpenAI API Key:")
        vbox.Add(lbl_key, flag=wx.LEFT | wx.TOP, border=10)

        self.txt_api_key = wx.TextCtrl(panel, value=self.api_key, style=wx.TE_PASSWORD)
        vbox.Add(self.txt_api_key, flag=wx.EXPAND | wx.LEFT | wx.RIGHT, border=10)

        # LLM Model
        lbl_model = wx.StaticText(panel, label="LLM Model:")
        vbox.Add(lbl_model, flag=wx.LEFT | wx.TOP, border=10)

        self.txt_llm_model = wx.TextCtrl(panel, value=self.llm_model)
        vbox.Add(self.txt_llm_model, flag=wx.EXPAND | wx.LEFT | wx.RIGHT, border=10)

        vbox.AddStretchSpacer()

        # Buttons
        hbox_btns = wx.BoxSizer(wx.HORIZONTAL)
        btn_save = wx.Button(panel, label="Save")
        btn_save.SetMinSize((90, 35))  # Ensure minimum size to prevent GTK warning
        btn_save.Bind(wx.EVT_BUTTON, self.on_save)
        btn_cancel = wx.Button(panel, label="Cancel")
        btn_cancel.SetMinSize((90, 35))  # Ensure minimum size to prevent GTK warning
        btn_cancel.Bind(wx.EVT_BUTTON, self.on_cancel)

        hbox_btns.Add(btn_save, flag=wx.RIGHT | wx.TOP | wx.BOTTOM, border=5)
        hbox_btns.Add(btn_cancel, flag=wx.TOP | wx.BOTTOM, border=5)

        vbox.Add(hbox_btns, flag=wx.ALIGN_RIGHT | wx.ALL, border=10)

        panel.SetSizer(vbox)

    def populate_channels(self):
        selection = self.device_choice.GetSelection()
        if selection != wx.NOT_FOUND:
            device_index = self.device_indices[selection]
            try:
                device_info = sd.query_devices(device_index, "input")
                max_channels = device_info["max_input_channels"]
                channels = [f"Channel {i}" for i in range(max_channels)]
                self.channel_choice.Set(channels)
                self.channel_choice.SetSelection(0)
            except Exception as e:
                print(f"Error listing channels: {e}")

    def on_device_change(self, event):
        self.populate_channels()

    def on_save(self, event):
        selection = self.device_choice.GetSelection()
        if selection != wx.NOT_FOUND:
            self.selected_device_index = self.device_indices[selection]

        channel_sel = self.channel_choice.GetSelection()
        if channel_sel != wx.NOT_FOUND:
            self.selected_channel_index = channel_sel

        self.api_url = self.txt_api_url.GetValue()
        self.api_key = self.txt_api_key.GetValue()
        self.llm_model = self.txt_llm_model.GetValue()
        self.EndModal(wx.ID_OK)

    def on_cancel(self, event):
        self.EndModal(wx.ID_CANCEL)


class MainFrame(wx.Frame):
    """
    The main application window that houses the UI and manages background threads
    for audio input and the simulated agent.
    """

    def __init__(
        self, agent_run_function=None, audio_queue=None, control_queue=None, debug=False
    ):
        super().__init__(
            None,
            title="Kurtis MLX Voice Agent",
            size=FRAME_SIZE,
            style=wx.DEFAULT_FRAME_STYLE & ~(wx.RESIZE_BORDER | wx.MAXIMIZE_BOX),
        )

        self.panel = PulsatingCirclePanel(self)
        self.SetBackgroundColour(BACKGROUND_COLOR)

        # Settings Data
        self.device_list = []
        self.device_indices = []
        self.current_device_index = None
        self.current_channel_index = 0

        # Import config here to avoid circular imports if any, or just use the module
        from kurtis_mlx import config

        self.current_api_url = config.OPENAI_API_URL
        self.current_api_key = config.OPENAI_API_KEY
        self.current_llm_model = config.LLM_MODEL

        # Load persisted settings
        self.load_persisted_settings()

        self.populate_devices()

        # Apply persisted device if available and valid
        if self.current_device_index is not None:
            # We need to check if this index is still valid or if we need to find by name
            # For simplicity, we trust the index for now, but ideally we should match by name
            pass

        # Layout
        sizer = wx.BoxSizer(wx.VERTICAL)

        # Top bar for settings button
        top_sizer = wx.BoxSizer(wx.HORIZONTAL)
        top_sizer.AddStretchSpacer()

        # Gear Button
        # Using text "Settings" to avoid unicode rendering issues
        self.btn_settings = wx.Button(self.panel, label="Settings")
        self.btn_settings.SetMinSize(
            (90, 35)
        )  # Ensure minimum size to prevent GTK warning
        self.btn_settings.Bind(wx.EVT_BUTTON, self.on_settings_click)

        # Pause Button
        self.is_paused = False
        self.btn_pause = wx.Button(self.panel, label="Pause")
        self.btn_pause.SetMinSize(
            (90, 35)
        )  # Ensure minimum size to prevent GTK warning
        self.btn_pause.Bind(wx.EVT_BUTTON, self.on_pause_click)

        top_sizer.Add(self.btn_settings, 0, wx.LEFT | wx.RIGHT, 5)
        top_sizer.Add(self.btn_pause, 0, wx.LEFT | wx.RIGHT, 5)

        sizer.Add(top_sizer, 0)
        sizer.AddStretchSpacer()
        self.panel.SetSizer(sizer)

        self.Center()

        self.audio_thread = None
        self.agent_thread = None
        self.stop_event = threading.Event()
        self.control_queue = control_queue
        self.debug = debug

        self.p = pyaudio.PyAudio()
        self.stream = None
        self.audio_queue = audio_queue

        self.agent_run_function = agent_run_function
        self.start_threads()

        # Apply initial settings to workers
        self.apply_initial_settings()

        # A wx.Timer is used to trigger frequent redraws for smooth animation
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.on_timer, self.timer)
        self.timer.Start(1000 // FPS)

        self.Bind(wx.EVT_CLOSE, self.on_close)

    def load_persisted_settings(self):
        """Loads settings from YAML and updates local state and config."""
        settings = SettingsManager.load_settings()
        if not settings:
            return

        print(f"Loaded settings: {settings}")

        if "audio_device_index" in settings:
            self.current_device_index = settings["audio_device_index"]

        if "audio_channel_index" in settings:
            self.current_channel_index = settings["audio_channel_index"]

        if "openai_api_url" in settings:
            self.current_api_url = settings["openai_api_url"]
            from kurtis_mlx import config

            config.OPENAI_API_URL = self.current_api_url

        if "openai_api_key" in settings:
            self.current_api_key = settings["openai_api_key"]
            from kurtis_mlx import config

            config.OPENAI_API_KEY = self.current_api_key

        if "llm_model" in settings:
            self.current_llm_model = settings["llm_model"]
            from kurtis_mlx import config

            config.LLM_MODEL = self.current_llm_model

    def apply_initial_settings(self):
        """Sends initial settings to workers after threads start."""
        # We need to wait a bit for workers to be ready, or just send it.
        # The workers process the queue in their loop.
        if self.control_queue:
            if self.current_device_index is not None:
                self.control_queue.put(
                    {"action": "set_device", "device_index": self.current_device_index}
                )
            if self.current_channel_index is not None:
                self.control_queue.put(
                    {
                        "action": "set_channel",
                        "channel_index": self.current_channel_index,
                    }
                )

    def populate_devices(self):
        """Populates the device list for settings."""
        try:
            devices = sd.query_devices()
            self.device_list = []
            self.device_indices = []

            default_input = sd.default.device[0]

            # If we haven't loaded a persisted device, use default
            if self.current_device_index is None:
                if default_input is None:
                    default_input = 0
                self.current_device_index = default_input

            for i, device in enumerate(devices):
                # Filter for devices that have at least one input channel
                if device["max_input_channels"] > 0:
                    # Construct a descriptive name
                    name = f"{i}: {device['name']} (In: {device['max_input_channels']}, Out: {device['max_output_channels']})"
                    self.device_list.append(name)
                    self.device_indices.append(i)

        except Exception as e:
            print(f"Error listing devices: {e}")

    def on_settings_click(self, event):
        """Opens the settings dialog."""
        dlg = SettingsDialog(
            self,
            self.current_device_index,
            self.current_channel_index,
            self.current_api_url,
            self.current_api_key,
            self.current_llm_model,
            self.device_list,
            self.device_indices,
        )

        if dlg.ShowModal() == wx.ID_OK:
            # Update local state
            new_device = dlg.selected_device_index
            new_channel = dlg.selected_channel_index
            new_api_url = dlg.api_url
            new_api_key = dlg.api_key
            new_llm_model = dlg.llm_model

            settings_changed = False

            # Check for changes and apply
            if new_device != self.current_device_index:
                self.current_device_index = new_device
                if self.control_queue:
                    print(f"Selected device index: {self.current_device_index}")
                    self.control_queue.put(
                        {
                            "action": "set_device",
                            "device_index": self.current_device_index,
                        }
                    )
                settings_changed = True

            if new_channel != self.current_channel_index:
                self.current_channel_index = new_channel
                if self.control_queue:
                    print(f"Selected channel index: {self.current_channel_index}")
                    self.control_queue.put(
                        {
                            "action": "set_channel",
                            "channel_index": self.current_channel_index,
                        }
                    )
                settings_changed = True

            if new_api_url != self.current_api_url:
                self.current_api_url = new_api_url
                from kurtis_mlx import config

                config.OPENAI_API_URL = self.current_api_url
                print(f"Updated OpenAI API URL to: {self.current_api_url}")
                settings_changed = True

            if new_api_key != self.current_api_key:
                self.current_api_key = new_api_key
                from kurtis_mlx import config

                config.OPENAI_API_KEY = self.current_api_key
                print("Updated OpenAI API Key")
                settings_changed = True

            if new_llm_model != self.current_llm_model:
                self.current_llm_model = new_llm_model
                from kurtis_mlx import config

                config.LLM_MODEL = self.current_llm_model
                print(f"Updated LLM Model to: {self.current_llm_model}")
                settings_changed = True

            if settings_changed:
                # Save settings
                settings = {
                    "audio_device_index": self.current_device_index,
                    "audio_channel_index": self.current_channel_index,
                    "openai_api_url": self.current_api_url,
                    "openai_api_key": self.current_api_key,
                    "llm_model": self.current_llm_model,
                }
                SettingsManager.save_settings(settings)

        dlg.Destroy()

    def on_pause_click(self, event):
        """Toggles the pause state."""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.btn_pause.SetLabel("Resume")
            # Optional: Visual indication on panel
        else:
            self.btn_pause.SetLabel("Pause")

        if self.control_queue:
            self.control_queue.put({"action": "toggle_pause"})

    def start_threads(self):
        """Initializes and starts the background threads."""
        # Thread for audio processing
        self.audio_thread = threading.Thread(target=self.audio_thread_run, daemon=True)
        self.audio_thread.start()

        # Thread for the Kurtis Agent, if a function is provided
        if self.agent_run_function:
            self.agent_thread = threading.Thread(
                target=self.agent_thread_run, daemon=True
            )
            self.agent_thread.start()

    def agent_thread_run(self):
        """Runs the provided agent function."""
        if self.agent_run_function:
            self.agent_run_function()

    def audio_thread_run(self):
        """
        Handles microphone input.
        If audio_queue is provided, reads RMS values from it.
        Otherwise, opens a local PyAudio stream (standalone mode).
        """
        if self.audio_queue:
            # Queue mode (Integrated)
            while not self.stop_event.is_set():
                try:
                    # Get message from queue with a timeout
                    msg = self.audio_queue.get(timeout=0.1)

                    if isinstance(msg, dict):
                        msg_type = msg.get("type")
                        if msg_type == "rms":
                            normalized_rms = msg.get("value", 0.0)
                            if self.debug and np.random.rand() < 0.01:
                                print(f"UI received RMS: {normalized_rms:.4f}")
                            wx.CallAfter(self.panel.set_amplitude, normalized_rms)
                        elif msg_type == "status":
                            state = msg.get("state")
                            if state in [
                                PulsatingCirclePanel.STATE_LISTENING,
                                PulsatingCirclePanel.STATE_SPEAKING,
                            ]:
                                wx.CallAfter(self.panel.set_state, state)
                    else:
                        # Fallback for float messages (legacy)
                        normalized_rms = float(msg)
                        wx.CallAfter(self.panel.set_amplitude, normalized_rms)

                except queue.Empty:
                    # Empty queue or timeout, just continue
                    continue
                except Exception as e:
                    print(f"UI Audio Thread Error: {e}")
        else:
            # PyAudio mode (Standalone)
            try:
                # Try to open stream with default device
                self.stream = self.p.open(
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                )

                while not self.stop_event.is_set():
                    data = self.stream.read(CHUNK, exception_on_overflow=False)
                    # Convert audio data to a numpy array
                    audio_np = np.frombuffer(data, dtype=np.int16)
                    # Calculate Root Mean Square (RMS) as a measure of amplitude
                    rms = np.sqrt(np.mean(audio_np.astype(np.float32) ** 2))
                    # Normalize RMS (int16 ranges from -32768 to 32767)
                    normalized_rms = rms / 32768.0

                    # wx.CallAfter is crucial for thread safety. It queues the call
                    # to `set_amplitude` to be executed on the main GUI thread.
                    wx.CallAfter(self.panel.set_amplitude, normalized_rms)

            except Exception as e:
                print(f"Audio Error: {e}")
            finally:
                if self.stream:
                    self.stream.stop_stream()
                    self.stream.close()

    def on_timer(self, event):
        """Forces the panel to be redrawn."""
        self.panel.Refresh()

    def on_close(self, event):
        """Handles window closing, ensuring clean shutdown of threads and pyaudio."""
        self.stop_event.set()
        if self.audio_thread:
            self.audio_thread.join(timeout=1)
        self.p.terminate()
        self.Destroy()


def run_ui(agent_run_function=None, audio_queue=None, control_queue=None, debug=False):
    """Initializes and runs the wxPython UI application."""
    app = wx.App(False)
    frame = MainFrame(
        agent_run_function=agent_run_function,
        audio_queue=audio_queue,
        control_queue=control_queue,
        debug=debug,
    )
    frame.Show()
    app.MainLoop()


if __name__ == "__main__":

    def placeholder_agent():
        """A placeholder agent function for standalone UI testing."""
        print("Kurtis Agent simulation started...")
        # Use a shorter sleep for testing
        time.sleep(10)  # Reduced from 3600 seconds
        print("Kurtis Agent simulation finished.")

    run_ui(agent_run_function=placeholder_agent)
