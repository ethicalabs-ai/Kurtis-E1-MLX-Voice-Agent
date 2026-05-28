import os
import sys
from unittest import mock

# Add project root to path to allow kurtis_ui import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Since wx, pyaudio and numpy may not be installed in the test environment,
# we need to mock them before importing from kurtis_ui.

# To allow PulsatingCirclePanel to properly inherit from wx.Panel, we can't
# just mock all of wx with a MagicMock. If wx.Panel is a MagicMock, then
# PulsatingCirclePanel becomes a MagicMock too, and its __init__ is not run.
# Instead, we create a mock for the wx module and assign a dummy class to wx.Panel.
mock_wx = mock.MagicMock()


class MockPanel:
    """A dummy class to stand in for wx.Panel for inheritance purposes."""

    def __init__(self, *args, **kwargs):
        # We also need to mock the methods that are called on the panel instance
        # within PulsatingCirclePanel's __init__, because this dummy class
        # does not have them.
        self.SetBackgroundStyle = mock.MagicMock()
        self.SetBackgroundColour = mock.MagicMock()
        self.Bind = mock.MagicMock()


class MockFrame:
    """A dummy class to stand in for wx.Frame for inheritance purposes."""

    def __init__(self, *args, **kwargs):
        self.Bind = mock.MagicMock()
        self.SetBackgroundColour = mock.MagicMock()
        self.Center = mock.MagicMock()
        self.Destroy = mock.MagicMock()
        self.Show = mock.MagicMock()


mock_wx.Panel = MockPanel
mock_wx.Frame = MockFrame

# This technique places mock objects into sys.modules, so when kurtis_ui
# is imported, it receives our mocks instead of trying to load the real libraries.
sys.modules['wx'] = mock_wx
sys.modules['pyaudio'] = mock.MagicMock()
sys.modules['numpy'] = mock.MagicMock()

# Now we can safely import from kurtis_mlx.ui
from kurtis_mlx.ui import (
    BASE_RADIUS,
    MainFrame,
    PulsatingCirclePanel,
    SCALING_FACTOR,
)


@mock.patch("threading.Thread")
@mock.patch("pyaudio.PyAudio")
def test_main_frame_starts_agent_thread(mock_pyaudio, mock_thread):
    """Tests that MainFrame starts a thread for the agent function."""
    agent_func_mock = mock.MagicMock()
    frame = MainFrame(agent_run_function=agent_func_mock)

    # Verify that a thread was created with the agent_thread_run as its target.
    # The audio thread is also created, so we check through all thread creations.
    agent_thread_created = any(
        call.kwargs.get("target") == frame.agent_thread_run
        for call in mock_thread.call_args_list
    )
    assert agent_thread_created, "Agent thread was not created."

    # Verify that start() was called on the thread instance(s).
    mock_thread.return_value.start.assert_called()

    # Verify that agent_thread_run() calls the provided function.
    frame.agent_thread_run()
    agent_func_mock.assert_called_once()


def test_pulsating_circle_panel_initialization():
    """Tests that the PulsatingCirclePanel initializes with correct values."""
    import wx  # This will be our mock wx

    parent_mock = mock.MagicMock()
    panel = PulsatingCirclePanel(parent_mock)

    assert panel.amplitude == 0.0
    assert panel.base_radius == BASE_RADIUS
    assert panel.scaling_factor == SCALING_FACTOR

    # Check that the panel binds the EVT_PAINT event to its on_paint method
    panel.Bind.assert_called_once_with(wx.EVT_PAINT, panel.on_paint)


def test_pulsating_circle_panel_set_amplitude():
    """Tests that set_amplitude correctly updates the amplitude value."""
    parent_mock = mock.MagicMock()
    panel = PulsatingCirclePanel(parent_mock)

    panel.set_amplitude(0.5)
    assert panel.amplitude == 0.5

    panel.set_amplitude(1.0)
    assert panel.amplitude == 1.0


def test_pulsating_circle_panel_on_paint():
    """
    Tests that the on_paint method performs the expected drawing operations
    by checking calls to the mocked wx.GraphicsContext.
    """
    import wx  # The mock wx

    parent_mock = mock.MagicMock()
    panel = PulsatingCirclePanel(parent_mock)

    # Mock the GetSize method to return a fixed size for consistent calculations
    panel.GetSize = mock.MagicMock(return_value=(400, 400))
    panel.set_amplitude(0.5)

    event_mock = mock.MagicMock()

    with mock.patch('wx.AutoBufferedPaintDC') as mock_dc_class, \
         mock.patch('wx.GraphicsContext.Create') as mock_gc_create:

        # Configure the mocks returned by the patched objects
        mock_dc = mock.MagicMock()
        mock_dc_class.return_value = mock_dc
        mock_gc = mock.MagicMock()
        mock_gc_create.return_value = mock_gc

        # Call the method under test
        panel.on_paint(event_mock)

        # Assert that the drawing setup was correct
        mock_dc_class.assert_called_once_with(panel)
        mock_dc.Clear.assert_called_once()
        mock_gc_create.assert_called_once_with(mock_dc)

        # Assert that the circle was drawn with the correct, calculated properties
        center_x, center_y = 200, 200
        expected_radius = BASE_RADIUS + (0.5 * SCALING_FACTOR)  # 50 + 50 = 100

        mock_gc.AddCircle.assert_called_once_with(center_x, center_y, expected_radius)
        mock_gc.FillPath.assert_called_once()
