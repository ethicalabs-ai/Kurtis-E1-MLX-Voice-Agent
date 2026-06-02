import os
import sys
from unittest import mock

from click.testing import CliRunner

# Add project root to path to allow kurtis_mlx imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kurtis_mlx.__main__ import main


@mock.patch("kurtis_mlx.__main__.run_ui")
@mock.patch("kurtis_mlx.__main__.Process")
@mock.patch("kurtis_mlx.__main__.handle_interaction")
def test_main_with_ui_flag(mock_handle_interaction, mock_process, mock_run_ui):
    """Test that --ui flag calls run_ui and agent logic is not started directly."""
    runner = CliRunner()
    result = runner.invoke(main, ["--ui"])
    assert result.exit_code == 0
    mock_run_ui.assert_called_once()
    mock_handle_interaction.assert_not_called()


@mock.patch("kurtis_mlx.__main__.run_ui")
@mock.patch("kurtis_mlx.__main__.Process")
@mock.patch("kurtis_mlx.__main__.handle_interaction")
def test_main_without_ui_flag(mock_handle_interaction, mock_process, mock_run_ui):
    """Test that no --ui flag calls agent logic directly and not run_ui."""
    runner = CliRunner()
    # The agent logic runs in an infinite loop. We can mock handle_interaction
    # to raise a KeyboardInterrupt to break the loop for the test.
    mock_handle_interaction.side_effect = KeyboardInterrupt("test exit")

    result = runner.invoke(main, [])

    # The KeyboardInterrupt is caught and the program exits gracefully.
    # We check the output to ensure our interrupt was the cause.
    assert "KeyboardInterrupt" in result.output
    mock_run_ui.assert_not_called()
    mock_handle_interaction.assert_called_once()


def test_main_ui_and_sip_are_incompatible():
    """Test that --ui and --sip flags cannot be used together."""
    runner = CliRunner()
    result = runner.invoke(main, ["--ui", "--sip"])
    assert result.exit_code == 0
    assert "UI mode is not compatible with SIP mode" in result.output


@mock.patch("kurtis_mlx.__main__.handle_sip_interaction")
def test_main_sip_mode_missing_args(mock_handle_sip_interaction):
    """Test that --sip mode requires server, user, and password arguments."""
    runner = CliRunner()
    # Invoke with --sip but without all required sub-options
    result = runner.invoke(main, ["--sip", "--sip-server", "test.com"])
    assert result.exit_code == 0
    assert (
        "you must provide --sip-server, --sip-user, and --sip-password" in result.output
    )
    mock_handle_sip_interaction.assert_not_called()
