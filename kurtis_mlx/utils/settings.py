import os
import yaml
from pathlib import Path


class SettingsManager:
    """
    Manages loading and saving of application settings to a YAML file.
    """

    APP_NAME = "kurtis"
    ORG_NAME = "ethicalabs-ai"
    CONFIG_FILENAME = "voice-agent.yaml"

    @staticmethod
    def get_config_dir():
        """Returns the configuration directory path."""
        # Use XDG_CONFIG_HOME if available, otherwise ~/.config
        xdg_config = os.environ.get("XDG_CONFIG_HOME")
        if xdg_config:
            base_path = Path(xdg_config)
        else:
            base_path = Path.home() / ".config"

        return base_path / SettingsManager.ORG_NAME / SettingsManager.APP_NAME

    @staticmethod
    def get_config_path():
        """Returns the full path to the configuration file."""
        return SettingsManager.get_config_dir() / SettingsManager.CONFIG_FILENAME

    @staticmethod
    def load_settings():
        """
        Loads settings from the YAML file.
        Returns a dictionary with settings or empty dict if file doesn't exist.
        """
        config_path = SettingsManager.get_config_path()
        if not config_path.exists():
            return {}

        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Error loading settings from {config_path}: {e}")
            return {}

    @staticmethod
    def save_settings(settings):
        """
        Saves the provided settings dictionary to the YAML file.
        Creates the directory if it doesn't exist.
        """
        config_dir = SettingsManager.get_config_dir()
        config_path = SettingsManager.get_config_path()

        try:
            config_dir.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w") as f:
                yaml.dump(settings, f, default_flow_style=False)
            print(f"Settings saved to {config_path}")
        except Exception as e:
            print(f"Error saving settings to {config_path}: {e}")
