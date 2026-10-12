"""Textual application shell. Models are imported only when preparing model seats."""

from textual.app import App

from .config import load_preset
from .setup import SetupScreen


class SkyjoApp(App):
    TITLE = "Skyjo"
    CSS_PATH = "skyjo.tcss"

    def __init__(self, *, preset=None, recording=None):
        super().__init__()
        self.preset_path = preset
        self.recording_path = recording

    def on_mount(self):
        config = None
        error = None
        if self.preset_path:
            try:
                config = load_preset(self.preset_path)
            except (
                OSError,
                ValueError,
                TypeError,
                KeyError,
                AttributeError,
            ) as failure:
                error = f"Could not load preset: {failure}"
        setup = SetupScreen(config)
        self.push_screen(setup)
        if error:
            self.call_after_refresh(setup.status, error)
        if self.recording_path:
            self.call_after_refresh(setup.open_recording, self.recording_path)
