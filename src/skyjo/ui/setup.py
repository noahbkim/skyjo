"""Matchup editing and recording selection."""

from pathlib import Path
from typing import ClassVar

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    Checkbox,
    Collapsible,
    Footer,
    Header,
    Input,
    Label,
    Select,
    Static,
)

from skyjo.search.mcts import SearchConfig
from skyjo.simulation import recording

from .config import MatchConfig, SeatConfig, load_preset, prepare_session, save_preset


class SeatEditor(Vertical):
    def __init__(self, number: int, seat: SeatConfig):
        super().__init__(id=f"seat-{number}")
        self.number = number
        self.seat = seat

    def compose(self) -> ComposeResult:
        with Collapsible(
            title=f"Seat {self.number}: {self.seat.name} ({self.seat.kind})",
            collapsed=True,
        ):
            yield Label("Name")
            yield Input(self.seat.name, classes="name")
            yield Select(
                [(name, name) for name in ("human", "random", "policy", "search")],
                value=self.seat.kind,
                allow_blank=False,
                classes="kind",
            )
            with VerticalScroll(classes="model-options"):
                yield Label("Model checkpoint path")
                yield Input(self.seat.checkpoint or "", classes="checkpoint")
                with Vertical(classes="search-options"):
                    yield Label("Search iterations / action temperature")
                    with Horizontal(classes="input-pair"):
                        yield Input(
                            str(self.seat.iterations),
                            classes="iterations",
                            type="integer",
                        )
                        yield Input(
                            str(self.seat.temperature),
                            classes="temperature",
                            type="number",
                        )
                    with Collapsible(title="Advanced search settings", collapsed=True):
                        for name in (
                            "c_puct",
                            "fpu_reduction",
                            "dirichlet_epsilon",
                            "boundary_samples",
                        ):
                            yield Label(name.replace("_", " "))
                            yield Input(
                                str(getattr(self.seat.search, name)), classes=name
                            )
                        yield Checkbox(
                            "Merge symmetric actions",
                            value=self.seat.search.merge_symmetric_actions,
                            classes="merge_symmetric_actions",
                        )
                        yield Checkbox(
                            "Evaluate all chance outcomes",
                            value=self.seat.search.after_state_evaluate_all_children,
                            classes="after_state_evaluate_all_children",
                        )
                        yield Label("Optional boundary checkpoint")
                        yield Input(
                            self.seat.boundary_checkpoint or "",
                            classes="boundary_checkpoint",
                        )

    def on_mount(self):
        self.update_kind()

    @on(Select.Changed, ".kind")
    def update_kind(self):
        kind = self.query_one(".kind", Select).value
        self.query_one(".model-options").display = kind in ("policy", "search")
        self.query_one(".search-options").display = kind == "search"
        self.update_title()

    @on(Input.Changed, ".name")
    def update_title(self):
        name = self.query_one(".name", Input).value
        kind = self.query_one(".kind", Select).value
        self.query_one(Collapsible).title = f"Seat {self.number}: {name} ({kind})"

    def read(self) -> SeatConfig:
        def value(name):
            return self.query_one(f".{name}", Input).value.strip()

        kind = self.query_one(".kind", Select).value
        search = SearchConfig()
        if kind == "search":
            search = SearchConfig(
                **{
                    name: float(value(name))
                    for name in ("c_puct", "fpu_reduction", "dirichlet_epsilon")
                },
                boundary_samples=int(value("boundary_samples")),
                **{
                    name: self.query_one(f".{name}", Checkbox).value
                    for name in (
                        "merge_symmetric_actions",
                        "after_state_evaluate_all_children",
                    )
                },
            )
        return SeatConfig(
            name=value("name"),
            kind=kind,
            checkpoint=value("checkpoint") or None,
            iterations=int(value("iterations")),
            temperature=float(value("temperature")),
            search=search,
            boundary_checkpoint=value("boundary_checkpoint") or None
            if kind == "search"
            else None,
        )


class SetupScreen(Screen):
    AUTO_FOCUS = "#recording-path"
    BINDINGS: ClassVar = [("q", "app.quit", "Quit")]

    def __init__(self, config=None):
        super().__init__()
        self.config = config or MatchConfig()
        self.busy = False

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="setup-body"):
            yield Static(
                "[bold]Skyjo · Play, watch, replay[/bold]\nConfigure seats below, or open a recorded game."
            )
            with Horizontal(classes="toolbar"):
                yield Input(placeholder="Recording JSON path", id="recording-path")
                yield Button("Open recording", id="open-recording")
            with Horizontal(classes="toolbar"):
                yield Input(placeholder="Matchup TOML path", id="preset-path")
                yield Button("Load preset", id="load-preset")
                yield Button("Save preset", id="save-preset")
            yield Label("Player count / seed")
            with Horizontal(classes="toolbar"):
                yield Select(
                    [(str(i), i) for i in range(2, 9)],
                    value=len(self.config.seats),
                    allow_blank=False,
                    id="seat-count",
                )
                yield Input(str(self.config.seed), type="integer", id="seed")
            with VerticalScroll(id="seat-editors"):
                for i in range(8):
                    seat = (
                        self.config.seats[i]
                        if i < len(self.config.seats)
                        else SeatConfig(f"Player {i + 1}")
                    )
                    yield SeatEditor(i + 1, seat)
        yield Static("", id="setup-status", markup=False)
        with Horizontal(classes="toolbar"):
            yield Button("Play", variant="primary", id="play")
            yield Button("Watch", id="watch")
        yield Footer()

    def on_mount(self):
        self.update_count()

    @on(Select.Changed, "#seat-count")
    def update_count(self):
        count = self.query_one("#seat-count", Select).value
        for i, editor in enumerate(self.query(SeatEditor)):
            editor.display = i < count

    def read(self):
        count = self.query_one("#seat-count", Select).value
        return MatchConfig(
            tuple(editor.read() for editor in list(self.query(SeatEditor))[:count]),
            int(self.query_one("#seed", Input).value),
        )

    def status(self, message):
        self.query_one("#setup-status", Static).update(message)

    @on(Button.Pressed)
    def pressed(self, event: Button.Pressed):
        if self.busy:
            return
        action = event.button.id
        try:
            if action == "load-preset":
                config = load_preset(self.path("preset-path"))
                self.app.switch_screen(SetupScreen(config))
            elif action == "save-preset":
                save_preset(self.read(), self.path("preset-path"))
                self.status("Preset saved.")
            elif action == "open-recording":
                self.busy = True
                self.status("Opening recording…")
                self.open_recording(self.path("recording-path"))
            elif action in ("play", "watch"):
                config = self.read()
                humans = sum(seat.kind == "human" for seat in config.seats)
                if action == "watch" and humans:
                    raise ValueError("Watch needs automated players in every seat")
                if action == "play" and not humans:
                    raise ValueError(
                        "Play needs at least one human seat; use Watch for automated games"
                    )
                self.busy = True
                self.status("Preparing players…")
                self.start_game(config, action)
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            self.busy = False
            self.status(str(error))

    def path(self, widget_id):
        text = self.query_one(f"#{widget_id}", Input).value.strip()
        if not text:
            raise ValueError("Enter a file path first")
        return Path(text).expanduser()

    @work(thread=True, exit_on_error=False)
    def start_game(self, config, mode):
        try:
            session = prepare_session(config)
            self.app.call_from_thread(self.open_table, session.recording, session, mode)
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
            self.app.call_from_thread(self.failed, str(error))

    @work(thread=True, exit_on_error=False)
    def open_recording(self, path):
        try:
            saved = recording.load(path)
            self.app.call_from_thread(self.open_table, saved, None, "replay")
        except (OSError, ValueError) as error:
            self.app.call_from_thread(self.failed, str(error))

    def failed(self, message):
        self.busy = False
        if self.is_mounted:
            self.status(message)

    def open_table(self, saved, session, mode):
        from .table import TableScreen

        self.busy = False
        if self.is_mounted:
            self.app.push_screen(TableScreen(saved, session=session, mode=mode))
