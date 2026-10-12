"""Live table and snapshot playback sharing one presentation and timeline."""

import uuid
from pathlib import Path
from typing import ClassVar

from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label, Select, Static

from skyjo.engine import game as sj
from skyjo.simulation import recording

from . import render


class TableScreen(Screen):
    BINDINGS: ClassVar = [
        ("left", "previous", "Previous"),
        ("right", "next", "Next"),
        ("space", "toggle_play", "Play/pause"),
        ("home", "first", "First"),
        ("end", "last", "Last"),
        ("[", "previous_round", "Previous round"),
        ("]", "next_round", "Next round"),
        ("l", "live", "Live"),
        ("escape", "setup", "Setup"),
        ("q", "app.quit", "Quit"),
        Binding("d", "human_draw", "Draw", show=False),
        Binding("t", "human_take", "Take", show=False),
        Binding("f", "flip_mode", "Flip", show=False),
        Binding("r", "replace_mode", "Replace", show=False),
    ]

    def __init__(self, saved, *, session=None, mode="replay", output_path=None):
        super().__init__()
        self.saved = saved
        self.session = session
        self.mode = mode
        self.cursor = 0
        self.playing = mode == "play"
        self.busy = False
        self.proposal = None
        self.request_id = 0
        self.card_mode = sj.MASK_REPLACE
        self.message = ""
        self.output_path = (
            Path(output_path)
            if output_path
            else Path(".runs/interactive") / f"game-{uuid.uuid4().hex}.json"
        )

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(id="game-status", markup=False)
        with Horizontal(id="card-context"):
            yield Static(id="current-card")
            yield Static(id="deck-summary")
        with Horizontal(classes="toolbar", id="playback"):
            yield Button("◀", id="previous")
            yield Button("Next ▶", id="next")
            yield Button("Play", id="toggle-play")
            yield Select(
                [("0.25 s", 0.25), ("0.5 s", 0.5), ("1 s", 1.0), ("2 s", 2.0)],
                value=1.0,
                allow_blank=False,
                id="speed",
            )
            yield Button("Live", id="live")
            yield Button("Setup", id="setup")
        with Horizontal(id="table-content"):
            with VerticalScroll(id="boards-pane"):
                with Grid(id="boards"):
                    for i in range(len(self.saved.players)):
                        yield Static(id=f"board-{i}", classes="board")
                with Vertical(id="human-controls"):
                    yield Label(
                        "Choose a legal action. For cards: select Flip/Replace, then a slot (1–9, a–c).",
                        id="human-prompt",
                    )
                    with Horizontal(classes="toolbar"):
                        yield Button("Below", id="action-0")
                        yield Button("Right", id="action-1")
                        yield Button(Text("Draw [d]"), id="action-2")
                        yield Button(Text("Take [t]"), id="action-3")
                        yield Button(Text("Flip [f]"), id="flip-mode")
                        yield Button(Text("Replace [r]"), id="replace-mode")
                    with Grid(id="card-actions"):
                        for i in range(sj.FINGER_COUNT):
                            yield Button(str(i + 1), id=f"card-{i}")
            with VerticalScroll(id="diagnostics-pane"):
                yield Static(
                    "Decision details · scroll ↔ for columns", classes="panel-title"
                )
                yield Static(id="diagnostics")
        yield Static("", id="table-message", markup=False)
        with Horizontal(classes="toolbar"):
            yield Input(placeholder="Save a copy to…", id="save-path")
            yield Button("Save As", id="save-as")
        yield Footer()

    def on_mount(self):
        self.timer = self.set_interval(1.0, self.tick)
        if self.session:
            self.persist()
        self.refresh_table()

    def on_unmount(self):
        self.request_id += 1
        self.playing = False
        self.timer.stop()

    def on_resize(self, event):
        self.set_class(event.size.width < 110, "narrow")

    def refresh_table(self):
        frame = self.saved.frames[self.cursor]
        state = frame.state
        player = sj.get_player(state)
        phase = render.PHASES[sj.get_action(state)]
        status = f"{self.mode.title()} · Round {frame.round_number} · Position {self.cursor + 1}/{len(self.saved.frames)} · Seat {player + 1}: {phase} · {'Thinking…' if self.busy else 'Playing' if self.playing else 'Paused'}"
        if frame.event != "decision":
            status = f"Round {frame.round_number} complete · Round scores {frame.round_scores} · Totals {frame.cumulative_scores}"
            if frame.winners is not None:
                status += "\nGame over · Winners: " + ", ".join(
                    self.saved.players[i]["name"] for i in frame.winners
                )
        if self.mode == "replay" and not self.saved.complete:
            status += " · Unfinished recording"
        self.query_one("#game-status", Static).update(status)
        self.query_one("#current-card", Static).update(render.current_card(state))
        self.query_one("#deck-summary", Static).update(render.deck_summary(state))
        for i, identity in enumerate(self.saved.players):
            self.query_one(f"#board-{i}", Static).update(
                render.board(state, i, identity["name"])
            )
        proposal = self.proposal if self.at_live else None
        details = render.diagnostics(frame, proposal)
        widget = self.query_one("#diagnostics", Static)
        widget.styles.min_width = render.diagnostic_width(details)
        widget.update(details)
        self.query_one("#toggle-play", Button).label = (
            "Pause" if self.playing else "Play"
        )
        human_hint = (
            "Your move — use the controls below the boards (scroll or Tab)."
            if self.session and self.at_live and self.session.human_turn
            else ""
        )
        previous = self.saved.frames[self.cursor - 1] if self.cursor else None
        last_move = (
            f"Previous move: seat {sj.get_player(previous.state) + 1} — "
            f"{sj.get_action_name(previous.action)}. "
            if previous is not None and previous.action is not None
            else ""
        )
        self.query_one("#table-message", Static).update(
            self.message
            or (last_move + human_hint if human_hint else "")
            or (
                "Decision ready. Next applies the move."
                if proposal
                else last_move or str(self.output_path)
                if self.session
                else "Saved snapshots · replay does not run players"
            )
        )
        self.refresh_human_controls()

    @property
    def at_live(self):
        return self.cursor == len(self.saved.frames) - 1

    def refresh_human_controls(self):
        active = self.session is not None and self.at_live and self.session.human_turn
        self.query_one("#human-controls").display = active
        if not active:
            return
        state = self.session.state
        legal = sj.actions(state)
        for i in range(4):
            self.query_one(f"#action-{i}", Button).display = bool(legal[i])
        can_flip = bool(legal[sj.MASK_FLIP : sj.MASK_REPLACE].any())
        can_replace = bool(legal[sj.MASK_REPLACE :].any())
        if self.card_mode == sj.MASK_FLIP and not can_flip:
            self.card_mode = sj.MASK_REPLACE
        if self.card_mode == sj.MASK_REPLACE and not can_replace:
            self.card_mode = sj.MASK_FLIP
        for name, enabled, mode in (
            ("flip-mode", can_flip, sj.MASK_FLIP),
            ("replace-mode", can_replace, sj.MASK_REPLACE),
        ):
            button = self.query_one(f"#{name}", Button)
            button.display = enabled
            button.variant = "primary" if self.card_mode == mode else "default"
        self.query_one("#card-actions").display = can_flip or can_replace
        for i in range(sj.FINGER_COUNT):
            button = self.query_one(f"#card-{i}", Button)
            row, col = divmod(i, sj.COLUMN_COUNT)
            card = sj.get_finger(state, row, col)
            button.label = Text.assemble(
                f"{'123456789abc'[i]}: ",
                (render.card_text(card), render.card_style(card)),
            )
            button.disabled = not legal[self.card_mode + i]

    @on(Button.Pressed)
    def pressed(self, event):
        name = event.button.id
        if name.startswith("action-"):
            self.human_action(int(name.removeprefix("action-")))
        elif name.startswith("card-"):
            self.human_action(self.card_mode + int(name.removeprefix("card-")))
        elif name == "save-as":
            text = self.query_one("#save-path", Input).value.strip()
            if not text:
                self.message = "Enter a destination path."
            else:
                try:
                    recording.save(self.saved, Path(text).expanduser())
                    self.message = f"Saved {text}"
                except (OSError, ValueError) as error:
                    self.message = f"Save failed: {error}"
            self.refresh_table()
        else:
            method = getattr(self, f"action_{name.replace('-', '_')}", None)
            if method:
                method()

    def on_key(self, event):
        if not isinstance(self.focused, Input) and event.character in tuple(
            "123456789abc"
        ):
            self.human_action(self.card_mode + "123456789abc".index(event.character))

    @on(Select.Changed, "#speed")
    def speed_changed(self, event):
        if hasattr(self, "timer"):
            self.timer.stop()
            self.timer = self.set_interval(float(event.value), self.tick)

    def tick(self):
        if self.playing:
            self.action_next()

    def action_next(self):
        if not self.at_live:
            self.cursor += 1
        elif self.session and not self.saved.complete:
            if sj.get_round_over(self.session.state):
                self.commit()
            elif self.session.human_turn:
                pass
            elif self.proposal is not None:
                self.commit(self.proposal)
            elif not self.busy:
                self.busy = True
                self.request_id += 1
                self.compute_decision(self.request_id)
        else:
            self.playing = False
        self.refresh_table()

    @work(thread=True, exit_on_error=False)
    def compute_decision(self, request_id):
        try:
            decision = self.session.propose()
            self.app.call_from_thread(self.receive_decision, request_id, decision, None)
        except Exception as error:  # noqa: BLE001 - isolate third-party player failures
            self.app.call_from_thread(
                self.receive_decision, request_id, None, str(error)
            )

    def receive_decision(self, request_id, decision, error):
        if request_id != self.request_id or not self.is_mounted:
            return
        self.busy = False
        self.proposal = decision
        if error:
            self.playing = False
            self.message = (
                f"Player failed: {error}. Next retries; Setup changes the matchup."
            )
        else:
            self.message = ""
        self.refresh_table()

    def commit(self, decision=None, *, action=None):
        try:
            self.session.advance(decision, action=action)
            self.proposal = None
            self.cursor = len(self.saved.frames) - 1
            self.message = ""
            self.persist()
        except (ValueError, OSError) as error:
            self.message = str(error)
            self.playing = False

    def persist(self):
        try:
            recording.save(self.saved, self.output_path)
        except (OSError, ValueError) as error:
            self.playing = False
            self.message = (
                f"Autosave failed: {error}. Use Save As to preserve this session."
            )

    def human_action(self, action):
        if self.session and self.at_live and self.session.human_turn:
            self.commit(action=action)
            self.refresh_table()

    def seek(self, index):
        self.playing = False
        self.cursor = max(0, min(index, len(self.saved.frames) - 1))
        self.refresh_table()

    def action_previous(self):
        self.seek(self.cursor - 1)

    def action_first(self):
        self.seek(0)

    def action_last(self):
        self.seek(len(self.saved.frames) - 1)

    def action_live(self):
        self.action_last()

    def action_previous_round(self):
        round_number = self.saved.frames[self.cursor].round_number
        starts = [
            i
            for i, frame in enumerate(self.saved.frames)
            if frame.round_number < round_number
        ]
        target = self.saved.frames[starts[-1]].round_number if starts else 1
        self.seek(
            next(
                i
                for i, frame in enumerate(self.saved.frames)
                if frame.round_number == target
            )
        )

    def action_next_round(self):
        round_number = self.saved.frames[self.cursor].round_number
        self.seek(
            next(
                (
                    i
                    for i, frame in enumerate(self.saved.frames)
                    if frame.round_number > round_number
                ),
                len(self.saved.frames) - 1,
            )
        )

    def action_toggle_play(self):
        self.playing = not self.playing
        self.refresh_table()

    def action_setup(self):
        self.playing = False
        self.request_id += 1
        self.app.pop_screen()

    def action_human_draw(self):
        self.human_action(sj.MASK_DRAW)

    def action_human_take(self):
        self.human_action(sj.MASK_TAKE)

    def action_flip_mode(self):
        self.card_mode = sj.MASK_FLIP
        self.refresh_human_controls()

    def action_replace_mode(self):
        self.card_mode = sj.MASK_REPLACE
        self.refresh_human_controls()
