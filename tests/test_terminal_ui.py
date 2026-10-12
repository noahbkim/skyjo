"""A few end-to-end checks of terminal input, playback, and worker lifetimes."""

import asyncio
import threading

from textual.widgets import Button, Input

from skyjo.engine import game as sj
from skyjo.simulation import recording
from skyjo.simulation.decisions import DiagnosticSection, PlayerDecision
from skyjo.simulation.player import RandomPlayer
from skyjo.simulation.session import GameSession
from skyjo.ui import render
from skyjo.ui.app import SkyjoApp
from skyjo.ui.setup import SetupScreen
from skyjo.ui.table import TableScreen


def test_setup_human_action_and_replay_navigation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    async def scenario():
        app = SkyjoApp()
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            assert isinstance(app.screen, SetupScreen)
            assert app.screen.query_one("#setup-body").scroll_y == 0
            app.screen.query_one("#play", Button).press()
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, TableScreen)
            screen.playing = False
            assert screen.query_one("#action-0").display
            assert not screen.query_one("#action-2").display
            screen.query_one("#action-1", Button).press()
            await pilot.pause()
            assert len(screen.saved.frames) == 2
            assert screen.saved.frames[0].action == sj.MASK_FLIP_SECOND_RIGHT
            assert (
                recording.load(screen.output_path).frames[0].action
                == sj.MASK_FLIP_SECOND_RIGHT
            )
            await pilot.press("left")
            assert screen.cursor == 0
            assert not screen.playing
            assert not screen.query_one("#human-controls").display
            await pilot.press("right")
            assert screen.cursor == 1
            screen.query_one("#save-path", Input).value = str(tmp_path / "copy.json")
            screen.query_one("#save-as", Button).press()
            await pilot.pause()
            copy = recording.load(tmp_path / "copy.json")
            screen.action_setup()
            await pilot.pause()
            app.push_screen(TableScreen(copy))
            await pilot.pause()
            assert not app.screen.playing
            await pilot.press("right", "left")
            assert app.screen.cursor == 0

    asyncio.run(scenario())


def test_pause_buffers_one_decision_and_late_results_cannot_advance_new_screen(
    tmp_path,
):
    ready = threading.Event()

    class SlowPlayer(RandomPlayer):
        calls = 0

        def get_decision(self, state, *, diagnostics=True):
            self.calls += 1
            ready.wait(5)
            return super().get_decision(state, diagnostics=diagnostics)

    async def scenario():
        player = SlowPlayer()
        session = GameSession(
            [player, player], seed=0, metadata=[{"name": "A"}, {"name": "B"}]
        )
        app = SkyjoApp()
        async with app.run_test(size=(80, 24)) as pilot:
            screen = TableScreen(
                session.recording,
                session=session,
                mode="watch",
                output_path=tmp_path / "watch.json",
            )
            app.push_screen(screen)
            await pilot.pause()
            screen.action_next()
            screen.action_next()
            assert screen.busy
            ready.set()
            await pilot.pause()
            assert player.calls == 1
            assert len(screen.saved.frames) == 1
            assert screen.proposal is not None
            screen.action_next()
            assert len(screen.saved.frames) == 2
            ready.clear()
            screen.action_next()
            screen.action_setup()
            await pilot.pause()
            ready.set()
            await pilot.pause()
            assert isinstance(app.screen, SetupScreen)
            assert len(session.recording.frames) == 2

    try:
        asyncio.run(scenario())
    finally:
        ready.set()


def test_eight_seat_narrow_table_has_scrollable_boards(tmp_path):
    async def scenario():
        session = GameSession(
            [RandomPlayer()] * 8, seed=0, metadata=[{"name": f"P{i}"} for i in range(8)]
        )
        app = SkyjoApp()
        async with app.run_test(size=(80, 24)) as pilot:
            app.push_screen(
                TableScreen(
                    session.recording,
                    session=session,
                    mode="watch",
                    output_path=tmp_path / "eight.json",
                )
            )
            await pilot.pause()
            screen = app.screen
            assert len(screen.query(".board")) == 8
            pane = screen.query_one("#boards-pane")
            assert pane.region.height > 0
            assert pane.max_scroll_y > 0
            screen.query_one("#board-7").scroll_visible(immediate=True)
            await pilot.pause()
            assert pane.scroll_y > 0

    asyncio.run(scenario())


def test_player_failure_pauses_without_losing_the_current_position(tmp_path):
    class FailingPlayer:
        def get_action_probabilities(self, state):
            raise RuntimeError("Checkpoint evaluator failed")

    async def scenario():
        session = GameSession(
            [FailingPlayer(), RandomPlayer()],
            seed=0,
            metadata=[{"name": "Broken"}, {"name": "Bot"}],
        )
        app = SkyjoApp()
        async with app.run_test() as pilot:
            screen = TableScreen(
                session.recording,
                session=session,
                mode="watch",
                output_path=tmp_path / "failed.json",
            )
            app.push_screen(screen)
            await pilot.pause()
            screen.playing = True
            screen.action_next()
            await pilot.pause()
            assert not screen.playing
            assert not screen.busy
            assert "Checkpoint evaluator failed" in screen.message
            assert len(screen.saved.frames) == 1
            assert not recording.load(screen.output_path).complete

    asyncio.run(scenario())


def test_card_context_distinguishes_draw_from_reveal_and_take():
    session = GameSession([None, None], seed=0, metadata=[{"name": "A"}, {"name": "B"}])
    assert "125 cards" in render.deck_summary(session.state).plain
    for _ in range(2):
        session.advance(action=sj.MASK_FLIP_SECOND_BELOW)
    assert "125 cards" in render.deck_summary(session.state).plain
    assert render.current_card(session.state).plain.startswith("DISCARD")
    session.advance(action=sj.MASK_DRAW)
    assert "124 cards" in render.deck_summary(session.state).plain
    assert render.current_card(session.state).plain.startswith("DRAWN CARD")
    session.advance(action=sj.MASK_REPLACE)
    session.advance(action=sj.MASK_TAKE)
    assert "124 cards" in render.deck_summary(session.state).plain
    assert render.current_card(session.state).plain.startswith("TAKEN FROM DISCARD")


def test_wide_diagnostics_scroll_without_rounding_away_small_differences(tmp_path):
    assert render.format_scalar(0.99999994) == "0.99999994"
    assert render.format_scalar(0.0000000123456789) == "1.23456789e-08"

    async def scenario():
        session = GameSession(
            [RandomPlayer()] * 2, seed=0, metadata=[{"name": "A"}, {"name": "B"}]
        )
        proposal = PlayerDecision(
            RandomPlayer().get_action_probabilities(session.state),
            (
                DiagnosticSection(
                    "Precise values",
                    columns=("A", "B", "C", "D"),
                    rows=((0.99999994, 0.123456789, 0.0000000123456789, 0.999999999),),
                ),
            ),
        )
        app = SkyjoApp()
        async with app.run_test(size=(120, 40)) as pilot:
            screen = TableScreen(
                session.recording,
                session=session,
                mode="watch",
                output_path=tmp_path / "precision.json",
            )
            app.push_screen(screen)
            await pilot.pause()
            screen.proposal = proposal
            screen.refresh_table()
            await pilot.pause()
            pane = screen.query_one("#diagnostics-pane")
            assert pane.max_scroll_x > 0
            pane.scroll_to(x=pane.max_scroll_x, animate=False)
            await pilot.pause()
            assert pane.scroll_x > 0

    asyncio.run(scenario())
