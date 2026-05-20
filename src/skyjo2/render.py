from __future__ import annotations

import shutil
from collections.abc import Iterable

from . import CARD_VALUES, HAND_COLUMNS, HAND_ROWS, Finger, Game


def _lrjust(left: object, right: object, width: int) -> str:
    left = str(left)
    right = str(right)
    space = " " * (width - len(left) - len(right))
    return f"{left}{space}{right}"


def render_card(card_index: int) -> str:
    return str(CARD_VALUES[card_index])


def render_finger(finger: Finger) -> str:
    if finger.is_hidden:
        return "[]"
    if finger.is_cleared:
        return "  "
    return str(CARD_VALUES[finger.card_index]).rjust(2)


def render(game: Game, *, column_max: int | None = None) -> Iterable[str]:
    """Render the game as ASCII art."""

    if column_max is None:
        column_max = shutil.get_terminal_size((0, 0)).columns

    yield (
        f"turn: {game.turn}  discard: {render_card(game.discarded_card_index)}"
        + (
            f"  draw: {render_card(game.drawn_card_index)}"
            if game.drawn_card_index is not None
            else ""
        )
    )

    yield ""

    hand_width = HAND_COLUMNS * 2 + HAND_COLUMNS - 1

    yield "  ".join(
        _lrjust(f"p{(i + game.turn) % len(game.players)}:", player.score, hand_width)
        for i, player in enumerate(game.players)
    )

    yield "  ".join("-" * hand_width for _ in game.players)

    for row_index in range(HAND_ROWS):
        yield "  ".join(
            " ".join(
                render_finger(finger) for finger in player.hand[row_index::HAND_ROWS]
            )
            for player in game.players
        )
