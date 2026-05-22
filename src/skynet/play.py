from __future__ import annotations

from collections.abc import Iterator

from skyjo2 import Game
from skyjo2.play import Action as Action
from skyjo2.play import ActionKind as ActionKind
from skyjo2.play import iter_actions as iter_actions

type Transition = (
    tuple[ActionKind.REVEAL_SECOND_CARD, int, int]
    | tuple[ActionKind.DRAW_CARD, int]
    | tuple[ActionKind.DISCARD_DRAW_AND_REVEAL_CARD, int, int]
    | tuple[ActionKind.REPLACE_WITH_DRAW, int, int | None]
    | tuple[ActionKind.REPLACE_WITH_DISCARD, int, int | None]
)


def iter_outcomes(game: Game, action: Action) -> Iterator[tuple[int | None, int]]:
    """Yields pairs of card index, weight as outcomes of an action."""

    if (
        # Revealing and drawing are guaranteed to roll a card.
        action[0] in {ActionKind.REVEAL_SECOND_CARD, ActionKind.DRAW_CARD}
        # Otherwise, we only roll if the selected finger is hidden.
        or not game.player.hand[action[1]].is_revealed
    ):
        yield from enumerate(game.draw_pile)
    else:
        yield None, 1


def iter_transitions(game: Game) -> Iterator[Transition]:
    """Yields all possible transitions from a game state."""

    for action in iter_actions(game):
        for outcome, _ in iter_outcomes(game, action):
            yield (*action, outcome)
