from __future__ import annotations

import random

from skyjo2 import (
    Action,
    ActionKind,
    Finger,
    Game,
    GameState,
    Player,
    play_game,
    play_round,
)


def draw(game: Game) -> Action:
    if game.state == GameState.REVEAL_SECOND_CARD:
        return (ActionKind.REVEAL_SECOND_CARD, 1)
    elif game.state == GameState.DRAW_OR_REPLACE_WITH_DISCARD:
        return (ActionKind.DRAW_CARD,)
    elif game.state == GameState.DISCARD_DRAW_AND_REVEAL_OR_REPLACE_WITH_DRAW:
        for i, finger in enumerate(game.player.hand):
            if finger.is_hidden:
                return (ActionKind.REPLACE_WITH_DRAW, i)
    assert False, f"unknown state {game}"


def stall(game: Game) -> Action:
    if game.state == GameState.REVEAL_SECOND_CARD:
        return (ActionKind.REVEAL_SECOND_CARD, 1)
    elif game.state == GameState.DRAW_OR_REPLACE_WITH_DISCARD:
        return (ActionKind.REPLACE_WITH_DISCARD, 2)
    elif game.state == GameState.DISCARD_DRAW_AND_REVEAL_OR_REPLACE_WITH_DRAW:
        assert False, f"unexpected state {game}"
    assert False, f"unknown state {game}"


def test_play_round_two(rng: random.Random) -> None:
    game = Game.new(players=2)
    players = (draw, draw)
    replay = list(play_round(game, players, rng))
    # 1 (deal)
    # 2 (reveal second)
    # 2 players * 10 cards * 2 actions (draws + replaces)
    # 1 (reveal hidden)
    assert len(replay) == 44
    result = replay[-1]
    assert result == Game(
        state=GameState.DEAL_FIRST_CARDS,
        turn_index=20,
        draw_pile=(4, 7, 11, 7, 7, 8, 7, 5, 6, 5, 7, 10, 7, 8, 6),
        drawn_card_index=None,
        discard_pile=(0, 2, 2, 2, 0, 1, 1, 2, 3, 3, 1, 0, 1, 0, 2),
        discarded_card_index=6,
        players=(
            Player(
                score=59,
                hand=(
                    Finger(card_index=10, is_revealed=True),
                    Finger(card_index=6, is_revealed=True),
                    Finger(card_index=12, is_revealed=True),
                    Finger(card_index=9, is_revealed=True),
                    Finger(card_index=7, is_revealed=True),
                    Finger(card_index=4, is_revealed=True),
                    Finger(card_index=9, is_revealed=True),
                    Finger(card_index=5, is_revealed=True),
                    Finger(card_index=10, is_revealed=True),
                    Finger(card_index=7, is_revealed=True),
                    Finger(card_index=4, is_revealed=True),
                    Finger(card_index=0, is_revealed=True),
                ),
            ),
            Player(
                score=69,
                hand=(
                    Finger(card_index=1, is_revealed=True),
                    Finger(card_index=13, is_revealed=True),
                    Finger(card_index=7, is_revealed=True),
                    Finger(card_index=14, is_revealed=True),
                    Finger(card_index=2, is_revealed=True),
                    Finger(card_index=2, is_revealed=True),
                    Finger(card_index=8, is_revealed=True),
                    Finger(card_index=4, is_revealed=True),
                    Finger(card_index=3, is_revealed=True),
                    Finger(card_index=14, is_revealed=True),
                    Finger(card_index=13, is_revealed=True),
                    Finger(card_index=12, is_revealed=True),
                ),
            ),
        ),
        start_player_index=0,
        end_player_index=0,
    )


def test_play_round_three(rng: random.Random) -> None:
    game = Game.new(players=3)
    players = (draw, draw, draw)
    replay = list(play_round(game, players, rng))
    # 1 (deal)
    # 3 (reveal second)
    # 3 players * 10 cards * 2 actions (draws + replaces)
    # 1 (reveal hidden)
    assert len(replay) == 65
    result = replay[-1]
    assert result == Game(
        state=GameState.DEAL_FIRST_CARDS,
        turn_index=30,
        drawn_card_index=None,
        draw_pile=(3, 6, 8, 6, 4, 7, 6, 5, 6, 4, 6, 6, 5, 8, 3),
        discarded_card_index=2,
        discard_pile=(0, 2, 3, 3, 1, 2, 2, 2, 3, 3, 1, 2, 1, 0, 5),
        players=(
            Player(
                score=49,
                hand=(
                    Finger(card_index=10, is_revealed=True),
                    Finger(card_index=13, is_revealed=True),
                    Finger(card_index=7, is_revealed=True),
                    Finger(card_index=7, is_revealed=True),
                    Finger(card_index=2, is_revealed=True),
                    Finger(card_index=5, is_revealed=True),
                    Finger(card_index=3, is_revealed=True),
                    Finger(card_index=4, is_revealed=True),
                    Finger(card_index=12, is_revealed=True),
                    Finger(card_index=0, is_revealed=True),
                    Finger(card_index=6, is_revealed=True),
                    Finger(card_index=4, is_revealed=True),
                ),
            ),
            Player(
                score=66,
                hand=(
                    Finger(card_index=1, is_revealed=True),
                    Finger(card_index=12, is_revealed=True),
                    Finger(card_index=9, is_revealed=True),
                    Finger(card_index=2, is_revealed=True),
                    Finger(card_index=9, is_revealed=True),
                    Finger(card_index=4, is_revealed=True),
                    Finger(card_index=7, is_revealed=True),
                    Finger(card_index=13, is_revealed=True),
                    Finger(card_index=12, is_revealed=True),
                    Finger(card_index=9, is_revealed=True),
                    Finger(card_index=1, is_revealed=True),
                    Finger(card_index=11, is_revealed=True),
                ),
            ),
            Player(
                score=71,
                hand=(
                    Finger(card_index=6, is_revealed=True),
                    Finger(card_index=10, is_revealed=True),
                    Finger(card_index=14, is_revealed=True),
                    Finger(card_index=4, is_revealed=True),
                    Finger(card_index=8, is_revealed=True),
                    Finger(card_index=10, is_revealed=True),
                    Finger(card_index=14, is_revealed=True),
                    Finger(card_index=0, is_revealed=True),
                    Finger(card_index=12, is_revealed=True),
                    Finger(card_index=4, is_revealed=True),
                    Finger(card_index=11, is_revealed=True),
                    Finger(card_index=2, is_revealed=True),
                ),
            ),
        ),
        start_player_index=0,
        end_player_index=0,
    )


def test_play_round_forfeit(rng: random.Random) -> None:
    game = Game.new(players=2)
    players = (draw, stall)
    replay = list(play_round(game, players, rng, no_progress_turn_max=5))
    # 1 (deal)
    # 2 (reveal)
    # 1 player * 5 cards * 2 actions (draws + replaces)
    # 1 player * 5 actions (replace with discard)
    # 1 forefit
    # 2 actions (draw + replace in endgame)
    # 1 (reveal hidden cards)
    assert len(replay) == 27
    assert replay[-2] == Game(
        state=GameState.REVEAL_HIDDEN_CARDS,
        turn_index=15,
        draw_pile=(5, 9, 12, 8, 9, 9, 9, 7, 9, 7, 8, 10, 8, 9, 9),
        drawn_card_index=None,
        discard_pile=(0, 0, 2, 0, 0, 0, 0, 2, 0, 2, 1, 0, 0, 0, 1),
        discarded_card_index=9,
        players=(
            Player(
                score=0,
                hand=(
                    Finger(card_index=10, is_revealed=True),
                    Finger(card_index=6, is_revealed=True),
                    Finger(card_index=12, is_revealed=True),
                    Finger(card_index=12, is_revealed=True),
                    Finger(card_index=5, is_revealed=True),
                    Finger(card_index=3, is_revealed=True),
                    Finger(card_index=3, is_revealed=True),
                    Finger(card_index=7, is_revealed=True),
                    Finger(card_index=8, is_revealed=True),
                    Finger(card_index=2, is_revealed=True),
                    Finger(card_index=None, is_revealed=False),
                    Finger(card_index=None, is_revealed=False),
                ),
            ),
            Player(
                score=0,
                hand=(
                    Finger(card_index=1, is_revealed=True),
                    Finger(card_index=13, is_revealed=True),
                    Finger(card_index=4, is_revealed=True),
                    Finger(card_index=None, is_revealed=False),
                    Finger(card_index=None, is_revealed=False),
                    Finger(card_index=None, is_revealed=False),
                    Finger(card_index=None, is_revealed=False),
                    Finger(card_index=None, is_revealed=False),
                    Finger(card_index=None, is_revealed=False),
                    Finger(card_index=None, is_revealed=False),
                    Finger(card_index=None, is_revealed=False),
                    Finger(card_index=None, is_revealed=False),
                ),
            ),
        ),
        start_player_index=0,
        end_player_index=1,
    )
    result = replay[-1]
    assert result == Game(
        state=GameState.DEAL_FIRST_CARDS,
        turn_index=15,
        draw_pile=(5, 8, 12, 7, 8, 8, 8, 7, 7, 6, 7, 10, 8, 9, 7),
        drawn_card_index=None,
        discard_pile=(0, 0, 2, 0, 0, 0, 0, 2, 0, 2, 1, 0, 0, 0, 1),
        discarded_card_index=9,
        players=(
            Player(
                score=66,
                hand=(
                    Finger(card_index=10, is_revealed=True),
                    Finger(card_index=6, is_revealed=True),
                    Finger(card_index=12, is_revealed=True),
                    Finger(card_index=12, is_revealed=True),
                    Finger(card_index=5, is_revealed=True),
                    Finger(card_index=3, is_revealed=True),
                    Finger(card_index=3, is_revealed=True),
                    Finger(card_index=7, is_revealed=True),
                    Finger(card_index=8, is_revealed=True),
                    Finger(card_index=2, is_revealed=True),
                    Finger(card_index=14, is_revealed=True),
                    Finger(card_index=8, is_revealed=True),
                ),
            ),
            Player(
                score=108,
                hand=(
                    Finger(card_index=1, is_revealed=True),
                    Finger(card_index=13, is_revealed=True),
                    Finger(card_index=4, is_revealed=True),
                    Finger(card_index=1, is_revealed=True),
                    Finger(card_index=5, is_revealed=True),
                    Finger(card_index=6, is_revealed=True),
                    Finger(card_index=4, is_revealed=True),
                    Finger(card_index=9, is_revealed=True),
                    Finger(card_index=10, is_revealed=True),
                    Finger(card_index=14, is_revealed=True),
                    Finger(card_index=3, is_revealed=True),
                    Finger(card_index=8, is_revealed=True),
                ),
            ),
        ),
        start_player_index=0,
        end_player_index=1,
    )


def test_play_game_three(rng: random.Random) -> None:
    game = Game.new(players=3)
    players = (draw, draw, draw)
    replay = list(play_game(game, players, rng))
    # 1 (deal)
    # 3 (reveal second)
    # 3 players * 10 cards * 2 actions (draws + replaces)
    # 1 (reveal hidden)
    # x2 (two rounds)
    assert len(replay) == 65 * 2
    result = replay[-1]
    assert result == Game(
        state=GameState.DEAL_FIRST_CARDS,
        turn_index=30,
        draw_pile=(4, 6, 9, 7, 7, 6, 6, 5, 6, 4, 6, 6, 3, 5, 3),
        drawn_card_index=None,
        discard_pile=(0, 2, 4, 2, 0, 1, 1, 1, 4, 3, 2, 3, 1, 2, 4),
        discarded_card_index=13,
        players=(
            Player(
                score=205,
                hand=(
                    Finger(card_index=13, is_revealed=True),
                    Finger(card_index=7, is_revealed=True),
                    Finger(card_index=3, is_revealed=True),
                    Finger(card_index=12, is_revealed=True),
                    Finger(card_index=5, is_revealed=True),
                    Finger(card_index=10, is_revealed=True),
                    Finger(card_index=12, is_revealed=True),
                    Finger(card_index=1, is_revealed=True),
                    Finger(card_index=12, is_revealed=True),
                    Finger(card_index=12, is_revealed=True),
                    Finger(card_index=9, is_revealed=True),
                    Finger(card_index=6, is_revealed=True),
                ),
            ),
            Player(
                score=153,
                hand=(
                    Finger(card_index=12, is_revealed=True),
                    Finger(card_index=14, is_revealed=True),
                    Finger(card_index=9, is_revealed=True),
                    Finger(card_index=10, is_revealed=True),
                    Finger(card_index=5, is_revealed=True),
                    Finger(card_index=7, is_revealed=True),
                    Finger(card_index=14, is_revealed=True),
                    Finger(card_index=14, is_revealed=True),
                    Finger(card_index=4, is_revealed=True),
                    Finger(card_index=7, is_revealed=True),
                    Finger(card_index=13, is_revealed=True),
                    Finger(card_index=2, is_revealed=True),
                ),
            ),
            Player(
                score=114,
                hand=(
                    Finger(card_index=2, is_revealed=True),
                    Finger(card_index=7, is_revealed=True),
                    Finger(card_index=5, is_revealed=True),
                    Finger(card_index=6, is_revealed=True),
                    Finger(card_index=9, is_revealed=True),
                    Finger(card_index=1, is_revealed=True),
                    Finger(card_index=0, is_revealed=True),
                    Finger(card_index=11, is_revealed=True),
                    Finger(card_index=4, is_revealed=True),
                    Finger(card_index=4, is_revealed=True),
                    Finger(card_index=6, is_revealed=True),
                    Finger(card_index=12, is_revealed=True),
                ),
            ),
        ),
        start_player_index=0,
        end_player_index=0,
    )
