from __future__ import annotations

import random

import skyjo2 as sj
import skyjo2.play as sj_play


class DrawPlayer(sj_play.Player):
    def play(self, game: sj.Game) -> sj_play.Action:
        if game.state == sj.GameState.REVEAL_SECOND_CARD:
            return (sj_play.ActionKind.REVEAL_SECOND_CARD, 1)  # same column
        elif game.state == sj.GameState.DRAW_OR_REPLACE_WITH_DISCARD:
            return (sj_play.ActionKind.DRAW_CARD,)
        elif game.state == sj.GameState.DISCARD_DRAW_AND_REVEAL_OR_REPLACE_WITH_DRAW:
            for i, finger in enumerate(game.player.hand):
                if finger.is_hidden:
                    return (sj_play.ActionKind.REPLACE_WITH_DRAW, i)
        assert False, f"unknown state {game}"


class StallPlayer(sj_play.Player):
    def play(self, game: sj.Game) -> sj_play.Action:
        if game.state == sj.GameState.REVEAL_SECOND_CARD:
            return (sj_play.ActionKind.REVEAL_SECOND_CARD, 1)  # same column
        elif game.state == sj.GameState.DRAW_OR_REPLACE_WITH_DISCARD:
            return (sj_play.ActionKind.REPLACE_WITH_DISCARD, 2)
        elif game.state == sj.GameState.DISCARD_DRAW_AND_REVEAL_OR_REPLACE_WITH_DRAW:
            assert False, f"unexpected state {game}"
        assert False, f"unknown state {game}"


def test_play_round_two(rng: random.Random) -> None:
    players = (DrawPlayer(), DrawPlayer())
    replay = list(sj_play.play(players, rng=rng, round_max=1))
    # 1 (new)
    # 1 (deal)
    # 2 (reveal second)
    # 2 players * 10 cards * 2 actions (draws + replaces)
    # 1 (reveal hidden)
    assert len(replay) == 45
    result = replay[-1]
    assert result == sj.Game(
        turn=22,
        end_turn=20,
        state=sj.GameState.DEAL_FIRST_CARDS,
        drawn_card_index=None,
        draw_pile=(4, 7, 11, 7, 7, 8, 7, 5, 6, 5, 7, 10, 7, 8, 6),
        discarded_card_index=6,
        discard_pile=(0, 2, 2, 2, 0, 1, 1, 2, 3, 3, 1, 0, 1, 0, 2),
        players=(
            sj.Player(
                index=0,
                score=59,
                hand=(
                    sj.Finger(card_index=10, is_revealed=True),
                    sj.Finger(card_index=6, is_revealed=True),
                    sj.Finger(card_index=12, is_revealed=True),
                    sj.Finger(card_index=9, is_revealed=True),
                    sj.Finger(card_index=7, is_revealed=True),
                    sj.Finger(card_index=4, is_revealed=True),
                    sj.Finger(card_index=9, is_revealed=True),
                    sj.Finger(card_index=5, is_revealed=True),
                    sj.Finger(card_index=10, is_revealed=True),
                    sj.Finger(card_index=7, is_revealed=True),
                    sj.Finger(card_index=4, is_revealed=True),
                    sj.Finger(card_index=0, is_revealed=True),
                ),
            ),
            sj.Player(
                index=1,
                score=69,
                hand=(
                    sj.Finger(card_index=1, is_revealed=True),
                    sj.Finger(card_index=13, is_revealed=True),
                    sj.Finger(card_index=7, is_revealed=True),
                    sj.Finger(card_index=14, is_revealed=True),
                    sj.Finger(card_index=2, is_revealed=True),
                    sj.Finger(card_index=2, is_revealed=True),
                    sj.Finger(card_index=8, is_revealed=True),
                    sj.Finger(card_index=4, is_revealed=True),
                    sj.Finger(card_index=3, is_revealed=True),
                    sj.Finger(card_index=14, is_revealed=True),
                    sj.Finger(card_index=13, is_revealed=True),
                    sj.Finger(card_index=12, is_revealed=True),
                ),
            ),
        ),
    )


def test_play_round_three(rng: random.Random) -> None:
    players = (DrawPlayer(), DrawPlayer(), DrawPlayer())
    replay = list(sj_play.play(players, rng=rng, round_max=1))
    # 1 (new)
    # 1 (deal)
    # 3 (reveal second)
    # 3 players * 10 cards * 2 actions (draws + replaces)
    # 1 (reveal hidden)
    assert len(replay) == 66
    result = replay[-1]
    assert result == sj.Game(
        turn=33,
        end_turn=30,
        state=sj.GameState.DEAL_FIRST_CARDS,
        drawn_card_index=None,
        draw_pile=(3, 6, 8, 6, 4, 7, 6, 5, 6, 4, 6, 6, 5, 8, 3),
        discarded_card_index=2,
        discard_pile=(0, 2, 3, 3, 1, 2, 2, 2, 3, 3, 1, 2, 1, 0, 5),
        players=(
            sj.Player(
                index=0,
                score=49,
                hand=(
                    sj.Finger(card_index=10, is_revealed=True),
                    sj.Finger(card_index=13, is_revealed=True),
                    sj.Finger(card_index=7, is_revealed=True),
                    sj.Finger(card_index=7, is_revealed=True),
                    sj.Finger(card_index=2, is_revealed=True),
                    sj.Finger(card_index=5, is_revealed=True),
                    sj.Finger(card_index=3, is_revealed=True),
                    sj.Finger(card_index=4, is_revealed=True),
                    sj.Finger(card_index=12, is_revealed=True),
                    sj.Finger(card_index=0, is_revealed=True),
                    sj.Finger(card_index=6, is_revealed=True),
                    sj.Finger(card_index=4, is_revealed=True),
                ),
            ),
            sj.Player(
                index=1,
                score=66,
                hand=(
                    sj.Finger(card_index=1, is_revealed=True),
                    sj.Finger(card_index=12, is_revealed=True),
                    sj.Finger(card_index=9, is_revealed=True),
                    sj.Finger(card_index=2, is_revealed=True),
                    sj.Finger(card_index=9, is_revealed=True),
                    sj.Finger(card_index=4, is_revealed=True),
                    sj.Finger(card_index=7, is_revealed=True),
                    sj.Finger(card_index=13, is_revealed=True),
                    sj.Finger(card_index=12, is_revealed=True),
                    sj.Finger(card_index=9, is_revealed=True),
                    sj.Finger(card_index=1, is_revealed=True),
                    sj.Finger(card_index=11, is_revealed=True),
                ),
            ),
            sj.Player(
                index=2,
                score=71,
                hand=(
                    sj.Finger(card_index=6, is_revealed=True),
                    sj.Finger(card_index=10, is_revealed=True),
                    sj.Finger(card_index=14, is_revealed=True),
                    sj.Finger(card_index=4, is_revealed=True),
                    sj.Finger(card_index=8, is_revealed=True),
                    sj.Finger(card_index=10, is_revealed=True),
                    sj.Finger(card_index=14, is_revealed=True),
                    sj.Finger(card_index=0, is_revealed=True),
                    sj.Finger(card_index=12, is_revealed=True),
                    sj.Finger(card_index=4, is_revealed=True),
                    sj.Finger(card_index=11, is_revealed=True),
                    sj.Finger(card_index=2, is_revealed=True),
                ),
            ),
        ),
    )


def test_play_round_forfeit(rng: random.Random) -> None:
    players = (DrawPlayer(), StallPlayer())
    replay = list(sj_play.play(players, rng=rng, no_progress_turn_max=5, round_max=1))
    # 1 (new)
    # 1 (deal)
    # 2 (reveal)
    # 1 player * 5 cards * 2 actions (draws + replaces)
    # 1 player * 5 actions (replace with discard)
    # 1 forefit
    # 2 actions (draw + replace in endgame)
    # 1 (reveal hidden cards)
    assert len(replay) == 25
    assert replay[-2] == sj.Game(
        turn=15,
        end_turn=13,
        state=sj.GameState.REVEAL_HIDDEN_CARDS,
        drawn_card_index=None,
        draw_pile=(5, 9, 13, 8, 9, 9, 9, 7, 9, 8, 8, 10, 8, 9, 9),
        discarded_card_index=2,
        discard_pile=(0, 0, 0, 0, 1, 0, 0, 2, 0, 2, 1, 0, 0, 0, 1),
        players=(
            sj.Player(
                index=1,
                score=0,
                hand=(
                    sj.Finger(card_index=1, is_revealed=True),
                    sj.Finger(card_index=13, is_revealed=True),
                    sj.Finger(card_index=2, is_revealed=True),
                    sj.Finger(card_index=None, is_revealed=False),
                    sj.Finger(card_index=None, is_revealed=False),
                    sj.Finger(card_index=None, is_revealed=False),
                    sj.Finger(card_index=None, is_revealed=False),
                    sj.Finger(card_index=None, is_revealed=False),
                    sj.Finger(card_index=None, is_revealed=False),
                    sj.Finger(card_index=None, is_revealed=False),
                    sj.Finger(card_index=None, is_revealed=False),
                    sj.Finger(card_index=None, is_revealed=False),
                ),
            ),
            sj.Player(
                index=0,
                score=0,
                hand=(
                    sj.Finger(card_index=10, is_revealed=True),
                    sj.Finger(card_index=6, is_revealed=True),
                    sj.Finger(card_index=12, is_revealed=True),
                    sj.Finger(card_index=12, is_revealed=True),
                    sj.Finger(card_index=5, is_revealed=True),
                    sj.Finger(card_index=3, is_revealed=True),
                    sj.Finger(card_index=3, is_revealed=True),
                    sj.Finger(card_index=7, is_revealed=True),
                    sj.Finger(card_index=8, is_revealed=True),
                    sj.Finger(card_index=None, is_revealed=False),
                    sj.Finger(card_index=None, is_revealed=False),
                    sj.Finger(card_index=None, is_revealed=False),
                ),
            ),
        ),
    )
    result = replay[-1]
    assert result == sj.Game(
        turn=15,
        end_turn=13,
        state=sj.GameState.DEAL_FIRST_CARDS,
        drawn_card_index=None,
        draw_pile=(5, 8, 12, 7, 8, 8, 8, 7, 8, 6, 7, 10, 8, 9, 7),
        discarded_card_index=2,
        discard_pile=(0, 0, 0, 0, 1, 0, 0, 2, 0, 2, 1, 0, 0, 0, 1),
        players=(
            sj.Player(
                index=1,
                score=100,
                hand=(
                    sj.Finger(card_index=1, is_revealed=True),
                    sj.Finger(card_index=13, is_revealed=True),
                    sj.Finger(card_index=2, is_revealed=True),
                    sj.Finger(card_index=2, is_revealed=True),
                    sj.Finger(card_index=9, is_revealed=True),
                    sj.Finger(card_index=14, is_revealed=True),
                    sj.Finger(card_index=8, is_revealed=True),
                    sj.Finger(card_index=1, is_revealed=True),
                    sj.Finger(card_index=5, is_revealed=True),
                    sj.Finger(card_index=6, is_revealed=True),
                    sj.Finger(card_index=4, is_revealed=True),
                    sj.Finger(card_index=9, is_revealed=True),
                ),
            ),
            sj.Player(
                index=0,
                score=69,
                hand=(
                    sj.Finger(card_index=10, is_revealed=True),
                    sj.Finger(card_index=6, is_revealed=True),
                    sj.Finger(card_index=12, is_revealed=True),
                    sj.Finger(card_index=12, is_revealed=True),
                    sj.Finger(card_index=5, is_revealed=True),
                    sj.Finger(card_index=3, is_revealed=True),
                    sj.Finger(card_index=3, is_revealed=True),
                    sj.Finger(card_index=7, is_revealed=True),
                    sj.Finger(card_index=8, is_revealed=True),
                    sj.Finger(card_index=10, is_revealed=True),
                    sj.Finger(card_index=14, is_revealed=True),
                    sj.Finger(card_index=3, is_revealed=True),
                ),
            ),
        ),
    )


def test_play_game_three(rng: random.Random) -> None:
    players = (DrawPlayer(), DrawPlayer(), DrawPlayer())
    replay = list(sj_play.play(players, rng=rng))
    # 1 (new)
    # ----------- x2 after this point for 2 rounds
    # 1 (deal)
    # 3 (reveal second)
    # 3 players * 10 cards * 2 actions (draws + replaces)
    # 1 (reveal hidden)
    assert len(replay) == 1 + 65 * 2
    result = replay[-1]
    assert result == sj.Game(
        turn=33,
        end_turn=30,
        state=sj.GameState.DEAL_FIRST_CARDS,
        drawn_card_index=None,
        draw_pile=(1, 2, 0, 2, 0, 1, 1, 1, 1, 1, 1, 0, 2, 3, 0),
        discarded_card_index=4,
        discard_pile=(1, 4, 5, 4, 3, 6, 4, 4, 4, 6, 3, 5, 2, 3, 6),
        players=(
            sj.Player(
                index=1,
                score=236,
                hand=(
                    sj.Finger(card_index=11, is_revealed=True),
                    sj.Finger(card_index=13, is_revealed=True),
                    sj.Finger(card_index=2, is_revealed=True),
                    sj.Finger(card_index=7, is_revealed=True),
                    sj.Finger(card_index=8, is_revealed=True),
                    sj.Finger(card_index=10, is_revealed=True),
                    sj.Finger(card_index=12, is_revealed=True),
                    sj.Finger(card_index=14, is_revealed=True),
                    sj.Finger(card_index=12, is_revealed=True),
                    sj.Finger(card_index=13, is_revealed=True),
                    sj.Finger(card_index=7, is_revealed=True),
                    sj.Finger(card_index=0, is_revealed=True),
                ),
            ),
            sj.Player(
                index=2,
                score=108,
                hand=(
                    sj.Finger(card_index=2, is_revealed=True),
                    sj.Finger(card_index=6, is_revealed=True),
                    sj.Finger(card_index=8, is_revealed=True),
                    sj.Finger(card_index=2, is_revealed=True),
                    sj.Finger(card_index=2, is_revealed=True),
                    sj.Finger(card_index=4, is_revealed=True),
                    sj.Finger(card_index=3, is_revealed=True),
                    sj.Finger(card_index=14, is_revealed=True),
                    sj.Finger(card_index=5, is_revealed=True),
                    sj.Finger(card_index=2, is_revealed=True),
                    sj.Finger(card_index=10, is_revealed=True),
                    sj.Finger(card_index=3, is_revealed=True),
                ),
            ),
            sj.Player(
                index=0,
                score=97,
                hand=(
                    sj.Finger(card_index=11, is_revealed=True),
                    sj.Finger(card_index=6, is_revealed=True),
                    sj.Finger(card_index=5, is_revealed=True),
                    sj.Finger(card_index=8, is_revealed=True),
                    sj.Finger(card_index=2, is_revealed=True),
                    sj.Finger(card_index=1, is_revealed=True),
                    sj.Finger(card_index=1, is_revealed=True),
                    sj.Finger(card_index=11, is_revealed=True),
                    sj.Finger(card_index=6, is_revealed=True),
                    sj.Finger(card_index=8, is_revealed=True),
                    sj.Finger(card_index=3, is_revealed=True),
                    sj.Finger(card_index=10, is_revealed=True),
                ),
            ),
        ),
    )
