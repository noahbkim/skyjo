from __future__ import annotations

import random

import skyjo2 as sj
import skyjo2.play as sj_play


class DrawPlayer(sj_play.Player):
    def play(self, game: sj.Game) -> sj_play.Action:
        if game.state == sj.State.REVEAL_SECOND_CARD:
            return sj_play.RevealSecondCard(1)  # same column
        elif game.state == sj.State.DRAW_OR_REPLACE_WITH_DISCARD:
            return sj_play.DrawCard()
        elif game.state == sj.State.DISCARD_DRAW_AND_REVEAL_OR_REPLACE_WITH_DRAW:
            for i, finger in enumerate(game.player.hand):
                if finger.is_hidden:
                    return sj_play.ReplaceCardWithDraw(i)
        assert False, f"unknown state {game}"


class StallPlayer(sj_play.Player):
    def play(self, game: sj.Game) -> sj_play.Action:
        if game.state == sj.State.REVEAL_SECOND_CARD:
            return sj_play.RevealSecondCard(1)  # same column
        elif game.state == sj.State.DRAW_OR_REPLACE_WITH_DISCARD:
            return sj_play.ReplaceCardWithDiscard(2)
        elif game.state == sj.State.DISCARD_DRAW_AND_REVEAL_OR_REPLACE_WITH_DRAW:
            assert False, f"unexpected state {game}"
        assert False, f"unknown state {game}"


def test_play_two(rng: random.Random) -> None:
    replay = list(sj_play.play((DrawPlayer(), DrawPlayer()), rng=rng))
    # 1 (new)
    # 1 (deal)
    # 2 (reveal)
    # 2 players * 10 cards * 2 actions (draws + replaces)
    assert len(replay) == 44
    result = replay[-1]
    assert result == sj.Game(
        turn=23,
        end_turn=21,
        state=sj.State.ENDED_BY_REVEAL,
        drawn_card_index=None,
        draw_pile=(4, 7, 11, 7, 7, 8, 7, 5, 6, 5, 7, 10, 7, 8, 6),
        discarded_card_index=6,
        discard_pile=(0, 2, 2, 2, 0, 1, 1, 2, 3, 3, 1, 0, 1, 0, 2),
        players=(
            sj.Player(
                score=0,
                hand=(
                    sj.Finger(card_index=1, is_revealed=True),
                    sj.Finger(card_index=13, is_revealed=True),
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
                score=0,
                hand=(
                    sj.Finger(card_index=10, is_revealed=True),
                    sj.Finger(card_index=6, is_revealed=True),
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
    assert result.players[0].hand_score == 57
    assert result.players[1].hand_score == 71
    assert result.final_scores == (57, 71)


def test_play_three(rng: random.Random) -> None:
    replay = list(sj_play.play((DrawPlayer(), DrawPlayer(), DrawPlayer()), rng=rng))
    # 1 (new)
    # 1 (deal)
    # 3 (reveal)
    # 3 players * 10 cards * 2 actions (draws + replaces)
    assert len(replay) == 65
    result = replay[-1]
    assert result == sj.Game(
        turn=34,
        end_turn=31,
        state=sj.State.ENDED_BY_REVEAL,
        drawn_card_index=None,
        draw_pile=(3, 6, 8, 6, 4, 7, 6, 5, 6, 4, 6, 6, 5, 8, 3),
        discarded_card_index=2,
        discard_pile=(0, 2, 3, 3, 1, 2, 2, 2, 3, 3, 1, 2, 1, 0, 5),
        players=(
            sj.Player(
                score=0,
                hand=(
                    sj.Finger(card_index=1, is_revealed=True),
                    sj.Finger(card_index=12, is_revealed=True),
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
                score=0,
                hand=(
                    sj.Finger(card_index=6, is_revealed=True),
                    sj.Finger(card_index=10, is_revealed=True),
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
                score=0,
                hand=(
                    sj.Finger(card_index=10, is_revealed=True),
                    sj.Finger(card_index=13, is_revealed=True),
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
    assert result.players[0].hand_score == 39
    assert result.players[1].hand_score == 69
    assert result.players[2].hand_score == 78
    assert result.final_scores == (39, 69, 78)


def test_play_forfeit(rng: random.Random) -> None:
    replay = list(
        sj_play.play(
            (DrawPlayer(), StallPlayer()),
            rng=rng,
            no_progress_turn_max=5,
        )
    )
    # 1 (new)
    # 1 (deal)
    # 2 (reveal)
    # 1 player * 5 cards * 2 actions (draws + replaces)
    # 1 player * 5 actions (replace with discard)
    # 1 forefit
    # 2 actions (draw + replace in endgame)
    # 1 (reveal hidden cards)
    assert len(replay) == 23
    assert replay[-2] == sj.Game(
        turn=15,
        end_turn=13,
        state=sj.State.ENDED_BY_FORFEIT,
        drawn_card_index=None,
        draw_pile=(5, 9, 14, 8, 9, 9, 9, 7, 10, 8, 8, 10, 8, 9, 9),
        discarded_card_index=4,
        discard_pile=(0, 0, 1, 0, 0, 0, 0, 1, 0, 2, 0, 0, 1, 0, 1),
        players=(
            sj.Player(  # staller
                score=0,
                hand=(
                    sj.Finger(card_index=1, is_revealed=True),
                    sj.Finger(card_index=13, is_revealed=True),
                    sj.Finger(card_index=7, is_revealed=True),
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
                score=0,
                hand=(
                    sj.Finger(card_index=10, is_revealed=True),
                    sj.Finger(card_index=6, is_revealed=True),
                    sj.Finger(card_index=10, is_revealed=True),  # goes first
                    sj.Finger(card_index=12, is_revealed=True),  # 1
                    sj.Finger(card_index=5, is_revealed=True),  # 2
                    sj.Finger(card_index=3, is_revealed=True),  # 3
                    sj.Finger(card_index=3, is_revealed=True),  # 4
                    sj.Finger(card_index=7, is_revealed=True),  # 5
                    sj.Finger(card_index=None, is_revealed=False),
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
        state=sj.State.ENDED_BY_FORFEIT,
        drawn_card_index=None,
        draw_pile=(5, 8, 12, 8, 8, 8, 8, 7, 8, 6, 7, 10, 8, 9, 7),
        discarded_card_index=4,
        discard_pile=(0, 0, 1, 0, 0, 0, 0, 1, 0, 2, 0, 0, 1, 0, 1),
        players=(
            sj.Player(
                score=0,
                hand=(
                    sj.Finger(card_index=1, is_revealed=True),
                    sj.Finger(card_index=13, is_revealed=True),
                    sj.Finger(card_index=7, is_revealed=True),
                    sj.Finger(card_index=8, is_revealed=True),
                    sj.Finger(card_index=2, is_revealed=True),
                    sj.Finger(card_index=2, is_revealed=True),
                    sj.Finger(card_index=9, is_revealed=True),
                    sj.Finger(card_index=14, is_revealed=True),
                    sj.Finger(card_index=8, is_revealed=True),
                    sj.Finger(card_index=1, is_revealed=True),
                    sj.Finger(card_index=5, is_revealed=True),
                    sj.Finger(card_index=6, is_revealed=True),
                ),
            ),
            sj.Player(
                score=0,
                hand=(
                    sj.Finger(card_index=10, is_revealed=True),
                    sj.Finger(card_index=6, is_revealed=True),
                    sj.Finger(card_index=10, is_revealed=True),
                    sj.Finger(card_index=12, is_revealed=True),
                    sj.Finger(card_index=5, is_revealed=True),
                    sj.Finger(card_index=3, is_revealed=True),
                    sj.Finger(card_index=3, is_revealed=True),
                    sj.Finger(card_index=7, is_revealed=True),
                    sj.Finger(card_index=4, is_revealed=True),
                    sj.Finger(card_index=9, is_revealed=True),
                    sj.Finger(card_index=10, is_revealed=True),
                    sj.Finger(card_index=14, is_revealed=True),
                ),
            ),
        ),
    )
    assert result.players[0].hand_score == 52
    assert result.players[1].hand_score == 69
    assert result.final_scores == (52 * 2, 69)  # forefit is always *2, even if lowest
