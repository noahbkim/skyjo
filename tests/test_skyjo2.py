from __future__ import annotations

import random

import pytest

import skyjo2 as sj


@pytest.fixture
def rng() -> random.Random:
    return random.Random(0)


class TestGame:
    def test_new(self) -> None:
        game = sj.Game.new(players=2)
        assert game == sj.Game(
            turn=0,
            action=None,
            drawn_card_index=None,
            draw_pile=(5, 10, 15, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10),
            discarded_card_index=None,
            discard_pile=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            players=(
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=None, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=True),
                    ),
                    last_novel_turn=0,
                ),
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=None, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=True),
                    ),
                    last_novel_turn=0,
                ),
            ),
        )

    def test_deal(self, rng: random.Random) -> None:
        game = sj.Game.new(players=2)
        game = game.with_random_deal(rng=rng)
        assert game == sj.Game(
            turn=0,
            action=None,
            drawn_card_index=None,
            draw_pile=(5, 9, 11, 8, 9, 8, 9, 7, 9, 7, 8, 10, 8, 9, 8),
            discarded_card_index=9,
            discard_pile=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            players=(
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=10, is_revealed=True),
                        sj.Finger(card_index=1, is_revealed=False),
                        sj.Finger(card_index=6, is_revealed=False),
                        sj.Finger(card_index=13, is_revealed=False),
                        sj.Finger(card_index=12, is_revealed=False),
                        sj.Finger(card_index=10, is_revealed=False),
                        sj.Finger(card_index=7, is_revealed=False),
                        sj.Finger(card_index=12, is_revealed=False),
                        sj.Finger(card_index=9, is_revealed=False),
                        sj.Finger(card_index=5, is_revealed=False),
                        sj.Finger(card_index=14, is_revealed=False),
                        sj.Finger(card_index=3, is_revealed=False),
                    ),
                    last_novel_turn=0,
                ),
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=7, is_revealed=True),
                        sj.Finger(card_index=3, is_revealed=False),
                        sj.Finger(card_index=2, is_revealed=False),
                        sj.Finger(card_index=7, is_revealed=False),
                        sj.Finger(card_index=4, is_revealed=False),
                        sj.Finger(card_index=8, is_revealed=False),
                        sj.Finger(card_index=2, is_revealed=False),
                        sj.Finger(card_index=2, is_revealed=False),
                        sj.Finger(card_index=9, is_revealed=False),
                        sj.Finger(card_index=14, is_revealed=False),
                        sj.Finger(card_index=2, is_revealed=False),
                        sj.Finger(card_index=5, is_revealed=False),
                    ),
                    last_novel_turn=0,
                ),
            ),
        )

    def test_with_second_card_revealed(self, rng: random.Random) -> None:
        game = sj.Game.new(players=2)
        game = game.with_random_deal(rng=rng)
        game = game.with_second_card_revealed(1)
        assert game == sj.Game(
            turn=0,
            action=sj.Action.REVEAL_SECOND_CARD,
            drawn_card_index=None,
            draw_pile=(5, 9, 11, 8, 9, 8, 9, 7, 9, 7, 8, 10, 8, 9, 8),
            discarded_card_index=9,
            discard_pile=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            players=(
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=7, is_revealed=True),
                        sj.Finger(card_index=3, is_revealed=False),
                        sj.Finger(card_index=2, is_revealed=False),
                        sj.Finger(card_index=7, is_revealed=False),
                        sj.Finger(card_index=4, is_revealed=False),
                        sj.Finger(card_index=8, is_revealed=False),
                        sj.Finger(card_index=2, is_revealed=False),
                        sj.Finger(card_index=2, is_revealed=False),
                        sj.Finger(card_index=9, is_revealed=False),
                        sj.Finger(card_index=14, is_revealed=False),
                        sj.Finger(card_index=2, is_revealed=False),
                        sj.Finger(card_index=5, is_revealed=False),
                    ),
                    last_novel_turn=0,
                ),
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=10, is_revealed=True),
                        sj.Finger(card_index=1, is_revealed=True),
                        sj.Finger(card_index=6, is_revealed=False),
                        sj.Finger(card_index=13, is_revealed=False),
                        sj.Finger(card_index=12, is_revealed=False),
                        sj.Finger(card_index=10, is_revealed=False),
                        sj.Finger(card_index=7, is_revealed=False),
                        sj.Finger(card_index=12, is_revealed=False),
                        sj.Finger(card_index=9, is_revealed=False),
                        sj.Finger(card_index=5, is_revealed=False),
                        sj.Finger(card_index=14, is_revealed=False),
                        sj.Finger(card_index=3, is_revealed=False),
                    ),
                    last_novel_turn=0,
                ),
            ),
        )
        game = game.with_second_card_revealed(sj.HAND_ROWS)
        assert game == sj.Game(
            turn=0,
            action=sj.Action.REVEAL_SECOND_CARD,
            drawn_card_index=None,
            draw_pile=(5, 9, 11, 8, 9, 8, 9, 7, 9, 7, 8, 10, 8, 9, 8),
            discarded_card_index=9,
            discard_pile=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            players=(
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=10, is_revealed=True),
                        sj.Finger(card_index=1, is_revealed=True),
                        sj.Finger(card_index=6, is_revealed=False),
                        sj.Finger(card_index=13, is_revealed=False),
                        sj.Finger(card_index=12, is_revealed=False),
                        sj.Finger(card_index=10, is_revealed=False),
                        sj.Finger(card_index=7, is_revealed=False),
                        sj.Finger(card_index=12, is_revealed=False),
                        sj.Finger(card_index=9, is_revealed=False),
                        sj.Finger(card_index=5, is_revealed=False),
                        sj.Finger(card_index=14, is_revealed=False),
                        sj.Finger(card_index=3, is_revealed=False),
                    ),
                    last_novel_turn=0,
                ),
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=7, is_revealed=True),
                        sj.Finger(card_index=3, is_revealed=False),
                        sj.Finger(card_index=2, is_revealed=False),
                        sj.Finger(card_index=7, is_revealed=True),
                        sj.Finger(card_index=4, is_revealed=False),
                        sj.Finger(card_index=8, is_revealed=False),
                        sj.Finger(card_index=2, is_revealed=False),
                        sj.Finger(card_index=2, is_revealed=False),
                        sj.Finger(card_index=9, is_revealed=False),
                        sj.Finger(card_index=14, is_revealed=False),
                        sj.Finger(card_index=2, is_revealed=False),
                        sj.Finger(card_index=5, is_revealed=False),
                    ),
                    last_novel_turn=0,
                ),
            ),
        )

    def test_with_random_drawn_card(self, rng: random.Random) -> None:
        game = sj.Game.new(players=2)
        game = game.with_random_deal(rng=rng)
        game = game.with_second_card_revealed(1)
        game = game.with_second_card_revealed(sj.HAND_ROWS)
        game = game.with_random_drawn_card(rng=rng)
        assert game == sj.Game(
            turn=0,
            action=sj.Action.DRAW_CARD,
            drawn_card_index=6,
            draw_pile=(5, 9, 11, 8, 9, 8, 8, 7, 9, 7, 8, 10, 8, 9, 8),
            discarded_card_index=9,
            discard_pile=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            players=(
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=10, is_revealed=True),
                        sj.Finger(card_index=1, is_revealed=True),
                        sj.Finger(card_index=6, is_revealed=False),
                        sj.Finger(card_index=13, is_revealed=False),
                        sj.Finger(card_index=12, is_revealed=False),
                        sj.Finger(card_index=10, is_revealed=False),
                        sj.Finger(card_index=7, is_revealed=False),
                        sj.Finger(card_index=12, is_revealed=False),
                        sj.Finger(card_index=9, is_revealed=False),
                        sj.Finger(card_index=5, is_revealed=False),
                        sj.Finger(card_index=14, is_revealed=False),
                        sj.Finger(card_index=3, is_revealed=False),
                    ),
                    last_novel_turn=0,
                ),
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=7, is_revealed=True),
                        sj.Finger(card_index=3, is_revealed=False),
                        sj.Finger(card_index=2, is_revealed=False),
                        sj.Finger(card_index=7, is_revealed=True),
                        sj.Finger(card_index=4, is_revealed=False),
                        sj.Finger(card_index=8, is_revealed=False),
                        sj.Finger(card_index=2, is_revealed=False),
                        sj.Finger(card_index=2, is_revealed=False),
                        sj.Finger(card_index=9, is_revealed=False),
                        sj.Finger(card_index=14, is_revealed=False),
                        sj.Finger(card_index=2, is_revealed=False),
                        sj.Finger(card_index=5, is_revealed=False),
                    ),
                    last_novel_turn=0,
                ),
            ),
        )

    def with_draw_discarded_and_card_revealed(self, rng: random.Random) -> None:
        game = sj.Game.new(players=2)
        game = game.with_random_deal(rng=rng)
        game = game.with_second_card_revealed(1)
        game = game.with_second_card_revealed(sj.HAND_ROWS)
        game = game.with_random_drawn_card(rng=rng)
        game = game.with_draw_discarded_and_card_revealed(2)
        assert game == sj.Game(
            turn=0,
            action=sj.Action.DRAW_CARD,
            drawn_card_index=None,
            draw_pile=(5, 9, 11, 8, 9, 8, 8, 7, 9, 7, 8, 10, 8, 9, 8),
            discarded_card_index=6,
            discard_pile=(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
            players=(
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=7, is_revealed=True),
                        sj.Finger(card_index=3, is_revealed=False),
                        sj.Finger(card_index=2, is_revealed=False),
                        sj.Finger(card_index=7, is_revealed=True),
                        sj.Finger(card_index=4, is_revealed=False),
                        sj.Finger(card_index=8, is_revealed=False),
                        sj.Finger(card_index=2, is_revealed=False),
                        sj.Finger(card_index=2, is_revealed=False),
                        sj.Finger(card_index=9, is_revealed=False),
                        sj.Finger(card_index=14, is_revealed=False),
                        sj.Finger(card_index=2, is_revealed=False),
                        sj.Finger(card_index=5, is_revealed=False),
                    ),
                    last_novel_turn=0,
                ),
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=10, is_revealed=True),
                        sj.Finger(card_index=1, is_revealed=True),
                        sj.Finger(card_index=6, is_revealed=True),
                        sj.Finger(card_index=13, is_revealed=False),
                        sj.Finger(card_index=12, is_revealed=False),
                        sj.Finger(card_index=10, is_revealed=False),
                        sj.Finger(card_index=7, is_revealed=False),
                        sj.Finger(card_index=12, is_revealed=False),
                        sj.Finger(card_index=9, is_revealed=False),
                        sj.Finger(card_index=5, is_revealed=False),
                        sj.Finger(card_index=14, is_revealed=False),
                        sj.Finger(card_index=3, is_revealed=False),
                    ),
                    last_novel_turn=0,
                ),
            ),
        )

    def with_draw_discarded_and_card_revealed(self, rng: random.Random) -> None:
        game = sj.Game.new()

    def test_with_card_replaced_with_draw(self, rng: random.Random) -> None:
        game = sj.Game.new(players=2)
        game = game.with_random_deal(rng=rng)
        game = game.with_second_card_revealed(1)
        game = game.with_second_card_revealed(sj.HAND_ROWS)
        game = game.with_random_drawn_card(rng=rng)
        game = game.with_card_replaced_with_draw(3)
        assert game == sj.Game(
            turn=1,
            action=sj.Action.DISCARD_DRAW_AND_REVEAL_CARD,
            drawn_card_index=None,
            draw_pile=(5, 9, 11, 8, 9, 8, 8, 7, 9, 7, 8, 10, 8, 9, 8),
            discarded_card_index=13,
            discard_pile=(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
            players=(
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=7, is_revealed=True),
                        sj.Finger(card_index=3, is_revealed=False),
                        sj.Finger(card_index=2, is_revealed=False),
                        sj.Finger(card_index=7, is_revealed=True),
                        sj.Finger(card_index=4, is_revealed=False),
                        sj.Finger(card_index=8, is_revealed=False),
                        sj.Finger(card_index=2, is_revealed=False),
                        sj.Finger(card_index=2, is_revealed=False),
                        sj.Finger(card_index=9, is_revealed=False),
                        sj.Finger(card_index=14, is_revealed=False),
                        sj.Finger(card_index=2, is_revealed=False),
                        sj.Finger(card_index=5, is_revealed=False),
                    ),
                    last_novel_turn=0,
                ),
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=10, is_revealed=True),
                        sj.Finger(card_index=1, is_revealed=True),
                        sj.Finger(card_index=6, is_revealed=False),
                        sj.Finger(card_index=6, is_revealed=True),
                        sj.Finger(card_index=12, is_revealed=False),
                        sj.Finger(card_index=10, is_revealed=False),
                        sj.Finger(card_index=7, is_revealed=False),
                        sj.Finger(card_index=12, is_revealed=False),
                        sj.Finger(card_index=9, is_revealed=False),
                        sj.Finger(card_index=5, is_revealed=False),
                        sj.Finger(card_index=14, is_revealed=False),
                        sj.Finger(card_index=3, is_revealed=False),
                    ),
                    last_novel_turn=0,
                ),
            ),
        )
