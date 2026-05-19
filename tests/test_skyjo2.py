from __future__ import annotations

import random

import skyjo2 as sj


class TestGame:
    def test_new(self) -> None:
        game = sj.Game.new(players=2)
        assert game == sj.Game(
            turn=0,
            state=sj.State.NULL,
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
                ),
            ),
        )

    def test_with_deal(self) -> None:
        game = sj.Game.new(players=2)
        game = game.with_deal((0, 1, 2))
        assert game == sj.Game(
            turn=0,
            state=sj.State.REVEAL_SECOND_CARD,
            drawn_card_index=None,
            draw_pile=(
                5 - 1,
                10 - 1,
                15 - 1,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
            ),
            discarded_card_index=0,
            discard_pile=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            players=(
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=1, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=False),
                        sj.Finger(card_index=None, is_revealed=False),
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
                        sj.Finger(card_index=None, is_revealed=False),
                        sj.Finger(card_index=None, is_revealed=False),
                    ),
                ),
            ),
        )

    def test_with_random_deal(self, rng: random.Random) -> None:
        game = sj.Game.new(players=2)
        game = game.with_random_deal(rng=rng)
        assert game == sj.Game(
            turn=0,
            state=sj.State.REVEAL_SECOND_CARD,
            drawn_card_index=None,
            draw_pile=(5, 9, 15, 10, 10, 10, 10, 10, 10, 9, 9, 10, 10, 10, 10),
            discarded_card_index=9,
            discard_pile=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            players=(
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=10, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=False),
                        sj.Finger(card_index=None, is_revealed=False),
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
                        sj.Finger(card_index=1, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=False),
                        sj.Finger(card_index=None, is_revealed=False),
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
            ),
        )

    def test_with_second_card_revealed(self, rng: random.Random) -> None:
        game = sj.Game.new(players=2)
        game = game.with_random_deal(rng=rng)
        game = game.with_random_second_card_revealed(1, rng=rng)
        assert game == sj.Game(
            turn=1,
            state=sj.State.REVEAL_SECOND_CARD,
            drawn_card_index=None,
            draw_pile=(5, 9, 15, 10, 10, 10, 9, 10, 10, 9, 9, 10, 10, 10, 10),
            discarded_card_index=9,
            discard_pile=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            players=(
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=1, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=False),
                        sj.Finger(card_index=None, is_revealed=False),
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
                        sj.Finger(card_index=None, is_revealed=False),
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
            ),
        )
        game = game.with_random_second_card_revealed(sj.HAND_ROWS, rng=rng)
        assert game == sj.Game(
            turn=3,  # skip one as lowest player gets to start
            state=sj.State.DRAW_OR_REPLACE_WITH_DISCARD,
            drawn_card_index=None,
            draw_pile=(5, 9, 15, 10, 10, 10, 9, 10, 10, 9, 9, 10, 10, 9, 10),
            discarded_card_index=9,
            discard_pile=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            players=(
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=1, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=False),
                        sj.Finger(card_index=None, is_revealed=False),
                        sj.Finger(card_index=13, is_revealed=True),
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
                        sj.Finger(card_index=None, is_revealed=False),
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
            ),
        )

    def test_with_random_drawn_card(self, rng: random.Random) -> None:
        game = sj.Game.new(players=2)
        game = game.with_random_deal(rng=rng)
        game = game.with_random_second_card_revealed(1, rng=rng)
        game = game.with_random_second_card_revealed(sj.HAND_ROWS, rng=rng)
        game = game.with_random_drawn_card(rng=rng)
        assert game == sj.Game(
            turn=3,
            state=sj.State.DISCARD_DRAW_AND_REVEAL_OR_REPLACE_WITH_DRAW,
            drawn_card_index=12,
            draw_pile=(5, 9, 15, 10, 10, 10, 9, 10, 10, 9, 9, 10, 9, 9, 10),
            discarded_card_index=9,
            discard_pile=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            players=(
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=1, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=False),
                        sj.Finger(card_index=None, is_revealed=False),
                        sj.Finger(card_index=13, is_revealed=True),
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
                        sj.Finger(card_index=None, is_revealed=False),
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
            ),
        )

    def test_with_draw_discarded_and_card_revealed(self, rng: random.Random) -> None:
        game = sj.Game.new(players=2)
        game = game.with_random_deal(rng=rng)
        game = game.with_random_second_card_revealed(1, rng=rng)
        game = game.with_random_second_card_revealed(sj.HAND_ROWS, rng=rng)
        game = game.with_random_drawn_card(rng=rng)
        game = game.with_draw_discarded_and_random_card_revealed(1, rng=rng)
        assert game == sj.Game(
            turn=4,
            state=sj.State.DRAW_OR_REPLACE_WITH_DISCARD,
            drawn_card_index=None,
            draw_pile=(5, 9, 15, 10, 10, 10, 9, 10, 10, 9, 8, 10, 9, 9, 10),
            discarded_card_index=12,
            discard_pile=(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
            players=(
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=10, is_revealed=True),
                        sj.Finger(card_index=6, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=False),
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
                        sj.Finger(card_index=1, is_revealed=True),
                        sj.Finger(card_index=10, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=False),
                        sj.Finger(card_index=13, is_revealed=True),
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
            ),
        )

    def test_with_draw_discarded_and_card_revealed_clear(self) -> None:
        game = sj.Game.new(players=2)
        game = game.with_deal((0, 1, 2))
        game = game.with_second_card_revealed(1, 1)  # player 1
        game = game.with_second_card_revealed(1, 2)  # player 2
        game = game.with_drawn_card(0)  # player 1
        game = game.with_draw_discarded_and_card_revealed(2, 1)  # player 1
        assert game == sj.Game(
            turn=3,
            state=sj.State.DRAW_OR_REPLACE_WITH_DISCARD,
            drawn_card_index=None,
            draw_pile=(
                5 - 2,  # initial discard, discarded draw
                10 - 3,  # deal, reveal, reveal
                15 - 2,  # deal, reveal
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
            ),
            discarded_card_index=1,
            discard_pile=(2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            players=(
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=2, is_revealed=True),
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
                        sj.Finger(card_index=None, is_revealed=False),
                    ),
                ),
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=None, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=True),
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
            ),
        )

    def test_with_draw_discarded_and_card_revealed_end(self) -> None:
        game = sj.Game(
            turn=100,
            state=sj.State.DISCARD_DRAW_AND_REVEAL_OR_REPLACE_WITH_DRAW,
            drawn_card_index=0,
            draw_pile=(
                5 - 2,  # initial discard, draw
                10 - 1,
                15 - 2,
                10 - 2,
                10 - 2,
                10 - 2,
                10 - 2,
                10 - 2,
                10 - 2,
                10 - 2,
                10 - 2,
                10 - 2,
                10 - 1,
                10,
                10,
            ),
            discarded_card_index=0,
            discard_pile=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            players=(
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=1, is_revealed=True),
                        sj.Finger(card_index=2, is_revealed=True),
                        sj.Finger(card_index=3, is_revealed=True),
                        sj.Finger(card_index=4, is_revealed=True),
                        sj.Finger(card_index=5, is_revealed=True),
                        sj.Finger(card_index=6, is_revealed=True),
                        sj.Finger(card_index=7, is_revealed=True),
                        sj.Finger(card_index=8, is_revealed=True),
                        sj.Finger(card_index=9, is_revealed=True),
                        sj.Finger(card_index=10, is_revealed=True),
                        sj.Finger(card_index=11, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=False),
                    ),
                ),
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=12, is_revealed=True),
                        sj.Finger(card_index=11, is_revealed=True),
                        sj.Finger(card_index=10, is_revealed=True),
                        sj.Finger(card_index=9, is_revealed=True),
                        sj.Finger(card_index=8, is_revealed=True),
                        sj.Finger(card_index=7, is_revealed=True),
                        sj.Finger(card_index=6, is_revealed=True),
                        sj.Finger(card_index=5, is_revealed=True),
                        sj.Finger(card_index=4, is_revealed=True),
                        sj.Finger(card_index=3, is_revealed=True),
                        sj.Finger(card_index=2, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=False),
                    ),
                ),
            ),
        )
        game = game.with_draw_discarded_and_card_revealed(11, 12)
        assert game == sj.Game(
            turn=101,
            state=sj.State.NULL,
            drawn_card_index=None,
            draw_pile=(
                5 - 2,
                10 - 1,
                15 - 2,
                10 - 2,
                10 - 2,
                10 - 2,
                10 - 2,
                10 - 2,
                10 - 2,
                10 - 2,
                10 - 2,
                10 - 2,
                10 - 2,
                10,
                10,
            ),
            discarded_card_index=0,
            discard_pile=(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            players=(
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=1, is_revealed=True),
                        sj.Finger(card_index=2, is_revealed=True),
                        sj.Finger(card_index=3, is_revealed=True),
                        sj.Finger(card_index=4, is_revealed=True),
                        sj.Finger(card_index=5, is_revealed=True),
                        sj.Finger(card_index=6, is_revealed=True),
                        sj.Finger(card_index=7, is_revealed=True),
                        sj.Finger(card_index=8, is_revealed=True),
                        sj.Finger(card_index=9, is_revealed=True),
                        sj.Finger(card_index=10, is_revealed=True),
                        sj.Finger(card_index=11, is_revealed=True),
                        sj.Finger(card_index=12, is_revealed=True),
                    ),
                ),
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=12, is_revealed=True),
                        sj.Finger(card_index=11, is_revealed=True),
                        sj.Finger(card_index=10, is_revealed=True),
                        sj.Finger(card_index=9, is_revealed=True),
                        sj.Finger(card_index=8, is_revealed=True),
                        sj.Finger(card_index=7, is_revealed=True),
                        sj.Finger(card_index=6, is_revealed=True),
                        sj.Finger(card_index=5, is_revealed=True),
                        sj.Finger(card_index=4, is_revealed=True),
                        sj.Finger(card_index=3, is_revealed=True),
                        sj.Finger(card_index=2, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=False),
                    ),
                ),
            ),
        )

    def test_with_card_replaced_with_draw(self, rng: random.Random) -> None:
        game = sj.Game.new(players=2)
        game = game.with_random_deal(rng=rng)
        game = game.with_random_second_card_revealed(1, rng=rng)
        game = game.with_random_second_card_revealed(sj.HAND_ROWS, rng=rng)
        game = game.with_random_drawn_card(rng=rng)
        game = game.with_random_card_replaced_with_draw(1, rng=rng)
        assert game == sj.Game(
            turn=4,
            state=sj.State.DRAW_OR_REPLACE_WITH_DISCARD,
            drawn_card_index=None,
            draw_pile=(5, 9, 15, 10, 10, 10, 9, 10, 10, 9, 8, 10, 9, 9, 10),
            discarded_card_index=10,
            discard_pile=(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
            players=(
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=10, is_revealed=True),
                        sj.Finger(card_index=6, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=False),
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
                        sj.Finger(card_index=1, is_revealed=True),
                        sj.Finger(card_index=12, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=False),
                        sj.Finger(card_index=13, is_revealed=True),
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
            ),
        )

    def test_with_card_replaced_with_draw_clear(self) -> None:
        game = sj.Game.new(players=2)
        game = game.with_deal((0, 1, 2))
        game = game.with_second_card_revealed(1, 1)  # player 1
        game = game.with_second_card_revealed(1, 2)  # player 2
        game = game.with_drawn_card(1)  # player 1
        game = game.with_card_replaced_with_draw(2, 0)  # player 1
        assert game == sj.Game(
            turn=3,
            state=sj.State.DRAW_OR_REPLACE_WITH_DISCARD,
            drawn_card_index=None,
            draw_pile=(
                5 - 2,  # initial discard, replaced card
                10 - 3,
                15 - 2,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
            ),
            discarded_card_index=1,
            discard_pile=(2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            players=(
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=2, is_revealed=True),
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
                        sj.Finger(card_index=None, is_revealed=False),
                    ),
                ),
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=None, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=True),
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
            ),
        )

    def test_with_card_replaced_with_draw_end(self) -> None:
        game = sj.Game(
            turn=100,
            state=sj.State.DISCARD_DRAW_AND_REVEAL_OR_REPLACE_WITH_DRAW,
            drawn_card_index=0,
            draw_pile=(
                5 - 2,  # initial discard, draw
                10 - 1,
                15 - 2,
                10 - 2,
                10 - 2,
                10 - 2,
                10 - 2,
                10 - 2,
                10 - 2,
                10 - 2,
                10 - 2,
                10 - 2,
                10 - 1,
                10,
                10,
            ),
            discarded_card_index=0,
            discard_pile=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            players=(
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=1, is_revealed=True),
                        sj.Finger(card_index=2, is_revealed=True),
                        sj.Finger(card_index=3, is_revealed=True),
                        sj.Finger(card_index=4, is_revealed=True),
                        sj.Finger(card_index=5, is_revealed=True),
                        sj.Finger(card_index=6, is_revealed=True),
                        sj.Finger(card_index=7, is_revealed=True),
                        sj.Finger(card_index=8, is_revealed=True),
                        sj.Finger(card_index=9, is_revealed=True),
                        sj.Finger(card_index=10, is_revealed=True),
                        sj.Finger(card_index=11, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=False),
                    ),
                ),
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=12, is_revealed=True),
                        sj.Finger(card_index=11, is_revealed=True),
                        sj.Finger(card_index=10, is_revealed=True),
                        sj.Finger(card_index=9, is_revealed=True),
                        sj.Finger(card_index=8, is_revealed=True),
                        sj.Finger(card_index=7, is_revealed=True),
                        sj.Finger(card_index=6, is_revealed=True),
                        sj.Finger(card_index=5, is_revealed=True),
                        sj.Finger(card_index=4, is_revealed=True),
                        sj.Finger(card_index=3, is_revealed=True),
                        sj.Finger(card_index=2, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=False),
                    ),
                ),
            ),
        )
        game = game.with_card_replaced_with_draw(11, 12)
        assert game == sj.Game(
            turn=101,
            state=sj.State.NULL,
            drawn_card_index=None,
            draw_pile=(
                5 - 2,
                10 - 1,
                15 - 2,
                10 - 2,
                10 - 2,
                10 - 2,
                10 - 2,
                10 - 2,
                10 - 2,
                10 - 2,
                10 - 2,
                10 - 2,
                10 - 2,
                10,
                10,
            ),
            discarded_card_index=12,
            discard_pile=(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            players=(
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=1, is_revealed=True),
                        sj.Finger(card_index=2, is_revealed=True),
                        sj.Finger(card_index=3, is_revealed=True),
                        sj.Finger(card_index=4, is_revealed=True),
                        sj.Finger(card_index=5, is_revealed=True),
                        sj.Finger(card_index=6, is_revealed=True),
                        sj.Finger(card_index=7, is_revealed=True),
                        sj.Finger(card_index=8, is_revealed=True),
                        sj.Finger(card_index=9, is_revealed=True),
                        sj.Finger(card_index=10, is_revealed=True),
                        sj.Finger(card_index=11, is_revealed=True),
                        sj.Finger(card_index=0, is_revealed=True),
                    ),
                ),
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=12, is_revealed=True),
                        sj.Finger(card_index=11, is_revealed=True),
                        sj.Finger(card_index=10, is_revealed=True),
                        sj.Finger(card_index=9, is_revealed=True),
                        sj.Finger(card_index=8, is_revealed=True),
                        sj.Finger(card_index=7, is_revealed=True),
                        sj.Finger(card_index=6, is_revealed=True),
                        sj.Finger(card_index=5, is_revealed=True),
                        sj.Finger(card_index=4, is_revealed=True),
                        sj.Finger(card_index=3, is_revealed=True),
                        sj.Finger(card_index=2, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=False),
                    ),
                ),
            ),
        )

    def test_with_card_replaced_with_discard(self, rng: random.Random) -> None:
        game = sj.Game.new(players=2)
        game = game.with_random_deal(rng=rng)
        game = game.with_random_second_card_revealed(1, rng=rng)
        game = game.with_random_second_card_revealed(sj.HAND_ROWS, rng=rng)
        game = game.with_random_card_replaced_with_discard(1, rng=rng)
        assert game == sj.Game(
            turn=4,
            state=sj.State.DRAW_OR_REPLACE_WITH_DISCARD,
            drawn_card_index=None,
            draw_pile=(5, 9, 15, 10, 10, 10, 9, 10, 10, 9, 9, 10, 9, 9, 10),
            discarded_card_index=12,
            discard_pile=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            players=(
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=10, is_revealed=True),
                        sj.Finger(card_index=6, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=False),
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
                        sj.Finger(card_index=1, is_revealed=True),
                        sj.Finger(card_index=9, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=False),
                        sj.Finger(card_index=13, is_revealed=True),
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
            ),
        )

    def test_with_card_replaced_with_discard_clear(self, rng: random.Random) -> None:
        game = sj.Game.new(players=2)
        game = game.with_deal((1, 1, 2))
        game = game.with_second_card_revealed(1, 1)  # player 1
        game = game.with_second_card_revealed(1, 2)  # player 2
        game = game.with_card_replaced_with_discard(2, 0)  # player 1
        assert game == sj.Game(
            turn=3,
            state=sj.State.DRAW_OR_REPLACE_WITH_DISCARD,
            drawn_card_index=None,
            draw_pile=(
                5 - 1,
                10 - 3,
                15 - 2,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
            ),
            discarded_card_index=1,
            discard_pile=(1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            players=(
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=2, is_revealed=True),
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
                        sj.Finger(card_index=None, is_revealed=False),
                    ),
                ),
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=None, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=True),
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
            ),
        )

    def test_with_card_replaced_with_discard_end(self) -> None:
        game = sj.Game(
            turn=100,
            state=sj.State.DRAW_OR_REPLACE_WITH_DISCARD,
            drawn_card_index=None,
            draw_pile=(
                5 - 1,  # initial discard
                10 - 1,
                15 - 2,
                10 - 2,
                10 - 2,
                10 - 2,
                10 - 2,
                10 - 2,
                10 - 2,
                10 - 2,
                10 - 2,
                10 - 2,
                10 - 1,
                10,
                10,
            ),
            discarded_card_index=0,
            discard_pile=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            players=(
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=1, is_revealed=True),
                        sj.Finger(card_index=2, is_revealed=True),
                        sj.Finger(card_index=3, is_revealed=True),
                        sj.Finger(card_index=4, is_revealed=True),
                        sj.Finger(card_index=5, is_revealed=True),
                        sj.Finger(card_index=6, is_revealed=True),
                        sj.Finger(card_index=7, is_revealed=True),
                        sj.Finger(card_index=8, is_revealed=True),
                        sj.Finger(card_index=9, is_revealed=True),
                        sj.Finger(card_index=10, is_revealed=True),
                        sj.Finger(card_index=11, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=False),
                    ),
                ),
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=12, is_revealed=True),
                        sj.Finger(card_index=11, is_revealed=True),
                        sj.Finger(card_index=10, is_revealed=True),
                        sj.Finger(card_index=9, is_revealed=True),
                        sj.Finger(card_index=8, is_revealed=True),
                        sj.Finger(card_index=7, is_revealed=True),
                        sj.Finger(card_index=6, is_revealed=True),
                        sj.Finger(card_index=5, is_revealed=True),
                        sj.Finger(card_index=4, is_revealed=True),
                        sj.Finger(card_index=3, is_revealed=True),
                        sj.Finger(card_index=2, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=False),
                    ),
                ),
            ),
        )
        game = game.with_card_replaced_with_discard(11, 12)
        assert game == sj.Game(
            turn=101,
            state=sj.State.NULL,
            drawn_card_index=None,
            draw_pile=(
                5 - 1,
                10 - 1,
                15 - 2,
                10 - 2,
                10 - 2,
                10 - 2,
                10 - 2,
                10 - 2,
                10 - 2,
                10 - 2,
                10 - 2,
                10 - 2,
                10 - 2,
                10,
                10,
            ),
            discarded_card_index=12,
            discard_pile=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            players=(
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=1, is_revealed=True),
                        sj.Finger(card_index=2, is_revealed=True),
                        sj.Finger(card_index=3, is_revealed=True),
                        sj.Finger(card_index=4, is_revealed=True),
                        sj.Finger(card_index=5, is_revealed=True),
                        sj.Finger(card_index=6, is_revealed=True),
                        sj.Finger(card_index=7, is_revealed=True),
                        sj.Finger(card_index=8, is_revealed=True),
                        sj.Finger(card_index=9, is_revealed=True),
                        sj.Finger(card_index=10, is_revealed=True),
                        sj.Finger(card_index=11, is_revealed=True),
                        sj.Finger(card_index=0, is_revealed=True),
                    ),
                ),
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=12, is_revealed=True),
                        sj.Finger(card_index=11, is_revealed=True),
                        sj.Finger(card_index=10, is_revealed=True),
                        sj.Finger(card_index=9, is_revealed=True),
                        sj.Finger(card_index=8, is_revealed=True),
                        sj.Finger(card_index=7, is_revealed=True),
                        sj.Finger(card_index=6, is_revealed=True),
                        sj.Finger(card_index=5, is_revealed=True),
                        sj.Finger(card_index=4, is_revealed=True),
                        sj.Finger(card_index=3, is_revealed=True),
                        sj.Finger(card_index=2, is_revealed=True),
                        sj.Finger(card_index=None, is_revealed=False),
                    ),
                ),
            ),
        )
