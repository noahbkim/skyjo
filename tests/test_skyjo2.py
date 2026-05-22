from __future__ import annotations

import random

import skyjo2 as sj


class TestGame:
    def test_new(self) -> None:
        game = sj.Game.new(players=2)
        assert game == sj.Game(
            turn=0,
            end_turn=None,
            state=sj.State.DEAL_FIRST_CARD,
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
        game = game.with_discard_and_first_cards_dealt((0, 1, 2))
        assert game == sj.Game(
            turn=0,
            end_turn=None,
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
        game = game.with_random_discard_and_first_cards_dealt(rng=rng)
        assert game == sj.Game(
            turn=0,
            end_turn=None,
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
        game = game.with_random_discard_and_first_cards_dealt(rng=rng)
        game = game.with_random_second_card_revealed(1, rng=rng)
        assert game == sj.Game(
            turn=1,
            end_turn=None,
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
            turn=2,
            end_turn=None,
            state=sj.State.DRAW_OR_REPLACE_WITH_DISCARD,
            drawn_card_index=None,
            draw_pile=(5, 9, 15, 10, 10, 10, 9, 10, 10, 9, 9, 10, 10, 9, 10),
            discarded_card_index=9,
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
            ),
        )

    def test_with_draw_discarded_and_card_revealed(self, rng: random.Random) -> None:
        game = sj.Game.new(players=2)
        game = game.with_random_discard_and_first_cards_dealt(rng=rng)
        game = game.with_random_second_card_revealed(1, rng=rng)
        game = game.with_random_second_card_revealed(sj.HAND_ROWS, rng=rng)
        game = game.with_random_drawn_card(rng=rng)
        game = game.with_draw_discarded_and_random_card_revealed(2, rng=rng)
        assert game == sj.Game(
            turn=3,
            end_turn=None,
            state=sj.State.DRAW_OR_REPLACE_WITH_DISCARD,
            drawn_card_index=None,
            draw_pile=(5, 9, 15, 10, 10, 10, 9, 10, 10, 9, 8, 10, 9, 9, 10),
            discarded_card_index=12,
            discard_pile=(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
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
                    ),
                ),
            ),
        )

    def test_with_draw_discarded_and_card_revealed_clear(self) -> None:
        game = sj.Game.new(players=2)
        game = game.with_discard_and_first_cards_dealt((0, 1, 2))
        game = game.with_second_card_revealed(1, 1)  # player 1
        game = game.with_second_card_revealed(1, 2)  # player 2
        game = game.with_drawn_card(0)  # player 1
        game = game.with_draw_discarded_and_card_revealed(2, 1)  # player 1
        assert game == sj.Game(
            turn=4,
            end_turn=None,
            state=sj.State.DRAW_OR_REPLACE_WITH_DISCARD,
            drawn_card_index=None,
            draw_pile=(3, 7, 13, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10),
            discarded_card_index=0,
            discard_pile=(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            players=(
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=1, is_revealed=True),
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
                    ),
                ),
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=2, is_revealed=True),
                        sj.Finger(card_index=2, is_revealed=True),
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
                    ),
                ),
            ),
        )

    def test_with_draw_discarded_and_card_revealed_end(self) -> None:
        game = sj.Game(
            turn=99,
            end_turn=None,
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
        game = game.with_drawn_card(0)
        game = game.with_draw_discarded_and_card_revealed(11, 12)
        assert game == sj.Game(
            turn=100,
            end_turn=99,
            state=sj.State.DRAW_OR_REPLACE_WITH_DISCARD,
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
            ),
        )
        game = game.with_drawn_card(0)
        game = game.with_draw_discarded_and_card_revealed(11, 1)
        assert game == sj.Game(
            turn=101,
            end_turn=99,
            state=sj.State.ENDED_BY_REVEAL,
            drawn_card_index=None,
            draw_pile=(
                5 - 3,
                10 - 2,
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
            discard_pile=(2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
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
                        sj.Finger(card_index=1, is_revealed=True),
                    ),
                ),
            ),
        )

    def test_with_card_replaced_with_draw(self, rng: random.Random) -> None:
        game = sj.Game.new(players=2)
        game = game.with_random_discard_and_first_cards_dealt(rng=rng)
        game = game.with_random_second_card_revealed(1, rng=rng)
        game = game.with_random_second_card_revealed(sj.HAND_ROWS, rng=rng)
        game = game.with_random_drawn_card(rng=rng)
        game = game.with_random_card_replaced_with_draw(1, rng=rng)
        assert game == sj.Game(
            turn=3,
            end_turn=None,
            state=sj.State.DRAW_OR_REPLACE_WITH_DISCARD,
            drawn_card_index=None,
            draw_pile=(5, 9, 15, 10, 10, 10, 9, 10, 10, 9, 9, 10, 9, 9, 10),
            discarded_card_index=6,
            discard_pile=(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
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
                        sj.Finger(card_index=12, is_revealed=True),
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

    def test_with_card_replaced_with_draw_clear(self) -> None:
        game = sj.Game.new(players=2)
        game = game.with_discard_and_first_cards_dealt((0, 1, 2))
        game = game.with_second_card_revealed(1, 1)  # player 1
        game = game.with_second_card_revealed(1, 2)  # player 2
        game = game.with_drawn_card(1)  # player 1
        game = game.with_card_replaced_with_draw(2, 0)  # player 1
        assert game == sj.Game(
            turn=4,
            end_turn=None,
            state=sj.State.DRAW_OR_REPLACE_WITH_DISCARD,
            drawn_card_index=None,
            draw_pile=(3, 7, 13, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10),
            discarded_card_index=0,
            discard_pile=(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            players=(
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=1, is_revealed=True),
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
                    ),
                ),
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=2, is_revealed=True),
                        sj.Finger(card_index=2, is_revealed=True),
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
                    ),
                ),
            ),
        )

    def test_with_card_replaced_with_draw_end(self) -> None:
        game = sj.Game(
            turn=99,
            end_turn=None,
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
        game = game.with_drawn_card(12)
        game = game.with_card_replaced_with_draw(11, 0)
        assert game == sj.Game(
            turn=100,
            end_turn=99,
            state=sj.State.DRAW_OR_REPLACE_WITH_DISCARD,
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
            ),
        )
        game = game.with_drawn_card(1)
        game = game.with_card_replaced_with_draw(11, 0)
        assert game == sj.Game(
            turn=101,
            end_turn=99,
            state=sj.State.ENDED_BY_REVEAL,
            drawn_card_index=None,
            draw_pile=(
                5 - 3,
                10 - 2,
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
            discard_pile=(2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
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
                        sj.Finger(card_index=1, is_revealed=True),
                    ),
                ),
            ),
        )

    def test_with_card_replaced_with_discard(self, rng: random.Random) -> None:
        game = sj.Game.new(players=2)
        game = game.with_random_discard_and_first_cards_dealt(rng=rng)
        game = game.with_random_second_card_revealed(1, rng=rng)
        game = game.with_random_second_card_revealed(sj.HAND_ROWS, rng=rng)
        game = game.with_random_card_replaced_with_discard(1, rng=rng)
        assert game == sj.Game(
            turn=3,
            end_turn=None,
            state=sj.State.DRAW_OR_REPLACE_WITH_DISCARD,
            drawn_card_index=None,
            draw_pile=(5, 9, 15, 10, 10, 10, 9, 10, 10, 9, 9, 10, 10, 9, 10),
            discarded_card_index=6,
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
                        sj.Finger(card_index=9, is_revealed=True),
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

    def test_with_card_replaced_with_discard_clear(self, rng: random.Random) -> None:
        game = sj.Game.new(players=2)
        game = game.with_discard_and_first_cards_dealt((1, 1, 2))
        game = game.with_second_card_revealed(1, 1)  # player 1
        game = game.with_second_card_revealed(1, 2)  # player 2
        game = game.with_card_replaced_with_discard(2, 0)  # player 1
        assert game == sj.Game(
            turn=4,
            end_turn=None,
            state=sj.State.DRAW_OR_REPLACE_WITH_DISCARD,
            drawn_card_index=None,
            draw_pile=(4, 7, 13, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10),
            discarded_card_index=0,
            discard_pile=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            players=(
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=1, is_revealed=True),
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
                    ),
                ),
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=2, is_revealed=True),
                        sj.Finger(card_index=2, is_revealed=True),
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
                    ),
                ),
            ),
        )

    def test_with_card_replaced_with_discard_end(self) -> None:
        game = sj.Game(
            turn=99,
            end_turn=None,
            state=sj.State.DRAW_OR_REPLACE_WITH_DISCARD,
            drawn_card_index=None,
            draw_pile=(
                5,
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
                10 - 2,  # initial discard
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
        game = game.with_card_replaced_with_discard(11, 1)
        assert game == sj.Game(
            turn=100,
            end_turn=99,
            state=sj.State.DRAW_OR_REPLACE_WITH_DISCARD,
            drawn_card_index=None,
            draw_pile=(
                5,
                10 - 2,
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
            discarded_card_index=1,
            discard_pile=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            players=(
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
            ),
        )
        game = game.with_card_replaced_with_discard(11, 0)
        assert game == sj.Game(
            turn=101,
            end_turn=99,
            state=sj.State.ENDED_BY_REVEAL,
            drawn_card_index=None,
            draw_pile=(
                5 - 1,
                10 - 2,
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
                        sj.Finger(card_index=1, is_revealed=True),
                    ),
                ),
            ),
        )

    def test_with_hidden_cards_revealed_tie(self) -> None:
        game = sj.Game(
            turn=100,
            end_turn=None,
            state=sj.State.ENDED_BY_REVEAL,
            drawn_card_index=None,
            draw_pile=(
                5 - 1,  # initial discard
                10 - 1,
                15 - 1,
                10 - 1,
                10 - 1,
                10 - 1,
                10 - 1,
                10 - 1,
                10 - 1,
                10 - 1,
                10 - 1,
                10 - 1,
                10 - 2,
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
                        sj.Finger(card_index=12, is_revealed=True),
                    ),
                ),
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=12, is_revealed=True),
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
        game = game.with_hidden_cards_revealed((11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1))
        assert game == sj.Game(
            turn=100,
            end_turn=None,
            state=sj.State.ENDED_BY_REVEAL,
            drawn_card_index=None,
            draw_pile=(
                5 - 1,  # initial discard
                10 - 2,
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
                        sj.Finger(card_index=1, is_revealed=True),
                    ),
                ),
            ),
        )
        assert game.final_scores == (54 * 2, 54)
        assert game.players[0].hand_score == 54
        assert game.players[1].hand_score == 54
        assert game.winner_index == 1

    def test_with_hidden_cards_revealed_lose(self) -> None:
        game = sj.Game(
            turn=100,
            end_turn=None,
            state=sj.State.ENDED_BY_REVEAL,
            drawn_card_index=None,
            draw_pile=(
                5 - 1,  # initial discard
                10 - 1,
                15 - 2,
                10 - 1,
                10 - 1,
                10 - 1,
                10 - 1,
                10 - 1,
                10 - 1,
                10 - 1,
                10 - 1,
                10 - 1,
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
                        sj.Finger(card_index=12, is_revealed=True),
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
        game = game.with_hidden_cards_revealed((2,) * 11)
        assert game == sj.Game(
            turn=100,
            end_turn=None,
            state=sj.State.ENDED_BY_REVEAL,
            drawn_card_index=None,
            draw_pile=(
                5 - 1,  # initial discard
                10 - 1,
                15 - 13,
                10 - 1,
                10 - 1,
                10 - 1,
                10 - 1,
                10 - 1,
                10 - 1,
                10 - 1,
                10 - 1,
                10 - 1,
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
                        sj.Finger(card_index=12, is_revealed=True),
                    ),
                ),
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=2, is_revealed=True),
                        sj.Finger(card_index=2, is_revealed=True),
                        sj.Finger(card_index=2, is_revealed=True),
                        sj.Finger(card_index=2, is_revealed=True),
                        sj.Finger(card_index=2, is_revealed=True),
                        sj.Finger(card_index=2, is_revealed=True),
                        sj.Finger(card_index=2, is_revealed=True),
                        sj.Finger(card_index=2, is_revealed=True),
                        sj.Finger(card_index=2, is_revealed=True),
                        sj.Finger(card_index=2, is_revealed=True),
                        sj.Finger(card_index=2, is_revealed=True),
                        sj.Finger(card_index=2, is_revealed=True),
                    ),
                ),
            ),
        )
        assert game.final_scores == (108, 0)
        assert game.players[0].hand_score == 54
        assert game.players[1].hand_score == 0
        assert game.winner_index == 1

    def test_with_hidden_cards_revealed_win(self) -> None:
        game = sj.Game(
            turn=100,
            end_turn=None,
            state=sj.State.ENDED_BY_REVEAL,
            drawn_card_index=None,
            draw_pile=(
                5 - 1,  # initial discard
                10 - 1,
                15 - 1,
                10 - 1,
                10 - 1,
                10 - 1,
                10 - 1,
                10 - 1,
                10 - 1,
                10 - 1,
                10 - 1,
                10 - 1,
                10 - 1,
                10,
                10 - 1,
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
                        sj.Finger(card_index=12, is_revealed=True),
                    ),
                ),
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=14, is_revealed=True),
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
        game = game.with_hidden_cards_revealed((14,) * 5 + (13,) * 6)
        assert game == sj.Game(
            turn=100,
            end_turn=None,
            state=sj.State.ENDED_BY_REVEAL,
            drawn_card_index=None,
            draw_pile=(
                5 - 1,  # initial discard
                10 - 1,
                15 - 1,
                10 - 1,
                10 - 1,
                10 - 1,
                10 - 1,
                10 - 1,
                10 - 1,
                10 - 1,
                10 - 1,
                10 - 1,
                10 - 1,
                10 - 6,
                10 - 6,
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
                        sj.Finger(card_index=12, is_revealed=True),
                    ),
                ),
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=14, is_revealed=True),
                        sj.Finger(card_index=14, is_revealed=True),
                        sj.Finger(card_index=14, is_revealed=True),
                        sj.Finger(card_index=14, is_revealed=True),
                        sj.Finger(card_index=14, is_revealed=True),
                        sj.Finger(card_index=14, is_revealed=True),
                        sj.Finger(card_index=13, is_revealed=True),
                        sj.Finger(card_index=13, is_revealed=True),
                        sj.Finger(card_index=13, is_revealed=True),
                        sj.Finger(card_index=13, is_revealed=True),
                        sj.Finger(card_index=13, is_revealed=True),
                        sj.Finger(card_index=13, is_revealed=True),
                    ),
                ),
            ),
        )
        assert game.final_scores == (54, 138)
        assert game.players[0].hand_score == 54
        assert game.players[1].hand_score == 138
        assert game.winner_index == 0

    def test_with_random_hidden_cards_revealed(self, rng: random.Random) -> None:
        game = sj.Game(
            turn=100,
            end_turn=None,
            state=sj.State.ENDED_BY_REVEAL,
            drawn_card_index=None,
            draw_pile=(
                5 - 1,  # initial discard
                10 - 1,
                15 - 1,
                10 - 1,
                10 - 1,
                10 - 1,
                10 - 1,
                10 - 1,
                10 - 1,
                10 - 1,
                10 - 1,
                10 - 1,
                10 - 2,
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
                        sj.Finger(card_index=12, is_revealed=True),
                    ),
                ),
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=12, is_revealed=True),
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
        game = game.with_random_hidden_cards_revealed(rng=rng)
        assert game == sj.Game(
            turn=100,
            end_turn=None,
            state=sj.State.ENDED_BY_REVEAL,
            drawn_card_index=None,
            draw_pile=(4, 8, 14, 9, 9, 8, 9, 8, 7, 9, 8, 8, 7, 10, 7),
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
                        sj.Finger(card_index=12, is_revealed=True),
                    ),
                ),
                sj.Player(
                    score=0,
                    hand=(
                        sj.Finger(card_index=12, is_revealed=True),
                        sj.Finger(card_index=10, is_revealed=True),
                        sj.Finger(card_index=12, is_revealed=True),
                        sj.Finger(card_index=1, is_revealed=True),
                        sj.Finger(card_index=7, is_revealed=True),
                        sj.Finger(card_index=14, is_revealed=True),
                        sj.Finger(card_index=14, is_revealed=True),
                        sj.Finger(card_index=11, is_revealed=True),
                        sj.Finger(card_index=8, is_revealed=True),
                        sj.Finger(card_index=14, is_revealed=True),
                        sj.Finger(card_index=5, is_revealed=True),
                        sj.Finger(card_index=8, is_revealed=True),
                    ),
                ),
            ),
        )
        assert game.final_scores == (54, 92)
        assert game.players[0].hand_score == 54
        assert game.players[1].hand_score == 92
        assert game.winner_index == 0
