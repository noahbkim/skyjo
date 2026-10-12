from __future__ import annotations

from skyjo2 import HAND_ROWS, Game, format_game


def test_render_new() -> None:
    game = Game.new(players=2)
    assert tuple(format_game(game)) == (
        "turn: 0",
        "",
        "> p0:     0  p1:       0",
        "-----------  -----------",
        "                        ",
        "                        ",
        "                        ",
    )
    game = Game.new(players=3)
    assert tuple(format_game(game)) == (
        "turn: 0",
        "",
        "> p0:     0  p1:       0  p2:       0",
        "-----------  -----------  -----------",
        "                                     ",
        "                                     ",
        "                                     ",
    )
    game = Game.new(players=4)
    assert tuple(format_game(game)) == (
        "turn: 0",
        "",
        "> p0:     0  p1:       0  p2:       0  p3:       0",
        "-----------  -----------  -----------  -----------",
        "                                                  ",
        "                                                  ",
        "                                                  ",
    )


def test_render_deal() -> None:
    game = Game.new(players=4)
    game = game.with_discard_and_first_cards_dealt((2, 0, 2, 3, 12))
    assert tuple(format_game(game)) == (
        "turn: 0  discard: 0",
        "",
        "> p0:     0  p1:       0  p2:       0  p3:       0",
        "-----------  -----------  -----------  -----------",
        "-2 [] [] []   0 [] [] []   1 [] [] []  10 [] [] []",
        "[] [] [] []  [] [] [] []  [] [] [] []  [] [] [] []",
        "[] [] [] []  [] [] [] []  [] [] [] []  [] [] [] []",
    )


def test_render_second_cards() -> None:
    game = Game.new(players=4)
    game = game.with_discard_and_first_cards_dealt((2, 0, 2, 3, 12))
    game = game.with_second_card_revealed(1, 0)
    game = game.with_second_card_revealed(HAND_ROWS, 2)
    game = game.with_second_card_revealed(1, 3)
    game = game.with_second_card_revealed(HAND_ROWS, 12)
    assert tuple(format_game(game)) == (
        "turn: 0  discard: 0",
        "",
        "p0:       0  p1:       0  p2:       0  > p3:     0",
        "-----------  -----------  -----------  -----------",
        "-2 [] [] []   0  0 [] []   1 [] [] []  10 10 [] []",
        "-2 [] [] []  [] [] [] []   1 [] [] []  [] [] [] []",
        "[] [] [] []  [] [] [] []  [] [] [] []  [] [] [] []",
    )


def test_render_draw() -> None:
    game = Game.new(players=4)
    game = game.with_discard_and_first_cards_dealt((2, 0, 2, 3, 12))
    game = game.with_second_card_revealed(1, 0)
    game = game.with_second_card_revealed(HAND_ROWS, 2)
    game = game.with_second_card_revealed(1, 3)
    game = game.with_second_card_revealed(HAND_ROWS, 12)
    game = game.with_card_drawn(14)
    assert tuple(format_game(game)) == (
        "turn: 0  discard: 0  draw: 12",
        "",
        "p0:       0  p1:       0  p2:       0  > p3:     0",
        "-----------  -----------  -----------  -----------",
        "-2 [] [] []   0  0 [] []   1 [] [] []  10 10 [] []",
        "-2 [] [] []  [] [] [] []   1 [] [] []  [] [] [] []",
        "[] [] [] []  [] [] [] []  [] [] [] []  [] [] [] []",
    )


def test_render_turn() -> None:
    game = Game.new(players=4)
    game = game.with_discard_and_first_cards_dealt((2, 0, 2, 3, 12))
    game = game.with_second_card_revealed(1, 0)
    game = game.with_second_card_revealed(HAND_ROWS, 2)
    game = game.with_second_card_revealed(1, 3)
    game = game.with_second_card_revealed(HAND_ROWS, 12)
    game = game.with_card_drawn(14)
    game = game.with_card_replaced_with_draw(2, 0)
    assert tuple(format_game(game)) == (
        "turn: 1  discard: -2",
        "",
        "> p0:     0  p1:       0  p2:       0  p3:       0",
        "-----------  -----------  -----------  -----------",
        "-2 [] [] []   0  0 [] []   1 [] [] []  10 10 [] []",
        "-2 [] [] []  [] [] [] []   1 [] [] []  [] [] [] []",
        "[] [] [] []  [] [] [] []  [] [] [] []  12 [] [] []",
    )


def test_render_forfeit() -> None:
    game = Game.new(players=4)
    game = game.with_discard_and_first_cards_dealt((2, 0, 2, 3, 12))
    game = game.with_second_card_revealed(1, 0)
    game = game.with_second_card_revealed(HAND_ROWS, 2)
    game = game.with_second_card_revealed(1, 3)
    game = game.with_second_card_revealed(HAND_ROWS, 12)
    game = game.with_card_drawn(14)
    game = game.with_card_replaced_with_draw(2, 0)
    game = game.with_forfeit()
    assert tuple(format_game(game)) == (
        "turn: 2  discard: -2",
        "",
        "p0:       0  > p1:     0  p2:       0  p3:       0",
        "-----------  -----------  -----------  -----------",
        "-2 [] [] []   0  0 [] []   1 [] [] []  10 10 [] []",
        "-2 [] [] []  [] [] [] []   1 [] [] []  [] [] [] []",
        "[] [] [] []  [] [] [] []  [] [] [] []  12 [] [] []",
    )
