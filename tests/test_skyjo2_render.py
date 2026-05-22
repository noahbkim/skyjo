from __future__ import annotations

import skyjo2 as sj
import skyjo2.render as sj_render


def test_render_new() -> None:
    game = sj.Game.new(players=2)
    assert tuple(sj_render.render(game)) == (
        "turn: 0",
        "",
        "p0:       0  p1:       0",
        "-----------  -----------",
        "                        ",
        "                        ",
        "                        ",
    )
    game = sj.Game.new(players=3)
    assert tuple(sj_render.render(game)) == (
        "turn: 0",
        "",
        "p0:       0  p1:       0  p2:       0",
        "-----------  -----------  -----------",
        "                                     ",
        "                                     ",
        "                                     ",
    )
    game = sj.Game.new(players=4)
    assert tuple(sj_render.render(game)) == (
        "turn: 0",
        "",
        "p0:       0  p1:       0  p2:       0  p3:       0",
        "-----------  -----------  -----------  -----------",
        "                                                  ",
        "                                                  ",
        "                                                  ",
    )


def test_render_deal() -> None:
    game = sj.Game.new(players=4)
    game = game.with_discard_and_first_cards_dealt((2, 0, 2, 3, 12))
    assert tuple(sj_render.render(game)) == (
        "turn: 0  discard: 0",
        "",
        "p0:       0  p1:       0  p2:       0  p3:       0",
        "-----------  -----------  -----------  -----------",
        "-2 [] [] []   0 [] [] []   1 [] [] []  10 [] [] []",
        "[] [] [] []  [] [] [] []  [] [] [] []  [] [] [] []",
        "[] [] [] []  [] [] [] []  [] [] [] []  [] [] [] []",
    )


def test_render_second_cards() -> None:
    game = sj.Game.new(players=4)
    game = game.with_discard_and_first_cards_dealt((2, 0, 2, 3, 12))
    game = game.with_second_card_revealed(1, 0)
    game = game.with_second_card_revealed(sj.HAND_ROWS, 2)
    game = game.with_second_card_revealed(1, 3)
    game = game.with_second_card_revealed(sj.HAND_ROWS, 12)
    assert tuple(sj_render.render(game)) == (
        "turn: 4  discard: 0",
        "",
        "p0:       0  p1:       0  p2:       0  p3:       0",
        "-----------  -----------  -----------  -----------",
        "10 10 [] []  -2 [] [] []   0  0 [] []   1 [] [] []",
        "[] [] [] []  -2 [] [] []  [] [] [] []   1 [] [] []",
        "[] [] [] []  [] [] [] []  [] [] [] []  [] [] [] []",
    )


def test_render_draw() -> None:
    game = sj.Game.new(players=4)
    game = game.with_discard_and_first_cards_dealt((2, 0, 2, 3, 12))
    game = game.with_second_card_revealed(1, 0)
    game = game.with_second_card_revealed(sj.HAND_ROWS, 2)
    game = game.with_second_card_revealed(1, 3)
    game = game.with_second_card_revealed(sj.HAND_ROWS, 12)
    game = game.with_drawn_card(14)
    assert tuple(sj_render.render(game)) == (
        "turn: 4  discard: 0  draw: 12",
        "",
        "p0:       0  p1:       0  p2:       0  p3:       0",
        "-----------  -----------  -----------  -----------",
        "10 10 [] []  -2 [] [] []   0  0 [] []   1 [] [] []",
        "[] [] [] []  -2 [] [] []  [] [] [] []   1 [] [] []",
        "[] [] [] []  [] [] [] []  [] [] [] []  [] [] [] []",
    )


def test_render_turn() -> None:
    game = sj.Game.new(players=4)
    game = game.with_discard_and_first_cards_dealt((2, 0, 2, 3, 12))
    game = game.with_second_card_revealed(1, 0)
    game = game.with_second_card_revealed(sj.HAND_ROWS, 2)
    game = game.with_second_card_revealed(1, 3)
    game = game.with_second_card_revealed(sj.HAND_ROWS, 12)
    game = game.with_drawn_card(14)
    game = game.with_card_replaced_with_draw(2, 0)
    assert tuple(sj_render.render(game)) == (
        "turn: 5  discard: -2",
        "",
        "p1:       0  p2:       0  p3:       0  p0:       0",
        "-----------  -----------  -----------  -----------",
        "-2 [] [] []   0  0 [] []   1 [] [] []  10 10 [] []",
        "-2 [] [] []  [] [] [] []   1 [] [] []  [] [] [] []",
        "[] [] [] []  [] [] [] []  [] [] [] []  12 [] [] []",
    )


def test_render_forfeit() -> None:
    game = sj.Game.new(players=4)
    game = game.with_discard_and_first_cards_dealt((2, 0, 2, 3, 12))
    game = game.with_second_card_revealed(1, 0)
    game = game.with_second_card_revealed(sj.HAND_ROWS, 2)
    game = game.with_second_card_revealed(1, 3)
    game = game.with_second_card_revealed(sj.HAND_ROWS, 12)
    game = game.with_drawn_card(14)
    game = game.with_card_replaced_with_draw(2, 0)
    game = game.with_forfeit()
    assert tuple(sj_render.render(game)) == (
        "turn: 6  discard: -2",
        "",
        "p2:       0  p3:       0  p0:       0  p1:       0",
        "-----------  -----------  -----------  -----------",
        " 0  0 [] []   1 [] [] []  10 10 [] []  -2 [] [] []",
        "[] [] [] []   1 [] [] []  [] [] [] []  -2 [] [] []",
        "[] [] [] []  [] [] [] []  12 [] [] []  [] [] [] []",
    )
