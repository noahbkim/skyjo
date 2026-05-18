from __future__ import annotations

import skyjo2 as sj
import skynet as sn


def test_embed_non_spatial() -> None:
    game = sj.Game.new(players=4)
    assert sn.embed_game_non_spatial(game).shape == (69,)


def test_embed_spatial() -> None:
    game = sj.Game.new(players=4)
    assert sn.embed_game_spatial(game).shape == (4, 3, 4, 17)
