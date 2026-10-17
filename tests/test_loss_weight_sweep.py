"""Whole-game holdout boundaries must survive position subsampling."""

import numpy as np

from skyjo.experiments.loss_weight_sweep import SPLITS, choose_split


def test_split_excludes_prior_games_and_never_shares_games_or_rows():
    identities = np.array([90, 41, 72, 63, 14, 55, 36, 87, 28, 19])
    lengths = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    offsets = np.r_[0, np.cumsum(lengths)]
    selection = choose_split(
        offsets, identities, np.arange(2, 10), [72, 19], [2, 2, 2], [5, 6, 7], 123
    )
    all_games, all_rows = [], []
    for name, count in zip(SPLITS, [5, 6, 7], strict=True):
        games, rows = selection[f"{name}_games"], selection[f"{name}_rows"]
        assert len(games) == 2
        assert len(rows) == count
        row_games = identities[np.searchsorted(offsets[1:], rows, side="right")]
        assert set(row_games) <= set(games)
        assert not set(games) & {90, 41, 72, 19}
        all_games.extend(games)
        all_rows.extend(rows)
    assert len(set(all_games)) == len(all_games)
    assert len(set(all_rows)) == len(all_rows)
    repeated = choose_split(
        offsets, identities, np.arange(2, 10), [72, 19], [2, 2, 2], [5, 6, 7], 123
    )
    for key in selection:
        np.testing.assert_array_equal(selection[key], repeated[key])
