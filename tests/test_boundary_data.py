"""Protect label construction and separation of games in the boundary experiment."""

import hashlib
import json

import numpy as np
import pytest

from skyjo.learning.boundary_data import load_boundaries, split_by_game


def write_rounds(path, games):
    records = []
    for game_index, rounds in enumerate(games):
        previous = np.zeros(len(rounds[0][0]), dtype=int)
        for number, (totals, starter) in enumerate(rounds, start=1):
            records.append(
                {
                    "run_id": "test-run",
                    "game_index": game_index,
                    "iteration": 41,
                    "play_seed": 900 + game_index,
                    "round_number": number,
                    "cumulative_scores": totals,
                    "scores": (np.array(totals) - previous).tolist(),
                    "ending_player": starter,
                    "partial_start": False,
                }
            )
            previous = np.array(totals)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return records


def test_observed_boundaries_use_eventual_winners_and_next_starter(tmp_path):
    path = tmp_path / "rounds.jsonl"
    write_rounds(
        path,
        [
            [([20, 40, 60], 1), ([70, 90, 50], 2), ([110, 110, 150], 0)],
            [
                ([10, 30, 50], 2)
            ],  # Unfinished: this current leader is not a winner label.
            [([120, 30, 70], 0)],  # Finished in one round: no continuation to predict.
        ],
    )
    original = path.read_bytes()
    data = load_boundaries([path])
    np.testing.assert_array_equal(data.scores, [[40, 60, 20], [50, 70, 90]])
    np.testing.assert_array_equal(data.targets, [[0.5, 0, 0.5], [0, 0.5, 0.5]])
    np.testing.assert_array_equal(data.round_numbers, [1, 2])
    assert data.scores.dtype == data.targets.dtype == np.float32
    assert data.summary["games_excluded_incomplete"] == 1
    assert data.summary["games_completed"] == 2
    assert data.summary["games_with_boundaries"] == 1
    assert data.sources[0]["sha256"] == hashlib.sha256(original).hexdigest()
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda rows: rows.append(rows[0]), "duplicate round"),
        (lambda rows: rows[1].update(round_number=3), "round sequence"),
        (lambda rows: rows[1].update(play_seed=2), "conflicting provenance"),
        (lambda rows: rows[0].update(ending_player=2), "ending_player"),
        (
            lambda rows: rows[0].update(cumulative_scores=[20, float("nan")]),
            "finite integer",
        ),
        (lambda rows: rows[1].update(cumulative_scores=[101, 90, 20]), "matching"),
        (lambda rows: rows[1].update(scores=[10, 20]), "inconsistent cumulative"),
    ],
)
def test_corrupted_logs_fail_instead_of_silently_creating_labels(
    tmp_path, change, message
):
    path = tmp_path / "rounds.jsonl"
    rows = write_rounds(path, [[([20, 40], 1), ([101, 90], 0)]])
    change(rows)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match=message):
        load_boundaries([path])


def test_game_identity_includes_run_and_splits_are_disjoint_and_reproducible(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_rounds(
        first,
        [[([20, 40], game % 2), ([40, 60], 1), ([110, 90], 0)] for game in range(12)],
    )
    second.write_text(first.read_text().replace("test-run", "different-run"))
    data = load_boundaries([first, second])
    assert len(np.unique(data.game_ids)) == 24
    splits = split_by_game(data, seed=14)
    reordered = load_boundaries([second, first])
    again = split_by_game(reordered, seed=14)
    groups = []
    for name, indices in splits.items():
        np.testing.assert_array_equal(indices, again[name])
        groups.append(set(data.game_ids[indices]))
        assert len(indices) == 2 * len(groups[-1])
    assert all(groups)
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    np.testing.assert_array_equal(
        np.sort(np.concatenate(list(splits.values()))), np.arange(48)
    )


def test_partial_games_are_excluded_and_post_terminal_records_are_errors(tmp_path):
    path = tmp_path / "rounds.jsonl"
    rows = write_rounds(
        path,
        [
            [([20, 40], 1), ([101, 90], 0)],
            [([50, 70], 1), ([105, 90], 0)],
        ],
    )
    rows[2]["partial_start"] = True
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    data = load_boundaries([path])
    assert data.summary["games_excluded_partial_start"] == 1
    assert len(data.scores) == 1
    rows[0].update(cumulative_scores=[110, 40], scores=[110, 40])
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="after game termination"):
        load_boundaries([path])
