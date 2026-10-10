import numpy as np
import pytest

from skyjo.engine import game
from skyjo.experiments.mcts_benchmark import fixture_cases, summarize_timings
from skyjo.search.symmetry import ActionGroups, safe_to_merge_actions


def test_fixtures_separate_asymmetric_and_unsafe_search_conditions():
    asymmetric, unsafe = fixture_cases()
    for case in (asymmetric, unsafe):
        game.validate(case.state)
    groups = ActionGroups.from_state(asymmetric.state)
    assert len(groups.members) == game.actions(asymmetric.state).sum()
    assert safe_to_merge_actions(asymmetric.state, 128)
    assert not safe_to_merge_actions(unsafe.state, 32)
    assert any(
        len(group) > 1 for group in ActionGroups.from_state(unsafe.state).members
    )


def test_timing_summary_uses_total_work_and_separates_fixture_and_phase():
    rows = []
    for cohort, phase, off, on in (
        ("control", "replace", 1.0, 0.5),
        ("variant", "draw_or_take", 9.0, 3.0),
        ("unsafe_fixture", "replace", 100.0, 100.0),
    ):
        for merge, seconds in ((False, off), (True, on)):
            rows.append(
                {
                    "cohort": cohort,
                    "phase": phase,
                    "evaluator": "next_deal",
                    "iterations": 32,
                    "merge_symmetric_actions": merge,
                    "seconds": seconds,
                }
            )
    summaries = summarize_timings(rows)
    replay = next(
        row
        for row in summaries
        if (row["cohort"], row["phase"]) == ("all_replay", "all")
    )
    assert replay["throughput_ratio_on_over_off"] == pytest.approx(10 / 3.5)
    assert replay["on"]["searches"] == 2
    assert replay["on"]["median_ms"] == 1750
    assert replay["on"]["p95_ms"] == pytest.approx(np.quantile([500, 3000], 0.95))
    replace = next(
        row
        for row in summaries
        if (row["cohort"], row["phase"]) == ("all_replay", "replace")
    )
    assert replace["throughput_ratio_on_over_off"] == 2
    with pytest.raises(ValueError, match="paired"):
        summarize_timings(rows[:-1])
