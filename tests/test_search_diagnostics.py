"""Portable fixtures and honest, nonperturbing search diagnostics."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
from test_mcts_core import UniformEvaluator, make_state

from skyjo.engine import game
from skyjo.experiments.search_diagnostics import model_diagnostic, run_search, summarize
from skyjo.experiments.search_positions import (
    Expectation,
    Witness,
    export_position,
    from_dict,
    load_suite,
    witness_states,
)
from skyjo.search import mcts
from skyjo.simulation import recording

SUITE = Path(__file__).resolve().parents[1] / "data/search-positions"


def test_recorded_clear_witnesses_hold_for_every_hidden_card():
    positions = load_suite(SUITE)
    assert len(positions) == 6
    for position in positions:
        prefix, outcomes = witness_states(position, position.witnesses[0])
        assert len(outcomes) > 1  # Clearing is certain; the revealed card isn't.
        assert sum(p for _, p in outcomes) == pytest.approx(1)
        actor = game.get_player(prefix[-1])
        for state, _ in outcomes:
            # Engine rotates after placement; fixed seat -> current orientation.
            relative = (actor - game.get_player(state)) % state.players
            assert state.table[relative, :, 3, game.FINGER_CLEARED].all()
            assert game.get_facedown_count(state, relative) == 0
        if "giveaway" in position.tags:
            assert not position.expectation.accepts(position.source["recorded_action"])
        else:
            assert position.expectation.accepts(position.source["recorded_action"])


def test_export_is_portable_and_rejects_stochastic_witness_prefix(tmp_path):
    state = make_state()
    source = tmp_path / "recording.json"
    recording.save(
        recording.Recording(
            [{"name": "A"}, {"name": "B"}], 1, [recording.make_frame(state, 1)]
        ),
        source,
    )
    path = export_position(
        source,
        0,
        tmp_path / "suite" / "case.json",
        id="opening",
        description="An unscored opening",
        expectation=Expectation("unscored", (), "Exploratory"),
    )
    source.unlink()
    case = load_suite(path.parent)[0]
    np.testing.assert_array_equal(case.state.table, state.table)
    invalid = case.to_dict()
    invalid["witnesses"] = [{"name": "random", "actions": [0, 2]}]
    with pytest.raises(ValueError, match="deterministic"):
        from_dict(invalid)
    invalid = case.to_dict()
    invalid["expectation"] = {"kind": "prefer", "actions": [27], "rationale": "Illegal"}
    with pytest.raises(ValueError, match="legal"):
        from_dict(invalid)


@pytest.mark.parametrize("exact", [False, True])
def test_tracing_preserves_search_and_counts_real_evaluator_work(exact):
    position = next(p for p in load_suite(SUITE) if p.id == "clear-9-giveaway")
    original = json.dumps(position.to_dict(), sort_keys=True)
    config = mcts.SearchConfig(
        after_state_evaluate_all_children=exact, merge_symmetric_actions=False
    )
    evaluator = UniformEvaluator()
    result = run_search(position, evaluator, 32, 71, config=config)
    root = mcts.run_mcts(
        position.state, evaluator, 32, config=config, rng=np.random.default_rng(71)
    )
    assert result["action"] == root.policy_targets(0).argmax()
    np.testing.assert_array_equal(result["value"], root.state_value)
    assert [g["visits"] for g in result["groups"]] == [
        n.visit_count for n in root.children.values()
    ]
    assert sum(g["visits"] for g in result["groups"]) == 32
    assert result["evaluator_states"] >= result["evaluator_calls"] >= 1
    assert result["trace"][-1]["iteration"] == 32
    assert json.dumps(position.to_dict(), sort_keys=True) == original
    repeated = run_search(position, evaluator, 32, 71, config=config)
    assert {k: v for k, v in result.items() if k != "seconds"} == {
        k: v for k, v in repeated.items() if k != "seconds"
    }
    baseline = model_diagnostic(position, evaluator)
    assert baseline["witnesses"][0]["after_value"] == pytest.approx([0.5, 0.5])


def test_evaluated_reply_is_distinct_from_visited_reply():
    # TAKE expansion evaluates its resulting decision, but that decision has no
    # visits yet. Merely creating an action edge does not evaluate that edge.
    position = next(p for p in load_suite(SUITE) if p.id == "clear-9-take")
    position = dataclasses.replace(
        position, witnesses=(Witness("take", (3,)), Witness("clear", (3, 27)))
    )

    class TakeEvaluator(UniformEvaluator):
        def evaluate(self, states):
            predictions = super().evaluate(states)
            for state, prediction in zip(states, predictions, strict=True):
                if game.get_action(state) == game.ACTION_DRAW_OR_TAKE:
                    prediction.policy[:] = 0
                    prediction.policy[3] = 1
            return predictions

    result = run_search(position, TakeEvaluator(), 1, 0)
    take, clear = result["witnesses"]
    assert take["first_evaluated_iteration"] == take["first_visited_iteration"] == 1
    assert clear["first_evaluated_iteration"] is None
    assert clear["first_visited_iteration"] is None
    assert result["stable_accepted_from"] == 1
    summary = summarize([dict(case=position.id, method="test", iterations=1, **result)])
    assert summary[0]["accepted"] == summary[0]["scored"] == 1
    assert summary[0]["witnesses"]["clear"] == {
        "evaluated": 0,
        "visited": 0,
        "available": 1,
        "median_first_evaluated_iteration": None,
    }


def test_public_runner_saves_replayable_inputs_and_a_complete_report(tmp_path):
    from run_search_diagnostics import run
    from skyjo.learning import checkpoint, models

    settings = {
        "name": "hierarchical_equivariant_v2",
        "embedding_dimensions": 4,
        "global_state_embedding_dimensions": 8,
        "num_heads": 2,
    }
    weights = tmp_path / "model.pth"
    checkpoint.save_checkpoint(
        weights,
        model=models.build(settings, players=2, device="cpu"),
        optimizer=None,
        configuration={"model": settings, "players": 2},
    )
    suite = tmp_path / "suite"
    suite.mkdir()
    position = load_suite(SUITE)[0]
    (suite / "one.json").write_text(json.dumps(position.to_dict()))
    plan = tmp_path / "plan.toml"
    plan.write_text(
        'version = 1\niterations = [2]\nseeds = [7]\n[[methods]]\nname = "test"\n'
    )
    output = run(weights, tmp_path / "result", suite=suite, plan=plan, allow_dirty=True)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["trials"] == 1
    assert load_suite(output / "positions")[0].id == position.id
    row = json.loads((output / "trials.jsonl").read_text())
    assert (row["case"], row["method"], row["iterations"], row["seed"]) == (
        position.id,
        "test",
        2,
        7,
    )
    assert position.id in (output / "report.md").read_text()
    with pytest.raises(FileExistsError):
        run(weights, output, suite=suite, plan=plan, allow_dirty=True)
