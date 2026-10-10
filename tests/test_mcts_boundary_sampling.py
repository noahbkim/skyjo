import dataclasses

import numpy as np
import pytest

from skyjo import boundary_inference
from skyjo import game as sj
from skyjo import mcts, skynet
from test_full_game import completed_round


class RootOnlyPredictor:
    """The score-boundary path must never request a next-deal prediction."""

    def __init__(self):
        self.states = []

    def predict(self, state):
        assert not sj.get_round_over(state)
        self.states.append(state)
        policy = sj.actions(state).astype(np.float32)
        return skynet.SkyNetPrediction(
            value_output=np.array([0.1, 0.2, 0.7], dtype=np.float32),
            policy_output=policy / policy.sum(),
        )

    def predict_many(self, states):
        return [self.predict(state) for state in states]


def test_search_caches_sampled_values_before_averaging_without_dealing(monkeypatch):
    first = completed_round(
        ((-2, 0, 1), (0, 1, 2), (1, 2, 3)), (10, 20, 30), ending_player=2
    )
    second = completed_round(
        ((-2, 0, 1), (0, 1, 2), (1, 2, 3)), (20, 30, 40), ending_player=2
    )
    state = sj.apply_action(dataclasses.replace(first, countdown=2), sj.MASK_TAKE)
    # The actor's replacements are visible; an opponent's hidden card still
    # makes the final reveal random and requires the full sample budget.
    card = int(np.argmax(state.table[1, 0, 0]))
    state.table[1, 0, 0] = 0
    state.table[1, 0, 0, sj.FINGER_HIDDEN] = 1
    state.deck[card] += 1
    assert sj.validate(state)
    applications = []
    evaluated = []

    def sample(unchanged, action):
        assert unchanged is state
        applications.append(action)
        return first if len(applications) % 2 else second

    def predict_completed(model, states):
        assert model == "frozen model"
        # Score conversion and player ordering have their own inference tests.
        # Here distinct outcomes must reach inference individually, never a
        # synthetic state representing their mean score.
        assert len(states) == 10
        assert all(item is first or item is second for item in states)
        evaluated.append(states)
        return np.array(
            [[0.1, 0.2, 0.7] if item is first else [0.5, 0.4, 0.1] for item in states],
            dtype=np.float32,
        )

    def no_next_round(*args, **kwargs):
        pytest.fail("Score boundary search must stop before the next deal")

    monkeypatch.setattr(sj, "apply_action", sample)
    monkeypatch.setattr(sj, "start_next_round", no_next_round)
    monkeypatch.setattr(
        boundary_inference, "load_boundary_model", lambda *args: "frozen model"
    )
    monkeypatch.setattr(
        boundary_inference, "predict_completed_rounds", predict_completed
    )
    client = RootOnlyPredictor()
    settings = {"boundary_samples": 10, "boundary_value_checkpoint": "frozen.pth"}
    root = mcts.run_mcts(state, client, iterations=100, **settings)

    assert len(evaluated) == len(root.children) == 3
    assert len(applications) == 30
    assert len(client.states) == 1
    np.testing.assert_allclose(root.state_value, [0.3, 0.3, 0.4], atol=1e-6)
    assert sum(child.visit_count for child in root.children.values()) == 100
    for child in root.children.values():
        assert child.next_round_state is None
        np.testing.assert_allclose(child.state_value, [0.3, 0.3, 0.4], atol=1e-6)

    mcts.run_mcts(state, client, iterations=10, root_node=root, **settings)
    assert len(applications) == 30
    assert len(evaluated) == 3

    for changed in (
        {"boundary_samples": 1}, {"boundary_value_checkpoint": "other.pth"}
    ):
        with pytest.raises(ValueError, match="boundary configuration"):
            mcts.run_mcts(
                state, client, iterations=1, root_node=root, **(settings | changed)
            )


def test_deterministic_terminal_boundaries_use_one_exact_tie_without_model(
    monkeypatch,
):
    completed = completed_round(((0, 1, 2), (1, 2, 3), (1, 2, 3)), (200, 0, 6))
    state = sj.apply_action(dataclasses.replace(completed, countdown=2), sj.MASK_TAKE)
    assert not state.table[:, :, :, sj.FINGER_HIDDEN].any()
    assert sj.get_round_about_to_end(state)
    applications = []
    apply = sj.apply_action

    def record_apply(state, action):
        applications.append(action)
        return apply(state, action)

    def no_model(*args, **kwargs):
        pytest.fail("Fully terminal outcomes must use exact winners")

    monkeypatch.setattr(sj, "apply_action", record_apply)
    monkeypatch.setattr(boundary_inference, "load_boundary_model", no_model)
    monkeypatch.setattr(sj, "start_next_round", no_model)
    client = RootOnlyPredictor()
    root = mcts.run_mcts(
        state,
        client,
        iterations=100,
        boundary_samples=10,
        boundary_value_checkpoint="unused.pth",
    )

    assert len(applications) == 3
    assert len(client.states) == 1
    np.testing.assert_allclose(root.state_value, [0, 0.5, 0.5], atol=1e-6)


@pytest.mark.parametrize("first_action", [sj.MASK_TAKE, sj.MASK_DRAW])
def test_boundary_settings_survive_in_round_decision_and_chance_nodes(
    monkeypatch, first_action
):
    completed = completed_round(((0, 1, 2), (1, 2, 3), (1, 2, 3)), (10, 20, 30))
    state = dataclasses.replace(completed, countdown=2)
    card = int(np.argmax(state.table[1, 0, 0]))
    state.table[1, 0, 0] = 0
    state.table[1, 0, 0, sj.FINGER_HIDDEN] = 1
    state.deck[card] += 1
    evaluated = []

    class FirstActionPredictor(RootOnlyPredictor):
        def predict(self, observed):
            prediction = super().predict(observed)
            if observed is state:
                prediction.policy_output.fill(0)
                prediction.policy_output[first_action] = 1
            return prediction

    def predict_completed(model, states):
        assert len(states) == 10
        assert all(sj.get_round_over(item) for item in states)
        evaluated.append(states)
        return np.tile([0.3, 0.3, 0.4], (len(states), 1)).astype(np.float32)

    def no_next_round(*args, **kwargs):
        pytest.fail("Boundary configuration was lost in an in-round child")

    monkeypatch.setattr(boundary_inference, "load_boundary_model", lambda *args: None)
    monkeypatch.setattr(
        boundary_inference, "predict_completed_rounds", predict_completed
    )
    monkeypatch.setattr(sj, "start_next_round", no_next_round)
    root = mcts.run_mcts(
        state,
        FirstActionPredictor(),
        iterations=30,
        fpu_reduction=1,
        boundary_samples=10,
        boundary_value_checkpoint="frozen.pth",
    )
    assert evaluated
    assert root.visit_count == 30


@pytest.mark.parametrize("samples", [0, 1.5, True])
def test_invalid_sample_budget_fails_before_search(samples):
    with pytest.raises(ValueError, match="positive integer"):
        mcts.run_mcts(None, None, iterations=0, boundary_samples=samples)


def test_sampling_requires_an_explicit_boundary_model():
    with pytest.raises(ValueError, match="requires boundary_value_checkpoint"):
        mcts.run_mcts(None, None, iterations=0, boundary_samples=10)
