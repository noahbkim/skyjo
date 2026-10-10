from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from skyjo import game as sj
from skyjo import mcts, observations, player, predictor, skynet


class InferenceCheckingSkyNet(skynet.EquivariantSkyNet):
    def forward(self, *args, **kwargs):
        assert torch.is_inference_mode_enabled()
        assert not self.training
        return super().forward(*args, **kwargs)


def make_model(auxiliary_objectives=None):
    torch.manual_seed(3)
    return InferenceCheckingSkyNet(
        spatial_input_shape=(2, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
        non_spatial_input_shape=observations.get_non_spatial_input_shape(2),
        value_output_shape=(2,),
        policy_output_shape=(sj.MASK_SIZE,),
        embedding_dimensions=4,
        global_state_embedding_dimensions=8,
        num_heads=1,
        auxiliary_objectives=auxiliary_objectives,
        device=torch.device("cpu"),
    )


def test_direct_predict_has_guaranteed_inference_mode() -> None:
    model = make_model()
    model.train()
    prediction = model.predict(sj.new(players=2, top=sj.CARD_0))
    assert not model.training
    assert isinstance(prediction.value_output, np.ndarray)
    assert isinstance(prediction.policy_output, np.ndarray)


@pytest.mark.parametrize(
    "auxiliary",
    [
        {},
        {"round_score": 0.1, "round_raw_score": 0.1, "round_doubled": 0.1},
    ],
)
def test_local_inference_preserves_all_predictions_across_chunks(auxiliary):
    model = make_model(auxiliary)
    inference = predictor.LocalPredictor(model, max_batch_size=2)
    # Distinct observations make output ordering observable across three chunks.
    states = [sj.new(players=2, top=card) for card in range(5)]
    expected = [model.predict(state) for state in states]
    model.train()
    actual = inference.predict_many(states)
    assert not model.training
    assert len(actual) == len(states)
    for state, observed, direct in zip(states, actual, expected, strict=True):
        np.testing.assert_allclose(
            observed.value_output, direct.value_output, atol=1e-6
        )
        np.testing.assert_allclose(
            observed.policy_output, direct.policy_output, atol=1e-6
        )
        np.testing.assert_allclose(
            observed.policy_logits, direct.policy_logits, atol=1e-6
        )
        assert np.isclose(observed.policy_output.sum(), 1.0)
        assert not observed.policy_output[sj.actions(state) == 0].any()
        assert set(observed.auxiliary_outputs or {}) == set(auxiliary)
        for name in auxiliary:
            np.testing.assert_allclose(
                observed.auxiliary_outputs[name],
                direct.auxiliary_outputs[name],
                atol=1e-6,
            )
    assert inference.predict_many([]) == []
    singleton = inference.predict(states[-1])
    np.testing.assert_allclose(
        singleton.value_output, expected[-1].value_output, atol=1e-6
    )
    # A subsequent request must contain only its own results.
    assert len(inference.predict_many(states[:1])) == 1


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_local_inference_rejects_invalid_batch_limit(limit):
    with pytest.raises(ValueError, match="positive integer"):
        predictor.LocalPredictor(make_model(), max_batch_size=limit)


@pytest.mark.parametrize("legal_scores,expected", [([2.0, 5.0], 1), ([5.0, 5.0], 0)])
def test_policy_player_chooses_legal_argmax_without_search(monkeypatch, legal_scores, expected):
    state = sj.new(players=2, top=sj.CARD_0)
    # Every illegal action outranks the two legal initial-reveal actions.
    scores = np.full(sj.MASK_SIZE, 100.0, dtype=np.float32)
    scores[:2] = legal_scores
    prediction = skynet.SkyNetPrediction(
        value_output=np.array([0.5, 0.5]),
        policy_output=np.exp(scores - scores.max()),
        policy_logits=scores,
    )
    calls = []

    def predict(observed):
        assert observed is state
        calls.append(observed)
        return prediction

    def forbidden_search(*args, **kwargs):
        pytest.fail("Policy-only play must not call MCTS")

    monkeypatch.setattr(mcts, "run_mcts", forbidden_search)
    agent = player.PolicyPlayer(SimpleNamespace(predict=predict))
    actual = agent.get_action_probabilities(state)
    target = np.zeros(sj.MASK_SIZE)
    target[expected] = 1
    np.testing.assert_array_equal(actual, target)
    assert len(calls) == 1
