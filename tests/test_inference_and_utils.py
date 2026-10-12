"""Inference owns batching, legal policies, and fixed-seat value conversion."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from skyjo.engine import game as sj
from skyjo.learning import batches, observations, predictor, skynet
from skyjo.search.evaluator import Prediction


class InferenceCheckingSkyNet(skynet.EquivariantSkyNet):
    def forward(self, *args, **kwargs):
        assert torch.is_inference_mode_enabled()
        assert not self.training
        return super().forward(*args, **kwargs)


def make_model(auxiliary=None):
    torch.manual_seed(3)
    return InferenceCheckingSkyNet(
        spatial_input_shape=(2, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
        non_spatial_input_shape=observations.get_non_spatial_input_shape(2),
        value_output_shape=(2,),
        policy_output_shape=(sj.MASK_SIZE,),
        embedding_dimensions=4,
        global_state_embedding_dimensions=8,
        num_heads=1,
        auxiliary_objectives=auxiliary,
        device=torch.device("cpu"),
    )


@pytest.mark.parametrize(
    "auxiliary",
    [{}, {"round_score": 0.1, "round_raw_score": 0.1, "round_doubled": 0.1}],
)
def test_inference_preserves_order_and_perspective_across_bounded_batches(auxiliary):
    model = make_model(auxiliary)
    inference = predictor.LocalPredictor(model, max_batch_size=2)
    rng = np.random.default_rng(5)
    states = [sj.start_round(sj.new(players=2, top=card), rng=rng) for card in range(5)]
    states[1] = sj.apply_action(states[1], sj.MASK_FLIP_SECOND_RIGHT, rng=rng)
    model.eval()
    tensors = batches.to_tensors(batches.states_to_batch(states), device=model.device)
    with torch.inference_mode():
        raw = model(
            tensors.spatial_inputs, tensors.non_spatial_inputs, tensors.action_masks
        )
        expected_policies = torch.softmax(raw.policy_logits, dim=-1).numpy()
    model.train()
    actual = inference.evaluate(states)
    assert not model.training
    assert len(actual) == len(states)
    for index, (state, prediction) in enumerate(zip(states, actual, strict=True)):
        np.testing.assert_allclose(
            prediction.value,
            np.roll(raw.value[index].numpy(), sj.get_player(state)),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            prediction.policy, expected_policies[index], atol=1e-6
        )
        assert prediction.policy.sum() == pytest.approx(1)
        assert not prediction.policy[sj.actions(state) == 0].any()
    assert inference.evaluate([]) == []
    assert len(inference.evaluate(states[:1])) == 1


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_inference_rejects_invalid_batch_limit(limit):
    with pytest.raises(ValueError, match="positive integer"):
        predictor.LocalPredictor(make_model(), limit)


@pytest.mark.parametrize("legal_scores,expected", [([2.0, 5.0], 1), ([5.0, 5.0], 0)])
def test_policy_player_chooses_legal_argmax(legal_scores, expected):
    state = sj.new(players=2, top=sj.CARD_0)
    scores = np.full(sj.MASK_SIZE, 100.0, dtype=np.float32)
    scores[:2] = legal_scores
    result = Prediction(np.array([0.5, 0.5]), scores)
    agent = predictor.PolicyPlayer(SimpleNamespace(evaluate=lambda states: [result]))
    actual = agent.get_action_probabilities(state)
    target = np.zeros(sj.MASK_SIZE)
    target[expected] = 1
    np.testing.assert_array_equal(actual, target)
