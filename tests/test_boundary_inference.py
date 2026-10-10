import numpy as np
import pytest
import torch

from skyjo.learning.boundary_inference import (
    load_boundary_model,
    predict_completed_rounds,
)
from skyjo.learning.boundary_value import BoundaryValueModel
from test_full_game import completed_round


def test_completed_scores_are_charged_once_and_values_return_in_fixed_seats():
    model = BoundaryValueModel("logistic", players=3)
    with torch.no_grad():
        model.network.weight.copy_(torch.eye(3))
        model.network.bias.zero_()
    state = completed_round(
        ((-2, 0, 1), (0, 1, 2), (1, 2, 3)), (10, 20, 30), ending_player=2
    )
    # Absolute totals are 9, 23, 42: finisher 2 pays double its six points.
    expected = np.exp(np.array([9, 23, 42]) / 100)
    expected /= expected.sum()
    actual = predict_completed_rounds(model, [state])
    np.testing.assert_allclose(actual[0], expected, rtol=1e-6)


def test_mixed_terminal_and_continuing_outcomes_keep_exact_tie_mass():
    model = BoundaryValueModel("logistic", players=3)
    with torch.no_grad():
        model.network.weight.zero_()
        model.network.bias.copy_(torch.log(torch.tensor([0.1, 0.3, 0.6])))
    continuing = completed_round(
        ((0, 1, 2), (1, 2, 3), (2, 3, 4)), (10, 20, 30), ending_player=2
    )
    terminal = completed_round(
        ((0, 1, 2), (1, 2, 3), (11, 12, 12)), (7, 4, 40), ending_player=2
    )
    actual = predict_completed_rounds(model, [continuing, terminal])
    np.testing.assert_allclose(actual, [[0.3, 0.6, 0.1], [0.5, 0.5, 0]], atol=1e-7)
    np.testing.assert_array_equal(
        predict_completed_rounds(None, [terminal]), [[0.5, 0.5, 0]]
    )
    with pytest.raises(ValueError, match="matching boundary model"):
        predict_completed_rounds(None, [continuing])


def test_checkpoint_load_is_frozen_cached_and_preserves_rng(tmp_path):
    model = BoundaryValueModel("mlp", 2)
    path = tmp_path / "value.pth"
    payload = {
        "format": "skyjo.boundary-value",
        "version": 1,
        "kind": "mlp",
        "players": 2,
        "hidden_width": 32,
        "score_scale": 100.0,
        "input_order": "next starter first, then cyclic seat order",
        "model_state_dict": model.state_dict(),
    }
    torch.save(payload, path)
    before = torch.get_rng_state().clone()
    loaded = load_boundary_model(path, 2)
    assert torch.equal(before, torch.get_rng_state())
    assert loaded is load_boundary_model(path, 2)
    assert not loaded.training
    assert not any(parameter.requires_grad for parameter in loaded.parameters())
    scores = torch.tensor([[20.0, 50.0]])
    torch.testing.assert_close(loaded(scores), model(scores))
    with pytest.raises(ValueError, match="player count"):
        load_boundary_model(path, 3)
    payload["score_scale"] = 1.0
    torch.save(payload, path)
    with pytest.raises(ValueError, match="score encoding"):
        load_boundary_model(path, 2)
