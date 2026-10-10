import numpy as np
import pytest
import torch

from skyjo.boundary_value import (
    BoundaryValueModel,
    fit_model,
    predict,
    probability_metrics,
)


def test_probability_metrics_preserve_tie_credit_and_distinguish_brier_from_mse():
    predictions = np.array([[0.75, 0.25], [0.5, 0.5], [1.0, 0.0]])
    targets = np.array([[1.0, 0.0], [0.5, 0.5], [1.0, 0.0]])
    scores = np.array([[5, 10], [80, 90], [90, 95]])

    metrics = probability_metrics(predictions, targets, scores)

    assert metrics["value_mse"] == pytest.approx(1 / 48)
    assert metrics["brier_score"] == pytest.approx(1 / 24)
    assert metrics["cross_entropy"] == pytest.approx((-np.log(0.75) + np.log(2)) / 3)
    assert metrics["ece"] == pytest.approx(1 / 12)
    calibration = metrics["calibration"]
    assert sum(item["count"] for item in calibration) == 6
    assert calibration[0]["count"] == calibration[9]["count"] == 1
    assert calibration[5]["observed_win_credit"] == 0.5
    assert calibration[5]["mean_prediction"] == 0.5
    empty = metrics["by_max_score"]["50_to_80"]
    assert empty["count"] == 0
    assert empty["value_mse"] is None
    assert empty["ece"] is None


def test_score_model_uses_absolute_totals_and_reconstructs_probabilities():
    model = BoundaryValueModel("logistic", players=2)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim == 2:
                parameter.copy_(torch.arange(4).reshape(2, 2))
            else:
                parameter.zero_()
    scores = np.array([[10, 20], [80, 90]], dtype=np.float32)

    probabilities = predict(model, scores)

    expected = torch.tensor([[0.2, 0.8], [0.9, 4.3]]).softmax(dim=-1).numpy()
    np.testing.assert_allclose(probabilities, expected, atol=1e-7)
    assert probabilities[0, 0] != pytest.approx(probabilities[1, 0])
    restored = BoundaryValueModel(model.kind, model.players, model.hidden_width)
    restored.load_state_dict(model.state_dict())
    np.testing.assert_array_equal(predict(restored, scores), probabilities)


@pytest.mark.parametrize("kind", ["logistic", "mlp"])
def test_fitting_is_reproducible_and_restores_best_validation_epoch(kind):
    # Identical observations with opposing train/validation labels make later
    # optimization worse on validation, exposing a last-epoch restore mistake.
    scores = np.zeros((12, 2), dtype=np.float32)
    targets = np.array([[1, 0]] * 8 + [[0, 1]] * 4, dtype=np.float32)
    train_indices = np.arange(8)
    validation_indices = np.arange(8, 12)
    options = dict(kind=kind, seed=5, batch_size=4, learn_rate=0.05)
    random_state = torch.random.get_rng_state().clone()

    fit = fit_model(
        scores, targets, train_indices, validation_indices, epochs=6, **options
    )

    assert torch.equal(torch.random.get_rng_state(), random_state)
    assert fit.best_epoch == 1
    assert fit.optimizer_steps == 12
    validation_mse = probability_metrics(
        predict(fit.model, scores[validation_indices]),
        targets[validation_indices],
        scores[validation_indices],
    )["value_mse"]
    assert validation_mse == pytest.approx(min(row["validation_mse"] for row in fit.history))
    repeated = fit_model(
        scores, targets, train_indices, validation_indices, epochs=1, **options
    )
    np.testing.assert_array_equal(predict(fit.model, scores), predict(repeated.model, scores))
    assert repeated.history == fit.history[:1]
    np.testing.assert_allclose(predict(fit.model, scores).sum(axis=1), 1.0)


def test_fit_rejects_training_validation_overlap():
    with pytest.raises(ValueError, match="disjoint"):
        fit_model(
            np.zeros((3, 2)), np.full((3, 2), 0.5),
            np.array([0, 1]), np.array([1, 2]), kind="logistic", seed=0,
        )
