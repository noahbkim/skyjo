"""Offline score-only prediction of full-game winners between rounds."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
from torch import nn


class BoundaryValueModel(nn.Module):
    """Use absolute totals, in next-starter order, to predict win credit."""

    def __init__(self, kind: str, players: int, hidden_width: int = 32):
        super().__init__()
        if kind not in {"logistic", "mlp"}:
            raise ValueError("kind must be 'logistic' or 'mlp'")
        if players < 2 or hidden_width < 1:
            raise ValueError("players must be at least two and hidden_width positive")
        self.kind = kind
        self.players = players
        self.hidden_width = hidden_width
        if kind == "logistic":
            self.network = nn.Linear(players, players)
        else:
            self.network = nn.Sequential(
                nn.Linear(players, hidden_width),
                nn.ReLU(),
                nn.Linear(hidden_width, hidden_width),
                nn.ReLU(),
                nn.Linear(hidden_width, players),
            )

    def logits(self, scores: torch.Tensor) -> torch.Tensor:
        return self.network(scores / 100.0)

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        return self.logits(scores).softmax(dim=-1)


@dataclass
class FitResult:
    model: BoundaryValueModel
    history: list[dict]
    best_epoch: int
    optimizer_steps: int
    training_seconds: float


def _matrix(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.ndim != 2 or result.shape[1] < 2 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite N-by-players matrix")
    return result


def _distribution(values: np.ndarray, name: str) -> np.ndarray:
    result = _matrix(values, name)
    if (result < 0).any() or not np.allclose(result.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError(f"{name} rows must be probability distributions")
    return result


def _indices(values: np.ndarray, name: str, count: int) -> np.ndarray:
    result = np.asarray(values)
    if (
        result.ndim != 1
        or not np.issubdtype(result.dtype, np.integer)
        or len(result) == 0
        or (result < 0).any()
        or (result >= count).any()
        or len(np.unique(result)) != len(result)
    ):
        raise ValueError(f"{name} must contain distinct, valid row indices")
    return result


def fit_model(
    scores: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    *,
    kind: str,
    seed: int,
    epochs: int = 200,
    batch_size: int = 256,
    learn_rate: float = 0.003,
    weight_decay: float = 0.0001,
    hidden_width: int = 32,
) -> FitResult:
    """Fit with soft-target CE; restore the epoch with least validation MSE.

    The local minibatch stream is independent of model initialization, so both
    architectures see the same ordering for a given seed. Test rows are unused.
    """
    scores = _matrix(scores, "scores")
    targets = _distribution(targets, "targets")
    if scores.shape != targets.shape:
        raise ValueError("scores and targets must have the same shape")
    train_indices = _indices(train_indices, "train_indices", len(scores))
    validation_indices = _indices(
        validation_indices, "validation_indices", len(scores)
    )
    if np.intersect1d(train_indices, validation_indices).size:
        raise ValueError("training and validation rows must be disjoint")
    if epochs < 1 or batch_size < 1 or learn_rate <= 0 or weight_decay < 0:
        raise ValueError("invalid optimizer or training budget settings")

    start = perf_counter()
    history = []
    optimizer_steps = 0
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    batches = np.random.default_rng(seed)
    score_tensor = torch.from_numpy(scores)
    target_tensor = torch.from_numpy(targets)
    validation_scores = score_tensor[validation_indices]
    validation_targets = target_tensor[validation_indices]

    # Initialize CPU weights without changing the caller's random state.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = BoundaryValueModel(kind, scores.shape[1], hidden_width)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learn_rate, weight_decay=weight_decay
        )
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            ordering = batches.permutation(train_indices)
            for offset in range(0, len(ordering), batch_size):
                indices = ordering[offset : offset + batch_size]
                logits = model.logits(score_tensor[indices])
                loss = -(
                    target_tensor[indices] * logits.log_softmax(dim=-1)
                ).sum(dim=-1).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(indices)
                optimizer_steps += 1
            model.eval()
            with torch.no_grad():
                validation_mse = (
                    (model(validation_scores) - validation_targets).square().mean().item()
                )
            history.append(
                {
                    "epoch": epoch,
                    "train_cross_entropy": total_loss / len(train_indices),
                    "validation_mse": validation_mse,
                }
            )
            if validation_mse < best_loss:
                best_loss = validation_mse
                best_epoch = epoch
                best_state = {
                    name: value.detach().clone()
                    for name, value in model.state_dict().items()
                }
        model.load_state_dict(best_state)
    return FitResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        optimizer_steps=optimizer_steps,
        training_seconds=perf_counter() - start,
    )


def predict(model: BoundaryValueModel, scores: np.ndarray) -> np.ndarray:
    """Return a CPU array of per-player win probabilities."""
    scores = _matrix(scores, "scores")
    if scores.shape[1] != model.players:
        raise ValueError("score player count does not match the model")
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(scores)).cpu().numpy()


def _metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    count, players = predictions.shape
    flat_predictions = predictions.ravel()
    flat_targets = targets.ravel()
    bin_ids = np.minimum((flat_predictions * 10).astype(int), 9)
    calibration = []
    weighted_error = 0.0
    for bin_id in range(10):
        mask = bin_ids == bin_id
        bin_count = int(mask.sum())
        mean_prediction = float(flat_predictions[mask].mean()) if bin_count else None
        win_credit = float(flat_targets[mask].mean()) if bin_count else None
        if bin_count:
            weighted_error += bin_count * abs(mean_prediction - win_credit)
        calibration.append(
            {
                "lower": bin_id / 10,
                "upper": (bin_id + 1) / 10,
                "count": bin_count,
                "mean_prediction": mean_prediction,
                "observed_win_credit": win_credit,
            }
        )
    squared_errors = (predictions - targets) ** 2
    return {
        "count": count,
        "player_predictions": count * players,
        "value_mse": float(squared_errors.mean()) if count else None,
        "brier_score": float(squared_errors.sum(axis=1).mean()) if count else None,
        "cross_entropy": (
            float(-(targets * np.log(np.clip(predictions, 1e-12, 1))).sum(axis=1).mean())
            if count else None
        ),
        "ece": weighted_error / (count * players) if count else None,
        "calibration": calibration,
    }


def probability_metrics(
    predictions: np.ndarray, targets: np.ndarray, scores: np.ndarray
) -> dict:
    """Report value MSE, sum-over-players Brier score, CE and pooled ECE.

    Calibration pools player probabilities in ten equal-width bins; the final
    bin includes probability one. Ties contribute fractional observed credit.
    Rows are equally weighted, so games with more boundaries contribute more.
    """
    predictions = _distribution(predictions, "predictions").astype(np.float64)
    targets = _distribution(targets, "targets").astype(np.float64)
    scores = _matrix(scores, "scores")
    if predictions.shape != targets.shape or predictions.shape != scores.shape:
        raise ValueError("predictions, targets and scores must have the same shape")
    result = _metrics(predictions, targets)
    max_scores = scores.max(axis=1)
    strata = {
        "below_50": max_scores < 50,
        "50_to_80": (max_scores >= 50) & (max_scores < 80),
        "at_least_80": max_scores >= 80,
    }
    result["by_max_score"] = {
        name: _metrics(predictions[mask], targets[mask])
        for name, mask in strata.items()
    }
    return result
