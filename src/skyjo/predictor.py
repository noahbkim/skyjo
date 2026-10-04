"""Synchronous local inference for gameplay and search."""

from collections.abc import Sequence

import torch

from . import batches, skynet
from . import game as sj


class LocalPredictor:
    """Evaluate states in bounded batches, preserving input order."""

    def __init__(self, model: skynet.SkyNet, max_batch_size: int):
        if type(max_batch_size) is not int or max_batch_size <= 0:
            raise ValueError("max_batch_size must be a positive integer")
        self.model = model
        self.max_batch_size = max_batch_size

    def predict(self, state: sj.Skyjo) -> skynet.SkyNetPrediction:
        return self.predict_many([state])[0]

    @torch.inference_mode()
    def predict_many(self, states: Sequence[sj.Skyjo]) -> list[skynet.SkyNetPrediction]:
        """Return every prediction in order; an empty request returns no results."""
        if not states:
            return []
        self.model.eval()
        predictions = []
        for start in range(0, len(states), self.max_batch_size):
            batch = states[start : start + self.max_batch_size]
            tensors = batches.to_tensors(
                batches.states_to_batch(batch), device=self.model.device
            )
            output = skynet.output_to_numpy(
                self.model(
                    tensors.spatial_inputs,
                    tensors.non_spatial_inputs,
                    tensors.action_masks,
                )
            )
            predictions.extend(
                skynet.SkyNetPrediction.from_numpy_output(
                    skynet.get_single_model_output(output, index)
                )
                for index in range(len(batch))
            )
        return predictions
