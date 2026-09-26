from __future__ import annotations

import numpy as np
import torch

from skyjo import game as sj
from skyjo import skynet


def test_numpy_batch_policy_renormalization_handles_zero_rows() -> None:
    policies = np.array([[0.2, 0.8, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    masks = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.int8)
    result = skynet.batch_mask_and_renormalize_policy_probabilities(policies, masks)
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, [[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]])


class InferenceCheckingSkyNet(skynet.SimpleSkyNet):
    def forward(self, *args, **kwargs):
        assert torch.is_inference_mode_enabled()
        return super().forward(*args, **kwargs)


def test_direct_predict_has_guaranteed_inference_mode() -> None:
    model = InferenceCheckingSkyNet(
        spatial_input_shape=(2, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
        non_spatial_input_shape=skynet.get_non_spatial_input_shape(2),
        value_output_shape=(2,),
        policy_output_shape=(sj.MASK_SIZE,),
        hidden_layers=[8],
        device=torch.device("cpu"),
    )
    model.train()
    prediction = model.predict(sj.new(players=2, top=sj.CARD_0))
    assert not model.training
    assert isinstance(prediction.value_output, np.ndarray)
    assert isinstance(prediction.policy_output, np.ndarray)
