import numpy as np
import torch

from skyjo import skynet, train, train_utils


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        self.linear = torch.nn.Linear(1, 1, bias=False)

    def forward(
        self,
        spatial_tensor: torch.Tensor,
        non_spatial_tensor: torch.Tensor,
        mask: torch.Tensor,
    ) -> skynet.SkyNetOutput:
        del non_spatial_tensor, mask
        value = self.linear(spatial_tensor.reshape(-1, 1))
        policy_logits = torch.zeros(value.shape[0], 2, device=value.device)
        return skynet.EquivariantOutput(value, policy_logits)


class FakeReplayBuffer:
    def __init__(self, batch: train_utils.TrainingBatch, length: int):
        self.batch = batch
        self.length = length

    def __len__(self) -> int:
        return self.length

    def sample_batch(self, batch_size: int) -> train_utils.TrainingBatch:
        del batch_size
        return self.batch


def _batch() -> train_utils.TrainingBatch:
    return train_utils.TrainingBatch(
        spatial_inputs=np.ones((2, 1), dtype=np.float32),
        non_spatial_inputs=np.zeros((2, 1), dtype=np.float32),
        action_masks=np.ones((2, 2), dtype=np.float32),
        target_arrays={
            train_utils.VALUE_TARGET_NAME: np.zeros((2, 1), dtype=np.float32),
            train_utils.POLICY_TARGET_NAME: np.zeros((2, 2), dtype=np.float32),
        },
    )


def _loss(
    model_output: skynet.SupportsCoreSkyNetOutput,
    targets: train_utils.TensorTrainingTargets,
) -> tuple[torch.Tensor, train_utils.LossDetails]:
    del targets
    loss = model_output.value.sum()
    return loss, {"loss": loss.item()}


def test_train_epoch_reuses_supplied_optimizer_across_batches():
    model = ToyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    replay_buffer = FakeReplayBuffer(_batch(), length=3)

    train.train_epoch(
        model,
        replay_buffer,
        training_batch_size=2,
        optimizer=optimizer,
        loss_function=_loss,
    )

    parameter = next(model.parameters())
    assert optimizer.state[parameter]["step"].item() == 2
