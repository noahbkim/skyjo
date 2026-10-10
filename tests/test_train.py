import numpy as np
import pytest
import torch

from skyjo.learning import batches, checkpoint, losses, skynet, train


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
    ) -> skynet.ModelOutput:
        del non_spatial_tensor, mask
        value = self.linear(spatial_tensor.reshape(-1, 1))
        policy_logits = torch.zeros(value.shape[0], 2, device=value.device)
        return skynet.ModelOutput(value, policy_logits)


class FakeReplayBuffer:
    def __init__(self, batch: batches.TrainingBatch, length: int):
        self.batch = batch
        self.length = length

    def __len__(self) -> int:
        return self.length

    def sample_batch(self, batch_size: int, *, rng) -> batches.TrainingBatch:
        del batch_size
        return self.batch


def _batch() -> batches.TrainingBatch:
    return batches.TrainingBatch(
        spatial_inputs=np.ones((2, 1), dtype=np.float32),
        non_spatial_inputs=np.zeros((2, 1), dtype=np.float32),
        action_masks=np.ones((2, 2), dtype=np.float32),
        targets={
            batches.VALUE_TARGET_NAME: np.zeros((2, 1), dtype=np.float32),
            batches.POLICY_TARGET_NAME: np.zeros((2, 2), dtype=np.float32),
        },
    )


def _loss(
    model_output: skynet.ModelOutput,
    targets: batches.TensorTargets,
) -> tuple[torch.Tensor, losses.LossDetails]:
    del targets
    loss = model_output.value.sum()
    return loss, {"loss": loss.item()}


def test_train_steps_runs_exact_optimizer_step_count():
    model = ToyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    replay_buffer = FakeReplayBuffer(_batch(), length=3)

    losses = train.train_steps(
        model,
        replay_buffer,
        training_batch_size=2,
        optimizer_steps=3,
        optimizer=optimizer,
        loss_function=_loss,
        sampling_rng=np.random.default_rng(0),
    )

    parameter = next(model.parameters())
    assert optimizer.state[parameter]["step"].item() == 3
    assert len(losses) == 3
    assert all("total_loss" in details for details in losses)


def test_evaluation_weights_selected_positions_and_restores_runtime():

    model = ToyModel()
    with torch.no_grad():
        model.linear.weight.fill_(2)
    model.train()
    model.linear.eval()  # Preserve mixed module modes, too.

    class Replay:
        def batch_indices(self, indices):
            values = np.array([[1], [2], [4]], dtype=np.float32)[indices]
            return batches.TrainingBatch(
                values,
                values,
                np.ones((len(indices), 2)),
                {"value": np.zeros_like(values), "policy": np.zeros((len(indices), 2))},
            )

    def observed_loss(output, targets):
        torch.rand(1)
        return losses.base_loss(output, targets, policy_scale=0)

    before = checkpoint.capture_rng_state()
    for batch_size in (1, 2):
        result = train.evaluate_loss(
            model, Replay(), batch_size, observed_loss, indices=np.array([2, 0, 2])
        )
        assert result["total_loss"] == pytest.approx(44)
        assert model.training and not model.linear.training
        assert torch.equal(torch.get_rng_state(), before["torch_cpu"])

    def failed_loss(output, targets):
        torch.rand(1)
        raise RuntimeError("evaluation failed")

    with pytest.raises(RuntimeError, match="evaluation failed"):
        train.evaluate_loss(model, Replay(), 2, failed_loss, indices=np.array([0]))
    assert model.training and not model.linear.training
    assert torch.equal(torch.get_rng_state(), before["torch_cpu"])
