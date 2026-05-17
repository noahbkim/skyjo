import collections.abc
import typing

import numpy as np
import pandas as pd
import torch

from . import play
from . import game as sj
from . import skynet

# MARK: Types

FloatArray: typing.TypeAlias = np.ndarray[tuple[int, ...], np.float32]
SpatialInput: typing.TypeAlias = FloatArray
NonSpatialInput: typing.TypeAlias = FloatArray
ActionMask: typing.TypeAlias = FloatArray
ValueTarget: typing.TypeAlias = FloatArray
PointsTarget: typing.TypeAlias = FloatArray
PolicyTarget: typing.TypeAlias = FloatArray
ClearedColumnsTarget: typing.TypeAlias = FloatArray


class NumpyTrainingTargets(typing.NamedTuple):
    value: ValueTarget
    points: PointsTarget | None
    policy: PolicyTarget
    cleared_columns: ClearedColumnsTarget | None


class TensorTrainingTargets(typing.NamedTuple):
    value: torch.Tensor
    points: torch.Tensor | None
    policy: torch.Tensor
    cleared_columns: torch.Tensor | None


TrainingTargets: typing.TypeAlias = TensorTrainingTargets


class TrainingDataPoint(typing.NamedTuple):
    spatial_input: SpatialInput
    non_spatial_input: NonSpatialInput
    action_mask: ActionMask
    value_target: ValueTarget
    points_target: PointsTarget
    policy_target: PolicyTarget
    cleared_columns_target: ClearedColumnsTarget

    @property
    def targets(self) -> NumpyTrainingTargets:
        return NumpyTrainingTargets(
            self.value_target,
            self.points_target,
            self.policy_target,
            self.cleared_columns_target,
        )


class TrainingBatch(typing.NamedTuple):
    spatial_inputs: SpatialInput
    non_spatial_inputs: NonSpatialInput
    action_masks: ActionMask
    value_targets: ValueTarget
    points_targets: PointsTarget
    policy_targets: PolicyTarget
    cleared_columns_targets: ClearedColumnsTarget

    @property
    def targets(self) -> NumpyTrainingTargets:
        return NumpyTrainingTargets(
            self.value_targets,
            self.points_targets,
            self.policy_targets,
            self.cleared_columns_targets,
        )


# MARK: Data Helpers

LossDetails: typing.TypeAlias = dict[
    str, float
]  # loss component name: loss component value


class LossFunction(typing.Protocol):
    def __call__(
        self,
        model_output: skynet.SupportsCoreSkyNetOutput,
        targets: TensorTrainingTargets,
    ) -> tuple[torch.Tensor, LossDetails]:
        ...


def as_numpy_training_targets(targets: typing.Any) -> NumpyTrainingTargets:
    if isinstance(targets, NumpyTrainingTargets):
        return targets
    if isinstance(targets, collections.abc.Mapping):
        return NumpyTrainingTargets(
            targets["value"],
            targets.get("points"),
            targets["policy"],
            targets.get("cleared_columns"),
        )
    if all(
        hasattr(targets, field)
        for field in ("value", "points", "policy", "cleared_columns")
    ):
        return NumpyTrainingTargets(
            targets.value,
            targets.points,
            targets.policy,
            targets.cleared_columns,
        )
    return NumpyTrainingTargets(*targets)


def game_stats_summary(game_stats_list: list[play.GameStats]) -> pd.DataFrame:
    stats_df = pd.DataFrame.from_records(
        [game_stats.to_record_dict() for game_stats in game_stats_list]
    )
    return stats_df.describe().T


def game_data_to_training_batch(
    game_data: play.GameData,
) -> TrainingBatch:
    spatial_states = [
        skynet.get_spatial_state_numpy(data_point.state) for data_point in game_data
    ]
    non_spatial_states = [
        skynet.get_non_spatial_state_numpy(data_point.state) for data_point in game_data
    ]
    action_masks = [sj.actions(data_point.state) for data_point in game_data]
    targets = [as_numpy_training_targets(data_point.targets) for data_point in game_data]
    value_targets = [target.value for target in targets]
    points_targets = [target.points for target in targets]
    policy_targets = [target.policy for target in targets]
    cleared_columns_targets = [target.cleared_columns for target in targets]
    return TrainingBatch(
        np.array(spatial_states, dtype=np.float32),
        np.array(non_spatial_states, dtype=np.float32),
        np.array(action_masks, dtype=np.float32),
        np.array(value_targets, dtype=np.float32),
        np.array(points_targets, dtype=np.float32),
        np.array(policy_targets, dtype=np.float32),
        np.array(cleared_columns_targets, dtype=np.float32),
    )


# MARK: Loss


def cross_entropy_policy_loss(
    predicted: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    return torch.nn.CrossEntropyLoss(reduction="mean")(predicted, target)


def mse_value_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.MSELoss(reduction="mean")(predicted, target)


def l1_value_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.L1Loss(reduction="mean")(predicted, target)


def mse_cleared_columns_loss(
    predicted: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    return torch.nn.MSELoss(reduction="mean")(predicted, target)


def policy_value_losses(
    model_output: skynet.SupportsCoreSkyNetOutput,
    targets: TensorTrainingTargets,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert model_output.policy_logits.shape == targets.policy.shape, (
        f"expected policy_logits of shape {targets.policy.shape}, got {model_output.policy_logits.shape}"
    )
    assert model_output.value.shape == targets.value.shape, (
        f"expected value of shape {targets.value.shape}, got {model_output.value.shape}"
    )
    policy_loss = cross_entropy_policy_loss(
        model_output.policy_logits,
        targets.policy,
    )
    value_loss = mse_value_loss(
        model_output.value,
        targets.value,
    )
    return value_loss, policy_loss


def policy_value_cleared_columns_losses(
    model_output: skynet.SupportsCoreSkyNetOutput,
    targets: TensorTrainingTargets,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert model_output.policy_logits.shape == targets.policy.shape, (
        f"expected policy_logits of shape {targets.policy.shape}, got {model_output.policy_logits.shape}"
    )
    assert model_output.value.shape == targets.value.shape, (
        f"expected value of shape {targets.value.shape}, got {model_output.value.shape}"
    )
    policy_loss = cross_entropy_policy_loss(
        model_output.policy_logits,
        targets.policy,
    )
    value_loss = mse_value_loss(
        model_output.value,
        targets.value,
    )
    # value_loss = l1_value_loss(
    #     value_output,
    #     value_targets,
    # )
    assert targets.cleared_columns is not None, (
        "expected cleared_columns target for cleared-columns loss"
    )
    cleared_columns_loss = mse_cleared_columns_loss(
        model_output.cleared_columns,
        targets.cleared_columns,
    )
    return value_loss, policy_loss, cleared_columns_loss


def base_loss(
    model_output: skynet.SupportsCoreSkyNetOutput,
    targets: TensorTrainingTargets,
    value_scale: float = 1 / (skynet.SCORE_DIFFERENTIAL_CAP**2),
    policy_scale: float = 1.0,
    cleared_columns_scale: float = 0.1,
) -> tuple[torch.Tensor, LossDetails]:
    value_loss, policy_loss, cleared_columns_loss = policy_value_cleared_columns_losses(
        model_output, targets
    )
    return (
        value_scale * value_loss
        + policy_scale * policy_loss
        + cleared_columns_scale * cleared_columns_loss,
        {
            "score_differential_value_loss": value_loss.item(),
            "policy_loss": policy_loss.item(),
            "cleared_columns_loss": cleared_columns_loss.item(),
        },
    )


def compute_model_loss_on_game_data(
    model: skynet.SkyNet,
    game_data: play.GameData,
    loss_function: typing.Callable[
        [skynet.SupportsCoreSkyNetOutput, TensorTrainingTargets], typing.Any
    ],
) -> typing.Any:
    batch = game_data_to_training_batch(game_data)
    spatial_tensor = torch.tensor(
        batch.spatial_inputs, dtype=torch.float32, device=model.device
    )
    non_spatial_tensor = torch.tensor(
        batch.non_spatial_inputs, dtype=torch.float32, device=model.device
    )
    mask_tensor = torch.tensor(
        batch.action_masks, dtype=torch.float32, device=model.device
    )
    policy_targets_tensor = torch.tensor(
        batch.policy_targets, dtype=torch.float32, device=model.device
    )
    value_targets_tensor = torch.tensor(
        batch.value_targets, dtype=torch.float32, device=model.device
    )
    points_targets_tensor = torch.tensor(
        batch.points_targets, dtype=torch.float32, device=model.device
    )
    cleared_columns_targets_tensor = torch.tensor(
        batch.cleared_columns_targets, dtype=torch.float32, device=model.device
    )
    model_output = model(spatial_tensor, non_spatial_tensor, mask_tensor)
    return loss_function(
        model_output,
        TensorTrainingTargets(
            value_targets_tensor,
            points_targets_tensor,
            policy_targets_tensor,
            cleared_columns_targets_tensor,
        ),
    )


def loss_details_summary(loss_details_list: list[LossDetails]) -> pd.DataFrame:
    return pd.DataFrame.from_records(loss_details_list).describe().T
