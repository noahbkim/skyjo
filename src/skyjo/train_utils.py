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
TargetArrays: typing.TypeAlias = dict[str, FloatArray | None]
CORE_TARGET_NAMES: typing.Final[tuple[str, ...]] = (
    "value",
    "points",
    "policy",
    "cleared_columns",
)


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
    target_arrays: TargetArrays

    @property
    def targets(self) -> TargetArrays:
        return self.target_arrays

    @property
    def value_target(self) -> ValueTarget:
        return self.target_arrays["value"]

    @property
    def points_target(self) -> PointsTarget | None:
        return self.target_arrays.get("points")

    @property
    def policy_target(self) -> PolicyTarget:
        return self.target_arrays["policy"]

    @property
    def cleared_columns_target(self) -> ClearedColumnsTarget | None:
        return self.target_arrays.get("cleared_columns")


class TrainingBatch(typing.NamedTuple):
    spatial_inputs: SpatialInput
    non_spatial_inputs: NonSpatialInput
    action_masks: ActionMask
    target_arrays: TargetArrays

    @property
    def targets(self) -> TargetArrays:
        return self.target_arrays

    @property
    def value_targets(self) -> ValueTarget:
        return self.target_arrays["value"]

    @property
    def points_targets(self) -> PointsTarget | None:
        return self.target_arrays.get("points")

    @property
    def policy_targets(self) -> PolicyTarget:
        return self.target_arrays["policy"]

    @property
    def cleared_columns_targets(self) -> ClearedColumnsTarget | None:
        return self.target_arrays.get("cleared_columns")


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
    target_dict = normalize_numpy_targets(targets, CORE_TARGET_NAMES)
    return NumpyTrainingTargets(
        target_dict["value"],
        target_dict.get("points"),
        target_dict["policy"],
        target_dict.get("cleared_columns"),
    )


def normalize_numpy_targets(
    targets: typing.Any,
    target_names: typing.Sequence[str] = CORE_TARGET_NAMES,
) -> TargetArrays:
    if isinstance(targets, NumpyTrainingTargets):
        return {
            "value": targets.value,
            "points": targets.points,
            "policy": targets.policy,
            "cleared_columns": targets.cleared_columns,
        }
    if isinstance(targets, collections.abc.Mapping):
        return {name: targets.get(name) for name in target_names}
    if all(
        hasattr(targets, field)
        for field in target_names
    ):
        return {name: getattr(targets, name) for name in target_names}
    target_values = tuple(targets)
    assert len(target_values) == len(target_names), (
        f"expected {len(target_names)} targets, got {len(target_values)}"
    )
    return {
        name: value for name, value in zip(target_names, target_values, strict=True)
    }


def numpy_targets_to_tensors(
    targets: NumpyTrainingTargets | TargetArrays,
    *,
    device: torch.device,
) -> TensorTrainingTargets:
    normalized_targets = as_numpy_training_targets(targets)
    return TensorTrainingTargets(
        torch.tensor(normalized_targets.value, dtype=torch.float32, device=device),
        None
        if normalized_targets.points is None
        else torch.tensor(
            normalized_targets.points, dtype=torch.float32, device=device
        ),
        torch.tensor(normalized_targets.policy, dtype=torch.float32, device=device),
        None
        if normalized_targets.cleared_columns is None
        else torch.tensor(
            normalized_targets.cleared_columns, dtype=torch.float32, device=device
        ),
    )


def game_stats_summary(game_stats_list: list[play.GameStats]) -> pd.DataFrame:
    stats_df = pd.DataFrame.from_records(
        [game_stats.to_record_dict() for game_stats in game_stats_list]
    )
    return stats_df.describe().T


def game_data_to_training_batch(
    game_data: play.GameData,
    target_names: typing.Sequence[str] = CORE_TARGET_NAMES,
) -> TrainingBatch:
    spatial_states = [
        skynet.get_spatial_state_numpy(data_point.state) for data_point in game_data
    ]
    non_spatial_states = [
        skynet.get_non_spatial_state_numpy(data_point.state) for data_point in game_data
    ]
    action_masks = [sj.actions(data_point.state) for data_point in game_data]
    normalized_targets = [
        normalize_numpy_targets(data_point.targets, target_names)
        for data_point in game_data
    ]
    batched_targets: TargetArrays = {}
    for name in target_names:
        field_values = [target[name] for target in normalized_targets]
        if any(value is None for value in field_values):
            assert all(value is None for value in field_values), (
                f"expected target field {name} to be consistently present or absent"
            )
            batched_targets[name] = None
        else:
            batched_targets[name] = np.array(field_values, dtype=np.float32)
    return TrainingBatch(
        np.array(spatial_states, dtype=np.float32),
        np.array(non_spatial_states, dtype=np.float32),
        np.array(action_masks, dtype=np.float32),
        batched_targets,
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
    tensor_targets = numpy_targets_to_tensors(
        batch.targets,
        device=model.device,
    )
    model_output = model(spatial_tensor, non_spatial_tensor, mask_tensor)
    return loss_function(
        model_output,
        tensor_targets,
    )


def loss_details_summary(loss_details_list: list[LossDetails]) -> pd.DataFrame:
    return pd.DataFrame.from_records(loss_details_list).describe().T
