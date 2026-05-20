import collections.abc
import typing

import numpy as np
import pandas as pd
import torch

from . import game as sj
from . import play, skynet

# MARK: Types

FloatArray: typing.TypeAlias = np.ndarray[tuple[int, ...], np.float32]
SpatialInput: typing.TypeAlias = FloatArray
NonSpatialInput: typing.TypeAlias = FloatArray
ActionMask: typing.TypeAlias = FloatArray
ValueTarget: typing.TypeAlias = FloatArray
PolicyTarget: typing.TypeAlias = FloatArray
TargetArrays: typing.TypeAlias = dict[str, FloatArray]
VALUE_TARGET_NAME: typing.Final[str] = "value"
POLICY_TARGET_NAME: typing.Final[str] = "policy"
ROUND_SCORE_TARGET_NAME: typing.Final[str] = skynet.ROUND_SCORE_TARGET_NAME
CORE_TARGET_NAMES: typing.Final[tuple[str, ...]] = (
    VALUE_TARGET_NAME,
    POLICY_TARGET_NAME,
)


class NumpyTrainingTargets(typing.NamedTuple):
    value: ValueTarget
    policy: PolicyTarget


class TensorTrainingTargets(typing.NamedTuple):
    value: torch.Tensor
    policy: torch.Tensor
    target_tensors: dict[str, torch.Tensor] | None = None

    @property
    def targets(self) -> dict[str, torch.Tensor]:
        if self.target_tensors is not None:
            return self.target_tensors
        return {
            VALUE_TARGET_NAME: self.value,
            POLICY_TARGET_NAME: self.policy,
        }

    def __getitem__(self, key: typing.Any) -> torch.Tensor:
        if isinstance(key, str):
            return self.targets[key]
        return tuple.__getitem__(self, key)


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
        return self.target_arrays[VALUE_TARGET_NAME]

    @property
    def policy_target(self) -> PolicyTarget:
        return self.target_arrays[POLICY_TARGET_NAME]


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
        return self.target_arrays[VALUE_TARGET_NAME]

    @property
    def policy_targets(self) -> PolicyTarget:
        return self.target_arrays[POLICY_TARGET_NAME]


# MARK: Data Helpers

LossDetails: typing.TypeAlias = dict[
    str, float
]  # loss component name: loss component value


class LossFunction(typing.Protocol):
    def __call__(
        self,
        model_output: skynet.SupportsCoreSkyNetOutput,
        targets: TensorTrainingTargets,
    ) -> tuple[torch.Tensor, LossDetails]: ...


def as_numpy_training_targets(targets: typing.Any) -> NumpyTrainingTargets:
    target_dict = normalize_numpy_targets(targets, CORE_TARGET_NAMES)
    return NumpyTrainingTargets(
        target_dict[VALUE_TARGET_NAME],
        target_dict[POLICY_TARGET_NAME],
    )


def normalize_numpy_targets(
    targets: typing.Any,
    target_names: typing.Sequence[str] = CORE_TARGET_NAMES,
) -> TargetArrays:
    if isinstance(targets, NumpyTrainingTargets):
        return {
            VALUE_TARGET_NAME: targets.value,
            POLICY_TARGET_NAME: targets.policy,
        }
    if isinstance(targets, collections.abc.Mapping):
        return {name: targets[name] for name in target_names}
    if all(hasattr(targets, field) for field in target_names):
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
    if isinstance(targets, collections.abc.Mapping):
        target_arrays = dict(targets)
    else:
        target_arrays = normalize_numpy_targets(targets, CORE_TARGET_NAMES)
    target_tensors = {
        name: torch.tensor(target, dtype=torch.float32, device=device)
        for name, target in target_arrays.items()
    }
    return TensorTrainingTargets(
        target_tensors[VALUE_TARGET_NAME],
        target_tensors[POLICY_TARGET_NAME],
        target_tensors,
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
    batched_targets: TargetArrays = {
        name: np.array(
            [target[name] for target in normalized_targets], dtype=np.float32
        )
        for name in target_names
    }
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


def base_loss(
    model_output: skynet.SupportsCoreSkyNetOutput,
    targets: TensorTrainingTargets,
    value_scale: float = 1.0,
    policy_scale: float = 1.0,
) -> tuple[torch.Tensor, LossDetails]:
    value_loss, policy_loss = policy_value_losses(model_output, targets)
    return (
        value_scale * value_loss + policy_scale * policy_loss,
        {
            "outcome_value_loss": value_loss.item(),
            "policy_loss": policy_loss.item(),
        },
    )


def outcome_policy_round_score_loss(
    model_output: skynet.SupportsCoreSkyNetOutput,
    targets: TensorTrainingTargets,
    value_scale: float = 1.0,
    policy_scale: float = 1.0,
    round_score_scale: float = 0.1,
) -> tuple[torch.Tensor, LossDetails]:
    base, details = base_loss(
        model_output,
        targets,
        value_scale=value_scale,
        policy_scale=policy_scale,
    )
    auxiliary_outputs = getattr(model_output, "auxiliary_outputs", None)
    assert auxiliary_outputs is not None, "expected model output with auxiliary_outputs"
    assert ROUND_SCORE_TARGET_NAME in auxiliary_outputs, (
        f"expected auxiliary output {ROUND_SCORE_TARGET_NAME}"
    )
    round_score_prediction = auxiliary_outputs[ROUND_SCORE_TARGET_NAME]
    round_score_target = targets[ROUND_SCORE_TARGET_NAME]
    assert round_score_prediction.shape == round_score_target.shape, (
        f"expected {ROUND_SCORE_TARGET_NAME} of shape {round_score_target.shape}, "
        f"got {round_score_prediction.shape}"
    )
    round_score_loss = mse_value_loss(round_score_prediction, round_score_target)
    details[f"{ROUND_SCORE_TARGET_NAME}_loss"] = round_score_loss.item()
    return base + round_score_scale * round_score_loss, details


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
