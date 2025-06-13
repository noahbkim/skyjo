import typing

import numpy as np
import torch

import play
import skyjo as sj
import skynet

# MARK: Types

TrainingDataPoint: typing.TypeAlias = tuple[
    np.ndarray[tuple[int], np.float32],  # spatial input
    np.ndarray[tuple[int], np.float32],  # non-spatial input
    np.ndarray[tuple[int], np.float32],  # outcome target
    np.ndarray[tuple[int], np.float32],  # points target
    np.ndarray[tuple[int], np.float32],  # policy target
]
TrainingBatch: typing.TypeAlias = tuple[
    np.ndarray[tuple[int, int, int, int, int], np.float32],  # spatial input
    np.ndarray[tuple[int, int], np.float32],  # non-spatial input
    np.ndarray[tuple[int, int], np.float32],  # outcome target
    np.ndarray[tuple[int, int], np.float32],  # points target
    np.ndarray[tuple[int, int], np.float32],  # policy target
    np.ndarray[tuple[int, int], np.float32],  # action mask
]
TrainingTargets: typing.TypeAlias = tuple[
    torch.Tensor,  # outcome target
    torch.Tensor,  # points target
    torch.Tensor,  # policy target
]


# MARK: Data Helpers


def game_data_to_training_batch(
    game_data: play.GameData,
) -> TrainingBatch:
    spatial_states = [
        skynet.get_spatial_state_numpy(state_tuple[0]) for state_tuple in game_data
    ]
    non_spatial_states = [
        skynet.get_non_spatial_state_numpy(state_tuple[0]) for state_tuple in game_data
    ]
    action_masks = [sj.actions(state_tuple[0]) for state_tuple in game_data]
    policy_targets = [state_tuple[1] for state_tuple in game_data]
    outcome_targets = [state_tuple[2] for state_tuple in game_data]
    points_targets = [state_tuple[3] for state_tuple in game_data]
    return (
        np.array(spatial_states, dtype=np.float32),
        np.array(non_spatial_states, dtype=np.float32),
        np.array(outcome_targets, dtype=np.float32),
        np.array(points_targets, dtype=np.float32),
        np.array(policy_targets, dtype=np.float32),
        np.array(action_masks, dtype=np.float32),
    )


# MARK: Loss


def cross_entropy_policy_loss(
    predicted: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    return torch.nn.CrossEntropyLoss(reduction="mean")(predicted, target)


def mse_value_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.MSELoss(reduction="mean")(predicted, target)


def policy_value_losses(
    model_output: skynet.SkyNetOutput,
    targets: TrainingTargets,
) -> torch.Tensor:
    value_output, points_output, policy_output = model_output
    value_targets, points_targets, policy_targets = targets
    assert policy_output.shape == policy_targets.shape, (
        f"expected policy_output of shape {policy_targets.shape}, got {policy_output.shape}"
    )
    assert value_output.shape == value_targets.shape, (
        f"expected value_output of shape {value_targets.shape}, got {value_output.shape}"
    )
    policy_loss = cross_entropy_policy_loss(
        policy_output,
        policy_targets,
    )
    value_loss = mse_value_loss(
        value_output,
        value_targets,
    )
    return value_loss, policy_loss


def base_policy_value_loss(
    model_output: skynet.SkyNetOutput,
    targets: TrainingTargets,
    value_scale: float = 3.0,
) -> torch.Tensor:
    value_loss, policy_loss = policy_value_losses(model_output, targets)
    return value_loss + value_scale * policy_loss


def compute_model_loss_on_game_data(
    model: skynet.SkyNet,
    game_data: play.GameData,
    loss_function: typing.Callable[[skynet.SkyNetOutput, TrainingTargets], typing.Any],
) -> torch.Tensor:
    (
        spatial_inputs,
        non_spatial_inputs,
        outcome_targets,
        points_targets,
        policy_targets,
        masks,
    ) = game_data_to_training_batch(game_data)
    spatial_tensor = torch.tensor(
        spatial_inputs, dtype=torch.float32, device=model.device
    )
    non_spatial_tensor = torch.tensor(
        non_spatial_inputs, dtype=torch.float32, device=model.device
    )
    mask_tensor = torch.tensor(masks, dtype=torch.float32, device=model.device)
    policy_targets_tensor = torch.tensor(
        policy_targets, dtype=torch.float32, device=model.device
    )
    outcome_targets_tensor = torch.tensor(
        outcome_targets, dtype=torch.float32, device=model.device
    )
    points_targets_tensor = torch.tensor(
        points_targets, dtype=torch.float32, device=model.device
    )
    model_output = model(spatial_tensor, non_spatial_tensor, mask_tensor)
    return loss_function(
        model_output,
        (outcome_targets_tensor, points_targets_tensor, policy_targets_tensor),
    )
