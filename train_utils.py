import typing

import numpy as np
import pandas as pd
import torch

import play
import skyjo as sj
import skynet

# MARK: Types

TrainingDataPoint: typing.TypeAlias = tuple[
    np.ndarray[tuple[int], np.float32],  # spatial input
    np.ndarray[tuple[int], np.float32],  # non-spatial input
    np.ndarray[tuple[int], np.float32],  # action mask
    np.ndarray[tuple[int], np.float32],  # outcome target
    np.ndarray[tuple[int], np.float32],  # points target
    np.ndarray[tuple[int], np.float32],  # policy target
    np.ndarray[tuple[int], np.float32],  # cleared columns target
]
TrainingBatch: typing.TypeAlias = tuple[
    np.ndarray[tuple[int, int, int, int, int], np.float32],  # spatial input
    np.ndarray[tuple[int, int], np.float32],  # non-spatial input
    np.ndarray[tuple[int, int], np.float32],  # action mask
    np.ndarray[tuple[int, int], np.float32],  # outcome target
    np.ndarray[tuple[int, int], np.float32],  # points target
    np.ndarray[tuple[int, int], np.float32],  # policy target
    np.ndarray[tuple[int, int], np.float32],  # cleared columns target
]
TrainingTargets: typing.TypeAlias = tuple[
    torch.Tensor,  # outcome target
    torch.Tensor,  # points target
    torch.Tensor,  # policy target
    torch.Tensor,  # cleared columns target
]


# MARK: Data Helpers

LossDetails: typing.TypeAlias = dict[
    str, float
]  # loss component name: loss component value


def game_stats_summary(game_stats_list: list[play.GameStats]) -> pd.DataFrame:
    stats_df = pd.DataFrame.from_records(
        [game_stats.to_record_dict() for game_stats in game_stats_list]
    )
    return stats_df.describe().T


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
    outcome_targets = [state_tuple[2] for state_tuple in game_data]
    points_targets = [state_tuple[3] for state_tuple in game_data]
    policy_targets = [state_tuple[4] for state_tuple in game_data]
    cleared_columns_targets = [state_tuple[5] for state_tuple in game_data]
    return (
        np.array(spatial_states, dtype=np.float32),
        np.array(non_spatial_states, dtype=np.float32),
        np.array(action_masks, dtype=np.float32),
        np.array(outcome_targets, dtype=np.float32),
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
    model_output: skynet.SkyNetOutput,
    targets: TrainingTargets,
) -> tuple[torch.Tensor, torch.Tensor]:
    value_output, points_output, policy_output, cleared_columns_output = model_output
    value_targets, points_targets, policy_targets, cleared_columns_targets = targets
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


def policy_value_cleared_columns_losses(
    model_output: skynet.SkyNetOutput,
    targets: TrainingTargets,
) -> tuple[torch.Tensor, torch.Tensor]:
    value_output, points_output, policy_output, cleared_columns_output = model_output
    value_targets, points_targets, policy_targets, cleared_columns_targets = targets
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
    # value_loss = l1_value_loss(
    #     value_output,
    #     value_targets,
    # )
    cleared_columns_loss = mse_cleared_columns_loss(
        cleared_columns_output,
        cleared_columns_targets,
    )
    return value_loss, policy_loss, cleared_columns_loss


def base_loss(
    model_output: skynet.SkyNetOutput,
    targets: TrainingTargets,
    value_scale: float = 1.0,
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
            "value_loss": value_loss.item(),
            "policy_loss": policy_loss.item(),
            "cleared_columns_loss": cleared_columns_loss.item(),
        },
    )


def compute_model_loss_on_game_data(
    model: skynet.SkyNet,
    game_data: play.GameData,
    loss_function: typing.Callable[[skynet.SkyNetOutput, TrainingTargets], typing.Any],
) -> tuple[torch.Tensor, LossDetails]:
    (
        spatial_inputs,
        non_spatial_inputs,
        masks,
        outcome_targets,
        points_targets,
        policy_targets,
        cleared_columns_targets,
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
    cleared_columns_targets_tensor = torch.tensor(
        cleared_columns_targets, dtype=torch.float32, device=model.device
    )
    model_output = model(spatial_tensor, non_spatial_tensor, mask_tensor)
    return loss_function(
        model_output,
        (
            outcome_targets_tensor,
            points_targets_tensor,
            policy_targets_tensor,
            cleared_columns_targets_tensor,
        ),
    )


def loss_details_summary(loss_details_list: list[LossDetails]) -> pd.DataFrame:
    return pd.DataFrame.from_records(loss_details_list).describe().T
