"""Construct a learner, replay and runtime state before the online loop starts."""

import dataclasses
from dataclasses import dataclass
from pathlib import Path

from skyjo.learning import buffer, checkpoint, continuation
from skyjo.learning.learner import Learner
from . import experiment_config
from .settings import SelfPlayRunConfig
from .state import TrainingState


@dataclass
class TrainingSession:
    learner: Learner
    replay: buffer.ReplayBuffer
    state: TrainingState
    checkpoint_configuration: dict


def initialize_training_data_buffer(
    config: buffer.Config,
    initial_dataset_path: Path | None = None,
    expected_dataset_id: str | None = None,
) -> buffer.ReplayBuffer:
    from skyjo.learning import replay_io

    if initial_dataset_path is None:
        return buffer.ReplayBuffer.from_config(config)
    replay = replay_io.load_for_config(config, initial_dataset_path)
    if expected_dataset_id is not None and replay.dataset_id != expected_dataset_id:
        raise ValueError(
            "Initial replay identity changed after configuration validation"
        )
    return replay


def prepare_training(
    settings: SelfPlayRunConfig,
    checkpoints_dir: Path,
    *,
    parent: continuation.Continuation | None,
    run_id: str | None,
    requested_settings: dict | None = None,
) -> TrainingSession:
    players, run_seed = settings.players, settings.seed
    training_config = settings.training
    training_data_buffer_config = settings.replay
    initial_training_dataset_path, expected_initial_dataset_id = (
        settings.initial_dataset,
        settings.dataset_id,
    )
    model_settings, model_player_config = (
        dataclasses.asdict(settings.model),
        settings.contestant,
    )
    auxiliary_targets, auxiliary_objectives = (
        dataclasses.asdict(settings.auxiliary_targets),
        settings.auxiliary_objectives.weights,
    )
    learn_rate = settings.optimizer.learn_rate
    value_scale, policy_scale = (
        settings.optimizer.value_scale,
        settings.optimizer.policy_scale,
    )
    process_count = settings.execution.workers
    torch_threads_per_worker = settings.execution.threads_per_worker
    games_per_task = settings.generation.games_per_task
    expected_specs = experiment_config.target_specs(players, auxiliary_objectives)
    if (
        buffer.resolve_target_specs(
            training_data_buffer_config.target_specs,
            spatial_input_shape=training_data_buffer_config.spatial_input_shape,
            action_mask_shape=training_data_buffer_config.action_mask_shape,
        )
        != expected_specs
    ):
        raise ValueError("Replay targets do not match enabled objectives")
    checkpoint_interval = settings.schedule.checkpoint_interval
    if checkpoint_interval < 1:
        raise ValueError("checkpoint interval must be positive")
    if process_count < 1:
        raise ValueError("process_count must be positive")
    if torch_threads_per_worker < 1:
        raise ValueError("torch_threads_per_worker must be positive")
    if games_per_task < 1:
        raise ValueError("games_per_task must be positive")
    if checkpoints_dir.exists() and any(checkpoints_dir.iterdir()):
        raise ValueError("Online training requires a fresh checkpoint directory")
    training_data_buffer = initialize_training_data_buffer(
        training_data_buffer_config,
        initial_training_dataset_path,
        expected_initial_dataset_id,
    )
    learner = Learner.create(
        model_settings,
        players=players,
        device=settings.execution.device,
        auxiliary_objectives=auxiliary_objectives,
        seed=run_seed,
        learn_rate=learn_rate,
        value_scale=value_scale,
        policy_scale=policy_scale,
    )
    model, optimizer = learner.model, learner.optimizer
    active_replay_ratio = training_config.replay_ratio
    replay_ratio_before_fill = training_config.replay_ratio
    # Retain an activated schedule only when its settings and capacity match.
    if parent is not None and training_config.replay_ratio_after_fill is not None:
        previous = parent.payload["configuration"]
        previous_training = previous["training"]
        if (
            previous.get("replay", {}).get("capacity")
            == training_data_buffer_config.max_size
            and previous_training.get("replay_ratio_before_fill")
            == replay_ratio_before_fill
            and previous_training.get("replay_ratio_after_fill")
            == training_config.replay_ratio_after_fill
        ):
            active_replay_ratio = parent.payload["continuation_state"]["replay_ratio"]
    run_configuration = {
        "requested": checkpoint.normalize_configuration(
            settings if requested_settings is None else requested_settings
        ),
        "seed": run_seed,
        "auxiliary_objectives": auxiliary_objectives or {},
        "run_id": run_id,
        "optimizer": {"type": "adam"},
        "players": players,
        "auxiliary_targets": auxiliary_targets or {"mode": "observed", "samples": 32},
        "model": dict(model_settings),
        "player": model_player_config,
        "learn": {
            "games_generated_per_iteration": (settings.generation.games_per_iteration),
            "checkpoint_interval": checkpoint_interval,
        },
        "training": {
            "batch_size": training_config.batch_size,
            "replay_ratio": training_config.replay_ratio,
            "replay_ratio_before_fill": replay_ratio_before_fill,
            "replay_ratio_after_fill": training_config.replay_ratio_after_fill,
            "learn_rate": learn_rate,
            "loss": {
                "value_scale": value_scale,
                "policy_scale": policy_scale,
                "auxiliary_objectives": auxiliary_objectives or {},
            },
        },
        "replay": {"capacity": training_data_buffer_config.max_size},
    }
    if parent is not None:
        if initial_training_dataset_path is None:
            raise ValueError("Continuation requires initial replay")
        parent.restore(model, optimizer, sampling_rng=learner.sampling_rng)
        progress = parent.progress
        checkpoint.restore_rng_state(parent.payload["rng_state"])
    else:
        progress = checkpoint.TrainingProgress()

    next_game = (
        parent.next_game(training_data_buffer)
        if parent
        else max(training_data_buffer.game_indices, default=-1) + 1
    )
    state = TrainingState(
        progress,
        inherited_progress=progress if parent else None,
        inherited_generated_positions=parent.generated_positions if parent else 0,
        next_game_index=next_game,
        replay_ratio=active_replay_ratio,
        diagnostic_done=bool(
            parent and parent.payload["continuation_state"].get("diagnostic_done")
        ),
    )
    return TrainingSession(learner, training_data_buffer, state, run_configuration)
