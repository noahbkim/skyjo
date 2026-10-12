"""Immutable recipe settings composed once from validated external configuration."""

from dataclasses import dataclass
from pathlib import Path

from skyjo.learning import buffer, objectives, train

from .contestants import ContestantConfig
from .experiment_training import ObservationConfig


@dataclass(frozen=True)
class ExecutionConfig:
    device: str
    workers: int
    threads_per_worker: int
    debug: bool = False


@dataclass(frozen=True)
class GenerationConfig:
    games_per_iteration: int
    games_per_task: int
    start_state: str


@dataclass(frozen=True)
class BudgetConfig:
    iterations: int
    checkpoint_interval: int
    max_seconds: float = 0.0


@dataclass(frozen=True)
class OptimizerConfig:
    learn_rate: float
    value_scale: float = 1.0
    policy_scale: float = 1.0


@dataclass(frozen=True)
class ModelSettings:
    name: str
    embedding_dimensions: int
    global_state_embedding_dimensions: int
    num_heads: int


@dataclass(frozen=True)
class AuxiliaryTargetSettings:
    mode: str
    samples: int


@dataclass(frozen=True)
class SelfPlayRunConfig:
    seed: int
    players: int
    model: ModelSettings
    optimizer: OptimizerConfig
    training: train.ReplayRatioTrainConfig
    generation: GenerationConfig
    execution: ExecutionConfig
    schedule: BudgetConfig
    contestant: ContestantConfig
    replay: buffer.Config
    observations: ObservationConfig
    auxiliary_objectives: objectives.ResolvedObjectives
    auxiliary_targets: AuxiliaryTargetSettings
    initial_dataset: Path | None = None
    dataset_id: str | None = None

    @classmethod
    def from_resolved(cls, config, *, search=None):
        t, shapes = config["training"], config["derived"]
        e, g, budget = config["execution"], config["selfplay"], config["budget"]
        return cls(
            config["seed"],
            config["players"],
            ModelSettings(**config["model"]),
            OptimizerConfig(t["learn_rate"], t["value_scale"], t["policy_scale"]),
            train.ReplayRatioTrainConfig(
                t["batch_size"],
                t["replay_ratio"],
                gradient_diagnostic=t["gradient_diagnostic"],
                replay_ratio_after_fill=t["replay_ratio_after_fill"],
            ),
            GenerationConfig(**g),
            ExecutionConfig(**e),
            BudgetConfig(**budget),
            ContestantConfig.from_search_settings(search or config["search"]),
            buffer.Config(
                config["replay"]["capacity"],
                tuple(shapes["spatial_input_shape"]),
                tuple(shapes["non_spatial_input_shape"]),
                tuple(shapes["action_mask_shape"]),
                tuple(
                    buffer.TargetShapeSpec(s["name"], tuple(s["shape"]))
                    for s in shapes["target_specs"]
                ),
            ),
            ObservationConfig(**config["logging"], **config["validation"]),
            objectives.resolve(config["auxiliary_objectives"]),
            AuxiliaryTargetSettings(**config["auxiliary_targets"]),
            Path(config["replay"]["initial_dataset"])
            if config["replay"]["initial_dataset"]
            else None,
            config["replay"]["dataset_id"],
        )


@dataclass(frozen=True)
class OfflineTrainingConfig:
    batch_size: int
    learn_rate: float
    value_scale: float
    policy_scale: float
    gradient_diagnostic: bool


@dataclass(frozen=True)
class OfflineRunConfig:
    players: int
    model: ModelSettings
    training: OfflineTrainingConfig
    execution: ExecutionConfig
    auxiliary_objectives: objectives.ResolvedObjectives
