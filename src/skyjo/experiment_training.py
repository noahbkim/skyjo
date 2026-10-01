"""Progress and evidence recording for continuous distributed training."""

from __future__ import annotations

import dataclasses
import pathlib
import time
from collections.abc import Callable

import numpy as np
import torch

from . import checkpoint, runs


@dataclasses.dataclass(frozen=True)
class Snapshot:
    path: pathlib.Path
    artifact_id: str | None


@dataclasses.dataclass(frozen=True)
class TrainingState:
    progress: checkpoint.TrainingProgress
    generated_positions: int = 0
    snapshot: Snapshot | None = None

    def point(self) -> dict:
        return {
            **dataclasses.asdict(self.progress),
            "generated_positions": self.generated_positions,
        }

    def generated(self, *, games: int, positions: int) -> TrainingState:
        return dataclasses.replace(
            self,
            progress=dataclasses.replace(
                self.progress, generated_games=self.progress.generated_games + games
            ),
            generated_positions=self.generated_positions + positions,
        )

    def trained(self, *, iteration: int, steps: int, batch_size: int) -> TrainingState:
        return TrainingState(
            progress=dataclasses.replace(
                self.progress,
                iteration=iteration,
                epoch=0,
                optimizer_steps=self.progress.optimizer_steps + steps,
                sampled_positions=self.progress.sampled_positions + steps * batch_size,
                trained_positions=self.progress.trained_positions + steps * batch_size,
            ),
            generated_positions=self.generated_positions,
        )


class RecipeRecording:
    """Own recipe-specific aggregation and replay lineage, not training decisions."""

    def __init__(
        self,
        recorder: runs.RunRecorder | None,
        replay,
        *,
        initial_dataset_path: pathlib.Path | None = None,
    ):
        self.recorder = recorder
        self.replay_artifact = None
        self.previous_dataset_id = replay.dataset_id
        self.initial_buffer = (
            None
            if initial_dataset_path is None
            else {
                "path": str(initial_dataset_path.resolve()),
                "dataset_id": replay.dataset_id,
            }
        )

    def event(self, kind: str, state: TrainingState, *, metrics=None, context=None):
        if self.recorder is not None:
            self.recorder.record_event(
                kind,
                progress=state.point(),
                metrics=metrics or {},
                context={
                    "checkpoint_artifact_id": state.snapshot.artifact_id
                    if state.snapshot
                    else None,
                    **(context or {}),
                },
            )

    def save_snapshot(
        self,
        factory,
        model,
        optimizer,
        configuration,
        state: TrainingState,
        *,
        role: str,
    ) -> Snapshot:
        started = time.perf_counter()
        path = factory.save_model(
            model,
            optimizer=optimizer,
            configuration=configuration,
            progress=state.progress,
        )
        seconds = time.perf_counter() - started
        snapshot = self.register_snapshot(path, state, role=role)
        self.event(
            "checkpoint_saved",
            dataclasses.replace(state, snapshot=snapshot),
            metrics={"time/checkpoint_save_seconds": seconds},
            context={"artifact_id": snapshot.artifact_id, "role": role},
        )
        return snapshot

    def register_snapshot(
        self, path: pathlib.Path, state: TrainingState, *, role: str
    ) -> Snapshot:
        artifact_id = None
        if self.recorder is not None:
            artifact_id = self.recorder.register_artifact(
                path,
                kind="checkpoint",
                progress=state.point(),
                metadata={"role": role},
            )
        return Snapshot(path, artifact_id)

    def validation(
        self,
        model,
        validate: Callable,
        state: TrainingState,
        *,
        initial: bool = False,
    ) -> None:
        """Measure a saved model without changing training mode or randomness."""
        started = time.perf_counter()
        rng = checkpoint.capture_rng_state()
        was_training = model.training
        try:
            model.eval()
            with torch.inference_mode():
                metrics = validate(model)
        finally:
            model.train(was_training)
            checkpoint.restore_rng_state(rng)
        self.event(
            "initial_validation" if initial else "validation",
            state,
            metrics=metrics or {},
            context={
                "suite": "built_in_examples",
                "seconds": time.perf_counter() - started,
            },
        )

    def faceoff(
        self,
        state: TrainingState,
        reference: Snapshot,
        result: dict,
        protocol: dict,
    ) -> None:
        self.event(
            "faceoff",
            state,
            metrics={
                "candidate_wins": result["candidate_wins"],
                "champion_wins": result["champion_wins"],
            },
            context={
                **protocol,
                "passed": result["passed"],
                "reference_artifact_id": reference.artifact_id,
            },
        )

    def save_replay(
        self,
        replay,
        *,
        state: TrainingState,
        generation: Snapshot | None,
        generation_iteration: int,
        game_count: int,
        generation_settings: dict,
    ) -> None:
        batch = {
            "run_id": self.recorder.manifest["run_id"] if self.recorder else None,
            "generation_iteration": generation_iteration,
            "game_count": game_count,
            "checkpoint_artifact_id": generation.artifact_id if generation else None,
            "checkpoint_path": str(generation.path) if generation else None,
            "settings": generation_settings,
        }
        metadata = {
            "initial_buffer": self.initial_buffer,
            "previous_dataset_id": self.previous_dataset_id,
            "latest_generated_batch": batch,
        }
        if self.recorder is not None and self.replay_artifact is not None:
            self.recorder.supersede_artifact(
                self.replay_artifact, progress=state.point()
            )
        # A replay snapshot can contain several models' games plus imported data.
        # Its manifest deliberately has no single source_checkpoint.
        path = replay.save(generation_metadata=metadata, source_checkpoint=None)
        self.previous_dataset_id = replay.dataset_id
        if self.recorder is not None:
            self.replay_artifact = self.recorder.register_artifact(
                path,
                kind="replay_data",
                progress=state.point(),
                metadata={
                    "dataset_id": replay.dataset_id,
                    "retention": "latest_only",
                    "manifest": replay.dataset_metadata,
                    **metadata,
                },
            )

    def training(self, state: TrainingState, losses: list[dict]) -> None:
        keys = sorted({key for detail in losses for key in detail})
        self.event(
            "training",
            state,
            metrics={
                f"loss/{key}": float(np.mean([d[key] for d in losses if key in d]))
                for key in keys
            },
            context={
                "aggregation": "mean_over_optimizer_steps",
                "sample_counts": {key: sum(key in d for d in losses) for key in keys},
                "replay_artifact_id": self.replay_artifact,
            },
        )

    def iteration(
        self,
        state: TrainingState,
        *,
        games,
        replay,
        before: tuple[int, int],
        timings: dict,
    ) -> None:
        records = [stats.to_record_dict() for stats in games]
        new_positions = sum(stats.game_length for stats in games)
        self.event(
            "iteration_completed",
            state,
            metrics={
                "generation/games": len(games),
                "generation/positions": new_positions,
                "replay/games": replay.game_count,
                "replay/positions": len(replay),
                "replay/evicted_games": max(
                    0, before[0] + len(games) - replay.game_count
                ),
                "replay/evicted_positions": max(
                    0, before[1] + new_positions - len(replay)
                ),
                **{f"time/{key}_seconds": value for key, value in timings.items()},
                **{
                    f"game_mean/{key}": float(np.mean([r[key] for r in records]))
                    for key in records[0]
                },
            },
            context={"game_sample_count": len(records)},
        )
