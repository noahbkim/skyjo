"""Execution state and explicit results shared by experiment recipes."""

from __future__ import annotations

import dataclasses
import pathlib

from skyjo.learning import checkpoint


@dataclasses.dataclass(frozen=True)
class Snapshot:
    path: pathlib.Path
    artifact_id: str | None


@dataclasses.dataclass(frozen=True)
class TrainingState:
    progress: checkpoint.TrainingProgress
    generated_positions: int = 0
    snapshot: Snapshot | None = None
    inherited_progress: checkpoint.TrainingProgress | None = None
    inherited_generated_positions: int | None = 0
    next_game_index: int = 0
    replay_ratio: float = 4.0
    diagnostic_done: bool = False

    def point(self) -> dict:
        point = {
            **dataclasses.asdict(self.progress),
            "generated_positions": self.generated_positions
            + self.inherited_generated_positions
            if self.inherited_generated_positions is not None
            else None,
        }
        if self.inherited_progress is not None:
            point.update(
                {
                    (
                        "additional_iterations"
                        if key == "iteration"
                        else f"additional_{key}"
                    ): getattr(self.progress, key)
                    - getattr(self.inherited_progress, key)
                    for key in (
                        "iteration",
                        "generated_games",
                        "optimizer_steps",
                        "sampled_positions",
                        "trained_positions",
                    )
                }
            )
            point["additional_generated_positions"] = self.generated_positions
        return point

    def continuation_state(self) -> dict:
        return {
            "next_game_index": self.next_game_index,
            "replay_ratio": self.replay_ratio,
            "diagnostic_done": self.diagnostic_done,
            "generated_positions": self.point()["generated_positions"],
        }

    def generated(self, *, games: int, positions: int) -> TrainingState:
        return dataclasses.replace(
            self,
            progress=dataclasses.replace(
                self.progress, generated_games=self.progress.generated_games + games
            ),
            generated_positions=self.generated_positions + positions,
            next_game_index=self.next_game_index + games,
        )

    def trained(self, *, iteration: int, steps: int, batch_size: int) -> TrainingState:
        return dataclasses.replace(
            self,
            progress=dataclasses.replace(
                self.progress,
                iteration=iteration,
                epoch=0,
                optimizer_steps=self.progress.optimizer_steps + steps,
                sampled_positions=self.progress.sampled_positions + steps * batch_size,
                trained_positions=self.progress.trained_positions + steps * batch_size,
            ),
            snapshot=None,
        )


@dataclasses.dataclass(frozen=True)
class TrainingRunResult:
    run_id: str | None
    path: pathlib.Path
    checkpoint: Snapshot
    progress: dict
    timings: dict[str, float]

    def to_dict(self):
        return {
            "run_id": self.run_id,
            "path": str(self.path),
            "checkpoint": {
                "absolute_path": str(self.checkpoint.path),
                "artifact_id": self.checkpoint.artifact_id,
            },
            "progress": self.progress,
            "timings": self.timings,
        }
