"""Progress and evidence recording for continuous distributed training."""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import pathlib
import time
import typing

import numpy as np

from . import checkpoint, explain, game_stats, runs, train, train_utils


@dataclasses.dataclass(frozen=True)
class ObservationConfig:
    progress_interval_seconds: float = 0.0
    concept_interval: int = 5

    def __post_init__(self):
        if (
            not math.isfinite(self.progress_interval_seconds)
            or self.progress_interval_seconds < 0
        ):
            raise ValueError("progress_interval_seconds must be finite and nonnegative")
        if self.concept_interval < 0:
            raise ValueError("concept_interval cannot be negative")

    def concepts_due(self, iteration: int, final_iteration: int) -> bool:
        return self.concept_interval > 0 and (
            iteration in (0, final_iteration) or iteration % self.concept_interval == 0
        )


class GenerationProgress:
    """One completion summary, with optional timed updates when interval > 0."""

    def __init__(self, total_games: int, interval: float):
        self.total_games = total_games
        self.interval = interval
        self.started = time.perf_counter()
        self.next_report = self.started + interval
        self.games = 0
        self.decisions = 0

    def wait_seconds(self) -> float | None:
        if self.interval == 0:
            return None
        return max(0, self.next_report - time.perf_counter())

    def report(self, *, final: bool = False) -> None:
        now = time.perf_counter()
        if not final and (self.interval == 0 or now < self.next_report):
            return
        elapsed = max(now - self.started, 1e-9)
        rate = self.games / elapsed
        eta = f"{(self.total_games - self.games) / rate:.1f}s" if rate else "unknown"
        logging.info(
            "[SELF-PLAY] %s/%s games in %.1fs | %.3f games/s | %.1f decisions/s | ETA %s%s",
            self.games,
            self.total_games,
            elapsed,
            rate,
            self.decisions / elapsed,
            eta,
            " | complete" if final else "",
        )
        self.next_report = now + self.interval


@dataclasses.dataclass(frozen=True)
class PreparedGames:
    games: list[game_stats.GameStats]
    provenance: list[tuple[int, int]]
    replay_before: tuple[int, int]
    seconds: float

    @property
    def positions(self) -> int:
        return sum(game.game_length for game in self.games)


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
        loss_stats_function: typing.Callable[[list[dict]], object] | None = None,
    ):
        self.recorder = recorder
        self.loss_stats_function = loss_stats_function
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

    def training(
        self,
        state: TrainingState,
        result: train.TrainingResult,
        *,
        new_positions: int,
        replay_positions: int,
    ) -> None:
        if result.gradient_scales:
            self.event(
                "gradient_scales",
                state,
                metrics=result.gradient_scales,
                context={
                    "batch": "first_actual_training_batch",
                    "scope": "shared_network",
                },
            )
        losses = result.losses
        keys = sorted({key for detail in losses for key in detail})
        mean_losses = {
            key: float(np.mean([d[key] for d in losses if key in d])) for key in keys
        }
        ratio = result.sampled_positions / new_positions
        passes = result.sampled_positions / replay_positions
        rate = result.sampled_positions / result.seconds
        self.event(
            "training",
            state,
            metrics={
                **result.diagnostics,
                "training/optimizer_steps": result.steps,
                "training/sampled_positions": result.sampled_positions,
                "training/replay_ratio": ratio,
                "training/replay_equivalent_passes": passes,
                "training/positions_per_second": rate,
                "training/steps_per_second": result.steps / result.seconds,
                **{f"loss/{key}": value for key, value in mean_losses.items()},
            },
            context={
                "aggregation": "mean_over_optimizer_steps",
                "sample_counts": {key: sum(key in d for d in losses) for key in keys},
                "replay_artifact_id": self.replay_artifact,
            },
        )
        logging.info(
            "[TRAIN] %s new positions | %s replay positions | %s updates | %s sampled "
            "| replay ratio %.2f | %.2f replay-equivalent passes | %.0f positions/s "
            "| loss %.4f",
            new_positions,
            replay_positions,
            result.steps,
            result.sampled_positions,
            ratio,
            passes,
            rate,
            mean_losses["total_loss"],
        )
        logging.debug(
            "[TRAIN] Mean losses: %s",
            {key: round(value, 5) for key, value in mean_losses.items()},
        )
        logging.debug("[TRAIN] Policy diagnostics: %s", result.diagnostics)
        if self.loss_stats_function is not None and logging.getLogger().isEnabledFor(
            logging.DEBUG
        ):
            logging.debug(
                "[TRAIN] Detailed losses:\n%s", self.loss_stats_function(losses)
            )

    def iteration(
        self,
        state: TrainingState,
        *,
        prepared: PreparedGames,
        replay,
        timings: dict,
    ) -> None:
        games, before = prepared.games, prepared.replay_before
        new_positions = prepared.positions
        round_metrics = game_stats.summarize_games(games)
        self.event(
            "iteration_completed",
            state,
            metrics={
                "generation/games": len(games),
                "generation/positions": new_positions,
                "generation/games_per_second": len(games) / timings["generation"],
                "generation/decisions_per_second": new_positions
                / timings["generation"],
                **round_metrics,
                "replay/games": replay.game_count,
                "replay/positions": len(replay),
                "replay/evicted_games": max(
                    0, before[0] + len(games) - replay.game_count
                ),
                "replay/evicted_positions": max(
                    0, before[1] + new_positions - len(replay)
                ),
                **{f"time/{key}_seconds": value for key, value in timings.items()},
            },
            context={"game_sample_count": len(games)},
        )
        logging.info("[GAMES] %s", game_stats.format_summary(round_metrics))
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(
                "[GAMES] Detailed summaries:\n%s\n%s",
                game_stats.format_summary(round_metrics, detailed=True),
                train_utils.game_stats_summary(games),
            )
        logging.info(
            "[LEARN] Completed iteration %s in %.1fs",
            state.progress.iteration,
            timings["iteration"],
        )
        logging.debug("[LEARN] Phase timings (seconds): %s", timings)

    def save_rounds(
        self, state: TrainingState, prepared: PreparedGames, generation: Snapshot | None
    ) -> None:
        if self.recorder is None:
            return
        context = {
            "run_id": self.recorder.manifest["run_id"],
            "iteration": state.progress.iteration,
            "checkpoint_artifact_id": generation.artifact_id if generation else None,
            "checkpoint_path": str(generation.path) if generation else None,
        }
        records = [
            {
                **context,
                "game_index": index,
                "play_seed": seed,
                "round_number": number,
                **dataclasses.asdict(stats),
            }
            for (index, seed), game in zip(
                prepared.provenance, prepared.games, strict=True
            )
            for number, stats in enumerate(game.rounds, start=1)
        ]
        self.save_records(
            state,
            f"rounds-{context['iteration']:06d}",
            records,
            kind="round_statistics",
        )

    def save_records(
        self, state: TrainingState, name: str, records: list[dict], *, kind: str
    ) -> None:
        if self.recorder is None:
            return
        path = self.recorder.path / "metrics" / f"{name}.jsonl"
        path.parent.mkdir(exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        self.recorder.register_artifact(
            path,
            kind=kind,
            progress=state.point(),
            metadata={"record_count": len(records)},
        )

    def concepts(self, state: TrainingState, report: explain.ConceptReport) -> None:
        self.save_records(
            state,
            f"concepts-{state.progress.iteration:06d}",
            report.examples,
            kind="heuristic_concept_checks",
        )
        summary = report.summary()
        self.event(
            "concept_checks",
            state,
            metrics=summary,
            context={
                "interpretation": "heuristic concepts, not calibrated playing strength"
            },
        )
        logging.info(
            "[CONCEPTS] Heuristic checks: %s/%s target actions | mean target probability %.3f",
            summary["target_action_matches"],
            summary["example_count"],
            summary["mean_target_probability"],
        )
        logging.debug("[CONCEPTS] Examples: %s", report.examples)
