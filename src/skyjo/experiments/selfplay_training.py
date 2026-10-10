"""Recorded self-play training with deterministic worker game streams."""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import logging
import math
import pathlib
import time
import typing

import numpy as np
import torch
import torch.multiprocessing as mp

from skyjo.learning import boundary_inference
from skyjo.learning import buffer
from skyjo.learning import checkpoint
from skyjo.learning import continuation
from skyjo.experiments import experiment_config
from skyjo.experiments import experiment_training
from skyjo.analytics import explain
from skyjo.learning import objectives, targets
from skyjo.analytics import game_stats
from skyjo.experiments.state import TrainingRunResult
from skyjo.experiments.artifacts import RunArtifacts
from skyjo.experiments.settings import SelfPlayRunConfig
from skyjo.experiments.training_setup import prepare_training
from skyjo.simulation.jobs import GeneratedGame, derive_game_seed
from skyjo.experiments import runs
from skyjo.learning import train
from skyjo.experiments import training_budget
from skyjo.engine import game as sj

from skyjo.experiments.generation import (
    configure_torch_worker,
    generate_iteration,
)


def add_generated_games_to_buffer(
    generated_games: typing.Iterable[GeneratedGame],
    training_data_buffer: buffer.ReplayBuffer,
    *,
    log_progress: bool = False,
    auxiliary_objectives=None,
    auxiliary_targets=None,
    run_seed=0,
) -> list[game_stats.GameStats]:
    """Convert observed full-game results in global game order."""
    ordered_games = sorted(
        generated_games,
        key=lambda game: game.global_game_index,
    )
    progress_interval = max(1, math.ceil(len(ordered_games) / 10))
    started_at = time.perf_counter()
    converted_positions = 0
    game_stats_list = []
    for completed_games, generated_game in enumerate(ordered_games, start=1):
        options = {}
        target_seed = -1
        if objectives.resolve(auxiliary_objectives).weights:
            target_seed = derive_game_seed(
                run_seed,
                generated_game.global_game_index,
                targets.AUXILIARY_SEED_STREAM,
            )
            options = dict(
                auxiliary_objectives=auxiliary_objectives,
                **(auxiliary_targets or {}),
                seed=target_seed,
                game_index=generated_game.global_game_index,
            )
        batch = targets.build_training_batch(generated_game.result, **options)
        training_data_buffer.append(
            batch,
            buffer.GameProvenance(
                generated_game.global_game_index,
                generated_game.play_seed,
                target_seed,
            ),
        )
        game_stats_list.append(game_stats.analyze_game(generated_game.result))
        converted_positions += len(batch.spatial_inputs)
        if log_progress and (
            completed_games % progress_interval == 0
            or completed_games == len(ordered_games)
        ):
            elapsed_seconds = time.perf_counter() - started_at
            logging.debug(
                "[TARGETS] Converted %s/%s games and %s positions in %.1fs "
                "(%.1f positions/s)",
                completed_games,
                len(ordered_games),
                converted_positions,
                elapsed_seconds,
                converted_positions / elapsed_seconds,
            )
    return game_stats_list


def prepare_iteration(
    generated: list[GeneratedGame], replay: buffer.ReplayBuffer, **target_settings
) -> experiment_training.PreparedGames:
    started = time.perf_counter()
    before = (replay.game_count, len(replay))
    ordered = sorted(generated, key=lambda game: game.global_game_index)
    games = add_generated_games_to_buffer(
        ordered, replay, log_progress=True, **target_settings
    )
    return experiment_training.PreparedGames(
        games,
        [(g.global_game_index, g.play_seed) for g in ordered],
        before,
        time.perf_counter() - started,
    )


def train_self_play(
    settings: SelfPlayRunConfig,
    *,
    checkpoints_dir: pathlib.Path,
    recorder: runs.RunRecorder | None = None,
    parent: continuation.Continuation | None = None,
    budget: training_budget.TrainingBudget | None = None,
    requested_settings: dict | None = None,
) -> TrainingRunResult:
    training_config = settings.training
    players, run_seed = settings.players, settings.seed
    model_settings, model_player_config = (
        dataclasses.asdict(settings.model),
        settings.contestant,
    )
    process_count = settings.execution.workers
    torch_threads_per_worker = settings.execution.threads_per_worker
    games_per_task = settings.generation.games_per_task
    initial_training_dataset_path = settings.initial_dataset
    observations = settings.observations
    auxiliary_targets, auxiliary_objectives = (
        dataclasses.asdict(settings.auxiliary_targets),
        settings.auxiliary_objectives.weights,
    )
    start_state_generator = (
        create_random_potential_clear_position
        if settings.generation.start_state == "potential_clear"
        else None
    )
    budget = budget or training_budget.TrainingBudget(
        settings.schedule.iterations, settings.schedule.max_seconds
    )
    observations = observations or experiment_training.ObservationConfig()
    if players != 2 and observations.concept_interval:
        logging.info(
            "[CONCEPTS] Skipped: handcrafted concepts require a two-player model"
        )
        observations = dataclasses.replace(observations, concept_interval=0)
    session = prepare_training(
        settings,
        checkpoints_dir,
        parent=parent,
        run_id=recorder.manifest["run_id"] if recorder else None,
        requested_settings=requested_settings,
    )
    learner, training_data_buffer, state = (
        session.learner,
        session.replay,
        session.state,
    )
    run_configuration = session.checkpoint_configuration
    model = learner.model
    initial_iteration = state.progress.iteration
    recording = experiment_training.RecipeRecording(recorder)
    artifacts = RunArtifacts(
        recorder,
        checkpoints_dir.parent / "data" / "replay",
        initial_dataset=initial_training_dataset_path,
        dataset_id=training_data_buffer.dataset_id,
    )
    initial = artifacts.save_checkpoint(
        checkpoints_dir / "checkpoint_000000.pth",
        learner,
        run_configuration,
        state,
        role="initial",
    )
    state = dataclasses.replace(state, snapshot=initial)
    if parent is not None:
        recording.event(
            "continuation_started",
            state,
            context={
                **parent.provenance,
                "source_replay_path": str(initial_training_dataset_path),
                "source_replay_dataset_id": training_data_buffer.dataset_id,
                "next_game_index": state.next_game_index,
            },
        )
    if initial_training_dataset_path is not None:
        artifacts.save_replay(
            training_data_buffer,
            state=state,
            generation=None,
            generation_iteration=None,
            game_count=0,
            generation_settings={},
        )

    def evaluate_concepts():
        rng = checkpoint.capture_rng_state()
        try:
            return explain.evaluate_concepts(model)
        finally:
            checkpoint.restore_rng_state(rng)

    last_concepts = None
    if observations.concept_interval:
        recording.concepts(state, evaluate_concepts())
        last_concepts = initial_iteration
    worker_settings = {
        "model_settings": model_settings,
        "auxiliary_objectives": auxiliary_objectives,
        "model_player_config": model_player_config,
        "players": players,
        "run_seed": run_seed,
        "start_state_generator": start_state_generator,
    }
    generation_settings = {
        "run_seed": run_seed,
        "players": players,
        "model": run_configuration["model"],
        "model_player": dataclasses.asdict(model_player_config),
        "auxiliary_targets": run_configuration["auxiliary_targets"],
    }
    phase_totals = {}
    final_saved = False
    with contextlib.ExitStack() as resources:
        if budget.stop_reason(0) is None:
            pool = resources.enter_context(
                mp.Pool(
                    processes=process_count,
                    initializer=configure_torch_worker,
                    initargs=(torch_threads_per_worker,),
                )
            )
        # Pool/model construction must not consume a continuation's learner RNG.
        if parent is not None:
            checkpoint.restore_rng_state(parent.payload["rng_state"])
        while budget.stop_reason(state.progress.iteration - initial_iteration) is None:
            iteration = state.progress.iteration + 1
            iteration_started = time.perf_counter()
            timings = {}
            logging.info(
                "[LEARN] Starting iteration %s | additional %s/%s | elapsed %.1fs / %ss",
                iteration,
                iteration - initial_iteration,
                settings.schedule.iterations or "unlimited",
                budget.elapsed(),
                budget.max_seconds or "unlimited",
            )

            generation = state.snapshot
            started = time.perf_counter()
            generated = generate_iteration(
                pool,
                total_games=settings.generation.games_per_iteration,
                games_per_task=games_per_task,
                first_game_index=state.next_game_index,
                worker_kwargs={
                    **worker_settings,
                    "model_state_dict": {
                        name: value.detach().cpu()
                        for name, value in model.state_dict().items()
                    },
                },
                progress_interval_seconds=observations.progress_interval_seconds,
            )
            timings["generation"] = time.perf_counter() - started

            prepared = prepare_iteration(
                generated,
                training_data_buffer,
                auxiliary_objectives=auxiliary_objectives,
                auxiliary_targets=auxiliary_targets,
                run_seed=run_seed,
            )
            timings["target"] = prepared.seconds
            state = state.generated(
                games=len(prepared.games), positions=prepared.positions
            )
            started = time.perf_counter()
            artifacts.save_replay(
                training_data_buffer,
                state=state,
                generation=generation,
                generation_iteration=iteration,
                game_count=len(prepared.games),
                generation_settings=generation_settings,
            )
            timings["replay_save"] = time.perf_counter() - started

            trained = train.train_iteration(
                learner,
                training_data_buffer,
                training_config,
                prepared.positions,
                replay_ratio=state.replay_ratio,
                gradient_diagnostic=training_config.gradient_diagnostic
                and not state.diagnostic_done,
            )
            timings["training"] = trained.seconds
            timings["gradient_diagnostic"] = (trained.gradient_scales or {}).get(
                "seconds", 0.0
            )
            state = state.trained(
                iteration=iteration,
                steps=trained.steps,
                batch_size=training_config.batch_size,
            )

            if trained.gradient_scales:
                state = dataclasses.replace(state, diagnostic_done=True)

            after_fill = training_config.replay_ratio_after_fill
            evicted_positions = (
                prepared.replay_before[1]
                + prepared.positions
                - len(training_data_buffer)
            )
            if (
                after_fill is not None
                and state.replay_ratio != after_fill
                and (
                    evicted_positions > 0
                    or len(training_data_buffer) == training_data_buffer.max_size
                )
            ):
                before_fill = state.replay_ratio
                # The filling iteration used the old ratio; save the next one.
                state = dataclasses.replace(state, replay_ratio=after_fill)
                recording.event(
                    "replay_ratio_changed",
                    state,
                    metrics={
                        "training/previous_replay_ratio": before_fill,
                        "training/replay_ratio": after_fill,
                        "replay/positions": len(training_data_buffer),
                        "replay/evicted_positions": evicted_positions,
                    },
                    context={
                        "reason": "replay_filled",
                        "applies_from_iteration": iteration + 1,
                    },
                )
                logging.info(
                    "[TRAIN] Replay filled; replay ratio %.2f -> %.2f from iteration %s",
                    before_fill,
                    after_fill,
                    iteration + 1,
                )

            concepts = None
            started = time.perf_counter()
            is_final = budget.stop_reason(iteration - initial_iteration) is not None
            if observations.concepts_due(iteration, iteration if is_final else -1):
                concepts = evaluate_concepts()
                last_concepts = iteration
            timings["validation"] = time.perf_counter() - started
            started = time.perf_counter()
            is_final = budget.stop_reason(iteration - initial_iteration) is not None
            if is_final or iteration % settings.schedule.checkpoint_interval == 0:
                saved = artifacts.save_checkpoint(
                    checkpoints_dir / f"checkpoint_{iteration:06d}.pth",
                    learner,
                    run_configuration,
                    state,
                    role="final" if is_final else "periodic",
                )
                state = dataclasses.replace(state, snapshot=saved)
                final_saved = is_final
            timings["checkpoint_save"] = time.perf_counter() - started

            started = time.perf_counter()
            recording.save_rounds(state, prepared, generation)
            timings["round_save"] = time.perf_counter() - started

            recording.training(
                state,
                trained,
                new_positions=prepared.positions,
                replay_positions=len(training_data_buffer),
                replay_artifact=artifacts.replay_artifact,
            )
            if concepts is not None:
                recording.concepts(state, concepts)
            timings["iteration"] = time.perf_counter() - iteration_started
            for phase, seconds in timings.items():
                phase_totals[f"time/{phase}_seconds"] = (
                    phase_totals.get(f"time/{phase}_seconds", 0) + seconds
                )
            recording.iteration(
                state,
                prepared=prepared,
                replay=training_data_buffer,
                timings=timings,
                budget_metrics=budget.metrics(),
            )

    if observations.concept_interval and last_concepts != state.progress.iteration:
        concepts = evaluate_concepts()
    else:
        concepts = None
    if not final_saved:
        saved = artifacts.save_checkpoint(
            checkpoints_dir / f"checkpoint_{state.progress.iteration:06d}_final.pth",
            learner,
            run_configuration,
            state,
            role="final",
        )
        state = dataclasses.replace(state, snapshot=saved)
    if concepts is not None:
        recording.concepts(state, concepts)
    metrics = budget.metrics()
    reason = budget.stop_reason(state.progress.iteration - initial_iteration)
    recording.event(
        "budget_completed", state, metrics=metrics, context={"stop_reason": reason}
    )
    logging.info(
        "[LEARN] Stopped: %s | %s additional iterations | elapsed %.1fs | overshoot %.1fs",
        reason,
        state.progress.iteration - initial_iteration,
        metrics["time/run_seconds"],
        metrics["time/budget_overshoot_seconds"],
    )
    phase_totals.update(
        {key: value for key, value in metrics.items() if key.startswith("time/")}
    )
    assert state.snapshot is not None
    return TrainingRunResult(
        recorder.manifest["run_id"] if recorder else None,
        recorder.path if recorder else checkpoints_dir.parent,
        state.snapshot,
        state.point(),
        phase_totals,
    )


def create_random_potential_clear_position(rng: np.random.Generator) -> sj.Skyjo:
    return explain.create_potential_clear_equal_position(
        int(rng.integers(0, sj.CARD_SIZE))
    )


def launch(
    config: pathlib.Path,
    runs_dir: pathlib.Path = pathlib.Path(".runs"),
    *,
    repository: pathlib.Path,
    allow_dirty: bool = False,
) -> TrainingRunResult:
    """Start one independent fresh or checkpoint-initialized recorded run."""
    invocation_started = time.perf_counter()
    config = config.resolve()
    supplied, configuration_sources = experiment_config.configuration_sources(config)
    input_bytes = configuration_sources[-1]["content"].encode()
    resolved = experiment_config.resolve_configuration(
        supplied, base_directory=config.parent
    )
    requested_settings = checkpoint.normalize_configuration(
        SelfPlayRunConfig.from_resolved(resolved)
    )
    boundary_source = resolved["search"]["boundary_value_checkpoint"]
    boundary_bytes = None
    if boundary_source:
        boundary_path = pathlib.Path(boundary_source)
        boundary_bytes = boundary_path.read_bytes()
        boundary_inference.load_boundary_model(boundary_source, resolved["players"])
        if boundary_path.read_bytes() != boundary_bytes:
            raise ValueError("Boundary value checkpoint changed while loading")
    budget = training_budget.TrainingBudget(
        resolved["budget"]["iterations"],
        resolved["budget"]["max_seconds"],
        invocation_started,
    )
    parent = (
        continuation.load(
            pathlib.Path(resolved["initial_checkpoint"]),
            resolved,
            requested_settings=requested_settings,
        )
        if resolved["initial_checkpoint"]
        else None
    )
    recorder = runs.RunRecorder.create(
        root=runs_dir,
        repository=repository,
        input_path=config,
        input_bytes=input_bytes,
        configuration=resolved,
        entrypoint="distributed_main.py:launch",
        invocation=[
            "--config",
            str(config),
            "--runs-dir",
            str(runs_dir),
            *(["--allow-dirty"] if allow_dirty else []),
        ],
        allow_dirty=allow_dirty,
    )
    logging.info("Run directory: %s", recorder.path)
    handler = logging.FileHandler(
        recorder.path / "logs" / "train.log", encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger = logging.getLogger()
    previous_level = logger.level
    logger.setLevel(logging.DEBUG if resolved["execution"]["debug"] else logging.INFO)
    logger.addHandler(handler)
    try:
        with recorder:
            runtime_search = dict(resolved["search"])
            if boundary_bytes is not None:
                boundary_snapshot = recorder.path / "data" / "boundary_value.pth"
                boundary_snapshot.write_bytes(boundary_bytes)
                recorder.register_artifact(
                    boundary_snapshot,
                    kind="boundary_value_checkpoint",
                    progress={},
                    metadata={
                        "source_path": boundary_source,
                        "source_sha256": hashlib.sha256(boundary_bytes).hexdigest(),
                        "frozen": True,
                    },
                )
                runtime_search["boundary_value_checkpoint"] = str(boundary_snapshot)
            sources_path = recorder.path / "input-sources.json"
            sources_path.write_text(json.dumps(configuration_sources, indent=2) + "\n")
            recorder.register_artifact(
                sources_path, kind="configuration_sources", progress={}
            )
            if resolved["experiment"]["suite_run_id"]:
                recorder.record_event(
                    "experiment_membership",
                    context={**resolved["experiment"], "seed": resolved["seed"]},
                )
            settings = SelfPlayRunConfig.from_resolved(resolved, search=runtime_search)
            torch.set_num_threads(settings.execution.threads_per_worker)
            result = train_self_play(
                settings,
                checkpoints_dir=recorder.path / "checkpoints",
                parent=parent,
                budget=budget,
                recorder=recorder,
                requested_settings=requested_settings,
            )
            result_path = recorder.path / "result.json"
            result_path.write_text(json.dumps(result.to_dict(), indent=2) + "\n")
            recorder.register_artifact(
                result_path, kind="training_result", progress=result.progress
            )
    finally:
        logger.removeHandler(handler)
        handler.close()
        logger.setLevel(previous_level)
    return result
