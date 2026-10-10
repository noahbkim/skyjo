"""Recorded self-play training with deterministic worker game streams."""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import json
import logging
import math
import pathlib
import queue
import random
import time
import typing

import numpy as np
import torch
import torch.multiprocessing as mp

from . import (
    buffer,
    checkpoint,
    continuation,
    experiment_config,
    experiment_training,
    explain,
    mcts,
    models,
    objectives,
    play,
    player,
    predictor,
    runs,
    train,
    training_budget,
)
from . import game as sj

StartStateGenerator: typing.TypeAlias = typing.Callable[[], sj.Skyjo | None]
PLAY_SEED_STREAM = 0


def configure_torch_worker(torch_thread_count: int) -> None:
    """Limit Torch CPU parallelism inside one multiprocessing worker."""
    if torch_thread_count < 1:
        raise ValueError("torch_thread_count must be positive")
    torch.set_num_threads(torch_thread_count)
    torch.set_num_interop_threads(1)


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedGame:
    global_game_index: int
    play_seed: int
    result: play.GameResult


def derive_game_seed(
    run_seed: int,
    global_game_index: int,
    stream: int,
) -> int:
    """Derive a stable uint32 seed for one game and randomness stream."""
    if global_game_index < 0:
        raise ValueError("global_game_index cannot be negative")
    if stream < 0:
        raise ValueError("stream cannot be negative")
    return int(
        np.random.SeedSequence([run_seed, global_game_index, stream]).generate_state(
            1, dtype=np.uint32
        )[0]
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def play_games_locally(
    model_state_dict: dict[str, torch.Tensor],
    model_settings: dict[str, typing.Any],
    model_player_config: player.ModelPlayerConfig,
    players: int,
    number_of_games: int,
    run_seed: int,
    first_game_index: int,
    start_state_generator: StartStateGenerator | None = None,
    auxiliary_objectives: dict | None = None,
) -> list[GeneratedGame]:
    """Pool worker entrypoint.

    Each task builds a local model from a weight snapshot, runs synchronous
    inference, and returns game histories to the parent process.
    """
    model = models.build(
        model_settings,
        players=players,
        device="cpu",
        auxiliary_objectives=auxiliary_objectives,
    )
    model.load_state_dict(model_state_dict)
    model.eval()

    inference = predictor.LocalPredictor(
        model=model,
        max_batch_size=512,
    )
    model_player = player.ModelPlayer(
        inference,
        **model_player_config.kwargs(),
    )
    model_players = [model_player for _ in range(players)]

    generated_games = []
    for offset in range(number_of_games):
        global_game_index = first_game_index + offset
        play_seed = derive_game_seed(
            run_seed,
            global_game_index,
            PLAY_SEED_STREAM,
        )
        set_seed(play_seed)
        start_state = None if start_state_generator is None else start_state_generator()
        result = play.play_game(model_players, start_state=start_state)
        generated_games.append(
            GeneratedGame(
                global_game_index=global_game_index,
                play_seed=play_seed,
                result=result,
            )
        )
    return generated_games


def game_batch_sizes(total_games: int, games_per_task: int) -> list[int]:
    return [
        min(games_per_task, total_games - start)
        for start in range(0, total_games, games_per_task)
    ]


def add_generated_games_to_buffer(
    generated_games: typing.Iterable[GeneratedGame],
    training_data_buffer: buffer.ReplayBuffer,
    *,
    log_progress: bool = False,
    auxiliary_objectives=None,
    auxiliary_targets=None,
    run_seed=0,
) -> list[play.GameStats]:
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
        if objectives.resolve(auxiliary_objectives).entries:
            options = dict(
                auxiliary_objectives=auxiliary_objectives,
                **(auxiliary_targets or {}),
                seed=run_seed,
                game_index=generated_game.global_game_index,
            )
        game_data, game_stats = play.game_result_to_game_data(
            generated_game.result, **options
        )
        training_data_buffer.add_game_data(
            game_data,
            game_index=generated_game.global_game_index,
            play_seed=generated_game.play_seed,
        )
        game_stats_list.append(game_stats)
        converted_positions += len(game_data)
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


def initialize_training_data_buffer(
    config: buffer.Config,
    initial_dataset_path: pathlib.Path | None = None,
    expected_dataset_id: str | None = None,
) -> buffer.ReplayBuffer:
    """Initialize fresh replay, optionally importing a stable dataset snapshot."""
    destination_manifest = (
        None if config.path is None else config.path / buffer.MANIFEST_FILE
    )
    if destination_manifest is not None and destination_manifest.is_file():
        raise ValueError("Online training requires a fresh replay destination")
    if initial_dataset_path is None:
        return buffer.ReplayBuffer.from_config(config)
    if not (initial_dataset_path / buffer.MANIFEST_FILE).is_file():
        raise FileNotFoundError(f"No replay dataset found at {initial_dataset_path}")

    def identity():
        files = [
            initial_dataset_path / buffer.MANIFEST_FILE,
            *sorted(initial_dataset_path.rglob("*.npy")),
        ]
        return [
            (
                str(path.relative_to(initial_dataset_path)),
                path.stat().st_ino,
                path.stat().st_size,
                path.stat().st_mtime_ns,
            )
            for path in files
        ]

    before = identity()
    manifest_before = (initial_dataset_path / buffer.MANIFEST_FILE).read_bytes()
    replay_buffer = buffer.ReplayBuffer.from_config_or_load(
        dataclasses.replace(config, path=initial_dataset_path)
    )
    # When capacity equals retained size, the loader returns read-only mmaps.
    # Detach them before training can overwrite replay rows.
    for name in ("spatial_input_buffer", "non_spatial_input_buffer", "action_masks"):
        array = getattr(replay_buffer, name)
        if not array.flags.writeable:
            setattr(replay_buffer, name, array.copy())
    for name, array in replay_buffer.target_buffers.items():
        if not array.flags.writeable:
            replay_buffer.target_buffers[name] = array.copy()
    if (
        before != identity()
        or manifest_before != (initial_dataset_path / buffer.MANIFEST_FILE).read_bytes()
    ):
        raise ValueError("Initial replay changed while loading; use a stable snapshot")
    if (
        expected_dataset_id is not None
        and replay_buffer.dataset_id != expected_dataset_id
    ):
        raise ValueError(
            "Initial replay identity changed after configuration validation"
        )
    replay_buffer.path = config.path
    return replay_buffer


def generate_iteration(
    pool,
    *,
    total_games: int,
    games_per_task: int,
    first_game_index: int,
    worker_kwargs: dict,
    progress_interval_seconds: float = 0.0,
) -> list[GeneratedGame]:
    """Collect actual completions; deterministic ordering is restored before replay."""
    sizes = game_batch_sizes(total_games, games_per_task)
    starts = np.cumsum([0, *sizes[:-1]]).tolist()
    completed = queue.Queue()
    progress = experiment_training.GenerationProgress(
        total_games, progress_interval_seconds
    )
    for size, offset in zip(sizes, starts, strict=True):
        pool.apply_async(
            play_games_locally,
            kwds={
                **worker_kwargs,
                "number_of_games": size,
                "first_game_index": first_game_index + offset,
            },
            callback=lambda result: completed.put((True, result)),
            error_callback=lambda error: completed.put((False, error)),
        )
    generated = []
    tasks = 0
    while tasks < len(sizes):
        try:
            succeeded, result = completed.get(timeout=progress.wait_seconds())
        except queue.Empty:
            progress.report()
            continue
        if not succeeded:
            raise result
        tasks += 1
        generated.extend(result)
        progress.games += len(result)
        progress.decisions += sum(
            len(r.history) - 1 for g in result for r in g.result.rounds
        )
        logging.debug("[SELF-PLAY] Completed %s/%s tasks", tasks, len(sizes))
        progress.report(final=tasks == len(sizes))
    return sorted(generated, key=lambda game: game.global_game_index)


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
    process_count: int,
    players: int,
    checkpoints_dir: pathlib.Path,
    learn_config: train.LearnConfig,
    training_config: train.ReplayRatioTrainConfig,
    training_data_buffer_config: buffer.Config,
    model_player_config: player.ModelPlayerConfig,
    model_settings: dict[str, typing.Any],
    torch_threads_per_worker: int = 1,
    run_seed: int = 0,
    games_per_task: int = 1,
    start_state_generator: StartStateGenerator | None = None,
    initial_training_dataset_path: pathlib.Path | None = None,
    recorder: runs.RunRecorder | None = None,
    observations: experiment_training.ObservationConfig | None = None,
    auxiliary_targets: dict | None = None,
    auxiliary_objectives: dict | None = None,
    parent: continuation.Continuation | None = None,
    budget: training_budget.TrainingBudget | None = None,
    expected_initial_dataset_id: str | None = None,
) -> None:
    budget = budget or training_budget.TrainingBudget(
        learn_config.learn_steps, learn_config.max_seconds
    )
    observations = observations or experiment_training.ObservationConfig()
    if players != 2 and observations.concept_interval:
        logging.info(
            "[CONCEPTS] Skipped: handcrafted concepts require a two-player model"
        )
        observations = dataclasses.replace(observations, concept_interval=0)
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
    checkpoint_interval = learn_config.checkpoint_interval
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
    model = models.build(
        model_settings,
        players=players,
        device=learn_config.torch_device,
        auxiliary_objectives=auxiliary_objectives,
    )
    optimizer = train.make_optimizer(model, training_config.learn_rate)
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
            training_config.replay_ratio = previous_training["replay_ratio"]
    run_configuration = {
        "seed": run_seed,
        "optimizer": {"type": "adam"},
        "players": players,
        "auxiliary_targets": auxiliary_targets or {"mode": "observed", "samples": 32},
        "model": {
            "name": getattr(model, "architecture_name", type(model).__name__),
            "non_spatial_input_shape": model.non_spatial_input_shape,
            **{key: value for key, value in model_settings.items() if key != "name"},
            "auxiliary_objectives": auxiliary_objectives or {},
        },
        "player": model_player_config,
        "learn": {
            "games_generated_per_iteration": (
                learn_config.games_generated_per_iteration
            ),
            "checkpoint_interval": checkpoint_interval,
        },
        "training": {
            "batch_size": training_config.batch_size,
            "replay_ratio": training_config.replay_ratio,
            "replay_ratio_before_fill": replay_ratio_before_fill,
            "replay_ratio_after_fill": training_config.replay_ratio_after_fill,
            "learn_rate": training_config.learn_rate,
            "loss_function": training_config.loss_function,
        },
        "replay": {"capacity": training_data_buffer_config.max_size},
    }
    if parent is not None:
        if initial_training_dataset_path is None:
            raise ValueError("Continuation requires initial replay")
        parent.restore(model, optimizer)
        progress = parent.progress
        checkpoint.restore_rng_state(parent.payload["rng_state"])
    else:
        progress = checkpoint.TrainingProgress()

    next_game = (
        parent.next_game(training_data_buffer)
        if parent
        else max(training_data_buffer.game_indices, default=-1) + 1
    )
    state = experiment_training.TrainingState(
        progress,
        inherited_progress=progress if parent else None,
        inherited_generated_positions=parent.generated_positions if parent else 0,
        next_game_index=next_game,
    )
    initial_iteration = progress.iteration
    recording = experiment_training.RecipeRecording(
        recorder,
        training_data_buffer,
        initial_dataset_path=initial_training_dataset_path,
    )
    initial_path = checkpoint.save_checkpoint(
        checkpoints_dir / "checkpoint_000000.pth",
        model=model,
        optimizer=optimizer,
        configuration=run_configuration,
        progress=progress,
        continuation_state=state.continuation_state(),
    )
    initial = recording.register_snapshot(initial_path, state, role="initial")
    state = dataclasses.replace(state, snapshot=initial)
    if parent is not None:
        recording.event(
            "continuation_started",
            state,
            context={
                **parent.provenance,
                "source_replay_path": str(initial_training_dataset_path),
                "source_replay_dataset_id": training_data_buffer.dataset_id,
                "next_game_index": next_game,
            },
        )
    if initial_training_dataset_path is not None:
        recording.save_replay(
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
                learn_config.learn_steps or "unlimited",
                budget.elapsed(),
                budget.max_seconds or "unlimited",
            )

            generation = state.snapshot
            started = time.perf_counter()
            generated = generate_iteration(
                pool,
                total_games=learn_config.games_generated_per_iteration,
                games_per_task=games_per_task,
                first_game_index=next_game,
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
            next_game += len(prepared.games)
            started = time.perf_counter()
            recording.save_replay(
                training_data_buffer,
                state=state,
                generation=generation,
                generation_iteration=iteration,
                game_count=len(prepared.games),
                generation_settings=generation_settings,
            )
            timings["replay_save"] = time.perf_counter() - started

            trained = train.train_iteration(
                model,
                training_data_buffer,
                optimizer,
                training_config,
                prepared.positions,
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

            after_fill = training_config.replay_ratio_after_fill
            evicted_positions = (
                prepared.replay_before[1]
                + prepared.positions
                - len(training_data_buffer)
            )
            if (
                after_fill is not None
                and training_config.replay_ratio != after_fill
                and (
                    evicted_positions > 0
                    or len(training_data_buffer) == training_data_buffer.max_size
                )
            ):
                before_fill = training_config.replay_ratio
                # The filling iteration used the old ratio; save the next one.
                training_config.replay_ratio = after_fill
                run_configuration["training"]["replay_ratio"] = after_fill
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
            if is_final or iteration % checkpoint_interval == 0:
                saved = recording.save_snapshot(
                    checkpoints_dir / f"checkpoint_{iteration:06d}.pth",
                    model,
                    optimizer,
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
            )
            if concepts is not None:
                recording.concepts(state, concepts)
            timings["iteration"] = time.perf_counter() - iteration_started
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
        saved = recording.save_snapshot(
            checkpoints_dir / f"checkpoint_{state.progress.iteration:06d}_final.pth",
            model,
            optimizer,
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


def create_random_potential_clear_position() -> sj.Skyjo:
    return explain.create_potential_clear_equal_position(
        np.random.randint(0, sj.CARD_SIZE)
    )


def launch(
    config: pathlib.Path,
    runs_dir: pathlib.Path = pathlib.Path(".runs"),
    *,
    repository: pathlib.Path,
    allow_dirty: bool = False,
) -> pathlib.Path:
    """Start one independent fresh or checkpoint-initialized recorded run."""
    invocation_started = time.perf_counter()
    config = config.resolve()
    supplied, configuration_sources = experiment_config.configuration_sources(config)
    input_bytes = configuration_sources[-1]["content"].encode()
    resolved = experiment_config.resolve_configuration(
        supplied, base_directory=config.parent
    )
    budget = training_budget.TrainingBudget(
        resolved["budget"]["iterations"],
        resolved["budget"]["max_seconds"],
        invocation_started,
    )
    parent = (
        continuation.load(pathlib.Path(resolved["initial_checkpoint"]), resolved)
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
            set_seed(resolved["seed"])
            execution, model_settings, training_settings = (
                resolved[key] for key in ("execution", "model", "training")
            )
            torch.set_num_threads(execution["threads_per_worker"])
            device = torch.device(execution["device"])
            loss = functools.partial(
                objectives.configured_loss,
                auxiliary_objectives=resolved["auxiliary_objectives"],
                value_scale=training_settings["value_scale"],
                policy_scale=training_settings["policy_scale"],
            )
            training_config = train.ReplayRatioTrainConfig(
                batch_size=training_settings["batch_size"],
                replay_ratio=training_settings["replay_ratio"],
                replay_ratio_after_fill=training_settings["replay_ratio_after_fill"],
                learn_rate=training_settings["learn_rate"],
                loss_function=loss,
                gradient_diagnostic=training_settings["gradient_diagnostic"],
            )
            learn_config = train.LearnConfig(
                torch_device=device,
                learn_steps=resolved["budget"]["iterations"],
                games_generated_per_iteration=resolved["selfplay"][
                    "games_per_iteration"
                ],
                checkpoint_interval=resolved["budget"]["checkpoint_interval"],
                max_seconds=resolved["budget"]["max_seconds"],
            )
            search = dict(resolved["search"])
            temperature = search.pop("action_softmax_temperature")
            player_config = player.ModelPlayerConfig(
                action_softmax_temperature=temperature,
                **mcts.MCTSConfig(**search).kwargs("mcts"),
            )
            shapes = resolved["derived"]
            buffer_config = buffer.Config(
                max_size=resolved["replay"]["capacity"],
                spatial_input_shape=tuple(shapes["spatial_input_shape"]),
                non_spatial_input_shape=tuple(shapes["non_spatial_input_shape"]),
                action_mask_shape=tuple(shapes["action_mask_shape"]),
                target_specs=tuple(
                    buffer.TargetShapeSpec(spec["name"], tuple(spec["shape"]))
                    for spec in shapes["target_specs"]
                ),
                path=recorder.path / "data" / "replay",
            )
            start_state = (
                create_random_potential_clear_position
                if resolved["selfplay"]["start_state"] == "potential_clear"
                else None
            )
            initial_dataset = resolved["replay"]["initial_dataset"]
            train_self_play(
                parent=parent,
                budget=budget,
                expected_initial_dataset_id=resolved["replay"]["dataset_id"],
                auxiliary_targets=resolved["auxiliary_targets"],
                process_count=execution["workers"],
                torch_threads_per_worker=execution["threads_per_worker"],
                players=resolved["players"],
                checkpoints_dir=recorder.path / "checkpoints",
                learn_config=learn_config,
                training_config=training_config,
                training_data_buffer_config=buffer_config,
                model_player_config=player_config,
                model_settings=model_settings,
                auxiliary_objectives=resolved["auxiliary_objectives"],
                run_seed=resolved["seed"],
                games_per_task=resolved["selfplay"]["games_per_task"],
                start_state_generator=start_state,
                initial_training_dataset_path=pathlib.Path(initial_dataset)
                if initial_dataset
                else None,
                recorder=recorder,
                observations=experiment_training.ObservationConfig(
                    **resolved["logging"],
                    **resolved["validation"],
                ),
            )
    finally:
        logger.removeHandler(handler)
        handler.close()
        logger.setLevel(previous_level)
    return recorder.path
