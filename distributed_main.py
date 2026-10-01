"""Status: manual
Purpose: Launch recorded, config-driven distributed self-play experiments.
Promote when: Multiple experiment entrypoints need this orchestration recipe.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import math
import pathlib
import random
import time
import typing

import numpy as np
import torch
import torch.multiprocessing as mp
import typer

from skyjo import (
    buffer,
    checkpoint,
    experiment_config,
    experiment_training,
    explain,
    faceoff,
    factory,
    mcts,
    play,
    player,
    predictor,
    runs,
    skynet,
    train,
    train_utils,
)
from skyjo import game as sj

StartStateGenerator: typing.TypeAlias = typing.Callable[[], sj.Skyjo | None]
PLAY_SEED_STREAM = 0
TARGET_SEED_STREAM = 1


def configure_torch_worker(torch_thread_count: int) -> None:
    """Limit Torch CPU parallelism inside one multiprocessing worker."""
    if torch_thread_count < 1:
        raise ValueError("torch_thread_count must be positive")
    torch.set_num_threads(torch_thread_count)
    torch.set_num_interop_threads(1)


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedGameHistory:
    global_game_index: int
    play_seed: int
    target_seed: int
    history: play.GameHistory


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


def build_local_model(
    model_callable: typing.Callable[..., skynet.SkyNet],
    model_kwargs: dict[str, typing.Any],
    players: int,
    model_state_dict: dict[str, torch.Tensor] | None = None,
) -> skynet.SkyNet:
    spatial_input_shape = (players, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE)
    non_spatial_input_shape = skynet.get_non_spatial_input_shape(players)
    value_output_shape = (players,)
    policy_output_shape = (sj.MASK_SIZE,)
    model = model_callable(
        spatial_input_shape=spatial_input_shape,
        non_spatial_input_shape=non_spatial_input_shape,
        value_output_shape=value_output_shape,
        policy_output_shape=policy_output_shape,
        device=torch.device("cpu"),
        **model_kwargs,
    )
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)
    model.eval()
    return model


def play_games_locally(
    model_callable: typing.Callable[..., skynet.SkyNet],
    model_state_dict: dict[str, torch.Tensor],
    model_kwargs: dict[str, typing.Any],
    model_player_config: player.ModelPlayerConfig,
    players: int,
    number_of_games: int,
    run_seed: int,
    first_game_index: int,
    start_state_generator: StartStateGenerator | None = None,
) -> list[GeneratedGameHistory]:
    """Pool worker entrypoint.

    This intentionally does not use PredictorProcess or any queues. Each task gets
    a snapshot of the model weights, builds a local predictor client, and returns
    game histories to the parent process.
    """
    model = build_local_model(
        model_callable=model_callable,
        model_kwargs=model_kwargs,
        players=players,
        model_state_dict=model_state_dict,
    )

    predictor_client = predictor.LocalPredictorClient(
        model=model,
        max_batch_size=512,
    )
    model_player = player.ModelPlayer(
        predictor_client,
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
        target_seed = derive_game_seed(
            run_seed,
            global_game_index,
            TARGET_SEED_STREAM,
        )
        set_seed(play_seed)
        start_state = None if start_state_generator is None else start_state_generator()
        history = play.distributed_play(
            model_players,
            start_state=start_state,
            number_of_games=1,
        )[0]
        generated_games.append(
            GeneratedGameHistory(
                global_game_index=global_game_index,
                play_seed=play_seed,
                target_seed=target_seed,
                history=history,
            )
        )
    return generated_games


def game_batch_sizes(total_games: int, games_per_task: int) -> list[int]:
    return [
        min(games_per_task, total_games - start)
        for start in range(0, total_games, games_per_task)
    ]


def add_generated_games_to_buffer(
    generated_games: typing.Iterable[GeneratedGameHistory],
    training_data_buffer: buffer.ReplayBuffer,
    *,
    outcome_rollouts: int,
    log_progress: bool = False,
) -> list[play.GameStats]:
    """Convert generated histories in global game order using target seeds."""
    ordered_games = sorted(
        generated_games,
        key=lambda game: game.global_game_index,
    )
    progress_interval = max(1, math.ceil(len(ordered_games) / 10))
    started_at = time.perf_counter()
    converted_positions = 0
    game_stats_list = []
    for completed_games, generated_game in enumerate(ordered_games, start=1):
        set_seed(generated_game.target_seed)
        game_data, game_stats = play.game_history_to_game_data(
            generated_game.history,
            terminal_rollouts=outcome_rollouts,
            include_future_clear_target=(
                skynet.FUTURE_CLEAR_TARGET_NAME in training_data_buffer.target_names
            ),
        )
        training_data_buffer.add_game_data(
            game_data,
            game_index=generated_game.global_game_index,
            play_seed=generated_game.play_seed,
            target_seed=generated_game.target_seed,
        )
        game_stats_list.append(game_stats)
        converted_positions += len(game_data)
        if log_progress and (
            completed_games % progress_interval == 0
            or completed_games == len(ordered_games)
        ):
            elapsed_seconds = time.perf_counter() - started_at
            logging.info(
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
) -> buffer.ReplayBuffer:
    """Load the destination dataset, or seed a fresh destination from a snapshot."""
    destination_manifest = (
        None if config.path is None else config.path / buffer.MANIFEST_FILE
    )
    if destination_manifest is not None and destination_manifest.is_file():
        return buffer.ReplayBuffer.from_config_or_load(config)
    if initial_dataset_path is None:
        return buffer.ReplayBuffer.from_config(config)
    if not (initial_dataset_path / buffer.MANIFEST_FILE).is_file():
        raise FileNotFoundError(f"No replay dataset found at {initial_dataset_path}")

    replay_buffer = buffer.ReplayBuffer.from_config_or_load(
        dataclasses.replace(config, path=initial_dataset_path)
    )
    replay_buffer.path = config.path
    return replay_buffer


def faceoff_models_locally(
    model_callable: typing.Callable[..., skynet.SkyNet],
    model_state_dict: dict[str, torch.Tensor],
    previous_model_state_dict: dict[str, torch.Tensor],
    model_kwargs: dict[str, typing.Any],
    players: int,
    paired_rounds: int,
    seed: int,
    model_player_config: player.ModelPlayerConfig,
    start_state_generator: StartStateGenerator | None = None,
) -> tuple[int, int]:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    trained_model = build_local_model(
        model_callable=model_callable,
        model_kwargs=model_kwargs,
        players=players,
        model_state_dict=model_state_dict,
    )
    previous_model = build_local_model(
        model_callable=model_callable,
        model_kwargs=model_kwargs,
        players=players,
        model_state_dict=previous_model_state_dict,
    )

    return faceoff.model_mcts_faceoff(
        trained_model,
        previous_model,
        model_player_config=model_player_config,
        paired_rounds=paired_rounds,
        seed=seed,
        start_state_generator=start_state_generator,
    )


def validate_model_faceoff(
    pool: mp.pool.Pool,
    model: skynet.SkyNet,
    players: int,
    previous_model_state_dict: dict[str, torch.Tensor],
    model_callable: typing.Callable[..., skynet.SkyNet],
    model_kwargs: dict[str, typing.Any],
    rounds: int,
    rounds_per_task: int,
    model_player_config: player.ModelPlayerConfig,
    start_state_generator: StartStateGenerator | None = None,
) -> dict[str, int | bool]:
    model_state_dict = {
        name: value.detach().cpu() for name, value in model.state_dict().items()
    }
    async_results = [
        pool.apply_async(
            faceoff_models_locally,
            (
                model_callable,
                model_state_dict,
                previous_model_state_dict,
                model_kwargs,
                players,
                batch_size,
                task_index,
                model_player_config,
                start_state_generator,
            ),
        )
        for task_index, batch_size in enumerate(
            game_batch_sizes(
                rounds,
                rounds_per_task,
            )
        )
    ]

    trained_model_wins = 0
    previous_model_wins = 0
    for result in async_results:
        task_trained_model_wins, task_previous_model_wins = result.get()
        trained_model_wins += task_trained_model_wins
        previous_model_wins += task_previous_model_wins

    logging.info(
        "[VALIDATION] Faceoff against previous model: trained model wins=%s previous model wins=%s",
        trained_model_wins,
        previous_model_wins,
    )
    return {
        "candidate_wins": trained_model_wins,
        "champion_wins": previous_model_wins,
        "promoted": trained_model_wins > previous_model_wins,
    }


def generate_iteration(
    pool,
    *,
    total_games: int,
    games_per_task: int,
    first_game_index: int,
    worker_kwargs: dict,
) -> list[GeneratedGameHistory]:
    """Dispatch one generation batch and collect its complete game histories."""
    sizes = game_batch_sizes(total_games, games_per_task)
    starts = np.cumsum([0, *sizes[:-1]]).tolist()
    results = [
        pool.apply_async(
            play_games_locally,
            kwds={
                **worker_kwargs,
                "number_of_games": size,
                "first_game_index": first_game_index + offset,
            },
        )
        for size, offset in zip(sizes, starts, strict=True)
    ]
    generated = []
    for completed, result in enumerate(results, start=1):
        generated.extend(result.get())
        logging.info(
            "[SELF-PLAY] Completed %s/%s tasks and %s/%s games",
            completed,
            len(results),
            len(generated),
            total_games,
        )
    return generated


def run_apply_async_local_selfplay_learning(
    process_count: int,
    players: int,
    model_factory: factory.SkyNetModelFactory,
    learn_config: train.LearnConfig,
    training_config: train.ReplayRatioTrainConfig,
    training_data_buffer_config: buffer.Config,
    model_player_config: player.ModelPlayerConfig,
    model_callable: typing.Callable[..., skynet.SkyNet],
    model_kwargs: dict[str, typing.Any],
    torch_threads_per_worker: int = 1,
    run_seed: int = 0,
    games_per_task: int = 1,
    start_state_generator: StartStateGenerator | None = None,
    outcome_rollouts: int = 1,
    faceoff_rounds: int = 100,
    faceoff_rounds_per_task: int = 1,
    initial_training_dataset_path: pathlib.Path | None = None,
    recorder: runs.RunRecorder | None = None,
    promotion_interval: int = 1,
) -> None:
    if promotion_interval < 1:
        raise ValueError("promotion_interval must be positive")
    if process_count < 1:
        raise ValueError("process_count must be positive")
    if torch_threads_per_worker < 1:
        raise ValueError("torch_threads_per_worker must be positive")
    if games_per_task < 1:
        raise ValueError("games_per_task must be positive")
    training_data_buffer = initialize_training_data_buffer(
        training_data_buffer_config,
        initial_training_dataset_path,
    )
    model = model_factory.get_latest_model()
    model.set_device(learn_config.torch_device)
    optimizer = train.make_optimizer(model, training_config.learn_rate)
    run_configuration = {
        "model": {
            "name": getattr(model, "architecture_name", type(model).__name__),
            "non_spatial_input_shape": model.non_spatial_input_shape,
            **model_kwargs,
        },
        "player": model_player_config,
        "learn": {
            "games_generated_per_iteration": (
                learn_config.games_generated_per_iteration
            ),
            "validation_interval": learn_config.validation_interval,
            "promotion_interval": promotion_interval,
        },
        "training": {
            "batch_size": training_config.batch_size,
            "replay_ratio": training_config.replay_ratio,
            "learn_rate": training_config.learn_rate,
            "loss_function": training_config.loss_function,
        },
    }
    progress = checkpoint.load_checkpoint(
        model_factory.get_latest_checkpoint_path(),
        model=model,
        optimizer=optimizer,
        expected_configuration=run_configuration,
        map_location=learn_config.torch_device,
    )

    state = experiment_training.TrainingState(progress, progress)
    recording = experiment_training.RecipeRecording(
        recorder,
        training_data_buffer,
        initial_dataset_path=initial_training_dataset_path,
    )
    initial = recording.register_snapshot(
        model_factory.get_latest_checkpoint_path(),
        state,
        role="initial",
    )
    # Rollback must restore even the initially empty optimizer state.
    champion = (
        recording.save_snapshot(
            model_factory, model, optimizer, run_configuration, state, role="champion"
        )
        if faceoff_rounds > 0
        else initial
    )
    state = dataclasses.replace(state, snapshot=champion)
    next_game = max(training_data_buffer.game_indices, default=-1) + 1

    with mp.Pool(
        processes=process_count,
        initializer=configure_torch_worker,
        initargs=(torch_threads_per_worker,),
    ) as pool:
        for iteration in range(progress.iteration, learn_config.learn_steps):
            iteration_started = time.perf_counter()
            timings = {}
            logging.info(
                "[LEARN] Starting iteration %s/%s",
                iteration + 1,
                learn_config.learn_steps,
            )

            # Validate the current state, then generate an explicitly attributed batch.
            if learn_config.validation_function is not None and train.interval_due(
                iteration, learn_config.validation_interval
            ):
                started = time.perf_counter()
                metrics = learn_config.validation_function(model)
                recording.event(
                    "validation",
                    state,
                    metrics=metrics or {},
                    context={
                        "suite": "built_in_examples",
                        "seconds": time.perf_counter() - started,
                    },
                )
            generation = state.snapshot or recording.save_snapshot(
                model_factory,
                model,
                optimizer,
                run_configuration,
                state,
                role="generation",
            )
            state = dataclasses.replace(state, snapshot=generation)
            started = time.perf_counter()
            generated = generate_iteration(
                pool,
                total_games=learn_config.games_generated_per_iteration,
                games_per_task=games_per_task,
                first_game_index=next_game,
                worker_kwargs={
                    "model_callable": model_callable,
                    "model_kwargs": model_kwargs,
                    "model_state_dict": {
                        name: value.detach().cpu()
                        for name, value in model.state_dict().items()
                    },
                    "model_player_config": model_player_config,
                    "players": players,
                    "run_seed": run_seed,
                    "start_state_generator": start_state_generator,
                },
            )
            timings["generation"] = time.perf_counter() - started
            before = (training_data_buffer.game_count, len(training_data_buffer))
            started = time.perf_counter()
            games = add_generated_games_to_buffer(
                generated,
                training_data_buffer,
                outcome_rollouts=outcome_rollouts,
                log_progress=True,
            )
            timings["target"] = time.perf_counter() - started
            new_positions = sum(stats.game_length for stats in games)
            state = state.generated(games=len(games), positions=new_positions)
            logging.info(
                "[LEARN] Generated game stats:\n%s",
                train_utils.game_stats_summary(games),
            )
            started = time.perf_counter()
            recording.save_replay(
                training_data_buffer,
                state=state,
                generation=generation,
                game_count=len(games),
                generation_settings={
                    "run_seed": run_seed,
                    "players": players,
                    "model": run_configuration["model"],
                    "model_player": dataclasses.asdict(model_player_config),
                    "outcome_rollouts": outcome_rollouts,
                },
            )
            timings["replay_save"] = time.perf_counter() - started
            next_game += len(games)

            # Training changes the candidate; work counters never roll back.
            steps = math.ceil(
                new_positions
                * training_config.replay_ratio
                / training_config.batch_size
            )
            started = time.perf_counter()
            losses = train.train_steps(
                model,
                training_data_buffer,
                training_batch_size=training_config.batch_size,
                optimizer_steps=steps,
                optimizer=optimizer,
                loss_function=training_config.loss_function,
            )
            timings["training"] = time.perf_counter() - started
            state = state.trained(
                iteration=iteration + 1,
                steps=steps,
                batch_size=training_config.batch_size,
            )
            recording.training(state, losses)
            if learn_config.loss_stats_function is not None:
                logging.info(
                    "[LEARN] Training stats:\n%s",
                    learn_config.loss_stats_function(losses),
                )

            # Only evaluation replaces the accepted champion, regardless of saving.
            if faceoff_rounds > 0 and train.interval_due(
                iteration + 1, promotion_interval
            ):
                champion_weights = torch.load(
                    champion.path, map_location="cpu", weights_only=False
                )["model_state_dict"]
                transition = experiment_training.promote_candidate(
                    model=model,
                    optimizer=optimizer,
                    factory=model_factory,
                    configuration=run_configuration,
                    state=state,
                    champion=champion,
                    evaluate=functools.partial(
                        validate_model_faceoff,
                        pool=pool,
                        model=model,
                        players=players,
                        previous_model_state_dict=champion_weights,
                        model_callable=model_callable,
                        model_kwargs=model_kwargs,
                        rounds=faceoff_rounds,
                        rounds_per_task=faceoff_rounds_per_task,
                        model_player_config=model_player_config,
                        start_state_generator=start_state_generator,
                    ),
                    recording=recording,
                    protocol={
                        "paired_rounds": faceoff_rounds,
                        "interval": promotion_interval,
                        "rounds_per_task": faceoff_rounds_per_task,
                        "seed_rule": "task index; each task increments by pair index",
                        "player": dataclasses.asdict(
                            dataclasses.replace(
                                model_player_config, action_softmax_temperature=0.0
                            )
                        ),
                        "start_state": "standard"
                        if start_state_generator is None
                        else start_state_generator.__qualname__,
                    },
                )
                state, champion = transition.state, transition.champion

            if train.interval_due(iteration + 1, learn_config.update_model_interval):
                saved = recording.save_snapshot(
                    model_factory,
                    model,
                    optimizer,
                    run_configuration,
                    state,
                    role="periodic",
                )
                state = dataclasses.replace(state, snapshot=saved)
            timings["iteration"] = time.perf_counter() - iteration_started
            recording.iteration(
                state,
                games=games,
                replay=training_data_buffer,
                before=before,
                timings=timings,
            )
            logging.info(
                "[LEARN] Completed iteration %s in %.1fs",
                iteration + 1,
                timings["iteration"],
            )

    recording.save_snapshot(
        model_factory, model, optimizer, run_configuration, state, role="final"
    )


def create_random_potential_clear_position() -> sj.Skyjo:
    return explain.create_potential_clear_equal_position(
        np.random.randint(0, sj.CARD_SIZE)
    )


MODEL_TYPES = {
    "equivariant": skynet.EquivariantSkyNet,
    "round_score": skynet.EquivariantSkyNetWithRoundScoreAux,
    "auxiliary": skynet.EquivariantSkyNetWithAuxiliaryHeads,
}
LOSS_TYPES = {
    "base": train_utils.base_loss,
    "auxiliary": train_utils.outcome_policy_auxiliary_loss,
}


def launch(
    config: typing.Annotated[pathlib.Path, typer.Option("--config")],
    runs_dir: typing.Annotated[pathlib.Path, typer.Option("--runs-dir")] = pathlib.Path(
        ".runs"
    ),
    allow_dirty: typing.Annotated[bool, typer.Option("--allow-dirty")] = False,
) -> pathlib.Path:
    """Start a fresh experiment and return its run directory."""
    config = config.resolve()
    input_bytes, resolved = experiment_config.load_configuration(config)
    repository = pathlib.Path(__file__).resolve().parent
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
    typer.echo(f"Run directory: {recorder.path}")
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
            set_seed(resolved["seed"])
            execution, model_settings, training_settings = (
                resolved[key] for key in ("execution", "model", "training")
            )
            torch.set_num_threads(execution["threads_per_worker"])
            device = torch.device(execution["device"])
            model_callable = MODEL_TYPES[model_settings["type"]]
            model_kwargs = {
                key: value for key, value in model_settings.items() if key != "type"
            }
            model_factory = factory.SkyNetModelFactory(
                model_callable=model_callable,
                players=resolved["players"],
                device=device,
                models_dir=recorder.path / "checkpoints",
                model_kwargs=model_kwargs,
            )
            loss_keys = ["value_scale", "policy_scale"]
            if training_settings["loss"] == "auxiliary":
                loss_keys += [
                    "round_score_scale",
                    "future_clear_scale",
                    "clear_positive_weight",
                ]
            loss = functools.partial(
                LOSS_TYPES[training_settings["loss"]],
                **{key: training_settings[key] for key in loss_keys},
            )
            training_config = train.ReplayRatioTrainConfig(
                batch_size=training_settings["batch_size"],
                replay_ratio=training_settings["replay_ratio"],
                learn_rate=training_settings["learn_rate"],
                loss_function=loss,
            )
            validation = resolved["validation"]
            validation_function = (
                functools.partial(
                    explain.validate_model,
                    value_loss_scale=validation["value_loss_scale"],
                    policy_loss_scale=validation["policy_loss_scale"],
                )
                if validation["enabled"]
                else None
            )
            learn_config = train.LearnConfig(
                torch_device=device,
                learn_steps=resolved["budget"]["iterations"],
                games_generated_per_iteration=resolved["selfplay"][
                    "games_per_iteration"
                ],
                training_epochs=0,
                training_batch_size=training_config.batch_size,
                training_learn_rate=training_config.learn_rate,
                training_loss_function=loss,
                loss_stats_function=train_utils.loss_details_summary,
                validation_interval=validation["interval"]
                if validation["enabled"]
                else None,
                validation_function=validation_function,
                update_model_interval=resolved["budget"]["checkpoint_interval"],
                model_faceoff_function=None,
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
            run_apply_async_local_selfplay_learning(
                process_count=execution["workers"],
                torch_threads_per_worker=execution["threads_per_worker"],
                players=resolved["players"],
                model_factory=model_factory,
                learn_config=learn_config,
                training_config=training_config,
                training_data_buffer_config=buffer_config,
                model_player_config=player_config,
                model_callable=model_callable,
                model_kwargs=model_kwargs,
                run_seed=resolved["seed"],
                games_per_task=resolved["selfplay"]["games_per_task"],
                start_state_generator=start_state,
                outcome_rollouts=resolved["selfplay"]["outcome_rollouts"],
                faceoff_rounds=resolved["faceoff"]["paired_rounds"],
                faceoff_rounds_per_task=resolved["faceoff"]["rounds_per_task"],
                promotion_interval=resolved["faceoff"]["interval"],
                initial_training_dataset_path=pathlib.Path(initial_dataset)
                if initial_dataset
                else None,
                recorder=recorder,
            )
    finally:
        logger.removeHandler(handler)
        handler.close()
        logger.setLevel(previous_level)
    return recorder.path


def main() -> None:
    typer.run(launch)


if __name__ == "__main__":
    main()
