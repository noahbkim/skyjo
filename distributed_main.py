"""Status: manual
Purpose: Run multiprocessing Skyjo self-play, training, and model validation.
Promote when: The distributed training orchestration becomes a maintained CLI.
"""

from __future__ import annotations

import dataclasses
import datetime
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

from skyjo import buffer, checkpoint, explain, faceoff, factory
from skyjo import game as sj
from skyjo import mcts, play, player, predictor, skynet, train, train_utils

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
                skynet.FUTURE_CLEAR_TARGET_NAME
                in training_data_buffer.target_names
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
) -> bool:
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
    return trained_model_wins > previous_model_wins


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
) -> None:
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
            "update_model_interval": learn_config.update_model_interval,
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

    starting_iteration = progress.iteration
    first_new_game_index = max(training_data_buffer.game_indices, default=-1) + 1
    tasks_per_iteration = len(
        game_batch_sizes(
            learn_config.games_generated_per_iteration,
            games_per_task,
        )
    )
    logging.info(
        "[LEARN] Starting distributed run: iterations=%s starting_iteration=%s "
        "games_per_iteration=%s workers=%s games_per_task=%s "
        "tasks_per_iteration=%s outcome_rollouts=%s replay_positions=%s/%s",
        learn_config.learn_steps,
        starting_iteration,
        learn_config.games_generated_per_iteration,
        process_count,
        games_per_task,
        tasks_per_iteration,
        outcome_rollouts,
        len(training_data_buffer),
        training_data_buffer.max_size,
    )
    with mp.Pool(
        processes=process_count,
        initializer=configure_torch_worker,
        initargs=(torch_threads_per_worker,),
    ) as pool:
        for iteration in range(starting_iteration, learn_config.learn_steps):
            iteration_started_at = time.perf_counter()
            logging.info(
                "[LEARN] Starting iteration %s/%s",
                iteration + 1,
                learn_config.learn_steps,
            )

            if learn_config.validation_function is not None and train.interval_due(
                iteration, learn_config.validation_interval
            ):
                validation_started_at = time.perf_counter()
                learn_config.validation_function(model)
                logging.info(
                    "[VALIDATION] Completed in %.1fs",
                    time.perf_counter() - validation_started_at,
                )

            model_state_dict = {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            }
            batch_sizes = game_batch_sizes(
                learn_config.games_generated_per_iteration,
                games_per_task,
            )
            batch_starts = np.cumsum([0, *batch_sizes[:-1]]).tolist()
            first_iteration_game = first_new_game_index + (
                (iteration - starting_iteration)
                * learn_config.games_generated_per_iteration
            )
            generation_started_at = time.perf_counter()
            logging.info(
                "[SELF-PLAY] Dispatching %s games across %s tasks "
                "(%s workers, up to %s games/task)",
                learn_config.games_generated_per_iteration,
                len(batch_sizes),
                process_count,
                games_per_task,
            )
            async_results = [
                pool.apply_async(
                    play_games_locally,
                    (
                        model_callable,
                        model_state_dict,
                        model_kwargs,
                        model_player_config,
                        players,
                        batch_size,
                        run_seed,
                        first_iteration_game + batch_start,
                        start_state_generator,
                    ),
                )
                for batch_size, batch_start in zip(
                    batch_sizes, batch_starts, strict=True
                )
            ]

            generated_games = []
            progress_interval = max(1, math.ceil(len(async_results) / 10))
            for completed_tasks, result in enumerate(async_results, start=1):
                generated_games.extend(result.get())
                if (
                    completed_tasks % progress_interval == 0
                    or completed_tasks == len(async_results)
                ):
                    generation_seconds = time.perf_counter() - generation_started_at
                    logging.info(
                        "[SELF-PLAY] Completed %s/%s tasks and %s/%s games in %.1fs "
                        "(%.2f games/s)",
                        completed_tasks,
                        len(async_results),
                        len(generated_games),
                        learn_config.games_generated_per_iteration,
                        generation_seconds,
                        len(generated_games) / generation_seconds,
                    )
            generation_seconds = time.perf_counter() - generation_started_at
            buffer_positions_before = len(training_data_buffer)
            buffer_games_before = training_data_buffer.game_count
            target_started_at = time.perf_counter()
            game_stats_list = add_generated_games_to_buffer(
                generated_games,
                training_data_buffer,
                outcome_rollouts=outcome_rollouts,
                log_progress=True,
            )
            target_seconds = time.perf_counter() - target_started_at
            new_position_count = sum(
                game_stats.game_length for game_stats in game_stats_list
            )
            evicted_positions = max(
                0,
                buffer_positions_before
                + new_position_count
                - len(training_data_buffer),
            )
            evicted_games = max(
                0,
                buffer_games_before
                + len(game_stats_list)
                - training_data_buffer.game_count,
            )

            logging.info(
                "[LEARN] Added %s games and %s positions to the replay buffer; "
                "evicted %s games and %s positions",
                len(game_stats_list),
                new_position_count,
                evicted_games,
                evicted_positions,
            )
            logging.info(
                "[LEARN] Replay buffer: %s games, %s/%s positions (%.1f%% full)",
                training_data_buffer.game_count,
                len(training_data_buffer),
                training_data_buffer.max_size,
                100 * len(training_data_buffer) / training_data_buffer.max_size,
            )
            logging.info(
                "[LEARN] Generation timing: self-play=%.1fs targets=%.1fs total=%.1fs",
                generation_seconds,
                target_seconds,
                generation_seconds + target_seconds,
            )
            logging.info(
                "[LEARN] Generated game stats:\n%s",
                train_utils.game_stats_summary(game_stats_list),
            )
            replay_save_started_at = time.perf_counter()
            replay_path = training_data_buffer.save(
                generation_metadata={
                    "run_seed": run_seed,
                    "players": players,
                    "model": run_configuration["model"],
                    "model_player": model_player_config,
                    "outcome_rollouts": outcome_rollouts,
                },
                source_checkpoint=model_factory.get_latest_checkpoint_path(),
            )
            logging.info(
                "[LEARN] Saved replay buffer to %s in %.1fs",
                replay_path,
                time.perf_counter() - replay_save_started_at,
            )

            previous_model_state_dict = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            champion_checkpoint_path = model_factory.save_model(
                model,
                optimizer=optimizer,
                configuration=run_configuration,
                progress=progress,
            )
            optimizer_steps = math.ceil(
                new_position_count
                * training_config.replay_ratio
                / training_config.batch_size
            )
            sampled_positions = optimizer_steps * training_config.batch_size
            training_started_at = time.perf_counter()
            loss_details = train.train_steps(
                model,
                training_data_buffer,
                training_batch_size=training_config.batch_size,
                optimizer_steps=optimizer_steps,
                optimizer=optimizer,
                loss_function=training_config.loss_function,
            )
            training_seconds = time.perf_counter() - training_started_at
            logging.info(
                "[LEARN] Trained for %s optimizer steps over %s sampled positions "
                "in %.1fs (%.2f steps/s, realized replay ratio %.4f)",
                optimizer_steps,
                sampled_positions,
                training_seconds,
                optimizer_steps / training_seconds,
                sampled_positions / new_position_count,
            )
            if learn_config.loss_stats_function is not None:
                logging.info(
                    "[LEARN] Training stats:\n%s",
                    learn_config.loss_stats_function(loss_details),
                )
            progress = checkpoint.TrainingProgress(
                iteration=iteration + 1,
                epoch=0,
                generated_games=progress.generated_games + len(game_stats_list),
                trained_positions=progress.trained_positions + sampled_positions,
                optimizer_steps=progress.optimizer_steps + optimizer_steps,
                sampled_positions=progress.sampled_positions + sampled_positions,
            )

            if train.interval_due(iteration + 1, learn_config.update_model_interval):
                if faceoff_rounds > 0:
                    logging.info("[LEARN] Model Faceoff")
                    faceoff_result = validate_model_faceoff(
                        pool=pool,
                        model=model,
                        players=players,
                        previous_model_state_dict=previous_model_state_dict,
                        model_callable=model_callable,
                        model_kwargs=model_kwargs,
                        rounds=faceoff_rounds,
                        rounds_per_task=faceoff_rounds_per_task,
                        model_player_config=model_player_config,
                        start_state_generator=start_state_generator,
                    )
                    if faceoff_result:
                        logging.info("[LEARN] New Model Faceoff Passed")
                    else:
                        logging.info(
                            "[LEARN] Model Faceoff Failed, reverting to previous model"
                        )
                        checkpoint.load_checkpoint(
                            champion_checkpoint_path,
                            model=model,
                            optimizer=optimizer,
                            expected_configuration=run_configuration,
                            map_location=learn_config.torch_device,
                        )

                logging.info(
                    "[LEARN] Saved model to %s",
                    model_factory.save_model(
                        model,
                        optimizer=optimizer,
                        configuration=run_configuration,
                        progress=progress,
                    ),
                )

            logging.info(
                "[LEARN] Completed iteration %s/%s in %.1fs",
                iteration + 1,
                learn_config.learn_steps,
                time.perf_counter() - iteration_started_at,
            )

    logging.info(
        "[LEARN] Saved final model to %s",
        model_factory.save_model(
            model,
            optimizer=optimizer,
            configuration=run_configuration,
            progress=progress,
        ),
    )


def create_random_potential_clear_position() -> sj.Skyjo:
    return explain.create_potential_clear_equal_position(
        np.random.randint(0, sj.CARD_SIZE)
    )


if __name__ == "__main__":
    seed = 0
    debug = False
    process_count = 8
    torch_threads_per_worker = 1
    players = 2
    games_per_task = 8
    faceoff_rounds = 0
    faceoff_rounds_per_task = 1
    start_state_generator = None
    outcome_rollouts = 100
    device = torch.device("cpu")
    initial_training_dataset_path = None

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"score_{timestamp}"
    log_dir = pathlib.Path("logs/apply_async_local_train") / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=log_dir / "main.log",
        filemode="w",
    )

    model_kwargs = {
        "embedding_dimensions": 16,
        "global_state_embedding_dimensions": 32,
        "num_heads": 2,
    }
    spatial_input_shape = (players, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE)
    non_spatial_input_shape = skynet.get_non_spatial_input_shape(players)
    value_output_shape = (players,)
    policy_output_shape = (sj.MASK_SIZE,)
    model = skynet.EquivariantSkyNetWithRoundScoreAux(
        spatial_input_shape=spatial_input_shape,
        non_spatial_input_shape=non_spatial_input_shape,
        value_output_shape=value_output_shape,
        policy_output_shape=policy_output_shape,
        device=device,
        **model_kwargs,
    )
    logging.info("Initialized model from random weights")
    logging.info("Using final-round-score auxiliary supervision")

    model_factory = factory.SkyNetModelFactory(
        model_callable=skynet.EquivariantSkyNetWithRoundScoreAux,
        players=players,
        model_kwargs=model_kwargs,
        device=device,
        models_dir=pathlib.Path("./models") / "apply_async_local" / run_name,
        initial_model=model,
    )

    training_config = train.ReplayRatioTrainConfig(
        replay_ratio=4.0,
        batch_size=256,
        learn_rate=1e-3,
        loss_function=functools.partial(
            train_utils.outcome_policy_auxiliary_loss,
            value_scale=1.0,
            policy_scale=1.0,
            round_score_scale=1.0,
            future_clear_scale=0.0,
        ),
    )
    learn_config = train.LearnConfig(
        torch_device=device,
        learn_steps=10,
        games_generated_per_iteration=1024,
        loss_stats_function=train_utils.loss_details_summary,
        validation_interval=1,
        validation_function=lambda model: explain.validate_model(
            model, value_loss_scale=1.0
        ),
        update_model_interval=1,
        model_faceoff_function=None,
        training_epochs=0,
        training_batch_size=training_config.batch_size,
        training_learn_rate=training_config.learn_rate,
        training_loss_function=training_config.loss_function,
    )

    mcts_config = mcts.MCTSConfig(
        iterations=100,
        after_state_evaluate_all_children=False,
        terminal_state_initial_rollouts=10,
        dirichlet_epsilon=0.25,
        c_puct=1.0,
        fpu_reduction=0.25,
        score_utility_weight=0.0,
    )
    model_player_config = player.ModelPlayerConfig(
        action_softmax_temperature=1.0,
        **mcts_config.kwargs("mcts"),
    )
    training_data_buffer_config = buffer.Config(
        max_size=2_000_000,
        spatial_input_shape=spatial_input_shape,
        non_spatial_input_shape=non_spatial_input_shape,
        action_mask_shape=policy_output_shape,
        target_specs=buffer.round_score_target_specs(players, policy_output_shape),
        path=pathlib.Path("./data/training_data") / run_name / "dataset",
    )

    run_apply_async_local_selfplay_learning(
        process_count=process_count,
        torch_threads_per_worker=torch_threads_per_worker,
        players=players,
        model_factory=model_factory,
        learn_config=learn_config,
        training_config=training_config,
        training_data_buffer_config=training_data_buffer_config,
        model_player_config=model_player_config,
        model_callable=skynet.EquivariantSkyNetWithRoundScoreAux,
        model_kwargs=model_kwargs,
        run_seed=seed,
        games_per_task=games_per_task,
        start_state_generator=start_state_generator,
        outcome_rollouts=outcome_rollouts,
        faceoff_rounds=faceoff_rounds,
        faceoff_rounds_per_task=faceoff_rounds_per_task,
        initial_training_dataset_path=initial_training_dataset_path,
    )
