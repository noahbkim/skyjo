from __future__ import annotations

import datetime
import dataclasses
import functools
import logging
import pathlib
import random
import typing

import numpy as np
import torch
import torch.multiprocessing as mp

from skyjo import (
    buffer,
    checkpoint,
    explain,
    faceoff,
    factory,
    mcts,
    play,
    player,
    predictor,
    skynet,
    train,
    train_utils,
)
from skyjo import game as sj

StartStateGenerator: typing.TypeAlias = typing.Callable[[], sj.Skyjo | None]
PLAY_SEED_STREAM = 0
TARGET_SEED_STREAM = 1


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
        np.random.SeedSequence(
            [run_seed, global_game_index, stream]
        ).generate_state(1, dtype=np.uint32)[0]
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
    non_spatial_input_shape = (sj.GAME_SIZE,)
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
        start_state = (
            None if start_state_generator is None else start_state_generator()
        )
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
) -> list[play.GameStats]:
    """Convert generated histories in global game order using target seeds."""
    game_stats_list = []
    for generated_game in sorted(
        generated_games,
        key=lambda game: game.global_game_index,
    ):
        set_seed(generated_game.target_seed)
        game_data, game_stats = play.game_history_to_game_data(
            generated_game.history,
            terminal_rollouts=outcome_rollouts,
        )
        training_data_buffer.add_game_data(
            game_data,
            game_index=generated_game.global_game_index,
            play_seed=generated_game.play_seed,
            target_seed=generated_game.target_seed,
        )
        game_stats_list.append(game_stats)
    return game_stats_list


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
    training_config: train.TrainConfig,
    training_data_buffer_config: buffer.Config,
    model_player_config: player.ModelPlayerConfig,
    model_callable: typing.Callable[..., skynet.SkyNet],
    model_kwargs: dict[str, typing.Any],
    run_seed: int = 0,
    games_per_task: int = 1,
    start_state_generator: StartStateGenerator | None = None,
    outcome_rollouts: int = 1,
    faceoff_rounds: int = 100,
    faceoff_rounds_per_task: int = 1,
) -> None:
    training_data_buffer = buffer.ReplayBuffer.from_config_or_load(
        training_data_buffer_config
    )
    model = model_factory.get_latest_model()
    model.set_device(learn_config.torch_device)
    optimizer = train.make_optimizer(model, training_config.learn_rate)
    run_configuration = {
        "model": model_kwargs,
        "player": model_player_config,
        "learn": {
            "games_generated_per_iteration": (
                learn_config.games_generated_per_iteration
            ),
            "validation_interval": learn_config.validation_interval,
            "update_model_interval": learn_config.update_model_interval,
        },
        "training": {
            "epochs": training_config.epochs,
            "batch_size": training_config.batch_size,
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

    with mp.Pool(processes=process_count) as pool:
        for iteration in range(progress.iteration, learn_config.learn_steps):
            logging.info("[LEARN] Starting iteration %s", iteration)

            if (
                learn_config.validation_function is not None
                and train.interval_due(iteration, learn_config.validation_interval)
            ):
                learn_config.validation_function(model)

            model_state_dict = {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            }
            batch_sizes = game_batch_sizes(
                learn_config.games_generated_per_iteration,
                games_per_task,
            )
            batch_starts = np.cumsum([0, *batch_sizes[:-1]]).tolist()
            first_iteration_game = (
                iteration * learn_config.games_generated_per_iteration
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
            for result in async_results:
                generated_games.extend(result.get())
            game_stats_list = add_generated_games_to_buffer(
                generated_games,
                training_data_buffer,
                outcome_rollouts=outcome_rollouts,
            )

            logging.info(
                "[LEARN] Added %s games and %s positions to the replay buffer",
                len(game_stats_list),
                sum(game_stats.game_length for game_stats in game_stats_list),
            )
            logging.info("[LEARN] Replay buffer size: %s", len(training_data_buffer))
            logging.info(
                "[LEARN] Generated game stats:\n%s",
                train_utils.game_stats_summary(game_stats_list),
            )
            logging.info(
                "[LEARN] Saved replay buffer to %s",
                training_data_buffer.save(
                    generation_metadata={
                        "run_seed": run_seed,
                        "players": players,
                        "model": model_kwargs,
                        "model_player": model_player_config,
                        "outcome_rollouts": outcome_rollouts,
                    },
                    source_checkpoint=model_factory.get_latest_checkpoint_path(),
                ),
            )

            if len(training_data_buffer) < training_config.batch_size:
                progress = dataclasses.replace(
                    progress,
                    iteration=iteration + 1,
                    epoch=0,
                    generated_games=progress.generated_games + len(game_stats_list),
                )
                logging.info(
                    "[LEARN] Skipping training until buffer has at least %s positions",
                    training_config.batch_size,
                )
                continue

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
            optimizer_steps_per_epoch = (
                len(training_data_buffer) // training_config.batch_size + 1
            )
            for epoch in range(training_config.epochs):
                loss_details = train.train_epoch(
                    model,
                    training_data_buffer,
                    training_batch_size=training_config.batch_size,
                    optimizer=optimizer,
                    loss_function=training_config.loss_function,
                )
                if learn_config.loss_stats_function is not None:
                    logging.info(
                        "[LEARN] Training epoch %s stats:\n%s",
                        epoch,
                        learn_config.loss_stats_function(loss_details),
                    )

            optimizer_steps = (
                optimizer_steps_per_epoch * training_config.epochs
            )
            sampled_positions = optimizer_steps * training_config.batch_size
            progress = checkpoint.TrainingProgress(
                iteration=iteration + 1,
                epoch=training_config.epochs,
                generated_games=progress.generated_games + len(game_stats_list),
                trained_positions=progress.trained_positions
                + sampled_positions,
                optimizer_steps=progress.optimizer_steps + optimizer_steps,
                sampled_positions=progress.sampled_positions
                + sampled_positions,
            )

            if (
                train.interval_due(iteration + 1, learn_config.update_model_interval)
            ):
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
    players = 2
    games_per_task = 1
    faceoff_rounds = 0
    faceoff_rounds_per_task = 1
    start_state_generator = None
    outcome_rollouts = 100
    device = torch.device("cpu")

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = pathlib.Path("logs/apply_async_local_train") / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=log_dir / "main.log",
        filemode="w",
    )

    model_kwargs = {
        "embedding_dimensions": 32,
        "global_state_embedding_dimensions": 64,
        "num_heads": 2,
    }
    spatial_input_shape = (players, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE)
    non_spatial_input_shape = (sj.GAME_SIZE,)
    value_output_shape = (players,)
    policy_output_shape = (sj.MASK_SIZE,)
    model = skynet.EquivariantSkyNet(
        spatial_input_shape=spatial_input_shape,
        non_spatial_input_shape=non_spatial_input_shape,
        value_output_shape=value_output_shape,
        policy_output_shape=policy_output_shape,
        device=device,
        **model_kwargs,
    )

    model_factory = factory.SkyNetModelFactory(
        model_callable=skynet.EquivariantSkyNet,
        players=players,
        model_kwargs=model_kwargs,
        device=device,
        models_dir=pathlib.Path("./models") / "apply_async_local" / timestamp,
        initial_model=model,
    )

    training_config = train.TrainConfig(
        epochs=2,
        batch_size=32,
        learn_rate=1e-3,
        loss_function=functools.partial(
            train_utils.base_loss,
            value_scale=1.0,
        ),
    )
    learn_config = train.LearnConfig(
        torch_device=device,
        learn_steps=2,
        games_generated_per_iteration=8,
        loss_stats_function=train_utils.loss_details_summary,
        validation_interval=1,
        validation_function=lambda model: explain.validate_model(
            model, value_loss_scale=1.0
        ),
        update_model_interval=1,
        model_faceoff_function=None,
        **training_config.kwargs("training"),
    )

    mcts_config = mcts.MCTSConfig(
        iterations=400,
        after_state_evaluate_all_children=False,
        terminal_state_initial_rollouts=10,
        dirichlet_epsilon=0.25,
        forced_playout_k=None,
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
        path=pathlib.Path("./data/training_data") / timestamp / "dataset",
    )

    run_apply_async_local_selfplay_learning(
        process_count=process_count,
        players=players,
        model_factory=model_factory,
        learn_config=learn_config,
        training_config=training_config,
        training_data_buffer_config=training_data_buffer_config,
        model_player_config=model_player_config,
        model_callable=skynet.EquivariantSkyNet,
        model_kwargs=model_kwargs,
        run_seed=seed,
        games_per_task=games_per_task,
        start_state_generator=start_state_generator,
        outcome_rollouts=outcome_rollouts,
        faceoff_rounds=faceoff_rounds,
        faceoff_rounds_per_task=faceoff_rounds_per_task,
    )
