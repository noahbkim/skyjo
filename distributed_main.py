from __future__ import annotations

import datetime
import logging
import pathlib
import pickle as pkl
import random
import typing

import numpy as np
import torch
import torch.multiprocessing as mp

from skyjo import (
    buffer,
    explain,
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


def play_games_locally(
    model_state_dict: dict[str, torch.Tensor],
    model_kwargs: dict[str, typing.Any],
    model_player_config: player.ModelPlayerConfig,
    players: int,
    number_of_games: int,
    seed: int,
    start_state_generator: StartStateGenerator | None = None,
) -> list[play.GameHistory]:
    """Pool worker entrypoint.

    This intentionally does not use PredictorProcess or any queues. Each task gets
    a snapshot of the model weights, builds a local predictor client, and returns
    game histories to the parent process.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    model = skynet.EquivariantSkyNet(
        spatial_input_shape=(players, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
        non_spatial_input_shape=(sj.GAME_SIZE,),
        value_output_shape=(players,),
        policy_output_shape=(sj.MASK_SIZE,),
        device=torch.device("cpu"),
        **model_kwargs,
    )
    model.load_state_dict(model_state_dict)
    model.eval()

    predictor_client = predictor.LocalPredictorClient(
        model=model,
        max_batch_size=512,
    )
    model_player = player.ModelPlayer(
        predictor_client,
        **model_player_config.kwargs(),
    )
    model_players = [model_player for _ in range(players)]

    if start_state_generator is None:
        return play.distributed_play(
            model_players,
            number_of_games=number_of_games,
        )

    game_histories = []
    for _ in range(number_of_games):
        game_histories.extend(
            play.distributed_play(
                model_players,
                start_state=start_state_generator(),
                number_of_games=1,
            )
        )
    return game_histories


def game_batch_sizes(total_games: int, games_per_task: int) -> list[int]:
    return [
        min(games_per_task, total_games - start)
        for start in range(0, total_games, games_per_task)
    ]


def run_apply_async_local_selfplay_learning(
    process_count: int,
    players: int,
    model_factory: factory.SkyNetModelFactory,
    learn_config: train.LearnConfig,
    training_config: train.TrainConfig,
    training_data_buffer_config: buffer.Config,
    model_player_config: player.ModelPlayerConfig,
    model_kwargs: dict[str, typing.Any],
    games_per_task: int = 1,
    start_state_generator: StartStateGenerator | None = None,
    outcome_rollouts: int = 1,
) -> None:
    training_data_buffer = buffer.ReplayBuffer(**training_data_buffer_config.kwargs())
    model = model_factory.get_latest_model()
    model.set_device(learn_config.torch_device)

    with mp.Pool(processes=process_count) as pool:
        for iteration in range(learn_config.learn_steps):
            logging.info("[LEARN] Starting iteration %s", iteration)

            if (
                learn_config.validation_interval is not None
                and learn_config.validation_function is not None
                and iteration % learn_config.validation_interval == 0
            ):
                learn_config.validation_function(model)

            model_state_dict = {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            }
            async_results = [
                pool.apply_async(
                    play_games_locally,
                    (
                        model_state_dict,
                        model_kwargs,
                        model_player_config,
                        players,
                        batch_size,
                        iteration * learn_config.games_generated_per_iteration
                        + task_index,
                        start_state_generator,
                    ),
                )
                for task_index, batch_size in enumerate(
                    game_batch_sizes(
                        learn_config.games_generated_per_iteration,
                        games_per_task,
                    )
                )
            ]

            game_histories = []
            for result in async_results:
                game_histories.extend(result.get())

            game_stats_list = []
            for game_history in game_histories:
                game_data, game_stats = play.game_history_to_game_data(
                    game_history,
                    terminal_rollouts=outcome_rollouts,
                )
                training_data_buffer.add_game_data(game_data)
                game_stats_list.append(game_stats)

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
                training_data_buffer.save(),
            )

            if len(training_data_buffer) < training_config.batch_size:
                logging.info(
                    "[LEARN] Skipping training until buffer has at least %s positions",
                    training_config.batch_size,
                )
                continue

            for epoch in range(training_config.epochs):
                loss_details = train.train_epoch(
                    model,
                    training_data_buffer,
                    training_batch_size=training_config.batch_size,
                    learn_rate=training_config.learn_rate,
                    loss_function=training_config.loss_function,
                )
                if learn_config.loss_stats_function is not None:
                    logging.info(
                        "[LEARN] Training epoch %s stats:\n%s",
                        epoch,
                        learn_config.loss_stats_function(loss_details),
                    )

            if (
                learn_config.update_model_interval is not None
                and (iteration + 1) % learn_config.update_model_interval == 0
            ):
                logging.info(
                    "[LEARN] Saved model to %s",
                    model_factory.save_model(model),
                )

    logging.info("[LEARN] Saved final model to %s", model_factory.save_model(model))


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

    validation_batch_path = pathlib.Path(
        "./data/validation/greedy_ev_validation_batch.pkl"
    )
    validation_function = None
    if validation_batch_path.exists():
        with open(validation_batch_path, "rb") as f:
            validation_batch = pkl.load(f)
        validation_function = lambda model: explain.validate_model(
            model,
            validation_batch,
        )

    model_kwargs = {
        "embedding_dimensions": 32,
        "global_state_embedding_dimensions": 64,
        "num_heads": 2,
    }
    model = skynet.EquivariantSkyNet(
        spatial_input_shape=(players, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
        non_spatial_input_shape=(sj.GAME_SIZE,),
        value_output_shape=(players,),
        policy_output_shape=(sj.MASK_SIZE,),
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
        batch_size=256,
        learn_rate=1e-3,
        loss_function=lambda model_outputs, targets: train_utils.base_loss(
            model_outputs,
            targets,
            value_scale=1 / (skynet.SCORE_DIFFERENTIAL_CAP**2),
        ),
    )
    learn_config = train.LearnConfig(
        torch_device=device,
        learn_steps=1000,
        games_generated_per_iteration=2500,
        loss_stats_function=train_utils.loss_details_summary,
        validation_interval=1,
        validation_function=validation_function,
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
        spatial_input_shape=(players, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
        non_spatial_input_shape=(sj.GAME_SIZE,),
        action_mask_shape=(sj.MASK_SIZE,),
        outcome_target_shape=(players,),  # score-differential value target
        points_target_shape=(players,),
        policy_target_shape=(sj.MASK_SIZE,),
        cleared_columns_target_shape=(players * sj.COLUMN_COUNT,),
        path=pathlib.Path("./data/training_data") / timestamp / "buffer.pkl",
    )

    run_apply_async_local_selfplay_learning(
        process_count=process_count,
        players=players,
        model_factory=model_factory,
        learn_config=learn_config,
        training_config=training_config,
        training_data_buffer_config=training_data_buffer_config,
        model_player_config=model_player_config,
        model_kwargs=model_kwargs,
        games_per_task=games_per_task,
        start_state_generator=start_state_generator,
        outcome_rollouts=outcome_rollouts,
    )
