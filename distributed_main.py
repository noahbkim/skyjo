from __future__ import annotations

import datetime
import logging
import pathlib
import pickle as pkl
import random
import time
import typing

import numpy as np
import torch
import torch.multiprocessing as mp

from skyjo import (
    buffer,
    explain,
    faceoff,
    factory,
    mcts,
    player,
    predictor,
    skynet,
    train,
    train_utils,
)
from skyjo import game as sj
from skyjo import play


StartStateGenerator: typing.TypeAlias = typing.Callable[[], sj.Skyjo | None]

_WORKER_ID: int | None = None
_WORKER_PLAYERS: list[player.AbstractPlayer] | None = None
_WORKER_START_STATE_GENERATOR: StartStateGenerator | None = None
_WORKER_OUTCOME_ROLLOUTS = 1
_WORKER_DEBUG = False


def create_obvious_clear_or_almost_clear_position() -> sj.Skyjo:
    if np.random.random() < 0.5:
        return explain.create_obvious_clear_position()
    return explain.create_almost_clear_position()


def create_random_clear_or_almost_clear_position() -> sj.Skyjo | None:
    r = np.random.random()
    if r < 0.1:
        return explain.create_random_clear_starting_position()
    if r < 0.2:
        return explain.create_random_almost_clear_position()
    return None


def create_random_potential_clear_position() -> sj.Skyjo:
    return explain.create_potential_clear_equal_position(
        np.random.randint(0, sj.CARD_SIZE)
    )


def get_start_state_generator(name: str) -> StartStateGenerator | None:
    if name == "none":
        return None
    if name == "obvious-clear":
        return create_obvious_clear_or_almost_clear_position
    if name == "random-clear":
        return create_random_clear_or_almost_clear_position
    if name == "random-potential-clear":
        return create_random_potential_clear_position
    raise ValueError(f"Unknown start state generator: {name}")


def model_faceoff_threshold(
    model: skynet.SkyNet,
    previous_model: skynet.SkyNet,
    policy_rounds: int,
    value_rounds: int,
    temperature: float,
    terminal_state_rollouts: int,
    win_percentage_threshold: float,
    start_state_generator: StartStateGenerator | None = None,
) -> bool:
    policy_faceoff_result = faceoff.model_policy_faceoff(
        model,
        previous_model,
        policy_rounds,
        temperature,
        start_state_generator,
    )
    value_faceoff_result = faceoff.model_value_faceoff(
        model,
        previous_model,
        value_rounds,
        terminal_state_rollouts,
        start_state_generator,
    )
    policy_total = policy_faceoff_result[0] + policy_faceoff_result[1]
    value_total = value_faceoff_result[0] + value_faceoff_result[1]
    return (
        policy_total > 0
        and policy_faceoff_result[0] / policy_total > win_percentage_threshold
    ) or (
        value_total > 0
        and value_faceoff_result[0] / value_total > win_percentage_threshold
    )


def _setup_worker(
    worker_id_queue: mp.Queue,
    predictor_input_queues: dict[int, predictor.PredictorInputQueue],
    predictor_output_queues: dict[int, predictor.PredictorOutputQueue],
    model_player_config: player.ModelPlayerConfig,
    player_count: int,
    start_state_generator: StartStateGenerator | None,
    outcome_rollouts: int,
    debug: bool,
    log_level: int,
    log_dir: pathlib.Path,
    seed: int,
) -> None:
    global _WORKER_ID
    global _WORKER_PLAYERS
    global _WORKER_START_STATE_GENERATOR
    global _WORKER_OUTCOME_ROLLOUTS
    global _WORKER_DEBUG

    _WORKER_ID = worker_id_queue.get()
    _WORKER_START_STATE_GENERATOR = start_state_generator
    _WORKER_OUTCOME_ROLLOUTS = outcome_rollouts
    _WORKER_DEBUG = debug

    np.random.seed(seed + _WORKER_ID)
    random.seed(seed + _WORKER_ID)
    torch.manual_seed(seed + _WORKER_ID)

    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if debug else log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=log_dir / f"pool_worker_{_WORKER_ID}.log",
        filemode="a",
    )

    predictor_client = predictor.DistributedPredictorClient(
        predictor_input_queues[_WORKER_ID],
        predictor_output_queues[_WORKER_ID],
    )
    model_player = player.ModelPlayer(
        predictor_client,
        **model_player_config.kwargs(),
    )
    _WORKER_PLAYERS = [model_player for _ in range(player_count)]
    logging.info("Initialized pool worker %s", _WORKER_ID)


def _generate_game_batch(
    number_of_games: int,
) -> list[tuple[play.GameData, play.GameStats]]:
    if _WORKER_PLAYERS is None:
        raise RuntimeError("Pool worker has not been initialized")

    if _WORKER_START_STATE_GENERATOR is None:
        game_histories = play.distributed_play(
            _WORKER_PLAYERS,
            number_of_games=number_of_games,
        )
    else:
        game_histories = []
        for _ in range(number_of_games):
            game_histories.extend(
                play.distributed_play(
                    _WORKER_PLAYERS,
                    start_state=_WORKER_START_STATE_GENERATOR(),
                    number_of_games=1,
                )
            )

    return [
        play.game_history_to_game_data(
            game_history,
            terminal_rollouts=_WORKER_OUTCOME_ROLLOUTS,
        )
        for game_history in game_histories
    ]


def _iter_game_batch_sizes(total_games: int, games_per_task: int) -> list[int]:
    batch_sizes = []
    remaining = total_games
    while remaining > 0:
        batch_size = min(games_per_task, remaining)
        batch_sizes.append(batch_size)
        remaining -= batch_size
    return batch_sizes


def run_apply_async_distributed_selfplay_learning(
    process_count: int,
    players: int,
    model_factory: factory.SkyNetModelFactory,
    learn_config: train.LearnConfig,
    training_config: train.TrainConfig,
    predictor_config: predictor.PredictorProcessConfig,
    training_data_buffer_config: buffer.Config,
    model_player_config: player.ModelPlayerConfig,
    games_per_task: int,
    start_state_generator: StartStateGenerator | None = None,
    outcome_rollouts: int = 1,
    debug: bool = False,
    log_level: int = logging.INFO,
    log_dir: pathlib.Path | None = None,
) -> None:
    logging.info("learning config: %s", learn_config)
    logging.info("training config: %s", training_config)
    logging.info("predictor config: %s", predictor_config)
    logging.info("training data buffer config: %s", training_data_buffer_config)
    logging.info("model player config: %s", model_player_config)
    logging.info("games per apply_async task: %s", games_per_task)
    logging.info("Using start position generator: %s", start_state_generator)

    if log_dir is None:
        log_dir = pathlib.Path(
            "logs/apply_async_distributed_train"
            f"/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    predictor_process = None
    pool = None
    completed = False
    try:
        training_data_buffer = buffer.ReplayBuffer(
            **training_data_buffer_config.kwargs()
        )
        model = model_factory.get_latest_model()
        model.set_device(learn_config.torch_device)

        predictor_model_update_queue = mp.Queue()
        predictor_input_queues = {
            i: predictor.PredictorInputQueue(
                queue_id=i,
                max_batch_size=predictor_config.max_batch_size,
            )
            for i in range(process_count)
        }
        predictor_output_queues = {
            i: predictor.PredictorOutputQueue(
                queue_id=i,
                max_batch_size=predictor_config.max_batch_size,
            )
            for i in range(process_count)
        }
        predictor_process = predictor.PredictorProcess(
            model_factory,
            predictor_model_update_queue,
            predictor_input_queues,
            predictor_output_queues,
            debug=debug,
            log_level=log_level,
            log_dir=log_dir,
            **predictor_config.kwargs(),
        )
        predictor_process.start()

        worker_id_queue = mp.Queue()
        for worker_id in range(process_count):
            worker_id_queue.put(worker_id)

        pool = mp.Pool(
            processes=process_count,
            initializer=_setup_worker,
            initargs=(
                worker_id_queue,
                predictor_input_queues,
                predictor_output_queues,
                model_player_config,
                players,
                start_state_generator,
                outcome_rollouts,
                debug,
                log_level,
                log_dir,
                0,
            ),
        )

        for iteration in range(learn_config.learn_steps):
            if (
                learn_config.validation_interval is not None
                and learn_config.validation_function is not None
                and iteration % learn_config.validation_interval == 0
            ):
                logging.info("[LEARN] Validating model")
                learn_config.validation_function(model)

            if (
                learn_config.update_model_interval is not None
                and iteration > 0
                and iteration % learn_config.update_model_interval == 0
            ):
                if learn_config.model_faceoff_function is not None:
                    logging.info("[LEARN] Model faceoff")
                    faceoff_result = learn_config.model_faceoff_function(
                        model,
                        model_factory.get_latest_model(),
                    )
                    if faceoff_result:
                        logging.info("[LEARN] New model faceoff passed")
                    else:
                        logging.info(
                            "[LEARN] Model faceoff failed, reverting to latest saved model"
                        )
                        model = model_factory.get_latest_model()
                        model.set_device(learn_config.torch_device)

                logging.info("[LEARN] Saving model")
                saved_path = model_factory.save_model(model)
                logging.info("[LEARN] Saved model to %s", saved_path)
                logging.info("[LEARN] Updating dedicated predictor process")
                predictor_model_update_queue.put(True)

            logging.info(
                "[LEARN] Generating %s games with pool.apply_async",
                learn_config.games_generated_per_iteration,
            )
            game_stats_list = []
            result_handles = [
                pool.apply_async(_generate_game_batch, (batch_size,))
                for batch_size in _iter_game_batch_sizes(
                    learn_config.games_generated_per_iteration,
                    games_per_task,
                )
            ]
            collected_tasks = 0
            pending_result_handles = list(result_handles)
            while pending_result_handles:
                if predictor_process is not None and not predictor_process.is_alive():
                    raise RuntimeError(
                        "Predictor process exited while game generation was running"
                    )

                ready_handles = [
                    result_handle
                    for result_handle in pending_result_handles
                    if result_handle.ready()
                ]
                if not ready_handles:
                    time.sleep(1.0)
                    continue

                result_handle = ready_handles[0]
                pending_result_handles.remove(result_handle)
                generated_games = result_handle.get()
                for game_data, game_stats in generated_games:
                    training_data_buffer.add_game_data(game_data)
                    game_stats_list.append(game_stats)
                collected_tasks += 1
                logging.info(
                    "[LEARN] Collected task %s/%s; total games collected: %s",
                    collected_tasks,
                    len(result_handles),
                    len(game_stats_list),
                )

            logging.info(
                "[LEARN] Finished generating %s games with %s data points",
                len(game_stats_list),
                sum(game_stats.game_length for game_stats in game_stats_list),
            )
            logging.info(
                "[LEARN] Training data buffer length: %s",
                len(training_data_buffer),
            )
            buffer_saved_path = training_data_buffer.save()
            logging.info("[LEARN] Saved training data buffer to %s", buffer_saved_path)
            logging.info(
                "[LEARN] Generated game stats:\n%s",
                train_utils.game_stats_summary(game_stats_list),
            )

            if len(training_data_buffer) < training_config.batch_size:
                logging.warning(
                    "[LEARN] Skipping training because buffer length %s is smaller "
                    "than batch size %s",
                    len(training_data_buffer),
                    training_config.batch_size,
                )
                continue

            logging.info(
                "[LEARN] Training for %s epochs",
                training_config.epochs,
            )
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

        final_model_path = model_factory.save_model(model)
        logging.info("[LEARN] Saved final model to %s", final_model_path)
        completed = True

    finally:
        if pool is not None:
            if completed:
                pool.close()
            else:
                pool.terminate()
            pool.join()
        if predictor_process is not None:
            predictor_process.cleanup(timeout=5)


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
    log_dir = pathlib.Path("logs/apply_async_distributed_train") / timestamp
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
    else:
        logging.warning(
            "Validation batch path does not exist, skipping validation: %s",
            validation_batch_path,
        )

    models_dir = pathlib.Path("./models") / "apply_async_distributed" / timestamp
    model = skynet.EquivariantSkyNet(
        spatial_input_shape=(
            players,
            sj.ROW_COUNT,
            sj.COLUMN_COUNT,
            sj.FINGER_SIZE,
        ),
        non_spatial_input_shape=(sj.GAME_SIZE,),
        value_output_shape=(players,),
        policy_output_shape=(sj.MASK_SIZE,),
        device=device,
        embedding_dimensions=32,
        global_state_embedding_dimensions=64,
        num_heads=2,
    )
    # model.load_state_dict(
    #     torch.load(
    #         "./models/special/20250811_164319/model_20250818_234645.pth",
    #         weights_only=True,
    #     )
    # )

    model_factory = factory.SkyNetModelFactory(
        model_callable=skynet.EquivariantSkyNet,
        players=players,
        model_kwargs={
            "embedding_dimensions": 32,
            "global_state_embedding_dimensions": 64,
            "num_heads": 2,
        },
        device=device,
        models_dir=models_dir,
        initial_model=model,
    )

    training_config = train.TrainConfig(
        epochs=2,
        batch_size=256,
        learn_rate=1e-3,
        loss_function=lambda model_outputs, targets: train_utils.base_loss(
            model_outputs,
            targets,
            value_scale=1.0,
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
        model_faceoff_function=lambda model, previous_model: model_faceoff_threshold(
            model,
            previous_model,
            500,
            25,
            1.0,
            10,
            0.50,
            start_state_generator,
        ),
        **training_config.kwargs("training"),
    )

    predictor_config = predictor.PredictorProcessConfig(
        min_batch_size=1,
        max_batch_size=512,
        max_wait_seconds=0.1,
        torch_device=device,
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
        spatial_input_shape=(
            players,
            sj.ROW_COUNT,
            sj.COLUMN_COUNT,
            sj.FINGER_SIZE,
        ),
        non_spatial_input_shape=(sj.GAME_SIZE,),
        action_mask_shape=(sj.MASK_SIZE,),
        policy_target_shape=(sj.MASK_SIZE,),
        outcome_target_shape=(players,),
        points_target_shape=(players,),
        cleared_columns_target_shape=(players * sj.COLUMN_COUNT,),
        path=pathlib.Path("./data/training_data") / timestamp / "buffer.pkl",
    )

    run_apply_async_distributed_selfplay_learning(
        process_count=process_count,
        players=players,
        model_factory=model_factory,
        learn_config=learn_config,
        training_config=training_config,
        predictor_config=predictor_config,
        training_data_buffer_config=training_data_buffer_config,
        model_player_config=model_player_config,
        games_per_task=games_per_task,
        start_state_generator=start_state_generator,
        outcome_rollouts=outcome_rollouts,
        debug=debug,
        log_level=logging.DEBUG if debug else logging.INFO,
        log_dir=log_dir,
    )
