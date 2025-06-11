import itertools
import logging
import pathlib
import random
import time
import typing

import numpy as np
import torch
import torch.multiprocessing as mp

import buffer
import explain
import model_factory
import play
import predictor
import skyjo as sj
import skynet


def constant_basic_learning_rate(train_iter: int) -> float:
    return 1e-3


def constant_basic_selfplay_params(learn_iter: int) -> dict[str, typing.Any]:
    return {
        "players": 2,
        "mcts_iterations": 1600,
        "mcts_temperature": 1.0,
        "afterstate_realizations": False,
        "virtual_loss": 0.5,
        "max_parallel_evaluations": 16,
        "terminal_rollouts": 100,
        "dirichlet_epsilon": 0.4,
    }


def train(
    model: skynet.SkyNet,
    batch: skynet.TrainingBatch,
    loss_function: typing.Callable[
        [
            skynet.SkyNetOutput,
            np.ndarray[tuple[int, int], np.float32],
            np.ndarray[tuple[int, int], np.float32],
            np.ndarray[tuple[int, int], np.float32],
        ],
        torch.Tensor,
    ],
    learning_rate: float = 1e-4,
) -> None:
    """Performs a single training step on the model."""
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    model.train()
    (
        spatial_inputs,
        non_spatial_inputs,
        policy_targets,
        outcome_targets,
        points_targets,
        masks,
    ) = batch
    spatial_inputs = torch.tensor(
        spatial_inputs,
        dtype=torch.float32,
        device=model.device,
    )
    non_spatial_inputs = torch.tensor(
        non_spatial_inputs,
        dtype=torch.float32,
        device=model.device,
    )
    masks = torch.tensor(masks, dtype=torch.float32, device=model.device)
    model_output = model(spatial_inputs, non_spatial_inputs, masks)

    loss = loss_function(model_output, outcome_targets, points_targets, policy_targets)
    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def train_epoch(
    model: skynet.SkyNet,
    training_data_buffer: buffer.ReplayBuffer,
    batch_count: int,
    training_batch_size: int,
    learning_rate: float,
    loss_function: typing.Callable[
        [skynet.SkyNetPrediction, skynet.TrainingBatch], torch.Tensor
    ],
):
    training_losses = []
    for _ in range(batch_count):
        batch = training_data_buffer.sample_batch(batch_size=training_batch_size)
        training_losses.append(train(model, batch, loss_function, learning_rate))
    logging.info(
        f"Mean batch training loss: {sum(training_losses) / len(training_losses)}"
    )


def multiprocessed_learn(
    factory: model_factory.SkyNetModelFactory,
    device: torch.device,
    learn_steps: int,
    training_data_buffer_max_size: int = 10_000_000,
    predictor_batch_size: int = 1024,
    selfplay_processes: int = 1,
    greedy_play_processes: int = 0,
    training_batch_size: int = 512,
    min_games_before_training: int = 250,
    validation_function: typing.Callable[[skynet.SkyNet], None] = None,
    validation_step_interval: int = 10,
    update_best_model_step_interval: int = 10,
    loss_function: typing.Callable[
        [skynet.SkyNetPrediction, skynet.TrainingBatch], torch.Tensor
    ] = skynet.base_total_loss,
    debug: bool = False,
):
    logging.info(f"""multiprocessed_learn(
    factory={factory},
    device={device},
    learn_steps={learn_steps},
    training_data_buffer_max_size={training_data_buffer_max_size},
    predictor_batch_size={predictor_batch_size},
    selfplay_processes={selfplay_processes},
    greedy_play_processes={greedy_play_processes},
    training_batch_size={training_batch_size},
    min_games_before_training={min_games_before_training},
    validation_step_interval={validation_step_interval},
    update_best_model_step_interval={update_best_model_step_interval},
    debug={debug},
)""")
    try:
        predictor_model_update_queue = mp.Queue()
        predictor_input_queues = {
            i: predictor.PredictorInputQueue(
                queue_id=i,
                max_batch_size=predictor_batch_size,
            )
            for i in range(selfplay_processes + greedy_play_processes)
        }
        predictor_output_queues = {
            i: predictor.PredictorOutputQueue(
                queue_id=i,
                max_batch_size=predictor_batch_size,
            )
            for i in range(selfplay_processes + greedy_play_processes)
        }
        predictor_process = predictor.PredictorProcess(
            factory,
            predictor_model_update_queue,
            predictor_input_queues,
            predictor_output_queues,
            min_batch_size=16,
            max_batch_size=predictor_batch_size,
            device=device,
            debug=debug,
        )
        predictor_process.start()
        predictor_clients = {
            i: predictor.MultiProcessPredictorClient(
                predictor_input_queues[i], predictor_output_queues[i]
            )
            for i in range(selfplay_processes + greedy_play_processes)
        }
        selfplay_data_queue = mp.Queue()
        logging.info(f"Starting {selfplay_processes} selfplay processes")
        selfplay_actors = [
            play.MultiProcessedSelfplayGenerator(
                predictor_clients[i],
                selfplay_data_queue,
                id=i,
                debug=debug,
                **constant_basic_selfplay_params(0),
            )
            for i in range(selfplay_processes)
        ]
        for actor in selfplay_actors:
            actor.start()
        greedy_play_actors = [
            play.MultiProcessedPlayGreedyPlayersGenerator(
                predictor_clients[i + selfplay_processes],
                selfplay_data_queue,
                id=i + selfplay_processes,
                debug=debug,
                episodes=500,
                **constant_basic_selfplay_params(0),
            )
            for i in range(greedy_play_processes)
        ]
        for actor in greedy_play_actors:
            actor.start()

        training_data_buffer = buffer.ReplayBuffer(
            max_size=training_data_buffer_max_size
        )
        logging.info(f"Training for {learn_steps} steps")
        model = factory.get_latest_model()
        model.set_device(device)
        start_time = time.time()
        games_count = 0
        last_games_count = 0
        for learn_step in range(learn_steps):
            if (
                learn_step % validation_step_interval == 0
                and validation_function is not None
            ):
                validation_function(model)

            if learn_step > 0 and learn_step % update_best_model_step_interval == 0:
                logging.info(
                    f"Training data buffer length: {len(training_data_buffer)}, total buffer size: {len(training_data_buffer)}"
                )
                logging.info(
                    f"Ran {update_best_model_step_interval} train steps in {time.time() - start_time} seconds"
                )
                saved_path = factory.save_model(model)
                logging.info(f"Saved model to {saved_path}")
                predictor_model_update_queue.put(f"new_model {saved_path}")
                start_time = time.time()

            # Add training data from the queue into the buffer
            while (
                not selfplay_data_queue.empty()
                or games_count - last_games_count <= min_games_before_training
            ):
                game_data = selfplay_data_queue.get()
                training_data_buffer.add_game_data(game_data)
                games_count += 1
            if learn_step == 0:
                logging.info("Enough selfplay data collected, starting training")
            last_games_count = games_count

            train_epoch(
                model,
                training_data_buffer,
                batch_count=len(training_data_buffer) // training_batch_size,
                training_batch_size=training_batch_size,
                learning_rate=constant_basic_learning_rate(learn_step),
                loss_function=loss_function,
            )

    finally:
        predictor_process.terminate()
        for actor in selfplay_actors:
            actor.terminate()
        for actor in greedy_play_actors:
            actor.terminate()
        for actor in selfplay_actors:
            actor.join()
        for actor in greedy_play_actors:
            actor.join()
        predictor_process.join()


def learn(
    model: skynet.SkyNet,
    players: int,
    learn_steps: int = 100,
    training_epochs: int = 2,
    training_episodes: int = 64,
    training_batch_size: int = 256,
    selfplay_kwargs: typing.Callable[[int], dict[str, typing.Any]]
    | dict[str, typing.Any] = {},
    training_data_buffer_max_games: int = 250,
    validation_function: typing.Callable[[skynet.SkyNet], None] = None,
    validation_step_interval: int = 10,
    update_best_model_step_interval: int = 10,
):
    training_data_buffer = buffer.ReplayBuffer(max_games=training_data_buffer_max_games)
    start_time = time.time()

    for learn_step in range(learn_steps):
        if callable(selfplay_kwargs):
            selfplay_kwargs = selfplay_kwargs(learn_step)
        selfplay_games_data = []
        for _ in range(training_episodes):
            selfplay_games_data.append(play.selfplay(model, players, **selfplay_kwargs))
        for game_data in selfplay_games_data:
            training_data_buffer.add_game_data_with_symmetry(game_data)

        training_losses = []
        for _ in range(training_epochs):
            loss = train_epoch(
                model,
                training_data_buffer,
                batch_count=len(training_data_buffer) // training_batch_size,
                training_batch_size=training_batch_size,
                learning_rate=constant_basic_learning_rate(learn_step),
                loss_function=skynet.base_total_loss,
            )
            training_losses.append(loss)

        if (
            learn_step % validation_step_interval == 0
            and validation_function is not None
        ):
            validation_function(model)

        if learn_step > 0 and learn_step % update_best_model_step_interval == 0:
            logging.info(
                f"Training data buffer length: {len(training_data_buffer)}, total buffer size: {len(training_data_buffer)}"
            )
            logging.info(
                f"Ran {update_best_model_step_interval} train steps in {time.time() - start_time} seconds"
            )
            saved_path = model.save(models_dir)
            logging.info(f"Saved model to {saved_path}")


def main_train_on_greedy_ev_player_games(
    model: skynet.SkyNet,
    models_dir: pathlib.Path,
    validation_games_data: list[play.GameData] | None = None,
    learn_steps: int = 100,
    training_epochs: int = 1,
    batch_count: int = 10_000,
    buffer_max_size: int = 10_000_000,
    games_per_step: int = 100_000,
    training_batch_size: int = 512,
):
    logging.info(
        f"Training model on greedy ev player games with "
        f"models_dir={models_dir}, "
        f"validation_games_data={validation_games_data}, "
        f"learn_steps={learn_steps}, "
        f"training_epochs={training_epochs}, "
        f"batch_count={batch_count}, "
        f"games_per_step={games_per_step}, "
        f"training_batch_size={training_batch_size}"
    )
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = model.save(models_dir)
    logging.info(f"Saved model to {model_path}")

    training_data_buffer = buffer.ReplayBuffer(max_size=buffer_max_size)
    logging.info("Starting Training")
    for _ in range(learn_steps):
        for _ in range(games_per_step):
            game_data = play.multiprocessed_play_greedy_players()
            training_data_buffer.add_game_data(game_data)

        logging.info(f"Added {games_per_step} games to the buffer")
        logging.info(f"Training data buffer size: {len(training_data_buffer)}")
        for learn_iter in range(training_epochs):
            explain.validate_model(model, validation_games_data)
            training_losses = []
            for batch_iter in range(batch_count):
                batch = training_data_buffer.sample_batch(
                    batch_size=training_batch_size
                )
                loss = train(
                    model,
                    batch,
                    loss_function=skynet.base_total_loss,
                    learning_rate=1e-3,
                )
                training_losses.append(loss)
            logging.info(
                f"Mean training loss: {sum(training_losses) / len(training_losses)}"
            )
        model_path = model.save(models_dir)
        logging.info(f"Saved model to {model_path}")


def main_overfit_small_training_sample(
    model: skynet.SkyNet,
    models_dir: pathlib.Path,
    training_sample: list[list[play.GameData]],
    learn_steps: int = 100,
    training_epochs: int = 1,
    batch_count: int = 1000,
    buffer_max_size: int = 10_000_000,
    training_batch_size: int = 512,
):
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = model.save(models_dir)
    logging.info(f"Saved model to {model_path}")
    training_data_buffer = buffer.ReplayBuffer(max_size=buffer_max_size)
    logging.info("Adding fixed training data sample to buffer")
    for game_data in training_sample:
        training_data_buffer.add_game_data(game_data)
    validation_games_data = list(itertools.chain.from_iterable(training_sample))
    logging.info("Starting Training")
    for _ in range(learn_steps):
        logging.info(f"Training data buffer size: {len(training_data_buffer)}")
        for learn_iter in range(training_epochs):
            explain.validate_model(model, validation_games_data)
            training_losses = []
            for batch_iter in range(batch_count):
                batch = training_data_buffer.sample_batch(
                    batch_size=training_batch_size
                )
                loss = train(
                    model,
                    batch,
                    loss_function=skynet.base_total_loss,
                    learning_rate=1e-3,
                )
                training_losses.append(loss)
            logging.info(
                f"Mean training loss: {sum(training_losses) / len(training_losses)}"
            )
        model_path = model.save(models_dir)
        logging.info(f"Saved model to {model_path}")


if __name__ == "__main__":
    import datetime
    import pickle as pkl

    import skyjo as sj

    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    debug = False
    log_dir = pathlib.Path(
        f"logs/multiprocessed_train/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}/"
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=log_dir / "main.log",
        filemode="w",
    )

    with open("./data/validation/greedy_ev_validation_games_data.pkl", "rb") as f:
        validation_games_data = pkl.load(f)
    with open("./data/validation/greedy_ev_fixed_training_games_data.pkl", "rb") as f:
        small_fixed_training_sample = pkl.load(f)

    models_dir = (
        pathlib.Path("./models")
        / "distributed"
        / f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    device = torch.device("cpu")
    model = skynet.EquivariantSkyNet(
        spatial_input_shape=(2, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
        non_spatial_input_shape=(sj.GAME_SIZE,),
        value_output_shape=(2,),
        policy_output_shape=(sj.MASK_SIZE,),
        device=device,
        card_embedding_dimensions=8,
        column_embedding_dimensions=16,
        board_embedding_dimensions=32,
        global_state_embedding_dimensions=64,
        non_spatial_embedding_dimensions=16,
    )
    # model.load_state_dict(
    #     torch.load(
    #         "./models/distributed/20250608_212522/model_20250609_023411.pth",
    #         weights_only=True,
    #     )
    # )
    # model = skynet.SimpleSkyNet(
    #     [256, 256, 256],
    #     spatial_input_shape=(2, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
    #     non_spatial_input_shape=(sj.GAME_SIZE,),
    #     value_output_shape=(2,),
    #     policy_output_shape=(sj.MASK_SIZE,),
    # )

    # main_train_on_greedy_ev_player_games(model, models_dir, validation_games_data)
    # main_overfit_small_training_sample(model, models_dir, small_fixed_training_sample)

    factory = model_factory.SkyNetModelFactory(
        model_callable=skynet.EquivariantSkyNet,
        players=2,
        model_kwargs={
            "card_embedding_dimensions": 8,
            "column_embedding_dimensions": 16,
            "board_embedding_dimensions": 32,
            "global_state_embedding_dimensions": 64,
            "non_spatial_embedding_dimensions": 16,
        },
        device=device,
        models_dir=models_dir,
        initial_model=model,
    )
    multiprocessed_learn(
        factory,
        device,
        learn_steps=1000,
        selfplay_processes=9,
        greedy_play_processes=0,
        predictor_batch_size=1024,
        validation_function=lambda model: explain.validate_model(
            model, validation_games_data
        ),
        training_data_buffer_max_size=10_000_000,
        update_best_model_step_interval=1,
        min_games_before_training=500,
        validation_step_interval=1,
        training_batch_size=128,
    )
