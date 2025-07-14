"""
Module to Train Skyjo models
"""

import dataclasses
import itertools
import logging
import pathlib
import time
import typing

import torch
import torch.multiprocessing as mp

import buffer
import config
import explain
import factory
import play
import player
import predictor
import skyjo as sj
import skynet
import train_utils

# MARK: Training


@dataclasses.dataclass(slots=True)
class TrainConfig(config.Config):
    batch_size: int
    epochs: int
    loss_function: typing.Callable[
        [skynet.SkyNetOutput, train_utils.TrainingTargets], torch.Tensor
    ]
    learn_rate: float


def train_step(
    model: skynet.SkyNet,
    batch: train_utils.TrainingBatch,
    loss_function: typing.Callable[
        [skynet.SkyNetOutput, train_utils.TrainingTargets], torch.Tensor
    ],
    learn_rate: float = 1e-4,
) -> None:
    """Performs a single training step on the model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=1e-4)
    model.train()
    (
        spatial_inputs_tensor,
        non_spatial_inputs_tensor,
        masks_tensor,
        outcome_targets_tensor,
        points_targets_tensor,
        policy_targets_tensor,
    ) = skynet.numpy_to_tensors(*batch, device=model.device, dtype=torch.float32)
    model_output = model(spatial_inputs_tensor, non_spatial_inputs_tensor, masks_tensor)
    loss = loss_function(
        model_output,
        (outcome_targets_tensor, points_targets_tensor, policy_targets_tensor),
    )
    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def train_epoch(
    model: skynet.SkyNet,
    training_data_buffer: buffer.ReplayBuffer,
    training_batch_size: int,
    learn_rate: float,
    loss_function: typing.Callable[
        [skynet.SkyNetOutput, train_utils.TrainingTargets], torch.Tensor
    ],
):
    """Performs a single training epoch.

    Runs train() for specified number of batches.

    Args:
        model (skynet.SkyNet): The model to train.
        training_data_buffer (buffer.ReplayBuffer): The training data buffer to sample batches from.
        batch_count (int): The number of batches to train on.
        training_batch_size (int): The size of each batch.
        learn_rate (float): The learning rate to use for training.
        loss_function (typing.Callable): The loss function to use for training.
    """
    training_losses = []
    for _ in range(len(training_data_buffer) // training_batch_size + 1):
        batch = training_data_buffer.sample_batch(batch_size=training_batch_size)
        training_losses.append(train_step(model, batch, loss_function, learn_rate))
    return sum(training_losses) / len(training_losses)


# MARK: Learning Loops


@dataclasses.dataclass(slots=True)
class LearnConfig(config.Config):
    torch_device: torch.device
    learn_steps: int
    games_generated_per_iteration: int
    training_epochs: int
    training_batch_size: int
    training_learn_rate: float
    training_loss_function: typing.Callable[
        [skynet.SkyNetOutput, train_utils.TrainingTargets], torch.Tensor
    ]
    validation_interval: int | None
    validation_function: typing.Callable[[skynet.SkyNet], None] | None
    update_model_interval: int | None
    model_faceoff_function: typing.Callable[[skynet.SkyNet, skynet.SkyNet], bool]


def learn(
    model_factory: factory.SkyNetModelFactory,
    predictor_clients: dict[int, predictor.AbstractPredictorClient],
    training_data_buffer: buffer.ReplayBuffer,
    training_data_queue: mp.Queue,
    torch_device: torch.device,
    learn_steps: int,
    games_generated_per_iteration: int,
    training_epochs: int,
    training_batch_size: int,
    training_learn_rate: float,
    training_loss_function: typing.Callable[
        [skynet.SkyNetOutput, train_utils.TrainingTargets], torch.Tensor
    ],
    validation_interval: int,
    validation_function: typing.Callable[[skynet.SkyNet], None] | None,
    update_model_interval: int,
    model_faceoff_function: typing.Callable[[skynet.SkyNet, skynet.SkyNet], bool]
    | None,
):
    """Core learning loop"""

    logging.info(f"[LEARN] Starting learning loop for {learn_steps} steps")
    logging.info(f"[LEARN] Parameters: {locals()}")

    model = model_factory.get_latest_model()
    model.set_device(torch_device)
    games_count = 0
    previous_games_count = 0
    for iteration in range(learn_steps):
        if iteration % validation_interval == 0 and validation_interval is not None:
            logging.info("[LEARN] Validating model")
            validation_function(model)

        if iteration > 0 and iteration % update_model_interval == 0:
            if model_faceoff_function is not None:
                logging.info("[LEARN] Model Faceoff")
                faceoff_result = model_faceoff_function(
                    model, model_factory.get_latest_model()
                )
                if faceoff_result:
                    logging.info("[LEARN] New Model Faceoff Passed")
                else:
                    logging.info(
                        "[LEARN] Model Faceoff Failed, reverting to previous model"
                    )
                    model = model_factory.get_latest_model()

            logging.info("[LEARN] Saving model")
            saved_path = model_factory.save_model(model)
            logging.info(f"[LEARN] Saved model to {saved_path}")
            logging.info("[LEARN] Updating predictor clients")
            for id, client in predictor_clients.items():
                client.trigger_model_update()

        # Add training data from the queue into the buffer
        logging.info(f"[LEARN] Generating {games_generated_per_iteration} games")
        game_stats_list = []
        while (
            not training_data_queue.empty()
            or len(game_stats_list) < games_generated_per_iteration
        ):
            game_data, game_stats = training_data_queue.get()
            training_data_buffer.add_game_data(game_data)
            game_stats_list.append(game_stats)
        logging.info(
            f"[LEARN] Finished generating {len(game_stats_list) - previous_games_count} games "
            f"with {sum([game_stats.game_length for game_stats in game_stats_list])} data points"
        )
        logging.info(
            f"[LEARN] Training data buffer length: {len(training_data_buffer)}, "
        )
        buffer_saved_path = training_data_buffer.save()
        logging.info(f"Saving training data buffer to {buffer_saved_path}")
        logging.info(
            f"Generated game stats:\n{train_utils.game_stats_summary(game_stats_list)}"
        )
        previous_games_count = len(game_stats_list)
        logging.info(f"[LEARN] Training for {training_epochs} epochs")
        for i in range(training_epochs):
            training_stats = train_epoch(
                model,
                training_data_buffer,
                training_batch_size=training_batch_size,
                learn_rate=training_learn_rate,
                loss_function=training_loss_function,
            )
            logging.info(f"[LEARN] Training Epoch {i} stats: {training_stats}")
        logging.info(
            f"[LEARN] Finished training for {training_epochs} epochs with "
            f"{len(training_data_buffer) // training_batch_size + 1} batches"
        )


# MARK: Full Learning


def run_multiprocessed_selfplay_with_dedicated_predictor_learning(
    process_count: int,
    players: int,
    model_factory: factory.SkyNetModelFactory,
    learn_config: LearnConfig,
    training_config: TrainConfig,
    predictor_config: predictor.PredictorProcessConfig,
    training_data_buffer_config: buffer.Config,
    model_player_config: player.ModelPlayerConfig,
    start_state_generator: typing.Callable[[], sj.Skyjo] | None = None,
    outcome_rollouts: int = 1,
    debug: bool = False,
    log_level: int = logging.INFO,
    log_dir: pathlib.Path | None = None,
):
    """Runs a distributed training loop.

    Runs a loop that:
    - Creates a predictor process and clients
    - Creates selfplay actors
    - Collects training data from the actors
    - Trains the model on the collected data
    - Saves the model periodically
    - Validates the model periodically
    - Terminates the actors and predictor process
    - Joins the actors and predictor process
    """
    logging.info(f"learning config: {learn_config}")
    logging.info(f"training config: {training_config}")
    logging.info(f"predictor config: {predictor_config}")
    logging.info(f"model player config: {model_player_config}")
    logging.info(f"Using start position generator: {start_state_generator}")
    predictor_process, selfplay_actors = None, []
    try:
        training_data_buffer = buffer.ReplayBuffer(
            **training_data_buffer_config.kwargs()
        )

        # Predictor setup
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
        predictor_clients = {
            i: predictor.DistributedPredictorClient(
                predictor_input_queues[i],
                predictor_output_queues[i],
            )
            for i in range(process_count)
        }

        # Selfplay setup
        selfplay_data_queue = mp.Queue()
        logging.info(f"Starting {process_count} selfplay processes")
        selfplay_actors = [
            play.SelfplayGenerator(
                id=f"selfplay_{i}",
                player=player.ModelPlayer(
                    predictor_clients[i], **model_player_config.kwargs()
                ),
                player_count=players,
                game_data_queue=selfplay_data_queue,
                start_state_generator=start_state_generator,
                outcome_rollouts=outcome_rollouts,
                debug=debug,
                log_level=log_level,
                log_dir=log_dir,
                play_callable=play.model_player_selfplay,
            )
            for i in range(process_count)
        ]
        for actor in selfplay_actors:
            actor.start()

        # Main training loop
        learn(
            model_factory=model_factory,
            predictor_clients=predictor_clients,
            training_data_buffer=training_data_buffer,
            training_data_queue=selfplay_data_queue,
            **(learn_config.kwargs() | training_config.kwargs(prefix="training")),
        )

    finally:
        predictor_process.cleanup(timeout=5)
        for actor in selfplay_actors:
            actor.cleanup(timeout=5)


def run_multiprocessed_batched_mcts_selfplay_with_dedicated_predictor_learning(
    process_count: int,
    players: int,
    model_factory: factory.SkyNetModelFactory,
    learn_config: LearnConfig,
    training_config: TrainConfig,
    predictor_config: predictor.PredictorProcessConfig,
    training_data_buffer_config: buffer.Config,
    batched_model_player_config: player.BatchedModelPlayerConfig,
    start_state_generator: typing.Callable[[], sj.Skyjo] | None = None,
    outcome_rollouts: int = 1,
    debug: bool = False,
    log_level: int = logging.INFO,
    log_dir: pathlib.Path | None = None,
):
    """Runs a distributed training loop.

    Runs a loop that:
    - Creates a predictor process and clients
    - Creates selfplay and greedy play actors
    - Collects training data from the actors
    - Trains the model on the collected data
    - Saves the model periodically
    - Validates the model periodically
    - Terminates the actors and predictor process
    - Joins the actors and predictor process
    """
    logging.info(f"learning config: {learn_config}")
    logging.info(f"training config: {training_config}")
    logging.info(f"predictor config: {predictor_config}")
    logging.info(f"training data buffer config: {training_data_buffer_config}")
    logging.info(f"batched model player config: {batched_model_player_config}")
    logging.info(f"Using start position generator: {start_state_generator}")
    predictor_process = None
    selfplay_actors = []
    try:
        # Predictor setup
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
            **predictor_config.kwargs(),
            debug=debug,
            log_level=log_level,
            log_dir=log_dir,
        )
        predictor_process.start()
        predictor_clients = {
            i: predictor.DistributedPredictorClient(
                predictor_input_queues[i], predictor_output_queues[i]
            )
            for i in range(process_count)
        }

        # Selfplay setup
        selfplay_data_queue = mp.Queue()
        logging.info(f"Starting {process_count} selfplay processes")
        selfplay_actors = [
            play.SelfplayGenerator(
                id=f"selfplay_{i}",
                player=player.BatchedModelPlayer(
                    predictor_clients[i], **batched_model_player_config.kwargs()
                ),
                player_count=players,
                game_data_queue=selfplay_data_queue,
                start_state_generator=start_state_generator,
                outcome_rollouts=outcome_rollouts,
                debug=debug,
                log_level=log_level,
                log_dir=log_dir,
                play_callable=play.batched_model_player_selfplay,
            )
            for i in range(process_count)
        ]
        for actor in selfplay_actors:
            actor.start()
        training_data_buffer = buffer.ReplayBuffer(
            **training_data_buffer_config.kwargs()
        )
        learn(
            model_factory=model_factory,
            predictor_clients=predictor_clients,
            training_data_buffer=training_data_buffer,
            training_data_queue=selfplay_data_queue,
            **(learn_config.kwargs() | training_config.kwargs(prefix="training")),
        )

    finally:
        if predictor_process is not None:
            predictor_process.cleanup(timeout=1)
        for actor in selfplay_actors:
            actor.cleanup(timeout=1)


def run_multiprocessed_selfplay_with_local_predictor_learning(
    process_count: int,
    players: int,
    model_factory: factory.SkyNetModelFactory,
    learn_config: LearnConfig,
    training_config: TrainConfig,
    training_data_buffer_config: buffer.Config,
    model_player_config: player.ModelPlayerConfig,
    start_state_generator: typing.Callable[[], sj.Skyjo] | None = None,
    outcome_rollouts: int = 1,
    debug: bool = False,
    log_level: int = logging.INFO,
    log_dir: pathlib.Path | None = None,
):
    """Runs a distributed training loop.

    Runs a loop that:
    - Creates selfplay actors each with local predictor
    - Collects training data from the actors
    - Trains the model on the collected data
    - Saves the model periodically
    - Validates the model periodically
    - Terminates the actors and predictor process
    - Joins the actors and predictor process
    """
    logging.info(f"learning config: {learn_config}")
    logging.info(f"training config: {training_config}")
    logging.info(f"model player config: {model_player_config}")
    logging.info(f"Using start position generator: {start_state_generator}")
    selfplay_actors = []
    try:
        training_data_buffer = buffer.ReplayBuffer(
            **training_data_buffer_config.kwargs()
        )

        # Predictor clients setup
        model_update_queues = {i: mp.Queue() for i in range(process_count)}
        predictor_clients = {
            i: predictor.LocalPredictorClient(
                model=model_factory.get_latest_model(),
                # TODO: make this a parameter (but properly not just add parameter)
                max_batch_size=512,
                factory=model_factory,
                model_update_queue=model_update_queues[i],
            )
            for i in range(process_count)
        }

        # Selfplay setup
        selfplay_data_queue = mp.Queue()
        logging.info(f"Starting {process_count} selfplay processes")
        selfplay_actors = [
            play.SelfplayGenerator(
                id=f"selfplay_{i}",
                player=player.ModelPlayer(
                    predictor_clients[i], **model_player_config.kwargs()
                ),
                player_count=players,
                game_data_queue=selfplay_data_queue,
                start_state_generator=start_state_generator,
                outcome_rollouts=outcome_rollouts,
                debug=debug,
                log_level=log_level,
                log_dir=log_dir,
                play_callable=play.model_player_selfplay,
            )
            for i in range(process_count)
        ]
        for actor in selfplay_actors:
            actor.start()

        # Main training loop
        learn(
            model_factory=model_factory,
            predictor_clients=predictor_clients,
            training_data_buffer=training_data_buffer,
            training_data_queue=selfplay_data_queue,
            **(learn_config.kwargs() | training_config.kwargs(prefix="training")),
        )

    finally:
        for actor in selfplay_actors:
            actor.cleanup(timeout=5)


def run_multiprocessed_batched_mcts_selfplay_with_local_predictor_learning(
    process_count: int,
    players: int,
    model_factory: factory.SkyNetModelFactory,
    learn_config: LearnConfig,
    training_config: TrainConfig,
    training_data_buffer_config: buffer.Config,
    batched_model_player_config: player.BatchedModelPlayerConfig,
    start_state_generator: typing.Callable[[], sj.Skyjo] | None = None,
    outcome_rollouts: int = 1,
    debug: bool = False,
    log_level: int = logging.INFO,
    log_dir: pathlib.Path | None = None,
):
    """Runs a distributed training loop.

    Runs a loop that:
    - Creates selfplay actors each with local predictor
    - Collects training data from the actors
    - Trains the model on the collected data
    - Saves the model periodically
    - Validates the model periodically
    - Terminates the actors and predictor process
    - Joins the actors and predictor process
    """
    logging.info(f"learning config: {learn_config}")
    logging.info(f"training config: {training_config}")
    logging.info(f"batched model player config: {batched_model_player_config}")
    logging.info(f"Using start position generator: {start_state_generator}")
    selfplay_actors = []
    try:
        training_data_buffer = buffer.ReplayBuffer(
            **training_data_buffer_config.kwargs()
        )

        # Predictor clients setup
        model_update_queues = {i: mp.Queue() for i in range(process_count)}
        predictor_clients = {
            i: predictor.LocalPredictorClient(
                model=model_factory.get_latest_model(),
                # TODO: make this a parameter (but properly not just add parameter)
                max_batch_size=512,
                factory=model_factory,
                model_update_queue=model_update_queues[i],
            )
            for i in range(process_count)
        }

        # Selfplay setup
        selfplay_data_queue = mp.Queue()
        logging.info(f"Starting {process_count} selfplay processes")
        selfplay_actors = [
            play.SelfplayGenerator(
                id=f"selfplay_{i}",
                player=player.BatchedModelPlayer(
                    predictor_clients[i], **batched_model_player_config.kwargs()
                ),
                player_count=players,
                game_data_queue=selfplay_data_queue,
                start_state_generator=start_state_generator,
                outcome_rollouts=outcome_rollouts,
                debug=debug,
                log_level=log_level,
                log_dir=log_dir,
                play_callable=play.batched_model_player_selfplay,
            )
            for i in range(process_count)
        ]
        for actor in selfplay_actors:
            actor.start()

        # Main training loop
        learn(
            model_factory=model_factory,
            predictor_clients=predictor_clients,
            training_data_buffer=training_data_buffer,
            training_data_queue=selfplay_data_queue,
            **(learn_config.kwargs() | training_config.kwargs(prefix="training")),
        )

    finally:
        for actor in selfplay_actors:
            actor.cleanup(timeout=5)


def run_single_process_learning(
    model: skynet.SkyNet,
    models_dir: pathlib.Path,
    learn_config: LearnConfig,
    training_config: TrainConfig,
    model_player_config: player.ModelPlayerConfig,
    buffer_config: buffer.Config,
    start_position_generator: typing.Callable[[], sj.Skyjo] | None = None,
    debug: bool = False,
):
    """Single-processed training loop.

    Runs a loop that:
    1. Generates selfplay games
    2. Adds the games to the training data buffer
    3. Trains the model on the collected data
    4. Saves and validates the model periodically
    """
    logging.info(f"learning config: {learn_config}")
    logging.info(f"training config: {training_config}")
    logging.info(f"training data buffer config: {buffer_config}")
    logging.info(f"model player config: {model_player_config}")
    logging.info(f"Using start position generator: {start_position_generator}")
    training_data_buffer = buffer.ReplayBuffer.from_config(buffer_config)
    start_time = time.time()
    predictor_client = predictor.NaivePredictorClient(model, max_batch_size=512)

    if learn_config.validation_function is not None:
        learn_config.validation_function(model)

    for learn_step in range(learn_config.iterations):
        selfplay_games_data = []
        game_generation_start_time = time.time()

        for game_count in range(learn_config.minimum_games_per_iteration):
            selfplay_games_data.append(
                play.model_selfplay(
                    predictor_client,
                    model_player_config.players,
                    model_player_config.mcts_config,
                    model_player_config.action_softmax_temperature,
                    model_player_config.outcome_rollouts,
                    start_position=None
                    if start_position_generator is None
                    else start_position_generator(),
                    debug=debug,
                )
            )
            if game_count % 1 == 0:
                logging.info(f"Generated {game_count} games")
        logging.info(
            f"Generated {len(selfplay_games_data)} games in {time.time() - game_generation_start_time} seconds"
        )
        for game_data in selfplay_games_data:
            training_data_buffer.add_game_data(game_data)
        training_start_time = time.time()
        training_losses = []
        for _ in range(learn_config.training_epochs):
            loss = train_epoch(
                model,
                training_data_buffer,
                training_batch_size=training_config.batch_size,
                learn_rate=training_config.learn_rate,
                loss_function=training_config.loss_function,
            )
            training_losses.append(loss)
        training_time = time.time() - training_start_time
        logging.info(
            f"Trained for {learn_config.training_epochs} epochs with "
            f"{len(training_data_buffer) // training_config.training_batch_size + 1} batches in {training_time} seconds"
        )

        if (
            learn_step % learn_config.validation_interval == 0
            and learn_config.validation_function is not None
        ):
            learn_config.validation_function(model)

        if learn_step > 0 and learn_step % learn_config.update_model_interval == 0:
            logging.info(
                f"Training data buffer length: {len(training_data_buffer)}, "
                f"total buffer size: {len(training_data_buffer)}"
            )
            logging.info(
                f"Ran {learn_config.update_model_interval} train steps in {time.time() - start_time} seconds"
            )
            saved_path = model.save(models_dir)
            logging.info(f"Saved model to {saved_path}")


def main_train_on_greedy_ev_player_games(
    model: skynet.SkyNet,
    models_dir: pathlib.Path,
    validation_batch: train_utils.TrainingBatch | None = None,
    learn_steps: int = 100,
    training_epochs: int = 1,
    batch_count: int = 10_000,
    buffer_max_size: int = 10_000_000,
    games_per_step: int = 100_000,
    training_batch_size: int = 512,
):
    """Trains the model on greedy ev player games.

    Runs a loop that:
    1. Generates games from greedy ev heuristic players
    2. Adds the games to the training data buffer
    3. Trains the model on the collected data
    4. Saves and validates the model periodically
    """
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
            explain.validate_model(model, validation_batch)
            training_losses = []
            for batch_iter in range(batch_count):
                batch = training_data_buffer.sample_batch(
                    batch_size=training_batch_size
                )
                loss = train_step(
                    model,
                    batch,
                    loss_function=train_utils.base_policy_value_loss,
                    learn_rate=1e-3,
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
    training_sample: train_utils.TrainingBatch,
    learn_steps: int = 100,
    training_epochs: int = 1,
    batch_count: int = 1000,
    buffer_max_size: int = 10_000_000,
    training_batch_size: int = 512,
):
    """Trains the model on a small fixed training sample.

    Mainly used for debugging to determine whether a model is capable of
    overfitting a small training sample.
    """
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
                loss = train_step(
                    model,
                    batch,
                    loss_function=train_utils.base_policy_value_loss,
                    learn_rate=1e-3,
                )
                training_losses.append(loss)
            logging.info(
                f"Mean training loss: {sum(training_losses) / len(training_losses)}"
            )
        model_path = model.save(models_dir)
        logging.info(f"Saved model to {model_path}")
