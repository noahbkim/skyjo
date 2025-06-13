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
import explain
import model_factory
import parallel_mcts
import play
import predictor
import skynet
import train_utils

# MARK: Configs


@dataclasses.dataclass(slots=True)
class TrainingEpochConfig:
    training_batch_size: int
    learning_rate: float
    loss_function: typing.Callable[
        [skynet.SkyNetOutput, train_utils.TrainingTargets], torch.Tensor
    ]


@dataclasses.dataclass(slots=True)
class BaseLearnConfig:
    iterations: int
    training_epochs: int
    training_epoch_config: TrainingEpochConfig
    validation_function: typing.Callable[[skynet.SkyNet], None]
    validation_interval: int | None
    update_model_interval: int | None


@dataclasses.dataclass(slots=True)
class MultiProcessedLearnConfig(BaseLearnConfig):
    selfplay_processes: int
    minimum_games_per_iteration: int
    torch_device: torch.device


# MARK: Training


def train(
    model: skynet.SkyNet,
    batch: train_utils.TrainingBatch,
    loss_function: typing.Callable[
        [skynet.SkyNetOutput, train_utils.TrainingTargets], torch.Tensor
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
        masks,
        outcome_targets,
        points_targets,
        policy_targets,
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

    outcome_targets_tensor = torch.tensor(
        outcome_targets, dtype=torch.float32, device=model.device
    )
    points_targets_tensor = torch.tensor(
        points_targets, dtype=torch.float32, device=model.device
    )
    policy_targets_tensor = torch.tensor(
        policy_targets, dtype=torch.float32, device=model.device
    )
    model_output = model(spatial_inputs, non_spatial_inputs, masks)
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
    learning_rate: float,
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
        learning_rate (float): The learning rate to use for training.
        loss_function (typing.Callable): The loss function to use for training.
    """
    training_losses = []
    for _ in range(len(training_data_buffer) // training_batch_size + 1):
        batch = training_data_buffer.sample_batch(batch_size=training_batch_size)
        training_losses.append(train(model, batch, loss_function, learning_rate))
    return sum(training_losses) / len(training_losses)


def train_epoch_with_config(
    model: skynet.SkyNet,
    training_data_buffer: buffer.ReplayBuffer,
    config: TrainingEpochConfig,
):
    return train_epoch(
        model,
        training_data_buffer,
        config.training_batch_size,
        config.learning_rate,
        config.loss_function,
    )


# MARK: Learning Loops


def multiprocessed_learn(
    factory: model_factory.SkyNetModelFactory,
    learn_config: MultiProcessedLearnConfig,
    training_config: TrainingEpochConfig,
    predictor_config: predictor.Config,
    training_data_buffer_config: buffer.Config,
    selfplay_config: play.Config,
    debug: bool = False,
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

    try:
        # Predictor setup
        predictor_model_update_queue = mp.Queue()
        predictor_input_queues = {
            i: predictor.PredictorInputQueue.from_config(
                queue_id=i,
                config=predictor_config,
            )
            for i in range(learn_config.selfplay_processes)
        }
        predictor_output_queues = {
            i: predictor.PredictorOutputQueue.from_config(
                queue_id=i,
                config=predictor_config,
            )
            for i in range(learn_config.selfplay_processes)
        }
        predictor_process = predictor.PredictorProcess.from_config(
            factory,
            predictor_model_update_queue,
            predictor_input_queues,
            predictor_output_queues,
            config=predictor_config,
            debug=debug,
        )
        predictor_process.start()
        predictor_clients = {
            i: predictor.MultiProcessPredictorClient(
                predictor_input_queues[i], predictor_output_queues[i]
            )
            for i in range(learn_config.selfplay_processes)
        }

        # Selfplay setup
        selfplay_data_queue = mp.Queue()
        logging.info(f"Starting {learn_config.selfplay_processes} selfplay processes")
        selfplay_actors = [
            play.MultiProcessedSelfplayGenerator.from_config(
                predictor_clients[i],
                selfplay_data_queue,
                id=i,
                config=selfplay_config,
                debug=debug,
            )
            for i in range(learn_config.selfplay_processes)
        ]
        for actor in selfplay_actors:
            actor.start()

        # Main training loop
        training_data_buffer = buffer.ReplayBuffer.from_config(
            training_data_buffer_config
        )
        logging.info(f"Training for {learn_config.iterations} iterations")
        model = factory.get_latest_model()
        model.set_device(learn_config.torch_device)
        start_time = time.time()
        games_count = 0
        last_games_count = 0
        for iteration in range(learn_config.iterations):
            if (
                iteration % learn_config.validation_interval == 0
                and learn_config.validation_function is not None
            ):
                learn_config.validation_function(model)

            if iteration > 0 and iteration % learn_config.update_model_interval == 0:
                logging.info(
                    f"Training data buffer length: {len(training_data_buffer)}, "
                    f"total buffer size: {len(training_data_buffer)}"
                )
                logging.info(
                    f"Ran {learn_config.update_model_interval} train steps "
                    f"in {time.time() - start_time} seconds"
                )
                saved_path = factory.save_model(model)
                logging.info(f"Saved model to {saved_path}")
                predictor_model_update_queue.put(f"new_model {saved_path}")
                start_time = time.time()

            # Add training data from the queue into the buffer
            game_generation_start_time = time.time()
            while (
                not selfplay_data_queue.empty()
                or games_count - last_games_count
                <= learn_config.minimum_games_per_iteration
            ):
                game_data = selfplay_data_queue.get()
                training_data_buffer.add_game_data(game_data)
                games_count += 1

            game_generation_time = time.time() - game_generation_start_time
            logging.info(
                f"Generated {games_count - last_games_count} games "
                f"in {game_generation_time} seconds"
            )
            last_games_count = games_count

            training_start_time = time.time()
            for i in range(learn_config.training_epochs):
                mean_training_loss = train_epoch_with_config(
                    model,
                    training_data_buffer,
                    config=training_config,
                )
                logging.info(f"Epoch {i} mean training loss: {mean_training_loss}")
            training_time = time.time() - training_start_time
            logging.info(
                f"Trained for {learn_config.training_epochs} epochs with "
                f"{len(training_data_buffer) // training_config.training_batch_size + 1} batches in {training_time} seconds"
            )

            if learn_config.validation_function is not None:
                learn_config.validation_function(model)
    finally:
        predictor_process.terminate()
        for actor in selfplay_actors:
            actor.terminate()
        for actor in selfplay_actors:
            actor.join()
        predictor_process.join()


def learn(
    model: skynet.SkyNet,
    models_dir: pathlib.Path,
    learn_config: BaseLearnConfig,
    training_config: TrainingEpochConfig,
    mcts_config: parallel_mcts.Config,
    buffer_config: buffer.Config,
):
    """Single-processed training loop.

    Runs a loop that:
    1. Generates selfplay games
    2. Adds the games to the training data buffer
    3. Trains the model on the collected data
    4. Saves and validates the model periodically
    """
    training_data_buffer = buffer.ReplayBuffer.from_config(buffer_config)
    start_time = time.time()

    for learn_step in range(learn_config.iterations):
        selfplay_games_data = []
        for _ in range(learn_config.minimum_games_per_iteration):
            selfplay_games_data.append(play.selfplay_with_config(model, mcts_config))
        for game_data in selfplay_games_data:
            training_data_buffer.add_game_data(game_data)

        training_losses = []
        for _ in range(learn_config.training_epochs):
            loss = train_epoch_with_config(
                model,
                training_data_buffer,
                config=training_config,
            )
            training_losses.append(loss)

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
                loss = train(
                    model,
                    batch,
                    loss_function=train_utils.base_policy_value_loss,
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
                loss = train(
                    model,
                    batch,
                    loss_function=train_utils.base_policy_value_loss,
                    learning_rate=1e-3,
                )
                training_losses.append(loss)
            logging.info(
                f"Mean training loss: {sum(training_losses) / len(training_losses)}"
            )
        model_path = model.save(models_dir)
        logging.info(f"Saved model to {model_path}")
