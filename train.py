import abc
import concurrent.futures
import itertools
import logging
import multiprocessing as mp
import pathlib
import random
import re
import typing

import numpy as np
import torch

import buffer
import mcts_new as mcts
import skyjo as sj
import skynet


class SkyNetModelFactory:
    def __init__(
        self,
        players: int = 2,
        dropout_rate: float = 0.5,
        device: torch.device = torch.device("cpu"),
        models_dir: pathlib.Path = pathlib.Path("models"),
        model_kwargs: dict[str, typing.Any] = {},
    ):
        self.models_dir = models_dir
        self.players = players
        self.dropout_rate = dropout_rate
        self.device = device
        self.model_kwargs = model_kwargs
        self.training_model = skynet.SkyNet1D(
            spatial_input_shape=(
                players,
                sj.ROW_COUNT,
                sj.COLUMN_COUNT,
                sj.FINGER_SIZE,
            ),
            non_spatial_input_shape=(sj.GAME_SIZE,),
            value_output_shape=(players,),
            policy_output_shape=(sj.MASK_SIZE,),
            dropout_rate=self.dropout_rate,
            device=self.device,
            **self.model_kwargs,
        )
        # Save initial model
        self.save_model(self.training_model)

    def _get_latest_model_path(self) -> pathlib.Path:
        """Finds the model file with the latest timestamp in the filename."""
        model_files = list(self.models_dir.glob("model_*.pth"))
        if not model_files:
            # If no models saved yet, return path for the initial model
            # Assuming the initial model is saved with a specific name or timestamp 0
            # For simplicity, let's assume save_model ensures one exists or handles it.
            # Re-evaluating: It's better to raise an error if called before first save.
            raise FileNotFoundError(f"No model files found in {self.models_dir}")

        # Regex to extract the timestamp YYYYMMDD_HHMMSS
        pattern = re.compile(r"model_(\d{8}_\d{6})\.pth")

        latest_file = None
        latest_timestamp_str = ""

        for file_path in model_files:
            match = pattern.match(file_path.name)
            if match:
                timestamp_str = match.group(1)
                # String comparison works for YYYYMMDD_HHMMSS format
                if timestamp_str > latest_timestamp_str:
                    latest_timestamp_str = timestamp_str
                    latest_file = file_path

        if latest_file is None:
            # This case should ideally not happen if files exist and saving adheres to format
            raise FileNotFoundError(
                f"No model files matching the pattern 'model_YYYYMMDD_HHMMSS.pth' found in {self.models_dir}"
            )

        return latest_file

    def save_model(self, model: skynet.SkyNet) -> pathlib.Path:
        return model.save(self.models_dir)

    def get_latest_model(self) -> skynet.SkyNet:
        latest_model_path = self._get_latest_model_path()
        model = skynet.SkyNet1D(
            spatial_input_shape=(
                self.players,
                sj.ROW_COUNT,
                sj.COLUMN_COUNT,
                sj.FINGER_SIZE,
            ),
            non_spatial_input_shape=(sj.GAME_SIZE,),
            value_output_shape=(self.players,),
            policy_output_shape=(sj.MASK_SIZE,),
            dropout_rate=self.dropout_rate,
            device=self.device,
            **self.model_kwargs,
        )
        model.load_state_dict(torch.load(latest_model_path, weights_only=True))
        model.to(self.device)
        return model


def constant_basic_learning_rate(train_iter: int) -> float:
    return 1e-4


def constant_basic_selfplay_params(learn_iter: int) -> dict[str, typing.Any]:
    return {
        "players": 2,
        "mcts_iterations": 100,
        "mcts_temperature": 1.0,
        "afterstate_initial_realizations": 50,
    }


class Launcher(abc.ABC):
    """Abstract base class for launching jobs to run a callable with args and kwargs.

    Specific implementations, might spawn a new process, thread, or submit a job to a distributed compute cluster."""

    def __init__(
        self,
        callable: typing.Callable[[], None],
        args: typing.Tuple[typing.Any, ...],
        kwargs: typing.Dict[str, typing.Any],
    ):
        self.callable = callable
        self.args = args
        self.kwargs = kwargs

    @abc.abstractmethod
    def launch(self):
        """Launches the job to run callable with args and kwargs."""
        pass


def get_skyjo_symmetries(
    skyjo: sj.Skyjo, policy_probs: np.ndarray[tuple[int], np.uint8]
) -> list[tuple[sj.Skyjo, np.ndarray[tuple[int], np.uint8]]]:
    """Return a list of symmetrically equivalent `Skyjo` states and corresponding policy."""
    symmetries = []
    for col_order in itertools.permutations(range(sj.COLUMN_COUNT)):
        new_table = skyjo[1].copy()
        new_table[0] = new_table[0][:, col_order, :]
        new_policy_probs = policy_probs.copy()
        flip_policy = new_policy_probs[sj.MASK_FLIP : sj.MASK_FLIP + sj.FINGER_COUNT]
        replace_policy = new_policy_probs[
            sj.MASK_REPLACE : sj.MASK_REPLACE + sj.FINGER_COUNT
        ]
        new_policy_probs[sj.MASK_FLIP : sj.MASK_FLIP + sj.FINGER_COUNT] = (
            flip_policy.reshape(sj.ROW_COUNT, sj.COLUMN_COUNT)[:, col_order].reshape(-1)
        )
        new_policy_probs[sj.MASK_REPLACE : sj.MASK_REPLACE + sj.FINGER_COUNT] = (
            replace_policy.reshape(sj.ROW_COUNT, sj.COLUMN_COUNT)[:, col_order].reshape(
                -1
            )
        )
        symmetries.append(
            (
                (
                    skyjo[0],
                    new_table,
                    skyjo[2],
                    skyjo[3],
                    skyjo[4],
                    skyjo[5],
                    skyjo[6],
                    skyjo[7],
                ),
                new_policy_probs,
            )
        )
    return symmetries


def training_data_from_game_data(
    game_data: list[tuple[sj.Skyjo, np.ndarray[tuple[int], np.uint8]]],
    outcome_state_value: np.ndarray[tuple[int], np.float32],
    fixed_perspective_score: np.ndarray[tuple[int], np.float32],
) -> list[skynet.TrainingDataPoint]:
    """Converts game data to training data.

    Game data is a tuple of skyjo state and mcts action probabilities. Converts this
    to a list of training data points which contain the current state, and targets
    for the network to train on.

    Args:
        game_data: A list of tuples containing a skyjo state and mcts action probabilities.
        outcome_state_value: The outcome state value.
        fixed_perspective_score: The fixed perspective score.

    Returns:
        A list of training data points.
    """
    training_data = []
    for game_state, mcts_probs in game_data:
        training_data.append(
            [
                (
                    game_state,  # game
                    mcts_probs,  # policy target
                    np.roll(
                        outcome_state_value, sj.get_player(game_state)
                    ),  # outcome target
                    np.roll(
                        fixed_perspective_score,
                        sj.get_player(game_state),
                    ),  # points target
                )
            ]
        )
    return training_data


def selfplay(
    model: skynet.SkyNet,
    players: int = 2,
    mcts_iterations: int = 100,
    mcts_temperature: float = 1.0,
    afterstate_initial_realizations: int = 50,
    debug: bool = False,
) -> list[skynet.TrainingDataPoint]:
    model.eval()
    with torch.no_grad():
        game_state = sj.new(players=players)
        game_state = sj.start_round(game_state)
        game_data = []
        while not sj.get_game_over(game_state):
            node = mcts.run_mcts(
                game_state, model, mcts_iterations, afterstate_initial_realizations
            )
            mcts_probs = node.sample_child_visit_probabilities(
                temperature=mcts_temperature
            )
            choice = np.random.choice(sj.MASK_SIZE, p=mcts_probs)
            assert sj.actions(game_state)[choice]
            for symmetric_game_state, symmetric_mcts_probs in get_skyjo_symmetries(
                game_state, mcts_probs
            ):
                game_data.append((symmetric_game_state, symmetric_mcts_probs))

            if debug:
                logging.info(f"{sj.visualize_state(game_state)}")
            game_state = sj.apply_action(game_state, choice)

        outcome = skynet.skyjo_to_state_value(game_state)
        fixed_perspective_score = sj.get_fixed_perspective_round_scores(game_state)
        if debug:
            logging.info("Game over")
            logging.info(f"Game state: {sj.visualize_state(game_state)}")
            logging.info(f"Outcome: {outcome}")
            logging.info(f"Scores: {fixed_perspective_score}")
            logging.info(f"total turns: {sj.get_turn(game_state)}")
    return training_data_from_game_data(game_data, outcome, fixed_perspective_score)


def _add_training_data_and_resubmit_selfplay(
    result: concurrent.futures.Future[list[skynet.TrainingDataPoint]],
    data_queue: mp.Queue,
    pool: concurrent.futures.ThreadPoolExecutor,
    model_factory: SkyNetModelFactory,
    model_update_queue: mp.Queue,
    selfplay_kwargs: dict[str, typing.Any],
) -> None:
    data_queue.put(result.result())
    if not model_update_queue.empty():
        model = model_factory.get_latest_model()
        model_update_queue.get()
    selfplay_future = pool.submit(selfplay, model, **selfplay_kwargs)
    selfplay_future.add_done_callback(
        lambda result: _add_training_data_and_resubmit_selfplay(
            result, data_queue, pool, model_factory, model_update_queue, selfplay_kwargs
        )
    )


def generate_multithreaded_selfplay_games(
    model_factory: SkyNetModelFactory,
    model_update_queue: mp.Queue,
    data_queue: mp.Queue,
    selfplay_kwargs: dict[str, typing.Any],
    thread_count: int = 8,
):
    with concurrent.futures.ThreadPoolExecutor(thread_count) as pool:
        model = model_factory.get_latest_model()
        # If learner has populated queue, it means a new model is available
        if not model_update_queue.empty():
            model = model_factory.get_latest_model()
            model_update_queue.get()
        for _ in range(thread_count):
            selfplay_future = pool.submit(selfplay, model, **selfplay_kwargs)
            selfplay_future.add_done_callback(
                lambda result: _add_training_data_and_resubmit_selfplay(
                    result,
                    data_queue,
                    pool,
                    model_factory,
                    model_update_queue,
                    selfplay_kwargs,
                )
            )


def train(
    model: skynet.SkyNet,
    batch: skynet.TrainingBatch,
    loss_function: typing.Callable,
    learning_rate: float = 1e-4,
) -> None:
    """Performs a single training step on the model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    game_state = batch[0]
    spatial_inputs = torch.tensor(game_state[0], dtype=torch.float32)
    non_spatial_inputs = torch.tensor(game_state[1], dtype=torch.float32)
    model_output = model(spatial_inputs, non_spatial_inputs)

    loss = loss_function(model_output, batch)
    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def faceoff(
    previous_model: skynet.SkyNet,
    new_model: skynet.SkyNet,
    mcts_iterations: int = 100,
    mcts_temperature: float = 1.0,
    afterstate_initial_realizations: int = 50,
):
    raise NotImplementedError


def learn(
    model_factory: SkyNetModelFactory,
    training_steps: int = 100,
    training_data_buffer_size: int = 10000,
    batch_size: int = 128,
    selfplay_processes: int = 2,
    selfplay_process_thread_count: int = 8,
    validation_function: typing.Callable[[skynet.SkyNet], None] = None,
    validation_step_interval: int = 10,
    update_best_model_step_interval: int = 10,
    loss_function: typing.Callable[
        [skynet.SkyNetPrediction, skynet.TrainingBatch], torch.Tensor
    ] = skynet.base_total_loss,
):
    selfplay_data_queue = mp.Queue()
    model_update_queue = mp.Queue()
    actors = []
    logging.info(
        f"Starting {selfplay_processes} selfplay processes each with {selfplay_process_thread_count} threads"
    )
    for _ in range(selfplay_processes):
        actor_process = mp.Process(
            target=generate_multithreaded_selfplay_games,
            args=[
                model_factory,
                model_update_queue,
                selfplay_data_queue,
                constant_basic_selfplay_params(0),
                selfplay_process_thread_count,
            ],
        )
        actor_process.start()
        actors.append(actor_process)

    training_data_buffer = buffer.ReplayBuffer(max_size=training_data_buffer_size)
    logging.info(f"Training for {training_steps} steps")
    for train_iter in range(training_steps):
        # Add training data from the queue into the buffer
        while not selfplay_data_queue.empty() or training_data_buffer.games_count < 16:
            game_data = selfplay_data_queue.get()
            training_data_buffer.add(game_data)
        if train_iter == 0:
            logging.info("Enough selfplay data collected, starting training")
        batch = training_data_buffer.sample_batch(batch_size=batch_size)
        model = model_factory.get_latest_model()
        train(model, batch, loss_function, constant_basic_learning_rate(train_iter))

        if train_iter % validation_step_interval == 0:
            validation_function(model)

        if train_iter > 0 and train_iter % update_best_model_step_interval == 0:
            saved_path = model_factory.save_model(model)
            logging.info(f"Saved model to {saved_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename="logs/train.log",
        filemode="a",
    )
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    players = 2
    model_factory = SkyNetModelFactory(
        players=players,
        dropout_rate=0.5,
        device=torch.device("mps"),
        models_dir=pathlib.Path("./models") / "parallel/",
    )
    learn(model_factory, training_steps=10)
