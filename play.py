"""Module for playing Skyjo games and generating training data."""

from __future__ import annotations

import abc
import dataclasses
import datetime
import itertools
import logging
import multiprocessing as mp
import pathlib
import typing

import numpy as np

import mcts
import parallel_mcts
import player
import predictor
import skyjo as sj
import skynet

# MARK: Config


@dataclasses.dataclass(slots=True)
class ModelMCTSSelfplayConfig:
    players: int
    action_softmax_temperature: float
    outcome_rollouts: int
    mcts_config: parallel_mcts.Config | mcts.Config


# MARK: Types

# Purely to represent what happened in a game.
GameHistory: typing.TypeAlias = list[
    tuple[sj.Skyjo, sj.SkyjoAction, np.ndarray[tuple[int], np.float32]]
    # (game state, action, action probabilities)
]

# Contains the targets for training
GameData: typing.TypeAlias = list[
    tuple[
        sj.Skyjo,  # game state
        np.ndarray[tuple[int], np.float32],  # outcome target
        np.ndarray[tuple[int], np.float32],  # points target
        np.ndarray[tuple[int], np.float32],  # policy target
        sj.SkyjoAction,  # realized action
    ]
]


# MARK: Symmetry
# NOTE: NOT CURRENTLY NOT USED


COLUMN_ORDER_PERMUTATIONS = list(itertools.permutations(range(sj.COLUMN_COUNT)))
ROW_ORDER_PERMUTATIONS = list(itertools.permutations(range(sj.ROW_COUNT)))


def get_symmetry_policy(
    original_policy: np.ndarray[tuple[int], np.float32],
    new_column_order: tuple[int, ...],
) -> np.ndarray[tuple[int], np.float32]:
    """Takes an original policy (sj.MASK_SIZE,) and returns a new policy based
    on the new_column_order permutation.
    """
    new_policy = original_policy.copy()
    flip_policy = new_policy[sj.MASK_FLIP : sj.MASK_FLIP + sj.FINGER_COUNT]
    flip_grid = flip_policy.reshape(sj.ROW_COUNT, sj.COLUMN_COUNT)
    permuted_flip_grid = flip_grid[:, new_column_order]
    new_policy[sj.MASK_FLIP : sj.MASK_FLIP + sj.FINGER_COUNT] = (
        permuted_flip_grid.reshape(-1)
    )
    replace_policy = new_policy[sj.MASK_REPLACE : sj.MASK_REPLACE + sj.FINGER_COUNT]
    replace_grid = replace_policy.reshape(sj.ROW_COUNT, sj.COLUMN_COUNT)
    permuted_replace_grid = replace_grid[:, new_column_order]
    new_policy[sj.MASK_REPLACE : sj.MASK_REPLACE + sj.FINGER_COUNT] = (
        permuted_replace_grid.reshape(-1)
    )
    return new_policy


def get_skyjo_symmetries(
    skyjo: sj.Skyjo, policy: np.ndarray[tuple[int], np.float32]
) -> list[tuple[sj.Skyjo, np.ndarray[tuple[int], np.float32]]]:
    """Return a list of symmetrically equivalent `Skyjo` states and corresponding policy."""
    # TODO: Make this work for within column permutations too
    symmetries = []
    symmetry_hashes = set()
    for player_column_orders in itertools.combinations_with_replacement(
        COLUMN_ORDER_PERMUTATIONS, sj.get_player_count(skyjo)
    ):
        new_table = sj.get_board(skyjo).copy()
        for player_idx, column_order in enumerate(player_column_orders):
            # Note: new_table[player_idx, :, column_order, :] results in a
            # shape that is different.
            # https://stackoverflow.com/questions/55829631/why-using-an-array-as-an-index-changes-the-shape-of-a-multidimensional-ndarray
            new_table[player_idx] = new_table[player_idx][:, column_order, :]
        if new_table.tobytes() not in symmetry_hashes:
            symmetry_hashes.add(new_table.tobytes())
            new_policy = get_symmetry_policy(policy, player_column_orders[0])
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
                    ),
                    new_policy,
                )
            )
    return symmetries


# MARK: Helpers


def simulate_game_end(
    penultimate_state: sj.Skyjo,
    last_action: sj.SkyjoAction,
    simulations: int = 1,
) -> tuple[np.ndarray[tuple[int], np.float32], np.ndarray[tuple[int], np.float32]]:
    """Returns the outcome of the game"""
    players = sj.get_player_count(penultimate_state)
    outcomes, scores = np.zeros(players), np.zeros(players)
    for _ in range(simulations):
        game_state = penultimate_state
        final_state = sj.apply_action(game_state, last_action)
        outcomes[sj.get_fixed_perspective_winner(final_state)] += 1 / simulations
        scores += sj.get_fixed_perspective_round_scores(final_state) / simulations
    return outcomes, scores


def game_history_to_game_data(
    game_history: GameHistory,
    terminal_rollouts: int = 1,
) -> GameData:
    """Converts game data to training data.

    Game data is a tuple of skyjo state and mcts action probabilities. Converts this
    to a list of training data points which contain the current state, and targets
    for the network to train on.

    Args:
        game_data: A list of tuples containing a skyjo state and mcts action probabilities.
        terminal_rollouts: The number of terminal rollouts to use to compute the outcome
            state value and fixed perspective score.

    Returns:
        A list of training data points.
    """
    training_data = []
    penultimate_state, penultimate_action = game_history[-2][0], game_history[-2][1]
    outcome_state_value, fixed_perspective_score = simulate_game_end(
        penultimate_state, penultimate_action, terminal_rollouts
    )

    # outcome_state_value = skynet.skyjo_to_state_value(game_data[-1][0])
    # fixed_perspective_score = sj.get_fixed_perspective_round_scores(game_data[-1][0])
    for game_state, action, mcts_probs in game_history[:-1]:
        training_data.append(
            (
                game_state,  # game
                np.roll(
                    outcome_state_value, -sj.get_player(game_state)
                ),  # outcome target
                np.roll(
                    fixed_perspective_score,
                    -sj.get_player(game_state),
                ),  # points target
                mcts_probs,  # policy target
                action,  # realized action
            )
        )
    return training_data


def print_game_history(
    game_history: GameHistory,
):
    for game_state, action, action_probabilities in game_history:
        print(sj.visualize_state(game_state))
        print(f"ACTION: {sj.get_action_name(action)}")
        print(f"ACTION PROBABILITIES: {action_probabilities}")


# MARK: Selfplay


def play(
    players: list[player.AbstractPlayer],
    debug: bool = False,
    start_state: sj.Skyjo | None = None,
    stop_event: mp.Event | None = None,
) -> GameHistory:
    if start_state is None:
        game_state = sj.new(players=len(players))
        game_state = sj.start_round(game_state)
    else:
        game_state = start_state
    game_history = []

    if debug:
        logging.info(f"{sj.visualize_state(game_state)}")
    while (stop_event is None or not stop_event.is_set()) and not sj.get_game_over(
        game_state
    ):
        action_probabilities = players[
            sj.get_player(game_state)
        ].get_action_probabilities(game_state)
        action = np.random.choice(sj.MASK_SIZE, p=action_probabilities)
        assert sj.actions(game_state)[action]
        game_history.append((game_state, action, action_probabilities))
        game_state = sj.apply_action(game_state, action)
        if debug:
            print(sj.get_action_name(action))
            logging.info(f"ACTION PROBABILITIES\n{action_probabilities}")
            logging.info(f"ACTION: {sj.get_action_name(action)}")
            logging.info(f"{sj.visualize_state(game_state)}")

    game_history.append((game_state, None, None))
    if debug:
        outcome = skynet.skyjo_to_state_value(game_state)
        fixed_perspective_score = sj.get_fixed_perspective_round_scores(game_state)
        logging.info("GAME OVER")
        logging.info(f"OUTCOME: {outcome}")
        logging.info(f"SCORES: {fixed_perspective_score}")
        logging.info(f"TOTAL TURNS: {sj.get_turn(game_state)}")
    return game_history


def model_player_selfplay(
    model_players: list[player.ModelPlayer],
    debug: bool = False,
    start_state: sj.Skyjo | None = None,
    stop_event: mp.Event | None = None,
) -> GameHistory:
    if start_state is None:
        game_state = sj.new(players=len(model_players))
        game_state = sj.start_round(game_state)
    else:
        game_state = start_state

    if debug:
        logging.info(f"{sj.visualize_state(game_state)}")

    root_node = None
    game_history = []
    while (stop_event is None or not stop_event.is_set()) and not sj.get_game_over(
        game_state
    ):
        model_player = model_players[sj.get_player(game_state)]
        mcts_iterations = model_player.mcts_iterations
        if root_node is not None:
            mcts_iterations -= root_node.visit_count
        root_node = mcts.run_mcts(
            game_state,
            model_player.predictor_client,
            mcts_iterations,
            model_player.mcts_dirichlet_epsilon,
            model_player.mcts_after_state_evaluate_all_children,
            model_player.mcts_terminal_state_initial_rollouts,
            root_node,
        )
        action_probabilities = root_node.sample_child_visit_probabilities(
            model_player.action_softmax_temperature
        )
        action = np.random.choice(sj.MASK_SIZE, p=action_probabilities)
        assert sj.actions(game_state)[action]
        game_history.append((game_state, action, action_probabilities))
        if sj.is_action_random(action, game_state):
            game_state = sj.apply_action(game_state, action)
            if not sj.get_game_over(game_state):
                root_node = root_node.children[action].children.get(
                    sj.hash_skyjo(game_state)
                )
        else:
            game_state = sj.apply_action(game_state, action)
            if not sj.get_game_over(game_state):
                root_node = root_node.children[action]

        if debug:
            print(sj.get_action_name(action))
            logging.info(f"ACTION PROBABILITIES\n{action_probabilities}")
            logging.info(f"ACTION: {sj.get_action_name(action)}")
            logging.info(f"{sj.visualize_state(game_state)}")

    game_history.append((game_state, None, None))
    if debug:
        outcome = skynet.skyjo_to_state_value(game_state)
        fixed_perspective_score = sj.get_fixed_perspective_round_scores(game_state)
        logging.info("GAME OVER")
        logging.info(f"OUTCOME: {outcome}")
        logging.info(f"SCORES: {fixed_perspective_score}")
        logging.info(f"TOTAL TURNS: {sj.get_turn(game_state)}")
    return game_history


# MARK: Data Generation Processes


class AbstractTrainingDataGenerator(mp.Process, abc.ABC):
    """Abstract Base class for training data generators.

    Implementations should override the `run_episode` method to run an episode
    and return the training data.
    """

    def __init__(
        self,
        id: str,
        debug: bool = False,
        log_level: int = logging.INFO,
        log_dir: pathlib.Path | None = None,
    ):
        super().__init__()
        self.id = id
        self.episode_count = 0

        self.debug = debug
        self.log_level = log_level
        if log_dir is None:
            log_dir = pathlib.Path(
                f"logs/multiprocessed_train/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}/"
            )
        self.log_dir = log_dir
        self._stop_event = mp.Event()

    @abc.abstractmethod
    def generate_episode(self) -> GameData:
        pass

    @abc.abstractmethod
    def add_game_data(self, game_data: GameData):
        pass

    def stop(self):
        self._stop_event.set()

    def cleanup(self, timeout: float = 1):
        logging.info(f"Cleaning up training data generator process {self.id}")
        self.stop()
        self.join(timeout=timeout)
        if self.is_alive():
            logging.warning(
                f"Training data generator process {self.id} is still alive, forcefully terminating"
            )
            self.terminate()
            self.join()

    def game_stats(self, game_data: GameData):
        """Computes statistics from episode game data. Can be overridden by subclasses."""
        actions = np.zeros(sj.MASK_SIZE)
        for _, _, _, _, realized_action in game_data:
            actions[realized_action] += 1
        return {
            "game_length": sj.get_turn(game_data[-1][0]),
            "outcome": game_data[-1][1],
            "scores": game_data[-1][2],
            "action_counts": actions,
            "action_frequencies": actions / len(game_data),
        }

    def run(self):
        # Setup logging
        level = logging.DEBUG if self.debug else self.log_level
        self.log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            filename=self.log_dir / f"{self.id}.log",
            filemode="a",
        )
        logging.info("Starting training data generator process")
        while not self._stop_event.is_set():
            if self.episode_count % 1 == 0:
                logging.info(f"Selfplay count: {self.episode_count}")
            episode_data = self.generate_episode()
            self.add_game_data(episode_data)
            self.episode_count += 1


class SelfplayGenerator(AbstractTrainingDataGenerator):
    def __init__(
        self,
        id: str,
        player: player.AbstractPlayer,
        player_count: int,
        game_data_queue: mp.Queue,
        start_state_generator: typing.Callable[[], sj.Skyjo] | None = None,
        debug: bool = False,
        log_level: int = logging.INFO,
        log_dir: pathlib.Path | None = None,
        outcome_rollouts: int = 1,
        play_callable: typing.Callable[[], GameHistory] | None = play,
    ):
        super().__init__(id=id, debug=debug, log_level=log_level, log_dir=log_dir)
        self.player = player
        self.players = [self.player for _ in range(player_count)]
        self.game_data_queue = game_data_queue
        self.start_state_generator = start_state_generator
        self.outcome_rollouts = outcome_rollouts
        self.play_callable = play_callable

    def generate_episode(self) -> GameData:
        start_state = None
        if self.start_state_generator is not None:
            start_state = self.start_state_generator()
        game_history = self.play_callable(
            self.players,
            debug=self.debug,
            start_state=start_state,
            stop_event=self._stop_event,
        )
        return game_history_to_game_data(game_history, self.outcome_rollouts)

    def add_game_data(self, game_data: GameData):
        self.game_data_queue.put(game_data)


if __name__ == "__main__":
    import torch

    import player

    device = torch.device("cpu")
    model = skynet.EquivariantSkyNet(
        spatial_input_shape=(2, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
        non_spatial_input_shape=(sj.GAME_SIZE,),
        value_output_shape=(2,),
        policy_output_shape=(sj.MASK_SIZE,),
        device=device,
        embedding_dimensions=16,
        global_state_embedding_dimensions=32,
        num_heads=2,
    )
    model.eval()
    model.to(device)
    predictor_client = predictor.NaivePredictorClient(model, max_batch_size=4096)
    while True:
        data = play(
            [
                player.ModelPlayer(predictor_client, 1.0, 400, 0.25, True, 25),
                player.ModelPlayer(predictor_client, 1.0, 400, 0.25, True, 25),
            ]
        )
        print("game done")
