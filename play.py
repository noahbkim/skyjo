"""Module for playing Skyjo games and generating training data."""

from __future__ import annotations

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
class Config:
    players: int
    action_softmax_temperature: float
    outcome_rollouts: int
    mcts_config: parallel_mcts.Config


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
    ]
]


# MARK: Symmetry


COLUMN_ORDER_PERMUTATIONS = list(itertools.permutations(range(sj.COLUMN_COUNT)))
ROW_ORDER_PERMUTATIONS = list(itertools.permutations(range(sj.ROW_COUNT)))


# NOTE: These are not currently used.
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


def batched_model_selfplay(
    predictor_client: predictor.PredictorClient,
    players: int,
    mcts_config: parallel_mcts.Config,
    action_softmax_temperature: float = 1.0,
    outcome_rollouts: int = 1,
    start_position: sj.Skyjo | None = None,
    debug: bool = False,
) -> GameData:
    game_players = [
        player.BatchedModelPlayer.from_mcts_config(
            predictor_client,
            action_softmax_temperature,
            mcts_config,
        )
        for _ in range(players)
    ]
    game_data = play(game_players, debug, start_position)
    return game_history_to_game_data(game_data, outcome_rollouts)


def model_selfplay(
    predictor_client: predictor.PredictorClient,
    players: int,
    mcts_config: mcts.Config,
    action_softmax_temperature: float = 1.0,
    outcome_rollouts: int = 1,
    start_position: sj.Skyjo | None = None,
    debug: bool = False,
) -> GameData:
    game_players = [
        player.ModelPlayer.from_mcts_config(
            predictor_client,
            action_softmax_temperature,
            mcts_config,
        )
        for _ in range(players)
    ]
    game_data = play(game_players, debug, start_position)
    return game_history_to_game_data(game_data, outcome_rollouts)


def capped_model_selfplay(
    predictor_client: predictor.PredictorClient,
    players: int,
    fast_mcts_config: mcts.Config,
    full_mcts_config: mcts.Config,
    action_softmax_temperature: float = 1.0,
    outcome_rollouts: int = 1,
    start_position: sj.Skyjo | None = None,
    debug: bool = False,
) -> GameData:
    game_players = [
        player.CappedModelPlayer.from_mcts_configs(
            predictor_client,
            action_softmax_temperature,
            fast_mcts_config,
            full_mcts_config,
        )
        for _ in range(players)
    ]
    game_data = play(game_players, debug, start_position)
    return game_history_to_game_data(game_data, outcome_rollouts)


def greedy_selfplay(
    players: int,
    debug: bool = False,
    outcome_rollouts: int = 1,
) -> GameData:
    game_players = [player.GreedyExpectedValuePlayer() for _ in range(players)]
    game_data = play(game_players, debug)
    return game_history_to_game_data(game_data, outcome_rollouts)


def play(
    players: list[player.AbstractPlayer],
    debug: bool = False,
    start_state: sj.Skyjo | None = None,
) -> GameHistory:
    if start_state is None:
        game_state = sj.new(players=len(players))
        game_state = sj.start_round(game_state)
    else:
        game_state = start_state
    game_history = []

    if debug:
        logging.info(f"{sj.visualize_state(game_state)}")

    while not sj.get_game_over(game_state):
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


# MARK: Data Generation Processes


class TrainingDataGenerator(mp.Process):
    """Base class for training data generators.

    Implementations should override the `run_episode` method to run an episode
    and return the training data.
    """

    def __init__(
        self,
        id: str,
        debug: bool = False,
        episodes: int | None = None,
    ):
        super().__init__()
        self.debug = debug
        self.id = id
        self.count = 0
        self.episodes = episodes

    def run_episode(self):
        raise NotImplementedError

    def add_game_data(self, game_data: GameData):
        raise NotImplementedError

    def run(self):
        level = logging.DEBUG if self.debug else logging.INFO
        log_dir = pathlib.Path(
            f"logs/multiprocessed_train/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}/"
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            filename=log_dir / f"play_{self.id}.log",
            filemode="a",
        )
        logging.info(f"Starting training data generator process id: {self.id}")
        while (
            self.episodes is not None and self.count < self.episodes
        ) or self.episodes is None:
            if self.count % 1 == 0:
                logging.info(f"Selfplay count: {self.count}")
            episode_data = self.run_episode()
            self.count += 1
            if self.debug:
                logging.debug(f"Finished episode ({len(episode_data)} data points)")
            self.add_game_data(episode_data)


class BatchedSelfplayGenerator(TrainingDataGenerator):
    def __init__(
        self,
        predictor_client: predictor.PredictorClient,
        selfplay_data_queue: mp.Queue,
        id: int,
        players: int,
        mcts_config: parallel_mcts.Config,
        action_softmax_temperature: float = 1.0,
        outcome_rollouts: int = 1,
        start_position_generator: typing.Callable[[], sj.Skyjo] | None = None,
        debug: bool = False,
    ):
        super().__init__(id, debug)
        self.predictor_client = predictor_client
        self.selfplay_data_queue = selfplay_data_queue
        self.players = players
        self.mcts_config = mcts_config
        self.action_softmax_temperature = action_softmax_temperature
        self.outcome_rollouts = outcome_rollouts
        self.start_position_generator = start_position_generator
        self.debug = debug
        self.id = id
        self.count = 0

    @classmethod
    def from_config(
        cls,
        predictor_client: predictor.PredictorClient,
        selfplay_data_queue: mp.Queue,
        id: int,
        config: Config,
        start_position_generator: typing.Callable[[], sj.Skyjo] | None = None,
        debug: bool = False,
    ):
        return cls(
            predictor_client,
            selfplay_data_queue,
            id,
            config.players,
            config.mcts_config,
            config.action_softmax_temperature,
            config.outcome_rollouts,
            start_position_generator,
            debug,
        )

    def add_game_data(self, game_data: GameData):
        self.selfplay_data_queue.put(game_data)

    def run_episode(self):
        return batched_model_selfplay(
            self.predictor_client,
            players=self.players,
            mcts_config=self.mcts_config,
            action_softmax_temperature=self.action_softmax_temperature,
            outcome_rollouts=self.outcome_rollouts,
            debug=self.debug,
            start_position=(
                self.start_position_generator()
                if self.start_position_generator is not None
                else None
            ),
        )


class SelfplayGenerator(TrainingDataGenerator):
    def __init__(
        self,
        predictor_client: predictor.PredictorClient,
        selfplay_data_queue: mp.Queue,
        id: int,
        players: int,
        mcts_config: mcts.Config,
        action_softmax_temperature: float = 1.0,
        outcome_rollouts: int = 1,
        start_position_generator: typing.Callable[[], sj.Skyjo] | None = None,
        debug: bool = False,
    ):
        super().__init__(id, debug)
        self.predictor_client = predictor_client
        self.selfplay_data_queue = selfplay_data_queue
        self.players = players
        self.mcts_config = mcts_config
        self.action_softmax_temperature = action_softmax_temperature
        self.outcome_rollouts = outcome_rollouts
        self.start_position_generator = start_position_generator
        self.debug = debug
        self.id = id
        self.count = 0

    @classmethod
    def from_config(
        cls,
        predictor_client: predictor.PredictorClient,
        selfplay_data_queue: mp.Queue,
        id: int,
        config: Config,
        start_position_generator: typing.Callable[[], sj.Skyjo] | None = None,
        debug: bool = False,
    ):
        return cls(
            predictor_client,
            selfplay_data_queue,
            id,
            config.players,
            config.mcts_config,
            config.action_softmax_temperature,
            config.outcome_rollouts,
            start_position_generator,
            debug,
        )

    def add_game_data(self, game_data: GameData):
        self.selfplay_data_queue.put(game_data)

    def run_episode(self):
        return model_selfplay(
            self.predictor_client,
            players=self.players,
            mcts_config=self.mcts_config,
            action_softmax_temperature=self.action_softmax_temperature,
            outcome_rollouts=self.outcome_rollouts,
            debug=self.debug,
            start_position=(
                self.start_position_generator()
                if self.start_position_generator is not None
                else None
            ),
        )


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
    saved_model_path = pathlib.Path(
        "models/distributed/20250620_101923/model_20250621_215207.pth"
    )
    model.load_state_dict(torch.load(saved_model_path, weights_only=True))
    model.eval()
    model.to(device)
    trained_predictor_client = predictor.NaivePredictorClient(
        model, max_batch_size=4096
    )
    while True:
        data = play(
            [
                player.ModelPlayer(trained_predictor_client, 1.0, 400, 0.25, True, 25),
                player.ModelPlayer(trained_predictor_client, 1.0, 400, 0.25, True, 25),
            ]
        )
        print("game done")
