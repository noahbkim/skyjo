from __future__ import annotations

import datetime
import itertools
import logging
import multiprocessing as mp
import pathlib
import random
import typing

import numpy as np

import mcts as mcts
import player
import predictor
import skyjo as sj
import skynet

# MARK: Symmetry


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


def normalize_table(skyjo: sj.Skyjo) -> np.ndarray[tuple[int], np.uint8]:
    """Returns normalized table"""
    raise NotImplementedError("Not implemented")


def normalize_action(action: int, skyjo: sj.Skyjo) -> int:
    """Returns normalized action"""
    raise NotImplementedError("Not implemented")


# MARK: Training data


def training_data_from_game_data(
    game_data: GameData,
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
    outcome_state_value = skynet.skyjo_to_state_value(game_data[-1][0])
    fixed_perspective_score = sj.get_fixed_perspective_round_scores(game_data[-1][0])
    for game_state, action, mcts_probs in game_data[:-1]:
        training_data.append(
            (
                game_state,  # game
                mcts_probs,  # policy target
                np.roll(
                    outcome_state_value, -sj.get_player(game_state)
                ),  # outcome target
                np.roll(
                    fixed_perspective_score,
                    -sj.get_player(game_state),
                ),  # points target
            )
        )
    return training_data


def outcome_and_scores_from_game_data(
    game_data: GameData,
    simulations: int = 1000,
) -> tuple[np.ndarray[tuple[int], np.float32], np.ndarray[tuple[int], np.float32]]:
    """Returns the outcome of the game"""
    penultimate_state, last_action, _ = game_data[-2]
    players = sj.get_player_count(penultimate_state)
    outcomes, scores = np.zeros(players), np.zeros(players)
    for _ in range(simulations):
        game_state = penultimate_state
        final_state = sj.apply_action(game_state, last_action)
        outcomes[sj.get_fixed_perspective_winner(final_state)] += 1
        scores += sj.get_fixed_perspective_round_scores(final_state)
    return outcomes / simulations, scores / simulations


# MARK: Selfplay


def multiprocessed_selfplay(
    predictor_client: predictor.PredictorClient,
    players: int = 2,
    mcts_iterations: int = 100,
    mcts_temperature: float = 1.0,
    afterstate_initial_realizations: int = 50,
    virtual_loss: float = 0.0,
    max_parallel_evaluations: int = 32,
    debug: bool = False,
):
    game_players = [
        player.ModelPlayer(
            predictor_client,
            mcts_temperature,
            mcts_iterations,
            afterstate_initial_realizations,
            virtual_loss,
            max_parallel_evaluations,
        )
        for _ in range(players)
    ]
    game_data = play(game_players, debug)
    # outcome, scores = outcome_and_scores_from_game_data(game_data)
    return training_data_from_game_data(game_data)


def selfplay(
    model: skynet.SkyNet,
    players: int = 2,
    mcts_iterations: int = 100,
    mcts_temperature: float = 1.0,
    afterstate_initial_realizations: int = 50,
    debug: bool = False,
) -> list[skynet.TrainingDataPoint]:
    game_players = [player.GreedyExpectedValuePlayer() for _ in range(players - 1)] + [
        player.Simple(
            model,
            mcts_iterations,
            mcts_temperature,
            afterstate_initial_realizations,
        )
    ]
    random.shuffle(game_players)
    game_data = play(game_players, debug)
    # outcome, scores = outcome_and_scores_from_game_data(game_data)
    return training_data_from_game_data(game_data)


def multiprocessed_play_greedy_players(
    players: int = 2,
    debug: bool = False,
) -> list[skynet.TrainingDataPoint]:
    game_players = [player.GreedyExpectedValuePlayer() for _ in range(players)]
    # random.shuffle(game_players)
    game_data = play(game_players, debug)
    # outcome, scores = outcome_and_scores_from_game_data(game_data)
    return training_data_from_game_data(game_data)


def play(
    players: list[player.AbstractPlayer],
    debug: bool = False,
    start_state: sj.Skyjo | None = None,
) -> GameData:
    if start_state is None:
        game_state = sj.new(players=len(players))
        game_state = sj.start_round(game_state)
    else:
        game_state = start_state
    game_data = []

    if debug:
        logging.info(f"{sj.visualize_state(game_state)}")
    while not sj.get_game_over(game_state):
        action_probabilities = players[
            sj.get_player(game_state)
        ].get_action_probabilities(game_state)
        action = np.random.choice(sj.MASK_SIZE, p=action_probabilities)
        assert sj.actions(game_state)[action]
        game_data.append((game_state, action, action_probabilities))
        game_state = sj.apply_action(game_state, action)
        if debug:
            logging.info(f"ACTION PROBABILITIES\n{action_probabilities}")
            logging.info(f"ACTION: {sj.get_action_name(action)}")
            logging.info(f"{sj.visualize_state(game_state)}")

    game_data.append((game_state, None, None))
    if debug:
        outcome = skynet.skyjo_to_state_value(game_state)
        fixed_perspective_score = sj.get_fixed_perspective_round_scores(game_state)
        logging.info("GAME OVER")
        logging.info(f"OUTCOME: {outcome}")
        logging.info(f"SCORES: {fixed_perspective_score}")
        logging.info(f"TOTAL TURNS: {sj.get_turn(game_state)}")
    return game_data


# MARK: Processes


class TrainingDataGenerator(mp.Process):
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
            if self.count % 10 == 0:
                logging.info(f"Selfplay count: {self.count}")
            episode_data = self.run_episode()
            self.count += 1
            if self.debug:
                logging.debug(f"Finished episode ({len(episode_data)} data points)")
            self.selfplay_data_queue.put(episode_data)


class MultiProcessedSelfplayGenerator(TrainingDataGenerator):
    def __init__(
        self,
        predictor_client: predictor.PredictorClient,
        selfplay_data_queue: mp.Queue,
        id: int = 0,
        players: int = 2,
        mcts_iterations: int = 100,
        mcts_temperature: float = 1.0,
        afterstate_initial_realizations: int = 50,
        virtual_loss: float = 0.0,
        max_parallel_evaluations: int = 32,
        debug: bool = False,
    ):
        super().__init__(id, debug)
        self.predictor_client = predictor_client
        self.selfplay_data_queue = selfplay_data_queue
        self.players = players
        self.mcts_iterations = mcts_iterations
        self.mcts_temperature = mcts_temperature
        self.afterstate_initial_realizations = afterstate_initial_realizations
        self.virtual_loss = virtual_loss
        self.max_parallel_evaluations = max_parallel_evaluations
        self.debug = debug
        self.id = id
        self.count = 0

    def run_episode(self):
        return multiprocessed_selfplay(
            self.predictor_client,
            players=self.players,
            mcts_iterations=self.mcts_iterations,
            mcts_temperature=self.mcts_temperature,
            afterstate_initial_realizations=self.afterstate_initial_realizations,
            virtual_loss=self.virtual_loss,
            max_parallel_evaluations=self.max_parallel_evaluations,
            debug=self.debug,
        )


class MultiProcessedPlayGreedyPlayersGenerator(TrainingDataGenerator):
    def __init__(
        self,
        predictor_client: predictor.PredictorClient,
        selfplay_data_queue: mp.Queue,
        id: str = "greedy_ev",
        players: int = 2,
        mcts_iterations: int = 100,
        mcts_temperature: float = 1.0,
        afterstate_initial_realizations: int = 50,
        virtual_loss: float = 0.0,
        max_parallel_evaluations: int = 32,
        episodes: int | None = None,
        debug: bool = False,
    ):
        super().__init__(id, debug, episodes)
        self.predictor_client = predictor_client
        self.selfplay_data_queue = selfplay_data_queue
        self.players = players
        self.mcts_iterations = mcts_iterations
        self.mcts_temperature = mcts_temperature
        self.afterstate_initial_realizations = afterstate_initial_realizations
        self.virtual_loss = virtual_loss
        self.max_parallel_evaluations = max_parallel_evaluations
        self.debug = debug
        self.id = id
        self.count = 0

    def run_episode(self):
        return multiprocessed_play_greedy_players(
            self.predictor_client,
            players=self.players,
            mcts_iterations=self.mcts_iterations,
            mcts_temperature=self.mcts_temperature,
            afterstate_initial_realizations=self.afterstate_initial_realizations,
            virtual_loss=self.virtual_loss,
            max_parallel_evaluations=self.max_parallel_evaluations,
            debug=self.debug,
        )


# MARK: Types

GameData: typing.TypeAlias = list[
    tuple[sj.Skyjo, sj.SkyjoAction, np.ndarray[tuple[int], np.float32]]
    # (game state, action, action probabilities)
]
