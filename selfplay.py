import datetime
import itertools
import logging
import multiprocessing as mp
import pathlib
import random

import numpy as np
import torch

import mcts as mcts
import parallel_mcts
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
                        skyjo[7],
                    ),
                    new_policy,
                )
            )
    return symmetries


def normalize_table(skyjo: sj.Skyjo) -> np.ndarray[tuple[int], np.uint8]:
    """Returns normalized table"""
    raise NotImplementedError("Not implemented")


def normalize_action(action: int, skyjo: sj.Skyjo) -> int:
    assert np.array_equal(sj.get_table(skyjo), normalize_table(skyjo))
    if action < sj.MASK_FLIP:
        return action
    if action < sj.MASK_REPLACE:
        row, col = divmod(action - sj.MASK_FLIP, sj.COLUMN_COUNT)
    else:
        row, col = divmod(action - sj.MASK_REPLACE, sj.COLUMN_COUNT)
    if sj.get_finger(skyjo, row, col) != sj.FINGER_HIDDEN:
        return action


# MARK: Training data


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
        )
    return training_data


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
    game_state = sj.new(players=players)
    game_state = sj.start_round(game_state)
    game_data = []
    if debug:
        logging.debug(f"{sj.visualize_state(game_state)}")
    while not sj.get_game_over(game_state):
        node = parallel_mcts.run_mcts(
            game_state,
            predictor_client,
            iterations=mcts_iterations,
            afterstate_initial_realizations=afterstate_initial_realizations,
            virtual_loss=virtual_loss,
            max_parallel_evaluations=max_parallel_evaluations,
        )
        mcts_probs = node.sample_child_visit_probabilities(temperature=mcts_temperature)
        action = np.random.choice(sj.MASK_SIZE, p=mcts_probs)
        assert sj.actions(game_state)[action]
        game_data.append((game_state, mcts_probs))
        game_state = sj.apply_action(game_state, action)
        if debug:
            logging.debug(f"ACTION PROBABILITIES: {mcts_probs}")
            logging.debug(
                f"MCTS VISIT PROBABILITIES: {node.sample_child_visit_probabilities()}"
            )
            logging.debug(f"ACTION: {sj.get_action_name(action)}")
            logging.debug(f"{sj.visualize_state(game_state)}")
    outcome = skynet.skyjo_to_state_value(game_state)
    fixed_perspective_score = sj.get_fixed_perspective_round_scores(game_state)
    if debug:
        logging.debug(f"OUTCOME: {outcome}")
        logging.debug(f"FIXED PERSPECTIVE SCORE: {fixed_perspective_score}")
    return training_data_from_game_data(game_data, outcome, fixed_perspective_score)


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
                game_state,
                model,
                mcts_iterations,
                afterstate_initial_realizations,
            )
            mcts_probs = node.sample_child_visit_probabilities(
                temperature=mcts_temperature
            )
            choice = np.random.choice(sj.MASK_SIZE, p=mcts_probs)
            assert sj.actions(game_state)[choice]
            game_data.append((game_state, mcts_probs))

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


def multiprocessed_play_greedy_players(
    predictor_client: predictor.PredictorClient,
    players: int = 2,
    mcts_iterations: int = 100,
    mcts_temperature: float = 1.0,
    afterstate_initial_realizations: int = 50,
    virtual_loss: float = 0.0,
    max_parallel_evaluations: int = 32,
    debug: bool = False,
) -> list[skynet.TrainingDataPoint]:
    greedy_player = player.GreedyExpectedValuePlayer()
    model_player_turn_order = random.randint(0, players - 1)
    game_state = sj.new(players=players)
    game_state = sj.start_round(game_state)
    game_data = []
    while not sj.get_game_over(game_state):
        if sj.get_player(game_state) == model_player_turn_order:
            node = parallel_mcts.run_mcts(
                game_state,
                predictor_client,
                iterations=mcts_iterations,
                afterstate_initial_realizations=afterstate_initial_realizations,
                virtual_loss=virtual_loss,
                max_parallel_evaluations=max_parallel_evaluations,
            )
            mcts_probs = node.sample_child_visit_probabilities(
                temperature=mcts_temperature
            )
            action = np.random.choice(sj.MASK_SIZE, p=mcts_probs)
            assert sj.actions(game_state)[action]
            game_data.append((game_state, mcts_probs))
        else:
            action = greedy_player.get_action(game_state)
            action_probabilities = np.zeros(sj.MASK_SIZE)
            action_probabilities[action] = 1.0
            assert sj.actions(game_state)[action]
            game_data.append((game_state, action_probabilities))
        game_state = sj.apply_action(game_state, action)
        if debug:
            logging.debug(f"{sj.visualize_state(game_state)}")

    outcome = skynet.skyjo_to_state_value(game_state)
    fixed_perspective_score = sj.get_fixed_perspective_round_scores(game_state)
    if debug:
        logging.info("Game over")
        logging.info(f"Game state: {sj.visualize_state(game_state)}")
        logging.info(f"Outcome: {outcome}")
        logging.info(f"Scores: {fixed_perspective_score}")
        logging.info(f"total turns: {sj.get_turn(game_state)}")
    return training_data_from_game_data(game_data, outcome, fixed_perspective_score)


# MARK: Processes


class MultiProcessedSelfplayGenerator(mp.Process):
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
        super().__init__()
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
            filename=log_dir / f"selfplay_{self.id}.log",
            filemode="a",
        )
        logging.info(f"Starting selfplay generator process for id: {self.id}")
        while True:
            episode_data = multiprocessed_selfplay(
                self.predictor_client,
                players=self.players,
                mcts_iterations=self.mcts_iterations,
                mcts_temperature=self.mcts_temperature,
                afterstate_initial_realizations=self.afterstate_initial_realizations,
                virtual_loss=self.virtual_loss,
                max_parallel_evaluations=self.max_parallel_evaluations,
                debug=self.debug,
            )
            self.count += 1
            if self.debug:
                logging.debug(
                    f"Finished selfplay episode ({len(episode_data)} data points)"
                )
            self.selfplay_data_queue.put(episode_data)
            if self.count % 10 == 1:
                logging.info(f"Selfplay count: {self.count}")


class MultiProcessedPlayGreedyPlayersGenerator(mp.Process):
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
        super().__init__()
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
            filename=log_dir / f"greedy_play_{self.id}.log",
            filemode="a",
        )
        logging.info(f"Starting greedy play generator process for id: {self.id}")
        while True:
            episode_data = multiprocessed_play_greedy_players(
                self.predictor_client,
                players=self.players,
                mcts_iterations=self.mcts_iterations,
                mcts_temperature=self.mcts_temperature,
                afterstate_initial_realizations=self.afterstate_initial_realizations,
                virtual_loss=self.virtual_loss,
                max_parallel_evaluations=self.max_parallel_evaluations,
                debug=self.debug,
            )
            self.count += 1
            if self.debug:
                logging.debug(
                    f"Finished greedy play episode ({len(episode_data)} data points)"
                )
            self.selfplay_data_queue.put(episode_data)
            if self.count % 10 == 1:
                logging.info(f"Greedy Play count: {self.count}")
