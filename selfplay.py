import itertools
import logging
import multiprocessing as mp

import numpy as np
import torch

import mcts as mcts
import parallel_mcts
import predictor
import skyjo as sj
import skynet


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
        game_state = sj.apply_action(game_state, action)
        if debug:
            logging.debug(f"ACTION PROBABILITIES: {mcts_probs}")
            logging.debug(
                f"MCTS VISIT PROBABILITIES: {node.sample_child_visit_probabilities()}"
            )
            logging.debug(f"ACTION: {sj.get_action_name(action)}")
            logging.debug(f"{sj.visualize_state(game_state)}")
        # Add all symmetric game states to game data
        for symmetric_game_state, symmetric_mcts_probs in get_skyjo_symmetries(
            game_state, mcts_probs
        ):
            game_data.append((symmetric_game_state, symmetric_mcts_probs))
    outcome = skynet.skyjo_to_state_value(game_state)
    fixed_perspective_score = sj.get_fixed_perspective_round_scores(game_state)
    if debug:
        logging.debug(f"OUTCOME: {outcome}")
        logging.debug(f"FIXED PERSPECTIVE SCORE: {fixed_perspective_score}")
    return training_data_from_game_data(game_data, outcome, fixed_perspective_score)


def selfplay(
    model: skynet.SkyNet,
    model_runner_input_queue: mp.Queue,
    model_runner_output_queue: mp.Queue,
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
            node = mcts.run_mcts_with_model_runner(
                game_state,
                model,
                mcts_iterations,
                afterstate_initial_realizations,
                model_runner_input_queue,
                model_runner_output_queue,
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
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            filename=f"logs/multiprocessed_train/selfplay_{self.id}.log",
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
            if self.count % 100 == 0:
                logging.info(f"Selfplay count: {self.count}")
