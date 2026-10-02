"""Module for playing Skyjo games and generating training data."""

from __future__ import annotations

import abc
import dataclasses
import datetime
import logging
import multiprocessing as mp
import pathlib
import typing

import numpy as np

from . import game as sj
from . import mcts, parallel_mcts, player, predictor, skynet

# MARK: Config


@dataclasses.dataclass(slots=True)
class ModelMCTSSelfplayConfig:
    players: int
    action_softmax_temperature: float
    outcome_rollouts: int
    mcts_config: parallel_mcts.Config | mcts.Config


# MARK: Types

FloatArray: typing.TypeAlias = np.ndarray[tuple[int, ...], np.float32]
ActionProbabilities: typing.TypeAlias = FloatArray


class GameHistoryEntry(typing.NamedTuple):
    """One observed decision point from self-play.

    Terminal entries use None for action and action_probabilities. The field
    order remains tuple-compatible with the previous GameHistory tuple.
    """

    state: sj.Skyjo
    action: sj.SkyjoAction | None
    action_probabilities: ActionProbabilities | None


GameHistory: typing.TypeAlias = list[GameHistoryEntry]


class GameDataPoint(typing.NamedTuple):
    """One generated training row.

    Targets are intentionally opaque to gameplay generation. Training utilities
    decide how to interpret them for a specific model/loss setup.
    """

    state: sj.Skyjo
    action: sj.SkyjoAction | None
    targets: typing.Any


GameData: typing.TypeAlias = list[GameDataPoint]


@dataclasses.dataclass(slots=True)
class GameStats:
    game_length: int
    outcome_state_value: sj.StateValue
    scores_state_value: sj.StateValue
    action_counts: np.ndarray[tuple[int], np.float32]
    action_possibility_counts: np.ndarray[tuple[int], np.float32]
    clear_count: int
    flip_count: int
    flip_possibility_count: int
    replace_face_up_count: int
    replace_face_down_count: int
    replace_possibility_count: int

    def to_record_dict(self) -> dict[str, typing.Any]:
        record_dict = {
            "game_length": self.game_length,
            "clear_count": self.clear_count,
        }
        for i, outcome in enumerate(self.outcome_state_value):
            record_dict[f"outcome_{i}"] = outcome
        for i, score in enumerate(self.scores_state_value):
            record_dict[f"score_{i}"] = score

        flip_rate = self.flip_count / max(self.flip_possibility_count, 1)
        replace_face_up_rate = self.replace_face_up_count / max(
            self.replace_possibility_count, 1
        )
        replace_face_down_rate = self.replace_face_down_count / max(
            self.replace_possibility_count, 1
        )
        record_dict["flip_rate"] = flip_rate
        record_dict["replace_face_up_rate"] = replace_face_up_rate
        record_dict["replace_face_down_rate"] = replace_face_down_rate

        action_rates = self.action_counts / np.maximum(
            self.action_possibility_counts, 1
        )
        for i in range(sj.MASK_SIZE):
            record_dict[f"{sj.get_action_name(i)}"] = action_rates[i]
        return record_dict


GeneratedEpisode: typing.TypeAlias = tuple[GameData, GameStats]


# MARK: Helpers


def simulate_game_end(penultimate_state, last_action, simulations=1, *, rng=None):
    """Compatibility adapter for the learner's terminal target computation."""
    from .targets import summarize_terminal

    summary = summarize_terminal(penultimate_state, last_action, simulations, rng=rng)
    return summary.outcome, summary.value, summary.scores, summary.cleared_columns


def game_history_to_game_data(
    game_history: GameHistory,
    terminal_rollouts: int = 1,
    *,
    auxiliary_objectives=None,
    rng=None,
) -> GeneratedEpisode:
    """Construct labels for a completed history; call on the learner side."""
    from .targets import build_targets

    return build_targets(
        game_history, auxiliary_objectives, terminal_rollouts=terminal_rollouts, rng=rng
    )


def print_game_history(
    game_history: GameHistory,
):
    for game_state, action, action_probabilities in game_history:
        print(sj.visualize_state(game_state))
        print(f"ACTION: {sj.get_action_name(action)}")
        print(f"ACTION PROBABILITIES: {action_probabilities}")


# MARK: Selfplay


def distributed_play(
    players: list[player.AbstractPlayer],
    start_state: sj.Skyjo | None = None,
    number_of_games: int = 1,
) -> list[GameHistory]:
    game_histories = []
    for _ in range(number_of_games):
        if start_state is None:
            game_state = sj.new(players=len(players))
            game_state = sj.start_round(game_state)
        else:
            game_state = start_state
        game_history = []
        while not sj.get_game_over(game_state):
            action_probabilities = players[
                sj.get_player(game_state)
            ].get_action_probabilities(game_state)
            action = np.random.choice(sj.MASK_SIZE, p=action_probabilities)
            assert sj.actions(game_state)[action]
            game_history.append(
                GameHistoryEntry(game_state, action, action_probabilities)
            )
            game_state = sj.apply_action(game_state, action)

        game_history.append(GameHistoryEntry(game_state, None, None))
        game_histories.append(game_history)
    return game_histories


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
        game_history.append(GameHistoryEntry(game_state, action, action_probabilities))
        game_state = sj.apply_action(game_state, action)
        if debug:
            print(sj.get_action_name(action))
            logging.info(f"ACTION PROBABILITIES\n{action_probabilities}")
            logging.info(f"ACTION: {sj.get_action_name(action)}")
            logging.info(f"{sj.visualize_state(game_state)}")

    game_history.append(GameHistoryEntry(game_state, None, None))
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
            model_player.mcts_forced_playout_k,
            root_node,
        )
        action_probabilities = root_node.policy_targets(
            model_player.action_softmax_temperature,
            model_player.mcts_forced_playout_k,
        )
        action = np.random.choice(sj.MASK_SIZE, p=action_probabilities)
        assert sj.actions(game_state)[action]
        game_history.append(GameHistoryEntry(game_state, action, action_probabilities))
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

    game_history.append(GameHistoryEntry(game_state, None, None))
    if debug:
        outcome = skynet.skyjo_to_state_value(game_state)
        fixed_perspective_score = sj.get_fixed_perspective_round_scores(game_state)
        logging.info("GAME OVER")
        logging.info(f"OUTCOME: {outcome}")
        logging.info(f"SCORES: {fixed_perspective_score}")
        logging.info(f"TOTAL TURNS: {sj.get_turn(game_state)}")
    return game_history


def batched_model_player_selfplay(
    model_players: list[player.BatchedModelPlayer],
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
        root_node = parallel_mcts.run_mcts(
            game_state,
            model_player.predictor_client,
            mcts_iterations,
            model_player.mcts_dirichlet_epsilon,
            model_player.mcts_after_state_evaluate_all_children,
            model_player.mcts_terminal_state_initial_rollouts,
            model_player.mcts_batched_leaf_count,
            model_player.mcts_virtual_loss,
            model_player.mcts_forced_playout_k,
            root_node,
        )
        action_probabilities = root_node.policy_targets(
            model_player.action_softmax_temperature,
            model_player.mcts_forced_playout_k,
        )

        action = np.random.choice(sj.MASK_SIZE, p=action_probabilities)
        assert sj.actions(game_state)[action]
        game_history.append(GameHistoryEntry(game_state, action, action_probabilities))
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

    game_history.append(GameHistoryEntry(game_state, None, None))
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
    def generate_episode(self) -> GameHistory:
        pass

    @abc.abstractmethod
    def add_game_data(self, game_data: GameHistory):
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
        for data_point in game_data:
            if data_point.action is not None:
                actions[data_point.action] += 1
        return {
            "game_length": sj.get_turn(game_data[-1].state),
            "targets": game_data[-1].targets,
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
            if self._stop_event.is_set():
                break
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
        play_callable: typing.Callable[[], GameHistory] | None = play,
    ):
        super().__init__(id=id, debug=debug, log_level=log_level, log_dir=log_dir)
        self.player = player
        self.players = [self.player for _ in range(player_count)]
        self.game_data_queue = game_data_queue
        self.start_state_generator = start_state_generator
        self.play_callable = play_callable

    def generate_episode(self) -> GameHistory:
        start_state = None
        if self.start_state_generator is not None:
            start_state = self.start_state_generator()
        game_history = self.play_callable(
            self.players,
            debug=self.debug,
            start_state=start_state,
            stop_event=self._stop_event,
        )
        return game_history

    def add_game_data(self, game_data: GameHistory):
        self.game_data_queue.put(game_data)


if __name__ == "__main__":
    import torch

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
