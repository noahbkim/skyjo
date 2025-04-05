import dataclasses
import logging
import time
import typing

import numpy as np
import torch

import mcts
import skyjo_immutable as sj
import skynet


@dataclasses.dataclass(slots=True)
class DataPoint:
    state: np.ndarray
    policy_target: np.ndarray
    value_target: float


@dataclasses.dataclass(slots=True)
class DataBatch:
    states_tensor: torch.Tensor
    policy_tensor: torch.Tensor
    values_tensor: torch.Tensor

    @property
    def batch_size(self) -> int:
        return len(self.states_tensor)

    @classmethod
    def from_data_points(
        cls, data_points: list[DataPoint], device: torch.device
    ) -> typing.Self:
        states_tensor = torch.tensor(
            [data_point.state for data_point in data_points], dtype=torch.float32
        )
        policy_tensor = torch.tensor(
            [data_point.policy_target for data_point in data_points],
            dtype=torch.float32,
        )
        values_tensor = torch.tensor(
            [data_point.value_target for data_point in data_points],
            dtype=torch.float32,
        )
        if device == torch.device("cuda"):
            return cls(
                states_tensor.contiguous().cuda(),
                policy_tensor.contiguous().cuda(),
                values_tensor.contiguous().cuda(),
            )
        return cls(states_tensor, policy_tensor, values_tensor)


def sample_action(action_probs: np.ndarray) -> int:
    """Returns an action index sampled according to the action probabilities"""
    actions = [action for action, _ in action_probs.items()]
    prob_dist = [action_prob for _, action_prob in action_probs.items()]
    return actions[np.random.choice(len(actions), p=prob_dist)]


def selfplay_game(
    model: skynet.SkyNet,
    num_players: int,
    mcts_iterations: int = 10,
    temperature: float = 1.0,
    c_puct: float = 1.0,
) -> list[tuple[np.array, np.array, float]]:
    """Runs a selfplay game and returns training data generated.
    Training data is a list of tuples of the form (numpy_state, mcts_action_probs, canonical_outcome_value)"""
    mcts_simulator = mcts.Tree(c_puct=c_puct)
    state = sj.ImmutableState(
        num_players=num_players,
        player_scores=np.zeros(num_players, dtype=np.int16),
        remaining_card_counts=sj.CardCounts.create_initial_deck_counts(),
        valid_actions=[sj.SkyjoAction(sj.SkyjoActionType.START_ROUND)],
    ).apply_action(sj.SkyjoAction(sj.SkyjoActionType.START_ROUND))
    training_data = []
    episode_step = 0
    while state.winning_player is None:
        episode_step += 1
        # This is the canonical representation of the state
        # (i.e. always oriented with the current player's perspective)
        numpy_state = state.numpy()
        root_node = mcts_simulator.run(state, model, iterations=mcts_iterations)
        mcts_action_probs = root_node.sample_action_probabilities(temperature)
        policy_target = sum(
            [
                action.numpy() * prob / action.numpy().sum()
                for action, prob in mcts_action_probs.items()
            ]
        )
        training_data.append(
            # TODO: we may want to create a dataclass for this
            (numpy_state, policy_target, state.curr_player)
        )

        # Move to next state based on sampled action
        random_action = sample_action(mcts_action_probs)
        state = state.apply_action(random_action)

        # Game is over
        if state.winning_player is not None:
            outcome = skynet.GameStateValue.from_winning_player(
                state.winning_player, state.num_players
            )
            return [
                DataPoint(
                    numpy_state,
                    policy_target,
                    outcome.value_from_perspective_of(data_player),
                )
                for numpy_state, policy_target, data_player in training_data
            ]


def train(
    model: skynet.SkyNet,
    num_epochs: int,
    batches: list[DataBatch],
):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    policy_losses = []
    value_losses = []

    for _ in range(num_epochs):
        model.train()

        for batch in batches:
            torch_predicted_policy, torch_predicted_value = model(batch.states_tensor)

            policy_loss = skynet.compute_policy_loss(
                torch_predicted_policy, batch.policy_tensor
            )
            value_loss = skynet.compute_value_loss(
                torch_predicted_value, batch.values_tensor
            )
            total_loss = policy_loss + value_loss

            logging.info(
                f"value loss: {value_loss.item()} policy loss: {policy_loss.item()} total loss: {total_loss.item()}"
            )

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())

            # compute gradient and do SGD step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()


# Simple example of training from a single selfplay game
if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    example_game_state = sj.ImmutableState(
        num_players=2,
        player_scores=np.array([0, 0], dtype=np.int16),
        remaining_card_counts=sj.CardCounts.create_initial_deck_counts(),
        valid_actions=[sj.SkyjoAction(sj.SkyjoActionType.START_ROUND)],
    )
    model = skynet.SkyNet(
        spatial_input_channels=sj.CARD_TYPES,
        non_spatial_features=example_game_state.non_spatial_numpy().shape[0],
        num_players=example_game_state.num_players,
    )
    training_data = []
    start_time = time.perf_counter()
    for num_episodes in range(1):
        training_data += selfplay_game(model, 2)
    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"Finished 1 selfplay games in {duration:.2f} seconds")

    # random.shuffle(training_data)
    # batches = [
    #     DataBatch.from_data_points(
    #         [
    #             training_data[data_idx]
    #             for data_idx in np.random.randint(len(training_data), size=64)
    #         ],
    #         torch.device("cpu"),
    #     )
    #     for _ in range(len(training_data) // 64)
    # ]
    # train(model, 10, batches)
