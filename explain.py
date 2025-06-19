"""Module to evaluate and understand current model behaivor"""

import logging

import numpy as np
import torch

import skyjo as sj
import skynet
import train_utils

# MARK: Game state creation


def create_initial_seperate_column_flip_game_state(
    player1_initial_flips: tuple[int, int] = (sj.CARD_P5, sj.CARD_P5),
    player2_initial_flips: tuple[int, int] = (sj.CARD_P5, sj.CARD_P5),
    top_card: int = sj.CARD_P5,
):
    """Create two player skyjo game with different column initial flips"""
    assert len(player1_initial_flips) == 2, (
        f"Please specify exactly two cards to be initially flipped for player 1, got {player1_initial_flips}"
    )
    assert len(player2_initial_flips) == 2, (
        f"Please specify exactly two cards to be initially flipped for player 2, got {player2_initial_flips}"
    )
    game_state = sj.new(players=2)
    game_state = sj.preordain(game_state, player1_initial_flips[0])
    game_state = sj.flip(game_state, 0, 0, set_draw_or_take_action=False)
    game_state = sj.preordain(game_state, player2_initial_flips[0])
    game_state = sj.flip(game_state, 0, 0, set_draw_or_take_action=False)
    game_state = sj.preordain(game_state, player1_initial_flips[1])
    game_state = sj.flip(game_state, 0, 1, set_draw_or_take_action=False)
    game_state = sj.preordain(game_state, player2_initial_flips[1])
    game_state = sj.flip(game_state, 0, 1, set_draw_or_take_action=False)
    game_state = sj.preordain(game_state, top_card)
    game_state = sj.begin(game_state)
    return game_state


def create_initial_same_column_flip_game_state(
    player1_initial_flips: tuple[int, int] = (sj.CARD_P5, sj.CARD_P5),
    player2_initial_flips: tuple[int, int] = (sj.CARD_P5, sj.CARD_P5),
    top_card: int = sj.CARD_P5,
):
    """Create two player skyjo game with different column initial flips"""
    assert len(player1_initial_flips) == 2, (
        f"Please specify exactly two cards to be initially flipped for player 1, got {player1_initial_flips}"
    )
    assert len(player2_initial_flips) == 2, (
        f"Please specify exactly two cards to be initially flipped for player 2, got {player2_initial_flips}"
    )
    game_state = sj.new(players=2)
    game_state = sj.preordain(game_state, player1_initial_flips[0])
    game_state = sj.flip(game_state, 0, 0, set_draw_or_take_action=False)
    game_state = sj.preordain(game_state, player2_initial_flips[0])
    game_state = sj.flip(game_state, 0, 0, set_draw_or_take_action=False)
    game_state = sj.preordain(game_state, player1_initial_flips[1])
    game_state = sj.flip(game_state, 1, 0, set_draw_or_take_action=False)
    game_state = sj.preordain(game_state, player2_initial_flips[1])
    game_state = sj.flip(game_state, 1, 0, set_draw_or_take_action=False)
    game_state = sj.preordain(game_state, top_card)
    game_state = sj.begin(game_state)
    return game_state


def create_almost_surely_winning_position() -> sj.Skyjo:
    """Creates an almost surely winning (current player perspective) position."""
    game_state = create_initial_seperate_column_flip_game_state(
        player1_initial_flips=(sj.CARD_P5, sj.CARD_P5),
        player2_initial_flips=(sj.CARD_P5, sj.CARD_P5),
        top_card=sj.CARD_0,
    )
    for i in range(2, 11):
        row, col = divmod(i, 4)
        game_state = sj.preordain(game_state, row)
        game_state = sj.flip(game_state, row, col)
        game_state = sj.preordain(game_state, row + sj.CARD_SIZE - sj.ROW_COUNT)
        game_state = sj.flip(game_state, row, col)
    return game_state


def create_almost_surely_losing_position() -> sj.Skyjo:
    """Returns an almost surely losing (current player perspective) game state.

        Returned game state has one face-down card remaining for both players.
    Current player will have every other card"""
    game_state = create_initial_seperate_column_flip_game_state(
        player1_initial_flips=(sj.CARD_P5, sj.CARD_P5),
        player2_initial_flips=(sj.CARD_P5, sj.CARD_P5),
        top_card=sj.CARD_P12,
    )
    for i in range(2, 11):
        row, col = divmod(i, 4)
        game_state = sj.preordain(game_state, row + sj.CARD_SIZE - sj.ROW_COUNT - 1)
        game_state = sj.flip(game_state, row, col)
        game_state = sj.preordain(game_state, row)
        game_state = sj.flip(game_state, row, col)
    return game_state


def create_obvious_take_and_clear_position() -> sj.Skyjo:
    """Creates a position where the current player can replace a card with a higher value card."""
    game_state = create_initial_seperate_column_flip_game_state(
        player1_initial_flips=(sj.CARD_P5, sj.CARD_P5),
        player2_initial_flips=(sj.CARD_P5, sj.CARD_P5),
        top_card=sj.CARD_P12,
    )
    for i in range(2, 7):
        row, col = divmod(i, sj.COLUMN_COUNT)
        game_state = sj.preordain(game_state, sj.CARD_P12)
        game_state = sj.flip(game_state, row, col)
        game_state = sj.preordain(game_state, sj.CARD_P11)
        game_state = sj.flip(game_state, row, col)
    return game_state


def create_obvious_take_position() -> sj.Skyjo:
    """Creates a position where the current player can replace a card with a higher value card."""
    game_state = create_initial_seperate_column_flip_game_state(
        player1_initial_flips=(sj.CARD_P5, sj.CARD_P5),
        player2_initial_flips=(sj.CARD_P5, sj.CARD_P5),
        top_card=sj.CARD_N2,
    )
    for i in range(2, 7):
        row, col = divmod(i, sj.COLUMN_COUNT)
        game_state = sj.preordain(game_state, sj.CARD_P12)
        game_state = sj.flip(game_state, row, col)
        game_state = sj.preordain(game_state, sj.CARD_P12)
        game_state = sj.flip(game_state, row, col)
    return game_state


def create_obvious_draw_position() -> sj.Skyjo:
    """Creates a position where the current player can replace a card with a higher value card."""
    game_state = create_initial_seperate_column_flip_game_state(
        player1_initial_flips=(sj.CARD_P5, sj.CARD_P5),
        player2_initial_flips=(sj.CARD_P5, sj.CARD_P5),
        top_card=sj.CARD_P12,
    )
    for i in range(2, 7):
        row, col = divmod(i, sj.COLUMN_COUNT)
        game_state = sj.preordain(game_state, sj.CARD_P10)
        game_state = sj.flip(game_state, row, col)
        game_state = sj.preordain(game_state, sj.CARD_P11)
        game_state = sj.flip(game_state, row, col)
    return game_state


def create_obvious_clear_position() -> sj.Skyjo:
    """Creates a position where the current player can replace a card with a higher value card."""
    game_state = create_initial_seperate_column_flip_game_state(
        player1_initial_flips=(sj.CARD_P10, sj.CARD_P11),
        player2_initial_flips=(sj.CARD_P11, sj.CARD_P12),
        top_card=sj.CARD_P10,
    )
    game_state = sj.preordain(game_state, sj.CARD_P10)
    game_state = sj.flip(game_state, 1, 0)
    game_state = sj.preordain(game_state, sj.CARD_P11)
    game_state = sj.flip(game_state, 1, 0)

    # game_state = sj.preordain(game_state, sj.CARD_P10)
    # game_state = sj.apply_action(game_state, sj.MASK_DRAW)
    return game_state


def create_almost_clear_position() -> sj.Skyjo:
    """Creates a position where the current player can replace a card with a higher value card."""
    game_state = create_initial_seperate_column_flip_game_state(
        player1_initial_flips=(sj.CARD_P10, sj.CARD_P11),
        player2_initial_flips=(sj.CARD_P11, sj.CARD_P12),
        top_card=sj.CARD_P11,
    )
    game_state = sj.preordain(game_state, sj.CARD_P10)
    game_state = sj.flip(game_state, 1, 0)
    game_state = sj.preordain(game_state, sj.CARD_P11)
    game_state = sj.flip(game_state, 1, 0)

    # game_state = sj.preordain(game_state, sj.CARD_P10)
    # game_state = sj.apply_action(game_state, sj.MASK_DRAW)
    return game_state


def create_almost_clear_draw_low_position() -> sj.Skyjo:
    game_state = create_almost_clear_position()
    return sj.draw(sj.preordain(game_state, sj.CARD_P1))


def create_random_clear_starting_position() -> sj.Skyjo:
    random_starting_cards = np.random.randint(0, sj.CARD_SIZE, size=5)
    game_state = create_initial_same_column_flip_game_state(
        player1_initial_flips=(random_starting_cards[0], random_starting_cards[1]),
        player2_initial_flips=(random_starting_cards[2], random_starting_cards[3]),
        top_card=random_starting_cards[4],
    )
    for i in range(5, 5 + np.random.randint(5)):
        row, col = divmod(i, 4)
        game_state = sj.preordain(game_state, np.random.randint(0, sj.CARD_SIZE))
        game_state = sj.flip(game_state, row, col)
        game_state = sj.preordain(game_state, np.random.randint(0, sj.CARD_SIZE))
        game_state = sj.flip(game_state, row, col)
    return game_state


def create_close_end_game_position() -> sj.Skyjo:
    game_state = create_initial_seperate_column_flip_game_state(
        player1_initial_flips=(sj.CARD_P6, sj.CARD_P7),
        player2_initial_flips=(sj.CARD_P6, sj.CARD_P7),
        top_card=sj.CARD_P5,
    )
    for i in range(2, 11):
        row, col = divmod(i, 4)
        game_state = sj.preordain(game_state, row + 2 + 3)
        game_state = sj.flip(game_state, row, col)
        game_state = sj.preordain(game_state, row + 2 + 3)
        game_state = sj.flip(game_state, row, col)
    return game_state


# MARK: Targets


def almost_surely_winning_position_targets():
    value_target = np.array([1.0, 0.0], dtype=np.float32)
    policy_target = np.zeros([sj.MASK_SIZE], dtype=np.float32)
    policy_target[sj.MASK_TAKE] = 1.0
    points_target = None
    return value_target, points_target, policy_target


def almost_surely_winning_take_position_targets():
    value_target = np.array([1.0, 0.0], dtype=np.float32)
    policy_target = np.zeros([sj.MASK_SIZE], dtype=np.float32)
    policy_target[sj.MASK_REPLACE + 11] = 1.0
    points_target = None
    return value_target, points_target, policy_target


def almost_surely_losing_position_targets():
    value_target = np.array([0.0, 1.0], dtype=np.float32)
    policy_target = np.zeros([sj.MASK_SIZE], dtype=np.float32)
    policy_target[sj.MASK_DRAW] = 1.0
    points_target = None
    return value_target, points_target, policy_target


def obvious_clear_position_targets():
    value_target = np.array([0.8, 0.2], dtype=np.float32)
    policy_target = np.zeros([sj.MASK_SIZE], dtype=np.float32)
    policy_target[sj.MASK_TAKE] = 1.0
    points_target = None
    return value_target, points_target, policy_target


def obvious_clear_take_position_targets():
    value_target = np.array([0.8, 0.2], dtype=np.float32)
    policy_target = np.zeros([sj.MASK_SIZE], dtype=np.float32)
    policy_target[sj.MASK_REPLACE + 8] = 1.0
    points_target = None
    return value_target, points_target, policy_target


def almost_clear_position_targets():
    value_target = np.array([0.55, 0.45], dtype=np.float32)
    policy_target = np.zeros([sj.MASK_SIZE], dtype=np.float32)
    policy_target[sj.MASK_DRAW] = 1.0
    points_target = None
    return value_target, points_target, policy_target


def almost_clear_draw_low_position_targets():
    value_target = np.array([0.6, 0.4], dtype=np.float32)
    policy_target = np.zeros([sj.MASK_SIZE], dtype=np.float32)
    policy_target[sj.MASK_REPLACE + 3] = 0.33
    policy_target[sj.MASK_REPLACE + 4] = 0.33
    policy_target[sj.MASK_REPLACE + 5] = 0.34
    points_target = None
    return value_target, points_target, policy_target


# MARK: Evaluation

VALIDATION_EXAMPLES = [
    (
        "almost surely winning position",
        create_almost_surely_winning_position(),
        almost_surely_winning_position_targets(),
    ),
    (
        "almost surely winning after take",
        sj.apply_action(create_almost_surely_winning_position(), sj.MASK_TAKE),
        almost_surely_winning_take_position_targets(),
    ),
    (
        "almost surely losing position",
        create_almost_surely_losing_position(),
        almost_surely_losing_position_targets(),
    ),
    (
        "obvious clear position",
        create_obvious_clear_position(),
        obvious_clear_position_targets(),
    ),
    (
        "obvious clear take position",
        sj.apply_action(create_obvious_clear_position(), sj.MASK_TAKE),
        obvious_clear_take_position_targets(),
    ),
    (
        "almost clear position",
        create_almost_clear_position(),
        almost_clear_position_targets(),
    ),
    (
        "leave clear option open",
        create_almost_clear_draw_low_position(),
        almost_clear_draw_low_position_targets(),
    ),
]


def validate_model_on_validation_examples(
    model: skynet.SkyNet,
):
    game_data = []
    for description, game_state, targets in VALIDATION_EXAMPLES:
        game_data.append(
            (
                game_state,
                targets[0],
                targets[1],
                targets[2],
            )
        )
    value_loss, policy_loss = train_utils.compute_model_loss_on_game_data(
        model, game_data, train_utils.policy_value_losses
    )
    logging.info("VALIDATION SET LOSS")
    logging.info(f"value loss: {value_loss.item()}")
    logging.info(f"policy loss: {policy_loss.item()}")

    logging.info("INDIVIDUAL EXAMPLES")
    for description, game_state, targets in VALIDATION_EXAMPLES:
        model_prediction = model.predict(game_state)
        tensor_targets = (
            torch.tensor(np.expand_dims(targets[0], 0), dtype=torch.float32),
            torch.tensor(np.expand_dims(targets[1], 0), dtype=torch.float32)
            if targets[1] is not None
            else None,
            torch.tensor(np.expand_dims(targets[2], 0), dtype=torch.float32),
        )
        value_loss, policy_loss = train_utils.policy_value_losses(
            model_prediction.to_output(), tensor_targets
        )
        logging.info(f"validation example: {description}")
        logging.info(f"value loss: {value_loss.item()}")
        logging.info(f"policy loss: {policy_loss.item()}")
        logging.info(f"model prediction:\n{model_prediction}")
        logging.info(f"value target: {targets[0]}")
        logging.info(f"points target: {targets[1]}")
        logging.info(f"policy target:\n{targets[2]}")


def validate_model_with_games_data(
    model: skynet.SkyNet,
    validation_batch: train_utils.TrainingBatch,
    value_loss_scale: float = 3.0,
):
    model.eval()
    with torch.no_grad():
        (
            spatial_inputs,
            non_spatial_inputs,
            masks,
            value_targets,
            points_targets,
            policy_targets,
        ) = validation_batch
        spatial_inputs_tensor = torch.tensor(
            spatial_inputs, dtype=torch.float32, device=model.device
        )
        non_spatial_inputs_tensor = torch.tensor(
            non_spatial_inputs, dtype=torch.float32, device=model.device
        )
        masks_tensor = torch.tensor(masks, dtype=torch.float32, device=model.device)
        value_targets_tensor = torch.tensor(
            value_targets, dtype=torch.float32, device=model.device
        )
        policy_targets_tensor = torch.tensor(
            policy_targets, dtype=torch.float32, device=model.device
        )
        model_output = model(
            spatial_inputs_tensor, non_spatial_inputs_tensor, masks_tensor
        )
        value_loss, policy_loss = train_utils.policy_value_losses(
            model_output, (value_targets_tensor, None, policy_targets_tensor)
        )
        total_loss = value_loss_scale * value_loss + policy_loss
        base_policy_entropies = -(
            policy_targets_tensor * torch.log(policy_targets_tensor + 1e-12)
        ).sum(dim=1)
        logging.info(
            f"value loss: {value_loss_scale * value_loss.item()} "
            # f"points loss: {points_loss_scale * points_loss.item()} "
            f"policy loss: {policy_loss.item()} "
            f"policy entropy: {base_policy_entropies.mean()} "
            f"total loss: {total_loss.item()} "
        )
        return total_loss


def validate_model(
    model: skynet.SkyNet,
    validation_batch: train_utils.TrainingBatch | None = None,
):
    validate_model_on_validation_examples(model)
    if validation_batch is not None:
        validate_model_with_games_data(model, validation_batch)


if __name__ == "__main__":
    import pathlib

    logging.basicConfig(level=logging.INFO)
    model = skynet.SimpleSkyNet(
        spatial_input_shape=(2, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
        non_spatial_input_shape=(sj.GAME_SIZE,),
        value_output_shape=(2,),
        policy_output_shape=(sj.MASK_SIZE,),
        hidden_layers=[64, 64],
        device=torch.device("cpu"),
    )
    saved_model_path = pathlib.Path("./models/model_20250423_141907.pth")
    model.load_state_dict(torch.load(saved_model_path, weights_only=True))
    validate_model_on_validation_examples(model)
    # node = mcts.run_mcts(
    #     create_almost_surely_losing_position(),
    #     model,
    #     iterations=100,
    #     num_afterstate_outcomes=10,
    # )
    # print(node.sample_child_visit_probabilities(temperature=1.0))
