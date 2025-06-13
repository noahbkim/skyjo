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


def create_random_clear_starting_position() -> sj.Skyjo:
    random_starting_card = np.random.randint(0, sj.CARD_SIZE)
    next_random_card = np.random.randint(0, sj.CARD_SIZE)
    game_state = create_initial_same_column_flip_game_state(
        player1_initial_flips=(random_starting_card, random_starting_card),
        player2_initial_flips=(next_random_card, next_random_card),
        top_card=random_starting_card,
    )
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


# MARK: Evaluation


def evaluate_almost_surely_winning_position(model: skynet.SkyNet):
    logging.info("Evaluating almost surely winning position")
    model.eval()
    with torch.no_grad():
        model_output = model.predict(create_almost_surely_winning_position())
        logging.debug(f"MODEL_EVALUATION:\n{model_output}")
        logging.info(f"{model_output.value_output}")
        logging.info(f"{model_output.policy_output}")
        logging.info(f"{model_output.points_output}")
    return model_output


def evaluate_almost_surely_losing_position(model: skynet.SkyNet):
    logging.info("Evaluating almost surely losing position")
    model.eval()
    with torch.no_grad():
        model_output = model.predict(create_almost_surely_losing_position())
        logging.debug(f"MODEL_EVALUATION:\n{model_output}")
        logging.info(f"{model_output.value_output}")
        logging.info(f"{model_output.policy_output}")
        logging.info(f"{model_output.points_output}")
    return model_output


def evaluate_almost_surely_winning_position_after_draw(model: skynet.SkyNet):
    logging.info("Evaluating almost surely winning position after draw")
    model.eval()
    game_state = create_almost_surely_winning_position()
    game_state = sj.apply_action(game_state, sj.MASK_DRAW)
    with torch.no_grad():
        model_output = model.predict(game_state)
        logging.debug(f"MODEL_EVALUATION:\n{model_output}")
        logging.info(f"drawn card: {sj.get_top(game_state)}")
        logging.info(f"{model_output.value_output}")
        logging.info(f"{model_output.policy_output}")
        logging.info(f"{model_output.points_output}")
    return model_output


def evaluate_almost_surely_winning_position_after_take(model: skynet.SkyNet):
    logging.info("Evaluating almost surely winning position after take")
    model.eval()
    game_state = create_almost_surely_winning_position()
    game_state = sj.apply_action(game_state, sj.MASK_TAKE)
    with torch.no_grad():
        model_output = model.predict(game_state)
        logging.debug(f"MODEL_EVALUATION:\n{model_output}")
        logging.info(f"{model_output.value_output}")
        logging.info(f"{model_output.policy_output}")
        logging.info(f"{model_output.points_output}")
    return model_output


def evaluate_almost_surely_losing_position_after_draw(model: skynet.SkyNet):
    logging.info("Evaluating almost surely losing position after draw")
    model.eval()
    game_state = create_almost_surely_losing_position()
    game_state = sj.apply_action(game_state, sj.MASK_DRAW)
    with torch.no_grad():
        model_output = model.predict(game_state)
        logging.debug(f"MODEL_EVALUATION:\n{model_output}")
        logging.info(f"drawn card: {sj.get_top(game_state)}")
        logging.info(f"{model_output.value_output}")
        logging.info(f"{model_output.policy_output}")
        logging.info(f"{model_output.points_output}")
    return model_output


def evaluate_obvious_clear_position(model: skynet.SkyNet):
    logging.info("Evaluating obvious clear position")
    model.eval()
    game_state = create_obvious_clear_position()
    with torch.no_grad():
        model_output = model.predict(game_state)
        take_model_output = model.predict(sj.apply_action(game_state, sj.MASK_TAKE))
    logging.debug(f"MODEL_EVALUATION:\n{model_output}")
    logging.info(f"{model_output.value_output}")
    logging.info(f"{model_output.policy_output}")
    logging.info(f"{model_output.points_output}")
    logging.info("Evaluating obvious clear position after take")
    logging.debug(f"MODEL_EVALUATION:\n{take_model_output}")
    logging.info(f"{take_model_output.value_output}")
    logging.info(f"{take_model_output.policy_output}")
    logging.info(f"{take_model_output.points_output}")
    return model_output


def validate_model_on_known_positions(model: skynet.SkyNet):
    _ = evaluate_almost_surely_winning_position(model)
    _ = evaluate_almost_surely_losing_position(model)
    _ = evaluate_almost_surely_winning_position_after_take(model)
    _ = evaluate_almost_surely_losing_position_after_draw(model)
    _ = evaluate_obvious_clear_position(model)


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
    validate_model_on_known_positions(model)
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
    validate_model_on_known_positions(model)
    # node = mcts.run_mcts(
    #     create_almost_surely_losing_position(),
    #     model,
    #     iterations=100,
    #     num_afterstate_outcomes=10,
    # )
    # print(node.sample_child_visit_probabilities(temperature=1.0))
