"""Handcrafted positions and standalone checks of learned Skyjo concepts.

The target values are heuristic round-level expectations, not calibrated full-game
win probabilities. These examples are qualitative diagnostics evaluated periodically by the pool
recipe; their heuristic labels never become replay targets.
"""

import dataclasses
import logging

import numpy as np
import torch

from skyjo.learning import checkpoint
from skyjo.learning import losses
from skyjo.learning import skynet, batches
from skyjo.engine import game as sj

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
    game_state = sj.new(players=2, top=top_card)
    game_state = sj.preordain(game_state, player1_initial_flips[0])
    game_state = sj.reveal_second_card(game_state, 0, 0)
    game_state = sj.preordain(game_state, player2_initial_flips[0])
    game_state = sj.reveal_second_card(game_state, 0, 0)
    game_state = sj.preordain(game_state, player1_initial_flips[1])
    game_state = sj.reveal_second_card(game_state, 0, 1)
    game_state = sj.preordain(game_state, player2_initial_flips[1])
    game_state = sj.reveal_second_card(game_state, 0, 1)
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
    game_state = sj.new(players=2, top=top_card)
    game_state = sj.preordain(game_state, player1_initial_flips[0])
    game_state = sj.reveal_second_card(game_state, 0, 0)
    game_state = sj.preordain(game_state, player2_initial_flips[0])
    game_state = sj.reveal_second_card(game_state, 0, 0)
    game_state = sj.preordain(game_state, player1_initial_flips[1])
    game_state = sj.reveal_second_card(game_state, 1, 0)
    game_state = sj.preordain(game_state, player2_initial_flips[1])
    game_state = sj.reveal_second_card(game_state, 1, 0)
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
    random_starting_card = np.random.randint(0, sj.CARD_SIZE)
    second_random_starting_card = np.random.randint(0, sj.CARD_SIZE)
    game_state = create_initial_same_column_flip_game_state(
        player1_initial_flips=(random_starting_card, random_starting_card),
        player2_initial_flips=(
            second_random_starting_card,
            second_random_starting_card,
        ),
        top_card=random_starting_card,
    )
    for i in np.random.choice(12, size=np.random.randint(5), replace=False):
        if i not in [0, 4]:
            row, col = divmod(i, 4)
            game_state = sj.randomize(game_state)
            game_state = sj.flip(game_state, row, col)
            game_state = sj.randomize(game_state)
            game_state = sj.flip(game_state, row, col)
    return game_state


def create_random_almost_clear_position() -> sj.Skyjo:
    random_starting_card = np.random.randint(0, sj.CARD_SIZE)
    second_random_starting_card = np.random.randint(0, sj.CARD_SIZE)
    game_state = create_initial_same_column_flip_game_state(
        player1_initial_flips=(random_starting_card, random_starting_card),
        player2_initial_flips=(
            second_random_starting_card,
            second_random_starting_card,
        ),
        top_card=np.random.randint(0, sj.CARD_SIZE),
    )
    for i in np.random.choice(12, size=np.random.randint(5), replace=False):
        if i not in [0, 4]:
            row, col = divmod(i, 4)
            game_state = sj.randomize(game_state)
            game_state = sj.flip(game_state, row, col)
            game_state = sj.randomize(game_state)
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


def create_early_flip_position() -> sj.Skyjo:
    game_state = create_initial_seperate_column_flip_game_state(
        player1_initial_flips=(sj.CARD_P12, sj.CARD_P11),
        player2_initial_flips=(sj.CARD_P5, sj.CARD_P5),
        top_card=sj.CARD_P8,
    )
    game_state = sj.preordain(game_state, sj.CARD_P10)
    game_state = sj.draw(game_state)
    return game_state


def create_negative_clear_position() -> sj.Skyjo:
    game_state = create_initial_same_column_flip_game_state(
        player1_initial_flips=(sj.CARD_N2, sj.CARD_N2),
        player2_initial_flips=(sj.CARD_N1, sj.CARD_N1),
        top_card=sj.CARD_N2,
    )
    game_state = sj.preordain(game_state, sj.CARD_P8)
    game_state = sj.flip(game_state, 0, 1)

    game_state = sj.preordain(game_state, sj.CARD_P9)
    game_state = sj.flip(game_state, 0, 1)
    return game_state


def create_negative_clear_take_position() -> sj.Skyjo:
    game_state = create_negative_clear_position()
    game_state = sj.apply_action(game_state, sj.MASK_TAKE)
    return game_state


def create_potential_clear_equal_position(top_card: int = sj.CARD_P10) -> sj.Skyjo:
    game_state = create_initial_same_column_flip_game_state(
        player1_initial_flips=(sj.CARD_P10, sj.CARD_P10),
        player2_initial_flips=(sj.CARD_P9, sj.CARD_P9),
        top_card=top_card,
    )
    game_state = sj.preordain(game_state, sj.CARD_P5)
    game_state = sj.flip(game_state, 0, 1)
    game_state = sj.preordain(game_state, sj.CARD_P6)
    game_state = sj.flip(game_state, 0, 1)
    game_state = sj.preordain(game_state, sj.CARD_N2)
    game_state = sj.flip(game_state, 0, 2)
    game_state = sj.preordain(game_state, sj.CARD_N1)
    game_state = sj.flip(game_state, 0, 2)
    game_state = sj.preordain(game_state, sj.CARD_P8)
    game_state = sj.flip(game_state, 0, 3)
    game_state = sj.preordain(game_state, sj.CARD_P7)
    game_state = sj.flip(game_state, 0, 3)
    game_state = sj.preordain(game_state, sj.CARD_N1)
    game_state = sj.flip(game_state, 1, 1)
    game_state = sj.preordain(game_state, sj.CARD_0)
    game_state = sj.flip(game_state, 1, 1)
    game_state = sj.preordain(game_state, sj.CARD_P4)
    game_state = sj.flip(game_state, 1, 2)
    game_state = sj.preordain(game_state, sj.CARD_N1)
    game_state = sj.flip(game_state, 1, 2)
    game_state = sj.preordain(game_state, sj.CARD_N1)
    game_state = sj.flip(game_state, 2, 1)
    game_state = sj.preordain(game_state, sj.CARD_P3)
    game_state = sj.flip(game_state, 2, 1)
    game_state = sj.preordain(game_state, sj.CARD_P2)
    game_state = sj.flip(game_state, 2, 2)
    game_state = sj.preordain(game_state, sj.CARD_P1)
    game_state = sj.flip(game_state, 2, 2)
    game_state = sj.preordain(game_state, sj.CARD_P5)
    game_state = sj.flip(game_state, 2, 0)
    game_state = sj.preordain(game_state, sj.CARD_P5)
    game_state = sj.flip(game_state, 2, 0)
    return game_state


# MARK: Targets


def almost_surely_winning_position_targets():
    value_target = np.array([1.0, 0.0], dtype=np.float32)
    policy_target = np.zeros([sj.MASK_SIZE], dtype=np.float32)
    policy_target[sj.MASK_TAKE] = 1.0
    return {"value": value_target, "policy": policy_target}


def almost_surely_winning_take_position_targets():
    value_target = np.array([1.0, 0.0], dtype=np.float32)
    policy_target = np.zeros([sj.MASK_SIZE], dtype=np.float32)
    policy_target[sj.MASK_REPLACE + 11] = 1.0
    return {"value": value_target, "policy": policy_target}


def almost_surely_losing_position_targets():
    value_target = np.array([0.0, 1.0], dtype=np.float32)
    policy_target = np.zeros([sj.MASK_SIZE], dtype=np.float32)
    policy_target[sj.MASK_DRAW] = 1.0
    return {"value": value_target, "policy": policy_target}


def obvious_clear_position_targets():
    value_target = np.array([0.7, 0.3], dtype=np.float32)
    policy_target = np.zeros([sj.MASK_SIZE], dtype=np.float32)
    policy_target[sj.MASK_TAKE] = 1.0
    return {"value": value_target, "policy": policy_target}


def obvious_clear_take_position_targets():
    value_target = np.array([0.7, 0.3], dtype=np.float32)
    policy_target = np.zeros([sj.MASK_SIZE], dtype=np.float32)
    policy_target[sj.MASK_REPLACE + 8] = 1.0
    return {"value": value_target, "policy": policy_target}


def almost_clear_position_targets():
    value_target = np.array([0.55, 0.45], dtype=np.float32)
    policy_target = np.zeros([sj.MASK_SIZE], dtype=np.float32)
    policy_target[sj.MASK_DRAW] = 1.0
    return {"value": value_target, "policy": policy_target}


def almost_clear_draw_low_position_targets():
    value_target = np.array([0.6, 0.4], dtype=np.float32)
    policy_target = np.zeros([sj.MASK_SIZE], dtype=np.float32)
    policy_target[sj.MASK_REPLACE + 1] = 1
    return {"value": value_target, "policy": policy_target}


def early_flip_position_targets():
    value_target = np.array([0.3, 0.7], dtype=np.float32)
    policy_target = np.zeros([sj.MASK_SIZE], dtype=np.float32)
    policy_target[sj.MASK_FLIP + 2 : sj.MASK_FLIP + 12] = 1 / 10
    return {"value": value_target, "policy": policy_target}


def negative_clear_position_targets():
    value_target = np.array([0.6, 0.4], dtype=np.float32)
    policy_target = np.zeros([sj.MASK_SIZE], dtype=np.float32)
    policy_target[sj.MASK_TAKE] = 1
    return {"value": value_target, "policy": policy_target}


def negative_clear_take_position_targets():
    value_target = np.array([0.6, 0.4], dtype=np.float32)
    policy_target = np.zeros([sj.MASK_SIZE], dtype=np.float32)
    policy_target[sj.MASK_REPLACE + 1] = 1
    return {"value": value_target, "policy": policy_target}


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
        "early flip position",
        create_early_flip_position(),
        early_flip_position_targets(),
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
    (
        "negative clear position",
        create_negative_clear_position(),
        negative_clear_position_targets(),
    ),
    (
        "negative clear take position",
        create_negative_clear_take_position(),
        negative_clear_take_position_targets(),
    ),
]


@dataclasses.dataclass(frozen=True)
class ConceptReport:
    examples: list[dict]

    def summary(self) -> dict[str, float | int]:
        return {
            "example_count": len(self.examples),
            "target_action_matches": sum(
                e["target_action_match"] for e in self.examples
            ),
            "mean_target_probability": float(
                np.mean([e["target_probability"] for e in self.examples])
            ),
            "heuristic_value_mse": float(
                np.mean([e["value_loss"] for e in self.examples])
            ),
            "policy_loss": float(np.mean([e["policy_loss"] for e in self.examples])),
        }


@torch.inference_mode()
def evaluate_concepts(model: skynet.SkyNet) -> ConceptReport:
    """Inspect heuristic positions without perturbing training mode or RNG streams."""
    rng = checkpoint.capture_rng_state()
    modes = [(module, module.training) for module in model.modules()]
    examples = []
    try:
        model.eval()
        for description, state, targets in VALIDATION_EXAMPLES:
            inputs = batches.to_tensors(
                batches.states_to_batch([state]), device=model.device
            )
            output = model(
                inputs.spatial_inputs, inputs.non_spatial_inputs, inputs.action_masks
            )
            policy = output.policy_logits[0].softmax(-1).detach().cpu().numpy()
            value = output.value[0].detach().cpu().numpy()
            tensor_targets = {
                "value": torch.tensor(
                    targets["value"][None, :], dtype=torch.float32, device=model.device
                ),
                "policy": torch.tensor(
                    targets["policy"][None, :], dtype=torch.float32, device=model.device
                ),
            }
            value_loss, policy_loss = losses.policy_value_losses(output, tensor_targets)
            action = int(policy.argmax())
            examples.append(
                {
                    "name": description,
                    "value": value.tolist(),
                    "policy": policy.tolist(),
                    "value_target": targets["value"].tolist(),
                    "policy_target": targets["policy"].tolist(),
                    "preferred_action": action,
                    "preferred_action_name": sj.get_action_name(action),
                    "target_probability": float(policy[targets["policy"] > 0].sum()),
                    "target_action_match": bool(targets["policy"][action] > 0),
                    "value_loss": value_loss.item(),
                    "policy_loss": policy_loss.item(),
                }
            )
        return ConceptReport(examples)
    finally:
        for module, training in modes:
            module.training = training
        checkpoint.restore_rng_state(rng)


def validate_model_on_validation_examples(
    model: skynet.SkyNet,
    value_loss_scale: float = 1.0,
    policy_loss_scale: float = 1.0,
) -> dict[str, float]:
    """Verbose standalone rendering of the heuristic concept suite."""
    report = evaluate_concepts(model)
    summary = report.summary()
    metrics = {
        "value_loss": value_loss_scale * summary["heuristic_value_mse"],
        "policy_loss": policy_loss_scale * summary["policy_loss"],
        "example_count": summary["example_count"],
    }
    logging.info(
        "[VALIDATION] Heuristic concept checks (not calibrated full-game validation): %s",
        metrics,
    )
    for example in report.examples:
        logging.info("[VALIDATION] %s", example)
    return metrics
