import math

import numpy as np
import pytest

import selfplay
import skyjo as sj


def test_get_skyjo_symmetries():
    """
    Tests the generation of symmetric game states and policies.
    - Sets up a board state with unique cards in row 0 of each column using flip(rotate=True).
    - Verifies the number of generated symmetries matches the function's logic.
    - Verifies correct table and policy permutation for a specific symmetry.
    """
    players = 2
    skyjo_state = sj.new(players=players)

    # --- Setup Board State using preordain and flip(rotate=True) ---
    initial_flips = {
        (0, 0): sj.CARD_P1,
        (0, 1): sj.CARD_P2,
        (0, 2): sj.CARD_P3,
        (0, 3): sj.CARD_P4,
        (1, 0): sj.CARD_P5,
        (1, 1): sj.CARD_P6,
        (1, 2): sj.CARD_P7,
        (1, 3): sj.CARD_P8,
    }
    for player, column in initial_flips.keys():
        card_value = initial_flips[(player, column)]
        skyjo_state = sj.preordain(skyjo_state, card_value)
        skyjo_state = sj.flip(
            skyjo_state,
            player,
            column,
            rotate=True,
            set_draw_or_take_action=False,
            is_turn=False,
        )
    assert sj.get_player(skyjo_state) == 0, "State did not rotate back to Player 0"
    initial_state = skyjo_state  # Keep initial state for comparison

    # --- Create Policy --- Use arange for distinct values
    policy = np.arange(sj.MASK_SIZE, dtype=np.float32)
    policy = policy / policy.sum()

    # --- Generate Symmetries ---
    symmetries = selfplay.get_skyjo_symmetries(skyjo_state, policy)

    # --- Assertions ---

    # 1. Verify the number of symmetries
    num_column_perms = math.factorial(sj.COLUMN_COUNT)
    expected_symmetries = math.comb(num_column_perms + players - 1, players)
    assert len(symmetries) == expected_symmetries, (
        f"Expected {expected_symmetries} symmetries based on combinations_with_replacement, but got {len(symmetries)}"
    )

    # 2. Verify a specific symmetry: Player 0 swaps cols 0,1; Player 1 swaps cols 1,2 and 3,0
    target_p0_perm = (1, 0, 2, 3)  # Swaps columns 0 and 1
    target_p1_perm = (2, 3, 1, 0)  # Swaps columns 1 and 2 and 3 and 0

    # Construct the expected target table
    expected_target_table = sj.get_board(initial_state).copy()
    expected_target_table[0] = expected_target_table[0][:, target_p0_perm, :]
    expected_target_table[1] = expected_target_table[1][:, target_p1_perm, :]

    # Construct the expected target policy
    expected_target_policy = selfplay.get_symmetry_policy(policy, target_p0_perm)

    # Find the expected symmetry in the results
    match_count = 0
    for sym_state, sym_policy in symmetries:
        # Check table and policy match the constructed targets
        if np.array_equal(
            sj.get_board(sym_state), expected_target_table
        ) and np.array_equal(sym_policy, expected_target_policy):
            match_count += 1

    assert match_count == 1, (
        f"Found {match_count} matches for the target symmetry, expected exactly 1"
    )


def test_get_symmetry_policy():
    """
    Tests the generation of symmetric policies.
    - Sets up a policy with distinct values.
    - Verifies the number of generated symmetries matches the function's logic.
    - Verifies correct policy permutation for a specific symmetry.
    """
    original_policy = np.arange(sj.MASK_SIZE, dtype=np.float32)
    original_policy = original_policy / original_policy.sum()

    column_order = (1, 0, 2, 3)
    column_order_map = {
        original_column_index: new_column_index
        for original_column_index, new_column_index in enumerate(column_order)
    }
    actual = selfplay.get_symmetry_policy(original_policy, column_order)

    expected = original_policy.copy()
    for original_row in range(sj.ROW_COUNT):
        for original_column in range(sj.COLUMN_COUNT):
            new_column_index = column_order_map[original_column]
            # Flip policy
            expected[
                sj.MASK_FLIP + original_row * sj.COLUMN_COUNT + original_column
            ] = original_policy[
                sj.MASK_FLIP + original_row * sj.COLUMN_COUNT + new_column_index
            ]
            # Replace policy
            expected[
                sj.MASK_REPLACE + original_row * sj.COLUMN_COUNT + original_column
            ] = original_policy[
                sj.MASK_REPLACE + original_row * sj.COLUMN_COUNT + new_column_index
            ]
    assert np.allclose(actual[: sj.MASK_FLIP], original_policy[: sj.MASK_FLIP]), (
        "Action policy prior to flip and replace should be the same"
    )
    assert np.allclose(
        actual[sj.MASK_FLIP : sj.MASK_FLIP + sj.FINGER_COUNT],
        expected[sj.MASK_FLIP : sj.MASK_FLIP + sj.FINGER_COUNT],
    ), "Flip policy does not match expected symmetric policy"
    assert np.allclose(
        actual[sj.MASK_REPLACE : sj.MASK_REPLACE + sj.FINGER_COUNT],
        expected[sj.MASK_REPLACE : sj.MASK_REPLACE + sj.FINGER_COUNT],
    ), "Replace policy does not match expected symmetric policy"
    assert np.allclose(actual, expected), (
        "Symmetric policy does not match expected policy"
    )


if __name__ == "__main__":
    pytest.main([__file__])
