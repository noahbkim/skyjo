import numpy as np
import pytest

import skyjo as sj


@pytest.mark.parametrize("players", [2, 3, 4, 5, 6, 7, 8])
def test_new_skyjo(players: int):
    (
        game_state,
        table_state,
        deck_state,
        num_players,
        turn_count,
        current_card,
        countdown,
    ) = sj.new(players=players)

    # Assert game state
    assert game_state.shape == (sj.GAME_SIZE,)
    assert game_state[sj.GAME_ACTION + sj.ACTION_FLIP_SECOND] == 1
    assert (
        np.sum(game_state[sj.GAME_TOP : sj.GAME_TOP + sj.CARD_SIZE]) == 0
    )  # No top card initially
    assert (
        np.sum(game_state[sj.GAME_DISCARDS : sj.GAME_DISCARDS + sj.CARD_SIZE]) == 0
    )  # No discards initially
    assert np.all(
        game_state[sj.GAME_SCORES : sj.GAME_SCORES + sj.PLAYER_COUNT] == 0
    )  # Scores are 0
    assert np.all(
        game_state[
            sj.GAME_LAST_REVEALED_TURNS : sj.GAME_LAST_REVEALED_TURNS + sj.PLAYER_COUNT
        ]
        == 0
    )  # Last revealed turns are 0

    # Assert table state
    expected_table_shape = (
        sj.PLAYER_COUNT,
        sj.ROW_COUNT,
        sj.COLUMN_COUNT,
        sj.FINGER_SIZE,
    )
    assert table_state.shape == expected_table_shape
    assert np.all(
        table_state[:players, :, :, sj.FINGER_HIDDEN] == 1
    )  # All player cards are hidden
    assert np.sum(table_state[players:]) == 0  # Unused player slots are empty

    # Assert deck state
    assert deck_state.shape == (sj.CARD_SIZE,)
    assert np.array_equal(deck_state, np.array(sj.CARD_COUNTS, dtype=np.int16))

    # Assert other parameters
    assert num_players == players
    assert turn_count == 0
    assert current_card is None
    assert countdown is None

    # Validate overall consistency
    assert sj.validate(
        (
            game_state,
            table_state,
            deck_state,
            num_players,
            turn_count,
            current_card,
            countdown,
        )
    )


@pytest.mark.parametrize("players", [2, 3, 4, 5, 6, 7, 8])
def test_no_progress_rule_ends_game_and_doubles_scores(players: int):
    s = sj.new(players=players)

    # Start round (flips (0,0) for all, sets up discard, action remains FLIP_SECOND)
    # P0 (original) is current player after this.
    s = sj.start_round(s)

    # All players flip their second card (1,0 using MASK_FLIP_SECOND_BELOW)
    # The last of these calls will trigger begin() internally, setting action to DRAW_OR_TAKE for P0 (original).
    for _ in range(players):
        s = sj.apply_action(
            s, sj.MASK_FLIP_SECOND_BELOW
        )  # Flips (1,0) for current player

    # Simulate (NO_PROGRESS_TURN_THRESHOLD - 1) full rounds of no-progress actions
    # Each player takes discard and replaces card (0,0) which is already face-up
    for _ in range(sj.NO_PROGRESS_TURN_THRESHOLD - 1):
        for _ in range(players):
            s = sj.apply_action(s, sj.MASK_TAKE)
            s = sj.apply_action(s, sj.MASK_REPLACE + 0)  # Replace (0,0)
            assert sj.get_countdown(s) is None  # Countdown should not be set yet

    # Player 0 (original, current player) is about to take a turn.
    # They have made (NO_PROGRESS_TURN_THRESHOLD - 1) no-progress rounds for themselves.
    # Their next action (flipping a new card) should trigger the countdown because the no-progress
    # check uses their Last Revealed Turn *before* this revealing flip.

    # Simulate remaining turns for the other (players - 1) players.
    # Each takes two actions, consuming the countdown.
    for i in range(players - 1):  # For each of the other players
        s = sj.apply_action(s, sj.MASK_TAKE)
        s = sj.apply_action(s, sj.MASK_REPLACE + 0)  # Replace (0,0)
        expected_countdown = (players - 1) * 2 - i * 2
        assert sj.get_countdown(s) == expected_countdown, (
            f"Countdown started: actual {sj.get_countdown(s)}, expected {expected_countdown}"
        )

    s = sj.apply_action(s, sj.MASK_DRAW)  # Player 0 (original) draws
    s = sj.apply_action(s, sj.MASK_FLIP + 1)

    # Game should be over. The last apply_action should have triggered end_round.
    assert sj.get_countdown(s) == 0, (
        f"Final countdown: actual {sj.get_countdown(s)}, expected 0"
    )
    assert sj.get_game_over(s)

    final_scores = sj.get_round_scores(s, round_ending_player=0)
    # After end_round (called by apply_action), all cards are visible. get_score gives the sum of card values.
    base_scores = np.array(
        [sj.get_score(s, player=i) for i in range(players)], dtype=np.int16
    )

    expected_scores = base_scores * 2
    expected_scores[-1] = base_scores[-1]

    assert np.array_equal(final_scores, expected_scores)
