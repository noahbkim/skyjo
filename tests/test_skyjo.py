import dataclasses

import numpy as np
import pytest

from skyjo.engine import game as sj


@pytest.mark.parametrize("players", [2, 3, 4, 5, 6, 7, 8])
def test_new_skyjo(players: int):
    initial = sj.new(players=players, top=sj.CARD_P8)
    game_state = initial.game
    table_state = initial.table
    deck_state = initial.deck
    num_players = initial.players
    turn_count = initial.turn
    current_card = initial.pending_card
    countdown = initial.countdown

    # Assert game state
    assert game_state.shape == (sj.GAME_SIZE,)
    assert game_state[sj.GAME_ACTION + sj.ACTION_FLIP_SECOND] == 1
    assert np.sum(game_state[sj.GAME_TOP : sj.GAME_TOP + sj.CARD_SIZE]) == 1
    assert game_state[sj.GAME_TOP + sj.CARD_P8] == 1
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
    expected_deck = np.array(sj.CARD_COUNTS, dtype=np.int16)
    expected_deck[sj.CARD_P8] -= 1
    assert np.array_equal(deck_state, expected_deck)

    # Assert other parameters
    assert num_players == players
    assert turn_count == 0
    assert current_card is None
    assert countdown is None

    # Validate overall consistency
    assert sj.validate(initial)


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

    # The limit is inclusive: exactly NO_PROGRESS_TURN_THRESHOLD completed
    # no-progress turns per player are allowed.
    for _ in range(sj.NO_PROGRESS_TURN_THRESHOLD):
        for _ in range(players):
            s = sj.apply_action(s, sj.MASK_TAKE)
            s = sj.apply_action(s, sj.MASK_REPLACE + 0)  # Replace (0,0)
            assert sj.get_countdown(s) is None

    # Player 0's next completed no-progress turn exceeds the limit and starts
    # the end-of-round countdown. The check occurs after the completed turn.
    s = sj.apply_action(s, sj.MASK_TAKE)
    assert sj.get_countdown(s) is None
    s = sj.apply_action(s, sj.MASK_REPLACE + 0)
    assert sj.get_countdown(s) == (players - 1) * 2

    # Every other player receives one final two-decision turn.
    for i in range(players - 1):
        s = sj.apply_action(s, sj.MASK_TAKE)
        s = sj.apply_action(s, sj.MASK_REPLACE + 0)  # Replace (0,0)
        expected_countdown = (players - 2 - i) * 2
        assert sj.get_countdown(s) == expected_countdown

    # Game should be over. The last apply_action should have triggered end_round.
    assert sj.get_countdown(s) == 0, (
        f"Final countdown: actual {sj.get_countdown(s)}, expected 0"
    )
    assert sj.get_round_over(s)

    final_scores = sj.get_round_scores(s, round_ending_player=0)
    # After end_round (called by apply_action), all cards are visible. get_score gives the sum of card values.
    base_scores = np.array(
        [sj.get_score(s, player=i) for i in range(players)], dtype=np.int16
    )

    assert np.array_equal(final_scores, base_scores * 2)


def test_revealing_replacement_resets_no_progress_before_checking_limit():
    rng = np.random.default_rng(0)

    class RandomAdapter:
        def random(self):
            return float(rng.random())

    random_adapter = RandomAdapter()
    state = sj.start_round(sj.new(players=2, top=sj.CARD_0), rng=random_adapter)
    for _ in range(2):
        state = sj.apply_action(state, sj.MASK_FLIP_SECOND_BELOW, rng=random_adapter)
    players = state.players
    stale_turn = (sj.NO_PROGRESS_TURN_THRESHOLD + 1) * players
    state = dataclasses.replace(state, turn=stale_turn)
    state = sj.apply_action(state, sj.MASK_TAKE, rng=random_adapter)
    state = sj.apply_action(state, sj.MASK_REPLACE + 1, rng=random_adapter)
    assert sj.get_countdown(state) is None
