from __future__ import annotations

import random

import numpy as np
from hypothesis import HealthCheck, given, settings, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, precondition, rule

import skyjo as sj


def assert_card_conservation(state: sj.Skyjo) -> None:
    game, table, deck, players, *_ = state
    on_table = table[:players, :, :, : sj.CARD_SIZE].sum(axis=(0, 1, 2))
    top = game[sj.GAME_TOP : sj.GAME_TOP + sj.CARD_SIZE]
    discarded = game[sj.GAME_DISCARDS : sj.GAME_DISCARDS + sj.CARD_SIZE]
    assert np.array_equal(deck + on_table + top + discarded, sj.CARD_COUNTS)


class LegacyGameStateMachine(RuleBasedStateMachine):
    @initialize(players=st.integers(min_value=2, max_value=8), seed=st.integers())
    def initialize_game(self, players: int, seed: int) -> None:
        self.rng = random.Random(seed)
        self.state = sj.start_round(sj.new(players=players, rng=self.rng), rng=self.rng)

    @precondition(lambda self: not sj.get_game_over(self.state))
    @rule(choice=st.integers(min_value=0, max_value=sj.MASK_SIZE - 1))
    def apply_legal_action(self, choice: int) -> None:
        valid_actions = np.flatnonzero(sj.actions(self.state))
        action = int(valid_actions[choice % len(valid_actions)])
        old_state = self.state
        old_turn = sj.get_turn(old_state)
        old_table = sj.get_table(old_state).copy()

        self.state = sj.apply_action(old_state, action, rng=self.rng)

        assert sj.validate(self.state)
        assert sj.get_player(self.state) == sj.get_turn(self.state) % sj.get_player_count(
            self.state
        )
        if sj.get_turn(self.state) == old_turn + 1 and not sj.get_game_over(self.state):
            # Completed turns rotate the non-acting players one slot toward the
            # active-player perspective without changing their boards.
            assert np.array_equal(sj.get_table(self.state)[:-1], old_table[1:])

    @precondition(lambda self: sj.get_game_over(self.state))
    @rule()
    def observe_terminal_state(self) -> None:
        scores = sj.get_round_scores(self.state)
        assert len(scores) == sj.get_player_count(self.state)
        assert sj.get_fixed_perspective_winner(self.state) == int(
            np.argmin(sj.get_fixed_perspective_round_scores(self.state))
        )

    @invariant()
    def state_is_valid_and_conserves_cards(self) -> None:
        assert sj.validate(self.state)
        assert_card_conservation(self.state)
        if not sj.get_game_over(self.state):
            assert sj.actions(self.state).sum() > 0


TestLegacyGameStateMachine = LegacyGameStateMachine.TestCase
TestLegacyGameStateMachine.settings = settings(
    max_examples=40,
    stateful_step_count=100,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)


@settings(max_examples=40, deadline=None)
@given(
    players=st.integers(min_value=2, max_value=8),
    top=st.integers(min_value=0, max_value=sj.CARD_SIZE - 1),
)
def test_initial_discard_is_conserved(players: int, top: int) -> None:
    state = sj.new(players=players, top=top)
    assert sj.get_top(state) == top
    assert sj.get_deck(state)[top] == sj.CARD_COUNTS[top] - 1
    assert_card_conservation(state)


@settings(max_examples=80, deadline=None)
@given(card=st.integers(min_value=0, max_value=sj.CARD_SIZE - 1))
def test_preordained_draw_realizes_exact_card(card: int) -> None:
    rng = random.Random(0)
    state = sj.start_round(sj.new(players=2, top=sj.CARD_0), rng=rng)
    for _ in range(2):
        state = sj.apply_action(state, sj.MASK_FLIP_SECOND_BELOW, rng=rng)

    if sj.get_deck(state)[card] == 0:
        return
    deck_before = sj.get_deck(state).copy()
    state = sj.apply_action(sj.preordain(state, card), sj.MASK_DRAW, rng=rng)
    assert sj.get_top(state) == card
    assert sj.get_deck(state)[card] == deck_before[card] - 1
    assert_card_conservation(state)


@settings(max_examples=60, deadline=None)
@given(card=st.integers(min_value=0, max_value=sj.CARD_SIZE - 1))
def test_preordained_reveal_and_hidden_replacement_realize_exact_card(card: int) -> None:
    rng = random.Random(4)
    state = sj.start_round(sj.new(players=2, top=sj.CARD_0), rng=rng)
    if sj.get_deck(state)[card] == 0:
        return
    state = sj.apply_action(
        sj.preordain(state, card), sj.MASK_FLIP_SECOND_BELOW, rng=rng
    )
    assert sj.get_table(state)[-1, 1, 0, card] == 1
    assert_card_conservation(state)

    # Complete setup, draw, then discard the draw and reveal a hidden finger.
    state = sj.apply_action(state, sj.MASK_FLIP_SECOND_BELOW, rng=rng)
    state = sj.apply_action(state, sj.MASK_DRAW, rng=rng)
    available = np.flatnonzero(sj.get_deck(state))
    reveal_card = int(available[0])
    state = sj.apply_action(
        sj.preordain(state, reveal_card), sj.MASK_FLIP + 1, rng=rng
    )
    assert sj.get_table(state)[-1, 0, 1, reveal_card] == 1
    assert_card_conservation(state)

    # Taking the discard and replacing a hidden card preordains the displaced
    # hidden card that becomes the new discard.
    previous_top = sj.get_top(state)
    state = sj.apply_action(state, sj.MASK_TAKE, rng=rng)
    available = np.flatnonzero(sj.get_deck(state))
    displaced_card = int(available[-1])
    state = sj.apply_action(
        sj.preordain(state, displaced_card), sj.MASK_REPLACE + 2, rng=rng
    )
    assert sj.get_top(state) == displaced_card
    assert sj.get_table(state)[-1, 0, 2, previous_top] == 1
    assert_card_conservation(state)


def test_hash_and_chance_node_key_contract() -> None:
    state = sj.new(players=2, top=sj.CARD_0)
    assert isinstance(sj.hash_skyjo(state), int)
