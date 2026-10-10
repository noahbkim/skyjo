"""Exact action groups and the simulator assumptions that permit sharing them."""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from fractions import Fraction

import numpy as np
import pytest

from skyjo.engine import game as sj
from skyjo.search import symmetry


SYMMETRIC_BOARD = (("H", 1, "H", "H"), ("H", "H", "H", 3), (1, "H", 2, 4))
RECYCLING_BOARD = (("H", 1, "H", "X"), (1, "H", 2, "X"), (1, 1, 3, "X"))


def board_state(
    board=SYMMETRIC_BOARD,
    *,
    phase=sj.ACTION_FLIP_OR_REPLACE,
    players=2,
    top=0,
    countdown=None,
    opponents_cleared=False,
    deck_counts=None,
):
    """Build a conserved state; numeric entries are card values, H/X are markers."""
    state = sj.new(players=players, top=top + 2)
    state.game[sj.GAME_ACTION : sj.GAME_ACTION + sj.ACTION_SIZE] = 0
    state.game[sj.GAME_ACTION + phase] = 1
    state.table[:players] = 0
    for player in range(players):
        for row in range(sj.ROW_COUNT):
            for column in range(sj.COLUMN_COUNT):
                value = "X" if player and opponents_cleared else board[row][column]
                if value == "H":
                    finger = sj.FINGER_HIDDEN
                elif value == "X":
                    finger = sj.FINGER_CLEARED
                else:
                    finger = value + 2
                state.table[player, row, column, finger] = 1
                if finger < sj.CARD_SIZE:
                    state.deck[finger] -= 1
    if deck_counts is not None:
        kept = np.zeros_like(state.deck)
        for value, count in deck_counts.items():
            kept[value + 2] = count
        assert np.all(kept <= state.deck)
        state.game[sj.GAME_DISCARDS : sj.GAME_DISCARDS + sj.CARD_SIZE] = (
            state.deck - kept
        )
        state.deck[:] = kept
    state = dataclasses.replace(state, countdown=countdown)
    assert sj.validate(state)
    return state


def with_slack(state, slack):
    """Move cards into discards until unseen minus hidden equals the given slack."""
    remaining = int(state.table[: state.players, :, :, sj.FINGER_HIDDEN].sum()) + slack
    deck = np.zeros_like(state.deck)
    for card, count in enumerate(state.deck):
        kept = min(remaining, int(count))
        deck[card] = kept
        remaining -= kept
    assert remaining == 0
    game = state.game.copy()
    game[sj.GAME_DISCARDS : sj.GAME_DISCARDS + sj.CARD_SIZE] += state.deck - deck
    result = dataclasses.replace(state, deck=deck, game=game)
    assert sj.validate(result)
    return result


def test_legal_groups_partition_actions_by_target_and_complete_column_context():
    state = board_state()
    before = sj.hash_skyjo(state)
    groups = symmetry.ActionGroups.from_state(state)
    assert groups.members == (
        (4, 8, 9, 13),
        (6, 10),
        (7,),
        (16, 20, 21, 25),
        (17, 24),
        (18, 22),
        (19,),
        (23,),
        (26,),
        (27,),
    )
    assert tuple(groups.representatives) == tuple(
        members[0] for members in groups.members
    )
    flattened = [action for members in groups.members for action in members]
    assert sorted(flattened) == list(sj.get_actions(state))
    assert len(flattened) == len(set(flattened))
    assert sj.hash_skyjo(state) == before

    # Equal sums cannot identify a column: H/0/6 and H/1/5 remain distinct.
    collision = board_state((("H", "H", "X", "X"), (0, 1, "X", "X"), (6, 5, "X", "X")))
    assert (sj.MASK_FLIP,) in symmetry.ActionGroups.from_state(collision).members
    assert (sj.MASK_FLIP + 1,) in symmetry.ActionGroups.from_state(collision).members


def test_disabled_merging_and_nonpositional_phases_keep_singleton_actions():
    for state, merge in (
        (board_state(), False),
        (sj.new(players=2, top=sj.CARD_0), True),
        (board_state(phase=sj.ACTION_DRAW_OR_TAKE), True),
    ):
        groups = symmetry.ActionGroups.from_state(state, merge=merge)
        assert groups.members == tuple(
            (int(action),) for action in sj.get_actions(state)
        )


def test_group_mass_is_summed_before_temperature_and_split_afterwards():
    groups = symmetry.ActionGroups.from_state(board_state(phase=sj.ACTION_REPLACE))
    original = np.arange(sj.MASK_SIZE, dtype=np.float32)
    untouched = original.copy()
    mass = groups.aggregate(original)
    assert mass[sj.MASK_REPLACE] == 16 + 20 + 21 + 25
    assert mass[sj.MASK_REPLACE + 1] == 17 + 24
    assert np.count_nonzero(mass) == len(groups.members)
    np.testing.assert_array_equal(original, untouched)

    policy = groups.policy_from_visits({16: 9, 17: 4}, temperature=2)
    np.testing.assert_allclose(policy[[16, 20, 21, 25]], 0.6 / 4)
    np.testing.assert_allclose(policy[[17, 24]], 0.4 / 2)
    assert np.count_nonzero(policy) == 6
    assert policy.sum() == pytest.approx(1)

    tied = groups.policy_from_visits({17: 9, 16: 9}, temperature=0)
    assert tied[16] == 1
    assert np.count_nonzero(tied) == 1
    tiny = groups.policy_from_visits({16: 31, 17: 1}, temperature=1e-320)
    np.testing.assert_allclose(tiny[[16, 20, 21, 25]], 0.25)
    assert np.isfinite(tiny).all()
    assert tiny.sum() == pytest.approx(1)
    for temperature in (0, 1):
        with pytest.raises(ValueError, match="visited legal"):
            groups.policy_from_visits({}, temperature=temperature)
    for temperature in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="temperature"):
            groups.policy_from_visits({16: 1}, temperature=temperature)


def canonical_outcome(state):
    """Compare complete successors modulo actual row/column permutations.

    This test oracle does not call any production symmetry or action-key helper.
    It preserves players, game metadata and unseen-card counts exactly.
    """
    boards = []
    for board in state.table[: state.players]:
        columns = [
            tuple(sorted(map(int, board[:, column].argmax(axis=-1))))
            for column in range(sj.COLUMN_COUNT)
        ]
        boards.append(tuple(sorted(columns)))
    return (
        tuple(boards),
        state.game.tobytes(),
        state.deck.tobytes(),
        state.turn,
        state.pending_card,
        state.countdown,
    )


@pytest.mark.parametrize("offset,top", [(sj.MASK_FLIP, 0), (sj.MASK_REPLACE, 1)])
def test_symmetric_random_transitions_have_identical_distributions_and_clears(
    offset, top
):
    state = board_state(
        RECYCLING_BOARD, top=top, opponents_cleared=True, deck_counts={1: 3, 2: 2}
    )
    before = sj.hash_skyjo(state)
    distributions = []
    cleared = False
    for action in (offset, offset + 5):
        distribution = defaultdict(Fraction)
        for card in np.flatnonzero(state.deck):
            successor = sj.apply_action(sj.preordain(state, int(card)), action)
            distribution[canonical_outcome(successor)] += Fraction(
                int(state.deck[card]), int(state.deck.sum())
            )
            cleared |= successor.table[1, :, :, sj.FINGER_CLEARED].sum() == 6
        distributions.append(distribution)
    assert distributions[0] == distributions[1]
    assert cleared
    assert sj.hash_skyjo(state) == before


def exact_final_score_distribution(state, action, monkeypatch):
    """Branch on every actual simulator draw, including recycled discards.

    Intercept only the random-choice seam: clearing, reveal order, recycling,
    score calculation and action execution all remain the real simulator.
    """

    class NeedDraw(Exception):
        def __init__(self, deck):
            self.deck = deck.copy()

    result = defaultdict(Fraction)
    pending = [((), Fraction(1))]
    with monkeypatch.context() as patch:
        while pending:
            tape, probability = pending.pop()
            choices = iter(tape)

            def choose(deck, rng):
                try:
                    card = next(choices)
                except StopIteration:
                    raise NeedDraw(deck) from None
                assert deck[card] > 0
                return card

            patch.setattr(sj, "_choose_card", choose)
            try:
                completed = sj.apply_action(state, action)
            except NeedDraw as draw:
                for card in np.flatnonzero(draw.deck):
                    weight = Fraction(int(draw.deck[card]), int(draw.deck.sum()))
                    pending.append((tape + (int(card),), probability * weight))
            else:
                assert sj.get_round_over(completed)
                result[tuple(sj.get_fixed_perspective_round_scores(completed))] += (
                    probability
                )
    assert sum(result.values()) == 1
    return result


def test_safe_final_reveals_preserve_scores_but_recycling_breaks_symmetry(monkeypatch):
    safe = board_state(
        RECYCLING_BOARD,
        phase=sj.ACTION_REPLACE,
        countdown=1,
        opponents_cleared=True,
        deck_counts={0: 1, 1: 3, 2: 1},
    )
    assert symmetry.safe_to_merge_actions(safe, 2)
    assert exact_final_score_distribution(
        safe, 16, monkeypatch
    ) == exact_final_score_distribution(safe, 21, monkeypatch)

    unsafe = board_state(
        RECYCLING_BOARD,
        phase=sj.ACTION_REPLACE,
        countdown=1,
        opponents_cleared=True,
        deck_counts={1: 2},
    )
    assert not symmetry.safe_to_merge_actions(unsafe, 0)
    first = exact_final_score_distribution(unsafe, 16, monkeypatch)
    second = exact_final_score_distribution(unsafe, 21, monkeypatch)
    assert first != second
    assert sum(scores[0] * weight for scores, weight in first.items()) == Fraction(
        2147, 141
    )
    assert sum(scores[0] * weight for scores, weight in second.items()) == Fraction(
        110, 9
    )


@pytest.mark.parametrize(
    "budget,required_slack", [(0, 1), (1, 2), (2, 2), (3, 3), (8, 5)]
)
def test_safety_threshold_counts_hidden_cards_across_all_players(
    budget, required_slack
):
    state = board_state(players=3)
    assert state.table[1:3, :, :, sj.FINGER_HIDDEN].sum() == 14
    assert symmetry.safe_to_merge_actions(with_slack(state, required_slack), budget)
    assert not symmetry.safe_to_merge_actions(
        with_slack(state, required_slack - 1), budget
    )
