from __future__ import annotations

import random
import types

import numpy as np
import pytest
import torch

from skyjo import game as sj
from skyjo import mcts, observations, predictor, skynet


def make_model() -> skynet.EquivariantSkyNet:
    torch.manual_seed(0)
    return skynet.EquivariantSkyNet(
        spatial_input_shape=(2, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
        non_spatial_input_shape=observations.get_non_spatial_input_shape(2),
        value_output_shape=(2,),
        policy_output_shape=(sj.MASK_SIZE,),
        embedding_dimensions=4,
        global_state_embedding_dimensions=8,
        num_heads=1,
        device=torch.device("cpu"),
    )


@pytest.mark.parametrize("exact_chance", [False, True])
def test_search_preserves_visit_accounting_and_legal_policy(exact_chance) -> None:
    random.seed(10)
    np.random.seed(10)
    state = sj.start_round(sj.new(players=2, top=sj.CARD_0), rng=random)
    # A one-state limit also exercises chunking when chance outcomes are expanded.
    inference = predictor.LocalPredictor(make_model(), max_batch_size=1)
    root = mcts.run_mcts(
        state,
        inference,
        iterations=8,
        after_state_evaluate_all_children=exact_chance,
    )
    policy = root.policy_targets()
    assert root.visit_count == 8
    assert sum(child.visit_count for child in root.children.values()) == 8
    assert np.isfinite(root.state_value).all()
    assert np.isclose(policy.sum(), 1.0)
    assert np.all(policy[sj.actions(root.state) == 0] == 0)
    if exact_chance:
        chances = [
            child
            for child in root.children.values()
            if isinstance(child, mcts.AfterStateNode) and child.is_expanded
        ]
        assert chances
        for chance in chances:
            assert len(chance.children) > 1
            assert all(
                child.model_prediction is not None for child in chance.children.values()
            )


def make_prediction(
    value: tuple[float, float], policy: dict[int, float]
) -> skynet.SkyNetPrediction:
    probabilities = np.zeros(sj.MASK_SIZE, dtype=np.float32)
    for action, probability in policy.items():
        probabilities[action] = probability
    return skynet.SkyNetPrediction(
        value_output=np.asarray(value, dtype=np.float32),
        policy_output=probabilities,
    )


def test_ucb_uses_parent_value_fpu_and_prior_on_first_selection() -> None:
    state = sj.start_round(sj.new(players=2, top=sj.CARD_0), rng=random)
    parent = mcts.DecisionStateNode(
        state=state,
        parent=None,
        action=None,
        c_puct=1.5,
        fpu_reduction=0.1,
    )
    parent.model_prediction = make_prediction(
        (0.7, 0.3), {sj.MASK_DRAW: 0.8, sj.MASK_TAKE: 0.2}
    )
    parent.is_expanded = True
    draw = types.SimpleNamespace(
        action=sj.MASK_DRAW,
        has_value_estimate=False,
        visit_count=0,
    )
    take = types.SimpleNamespace(
        action=sj.MASK_TAKE,
        has_value_estimate=False,
        visit_count=0,
    )
    parent.children = {sj.MASK_DRAW: draw, sj.MASK_TAKE: take}

    assert mcts.ucb_score(draw, parent) == pytest.approx(1.8)
    assert mcts.ucb_score(take, parent) == pytest.approx(0.9)
    assert parent.select_child() is draw


def test_exact_chance_update_replaces_propagated_return() -> None:
    state = sj.start_round(sj.new(players=2, top=sj.CARD_0), rng=random)
    root = mcts.DecisionStateNode(state=state, parent=None, action=None)
    root.model_prediction = make_prediction((0.5, 0.5), {sj.MASK_DRAW: 1.0})
    chance = mcts.AfterStateNode(state=state, action=sj.MASK_DRAW, parent=root)
    child_state = sj.draw(sj.preordain(state, sj.CARD_P5))
    child = mcts.DecisionStateNode(
        state=child_state,
        parent=chance,
        action=sj.MASK_DRAW,
    )
    child.model_prediction = make_prediction((0.6, 0.4), {})
    child.is_expanded = True
    child_hash = sj.hash_skyjo(child_state)
    chance.children = {child_hash: child}
    chance.child_weights = {child_hash: 0.25}
    chance.child_weight_total = 1.0
    chance.state_value_total = np.array([0.45, 0.55], dtype=np.float32)
    chance.all_children_discovered = True
    chance.is_expanded = True

    mcts.backpropagate([root, chance, child], np.array([0.8, 0.2], dtype=np.float32))

    assert np.allclose(child.state_value, [0.8, 0.2])
    assert np.allclose(chance.state_value, [0.5, 0.5])
    assert np.allclose(root.state_value, [0.5, 0.5])


def test_reused_root_requires_matching_scoring_and_resets_noise() -> None:
    state = sj.start_round(sj.new(players=2, top=sj.CARD_0), rng=random)
    client = predictor.LocalPredictor(make_model(), max_batch_size=64)
    root = mcts.run_mcts(
        state,
        client,
        iterations=0,
        dirichlet_epsilon=0.25,
        c_puct=1.5,
        fpu_reduction=0.0,
    )
    assert root.dirichlet_noise.sum() == pytest.approx(1.0)

    reused = mcts.run_mcts(
        state,
        client,
        iterations=0,
        dirichlet_epsilon=0.0,
        c_puct=1.5,
        fpu_reduction=0.0,
        root_node=root,
    )
    assert reused is root
    assert root.dirichlet_epsilon == 0.0
    assert not root.dirichlet_noise.any()

    with pytest.raises(ValueError, match="scoring configuration"):
        mcts.run_mcts(
            state,
            client,
            iterations=0,
            c_puct=1.0,
            root_node=root,
        )


def test_policy_temperature_handles_extremes_and_unvisited_roots():
    state = sj.start_round(sj.new(players=2, top=sj.CARD_0), rng=random.Random(1))
    root = mcts.DecisionStateNode(state, None, None)
    actions = sj.get_actions(state)
    root.children = {action: types.SimpleNamespace(visit_count=0) for action in actions}
    for temperature in (0, 1):
        with pytest.raises(ValueError, match="visited legal"):
            root.policy_targets(temperature)
    root.children[actions[0]].visit_count = 31
    root.children[actions[1]].visit_count = 1
    for temperature in (0, 0.01, 1e-320, 1):
        policy = root.policy_targets(temperature)
        assert np.isfinite(policy).all()
        assert policy.sum() == pytest.approx(1)
        assert policy.argmax() == actions[0]
        assert not policy[~sj.actions(state).astype(bool)].any()
    for temperature in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="temperature"):
            root.policy_targets(temperature)


@pytest.mark.parametrize("predetermined", [False, True])
def test_exact_chance_recycles_discards_without_changing_input(predetermined):
    rng = random.Random(8)
    state = sj.start_round(sj.new(players=2, top=sj.CARD_0), rng=rng)
    for _ in range(2):
        state = sj.apply_action(state, sj.MASK_FLIP_SECOND_RIGHT, rng=rng)
    remaining_after_draw = state.deck.sum() - 1
    assert sj.MASK_DRAW in sj.get_actions(state)
    # Move all unseen cards to recyclable discards, preserving card conservation.
    state.game[sj.GAME_DISCARDS : sj.GAME_DISCARDS + sj.CARD_SIZE] = state.deck
    expected = state.deck.copy() / state.deck.sum()
    state.deck.fill(0)
    if predetermined:
        state = sj.preordain(state, sj.CARD_P1)
        expected = [1.0]
    before = sj.hash_skyjo(state)
    root = mcts.DecisionStateNode(state, None, None)
    chance = mcts.AfterStateNode(state, sj.MASK_DRAW, root)
    chance.discover(discover_all_children=True)
    assert sorted(chance.child_weights.values()) == pytest.approx(sorted(expected))
    assert sj.hash_skyjo(state) == before
    assert all(
        sj.get_deck(child.state).sum() == remaining_after_draw
        for child in chance.children.values()
    )
    sampled = sj.apply_action(state, sj.MASK_DRAW)
    assert sj.hash_skyjo(sampled) in chance.children
