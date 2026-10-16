"""Group probabilities survive model, loss, search and checkpoint boundaries."""

import numpy as np
import pytest
import torch
from test_action_symmetry import board_state

from skyjo.engine import game
from skyjo.experiments.contestants import load_model
from skyjo.learning import batches, checkpoint, losses, models, predictor
from skyjo.learning.policy import group_normalized_logits
from skyjo.search import mcts
from skyjo.search.runner import search
from skyjo.search.symmetry import ActionGroups


def test_pool_logits_before_softmax_and_train_on_split_group_targets():
    inputs = batches.to_tensors(batches.states_to_batch([board_state()]), device="cpu")
    # Two distinct choices each have four representations; one is a singleton.
    groups = ((4, 8, 9, 13), (7,), (16, 20, 21, 25))
    mask = torch.zeros(1, game.MASK_SIZE)
    logits = torch.full_like(mask, 1000.0)  # Illegal scores must not enter pooling.
    for group in groups:
        mask[0, list(group)] = 1
    logits[0, list(groups[0])] = torch.tensor([0.0, 2.0, 4.0, 2.0])
    logits[0, 7] = 0
    logits[0, list(groups[2])] = np.log(2)
    logits.requires_grad_()
    adjusted = group_normalized_logits(logits, inputs.spatial_inputs, mask)
    probabilities = adjusted.softmax(-1)
    expected = torch.tensor([2.0, 0.0, np.log(2)], dtype=torch.float32).softmax(0)
    target = torch.zeros_like(mask)
    group_target = (0.2, 0.5, 0.3)
    for i, group in enumerate(groups):
        torch.testing.assert_close(
            probabilities[0, list(group)], expected[i].expand(len(group)) / len(group)
        )
        target[0, list(group)] = group_target[i] / len(group)
    assert not probabilities[~mask.bool()].any()
    losses.cross_entropy_policy_loss(adjusted, target).backward()
    for i, group in enumerate(groups):
        torch.testing.assert_close(
            logits.grad[0, list(group)],
            (expected[i] - group_target[i]).expand(len(group)) / len(group),
        )
    assert not logits.grad[~mask.bool()].any()


def test_default_model_search_and_visit_targets_share_group_probabilities():
    state = board_state()
    model = models.build({}, players=2, device="cpu")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    evaluator = predictor.LocalPredictor(model, 32)
    result = search(
        state,
        evaluator,
        16,
        config=mcts.SearchConfig(target_strategy="visits"),
        rng=np.random.default_rng(71),
    )
    groups = result.summary.groups.members
    assert len(groups) == 10
    np.testing.assert_allclose(result.summary.priors, np.full(10, 0.1))
    prediction = evaluator.evaluate([state])[0]
    # A larger group must not lose greedy policy play to a singleton merely
    # because its probability is split across several representations.
    assert prediction.policy.argmax() == 7
    assert (
        predictor.PolicyPlayer(evaluator).get_action_probabilities(state).argmax() == 4
    )
    for i, group in enumerate(groups):
        np.testing.assert_allclose(prediction.policy[list(group)], 0.1 / len(group))
        np.testing.assert_allclose(
            result.training_policy[list(group)],
            result.summary.visits[i] / 16 / len(group),
        )
    assert (
        result.action_probabilities.max() == 1
    )  # Greedy acting still has split targets.
    model.policy_normalization = "actions"
    legacy = evaluator.evaluate([state])[0]
    np.testing.assert_allclose(legacy.policy[game.actions(state).astype(bool)], 1 / 19)


@pytest.mark.parametrize("mode", ["groups", "actions"])
def test_checkpoints_preserve_policy_semantics_and_reject_wrong_exact_resume(
    tmp_path, mode
):
    model = models.build({"policy_normalization": mode}, players=2, device="cpu")
    configuration = {"model": {}, "players": 2}
    path = checkpoint.save_checkpoint(
        tmp_path / "model.pth", model=model, optimizer=None, configuration=configuration
    )
    assert configuration["model"] == {}  # Recording does not mutate the caller.
    saved = checkpoint.decode_checkpoint(path)
    assert saved["configuration"]["model"]["policy_normalization"] == mode
    reloaded = load_model(path)
    state = board_state()
    expected = predictor.LocalPredictor(model, 1).evaluate([state])[0]
    actual = predictor.LocalPredictor(reloaded, 1).evaluate([state])[0]
    np.testing.assert_array_equal(expected.policy, actual.policy)
    if mode == "actions":
        # Historical checkpoints have no normalization metadata.
        del saved["configuration"]["model"]["policy_normalization"]
        torch.save(saved, path)
        legacy = load_model(path)
        np.testing.assert_array_equal(
            predictor.LocalPredictor(legacy, 1).evaluate([state])[0].policy,
            expected.policy,
        )
    wrong = models.build(
        {"policy_normalization": "actions" if mode == "groups" else "groups"},
        players=2,
        device="cpu",
    )
    for expected_config in (configuration, saved["configuration"]):
        with pytest.raises(ValueError, match="configuration does not match"):
            checkpoint.load_checkpoint(
                path, model=wrong, expected_configuration=expected_config
            )


@pytest.mark.parametrize("group_noise", [True, False])
def test_root_noise_uses_the_selected_probability_space(group_noise):
    from test_mcts_symmetry import FixedEvaluator

    state = board_state()
    config = mcts.SearchConfig(dirichlet_epsilon=1, group_root_noise=group_noise)
    root = mcts.run_mcts(
        state, FixedEvaluator(), 0, config=config, rng=np.random.default_rng(81)
    )
    groups = ActionGroups.from_state(state, merge=group_noise)
    # The symmetric Dirichlet has one component per choice, not per representation.
    expected = np.random.default_rng(81).dirichlet(
        np.full(len(groups.members), 10 / len(groups.members))
    )
    for group, mass in zip(groups.members, expected, strict=True):
        assert root.dirichlet_noise[list(group)].sum() == pytest.approx(mass)
    np.testing.assert_allclose(
        root.action_priors, root.action_groups.aggregate(root.dirichlet_noise)
    )
