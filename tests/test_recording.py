"""Game records preserve observed decisions independently of model execution."""

import json
import subprocess
import sys

import numpy as np
import pytest

from skyjo.engine import game as sj
from skyjo.search.configuration import preset
from skyjo.search.evaluator import Prediction
from skyjo.search.mcts import SearchConfig
from skyjo.search.player import SearchPlayer
from skyjo.simulation import play, recording
from skyjo.simulation.decisions import PlayerDecision, get_decision
from skyjo.simulation.jobs import GameRandomStreams
from skyjo.simulation.player import RandomPlayer
from skyjo.simulation.session import GameSession
from skyjo.ui.config import (
    MatchConfig,
    SeatConfig,
    load_preset,
    prepare_session,
    save_preset,
)


def random_session(players=2):
    return GameSession(
        [RandomPlayer() for _ in range(players)],
        seed=23,
        metadata=[{"name": f"Seat {i + 1}"} for i in range(players)],
    )


def test_session_matches_simulation_and_saved_game_replays_exactly(tmp_path):
    session = random_session()
    while not session.recording.complete:
        session.advance(None if sj.get_round_over(session.state) else session.propose())
    streams = GameRandomStreams.from_seed(23)
    reference = play.play_game(
        [RandomPlayer(), RandomPlayer()],
        environment_rng=streams.environment,
        action_rng=streams.actions,
    )
    recorded = session.recording
    entries = [entry for round_ in reference.rounds for entry in round_.history]
    assert len(entries) == len(recorded.frames)
    assert recorded.frames[-1].cumulative_scores == reference.final_scores
    assert recorded.frames[-1].winners == reference.winners
    for entry, frame in zip(entries, recorded.frames, strict=True):
        assert sj.hash_skyjo(entry.state) == sj.hash_skyjo(frame.state)
        assert entry.action == frame.action
    path = recording.save(recorded, tmp_path / "game.json")
    restored = recording.load(path)
    assert recording.to_dict(restored) == recording.to_dict(recorded)


def test_partial_recording_and_bad_file_do_not_destroy_previous_save(tmp_path):
    session = random_session(8)
    session.advance(session.propose())
    path = recording.save(session.recording, tmp_path / "partial.json")
    saved = path.read_bytes()
    restored = recording.load(path)
    assert not restored.complete
    assert restored.frames[-1].action is None
    assert restored.frames[0].decision.diagnostics
    session.recording.players[0]["invalid"] = float("nan")
    with pytest.raises(ValueError):
        recording.save(session.recording, path)
    assert path.read_bytes() == saved
    data = json.loads(saved)
    data["frames"][0]["action"] = sj.MASK_DRAW
    with pytest.raises(recording.RecordingError, match="Illegal recorded action"):
        recording.from_dict(data)
    data = json.loads(saved)
    data["frames"][0]["diagnostics"][0]["rows"] = [["not enough columns"]]
    with pytest.raises(recording.RecordingError, match="diagnostic"):
        recording.from_dict(data)


def test_human_actions_and_legacy_player_distributions_are_checked():
    session = GameSession(
        [None, RandomPlayer()], seed=3, metadata=[{"name": "Human"}, {"name": "Bot"}]
    )
    before = sj.hash_skyjo(session.state)
    with pytest.raises(ValueError, match="legal human action"):
        session.advance(action=sj.MASK_DRAW)
    assert sj.hash_skyjo(session.state) == before
    session.advance(action=sj.MASK_FLIP_SECOND_RIGHT)
    assert session.recording.frames[0].action == sj.MASK_FLIP_SECOND_RIGHT

    class LegacyPlayer:
        def get_action_probabilities(self, state):
            return RandomPlayer().get_action_probabilities(state)

    decision = get_decision(LegacyPlayer(), session.state)
    assert not decision.diagnostics
    with pytest.raises(ValueError, match="normalized distribution"):
        session.advance(PlayerDecision(np.zeros(sj.MASK_SIZE)))


def test_independent_training_policy_survives_recording_and_legacy_loading():
    session = random_session()
    actions = np.flatnonzero(sj.actions(session.state))
    acting = np.zeros(sj.MASK_SIZE, dtype=np.float32)
    target = np.zeros_like(acting)
    acting[actions[0]] = target[actions[-1]] = 1
    session.advance(PlayerDecision(acting, training_policy=target))
    assert session.recording.frames[0].action == actions[0]
    data = recording.to_dict(session.recording)
    restored = recording.from_dict(data)
    np.testing.assert_array_equal(restored.frames[0].decision.probabilities, acting)
    np.testing.assert_array_equal(restored.frames[0].decision.training_policy, target)
    assert recording.to_dict(restored) == data

    data["version"] = 1
    for frame in data["frames"]:
        frame.pop("training_policy")
    legacy = recording.from_dict(data)
    assert legacy.frames[0].decision.training_policy is None
    np.testing.assert_array_equal(legacy.frames[0].decision.probabilities, acting)


def test_invalid_training_policy_is_rejected_before_play_or_replay():
    session = random_session()
    decision = session.propose()
    invalid = PlayerDecision(
        decision.probabilities, training_policy=np.zeros(sj.MASK_SIZE)
    )

    class InvalidPlayer:
        def get_decision(self, state, *, diagnostics=True):
            return invalid

    with pytest.raises(ValueError, match="normalized distribution"):
        get_decision(InvalidPlayer(), session.state)
    before = sj.hash_skyjo(session.state)
    with pytest.raises(ValueError, match="normalized distribution"):
        session.advance(invalid)
    assert sj.hash_skyjo(session.state) == before
    session.advance(decision)
    data = recording.to_dict(session.recording)
    data["frames"][0]["training_policy"] = invalid.training_policy.tolist()
    with pytest.raises(recording.RecordingError, match="normalized distribution"):
        recording.from_dict(data)


def test_diagnostic_action_metadata_round_trips_and_is_optional():
    session = random_session()
    session.advance(session.propose())
    payload = recording.to_dict(session.recording)
    restored = recording.from_dict(payload)
    section = restored.frames[0].decision.diagnostics[0]
    assert section.row_actions == (sj.MASK_FLIP_SECOND_BELOW, sj.MASK_FLIP_SECOND_RIGHT)
    assert section == session.recording.frames[0].decision.diagnostics[0]

    raw_section = payload["frames"][0]["diagnostics"][0]
    del raw_section["row_actions"]
    legacy = recording.from_dict(payload).frames[0].decision.diagnostics[0]
    assert legacy.rows == section.rows
    assert legacy.row_actions == ()
    for invalid in ((0,), (0, sj.MASK_SIZE), (0, True)):
        raw_section["row_actions"] = invalid
        with pytest.raises(recording.RecordingError, match="diagnostic row actions"):
            recording.from_dict(payload)


class UniformEvaluator:
    def evaluate(self, states):
        return [
            Prediction(
                np.full(state.players, 1 / state.players, np.float32),
                RandomPlayer().get_action_probabilities(state),
            )
            for state in states
        ]


def test_search_diagnostics_share_the_same_search_and_rng():
    state = random_session().state
    config = SearchConfig()
    a = SearchPlayer(
        UniformEvaluator(),
        12,
        config=config,
        temperature=0,
        rng=np.random.default_rng(45),
    )
    b = SearchPlayer(
        UniformEvaluator(),
        12,
        config=config,
        temperature=0,
        rng=np.random.default_rng(45),
    )
    detailed = a.get_decision(state)
    plain = b.get_decision(state, diagnostics=False)
    np.testing.assert_array_equal(detailed.probabilities, plain.probabilities)
    assert a.rng.bit_generator.state == b.rng.bit_generator.state
    actions, values = detailed.diagnostics
    assert sum(row[3] for row in actions.rows) == 12
    assert sum(row[4] for row in actions.rows) == pytest.approx(1)
    assert all(row[4] == row[3] / 12 for row in actions.rows)
    assert sum(row[3] > 0 for row in actions.rows) > 1
    assert np.count_nonzero(detailed.probabilities) == 1
    assert actions.row_actions == tuple(map(int, np.flatnonzero(sj.actions(state))))
    assert [row[5] for row in actions.rows] == [
        float(detailed.probabilities[action]) for action in actions.row_actions
    ]
    assert sum(row[1] for row in actions.rows) == pytest.approx(1)
    assert values.title == "Root win estimates"


@pytest.mark.parametrize("temperature", [0, 0.5])
def test_grouped_diagnostics_preserve_model_search_and_selection_meanings(temperature):
    class WeightedEvaluator:
        def evaluate(self, states):
            predictions = []
            for state in states:
                policy = np.arange(1, sj.MASK_SIZE + 1, dtype=np.float32) * sj.actions(
                    state
                )
                predictions.append(
                    Prediction(np.array([0.3, 0.7]), policy / policy.sum())
                )
            return predictions

    state = random_session().state
    rng = np.random.default_rng(2)
    while sj.get_action(state) != sj.ACTION_FLIP_OR_REPLACE:
        action = int(np.flatnonzero(sj.actions(state))[0])
        state = sj.apply_action(state, action, rng=rng)
    config = SearchConfig(dirichlet_epsilon=0.25)
    player = SearchPlayer(
        WeightedEvaluator(),
        2,
        config=config,
        temperature=temperature,
        rng=np.random.default_rng(45),
    )
    reference = SearchPlayer(
        WeightedEvaluator(),
        2,
        config=config,
        temperature=temperature,
        rng=np.random.default_rng(45),
    ).run_mcts(state)
    decision = player.get_decision(state)
    actions, values = decision.diagnostics
    summaries = [row for row in actions.rows if row[3] != ""]
    assert len(summaries) == len(reference.action_groups.members)
    assert sum(row[3] for row in summaries) == 2
    assert any(row[6] is None for row in summaries)
    assert sum(row[1] for row in summaries) == pytest.approx(1)
    assert sum(row[2] for row in summaries) == pytest.approx(1)
    assert sum(row[5] for row in summaries) == pytest.approx(1)
    assert any(action is None for action in actions.row_actions)
    for row, group in zip(summaries, reference.action_groups.members, strict=True):
        representative = group[0]
        child = reference.children[representative]
        assert row[1] == pytest.approx(
            reference.model_prediction.policy[list(group)].sum()
        )
        assert row[2] == reference.action_priors[representative]
        assert row[3:5] == (child.visit_count, child.visit_count / 2)
        assert row[5] == pytest.approx(decision.probabilities[list(group)].sum())
        expected_value = (
            child.state_value[sj.get_player(state)]
            if child.has_value_estimate
            else None
        )
        assert row[6] == expected_value
    members = {
        action: row
        for action, row in zip(actions.row_actions, actions.rows, strict=True)
        if action is not None
    }
    assert list(members) == [
        action for group in reference.action_groups.members for action in group
    ]
    for group in reference.action_groups.members:
        for action in group:
            row = members[action]
            assert row[1] == reference.model_prediction.policy[action]
            assert row[5] == decision.probabilities[action]
            if len(group) > 1:
                assert row[2:5] == ("", "", "")
                assert row[6] == ""
    assert values.rows == tuple(
        (i + 1, float(model), float(search))
        for i, (model, search) in enumerate(
            zip(reference.model_prediction.value, reference.state_value, strict=True)
        )
    )
    np.testing.assert_array_equal(
        decision.probabilities, reference.policy_targets(temperature)
    )


def test_preset_paths_settings_and_seat_counts_round_trip(tmp_path):
    config = MatchConfig(tuple(SeatConfig(f"P{i}") for i in range(8)), seed=19)
    path = tmp_path / "match.toml"
    save_preset(config, path)
    assert load_preset(path) == config
    assert prepare_session(config).state.players == 8
    path.write_text(
        'version = 1\n[[seats]]\nname = "Human"\nkind = "human"\n[[seats]]\nname = "Model"\nkind = "search"\ncheckpoint = "model.pth"\n[seats.search]\nfpu_reduction = 0.2\n'
    )
    loaded = load_preset(path)
    assert loaded.seats[1].checkpoint == str(tmp_path / "model.pth")
    assert loaded.seats[1].search.fpu_reduction == 0.2


@pytest.mark.parametrize("name", ["alphazero", "gumbel", "lower_confidence"])
def test_consolidated_diagnostics_preserve_strategy_targets_and_replica_estimates(name):
    state = random_session().state
    config = preset(name)

    def player():
        return SearchPlayer(
            UniformEvaluator(),
            32,
            config=config,
            temperature=0,
            rng=np.random.default_rng(45),
        )

    result = player().search(state)
    decision = player().get_decision(state)
    actions, values = decision.diagnostics
    view = result.summary
    assert sum(row[3] for row in actions.rows) == 32
    assert actions.columns[7] == "Training probability"
    for index, group in enumerate(view.groups.members):
        row = actions.rows[index]
        assert row[5] == float(result.action_probabilities[list(group)].sum())
        assert row[6] == float(view.q[index])
        assert row[7] == float(result.training_policy[list(group)].sum())
        if config.replicas > 1:
            assert actions.columns[8] == "Standard error"
            assert row[8] == float(view.standard_error[index, view.player])
    assert [row[2] for row in values.rows] == list(view.state_value)
    np.testing.assert_array_equal(decision.training_policy, result.training_policy)


@pytest.mark.parametrize("alpha", [None, 0.3])
def test_strategy_presets_preserve_optional_parameters_in_toml(tmp_path, alpha):
    config = MatchConfig(
        (
            SeatConfig("Human", "human"),
            SeatConfig(
                "Search",
                "search",
                str(tmp_path / "model.pth"),
                search=preset("gumbel", dirichlet_alpha=alpha),
            ),
        )
    )
    path = tmp_path / "strategies.toml"
    save_preset(config, path)
    assert load_preset(path) == config


@pytest.mark.parametrize("legacy_metadata", [False, True])
def test_model_seats_load_checkpoints_and_produce_distinct_diagnostics(
    tmp_path, legacy_metadata
):
    from skyjo.learning import checkpoint, models

    settings = {
        "name": "hierarchical_equivariant_v2",
        "embedding_dimensions": 4,
        "global_state_embedding_dimensions": 8,
        "num_heads": 2,
    }
    path = tmp_path / "model.pth"
    objectives = {"round_doubled": 0.1, "round_raw_score": 1.0}
    configuration = {
        "model": settings,
        "players": 2,
        "auxiliary_objectives": objectives,
    }
    if legacy_metadata:
        configuration = {
            "players": 2,
            "model": {
                **settings,
                "non_spatial_input_shape": [69],
                "auxiliary_objectives": objectives,
            },
        }
    checkpoint.save_checkpoint(
        path,
        model=models.build(
            settings, players=2, device="cpu", auxiliary_objectives=objectives
        ),
        optimizer=None,
        configuration=configuration,
    )
    seats = (
        SeatConfig("Policy", "policy", str(path)),
        SeatConfig("Search", "search", str(path), iterations=2),
    )
    session = prepare_session(MatchConfig(seats))
    policy_decision = session.propose()
    assert np.count_nonzero(policy_decision.probabilities) == 1
    assert policy_decision.diagnostics[0].title.startswith("Model policy")
    session.advance(policy_decision)
    search_decision = session.propose()
    assert search_decision.diagnostics[0].title == "Search actions"
    assert (
        session.recording.players[0]["checkpoint_sha256"]
        == session.recording.players[1]["checkpoint_sha256"]
    )
    with pytest.raises(ValueError, match="table has 3"):
        prepare_session(MatchConfig((*seats, SeatConfig())))


def test_recording_and_random_ui_layers_do_not_import_torch():
    program = """
import importlib.abc
import sys
class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise AssertionError("Model-free playback imported Torch")
sys.meta_path.insert(0, BlockTorch())
from skyjo.ui.app import SkyjoApp
from skyjo.ui.config import MatchConfig, prepare_session
from skyjo.simulation.recording import from_dict, to_dict
session = prepare_session(MatchConfig())
session.advance(action=0)
from_dict(to_dict(session.recording))
assert "torch" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", program], check=True, timeout=30)
