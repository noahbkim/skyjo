"""The game, search and simulation contracts remain usable without Torch."""

import os
import subprocess
import sys

import numpy as np

from skyjo.engine import game as sj
from skyjo.search import mcts
from skyjo.simulation import play, player
from test_mcts_core import UniformEvaluator


def test_model_free_layers_import_and_play_with_torch_blocked():
    program = """
import importlib.abc
import sys
class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise AssertionError("A model-free layer tried to import Torch")
sys.meta_path.insert(0, BlockTorch())
from skyjo.engine import game, values, symmetry
from skyjo.search import mcts, evaluator, player as search_player
from skyjo.simulation import play, player, jobs
result = play.play_game([player.RandomPlayer(), player.RandomPlayer()])
assert max(result.final_scores) >= 100
assert "torch" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", program], check=True, env=os.environ.copy(), timeout=30
    )


def test_owned_streams_make_game_and_search_independent_of_process_rng(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected process-global RNG use")

    import random

    monkeypatch.setattr(random, "random", forbidden)
    monkeypatch.setattr(np.random, "choice", forbidden)
    monkeypatch.setattr(np.random, "dirichlet", forbidden)

    def generate():
        return play.play_game(
            [player.RandomPlayer(), player.RandomPlayer()],
            environment_rng=np.random.default_rng(23),
            action_rng=np.random.default_rng(31),
        )

    first, second = generate(), generate()
    assert first.final_scores == second.final_scores
    assert [
        [sj.hash_skyjo(entry.state) for entry in round_.history]
        for round_ in first.rounds
    ] == [
        [sj.hash_skyjo(entry.state) for entry in round_.history]
        for round_ in second.rounds
    ]
    state = first.rounds[0].history[0].state
    config = mcts.SearchConfig(dirichlet_epsilon=0.2)
    evaluator = UniformEvaluator()
    roots = [
        mcts.run_mcts(
            state, evaluator, 12, config=config, rng=np.random.default_rng(44)
        )
        for _ in range(2)
    ]
    np.testing.assert_array_equal(roots[0].policy_targets(), roots[1].policy_targets())
    np.testing.assert_array_equal(roots[0].state_value, roots[1].state_value)
