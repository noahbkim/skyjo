"""Persistence contracts; no model or measurement schema assumptions."""

import json
import random
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from skyjo.experiments import runs


def read_events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture
def repository(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "uv.lock").write_text("test lock\n")
    (repo / ".gitignore").write_text(".runs/\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=repo,
        check=True,
    )
    return repo


def create_run(repo, *, allow_dirty=False):
    return runs.RunRecorder.create(
        root=repo / ".runs",
        repository=repo,
        input_path=Path("input.toml"),
        input_bytes=b'name = "example"\n',
        configuration={
            "name": "example",
            "description": "Question",
            "notes": "Observation",
        },
        entrypoint="test",
        invocation=[],
        allow_dirty=allow_dirty,
    )


@pytest.mark.parametrize("change", ["tracked", "staged", "untracked"])
def test_dirty_launch_requires_recorded_override(repository, change):
    clean = create_run(repository)
    assert clean.manifest["git"]["dirty"] is False
    path = repository / ("new.py" if change == "untracked" else "uv.lock")
    path.write_text("changed\n")
    if change == "staged":
        subprocess.run(["git", "add", "uv.lock"], cwd=repository, check=True)
    before = set((repository / ".runs").iterdir())
    with pytest.raises(ValueError, match="allow-dirty"):
        create_run(repository)
    assert set((repository / ".runs").iterdir()) == before
    overridden = create_run(repository, allow_dirty=True)
    assert overridden.path != clean.path
    assert overridden.manifest["git"] == {
        "commit": clean.manifest["git"]["commit"],
        "dirty": True,
        "allow_dirty": True,
    }


def test_extensible_events_artifacts_and_rng_neutrality(repository):
    random.seed(18)
    np.random.seed(18)
    torch.manual_seed(18)
    expected_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    with create_run(repository) as run:
        run.record_event(
            "new_diagnostic",
            progress={"tokens": 17},
            metrics={"novel/metric": 0.25},
            context={"nested": [1, {"a": True}]},
        )
        data = run.path / "data" / "test.json"
        data.write_text("{}")
        old = run.register_artifact(data, kind="custom", progress={"tokens": 17})
        run.supersede_artifact(old, progress={"tokens": 18})
        data.write_text('{"replacement":true}')
        new = run.register_artifact(data, kind="custom", progress={"tokens": 18})
    assert old != new
    events = read_events(run.path / "trajectory.jsonl")
    diagnostic = next(event for event in events if event["kind"] == "new_diagnostic")
    assert diagnostic["context"] == {"nested": [1, {"a": True}]}
    assert diagnostic["metrics"] == {"novel/metric": 0.25}
    artifacts = read_events(run.path / "artifacts.jsonl")
    assert [event["kind"] for event in artifacts] == [
        "registered",
        "superseded",
        "registered",
    ]
    assert artifacts[1]["artifact_id"] == old
    assert all(
        event["path"] == "data/test.json" for event in artifacts if "path" in event
    )
    sequence = sorted(event["sequence"] for event in events + artifacts)
    assert sequence == list(range(len(sequence)))
    assert events[-1]["kind"] == "completed"
    assert json.loads((run.path / "run.json").read_text())["status"] == "completed"
    assert random.getstate() == expected_rng[0]
    actual_numpy = np.random.get_state()
    assert np.array_equal(actual_numpy[1], expected_rng[1][1])
    assert actual_numpy[2:] == expected_rng[1][2:]
    assert torch.equal(torch.get_rng_state(), expected_rng[2])


@pytest.mark.parametrize(
    "error,status",
    [(RuntimeError("training failed"), "failed"), (KeyboardInterrupt(), "interrupted")],
)
def test_failure_preserves_prior_history_and_propagates(repository, error, status):
    run = create_run(repository)
    with pytest.raises(type(error)):
        with run:
            run.record_event("training", progress={"optimizer_steps": 7})
            raise error
    events = read_events(run.path / "trajectory.jsonl")
    assert [e["kind"] for e in events] == ["started", "training", status]
    manifest = json.loads((run.path / "run.json").read_text())
    assert manifest["status"] == status
    assert manifest["final_progress"]["optimizer_steps"] == 7


def test_recording_io_failure_is_not_swallowed(repository, monkeypatch):
    run = create_run(repository)
    original_open = Path.open

    def failing_open(path, *args, **kwargs):
        if path.name == "trajectory.jsonl":
            raise OSError("disk full")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_open)
    with pytest.raises(OSError, match="disk full"):
        run.record_event("training", metrics={"loss": 1.0})
