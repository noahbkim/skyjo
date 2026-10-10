"""Local experiment records. Payload semantics belong to their producers."""

from __future__ import annotations

import datetime
import hashlib
import importlib.metadata
import json
import logging
import pathlib
import platform
import re
import subprocess
import time
import uuid
from typing import Any, Self


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def file_digest(path: pathlib.Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git_provenance(repository: pathlib.Path, *, allow_dirty: bool) -> dict:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repository, check=True, capture_output=True, text=True
        ).stdout.strip()

    revision = git("rev-parse", "--verify", "HEAD")
    dirty = bool(git("status", "--porcelain", "--untracked-files=all"))
    if dirty and not allow_dirty:
        raise ValueError(
            "Repository has uncommitted changes; commit them or use --allow-dirty"
        )
    return {"commit": revision, "dirty": dirty, "allow_dirty": allow_dirty}


def _atomic_json(path: pathlib.Path, data: Any) -> None:
    text = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


class RunRecorder:
    """One parent-process writer; a fresh directory for each execution.

    No resume, schema migration, tensor serialization, or metric registry.
    Methods do not use Python, NumPy, or Torch random-number generators.
    """

    def __init__(self, path: pathlib.Path, manifest: dict):
        self.path = path
        self.manifest = manifest
        self.sequence = 0
        self.started = time.perf_counter()
        self.last_progress: dict = {}
        self._artifacts: dict[str, dict] = {}

    @classmethod
    def create(
        cls,
        *,
        root: pathlib.Path,
        repository: pathlib.Path,
        input_path: pathlib.Path,
        input_bytes: bytes,
        configuration: dict,
        entrypoint: str,
        invocation: list[str],
        allow_dirty: bool = False,
    ) -> RunRecorder:
        provenance = git_provenance(repository, allow_dirty=allow_dirty)
        # Validate serializability before leaving a run directory behind.
        json.dumps(configuration, allow_nan=False)
        name = configuration.get("name", "experiment")
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-")[:60] or "run"
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        run_id = f"{timestamp}-{slug}-{uuid.uuid4().hex[:10]}"
        path = root.resolve() / run_id
        manifest = {
            "run_id": run_id,
            "name": name,
            "created_at": utc_now(),
            "status": "running",
            "git": provenance,
            "entrypoint": entrypoint,
            "invocation": invocation,
            "working_directory": str(pathlib.Path.cwd()),
            "seed": configuration.get("seed"),
            "environment": {
                "python": platform.python_version(),
                "numpy": importlib.metadata.version("numpy"),
                "torch": importlib.metadata.version("torch"),
                "platform": platform.platform(),
                "device": configuration.get("execution", {}).get("device"),
                "dependency_lock_sha256": file_digest(repository / "uv.lock"),
            },
            "config_sha256": hashlib.sha256(
                json.dumps(configuration, sort_keys=True, allow_nan=False).encode()
            ).hexdigest(),
            "recovery": "Independent checkpoints and latest replay only; coherent run resume is not supported",
        }
        path.mkdir(parents=True, exist_ok=False)
        for directory in ("checkpoints", "data", "logs"):
            (path / directory).mkdir()
        (path / f"input-config{input_path.suffix}").write_bytes(input_bytes)
        _atomic_json(path / "resolved-config.json", configuration)
        notes = configuration.get("notes", "").strip()
        if notes:
            (path / "notes.md").write_text(f"# {name}\n\n{notes}\n", encoding="utf-8")
        for filename in ("trajectory.jsonl", "artifacts.jsonl"):
            (path / filename).touch(exist_ok=False)
        _atomic_json(path / "run.json", manifest)
        return cls(path, manifest)

    def _append(self, filename: str, kind: str, payload: dict) -> dict:
        event = {
            "run_id": self.manifest["run_id"],
            "sequence": self.sequence,
            "timestamp": utc_now(),
            "kind": kind,
            **payload,
        }
        line = json.dumps(event, sort_keys=True, allow_nan=False) + "\n"
        with (self.path / filename).open("a", encoding="utf-8") as stream:
            stream.write(line)
            stream.flush()
        self.sequence += 1
        return event

    def record_event(
        self,
        kind: str,
        *,
        progress: dict | None = None,
        metrics: dict | None = None,
        context: dict | None = None,
    ) -> dict:
        point = {
            **(progress or {}),
            "elapsed_seconds": time.perf_counter() - self.started,
        }
        event = self._append(
            "trajectory.jsonl",
            kind,
            {
                "progress": point,
                "metrics": metrics or {},
                "context": context or {},
            },
        )
        self.last_progress = point
        return event

    def register_artifact(
        self,
        path: pathlib.Path,
        *,
        kind: str,
        progress: dict,
        metadata: dict | None = None,
    ) -> str:
        path = path.resolve()
        relative = str(path.relative_to(self.path))
        if not path.exists():
            raise FileNotFoundError(path)
        artifact_id = uuid.uuid4().hex
        record = {
            "artifact_id": artifact_id,
            "artifact_kind": kind,
            "path": relative,
            "progress": progress,
            "metadata": metadata or {},
            "sha256": file_digest(path) if path.is_file() else None,
        }
        self._append("artifacts.jsonl", "registered", record)
        self._artifacts[artifact_id] = record
        self.record_event(
            "artifact_saved", progress=progress, context={"artifact_id": artifact_id}
        )
        return artifact_id

    def supersede_artifact(self, artifact_id: str, *, progress: dict) -> None:
        if artifact_id not in self._artifacts:
            raise KeyError(artifact_id)
        self._append(
            "artifacts.jsonl",
            "superseded",
            {
                "artifact_id": artifact_id,
                "progress": progress,
            },
        )

    def __enter__(self) -> Self:
        self.record_event("started")
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        status = (
            "completed"
            if exc is None
            else ("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed")
        )
        context = (
            {}
            if exc is None
            else {"exception": type(exc).__name__, "message": str(exc)}
        )
        if exc is not None:
            logging.getLogger(__name__).error(
                "Run %s", status, exc_info=(exc_type, exc, traceback)
            )
        self.record_event(status, progress=self.last_progress, context=context)
        self.manifest.update(
            status=status, finished_at=utc_now(), final_progress=self.last_progress
        )
        if context:
            self.manifest["error"] = context
        _atomic_json(self.path / "run.json", self.manifest)
        return False
