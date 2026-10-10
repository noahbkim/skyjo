"""Explicit checkpoint/replay persistence called by experiment recipes."""

from pathlib import Path

from skyjo.learning import checkpoint, replay_io

from .state import Snapshot, TrainingState


class RunArtifacts:
    def __init__(
        self, recorder, replay_path: Path, *, initial_dataset=None, dataset_id=None
    ):
        self.recorder = recorder
        self.replay_path = replay_path
        self.replay_artifact = None
        self.previous_dataset_id = dataset_id
        self.initial_buffer = (
            None
            if initial_dataset is None
            else {
                "path": str(initial_dataset),
                "dataset_id": dataset_id,
            }
        )

    def save_checkpoint(
        self, path, learner, configuration, state: TrainingState, *, role
    ):
        path = checkpoint.save_checkpoint(
            path,
            model=learner.model,
            optimizer=learner.optimizer,
            configuration=configuration,
            progress=state.progress,
            continuation_state=state.continuation_state(),
            sampling_rng=learner.sampling_rng,
        )
        artifact_id = None
        if self.recorder is not None:
            artifact_id = self.recorder.register_artifact(
                path,
                kind="checkpoint",
                progress=state.point(),
                metadata={"role": role},
            )
            self.recorder.record_event(
                "checkpoint_saved",
                progress=state.point(),
                context={
                    "artifact_id": artifact_id,
                    "role": role,
                },
            )
        return Snapshot(path, artifact_id)

    def save_replay(
        self,
        replay,
        *,
        state,
        generation,
        generation_iteration,
        game_count,
        generation_settings,
    ):
        batch = (
            None
            if generation_iteration is None
            else {
                "run_id": self.recorder.manifest["run_id"] if self.recorder else None,
                "generation_iteration": generation_iteration,
                "game_count": game_count,
                "checkpoint_artifact_id": generation.artifact_id
                if generation
                else None,
                "checkpoint_path": str(generation.path) if generation else None,
                "settings": generation_settings,
            }
        )
        metadata = {
            "initial_buffer": self.initial_buffer,
            "previous_dataset_id": self.previous_dataset_id,
            "latest_generated_batch": batch,
        }
        if self.recorder is not None and self.replay_artifact is not None:
            self.recorder.supersede_artifact(
                self.replay_artifact, progress=state.point()
            )
        path = replay_io.save(replay, self.replay_path, generation_metadata=metadata)
        self.previous_dataset_id = replay.dataset_id
        if self.recorder is not None:
            self.replay_artifact = self.recorder.register_artifact(
                path,
                kind="replay_data",
                progress=state.point(),
                metadata={
                    "dataset_id": replay.dataset_id,
                    "retention": "latest_only",
                    "manifest": replay.dataset_metadata,
                    **metadata,
                },
            )
