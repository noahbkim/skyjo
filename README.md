# Skyjo

AI model, training, and gameplay for Skyjo

## Usage

Install the project in editable mode while developing:

```sh
uv sync
```

Then import the core game API directly from `skyjo`, or import supporting
modules from the package:

```python
import skyjo as sj
from skyjo import play, skynet

state = sj.new(players=2)
model_cls = skynet.EquivariantSkyNet
```

## Model checkpoint faceoffs

Run an even number of games so each checkpoint plays both seats from the same
seeded starting conditions:

```sh
uv run skyjo-faceoff \
  models/run_a/checkpoint_a.pth \
  models/run_b/checkpoint_b.pth \
  --games 20 \
  --mcts-iterations 100 \
  --workers 4
```

The command reads the `EquivariantSkyNet` architecture settings stored in the
checkpoints. Use `uv run skyjo-faceoff --help` to see model-setting overrides,
the random seed, device, terminal-rollout, progress, and worker options.

## Offline training

Replay buffers are saved as versioned NumPy dataset directories containing only
retained positions and complete-game metadata. Legacy replay pickle files are
not supported.

Train to an exact cumulative optimizer-step target with a deterministic
game-level validation split:

```sh
uv run python run_train_epoch.py data/training_data/RUN/dataset \
  --steps 100 \
  --output-checkpoint models/offline/step_100.pth
```

Resume by passing the prior checkpoint and a larger cumulative target:

```sh
uv run python run_train_epoch.py data/training_data/RUN/dataset \
  --checkpoint models/offline/step_100.pth \
  --steps 200 \
  --output-checkpoint models/offline/step_200.pth
```
