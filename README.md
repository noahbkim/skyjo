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

## Auxiliary objectives

Set `auxiliary_objectives` on `EquivariantSkyNet` (or in the model factory's
`model_kwargs`). The experiment configurations in `main.py` and
`distributed_main.py` expose this mapping and use 32 terminal samples for every
ablation, including the baseline:

```python
auxiliary_objectives = {}  # baseline
auxiliary_objectives = {"round_raw_score": 0.1}  # score only
auxiliary_objectives = {"round_doubled": 0.1}  # doubling only
auxiliary_objectives = {"round_raw_score": 0.1, "round_doubled": 0.1}
```

The named configuration `configs/round_score_doubling.json` enables both
objectives at weight 0.1, with 32 terminal samples and target seed 0:

```sh
uv run python distributed_main.py --config configs/round_score_doubling.json
```

This selects the learning objectives while retaining the distributed runner's
other settings (currently 2 learning iterations, 8 games per iteration, and
2 training epochs). Running without `--config` selects the core-only baseline.

The model's resolved configuration controls its heads, replay fields, and
additional losses in `train_step`. Omitted or zero-weight objectives are fully
disabled. Core-only forward calls retain their original two-field output;
enabled models additionally expose an `auxiliary` mapping. Search and gameplay
adapters use only the existing value and policy outputs.

Self-play workers return completed histories. The learner calls
`targets.build_targets(history, model.objectives, terminal_rollouts=32,
rng=target_rng)` before inserting rows into replay. Create one dedicated
`random.Random(target_seed)` per learning run and reuse it across histories.
Both auxiliary objectives share the terminal sampling pass already needed by
the core value target. Minibatch training never resimulates an ending.

The sampler fixes the final decision and resamples its random transition and
automatic reveals. Raw scores are measured after clearing and before doubling;
doubling labels are the frequency of the rule applying, including ties and
no-progress penalties. Each earlier state receives these averaged labels in
current-player order. This is supervision conditional on the played trajectory,
not a continuation search from every state. Penalized scores and value targets
are computed per sampled ending before averaging.

Raw score uses `MSE(prediction / 144, target / 144)`; doubling uses
`BCEWithLogits` against the sampled frequency. Logs include each unweighted and
weighted loss and raw-score MAE in points. The default experimental weight is
0.1 for each enabled objective.

New replay buffers are configured with
`buffer.for_objectives(buffer_config, model.objectives)`. Saved buffers must
contain all enabled labels; extra labels can be ignored when training a subset
of objectives. Start each ablation with fresh weights. Checkpoints retain a
`.pth` state dictionary and add a neighboring `.json` model configuration; keep
both files together. `EquivariantSkyNet.from_checkpoint(path, device)` restores
the architecture and objective weights. Model factories also use this metadata.

To train one epoch from labeled replay:

```sh
uv run python run_train_epoch.py data/replay.pkl \
  --auxiliary-objectives '{"round_raw_score": 0.1, "round_doubled": 0.1}'
```

When `--weights` references a new checkpoint, the CLI inherits its objective
configuration unless an explicitly supplied configuration matches it.

To run a reproducible offline comparison of all four ablations:

```sh
uv run python run_auxiliary_ablation.py --seeds 0,1 --terminal-rollouts 32
```

This shares initial-model MCTS histories, core initialization, target samples,
and minibatch indices across arms. The JSON report records target construction
time, training time, losses, and policy-only play against greedy EV in both
seats. Small budgets validate wiring; larger multi-seed experiments are needed
to establish playing strength.

### Adding another auxiliary objective

Register an `objectives.Objective` with a target-shape function, context
dependency name, per-state target extractor, head factory, and loss/metrics
function. Add a builder to `targets.CONTEXT_BUILDERS` if it needs new
history-derived context. Builders run once per history; objectives sharing a
dependency reuse its result. Extractors return labels in the state's current
perspective. No changes to the generic training loop are needed. Context
builders receive isolated random streams keyed by dependency name so adding
one cannot perturb existing targets.
