# Does symmetric action grouping make search faster?

**Conclusion, October 9, 2026 benchmark:** grouping equivalent actions reduced
tree nodes but produced essentially unchanged replay-wide search throughput.
The saved result supports a structural reduction, not a general speedup or a
playing-strength improvement.

The benchmark used 128 frozen positions from each of the control/variant
iteration-119 replays, one gameplay checkpoint, fresh roots, one CPU Torch
thread, no root noise, warm-up, and three alternating paired sweeps. Each table
row covers 768 timed searches per mode. Ratio greater than one favors grouping.

| Boundary evaluator | Visits | Throughput ratio, on/off | Mean nodes, off → on |
| --- | ---: | ---: | ---: |
| Single-ending / next-deal | 32 | 1.002 | 288.9 → 240.2 |
| Single-ending / next-deal | 128 | 1.001 | 1008.6 → 875.7 |
| K10 / frozen score | 32 | 0.993 | 286.6 → 239.9 |
| K10 / frozen score | 128 | 1.001 | 1015.1 → 876.0 |

Diagnostics were collected separately from timings on 256 searches per mode.
Gameplay inference counts barely changed: approximately 32.7 states at 32
visits and 127.2–127.4 at 128 visits. Fewer retained nodes therefore did not
remove the dominant fixed-budget evaluation work in this setup. This is an
interpretation of the counters, not a measured attribution of all runtime.

Timing includes inference and expansion; it excludes setup, writes, diagnostics,
and warm-up. GC remained enabled. Equal seeds improve reproducibility but cannot
keep trajectories identical when action grouping changes search. No confidence
interval or performance gate was specified, and synthetic fixtures must not be
pooled with replay positions to advertise a larger gain.

**Next question:** profile time within transition, expansion and inference before
optimizing further, then repeat on the target environment. Preserve symmetry
safety checks and assess strength separately if grouping behavior changes.

[Selected settings, timing rows and diagnostics](evidence/symmetry.json) come
from completed run `20261009T134343Z-symmetric-mcts-benchmark-cc332d596e`.
The run recorded a dirty revision; its commit alone does not reconstruct the
executed source. These figures are a historical baseline, not current guarantees.
