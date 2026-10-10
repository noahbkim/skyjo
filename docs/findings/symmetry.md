# Does symmetric action grouping make search faster?

**Conclusion, October 9, 2026:** grouping equivalent actions reduced tree nodes
without improving replay-wide search throughput. Playing strength was not tested.

Benchmark: 256 positions split equally between iteration-119 control/variant
replays, one checkpoint, fresh roots, one CPU Torch thread, no root noise.
Three alternating paired sweeps after warmup gave 768 searches per mode per row.
Ratios above one favor grouping.

| Boundary evaluator | Visits | Throughput ratio, on/off | Mean nodes, off → on |
| --- | ---: | ---: | ---: |
| Single-ending / next-deal | 32 | 1.002 | 288.9 → 240.2 |
| Single-ending / next-deal | 128 | 1.001 | 1008.6 → 875.7 |
| K10 / frozen score | 32 | 0.993 | 286.6 → 239.9 |
| K10 / frozen score | 128 | 1.001 | 1015.1 → 876.0 |

Separate diagnostics on 256 searches per mode found almost unchanged inference
counts: 32.7 states at 32 visits and 127.2–127.4 at 128. This suggests node savings
did not remove the dominant evaluation work; runtime was not directly attributed.

Timings include inference/expansion, exclude setup/diagnostics, and keep GC
enabled. Grouping changes trajectories despite equal seeds. No confidence
interval was specified; synthetic fixtures cannot establish replay-wide gains.

**Next question:** profile transition, expansion and inference on the target
environment. Preserve symmetry safety checks and assess strength separately.

[Evidence](evidence/symmetry.json): run
`20261009T134343Z-symmetric-mcts-benchmark-cc332d596e`, settings, timings and
diagnostics. The recorded dirty revision prevents exact reconstruction from
its commit alone.
