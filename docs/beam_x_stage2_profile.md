# BEAM-X Stage 2: Persistent QuotientGraph Results

## Status

Stage 2 passes correctness and work-elimination gates but fails the runtime
gate. Stage 3 must not start yet. The retained implementation remains behind
`--state-backend persistent`; legacy is still the default.

## Configuration

The release-equivalent binary was compiled with GCC using:

```text
-std=c++17 -O3 -DNDEBUG -fopenmp -Iinclude
```

CUDA was disabled. Runs used 24 OMP threads, `OMP_PROC_BIND=close`,
`OMP_PLACES=cores`, seed 1, 20 iterations, top-k 16, group batch 64, reciprocal
threshold, legacy objective, persistent state, and auto quotient update.

## Correctness

`tests/quotient_graph_test.cpp` covers named graphs and 300 random graphs with
multiple merge orders. After every merge it compares size, internal edges,
all cross counts, exported legacy rows, exact partition cost, and every legal
exact merge gain against full reconstruction from the original graph.

`tests/quotient_graph_integration_test.py` compares legacy, incremental,
bulk-rebuild, and auto state configurations across both CPU merge modes and
seven small graphs. All produce identical `G/P/Cp/Cm` and pass lossless
reconstruction. Debug mode validates quotient state after every commit.

All Stage 0 and Stage 1 freeze checksums still pass.

## Seven-Dataset Results

Runtime is algorithm time and excludes graph loading. The legacy column is the
frozen Stage 0 run; persistent results were collected in a later system-load
window. Current load and CPU frequency varied substantially, so the cross-run
runtime ratios are evidence of failure to demonstrate a speedup, but they are
not a controlled paired estimate of the exact regression size.

| Dataset | Legacy ms | Persistent ms | Speedup | Legacy ratio | Persistent ratio | Prepare CSR scans |
|---|---:|---:|---:|---:|---:|---:|
| FA | 275.951 | 347.109 | 0.795x | 0.491347 | 0.491347 | 0 |
| EM | 1,783.740 | 2,211.289 | 0.807x | 0.695666 | 0.695666 | 0 |
| AM | 9,385.633 | 17,991.926 | 0.522x | 0.598288 | 0.598288 | 0 |
| DB | 8,815.505 | 18,731.440 | 0.471x | 0.542040 | 0.542040 | 0 |
| YO | 58,429.643 | 106,693.647 | 0.548x | 0.732916 | 0.732916 | 0 |
| RN | 95,987.109 | 166,795.329 | 0.575x | 0.707981 | 0.707981 | 0 |
| LJ | 266,674.080 | 478,325.622 | 0.558x | 0.749757 | 0.749757 | 0 |

All final supernode and `P/Cp/Cm` counts also match Stage 0 exactly. Full data
and quotient work counters are in `docs/beam_x_stage2_results.csv`.

## What the Data Establishes

Confirmed:

- Prepare no longer scans original CSR arcs after quotient initialization.
- Persistent rows preserve all legacy decisions and final compression.
- Incremental quotient gain exactly matches full CostOracle recomputation.
- Bulk rebuild from the quotient stream is correct and scans no original edge.
- Conservative auto mode selected zero bulk rebuilds for all seven datasets,
  because current group-level commit batches are too small.

Not confirmed:

- Persistent state is not faster in the current integration.
- The runtime regression cannot be dismissed as instrumentation overhead.
- Stage 2 therefore does not satisfy its runtime acceptance requirement.

## Bottleneck Evidence

The retained path still copies quotient rows into every prepared group. More
importantly, persistent state adds a large memory footprint and irregular
reciprocal row mutations. LJ finishes with 48,482,464 quotient NNZ and touches
52,521,601 entries during updates. Its measured phase times were approximately:

```text
divide       113.5 s
prepare       92.4 s
exact gain   157.2 s
update        89.8 s
```

Thus eliminating CSR scans did not eliminate row materialization, candidate
work, or exact scoring work. High-degree flat-row reciprocal insert/erase also
becomes expensive. A forced bulk-rebuild AM ablation exceeded three minutes
before termination because small group commits repeatedly scanned the entire
quotient stream. Auto initially triggered this failure mode; requiring at least
5% active-supernode reduction prevented it.

## Required Work Before Stage 3

1. Add a read-only quotient row view so prepare/scoring consumes persistent
   storage directly instead of copying rows per group.
2. Replace high-degree reciprocal vector insert/erase with a hybrid mutable
   representation or deferred delta overlay, while preserving sorted scoring
   views.
3. Accumulate commits at an iteration-level boundary before considering bulk
   rebuild; group-level full rebuild is structurally inappropriate.
4. Repeat paired `legacy/persistent` runs on FA/EM/AM/DB and report medians.
5. Repeat seven-dataset profiling only after the paired runtime gate improves.

Until these steps succeed, Stage 2 remains open and Stage 3 candidate-index work
is deferred.
