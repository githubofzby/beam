# BEAM-X Stage 1B: CostOracle Integration

## Design

Stage 1B is a correctness reference path, not a performance optimization.
`--cost-objective legacy` remains the default and preserves Stage 0 behavior.
`--cost-objective mags-compatible` routes CPU exact candidate scoring through
the Stage 1A block formula and revalidates every selected merge against a full
recomputation of the current partition immediately before commit.

The authoritative mode also materializes undirected internal-block edges and
negative corrections once. Its three cost views must agree:

```text
encoding payload count
== ExactPartitionCost(partition)
== encoding_cost_mags_compatible
```

CUDA is rejected for this mode because its existing kernel implements legacy
row semantics. This prevents silent objective mixing. Stage 2 will replace the
deliberately expensive pre-commit full recomputation with persistent quotient
state.

## Files and Interfaces

- `include/sweg.hpp`: `CostObjective`, exact partition cost access, validation
  bridge.
- `src/sweg.cpp`: authoritative local gain, pre-commit monotonic gate, and
  internal-block materialization.
- `include/compression_metrics.hpp`: objective-aware final reporting.
- `src/main.cpp`: `--cost-objective` CLI, metrics, and final cost invariant.
- `tests/cost_oracle_integration_test.py`: lossless and cost-equality gate for
  both CPU merge modes on seven small graphs.

## Correctness Commands

```bash
g++ -std=c++17 -O2 -fopenmp -Iinclude \
  src/main.cpp src/sweg.cpp src/scoring_cpu.cpp src/cost_oracle.cpp \
  src/graph_io/graph_io.cpp src/cuda_scoring_stub.cpp -o /tmp/beam_stage1

/tmp/cost_oracle_test_stage1
/tmp/compression_metrics_test_stage1
OMP_NUM_THREADS=1 python3 tests/profiling_equivalence_test.py \
  --binary /tmp/beam_stage1
OMP_NUM_THREADS=1 python3 tests/cost_oracle_integration_test.py \
  --binary /tmp/beam_stage1
```

Expected results: all standalone CostOracle tests pass, legacy profiling output
remains byte-identical, all authoritative-mode summaries reconstruct exactly,
and every final payload cost equals the full partition recomputation.

## Known Limits

- The new mode is CPU-only.
- Full partition reconstruction before each selected merge is intentionally
  slow and may scan the original graph many times.
- Stage 1B does not claim a large-dataset speedup and must not be used as the
  Stage 2 performance baseline.
