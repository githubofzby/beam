# BEAM-X Stage 2 Task 1: Zero-Copy Quotient Row Views

## Change

Persistent CPU prepare no longer exports each quotient row into a group-local
`vector<pair<int,int64_t>>`. `QuotientRowView` references the sorted cross-row
storage and exposes the legacy doubled internal-edge entry through a logical
iterator at its deterministic sorted position. No temporary vector is created
for the internal entry.

`PreparedRow` gives candidate discovery and CPU exact scoring one read-only
range abstraction. Legacy state owns its prepared rows; persistent state stores
views. CUDA flattening still explicitly packs rows because device input must be
contiguous. Views are created only during prepare and are discarded before the
corresponding commit. A representative version check detects stale iteration.

Modified files:

- `include/quotient_graph.hpp`, `src/quotient_graph.cpp`: row view and iterator.
- `include/sweg.hpp`, `src/sweg.cpp`: prepared range and profiling.
- `src/scoring_cpu.cpp`: direct CPU scoring over views.
- `src/main.cpp`: Task 1/2 metric output.
- `tests/quotient_row_view_test.cpp`: checkpoint rebuild comparison.

## Profiling Metrics

Task 1 adds:

```text
prepare_row_entries_copied
prepare_row_copy_bytes
prepare_row_views_created
quotient_row_lookup_ms
quotient_row_copy_ms
quotient_exact_gain_ms
```

Task 2 adds batch/merge-boundary metrics rather than per-lookup clocks:

```text
quotient_incremental_update_ms
quotient_reciprocal_update_ms
quotient_memory_allocation_ms
quotient_sort_or_merge_ms
quotient_rebuild_ms
quotient_rows_updated
quotient_entries_inserted/removed/shifted
quotient_high_degree_rows_touched
quotient_max_row_degree
quotient_allocated_bytes
quotient_peak_nnz
```

Allocated bytes is a vector-capacity estimate, not process RSS. Profiling-off
runs do not collect quotient substage clocks or shift counters.

## Validation

The routine smoke gate used triangle, path, clique, and two-community with one
fixed blocked merge path. Legacy and persistent produced identical partition,
exact cost, MAGS-compatible ratio, and byte-identical `G/P/Cp/Cm`; all four
reconstructed losslessly. Persistent copied entries and bytes were zero.

Because this is a merge-ready data-structure milestone, a medium test used 25
fixed-seed random graphs and one fixed merge order. It compared row views and
ExactCost with full rebuild at every third merge and at completion. It passed.
The 300-graph full suite was intentionally not repeated.

## Paired Runtime

Runs used the order `L P / P L / L P`, 24 OMP threads, close/core affinity,
seed 1, and Stage 0 algorithm parameters. Runtime runs used profiling off; one
additional persistent summary run collected the work breakdown. At experiment
start load average was 0.49 and reported CPU scaling was 46%, so paired ratios
are preferred over comparison with older runs.

| Dataset | Legacy median ms | Persistent median ms | Paired ratio median |
|---|---:|---:|---:|
| FA | 275.651 | 279.898 | 0.993 |
| EM | 1,807.963 | 1,777.474 | 0.981 |
| AM | 9,132.153 | 9,463.822 | 1.031 |
| DB | 8,786.514 | 9,008.139 | 1.025 |

All four satisfy the current gate `persistent <= 1.10 * legacy`. Complete
min/median/max and all three paired ratios are in
`docs/beam_x_stage2_task1_paired.csv`.

## Remaining Bottleneck and Task 3 Decision

Persistent prepare reported zero copied entries/bytes on all profiled graphs.
The additional persistent summary runs showed:

| Dataset | Row lookup ms | Exact gain ms | Incremental update ms | Reciprocal ms | Max row degree | High-degree touches |
|---|---:|---:|---:|---:|---:|---:|
| FA | 9.6 | 114.4 | 11.8 | 10.0 | 1,044 | 0 |
| EM | 92.9 | 659.7 | 32.2 | 22.9 | 1,382 | 0 |
| AM | 1,314.6 | 5,257.1 | 191.9 | 114.4 | 548 | 0 |
| DB | 1,080.1 | 5,025.3 | 187.2 | 111.0 | 423 | 0 |

Sorted-row shifts occur, but reciprocal update is only a small fraction of
algorithm time and no row reached the provisional degree-4096 threshold.
Therefore profiling does not justify a delta-overlay implementation yet.
Task 3 is deferred. The next Stage 2 work should reduce row-view acquisition
overhead and avoid millions of repeated prepared views, while leaving candidate
count, scheduler, and objective unchanged.
