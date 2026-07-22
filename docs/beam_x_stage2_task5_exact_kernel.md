# BEAM-X Stage 2 Task 5: Specialized Persistent Exact-Gain Kernel

## Status

The specialized persistent CPU kernel is retained. Stage 2 is not closed and
Stage 3 must not start: the closing runtime gate passed on six datasets, but the
three-pair DB median was 1.120, above the required 1.10.

Task 4's negative result remains unchanged. Acquisition requests equal unique
representatives, so no `PreparedRowRegistry` was implemented. The delta overlay
also remains deferred because quotient update is not the dominant component.

## Implementation

`QuotientGraph::ExactMergeGainPersistent` reads the two immutable sorted raw
cross-adjacency rows directly. It uses a two-pointer union, excludes A and B
from the external-neighbor sum, records the A-B edge once, and handles the four
internal terms explicitly:

```text
cost(A,A) + cost(B,B) + cost(A,B) - cost(W,W)
```

Single-sided external neighbors use two block-cost evaluations; overlapping
neighbors use three. The kernel creates no synthetic internal entry, temporary
union row, hash table, or per-entry heap object. Backend/objective dispatch is
at the scoring-batch boundary. Version validation is once per prepared row
before scoring, not inside the entry loop.

Raw views are acquired during prepare and remain valid only until the next
commit. The specialized kernel obtains its raw rows from the current quotient
for each exact call and retains no row pointer after the call.

The kernel continues to call the frozen `CostOracle::BlockCost`. A proposed
capacity-based helper was not retained because Stage 1 freezes
`include/cost_oracle.hpp` and `src/cost_oracle.cpp`. Duplicating the block-cost
formula locally would create a second objective implementation. Consequently,
Task 5 removes iterator, synthetic-entry, repeated edge lookup, and temporary
row costs, but does not yet replace the merged-capacity multiplication with a
precomputed-capacity CostOracle API.

Modified files:

- `include/quotient_graph.hpp`, `src/quotient_graph.cpp`: raw-row kernel and
  work counters.
- `include/sweg.hpp`, `src/scoring_cpu.cpp`, `src/sweg.cpp`: persistent-only
  batch dispatch and padded per-thread profiling reduction.
- `src/main.cpp`: summary and long-form CSV metrics.
- `tests/quotient_graph_test.cpp`, `tests/quotient_row_view_test.cpp`:
  specialized/reference comparisons.

## Profiling Counters

Task 5 adds:

```text
exact_persistent_pairs
exact_raw_entries_a
exact_raw_entries_b
exact_union_neighbors
exact_overlap_neighbors
exact_single_sided_neighbors
exact_internal_block_terms
exact_block_cost_evaluations
exact_capacity_multiplications
```

Profiling-off selects a compile-time no-counter kernel. Profiling-on uses local
candidate scalars and one reduction to padded per-thread counters; there are no
atomics or clocks in the entry loop.

For the two-community smoke graph, 19 exact pairs scanned 78 raw row entries
and 38 external union neighbors: 10 overlap and 28 single-sided. It evaluated
162 block costs and four internal terms per candidate.

## Correctness

The deterministic smoke gate passed on triangle, path, clique, and
two-community. Legacy and persistent produced identical exact cost, partition,
supernode count, exact calls, merge count, and byte-identical `G/P/Cp/Cm`; all
outputs reconstructed losslessly. Persistent copied entries and bytes remain
zero.

A medium gate used 20 fixed-seed random graphs, one fixed merge order, every
third merge as a checkpoint, and the endpoint as the final checkpoint. Every
legal specialized gain equaled the existing quotient reference gain; checkpoint
rows and exact cost equaled a full quotient rebuild.

The Stage 2 closing correctness suite was then run once. It covered 300 random
graphs, 20 merge orders, full CostOracle comparison after every merge, all
quotient update modes, both CPU merge modes, seven deterministic graphs, and
lossless reconstruction. It passed. Stage 0 and Stage 1 freeze checksums also
pass.

## Task 5 Old/New Kernel Results

These runs compare persistent Task 4 and persistent Task 5 under the
MAGS-compatible objective. Absolute time varied, so old/new same-window ratios
are the useful evidence.

| Dataset | Old exact median ms | New exact median ms | Exact change | Calls unchanged |
|---|---:|---:|---:|---:|
| FA | 187.331 | 126.396 | -32.5% | yes |
| EM | 897.372 | 597.640 | -33.4% | yes |
| AM | 5,322.077 | 5,775.099 | +8.5% | yes |
| DB | 5,671.294 | 6,494.551 | +14.5% | yes |

FA and EM satisfy the Task 5 performance gate. AM and especially DB are noisy
or negative; DB's old runs differed by 48%, so those two-pair results are not a
stable regression estimate. No further branchless, SIMD, prefetch, allocator,
or delta-overlay work was attempted.

## Stage 2 Closing Runtime Gate

The state-backend closing comparison uses the frozen legacy objective and
Stage 0 algorithm parameters. This is intentionally separate from the Task 5
MAGS-compatible old/new kernel experiment: combining the two would invoke the
full legacy CostOracle path and would not reproduce the frozen baseline.

| Dataset | Legacy median ms | Persistent median ms | Paired ratio median | Gate |
|---|---:|---:|---:|---|
| FA | 301.968 | 267.875 | 0.907 | pass |
| EM | 2,002.351 | 2,001.212 | 1.011 | pass |
| AM | 11,629.772 | 11,854.869 | 1.032 | pass |
| DB | 11,000.188 | 12,451.065 | 1.120 | fail |

One adjacent pair was run on each large dataset:

| Dataset | Legacy ms | Persistent ms | Ratio | Compression identical |
|---|---:|---:|---:|---|
| YO | 44,148.782 | 45,659.334 | 1.034 | yes |
| RN | 65,191.372 | 71,268.111 | 1.093 | yes |
| LJ | 254,394.913 | 258,013.994 | 1.014 | yes |

All datasets have identical exact-call count, merge count, supernode count, and
MAGS-compatible compression ratio between legacy and persistent state.

## Decision

Task 5 meets its local retention gate and is correctness-complete. Stage 2 is
not ready to close because DB misses the state-backend runtime gate by about two
percentage points. The next action should be a narrowly scoped DB paired audit
that separates prepare, candidate generation, exact scoring, and update under
the legacy objective. It must not change CandidateIndex or enter Stage 3.
