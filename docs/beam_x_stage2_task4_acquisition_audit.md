# BEAM-X Stage 2 Task 4: Prepared-Row Acquisition Audit

## Decision

The proposed `PreparedRowRegistry` was not implemented. Instrumentation of the
actual prepare/commit epoch found that every active representative is acquired
exactly once. A registry would therefore eliminate no `GetRowView` calls and
would add a row-id lookup and another indirection to candidate discovery and
exact scoring.

This follows the Task 4 stop condition. Stage 2 remains active; no Stage 3
candidate, scheduler, top-k, threshold, merge, objective, or CUDA behavior was
changed.

## Acquisition Audit

The persistent path uses a reusable versioned marker array when profiling is
enabled. In blocked mode an epoch is the set of groups prepared before one
commit; in non-blocked mode it is one group before its commit. The audit is
outside adjacency-entry loops and performs no work when profiling is off.

Added metrics:

```text
prepare_unique_representatives
prepare_row_acquisition_requests
prepare_row_acquisition_hits
prepare_row_acquisition_misses
prepare_duplicate_acquisitions_avoided
prepare_row_registry_entries
prepare_row_registry_bytes
```

The last two fields describe an actually allocated registry and remain zero,
because the audit rejected the registry hypothesis.

| Dataset | Requests | Unique | Hits | Misses | Requests / unique | Avoided / requests |
|---|---:|---:|---:|---:|---:|---:|
| triangle | 5 | 5 | 0 | 5 | 1.000 | 0.000% |
| path | 17 | 17 | 0 | 17 | 1.000 | 0.000% |
| clique | 6 | 6 | 0 | 6 | 1.000 | 0.000% |
| two-community | 18 | 18 | 0 | 18 | 1.000 | 0.000% |
| FA | 45,120 | 45,120 | 0 | 45,120 | 1.000 | 0.000% |
| EM | 318,150 | 318,150 | 0 | 318,150 | 1.000 | 0.000% |

All runs satisfy:

```text
requests == hits + misses
views_created == misses
misses == unique_representatives
```

## Exact-Scoring Constant-Cost Optimization

After rejecting acquisition deduplication, inspection found that
`QuotientRowView::Iterator::operator*` and `operator[]` were out-of-line in a
separate translation unit. Without LTO, the two-row exact-gain scan could make
a function call for every logical row entry. Both operations are now inline in
`include/quotient_graph.hpp`. Stale-view validation moved from iterator entry to
the prepared-group scoring boundary, so each row version is checked once before
parallel scoring rather than during the entry scan.

Views still exist only between prepare and commit. A quotient update invalidates
their recorded version; no view is retained across a commit or epoch.

An old/new same-window isolation run reported:

| Dataset | Build | Algorithm ms | Quotient exact ms | Row lookup ms |
|---|---|---:|---:|---:|
| FA | old | 319.800 | 133.752 | 7.603 |
| FA | new | 239.673 | 74.235 | 9.137 |
| FA | new | 226.614 | 75.559 | 10.445 |
| FA | old | 270.051 | 117.278 | 9.705 |
| EM | old | 1,657.946 | 596.662 | 79.314 |
| EM | new | 1,746.719 | 597.858 | 95.428 |
| EM | new | 1,602.894 | 512.539 | 85.714 |
| EM | old | 1,679.139 | 600.734 | 83.783 |

FA shows a clear exact-scoring reduction. EM is mixed-positive and affected by
run-to-run noise. Row lookup itself does not improve, which is consistent with
the measured zero duplicate acquisitions. The requested 20% lookup reduction
is therefore not a valid success gate for this workload.

## Lightweight Correctness Gate

The release-equivalent build used:

```text
-std=c++17 -O3 -DNDEBUG -fopenmp -Iinclude
```

Triangle, path, clique, and two-community were run in one fixed blocked merge
mode for legacy and persistent state. All four had identical exact cost,
partition, supernode count, exact-call count, selected merge count, and
byte-identical `G/P/Cp/Cm`; all reconstructed losslessly. Persistent copied row
entries and bytes remained zero. Stage 0 and Stage 1 freeze checksums pass.

The full 300-random-graph suite was intentionally not rerun during this
optimization loop.

## Lightweight FA/EM Paired Check

Order was `L P / P L`, with two runs per backend:

| Dataset | L1 ms | P1 ms | P2 ms | L2 ms | P1/L1 | P2/L2 |
|---|---:|---:|---:|---:|---:|---:|
| FA | 302.652 | 325.536 | 258.162 | 270.111 | 1.076 | 0.956 |
| EM | 1,699.207 | 1,719.935 | 1,748.169 | 1,703.125 | 1.012 | 1.026 |

Both datasets remain within the current `persistent <= 1.10 * legacy` gate.
These are lightweight checks, not the three-replicate four-dataset acceptance
benchmark, so AM/DB were not run.

## Next Bottleneck

Prepared-row acquisition should not be optimized further. Task 3 delta overlay
also remains unjustified because update and reciprocal-update time are not the
dominant components. The next Stage 2 target is the constant cost of the CPU
exact-gain two-row merge: iterator representation, internal logical-entry
handling, and repeated capacity/block-cost evaluation. Any further change must
preserve candidate and exact-call counts and use the same lightweight gate
before a milestone validation.
