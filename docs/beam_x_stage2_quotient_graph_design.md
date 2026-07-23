# BEAM-X Stage 2: Persistent QuotientGraph Design

## Legacy Rebuild Path

For every active supernode participating in a group,
`CreateWForSupernodeWithScratch` walks all original CSR arcs incident to its
member nodes. `AggregateWBySupernodeWithScratch` then maps original endpoints
through `S_` and sorts the resulting quotient row. This happens again in later
iterations even if most of the partition is unchanged. `Update_S` updates only
the member linked list and labels; it retains no supernode adjacency.

## State and Semantics

`QuotientGraph` owns state indexed by stable representative id:

```text
size[A]
internal_edges[A]
adj[A][B]
version[A]
children[A]
active[A]
```

The graph is simple, undirected, and loop-free. `internal_edges[A]` counts each
original undirected edge once. A cross-block count is stored symmetrically in
both endpoint rows. Zero-count blocks are implicit.

For compatibility with the existing scorer, `ExportLegacyRow(A)` emits cross
counts once and the internal entry as `2 * internal_edges[A]`, matching the
repository CSR arc convention. CostOracle-facing methods always use the
undirected count.

## Incremental Merge

Merging `remove` into `keep` computes:

```text
size[keep] += size[remove]
internal[keep] += internal[remove] + edges(keep, remove)
edges(keep, x) = edges(keep, x) + edges(remove, x)
```

The smaller adjacency row is iterated first where possible, while reciprocal
entries are updated in place. The removed row is cleared only after its
neighbors have been redirected. Versions increment for the kept endpoint,
removed endpoint, and every neighbor whose reciprocal count changes.

## Validation and Rollout

The standalone gate compares all state, rows, capacities, block costs, and
partition cost against reconstruction from the immutable original graph after
every merge in exhaustive named and fixed-seed random small graphs.

Integration is feature flagged as `--state-backend legacy|persistent`, with
legacy remaining the default. The first integrated version changes only how
prepared aggregate rows are obtained; grouping, proxy ranking, exact scoring,
matching, and `Update_S` remain unchanged. Debug validation retains the Stage
1 full recomputation oracle.

## Delta-or-Rebuild

Three feature-flagged update modes are implemented:

- `incremental`: merge sorted rows and update reciprocal frontier rows.
- `bulk_rebuild`: apply endpoint-disjoint mappings while scanning the current
  quotient edge stream once; it never scans original CSR edges.
- `auto`: select bulk only when the batch reduces at least 5% of active
  supernodes and estimated incremental work exceeds current quotient NNZ.

Forced bulk rebuild is an ablation mode. With the current group-level scheduler
it repeatedly scans the whole quotient for small commit batches and is slow.
The conservative auto condition keeps those batches incremental; later
large-batch stages can create conditions where rebuild is appropriate.

## Profiling Findings During Implementation

The first prototype used `vector<unordered_map>` and copied/sorted every row.
It eliminated original-edge scans but regressed AM and DB due to hash updates
and locality. A second prototype cached sorted copies; on LJ this duplicated
tens of millions of entries and worsened divide, prepare, and scoring time.
Both designs were rejected based on profiling.

The retained representation uses one sorted flat row per active supernode,
linear keep/remove row merge, binary-search reciprocal updates, and no duplicate
child storage in the integrated path. Standalone validation still enables
children for state-completeness tests.
