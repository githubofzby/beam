# BEAM-X Stage 1A: Standalone CostOracle

## Scope

Stage 1A implements the exact mathematical objective without calling or
changing legacy grouping, candidate scoring, matching, merge commit, or final
encoding. The module accepts only an immutable simple-undirected CSR graph and
a partition label vector.

Public API:

```cpp
EncodingCost BlockCost(size_a, size_b, edge_count, internal_block);
MergeGain ExactMergeGain(graph, labels, label_a, label_b);
EncodingCost ExactPartitionCost(graph, labels);
```

All costs and gains are signed 64-bit integers. Capacity arithmetic uses a
wider intermediate and rejects overflow. Gain is always `old_cost - new_cost`.

## Graph Semantics

- The graph is simple, undirected, and contains no self-loops.
- Repository CSR stores every undirected edge as two arcs; CostOracle counts
  only the `u < v` arc.
- Cross-block capacity is `|A| * |B|`.
- Internal-block capacity is `|A| * (|A| - 1) / 2`.
- A block with no actual edge contributes zero and therefore need not be
  materialized.

The exact block objective is:

```text
min(edges, 1 + capacity - edges)
```

## Correctness Tests

`tests/cost_oracle_test.cpp` independently recomputes complete partition cost
by enumerating all partition block pairs. It checks every legal merge against:

```text
ExactMergeGain(old, A, B)
==
ExactPartitionCost(old) - ExactPartitionCost(merge(old, A, B))
```

Coverage includes an empty in-memory graph, clique, path, star, complete
bipartite, two communities, internal blocks, cross blocks, 500 fixed-seed
random graphs, and validation failures. This empty graph is constructed in
memory; the repository text loader continues to reject empty input.

Test command:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -Iinclude \
  tests/cost_oracle_test.cpp src/cost_oracle.cpp -o /tmp/cost_oracle_test
/tmp/cost_oracle_test
```

Result: `CostOracle tests passed`.

## Compatibility Decision for Stage 1B

The requested integer objective and the current reporting-only
`cost_ratio_mags_compatible` are identical for sparse blocks and dense cross
blocks, but not for dense internal blocks.

For a clique represented by one internal supernode:

```text
CostOracle BlockCost = 1
current final MAGS-compatible reporting = 0.5
```

For an incomplete dense internal block, the current encoder also materializes
missing ordered pairs in both directions. Its reporting cost is therefore not
obtained from the specified block formula by one global scale factor.

Legacy exact gain has a related issue: internal adjacency is built from two CSR
arcs, while `EncodeCostForPair` uses undirected internal capacity. Stage 1A does
not preserve this drift; it implements the authoritative formula exactly.

Stage 1B adopts the first policy below after explicit approval:

1. Make the specified integer block objective authoritative and update final
   internal encoding/reporting to match it.
2. Define a different integer-scaled MAGS objective that exactly reproduces
   current loop-superedge reporting, then revise the requested formula.

The first policy matches the BEAM-X specification as written. The historical
reporting and encoding remain available as `--cost-objective legacy`; the
authoritative integer objective is enabled only by
`--cost-objective mags-compatible`.
