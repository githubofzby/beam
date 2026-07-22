# BEAM-X Stage 3A: Quotient-Neighbor CandidateIndex Prototype

## Stage Transition

Stage 2 is closed as functionally complete and correctness accepted, with DB's
1.120 persistent/legacy runtime ratio recorded as performance debt. Stage 3A
starts behind an opt-in feature flag; legacy candidate generation remains the
default.

## Interface and Lifetime

`CandidateIndex` separates proposal from exact scoring. It exposes
`BuildOrRefresh`, per-source `Propose`, and `OnMerge`. The first implementation,
`QuotientNeighborCandidateIndex`, holds only epoch-local references to the
persistent quotient and active representative list. It stores no row view
across a commit.

The blocked prepare path treats original group members as sources, but proposals
may target any active quotient representative. External targets are appended as
read-only prepared rows, so grouping is no longer a hard candidate boundary.
Pairs are deterministically sorted and deduplicated both per source and across
the commit epoch. Stale sources and targets created by an earlier cross-group
merge are rejected at the proposal boundary.

Flags:

```text
--candidate-index legacy|quotient-neighbor
--candidate-budget <positive integer>
```

The quotient-neighbor path currently requires persistent CPU state. CUDA and
legacy candidate behavior are unchanged.

## Proposal Rules

For each source, the prototype considers:

1. strongest direct quotient neighbors by block edge count;
2. candidates reached through the two strongest direct-neighbor pivots, ranked
   by the minimum of the two path weights;
3. one fixed-seed deterministic exploration representative.

Each channel uses bounded ordered selection and stores O(budget) proposals; it
does not copy or sort a full high-degree row. These scores only rank proposals.
The existing exact gain, threshold, matching, and commit paths remain decisive.

## Profiling

Added metrics:

```text
candidate_index_build_ms
candidate_index_refresh_ms
candidate_proposal_ms
candidate_proposals_raw
candidate_proposals_unique
candidate_duplicates_removed
candidate_budget_exhausted_nodes
candidate_nodes_with_zero_proposals
candidate_direct_neighbor_count
candidate_shared_neighbor_count
candidate_exploration_count
```

They are available in stdout and long-form rounds CSV. CandidateIndex mode and
budget are recorded in the results CSV.

## Validation

Triangle, path, clique, and two-community pass deterministic repeated-output,
valid-endpoint, exact-decision, and lossless reconstruction checks. The new path
is allowed to choose a different partition; repeated runs with the same seed
produce byte-identical `G/P/Cp/Cm`.

A same-checkpoint recall test used 20 fixed-seed random graphs and three
checkpoints per graph. With budget 8:

```text
positive-gain pair recall = 97.39%
best-positive-pair recall  = 95.00%
```

This passes the 90% diagnostic recall line, but static small-graph recall did
not predict multi-round compression quality on FA.

## FA Funnel and Stop Decision

The persistent MAGS-compatible legacy-index reference produced 274,635 exact
calls, 2,346 committed merges, ratio 0.483725, and 328.689 ms algorithm time.

| Budget | Exact calls | Change | Merges | Cost ratio | Ratio degradation | Algorithm ms |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 118,226 | -57.0% | 1,484 | 0.704672 | +45.7% | 494.555 |
| 4 | 194,093 | -29.3% | 2,049 | 0.582508 | +20.4% | 425.678 |
| 8 | 351,503 | +28.0% | 2,256 | 0.551930 | +14.1% | 456.731 |

No budget satisfies the Stage 3A quality gate of at most 1% per-dataset ratio
degradation. Budget 2 meets the work-reduction gate but loses too many useful
merge opportunities; budget 8 increases exact work while still degrading
compression. Increasing a fixed budget would therefore mask the proposal
quality problem rather than solve it.

EM and DB milestone runs were intentionally not started after FA triggered the
stop condition. The prototype interface and opt-in implementation are retained
for A/B tests, but quotient-neighbor-only is not a recommended algorithm mode.

## Next Step

Proceed to Stage 3B with a residual-signature proposal source. Evaluation must
measure selected-pair/high-gain recall on the same partition checkpoint, not
only all-positive-pair recall. Direct neighbor plus deterministic exploration
should remain a small complementary channel; shared-neighbor expansion should
not be enlarged to recover quality.
