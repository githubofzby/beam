# BEAM-X Stage 3B: Residual Signatures and Ranking Audit

## Outcome

The residual-signature prototype reduces FA exact calls to 57.7% of the legacy
candidate path, but fails the hard compression and runtime gates. Stage 3B
therefore stops before EM/DB. The source is retained only as an experimental
ablation; legacy candidate generation remains the recommended path for the
subsequent sequence-preserving certification experiment.

Legacy candidate generation remains the default. Both Stage 3A and Stage 3B
paths remain opt-in diagnostic modes.

## Ranking Audit

The same-partition audit used 20 fixed-seed random graphs and 60 checkpoints.
It exhaustively scored legal pairs at each checkpoint and compared the bounded
proposal set with the exact ranking.

| Metric | Quotient-neighbor budget 8 | Residual budget 4 |
|---|---:|---:|
| Ordinary positive-pair recall | 97.39% | 68.13% |
| Gain-weighted recall | 96.01% | 69.76% |
| Best-pair recall | 95.00% | 100.00% |
| Top-1 cheap-score exact hit | 15.00% | 51.67% |
| Top-4 exact recall | 98.28% | 97.84% |
| Median best-gain regret | 0.000 | 0.000 |
| p95 best-gain regret | 0.125 | 0.000 |

Residual ranking intentionally sacrifices ordinary recall while improving the
top of the exact-gain ranking. This confirms that ordinary positive-pair recall
was a misleading Stage 3A metric.

`legacy_commit_pair_covered` was not fabricated from the exhaustive global-best
pair. A valid value requires shadow-running the legacy proposal, exact filter,
and existing matching on the same online partition. Because FA crossed the
immediate stop line, the prototype was stopped rather than adding a second
online matching pipeline solely to complete this diagnostic.

## Residual Signature

For each bounded top feature `(A,X)`:

```text
residual(A,X) = 2 * edges(A,X) - size(A) * size(X)
```

The cached feature stores neighbor id, sign, absolute magnitude, normalized
magnitude, and a magnitude bin. Each signature also stores size and quotient
degree bins. Only the strongest eight normalized features are retained; no
dense vector or original CSR scan is used.

The deterministic inverted key is:

```text
(neighbor representative, residual sign, magnitude bin)
```

Each bucket is sorted by normalized magnitude and representative id. Proposal
fanout is capped at eight entries per source feature. Large buckets never emit
all pairs.

## Ranking and Budget

Residual cheap score combines:

```text
+ 4 * same-sign residual alignment
- 3 * opposite-sign residual conflict
+ size-bin compatibility
+ direct quotient-edge evidence
```

It is only a ranking score and does not duplicate or approximate CostOracle.

Budget 4 is fixed to:

```text
2 residual-signature
1 direct quotient neighbor
1 deterministic exploration
```

Budget 8 additionally permits four residual, two direct, one shared, and one
exploration proposal, but was not used to repair the failed budget-4 result.

## Cache Lifetime

Each signature records the quotient representative version. At every blocked
commit epoch boundary, the index rebuilds the current active representative
list, reuses version-matching signatures, refreshes only changed rows, and then
rebuilds bounded inverted buckets from cached features. Proposal workers only
read signatures and buckets. No view or signature is used across a mismatched
version.

On the two-community smoke graph the final implementation reported nine row
scans, nine misses, nine hits, 18 created features, 22 accumulated buckets, and
a maximum bucket size of three.

## FA Gate

| Metric | Legacy candidate path | Residual budget 4 | Change/gate |
|---|---:|---:|---:|
| Exact calls | 274,635 | 158,451 | -42.3%, pass |
| Committed merges | 2,346 | 2,317 | -1.2% |
| MAGS-compatible ratio | 0.483725 | 0.510619 | +5.56%, fail |
| Algorithm time | 328.689 ms | 549.620 ms | +67.2%, fail |
| Index refresh time | 0 | 222.416 ms | diagnostic |
| Proposal task-sum | 0 | 469.247 ms | diagnostic |

The ratio exceeds the 3% immediate-stop condition, so no second score-adjustment
round, budget increase, EM run, or DB run was performed. Index refresh and
proposal work also offset the exact-work reduction.

## Sequence Divergence

The first submitted partition divergence occurs in epoch 1. At that point:

| Path | Exact partition cost | Cumulative reported local gain | Merges |
|---|---:|---:|---:|
| Legacy | 88,179 | 56 | 17 |
| Residual | 88,167 | 68 | 15 |

Residual is initially better by exact partition cost, but finishes worse:

| Path | Final exact cost | Final ratio | Reported cumulative local gain |
|---|---:|---:|---:|
| Legacy | 42,681 | 0.483725 | 53,137 |
| Residual | 45,054 | 0.510619 | 54,594 |

Thus the first divergent choices are not simply lower-gain merges. Endpoint
competition changes the later reachable merge sequence. The mismatch between
reported local-gain sum and final partition-cost ordering is also consistent
with batch interaction, but Stage 3B does not modify matching, scheduler, or
batch semantics.

## Decision

The proposal-source approach has now failed twice under bounded budgets:
quotient-neighbor misses sequence quality, while residual signatures improve
top-rank checkpoint metrics but still lose online compression and runtime.
Further residual buckets, larger fixed budgets, ANN, original-edge rescans, or
GPU candidate generation are not justified.

The evidence points to endpoint competition and sequence-aware
matching/scheduling rather than proposal coverage alone. Stage 4A may only use
the legacy candidate path and provably safe certification; residual proposals
must not be combined with it.
