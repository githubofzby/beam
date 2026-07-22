# BEAM-X Stage 4A: Sequence-Preserving Safe Certification

## Decision

Stage 4A is correctness-accepted but does not pass the runtime gate. The
implementation remains opt-in as `--certification safe`; the recommended path
keeps certification off. No candidate, matching, commit, or objective behavior
was changed.

## Safe Bounds

For merge `(A,B)`, `before` is the exact cost of all affected old blocks and
`W=A union B`. Since every external merged-block cost is nonnegative:

```text
ExactGain(A,B) <= before - BlockCost(W,W)
```

During the sorted two-row scan, subtracting each known external
`BlockCost(W,X)` gives a monotonically tighter upper bound. The implementation
aborts only if this bound is non-positive or its ratio is strictly below the
active threshold. Threshold equality is retained. Candidates that survive are
fully scanned and receive the same integer gain as the reference kernel.

The threshold denominator is computed once per prepared representative. An
O(1) persistent cache was tested and rejected: changing a supernode size changes
the capacity and cost of every incident block, so cache maintenance requires
recomputing neighboring incident rows and worsened EM runtime.

## Validation

- The quotient test exhaustively checked the bound and certified decision over
  its deterministic named graphs and 300 small random graphs; runtime was 2.4 s.
- Triangle, path, clique, and two-community off/safe runs produced byte-identical
  `G/P/Cp/Cm`; all safe summaries reconstructed exactly.
- FA off/safe produced byte-identical payloads and reconstructed all 88,234
  original edges.
- Stage 0 and Stage 1 frozen checksums remain valid.

## Work And Runtime

All milestone runs used 24 OpenMP threads, close binding, cores placement,
20 iterations, seed 1, top-k 16, reciprocal threshold, MAGS-compatible cost,
persistent state, CPU scoring, and the legacy candidate index. Runtime values
are lightweight single/paired diagnostics, not final benchmark medians.

| Dataset | Logical exact | Full scans | Full-scan reduction | Entries skipped | Entry reduction | Off ms | Safe ms | Runtime change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FA | 275,736 | 11,884 | 95.7% | 5,408,083 | 19.9% | 514.5 / 413.5 | 484.2 / 449.1 | paired median about +1.4% |
| EM | 1,340,517 | 374,121 | 72.1% | 10,883,981 | 17.4% | 2,100.6 | 2,307.8 | +9.9% |
| DB | 5,264,070 | 427,356 | 91.9% | 27,331,510 | 23.3% | 10,505.2 | 13,652.0 | +30.0% |

Compression ratio, selected merge count, and logical exact-call count were
identical for off/safe on each milestone dataset. FA passes the requested
full-scan work gate, but safe certification does not produce stable algorithm
speedup and DB has a material regression. Stage 4A therefore stops here: no
full Stage 4 promotion and no default-path change. The next research target is
the scheduler/matching path rather than a more complex upper bound.
