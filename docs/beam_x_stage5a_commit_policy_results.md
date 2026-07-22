# BEAM-X Stage 5A: Fixed-Candidate Commit Policy Results

## Decision

Phase A triggers the BEAM scheduler stop condition. Transactional commit exposes
measurable gain decay, but changes final cost by far less than 1% on FA and EM.
Micro-batch sizes 1/2/4/8 and mutual-best priority converge to the same final
partition on FA. Phase B candidate refresh and the full Stage 5 scheduler are
therefore not justified.

## Policies And Correctness

The audit retained the same legacy candidate discovery, exact-scored graph,
threshold, and epoch schedule. Candidate refresh count remains zero because the
legacy CandidateIndex has no refresh pipeline. Every committed pair was
rescored against the latest quotient state and required strictly positive exact
gain. Optional `ExactCost()` checks verified that realized marginal-gain sums
equal actual batch cost reduction.

The clear two-community graph and deterministic noisy SBM passed lossless
reconstruction under G0/S1/T2/T4/T8/M4. On the noisy SBM all policies reached
exact cost 16 and the same partition. Its isolated gain sum was 18 versus 14
realized, but no alternative commit order improved the endpoint-disjoint merge
set.

## FA Audit

FA used the frozen Stage 3 legacy configuration and exact-cost trajectory audit.

| Policy | Final cost | Ratio | Improvement vs G0 | Negative marginal | Validation calls | Algorithm ms | Audit oracle ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| G0 | 42,681 | 0.483725 | reference | 0 | 2,346 | 497.6 | 135.5 |
| S1 | 42,676 | 0.483668 | 0.0117% | 2 | 114,291 | 2,093.9 | 1,643.9 |
| T2 | 42,676 | 0.483668 | 0.0117% | 2 | 58,925 | 1,154.9 | 851.3 |
| T4 | 42,676 | 0.483668 | 0.0117% | 2 | 31,258 | 808.1 | 473.9 |
| T8 | 42,676 | 0.483668 | 0.0117% | 2 | 17,462 | 526.9 | 256.0 |
| M4 | 42,676 | 0.483668 | 0.0117% | 2 | 31,258 | 736.5 | 449.3 |

G0 isolated gains sum to 53,137 while actual partition reduction is 45,553, an
interaction delta of 7,584 (14.27%). Transactional policies detect about 970
decreased gains, but only two become nonpositive. All five alternatives produce
the same partition hash. The large interaction sum therefore mostly changes
gain magnitude, not which endpoint-disjoint merges remain profitable.

## EM Milestone

EM disabled trajectory audit and reports production validation overhead only.

| Policy | Final cost | Ratio | Improvement vs G0 | Negative marginal | Validation calls | Validation ms | Algorithm ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| G0 | 126,992 | 0.690808 | reference | 0 | 22,466 | 4.17 | 1,740.2 |
| T4 | 126,979 | 0.690738 | 0.0102% | 26 | 107,806 | 10.08 | 1,616.8 |
| T8 | 126,979 | 0.690738 | 0.0102% | 26 | 71,814 | 8.65 | 1,649.7 |
| M4 | 126,979 | 0.690738 | 0.0102% | 26 | 107,806 | 11.22 | 2,261.2 |

Absolute runtime is noisy and was not used to claim speedup. Compression changes
are two orders of magnitude below the 1% feasibility gate.

## Conclusion

Stage 5A fails its quality gate:

```text
FA improvement = 0.0117% < 1%
EM improvement = 0.0102% < 1%
```

Negative marginal rates are tiny and mutual-best does not improve the final
partition. BEAM's quality gap is not primarily caused by unvalidated
endpoint-disjoint batch interaction. No deferred-conflict policy, Phase B
candidate refresh, AM/DB benchmark, or full correctness suite is run.

The evidence supports moving the primary research path to MAGS-DM plus the
shared exact CostOracle audit and, if warranted, bounded local repair.
