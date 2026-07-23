# BEAM-X Stage 0 Profiling Report

This report freezes the evidence used to choose the next BEAM-X stages. It is
based on the legacy `batch-ea-blocked` CPU path, 20 iterations, top-k 16,
reciprocal threshold, seed 1, and 24 pinned OpenMP threads. Full environment and
build metadata are recorded in `beam_x_baseline.md`.

## Acceptance

- Seven datasets produced exactly 20 long-form rows each.
- Every row satisfies `actual <= matching <= threshold <= positive <= exact`.
- Profiling does not change the final partition counts, `P/Cp/Cm`, or
  `cost_ratio_mags_compatible`.
- Triangle, path, clique, two-block, star, two-community, and seeded-random
  graphs have byte-identical off/on output and lossless reconstruction.
- Empty input is a parser negative test and is rejected as expected.
- Paired median overhead for both summary and rounds modes is below 5% on FA,
  EM, AM, and DB.

## Work And Time

`prepare scans = prepare_original_edges_scanned / (2m)` and
`exact work = exact_gain_input_nnz / (2m)`.

| Dataset | Prepare scans | Exact/merge | Exact work | Prepare ms | Exact ms | Update ms |
|---|---:|---:|---:|---:|---:|---:|
| FA | 19.65 | 133.37 | 161.53 | 143.32 | 83.61 | 0.32 |
| EM | 18.44 | 68.70 | 174.52 | 940.83 | 542.17 | 3.03 |
| AM | 17.82 | 38.55 | 41.88 | 1,355.97 | 5,714.76 | 23.68 |
| DB | 17.99 | 37.80 | 63.16 | 1,536.49 | 5,413.08 | 23.33 |
| YO | 17.25 | 51.00 | 127.83 | 23,185.31 | 14,908.45 | 93.17 |
| RN | 16.46 | 25.67 | 20.33 | 4,205.45 | 36,781.01 | 95.50 |
| LJ | 19.13 | 96.96 | 179.44 | 69,250.60 | 76,557.51 | 242.37 |

## Stage Mapping

| Evidence | Conclusion | Required stage |
|---|---|---|
| Prepare rereads 16.46--19.65 full graphs | Supernode adjacency is rebuilt nearly every iteration | Stage 2 persistent QuotientGraph |
| Exact/merge is 25.67--133.37 | Fixed top-k exact-scores too many rejected pairs | Stage 4 progressive certification and safe pruning |
| Exact work reaches 179.44 full graphs | Several datasets also have expensive row unions | Stage 2 compact persistent rows, then Stage 4 early abort |
| Threshold survival is 2.61% on LJ and 4.05% on FA | Candidate quality/pruning, not just kernel speed, is limiting | Stages 3--4 CandidateIndex and certification |
| Update time is at most 242 ms | Partition commit is not the current primary bottleneck | Keep counter; do not prioritize partition rewrite |
| Divide costs 69.2 s on LJ and 25.5 s on RN | Original-neighborhood grouping is itself material at scale | Stage 3 affected-frontier CandidateIndex |

The evidence supports the original priority order. Stage 1 must first provide a
single exact mathematical objective; Stage 2 then removes repeated preparation;
Stage 4 reduces exact-call amplification. CUDA work remains deferred.

## Frozen Artifacts

- `beam_x_baseline.md`: build, hardware, parameters, frozen compression baseline.
- `beam_x_stage0_profile.md`: this evidence and stage mapping.
- `beam_x_stage0_overhead.csv`: raw 36-run overhead experiment.
- `beam_x_stage0_profile_rounds.csv`: raw 7 x 20 long-form profiling data.
- `beam_x_stage0_summary.csv`: machine-readable derived metrics.

These artifacts are Stage 0 references and must not be edited during later
algorithm stages. Corrections require a new versioned freeze rather than an
in-place rewrite.
