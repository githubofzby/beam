# BEAM-X Stage 0 Baseline Freeze

Status: Stage 0 baseline and first profiling pass complete. The legacy
implementation is unchanged algorithmically; the executable was built from the
current working tree because it already contains the user-provided
MAGS-compatible reporting changes.

## Reproducibility

| Item | Value |
|---|---|
| Git commit | `62365caabba8e59a4f0e885ce7f57d72dae0bca1` |
| Working tree | Dirty: user changes in `CMakeLists.txt`, `README.md`, `src/main.cpp`, `运行方式.md`; user-added `include/compression_metrics.hpp` |
| Build type | Manual release-equivalent |
| Compiler | GCC 16.1.1 (20260625) |
| Compile flags | `-std=c++17 -O3 -DNDEBUG -fopenmp -Iinclude` |
| CUDA | Disabled; `src/cuda_scoring_stub.cpp` |
| CPU | AMD Ryzen 7 8840HX with Radeon Graphics; 12 cores / 24 logical CPUs |
| OMP_NUM_THREADS | 24 |
| OMP_PROC_BIND | `close` |
| OMP_PLACES | `cores` |
| Random seed | 1 |
| Input representation | Undirected edge list; internally stored as two CSR arcs per edge |

The baseline executable is `build_stage0_baseline/beam`. Its source-level
configuration is:

```text
--merge-mode batch-ea-blocked
--scoring-backend cpu
--top-k 16
--group-batch-size 64
--ea-use-threshold
--threshold-policy reciprocal
--iterations 20
--seed 1
--error-bound 0.0
```

The raw complete CSV is `docs/beam_x_stage0_baseline.csv`. Per-dataset stdout
logs are stored under `docs/stage0_logs/`. The external `/usr/bin/time` tool is
not available in this environment, so only BEAM's own `runtime_algorithm_ms` is
frozen here.

## Frozen Results

`runtime_algorithm_ms` includes `Run + Encode` and excludes graph loading and
output writing. `cost_ratio_mags_compatible` is the official comparison metric.

| Dataset | Alias | n | Undirected edges | runtime_algorithm_ms | cost_ratio_mags_compatible | Supernodes | P | Cp | Cm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Facebook | FA | 4,039 | 88,234 | 275.951 | 0.491347 | 1,872 | 13,478 | 22,681 | 7,662 |
| Email-Enron | EM | 36,692 | 183,831 | 1,783.740 | 0.695666 | 15,957 | 61,825 | 65,646 | 3,144 |
| Amazon | AM | 334,863 | 925,872 | 9,385.633 | 0.598288 | 179,322 | 268,658 | 264,818 | 39,286 |
| DBLP | DB | 317,080 | 1,049,866 | 8,815.505 | 0.542040 | 155,835 | 348,921 | 229,747 | 21,153 |
| YouTube | YO | 1,134,890 | 2,987,624 | 58,429.643 | 0.732916 | 519,883 | 1,078,907 | 1,101,708 | 16,065 |
| RoadNet | RN | 1,965,206 | 2,766,607 | 95,987.109 | 0.707981 | 1,278,139 | 698,766 | 1,230,219 | 50,607 |
| LiveJournal | LJ | 3,997,962 | 34,681,189 | 266,674.080 | 0.749757 | 2,348,740 | 12,388,291 | 13,093,280 | 720,646 |

## Instrumentation Design

Stage 0 adds a legacy-only `ProfilingMode` without changing merge decisions:

- `off`: no profile containers, counter updates, or profiling output.
- `summary`: one global accumulated profile is printed after the run.
- `rounds`: the same summary plus one long-form row per iteration written after
  the algorithm completes.

The long-form profile CSV begins with stable identity columns:

```text
dataset,iteration,algorithm,profiling_mode,
active_supernodes,group_count,sum_group_sizes
```

It then contains the decision-funnel counters, work counters, and exclusive
stage timings. It is deliberately separate from `--results-csv`, whose schema
is an existing result contract.

Current counter definitions:

| Counter | Definition |
|---|---|
| `candidate_proxy_pairs_examined` | Candidate pairs actually examined in proxy bucket enumeration |
| `candidate_pairs_after_topk` | Deduplicated pairs surviving fixed per-node top-k |
| `candidate_pairs_submitted_for_scoring` | Pairs handed to CPU/CUDA exact scoring |
| `candidate_pairs_scored` | Pairs for which an exact result is returned |
| `exact_gain_calls` | Exact evaluations, currently equal to scored pairs |
| `exact_gain_positive_count` | Exact gains strictly greater than zero |
| `above_threshold_count` | Positive exact gains retained after optional threshold filtering |
| `matching_selected_count` | Greedy matching selections |
| `actual_merge_count` | Successful `Update_S` calls |
| `prepare_original_edges_scanned` | Original CSR arcs read while preparing rows; a full graph scan is about `2m` |
| `prepare_aggregated_nnz` | Sum of materialized supernode-level sparse row lengths |
| `exact_gain_input_nnz` | Sum of `|agg[A]| + |agg[B]|` over scored candidate pairs |
| `update_partition_nodes_touched` | Nodes traversed/relabelled by `Update_S`; this is not quotient work |

For the current CPU path, the expected funnel invariant is:

```text
actual_merge_count <= matching_selected_count <= above_threshold_count
<= exact_gain_positive_count <= exact_gain_calls
```

`candidate_pairs_after_topk == candidate_pairs_submitted_for_scoring ==
candidate_pairs_scored == exact_gain_calls` currently holds for CPU legacy
scoring. The fields remain separate because CUDA slicing and future filtering
may legitimately distinguish them.

Timing convention: `prepare_ms` is the wall time of the prepare stage;
`candidate_discovery_task_sum_ms` is a parallel task-time sum and can exceed
wall time. They must not be summed to estimate algorithm time. CSV is written
only after the run, not from a hot loop.

## Initial Correctness Gate

With `OMP_NUM_THREADS=1`, profiling `off` and `rounds` produced byte-identical
`G.txt`, `P.txt`, `Cp.txt`, and `Cm.txt` on triangle, path, clique, and two-block
graphs. The reconstruction checker reported zero missing and extra edges for
each. The final MAGS-compatible ratio and summary counts were identical.

Still required before Stage 0 acceptance: overhead medians and profile runs for
all frozen datasets. No performance conclusion is drawn from this baseline
table yet.

## Facebook Instrumentation Smoke Test

The Stage 0 CPU profile path was run once on Facebook with the frozen
configuration and `--profiling rounds`. This is a semantic smoke test, not an
overhead measurement and not a conclusion about the global bottleneck order.

| Metric | Value |
|---|---:|
| `candidate_proxy_pairs_examined` | 16,679,445 |
| `candidate_pairs_after_topk` | 289,013 |
| `candidate_pairs_submitted_for_scoring` | 289,013 |
| `candidate_pairs_scored` | 289,013 |
| `exact_gain_positive_count` | 99,331 |
| `above_threshold_count` | 11,713 |
| `matching_selected_count` | 2,167 |
| `actual_merge_count` | 2,167 |
| `prepare_original_edges_scanned` | 3,468,052 |
| `prepare_aggregated_nnz` | 1,619,002 |
| `exact_gain_input_nnz` | 28,505,384 |
| `update_partition_nodes_touched` | 5,439 |
| profile CSV rows | 20 iterations + header |

The funnel invariant passed for every iteration. The profiled final compression
ratio was `0.491347`, identical to the frozen unprofiled baseline. The raw
smoke CSV and stdout are intentionally temporary (`/tmp`) and are not treated
as the official seven-dataset profiling table.

## Remaining Stage 0 Runs

The small-graph gate has since been extended and passed for triangle, path,
clique, two blocks, star, two communities, and a fixed seed-7 random graph.
For every graph, `off` and `rounds` had identical final `G/P/Cp/Cm` files and
the profile CSV satisfied the funnel invariant on every iteration. Empty input
is intentionally a negative parser test: the current loader rejects it with
`Invalid node count`, so no empty-graph compression result is claimed.

Run the following on the host shell to finish the long-running acceptance
experiments without an interactive execution timeout:

```bash
export OMP_NUM_THREADS=24
export OMP_PROC_BIND=close
export OMP_PLACES=cores
bash tools/run_stage0_overhead.sh build_stage0_baseline/beam_profile
bash tools/run_stage0_profiling.sh build_stage0_baseline/beam_profile
```

The overhead runner alternates `off -> rounds` three times for FA, EM, AM, and
DB, writing each completed result immediately to
`docs/beam_x_stage0_overhead.csv`. The profiling runner writes the 20-row-per-
dataset long-form CSV to `docs/beam_x_stage0_profile_rounds.csv` and a compact
per-dataset summary to `docs/beam_x_stage0_profile_summary.csv`. Both scripts
use the frozen parameters above.

Do not treat the partial interactive overhead samples as a pass/fail result:
this environment kills foreground child processes around 25 seconds, which
invalidates the AM/DB repetitions. The runners are the authoritative path for
the required median calculation and seven-dataset bottleneck table.

## Stage 0 Results

The authoritative runner completed successfully at 2026-07-21 20:31. The raw
per-iteration measurements are in `docs/beam_x_stage0_profile_rounds.csv`; the
per-dataset runtime summary is in `docs/beam_x_stage0_profile_summary.csv`.
Every dataset has exactly 20 profile rows and every row satisfies:

```text
actual_merge_count <= matching_selected_count <= above_threshold_count
<= exact_gain_positive_count <= exact_gain_calls
```

The profile-run final `cost_ratio_mags_compatible`, supernode count, and
`P/Cp/Cm` counts exactly match the frozen baseline for all seven datasets.

### Instrumentation Overhead

The overhead runner completed three paired `off -> rounds` repetitions on FA,
EM, AM, and DB. The table reports the median of paired percentage changes in
`runtime_algorithm_ms`; negative values are ordinary run-to-run noise, not a
claim that instrumentation improves the algorithm.

| Dataset | Summary median overhead | Rounds median overhead |
|---|---:|---:|
| FA | +0.31% | -0.64% |
| EM | -4.07% | -5.66% |
| AM | +2.78% | -0.03% |
| DB | +0.97% | +2.29% |

All medians are below the 5% acceptance limit. Negative values are measurement
noise, not a speedup claim. The raw 36 runs are retained in
`docs/beam_x_stage0_overhead.csv`.

### Per-Dataset Work Profile

`prepare_ms` is prepare-stage wall time. `proxy_task_sum_ms` is a parallel task
sum and must not be added to wall times. `exact/merge` and `exact-work/(2m)`
are the requested work-efficiency diagnostics.

| Dataset | Algorithm ms | Divide ms | Prepare ms | Exact ms | Exact calls | Merges | exact/merge | prepare scans | exact-work/(2m) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FA | 272.583 | 17.348 | 143.321 | 83.610 | 289,013 | 2,167 | 133.37 | 19.65 | 161.53 |
| EM | 1,750.264 | 113.272 | 940.828 | 542.169 | 1,424,396 | 20,735 | 68.70 | 18.44 | 174.52 |
| AM | 9,973.209 | 1,475.156 | 1,355.970 | 5,714.762 | 5,996,392 | 155,541 | 38.55 | 17.82 | 41.88 |
| DB | 9,537.965 | 1,229.404 | 1,536.493 | 5,413.078 | 6,095,695 | 161,245 | 37.80 | 17.99 | 63.16 |
| YO | 50,065.350 | 7,796.552 | 23,185.306 | 14,908.452 | 31,367,993 | 615,007 | 51.00 | 17.25 | 127.83 |
| RN | 75,561.054 | 25,492.988 | 4,205.451 | 36,781.014 | 17,636,358 | 687,067 | 25.67 | 16.46 | 20.33 |
| LJ | 236,713.884 | 69,212.188 | 69,250.602 | 76,557.513 | 159,916,415 | 1,649,222 | 96.96 | 19.13 | 179.44 |

### Evidence-Based Diagnosis

1. **Repeated prepare is real and material.** Every dataset reads the original
   graph roughly 16.5--19.7 equivalent times during prepare alone. This directly
   validates the first BEAM-X hypothesis: a persistent quotient graph can
   eliminate a large source of repeated original-CSR work.
2. **Candidate certification is a primary next target.** Exact calls per
   committed merge range from 25.67 (RN) to 133.37 (FA), far above the target
   of 5. On FA only 4.05% of exact calls survive thresholding; on LJ it is
   2.61%. This validates the need for progressive certification and safe
   pruning, but does not yet identify a correct upper bound.
3. **Single-call exact work is also high on several graphs.** Exact row work
   is 161.53 graph equivalents on FA, 174.52 on EM, and 179.44 on LJ. This is
   evidence for compact persistent rows and later early abort; it is not an
   argument for a CUDA rewrite before reducing candidate work.
4. **Divide remains substantial at scale.** RN and LJ spend about 25.5 s and
   69.2 s respectively in divide. The persistent quotient graph will not by
   itself remove this cost; the CandidateIndex design must avoid replacing it
   with another full original-neighborhood scan.
5. **Partition commit is not the current leading bottleneck.** Measured update
   time is below 0.25 s even on LJ, although `update_partition_nodes_touched`
   should remain tracked when larger merge trees are introduced.

The Stage 0 core acceptance criteria are met: instrumentation perturbation is
below 5% by the prescribed median test, profile data explains repeated prepare,
candidate-to-merge filtering, exact-call amplification, and update cost, and
the instrumentation leaves lossless output and formal MAGS-compatible metrics
unchanged. Before entering CostOracle, the remaining Stage 0 implementation
cleanup is to expose zero-valued legacy-only early-abort/batch-validation and
quotient-update counters plus split aggregate timing; this does not change the
diagnosis above.
