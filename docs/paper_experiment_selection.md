# 论文实验选择建议

## Level 1：主论文证据

1. **Exact objective correctness（E10/E11）**：用一行或紧凑表说明 500 random oracle tests、每个合法 merge 的 local/global delta equality、7 named graph × 2 merge modes 的 payload/partition-cost equality 和 lossless reconstruction。
2. **Persistent work elimination（E01/E20/E21）**：报告 legacy prepare 的 16.461--19.653 graph-equivalent scans、persistent post-init scans=0、zero-copy entries/bytes=0。runtime 与 work 分列，不能声称同比加速。
3. **Final end-to-end runtime/compression**：当前仓库没有一组同时具备 final config、同机器、明确 commit、raw repetitions 的权威表。历史 PVLDB Intel/P100 表只有在映射回 raw runs/commit 后才能保留；不要用 Ryzen Stage 0--5 拼接替代。
4. **Default proxy-exact necessity**：当前仅有 FA funnel 的有限证据，不足以形成 ranking 表。若论文需要 top-k recall/rank-quality claim，必须执行 `paper_missing_experiments.md` 的最小审计。
5. **Commit design validation（E50/E51）**：科学结论有价值（FA/EM 改善约 0.01%），但 raw/provenance 不完整。恢复或重做轻量 raw audit 后可进主文；否则降到 supplement。

## Level 2：补充材料 / artifact

- Stage 0 全七图 per-round funnel、timing 与 exact row-entry work（E01），以及 instrumentation overhead（E02）。
- Stage 2 persistent/legacy correctness matrix、update modes、300 random graph closing gate（E20）。
- Zero-copy 25 random graph checkpoint rebuild test和全部 paired min/median/max（E21）。
- Specialized exact kernel work counters与 closing runtime，包括 DB 1.120 debt（E23）。
- Safe certification 的 bound proof、300-graph validation、full-scan/entry reductions（E40），同时明确 runtime failed。
- CUDA counters只用于说明 optional scoring backend 的开销分解；它们不是 algorithmic improvement，且只能来自同一历史 Intel/P100 evidence set。

## Level 3：负面消融

| Evidence | 单一科学教训 |
|---|---|
| E22 PreparedRow acquisition | requests/unique=1 且 hits=0；profiling 否决了不必要的 registry/caching layer。 |
| E31 quotient-neighbor budgets | 少 exact calls 不等于好 partition；bounded local proposals 严重破坏 FA compression。 |
| E32/E33 residual signature | 改善 checkpoint top-rank metrics 仍可能因 online sequence和 index overhead而失败。 |
| E40 safe certification | full exact scan reduction可很大，但 row-entry reduction较小且 runtime可退化；逻辑 call count不是充分性能指标。 |
| E23 exact-kernel micro-optimization | constant-factor kernel改进受数据集与系统噪声制约；应完整披露 DB exception。 |
| Historical larger fixed budgets（E62） | 只有在 provenance恢复后保留；lesson 应聚焦 quality/work trade-off，不写工程时间线。 |

## Level 4：仅内部使用

- `results/beam_mags.csv`：同时打印 legacy 与 MAGS-compatible reporting，但 decisions 属于 legacy objective；machine、commit、threads缺失。
- `results/beam_all_*.csv` 与 sensitivity CSV：在无法恢复 command、commit、dataset checksum 前，不与 current Ryzen 数据或 final system 表混用。
- `results/mags_logs.docx`：40-thread MAGS-DM 日志有输入路径和时间，但 machine/commit/dataset checksum未知；不可做严格 speedup/quality gate。
- Stage 4/5 prose-only数值：raw CSV/log不在仓库；可以指导研究叙事，正式数表需 provenance补全。
- `tests/outputs/**` 与 `.pyc`：fixture/派生缓存，不是独立实验。
- `../paper/exsum/main.tex`、`../paper/paper1/main.tex`：不同算法、objective与任务，不属于 BEAM evidence。

## 推荐主线

主论文应形成“exact objective正确 → persistent state消除已测重建工作 → default proxy只作筛选且 exact validation保障决策 → bounded candidate/commit alternatives未改善最终系统”的短闭环。当前第三环只能用 nonpositive retained funnel谨慎表述；若需要 ranking-quality 数字，应补最小 default proxy audit，而不是借用 Stage 3 CandidateIndex recall。

