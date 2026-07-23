# 论文可用实验表（基于现存兼容证据）

这些表不把历史 Intel/P100、当前 Ryzen、legacy objective 与 MAGS-compatible objective 的数值合并。`UNKNOWN` provenance 的结果只作为设计/消融证据候选，正式投稿前应补 raw artifact 和 commit。

## Table A：正确性与目标一致性

**建议 caption：** “Exact integer objective and lossless-summary validation. All tests are deterministic correctness checks rather than performance measurements.”

| 检查 | 覆盖 | 结果 | evidence |
|---|---|---:|---|
| `ExactMergeGain = Cost(before)-Cost(after)` | 500 fixed-seed random graphs + named graph families；每个合法 merge | PASS | E10_COST_ORACLE_UNIT |
| 64-bit objective/capacity overflow semantics | standalone CostOracle validation cases | PASS | E10_COST_ORACLE_UNIT |
| payload cost = `ExactPartitionCost` = reported MAGS-compatible cost | 7 named small graphs × 2 CPU merge modes | PASS | E11_COST_ORACLE_INTEGRATION |
| lossless reconstruction | 同上 | PASS | E11_COST_ORACLE_INTEGRATION |
| persistent gain/update vs full rebuild | 300 random graphs、多 merge 顺序与 update modes（Stage 2 closing report） | PASS | E20_PERSISTENT_INITIAL |
| persistent vs legacy partition and payload | 7 named graphs；Stage 2 frozen gate | byte-identical | E20_PERSISTENT_INITIAL |
| Stage 0/1 frozen checksum | Stage 2/4 closing checks | PASS | E11_COST_ORACLE_INTEGRATION / E40_SAFE_CERTIFICATION |

**解释。** 独立 oracle 从全局 partition cost 定义验证局部整数 gain；integration gate 再把数学目标、最终 payload 和 lossless reconstruction 连起来。这组证据适合支撑 exact objective 的正确性，不依赖大图 runtime。

**可支持：** local gain 与 global cost delta 一致；MAGS-compatible payload count 与 ExactPartitionCost 一致；生成结果 lossless。

**不可支持：** 大图上的统计错误率为零、所有 CUDA objective 语义相同，或 Stage 1B reference path 的性能。

## Table B：Persistent state 的工作消除

**建议 caption：** “Eliminated reconstruction and row-copy work under persistent quotient state; runtime ratios are reported separately from deterministic work counters.”

### B1. Frozen legacy profile（legacy objective，commit `62365ca...`）

| Dataset | Prepare scans (`2m` equivalents) |
|---|---:|
| Facebook | 19.653 |
| Email-Enron | 18.437 |
| Amazon | 17.815 |
| DBLP | 17.992 |
| YouTube | 17.250 |
| RoadNet | 16.461 |
| LiveJournal | 19.131 |

### B2. Persistent MAGS-compatible work profile（Stage 2 intermediate）

| Dataset | Post-init original scans |
|---|---:|
| Facebook | 0 |
| Email-Enron | 0 |
| Amazon | 0 |
| DBLP | 0 |
| YouTube | 0 |
| RoadNet | 0 |
| LiveJournal | 0 |

### B3. Zero-copy paired milestone（同一 Task 1 paired window）

| Dataset | Copied entries / bytes | Persistent/legacy paired runtime ratio |
|---|---:|---:|
| Facebook | 0 / 0 | 0.992615 |
| Email-Enron | 0 / 0 | 0.980934 |
| Amazon | 0 / 0 | 1.030848 |
| DBLP | 0 / 0 | 1.025223 |

来源：E01_STAGE0_PROFILE、E20_PERSISTENT_INITIAL、E21_ZERO_COPY。三个 panel 是不同 evidence configurations，不做逐行数值合并。runtime ratio 是 Ryzen 上 3 个相邻 paired comparisons 的 median；工作列不是 runtime 推断。`copied entries/bytes=0` 由 Task 1 persistent profiling/smoke 报告确认。

**解释。** Persistent quotient state 把 legacy 每次 prepare 的 16.461--19.653 次等价原图扫描降为初始化后零扫描；zero-copy view 又消除 persistent prepare 的 row materialization。前四图 paired runtime 接近 parity，但这不是“工作量按比例转化为加速”的证据。

**可支持：** repeated original-CSR reconstruction 和 prepared-row copy 被消除；FA/EM/AM/DB 当时实现未出现大幅 runtime regression。

**不可支持：** 七图 final-system speedup、persistent 一定更快、或跨机器 geometric mean。

## Table C：Candidate ranking evidence（必须分栏）

### C-A. Default EA-proxy audit

**建议 caption：** “Status of empirical evidence for the default EA proxy.”

| Default path observation | Value | Reference scope |
|---|---:|---|
| FA retained candidates exact-scored | 289,013 | legacy per-source top-k 后去重 pair |
| FA retained candidates with exact gain > 0 | 99,331 | 同一 retained set |
| Per-pair proxy score/rank retained | No | — |
| Exact-best recall / rank correlation / regret | Not measured | — |

来源：E01_STAGE0_PROFILE。**解释。** 该 funnel 说明 proxy/top-k retained set 中存在大量 exact-nonpositive pair，但不是完整 proxy-versus-exact ranking audit。

**可支持：** proxy screening 不能替代 exact validation。**不可支持：** default proxy 的 top-k recall、Spearman correlation、top1 hit 或 regret。

### C-B. Quotient-neighbor audit（实验 CandidateIndex）

**建议 caption：** “Offline ranking audit of the experimental quotient-neighbor CandidateIndex over all active legal pairs at 60 small-graph checkpoints.”

| Budget | Positive recall | Gain-weighted recall | Best-pair recall | Top-1 cheap/exact hit | Top-4 exact recall | p95 regret |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.97392 | 0.960124 | 0.950000 | 0.150000 | 0.982759 | 0.125000 |

来源：E30_QN_CHECKPOINT_AUDIT。**解释。** broad recall 很高，但 cheap-score top1 exact hit 只有 15%；这说明覆盖率与 top-rank fidelity不同。

**警告：** 不是 default EA-proxy，不是 production divide group，不代表大图 early/middle/late epoch。

### C-C. Residual-signature audit（实验 CandidateIndex）

**建议 caption：** “Offline ranking audit of the experimental residual-signature CandidateIndex under the same checkpoint reference scope.”

| Budget | Positive recall | Gain-weighted recall | Best-pair recall | Top-1 cheap/exact hit | Top-4 exact recall | p95 regret |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.681337 | 0.697557 | 1.000000 | 0.516667 | 0.978448 | 0.000000 |

来源：E32_RESIDUAL_CHECKPOINT_AUDIT。**解释。** residual source牺牲 broad recall 并改善 top rank，但 online FA 结果仍变差。

**警告：** 不得把 100% best-pair recall 归给 default EA proxy，也不能从 60 个合成 checkpoint 推断 final compression。

## Table D：Online candidate-path outcomes

**建议 caption：** “Reducing exact candidate evaluations does not necessarily preserve the final partition quality.”

| FA path | Budget | Exact calls | Merges | Exact cost | MAGS-compatible ratio | Algorithm ms |
|---|---:|---:|---:|---:|---:|---:|
| legacy CandidateIndex | 16 | 274,635 | 2,346 | 42,681 | 0.483725 | 328.688803 |
| quotient-neighbor | 2 | 118,226 | 1,484 | — | 0.704672 | 494.554892 |
| quotient-neighbor | 4 | 194,093 | 2,049 | — | 0.582508 | 425.677965 |
| quotient-neighbor | 8 | 351,503 | 2,256 | — | 0.551930 | 456.730899 |
| residual-signature | 4 | 158,451 | 2,317 | 45,054 | 0.510619 | 549.619734 |

来源：E31_QN_ONLINE_FA、E33_RESIDUAL_ONLINE_FA、E34_SEQUENCE_DIVERGENCE。**解释。** QN budget 2 和 residual budget 4 分别显著降低 exact calls，却恶化 final compression；residual 还因 index refresh/proposal overhead 变慢。这直接说明 exact-call reduction 不是充分目标。

**可支持：** C6/C7 的 bounded negative ablation。**不可支持：** CandidateIndex 在所有数据集都失败、default proxy 最优、或跨 run runtime significance；这些是单次 FA funnel。

## Table E：Commit-policy 设计验证（provisional）

**建议 caption：** “Fixed-candidate commit-policy audit. Transactional validation exposes gain decay but changes final cost by only about 0.01%.”

| Dataset | Policy | Final cost | Improvement vs G0 | Selected/accepted merges | Nonpositive rejects | Interaction delta ratio | Gain decay ratio | Validation calls | Validation ms | Partition relation |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| FA | G0 | 42,681 | ref. | 2,346 / 2,346 | 0 | 14.27% | — | 2,346 | — | differs from alternatives |
| FA | S1 | 42,676 | 0.0117% | not retained | 2 | — | not retained | 114,291 | — | same as T2/T4/T8/M4 |
| FA | T4 | 42,676 | 0.0117% | not retained | 2 | — | not retained | 31,258 | — | same as S1/T2/T8/M4 |
| FA | M4 | 42,676 | 0.0117% | not retained | 2 | — | not retained | 31,258 | — | same as S1/T2/T4/T8 |
| EM | G0 | 126,992 | ref. | not retained | 0 | — | — | 22,466 | 4.17 | not reported |
| EM | T4 | 126,979 | 0.0102% | not retained | 26 | — | not retained | 107,806 | 10.08 | not reported |
| EM | M4 | 126,979 | 0.0102% | not retained | 26 | — | not retained | 107,806 | 11.22 | not reported |

来源：E50_COMMIT_POLICY_FA、E51_COMMIT_POLICY_EM。缺失值保持为空，不能从 final merges 或 prose推断。

**解释。** FA 的 isolated gain 与 actual reduction 相差 14.27%，但只有两个 marginal gain 变为非正，alternative policies 最终只改善 5 cost items；EM 只改善 13。interaction 可测并不代表 scheduler 是主要 quality bottleneck。

**可支持：** 在这两个数据集/配置上，复杂 current-state validation 没有 material final-cost benefit；FA alternatives 彼此 partition hash相同。

**不可支持：** G0 与 alternatives partition 相同、EM policies partition相同、所有 selected/accepted merge counts相同、精确 gain-decay ratio，或 runtime speedup。原始 Stage 5 CSV/log 缺失，正式主文使用前应补 provenance。
