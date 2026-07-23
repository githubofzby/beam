C. ABSENT:
Existing ranking audits concern experimental CandidateIndex paths and
cannot support a claim about the default EA-proxy.

# Default EA-Proxy / Exact-Gain 审计状态

## 判定依据

默认路径确实计算 EA proxy，但仓库中没有把其逐候选分值、名次与同一候选的 exact gain 联结保存。`src/main.cpp` 的默认值是 `CandidateIndexMode::kLegacy`；`src/sweg.cpp:1804` 定义 `proxy_gain`/`proxy_ratio`，`src/sweg.cpp:1926` 按 ratio、gain 和端点确定性排序，`src/sweg.cpp:1972` 执行 per-source top-k，保留 pair 才进入 exact scoring。Stage 0 的运行命令见 `tools/run_stage0_profiling.sh`，等价于 legacy candidate index（当时尚无显式 CandidateIndex 选项），但输出只有漏斗计数，没有逐 pair proxy 数据。

现有 ranking audit 来自 `tests/candidate_index_recall_test.cpp:246` 和 `tests/candidate_index_recall_test.cpp:248` 显式构造的 `QuotientNeighborCandidateIndex` 与 `ResidualSignatureCandidateIndex`。它在 20 个随机小图、每图 3 个 checkpoint 上，对所有 active representative 的无序合法 pair 调用 `ExactMergeGainPersistent`（`tests/candidate_index_recall_test.cpp:123`），再与 bounded proposal set 比较。其结果见 `docs/beam_x_stage3b_results.csv`，不是 legacy EA proxy 的 divide-group、per-source top-k 审计。

## 七个具体问题

1. **是否使用 `--candidate-index legacy`？** Stage 0 漏斗数据是默认 legacy 路径；但它不是 ranking audit。Stage 3 ranking audit 否，它直接实例化两个实验 CandidateIndex。仓库中没有一组同时满足“legacy + 逐 pair proxy/exact 对照”的结果。
2. **是否记录实际 `proxy_gain` 或 `proxy_ratio`？** 否。两者只存在于 legacy candidate discovery 的临时对象中。结果 CSV、rounds CSV 和日志只记录 `candidate_proxy_pairs_examined`、`candidate_pairs_after_topk` 等聚合计数。
3. **exact gain 的覆盖域是什么？** Default legacy 运行只 exact-score top-k 后去重的 retained pairs。Stage 3 checkpoint test 则离线 exact-score checkpoint 上全部 active legal pairs。仓库没有对全部 bucket-induced pairs、divide-group 全 pair 或 default-proxy sampled reference set 的逐 pair输出。
4. **“best-pair recall”的 reference scope 是什么？** Stage 3 的 best pair 是 checkpoint 上全部 active legal unordered pairs中的 exact-best positive pair，不是 legacy divide group，也不是 bucket-induced 或 bounded reference set。见 `tests/candidate_index_recall_test.cpp:123`、`:154`、`:168`。
5. **是否覆盖 early/middle/late epoch？** 否。Stage 3 使用三个合成 checkpoint，并在 checkpoint 间固定合并 `active[0], active[1]`；没有真实大图 early/middle/late epoch 分层。Stage 0 有 20 轮漏斗计数，但没有 proxy/exact pair 字段。
6. **gain 是否使用 MAGS-compatible CostOracle？** Stage 3 test 调用 persistent `ExactMergeGainPersistent`，设计文档将其定义为统一 MAGS-compatible integer objective；但测试不是通过带完整 CLI provenance 的 production run 生成。Stage 0 frozen profile 的最终报告含 `cost_ratio_mags_compatible`，其 candidate gain 仍属于当时 legacy objective，不能据此称为 MAGS-compatible proxy/exact audit。
7. **可支持哪些论文说法？** 见下表。

## 潜在论文说法

| 说法 | 支持状态 | 证据与限制 |
|---|---|---|
| proxy ranking is imperfect | PARTIALLY_SUPPORTED | Stage 0 retained 289,013 个 FA pair 中只有 99,331 个 exact gain 为正，说明 proxy-positive/top-k retained set 含大量无益 pair；但没有 rank correlation、regret 或逐 pair proxy 值。 |
| proxy-selected candidates can be nonpositive | PARTIALLY_SUPPORTED | Stage 0 定义保证 legacy retained pairs 才 exact-score，且 `exact_gain_positive_count < exact_gain_calls`；可说明 retained candidate 中存在 exact-nonpositive pair，不能给出分布或按 epoch 比例之外的排名解释。 |
| exact rescoring changes candidate order | NOT_SUPPORTED | 实现会对 exact result 排序/过滤，但没有保存 proxy order 与 exact order 的经验对照。实现事实不能替代审计。 |
| top-k screening retains the best positive candidate frequently | NOT_SUPPORTED | 现有 best-pair recall 属于实验 CandidateIndex，并且 scope 是所有 active pair；不能迁移到 default EA proxy。 |
| exact rescoring is necessary for final merge decisions | PARTIALLY_SUPPORTED | 实现语义和大量 retained-exact-nonpositive pair支持“exact validation 防止直接提交明显无益候选”；没有 proxy-only counterfactual final partition，故“经验上必要”仍过强。 |

## 当前可安全使用的措辞

可写：“在 frozen Facebook profile 中，默认筛选路径提交 289,013 个候选做 exact evaluation，其中 99,331 个具有正 exact gain；这说明 proxy/top-k screening 不能替代 exact validation。”不可写：“EA proxy 以某个 recall 保留 exact-best pair”“proxy 与 exact rank 的相关系数为某值”或将 Stage 3 的 95% best-pair recall 归给默认 proxy。
