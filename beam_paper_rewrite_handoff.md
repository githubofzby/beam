# BEAM 论文改写交接包

## 0. 用途

本文件用于把 BEAM 项目的算法、工程改造、实验审计结论和论文表述边界，交接给另一个负责改写论文的 ChatGPT 对话。

该对话还应同时读取：

1. 论文项目中的当前论文源文件或 PDF；
2. 论文项目中的 `experiments.md`；
3. 本交接文件；
4. 必要时再读取下面列出的少量 BEAM `docs/` 文件。

不要把 BEAM 仓库中所有阶段文档一次性塞给论文改写对话。大量历史数据来自不同机器、commit、cost objective 和算法配置，容易造成数值混用。

---

# 1. Source of Truth Hierarchy

论文改写时，严格按以下优先级使用信息。

## Level A：最终实验数值

**论文项目中的 `experiments.md` 是最终实验数字的唯一权威来源。**

所有 runtime、compression ratio、speedup、逐数据集结果、平均值和表格数字，都必须优先从 `experiments.md` 获取。

若 `experiments.md` 与 BEAM 历史 docs 不一致：

- 使用 `experiments.md`；
- 不对不同环境的数据做平均或拼接；
- 在无法确认来源时停止写数字，而不是猜测。

## Level B：当前论文内容

当前论文的 LaTeX/PDF 是以下内容的权威来源：

- 现有章节结构；
- 已引用文献；
- 当前算法名称和符号；
- 当前表格、图和篇幅限制；
- 需要被修改或删除的旧 claim。

## Level C：BEAM 项目交接资料

本文件及精简后的 BEAM docs 用于说明：

- BEAM 实际做了什么；
- 哪些设计已经验证；
- 哪些方向被否决；
- 哪些结论可以写进论文；
- 哪些表述会 overclaim。

## Level D：历史阶段文档

历史 Stage 0–5A 文档只用于：

- correctness provenance；
- supplementary material；
- negative ablation；
- artifact documentation。

除非与最终 `experiments.md` 完全同环境、同 commit、同配置，否则不能作为最终主实验数字。

---

# 2. BEAM 项目实际完成了什么

## 2.1 研究定位

BEAM 是一个面向无损 flat correction-set graph summarization 的 merge-engine 设计。

论文应将它定位为：

> 对 SWeG-style divide-and-merge pipeline 的算法工程重构：将廉价 candidate screening 与 exact correction-set objective evaluation 分离，在有界候选工作量下对 retained pairs 做精确评分。

BEAM 不是：

- 新的图摘要表示模型；
- 全局最优算法；
- 以 CUDA 为主要贡献的 GPU 论文；
- MAGS-DM 的替代或全面 SOTA 声明。

## 2.2 核心算法语义

输入是无向简单图。给定 flat partition，每个 unordered block pair 的成本为：

```text
BlockCost(A,B)
=
min(
    edges(A,B),
    1 + capacity(A,B) - edges(A,B)
)
```

其中：

```text
cross block:
capacity(A,B) = |A| * |B|

internal block:
capacity(A,A) = |A| * (|A|-1) / 2
```

全局 partition cost：

```text
ExactPartitionCost(P)
=
sum over unordered block pairs BlockCost(A,B)
```

merge gain：

```text
gain(A,B)
=
ExactPartitionCost(P)
-
ExactPartitionCost(P after merging A and B)
```

只有 `gain > 0` 的 merge 才降低当前编码成本。

## 2.3 BEAM 的执行流程

推荐的论文级抽象：

```text
divide/group current supernodes
→ use lightweight profiles to construct a bounded candidate graph
→ evaluate every retained pair with exact integer local gain
→ keep positive-gain pairs
→ deterministic descending-gain endpoint-disjoint matching
→ blocked partition update
→ repeat
→ final lossless correction-set encoding
```

关键边界：

- proxy 只负责控制 candidate search budget；
- proxy 不是最终 objective；
- exact scoring 决定 retained pair 是否有正收益；
- greedy matching 的保证只适用于固定 retained candidate graph；
- 最终 partition 仍受 candidate sparsification 和长期 merge sequence 影响。

---

# 3. 已完成的主要工程与验证

## Stage 0：冻结基线和 profiling

发现：

- legacy prepare 在多轮中重复从原图重建 quotient row；
- 总扫描量约为 16.461–19.653 个完整原图等价量；
- exact calls per committed merge 很高；
- CUDA 不能解决候选过多、原图重复扫描和 divide/candidate-generation 工作量。

该阶段适合用于 motivation 和 supplementary work breakdown。

## Stage 1：统一整数 CostOracle

实现：

```text
BlockCost
ExactMergeGain
ExactPartitionCost
```

验证：

- 500 个固定随机小图；
- 每个合法 merge 都满足 local exact gain 等于 global partition cost difference；
- lossless reconstruction；
- payload cost、partition cost 和 MAGS-compatible encoding cost 一致；
- frozen checksums 保持不变。

这是主论文中最强的 correctness contribution 之一。

## Stage 2：Persistent QuotientGraph

维护：

```text
supernode size
internal edge count
sorted quotient adjacency row
version/active state
optional children
```

实现：

```text
GetEdgeCount
GetCapacity
GetBlockCost
ExactCost
ExactMergeGain
Merge
MergeBatchBulk
ValidateAgainstOriginalGraph
```

主要价值：

- 初始化后不再从原始 CSR 重建 prepare rows；
- zero-copy quotient row view；
- prepared row entries copied = 0；
- prepared row copy bytes = 0；
- specialized raw-row exact-gain kernel。

正确性：

- random graphs、multiple merge orders、deterministic graphs；
- legacy/persistent partition 和编码结果一致；
- lossless reconstruction 和 checksums 通过。

论文中不要声称 persistent state 必然提升端到端 runtime。安全表述是：

> It removes post-initialization original-graph row reconstruction and prepared-row copying while preserving exact semantics.

## Stage 3A：Quotient-neighbor CandidateIndex

结论：

- checkpoint static recall 可以很高；
- online final compression 明显恶化；
- 增大固定 candidate budget 不能修复；
- 该路径被否决并默认关闭。

科学结论：

> Static positive-pair recall does not imply a high-quality online merge sequence.

只适合作为 negative ablation。

## Stage 3B：Residual-signature CandidateIndex

结果：

- 顶部局部 ranking 指标有所改善；
- FA exact calls 明显下降；
- runtime 和 final compression 均恶化；
- early local gain 更好，但最终 partition 更差。

科学结论：

> Better local pair ranking and lower scoring work do not guarantee a better final partition.

只适合作为 negative ablation，不可用于证明默认 EA-proxy 的 ranking quality。

## Stage 4A：Safe Certification

结果：

- full exact scans 大幅减少；
- 真实 row-entry work 只小幅下降；
- 没有稳定端到端 runtime 改善；
- EM/DB 出现回归。

科学结论：

> Reducing a prominent counter does not necessarily reduce the dominant work.

作为 supplementary 或 negative ablation。

## Stage 5A：Commit-policy audit

固定：

```text
candidate source
candidate ordering
exact-scored candidate graph
threshold
candidate refresh frequency
```

只改变：

```text
matching policy
commit order
micro-batch size
current-state marginal validation
mutual-best priority
```

主要结果：

```text
Facebook:
G0 cost 42,681
best cost 42,676
improvement 0.0117%

Email-Enron:
G0 cost 126,992
best cost 126,979
improvement 0.0102%
```

其他结论：

- FA interaction delta 可测，但 nonpositive current marginals 极少；
- EM negative marginals 也很少；
- sequential、transactional、micro-batched、mutual-best policy 几乎不改变最终结果；
- mutual-best 没有改善最终 partition；
- Phase B 不再进行。

论文中的正面用途：

> 更昂贵的 current-state transactional commit 对最终 exact cost 的改善不超过 0.012%，因此保留简单 deterministic blocked commit。

不要把它写成“所有数据集结果完全相同”。

---

# 4. 最终推荐配置

BEAM 工程结论中的稳定推荐路径：

```text
state backend: persistent
candidate index: legacy/default EA-proxy
certification: off
commit policy: g0
cost objective: MAGS-compatible
```

但论文的最终实验配置必须以论文项目 `experiments.md` 为准。

如 `experiments.md` 使用了不同 backend 或参数，应在论文中明确区分：

- original implementation；
- optimized/recommended implementation；
- experimental ablation paths。

---

# 5. Default EA-proxy 证据边界

证据审计结论：

```text
Default EA-proxy versus exact-gain ranking audit: ABSENT
```

当前默认路径最强证据是：

```text
Facebook audited run:
retained candidates = 289,013
exact-positive retained candidates = 99,331
```

该数据只支持：

> A large fraction of proxy-retained candidates are not exact-positive, so exact validation cannot be replaced by proxy screening.

它不支持：

- best-positive recall；
- proxy/exact Spearman correlation；
- top-1 exact hit；
- regret；
- early/middle/late ranking stability；
- “proxy reliably retains the globally best exact pair”。

论文中应称为：

> retained-candidate exact-positive rate in the audited Facebook run

不要称为普遍的 “proxy precision”。

---

# 6. 对写论文真正有用的 docs 文件

建议从 BEAM 仓库整理出以下文件，而不是整个 `docs/`。

## 6.1 必须交给论文改写对话

### 1. `docs/paper_experiment_selection.md`

用途：

- 告诉改写对话哪些证据放主论文、supplement、negative ablation 或 internal only；
- 是最适合快速理解证据层级的入口。

### 2. `docs/paper_claim_evidence_matrix.md`

用途：

- 每个 claim 对应什么证据；
- 防止摘要、引言和结论 overclaim；
- 改写 Contributions、Evaluation findings 和 Conclusion 时最重要。

### 3. `docs/paper_ready_experiment_tables.md`

用途：

- 已整理成论文可用形式的表格；
- 包含可支持和不可支持的表述边界；
- 可作为重新制作 LaTeX tables 的基础。

### 4. `docs/paper_proxy_exact_audit_status.md`

用途：

- 明确 default proxy audit 为 ABSENT；
- 防止误用 Stage 3 ranking audit；
- 约束 proxy 相关叙事。

### 5. `docs/beam_x_stage5a_commit_policy_results.md`

用途：

- blocked commit 设计验证；
- FA/EM 的 0.0117% 和 0.0102%；
- 可用于正文 compact table。

### 6. CostOracle correctness 文档

选择实际仓库中最完整的一份 Stage 1 / CostOracle 设计与验证文档。

必须包含：

- exact integer cost semantics；
- local-global equality；
- lossless reconstruction；
- objective consistency；
- checksums。

### 7. Persistent QuotientGraph 总结文档

优先选择一份能汇总 Stage 2 的文档，而不是把所有 task 文档上传。

应覆盖：

- quotient state；
- merge identities；
- zero-copy row view；
- original-graph scan elimination；
- correctness equivalence；
- runtime claim boundary。

## 6.2 建议同时提供的机器可读文件

### 8. `docs/paper_evidence_table.csv`

用途：

- 查具体数值和 provenance；
- 不建议让改写对话直接从 CSV 自行发明新的聚合值；
- 只有在需要核对表格时使用。

### 9. `docs/paper_evidence_inventory.md`

用途：

- 追踪某个 claim 的原始文档、配置和环境；
- 作为 reference index，不作为首先阅读的长文档。

## 6.3 只在需要时提供

- `docs/beam_x_stage5a_commit_policy_design.md`
- Stage 0 profiling 文档
- `docs/beam_x_stage2_task5_exact_kernel.md`
- full correctness matrix
- safe-certification results
- quotient-neighbor results
- residual-signature results
- PreparedRowRegistry audit

这些不应该一次性全部交给改写对话。

---

# 7. 文件按论文角色分类

## Main paper

优先使用：

1. 论文项目 `experiments.md` 中的最终 end-to-end 表格；
2. CostOracle local-global equality 和 lossless correctness；
3. persistent state 消除原图重建和 row copies；
4. default retained candidates 需要 exact validation 的有限证据；
5. Stage 5A blocked commit design validation；
6. 最终 runtime/compression/scaling 结果，以 `experiments.md` 为准。

## Supplementary / Artifact

- Stage 0 详细 work counters；
- full correctness matrices；
- legacy/persistent paired runs；
- CUDA counters；
- detailed commit policies；
- safe-certification scan and row-work counters；
- CLI、commit、checksums 和 commands。

## Negative ablation

用一张紧凑表或一个 subsection 总结：

- PreparedRowRegistry；
- quotient-neighbor CandidateIndex；
- residual-signature CandidateIndex；
- safe certification；
- transactional/mutual-best commit。

每个方向只保留：

```text
hypothesis
measured effect
decision
scientific lesson
```

不要按开发时间线写成长篇工程日志。

## Internal only

不要进入论文：

- provenance 不完整的 runtime；
- mixed-machine numbers；
- 旧 objective 的 cost ratio；
- historical CSV 中缺少 commit、machine、threads 或 dataset checksum 的值；
- task-sum 与 wall-clock 混合值；
- MAGS-DM 历史不同机器的直接 speedup。

---

# 8. 论文可以安全使用的核心 claim

## Claim A：exact local gain 具有全局一致性

推荐表述：

> For every valid merge, the local integer gain computed over the affected quotient rows equals the exact difference in global partition cost.

## Claim B：persistent state 消除重复重建工作

推荐表述：

> The persistent quotient representation removes post-initialization reconstruction of supernode adjacency from the original CSR and supports zero-copy row access.

不要写：

> Persistent state always improves end-to-end runtime.

## Claim C：proxy 不能替代 exact validation

推荐表述：

> In the audited Facebook run, only 99,331 of 289,013 retained candidates had positive exact gain, demonstrating that screening alone is insufficient for merge acceptance.

不要写：

> The proxy has poor precision on all datasets.

## Claim D：简单 blocked commit 足够

推荐表述：

> On Facebook and Email-Enron, more expensive sequential, transactional, micro-batched, and mutual-best commit policies improve final exact cost by at most 0.012% over the default blocked policy.

不要写：

> Batch interactions do not exist.

## Claim E：局部候选指标不能预测最终 partition

推荐表述：

> Experimental candidate indexes can improve checkpoint ranking metrics or reduce exact evaluations while producing a worse online merge trajectory and final summary.

## Claim F：CUDA 只是可选 backend

推荐表述：

> CPU and CUDA implementations share the same retained-candidate scoring semantics; CUDA is an optional execution backend rather than an algorithmic requirement.

---

# 9. 不应在论文中使用的 claim

不要写：

- state-of-the-art；
- BEAM makes all flat summarization faster；
- BEAM is better than all current flat summarizers；
- SWeG is the primary same-model baseline；
- default proxy has high recall；
- proxy ranking closely matches exact gain；
- persistent state guarantees runtime speedup；
- endpoint-disjoint merges are objective-independent；
- exact-positive isolated gains are additive；
- all commit policies produce identical partitions；
- Stage 3 ranking audit validates the default EA-proxy；
- CUDA is the main algorithmic contribution；
- negative results mathematically prove that BEAM cannot be improved。

---

# 10. 建议的论文改写流程

## Step 1：冻结最终数字

让新对话先读取 `experiments.md`，列出：

- datasets；
- final algorithm configurations；
- runtime metric；
- compression metric；
- number of runs；
- aggregation rule；
- authoritative tables；
- machine and commit。

没有完成这一步前，不改摘要。

## Step 2：建立 claim-to-result map

用 `paper_claim_evidence_matrix.md` 和 `experiments.md` 生成：

```text
claim
→ exact supporting table/figure
→ applicable datasets
→ limitations of scope
```

## Step 3：先改 Evaluation

顺序：

1. experimental setup；
2. authoritative end-to-end table；
3. runtime/compression tradeoff；
4. correctness/objective consistency；
5. work elimination；
6. blocked commit validation；
7. compact negative ablation；
8. CUDA/backend discussion。

## Step 4：再改 Method

保证算法描述与最终实验配置一致：

- cost objective；
- state backend；
- candidate path；
- commit policy；
- CPU/CUDA semantics；
- final encoder。

## Step 5：改 Introduction 和 Contributions

Contributions 应只写已经由正文实验直接支撑的内容。

## Step 6：最后写 Abstract 和 Conclusion

摘要中的每一个数字必须能够指向 `experiments.md` 中的一行或一张表。

---

# 11. 给新论文改写对话的启动 Prompt

复制下面的 Prompt，并同时上传：

- 当前论文源文件或 PDF；
- 论文项目的 `experiments.md`；
- 本交接文件；
- `paper_claim_evidence_matrix.md`；
- `paper_ready_experiment_tables.md`；
- `paper_experiment_selection.md`；
- `paper_proxy_exact_audit_status.md`；
- Stage 5A results；
- 一份 CostOracle summary；
- 一份 Persistent QuotientGraph summary。

```text
You are helping rewrite a research paper about BEAM, a merge engine for
lossless flat correction-set graph summarization.

Read all uploaded files before editing anything.

SOURCE-OF-TRUTH RULES

1. The paper project's experiments.md is the sole authoritative source
   for final runtime, compression, speedup, average, and per-dataset
   numbers.

2. The current paper source/PDF is authoritative for the existing
   structure, notation, citations, and text that must be revised.

3. The BEAM handoff document and selected BEAM docs are authoritative
   for method history, correctness evidence, design decisions, negative
   ablations, and claim boundaries.

4. Never combine values from different machines, commits, cost
   objectives, algorithm paths, or dataset versions.

5. When a result in BEAM docs conflicts with experiments.md, use
   experiments.md for final paper numbers and report the conflict.

PAPER POSITIONING

Position BEAM as an algorithm-engineering redesign of a SWeG-style
divide-and-merge merge engine:

- lightweight profiles bound candidate search;
- every retained candidate is evaluated by an exact integer
  correction-set CostOracle;
- positive-gain endpoint-disjoint pairs are selected by deterministic
  greedy matching;
- a persistent quotient representation removes repeated reconstruction
  of current supernode rows from the original graph;
- blocked commit is retained because more expensive current-state commit
  policies improve final exact cost by at most 0.012% on the audited
  Facebook and Email-Enron runs;
- CUDA is an optional scoring backend, not the primary contribution.

CLAIM CONSTRAINTS

Do not claim:

- state-of-the-art performance;
- that BEAM improves every flat graph summarization method;
- that the default proxy has high recall, high correlation with exact
  gain, or low regret;
- that persistent state necessarily improves end-to-end runtime;
- that simultaneous endpoint-disjoint merges are independent;
- that all commit policies always produce identical partitions;
- that Stage 3 experimental CandidateIndex audits validate the default
  EA-proxy;
- that negative experiments prove BEAM is mathematically unoptimizable.

Safe proxy statement:

The default EA-proxy only bounds candidate generation. In the audited
Facebook run, 99,331 of 289,013 retained candidates had positive exact
gain, showing why exact validation is necessary. This is not a
cross-dataset proxy precision or recall result.

Safe commit statement:

On Facebook and Email-Enron, sequential, transactional, micro-batched,
and mutual-best commit variants improve final exact cost by at most
0.012% over the default blocked policy.

REWRITE ORDER

1. First summarize experiments.md and identify every authoritative final
   table and metric.
2. Build a claim-evidence map.
3. Rewrite Evaluation first.
4. Rewrite Method so it exactly matches the evaluated implementation.
5. Rewrite Introduction and Contributions.
6. Write Abstract and Conclusion last.
7. For every numerical claim, identify its exact source in
   experiments.md.
8. Mark any missing evidence instead of inventing a result.

NEGATIVE ABLATIONS

Do not narrate the entire engineering chronology. Compress the negative
results into a design-space validation section. For each path report:

- hypothesis;
- measured result;
- decision;
- scientific lesson.

The relevant lessons are:

- static candidate recall does not guarantee a good online sequence;
- improved local ranking can worsen the final partition;
- reducing full-scan counters may not reduce dominant row-entry work;
- transactional commit does not repair a deficient candidate
  trajectory.

INITIAL OUTPUT

Before rewriting the paper, produce:

A. a one-page summary of the current paper's main claim;
B. a table of authoritative experiments from experiments.md;
C. a claim-evidence matrix;
D. a list of current sentences that overclaim or use stale method
   descriptions;
E. a proposed new section outline;
F. a prioritized edit plan.

Do not edit files until these six outputs have been reviewed.
```

---

# 12. 最小上传集合

上下文有限时，只上传这 7 项：

1. 当前 paper PDF 或主要 `.tex` 文件；
2. `experiments.md`；
3. 本交接文件；
4. `paper_claim_evidence_matrix.md`；
5. `paper_ready_experiment_tables.md`；
6. `paper_experiment_selection.md`；
7. `paper_proxy_exact_audit_status.md`。

随后按需补：

- Stage 5A results；
- CostOracle summary；
- Persistent QuotientGraph summary；
- evidence table CSV。

这样足以让新对话理解 BEAM 做了什么，同时避免被大量历史 engineering docs 淹没。
