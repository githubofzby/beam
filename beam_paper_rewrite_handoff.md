# BEAM 论文重写与 VLDB 投稿指导
## `beam_paper_rewrite_handoff.md`（审计后修订版）

> 适用目标：PVLDB / VLDB Regular Research Paper  
> 项目名称：论文中统一使用 **BEAM**；`BEAM-X` 仅是内部重构与诊断工程名，不得作为论文算法名。  
> 仓库：https://github.com/githubofzby/beam  
> 本版用途：指导后续对论文 LaTeX/PDF、`experiments.md`、图表、贡献列表和实验叙事进行系统重写。

---

# 0. 先给结论

现有 handoff 的基本方向是对的：降低 CUDA 的地位，突出 exact objective、persistent quotient state、bounded candidate work 和系统化设计验证。

但为了投 VLDB，必须做三项关键修正。

第一，不能把论文写成“BEAM 比当前图摘要方法更快”或“BEAM 达到新的 runtime–compression Pareto frontier”。现有证据不足以支持这类 headline claim，尤其 MAGS-DM 是同一 lossless flat-summary 目标下的重要强基线。

第二，不能用下面这类说法解释 MAGS-DM 的速度优势：

> BEAM 的速度差距只是算法模型造成的，换一个算法即可消除这部分影响。

这句话不适合论文，原因是：

- BEAM 与 MAGS-DM 解决的是同一类 lossless flat correction-set summarization 问题；
- 搜索骨架和 merge trajectory 的差异正是算法比较的核心，而不是可以从比较中“扣除”的干扰因素；
- 当前运行时间差距可能同时包含算法工作量、并行结构、数据结构和实现效率差异；
- “换算法即可消除”是未来工作推测，不是当前 BEAM 的实证贡献；
- 这种表述反而会让 reviewer 认为当前算法本身缺乏竞争力。

第三，论文的创新性不能靠提高语气获得，而要靠重新抽象。最可辩护、也最适合 VLDB systems/algorithms reviewer 的核心定位是：

> **BEAM is an objective-faithful, work-bounded merge engine for lossless flat graph summarization. It separates approximate candidate proposal from exact merge decisions, maintains the evolving partition as a persistent quotient graph, and executes deterministic blocked merges under an exact correction-set cost contract.**

中文概括：

> BEAM 不是新的摘要表示，也不是 GPU 算法，而是一套面向动态 divide-and-merge 图摘要流程的、目标一致且工作量受控的 merge-engine 架构。

这应成为全文唯一主线。

---

# 1. 当前仓库审计结论

## 1.1 当前根目录 `experiments.md` 不能直接作为最终论文唯一数据源

仓库当前 `experiments.md` 同时混入了至少三类不同证据：

1. 历史 SWeG / SLUGGER / BEAM 表格；
2. 历史 CUDA breakdown；
3. MAGS-DM 历史对比和不同阶段的 BEAM 数值。

其中存在明显的不一致，例如：

- FA 的 BEAM relative size 在不同表中出现 `0.484` 和 `0.497`；
- 同一数据集的 BEAM runtime 同时出现历史秒数和当前 CPU 路径秒数；
- 历史 CUDA 路径使用 legacy objective，而当前正式 CPU 路径使用 MAGS-compatible objective；
- SWeG、SLUGGER、旧 BEAM 和当前 persistent BEAM 不是同一 commit、机器或算法路径；
- MAGS-DM 历史时间缺少足以支持严格 speedup 的统一 provenance。

因此，本 handoff 修订后的 source-of-truth 顺序为：

```text
frozen raw CSV/log + run manifest
>
由这些原始文件生成并审计过的 experiments.md
>
论文中的最终表格
>
paper-ready docs
>
Stage 0–5A 历史文档
```

在清理完成前，不得继续宣称“`experiments.md` 中出现的所有数值都是最终权威值”。

## 1.2 当前代码默认值与论文推荐配置不同

当前 CLI 兼容性默认值仍是：

```text
cost objective: legacy
state backend: legacy
candidate index: legacy
commit policy: g0
```

而论文正式 CPU 路径应显式写全：

```text
--scoring-backend cpu
--cost-objective mags-compatible
--state-backend persistent
--candidate-index legacy
--certification off
--commit-policy g0
--merge-mode batch-ea-blocked
--top-k 16
--group-batch-size 64
--iterations 20
--seed 1
```

论文不能只写“we use the default configuration”，因为程序默认值不等于论文主实验配置。

## 1.3 当前 MAGS-compatible 路径不是 CUDA 路径

代码对下面组合显式拒绝：

```text
--cost-objective mags-compatible
--scoring-backend cuda
```

因此当前 CPU 和 CUDA CSV 不是“同一算法和同一目标，仅更换 scoring backend”的纯 backend comparison。

论文必须明确：

- 正式算法与正确性结果基于 CPU MAGS-compatible 路径；
- CUDA 是历史或可选实现路径；
- 未统一 objective 和 merge semantics 前，不报告“保持相同输出的 CUDA speedup”；
- Abstract、Introduction、Contributions 和 Conclusion 中不把 CUDA 列为主要贡献。

## 1.4 当前仓库未包含可供逐句审计的论文源文件

仓库根目录包含代码、实验记录和 docs，但没有当前论文的主 `.tex` 或论文 PDF。

因此本文件给出的是完整的论文级改写指南，而不是对现有论文逐句修改后的结果。下一步真正改论文时，必须同时读取：

- 当前主 `.tex` 及所有 `\input{}` 文件，或完整 PDF；
- bibliography；
- 当前 figures/tables；
- 清理后的 `experiments.md`；
- 本 handoff。

---

# 2. VLDB 投稿类别与论文形态

## 2.1 推荐类别

优先按 **Regular Research Paper — Systems / Algorithms** 写，而不是把论文写成纯 benchmark 或工程报告。

理由：

- BEAM 有明确的新执行架构；
- 有 exact CostOracle 和 local–global equality；
- 有 persistent quotient state、zero-copy row access 和 specialized exact-gain kernel；
- 有完整原型和大图实验；
- 有可复用的设计原则和失败路径分析。

但论文必须达到 systems paper 的基本形态：

```text
principled design
+ precise method
+ working end-to-end prototype
+ controlled evaluation
+ honest comparison
+ reproducible artifact
```

## 2.2 EA&B 不应作为当前首选

如果论文主要写成“我们测试了多种候选索引和 scheduler，它们都失败了”，则更像 Experiment, Analysis & Benchmark paper。

但 EA&B 需要：

- 更广泛且公平的现有方法比较；
- 当前 MAGS/MAGS-DM、SWeG、SLUGGER 等基线；
- 完整实验数据和强制可复现 artifact；
- 新的、具有普适价值的系统性结论。

在不做当前 MAGS-DM 控制实验的约束下，EA&B 反而更难成立。因此本次仍建议 Regular Research Paper，以 BEAM 的新 merge-engine 架构为主，negative ablation 只作为设计验证。

## 2.3 篇幅规划

按 PVLDB Regular Research Paper 的 12 页正文上限规划，参考分配如下：

| Section | 建议页数 |
|---|---:|
| 1. Introduction | 1.25 |
| 2. Problem and Design Goals | 1.00 |
| 3. BEAM Overview | 0.75 |
| 4. Exact Objective and Merge Gain | 1.25 |
| 5. Work-Bounded Merge Processing | 1.75 |
| 6. Persistent Quotient Execution | 1.25 |
| 7. Evaluation | 3.50 |
| 8. Related Work | 0.75 |
| 9. Limitations and Conclusion | 0.50 |
| 图表、留白和浮动开销 | 约 1.00 |

References 不计入正文页数，但 appendix 计入正文页数。详细日志和大型表格放 supplementary material。

---

# 3. 论文的唯一中心问题

论文应围绕一个清晰问题展开：

> **How can a dynamic divide-and-merge summarizer bound candidate-search work without allowing approximate screening heuristics to determine the actual correction-set objective?**

中文：

> 在动态 divide-and-merge 图摘要中，如何限制候选搜索工作量，同时保证真正的 merge 决策仍由精确 correction-set 编码目标决定？

全文不再同时追逐以下多个松散主题：

- GPU acceleration；
- matching approximation；
- adaptive threshold；
- 新 summary representation；
- 全领域 SOTA；
- 多种 CandidateIndex 的竞争；
- scheduler optimization。

这些都只能作为实现细节、设计选择或负面消融。

---

# 4. 推荐的论文 thesis

## 4.1 一句话 thesis

> BEAM decouples bounded candidate proposal from exact objective evaluation and maintains the evolving partition as a persistent quotient graph, enabling objective-faithful merge processing without repeatedly reconstructing current supernode state from the original graph.

## 4.2 三个设计矛盾

Introduction 中可用三个 tension 组织问题：

### Tension 1：Candidate work 与 search quality

全局搜索代价高，bounded proposal 必须缩小候选空间；但 proposal heuristic 会影响长期不可逆 merge trajectory。

### Tension 2：Cheap screening 与 objective fidelity

proxy 可以筛选候选，但不能作为最终 merge objective。每个 retained pair 必须经过 exact gain 评估。

### Tension 3：Dynamic partition 与 state maintenance

每轮从原图重建当前 supernode adjacency 会重复扫描；持久 quotient state 能消除重建，但必须保持精确计数、更新正确性和可控访问成本。

## 4.3 论文回答

BEAM 的回答由四层组成：

1. **Exact cost contract**：统一整数 BlockCost、ExactPartitionCost 和 ExactMergeGain；
2. **Bounded proposal**：proxy 只生成受控 retained candidate graph；
3. **Persistent state**：维护 supernode size、internal edges 和 quotient adjacency；
4. **Deterministic execution**：在 exact-positive candidate graph 上执行确定性 endpoint-disjoint blocked merge。

---

# 5. 如何合法“拔高”创新性

创新性不应写成“我们用了 top-k、OpenMP 和 CUDA”，而应提升到以下系统抽象。

## 5.1 Contribution A：Objective contract

把 CostOracle 写成 BEAM 的语义核心，而不是测试工具。

推荐表述：

> We define a single integer correction-set cost contract shared by candidate validation, partition auditing, and final encoding. For every legal merge, the local quotient-row computation equals the independently recomputed global partition-cost difference.

重点：

- 一个 objective definition；
- local scoring、global auditing、final payload 三者一致；
- 避免 legacy reporting、internal block 和 double-arc 语义漂移；
- lossless reconstruction 与 payload consistency 构成 end-to-end correctness chain。

不要声称：

- 这是新的摘要模型；
- 形式化证明覆盖所有后端和所有实现；
- CUDA 已共享完全相同语义。

## 5.2 Contribution B：Proposal–decision separation

这应是论文最重要的算法设计原则。

推荐表述：

> BEAM treats approximate similarity only as a proposal mechanism. The proxy bounds the number of pairs entering the expensive stage, while merge acceptance and ordering are based on exact local encoding gain.

强调：

- proposal 决定看哪些 pair；
- exact oracle 决定 retained pair 是否有益；
- correctness 不依赖 proxy score；
- quality 仍受 proposal recall 和长期 sequence 影响；
- 这是 objective fidelity，不是 global optimality。

## 5.3 Contribution C：Persistent quotient execution

推荐表述：

> BEAM materializes the evolving partition as a persistent quotient graph and exposes immutable row views to CPU scoring. This removes post-initialization original-CSR reconstruction and prepared-row copying while retaining exact merge semantics.

强调 deterministic work elimination：

- post-init original CSR scans = 0；
- prepared row entries copied = 0；
- prepared row bytes copied = 0；
- local exact gain uses two sorted quotient rows；
- merge update obeys exact edge-count identities。

不要把这一贡献写成“persistent graph always speeds up execution”。当前证据支持 work elimination，不支持全数据集必然 runtime speedup。

## 5.4 Contribution D：Design-space diagnosis

把失败实验压缩成可推广的科学结论：

> We systematically separate logical counters from dominant work and offline candidate diagnostics from online merge trajectories.

四个核心结论：

1. 高静态 recall 不保证高质量 online trajectory；
2. 更好的局部 ranking 不保证更好的 final partition；
3. 减少 full scans 或 exact calls 不保证减少 dominant work；
4. 更复杂 transactional commit 不能修复有缺陷的 candidate trajectory。

这部分是设计验证，不是论文唯一创新。

---

# 6. 推荐的 Contributions 段落

Introduction 末尾控制为三项，不要列六七项工程改动。

可直接使用下面的英文骨架：

> **First**, we formulate an objective contract for lossless flat correction-set summaries and implement an integer CostOracle whose local merge gain is exactly consistent with global partition cost and final encoding payload.
>
> **Second**, we design a work-bounded merge engine that separates approximate candidate proposal from exact merge decisions and executes deterministic positive-gain matching over a bounded retained graph.
>
> **Third**, we develop a persistent quotient representation with zero-copy CPU row access and a specialized local-gain kernel, and use a comprehensive design-space evaluation to identify which reductions in logical work translate—or fail to translate—into end-to-end improvements.

若最终有受控主实验，可以在第三项末尾加入具体结果。没有冻结表格前，不写任何平均 speedup。

---

# 7. MAGS / MAGS-DM 的正确处理

## 7.1 首先明确：MAGS-DM 是同问题的重要相关方法

MAGS-DM 不能被写成：

- 不可比较；
- 使用不同 lossless representation；
- 只是工程优化；
- 没有开源；
- 与 SWeG-style 方法完全无关。

MAGS-DM 同样优化 lossless summary graph + corrections，并采用 divide-and-merge 结构。其方法包含：

- MinHash-based dividing；
- 从每个 source node 出发选择 shortlist；
- 在 shortlist 中按 saving 选择 merge partner；
- redesigned merge threshold；
- 针对 dividing 和 merging 的完整算法重设计。

因此 Related Work 必须正面说明其强项。

## 7.2 不推荐的 MAGS 叙事

删除或禁止以下表述：

> MAGS-DM 速度快只是因为算法模型不同，因此这部分差距不属于 BEAM 的问题。

> 更换 MAGS-DM 的算法模型后即可消除速度影响。

> BEAM 与 MAGS-DM 的压缩率几乎一样。

> MAGS-DM 不能作为 BEAM 的公平 baseline。

这些说法会被 reviewer 直接质疑。

## 7.3 推荐的 MAGS 叙事

安全的总表述：

> BEAM and MAGS-DM optimize the same lossless flat-summary representation but traverse the partition space using substantially different proposal and merge trajectories. BEAM is a controlled redesign of the SWeG-derived merge execution path, whereas MAGS-DM also redesigns dividing, node selection, similarity estimation, and threshold scheduling.

进一步说明：

> Our diagnostics indicate that BEAM's remaining quality gap is primarily associated with candidate search and the resulting long-term merge sequence, rather than stale endpoint-disjoint commit alone. The available runtime data, however, do not isolate algorithmic work from implementation and environment effects.

这比“完全是算法模型差异”更准确。

## 7.4 历史九数据集结果如何写

如果保留历史审计结果，只能写成 motivation 或 limitation：

> In a historical nine-dataset audit, BEAM was slower than MAGS-DM on all datasets and achieved a lower encoding cost on only Facebook. Across the other eight datasets, its cost was worse by roughly 3.76% on average, with a largest observed gap of about 9.67%. Because these runs came from different code states or experimental environments, we do not use them to derive a controlled speedup claim.

不要写“压缩差距可以忽略”。更准确的说法是：

```text
close on average in the historical overlap,
but not uniformly close and not a controlled final-system comparison
```

## 7.5 三种可选的 MAGS 呈现级别

### Level A：最佳方案——当前控制实验

若最终愿意重新运行 MAGS-DM，必须统一：

- dataset checksum；
- undirected edge semantics；
- objective denominator；
- machine；
- physical cores / threads；
- compiler flags；
- repetitions；
- runtime boundaries；
- input loading是否计时；
- algorithm iteration budget；
- raw logs。

此时可在主文加入 runtime–compression Pareto 图。

### Level B：本次现实方案——历史质量审计，不报 speedup

如果不重新运行：

- MAGS-DM 不进入 headline runtime 表；
- 历史 compression audit 放 limitation、discussion 或 supplement；
- 不计算历史几何平均 speedup；
- 主文只陈述 algorithmic differences；
- 明确这仍是论文最大 evaluation risk。

### Level C：最保守方案——只放 Related Work

如果历史 provenance 无法清楚解释：

- 删除定量 MAGS 表；
- Related Work 正面讨论 MAGS/MAGS-DM；
- Conclusion 中承认没有做 matched current-system comparison。

## 7.6 对 reviewer 的诚实边界

可以写：

> The controlled predecessor comparison centers on SWeG because BEAM directly redesigns its merge-stage execution path.

但必须紧接：

> MAGS-DM is a stronger contemporary divide-and-merge system and represents an important external frontier. A matched end-to-end comparison remains necessary to determine whether BEAM establishes a new Pareto point.

不能暗示 SWeG 是唯一同模型 baseline。

---

# 8. 数学与算法部分应如何重写

## 8.1 Problem definition

定义：

```text
G = (V,E): simple undirected graph
P: flat partition of V
S: summary graph over supernodes
C+ / C-: positive and negative corrections
```

每条无向边只计算一次。

Cross block capacity：

```text
cap(A,B) = |A||B|, A != B
```

Internal block capacity：

```text
cap(A,A) = |A|(|A|-1)/2
```

Block cost：

```text
c(A,B) = min{ m(A,B), 1 + cap(A,B) - m(A,B) }
```

Global cost：

```text
C(P) = sum_{unordered block pairs {A,B}} c(A,B)
```

Merge gain：

```text
g(A,B) = C(P) - C(P after A,B -> W)
```

只允许 `g(A,B) > 0` 的 exact-positive candidate 进入 selected graph。

## 8.2 Proposition 1：Fixed-partition optimal encoding

可正式写成 proposition：

> For a fixed partition, the optimal representation of each unordered block pair is either to store all present edges as positive corrections or to store one superedge and all missing edges as negative corrections. Therefore its exact cost is `min(m, 1 + cap - m)`.

证明只需一段。

## 8.3 Proposition 2：Merge locality

设 `W=A∪B`，则 exact gain 只涉及：

- `(A,A)`；
- `(B,B)`；
- `(A,B)`；
- `(W,W)`；
- 对所有外部 active block `X` 的 `(A,X)`、`(B,X)`、`(W,X)`。

可写：

```text
g(A,B)
=
c(A,A)+c(B,B)+c(A,B)-c(W,W)
+
sum_X [c(A,X)+c(B,X)-c(W,X)]
```

其中：

```text
m(W,X)=m(A,X)+m(B,X)
m(W,W)=m(A,A)+m(B,B)+m(A,B)
```

由于 zero-edge block cost 为零，只需遍历 `N_Q(A) ∪ N_Q(B)`。

这使 specialized kernel 的复杂度可表述为：

```text
O(deg_Q(A) + deg_Q(B))
```

前提是两行已排序并使用 two-pointer union。

## 8.4 Proposition 3：Current-state positive merge

仅对当前状态重新验证后的单个 merge 可以写：

> If a merge has positive gain under the current partition, committing it strictly decreases the exact partition cost by that gain.

不要把这一结论直接扩展到整批 endpoint-disjoint merge。Endpoint-disjoint 不代表 correction-set objective 下彼此独立。

## 8.5 Candidate work bound

若每个 source supernode 最多保留 `k` 个 proposal：

```text
directed retained proposals <= k|P|
```

去重后 exact-scored unordered pairs 不超过该数量。

但论文要补一句：

> The number of exact calls is bounded by retained-pair count, whereas exact scoring time is degree-weighted and depends on the quotient-row union scanned by each pair.

这能自然连接 safe-certification 的负面结果：call count 和 dominant row-entry work 不是同一个量。

## 8.6 Matching 的表述

只写：

- candidate graph 的 edge weight 是 exact gain；
- 按 gain 降序和 deterministic tie-break；
- 选择 endpoint-disjoint pairs；
- blocked commit 降低同步和更新频率。

不要把标准 greedy matching 的 `1/2` guarantee 写成核心贡献。若保留，只能限定为：

> On a fixed weighted retained graph, descending-weight greedy gives the standard approximation for static maximum-weight matching.

随后必须强调：

- candidate graph 已经 sparsified；
- gains 会随 commit 改变；
- 该保证不等价于对最终 graph-summary objective 的 approximation guarantee。

---

# 9. 推荐的论文结构

# 1 Introduction

按六段写。

### Paragraph 1：问题价值

说明 lossless graph summaries 对存储、传输、cache locality 和查询的意义。

### Paragraph 2：真正困难

固定 partition 下编码容易，困难在于寻找 partition。全局 merge 搜索昂贵，divide-and-merge 通过局部候选实现扩展性。

### Paragraph 3：现有执行矛盾

指出 SWeG-style pipeline 中：

- cheap similarity 与真正编码目标不一致；
- 动态 supernode state 容易被重复重建；
- 批量候选和 exact scoring 产生大量工作；
- 单个 counter 的下降未必转化为 runtime。

不要在这里先介绍 CUDA。

### Paragraph 4：BEAM 核心设计

用一段描述四层架构：

```text
bounded proposal
→ exact CostOracle
→ deterministic positive-gain matching
→ persistent quotient commit
```

### Paragraph 5：主要发现

只写最终冻结、可追溯的数字。建议包含：

- exact local/global equality coverage；
- post-init CSR scans 和 row-copy elimination；
- 11-dataset current CPU result中的一个或两个稳健 aggregate；
- commit-policy改善上限；
- 不加入未统一的 CUDA speedup。

### Paragraph 6：Contributions

使用三项贡献列表。

---

# 2 Problem and Design Goals

包括：

1. lossless flat correction-set representation；
2. MAGS-compatible exact block cost；
3. graph and CSR semantics；
4. merge gain；
5. design goals。

Design goals 建议写成：

- G1 Objective fidelity；
- G2 Work boundedness；
- G3 Incremental state maintenance；
- G4 Determinism and auditability；
- G5 Backend independence。

其中 backend independence 是接口目标，不表示当前 CUDA 与 CPU 已有相同 objective。

---

# 3 BEAM Overview

放一张系统图：

```text
Original CSR
    |
    v
Persistent Quotient State
    |
Divide / Group
    |
EA-proxy top-k proposal
    |
Exact integer gain
    |
Positive weighted candidate graph
    |
Deterministic endpoint-disjoint matching
    |
Blocked quotient update
    |
Repeat
    |
Lossless encoder
```

图中用两条虚线标出：

- **work boundary**：top-k proposal；
- **correctness boundary**：exact CostOracle。

Overview 只讲接口和数据流，不深入公式。

---

# 4 Exact Objective and Merge Gain

内容顺序：

1. BlockCost；
2. fixed-partition optimality；
3. local merge formula；
4. exact CostOracle API；
5. local–global equality；
6. encoder consistency。

将 correctness tests 放在 Evaluation，不要在 Method 中堆测试数量。

---

# 5 Work-Bounded Merge Processing

建议小节：

## 5.1 Divide and group construction

准确描述最终实际 divide 实现，不再沿用旧论文中已经失效的描述。

## 5.2 EA-proxy proposal

明确：

- proxy 的输入；
- proxy_gain / proxy_ratio；
- per-source top-k；
- deterministic tie-breaking；
- deduplication；
- proxy 不接受 merge。

安全句：

> The proxy is used solely to bound candidate generation. It is neither a surrogate objective nor a correctness certificate.

## 5.3 Exact candidate graph

所有 retained pairs 通过 exact local gain。

FA funnel 只能写：

> In the audited Facebook run, 99,331 of 289,013 retained candidates had positive exact gain.

不能称为 cross-dataset proxy precision。

## 5.4 Deterministic matching

描述 descending exact gain、endpoint-disjoint selection 和 tie-break。

## 5.5 Blocked execution

讲 cache、parallel work organization 和 update boundary。

不要把 batch independence 当成数学事实。

---

# 6 Persistent Quotient Execution

建议小节：

## 6.1 State layout

每个 active supernode：

```text
size
internal_edges
sorted cross-adjacency row
active/version
optional child relation
```

## 6.2 Merge update

写出：

```text
size[W] = size[A] + size[B]
m(W,X) = m(A,X) + m(B,X)
m(W,W) = m(A,A) + m(B,B) + m(A,B)
```

## 6.3 Zero-copy row access

解释 `QuotientRowView` / `PreparedRow`，强调 CPU path。

## 6.4 Specialized exact-gain kernel

two-pointer union、internal four-term handling、无 temporary hash/union row。

## 6.5 Correctness and lifetime

- row view 在 commit 前有效；
- version 检查；
- persistent update与full rebuild核对；
- final encoder仍可访问原图，不能说“初始化后完全不读原图”。

---

# 7 Evaluation

Evaluation 必须先改，并决定全文能写什么。

## 7.1 Research questions

建议按以下 RQ 组织：

### RQ1：Objective correctness

Does local exact gain match global partition-cost change, and does the final payload reconstruct the graph losslessly?

### RQ2：Work elimination

Which reconstruction and copy work is deterministically removed by persistent quotient state?

### RQ3：End-to-end behavior

What runtime and MAGS-compatible compression does the frozen BEAM CPU configuration achieve on the 11 datasets?

### RQ4：Why exact validation is needed

How many proxy-retained candidates are rejected by exact gain?

### RQ5：Which alternatives fail and why

Do alternative CandidateIndex, safe certification, or transactional commit improve the final runtime–compression outcome?

### RQ6：External frontier

How does BEAM relate to SWeG, SLUGGER, MAGS and MAGS-DM under the available evidence?

若没有 matched MAGS-DM run，RQ6 必须写成 scope/relationship，而不是 controlled performance question。

## 7.2 Experimental setup 必须包含

- machine model；
- CPU physical/logical cores；
- RAM；
- OS；
- compiler/version/flags；
- OpenMP threads/binding；
- CUDA disabled/enabled；
- git commit；
- input graph checksum；
- repetitions；
- warmup；
- aggregation；
- timeout；
- runtime boundary；
- output writing是否计时；
- exact CLI；
- raw CSV路径；
- cost objective；
- state backend；
- candidate path；
- seed；
- iterations。

## 7.3 正式 CPU 配置

主文应完整列出：

```text
scoring_backend = cpu
cost_objective = mags-compatible
state_backend = persistent
candidate_index = legacy
certification = off
commit_policy = g0
merge_mode = batch-ea-blocked
top_k = 16
group_batch_size = 64
iterations = 20
seed = 1
```

## 7.4 当前 11 数据集 CPU 表

以下值可作为当前待冻结主表，但最终必须由 raw CSV 和 derivation script 再生成一次：

| Dataset | Runtime algorithm (s) | MAGS-compatible ratio |
|---|---:|---:|
| FA | 0.720 | 0.483725 |
| EM | 3.759 | 0.690808 |
| AM | 20.532 | 0.580427 |
| DB | 18.738 | 0.515487 |
| YO | 100.473 | 0.729461 |
| RN | 168.916 | 0.694160 |
| LJ | 479.854 | 0.747223 |
| HO | 352.641 | 0.535241 |
| OR | 936.755 | 0.869938 |
| U2 | 2,545.322 | 0.113047 |
| U5 | 28,441.264 | 0.093765 |

不要与旧 Intel/P100/CUDA 表逐行拼接。

## 7.5 主文建议表格

### Table 1：Datasets and provenance

列：

```text
dataset
|V|
|E|
source
checksum
```

### Table 2：Current BEAM CPU end-to-end result

列：

```text
dataset
runtime_algorithm
cost_ratio_mags_compatible
supernodes
exact calls
committed merges
```

若 exact calls 太宽，移到 supplement。

### Table 3：Correctness and objective consistency

紧凑报告：

- 500 random graphs；
- every legal merge equality；
- 7 deterministic graphs；
- payload = partition cost = reported cost；
- lossless reconstruction；
- 300 persistent random graphs；
- multiple merge orders/update modes。

### Table 4：Deterministic work elimination

分 panel：

- legacy graph-equivalent scans；
- persistent post-init scans = 0；
- zero-copy copied entries/bytes = 0；
- runtime ratio单独列，明确不是因果换算。

### Table 5：Design validation / negative ablation

| Hypothesis | Observation | Decision | Lesson |
|---|---|---|---|
| 高 static recall 可安全降工作量 | online compression恶化 | Reject | Offline recall不预测trajectory |
| 更好 local ranking改善final summary | early gain更好但final cost更差 | Reject | Local ranking不等于long-term quality |
| Safe bounds降低runtime | scans大降但row work和runtime未改善 | Reject | Counter reduction不等于dominant work |
| Transactional commit修复batch interaction | final改善不超过0.012% | Keep G0 | Commit policy不是主要quality bottleneck |

## 7.6 主文建议图

### Figure 1：BEAM architecture

必须有。

### Figure 2：Runtime–compression view

只有当所有点来自同一可比实验时才画。

不能把历史 SWeG/SLUGGER、当前 CPU BEAM 和 legacy CUDA MAGS 混在一张 Pareto 图中。

### Figure 3：Work funnel

建议展示：

```text
proxy pairs examined
→ retained after top-k
→ exact positive
→ matched
→ committed
```

只用同一配置和数据集。

### Figure 4：Logical work vs real work

可展示 safe certification：

```text
full scans ↓95.7%
row-entry work ↓19.9%
runtime no stable improvement
```

这是一张很有 VLDB systems 味道的图，但需保证来源和配置清晰。

## 7.7 Runtime 报告规则

禁止：

- task-sum 与 wall time相加；
- 跨机器计算speedup；
- 单次运行声称统计显著；
- 把 graph loading、algorithm、encoding 和 output 混为同一口径；
- CPU/CUDA不同objective直接计算backend speedup。

优先报告：

```text
runtime_algorithm = run + encode
```

并单独列：

```text
load
output
```

如论文使用别的边界，必须全表统一。

---

# 10. `experiments.md` 的强制整改

在改 Abstract 前，先重写 `experiments.md`。

## 10.1 建议结构

```text
# 0. Frozen Paper Configuration
# 1. Machine and Software
# 2. Dataset Manifest and Checksums
# 3. Current CPU Main Results
# 4. Correctness Results
# 5. Work-Elimination Results
# 6. Design Validation / Negative Ablations
# 7. External Baselines
# 8. Historical Results — Not Used for Headline Claims
# 9. CUDA Legacy Path — Not Comparable to Current CPU
# 10. Derivation Scripts and Raw File Map
```

## 10.2 每个表必须有 metadata block

示例：

```yaml
table_id: current_cpu_main
git_commit: ...
machine: ...
compiler: ...
threads: 24
cost_objective: mags-compatible
state_backend: persistent
candidate_index: legacy
commit_policy: g0
iterations: 20
seed: 1
runtime_metric: runtime_algorithm
repetitions: ...
aggregation: median
raw_csv:
  - ...
derivation_script: ...
```

## 10.3 历史表必须显式隔离

标题必须带：

```text
Historical / not used for current headline comparison
```

旧十数据集 speedup、Intel/P100/CUDA breakdown、历史 MAGS-DM 表不得与 current CPU main results 处于同一主表。

## 10.4 冻结检查

最终 `experiments.md` 中，每个数字必须能追踪到：

```text
dataset
command
config
git commit
machine
threads
objective
raw row
derivation
```

---

# 11. Default EA-proxy 的证据边界

当前完整 default proxy ranking audit 为：

```text
ABSENT
```

当前只能支持：

```text
FA retained candidates = 289,013
FA exact-positive retained candidates = 99,331
exact-positive rate ≈ 34.4%
```

安全写法：

> In the audited Facebook run, only 99,331 of 289,013 proxy-retained pairs had positive exact gain, showing that screening alone is insufficient for merge acceptance.

不能写：

- high positive-pair recall；
- high best-pair recall；
- strong proxy/exact correlation；
- low regret；
- stable ranking across epochs；
- proxy approximates global exact order；
- proxy precision across all datasets。

Stage 3 quotient-neighbor 和 residual ranking audit 是 experimental CandidateIndex，不是 default EA-proxy。

---

# 12. Commit-policy 结果如何写

主文只保留一个紧凑设计验证。

推荐表述：

> On Facebook and Email-Enron, sequential, transactional, micro-batched, and mutual-best commit variants improved final exact cost by at most 0.012% over the default blocked policy under a fixed candidate graph.

随后解释：

- interaction delta 存在；
- isolated gains 并不严格可加；
- 但变为 nonpositive 的 current marginal 很少；
- 更复杂 commit 没有 materially 改善 final partition；
- 质量瓶颈更可能在 candidate trajectory。

不能写：

- batch interaction 不存在；
- endpoint-disjoint merge 对 objective 独立；
- 所有 policy 在所有图上相同；
- scheduler 永远无效。

---

# 13. CUDA 的降级方案

## 13.1 论文位置

CUDA 只放在：

- Implementation 小节最后一段；
- Evaluation 的 optional backend discussion；
- Supplementary artifact。

## 13.2 Abstract

不提 CUDA，除非未来完成：

```text
same objective
same candidate graph
same merge sequence
same output
only backend differs
```

## 13.3 Contributions

不列 CUDA。

## 13.4 Main evaluation

不使用当前 CPU/CUDA 几何平均 speedup，因为：

- CPU 使用 MAGS-compatible objective；
- CUDA 使用 legacy objective；
- partition、candidate sequence 和 final cost不同；
- 11个数据集中CUDA只在4个更快；
- 当前观测的几何平均 speedup约0.944×，不构成正面系统结论。

## 13.5 安全句

> CUDA is retained as an optional historical scoring backend. The objective-faithful BEAM configuration evaluated in this paper uses the CPU implementation.

---

# 14. Related Work 重写框架

按算法谱系写，不按“谁比我们差”写。

## 14.1 Lossless flat graph summarization

介绍 summary graph + corrections 目标和 fixed-partition encoding。

## 14.2 Greedy and divide-and-merge methods

介绍：

- Navlakha-style Greedy；
- SWeG；
- LDME；
- SLUGGER。

## 14.3 MAGS and MAGS-DM

必须说明：

- MAGS 改进 greedy search；
- MAGS-DM 改进 divide-and-merge；
- MinHash divide 和 shortlist；
- saving-based partner selection；
- threshold schedule；
- parallel implementation；
- 与 BEAM 同属 lossless flat-summary family。

推荐对比：

> MAGS-DM redesigns the search trajectory to improve both compactness and efficiency. BEAM instead focuses on objective-faithful execution and state maintenance within a SWeG-derived bounded-candidate pipeline.

不要说：

> MAGS-DM is only an implementation optimization.

## 14.4 Parallel and GPU graph processing

一段即可。不要把论文变成 GPU related-work survey。

---

# 15. 推荐标题

首选：

> **BEAM: Objective-Faithful and Work-Bounded Merge Processing for Lossless Graph Summarization**

备选：

> **BEAM: Exact-Gain Merge Engineering over Persistent Quotient Graphs**

> **BEAM: A Work-Bounded Merge Engine for Lossless Flat Graph Summarization**

不推荐：

- “Faster Graph Summarization with CUDA”；
- “State-of-the-Art Lossless Graph Summarization”；
- “BEAM-X”；
- “Optimal Graph Summarization”；
- “GPU-Accelerated SWeG”。

---

# 16. Abstract 最后再写

Abstract 采用五句结构：

1. **Problem**：lossless summary partition search昂贵；
2. **Gap**：bounded heuristics与exact objective、dynamic state maintenance之间存在矛盾；
3. **Method**：proposal–decision separation + persistent quotient + deterministic blocked processing；
4. **Correctness**：local/global equality + lossless payload；
5. **Results**：只使用最终 frozen current CPU table的数字和严格限定的design validation结果。

可用骨架：

> Lossless flat graph summarization represents a graph using supernodes, superedges, and edge corrections, but finding a compact partition requires repeatedly evaluating merges over an evolving graph state. We present BEAM, an objective-faithful and work-bounded merge engine that uses lightweight profiles only to propose a bounded candidate graph and evaluates every retained pair with an exact integer correction-set objective. BEAM maintains the current partition as a persistent quotient graph, supports zero-copy CPU row access, and executes deterministic blocked matching over exact-positive candidates. Our validation establishes equality between local merge gain, global partition-cost change, and final encoding payload, while eliminating post-initialization original-graph row reconstruction and prepared-row copying. Under the frozen CPU configuration, BEAM [insert only final audited runtime/compression results and scope].

不要在 freeze 前填最后一句。

---

# 17. Conclusion 的结构

Conclusion 不重复所有数字，写三层：

1. BEAM 解决了什么执行问题；
2. 哪些设计原则被验证；
3. 当前限制是什么。

推荐限制句：

> BEAM does not eliminate the trajectory dependence introduced by bounded candidate proposal, and the current evaluation does not establish a new Pareto frontier over MAGS-DM. Our results instead provide an exact and auditable merge-engine foundation on which stronger search strategies can be evaluated.

这句话诚实，但不会贬低贡献。

未来工作可以写：

> A promising next step is to combine the shared CostOracle and persistent execution machinery with a stronger MAGS-DM-style search trajectory or carefully bounded local repair.

不能写成当前已完成贡献。

---

# 18. 必须删除或降级的旧内容

## 从 Abstract 删除

- CUDA headline；
- “state-of-the-art”；
- 混合旧十数据集和新11数据集的平均值；
- SWeG/SLUGGER旧 speedup与current CPU结果混合；
- MAGS-DM未控制speedup；
- greedy matching的标准1/2 guarantee作为主要创新。

## 从 Introduction 删除或改写

- “SWeG is the primary same-model baseline”；
- “existing methods must choose either compactness or efficiency”，除非严谨讨论MAGS；
- “BEAM makes flat summarization faster”；
- “our proxy accurately estimates exact gain”；
- “CUDA solves merge bottleneck”。

## 从 Method 更新

- legacy state rebuild描述；
- 旧 cost ratio定义；
- 把 proxy当objective的措辞；
- 把 endpoint-disjoint视为无interaction；
- 与最终 CLI 不一致的参数和默认值。

## 从 Evaluation 删除或隔离

- task-sum + wall time；
- mixed-machine speedup；
- old objective与MAGS-compatible ratio同表；
- current CPU与legacy CUDA纯backend comparison；
- provenance不完整的commit-policy runtime；
- experimental CandidateIndex recall冒充default proxy evidence。

---

# 19. Reviewer 风险与应对

## Risk 1：MAGS-DM 已经更快且压缩更好，BEAM为何值得发表？

回答不能是“算法不同”。

论文应回答：

- BEAM 提供 objective-faithful merge execution architecture；
- exact local/global/payload consistency 是可复用契约；
- persistent quotient state消除确定性的重复重建；
- design-space audit揭示常见优化指标为何失效；
- 当前贡献是 merge-engine design，不是新Pareto冠军。

但必须承认：没有 matched MAGS-DM baseline 仍是最大风险。

## Risk 2：这些只是工程优化吗？

用以下内容回应：

- exact objective contract；
- merge-locality formulation；
- bounded proposal / exact decision separation；
- persistent mutable quotient abstraction；
- correctness invariants；
- degree-weighted work analysis；
- trajectory dependence的实证结论。

不要用代码行数回应。

## Risk 3：Persistent state没有稳定加速，为何是贡献？

回答：

- 它消除了可测且确定性的重复工作；
- 它建立了后续 exact local kernel 的状态基础；
- runtime由候选数量和row-entry work主导，说明 eliminating one work class不必然得到端到端加速；
- 这是系统设计中有价值的因果分解。

不要声称它单独提高runtime。

## Risk 4：Proxy质量没有审计

主动限定：

- proxy只作proposal；
- exact validation是correctness boundary；
- 只报告FA retained exact-positive funnel；
- 不做ranking claim。

## Risk 5：为什么固定20轮？

说明：

- 20是frozen configuration；
- 不声称普遍最优；
- 若保留历史参数分析，必须同配置；
- Related Work可说明MAGS-DM也采用多轮threshold schedule，但不能据此宣布公平。

## Risk 6：负面消融太多，主线不清

正文只保留一张设计验证表。完整开发时间线放 supplement。

---

# 20. Artifact 与 reproducibility

PVLDB 当前强调公开 supplemental material。Artifact 至少应包含：

```text
README_ARTIFACT.md
build instructions
compiler and dependency versions
dataset download/preprocess scripts
dataset checksums
exact paper commands
raw CSV/logs
table-generation scripts
figure-generation scripts
correctness tests
expected small-graph outputs
machine-readable config manifest
```

建议目录：

```text
artifact/
  configs/
  scripts/
  raw/
  derived/
  figures/
  tables/
  checksums/
  tests/
```

主表必须由脚本从 raw CSV 生成，不手工复制。

建议在仓库增加：

```text
docs/paper_run_manifest.yaml
docs/paper_table_manifest.yaml
```

---

# 21. 实际改写顺序

严格按以下顺序执行。

## Step 1：清理证据

- 重写 `experiments.md`；
- 冻结 current CPU raw CSV；
- 生成 dataset checksum；
- 冻结 commit、machine、threads和CLI；
- 标记所有historical表。

## Step 2：建立 claim–evidence map

每个 claim 写：

```text
claim
supporting raw result
applicable datasets
config
scope limitation
paper location
```

## Step 3：先重写 Evaluation

直到每一张表、每个 aggregate 和每个 caption都可追溯。

## Step 4：重写 Method

使文字与：

```text
persistent
legacy CandidateIndex
MAGS-compatible objective
G0
CPU
```

完全一致。

## Step 5：加入形式化 proposition

- fixed-partition block optimality；
- merge locality；
- current-state positive gain；
- candidate work bound。

## Step 6：写 Overview 图和 Introduction

用唯一中心问题组织全文。

## Step 7：更新 Related Work

正面加入 MAGS/MAGS-DM。

## Step 8：写 Limitations 与 Conclusion

提前化解 reviewer 对MAGS、proxy、CUDA和trajectory的质疑。

## Step 9：最后写 Abstract

每一个数字都要指向最终 frozen table。

---

# 22. 论文改写时的文件优先级

## 第一优先级

1. 当前论文主 `.tex` / PDF；
2. 清理后的 `experiments.md`；
3. raw current CPU CSV；
4. `docs/paper_claim_evidence_matrix.md`；
5. `docs/paper_ready_experiment_tables.md`；
6. `docs/paper_experiment_selection.md`；
7. `docs/paper_proxy_exact_audit_status.md`。

## 第二优先级

8. `docs/beam_x_stage1a_cost_oracle.md`；
9. `docs/beam_x_stage1b_cost_oracle_integration.md`；
10. `docs/beam_x_stage2_task1_row_view.md`；
11. `docs/beam_x_stage2_task5_exact_kernel.md`；
12. Stage 2 closing correctness report；
13. `docs/beam_x_stage5a_commit_policy_results.md`。

## 第三优先级

14. Stage 0 profiling；
15. quotient-neighbor results；
16. residual-signature results；
17. safe certification；
18. PreparedRowRegistry audit；
19. full CUDA counters。

不要先把所有历史日志塞进论文改写上下文。

---

# 23. 最终 claim checklist

提交前逐项回答“是/否”。

## Objective

- [ ] 全文只使用一个正式compression metric：`cost_ratio_mags_compatible`。
- [ ] internal block按无向边一次计数。
- [ ] local gain定义为global cost before minus after。
- [ ] payload cost与partition cost口径一致。

## Configuration

- [ ] 主实验写出完整CLI。
- [ ] 不使用“default”代替配置。
- [ ] CPU/CUDA objective区别已明确。
- [ ] `BEAM-X`未作为论文算法名。

## Baselines

- [ ] SWeG controlled predecessor定位清楚。
- [ ] MAGS和MAGS-DM已正面引用。
- [ ] 未声称MAGS-DM不可比。
- [ ] 未使用不受控runtime speedup。
- [ ] 若无matched MAGS实验，limitations明确承认。

## Claims

- [ ] 未声称SOTA。
- [ ] 未声称persistent必然加速。
- [ ] 未声称default proxy高recall。
- [ ] 未把experimental CandidateIndex audit写成default audit。
- [ ] 未声称endpoint-disjoint merge独立。
- [ ] 未声称所有commit policy总是相同。
- [ ] 未声称CUDA保持相同输出获得speedup。
- [ ] 未把future MAGS-DM hybrid写成当前贡献。

## Evidence

- [ ] 每个数字有raw source。
- [ ] 每个aggregate有derivation script。
- [ ] 不混机器、commit、objective或algorithm path。
- [ ] 图表可由artifact重新生成。
- [ ] negative ablation只占一张表或一个紧凑小节。

---

# 24. 给后续论文改写对话的启动 Prompt

```text
You are rewriting a PVLDB paper about BEAM, an objective-faithful and work-bounded merge engine for lossless flat graph summarization.

Read the current paper source/PDF, the cleaned experiments.md, the raw-table manifest, and beam_paper_rewrite_handoff.md before editing.

Core thesis:
BEAM separates bounded approximate candidate proposal from exact correction-set merge decisions, maintains the evolving partition as a persistent quotient graph, and executes deterministic blocked merges over exact-positive retained candidates.

Non-negotiable rules:
1. Use BEAM, not BEAM-X, as the paper method name.
2. Use cost_ratio_mags_compatible as the formal compression metric.
3. Do not combine values from different machines, commits, objectives, dataset versions, or algorithm paths.
4. Do not treat the current CPU and CUDA files as a pure backend comparison.
5. Do not claim that the default EA proxy has high recall, correlation, or low regret.
6. Do not claim that endpoint-disjoint gains are independent or additive.
7. Do not claim state of the art or a new Pareto frontier unless the final controlled tables support it.
8. Discuss MAGS and MAGS-DM as same-objective contemporary methods. Do not dismiss their comparison as an algorithm-model difference.
9. Position SWeG as the controlled predecessor because BEAM redesigns its execution path, not as the only comparable method.
10. Write the Abstract last.

Rewrite order:
A. audit experiments and reconstruct all tables;
B. rewrite Evaluation;
C. rewrite Method and formal propositions;
D. write the design-validation subsection;
E. rewrite Introduction and Contributions;
F. rewrite Related Work;
G. write Limitations and Conclusion;
H. write Abstract.

Before editing, output:
1. the current paper's actual main claim;
2. every stale or overclaimed sentence;
3. a table mapping claims to evidence;
4. a new section outline with page budgets;
5. a list of figures and tables to keep, replace, or delete;
6. unresolved evidence gaps.
```

---

# 25. 最终判断

这个重写方向可以投 VLDB，但必须接受一个现实边界：

> 论文最强的贡献不是“BEAM 已经击败 MAGS-DM”，而是“BEAM 建立了一套目标一致、工作量受控、可审计的 merge-engine 架构，并用系统化实验揭示了动态图摘要中 proposal quality、真实工作量和长期 merge trajectory 之间的关系”。

这是比“CUDA 加速 SWeG”更强、更完整、也更经得住 reviewer 追问的论文故事。

同时，不能用措辞掩盖当前缺少 matched MAGS-DM baseline 的问题。若本次不补该实验，就要：

- 降低 performance superiority claim；
- 强化 exactness、architecture 和 design insight；
- 把 MAGS-DM 作为外部 frontier 正面讨论；
- 在 Limitations 中明确承认比较边界；
- 确保 artifact 和证据治理足够强。
