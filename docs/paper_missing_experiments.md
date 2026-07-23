# 最小缺失实验：Default EA-Proxy / Exact-Gain Audit

当前状态为 **ABSENT**：需要一个小型、checkpoint-sampled audit；不需要重跑七图完整 benchmark，也不应改算法行为。

## 固定 production 配置

```text
state-backend persistent
candidate-index legacy
certification off
commit-policy g0
cost-objective mags-compatible
scoring-backend cpu
iterations 20
top-k 16
seed 1
OMP_NUM_THREADS 24（或最终论文固定线程数）
```

在正式采数前记录：完整 command、git commit、dirty diff hash、compiler/flags、CPU/memory/OS、OpenMP affinity、dataset absolute identity（n、m、file SHA-256）与 machine id。Facebook、Email-Enron、DBLP、LiveJournal 必须在同一机器/build上运行；runtime只作 audit overhead说明，不与历史 Intel/P100表混合。

## Reference scope（必须预先固定）

主 scope 推荐为：**sampled divide groups 中所有 generated pre-top-k pairs**。它直接回答 EA proxy top-k screening 保留了哪些 proxy-generated pairs，又避免对大 group/all legal graph pairs做二次方枚举。

- `all generated pre-top-k pairs`：由 legacy bucket/direct-edge规则产生且 `proxy_gain>0` 的去重 pair；这是 primary rank/recall denominator。
- `all legal within-group pairs`：只在预设小 group（例如 active size ≤128）做 secondary exhaustive reference，单独报告，不与 primary scope平均。
- `all bucket-induced pairs`：若包含 proxy delta≤0或不含 direct signal，必须另起 scope label。
- `sampled reference pairs`：只在预计 pre-top-k pair过大时使用；记录 inclusion probability/采样规则，不能和 exhaustive recall混称。
- `retained candidates only`：只能计算 retained nonpositive rate/order change，不能计算 top-k best-positive recall。

## Checkpoint sampling

每个 dataset 固定三类 epoch：early（例如第 1 个非空 merge epoch）、middle（按实际非空 epoch的50%位置）、late（最后一个仍产生 candidate 的 epoch）。每类预先选定少量 group，例如按 group id hash和 size strata确定性抽取 8--16 个，避免只挑容易 group。记录空/过大 group被跳过的原因。

推荐先做 overhead pilot：FA 的三个 checkpoint全部记录；根据 candidate row数量与日志大小决定 EM/DB/LJ 每 checkpoint group数。不要先做 full seven-dataset audit。

## Raw schema

每个 sampled candidate 一行：

```text
dataset,epoch,epoch_phase,group_id,reference_scope,
source_id,target_id,proxy_gain,proxy_ratio,proxy_rank_for_source,
retained_by_top_k,exact_gain,exact_rank_in_reference_scope,
positive_exact_gain,selected_for_matching,committed,
source_size,target_size,affected_row_entries
```

另加 run header/sidecar：machine id、commit、dirty status、build flags、dataset checksum、threads、seed、iterations、top-k、objective、candidate index、certification和commit policy。Pair ids 应在当前 representative namespace 中解释；若同一 unordered pair由两个 source方向保留，raw row需注明 per-source rank，同时 aggregate pair dedup规则与 production一致。

## Aggregate definitions

- **retained nonpositive rate** = retained且 `exact_gain<=0` / retained pairs。
- **proxy top-1 exact top-1 hit**：在每个 source 的 primary reference set 内，proxy rank 1 是否等于 exact gain rank 1；只对存在 positive exact pair 的 source统计。
- **best-positive recall@k**：每个 source reference set的 exact-best positive pair是否进入其 proxy top-k；另可报告 group-global版本，但名称必须区分。
- **exact top-k recall**：exact top-k pair中被 proxy top-k保留的比例；k及 per-source/group scope固定。
- **Spearman correlation**：对同一 reference scope内 proxy/exact ranks计算；ties采用预先声明的 average-rank或 deterministic endpoint tie-break。
- **gain-weighted recall** = retained positive exact gains之和 / reference scope全部 positive exact gains之和。
- **best-gain regret** = `(reference_best_positive_gain - best_retained_positive_gain) / max(1, reference_best_positive_gain)`；报告 median/p95。
- 所有指标按 dataset × early/middle/late分别给出；再给 macro aggregation时不得让大 group candidate count暗中变成 dataset权重。

## 开销与存储预估

Stage 0 FA 已有约 16.68M bucket pair examinations和 289k retained pairs；若对全部 examinations逐 pair exact-score，开销可能远高于正常 run。Primary scope应在**抽样 group/checkpoint**内 materialize去重 pre-top-k pairs，预计成本约为这些 group reference pairs的 two-row exact work，而非全图 16.68M examinations。

以每 raw row 约 120--220 bytes CSV估算：10万 rows约12--22 MB，100万 rows约120--220 MB；压缩 CSV通常更小。先从每 dataset约 50k--200k reference pairs设 storage cap（约6--44 MB未压缩），并记录 dropped/unsampled counts。runtime pilot应分别报告 normal algorithm、audit exact work、serialization时间；若 FA audit超过正常 runtime 10×或预计 LJ超过数小时，降低 deterministic group sample，而不是改变 reference definition。

## 最小执行顺序

1. 只在 instrumented build 上做 triangle/path smoke，确认 instrumentation off 时 payload和merge sequence byte-identical。
2. FA early/mid/late pilot，验证 schema、不变量、rank scope和开销。
3. 固定 sampling caps后运行 EM、DB、LJ；每个 dataset一次 deterministic audit即可，因为这是非runtime measurement。
4. 使用相同 raw rows离线生成全部 aggregates；不要为每个指标重跑算法。
5. 将 raw/sidecar/aggregate CSV纳入 artifact，并用未instrumented final build单独跑 end-to-end benchmark。

这个实验能填补 default proxy audit，但不自动证明 exact rescoring对 final partition的因果必要性。若论文需要该反事实，还需一个独立、受控的 proxy-only decision ablation；当前不建议把它与本次最小 ranking audit绑定。

