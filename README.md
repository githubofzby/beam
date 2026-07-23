# BEAM / BEAM-X 图摘要原型

本仓库实现无向图的无损摘要算法 BEAM，并保留 BEAM-X 各阶段的实验路径。程序可输出：

- supernode 划分；
- superedge 集合 `P`；
- positive corrections `Cp`；
- negative corrections `Cm`；
- `cost_ratio_mags_compatible` 等压缩指标；
- runtime、候选漏斗、quotient graph 和 commit interaction 统计。

`error_bound=0.0` 时，输出可以无损恢复原始边集合。

## 当前项目状态

当前推荐的 CPU reference 路径为：

```text
MAGS-compatible CostOracle
+ Persistent QuotientGraph
+ legacy CandidateIndex
+ certification off
+ greedy full-batch commit (g0)
```

以下功能保留用于 A/B、profiling 和消融实验，但没有通过成为推荐默认路径的质量或性能门槛：

| 功能 | 开关 | 当前结论 |
|---|---|---|
| Quotient-neighbor 候选 | `--candidate-index quotient-neighbor` | 候选减少，但压缩率明显下降 |
| Residual-signature 候选 | `--candidate-index residual-signature` | checkpoint ranking 改善，但在线压缩率和 runtime 未通过 |
| Safe certification | `--certification safe` | full exact scan 显著减少，但整体 runtime 未稳定改善 |
| Transactional commit | `--commit-policy s1/t2/t4/t8/m4` | 可测得 interaction，但 FA/EM 压缩改善约为 0.01% |

因此，正式复现实验应显式使用 `--candidate-index legacy --certification off --commit-policy g0`。

详细阶段报告位于 `docs/`。

## 依赖

CPU 构建需要：

- 支持 C++17 的编译器；
- CMake 3.16 或更高版本；
- OpenMP（建议安装，用于 CPU 并行）；
- Python 3（仅测试和 lossless reconstruction 检查需要）。

CUDA 路径还需要 CUDA Toolkit 和可用的 NVIDIA GPU。CUDA 不是当前推荐主线。

## 编译

所有命令均在仓库根目录执行。

### CPU Release 构建

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DSWEG_ENABLE_CUDA=OFF

cmake --build build -j
```

生成的可执行文件为：

```text
build/beam
```

### CUDA 构建

```bash
cmake -S . -B build_cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DSWEG_ENABLE_CUDA=ON

cmake --build build_cuda -j
```

如果 CMake 找不到 CUDA 编译器，会给出警告并使用 CPU stub，而不是真实 CUDA scoring。

### 不使用 CMake

```bash
mkdir -p build_gpp

g++ -std=c++17 -O3 -DNDEBUG -fopenmp -Iinclude \
  src/main.cpp \
  src/sweg.cpp \
  src/scoring_cpu.cpp \
  src/candidate_index.cpp \
  src/cost_oracle.cpp \
  src/quotient_graph.cpp \
  src/cuda_scoring_stub.cpp \
  src/graph_io/graph_io.cpp \
  -o build_gpp/beam
```

查看程序实际支持的参数：

```bash
./build/beam --help
```

## 输入格式

程序读取普通文本无向单边 edge list：

```text
0 1
0 2
1 3
```

要求：

- 每个非注释行包含两个整数 `u v`；
- 每条无向边只出现一次；
- 程序内部自动生成双向 CSR arcs；
- 空行和以 `#` 开头的注释行会被跳过；
- 不要添加 `n m` header；
- 不支持 self-loop；
- 不要输入已经双向化的边集合，否则边数会翻倍。

对于包含 `m` 条边的输入：

```text
input_edges_raw = m
graph_arcs = 2m
original_undirected_edges = m
```

仓库实验数据默认位于：

```text
../../datasets/
```

例如：

```text
../../datasets/facebook/facebook.undir.single.txt
../../datasets/email_enron/email_enron.undir.single.txt
../../datasets/amazon/amazon.undir.single.txt
../../datasets/dblp/dblp.undir.single.txt
../../datasets/youtube/youtube.undir.single.txt
../../datasets/roadnet/roadnet.undir.single.txt
../../datasets/livejournal/livejournal.undir.single.txt
```

## 推荐运行方式

### 设置 CPU 线程

正式运行前建议显式固定 OpenMP 配置：

```bash
export OMP_NUM_THREADS=24
export OMP_PROC_BIND=close
export OMP_PLACES=cores
```

线程数应根据机器的物理核心数调整。比较实验必须记录相同的线程数和 affinity。

### 当前推荐 CPU reference 配置

```bash
mkdir -p results

./build/beam \
  --input ../../datasets/facebook/facebook.undir.single.txt \
  --merge-mode batch-ea-blocked \
  --scoring-backend cpu \
  --cost-objective mags-compatible \
  --state-backend persistent \
  --quotient-update incremental \
  --candidate-index legacy \
  --certification off \
  --commit-policy g0 \
  --top-k 16 \
  --group-batch-size 64 \
  --ea-use-threshold \
  --threshold-policy reciprocal \
  --iterations 20 \
  --print-offset 0 \
  --seed 1 \
  --error-bound 0.0 \
  --profiling summary \
  --results-csv results/facebook.csv
```

注意：程序自身默认仍保留 legacy objective 和 legacy state，便于历史 A/B。要运行 BEAM-X reference，必须显式传入上面的 objective、state 和算法开关。

### 输出摘要并验证无损恢复

```bash
mkdir -p outputs/facebook

./build/beam \
  --input ../../datasets/facebook/facebook.undir.single.txt \
  --merge-mode batch-ea-blocked \
  --scoring-backend cpu \
  --cost-objective mags-compatible \
  --state-backend persistent \
  --quotient-update incremental \
  --candidate-index legacy \
  --certification off \
  --commit-policy g0 \
  --top-k 16 \
  --group-batch-size 64 \
  --ea-use-threshold \
  --threshold-policy reciprocal \
  --iterations 20 \
  --seed 1 \
  --error-bound 0.0 \
  --write-output \
  --out outputs/facebook

python3 tools/check_reconstruction.py \
  --graph ../../datasets/facebook/facebook.undir.single.txt \
  --compressed outputs/facebook
```

摘要目录包含：

```text
G.txt
P.txt
Cp.txt
Cm.txt
```

正式 algorithm runtime 不建议包含写盘时间；只有需要检查 payload 时才使用 `--write-output`。

## 批量运行七个数据集

```bash
export OMP_NUM_THREADS=40
export OMP_PROC_BIND=close
export OMP_PLACES=cores

BEAM=./build/beam
DATA=/data/zby/datasets
RESULTS=results/beam_x_cpu.csv

mkdir -p results logs

for item in \
    "facebook|$DATA/facebook/facebook.undir.single.txt" \
    "email_enron|$DATA/email_enron/email_enron.undir.single.txt" \
    "amazon|$DATA/amazon/amazon.undir.single.txt" \
    "dblp|$DATA/dblp/dblp.undir.single.txt" \
    "youtube|$DATA/youtube/youtube.undir.single.txt" \
    "roadnet|$DATA/roadnet/roadnet.undir.single.txt" \
    "livejournal|$DATA/livejournal/livejournal.undir.single.txt" \
    "hollywood|$DATA/hollywood/hollywood.undir.single.txt" \
    "orkut|$DATA/orkut/orkut.undir.single.txt" \
    "uk-2002|$DATA/uk-2002/uk-2002.undir.single.txt" \
    "uk-2005|$DATA/uk-2005/uk-2005.undir.single.txt"
do
  IFS='|' read -r name input <<< "$item"

  "$BEAM" \
    --input "$input" \
    --merge-mode batch-ea-blocked \
    --scoring-backend cpu \
    --cost-objective mags-compatible \
    --state-backend persistent \
    --quotient-update incremental \
    --candidate-index legacy \
    --certification off \
    --commit-policy g0 \
    --top-k 16 \
    --group-batch-size 64 \
    --ea-use-threshold \
    --threshold-policy reciprocal \
    --iterations 20 \
    --print-offset 0 \
    --seed 1 \
    --error-bound 0.0 \
    --profiling summary \
    --results-csv "$RESULTS" \
    > "logs/${name}.log"
done
```

如果 CSV 已存在但 schema 与当前版本不同，程序会拒绝追加。此时请使用新文件名或先移动旧 CSV。



cuda模式

```bash
export OMP_NUM_THREADS=40
export OMP_PROC_BIND=close
export OMP_PLACES=cores

BEAM=./build_cuda/beam
DATA=/data/zby/datasets
RESULTS=results/beam_x_cuda.csv

mkdir -p results logs

for item in \
    "facebook|$DATA/facebook/facebook.undir.single.txt" \
    "email_enron|$DATA/email_enron/email_enron.undir.single.txt" \
    "amazon|$DATA/amazon/amazon.undir.single.txt" \
    "dblp|$DATA/dblp/dblp.undir.single.txt" \
    "youtube|$DATA/youtube/youtube.undir.single.txt" \
    "roadnet|$DATA/roadnet/roadnet.undir.single.txt" \
    "livejournal|$DATA/livejournal/livejournal.undir.single.txt" \
    "hollywood|$DATA/hollywood/hollywood.undir.single.txt" \
    "orkut|$DATA/orkut/orkut.undir.single.txt" \
    "uk-2002|$DATA/uk-2002/uk-2002.undir.single.txt" \
    "uk-2005|$DATA/uk-2005/uk-2005.undir.single.txt"
do
  IFS='|' read -r name input <<< "$item"

  "$BEAM" \
    --input "$input" \
    --merge-mode batch-ea-blocked \
    --scoring-backend cuda \
    --cost-objective legacy \
    --state-backend persistent \
    --quotient-update incremental \
    --candidate-index legacy \
    --certification off \
    --commit-policy g0 \
    --top-k 16 \
    --group-batch-size 64 \
    --ea-use-threshold \
    --threshold-policy reciprocal \
    --iterations 20 \
    --print-offset 0 \
    --seed 1 \
    --error-bound 0.0 \
    --profiling summary \
    --results-csv "$RESULTS" \
    > "logs/${name}.log"
done
```



## 主要参数

### 基本运行参数

| 参数 | 含义 | 默认值 |
|---|---|---:|
| `--input <path>` | 输入 edge list，必填 | 无 |
| `--iterations <int>` | Divide + Merge 轮数 | `20` |
| `--seed <int>` | 随机种子 | `1` |
| `--print-offset <int>` | 每隔多少轮打印一次 | `1` |
| `--error-bound <double>` | 丢边比例，`0.0` 为 lossless | `0.0` |
| `--results-csv <path>` | 追加一行结构化结果 | 不写 |
| `--write-output` | 生成 `G/P/Cp/Cm` | 关闭 |
| `--out <dir>` | 摘要输出目录 | 无 |

`--write-output` 必须与 `--out` 一起使用。

### 算法与状态

| 参数 | 可选值 | 默认值 |
|---|---|---:|
| `--merge-mode` | `batch-ea`, `batch-ea-blocked` | `batch-ea` |
| `--scoring-backend` | `cpu`, `cuda` | `cpu` |
| `--cost-objective` | `legacy`, `mags-compatible` | `legacy` |
| `--state-backend` | `legacy`, `persistent` | `legacy` |
| `--quotient-update` | `incremental`, `bulk_rebuild`, `auto` | `incremental` |
| `--top-k <int>` | 每个 supernode 的 EA-proxy top-k | `16` |
| `--group-batch-size <int>` | blocked 模式每批 group 数 | `64` |
| `--divide-hash-dims <int>` | divide hash 维数 | `16` |
| `--divide-max-group <int>` | divide group 大小上限 | `512` |

`mags-compatible` 当前只支持 CPU scoring。`--validate-quotient` 会在 merge 后用原图重建检查，仅用于小图调试。

### Threshold

启用 threshold：

```bash
--ea-use-threshold --threshold-policy reciprocal
```

支持：

- `reciprocal`：`theta(t) = 1 / (1 + t)`；
- `mags-geom`：从 `threshold-high` 到 `threshold-low` 的几何衰减；
- `adaptive`：根据上一轮 positive saving ratio 样本自适应调整。

相关参数：

| 参数 | 默认值 |
|---|---:|
| `--threshold-high` | `0.5` |
| `--threshold-low` | `0.005` |
| `--threshold-min-low` | `0.005` |
| `--adaptive-q-high` | `0.85` |
| `--adaptive-q-low` | `0.15` |
| `--adaptive-sample-limit` | `4096` |
| `--acceptance-target` | `0.15` |

## 实验和调试开关

### CandidateIndex 消融

```bash
--candidate-index legacy
--candidate-index quotient-neighbor --candidate-budget 8
--candidate-index residual-signature --candidate-budget 4
```

后两种模式要求 persistent CPU state，只用于消融，不推荐替代 legacy reference。

### Safe certification 消融

```bash
--cost-objective mags-compatible \
--state-backend persistent \
--scoring-backend cpu \
--candidate-index legacy \
--certification safe
```

该路径提供 safe upper bound 和 exact early abort。逻辑 exact request 数仍记录在 `merge_exact_gain_calls`，真正完整扫描数记录在 `exact_full_scan_count`。

### Commit policy 消融

支持：

| Policy | 含义 |
|---|---|
| `g0` | 当前 greedy endpoint-disjoint full batch，推荐 reference |
| `s1` | 固定候选图，逐 pair current-state validation |
| `t2` | transactional micro-batch 2 |
| `t4` | transactional micro-batch 4 |
| `t8` | transactional micro-batch 8 |
| `m4` | mutual-best 优先 + transactional batch 4 |

启用 exact-cost trajectory audit：

```bash
--commit-policy t4 \
--commit-audit \
--commit-audit-csv results/commit_trajectory.csv
```

`--commit-audit` 会调用额外的 exact quotient cost 检查。其开销单独记录为 `audit_oracle_ms`，不能作为生产 runtime。

### Profiling

```bash
--profiling off
--profiling summary
--profiling rounds --profile-csv results/profile_rounds.csv
```

- `off`：不保留逐轮 profile；
- `summary`：输出全局计数和阶段时间；
- `rounds`：额外保存每轮长格式 CSV；
- `--profile-csv` 必须与 `--profiling rounds` 一起使用。

Stage 0 runner：

```bash
bash tools/run_stage0_overhead.sh ./build/beam
bash tools/run_stage0_profiling.sh ./build/beam
```

脚本会写入 `docs/`，不要用于覆盖已冻结的 Stage 0 artifacts。

## 输出指标

正式压缩比较必须使用：

```text
cost_ratio_mags_compatible
```

不要使用 `cost_ratio` 或 `cost_ratio_standard` 替代正式 MAGS-compatible 结果。

无向图的兼容成本定义为：

```text
cost_ratio_mags_compatible
= (|P_nonloop| + 0.5 * |P_loop| + |Cp| + |Cm|) / m
```

程序同时输出 `encoding_cost_mags_x2`，用整数保存两倍兼容成本，避免浮点歧义。

### Runtime

| 字段 | 含义 |
|---|---|
| `runtime_run_ms` | `Sweg::Run()`，包含 divide 和 merge |
| `runtime_divide_ms` | Divide 累计时间 |
| `runtime_merge_ms` | Merge 累计时间 |
| `runtime_encode_ms` | 最终 encoding materialization 时间 |
| `runtime_output_ms` | `G/P/Cp/Cm` 写盘时间 |
| `runtime_algorithm_ms` | `runtime_run_ms + runtime_encode_ms` |
| `runtime_end_to_end_ms` | algorithm + output |

正式与 MAGS-DM 比较使用 `runtime_algorithm_ms`，不包含读图和写盘。

### 搜索和 commit

重点字段包括：

```text
merge_exact_gain_calls
merge_selected_pairs
exact_gain_input_nnz
exact_full_scan_count
exact_entries_scanned
exact_entries_skipped
isolated_gain_sum
realized_marginal_gain_sum
interaction_delta
negative_marginal
validation_exact_calls
commit_validation_ms
audit_oracle_ms
partition_hash
```

## 测试

### CTest

```bash
ctest --test-dir build --output-on-failure
```

包含 CostOracle、compression metrics、QuotientGraph、row view、CandidateIndex 和 Python integration tests。

### 小图 lossless 测试

```bash
python3 tools/run_small_tests.py \
  --binary ./build/beam
```

### 单独验证已有摘要

```bash
python3 tools/check_reconstruction.py \
  --graph tests/two_communities6.edgelist \
  --compressed outputs/two_communities
```

## 冻结结果与设计文档

关键报告包括：

```text
docs/beam_x_baseline.md
docs/beam_x_stage0_profile.md
docs/beam_x_stage1a_cost_oracle.md
docs/beam_x_stage1b_cost_oracle_integration.md
docs/beam_x_stage2_task5_exact_kernel.md
docs/beam_x_stage3a_candidate_index.md
docs/beam_x_stage3b_residual_signature.md
docs/beam_x_stage4a_safe_certification.md
docs/beam_x_stage5a_commit_policy_design.md
docs/beam_x_stage5a_commit_policy_results.md
```

验证 Stage 0/1 冻结文件：

```bash
sha256sum -c docs/beam_x_stage0_freeze.sha256
sha256sum -c docs/beam_x_stage1_freeze.sha256
```

## 可复现性记录

正式实验应同时记录：

- git commit hash；
- build type；
- compiler 和版本；
- compile flags；
- CPU 型号；
- `OMP_NUM_THREADS`；
- `OMP_PROC_BIND` / `OMP_PLACES`；
- GPU 和 CUDA 版本；
- 完整参数；
- 随机种子。

Stage 0 的冻结环境与七数据集基线见 `docs/beam_x_baseline.md`。
