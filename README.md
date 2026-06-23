# BEAM C++ Prototype

`cpp/` 目录提供一个可运行的 SWeG C++ 原型，保留现有压缩核心与 merge/scoring 逻辑，当前支持：

- `batch-ea`
- `batch-ea-blocked`

默认目标：

- 保持现有 `Encode` 语义不变
- 可选输出兼容的 `G.txt / P.txt / Cp.txt / Cm.txt`
- 在 `error_bound=0.0` 下保证可重建
- 支持结构化指标输出和 `results.csv`

## 构建

### 用 CMake

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

如需启用 CUDA scoring backend 构建骨架：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSWEG_ENABLE_CUDA=ON
cmake --build build -j
```

### 没有 CMake 时直接用 g++

```bash
mkdir -p build_gpp
g++ -std=c++17 -O3 -Iinclude \
  src/main.cpp src/sweg.cpp src/scoring_cpu.cpp \
  src/cuda_scoring_stub.cpp src/graph_io/graph_io.cpp \
  -o build_gpp/beam
```

## 输入格式

程序只支持一种输入格式：

- 普通文本无向 edge list
- 每个非注释行包含两个整数：`u v`
- 表示一条无向边 `{u, v}`
- 每条无向边只出现一次
- 程序内部会自动补成 `(u, v)` 和 `(v, u)` 两条 arc

额外约束：

- 跳过空行
- 跳过以 `#` 开头的注释行
- 不支持 self-loop，遇到 `u == v` 会报错
- 不支持 header；不要把第一行写成 `n m`
- 不要提供已经双向化的 edge list，否则边数会翻倍

例如：

```text
0 1
0 2
1 3
```

程序内部会构建：

- `input_edges_raw = 3`
- `graph_arcs = 6`

## CLI 参数

当前 `beam` 支持的主要参数如下。

本次 threshold policy 改动后，`batch-ea` / `batch-ea-blocked` 新增了一组可选阈值参数。
如果你不传这些参数，默认行为仍然是旧版 reciprocal threshold，也就是：

- `theta(t) = 1 / (1 + t)`
- 对应 `--threshold-policy reciprocal`

另外有几个实际重要的运行前提会明显影响结果或性能：

- `batch-ea` 和 `batch-ea-blocked` 现在共用同一条 EA-proxy candidate generation 路径。
- `--top-k <= 0` 不会禁用 candidate generation，程序仍会按至少 `1` 处理。
- CPU 路径性能强依赖 OpenMP 线程环境；如果不显式设置 `OMP_NUM_THREADS`，性能可能和实验记录差很多。
- `results.csv` 表头已经新增 threshold 相关字段；如果复用旧 CSV，程序会因为 schema 不匹配而拒绝追加写入。建议使用新文件名或先删除旧 CSV。

### 基本参数

- `--input <path>`
  输入图路径。必填。

- `--results-csv <path>`
  将本次运行结果追加写入 CSV。

- `--write-output`
  显式写出 `G.txt / P.txt / Cp.txt / Cm.txt`。

- `--out <dir>`
  输出目录。只有和 `--write-output` 一起使用时才需要提供。

- `--iterations <int>`
  `Divide + Merge` 迭代次数。默认 `20`。

- `--print-offset <int>`
  每隔多少轮打印一次迭代日志。默认 `1`。

- `--seed <int>`
  随机种子。默认 `1`。

- `--divide-hash-dims <int>`
  Adaptive Multi-Hash Divide 使用的独立 hash 维度数。默认 `16`。

- `--divide-max-group <int>`
  Adaptive Multi-Hash Divide 的目标最大 group 大小。默认 `512`。

- `--error-bound <double>`
  当前建议保持 `0.0`。默认 `0.0`。

### Adaptive Multi-Hash Divide

- `--divide-hash-dims`
  Number of independent hash dimensions used by adaptive multi-hash divide.
  Default: `16`.
  Larger values provide more chances to split oversized groups, but may increase divide time.

- `--divide-max-group`
  Maximum target group size during adaptive divide.
  Default: `512`.
  Groups larger than this value are recursively split using additional hash dimensions.

If these parameters are not provided, BEAM uses the default values.
The existing `--seed` controls the deterministic randomness of the divide process.
Using the same seed gives reproducible group construction.
`--divide-max-group` is a hard cap on final divide group size. If all hash
dimensions are exhausted and a bucket is still larger than this value, BEAM
performs a deterministic fallback split using the existing seed and iteration
number, so final groups remain reproducible and do not exceed the cap.

### Merge 模式

- `--merge-mode batch-ea`
- `--merge-mode batch-ea-blocked`

### batch-ea 相关参数

- `--top-k <int>`
  `batch-ea` / `batch-ea-blocked` 中每个 supernode 保留的 EA-proxy 候选数。默认 `16`。

`batch-ea` 不再使用 SuperJaccard 作为 candidate proxy。
当前 candidate generation 使用基于 `EncodeCostForPair` 风格局部代价模型的 encoding-aware proxy。
`--top-k` 控制每个 supernode 在 exact scoring 之前保留多少个 EA-proxy candidate。
如果传入 `0` 或负数，当前实现仍会按至少 `1` 处理，而不是禁用 candidate。

- `--group-batch-size <int>`
  仅对 `batch-ea-blocked` 有效。默认 `64`。

- `--candidate-batch-budget <int>`
  仅对 `batch-ea-blocked + --scoring-backend cuda` 有效。默认 `0`。
  该参数只控制 CUDA exact-gain scoring 中 candidate pairs 的 chunk 大小。
  它不会再因为 candidate 分块而重复上传同一个 blocked slice 的 FlatAggCSR。

- `--overflow-group-gmax <int>`
  大 group 的局部细化阈值。默认 `0`，表示关闭。

- `--overflow-refine-rounds <int>`
  overflow group 的最大细化轮数。默认 `0`。

- `--ea-use-threshold`
  对 `batch-ea` / `batch-ea-blocked` 有效，启用 local gain ratio threshold 过滤。

- `--threshold-policy <reciprocal|mags-geom|adaptive>`
  仅对 `batch-ea` / `batch-ea-blocked` 生效。默认 `reciprocal`。

  - `reciprocal`
    旧版行为，`theta(t) = 1 / (1 + t)`。
  - `mags-geom`
    MAGS-DM 几何阈值：
    `high * pow(low / high, double(iter - 1) / double(T - 1))`。
  - `adaptive`
    使用上一轮采样到的 positive local gain ratio 分布，结合几何上界和 acceptance controller 自适应调节阈值。

- `--threshold-high <double>`
  `mags-geom` / `adaptive` 的几何阈值起点。默认 `0.5`。

- `--threshold-low <double>`
  `mags-geom` / `adaptive` 的几何阈值终点。默认 `0.005`。

- `--threshold-min-low <double>`
  `adaptive` 最终阈值下界。默认 `0.005`。
  如果想探索更激进的 EA threshold，可设成 `0.001`。

- `--adaptive-q-high <double>`
  `adaptive` 第 1 轮使用的高分位数。默认 `0.85`。

- `--adaptive-q-low <double>`
  `adaptive` 后期轮次使用的低分位数。默认 `0.15`。

- `--adaptive-sample-limit <int>`
  `adaptive` reservoir sampling 的样本上限。默认 `4096`。

- `--acceptance-target <double>`
  `adaptive` acceptance controller 的目标接受率。默认 `0.15`。

### threshold policy 说明

`batch-ea` / `batch-ea-blocked` 在启用 `--ea-use-threshold` 时，会用 exact local gain ratio：

- `saving_ratio = gain / before_cost`

再结合 `--threshold-policy` 做过滤。

如果没有传 `--ea-use-threshold`：

- 仍然会输出 threshold 相关 metrics
- 但 threshold 不参与 pair 过滤
- 这时这些指标主要用于对照实验和日志记录

`adaptive` 不会额外做一遍 prepass。
第 `t` 轮使用第 `t-1` 轮记录到的 positive saving ratio 样本，并用 reservoir sampling 控制样本规模。

### Scoring backend

- `--scoring-backend cpu`
  默认选项。建议配合 `OMP_NUM_THREADS` 使用。

- `--scoring-backend cuda`
  使用 CUDA scoring backend；当前主线 CUDA scoring 说明和 candidate batch 预算控制以 `batch-ea-blocked` 路径为准。若当前构建未启用真实 CUDA backend，则会报错或回退到 stub 能力边界。
  当前 blocked CUDA V1 会为每个 scoring slice 构造一个 combined `FlatAggCSR`，
  将 row arrays 上传一次并在 GPU 驻留，然后把 candidates 按 `--candidate-batch-budget`
  切成多个 chunk，用两个 non-blocking streams 做
  `H2D pairs -> kernel -> D2H results` 双缓冲流水。

- `--verify-cuda-gain`
  调试检查开关，只增加校验，不改变最终压缩结果。

CUDA metrics 说明：

- `cuda_row_h2d_ms`
  resident CSR 上传时间。
- `cuda_pair_h2d_ms`
  candidate pair chunk 的 H2D 时间。
- `cuda_kernel_ms`
  candidate chunk kernel 时间。
- `cuda_d2h_ms`
  candidate chunk result 的 D2H 时间。
- `cuda_total_ms`
  每次高层 `ScoreCandidatesCuda()` 调用的墙钟时间总和。
- `cuda_row_uploads`
  resident CSR 上传次数。
- `cuda_kernel_launches`
  candidate chunk / kernel launch 次数。
- `cuda_h2d_bytes`
  H2D 总字节数，包含 resident CSR 和 candidate pairs。
- `cuda_d2h_bytes`
  result D2H 总字节数。

由于两个 stream 可以 overlap，`cuda_row_h2d_ms + cuda_pair_h2d_ms + cuda_kernel_ms + cuda_d2h_ms`
可能大于 `cuda_total_ms`。这表示阶段时间按 stream 累加，而 `cuda_total_ms` 是墙钟时间。

## 使用方式（运行命令）

### 批量运行多个数据集（自动化命令）

如果需要批量在多个数据集上运行，可以参考以下 bash 脚本：

```bash
THREADS=40
BEAM=./build/beam
DATA=/data/zby/datasets

MODE=batch-ea
BACKEND=cpu
TOPK=32
ITERS=20
SEED=1
ERROR_BOUND=0.0

DIVIDE_HASH_DIMS=16
DIVIDE_MAX_GROUP=512

LOG=./logs_beam_ea_proxy_${MODE}_it${ITERS}_th${THREADS}_topk${TOPK}
CSV_DIR=./results/beam_ea_proxy_${MODE}_it${ITERS}_th${THREADS}_topk${TOPK}

mkdir -p "$LOG" "$CSV_DIR"

export OMP_NUM_THREADS=$THREADS
export OMP_PROC_BIND=close
export OMP_PLACES=cores

for item in \
  "facebook|$DATA/facebook/facebook.undir.single.txt" \
  "caida|$DATA/caida/caida.undir.single.txt" \
  "email_enron|$DATA/email_enron/email_enron.undir.single.txt" \
  "amazon|$DATA/amazon/amazon.undir.single.txt" \
  "dblp|$DATA/dblp/dblp.undir.single.txt"
do
  IFS="|" read -r name input <<< "$item"

  log_file="$LOG/${name}.log"
  csv_file="$CSV_DIR/${name}.csv"

  rm -f "$csv_file"

  echo "===== ${name} start $(date) =====" | tee "$log_file"
  echo "input=${input}" | tee -a "$log_file"
  echo "csv=${csv_file}" | tee -a "$log_file"
  echo "threads=${THREADS}, mode=${MODE}, backend=${BACKEND}, topk=${TOPK}, iterations=${ITERS}, seed=${SEED}" | tee -a "$log_file"
  echo "divide_hash_dims=${DIVIDE_HASH_DIMS}, divide_max_group=${DIVIDE_MAX_GROUP}" | tee -a "$log_file"

  /usr/bin/time -v "$BEAM" \
    --input "$input" \
    --merge-mode "$MODE" \
    --scoring-backend "$BACKEND" \
    --top-k "$TOPK" \
    --ea-use-threshold \
    --threshold-policy reciprocal \
    --iterations "$ITERS" \
    --seed "$SEED" \
    --divide-hash-dims "$DIVIDE_HASH_DIMS" \
    --divide-max-group "$DIVIDE_MAX_GROUP" \
    --error-bound "$ERROR_BOUND" \
    --results-csv "$csv_file" \
    2>&1 | tee -a "$log_file"

  status=${PIPESTATUS[0]}

  echo "exit_status=${status}" | tee -a "$log_file"
  echo "===== ${name} end $(date) =====" | tee -a "$log_file"
  echo ""
done

```



### 单个数据集运行示例

对当前主线算法，`batch-ea` 最常用、也是推荐的最小必要参数集合是：

```bash
./build/beam \
  --input <graph> \
  --merge-mode batch-ea \
  --scoring-backend cpu \
  --top-k 16 \
  --ea-use-threshold \
  --threshold-policy reciprocal \
  --iterations 20 \
  --seed 1 \
  --error-bound 0.0
```

其中真正重要的项是：

- `--merge-mode batch-ea`
- `--scoring-backend cpu` 或 `cuda`
- `--top-k`
- `--ea-use-threshold`
- `--threshold-policy`
- `--iterations`
- `--seed`

如果你要复现实验中的 CPU 性能，通常还需要显式设置：

```bash
export OMP_NUM_THREADS=<threads>
export OMP_PROC_BIND=close
export OMP_PLACES=cores
```

这些不是 BEAM 的 CLI 参数，但对运行速度非常重要。

正式实验默认不写 `G/P/Cp/Cm`，也不需要 `--out`：

```bash
./build/beam \
  --input data/facebook/facebook_combined.txt \
  --merge-mode batch-ea \
  --scoring-backend cpu \
  --top-k 4 \
  --ea-use-threshold \
  --threshold-policy reciprocal \
  --iterations 20 \
  --seed 1 \
  --error-bound 0.0 \
  --results-csv results.csv
```

一个完整示例：

```bash
OMP_NUM_THREADS=40 ./build/beam \
  --input data/facebook/facebook_combined.txt \
  --merge-mode batch-ea \
  --scoring-backend cpu \
  --top-k 4 \
  --ea-use-threshold \
  --threshold-policy reciprocal \
  --iterations 20 \
  --seed 1 \
  --error-bound 0.0 \
  --results-csv results/sweg_batch_ea.csv
```

如果需要切到 MAGS-DM 几何阈值：

```bash
OMP_NUM_THREADS=40 ./build/beam \
  --input data/facebook/facebook_combined.txt \
  --merge-mode batch-ea \
  --scoring-backend cpu \
  --top-k 4 \
  --ea-use-threshold \
  --threshold-policy mags-geom \
  --threshold-high 0.5 \
  --threshold-low 0.005 \
  --iterations 20 \
  --seed 1 \
  --error-bound 0.0 \
  --results-csv results/beam_batch_ea_mags_geom.csv
```

如果需要切到 adaptive threshold：

```bash
OMP_NUM_THREADS=40 ./build/beam \
  --input data/facebook/facebook_combined.txt \
  --merge-mode batch-ea \
  --scoring-backend cpu \
  --top-k 4 \
  --ea-use-threshold \
  --threshold-policy adaptive \
  --threshold-high 0.5 \
  --threshold-low 0.005 \
  --threshold-min-low 0.005 \
  --adaptive-q-high 0.85 \
  --adaptive-q-low 0.15 \
  --adaptive-sample-limit 4096 \
  --acceptance-target 0.15 \
  --iterations 20 \
  --seed 1 \
  --error-bound 0.0 \
  --results-csv results/beam_batch_ea_adaptive.csv
```

如果你要跑 `batch-ea-blocked`：

```bash
OMP_NUM_THREADS=40 ./build/beam \
  --input data/facebook/facebook_combined.txt \
  --merge-mode batch-ea-blocked \
  --scoring-backend cpu \
  --top-k 4 \
  --group-batch-size 64 \
  --ea-use-threshold \
  --threshold-policy adaptive \
  --iterations 20 \
  --seed 1 \
  --error-bound 0.0 \
  --results-csv results/beam_batch_ea_blocked_adaptive.csv
```

如果需要写出 `G/P/Cp/Cm` 文件：

```bash
OMP_NUM_THREADS=40 ./build/beam \
  --input data/facebook/facebook_combined.txt \
  --merge-mode batch-ea \
  --scoring-backend cpu \
  --top-k 4 \
  --ea-use-threshold \
  --threshold-policy reciprocal \
  --iterations 20 \
  --seed 1 \
  --divide-hash-dims 16 \
  --divide-max-group 512 \
  --error-bound 0.0 \
  --write-output \
  --out out/facebook \
  --results-csv results/sweg_batch_ea.csv
```

最小运行示例：

```bash
./beam --input <path> --seed 1
```

这个命令能运行，但它不会显式锁定你真正关心的 merge 模式、backend 和 top-k。
如果是做 `batch-ea` 实验，不建议只用这一条最小命令。

带 divide 参数的示例：

```bash
./beam --input <path> --seed 1 --divide-hash-dims 16 --divide-max-group 512
```

如果传了 `--write-output` 但没有传 `--out`，程序会直接报错。

## 计时字段

程序输出以下时间字段：

- `runtime_run_ms`
  `Sweg::Run()` 的总时间，也就是 `divide + merge`。

- `runtime_divide_ms`
  `Divide()` 累计时间。

- `runtime_merge_ms`
  `Merge()` 累计时间。

- `divide_max_group_size`
  最终 divide groups 的最大大小。启用 hard cap 后应不超过 `--divide-max-group`。

- `divide_fallback_splits`
  Adaptive Multi-Hash Divide 在 hash 维度耗尽后触发 deterministic fallback split 的次数。

- `runtime_encode_ms`
  `Encode()` 时间。

- `runtime_output_ms`
  `WriteOutput()` 时间。默认不写输出时为 `0`。

- `runtime_algorithm_ms`
  `runtime_run_ms + runtime_encode_ms`。
  正式与 MAGS 对比时使用这个字段。

- `runtime_end_to_end_ms`
  `runtime_algorithm_ms + runtime_output_ms`。
  只有显式写输出时才会大于 `runtime_algorithm_ms`。

默认不写输出文件，因此：

- `runtime_output_ms = 0`
- 正式实验时间使用 `runtime_algorithm_ms`

如果显式开启 `--write-output`：

- 会生成 `G.txt / P.txt / Cp.txt / Cm.txt`
- `runtime_end_to_end_ms = runtime_algorithm_ms + runtime_output_ms`

如果你修改了 `results.csv` 表头，旧 CSV 文件会因为 schema 不匹配而报错。
建议删除旧 CSV，或使用新的文件名。

## threshold 相关指标

现在的 metrics / CSV 中额外包含以下 threshold 字段：

- `threshold_policy`
  当前实验使用的 threshold policy 名称。

- `threshold_last`
  最后一轮实际使用的阈值。

- `threshold_geom_last`
  最后一轮几何阈值。

- `threshold_adaptive_last`
  最后一轮 adaptive 分位数阈值。
  对 `reciprocal` / `mags-geom`，这个值会退化成对应主阈值。

- `threshold_acceptance_scale`
  `adaptive` acceptance controller 在最后一轮结束后的缩放因子。

- `threshold_acceptance_rate_last`
  最后一轮记录到的 acceptance rate。

- `threshold_sample_count_last`
  最后一轮结束后保留下来的 sampled positive saving ratio 个数。

## 批量运行

正式实验如果不需要写 `G/P/Cp/Cm`，不要传 `--write-output`，也不需要 `--out`：

```bash
THREADS=40
SWEG=./build/beam
DATA=/data/zby/datasets

MODE=batch-ea
BACKEND=cpu
TOPK=4
ITERS=20
SEED=1
ERROR_BOUND=0.0

LOG=./logs_sweg_${MODE}_th${THREADS}
CSV=./results/sweg_${MODE}_th${THREADS}.csv

mkdir -p "$LOG" "$(dirname "$CSV")"

export OMP_NUM_THREADS=$THREADS
export OMP_PROC_BIND=close
export OMP_PLACES=cores

for item in \
  "facebook|$DATA/facebook/facebook.undir.single.txt" \
  "caida|$DATA/caida/caida.undir.single.txt" \
  "email_enron|$DATA/email_enron/email_enron.undir.single.txt" \
  "amazon|$DATA/amazon/amazon.undir.single.txt" \
  "dblp|$DATA/dblp/dblp.undir.single.txt"
do
  IFS="|" read -r name input <<< "$item"

  log_file="$LOG/${name}.log"

  echo "===== ${name} start $(date) =====" | tee "$log_file"
  echo "input=${input}" | tee -a "$log_file"
  echo "csv=${CSV}" | tee -a "$log_file"

  /usr/bin/time -v "$SWEG" \
    --input "$input" \
    --merge-mode "$MODE" \
    --scoring-backend "$BACKEND" \
    --top-k "$TOPK" \
    --ea-use-threshold \
    --threshold-policy reciprocal \
    --iterations "$ITERS" \
    --seed "$SEED" \
    --error-bound "$ERROR_BOUND" \
    --results-csv "$CSV" \
    2>&1 | tee -a "$log_file"

  status=${PIPESTATUS[0]}

  echo "exit_status=${status}" | tee -a "$log_file"
  echo "===== ${name} end $(date) =====" | tee -a "$log_file"
  echo ""
done
```



## 指标口径

对于输入无向单边 edge list：

- `metric.input_edges_raw = m`
- `metric.graph_arcs = 2m`
- `metric.original_undirected_edges = m`
- `metric.original_edges_eval = m`

压缩率口径保持不变：

- `encoding_cost = |P| + |Cp| + |Cm|`
- `cost_ratio = encoding_cost / original_edges_eval`
- `compression_gain = 1 - cost_ratio`

因此在当前唯一输入格式下：

- `relative_size == cost_ratio`

## 测试脚本

仓库内的小图测试脚本会显式传 `--write-output --out ...`，因为 reconstruction 需要读取 `G/P/Cp/Cm`：

```bash
python3 cpp/tools/run_small_tests.py \
  --binary cpp/build/beam
```
