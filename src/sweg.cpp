#include "sweg.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

uint64_t PackEdge(int u, int v) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(u)) << 32U) |
         static_cast<uint32_t>(v);
}

bool EdgePairLess(const EdgePair& a, const EdgePair& b) {
  if (a.first != b.first) {
    return a.first < b.first;
  }
  return a.second < b.second;
}

double ElapsedMs(std::chrono::steady_clock::time_point start,
                 std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

double SafeRatio(uint64_t numerator, uint64_t denominator) {
  return denominator > 0
             ? static_cast<double>(numerator) /
                   static_cast<double>(denominator)
             : 0.0;
}

size_t AlignUp(size_t value, size_t alignment) {
  if (alignment <= 1) {
    return value;
  }
  const size_t remainder = value % alignment;
  return remainder == 0 ? value : (value + alignment - remainder);
}

}  // namespace

const char* MergeModeToString(MergeMode mode) {
  switch (mode) {
    case MergeMode::kBatchEncodingAware:
      return "batch-ea";
    case MergeMode::kBatchEncodingAwareBlocked:
      return "batch-ea-blocked";
  }
  return "unknown";
}

const char* ScoringBackendToString(ScoringBackend backend) {
  switch (backend) {
    case ScoringBackend::kCpu:
      return "cpu";
    case ScoringBackend::kCuda:
      return "cuda";
  }
  return "unknown";
}

const char* ThresholdPolicyToString(ThresholdPolicy policy) {
  switch (policy) {
    case ThresholdPolicy::kReciprocal:
      return "reciprocal";
    case ThresholdPolicy::kMagsGeom:
      return "mags-geom";
    case ThresholdPolicy::kAdaptive:
      return "adaptive";
  }
  return "unknown";
}

Sweg::Sweg(const CSRGraph& graph, MergeMode merge_mode, int top_k,
           bool ea_use_threshold, uint64_t seed,
           ScoringBackend scoring_backend, bool verify_cuda_gain,
           int group_batch_size, int candidate_batch_budget,
           int cuda_slice_memory_mb,
           int overflow_group_gmax, int overflow_refine_rounds,
           int divide_hash_dims, int divide_max_group,
           const ThresholdConfig& threshold_config)
    : graph_(&graph),
      n_(graph.n),
      merge_mode_(merge_mode),
      top_k_(top_k),
      ea_use_threshold_(ea_use_threshold),
      scoring_backend_(scoring_backend),
      verify_cuda_gain_(verify_cuda_gain),
      group_batch_size_(std::max(1, group_batch_size)),
      candidate_batch_budget_(std::max(0, candidate_batch_budget)),
      cuda_slice_memory_mb_(std::max(0, cuda_slice_memory_mb)),
      overflow_group_gmax_(std::max(0, overflow_group_gmax)),
      overflow_refine_rounds_(std::max(0, overflow_refine_rounds)),
      divide_hash_dims_(divide_hash_dims > 0 ? divide_hash_dims : 16),
      divide_max_group_(divide_max_group > 0 ? divide_max_group : 512),
      threshold_config_(threshold_config),
      seed64_(seed) {
  h_.resize(n_);
  S_.resize(n_);
  I_.resize(n_);
  J_.resize(n_);
  supernode_sizes_by_rep_.assign(n_, 0);

  for (int i = 0; i < n_; ++i) {
    h_[i] = i;
    S_[i] = i;
    I_[i] = i;
    J_[i] = -1;
    supernode_sizes_by_rep_[i] = 1;
  }

  F_.assign(n_, -1);
  G_.resize(n_);
  active_reps_.reserve(n_);
  groups_.reserve(n_);
  dim_order_.reserve(static_cast<size_t>(divide_hash_dims_));
  rng_.seed(static_cast<uint32_t>(seed));
  cuda_verify_samples_remaining_ = verify_cuda_gain_ ? static_cast<size_t>(4096) : 0;
  serial_workspace_.counts.assign(static_cast<size_t>(n_), 0);
  serial_workspace_.marks.assign(static_cast<size_t>(n_), 0);
  serial_workspace_.touched.reserve(static_cast<size_t>(std::min(n_, 1 << 20)));
  serial_workspace_.epoch = 1;
  EnsurePrepareWorkspace(&sequential_prepare_workspace_);
  scratch_counts_.assign(static_cast<size_t>(n_), 0);
  scratch_marks_.assign(static_cast<size_t>(n_), 0);
  scratch_touched_.reserve(static_cast<size_t>(std::min(n_, 1 << 20)));
}

void Sweg::ResetRuntimeStats() {
  stats_ = RuntimeStats{};
  stats_.threshold_acceptance_scale = threshold_acceptance_scale_;
}

double Sweg::GeometricThreshold(int iter, int total_iterations) const {
  const double high = threshold_config_.high;
  const double low = threshold_config_.low;
  if (total_iterations <= 1) {
    return high;
  }
  const double alpha = static_cast<double>(iter - 1) /
                       static_cast<double>(total_iterations - 1);
  return high * std::pow(low / high, alpha);
}

double Sweg::AdaptiveQuantileForIter(int iter, int total_iterations) const {
  if (total_iterations <= 1) {
    return threshold_config_.q_high;
  }
  const double alpha = static_cast<double>(iter - 1) /
                       static_cast<double>(total_iterations - 1);
  return threshold_config_.q_high +
         (threshold_config_.q_low - threshold_config_.q_high) * alpha;
}

double Sweg::Quantile(std::vector<double> values, double q) const {
  if (values.empty()) {
    return 0.0;
  }
  q = std::clamp(q, 0.0, 1.0);
  const size_t idx = static_cast<size_t>(
      std::floor(q * static_cast<double>(values.size() - 1)));
  std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(idx),
                   values.end());
  return values[idx];
}

double Sweg::ComputeMergeThreshold(int iter, int total_iterations) {
  const bool use_configurable_policy =
      merge_mode_ == MergeMode::kBatchEncodingAware ||
      merge_mode_ == MergeMode::kBatchEncodingAwareBlocked;
  if (!use_configurable_policy) {
    stats_.threshold_geom_last = 1.0 / static_cast<double>(iter + 1);
    stats_.threshold_adaptive_last = stats_.threshold_geom_last;
    stats_.threshold_last = stats_.threshold_geom_last;
    stats_.threshold_acceptance_scale = threshold_acceptance_scale_;
    return stats_.threshold_last;
  }

  if (threshold_config_.policy == ThresholdPolicy::kReciprocal) {
    stats_.threshold_geom_last = 1.0 / static_cast<double>(iter + 1);
    stats_.threshold_adaptive_last = stats_.threshold_geom_last;
    stats_.threshold_last = stats_.threshold_geom_last;
    stats_.threshold_acceptance_scale = threshold_acceptance_scale_;
    return stats_.threshold_last;
  }

  const double geom = GeometricThreshold(iter, total_iterations);
  double adaptive = geom;
  if (threshold_config_.policy == ThresholdPolicy::kAdaptive &&
      !prev_positive_savings_.empty()) {
    adaptive = Quantile(prev_positive_savings_,
                        AdaptiveQuantileForIter(iter, total_iterations));
  }

  double threshold = geom;
  if (threshold_config_.policy == ThresholdPolicy::kAdaptive) {
    threshold = std::min(geom, adaptive * threshold_acceptance_scale_);
  }

  threshold = std::clamp(threshold, threshold_config_.min_low,
                         threshold_config_.high);
  stats_.threshold_geom_last = geom;
  stats_.threshold_adaptive_last = adaptive;
  stats_.threshold_last = threshold;
  stats_.threshold_acceptance_scale = threshold_acceptance_scale_;
  return threshold;
}

void Sweg::BeginMergeIteration() {
  cur_positive_savings_.clear();
  cur_positive_savings_seen_ = 0;
  cur_tested_pairs_ = 0;
  cur_accepted_pairs_ = 0;
  stats_.threshold_sample_count_last = prev_positive_savings_.size();
  stats_.threshold_acceptance_rate_last = 0.0;
}

void Sweg::EndMergeIteration() {
  if (threshold_config_.policy == ThresholdPolicy::kAdaptive &&
      cur_tested_pairs_ > 0) {
    stats_.threshold_acceptance_rate_last =
        static_cast<double>(cur_accepted_pairs_) /
        static_cast<double>(cur_tested_pairs_);
    if (stats_.threshold_acceptance_rate_last <
        threshold_config_.acceptance_target * 0.5) {
      threshold_acceptance_scale_ *= threshold_config_.acceptance_down;
    } else if (stats_.threshold_acceptance_rate_last >
               threshold_config_.acceptance_target * 2.0) {
      threshold_acceptance_scale_ *= threshold_config_.acceptance_up;
    }
    threshold_acceptance_scale_ =
        std::clamp(threshold_acceptance_scale_,
                   threshold_config_.acceptance_scale_min,
                   threshold_config_.acceptance_scale_max);
  } else {
    stats_.threshold_acceptance_rate_last = 0.0;
  }

  prev_positive_savings_.swap(cur_positive_savings_);
  cur_positive_savings_.clear();
  stats_.threshold_sample_count_last = prev_positive_savings_.size();
  stats_.threshold_acceptance_scale = threshold_acceptance_scale_;
}

void Sweg::RecordSavingSample(double saving_ratio) {
  if (threshold_config_.policy != ThresholdPolicy::kAdaptive) {
    return;
  }
  if (!(saving_ratio > 0.0)) {
    return;
  }

  ++cur_positive_savings_seen_;
  const size_t limit = static_cast<size_t>(
      std::max(1, threshold_config_.sample_limit));
  if (cur_positive_savings_.size() < limit) {
    cur_positive_savings_.push_back(saving_ratio);
    return;
  }

  std::uniform_int_distribution<uint64_t> dist(0, cur_positive_savings_seen_ - 1);
  const uint64_t pos = dist(rng_);
  if (pos < limit) {
    cur_positive_savings_[static_cast<size_t>(pos)] = saving_ratio;
  }
}

void Sweg::ResetScratch() const {
  ++scratch_epoch_;
  if (scratch_epoch_ == 0) {
    std::fill(scratch_marks_.begin(), scratch_marks_.end(), 0);
    scratch_epoch_ = 1;
  }
  scratch_touched_.clear();
}

void Sweg::ResetParallelScratch(ParallelScratch& scratch) const {
  ++scratch.epoch;
  if (scratch.epoch == 0) {
    std::fill(scratch.marks.begin(), scratch.marks.end(), 0);
    scratch.epoch = 1;
  }
  scratch.touched.clear();
}

void Sweg::EnsurePrepareWorkspace(PrepareWorkspace* workspace) const {
  if (workspace == nullptr) {
    return;
  }
  if (workspace->scratch.counts.size() == static_cast<size_t>(n_)) {
    return;
  }
  workspace->scratch.counts.assign(static_cast<size_t>(n_), 0);
  workspace->scratch.marks.assign(static_cast<size_t>(n_), 0);
  workspace->scratch.touched.reserve(static_cast<size_t>(std::min(n_, 1 << 20)));
  workspace->scratch.epoch = 1;
}

void Sweg::EnsurePrepareWorkspaces(int thread_count) const {
  if (thread_count <= 0) {
    thread_count = 1;
  }
  if (static_cast<int>(prepare_workspaces_.size()) >= thread_count) {
    return;
  }
  const size_t old_size = prepare_workspaces_.size();
  prepare_workspaces_.resize(static_cast<size_t>(thread_count));
  for (size_t i = old_size; i < prepare_workspaces_.size(); ++i) {
    EnsurePrepareWorkspace(&prepare_workspaces_[i]);
  }
}

void Sweg::AccumulatePrepareStats(const EaPrepareStats& prepare_stats) {
  stats_.merge_prepare_task_sum_ms += prepare_stats.wall_ms;
  stats_.merge_create_w_ms += prepare_stats.create_w_ms;
  stats_.merge_candidate_gen_ms += prepare_stats.candidate_gen_ms;
  stats_.merge_scoring_ms += prepare_stats.candidate_gen_ms;
  stats_.merge_group_count += prepare_stats.group_count;
  stats_.merge_group_max_size =
      std::max<uint64_t>(stats_.merge_group_max_size,
                         prepare_stats.group_max_size);
  stats_.merge_raw_pair_count += prepare_stats.raw_pair_count;
  stats_.merge_candidate_pairs_after_prune +=
      prepare_stats.candidate_pairs_after_prune;
}

void Sweg::Run(int iterations, int print_offset) {
  ResetRuntimeStats();
  cuda_verify_samples_remaining_ =
      verify_cuda_gain_ ? static_cast<size_t>(4096) : 0;
  const auto total_start = std::chrono::steady_clock::now();
  for (int iter = 1; iter <= iterations; ++iter) {
    const auto divide_start = std::chrono::steady_clock::now();
    Divide(iter);
    const auto divide_end = std::chrono::steady_clock::now();
    stats_.runtime_divide_ms += ElapsedMs(divide_start, divide_end);

    const auto merge_start = std::chrono::steady_clock::now();
    Merge(iter, iterations);
    const auto merge_end = std::chrono::steady_clock::now();
    stats_.runtime_merge_ms += ElapsedMs(merge_start, merge_end);

    if (print_offset > 0 &&
        (iter % print_offset == 0 || iter == iterations)) {
      std::cout << "Iter " << iter << ": groups=" << CountGroups()
                << " active_supernodes=" << CountActiveSupernodes() << "\n";
    }
  }
  const auto total_end = std::chrono::steady_clock::now();
  stats_.runtime_run_ms = ElapsedMs(total_start, total_end);
  stats_.merge_exact_gain_calls_per_selected =
      SafeRatio(stats_.merge_exact_gain_calls,
                std::max<uint64_t>(1, stats_.merge_selected_pairs));
  stats_.merge_positive_gain_ratio =
      SafeRatio(stats_.merge_positive_gain_pairs,
                std::max<uint64_t>(1, stats_.merge_exact_gain_calls));
}

void Sweg::ShuffleArray() {
  for (int i = n_ - 1; i > 0; --i) {
    std::uniform_int_distribution<int> dist(0, i);
    int index = dist(rng_);
    std::swap(h_[index], h_[i]);
  }
}

int Sweg::NodeShingle(int v) const {
  int fv = h_[v];
  auto neighbors = graph_->neighbors(v);
  for (const int* it = neighbors.first; it != neighbors.second; ++it) {
    const int u = *it;
    if (fv > h_[u]) {
      fv = h_[u];
    }
  }
  return fv;
}

uint64_t Sweg::HashRank64(int node, int dim, int iter) const {
  uint64_t x = seed64_;
  x ^= 0x9e3779b97f4a7c15ULL * static_cast<uint64_t>(node + 1);
  x ^= 0xbf58476d1ce4e5b9ULL * static_cast<uint64_t>(dim + 1);
  x ^= 0x94d049bb133111ebULL * static_cast<uint64_t>(iter + 1);
  return SplitMix64(x);
}

uint64_t Sweg::DimRank(int dim, int iter) const {
  uint64_t x = seed64_;
  x ^= 0x9e3779b97f4a7c15ULL * static_cast<uint64_t>(dim + 1);
  x ^= 0xbf58476d1ce4e5b9ULL * static_cast<uint64_t>(iter + 1);
  return SplitMix64(x);
}

uint64_t Sweg::FallbackRank(int rep, int iter) const {
  uint64_t x = seed64_;
  x ^= 0x9e3779b97f4a7c15ULL * static_cast<uint64_t>(rep + 1);
  x ^= 0xbf58476d1ce4e5b9ULL * static_cast<uint64_t>(iter + 1);
  x ^= 0xd6e8feb86659fd93ULL;
  return SplitMix64(x);
}

uint64_t Sweg::NodeShingleDim(int v, int dim, int iter) const {
  uint64_t best_rank = HashRank64(v, dim, iter);
  auto neighbors = graph_->neighbors(v);
  for (const int* it = neighbors.first; it != neighbors.second; ++it) {
    const uint64_t rank = HashRank64(*it, dim, iter);
    if (rank < best_rank) {
      best_rank = rank;
    }
  }
  return best_rank;
}

uint64_t Sweg::SupernodeMinHash(int rep, int dim, int iter) {
  const uint64_t key =
      (static_cast<uint64_t>(static_cast<uint32_t>(rep)) << 32U) |
      static_cast<uint32_t>(dim);
  auto it = divide_minhash_cache_.find(key);
  if (it != divide_minhash_cache_.end()) {
    return it->second;
  }

  uint64_t best = std::numeric_limits<uint64_t>::max();
  for (int node = I_[rep]; node != -1; node = J_[node]) {
    const uint64_t shingle = NodeShingleDim(node, dim, iter);
    if (shingle < best) {
      best = shingle;
    }
  }
  divide_minhash_cache_.emplace(key, best);
  return best;
}

void Sweg::FallbackSplitGroup(std::vector<int>& reps, int l, int r, int iter) {
  if (l >= r) {
    return;
  }
  stats_.divide_fallback_splits += 1;

  std::stable_sort(reps.begin() + l, reps.begin() + r,
                   [this, iter](int a, int b) {
                     const uint64_t ha = FallbackRank(a, iter);
                     const uint64_t hb = FallbackRank(b, iter);
                     if (ha != hb) {
                       return ha < hb;
                     }
                     return a < b;
                   });

  for (int start = l; start < r; start += divide_max_group_) {
    const int end = std::min(start + divide_max_group_, r);
    if (end > start) {
      groups_.push_back(GroupSpan{start, end - start});
      stats_.divide_max_group_size =
          std::max<int64_t>(stats_.divide_max_group_size, end - start);
    }
  }
}

void Sweg::SplitGroupByHashDims(std::vector<int>& reps, int l, int r, int depth,
                                int iter) {
  if (l >= r) {
    return;
  }
  const int size = r - l;
  if (size <= divide_max_group_) {
    groups_.push_back(GroupSpan{l, size});
    stats_.divide_max_group_size =
        std::max<int64_t>(stats_.divide_max_group_size, size);
    return;
  }
  if (depth >= divide_hash_dims_) {
    FallbackSplitGroup(reps, l, r, iter);
    return;
  }

  const int dim = dim_order_[static_cast<size_t>(depth)];
  std::stable_sort(reps.begin() + l, reps.begin() + r,
                   [this, dim, iter](int a, int b) {
                     const uint64_t ha = SupernodeMinHash(a, dim, iter);
                     const uint64_t hb = SupernodeMinHash(b, dim, iter);
                     if (ha != hb) {
                       return ha < hb;
                     }
                     return a < b;
                   });

  int begin = l;
  while (begin < r) {
    const uint64_t bucket_hash = SupernodeMinHash(reps[begin], dim, iter);
    int end = begin + 1;
    while (end < r &&
           SupernodeMinHash(reps[end], dim, iter) == bucket_hash) {
      ++end;
    }
    SplitGroupByHashDims(reps, begin, end, depth + 1, iter);
    begin = end;
  }
}

void Sweg::Divide(int iter) {
  active_reps_.clear();
  groups_.clear();
  dim_order_.resize(static_cast<size_t>(divide_hash_dims_));
  divide_minhash_cache_.clear();
  std::fill(F_.begin(), F_.end(), -1);
  std::fill(G_.begin(), G_.end(), -1);

  for (int rep = 0; rep < n_; ++rep) {
    if (I_[rep] != -1) {
      active_reps_.push_back(rep);
    }
  }

  std::iota(dim_order_.begin(), dim_order_.end(), 0);
  std::stable_sort(dim_order_.begin(), dim_order_.end(),
                   [this, iter](int a, int b) {
                     const uint64_t ra = DimRank(a, iter);
                     const uint64_t rb = DimRank(b, iter);
                     if (ra != rb) {
                       return ra < rb;
                     }
                     return a < b;
                   });

  SplitGroupByHashDims(active_reps_, 0, static_cast<int>(active_reps_.size()),
                       0, iter);

  for (const GroupSpan& group : groups_) {
    for (int offset = 0; offset < group.length; ++offset) {
      const int rep = active_reps_[static_cast<size_t>(group.start + offset)];
      F_[rep] = group.start;
      G_[static_cast<size_t>(group.start + offset)] = rep;
    }
  }

  gstart_ = active_reps_.empty() ? n_ : 0;
}

std::vector<Sweg::GroupSpan> Sweg::BuildGroups() const {
  return groups_;
}

Sweg::SparseCounts Sweg::CreateWForSupernode(int rep,
                                             int64_t* self_loop_count) const {
  return CreateWForSupernodeWithScratch(rep, serial_workspace_,
                                        self_loop_count);
}

Sweg::SparseCounts Sweg::CreateWForSupernodeWithScratch(
    int rep, ParallelScratch& scratch, int64_t* self_loop_count) const {
  ResetParallelScratch(scratch);
  int64_t loop_count = 0;

  for (int node = I_[rep]; node != -1; node = J_[node]) {
    auto neighbors = graph_->neighbors(node);
    for (const int* it = neighbors.first; it != neighbors.second; ++it) {
      const int neighbor = *it;
      if (scratch.marks[static_cast<size_t>(neighbor)] != scratch.epoch) {
        scratch.marks[static_cast<size_t>(neighbor)] = scratch.epoch;
        scratch.counts[static_cast<size_t>(neighbor)] = 0;
        scratch.touched.push_back(neighbor);
      }
      ++scratch.counts[static_cast<size_t>(neighbor)];
      if (*it == node) {
        ++loop_count;
      }
    }
  }

  SparseCounts w;
  w.reserve(scratch.touched.size());
  for (int neighbor : scratch.touched) {
    w.emplace_back(neighbor, scratch.counts[static_cast<size_t>(neighbor)]);
  }
  std::sort(w.begin(), w.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
  if (self_loop_count != nullptr) {
    *self_loop_count = loop_count;
  }
  return w;
}

void Sweg::Merge(int iter, int total_iterations) {
  BeginMergeIteration();
  const auto groups = BuildGroups();
  const double threshold = ComputeMergeThreshold(iter, total_iterations);
  OverflowRefineCounters overflow_counters;

  auto build_refined_group_list = [&](const GroupSpan& group) {
    std::vector<std::vector<int>> refined_groups;
    if (group.length < 2) {
      return refined_groups;
    }
    std::vector<int> q;
    q.reserve(static_cast<size_t>(group.length));
    for (int j = 0; j < group.length; ++j) {
      q.push_back(G_[group.start + j]);
    }
    if (overflow_group_gmax_ > 0) {
      return RefineOverflowGroup(q, &overflow_counters);
    }
    refined_groups.push_back(std::move(q));
    return refined_groups;
  };

  if (merge_mode_ == MergeMode::kBatchEncodingAwareBlocked) {
    const int block_size = std::max(1, group_batch_size_);
    const size_t candidate_chunk_size =
        candidate_batch_budget_ > 0
            ? static_cast<size_t>(candidate_batch_budget_)
            : 0;
    for (size_t block_start = 0; block_start < groups.size();
         block_start += static_cast<size_t>(block_size)) {
      const size_t block_end =
          std::min(groups.size(), block_start + static_cast<size_t>(block_size));
      std::vector<std::pair<int, int>> block_selected_pairs;
      std::vector<std::vector<int>> work_items;
      work_items.reserve((block_end - block_start) * 2);
      for (size_t gi = block_start; gi < block_end; ++gi) {
        const GroupSpan& group = groups[gi];
        const auto refined_groups = build_refined_group_list(group);
        for (const auto& q : refined_groups) {
          work_items.push_back(q);
        }
      }

      std::vector<EaGroupPrepared> prepared_groups(work_items.size());
      std::vector<EaPrepareStats> prepare_stats(work_items.size());
      if (!work_items.empty()) {
        const auto prepare_start = std::chrono::steady_clock::now();
#ifdef _OPENMP
        const int thread_count = std::max(1, omp_get_max_threads());
        EnsurePrepareWorkspaces(thread_count);
#pragma omp parallel for schedule(dynamic, 1) if(work_items.size() > 1)
        for (int i = 0; i < static_cast<int>(work_items.size()); ++i) {
          PrepareWorkspace* workspace = nullptr;
#ifdef _OPENMP
          workspace = &prepare_workspaces_[static_cast<size_t>(omp_get_thread_num())];
#else
          workspace = &prepare_workspaces_[0];
#endif
          EaPrepareResult result = PrepareBatchEncodingAwareGroupWithWorkspace(
              work_items[static_cast<size_t>(i)], workspace, false);
          prepared_groups[static_cast<size_t>(i)] = std::move(result.prepared);
          prepare_stats[static_cast<size_t>(i)] = result.stats;
        }
#else
        EnsurePrepareWorkspaces(1);
        for (size_t i = 0; i < work_items.size(); ++i) {
          EaPrepareResult result = PrepareBatchEncodingAwareGroupWithWorkspace(
              work_items[i], &prepare_workspaces_[0], false);
          prepared_groups[i] = std::move(result.prepared);
          prepare_stats[i] = result.stats;
        }
#endif
        stats_.merge_prepare_wall_ms +=
            ElapsedMs(prepare_start, std::chrono::steady_clock::now());
      }

      for (size_t i = 0; i < prepare_stats.size(); ++i) {
        AccumulatePrepareStats(prepare_stats[i]);
      }

      std::vector<CudaSliceSpan> slice_plan;
      if (scoring_backend_ == ScoringBackend::kCuda) {
        slice_plan = BuildCudaSlicePlan(prepared_groups, candidate_chunk_size);
        stats_.cuda_slice_count += static_cast<int64_t>(slice_plan.size());
        if (slice_plan.size() <= 1) {
          stats_.cuda_blocks_single_slice += 1;
        } else {
          stats_.cuda_blocks_multi_slice += 1;
        }
        for (const CudaSliceSpan& slice : slice_plan) {
          stats_.cuda_max_slice_rows =
              std::max<int64_t>(stats_.cuda_max_slice_rows,
                                static_cast<int64_t>(slice.rows));
          stats_.cuda_max_slice_nnz =
              std::max<int64_t>(stats_.cuda_max_slice_nnz,
                                static_cast<int64_t>(slice.nnz));
          stats_.cuda_max_slice_candidates =
              std::max<int64_t>(stats_.cuda_max_slice_candidates,
                                static_cast<int64_t>(slice.candidates));
        }
      }

      if (scoring_backend_ == ScoringBackend::kCuda) {
        for (const CudaSliceSpan& slice : slice_plan) {
          ScoreBatchEncodingAwarePreparedGroupsBlockedRange(
              &prepared_groups, slice.begin, slice.end, threshold);
        }
      } else {
        for (EaGroupPrepared& prepared : prepared_groups) {
          ScoreBatchEncodingAwarePreparedGroup(&prepared, threshold);
        }
      }

      for (const EaGroupPrepared& prepared : prepared_groups) {
        std::vector<std::pair<int, int>> group_pairs =
            SelectScoredBatchEncodingAwareGroupPairs(prepared);
        block_selected_pairs.insert(block_selected_pairs.end(),
                                    group_pairs.begin(), group_pairs.end());
      }

      const auto update_start = std::chrono::steady_clock::now();
      for (const auto& pair : block_selected_pairs) {
        const int rep_a = pair.first;
        const int rep_b = pair.second;
        if (I_[rep_a] == -1 || I_[rep_b] == -1) {
          continue;
        }
        Update_S(rep_a, rep_b);
      }
      const auto update_end = std::chrono::steady_clock::now();
      stats_.merge_update_ms += ElapsedMs(update_start, update_end);
    }
    if (overflow_counters.overflow_groups_seen > 0) {
      std::cout << "  Overflow refinement: groups="
                << overflow_counters.overflow_groups_seen
                << " refined_subgroups="
                << overflow_counters.refined_subgroups
                << " forced_chunks=" << overflow_counters.forced_chunks
                << " max_before=" << overflow_counters.max_group_before
                << " max_after=" << overflow_counters.max_group_after << "\n";
      stats_.overflow_groups_seen += overflow_counters.overflow_groups_seen;
      stats_.overflow_refined_subgroups +=
          overflow_counters.refined_subgroups;
      stats_.overflow_forced_chunks += overflow_counters.forced_chunks;
      stats_.overflow_max_group_before =
          std::max(stats_.overflow_max_group_before,
                   overflow_counters.max_group_before);
      stats_.overflow_max_group_after =
          std::max(stats_.overflow_max_group_after,
                   overflow_counters.max_group_after);
    }
    if (stats_.cuda_slice_count > 0 && scoring_backend_ == ScoringBackend::kCuda) {
      std::cout << "  Blocked scoring slices: count=" << stats_.cuda_slice_count
                << " max_candidates=" << stats_.cuda_max_slice_candidates
                << " max_rows=" << stats_.cuda_max_slice_rows
                << " max_nnz=" << stats_.cuda_max_slice_nnz << "\n";
    }
    EndMergeIteration();
    return;
  }

  for (const GroupSpan& group : groups) {
    const auto refined_groups = build_refined_group_list(group);
    for (const auto& q : refined_groups) {
      switch (merge_mode_) {
        case MergeMode::kBatchEncodingAware:
          MergeBatchEncodingAwareGroup(q, threshold);
          break;
        case MergeMode::kBatchEncodingAwareBlocked:
          break;
      }
    }
  }

  if (overflow_counters.overflow_groups_seen > 0) {
    std::cout << "  Overflow refinement: groups="
              << overflow_counters.overflow_groups_seen
              << " refined_subgroups="
              << overflow_counters.refined_subgroups
              << " forced_chunks=" << overflow_counters.forced_chunks
              << " max_before=" << overflow_counters.max_group_before
              << " max_after=" << overflow_counters.max_group_after << "\n";
    stats_.overflow_groups_seen += overflow_counters.overflow_groups_seen;
    stats_.overflow_refined_subgroups += overflow_counters.refined_subgroups;
    stats_.overflow_forced_chunks += overflow_counters.forced_chunks;
    stats_.overflow_max_group_before =
        std::max(stats_.overflow_max_group_before,
                 overflow_counters.max_group_before);
    stats_.overflow_max_group_after =
        std::max(stats_.overflow_max_group_after,
                 overflow_counters.max_group_after);
  }
  EndMergeIteration();
}

Sweg::SparseCounts Sweg::AggregateWBySupernode(const SparseCounts& w) const {
  return AggregateWBySupernodeWithScratch(w, serial_workspace_);
}

Sweg::SparseCounts Sweg::AggregateWBySupernodeWithScratch(
    const SparseCounts& w, ParallelScratch& scratch) const {
  ResetParallelScratch(scratch);
  for (const auto& kv : w) {
    const int rep = S_[kv.first];
    if (scratch.marks[static_cast<size_t>(rep)] != scratch.epoch) {
      scratch.marks[static_cast<size_t>(rep)] = scratch.epoch;
      scratch.counts[static_cast<size_t>(rep)] = 0;
      scratch.touched.push_back(rep);
    }
    scratch.counts[static_cast<size_t>(rep)] += kv.second;
  }
  SparseCounts aggregated;
  aggregated.reserve(scratch.touched.size());
  for (int rep : scratch.touched) {
    aggregated.emplace_back(rep, scratch.counts[static_cast<size_t>(rep)]);
  }
  std::sort(aggregated.begin(), aggregated.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
  return aggregated;
}

size_t Sweg::EstimateCudaSliceBytes(size_t total_rows, size_t total_nnz,
                                    size_t total_candidates,
                                    size_t candidate_chunk_size) const {
  using PackedPair = std::pair<int, int>;
  const size_t offsets_bytes =
      AlignUp(sizeof(int) * (total_rows + 1), alignof(int));
  const size_t row_rep_ids_bytes =
      AlignUp(sizeof(int) * total_rows, alignof(int));
  const size_t neighbors_bytes =
      AlignUp(sizeof(int) * total_nnz, alignof(int));
  const size_t weights_bytes =
      AlignUp(sizeof(int64_t) * total_nnz, alignof(int64_t));
  const size_t neighbor_sizes_bytes =
      AlignUp(sizeof(int64_t) * total_nnz, alignof(int64_t));
  const size_t row_sizes_bytes =
      AlignUp(sizeof(int64_t) * total_rows, alignof(int64_t));
  const size_t row_self_loops_bytes =
      AlignUp(sizeof(int64_t) * total_rows, alignof(int64_t));
  const size_t pairs_bytes =
      AlignUp(sizeof(PackedPair) * total_candidates, alignof(PackedPair));
  const size_t output_bytes = AlignUp(
      sizeof(LocalGainResult) * total_candidates, alignof(LocalGainResult));
  const size_t max_chunk_candidates =
      candidate_chunk_size == 0
          ? total_candidates
          : std::max<size_t>(1, std::min(candidate_chunk_size, total_candidates));
  const size_t chunk_pair_bytes =
      AlignUp(sizeof(PackedPair) * max_chunk_candidates, alignof(PackedPair));
  const size_t chunk_result_bytes = AlignUp(
      sizeof(LocalGainResult) * max_chunk_candidates, alignof(LocalGainResult));

  const size_t resident_bytes =
      offsets_bytes + row_rep_ids_bytes + neighbors_bytes + weights_bytes +
      neighbor_sizes_bytes + row_sizes_bytes + row_self_loops_bytes;
  const size_t host_input_bytes = resident_bytes + pairs_bytes;
  const size_t device_input_bytes = resident_bytes + pairs_bytes;
  const size_t stream_bytes = (chunk_pair_bytes + chunk_result_bytes) * 2;
  const size_t io_bytes = output_bytes * 2;
  const size_t staging_overhead = AlignUp(64 * 1024, 256);

  return host_input_bytes + device_input_bytes + io_bytes + stream_bytes +
         staging_overhead;
}

std::vector<Sweg::CudaSliceSpan> Sweg::BuildCudaSlicePlan(
    const std::vector<EaGroupPrepared>& prepared_groups,
    size_t candidate_chunk_size) {
  std::vector<CudaSliceSpan> slices;
  if (prepared_groups.empty()) {
    return slices;
  }

  size_t total_rows = 0;
  size_t total_nnz = 0;
  size_t total_candidates = 0;
  std::vector<size_t> group_rows(prepared_groups.size(), 0);
  std::vector<size_t> group_nnz(prepared_groups.size(), 0);
  std::vector<size_t> group_candidates(prepared_groups.size(), 0);
  for (size_t i = 0; i < prepared_groups.size(); ++i) {
    const EaGroupPrepared& group = prepared_groups[i];
    group_rows[i] = group.q.size();
    group_candidates[i] = group.candidate_pairs.size();
    for (const SparseCounts& row : group.agg_by_idx) {
      group_nnz[i] += row.size();
    }
    total_rows += group_rows[i];
    total_nnz += group_nnz[i];
    total_candidates += group_candidates[i];
  }

  const size_t budget_bytes = GetCudaSliceMemoryBudgetBytes();
  stats_.cuda_slice_memory_budget_bytes =
      static_cast<int64_t>(budget_bytes);
  if (budget_bytes == 0 ||
      EstimateCudaSliceBytes(total_rows, total_nnz, total_candidates,
                             candidate_chunk_size) <= budget_bytes) {
    slices.push_back(CudaSliceSpan{0, prepared_groups.size(), total_rows,
                                   total_nnz, total_candidates,
                                   EstimateCudaSliceBytes(
                                       total_rows, total_nnz, total_candidates,
                                       candidate_chunk_size)});
    return slices;
  }

  size_t begin = 0;
  while (begin < prepared_groups.size()) {
    size_t end = begin;
    size_t slice_rows = 0;
    size_t slice_nnz = 0;
    size_t slice_candidates = 0;
    while (end < prepared_groups.size()) {
      const size_t next_rows = slice_rows + group_rows[end];
      const size_t next_nnz = slice_nnz + group_nnz[end];
      const size_t next_candidates = slice_candidates + group_candidates[end];
      const size_t estimated =
          EstimateCudaSliceBytes(next_rows, next_nnz, next_candidates,
                                 candidate_chunk_size);
      if (end > begin && estimated > budget_bytes) {
        break;
      }
      slice_rows = next_rows;
      slice_nnz = next_nnz;
      slice_candidates = next_candidates;
      ++end;
      if (estimated > budget_bytes) {
        break;
      }
    }
    if (end == begin) {
      ++end;
      slice_rows = group_rows[begin];
      slice_nnz = group_nnz[begin];
      slice_candidates = group_candidates[begin];
    }
    slices.push_back(CudaSliceSpan{
        begin,
        end,
        slice_rows,
        slice_nnz,
        slice_candidates,
        EstimateCudaSliceBytes(slice_rows, slice_nnz, slice_candidates,
                               candidate_chunk_size)});
    begin = end;
  }
  return slices;
}

int64_t Sweg::EdgeCountToSupernode(const SparseCounts& aggregated, int target_rep,
                                   bool self_loop) const {
  auto it = std::lower_bound(
      aggregated.begin(), aggregated.end(), target_rep,
      [](const auto& entry, int value) { return entry.first < value; });
  if (it == aggregated.end() || it->first != target_rep) {
    return 0;
  }
  int64_t edges = it->second;
  if (self_loop) {
    const auto nodes = Recover_S(target_rep);
    for (int node : nodes) {
      auto neighbors = graph_->neighbors(node);
      for (const int* succ = neighbors.first; succ != neighbors.second; ++succ) {
        if (*succ == node) {
          --edges;
        }
      }
    }
  }
  return std::max<int64_t>(0, edges);
}

int64_t Sweg::EncodeCostForPair(int rep_u, int64_t size_u, int rep_x,
                                int64_t size_x, int64_t edges) const {
  int64_t capacity = 0;
  if (rep_u == rep_x) {
    capacity = size_u * (size_u - 1) / 2;
  } else {
    capacity = size_u * size_x;
  }
  const int64_t complement_cost = 1 + capacity - edges;
  return std::min(edges, complement_cost);
}

Sweg::LocalGainResult Sweg::ComputeLocalEncodingGain(const SparseCounts& agg_a,
                                                     const SparseCounts& agg_b,
                                                     int rep_a, int rep_b,
                                                     int64_t size_a,
                                                     int64_t size_b,
                                                     int64_t self_loops_a,
                                                     int64_t self_loops_b) const {
  const int64_t size_m = size_a + size_b;

  int64_t before_total = 0;
  int64_t after_total = 0;
  bool seen_rep_a = false;
  bool seen_rep_b = false;
  auto accumulate_rep = [&](int rep_x, int64_t edges_a_raw, int64_t edges_b_raw) {
    if (rep_x == rep_a) {
      seen_rep_a = true;
    }
    if (rep_x == rep_b) {
      seen_rep_b = true;
    }
    const int64_t size_x =
        static_cast<int64_t>(supernode_sizes_by_rep_[static_cast<size_t>(rep_x)]);
    const int64_t edges_a =
        (rep_x == rep_a) ? std::max<int64_t>(0, edges_a_raw - self_loops_a)
                         : edges_a_raw;
    const int64_t edges_b =
        (rep_x == rep_b) ? std::max<int64_t>(0, edges_b_raw - self_loops_b)
                         : edges_b_raw;
    const int64_t edges_m = edges_a + edges_b;

    before_total +=
        EncodeCostForPair(rep_a, size_a, rep_x, size_x, edges_a) +
        EncodeCostForPair(rep_b, size_b, rep_x, size_x, edges_b);

    const int merged_rep = (rep_x == rep_b) ? rep_a : rep_x;
    const int64_t merged_size_x = (rep_x == rep_b) ? size_m : size_x;
    after_total +=
        EncodeCostForPair(rep_a, size_m, merged_rep, merged_size_x, edges_m);
  };

  size_t ia = 0;
  size_t ib = 0;
  while (ia < agg_a.size() || ib < agg_b.size()) {
    int rep_x = -1;
    int64_t edges_a_raw = 0;
    int64_t edges_b_raw = 0;
    if (ib >= agg_b.size() ||
        (ia < agg_a.size() &&
         agg_a[static_cast<size_t>(ia)].first <
             agg_b[static_cast<size_t>(ib)].first)) {
      rep_x = agg_a[static_cast<size_t>(ia)].first;
      edges_a_raw = agg_a[static_cast<size_t>(ia)].second;
      ++ia;
    } else if (ia >= agg_a.size() ||
               agg_b[static_cast<size_t>(ib)].first <
                   agg_a[static_cast<size_t>(ia)].first) {
      rep_x = agg_b[static_cast<size_t>(ib)].first;
      edges_b_raw = agg_b[static_cast<size_t>(ib)].second;
      ++ib;
    } else {
      rep_x = agg_a[static_cast<size_t>(ia)].first;
      edges_a_raw = agg_a[static_cast<size_t>(ia)].second;
      edges_b_raw = agg_b[static_cast<size_t>(ib)].second;
      ++ia;
      ++ib;
    }

    accumulate_rep(rep_x, edges_a_raw, edges_b_raw);
  }

  if (!seen_rep_a) {
    accumulate_rep(rep_a, 0, 0);
  }
  if (rep_b != rep_a && !seen_rep_b) {
    accumulate_rep(rep_b, 0, 0);
  }

  return LocalGainResult{before_total - after_total, before_total};
}

Sweg::EaGroupPrepared Sweg::PrepareBatchEncodingAwareGroup(
    const std::vector<int>& q) {
  EaPrepareResult result = PrepareBatchEncodingAwareGroupWithWorkspace(
      q, &sequential_prepare_workspace_, true);
  AccumulatePrepareStats(result.stats);
  return std::move(result.prepared);
}

Sweg::EaPrepareResult Sweg::PrepareBatchEncodingAwareGroupWithWorkspace(
    const std::vector<int>& q, PrepareWorkspace* workspace,
    bool allow_inner_parallel) {
  EaPrepareResult result;
  result.prepared.q = q;

  const auto prepare_start = std::chrono::steady_clock::now();
  EnsurePrepareWorkspace(workspace);

  const int group_size = static_cast<int>(result.prepared.q.size());
  if (group_size < 2) {
    result.stats.wall_ms =
        ElapsedMs(prepare_start, std::chrono::steady_clock::now());
    return result;
  }

  result.stats.group_count = 1;
  result.stats.group_max_size = static_cast<uint64_t>(group_size);
  result.stats.raw_pair_count =
      static_cast<uint64_t>(group_size) *
      static_cast<uint64_t>(group_size - 1) / 2ULL;

  const int top_k = std::max(1, std::min(top_k_, group_size - 1));
  const auto create_w_start = std::chrono::steady_clock::now();
  std::vector<SparseCounts> w_by_idx(static_cast<size_t>(group_size));
  result.prepared.agg_by_idx.resize(static_cast<size_t>(group_size));
  result.prepared.self_loops_by_idx.assign(static_cast<size_t>(group_size), 0);
  result.prepared.size_by_idx.assign(static_cast<size_t>(group_size), 0);
  const bool use_parallel_create = allow_inner_parallel && group_size >= 64;
#ifdef _OPENMP
  if (use_parallel_create) {
    const int thread_count = std::max(1, omp_get_max_threads());
    EnsurePrepareWorkspaces(thread_count);
#pragma omp parallel for schedule(dynamic, 8)
    for (int i = 0; i < group_size; ++i) {
      const int rep = result.prepared.q[static_cast<size_t>(i)];
      if (I_[rep] == -1) {
        continue;
      }
      const int tid = omp_get_thread_num();
      PrepareWorkspace& ws = prepare_workspaces_[static_cast<size_t>(tid)];
      w_by_idx[static_cast<size_t>(i)] =
          CreateWForSupernodeWithScratch(
              rep, ws.scratch,
              &result.prepared.self_loops_by_idx[static_cast<size_t>(i)]);
      result.prepared.agg_by_idx[static_cast<size_t>(i)] =
          AggregateWBySupernodeWithScratch(w_by_idx[static_cast<size_t>(i)],
                                           ws.scratch);
      result.prepared.size_by_idx[static_cast<size_t>(i)] = static_cast<int64_t>(
          supernode_sizes_by_rep_[static_cast<size_t>(rep)]);
    }
  } else
#endif
  {
    for (int i = 0; i < group_size; ++i) {
      const int rep = result.prepared.q[static_cast<size_t>(i)];
      if (I_[rep] != -1) {
        w_by_idx[static_cast<size_t>(i)] =
            CreateWForSupernodeWithScratch(
                rep, workspace->scratch,
                &result.prepared.self_loops_by_idx[static_cast<size_t>(i)]);
        result.prepared.agg_by_idx[static_cast<size_t>(i)] =
            AggregateWBySupernodeWithScratch(w_by_idx[static_cast<size_t>(i)],
                                             workspace->scratch);
        result.prepared.size_by_idx[static_cast<size_t>(i)] =
            static_cast<int64_t>(
                supernode_sizes_by_rep_[static_cast<size_t>(rep)]);
      }
    }
  }
  const auto create_w_end = std::chrono::steady_clock::now();
  result.stats.create_w_ms = ElapsedMs(create_w_start, create_w_end);

  const auto candidate_gen_start = std::chrono::steady_clock::now();
  struct TargetEntry {
    int row_idx = -1;
    int64_t edges = 0;
  };
  struct PairProxyInfo {
    int64_t proxy_gain = 0;
    double proxy_ratio = 0.0;
  };
  struct RankedCandidate {
    int other_idx = -1;
    double proxy_ratio = 0.0;
    int64_t proxy_gain = 0;
    int rep_lo = -1;
    int rep_hi = -1;
  };

  std::unordered_map<int, int> row_index_by_rep;
  row_index_by_rep.reserve(static_cast<size_t>(group_size) * 2U + 1U);
  for (int i = 0; i < group_size; ++i) {
    const int rep = result.prepared.q[static_cast<size_t>(i)];
    if (I_[rep] == -1) {
      continue;
    }
    row_index_by_rep.emplace(rep, i);
  }

  std::unordered_map<int, std::vector<TargetEntry>> buckets_by_target;
  buckets_by_target.reserve(static_cast<size_t>(group_size) * 4U + 1U);
  std::vector<int64_t> approx_old_cost_by_idx(static_cast<size_t>(group_size), 0);
  for (int i = 0; i < group_size; ++i) {
    const int rep_i = result.prepared.q[static_cast<size_t>(i)];
    if (I_[rep_i] == -1) {
      continue;
    }
    const int64_t size_i = result.prepared.size_by_idx[static_cast<size_t>(i)];
    const int64_t self_loops_i =
        result.prepared.self_loops_by_idx[static_cast<size_t>(i)];
    for (const auto& kv : result.prepared.agg_by_idx[static_cast<size_t>(i)]) {
      const int target_rep = kv.first;
      const int64_t edges_raw = kv.second;
      const int64_t target_size = static_cast<int64_t>(
          supernode_sizes_by_rep_[static_cast<size_t>(target_rep)]);
      const int64_t edges_for_cost =
          target_rep == rep_i ? std::max<int64_t>(0, edges_raw - self_loops_i)
                              : edges_raw;
      approx_old_cost_by_idx[static_cast<size_t>(i)] +=
          EncodeCostForPair(rep_i, size_i, target_rep, target_size,
                            edges_for_cost);
      buckets_by_target[target_rep].push_back(TargetEntry{i, edges_raw});
    }
  }

  std::unordered_map<uint64_t, PairProxyInfo> pair_proxy;
  pair_proxy.reserve(static_cast<size_t>(group_size) *
                     static_cast<size_t>(top_k) * 2U + 1U);

  auto pair_key = [](int a_idx, int b_idx) {
    const int lo = std::min(a_idx, b_idx);
    const int hi = std::max(a_idx, b_idx);
    return (static_cast<uint64_t>(static_cast<uint32_t>(lo)) << 32U) |
           static_cast<uint32_t>(hi);
  };

  for (const auto& bucket_kv : buckets_by_target) {
    const int target_rep = bucket_kv.first;
    const auto& entries = bucket_kv.second;
    if (entries.size() < 2) {
      continue;
    }
    const int64_t target_size = static_cast<int64_t>(
        supernode_sizes_by_rep_[static_cast<size_t>(target_rep)]);
    for (size_t ia = 0; ia < entries.size(); ++ia) {
      const int a_idx = entries[ia].row_idx;
      const int rep_a = result.prepared.q[static_cast<size_t>(a_idx)];
      if (target_rep == rep_a) {
        continue;
      }
      const int64_t size_a =
          result.prepared.size_by_idx[static_cast<size_t>(a_idx)];
      const int64_t edges_a = entries[ia].edges;
      for (size_t ib = ia + 1; ib < entries.size(); ++ib) {
        const int b_idx = entries[ib].row_idx;
        const int rep_b = result.prepared.q[static_cast<size_t>(b_idx)];
        if (target_rep == rep_b) {
          continue;
        }
        const int64_t size_b =
            result.prepared.size_by_idx[static_cast<size_t>(b_idx)];
        const int64_t edges_b = entries[ib].edges;
        const int64_t old_cost =
            EncodeCostForPair(rep_a, size_a, target_rep, target_size, edges_a) +
            EncodeCostForPair(rep_b, size_b, target_rep, target_size, edges_b);
        const int64_t new_cost = EncodeCostForPair(
            rep_a, size_a + size_b, target_rep, target_size,
            edges_a + edges_b);
        const int64_t delta = old_cost - new_cost;
        if (delta <= 0) {
          continue;
        }
        pair_proxy[pair_key(a_idx, b_idx)].proxy_gain += delta;
      }
    }
  }

  for (int i = 0; i < group_size; ++i) {
    const int rep_i = result.prepared.q[static_cast<size_t>(i)];
    if (I_[rep_i] == -1) {
      continue;
    }
    for (const auto& kv : result.prepared.agg_by_idx[static_cast<size_t>(i)]) {
      const int target_rep = kv.first;
      const auto row_it = row_index_by_rep.find(target_rep);
      if (row_it == row_index_by_rep.end()) {
        continue;
      }
      const int j = row_it->second;
      if (i >= j) {
        continue;
      }
      // Direct edges are used only as a heuristic candidate signal.
      // Exact gain scoring still decides whether the pair is actually beneficial.
      pair_proxy[pair_key(i, j)].proxy_gain += kv.second;
    }
  }

  auto ranked_better = [](const RankedCandidate& lhs,
                          const RankedCandidate& rhs) {
    if (lhs.proxy_ratio != rhs.proxy_ratio) {
      return lhs.proxy_ratio > rhs.proxy_ratio;
    }
    if (lhs.proxy_gain != rhs.proxy_gain) {
      return lhs.proxy_gain > rhs.proxy_gain;
    }
    if (lhs.rep_lo != rhs.rep_lo) {
      return lhs.rep_lo < rhs.rep_lo;
    }
    return lhs.rep_hi < rhs.rep_hi;
  };

  std::vector<std::vector<RankedCandidate>> ranked_by_idx(
      static_cast<size_t>(group_size));
  for (const auto& kv : pair_proxy) {
    const int a_idx = static_cast<int>(kv.first >> 32U);
    const int b_idx = static_cast<int>(kv.first & 0xffffffffU);
    const int64_t proxy_gain = kv.second.proxy_gain;
    if (proxy_gain <= 0) {
      continue;
    }
    const int rep_a = result.prepared.q[static_cast<size_t>(a_idx)];
    const int rep_b = result.prepared.q[static_cast<size_t>(b_idx)];
    const double proxy_ratio = static_cast<double>(proxy_gain) /
                               static_cast<double>(std::max<int64_t>(
                                   1, approx_old_cost_by_idx[static_cast<size_t>(a_idx)] +
                                          approx_old_cost_by_idx[static_cast<size_t>(b_idx)]));

    RankedCandidate cand_ab{b_idx, proxy_ratio, proxy_gain,
                            std::min(rep_a, rep_b), std::max(rep_a, rep_b)};
    RankedCandidate cand_ba{a_idx, proxy_ratio, proxy_gain,
                            std::min(rep_a, rep_b), std::max(rep_a, rep_b)};
    ranked_by_idx[static_cast<size_t>(a_idx)].push_back(cand_ab);
    ranked_by_idx[static_cast<size_t>(b_idx)].push_back(cand_ba);
  }

  std::vector<uint64_t> candidate_keys;
  candidate_keys.reserve(static_cast<size_t>(group_size) *
                         static_cast<size_t>(top_k));
  for (int i = 0; i < group_size; ++i) {
    auto& ranked = ranked_by_idx[static_cast<size_t>(i)];
    if (ranked.empty()) {
      continue;
    }
    std::sort(ranked.begin(), ranked.end(), ranked_better);
    const size_t keep = std::min(ranked.size(), static_cast<size_t>(top_k));
    for (size_t pos = 0; pos < keep; ++pos) {
      candidate_keys.push_back(pair_key(i, ranked[pos].other_idx));
    }
  }

  std::sort(candidate_keys.begin(), candidate_keys.end());
  candidate_keys.erase(
      std::unique(candidate_keys.begin(), candidate_keys.end()),
      candidate_keys.end());

  for (uint64_t key : candidate_keys) {
    const int a_idx = static_cast<int>(key >> 32U);
    const int b_idx = static_cast<int>(key & 0xffffffffU);
    result.prepared.candidate_pairs.push_back(
        EaCandidatePair{a_idx, b_idx, 0, 0});
  }
  result.stats.candidate_pairs_after_prune =
      static_cast<uint64_t>(result.prepared.candidate_pairs.size());

  const auto candidate_gen_end = std::chrono::steady_clock::now();
  result.stats.candidate_gen_ms =
      ElapsedMs(candidate_gen_start, candidate_gen_end);
  result.stats.wall_ms =
      ElapsedMs(prepare_start, std::chrono::steady_clock::now());
  return result;
}

void Sweg::ScoreBatchEncodingAwarePreparedGroup(EaGroupPrepared* prepared,
                                                double threshold) {
  if (prepared == nullptr || prepared->candidate_pairs.empty()) {
    return;
  }
  stats_.merge_exact_gain_calls +=
      static_cast<uint64_t>(prepared->candidate_pairs.size());

  std::vector<std::pair<int, int>> candidate_pairs;
  candidate_pairs.reserve(prepared->candidate_pairs.size());
  for (const EaCandidatePair& pair : prepared->candidate_pairs) {
    candidate_pairs.emplace_back(pair.a_idx, pair.b_idx);
  }

  const auto gain_scoring_start = std::chrono::steady_clock::now();
  std::vector<LocalGainResult> gain_results;
  if (scoring_backend_ == ScoringBackend::kCpu) {
    gain_results =
        ScoreCandidatesCpu(candidate_pairs, prepared->agg_by_idx, prepared->q,
                           prepared->size_by_idx, prepared->self_loops_by_idx);
  } else {
    const FlatAggCSR flat_agg =
        BuildFlatAggCsr(prepared->agg_by_idx, prepared->q, prepared->size_by_idx,
                        prepared->self_loops_by_idx);
    gain_results = ScoreCandidatesCuda(candidate_pairs, flat_agg, 0);
  }
  const auto gain_scoring_end = std::chrono::steady_clock::now();
  const double gain_scoring_ms =
      ElapsedMs(gain_scoring_start, gain_scoring_end);
  stats_.merge_gain_scoring_ms += gain_scoring_ms;
  stats_.merge_scoring_ms += gain_scoring_ms;
  AssignGainResultsToPreparedGroup(prepared, gain_results);
  FilterPreparedCandidatePairsByThreshold(prepared, threshold);
}

void Sweg::AssignGainResultsToPreparedGroup(
    EaGroupPrepared* prepared,
    const std::vector<LocalGainResult>& gain_results) const {
  if (prepared == nullptr) {
    return;
  }
  if (gain_results.size() != prepared->candidate_pairs.size()) {
    throw std::runtime_error("gain result size mismatch for prepared group");
  }
  for (size_t i = 0; i < prepared->candidate_pairs.size(); ++i) {
    prepared->candidate_pairs[i].gain = gain_results[i].gain;
    prepared->candidate_pairs[i].before_cost = gain_results[i].before_cost;
  }
}

void Sweg::FilterPreparedCandidatePairsByThreshold(EaGroupPrepared* prepared,
                                                   double threshold) {
  if (prepared == nullptr || prepared->candidate_pairs.empty()) {
    return;
  }
  std::vector<EaCandidatePair> filtered;
  filtered.reserve(prepared->candidate_pairs.size());
  for (const EaCandidatePair& pair : prepared->candidate_pairs) {
    const double gain_ratio =
        pair.before_cost > 0
            ? static_cast<double>(pair.gain) /
                  static_cast<double>(pair.before_cost)
            : 0.0;
    RecordSavingSample(gain_ratio);
    ++cur_tested_pairs_;
    if (pair.gain <= 0) {
      stats_.merge_rejected_by_threshold += 1;
      continue;
    }
    if (ea_use_threshold_) {
      if (pair.before_cost <= 0) {
        stats_.merge_rejected_by_threshold += 1;
        continue;
      }
      if (gain_ratio < threshold) {
        stats_.merge_rejected_by_threshold += 1;
        continue;
      }
    }
    ++cur_accepted_pairs_;
    stats_.merge_positive_gain_pairs += 1;
    filtered.push_back(pair);
  }
  prepared->candidate_pairs.swap(filtered);
}

std::vector<std::pair<int, int>> Sweg::SelectScoredBatchEncodingAwareGroupPairs(
    const EaGroupPrepared& prepared) {
  const int group_size = static_cast<int>(prepared.q.size());
  if (group_size < 2 || prepared.candidate_pairs.empty()) {
    return {};
  }

  std::vector<EaCandidatePair> scored_pairs = prepared.candidate_pairs;
  const auto selection_start = std::chrono::steady_clock::now();
  std::sort(scored_pairs.begin(), scored_pairs.end(),
            [](const EaCandidatePair& lhs, const EaCandidatePair& rhs) {
              if (lhs.gain != rhs.gain) {
                return lhs.gain > rhs.gain;
              }
              if (lhs.a_idx != rhs.a_idx) {
                return lhs.a_idx < rhs.a_idx;
              }
              return lhs.b_idx < rhs.b_idx;
            });

  std::vector<char> used(static_cast<size_t>(group_size), 0);
  std::vector<std::pair<int, int>> selected_rep_pairs;
  selected_rep_pairs.reserve(scored_pairs.size());
  int64_t total_local_gain = 0;
  for (const EaCandidatePair& pair : scored_pairs) {
    if (used[static_cast<size_t>(pair.a_idx)] ||
        used[static_cast<size_t>(pair.b_idx)]) {
      stats_.merge_rejected_by_overlap += 1;
      continue;
    }
    if (I_[prepared.q[static_cast<size_t>(pair.a_idx)]] == -1 ||
        I_[prepared.q[static_cast<size_t>(pair.b_idx)]] == -1) {
      continue;
    }
    used[static_cast<size_t>(pair.a_idx)] = 1;
    used[static_cast<size_t>(pair.b_idx)] = 1;
    selected_rep_pairs.emplace_back(
        prepared.q[static_cast<size_t>(pair.a_idx)],
        prepared.q[static_cast<size_t>(pair.b_idx)]);
    total_local_gain += pair.gain;
  }
  const auto selection_end = std::chrono::steady_clock::now();
  stats_.merge_selection_ms += ElapsedMs(selection_start, selection_end);
  stats_.merge_candidate_pairs +=
      static_cast<int64_t>(prepared.candidate_pairs.size());
  stats_.merge_selected_pairs += static_cast<int64_t>(selected_rep_pairs.size());
  stats_.merge_total_local_gain += total_local_gain;

  return selected_rep_pairs;
}

std::vector<std::pair<int, int>> Sweg::SelectBatchEncodingAwareGroupPairs(
    const std::vector<int>& q, double threshold) {
  EaGroupPrepared prepared = PrepareBatchEncodingAwareGroup(q);
  ScoreBatchEncodingAwarePreparedGroup(&prepared, threshold);
  return SelectScoredBatchEncodingAwareGroupPairs(prepared);
}

uint64_t Sweg::SplitMix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27U)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31U);
}

uint64_t Sweg::OverflowNodeRank(int node, uint64_t seed) const {
  return SplitMix64(seed ^ static_cast<uint64_t>(static_cast<uint32_t>(node)));
}

int Sweg::OverflowNodeShingle(int node, uint64_t seed) const {
  uint64_t best_rank = OverflowNodeRank(node, seed);
  int best_node = node;
  auto neighbors = graph_->neighbors(node);
  for (const int* it = neighbors.first; it != neighbors.second; ++it) {
    const int neighbor = *it;
    const uint64_t rank = OverflowNodeRank(neighbor, seed);
    if (rank < best_rank || (rank == best_rank && neighbor < best_node)) {
      best_rank = rank;
      best_node = neighbor;
    }
  }
  return best_node;
}

int Sweg::OverflowSupernodeShingle(int rep, uint64_t seed) const {
  int best_shingle = n_;
  for (int node = I_[rep]; node != -1; node = J_[node]) {
    const int shingle = OverflowNodeShingle(node, seed);
    if (shingle < best_shingle) {
      best_shingle = shingle;
    }
  }
  return best_shingle;
}

std::vector<std::vector<int>> Sweg::RefineOverflowGroup(
    const std::vector<int>& q, OverflowRefineCounters* counters) {
  if (overflow_group_gmax_ <= 0 ||
      static_cast<int>(q.size()) <= overflow_group_gmax_) {
    return {q};
  }

  const auto refine_start = std::chrono::steady_clock::now();
  std::vector<std::vector<int>> output_groups;
  std::vector<std::vector<int>> pending_groups;
  pending_groups.push_back(q);

  if (counters != nullptr) {
    counters->overflow_groups_seen += 1;
    counters->max_group_before =
        std::max<int64_t>(counters->max_group_before,
                          static_cast<int64_t>(q.size()));
  }

  for (int round = 0; round < overflow_refine_rounds_ &&
                      !pending_groups.empty();
       ++round) {
    std::vector<std::vector<int>> next_pending;
    const uint64_t seed =
        static_cast<uint64_t>(rng_()) ^
        (static_cast<uint64_t>(round + 1) * 0x9e3779b97f4a7c15ULL);

    for (const auto& group : pending_groups) {
      if (static_cast<int>(group.size()) <= overflow_group_gmax_) {
        output_groups.push_back(group);
        if (counters != nullptr) {
          counters->max_group_after =
              std::max<int64_t>(counters->max_group_after,
                                static_cast<int64_t>(group.size()));
        }
        continue;
      }

      std::vector<std::pair<int, int>> keyed_group;
      keyed_group.reserve(group.size());
      for (int rep : group) {
        keyed_group.emplace_back(OverflowSupernodeShingle(rep, seed), rep);
      }
      std::stable_sort(
          keyed_group.begin(), keyed_group.end(),
          [](const auto& lhs, const auto& rhs) {
            if (lhs.first != rhs.first) {
              return lhs.first < rhs.first;
            }
            return lhs.second < rhs.second;
          });

      bool split_happened = false;
      size_t begin = 0;
      while (begin < keyed_group.size()) {
        size_t end = begin + 1;
        const int shingle = keyed_group[begin].first;
        while (end < keyed_group.size() && keyed_group[end].first == shingle) {
          ++end;
        }
        std::vector<int> bucket;
        bucket.reserve(end - begin);
        for (size_t i = begin; i < end; ++i) {
          bucket.push_back(keyed_group[i].second);
        }
        if (bucket.size() != group.size()) {
          split_happened = true;
        }
        if (static_cast<int>(bucket.size()) <= overflow_group_gmax_) {
          output_groups.push_back(bucket);
          if (counters != nullptr) {
            counters->refined_subgroups += 1;
            counters->max_group_after =
                std::max<int64_t>(counters->max_group_after,
                                  static_cast<int64_t>(bucket.size()));
          }
        } else {
          next_pending.push_back(std::move(bucket));
        }
        begin = end;
      }

    }

    pending_groups.swap(next_pending);
  }

  for (auto& group : pending_groups) {
    if (static_cast<int>(group.size()) <= overflow_group_gmax_) {
      output_groups.push_back(group);
      if (counters != nullptr) {
        counters->max_group_after =
            std::max<int64_t>(counters->max_group_after,
                              static_cast<int64_t>(group.size()));
      }
      continue;
    }

    std::stable_sort(group.begin(), group.end());
    for (size_t begin = 0; begin < group.size();
         begin += static_cast<size_t>(overflow_group_gmax_)) {
      const size_t end = std::min(
          group.size(), begin + static_cast<size_t>(overflow_group_gmax_));
      std::vector<int> chunk;
      chunk.reserve(end - begin);
      for (size_t i = begin; i < end; ++i) {
        chunk.push_back(group[i]);
      }
      output_groups.push_back(std::move(chunk));
      if (counters != nullptr) {
        counters->forced_chunks += 1;
        counters->refined_subgroups += 1;
        counters->max_group_after =
            std::max<int64_t>(counters->max_group_after,
                              static_cast<int64_t>(output_groups.back().size()));
      }
    }
  }

  if (counters != nullptr) {
    stats_.overflow_refine_ms += ElapsedMs(
        refine_start, std::chrono::steady_clock::now());
  }
  return output_groups;
}

void Sweg::ScoreBatchEncodingAwarePreparedGroupsBlocked(
    std::vector<EaGroupPrepared>* prepared_groups, double threshold) {
  if (prepared_groups == nullptr || prepared_groups->empty()) {
    return;
  }

  ScoreBatchEncodingAwarePreparedGroupsBlockedRange(
      prepared_groups, 0, prepared_groups->size(), threshold);
}

void Sweg::ScoreBatchEncodingAwarePreparedGroupsBlockedRange(
    std::vector<EaGroupPrepared>* prepared_groups, size_t begin, size_t end,
    double threshold) {
  if (prepared_groups == nullptr || begin >= end || end > prepared_groups->size()) {
    return;
  }

  if (scoring_backend_ == ScoringBackend::kCpu) {
    for (size_t i = begin; i < end; ++i) {
      EaGroupPrepared& prepared = (*prepared_groups)[i];
      ScoreBatchEncodingAwarePreparedGroup(&prepared, threshold);
    }
    return;
  }
  const size_t candidate_chunk_size =
      candidate_batch_budget_ > 0
          ? static_cast<size_t>(candidate_batch_budget_)
          : 0;
  ScoreBatchEncodingAwarePreparedGroupsBlockedCudaRange(
      prepared_groups, begin, end, threshold, candidate_chunk_size);
}

Sweg::CudaBlockedBatch Sweg::BuildCudaBlockedBatch(
    const std::vector<EaGroupPrepared>& groups) const {
  return BuildCudaBlockedBatch(groups, 0, groups.size());
}

Sweg::CudaBlockedBatch Sweg::BuildCudaBlockedBatch(
    const std::vector<EaGroupPrepared>& groups, size_t begin,
    size_t end) const {
  CudaBlockedBatch batch;

  size_t total_rows = 0;
  size_t total_nnz = 0;
  size_t total_candidates = 0;
  for (size_t i = begin; i < end; ++i) {
    const EaGroupPrepared& group = groups[i];
    total_rows += group.q.size();
    total_candidates += group.candidate_pairs.size();
    for (const SparseCounts& row : group.agg_by_idx) {
      total_nnz += row.size();
    }
  }

  batch.flat_agg.offsets.reserve(total_rows + 1);
  batch.flat_agg.row_rep_ids.reserve(total_rows);
  batch.flat_agg.row_sizes.reserve(total_rows);
  batch.flat_agg.row_self_loops.reserve(total_rows);
  batch.flat_agg.neighbors.reserve(total_nnz);
  batch.flat_agg.weights.reserve(total_nnz);
  batch.flat_agg.neighbor_sizes.reserve(total_nnz);
  batch.candidate_pairs.reserve(total_candidates);
  batch.destinations.reserve(total_candidates);
  batch.flat_agg.offsets.push_back(0);

  int row_base = 0;
  for (size_t group_idx = begin; group_idx < end; ++group_idx) {
    const EaGroupPrepared& group = groups[group_idx];
    if (group.agg_by_idx.size() != group.q.size() ||
        group.size_by_idx.size() != group.q.size() ||
        group.self_loops_by_idx.size() != group.q.size()) {
      throw std::runtime_error("prepared group row metadata size mismatch");
    }

    for (size_t row_idx = 0; row_idx < group.q.size(); ++row_idx) {
      batch.flat_agg.row_rep_ids.push_back(group.q[row_idx]);
      batch.flat_agg.row_sizes.push_back(group.size_by_idx[row_idx]);
      batch.flat_agg.row_self_loops.push_back(group.self_loops_by_idx[row_idx]);

      for (const auto& [neighbor_rep, weight] : group.agg_by_idx[row_idx]) {
        if (neighbor_rep < 0 ||
            static_cast<size_t>(neighbor_rep) >= supernode_sizes_by_rep_.size()) {
          throw std::runtime_error("neighbor representative out of range");
        }
        batch.flat_agg.neighbors.push_back(neighbor_rep);
        batch.flat_agg.weights.push_back(weight);
        batch.flat_agg.neighbor_sizes.push_back(
            static_cast<int64_t>(
                supernode_sizes_by_rep_[static_cast<size_t>(neighbor_rep)]));
      }
      batch.flat_agg.offsets.push_back(
          static_cast<int>(batch.flat_agg.neighbors.size()));
    }

    for (size_t candidate_idx = 0; candidate_idx < group.candidate_pairs.size();
         ++candidate_idx) {
      const EaCandidatePair& candidate = group.candidate_pairs[candidate_idx];
      if (candidate.a_idx < 0 || candidate.b_idx < 0 ||
          static_cast<size_t>(candidate.a_idx) >= group.q.size() ||
          static_cast<size_t>(candidate.b_idx) >= group.q.size()) {
        throw std::runtime_error("candidate row index out of range");
      }
      batch.candidate_pairs.emplace_back(row_base + candidate.a_idx,
                                         row_base + candidate.b_idx);
      batch.destinations.push_back(
          CudaBlockedBatch::Destination{group_idx - begin, candidate_idx});
    }

    row_base += static_cast<int>(group.q.size());
  }

  assert(batch.flat_agg.offsets.size() ==
         batch.flat_agg.row_rep_ids.size() + 1);
  assert(batch.flat_agg.row_sizes.size() == batch.flat_agg.row_rep_ids.size());
  assert(batch.flat_agg.row_self_loops.size() ==
         batch.flat_agg.row_rep_ids.size());
  assert(batch.flat_agg.neighbors.size() == batch.flat_agg.weights.size());
  assert(batch.flat_agg.neighbors.size() ==
         batch.flat_agg.neighbor_sizes.size());
  assert(batch.candidate_pairs.size() == batch.destinations.size());

  return batch;
}

void Sweg::ScoreBatchEncodingAwarePreparedGroupsBlockedCuda(
    std::vector<EaGroupPrepared>* prepared_groups, double threshold,
    size_t candidate_chunk_size) {
  if (prepared_groups == nullptr || prepared_groups->empty()) {
    return;
  }

  ScoreBatchEncodingAwarePreparedGroupsBlockedCudaRange(
      prepared_groups, 0, prepared_groups->size(), threshold,
      candidate_chunk_size);
}

void Sweg::ScoreBatchEncodingAwarePreparedGroupsBlockedCudaRange(
    std::vector<EaGroupPrepared>* prepared_groups, size_t begin, size_t end,
    double threshold, size_t candidate_chunk_size) {
  if (prepared_groups == nullptr || begin >= end || end > prepared_groups->size()) {
    return;
  }

  CudaBlockedBatch batch = BuildCudaBlockedBatch(*prepared_groups, begin, end);
  if (batch.candidate_pairs.empty()) {
    return;
  }

  const auto gain_scoring_start = std::chrono::steady_clock::now();
  std::vector<LocalGainResult> gain_results = ScoreCandidatesCuda(
      batch.candidate_pairs, batch.flat_agg, candidate_chunk_size);
  const auto gain_scoring_end = std::chrono::steady_clock::now();
  const double gain_scoring_ms =
      ElapsedMs(gain_scoring_start, gain_scoring_end);
  stats_.merge_gain_scoring_ms += gain_scoring_ms;
  stats_.merge_scoring_ms += gain_scoring_ms;
  stats_.merge_exact_gain_calls +=
      static_cast<uint64_t>(batch.candidate_pairs.size());

  if (gain_results.size() != batch.destinations.size()) {
    throw std::runtime_error("CUDA scoring result size mismatch");
  }

  for (size_t i = 0; i < gain_results.size(); ++i) {
    const CudaBlockedBatch::Destination& dst = batch.destinations[i];
    if (begin + dst.group_idx >= prepared_groups->size()) {
      throw std::runtime_error("destination group index out of range");
    }
    EaGroupPrepared& group = (*prepared_groups)[begin + dst.group_idx];
    if (dst.candidate_idx >= group.candidate_pairs.size()) {
      throw std::runtime_error("destination candidate index out of range");
    }
    group.candidate_pairs[dst.candidate_idx].gain = gain_results[i].gain;
    group.candidate_pairs[dst.candidate_idx].before_cost =
        gain_results[i].before_cost;
  }

  for (size_t i = begin; i < end; ++i) {
    FilterPreparedCandidatePairsByThreshold(&(*prepared_groups)[i], threshold);
  }
}

void Sweg::MergeBatchEncodingAwareGroup(const std::vector<int>& q,
                                        double threshold) {
  std::vector<std::pair<int, int>> selected_rep_pairs =
      SelectBatchEncodingAwareGroupPairs(q, threshold);
  const auto update_start = std::chrono::steady_clock::now();
  for (const auto& pair : selected_rep_pairs) {
    const int rep_a = pair.first;
    const int rep_b = pair.second;
    if (I_[rep_a] == -1 || I_[rep_b] == -1) {
      continue;
    }
    Update_S(rep_a, rep_b);
  }
  const auto update_end = std::chrono::steady_clock::now();
  stats_.merge_update_ms += ElapsedMs(update_start, update_end);
}

int Sweg::SupernodeLength(int key) const {
  int count = 0;
  int cur = I_[key];
  while (cur != -1) {
    ++count;
    cur = J_[cur];
  }
  return count;
}

std::vector<int> Sweg::Recover_S(int key) const {
  std::vector<int> nodes;
  nodes.reserve(static_cast<size_t>(std::max(0, SupernodeLength(key))));
  int cur = I_[key];
  while (cur != -1) {
    nodes.push_back(cur);
    cur = J_[cur];
  }
  return nodes;
}

void Sweg::Update_S(int A, int B) {
  std::vector<int> nodes_a = Recover_S(A);
  std::vector<int> nodes_b = Recover_S(B);
  if (nodes_a.empty() || nodes_b.empty()) {
    return;
  }

  J_[nodes_a.back()] = I_[B];
  I_[B] = -1;
  const int head = I_[A];
  for (int node : nodes_a) {
    S_[node] = head;
  }
  for (int node : nodes_b) {
    S_[node] = head;
  }
  supernode_sizes_by_rep_[A] = static_cast<int>(nodes_a.size() + nodes_b.size());
  supernode_sizes_by_rep_[B] = 0;
}

EncodingResult Sweg::Encode() {
  EncodingResult result;
  const auto relabel_start = std::chrono::steady_clock::now();
  result.node_to_supernode.assign(static_cast<size_t>(n_), -1);

  std::vector<int> rep_to_new(static_cast<size_t>(n_), -1);
  for (int rep = 0; rep < n_; ++rep) {
    if (I_[rep] == -1) {
      continue;
    }
    std::vector<int> nodes = Recover_S(rep);
    const int new_id = static_cast<int>(result.supernodes.size());
    rep_to_new[static_cast<size_t>(rep)] = new_id;
    result.supernode_sizes.push_back(static_cast<int>(nodes.size()));
    for (int node : nodes) {
      result.node_to_supernode[static_cast<size_t>(node)] = new_id;
    }
    result.supernodes.push_back(std::move(nodes));
  }
  const auto relabel_end = std::chrono::steady_clock::now();
  stats_.encode_relabel_ms += ElapsedMs(relabel_start, relabel_end);

  struct PairArc {
    uint64_t pair_key = 0;
    int u = 0;
    int v = 0;
  };

  const auto count_start = std::chrono::steady_clock::now();
  std::vector<PairArc> pair_arcs;
  pair_arcs.reserve(static_cast<size_t>(graph_->m));

  for (int u = 0; u < n_; ++u) {
    const int su = result.node_to_supernode[static_cast<size_t>(u)];
    if (su < 0) {
      throw std::runtime_error("Invalid node_to_supernode during encode.");
    }
    auto neighbors = graph_->neighbors(u);
    for (const int* it = neighbors.first; it != neighbors.second; ++it) {
      const int v = *it;
      const int sv = result.node_to_supernode[static_cast<size_t>(v)];
      if (sv < 0) {
        throw std::runtime_error("Invalid node_to_supernode during encode.");
      }
      // Preserve the original encode_old-style semantics: only keep
      // arcs whose source supernode id is <= destination supernode id.
      if (su > sv) {
        continue;
      }
      pair_arcs.push_back(PairArc{PackEdge(su, sv), u, v});
    }
  }

  std::sort(pair_arcs.begin(), pair_arcs.end(),
            [](const PairArc& lhs, const PairArc& rhs) {
              if (lhs.pair_key != rhs.pair_key) {
                return lhs.pair_key < rhs.pair_key;
              }
              if (lhs.u != rhs.u) {
                return lhs.u < rhs.u;
              }
              return lhs.v < rhs.v;
            });
  const auto count_end = std::chrono::steady_clock::now();
  stats_.encode_count_edges_ms += ElapsedMs(count_start, count_end);

  size_t begin = 0;
  while (begin < pair_arcs.size()) {
    size_t end = begin + 1;
    const uint64_t pair_key = pair_arcs[begin].pair_key;
    while (end < pair_arcs.size() && pair_arcs[end].pair_key == pair_key) {
      ++end;
    }

    const int A = static_cast<int>(pair_key >> 32U);
    const int B = static_cast<int>(pair_key & 0xffffffffU);
    const int64_t real_edges = static_cast<int64_t>(end - begin);
    const int64_t size_a =
        static_cast<int64_t>(result.supernode_sizes[static_cast<size_t>(A)]);
    const int64_t size_b =
        static_cast<int64_t>(result.supernode_sizes[static_cast<size_t>(B)]);

    const auto decision_start = std::chrono::steady_clock::now();
    const double edge_compare_cond =
        (A == B)
            ? (static_cast<double>(size_a) *
               static_cast<double>(size_a - 1) / 4.0)
            : (static_cast<double>(size_a) * static_cast<double>(size_b) /
               2.0);

    if (static_cast<double>(real_edges) <= edge_compare_cond) {
      auto& dst = result.Cp;
      for (size_t i = begin; i < end; ++i) {
        dst.push_back(EdgePair{pair_arcs[i].u, pair_arcs[i].v});
      }
      const auto decision_end = std::chrono::steady_clock::now();
      stats_.encode_decision_ms += ElapsedMs(decision_start, decision_end);
      begin = end;
      continue;
    }

    result.P.push_back(EdgePair{A, B});
    const auto decision_end = std::chrono::steady_clock::now();
    stats_.encode_decision_ms += ElapsedMs(decision_start, decision_end);

    const auto correction_start = std::chrono::steady_clock::now();
    std::unordered_set<uint64_t> edge_set;
    edge_set.reserve((end - begin) * 2U + 1U);
    for (size_t i = begin; i < end; ++i) {
      edge_set.insert(PackEdge(pair_arcs[i].u, pair_arcs[i].v));
    }

    const std::vector<int>& in_a = result.supernodes[static_cast<size_t>(A)];
    const std::vector<int>& in_b = result.supernodes[static_cast<size_t>(B)];
    for (int u : in_a) {
      for (int v : in_b) {
        if (A == B && u == v) {
          continue;
        }
        if (edge_set.find(PackEdge(u, v)) == edge_set.end()) {
          result.Cm.push_back(EdgePair{u, v});
        }
      }
    }
    const auto correction_end = std::chrono::steady_clock::now();
    stats_.encode_correction_generation_ms +=
        ElapsedMs(correction_start, correction_end);

    begin = end;
  }

  std::sort(result.P.begin(), result.P.end(), EdgePairLess);
  std::sort(result.Cp.begin(), result.Cp.end(), EdgePairLess);
  std::sort(result.Cm.begin(), result.Cm.end(), EdgePairLess);
  return result;
}

void Sweg::Drop(double error_bound, EncodingResult& result) const {
  if (error_bound <= 0.0) {
    return;
  }

  std::vector<double> cv(static_cast<size_t>(n_), 0.0);
  for (int v = 0; v < n_; ++v) {
    cv[static_cast<size_t>(v)] =
        error_bound * static_cast<double>(graph_->out_degree(v));
  }

  auto drop_edges = [&cv](std::vector<EdgePair>& edges) {
    std::vector<EdgePair> kept;
    kept.reserve(edges.size());
    for (const EdgePair& edge : edges) {
      if (cv[static_cast<size_t>(edge.first)] >= 1.0 &&
          cv[static_cast<size_t>(edge.second)] >= 1.0) {
        cv[static_cast<size_t>(edge.first)] -= 1.0;
        cv[static_cast<size_t>(edge.second)] -= 1.0;
      } else {
        kept.push_back(edge);
      }
    }
    edges.swap(kept);
  };

  drop_edges(result.Cp);
  drop_edges(result.Cm);

  std::stable_sort(result.P.begin(), result.P.end(),
                   [&result](const EdgePair& lhs, const EdgePair& rhs) {
                     const int64_t lhs_val =
                         static_cast<int64_t>(
                             result.supernode_sizes[static_cast<size_t>(lhs.first)]) *
                         static_cast<int64_t>(
                             result.supernode_sizes[static_cast<size_t>(lhs.second)]);
                     const int64_t rhs_val =
                         static_cast<int64_t>(
                             result.supernode_sizes[static_cast<size_t>(rhs.first)]) *
                         static_cast<int64_t>(
                             result.supernode_sizes[static_cast<size_t>(rhs.second)]);
                     return lhs_val < rhs_val;
                   });

  std::vector<EdgePair> kept_p;
  kept_p.reserve(result.P.size());
  for (const EdgePair& edge : result.P) {
    const int A = edge.first;
    const int B = edge.second;
    if (A == B) {
      kept_p.push_back(edge);
      continue;
    }

    const int size_b = result.supernode_sizes[static_cast<size_t>(B)];
    bool cond_a = true;
    for (int node : result.supernodes[static_cast<size_t>(A)]) {
      if (cv[static_cast<size_t>(node)] < static_cast<double>(size_b)) {
        cond_a = false;
        break;
      }
    }
    if (!cond_a) {
      kept_p.push_back(edge);
      continue;
    }

    const int size_a = result.supernode_sizes[static_cast<size_t>(A)];
    bool cond_b = true;
    for (int node : result.supernodes[static_cast<size_t>(B)]) {
      if (cv[static_cast<size_t>(node)] < static_cast<double>(size_a)) {
        cond_b = false;
        break;
      }
    }
    if (!cond_b) {
      kept_p.push_back(edge);
      continue;
    }

    for (int node : result.supernodes[static_cast<size_t>(A)]) {
      cv[static_cast<size_t>(node)] -= static_cast<double>(size_b);
    }
    for (int node : result.supernodes[static_cast<size_t>(B)]) {
      cv[static_cast<size_t>(node)] -= static_cast<double>(size_a);
    }
  }
  result.P.swap(kept_p);
}

void Sweg::WriteOutput(const std::string& out_dir,
                       const EncodingResult& result) const {
  std::filesystem::create_directories(out_dir);

  {
    std::ofstream out(out_dir + "/G.txt");
    if (!out) {
      throw std::runtime_error("Failed to open G.txt for writing.");
    }
    for (size_t sid = 0; sid < result.supernodes.size(); ++sid) {
      out << sid;
      for (int node : result.supernodes[sid]) {
        out << '\t' << node;
      }
      out << '\n';
    }
  }

  auto write_edges = [&out_dir](const std::string& name,
                                const std::vector<EdgePair>& edges) {
    std::ofstream out(out_dir + "/" + name);
    if (!out) {
      throw std::runtime_error("Failed to open output file: " + name);
    }
    for (const EdgePair& edge : edges) {
      out << edge.first << '\t' << edge.second << '\n';
    }
  };

  write_edges("P.txt", result.P);
  write_edges("Cp.txt", result.Cp);
  write_edges("Cm.txt", result.Cm);
}

int Sweg::CountGroups() const {
  return static_cast<int>(BuildGroups().size());
}

int Sweg::CountActiveSupernodes() const {
  int count = 0;
  for (int rep = 0; rep < n_; ++rep) {
    if (I_[rep] != -1) {
      ++count;
    }
  }
  return count;
}
