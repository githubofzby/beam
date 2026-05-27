#include "sweg.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
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

}  // namespace

const char* MergeModeToString(MergeMode mode) {
  switch (mode) {
    case MergeMode::kSequential:
      return "sequential";
    case MergeMode::kBatchSuperJaccard:
      return "batch-sj";
    case MergeMode::kBatchEncodingAware:
      return "batch-ea";
    case MergeMode::kBatchEncodingAwareFrozen:
      return "batch-ea-frozen";
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

Sweg::Sweg(const CSRGraph& graph, MergeMode merge_mode, int top_k,
           bool ea_use_threshold, uint64_t seed,
           ScoringBackend scoring_backend, bool verify_cuda_gain,
           int group_batch_size, int candidate_batch_budget,
           int overflow_group_gmax, int overflow_refine_rounds)
    : graph_(&graph),
      n_(graph.n),
      merge_mode_(merge_mode),
      top_k_(top_k),
      ea_use_threshold_(ea_use_threshold),
      scoring_backend_(scoring_backend),
      verify_cuda_gain_(verify_cuda_gain),
      group_batch_size_(std::max(1, group_batch_size)),
      candidate_batch_budget_(std::max(0, candidate_batch_budget)),
      overflow_group_gmax_(std::max(0, overflow_group_gmax)),
      overflow_refine_rounds_(std::max(0, overflow_refine_rounds)) {
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
  rng_.seed(static_cast<uint32_t>(seed));
  cuda_verify_samples_remaining_ = verify_cuda_gain_ ? static_cast<size_t>(4096) : 0;
  serial_workspace_.counts.assign(static_cast<size_t>(n_), 0);
  serial_workspace_.marks.assign(static_cast<size_t>(n_), 0);
  serial_workspace_.touched.reserve(static_cast<size_t>(std::min(n_, 1 << 20)));
  serial_workspace_.epoch = 1;
  scratch_counts_.assign(static_cast<size_t>(n_), 0);
  scratch_marks_.assign(static_cast<size_t>(n_), 0);
  scratch_touched_.reserve(static_cast<size_t>(std::min(n_, 1 << 20)));
}

void Sweg::ResetRuntimeStats() {
  stats_ = RuntimeStats{};
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

void Sweg::EnsureParallelWorkspaces(int thread_count) const {
  if (thread_count <= 0) {
    thread_count = 1;
  }
  if (static_cast<int>(parallel_workspaces_.size()) >= thread_count) {
    return;
  }
  const size_t old_size = parallel_workspaces_.size();
  parallel_workspaces_.resize(static_cast<size_t>(thread_count));
  for (size_t i = old_size; i < parallel_workspaces_.size(); ++i) {
    auto& ws = parallel_workspaces_[i];
    ws.counts.assign(static_cast<size_t>(n_), 0);
    ws.marks.assign(static_cast<size_t>(n_), 0);
    ws.touched.reserve(static_cast<size_t>(std::min(n_, 1 << 20)));
    ws.epoch = 1;
  }
}

void Sweg::Run(int iterations, int print_offset) {
  ResetRuntimeStats();
  cuda_verify_samples_remaining_ =
      verify_cuda_gain_ ? static_cast<size_t>(4096) : 0;
  const auto total_start = std::chrono::steady_clock::now();
  for (int iter = 1; iter <= iterations; ++iter) {
    const auto divide_start = std::chrono::steady_clock::now();
    Divide();
    const auto divide_end = std::chrono::steady_clock::now();
    stats_.runtime_divide_ms += ElapsedMs(divide_start, divide_end);

    const auto merge_start = std::chrono::steady_clock::now();
    Merge(iter);
    const auto merge_end = std::chrono::steady_clock::now();
    stats_.runtime_merge_ms += ElapsedMs(merge_start, merge_end);

    if (print_offset > 0 &&
        (iter % print_offset == 0 || iter == iterations)) {
      std::cout << "Iter " << iter << ": groups=" << CountGroups()
                << " active_supernodes=" << CountActiveSupernodes() << "\n";
    }
  }
  const auto total_end = std::chrono::steady_clock::now();
  stats_.runtime_total_ms = ElapsedMs(total_start, total_end);
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

void Sweg::Divide() {
  ShuffleArray();
  std::fill(F_.begin(), F_.end(), -1);

  for (int A = 0; A < n_; ++A) {
    if (I_[A] == -1) {
      continue;
    }
    F_[A] = n_;
    for (int v = I_[A]; v != -1; v = J_[v]) {
      const int fv = NodeShingle(v);
      if (F_[A] > fv) {
        F_[A] = fv;
      }
    }
  }

  std::iota(G_.begin(), G_.end(), 0);
  std::stable_sort(G_.begin(), G_.end(),
                   [this](int a, int b) { return F_[a] < F_[b]; });

  gstart_ = 0;
  while (gstart_ < n_ && F_[G_[gstart_]] == -1) {
    ++gstart_;
  }
}

std::vector<Sweg::GroupSpan> Sweg::BuildGroups() const {
  std::vector<GroupSpan> groups;
  if (gstart_ >= n_) {
    return groups;
  }

  int current = F_[G_[gstart_]];
  int start = gstart_;
  for (int i = gstart_ + 1; i < n_; ++i) {
    if (F_[G_[i]] != current) {
      groups.push_back(GroupSpan{current, start, i - start});
      current = F_[G_[i]];
      start = i;
    }
  }
  groups.push_back(GroupSpan{current, start, n_ - start});
  return groups;
}

Sweg::SparseCounts Sweg::CreateWForSupernode(int rep,
                                             int64_t* self_loop_count) const {
  return CreateWForSupernodeWithScratch(rep, serial_workspace_,
                                        self_loop_count);
}

Sweg::SparseCounts Sweg::CreateWForSupernodeWithScratch(
    int rep, ParallelScratch& scratch, int64_t* self_loop_count) const {
  ResetParallelScratch(scratch);
  ResetScratch();
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

Sweg::SparseCounts Sweg::UpdateW(const SparseCounts& w_a,
                                 const SparseCounts& w_b) const {
  SparseCounts result;
  result.reserve(w_a.size() + w_b.size());
  size_t ia = 0;
  size_t ib = 0;
  while (ia < w_a.size() && ib < w_b.size()) {
    if (w_a[ia].first == w_b[ib].first) {
      result.emplace_back(w_a[ia].first, w_a[ia].second + w_b[ib].second);
      ++ia;
      ++ib;
    } else if (w_a[ia].first < w_b[ib].first) {
      result.push_back(w_a[ia++]);
    } else {
      result.push_back(w_b[ib++]);
    }
  }
  while (ia < w_a.size()) {
    result.push_back(w_a[ia++]);
  }
  while (ib < w_b.size()) {
    result.push_back(w_b[ib++]);
  }
  return result;
}

std::pair<double, double> Sweg::JacSimSavCost(const SparseCounts& w_a,
                                              const SparseCounts& w_b) const {
  int64_t down = 0;
  int64_t up = 0;
  const int64_t savings_down =
      static_cast<int64_t>(w_a.size()) + static_cast<int64_t>(w_b.size());
  int64_t savings_up = 0;

  size_t ia = 0;
  size_t ib = 0;
  while (ia < w_a.size() && ib < w_b.size()) {
    if (w_a[ia].first == w_b[ib].first) {
      ++savings_up;
      up += std::min(w_a[ia].second, w_b[ib].second);
      down += std::max(w_a[ia].second, w_b[ib].second);
      ++ia;
      ++ib;
    } else if (w_a[ia].first < w_b[ib].first) {
      ++savings_up;
      down += w_a[ia].second;
      ++ia;
    } else {
      ++savings_up;
      down += w_b[ib].second;
      ++ib;
    }
  }
  while (ia < w_a.size()) {
    ++savings_up;
    down += w_a[ia].second;
    ++ia;
  }
  while (ib < w_b.size()) {
    ++savings_up;
    down += w_b[ib].second;
    ++ib;
  }

  const double jaccard =
      down == 0 ? 0.0 : static_cast<double>(up) / static_cast<double>(down);
  const double savings = savings_down == 0
                             ? 0.0
                             : 1.0 - static_cast<double>(savings_up) /
                                         static_cast<double>(savings_down);
  return {jaccard, savings};
}

double Sweg::WeightedJaccard(const SparseCounts& w_a,
                             const SparseCounts& w_b) const {
  int64_t down = 0;
  int64_t up = 0;

  size_t ia = 0;
  size_t ib = 0;
  while (ia < w_a.size() && ib < w_b.size()) {
    if (w_a[ia].first == w_b[ib].first) {
      up += std::min(w_a[ia].second, w_b[ib].second);
      down += std::max(w_a[ia].second, w_b[ib].second);
      ++ia;
      ++ib;
    } else if (w_a[ia].first < w_b[ib].first) {
      down += w_a[ia].second;
      ++ia;
    } else {
      down += w_b[ib].second;
      ++ib;
    }
  }
  while (ia < w_a.size()) {
    down += w_a[ia].second;
    ++ia;
  }
  while (ib < w_b.size()) {
    down += w_b[ib].second;
    ++ib;
  }
  return down == 0 ? 0.0
                   : static_cast<double>(up) / static_cast<double>(down);
}

double Sweg::CostSaving(const SparseCounts& w_a, const SparseCounts& w_b,
                        int rep_a, int rep_b) const {
  const std::vector<int> nodes_a = Recover_S(rep_a);
  const std::vector<int> nodes_b = Recover_S(rep_b);
  double cost_a = 0.0;
  double cost_b = 0.0;
  double cost_union = 0.0;

  std::unordered_map<int, int> candidate_size;
  std::unordered_map<int, int64_t> candidate_a;
  std::unordered_map<int, int64_t> candidate_b;

  for (const auto& kv : w_a) {
    const int supernode = S_[kv.first];
    if (candidate_size.find(supernode) == candidate_size.end()) {
      candidate_size.emplace(
          supernode,
          supernode_sizes_by_rep_[static_cast<size_t>(supernode)]);
    }
    candidate_a[supernode] += kv.second;
  }
  for (const auto& kv : w_b) {
    const int supernode = S_[kv.first];
    if (candidate_size.find(supernode) == candidate_size.end()) {
      candidate_size.emplace(
          supernode,
          supernode_sizes_by_rep_[static_cast<size_t>(supernode)]);
    }
    candidate_b[supernode] += kv.second;
  }

  const double size_a = static_cast<double>(nodes_a.size());
  const double size_b = static_cast<double>(nodes_b.size());
  const double size_union = size_a + size_b;

  for (const auto& kv : candidate_a) {
    const int key = kv.first;
    const double val_a = static_cast<double>(kv.second);
    const double cand_size = static_cast<double>(candidate_size[key]);
    if (key == rep_a) {
      if (val_a >= (size_a * size_a) / 4.0) {
        cost_a += size_a * size_a - val_a;
      } else {
        cost_a += val_a;
      }
      continue;
    }
    if (val_a >= (size_a * cand_size) / 2.0) {
      cost_a += cand_size * size_a - val_a + 1.0;
    } else {
      cost_a += val_a;
    }

    if (key == rep_b) {
      continue;
    }

    auto it_b = candidate_b.find(key);
    if (it_b != candidate_b.end()) {
      const double combined = val_a + static_cast<double>(it_b->second);
      if (combined >= (size_union * cand_size) / 2.0) {
        cost_union += cand_size * size_union - combined + 1.0;
      } else {
        cost_union += combined;
      }
    } else {
      if (val_a >= (size_union * cand_size) / 2.0) {
        cost_union += cand_size * size_union - val_a + 1.0;
      } else {
        cost_union += val_a;
      }
    }
  }

  for (const auto& kv : candidate_b) {
    const int key = kv.first;
    const double val_b = static_cast<double>(kv.second);
    const double cand_size = static_cast<double>(candidate_size[key]);
    if (key == rep_b) {
      if (val_b >= (size_b * size_b) / 4.0) {
        cost_b += size_b * size_b - val_b;
      } else {
        cost_b += val_b;
      }
      continue;
    }

    if (val_b >= (size_b * cand_size) / 2.0) {
      cost_b += cand_size * size_b - val_b + 1.0;
    } else {
      cost_b += val_b;
    }

    if (candidate_a.find(key) != candidate_a.end() || key == rep_a) {
      continue;
    }
    if (val_b >= (size_union * cand_size) / 2.0) {
      cost_union += cand_size * size_union - val_b + 1.0;
    } else {
      cost_union += val_b;
    }
  }

  int64_t union_edges = 0;
  auto it_aa = candidate_a.find(rep_a);
  if (it_aa != candidate_a.end()) {
    union_edges += it_aa->second;
  }
  auto it_ab = candidate_a.find(rep_b);
  if (it_ab != candidate_a.end()) {
    union_edges += it_ab->second;
  }
  auto it_bb = candidate_b.find(rep_b);
  if (it_bb != candidate_b.end()) {
    union_edges += it_bb->second;
  }
  if (union_edges > 0) {
    const double val = static_cast<double>(union_edges);
    if (val >= (size_union * size_union) / 4.0) {
      cost_union += size_union * size_union - val;
    } else {
      cost_union += val;
    }
  }

  const double denom = cost_a + cost_b;
  if (denom == 0.0) {
    return 0.0;
  }
  return 1.0 - cost_union / denom;
}

void Sweg::Merge(int iter) {
  const auto groups = BuildGroups();
  const double threshold = 1.0 / static_cast<double>(iter + 1);
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

  if (merge_mode_ == MergeMode::kBatchEncodingAwareFrozen) {
    std::vector<std::pair<int, int>> all_selected_pairs;
    for (const GroupSpan& group : groups) {
      const auto refined_groups = build_refined_group_list(group);
      for (const auto& q : refined_groups) {
        std::vector<std::pair<int, int>> group_pairs =
            SelectBatchEncodingAwareGroupPairs(q, threshold);
        all_selected_pairs.insert(all_selected_pairs.end(), group_pairs.begin(),
                                  group_pairs.end());
      }
    }

    const auto update_start = std::chrono::steady_clock::now();
    for (const auto& pair : all_selected_pairs) {
      const int rep_a = pair.first;
      const int rep_b = pair.second;
      if (I_[rep_a] == -1 || I_[rep_b] == -1) {
        continue;
      }
      Update_S(rep_a, rep_b);
    }
    const auto update_end = std::chrono::steady_clock::now();
    stats_.merge_update_ms += ElapsedMs(update_start, update_end);
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
    return;
  }

  if (merge_mode_ == MergeMode::kBatchEncodingAwareBlocked) {
    const int block_size = std::max(1, group_batch_size_);
    int64_t blocked_slice_count = 0;
    int64_t blocked_slice_max_candidates = 0;
    int64_t blocked_slice_max_rows = 0;
    int64_t blocked_slice_max_nnz = 0;
    bool used_overflow_cuda_group_fallback = false;
    for (size_t block_start = 0; block_start < groups.size();
         block_start += static_cast<size_t>(block_size)) {
      const size_t block_end =
          std::min(groups.size(), block_start + static_cast<size_t>(block_size));
      std::vector<std::pair<int, int>> block_selected_pairs;

      std::vector<EaGroupPrepared> prepared_slice;
      prepared_slice.reserve(block_end - block_start);
      size_t slice_candidate_count = 0;
      size_t slice_rows = 0;
      size_t slice_nnz = 0;
      const size_t candidate_budget =
          candidate_batch_budget_ > 0
              ? static_cast<size_t>(candidate_batch_budget_)
              : std::numeric_limits<size_t>::max();
      const size_t row_budget = 65536;
      const size_t nnz_budget = 2000000;

      auto flush_prepared_slice = [&]() {
        if (prepared_slice.empty()) {
          return;
        }
        blocked_slice_count += 1;
        blocked_slice_max_candidates =
            std::max<int64_t>(blocked_slice_max_candidates,
                              static_cast<int64_t>(slice_candidate_count));
        blocked_slice_max_rows =
            std::max<int64_t>(blocked_slice_max_rows,
                              static_cast<int64_t>(slice_rows));
        blocked_slice_max_nnz =
            std::max<int64_t>(blocked_slice_max_nnz,
                              static_cast<int64_t>(slice_nnz));

        if (scoring_backend_ == ScoringBackend::kCuda &&
            overflow_group_gmax_ > 0) {
          used_overflow_cuda_group_fallback = true;
          for (EaGroupPrepared& prepared : prepared_slice) {
            ScoreBatchEncodingAwarePreparedGroup(&prepared, threshold);
          }
        } else if (scoring_backend_ == ScoringBackend::kCuda &&
                   candidate_batch_budget_ > 0) {
          ScoreBatchEncodingAwarePreparedGroupsBlockedCudaBudgeted(
              &prepared_slice, threshold);
        } else if (scoring_backend_ == ScoringBackend::kCuda) {
          ScoreBatchEncodingAwarePreparedGroupsBlocked(&prepared_slice,
                                                       threshold);
        } else {
          for (EaGroupPrepared& prepared : prepared_slice) {
            ScoreBatchEncodingAwarePreparedGroup(&prepared, threshold);
          }
        }

        for (const EaGroupPrepared& prepared : prepared_slice) {
          std::vector<std::pair<int, int>> group_pairs =
              SelectScoredBatchEncodingAwareGroupPairs(prepared);
          block_selected_pairs.insert(block_selected_pairs.end(),
                                      group_pairs.begin(), group_pairs.end());
        }
        prepared_slice.clear();
        slice_candidate_count = 0;
        slice_rows = 0;
        slice_nnz = 0;
      };

      for (size_t gi = block_start; gi < block_end; ++gi) {
        const GroupSpan& group = groups[gi];
        const auto refined_groups = build_refined_group_list(group);
        for (const auto& q : refined_groups) {
          EaGroupPrepared prepared = PrepareBatchEncodingAwareGroup(q);
          size_t prepared_nnz = 0;
          for (const SparseCounts& row : prepared.agg_by_idx) {
            prepared_nnz += row.size();
          }
          const size_t prepared_rows = prepared.q.size();
          const size_t prepared_candidates = prepared.candidate_pairs.size();

          const bool would_exceed_budget =
              !prepared_slice.empty() &&
              (slice_candidate_count + prepared_candidates > candidate_budget ||
               slice_rows + prepared_rows > row_budget ||
               slice_nnz + prepared_nnz > nnz_budget);
          if (would_exceed_budget) {
            flush_prepared_slice();
          }

          slice_candidate_count += prepared_candidates;
          slice_rows += prepared_rows;
          slice_nnz += prepared_nnz;
          prepared_slice.push_back(std::move(prepared));

          if (slice_candidate_count >= candidate_budget ||
              slice_rows >= row_budget || slice_nnz >= nnz_budget) {
            flush_prepared_slice();
          }
        }
      }
      flush_prepared_slice();

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
    if (blocked_slice_count > 0) {
      std::cout << "  Blocked scoring slices: count=" << blocked_slice_count
                << " max_candidates=" << blocked_slice_max_candidates
                << " max_rows=" << blocked_slice_max_rows
                << " max_nnz=" << blocked_slice_max_nnz << "\n";
    }
    if (used_overflow_cuda_group_fallback) {
      std::cout << "  CUDA blocked scoring fallback: per-group scoring enabled "
                   "because overflow refinement is active\n";
    }
    return;
  }

  for (const GroupSpan& group : groups) {
    const auto refined_groups = build_refined_group_list(group);
    for (const auto& q : refined_groups) {
      switch (merge_mode_) {
        case MergeMode::kSequential:
          MergeSequentialGroup(q, threshold);
          break;
        case MergeMode::kBatchSuperJaccard:
          MergeBatchSuperJaccardGroup(q, threshold);
          break;
        case MergeMode::kBatchEncodingAware:
          MergeBatchEncodingAwareGroup(q, threshold);
          break;
        case MergeMode::kBatchEncodingAwareFrozen:
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
}

void Sweg::MergeSequentialGroup(const std::vector<int>& q, double threshold) {
  const auto create_w_start = std::chrono::steady_clock::now();
  std::unordered_map<int, SparseCounts> active_w;
  active_w.reserve(q.size());
  for (size_t idx = 0; idx < q.size(); ++idx) {
    active_w.emplace(static_cast<int>(idx), CreateWForSupernode(q[idx]));
  }
  const auto create_w_end = std::chrono::steady_clock::now();
  stats_.merge_create_w_ms += ElapsedMs(create_w_start, create_w_end);

  const int initial_size = static_cast<int>(q.size());
  while (active_w.size() > 1) {
    const auto selection_start = std::chrono::steady_clock::now();
    std::uniform_int_distribution<int> pick(0, initial_size - 1);
    const int a_idx = pick(rng_);
    auto it_a = active_w.find(a_idx);
    if (it_a == active_w.end()) {
      const auto selection_end = std::chrono::steady_clock::now();
      stats_.merge_selection_ms += ElapsedMs(selection_start, selection_end);
      continue;
    }

    const auto scoring_start = std::chrono::steady_clock::now();
    double max_jaccard = 0.0;
    int best_idx = -1;
    for (const auto& kv : active_w) {
      const int b_idx = kv.first;
      if (b_idx == a_idx) {
        continue;
      }
      const auto sim = JacSimSavCost(it_a->second, kv.second);
      if (sim.first > max_jaccard) {
        max_jaccard = sim.first;
        best_idx = b_idx;
      }
    }
    const auto scoring_end = std::chrono::steady_clock::now();
    stats_.merge_scoring_ms += ElapsedMs(scoring_start, scoring_end);

    if (best_idx == -1) {
      active_w.erase(it_a);
      const auto selection_end = std::chrono::steady_clock::now();
      stats_.merge_selection_ms += ElapsedMs(selection_start, selection_end);
      continue;
    }

    auto it_b = active_w.find(best_idx);
    const auto saving_start = std::chrono::steady_clock::now();
    const double savings =
        CostSaving(it_a->second, it_b->second, q[a_idx], q[best_idx]);
    const auto saving_end = std::chrono::steady_clock::now();
    stats_.merge_scoring_ms += ElapsedMs(saving_start, saving_end);
    const auto selection_end = std::chrono::steady_clock::now();
    stats_.merge_selection_ms += ElapsedMs(selection_start, selection_end);

    if (savings >= threshold) {
      const auto update_start = std::chrono::steady_clock::now();
      it_a->second = UpdateW(it_a->second, it_b->second);
      active_w.erase(it_b);
      Update_S(q[a_idx], q[best_idx]);
      const auto update_end = std::chrono::steady_clock::now();
      stats_.merge_update_ms += ElapsedMs(update_start, update_end);
    } else {
      active_w.erase(it_a);
    }
  }
}

void Sweg::MergeBatchSuperJaccardGroup(const std::vector<int>& q,
                                       double threshold) {
  struct Candidate {
    int best_idx = -1;
    double best_jaccard = -1.0;
  };

  const int n = static_cast<int>(q.size());
  const auto create_w_start = std::chrono::steady_clock::now();
  std::vector<SparseCounts> w_by_idx(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    if (I_[q[static_cast<size_t>(i)]] != -1) {
      w_by_idx[static_cast<size_t>(i)] = CreateWForSupernode(q[static_cast<size_t>(i)]);
    }
  }
  const auto create_w_end = std::chrono::steady_clock::now();
  stats_.merge_create_w_ms += ElapsedMs(create_w_start, create_w_end);

  const auto scoring_start = std::chrono::steady_clock::now();
  std::vector<Candidate> choice(static_cast<size_t>(n));
  for (int a_idx = 0; a_idx < n; ++a_idx) {
    if (I_[q[static_cast<size_t>(a_idx)]] == -1) {
      continue;
    }
    for (int b_idx = 0; b_idx < n; ++b_idx) {
      if (a_idx == b_idx || I_[q[static_cast<size_t>(b_idx)]] == -1) {
        continue;
      }
      const auto sim = JacSimSavCost(w_by_idx[static_cast<size_t>(a_idx)],
                                     w_by_idx[static_cast<size_t>(b_idx)]);
      if (sim.first > choice[static_cast<size_t>(a_idx)].best_jaccard) {
        choice[static_cast<size_t>(a_idx)].best_jaccard = sim.first;
        choice[static_cast<size_t>(a_idx)].best_idx = b_idx;
      }
    }
  }
  const auto scoring_end = std::chrono::steady_clock::now();
  stats_.merge_scoring_ms += ElapsedMs(scoring_start, scoring_end);

  const auto selection_start = std::chrono::steady_clock::now();
  std::vector<std::pair<int, int>> accepted_pairs;
  accepted_pairs.reserve(static_cast<size_t>(n / 2));
  std::vector<char> used(static_cast<size_t>(n), 0);

  for (int a_idx = 0; a_idx < n; ++a_idx) {
    if (used[static_cast<size_t>(a_idx)] ||
        I_[q[static_cast<size_t>(a_idx)]] == -1) {
      continue;
    }
    const int b_idx = choice[static_cast<size_t>(a_idx)].best_idx;
    if (b_idx < 0 || used[static_cast<size_t>(b_idx)] ||
        I_[q[static_cast<size_t>(b_idx)]] == -1) {
      continue;
    }
    if (choice[static_cast<size_t>(b_idx)].best_idx != a_idx) {
      continue;
    }

    const int rep_a = q[static_cast<size_t>(a_idx)];
    const int rep_b = q[static_cast<size_t>(b_idx)];
    const auto saving_start = std::chrono::steady_clock::now();
    const double savings =
        CostSaving(w_by_idx[static_cast<size_t>(a_idx)],
                   w_by_idx[static_cast<size_t>(b_idx)], rep_a, rep_b);
    const auto saving_end = std::chrono::steady_clock::now();
    stats_.merge_scoring_ms += ElapsedMs(saving_start, saving_end);
    if (savings < threshold) {
      continue;
    }

    used[static_cast<size_t>(a_idx)] = 1;
    used[static_cast<size_t>(b_idx)] = 1;
    if (rep_a < rep_b) {
      accepted_pairs.emplace_back(a_idx, b_idx);
    } else {
      accepted_pairs.emplace_back(b_idx, a_idx);
    }
  }
  const auto selection_end = std::chrono::steady_clock::now();
  stats_.merge_selection_ms += ElapsedMs(selection_start, selection_end);
  stats_.merge_candidate_pairs += static_cast<int64_t>(accepted_pairs.size());
  stats_.merge_selected_pairs += static_cast<int64_t>(accepted_pairs.size());

  const auto update_start = std::chrono::steady_clock::now();
  for (const auto& pair : accepted_pairs) {
    const int keep_idx = pair.first;
    const int merge_idx = pair.second;
    if (I_[q[static_cast<size_t>(keep_idx)]] == -1 ||
        I_[q[static_cast<size_t>(merge_idx)]] == -1) {
      continue;
    }
    Update_S(q[static_cast<size_t>(keep_idx)],
             q[static_cast<size_t>(merge_idx)]);
  }
  const auto update_end = std::chrono::steady_clock::now();
  stats_.merge_update_ms += ElapsedMs(update_start, update_end);
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
  auto candidate_better = [](const std::pair<double, int>& lhs,
                             const std::pair<double, int>& rhs) {
    if (lhs.first != rhs.first) {
      return lhs.first > rhs.first;
    }
    return lhs.second < rhs.second;
  };

  EaGroupPrepared prepared;
  prepared.q = q;

  const int group_size = static_cast<int>(prepared.q.size());
  if (group_size < 2) {
    return prepared;
  }

  const int top_k = std::max(1, std::min(top_k_, group_size - 1));
  const auto create_w_start = std::chrono::steady_clock::now();
  std::vector<SparseCounts> w_by_idx(static_cast<size_t>(group_size));
  prepared.agg_by_idx.resize(static_cast<size_t>(group_size));
  prepared.self_loops_by_idx.assign(static_cast<size_t>(group_size), 0);
  prepared.size_by_idx.assign(static_cast<size_t>(group_size), 0);
  const bool use_parallel_create =
      group_size >= 64;
#ifdef _OPENMP
  if (use_parallel_create) {
    const int thread_count = std::max(1, omp_get_max_threads());
    EnsureParallelWorkspaces(thread_count);
#pragma omp parallel for schedule(dynamic, 8)
    for (int i = 0; i < group_size; ++i) {
      const int rep = prepared.q[static_cast<size_t>(i)];
      if (I_[rep] == -1) {
        continue;
      }
      const int tid = omp_get_thread_num();
      auto& ws = parallel_workspaces_[static_cast<size_t>(tid)];
      w_by_idx[static_cast<size_t>(i)] =
          CreateWForSupernodeWithScratch(
              rep, ws, &prepared.self_loops_by_idx[static_cast<size_t>(i)]);
      prepared.agg_by_idx[static_cast<size_t>(i)] =
          AggregateWBySupernodeWithScratch(w_by_idx[static_cast<size_t>(i)], ws);
      prepared.size_by_idx[static_cast<size_t>(i)] = static_cast<int64_t>(
          supernode_sizes_by_rep_[static_cast<size_t>(rep)]);
    }
  } else
#endif
  {
    for (int i = 0; i < group_size; ++i) {
      const int rep = prepared.q[static_cast<size_t>(i)];
      if (I_[rep] != -1) {
        w_by_idx[static_cast<size_t>(i)] =
            CreateWForSupernode(rep,
                                &prepared.self_loops_by_idx[static_cast<size_t>(i)]);
        prepared.agg_by_idx[static_cast<size_t>(i)] =
            AggregateWBySupernode(w_by_idx[static_cast<size_t>(i)]);
        prepared.size_by_idx[static_cast<size_t>(i)] =
            static_cast<int64_t>(
                supernode_sizes_by_rep_[static_cast<size_t>(rep)]);
      }
    }
  }
  const auto create_w_end = std::chrono::steady_clock::now();
  stats_.merge_create_w_ms += ElapsedMs(create_w_start, create_w_end);

  const auto candidate_gen_start = std::chrono::steady_clock::now();
  std::vector<std::vector<std::pair<double, int>>> top_candidates_by_idx(
      static_cast<size_t>(group_size));
  for (int i = 0; i < group_size; ++i) {
    top_candidates_by_idx[static_cast<size_t>(i)].reserve(
        static_cast<size_t>(top_k));
  }

  auto try_push_top_candidate =
      [&](int src_idx, int dst_idx, double score) {
        auto& top_candidates =
            top_candidates_by_idx[static_cast<size_t>(src_idx)];
        if (static_cast<int>(top_candidates.size()) < top_k) {
          top_candidates.emplace_back(score, dst_idx);
          size_t pos = top_candidates.size() - 1;
          while (pos > 0 &&
                 candidate_better(top_candidates[pos],
                                  top_candidates[pos - 1])) {
            std::swap(top_candidates[pos], top_candidates[pos - 1]);
            --pos;
          }
          return;
        }
        if (score <= top_candidates.back().first) {
          return;
        }
        top_candidates.back() = {score, dst_idx};
        size_t pos = top_candidates.size() - 1;
        while (pos > 0 &&
               candidate_better(top_candidates[pos], top_candidates[pos - 1])) {
          std::swap(top_candidates[pos], top_candidates[pos - 1]);
          --pos;
        }
      };

  std::vector<uint64_t> candidate_keys;
  candidate_keys.reserve(static_cast<size_t>(group_size) *
                         static_cast<size_t>(top_k));
  for (int a_idx = 0; a_idx < group_size; ++a_idx) {
    if (I_[prepared.q[static_cast<size_t>(a_idx)]] == -1) {
      continue;
    }
    for (int b_idx = a_idx + 1; b_idx < group_size; ++b_idx) {
      if (I_[prepared.q[static_cast<size_t>(b_idx)]] == -1) {
        continue;
      }
      const double jaccard = WeightedJaccard(
          w_by_idx[static_cast<size_t>(a_idx)],
          w_by_idx[static_cast<size_t>(b_idx)]);
      try_push_top_candidate(a_idx, b_idx, jaccard);
      try_push_top_candidate(b_idx, a_idx, jaccard);
    }
  }

  for (int a_idx = 0; a_idx < group_size; ++a_idx) {
    if (I_[prepared.q[static_cast<size_t>(a_idx)]] == -1) {
      continue;
    }
    const auto& top_candidates =
        top_candidates_by_idx[static_cast<size_t>(a_idx)];
    for (const auto& candidate : top_candidates) {
      const int b_idx = candidate.second;
      const int u = std::min(a_idx, b_idx);
      const int v = std::max(a_idx, b_idx);
      candidate_keys.push_back(
          (static_cast<uint64_t>(static_cast<uint32_t>(u)) << 32U) |
          static_cast<uint32_t>(v));
    }
  }

  std::sort(candidate_keys.begin(), candidate_keys.end());
  candidate_keys.erase(
      std::unique(candidate_keys.begin(), candidate_keys.end()),
      candidate_keys.end());

  for (uint64_t key : candidate_keys) {
    const int a_idx = static_cast<int>(key >> 32U);
    const int b_idx = static_cast<int>(key & 0xffffffffU);
    prepared.candidate_pairs.push_back(EaCandidatePair{a_idx, b_idx, 0, 0});
  }

  const auto candidate_gen_end = std::chrono::steady_clock::now();
  const double candidate_gen_ms =
      ElapsedMs(candidate_gen_start, candidate_gen_end);
  stats_.merge_candidate_gen_ms += candidate_gen_ms;
  stats_.merge_scoring_ms += candidate_gen_ms;
  return prepared;
}

void Sweg::ScoreBatchEncodingAwarePreparedGroup(EaGroupPrepared* prepared,
                                                double threshold) {
  if (prepared == nullptr || prepared->candidate_pairs.empty()) {
    return;
  }

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
    gain_results = ScoreCandidatesCuda(candidate_pairs, flat_agg);
  }
  const auto gain_scoring_end = std::chrono::steady_clock::now();
  const double gain_scoring_ms =
      ElapsedMs(gain_scoring_start, gain_scoring_end);
  stats_.merge_gain_scoring_ms += gain_scoring_ms;
  stats_.merge_scoring_ms += gain_scoring_ms;
  std::vector<EaCandidatePair> filtered;
  filtered.reserve(prepared->candidate_pairs.size());
  for (size_t i = 0; i < prepared->candidate_pairs.size(); ++i) {
    EaCandidatePair pair = prepared->candidate_pairs[i];
    pair.gain = gain_results[i].gain;
    pair.before_cost = gain_results[i].before_cost;
    if (pair.gain <= 0) {
      continue;
    }
    if (ea_use_threshold_) {
      if (pair.before_cost <= 0) {
        continue;
      }
      const double gain_ratio =
          static_cast<double>(pair.gain) / static_cast<double>(pair.before_cost);
      if (gain_ratio < threshold) {
        continue;
      }
    }
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

uint64_t Sweg::SplitMix64(uint64_t x) const {
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

  size_t total_rows = 0;
  size_t total_candidates = 0;
  size_t total_nnz = 0;
  for (const EaGroupPrepared& prepared : *prepared_groups) {
    total_rows += prepared.q.size();
    total_candidates += prepared.candidate_pairs.size();
    for (const SparseCounts& row : prepared.agg_by_idx) {
      total_nnz += row.size();
    }
  }
  if (total_candidates == 0) {
    return;
  }

  std::vector<std::pair<int, int>> global_candidate_pairs;
  global_candidate_pairs.reserve(total_candidates);

  FlatAggCSR flat;
  flat.offsets.resize(total_rows + 1, 0);
  flat.row_rep_ids.reserve(total_rows);
  flat.row_sizes.reserve(total_rows);
  flat.row_self_loops.reserve(total_rows);
  flat.neighbors.reserve(total_nnz);
  flat.weights.reserve(total_nnz);
  flat.neighbor_sizes.reserve(total_nnz);

  struct GroupMeta {
    size_t row_base = 0;
    size_t candidate_base = 0;
    size_t candidate_count = 0;
  };
  std::vector<GroupMeta> metas;
  metas.reserve(prepared_groups->size());

  size_t row_base = 0;
  size_t candidate_base = 0;
  size_t nnz = 0;
  for (EaGroupPrepared& prepared : *prepared_groups) {
    GroupMeta meta;
    meta.row_base = row_base;
    meta.candidate_base = candidate_base;
    meta.candidate_count = prepared.candidate_pairs.size();

    for (size_t i = 0; i < prepared.q.size(); ++i) {
      flat.offsets[row_base + i] = static_cast<int>(nnz);
      flat.row_rep_ids.push_back(prepared.q[i]);
      flat.row_sizes.push_back(prepared.size_by_idx[i]);
      flat.row_self_loops.push_back(prepared.self_loops_by_idx[i]);
      const SparseCounts& row = prepared.agg_by_idx[i];
      for (const auto& kv : row) {
        flat.neighbors.push_back(kv.first);
        flat.weights.push_back(kv.second);
        flat.neighbor_sizes.push_back(
            static_cast<int64_t>(
                supernode_sizes_by_rep_[static_cast<size_t>(kv.first)]));
        ++nnz;
      }
    }
    row_base += prepared.q.size();

    for (const EaCandidatePair& pair : prepared.candidate_pairs) {
      global_candidate_pairs.emplace_back(
          static_cast<int>(meta.row_base + static_cast<size_t>(pair.a_idx)),
          static_cast<int>(meta.row_base + static_cast<size_t>(pair.b_idx)));
    }
    candidate_base += prepared.candidate_pairs.size();
    metas.push_back(meta);
  }
  flat.offsets[row_base] = static_cast<int>(nnz);

  const auto gain_scoring_start = std::chrono::steady_clock::now();
  std::vector<LocalGainResult> gain_results;
  if (scoring_backend_ == ScoringBackend::kCpu) {
    gain_results = ScoreCandidatesCuda(global_candidate_pairs, flat);
  } else {
    gain_results = ScoreCandidatesCuda(global_candidate_pairs, flat);
  }
  const auto gain_scoring_end = std::chrono::steady_clock::now();
  const double gain_scoring_ms =
      ElapsedMs(gain_scoring_start, gain_scoring_end);
  stats_.merge_gain_scoring_ms += gain_scoring_ms;
  stats_.merge_scoring_ms += gain_scoring_ms;

  for (size_t gi = 0; gi < prepared_groups->size(); ++gi) {
    EaGroupPrepared& prepared = (*prepared_groups)[gi];
    const GroupMeta& meta = metas[gi];
    std::vector<EaCandidatePair> filtered;
    filtered.reserve(prepared.candidate_pairs.size());
    for (size_t i = 0; i < meta.candidate_count; ++i) {
      EaCandidatePair pair = prepared.candidate_pairs[i];
      const LocalGainResult& result = gain_results[meta.candidate_base + i];
      pair.gain = result.gain;
      pair.before_cost = result.before_cost;
      if (pair.gain <= 0) {
        continue;
      }
      if (ea_use_threshold_) {
        if (pair.before_cost <= 0) {
          continue;
        }
        const double gain_ratio =
            static_cast<double>(pair.gain) /
            static_cast<double>(pair.before_cost);
        if (gain_ratio < threshold) {
          continue;
        }
      }
      filtered.push_back(pair);
    }
    prepared.candidate_pairs.swap(filtered);
  }
}

void Sweg::ScoreBatchEncodingAwarePreparedGroupsBlockedCudaBudgeted(
    std::vector<EaGroupPrepared>* prepared_groups, double threshold) {
  if (prepared_groups == nullptr || prepared_groups->empty()) {
    return;
  }

  const int budget = std::max(1, candidate_batch_budget_);
  size_t begin = 0;
  while (begin < prepared_groups->size()) {
    size_t end = begin;
    size_t candidate_count = 0;
    while (end < prepared_groups->size()) {
      const size_t next =
          candidate_count + (*prepared_groups)[end].candidate_pairs.size();
      if (end > begin && next > static_cast<size_t>(budget)) {
        break;
      }
      candidate_count = next;
      ++end;
      if (candidate_count >= static_cast<size_t>(budget)) {
        break;
      }
    }

    std::vector<EaGroupPrepared> slice;
    slice.reserve(end - begin);
    for (size_t i = begin; i < end; ++i) {
      slice.push_back(std::move((*prepared_groups)[i]));
    }
    ScoreBatchEncodingAwarePreparedGroupsBlocked(&slice, threshold);
    for (size_t i = 0; i < slice.size(); ++i) {
      (*prepared_groups)[begin + i] = std::move(slice[i]);
    }
    begin = end;
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
