#include "sweg.hpp"

#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

Sweg::FlatAggCSR Sweg::BuildFlatAggCsr(
    const std::vector<PreparedRow>& agg_by_idx, const std::vector<int>& q,
    const std::vector<int64_t>& size_by_idx,
    const std::vector<int64_t>& self_loops_by_idx) const {
  FlatAggCSR flat;
  const int rows = static_cast<int>(agg_by_idx.size());
  flat.offsets.resize(static_cast<size_t>(rows + 1), 0);
  flat.row_rep_ids.resize(static_cast<size_t>(rows), -1);
  flat.row_sizes = size_by_idx;
  flat.row_self_loops = self_loops_by_idx;

  size_t total_nnz = 0;
  for (int i = 0; i < rows; ++i) {
    flat.row_rep_ids[static_cast<size_t>(i)] = q[static_cast<size_t>(i)];
    flat.offsets[static_cast<size_t>(i)] = static_cast<int>(total_nnz);
    total_nnz += agg_by_idx[static_cast<size_t>(i)].size();
  }
  flat.offsets[static_cast<size_t>(rows)] = static_cast<int>(total_nnz);
  flat.neighbors.reserve(total_nnz);
  flat.weights.reserve(total_nnz);
  flat.neighbor_sizes.reserve(total_nnz);
  for (const auto& row : agg_by_idx) {
    for (const auto& kv : row) {
      flat.neighbors.push_back(kv.first);
      flat.weights.push_back(kv.second);
      flat.neighbor_sizes.push_back(
          static_cast<int64_t>(supernode_sizes_by_rep_[static_cast<size_t>(kv.first)]));
    }
  }
  return flat;
}

std::vector<Sweg::LocalGainResult> Sweg::ScoreCandidatesCpu(
    const std::vector<std::pair<int, int>>& candidate_pairs,
    const std::vector<PreparedRow>& agg_by_idx, const std::vector<int>& q,
    const std::vector<int64_t>& size_by_idx,
    const std::vector<int64_t>& self_loops_by_idx) const {
  std::vector<LocalGainResult> results(candidate_pairs.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 64)
#endif
  for (int i = 0; i < static_cast<int>(candidate_pairs.size()); ++i) {
    const auto& pair = candidate_pairs[static_cast<size_t>(i)];
    const int a_idx = pair.first;
    const int b_idx = pair.second;
    const int rep_a = q[static_cast<size_t>(a_idx)];
    const int rep_b = q[static_cast<size_t>(b_idx)];
    results[static_cast<size_t>(i)] = ComputeLocalEncodingGain(
        agg_by_idx[static_cast<size_t>(a_idx)].view,
        agg_by_idx[static_cast<size_t>(b_idx)].view, rep_a, rep_b,
        size_by_idx[static_cast<size_t>(a_idx)],
        size_by_idx[static_cast<size_t>(b_idx)],
        self_loops_by_idx[static_cast<size_t>(a_idx)],
        self_loops_by_idx[static_cast<size_t>(b_idx)]);
  }
  return results;
}

std::vector<Sweg::LocalGainResult> Sweg::ScoreCandidatesPersistentCpu(
    const std::vector<std::pair<int, int>>& candidate_pairs,
    const std::vector<int>& q, ExactGainWorkCounters* work) const {
  std::vector<LocalGainResult> results(candidate_pairs.size());
  if (work == nullptr) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 64)
#endif
    for (int i = 0; i < static_cast<int>(candidate_pairs.size()); ++i) {
      const auto pair = candidate_pairs[static_cast<size_t>(i)];
      const QuotientMergeGainResult gain =
          quotient_graph_->ExactMergeGainPersistent(
              q[static_cast<size_t>(pair.first)],
              q[static_cast<size_t>(pair.second)]);
      results[static_cast<size_t>(i)] = {gain.gain, gain.before_cost};
    }
    return results;
  }

#ifdef _OPENMP
  const int thread_count = std::max(1, omp_get_max_threads());
#else
  const int thread_count = 1;
#endif
  struct alignas(64) PaddedWork {
    ExactGainWorkCounters counters;
  };
  std::vector<PaddedWork> thread_work(static_cast<size_t>(thread_count));
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 64)
#endif
  for (int i = 0; i < static_cast<int>(candidate_pairs.size()); ++i) {
#ifdef _OPENMP
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif
    const auto pair = candidate_pairs[static_cast<size_t>(i)];
    const QuotientMergeGainResult gain =
        quotient_graph_->ExactMergeGainPersistent(
            q[static_cast<size_t>(pair.first)],
            q[static_cast<size_t>(pair.second)],
            &thread_work[static_cast<size_t>(tid)].counters);
    results[static_cast<size_t>(i)] = {gain.gain, gain.before_cost};
  }
  for (const PaddedWork& local : thread_work) {
    work->Add(local.counters);
  }
  return results;
}

std::vector<Sweg::LocalGainResult>
Sweg::ScoreCandidatesCertifiedPersistentCpu(
    const std::vector<std::pair<int, int>>& candidate_pairs,
    const std::vector<int>& q,
    const std::vector<EncodingCost>& incident_cost_by_idx, double threshold,
    CertificationWorkCounters* work, double* upper_bound_ms,
    double* exact_ms) const {
  std::vector<LocalGainResult> results(candidate_pairs.size());
  std::vector<EncodingCost> before_costs(candidate_pairs.size(), 0);
  std::vector<uint8_t> upper_passed(candidate_pairs.size(), 0);

  const auto upper_start = std::chrono::steady_clock::now();
  uint64_t entries_available = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+ : entries_available)
#endif
  for (int i = 0; i < static_cast<int>(candidate_pairs.size()); ++i) {
    const auto pair = candidate_pairs[static_cast<size_t>(i)];
    const int rep_a = q[static_cast<size_t>(pair.first)];
    const int rep_b = q[static_cast<size_t>(pair.second)];
    entries_available +=
        static_cast<uint64_t>(quotient_graph_->GetRowView(rep_a).cross_size +
                              quotient_graph_->GetRowView(rep_b).cross_size);
    const EncodingCost cross_cost = quotient_graph_->GetBlockCost(rep_a, rep_b);
    const EncodingCost before =
        incident_cost_by_idx[static_cast<size_t>(pair.first)] +
        incident_cost_by_idx[static_cast<size_t>(pair.second)] - cross_cost;
    before_costs[static_cast<size_t>(i)] = before;
    const MergeGain upper =
        quotient_graph_->MergeGainUpperBound(rep_a, rep_b, before);
    const bool passes =
        upper > 0 &&
        (!ea_use_threshold_ ||
         (before > 0 && static_cast<double>(upper) /
                            static_cast<double>(before) >= threshold));
    upper_passed[static_cast<size_t>(i)] = passes ? 1U : 0U;
    if (!passes) {
      results[static_cast<size_t>(i)] = {0, before};
    }
  }
  const auto upper_end = std::chrono::steady_clock::now();
  if (upper_bound_ms != nullptr) {
    *upper_bound_ms =
        std::chrono::duration<double, std::milli>(upper_end - upper_start)
            .count();
  }

#ifdef _OPENMP
  const int thread_count = std::max(1, omp_get_max_threads());
#else
  const int thread_count = 1;
#endif
  struct alignas(64) PaddedWork {
    CertificationWorkCounters counters;
  };
  std::vector<PaddedWork> thread_work(static_cast<size_t>(thread_count));
  const auto exact_start = std::chrono::steady_clock::now();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 64)
#endif
  for (int i = 0; i < static_cast<int>(candidate_pairs.size()); ++i) {
#ifdef _OPENMP
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif
    CertificationWorkCounters& local =
        thread_work[static_cast<size_t>(tid)].counters;
    ++local.candidates_seen;
    if (upper_passed[static_cast<size_t>(i)] == 0) {
      ++local.upper_bound_pruned;
      continue;
    }
    ++local.upper_bound_passed;
    const auto pair = candidate_pairs[static_cast<size_t>(i)];
    const CertifiedMergeGainResult gain =
        quotient_graph_->CertifiedMergeGainPersistent(
            q[static_cast<size_t>(pair.first)],
            q[static_cast<size_t>(pair.second)],
            before_costs[static_cast<size_t>(i)], threshold,
            ea_use_threshold_, &local, true);
    results[static_cast<size_t>(i)] = {gain.gain, gain.before_cost};
  }
  const auto exact_end = std::chrono::steady_clock::now();
  if (exact_ms != nullptr) {
    *exact_ms =
        std::chrono::duration<double, std::milli>(exact_end - exact_start)
            .count();
  }
  if (work != nullptr) {
    work->entries_available += entries_available;
    for (const PaddedWork& local : thread_work) {
      work->candidates_seen += local.counters.candidates_seen;
      work->upper_bound_pruned += local.counters.upper_bound_pruned;
      work->upper_bound_passed += local.counters.upper_bound_passed;
      work->early_abort_count += local.counters.early_abort_count;
      work->exact_full_scan_count += local.counters.exact_full_scan_count;
      work->entries_scanned += local.counters.entries_scanned;
    }
  }
  return results;
}
