#include "sweg.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

Sweg::FlatAggCSR Sweg::BuildFlatAggCsr(
    const std::vector<SparseCounts>& agg_by_idx, const std::vector<int>& q,
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
    const std::vector<SparseCounts>& agg_by_idx, const std::vector<int>& q,
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
        agg_by_idx[static_cast<size_t>(a_idx)],
        agg_by_idx[static_cast<size_t>(b_idx)], rep_a, rep_b,
        size_by_idx[static_cast<size_t>(a_idx)],
        size_by_idx[static_cast<size_t>(b_idx)],
        self_loops_by_idx[static_cast<size_t>(a_idx)],
        self_loops_by_idx[static_cast<size_t>(b_idx)]);
  }
  return results;
}
