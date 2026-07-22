#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <memory>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

#include "graph_io/graph.hpp"
#include "candidate_index.hpp"
#include "cost_oracle.hpp"
#include "quotient_graph.hpp"

struct EdgePair {
  int first = 0;
  int second = 0;
};

enum class MergeMode {
  kBatchEncodingAware,
  kBatchEncodingAwareBlocked,
};

enum class ScoringBackend {
  kCpu,
  kCuda,
};

enum class CostObjective {
  kLegacy,
  kMagsCompatible,
};

enum class StateBackend {
  kLegacy,
  kPersistent,
};

enum class QuotientUpdateMode {
  kIncremental,
  kBulkRebuild,
  kAuto,
};

enum class CertificationMode {
  kOff,
  kSafe,
};

enum class CommitPolicy {
  kGreedyFull,
  kSequentialOne,
  kTransactional2,
  kTransactional4,
  kTransactional8,
  kMutualBest4,
};

struct CommitAuditRow {
  uint64_t batch_id = 0;
  uint64_t batch_size = 0;
  EncodingCost partition_exact_cost = 0;
  int64_t cumulative_realized_gain = 0;
  uint64_t rejected_pair_count = 0;
};

enum class ThresholdPolicy {
  kReciprocal,
  kMagsGeom,
  kAdaptive,
};

// Off avoids allocating or updating any per-iteration profiling state.
enum class ProfilingMode {
  kOff,
  kSummary,
  kRounds,
};

struct IterationProfile {
  int iteration = 0;
  int64_t active_supernodes = 0;
  int64_t group_count = 0;
  int64_t sum_group_sizes = 0;
  uint64_t candidate_proxy_pairs_examined = 0;
  uint64_t candidate_pairs_after_topk = 0;
  uint64_t candidate_pairs_submitted_for_scoring = 0;
  uint64_t candidate_pairs_scored = 0;
  uint64_t exact_gain_calls = 0;
  uint64_t exact_gain_positive_count = 0;
  uint64_t above_threshold_count = 0;
  uint64_t matching_selected_count = 0;
  uint64_t actual_merge_count = 0;
  uint64_t prepare_original_edges_scanned = 0;
  uint64_t prepare_aggregated_nnz = 0;
  uint64_t exact_gain_input_nnz = 0;
  uint64_t update_partition_nodes_touched = 0;
  uint64_t update_touched_quotient_entries = 0;
  uint64_t prepare_row_entries_copied = 0;
  uint64_t prepare_row_copy_bytes = 0;
  uint64_t prepare_row_views_created = 0;
  uint64_t prepare_unique_representatives = 0;
  uint64_t prepare_row_acquisition_requests = 0;
  uint64_t prepare_row_acquisition_hits = 0;
  uint64_t prepare_row_acquisition_misses = 0;
  uint64_t prepare_duplicate_acquisitions_avoided = 0;
  uint64_t prepare_row_registry_entries = 0;
  uint64_t prepare_row_registry_bytes = 0;
  uint64_t exact_persistent_pairs = 0;
  uint64_t exact_raw_entries_a = 0;
  uint64_t exact_raw_entries_b = 0;
  uint64_t exact_union_neighbors = 0;
  uint64_t exact_overlap_neighbors = 0;
  uint64_t exact_single_sided_neighbors = 0;
  uint64_t exact_internal_block_terms = 0;
  uint64_t exact_block_cost_evaluations = 0;
  uint64_t exact_capacity_multiplications = 0;
  double candidate_index_build_ms = 0.0;
  double candidate_index_refresh_ms = 0.0;
  double candidate_proposal_ms = 0.0;
  uint64_t candidate_proposals_raw = 0;
  uint64_t candidate_proposals_unique = 0;
  uint64_t candidate_duplicates_removed = 0;
  uint64_t candidate_budget_exhausted_nodes = 0;
  uint64_t candidate_nodes_with_zero_proposals = 0;
  uint64_t candidate_direct_neighbor_count = 0;
  uint64_t candidate_shared_neighbor_count = 0;
  uint64_t candidate_exploration_count = 0;
  double residual_signature_build_ms = 0.0;
  double residual_signature_refresh_ms = 0.0;
  uint64_t residual_signature_rows_scanned = 0;
  uint64_t residual_signature_features_created = 0;
  uint64_t residual_signature_cache_hits = 0;
  uint64_t residual_signature_cache_misses = 0;
  uint64_t residual_bucket_count = 0;
  uint64_t residual_bucket_max_size = 0;
  uint64_t residual_bucket_candidates_considered = 0;
  uint64_t residual_bucket_candidates_dropped_by_cap = 0;
  int64_t residual_alignment_score_sum = 0;
  int64_t residual_conflict_penalty_sum = 0;
  int64_t residual_direct_score_sum = 0;
  int64_t residual_size_compatibility_sum = 0;
  uint64_t certification_candidates_seen = 0;
  uint64_t upper_bound_pruned = 0;
  uint64_t upper_bound_passed = 0;
  uint64_t early_abort_count = 0;
  uint64_t exact_full_scan_count = 0;
  uint64_t exact_entries_available = 0;
  uint64_t exact_entries_scanned = 0;
  uint64_t exact_entries_skipped = 0;
  double upper_bound_ms = 0.0;
  double early_abort_exact_ms = 0.0;
  double divide_ms = 0.0;
  double prepare_ms = 0.0;
  double candidate_discovery_task_sum_ms = 0.0;
  double exact_gain_ms = 0.0;
  double matching_ms = 0.0;
  double update_ms = 0.0;
  double quotient_row_lookup_ms = 0.0;
  double quotient_row_copy_ms = 0.0;
  double quotient_exact_gain_ms = 0.0;
  double quotient_incremental_update_ms = 0.0;
  double quotient_reciprocal_update_ms = 0.0;
  double quotient_memory_allocation_ms = 0.0;
  double quotient_sort_or_merge_ms = 0.0;
  double quotient_rebuild_ms = 0.0;
  uint64_t quotient_rows_updated = 0;
  uint64_t quotient_entries_inserted = 0;
  uint64_t quotient_entries_removed = 0;
  uint64_t quotient_entries_shifted = 0;
  uint64_t quotient_high_degree_rows_touched = 0;
  uint64_t quotient_max_row_degree = 0;
  uint64_t quotient_allocated_bytes = 0;
  uint64_t quotient_peak_nnz = 0;
};

struct RuntimeProfile {
  ProfilingMode mode = ProfilingMode::kOff;
  int64_t iterations_observed = 0;
  int64_t active_supernodes_last = 0;
  int64_t group_count_last = 0;
  int64_t sum_group_sizes_last = 0;
  std::vector<IterationProfile> rounds;
  IterationProfile current;
  IterationProfile total;
  double profiling_divide_ms = 0.0;
  double profiling_prepare_ms = 0.0;
  double profiling_candidate_discovery_task_sum_ms = 0.0;
  double profiling_exact_gain_ms = 0.0;
  double profiling_matching_ms = 0.0;
  double profiling_update_ms = 0.0;
};

struct ThresholdConfig {
  ThresholdPolicy policy = ThresholdPolicy::kReciprocal;
  double high = 0.5;
  double low = 0.005;
  double min_low = 0.005;
  double q_high = 0.85;
  double q_low = 0.15;
  int sample_limit = 4096;
  double acceptance_target = 0.15;
  double acceptance_down = 0.85;
  double acceptance_up = 1.05;
  double acceptance_scale_min = 0.25;
  double acceptance_scale_max = 4.0;
};

struct EncodingResult {
  std::vector<std::vector<int>> supernodes;
  std::vector<int> supernode_sizes;
  std::vector<int> node_to_supernode;
  std::vector<EdgePair> P;
  std::vector<EdgePair> Cp;
  std::vector<EdgePair> Cm;
};

struct RuntimeStats {
  double runtime_run_ms = 0.0;
  double runtime_divide_ms = 0.0;
  double runtime_merge_ms = 0.0;
  double runtime_encode_ms = 0.0;
  double runtime_output_ms = 0.0;
  int64_t divide_max_group_size = 0;
  int64_t divide_fallback_splits = 0;

  double merge_prepare_wall_ms = 0.0;
  double merge_prepare_task_sum_ms = 0.0;
  double merge_create_w_ms = 0.0;
  double merge_scoring_ms = 0.0;
  double merge_candidate_gen_ms = 0.0;
  double merge_gain_scoring_ms = 0.0;
  double merge_selection_ms = 0.0;
  double merge_update_ms = 0.0;
  int64_t merge_candidate_pairs = 0;
  int64_t merge_selected_pairs = 0;
  int64_t merge_total_local_gain = 0;
  uint64_t merge_group_count = 0;
  uint64_t merge_group_max_size = 0;
  uint64_t merge_raw_pair_count = 0;
  uint64_t merge_candidate_pairs_after_prune = 0;
  uint64_t merge_exact_gain_calls = 0;
  uint64_t merge_positive_gain_pairs = 0;
  int64_t isolated_gain_sum = 0;
  int64_t realized_marginal_gain_sum = 0;
  int64_t actual_batch_cost_reduction = 0;
  int64_t interaction_delta = 0;
  uint64_t selected_merges_for_validation = 0;
  uint64_t accepted_merges_after_validation = 0;
  uint64_t rejected_nonpositive = 0;
  uint64_t stale_endpoint = 0;
  uint64_t gain_decreased = 0;
  uint64_t gain_increased = 0;
  uint64_t negative_marginal = 0;
  uint64_t candidate_refresh_count = 0;
  uint64_t original_exact_calls = 0;
  uint64_t validation_exact_calls = 0;
  uint64_t validation_exact_row_entry_work = 0;
  double commit_validation_ms = 0.0;
  double audit_oracle_ms = 0.0;
  uint64_t update_touched_quotient_entries = 0;
  uint64_t quotient_nnz_final = 0;
  uint64_t quotient_incremental_batch_count = 0;
  uint64_t quotient_bulk_rebuild_count = 0;
  uint64_t prepare_row_entries_copied = 0;
  uint64_t prepare_row_copy_bytes = 0;
  uint64_t prepare_row_views_created = 0;
  uint64_t prepare_unique_representatives = 0;
  uint64_t prepare_row_acquisition_requests = 0;
  uint64_t prepare_row_acquisition_hits = 0;
  uint64_t prepare_row_acquisition_misses = 0;
  uint64_t prepare_duplicate_acquisitions_avoided = 0;
  uint64_t prepare_row_registry_entries = 0;
  uint64_t prepare_row_registry_bytes = 0;
  uint64_t exact_persistent_pairs = 0;
  uint64_t exact_raw_entries_a = 0;
  uint64_t exact_raw_entries_b = 0;
  uint64_t exact_union_neighbors = 0;
  uint64_t exact_overlap_neighbors = 0;
  uint64_t exact_single_sided_neighbors = 0;
  uint64_t exact_internal_block_terms = 0;
  uint64_t exact_block_cost_evaluations = 0;
  uint64_t exact_capacity_multiplications = 0;
  double candidate_index_build_ms = 0.0;
  double candidate_index_refresh_ms = 0.0;
  double candidate_proposal_ms = 0.0;
  uint64_t candidate_proposals_raw = 0;
  uint64_t candidate_proposals_unique = 0;
  uint64_t candidate_duplicates_removed = 0;
  uint64_t candidate_budget_exhausted_nodes = 0;
  uint64_t candidate_nodes_with_zero_proposals = 0;
  uint64_t candidate_direct_neighbor_count = 0;
  uint64_t candidate_shared_neighbor_count = 0;
  uint64_t candidate_exploration_count = 0;
  double residual_signature_build_ms = 0.0;
  double residual_signature_refresh_ms = 0.0;
  uint64_t residual_signature_rows_scanned = 0;
  uint64_t residual_signature_features_created = 0;
  uint64_t residual_signature_cache_hits = 0;
  uint64_t residual_signature_cache_misses = 0;
  uint64_t residual_bucket_count = 0;
  uint64_t residual_bucket_max_size = 0;
  uint64_t residual_bucket_candidates_considered = 0;
  uint64_t residual_bucket_candidates_dropped_by_cap = 0;
  int64_t residual_alignment_score_sum = 0;
  int64_t residual_conflict_penalty_sum = 0;
  int64_t residual_direct_score_sum = 0;
  int64_t residual_size_compatibility_sum = 0;
  uint64_t certification_candidates_seen = 0;
  uint64_t upper_bound_pruned = 0;
  uint64_t upper_bound_passed = 0;
  uint64_t early_abort_count = 0;
  uint64_t exact_full_scan_count = 0;
  uint64_t exact_entries_available = 0;
  uint64_t exact_entries_scanned = 0;
  uint64_t exact_entries_skipped = 0;
  double upper_bound_ms = 0.0;
  double early_abort_exact_ms = 0.0;
  double quotient_row_lookup_ms = 0.0;
  double quotient_row_copy_ms = 0.0;
  double quotient_exact_gain_ms = 0.0;
  double quotient_incremental_update_ms = 0.0;
  double quotient_reciprocal_update_ms = 0.0;
  double quotient_memory_allocation_ms = 0.0;
  double quotient_sort_or_merge_ms = 0.0;
  double quotient_rebuild_ms = 0.0;
  uint64_t quotient_rows_updated = 0;
  uint64_t quotient_entries_inserted = 0;
  uint64_t quotient_entries_removed = 0;
  uint64_t quotient_entries_shifted = 0;
  uint64_t quotient_high_degree_rows_touched = 0;
  uint64_t quotient_max_row_degree = 0;
  uint64_t quotient_allocated_bytes = 0;
  uint64_t quotient_peak_nnz = 0;
  uint64_t merge_rejected_by_overlap = 0;
  uint64_t merge_rejected_by_threshold = 0;
  double merge_exact_gain_calls_per_selected = 0.0;
  double merge_positive_gain_ratio = 0.0;
  double threshold_last = 0.0;
  double threshold_geom_last = 0.0;
  double threshold_adaptive_last = 0.0;
  double threshold_acceptance_scale = 1.0;
  double threshold_acceptance_rate_last = 0.0;
  uint64_t threshold_sample_count_last = 0;
  double cuda_init_ms = 0.0;
  double cuda_h2d_ms = 0.0;
  double cuda_row_h2d_ms = 0.0;
  double cuda_pair_h2d_ms = 0.0;
  double cuda_kernel_ms = 0.0;
  double cuda_d2h_ms = 0.0;
  double cuda_total_ms = 0.0;
  double cuda_packed_h2d_ms = 0.0;
  double cuda_packed_d2h_ms = 0.0;
  int64_t cuda_num_calls = 0;
  int64_t cuda_row_uploads = 0;
  int64_t cuda_kernel_launches = 0;
  int64_t cuda_h2d_bytes = 0;
  int64_t cuda_d2h_bytes = 0;
  int64_t cuda_max_candidates_per_launch = 0;
  int64_t cuda_packed_h2d_calls = 0;
  int64_t cuda_packed_d2h_calls = 0;
  int64_t cuda_packed_input_bytes = 0;
  int64_t cuda_packed_output_bytes = 0;
  int64_t cuda_slice_count = 0;
  int64_t cuda_blocks_single_slice = 0;
  int64_t cuda_blocks_multi_slice = 0;
  int64_t cuda_max_slice_rows = 0;
  int64_t cuda_max_slice_nnz = 0;
  int64_t cuda_max_slice_candidates = 0;
  int64_t cuda_slice_memory_budget_bytes = 0;

  int64_t overflow_groups_seen = 0;
  int64_t overflow_refined_subgroups = 0;
  int64_t overflow_forced_chunks = 0;
  int64_t overflow_max_group_before = 0;
  int64_t overflow_max_group_after = 0;
  double overflow_refine_ms = 0.0;

  double encode_relabel_ms = 0.0;
  double encode_count_edges_ms = 0.0;
  double encode_decision_ms = 0.0;
  double encode_correction_generation_ms = 0.0;
};

class Sweg {
 public:
  struct LocalGainResult {
    int64_t gain = 0;
    int64_t before_cost = 0;
  };

  struct FlatAggCSR {
    std::vector<int> offsets;
    std::vector<int> row_rep_ids;
    std::vector<int> neighbors;
    std::vector<int64_t> weights;
    std::vector<int64_t> neighbor_sizes;
    std::vector<int64_t> row_sizes;
    std::vector<int64_t> row_self_loops;
  };

  explicit Sweg(const CSRGraph& graph, MergeMode merge_mode, int top_k,
                bool ea_use_threshold, uint64_t seed,
                ScoringBackend scoring_backend, bool verify_cuda_gain,
                int group_batch_size, int candidate_batch_budget,
                int cuda_slice_memory_mb,
                int overflow_group_gmax, int overflow_refine_rounds,
                int divide_hash_dims, int divide_max_group,
                const ThresholdConfig& threshold_config,
                ProfilingMode profiling_mode = ProfilingMode::kOff,
                CostObjective cost_objective = CostObjective::kLegacy,
                StateBackend state_backend = StateBackend::kLegacy,
                bool validate_quotient = false,
                QuotientUpdateMode quotient_update_mode =
                    QuotientUpdateMode::kIncremental,
                CandidateIndexMode candidate_index_mode =
                    CandidateIndexMode::kLegacy,
                int candidate_budget = 8,
                CertificationMode certification_mode =
                    CertificationMode::kOff,
                CommitPolicy commit_policy = CommitPolicy::kGreedyFull,
                bool commit_audit = false);

  void Run(int iterations, int print_offset);
  void Divide(int iter);
  void Merge(int iter, int total_iterations);
  EncodingResult Encode();
  void Drop(double error_bound, EncodingResult& result) const;
  void WriteOutput(const std::string& out_dir,
                   const EncodingResult& result) const;
  void Update_S(int A, int B);

  int CountGroups() const;
  int CountActiveSupernodes() const;
  int gstart() const { return gstart_; }
  const RuntimeStats& runtime_stats() const { return stats_; }
  const RuntimeProfile& runtime_profile() const { return runtime_profile_; }
  const std::vector<CommitAuditRow>& commit_audit_rows() const {
    return commit_audit_rows_;
  }
  uint64_t PartitionHash() const;
  const std::vector<int>& supernode_sizes_by_rep() const {
    return supernode_sizes_by_rep_;
  }
  EncodingCost ExactCurrentPartitionCost() const;
  ~Sweg();

 private:
  struct CudaScoringCache;
  using SparseCounts = std::vector<std::pair<int, int64_t>>;

  struct PreparedRow {
    SparseCounts owned;
    QuotientRowView view;

    void SetOwned(SparseCounts row) {
      owned = std::move(row);
      view = QuotientRowView::Raw(owned.data(), owned.size());
    }
    void SetView(QuotientRowView row_view) {
      owned.clear();
      view = row_view;
    }
    size_t size() const { return view.size(); }
    auto begin() const { return view.begin(); }
    auto end() const { return view.end(); }
  };

  struct GroupSpan {
    int start = 0;
    int length = 0;
  };

  struct EaCandidatePair {
    int a_idx = -1;
    int b_idx = -1;
    int64_t gain = 0;
    int64_t before_cost = 0;
  };

  struct EaGroupPrepared {
    std::vector<int> q;
    std::vector<PreparedRow> agg_by_idx;
    std::vector<int64_t> size_by_idx;
    std::vector<int64_t> self_loops_by_idx;
    std::vector<EncodingCost> incident_cost_by_idx;
    std::vector<EaCandidatePair> candidate_pairs;
  };

  struct ParallelScratch {
    std::vector<int64_t> counts;
    std::vector<uint32_t> marks;
    std::vector<int> touched;
    uint32_t epoch = 1;
  };

  struct PrepareWorkspace {
    ParallelScratch scratch;
  };

  struct EaPrepareStats {
    double wall_ms = 0.0;
    double create_w_ms = 0.0;
    double candidate_gen_ms = 0.0;
    uint64_t group_count = 0;
    uint64_t group_max_size = 0;
    uint64_t raw_pair_count = 0;
    uint64_t candidate_pairs_after_prune = 0;
    uint64_t candidate_proxy_pairs_examined = 0;
    uint64_t prepare_original_edges_scanned = 0;
    uint64_t prepare_aggregated_nnz = 0;
    uint64_t prepare_row_entries_copied = 0;
    uint64_t prepare_row_copy_bytes = 0;
    uint64_t prepare_row_views_created = 0;
    double quotient_row_lookup_ms = 0.0;
    double quotient_row_copy_ms = 0.0;
    double candidate_proposal_ms = 0.0;
    CandidateIndexStats candidate_index;
  };

  struct EaPrepareResult {
    EaGroupPrepared prepared;
    EaPrepareStats stats;
  };

  struct OverflowRefineCounters {
    int64_t overflow_groups_seen = 0;
    int64_t refined_subgroups = 0;
    int64_t forced_chunks = 0;
    int64_t max_group_before = 0;
    int64_t max_group_after = 0;
  };

  struct CudaBlockedBatch {
    struct Destination {
      size_t group_idx = 0;
      size_t candidate_idx = 0;
    };

    FlatAggCSR flat_agg;
    std::vector<std::pair<int, int>> candidate_pairs;
    std::vector<Destination> destinations;
  };

  struct CudaSliceSpan {
    size_t begin = 0;
    size_t end = 0;
    size_t rows = 0;
    size_t nnz = 0;
    size_t candidates = 0;
    size_t estimated_bytes = 0;
  };

  void ShuffleArray();
  void ResetRuntimeStats();
  void RecordIterationProfile(int iter);
  void FinalizeIterationProfile();
  bool ProfilingEnabled() const {
    return runtime_profile_.mode != ProfilingMode::kOff;
  }
  int NodeShingle(int v) const;
  uint64_t HashRank64(int node, int dim, int iter) const;
  uint64_t DimRank(int dim, int iter) const;
  uint64_t FallbackRank(int rep, int iter) const;
  uint64_t NodeShingleDim(int v, int dim, int iter) const;
  uint64_t SupernodeMinHash(int rep, int dim, int iter);
  void FallbackSplitGroup(std::vector<int>& reps, int l, int r, int iter);
  void SplitGroupByHashDims(std::vector<int>& reps, int l, int r, int depth,
                            int iter);
  int SupernodeLength(int key) const;
  std::vector<int> Recover_S(int key) const;
  std::vector<GroupSpan> BuildGroups() const;
  SparseCounts CreateWForSupernode(int rep,
                                   int64_t* self_loop_count = nullptr) const;
  double GeometricThreshold(int iter, int total_iterations) const;
  double AdaptiveQuantileForIter(int iter, int total_iterations) const;
  double Quantile(std::vector<double> values, double q) const;
  double ComputeMergeThreshold(int iter, int total_iterations);
  void BeginMergeIteration();
  void EndMergeIteration();
  void RecordSavingSample(double saving_ratio);
  void MergeBatchEncodingAwareGroup(const std::vector<int>& q, double threshold);
  std::vector<std::pair<int, int>> SelectBatchEncodingAwareGroupPairs(
      const std::vector<int>& q, double threshold);
  std::vector<std::vector<int>> RefineOverflowGroup(
      const std::vector<int>& q, OverflowRefineCounters* counters);
  EaGroupPrepared PrepareBatchEncodingAwareGroup(const std::vector<int>& q);
  EaPrepareResult PrepareBatchEncodingAwareGroupWithWorkspace(
      const std::vector<int>& q, PrepareWorkspace* workspace,
      bool allow_inner_parallel);
  void ScoreBatchEncodingAwarePreparedGroup(EaGroupPrepared* prepared,
                                            double threshold);
  void AssignGainResultsToPreparedGroup(
      EaGroupPrepared* prepared,
      const std::vector<LocalGainResult>& gain_results) const;
  void FilterPreparedCandidatePairsByThreshold(EaGroupPrepared* prepared,
                                               double threshold);
  std::vector<std::pair<int, int>> SelectScoredBatchEncodingAwareGroupPairs(
      const EaGroupPrepared& prepared);
  void ScoreBatchEncodingAwarePreparedGroupsBlocked(
      std::vector<EaGroupPrepared>* prepared_groups, double threshold);
  void ScoreBatchEncodingAwarePreparedGroupsBlockedCuda(
      std::vector<EaGroupPrepared>* prepared_groups, double threshold,
      size_t candidate_chunk_size);
  void ScoreBatchEncodingAwarePreparedGroupsBlockedRange(
      std::vector<EaGroupPrepared>* prepared_groups, size_t begin, size_t end,
      double threshold);
  void ScoreBatchEncodingAwarePreparedGroupsBlockedCudaRange(
      std::vector<EaGroupPrepared>* prepared_groups, size_t begin, size_t end,
      double threshold, size_t candidate_chunk_size);
  CudaBlockedBatch BuildCudaBlockedBatch(
      const std::vector<EaGroupPrepared>& groups) const;
  CudaBlockedBatch BuildCudaBlockedBatch(
      const std::vector<EaGroupPrepared>& groups, size_t begin,
      size_t end) const;
  SparseCounts AggregateWBySupernode(const SparseCounts& w) const;
  int64_t EdgeCountToSupernode(const SparseCounts& aggregated, int target_rep,
                               bool self_loop) const;
  int64_t EncodeCostForPair(int rep_u, int64_t size_u, int rep_x,
                            int64_t size_x, int64_t edges) const;
  LocalGainResult ComputeMagsCompatibleGain(
      const QuotientRowView& agg_a, const QuotientRowView& agg_b, int rep_a,
      int rep_b, int64_t size_a, int64_t size_b) const;
  std::vector<int> CurrentPartitionLabels() const;
  bool ValidateMergeAgainstCurrentPartition(int rep_a, int rep_b) const;
  void CommitSelectedPairs(const std::vector<std::pair<int, int>>& pairs);
  void CommitSelectedPairsTransactional(
      const std::vector<std::pair<int, int>>& pairs);
  LocalGainResult ComputeLocalEncodingGain(const QuotientRowView& agg_a,
                                           const QuotientRowView& agg_b, int rep_a,
                                           int rep_b, int64_t size_a,
                                           int64_t size_b,
                                           int64_t self_loops_a,
                                           int64_t self_loops_b) const;
  FlatAggCSR BuildFlatAggCsr(const std::vector<PreparedRow>& agg_by_idx,
                             const std::vector<int>& q,
                             const std::vector<int64_t>& size_by_idx,
                             const std::vector<int64_t>& self_loops_by_idx) const;
  std::vector<LocalGainResult> ScoreCandidatesCpu(
      const std::vector<std::pair<int, int>>& candidate_pairs,
      const std::vector<PreparedRow>& agg_by_idx, const std::vector<int>& q,
      const std::vector<int64_t>& size_by_idx,
      const std::vector<int64_t>& self_loops_by_idx) const;
  std::vector<LocalGainResult> ScoreCandidatesPersistentCpu(
      const std::vector<std::pair<int, int>>& candidate_pairs,
      const std::vector<int>& q, ExactGainWorkCounters* work) const;
  std::vector<LocalGainResult> ScoreCandidatesCertifiedPersistentCpu(
      const std::vector<std::pair<int, int>>& candidate_pairs,
      const std::vector<int>& q,
      const std::vector<EncodingCost>& incident_cost_by_idx,
      double threshold, CertificationWorkCounters* work,
      double* upper_bound_ms, double* exact_ms) const;
  std::vector<LocalGainResult> ScoreCandidatesCuda(
      const std::vector<std::pair<int, int>>& candidate_pairs,
      const FlatAggCSR& flat_agg, size_t candidate_chunk_size);
  static uint64_t SplitMix64(uint64_t x);
  uint64_t OverflowNodeRank(int node, uint64_t seed) const;
  int OverflowNodeShingle(int node, uint64_t seed) const;
  int OverflowSupernodeShingle(int rep, uint64_t seed) const;
  void ResetScratch() const;
  void ResetParallelScratch(ParallelScratch& scratch) const;
  void EnsurePrepareWorkspace(PrepareWorkspace* workspace) const;
  void EnsurePrepareWorkspaces(int thread_count) const;
  void AccumulatePrepareStats(const EaPrepareStats& prepare_stats);
  void RecordPreparedRowAcquisitionAudit(
      const std::vector<std::vector<int>>& work_items);
  void RefreshCandidateIndex(uint64_t epoch_seed);
  void AccumulateCandidateIndexStats(const CandidateIndexStats& index_stats);
  SparseCounts CreateWForSupernodeWithScratch(
      int rep, ParallelScratch& scratch,
      int64_t* self_loop_count = nullptr,
      uint64_t* original_edges_scanned = nullptr) const;
  SparseCounts AggregateWBySupernodeWithScratch(
      const SparseCounts& w, ParallelScratch& scratch) const;
  size_t EstimateCudaSliceBytes(size_t total_rows, size_t total_nnz,
                                size_t total_candidates,
                                size_t candidate_chunk_size) const;
  std::vector<CudaSliceSpan> BuildCudaSlicePlan(
      const std::vector<EaGroupPrepared>& prepared_groups,
      size_t candidate_chunk_size);
  void EnsureCudaScoringCache();
  size_t GetCudaSliceMemoryBudgetBytes();
  size_t QueryCudaSliceMemoryBudgetBytes();
  const CSRGraph* graph_;
  int n_;
  std::vector<int> h_;
  std::vector<int> S_;
  std::vector<int> I_;
  std::vector<int> J_;
  std::vector<int> F_;
  std::vector<int> G_;
  std::vector<int> supernode_sizes_by_rep_;
  int gstart_ = 0;
  MergeMode merge_mode_;
  int top_k_;
  bool ea_use_threshold_;
  ScoringBackend scoring_backend_;
  bool verify_cuda_gain_;
  int group_batch_size_;
  int candidate_batch_budget_;
  int cuda_slice_memory_mb_;
  int overflow_group_gmax_;
  int overflow_refine_rounds_;
  int divide_hash_dims_;
  int divide_max_group_;
  ThresholdConfig threshold_config_;
  CostObjective cost_objective_;
  CostOracle cost_oracle_;
  StateBackend state_backend_;
  bool validate_quotient_;
  QuotientUpdateMode quotient_update_mode_;
  CandidateIndexMode candidate_index_mode_ = CandidateIndexMode::kLegacy;
  int candidate_budget_ = 8;
  CertificationMode certification_mode_ = CertificationMode::kOff;
  CommitPolicy commit_policy_ = CommitPolicy::kGreedyFull;
  bool commit_audit_ = false;
  uint64_t commit_batch_id_ = 0;
  std::vector<CommitAuditRow> commit_audit_rows_;
  std::unordered_map<uint64_t, MergeGain> selected_isolated_gain_;
  std::unique_ptr<CandidateIndex> candidate_index_;
  bool candidate_index_built_ = false;
  bool quotient_batch_precommitted_ = false;
  std::unique_ptr<QuotientGraph> quotient_graph_;
  RuntimeStats stats_;
  RuntimeProfile runtime_profile_;
  double threshold_acceptance_scale_ = 1.0;
  std::vector<double> prev_positive_savings_;
  std::vector<double> cur_positive_savings_;
  uint64_t cur_positive_savings_seen_ = 0;
  uint64_t cur_tested_pairs_ = 0;
  uint64_t cur_accepted_pairs_ = 0;
  uint64_t seed64_;
  std::mt19937 rng_;
  std::shared_ptr<CudaScoringCache> cuda_cache_;
  size_t cuda_verify_samples_remaining_ = 0;
  uint64_t cuda_verify_call_counter_ = 0;
  mutable PrepareWorkspace sequential_prepare_workspace_;
  mutable ParallelScratch serial_workspace_;
  mutable std::vector<PrepareWorkspace> prepare_workspaces_;
  mutable std::vector<int64_t> scratch_counts_;
  mutable std::vector<uint32_t> scratch_marks_;
  mutable std::vector<int> scratch_touched_;
  std::vector<uint32_t> acquisition_audit_marks_;
  uint32_t acquisition_audit_epoch_ = 1;
  mutable uint32_t scratch_epoch_ = 1;
  std::vector<int> active_reps_;
  std::vector<int> candidate_active_reps_;
  std::vector<GroupSpan> groups_;
  std::vector<int> dim_order_;
  std::unordered_map<uint64_t, uint64_t> divide_minhash_cache_;
};

const char* MergeModeToString(MergeMode mode);
const char* ScoringBackendToString(ScoringBackend backend);
const char* ThresholdPolicyToString(ThresholdPolicy policy);
const char* CommitPolicyToString(CommitPolicy policy);
