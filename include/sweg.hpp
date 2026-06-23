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

enum class ThresholdPolicy {
  kReciprocal,
  kMagsGeom,
  kAdaptive,
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
                const ThresholdConfig& threshold_config);

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
  const std::vector<int>& supernode_sizes_by_rep() const {
    return supernode_sizes_by_rep_;
  }
  ~Sweg();

 private:
  struct CudaScoringCache;
  using SparseCounts = std::vector<std::pair<int, int64_t>>;

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
    std::vector<SparseCounts> agg_by_idx;
    std::vector<int64_t> size_by_idx;
    std::vector<int64_t> self_loops_by_idx;
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
  LocalGainResult ComputeLocalEncodingGain(const SparseCounts& agg_a,
                                           const SparseCounts& agg_b, int rep_a,
                                           int rep_b, int64_t size_a,
                                           int64_t size_b,
                                           int64_t self_loops_a,
                                           int64_t self_loops_b) const;
  FlatAggCSR BuildFlatAggCsr(const std::vector<SparseCounts>& agg_by_idx,
                             const std::vector<int>& q,
                             const std::vector<int64_t>& size_by_idx,
                             const std::vector<int64_t>& self_loops_by_idx) const;
  std::vector<LocalGainResult> ScoreCandidatesCpu(
      const std::vector<std::pair<int, int>>& candidate_pairs,
      const std::vector<SparseCounts>& agg_by_idx, const std::vector<int>& q,
      const std::vector<int64_t>& size_by_idx,
      const std::vector<int64_t>& self_loops_by_idx) const;
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
  SparseCounts CreateWForSupernodeWithScratch(
      int rep, ParallelScratch& scratch,
      int64_t* self_loop_count = nullptr) const;
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
  RuntimeStats stats_;
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
  mutable ParallelScratch serial_workspace_;
  mutable std::vector<PrepareWorkspace> prepare_workspaces_;
  mutable std::vector<int64_t> scratch_counts_;
  mutable std::vector<uint32_t> scratch_marks_;
  mutable std::vector<int> scratch_touched_;
  mutable uint32_t scratch_epoch_ = 1;
  std::vector<int> active_reps_;
  std::vector<GroupSpan> groups_;
  std::vector<int> dim_order_;
  std::unordered_map<uint64_t, uint64_t> divide_minhash_cache_;
};

const char* MergeModeToString(MergeMode mode);
const char* ScoringBackendToString(ScoringBackend backend);
const char* ThresholdPolicyToString(ThresholdPolicy policy);
