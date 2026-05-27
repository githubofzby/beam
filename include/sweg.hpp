#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "graph_io/graph.hpp"

struct EdgePair {
  int first = 0;
  int second = 0;
};

enum class MergeMode {
  kSequential,
  kBatchSuperJaccard,
  kBatchEncodingAware,
  kBatchEncodingAwareFrozen,
  kBatchEncodingAwareBlocked,
};

enum class ScoringBackend {
  kCpu,
  kCuda,
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
  double runtime_total_ms = 0.0;
  double runtime_divide_ms = 0.0;
  double runtime_merge_ms = 0.0;
  double runtime_encode_ms = 0.0;
  double runtime_output_ms = 0.0;

  double merge_create_w_ms = 0.0;
  double merge_scoring_ms = 0.0;
  double merge_candidate_gen_ms = 0.0;
  double merge_gain_scoring_ms = 0.0;
  double merge_selection_ms = 0.0;
  double merge_update_ms = 0.0;
  int64_t merge_candidate_pairs = 0;
  int64_t merge_selected_pairs = 0;
  int64_t merge_total_local_gain = 0;
  double cuda_h2d_ms = 0.0;
  double cuda_kernel_ms = 0.0;
  double cuda_d2h_ms = 0.0;
  double cuda_total_ms = 0.0;
  int64_t cuda_num_calls = 0;

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
                int overflow_group_gmax, int overflow_refine_rounds);

  void Run(int iterations, int print_offset);
  void Divide();
  void Merge(int iter);
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
    int shingle = -1;
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

  struct OverflowRefineCounters {
    int64_t overflow_groups_seen = 0;
    int64_t refined_subgroups = 0;
    int64_t forced_chunks = 0;
    int64_t max_group_before = 0;
    int64_t max_group_after = 0;
  };

  void ShuffleArray();
  void ResetRuntimeStats();
  int NodeShingle(int v) const;
  int SupernodeLength(int key) const;
  std::vector<int> Recover_S(int key) const;
  std::vector<GroupSpan> BuildGroups() const;
  SparseCounts CreateWForSupernode(int rep,
                                   int64_t* self_loop_count = nullptr) const;
  SparseCounts UpdateW(const SparseCounts& w_a, const SparseCounts& w_b) const;
  void MergeSequentialGroup(const std::vector<int>& q, double threshold);
  void MergeBatchSuperJaccardGroup(const std::vector<int>& q, double threshold);
  void MergeBatchEncodingAwareGroup(const std::vector<int>& q, double threshold);
  std::vector<std::pair<int, int>> SelectBatchEncodingAwareGroupPairs(
      const std::vector<int>& q, double threshold);
  std::vector<std::vector<int>> RefineOverflowGroup(
      const std::vector<int>& q, OverflowRefineCounters* counters);
  EaGroupPrepared PrepareBatchEncodingAwareGroup(const std::vector<int>& q);
  void ScoreBatchEncodingAwarePreparedGroup(EaGroupPrepared* prepared,
                                            double threshold);
  std::vector<std::pair<int, int>> SelectScoredBatchEncodingAwareGroupPairs(
      const EaGroupPrepared& prepared);
  void ScoreBatchEncodingAwarePreparedGroupsBlocked(
      std::vector<EaGroupPrepared>* prepared_groups, double threshold);
  void ScoreBatchEncodingAwarePreparedGroupsBlockedCudaBudgeted(
      std::vector<EaGroupPrepared>* prepared_groups, double threshold);
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
      const FlatAggCSR& flat_agg);
  std::pair<double, double> JacSimSavCost(const SparseCounts& w_a,
                                          const SparseCounts& w_b) const;
  double WeightedJaccard(const SparseCounts& w_a,
                         const SparseCounts& w_b) const;
  double CostSaving(const SparseCounts& w_a, const SparseCounts& w_b,
                    int rep_a, int rep_b) const;
  uint64_t SplitMix64(uint64_t x) const;
  uint64_t OverflowNodeRank(int node, uint64_t seed) const;
  int OverflowNodeShingle(int node, uint64_t seed) const;
  int OverflowSupernodeShingle(int rep, uint64_t seed) const;
  void ResetScratch() const;
  void ResetParallelScratch(ParallelScratch& scratch) const;
  void EnsureParallelWorkspaces(int thread_count) const;
  SparseCounts CreateWForSupernodeWithScratch(
      int rep, ParallelScratch& scratch,
      int64_t* self_loop_count = nullptr) const;
  SparseCounts AggregateWBySupernodeWithScratch(
      const SparseCounts& w, ParallelScratch& scratch) const;
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
  int overflow_group_gmax_;
  int overflow_refine_rounds_;
  RuntimeStats stats_;
  std::mt19937 rng_;
  std::shared_ptr<CudaScoringCache> cuda_cache_;
  size_t cuda_verify_samples_remaining_ = 0;
  mutable ParallelScratch serial_workspace_;
  mutable std::vector<ParallelScratch> parallel_workspaces_;
  mutable std::vector<int64_t> scratch_counts_;
  mutable std::vector<uint32_t> scratch_marks_;
  mutable std::vector<int> scratch_touched_;
  mutable uint32_t scratch_epoch_ = 1;
};

const char* MergeModeToString(MergeMode mode);
const char* ScoringBackendToString(ScoringBackend backend);
