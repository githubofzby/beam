#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "quotient_graph.hpp"

enum class CandidateIndexMode {
  kLegacy,
  kQuotientNeighbor,
  kResidualSignature,
};

enum class CandidateSource : uint8_t {
  kDirectNeighbor,
  kSharedNeighbor,
  kExploration,
  kResidualSignature,
};

struct CandidateProposal {
  int source = -1;
  int target = -1;
  CandidateSource source_type = CandidateSource::kDirectNeighbor;
  int64_t cheap_score = 0;
};

struct CandidateIndexStats {
  uint64_t proposals_raw = 0;
  uint64_t proposals_unique = 0;
  uint64_t duplicates_removed = 0;
  uint64_t budget_exhausted_nodes = 0;
  uint64_t nodes_with_zero_proposals = 0;
  uint64_t direct_neighbor_count = 0;
  uint64_t shared_neighbor_count = 0;
  uint64_t exploration_count = 0;
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
};

class CandidateIndex {
 public:
  virtual ~CandidateIndex() = default;
  virtual void BuildOrRefresh(const QuotientGraph& quotient,
                              const std::vector<int>& active_reps,
                              uint64_t seed) = 0;
  virtual void Propose(int source, int budget,
                       std::vector<CandidateProposal>* output,
                       CandidateIndexStats* stats) const = 0;
  virtual void CollectRefreshStats(CandidateIndexStats* stats) const = 0;
  virtual void OnMerge(int keep, int remove) = 0;
};

class QuotientNeighborCandidateIndex final : public CandidateIndex {
 public:
  void BuildOrRefresh(const QuotientGraph& quotient,
                      const std::vector<int>& active_reps,
                      uint64_t seed) override;
  void Propose(int source, int budget,
               std::vector<CandidateProposal>* output,
               CandidateIndexStats* stats) const override;
  void OnMerge(int keep, int remove) override;
  void CollectRefreshStats(CandidateIndexStats* stats) const override;

 private:
  const QuotientGraph* quotient_ = nullptr;
  const std::vector<int>* active_reps_ = nullptr;
  uint64_t seed_ = 1;
};

class ResidualSignatureCandidateIndex final : public CandidateIndex {
 public:
  explicit ResidualSignatureCandidateIndex(int feature_limit = 8,
                                           int bucket_fanout = 8);
  void BuildOrRefresh(const QuotientGraph& quotient,
                      const std::vector<int>& active_reps,
                      uint64_t seed) override;
  void Propose(int source, int budget,
               std::vector<CandidateProposal>* output,
               CandidateIndexStats* stats) const override;
  void OnMerge(int keep, int remove) override;
  void CollectRefreshStats(CandidateIndexStats* stats) const override;

 private:
  struct Feature {
    int neighbor = -1;
    int8_t sign = 0;
    uint8_t magnitude_bin = 0;
    int64_t magnitude = 0;
    int64_t normalized = 0;
  };
  struct Signature {
    uint64_t version = UINT64_MAX;
    int size_bin = 0;
    int degree_bin = 0;
    std::vector<Feature> features;
  };
  struct BucketMember {
    int representative = -1;
    int64_t magnitude = 0;
  };

  uint64_t FeatureKey(const Feature& feature) const;
  const Signature& RefreshSignature(int representative,
                                    CandidateIndexStats* stats);
  const QuotientGraph* quotient_ = nullptr;
  const std::vector<int>* active_reps_ = nullptr;
  uint64_t seed_ = 1;
  int feature_limit_ = 8;
  int bucket_fanout_ = 8;
  std::vector<Signature> signatures_;
  std::unordered_map<uint64_t, std::vector<BucketMember>> buckets_;
  CandidateIndexStats build_stats_;
};

const char* CandidateIndexModeToString(CandidateIndexMode mode);
