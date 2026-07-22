#include "candidate_index.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

uint64_t SplitMix64(uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

bool Better(const CandidateProposal& lhs, const CandidateProposal& rhs) {
  if (lhs.cheap_score != rhs.cheap_score) return lhs.cheap_score > rhs.cheap_score;
  if (lhs.source_type != rhs.source_type) {
    return static_cast<int>(lhs.source_type) < static_cast<int>(rhs.source_type);
  }
  return lhs.target < rhs.target;
}

void InsertBounded(const CandidateProposal& proposal, size_t limit,
                   std::vector<CandidateProposal>* selected) {
  auto position = std::lower_bound(selected->begin(), selected->end(), proposal,
                                   Better);
  selected->insert(position, proposal);
  if (selected->size() > limit) selected->pop_back();
}

}  // namespace

void QuotientNeighborCandidateIndex::BuildOrRefresh(
    const QuotientGraph& quotient, const std::vector<int>& active_reps,
    uint64_t seed) {
  quotient_ = &quotient;
  active_reps_ = &active_reps;
  seed_ = seed;
}

void QuotientNeighborCandidateIndex::Propose(
    int source, int budget, std::vector<CandidateProposal>* output,
    CandidateIndexStats* stats) const {
  if (quotient_ == nullptr || active_reps_ == nullptr || output == nullptr ||
      stats == nullptr || budget <= 0) {
    throw std::logic_error("candidate index is not initialized");
  }
  output->clear();
  if (!quotient_->IsActive(source)) {
    ++stats->nodes_with_zero_proposals;
    return;
  }
  const QuotientRowView source_row = quotient_->GetRowView(source);
  std::vector<CandidateProposal> raw;
  raw.reserve(static_cast<size_t>(3 * budget + 1));
  std::vector<CandidateProposal> direct;
  direct.reserve(static_cast<size_t>(budget));
  for (size_t i = 0; i < source_row.cross_size; ++i) {
    const auto entry = source_row.cross_data[i];
    InsertBounded({source, entry.first, CandidateSource::kDirectNeighbor,
                   entry.second}, static_cast<size_t>(budget), &direct);
  }
  raw.insert(raw.end(), direct.begin(), direct.end());

  const size_t pivot_limit = std::min(direct.size(), static_cast<size_t>(2));
  for (size_t p = 0; p < pivot_limit; ++p) {
    const QuotientRowView shared = quotient_->GetRowView(direct[p].target);
    std::vector<CandidateProposal> shared_best;
    shared_best.reserve(static_cast<size_t>(budget));
    for (size_t i = 0; i < shared.cross_size; ++i) {
      const auto entry = shared.cross_data[i];
      if (entry.first != source) {
        InsertBounded({source, entry.first, CandidateSource::kSharedNeighbor,
                       std::min(direct[p].cheap_score, entry.second)},
                      static_cast<size_t>(budget), &shared_best);
      }
    }
    raw.insert(raw.end(), shared_best.begin(), shared_best.end());
  }

  if (active_reps_->size() > 1) {
    size_t pos = static_cast<size_t>(SplitMix64(
        seed_ ^ static_cast<uint64_t>(static_cast<uint32_t>(source)))) %
        active_reps_->size();
    for (size_t attempt = 0; attempt < active_reps_->size(); ++attempt) {
      const int target = (*active_reps_)[pos];
      if (target != source && quotient_->IsActive(target)) {
        raw.push_back({source, target, CandidateSource::kExploration, 0});
        break;
      }
      pos = (pos + 1) % active_reps_->size();
    }
  }

  stats->proposals_raw += raw.size();
  std::sort(raw.begin(), raw.end(), [](const auto& lhs, const auto& rhs) {
    if (lhs.target != rhs.target) return lhs.target < rhs.target;
    return Better(lhs, rhs);
  });
  for (const CandidateProposal& proposal : raw) {
    if (output->empty() || output->back().target != proposal.target) {
      output->push_back(proposal);
    }
  }
  const size_t deduplicated_size = output->size();
  stats->duplicates_removed += raw.size() - deduplicated_size;
  std::sort(output->begin(), output->end(), Better);
  if (output->size() > static_cast<size_t>(budget)) {
    output->resize(static_cast<size_t>(budget));
    ++stats->budget_exhausted_nodes;
  }
  if (output->empty()) ++stats->nodes_with_zero_proposals;
  stats->proposals_unique += output->size();
  for (const CandidateProposal& proposal : *output) {
    switch (proposal.source_type) {
      case CandidateSource::kDirectNeighbor: ++stats->direct_neighbor_count; break;
      case CandidateSource::kSharedNeighbor: ++stats->shared_neighbor_count; break;
      case CandidateSource::kExploration: ++stats->exploration_count; break;
    }
  }
}

void QuotientNeighborCandidateIndex::OnMerge(int, int) {}

void QuotientNeighborCandidateIndex::CollectRefreshStats(
    CandidateIndexStats*) const {}

ResidualSignatureCandidateIndex::ResidualSignatureCandidateIndex(
    int feature_limit, int bucket_fanout)
    : feature_limit_(std::max(1, feature_limit)),
      bucket_fanout_(std::max(1, bucket_fanout)) {}

uint64_t ResidualSignatureCandidateIndex::FeatureKey(
    const Feature& feature) const {
  return (static_cast<uint64_t>(static_cast<uint32_t>(feature.neighbor)) << 16U) |
         (static_cast<uint64_t>(feature.sign > 0 ? 1U : 0U) << 8U) |
         feature.magnitude_bin;
}

const ResidualSignatureCandidateIndex::Signature&
ResidualSignatureCandidateIndex::RefreshSignature(
    int representative, CandidateIndexStats* stats) {
  Signature& signature = signatures_[static_cast<size_t>(representative)];
  const uint64_t version = quotient_->GetVersion(representative);
  if (signature.version == version) {
    ++stats->residual_signature_cache_hits;
    return signature;
  }
  ++stats->residual_signature_cache_misses;
  ++stats->residual_signature_rows_scanned;
  signature.features.clear();
  const int64_t size = quotient_->GetSize(representative);
  const QuotientRowView row = quotient_->GetRowView(representative);
  signature.size_bin = size > 0 ? static_cast<int>(std::log2(size)) : 0;
  signature.degree_bin = row.cross_size > 0
      ? static_cast<int>(std::log2(row.cross_size)) : 0;
  for (size_t i = 0; i < row.cross_size; ++i) {
    const auto entry = row.cross_data[i];
    const int64_t capacity = size * quotient_->GetSize(entry.first);
    const int64_t residual = 2 * entry.second - capacity;
    const int64_t magnitude = std::abs(residual);
    Feature feature;
    feature.neighbor = entry.first;
    feature.sign = residual >= 0 ? 1 : -1;
    feature.magnitude = magnitude;
    feature.normalized = capacity > 0 ? (magnitude * 1024) / capacity : 0;
    feature.magnitude_bin = static_cast<uint8_t>(std::min<int64_t>(
        15, feature.normalized > 0
                ? static_cast<int64_t>(std::log2(feature.normalized))
                : 0));
    auto better_feature = [](const Feature& lhs, const Feature& rhs) {
      if (lhs.normalized != rhs.normalized) return lhs.normalized > rhs.normalized;
      if (lhs.magnitude != rhs.magnitude) return lhs.magnitude > rhs.magnitude;
      return lhs.neighbor < rhs.neighbor;
    };
    auto pos = std::lower_bound(signature.features.begin(),
                                signature.features.end(), feature,
                                better_feature);
    signature.features.insert(pos, feature);
    if (signature.features.size() > static_cast<size_t>(feature_limit_)) {
      signature.features.pop_back();
    }
  }
  signature.version = version;
  stats->residual_signature_features_created += signature.features.size();
  return signature;
}

void ResidualSignatureCandidateIndex::BuildOrRefresh(
    const QuotientGraph& quotient, const std::vector<int>& active_reps,
    uint64_t seed) {
  const auto start = std::chrono::steady_clock::now();
  const bool first_build = quotient_ == nullptr;
  quotient_ = &quotient;
  active_reps_ = &active_reps;
  seed_ = seed;
  if (!active_reps.empty()) {
    const int max_rep = *std::max_element(active_reps.begin(), active_reps.end());
    if (signatures_.size() <= static_cast<size_t>(max_rep)) {
      signatures_.resize(static_cast<size_t>(max_rep + 1));
    }
  }
  buckets_.clear();
  CandidateIndexStats refresh;
  for (int representative : active_reps) {
    const Signature& signature = RefreshSignature(representative, &refresh);
    for (const Feature& feature : signature.features) {
      buckets_[FeatureKey(feature)].push_back(
          {representative, feature.normalized});
    }
  }
  for (auto& item : buckets_) {
    auto& members = item.second;
    std::sort(members.begin(), members.end(), [](const auto& lhs, const auto& rhs) {
      if (lhs.magnitude != rhs.magnitude) return lhs.magnitude > rhs.magnitude;
      return lhs.representative < rhs.representative;
    });
    refresh.residual_bucket_max_size =
        std::max<uint64_t>(refresh.residual_bucket_max_size, members.size());
  }
  refresh.residual_bucket_count = buckets_.size();
  const double elapsed = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - start).count();
  if (first_build) refresh.residual_signature_build_ms = elapsed;
  else refresh.residual_signature_refresh_ms = elapsed;
  build_stats_ = refresh;
}

void ResidualSignatureCandidateIndex::Propose(
    int source, int budget, std::vector<CandidateProposal>* output,
    CandidateIndexStats* stats) const {
  if (quotient_ == nullptr || active_reps_ == nullptr || output == nullptr ||
      stats == nullptr || budget <= 0) {
    throw std::logic_error("residual candidate index is not initialized");
  }
  output->clear();
  if (!quotient_->IsActive(source)) {
    ++stats->nodes_with_zero_proposals;
    return;
  }
  const Signature& signature = signatures_[static_cast<size_t>(source)];
  const int residual_budget = budget >= 8 ? 4 : 2;
  const int direct_budget = budget >= 8 ? 2 : 1;
  const int shared_budget = budget >= 8 ? 1 : 0;
  std::unordered_map<int, CandidateProposal> ranked;
  for (const Feature& feature : signature.features) {
    const auto bucket_it = buckets_.find(FeatureKey(feature));
    if (bucket_it == buckets_.end()) continue;
    const auto& members = bucket_it->second;
    const size_t considered = std::min(members.size(),
        static_cast<size_t>(bucket_fanout_));
    stats->residual_bucket_candidates_considered += considered;
    if (members.size() > considered) {
      stats->residual_bucket_candidates_dropped_by_cap += members.size() - considered;
    }
    for (size_t i = 0; i < considered; ++i) {
      const int target = members[i].representative;
      if (target == source || !quotient_->IsActive(target)) continue;
      const Signature& target_signature = signatures_[static_cast<size_t>(target)];
      int64_t alignment = 0;
      int64_t conflict = 0;
      for (const Feature& lhs : signature.features) {
        for (const Feature& rhs : target_signature.features) {
          if (lhs.neighbor != rhs.neighbor) continue;
          const int64_t strength = std::min(lhs.normalized, rhs.normalized);
          if (lhs.sign == rhs.sign) alignment += strength;
          else conflict += strength;
        }
      }
      const int64_t size_compatibility =
          128 / (1 + std::abs(signature.size_bin - target_signature.size_bin));
      const int64_t direct = quotient_->GetEdgeCount(source, target) > 0 ? 128 : 0;
      const int64_t score = 4 * alignment - 3 * conflict +
                            size_compatibility + direct;
      CandidateProposal proposal{source, target,
          CandidateSource::kResidualSignature, score};
      auto it = ranked.find(target);
      if (it == ranked.end() || Better(proposal, it->second)) ranked[target] = proposal;
      stats->residual_alignment_score_sum += alignment;
      stats->residual_conflict_penalty_sum += conflict;
      stats->residual_direct_score_sum += direct;
      stats->residual_size_compatibility_sum += size_compatibility;
    }
  }
  std::vector<CandidateProposal> residual;
  for (const auto& item : ranked) residual.push_back(item.second);
  std::sort(residual.begin(), residual.end(), Better);
  if (residual.size() > static_cast<size_t>(residual_budget)) {
    residual.resize(static_cast<size_t>(residual_budget));
  }
  output->insert(output->end(), residual.begin(), residual.end());

  const QuotientRowView row = quotient_->GetRowView(source);
  std::vector<CandidateProposal> direct;
  for (size_t i = 0; i < row.cross_size; ++i) {
    const auto entry = row.cross_data[i];
    InsertBounded({source, entry.first, CandidateSource::kDirectNeighbor,
                   entry.second}, direct_budget, &direct);
  }
  output->insert(output->end(), direct.begin(), direct.end());
  if (shared_budget > 0 && !direct.empty()) {
    const QuotientRowView shared = quotient_->GetRowView(direct.front().target);
    std::vector<CandidateProposal> shared_best;
    for (size_t i = 0; i < shared.cross_size; ++i) {
      const auto entry = shared.cross_data[i];
      if (entry.first != source) {
        InsertBounded({source, entry.first, CandidateSource::kSharedNeighbor,
                       std::min(direct.front().cheap_score, entry.second)},
                      shared_budget, &shared_best);
      }
    }
    output->insert(output->end(), shared_best.begin(), shared_best.end());
  }
  if (active_reps_->size() > 1) {
    size_t pos = SplitMix64(seed_ ^ static_cast<uint32_t>(source)) %
                 active_reps_->size();
    for (size_t i = 0; i < active_reps_->size(); ++i) {
      const int target = (*active_reps_)[pos];
      if (target != source && quotient_->IsActive(target)) {
        output->push_back({source, target, CandidateSource::kExploration, 0});
        break;
      }
      pos = (pos + 1) % active_reps_->size();
    }
  }
  stats->proposals_raw += output->size();
  std::sort(output->begin(), output->end(), [](const auto& lhs, const auto& rhs) {
    if (lhs.target != rhs.target) return lhs.target < rhs.target;
    return Better(lhs, rhs);
  });
  output->erase(std::unique(output->begin(), output->end(),
      [](const auto& lhs, const auto& rhs) { return lhs.target == rhs.target; }),
      output->end());
  std::sort(output->begin(), output->end(), Better);
  if (output->size() > static_cast<size_t>(budget)) output->resize(budget);
  stats->proposals_unique += output->size();
  for (const auto& proposal : *output) {
    switch (proposal.source_type) {
      case CandidateSource::kResidualSignature: break;
      case CandidateSource::kDirectNeighbor: ++stats->direct_neighbor_count; break;
      case CandidateSource::kSharedNeighbor: ++stats->shared_neighbor_count; break;
      case CandidateSource::kExploration: ++stats->exploration_count; break;
    }
  }
}

void ResidualSignatureCandidateIndex::OnMerge(int, int) {}

void ResidualSignatureCandidateIndex::CollectRefreshStats(
    CandidateIndexStats* stats) const {
  stats->residual_signature_build_ms += build_stats_.residual_signature_build_ms;
  stats->residual_signature_refresh_ms += build_stats_.residual_signature_refresh_ms;
  stats->residual_signature_rows_scanned += build_stats_.residual_signature_rows_scanned;
  stats->residual_signature_features_created += build_stats_.residual_signature_features_created;
  stats->residual_signature_cache_hits += build_stats_.residual_signature_cache_hits;
  stats->residual_signature_cache_misses += build_stats_.residual_signature_cache_misses;
  stats->residual_bucket_count += build_stats_.residual_bucket_count;
  stats->residual_bucket_max_size = std::max(stats->residual_bucket_max_size,
                                            build_stats_.residual_bucket_max_size);
}

const char* CandidateIndexModeToString(CandidateIndexMode mode) {
  if (mode == CandidateIndexMode::kQuotientNeighbor) return "quotient-neighbor";
  if (mode == CandidateIndexMode::kResidualSignature) return "residual-signature";
  return "legacy";
}
