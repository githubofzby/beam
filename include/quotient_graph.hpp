#pragma once

#include <cstdint>
#include <iterator>
#include <utility>
#include <vector>

#include "cost_oracle.hpp"
#include "graph_io/graph.hpp"

struct QuotientRowView {
  using NeighborEntry = std::pair<int, int64_t>;

  class Iterator {
   public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = NeighborEntry;
    using difference_type = std::ptrdiff_t;
    using pointer = void;
    using reference = NeighborEntry;

    Iterator() = default;
    Iterator(const QuotientRowView* view, size_t position)
        : view_(view), position_(position) {}
    NeighborEntry operator*() const;
    Iterator& operator++() {
      ++position_;
      return *this;
    }
    bool operator==(const Iterator& other) const {
      return view_ == other.view_ && position_ == other.position_;
    }
    bool operator!=(const Iterator& other) const { return !(*this == other); }

   private:
    const QuotientRowView* view_ = nullptr;
    size_t position_ = 0;
  };

  static QuotientRowView Raw(const NeighborEntry* data, size_t size);
  Iterator begin() const;
  Iterator end() const;
  NeighborEntry operator[](size_t position) const;
  void ValidateVersion() const;
  size_t size() const { return cross_size + (has_internal ? 1U : 0U); }
  bool empty() const { return size() == 0; }

  const NeighborEntry* cross_data = nullptr;
  size_t cross_size = 0;
  int representative = -1;
  int64_t internal_arc_count = 0;
  size_t internal_position = 0;
  bool has_internal = false;
  const uint64_t* version_address = nullptr;
  uint64_t expected_version = 0;
};

inline QuotientRowView::NeighborEntry QuotientRowView::Iterator::operator*()
    const {
  if (view_->has_internal && position_ == view_->internal_position) {
    return {view_->representative, view_->internal_arc_count};
  }
  const size_t cross_position =
      position_ - (view_->has_internal && position_ > view_->internal_position
                       ? 1U
                       : 0U);
  return view_->cross_data[cross_position];
}

inline QuotientRowView::NeighborEntry QuotientRowView::operator[](
    size_t position) const {
  if (has_internal && position == internal_position) {
    return {representative, internal_arc_count};
  }
  const size_t cross_position =
      position - (has_internal && position > internal_position ? 1U : 0U);
  return cross_data[cross_position];
}

struct QuotientProfileStats {
  double incremental_update_ms = 0.0;
  double reciprocal_update_ms = 0.0;
  double memory_allocation_ms = 0.0;
  double sort_or_merge_ms = 0.0;
  double rebuild_ms = 0.0;
  uint64_t rows_updated = 0;
  uint64_t entries_inserted = 0;
  uint64_t entries_removed = 0;
  uint64_t entries_shifted = 0;
  uint64_t high_degree_rows_touched = 0;
  uint64_t max_row_degree = 0;
  uint64_t allocated_bytes = 0;
  uint64_t peak_quotient_nnz = 0;
};

struct ExactGainWorkCounters {
  uint64_t pairs = 0;
  uint64_t raw_entries_a = 0;
  uint64_t raw_entries_b = 0;
  uint64_t union_neighbors = 0;
  uint64_t overlap_neighbors = 0;
  uint64_t single_sided_neighbors = 0;
  uint64_t internal_block_terms = 0;
  uint64_t block_cost_evaluations = 0;
  uint64_t capacity_multiplications = 0;

  void Add(const ExactGainWorkCounters& other) {
    pairs += other.pairs;
    raw_entries_a += other.raw_entries_a;
    raw_entries_b += other.raw_entries_b;
    union_neighbors += other.union_neighbors;
    overlap_neighbors += other.overlap_neighbors;
    single_sided_neighbors += other.single_sided_neighbors;
    internal_block_terms += other.internal_block_terms;
    block_cost_evaluations += other.block_cost_evaluations;
    capacity_multiplications += other.capacity_multiplications;
  }
};

struct QuotientMergeGainResult {
  MergeGain gain = 0;
  EncodingCost before_cost = 0;
};

struct CertificationWorkCounters {
  uint64_t candidates_seen = 0;
  uint64_t upper_bound_pruned = 0;
  uint64_t upper_bound_passed = 0;
  uint64_t early_abort_count = 0;
  uint64_t exact_full_scan_count = 0;
  uint64_t entries_available = 0;
  uint64_t entries_scanned = 0;
};

struct CertifiedMergeGainResult {
  MergeGain gain = 0;
  EncodingCost before_cost = 0;
  bool upper_bound_pruned = false;
  bool early_aborted = false;
  bool full_scan = false;
};

class QuotientGraph {
 public:
  using SparseRow = std::vector<std::pair<int, int64_t>>;

  QuotientGraph() = default;
  explicit QuotientGraph(const CSRGraph& graph, bool track_children = true,
                         bool collect_profile = false);

  void Initialize(const CSRGraph& graph);
  void Rebuild(const CSRGraph& graph, const std::vector<int>& labels);
  uint64_t Merge(int keep, int remove);
  uint64_t MergeBatchBulk(
      const std::vector<std::pair<int, int>>& keep_remove_pairs);
  uint64_t EstimateIncrementalWork(
      const std::vector<std::pair<int, int>>& keep_remove_pairs) const;

  bool IsActive(int supernode) const;
  int64_t GetSize(int supernode) const;
  int64_t GetInternalEdgeCount(int supernode) const;
  int64_t GetEdgeCount(int a, int b) const;
  int64_t GetCapacity(int a, int b) const;
  EncodingCost GetBlockCost(int a, int b) const;
  EncodingCost ExactCost() const;
  uint64_t Nnz() const;
  MergeGain ExactMergeGain(int a, int b) const;
  QuotientMergeGainResult ExactMergeGainPersistent(
      int a, int b, ExactGainWorkCounters* work = nullptr) const;
  EncodingCost ExactIncidentCost(int representative) const;
  MergeGain MergeGainUpperBound(int a, int b,
                                EncodingCost before_cost) const;
  CertifiedMergeGainResult CertifiedMergeGainPersistent(
      int a, int b, EncodingCost before_cost, double threshold,
      bool use_threshold, CertificationWorkCounters* work = nullptr,
      bool upper_bound_already_passed = false) const;
  uint64_t GetVersion(int supernode) const;
  const std::vector<int>& GetChildren(int supernode) const;
  QuotientRowView GetRowView(int supernode) const;
  const QuotientProfileStats& profile_stats() const { return profile_stats_; }
  void FinalizeProfileSnapshot();
  SparseRow ExportLegacyRow(int supernode) const;

  void ValidateAgainstOriginalGraph(const CSRGraph& graph,
                                    const std::vector<int>& labels) const;

 private:
  template <bool TrackWork>
  QuotientMergeGainResult ExactMergeGainPersistentImpl(
      int a, int b, ExactGainWorkCounters* work) const;
  void CheckIndex(int supernode) const;
  void CheckActive(int supernode) const;
  int64_t RowEdgeCount(int row, int neighbor) const;
  void SetRowEdgeCount(int row, int neighbor, int64_t count);
  uint64_t EstimateAllocatedBytes() const;

  std::vector<int64_t> sizes_;
  std::vector<int64_t> internal_edges_;
  std::vector<SparseRow> adjacency_;
  std::vector<uint64_t> versions_;
  std::vector<std::vector<int>> children_;
  std::vector<uint8_t> active_;
  bool track_children_ = true;
  bool collect_profile_ = false;
  CostOracle cost_oracle_;
  QuotientProfileStats profile_stats_;
};
