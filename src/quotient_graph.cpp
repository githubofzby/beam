#include "quotient_graph.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

int64_t CheckedCapacity(__int128 value) {
  if (value < 0 || value > std::numeric_limits<int64_t>::max()) {
    throw std::overflow_error("quotient block capacity overflow");
  }
  return static_cast<int64_t>(value);
}

void CheckedAdd(int64_t* target, int64_t value, const char* context) {
  if (value < 0 || *target > std::numeric_limits<int64_t>::max() - value) {
    throw std::overflow_error(context);
  }
  *target += value;
}

bool CannotMeetAcceptance(MergeGain upper_bound, EncodingCost before_cost,
                          double threshold, bool use_threshold) {
  if (upper_bound <= 0) {
    return true;
  }
  if (!use_threshold) {
    return false;
  }
  if (before_cost <= 0) {
    return true;
  }
  // Match the downstream comparison exactly. Since floating-point division is
  // monotone here, a bound below the threshold proves every smaller gain fails.
  const double upper_ratio = static_cast<double>(upper_bound) /
                             static_cast<double>(before_cost);
  return upper_ratio < threshold;
}

}  // namespace

QuotientRowView QuotientRowView::Raw(const NeighborEntry* data, size_t size) {
  QuotientRowView view;
  view.cross_data = data;
  view.cross_size = size;
  return view;
}

QuotientRowView::Iterator QuotientRowView::begin() const {
  return Iterator(this, 0);
}

QuotientRowView::Iterator QuotientRowView::end() const {
  return Iterator(this, size());
}

void QuotientRowView::ValidateVersion() const {
  if (version_address != nullptr && *version_address != expected_version) {
    throw std::runtime_error("stale quotient row view");
  }
}

QuotientGraph::QuotientGraph(const CSRGraph& graph, bool track_children,
                             bool collect_profile)
    : track_children_(track_children), collect_profile_(collect_profile) {
  Initialize(graph);
}

void QuotientGraph::Initialize(const CSRGraph& graph) {
  std::vector<int> labels(static_cast<size_t>(graph.n));
  for (int v = 0; v < graph.n; ++v) {
    labels[static_cast<size_t>(v)] = v;
  }
  Rebuild(graph, labels);
}

void QuotientGraph::Rebuild(const CSRGraph& graph,
                            const std::vector<int>& labels) {
  if (graph.n < 0 || labels.size() != static_cast<size_t>(graph.n) ||
      graph.row_ptr.size() != static_cast<size_t>(graph.n + 1) ||
      graph.col_idx.size() != static_cast<size_t>(graph.m)) {
    throw std::invalid_argument("invalid graph or partition shape");
  }
  sizes_.assign(static_cast<size_t>(graph.n), 0);
  internal_edges_.assign(static_cast<size_t>(graph.n), 0);
  adjacency_.assign(static_cast<size_t>(graph.n), {});
  versions_.assign(static_cast<size_t>(graph.n), 0);
  children_.assign(static_cast<size_t>(graph.n), {});
  active_.assign(static_cast<size_t>(graph.n), 0);

  for (int v = 0; v < graph.n; ++v) {
    const int label = labels[static_cast<size_t>(v)];
    if (label < 0 || label >= graph.n) {
      throw std::invalid_argument("partition label is out of range");
    }
    active_[static_cast<size_t>(label)] = 1;
    ++sizes_[static_cast<size_t>(label)];
    if (track_children_) {
      children_[static_cast<size_t>(label)].push_back(v);
    }
  }

  for (int u = 0; u < graph.n; ++u) {
    const auto neighbors = graph.neighbors(u);
    for (const int* it = neighbors.first; it != neighbors.second; ++it) {
      const int v = *it;
      if (v < 0 || v >= graph.n || u == v) {
        throw std::invalid_argument(
            "QuotientGraph requires a loop-free graph with valid neighbors");
      }
      if (u >= v) {
        continue;
      }
      const int a = labels[static_cast<size_t>(u)];
      const int b = labels[static_cast<size_t>(v)];
      if (a == b) {
        CheckedAdd(&internal_edges_[static_cast<size_t>(a)], 1,
                   "internal edge count overflow");
      } else {
        adjacency_[static_cast<size_t>(a)].emplace_back(b, 1);
        adjacency_[static_cast<size_t>(b)].emplace_back(a, 1);
      }
    }
  }
  for (SparseRow& row : adjacency_) {
    std::sort(row.begin(), row.end());
    size_t out = 0;
    for (const auto& entry : row) {
      if (out > 0 && row[out - 1].first == entry.first) {
        CheckedAdd(&row[out - 1].second, entry.second,
                   "quotient edge count overflow");
      } else {
        row[out++] = entry;
      }
    }
    row.resize(out);
  }
  if (collect_profile_) {
    profile_stats_.allocated_bytes = EstimateAllocatedBytes();
    profile_stats_.peak_quotient_nnz = Nnz();
  }
}

void QuotientGraph::CheckIndex(int supernode) const {
  if (supernode < 0 || supernode >= static_cast<int>(active_.size())) {
    throw std::out_of_range("quotient supernode is out of range");
  }
}

void QuotientGraph::CheckActive(int supernode) const {
  CheckIndex(supernode);
  if (!active_[static_cast<size_t>(supernode)]) {
    throw std::invalid_argument("quotient supernode is inactive: " +
                                std::to_string(supernode));
  }
}

bool QuotientGraph::IsActive(int supernode) const {
  CheckIndex(supernode);
  return active_[static_cast<size_t>(supernode)] != 0;
}

int64_t QuotientGraph::GetSize(int supernode) const {
  CheckActive(supernode);
  return sizes_[static_cast<size_t>(supernode)];
}

int64_t QuotientGraph::GetInternalEdgeCount(int supernode) const {
  CheckActive(supernode);
  return internal_edges_[static_cast<size_t>(supernode)];
}

int64_t QuotientGraph::GetEdgeCount(int a, int b) const {
  CheckActive(a);
  CheckActive(b);
  if (a == b) {
    return internal_edges_[static_cast<size_t>(a)];
  }
  return RowEdgeCount(a, b);
}

int64_t QuotientGraph::RowEdgeCount(int row, int neighbor) const {
  const SparseRow& entries = adjacency_[static_cast<size_t>(row)];
  const auto it = std::lower_bound(
      entries.begin(), entries.end(), neighbor,
      [](const auto& entry, int value) { return entry.first < value; });
  return it == entries.end() || it->first != neighbor ? 0 : it->second;
}

void QuotientGraph::SetRowEdgeCount(int row, int neighbor, int64_t count) {
  SparseRow& entries = adjacency_[static_cast<size_t>(row)];
  const auto it = std::lower_bound(
      entries.begin(), entries.end(), neighbor,
      [](const auto& entry, int value) { return entry.first < value; });
  if (it != entries.end() && it->first == neighbor) {
    if (count == 0) {
      if (collect_profile_) {
        ++profile_stats_.entries_removed;
        profile_stats_.entries_shifted +=
            static_cast<uint64_t>(entries.end() - it - 1);
      }
      entries.erase(it);
    } else {
      it->second = count;
    }
  } else if (count != 0) {
    if (collect_profile_) {
      ++profile_stats_.entries_inserted;
      profile_stats_.entries_shifted +=
          static_cast<uint64_t>(entries.end() - it);
    }
    entries.insert(it, {neighbor, count});
  }
}

uint64_t QuotientGraph::EstimateAllocatedBytes() const {
  uint64_t bytes = 0;
  for (const SparseRow& row : adjacency_) {
    bytes += static_cast<uint64_t>(row.capacity()) * sizeof(SparseRow::value_type);
  }
  return bytes;
}

int64_t QuotientGraph::GetCapacity(int a, int b) const {
  const int64_t size_a = GetSize(a);
  const int64_t size_b = GetSize(b);
  return CheckedCapacity(a == b
                             ? static_cast<__int128>(size_a) * (size_a - 1) / 2
                             : static_cast<__int128>(size_a) * size_b);
}

EncodingCost QuotientGraph::GetBlockCost(int a, int b) const {
  return cost_oracle_.BlockCost(GetSize(a), GetSize(b), GetEdgeCount(a, b),
                                a == b);
}

EncodingCost QuotientGraph::ExactCost() const {
  EncodingCost total = 0;
  for (int a = 0; a < static_cast<int>(active_.size()); ++a) {
    if (!active_[static_cast<size_t>(a)]) {
      continue;
    }
    CheckedAdd(&total, GetBlockCost(a, a), "quotient cost overflow");
    for (const auto& [b, edge_count] : adjacency_[static_cast<size_t>(a)]) {
      if (a < b) {
        CheckedAdd(&total,
                   cost_oracle_.BlockCost(GetSize(a), GetSize(b), edge_count,
                                          false),
                   "quotient cost overflow");
      }
    }
  }
  return total;
}

uint64_t QuotientGraph::Nnz() const {
  uint64_t total = 0;
  for (size_t rep = 0; rep < active_.size(); ++rep) {
    if (active_[rep]) {
      total += adjacency_[rep].size();
      if (internal_edges_[rep] > 0) {
        ++total;
      }
    }
  }
  return total;
}

MergeGain QuotientGraph::ExactMergeGain(int a, int b) const {
  CheckActive(a);
  CheckActive(b);
  if (a == b) {
    throw std::invalid_argument("merge endpoints must differ");
  }
  const int64_t size_a = GetSize(a);
  const int64_t size_b = GetSize(b);
  const int64_t merged_size = CheckedCapacity(
      static_cast<__int128>(size_a) + size_b);
  const int64_t edges_ab = GetEdgeCount(a, b);

  EncodingCost before = 0;
  EncodingCost after = 0;
  CheckedAdd(&before, GetBlockCost(a, a), "merge gain cost overflow");
  CheckedAdd(&before, GetBlockCost(b, b), "merge gain cost overflow");
  CheckedAdd(&before, GetBlockCost(a, b), "merge gain cost overflow");
  CheckedAdd(&after,
             cost_oracle_.BlockCost(
                 merged_size, merged_size,
                 GetInternalEdgeCount(a) + GetInternalEdgeCount(b) + edges_ab,
                 true),
             "merge gain cost overflow");

  SparseRow union_counts;
  const SparseRow& row_a = adjacency_[static_cast<size_t>(a)];
  const SparseRow& row_b = adjacency_[static_cast<size_t>(b)];
  union_counts.reserve(row_a.size() + row_b.size());
  size_t ia = 0;
  size_t ib = 0;
  while (ia < row_a.size() || ib < row_b.size()) {
    int neighbor = -1;
    int64_t count = 0;
    if (ib >= row_b.size() ||
        (ia < row_a.size() && row_a[ia].first < row_b[ib].first)) {
      neighbor = row_a[ia].first;
      count = row_a[ia++].second;
    } else if (ia >= row_a.size() || row_b[ib].first < row_a[ia].first) {
      neighbor = row_b[ib].first;
      count = row_b[ib++].second;
    } else {
      neighbor = row_a[ia].first;
      count = row_a[ia++].second;
      CheckedAdd(&count, row_b[ib++].second, "merge gain edge overflow");
    }
    if (neighbor != a && neighbor != b) {
      union_counts.emplace_back(neighbor, count);
    }
  }
  for (const auto& [neighbor, merged_edges] : union_counts) {
    const int64_t edges_a = GetEdgeCount(a, neighbor);
    const int64_t edges_b = GetEdgeCount(b, neighbor);
    CheckedAdd(&before,
               cost_oracle_.BlockCost(size_a, GetSize(neighbor), edges_a,
                                      false),
               "merge gain cost overflow");
    CheckedAdd(&before,
               cost_oracle_.BlockCost(size_b, GetSize(neighbor), edges_b,
                                      false),
               "merge gain cost overflow");
    CheckedAdd(&after,
               cost_oracle_.BlockCost(merged_size, GetSize(neighbor),
                                      merged_edges, false),
               "merge gain cost overflow");
  }
  return before - after;
}

template <bool TrackWork>
QuotientMergeGainResult QuotientGraph::ExactMergeGainPersistentImpl(
    int a, int b, ExactGainWorkCounters* work) const {
  CheckActive(a);
  CheckActive(b);
  if (a == b) {
    throw std::invalid_argument("merge endpoints must differ");
  }

  const int64_t size_a = sizes_[static_cast<size_t>(a)];
  const int64_t size_b = sizes_[static_cast<size_t>(b)];
  const int64_t size_w = CheckedCapacity(
      static_cast<__int128>(size_a) + static_cast<__int128>(size_b));
  const SparseRow& row_a = adjacency_[static_cast<size_t>(a)];
  const SparseRow& row_b = adjacency_[static_cast<size_t>(b)];

  uint64_t union_neighbors = 0;
  uint64_t overlap_neighbors = 0;

  EncodingCost before = 0;
  EncodingCost after = 0;
  int64_t edges_ab_a = 0;
  int64_t edges_ab_b = 0;
  size_t ia = 0;
  size_t ib = 0;
  while (ia < row_a.size() || ib < row_b.size()) {
    int neighbor = -1;
    int64_t edges_a = 0;
    int64_t edges_b = 0;
    bool overlap = false;
    if (ib >= row_b.size() ||
        (ia < row_a.size() && row_a[ia].first < row_b[ib].first)) {
      neighbor = row_a[ia].first;
      edges_a = row_a[ia].second;
      ++ia;
    } else if (ia >= row_a.size() || row_b[ib].first < row_a[ia].first) {
      neighbor = row_b[ib].first;
      edges_b = row_b[ib].second;
      ++ib;
    } else {
      neighbor = row_a[ia].first;
      edges_a = row_a[ia].second;
      edges_b = row_b[ib].second;
      ++ia;
      ++ib;
      overlap = true;
    }

    if (neighbor == b) {
      edges_ab_a = edges_a;
      continue;
    }
    if (neighbor == a) {
      edges_ab_b = edges_b;
      continue;
    }

    const int64_t size_x = sizes_[static_cast<size_t>(neighbor)];
    if (overlap) {
      CheckedAdd(&before,
                 cost_oracle_.BlockCost(size_a, size_x, edges_a, false),
                 "merge gain cost overflow");
      CheckedAdd(&before,
                 cost_oracle_.BlockCost(size_b, size_x, edges_b, false),
                 "merge gain cost overflow");
      if constexpr (TrackWork) {
        ++overlap_neighbors;
      }
    } else {
      const int64_t one_sided_edges = edges_a + edges_b;
      CheckedAdd(&before,
                 edges_a != 0
                     ? cost_oracle_.BlockCost(size_a, size_x, one_sided_edges,
                                              false)
                     : cost_oracle_.BlockCost(size_b, size_x, one_sided_edges,
                                              false),
                 "merge gain cost overflow");
    }
    CheckedAdd(&after,
               cost_oracle_.BlockCost(size_w, size_x, edges_a + edges_b,
                                      false),
               "merge gain cost overflow");
    if constexpr (TrackWork) {
      ++union_neighbors;
    }
  }

  if (edges_ab_a != edges_ab_b) {
    throw std::runtime_error("asymmetric quotient edge count during scoring");
  }
  CheckedAdd(&before,
             cost_oracle_.BlockCost(
                 size_a, size_a, internal_edges_[static_cast<size_t>(a)], true),
             "merge gain cost overflow");
  CheckedAdd(&before,
             cost_oracle_.BlockCost(
                 size_b, size_b, internal_edges_[static_cast<size_t>(b)], true),
             "merge gain cost overflow");
  CheckedAdd(&before,
             cost_oracle_.BlockCost(size_a, size_b, edges_ab_a, false),
             "merge gain cost overflow");
  CheckedAdd(&after,
             cost_oracle_.BlockCost(
                 size_w, size_w, internal_edges_[static_cast<size_t>(a)] +
                                     internal_edges_[static_cast<size_t>(b)] +
                                     edges_ab_a,
                 true),
             "merge gain cost overflow");
  if constexpr (TrackWork) {
    ++work->pairs;
    work->raw_entries_a += row_a.size();
    work->raw_entries_b += row_b.size();
    work->union_neighbors += union_neighbors;
    work->overlap_neighbors += overlap_neighbors;
    work->single_sided_neighbors += union_neighbors - overlap_neighbors;
    work->internal_block_terms += 4;
    work->block_cost_evaluations +=
        4 + 2 * union_neighbors + overlap_neighbors;
    work->capacity_multiplications +=
        4 + 2 * union_neighbors + overlap_neighbors;
  }
  return QuotientMergeGainResult{before - after, before};
}

QuotientMergeGainResult QuotientGraph::ExactMergeGainPersistent(
    int a, int b, ExactGainWorkCounters* work) const {
  if (work != nullptr) {
    return ExactMergeGainPersistentImpl<true>(a, b, work);
  }
  return ExactMergeGainPersistentImpl<false>(a, b, nullptr);
}

EncodingCost QuotientGraph::ExactIncidentCost(int representative) const {
  CheckActive(representative);
  const size_t rep = static_cast<size_t>(representative);
  EncodingCost cost = cost_oracle_.BlockCost(
      sizes_[rep], sizes_[rep], internal_edges_[rep], true);
  for (const auto& [neighbor, edges] : adjacency_[rep]) {
    CheckedAdd(&cost,
               cost_oracle_.BlockCost(sizes_[rep], GetSize(neighbor), edges,
                                      false),
               "incident cost overflow");
  }
  return cost;
}

MergeGain QuotientGraph::MergeGainUpperBound(
    int a, int b, EncodingCost before_cost) const {
  CheckActive(a);
  CheckActive(b);
  if (a == b) {
    throw std::invalid_argument("merge endpoints must differ");
  }
  const int64_t size_w = CheckedCapacity(
      static_cast<__int128>(GetSize(a)) + static_cast<__int128>(GetSize(b)));
  const int64_t edges_w = CheckedCapacity(
      static_cast<__int128>(GetInternalEdgeCount(a)) +
      static_cast<__int128>(GetInternalEdgeCount(b)) +
      static_cast<__int128>(GetEdgeCount(a, b)));
  const EncodingCost internal_after =
      cost_oracle_.BlockCost(size_w, size_w, edges_w, true);
  return before_cost - internal_after;
}

CertifiedMergeGainResult QuotientGraph::CertifiedMergeGainPersistent(
    int a, int b, EncodingCost before_cost, double threshold,
    bool use_threshold, CertificationWorkCounters* work,
    bool upper_bound_already_passed) const {
  CheckActive(a);
  CheckActive(b);
  if (a == b) {
    throw std::invalid_argument("merge endpoints must differ");
  }

  const SparseRow& row_a = adjacency_[static_cast<size_t>(a)];
  const SparseRow& row_b = adjacency_[static_cast<size_t>(b)];
  if (work != nullptr && !upper_bound_already_passed) {
    work->entries_available += row_a.size() + row_b.size();
  }

  MergeGain running_upper = MergeGainUpperBound(a, b, before_cost);
  if (!upper_bound_already_passed &&
      CannotMeetAcceptance(running_upper, before_cost, threshold,
                           use_threshold)) {
    if (work != nullptr) {
      ++work->upper_bound_pruned;
    }
    return CertifiedMergeGainResult{0, before_cost, true, false, false};
  }

  const int64_t size_w = CheckedCapacity(
      static_cast<__int128>(GetSize(a)) + static_cast<__int128>(GetSize(b)));
  size_t ia = 0;
  size_t ib = 0;
  while (ia < row_a.size() || ib < row_b.size()) {
    int neighbor = -1;
    int64_t merged_edges = 0;
    uint64_t consumed = 0;
    if (ib >= row_b.size() ||
        (ia < row_a.size() && row_a[ia].first < row_b[ib].first)) {
      neighbor = row_a[ia].first;
      merged_edges = row_a[ia].second;
      ++ia;
      consumed = 1;
    } else if (ia >= row_a.size() || row_b[ib].first < row_a[ia].first) {
      neighbor = row_b[ib].first;
      merged_edges = row_b[ib].second;
      ++ib;
      consumed = 1;
    } else {
      neighbor = row_a[ia].first;
      merged_edges = CheckedCapacity(static_cast<__int128>(row_a[ia].second) +
                                     row_b[ib].second);
      ++ia;
      ++ib;
      consumed = 2;
    }
    if (work != nullptr) {
      work->entries_scanned += consumed;
    }
    if (neighbor == a || neighbor == b) {
      continue;
    }

    const EncodingCost merged_block_cost = cost_oracle_.BlockCost(
        size_w, GetSize(neighbor), merged_edges, false);
    running_upper -= merged_block_cost;
    if (CannotMeetAcceptance(running_upper, before_cost, threshold,
                             use_threshold)) {
      if (work != nullptr) {
        ++work->early_abort_count;
      }
      return CertifiedMergeGainResult{0, before_cost, false, true, false};
    }
  }

  if (work != nullptr) {
    ++work->exact_full_scan_count;
  }
  return CertifiedMergeGainResult{running_upper, before_cost, false, false,
                                  true};
}

uint64_t QuotientGraph::GetVersion(int supernode) const {
  CheckIndex(supernode);
  return versions_[static_cast<size_t>(supernode)];
}

const std::vector<int>& QuotientGraph::GetChildren(int supernode) const {
  CheckActive(supernode);
  if (!track_children_) {
    throw std::logic_error("quotient child tracking is disabled");
  }
  return children_[static_cast<size_t>(supernode)];
}

QuotientRowView QuotientGraph::GetRowView(int supernode) const {
  CheckActive(supernode);
  const SparseRow& row = adjacency_[static_cast<size_t>(supernode)];
  QuotientRowView view;
  view.cross_data = row.data();
  view.cross_size = row.size();
  view.representative = supernode;
  const int64_t internal = internal_edges_[static_cast<size_t>(supernode)];
  if (internal > std::numeric_limits<int64_t>::max() / 2) {
    throw std::overflow_error("legacy internal arc count overflow");
  }
  view.internal_arc_count = internal * 2;
  view.has_internal = internal > 0;
  view.internal_position = static_cast<size_t>(std::lower_bound(
      row.begin(), row.end(), supernode,
      [](const auto& entry, int value) { return entry.first < value; }) -
                                              row.begin());
  view.version_address = &versions_[static_cast<size_t>(supernode)];
  view.expected_version = versions_[static_cast<size_t>(supernode)];
  return view;
}

QuotientGraph::SparseRow QuotientGraph::ExportLegacyRow(int supernode) const {
  const QuotientRowView view = GetRowView(supernode);
  SparseRow row;
  row.reserve(view.size());
  for (const auto entry : view) {
    row.push_back(entry);
  }
  return row;
}

uint64_t QuotientGraph::Merge(int keep, int remove) {
  const auto update_start = collect_profile_ ? std::chrono::steady_clock::now()
                                             : std::chrono::steady_clock::time_point{};
  CheckActive(keep);
  CheckActive(remove);
  if (keep == remove) {
    throw std::invalid_argument("merge endpoints must differ");
  }

  const int64_t cross = GetEdgeCount(keep, remove);
  const auto allocation_start = collect_profile_
                                    ? std::chrono::steady_clock::now()
                                    : std::chrono::steady_clock::time_point{};
  CheckedAdd(&sizes_[static_cast<size_t>(keep)],
             sizes_[static_cast<size_t>(remove)], "quotient size overflow");
  CheckedAdd(&internal_edges_[static_cast<size_t>(keep)],
             internal_edges_[static_cast<size_t>(remove)],
             "internal edge count overflow");
  CheckedAdd(&internal_edges_[static_cast<size_t>(keep)], cross,
             "internal edge count overflow");

  SetRowEdgeCount(keep, remove, 0);
  SetRowEdgeCount(remove, keep, 0);
  const SparseRow old_keep_edges = adjacency_[static_cast<size_t>(keep)];
  SparseRow removed_edges = adjacency_[static_cast<size_t>(remove)];
  const auto row_merge_start = collect_profile_ ? std::chrono::steady_clock::now()
                                                : std::chrono::steady_clock::time_point{};
  SparseRow merged_keep_edges;
  merged_keep_edges.reserve(old_keep_edges.size() + removed_edges.size());
  if (collect_profile_) {
    profile_stats_.memory_allocation_ms +=
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - allocation_start)
            .count();
  }
  size_t keep_index = 0;
  size_t remove_index = 0;
  while (keep_index < old_keep_edges.size() ||
         remove_index < removed_edges.size()) {
    if (remove_index >= removed_edges.size() ||
        (keep_index < old_keep_edges.size() &&
         old_keep_edges[keep_index].first <
             removed_edges[remove_index].first)) {
      merged_keep_edges.push_back(old_keep_edges[keep_index++]);
    } else if (keep_index >= old_keep_edges.size() ||
               removed_edges[remove_index].first <
                   old_keep_edges[keep_index].first) {
      merged_keep_edges.push_back(removed_edges[remove_index++]);
    } else {
      int64_t merged_count = old_keep_edges[keep_index].second;
      CheckedAdd(&merged_count, removed_edges[remove_index].second,
                 "quotient edge count overflow");
      merged_keep_edges.emplace_back(old_keep_edges[keep_index].first,
                                      merged_count);
      ++keep_index;
      ++remove_index;
    }
  }
  adjacency_[static_cast<size_t>(keep)] = std::move(merged_keep_edges);
  if (collect_profile_) {
    profile_stats_.sort_or_merge_ms +=
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - row_merge_start)
            .count();
  }
  const auto reciprocal_start = collect_profile_ ? std::chrono::steady_clock::now()
                                                 : std::chrono::steady_clock::time_point{};
  uint64_t touched_entries = 2;
  for (const auto& [neighbor, count] : removed_edges) {
    (void)count;
    const int64_t merged_count = RowEdgeCount(keep, neighbor);
    SetRowEdgeCount(neighbor, remove, 0);
    SetRowEdgeCount(neighbor, keep, merged_count);
    ++versions_[static_cast<size_t>(neighbor)];
    touched_entries += 3;
    if (collect_profile_) {
      ++profile_stats_.rows_updated;
      const uint64_t degree = adjacency_[static_cast<size_t>(neighbor)].size();
      profile_stats_.max_row_degree =
          std::max(profile_stats_.max_row_degree, degree);
      if (degree >= 4096) {
        ++profile_stats_.high_degree_rows_touched;
      }
    }
  }
  if (collect_profile_) {
    profile_stats_.reciprocal_update_ms +=
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - reciprocal_start)
            .count();
  }
  adjacency_[static_cast<size_t>(remove)].clear();

  if (track_children_) {
    auto& keep_children = children_[static_cast<size_t>(keep)];
    auto& remove_children = children_[static_cast<size_t>(remove)];
    keep_children.insert(keep_children.end(), remove_children.begin(),
                         remove_children.end());
    std::sort(keep_children.begin(), keep_children.end());
    remove_children.clear();
  }
  sizes_[static_cast<size_t>(remove)] = 0;
  internal_edges_[static_cast<size_t>(remove)] = 0;
  active_[static_cast<size_t>(remove)] = 0;
  ++versions_[static_cast<size_t>(keep)];
  ++versions_[static_cast<size_t>(remove)];
  if (collect_profile_) {
    profile_stats_.rows_updated += 2;
    profile_stats_.incremental_update_ms +=
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - update_start)
            .count();
  }
  return touched_entries;
}

uint64_t QuotientGraph::EstimateIncrementalWork(
    const std::vector<std::pair<int, int>>& pairs) const {
  uint64_t work = 0;
  for (const auto& [keep, remove] : pairs) {
    CheckActive(keep);
    CheckActive(remove);
    if (keep == remove) {
      throw std::invalid_argument("merge endpoints must differ");
    }
    work += adjacency_[static_cast<size_t>(keep)].size();
    work += 3 * adjacency_[static_cast<size_t>(remove)].size() + 2;
  }
  return work;
}

uint64_t QuotientGraph::MergeBatchBulk(
    const std::vector<std::pair<int, int>>& pairs) {
  const auto rebuild_start = collect_profile_ ? std::chrono::steady_clock::now()
                                              : std::chrono::steady_clock::time_point{};
  if (pairs.empty()) {
    return 0;
  }
  std::vector<int> destination(active_.size(), -1);
  for (size_t rep = 0; rep < active_.size(); ++rep) {
    if (active_[rep]) {
      destination[rep] = static_cast<int>(rep);
    }
  }
  for (const auto& [keep, remove] : pairs) {
    CheckActive(keep);
    CheckActive(remove);
    if (keep == remove || destination[static_cast<size_t>(keep)] != keep ||
        destination[static_cast<size_t>(remove)] != remove) {
      throw std::invalid_argument(
          "bulk quotient merge requires endpoint-disjoint active pairs");
    }
    destination[static_cast<size_t>(remove)] = keep;
  }

  std::vector<int64_t> new_sizes = sizes_;
  std::vector<int64_t> new_internal = internal_edges_;
  std::vector<SparseRow> new_adjacency(active_.size());
  for (const auto& [keep, remove] : pairs) {
    CheckedAdd(&new_sizes[static_cast<size_t>(keep)],
               sizes_[static_cast<size_t>(remove)], "quotient size overflow");
    CheckedAdd(&new_internal[static_cast<size_t>(keep)],
               internal_edges_[static_cast<size_t>(remove)],
               "internal edge count overflow");
    new_sizes[static_cast<size_t>(remove)] = 0;
    new_internal[static_cast<size_t>(remove)] = 0;
  }

  uint64_t scanned_entries = 0;
  for (int a = 0; a < static_cast<int>(active_.size()); ++a) {
    if (!active_[static_cast<size_t>(a)]) {
      continue;
    }
    for (const auto& [b, count] : adjacency_[static_cast<size_t>(a)]) {
      ++scanned_entries;
      if (a >= b) {
        continue;
      }
      const int mapped_a = destination[static_cast<size_t>(a)];
      const int mapped_b = destination[static_cast<size_t>(b)];
      if (mapped_a == mapped_b) {
        CheckedAdd(&new_internal[static_cast<size_t>(mapped_a)], count,
                   "internal edge count overflow");
      } else {
        new_adjacency[static_cast<size_t>(mapped_a)].emplace_back(mapped_b,
                                                                  count);
        new_adjacency[static_cast<size_t>(mapped_b)].emplace_back(mapped_a,
                                                                  count);
      }
    }
  }
  for (SparseRow& row : new_adjacency) {
    std::sort(row.begin(), row.end());
    size_t out = 0;
    for (const auto& entry : row) {
      if (out > 0 && row[out - 1].first == entry.first) {
        CheckedAdd(&row[out - 1].second, entry.second,
                   "quotient edge count overflow");
      } else {
        row[out++] = entry;
      }
    }
    row.resize(out);
  }

  for (const auto& [keep, remove] : pairs) {
    if (track_children_) {
      auto& keep_children = children_[static_cast<size_t>(keep)];
      auto& remove_children = children_[static_cast<size_t>(remove)];
      keep_children.insert(keep_children.end(), remove_children.begin(),
                           remove_children.end());
      std::sort(keep_children.begin(), keep_children.end());
      remove_children.clear();
    }
    active_[static_cast<size_t>(remove)] = 0;
    ++versions_[static_cast<size_t>(keep)];
    ++versions_[static_cast<size_t>(remove)];
  }
  sizes_.swap(new_sizes);
  internal_edges_.swap(new_internal);
  adjacency_.swap(new_adjacency);
  if (collect_profile_) {
    profile_stats_.rebuild_ms +=
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - rebuild_start)
            .count();
  }
  return scanned_entries;
}

void QuotientGraph::FinalizeProfileSnapshot() {
  if (!collect_profile_) {
    return;
  }
  profile_stats_.allocated_bytes = EstimateAllocatedBytes();
  profile_stats_.peak_quotient_nnz =
      std::max(profile_stats_.peak_quotient_nnz, Nnz());
}

void QuotientGraph::ValidateAgainstOriginalGraph(
    const CSRGraph& graph, const std::vector<int>& labels) const {
  QuotientGraph expected;
  expected.Rebuild(graph, labels);
  if (active_ != expected.active_ || sizes_ != expected.sizes_ ||
      internal_edges_ != expected.internal_edges_) {
    throw std::runtime_error("quotient node state differs from full rebuild");
  }
  for (size_t rep = 0; rep < active_.size(); ++rep) {
    if (active_[rep] &&
        (adjacency_[rep] != expected.adjacency_[rep] ||
         (track_children_ && children_[rep] != expected.children_[rep]))) {
      throw std::runtime_error("quotient row differs from full rebuild at " +
                               std::to_string(rep));
    }
  }
  if (ExactCost() != cost_oracle_.ExactPartitionCost(graph, labels)) {
    throw std::runtime_error("quotient cost differs from CostOracle");
  }
}
