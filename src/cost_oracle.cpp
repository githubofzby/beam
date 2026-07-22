#include "cost_oracle.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct PartitionState {
  std::vector<int64_t> sizes;
  std::unordered_map<uint64_t, int64_t> edges;
  std::unordered_map<int, int> compact_by_label;
};

uint64_t PairKey(int a, int b) {
  const uint32_t lo = static_cast<uint32_t>(std::min(a, b));
  const uint32_t hi = static_cast<uint32_t>(std::max(a, b));
  return (static_cast<uint64_t>(lo) << 32U) | hi;
}

int64_t CheckedInt64(__int128 value, const char* context) {
  if (value < 0 || value > std::numeric_limits<int64_t>::max()) {
    throw std::overflow_error(context);
  }
  return static_cast<int64_t>(value);
}

void CheckedAdd(EncodingCost* total, EncodingCost value) {
  if (value > std::numeric_limits<EncodingCost>::max() - *total) {
    throw std::overflow_error("partition cost overflow");
  }
  *total += value;
}

PartitionState BuildState(const CSRGraph& graph,
                          const std::vector<int>& labels) {
  if (graph.n < 0 || labels.size() != static_cast<size_t>(graph.n)) {
    throw std::invalid_argument("partition labels must match graph.n");
  }
  if (graph.row_ptr.size() != static_cast<size_t>(graph.n + 1) ||
      graph.col_idx.size() != static_cast<size_t>(graph.m)) {
    throw std::invalid_argument("invalid CSR graph shape");
  }

  PartitionState state;
  state.compact_by_label.reserve(labels.size());
  std::vector<int> compact(labels.size(), -1);
  for (int v = 0; v < graph.n; ++v) {
    const int label = labels[static_cast<size_t>(v)];
    if (label < 0) {
      throw std::invalid_argument("partition labels must be non-negative");
    }
    auto [it, inserted] = state.compact_by_label.emplace(
        label, static_cast<int>(state.sizes.size()));
    if (inserted) {
      state.sizes.push_back(0);
    }
    compact[static_cast<size_t>(v)] = it->second;
    ++state.sizes[static_cast<size_t>(it->second)];
  }

  state.edges.reserve(static_cast<size_t>(graph.input_edges_raw) + 1U);
  for (int u = 0; u < graph.n; ++u) {
    const auto neighbors = graph.neighbors(u);
    for (const int* it = neighbors.first; it != neighbors.second; ++it) {
      const int v = *it;
      if (v < 0 || v >= graph.n) {
        throw std::invalid_argument("CSR neighbor is out of range");
      }
      if (u == v) {
        throw std::invalid_argument("CostOracle does not support self-loops");
      }
      // Repository graphs store each undirected edge as two arcs.
      if (u < v) {
        ++state.edges[PairKey(compact[static_cast<size_t>(u)],
                              compact[static_cast<size_t>(v)])];
      }
    }
  }
  return state;
}

int64_t EdgeCount(const PartitionState& state, int a, int b) {
  const auto it = state.edges.find(PairKey(a, b));
  return it == state.edges.end() ? 0 : it->second;
}

}  // namespace

EncodingCost CostOracle::BlockCost(int64_t size_a, int64_t size_b,
                                   int64_t edge_count,
                                   bool internal_block) const {
  if (size_a < 0 || size_b < 0 || edge_count < 0) {
    throw std::invalid_argument("block sizes and edge count must be non-negative");
  }
  if (internal_block && size_a != size_b) {
    throw std::invalid_argument("internal block sizes must match");
  }
  const __int128 capacity_wide = internal_block
      ? static_cast<__int128>(size_a) * (size_a - 1) / 2
      : static_cast<__int128>(size_a) * size_b;
  const int64_t capacity = CheckedInt64(capacity_wide, "block capacity overflow");
  if (edge_count > capacity) {
    throw std::invalid_argument("edge count exceeds block capacity");
  }
  return std::min(edge_count, 1 + capacity - edge_count);
}

EncodingCost CostOracle::ExactPartitionCost(
    const CSRGraph& graph, const std::vector<int>& labels) const {
  const PartitionState state = BuildState(graph, labels);
  EncodingCost total = 0;
  for (const auto& [key, edge_count] : state.edges) {
    const int a = static_cast<int>(key >> 32U);
    const int b = static_cast<int>(key & 0xffffffffU);
    CheckedAdd(&total, BlockCost(state.sizes[static_cast<size_t>(a)],
                                 state.sizes[static_cast<size_t>(b)],
                                 edge_count, a == b));
  }
  return total;
}

MergeGain CostOracle::ExactMergeGain(const CSRGraph& graph,
                                     const std::vector<int>& labels,
                                     int label_a, int label_b) const {
  if (label_a == label_b) {
    throw std::invalid_argument("merge endpoints must have different labels");
  }
  const PartitionState state = BuildState(graph, labels);
  const auto a_it = state.compact_by_label.find(label_a);
  const auto b_it = state.compact_by_label.find(label_b);
  if (a_it == state.compact_by_label.end() ||
      b_it == state.compact_by_label.end()) {
    throw std::invalid_argument("merge endpoint label is not active");
  }
  const int a = a_it->second;
  const int b = b_it->second;
  const int64_t size_a = state.sizes[static_cast<size_t>(a)];
  const int64_t size_b = state.sizes[static_cast<size_t>(b)];
  const int64_t merged_size = CheckedInt64(
      static_cast<__int128>(size_a) + size_b, "merged size overflow");

  EncodingCost before = 0;
  EncodingCost after = 0;
  const int64_t internal_edges =
      EdgeCount(state, a, a) + EdgeCount(state, b, b) + EdgeCount(state, a, b);
  CheckedAdd(&before, BlockCost(size_a, size_a, EdgeCount(state, a, a), true));
  CheckedAdd(&before, BlockCost(size_b, size_b, EdgeCount(state, b, b), true));
  CheckedAdd(&before, BlockCost(size_a, size_b, EdgeCount(state, a, b), false));
  CheckedAdd(&after, BlockCost(merged_size, merged_size, internal_edges, true));

  for (int x = 0; x < static_cast<int>(state.sizes.size()); ++x) {
    if (x == a || x == b) {
      continue;
    }
    const int64_t size_x = state.sizes[static_cast<size_t>(x)];
    const int64_t edges_a = EdgeCount(state, a, x);
    const int64_t edges_b = EdgeCount(state, b, x);
    CheckedAdd(&before, BlockCost(size_a, size_x, edges_a, false));
    CheckedAdd(&before, BlockCost(size_b, size_x, edges_b, false));
    CheckedAdd(&after,
               BlockCost(merged_size, size_x, edges_a + edges_b, false));
  }
  return before - after;
}
