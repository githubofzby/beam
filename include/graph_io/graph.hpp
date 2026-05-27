#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

struct CSRGraph {
  int n = 0;
  int64_t m = 0;
  int64_t input_edges_raw = 0;
  std::vector<int64_t> row_ptr;
  std::vector<int> col_idx;

  int64_t out_degree(int v) const {
    return row_ptr[static_cast<size_t>(v + 1)] - row_ptr[static_cast<size_t>(v)];
  }

  std::pair<const int*, const int*> neighbors(int v) const {
    size_t start = static_cast<size_t>(row_ptr[static_cast<size_t>(v)]);
    size_t end = static_cast<size_t>(row_ptr[static_cast<size_t>(v + 1)]);
    return {col_idx.data() + start, col_idx.data() + end};
  }

  bool HasArc(int u, int v) const {
    auto range = neighbors(u);
    return std::binary_search(range.first, range.second, v);
  }
};
