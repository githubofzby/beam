#pragma once

#include <cstdint>
#include <vector>

#include "graph_io/graph.hpp"

using EncodingCost = int64_t;
using MergeGain = int64_t;

// Exact integer objective for simple undirected graphs without self-loops.
class CostOracle {
 public:
  EncodingCost BlockCost(int64_t size_a, int64_t size_b,
                         int64_t edge_count, bool internal_block) const;

  MergeGain ExactMergeGain(const CSRGraph& graph,
                           const std::vector<int>& labels,
                           int label_a, int label_b) const;

  EncodingCost ExactPartitionCost(const CSRGraph& graph,
                                  const std::vector<int>& labels) const;
};
