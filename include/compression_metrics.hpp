#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "sweg.hpp"

// Reporting-only metrics for the final flat encoding.  This deliberately does
// not participate in encoding or merge decisions.
struct CompressionMetrics {
  int64_t num_superedges_nonloop = 0;
  int64_t num_superedges_loop = 0;
  int64_t num_positive_corrections = 0;
  int64_t num_negative_corrections = 0;
  int64_t encoding_cost_standard = 0;
  double cost_ratio_standard = 0.0;
  int64_t encoding_cost_mags_x2 = 0;
  double encoding_cost_mags_compatible = 0.0;
  double cost_ratio_mags_compatible = 0.0;
};

inline int64_t CheckedCount(size_t value, const char* name) {
  if (value > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
    throw std::runtime_error(std::string("Count exceeds int64_t range: ") + name);
  }
  return static_cast<int64_t>(value);
}

inline int64_t CheckedAdd(int64_t left, int64_t right, const char* name) {
  if (right > std::numeric_limits<int64_t>::max() - left) {
    throw std::runtime_error(std::string("Metric count overflow: ") + name);
  }
  return left + right;
}

inline CompressionMetrics ComputeCompressionMetrics(
    const EncodingResult& result, int64_t original_undirected_edges,
    CostObjective objective = CostObjective::kLegacy) {
  if (original_undirected_edges < 0) {
    throw std::runtime_error("Original undirected edge count cannot be negative.");
  }

  CompressionMetrics metrics;
  for (const EdgePair& edge : result.P) {
    if (edge.first == edge.second) {
      metrics.num_superedges_loop = CheckedAdd(
          metrics.num_superedges_loop, 1, "num_superedges_loop");
    } else {
      metrics.num_superedges_nonloop = CheckedAdd(
          metrics.num_superedges_nonloop, 1, "num_superedges_nonloop");
    }
  }
  metrics.num_positive_corrections = CheckedCount(
      result.Cp.size(), "num_positive_corrections");
  metrics.num_negative_corrections = CheckedCount(
      result.Cm.size(), "num_negative_corrections");

  const int64_t num_superedges = CheckedAdd(
      metrics.num_superedges_nonloop, metrics.num_superedges_loop,
      "num_superedges");
  if (num_superedges != CheckedCount(result.P.size(), "P")) {
    throw std::runtime_error("Superedge count invariant violated.");
  }
  metrics.encoding_cost_standard = CheckedAdd(
      CheckedAdd(num_superedges, metrics.num_positive_corrections,
                 "encoding_cost_standard"),
      metrics.num_negative_corrections, "encoding_cost_standard");
  if (objective == CostObjective::kMagsCompatible) {
    if (metrics.encoding_cost_standard >
        std::numeric_limits<int64_t>::max() / 2) {
      throw std::runtime_error("MAGS-compatible metric count overflow");
    }
    metrics.encoding_cost_mags_x2 = metrics.encoding_cost_standard * 2;
  } else {
    metrics.encoding_cost_mags_x2 = CheckedAdd(
        metrics.encoding_cost_standard,
        metrics.encoding_cost_standard - metrics.num_superedges_loop,
        "encoding_cost_mags_x2");
  }
  if (metrics.encoding_cost_mags_x2 < 0) {
    throw std::runtime_error("MAGS-compatible encoding cost is negative.");
  }

  metrics.encoding_cost_mags_compatible =
      static_cast<double>(metrics.encoding_cost_mags_x2) / 2.0;
  if (original_undirected_edges > 0) {
    metrics.cost_ratio_standard =
        static_cast<double>(metrics.encoding_cost_standard) /
        static_cast<double>(original_undirected_edges);
    metrics.cost_ratio_mags_compatible =
        static_cast<double>(metrics.encoding_cost_mags_x2) /
        (2.0 * static_cast<double>(original_undirected_edges));
  }
  return metrics;
}
