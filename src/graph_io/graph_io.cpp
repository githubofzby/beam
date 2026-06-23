#include "graph_io/graph_io.hpp"

#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

CSRGraph LoadUndirectedEdgeList(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Failed to open file: " + path);
  }

  std::vector<std::pair<int, int>> edges;
  int64_t max_node = -1;
  int64_t raw_edges = 0;
  std::string line;
  int64_t line_no = 0;

  while (std::getline(in, line)) {
    ++line_no;
    const auto first = line.find_first_not_of(" \t\r");
    if (first == std::string::npos) {
      continue;
    }
    if (line[first] == '#') {
      continue;
    }

    std::istringstream iss(line);
    int64_t u64 = 0;
    int64_t v64 = 0;
    if (!(iss >> u64 >> v64)) {
      throw std::runtime_error("Malformed edge list line " +
                               std::to_string(line_no) +
                               ": expected at least two integers 'u v'.");
    }
    if (u64 < 0 || v64 < 0) {
      throw std::runtime_error("Negative node id at line " +
                               std::to_string(line_no) + ".");
    }
    if (u64 > std::numeric_limits<int>::max() ||
        v64 > std::numeric_limits<int>::max()) {
      throw std::runtime_error("Node id exceeds int range at line " +
                               std::to_string(line_no) + ".");
    }

    const int u = static_cast<int>(u64);
    const int v = static_cast<int>(v64);
    if (u == v) {
      throw std::runtime_error("Self-loop is not supported at line " +
                               std::to_string(line_no) + ".");
    }
    ++raw_edges;
    edges.emplace_back(u, v);
    edges.emplace_back(v, u);
    if (u > max_node) {
      max_node = u;
    }
    if (v > max_node) {
      max_node = v;
    }
  }

  const int64_t n = max_node + 1;
  if (n <= 0) {
    throw std::runtime_error("Invalid node count.");
  }

  CSRGraph g;
  g.n = static_cast<int>(n);
  g.m = static_cast<int64_t>(edges.size());
  g.input_edges_raw = raw_edges;
  g.row_ptr.assign(static_cast<size_t>(g.n + 1), 0);

  for (const auto& e : edges) {
    if (e.first < 0 || e.first >= g.n) {
      throw std::runtime_error("Edge list has node id out of range.");
    }
    g.row_ptr[static_cast<size_t>(e.first + 1)]++;
  }

  for (int i = 0; i < g.n; ++i) {
    g.row_ptr[static_cast<size_t>(i + 1)] += g.row_ptr[static_cast<size_t>(i)];
  }

  g.col_idx.assign(static_cast<size_t>(g.m), 0);
  std::vector<int64_t> cur = g.row_ptr;
  for (const auto& e : edges) {
    size_t idx = static_cast<size_t>(cur[static_cast<size_t>(e.first)]++);
    g.col_idx[idx] = e.second;
  }

  for (int v = 0; v < g.n; ++v) {
    auto start = static_cast<size_t>(g.row_ptr[static_cast<size_t>(v)]);
    auto end = static_cast<size_t>(g.row_ptr[static_cast<size_t>(v + 1)]);
    std::sort(g.col_idx.begin() + static_cast<std::ptrdiff_t>(start),
              g.col_idx.begin() + static_cast<std::ptrdiff_t>(end));
  }

  return g;
}
