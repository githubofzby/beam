#include "graph_io/graph_io.hpp"
#include "graph_io/utils.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

CSRGraph LoadEdgeList(const std::string& path, bool has_header,
                      bool make_undirected) {
  int64_t n = 0;
  int64_t m = 0;
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Failed to open file: " + path);
  }

  std::vector<std::pair<int, int>> edges;
  int64_t max_node = -1;
  int64_t raw_edges = 0;
  bool header_consumed = !has_header;
  std::string line;

  while (std::getline(in, line)) {
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
    if (!header_consumed) {
      if (!(iss >> n >> m)) {
        throw std::runtime_error("Failed to read edge list header (n m).");
      }
      header_consumed = true;
      if (m > 0) {
        edges.reserve(static_cast<size_t>(m));
      }
      continue;
    }

    if (!(iss >> u64 >> v64)) {
      throw std::runtime_error("Malformed edge list line: expected 'u v'.");
    }

    const int u = static_cast<int>(u64);
    const int v = static_cast<int>(v64);
    ++raw_edges;
    edges.emplace_back(u, v);
    if (make_undirected && u != v) {
      edges.emplace_back(v, u);
    }
    if (u > max_node) {
      max_node = u;
    }
    if (v > max_node) {
      max_node = v;
    }
  }

  if (!has_header) {
    n = max_node + 1;
    m = static_cast<int64_t>(edges.size());
  } else if (!header_consumed) {
    throw std::runtime_error("Failed to read edge list header (n m).");
  } else if (m != static_cast<int64_t>(edges.size())) {
    std::cerr << "Warning: header m (" << m << ") does not match edge count ("
              << edges.size() << "). Using edge count.\n";
    m = static_cast<int64_t>(edges.size());
  }

  if (n <= 0) {
    throw std::runtime_error("Invalid node count.");
  }

  if (max_node >= n) {
    std::cerr << "Warning: max node id (" << max_node << ") exceeds n-1 ("
              << (n - 1) << "). Expanding n.\n";
    n = max_node + 1;
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

  return g;
}

CSRGraph LoadCsrBinary(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Failed to open CSR binary: " + path);
  }

  int64_t n = 0;
  int64_t m = 0;
  in.read(reinterpret_cast<char*>(&n), sizeof(int64_t));
  in.read(reinterpret_cast<char*>(&m), sizeof(int64_t));
  if (!in) {
    throw std::runtime_error("Failed to read CSR header.");
  }

  CSRGraph g;
  g.n = static_cast<int>(n);
  g.m = m;
  g.input_edges_raw = m;
  g.row_ptr.resize(static_cast<size_t>(g.n + 1));
  g.col_idx.resize(static_cast<size_t>(g.m));

  in.read(reinterpret_cast<char*>(g.row_ptr.data()),
          static_cast<std::streamsize>((g.n + 1) * sizeof(int64_t)));
  in.read(reinterpret_cast<char*>(g.col_idx.data()),
          static_cast<std::streamsize>(g.m * sizeof(int)));

  if (!in) {
    throw std::runtime_error("Failed to read CSR arrays.");
  }

  for (int v = 0; v < g.n; ++v) {
    auto start = static_cast<size_t>(g.row_ptr[static_cast<size_t>(v)]);
    auto end = static_cast<size_t>(g.row_ptr[static_cast<size_t>(v + 1)]);
    std::sort(g.col_idx.begin() + static_cast<std::ptrdiff_t>(start),
              g.col_idx.begin() + static_cast<std::ptrdiff_t>(end));
  }

  return g;
}
