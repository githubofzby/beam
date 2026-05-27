#pragma once

#include <string>

#include "graph_io/graph.hpp"

enum class GraphInputFormat {
  kEdgeList,
  kCsrBinary
};

CSRGraph LoadEdgeList(const std::string& path,
                      bool has_header = false,
                      bool make_undirected = false);
CSRGraph LoadCsrBinary(const std::string& path);
