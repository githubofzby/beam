#include "graph_io/graph_io.hpp"
#include "cuda_scoring.hpp"
#include "sweg.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

struct CliOptions {
  std::string input;
  std::string format = "edgelist";
  std::string dataset;
  std::string results_csv;
  MergeMode merge_mode = MergeMode::kSequential;
  ScoringBackend scoring_backend = ScoringBackend::kCpu;
  int top_k = 16;
  int group_batch_size = 64;
  int candidate_batch_budget = 0;
  int overflow_group_gmax = 0;
  int overflow_refine_rounds = 0;
  bool ea_use_threshold = false;
  int iterations = 1;
  int print_offset = 1;
  double error_bound = 0.0;
  std::string out_dir = "compressed";
  uint64_t seed = 1;
  bool has_header = false;
  bool make_undirected = false;
  bool verify_cuda_gain = false;
  bool show_help = false;
};

static void PrintUsage() {
  std::cout
      << "Usage:\n"
      << "  sweg_cpu --input <graph.txt> --format edgelist --out <dir>\n"
      << "           [--undirected] [--has-header] [--merge-mode ...]\n"
      << "  sweg_cpu --input <graph.csrbin> --format csrbin --out <dir>\n"
      << "  sweg_cpu <graph> <unused> <iterations> <print_offset>  (legacy, deprecated)\n"
      << "\n"
      << "Options:\n"
      << "  --input <path>           Input graph\n"
      << "  --format <edgelist|csrbin>\n"
      << "  --has-header            Edge list starts with 'n m'\n"
      << "  --directed              Treat edge list as directed (default)\n"
      << "  --undirected            Add reverse edges for each 'u v'\n"
      << "  --dataset <name>        Dataset name for metrics/CSV\n"
      << "  --results-csv <path>    Append one structured metrics row\n"
      << "  --merge-mode <sequential|batch-sj|batch-ea|batch-ea-frozen|batch-ea-blocked>\n"
      << "  --scoring-backend <cpu|cuda>\n"
      << "  --verify-cuda-gain      Debug hook for CPU/GPU gain checks\n"
      << "  --top-k <int>           Top-k SuperJaccard candidates for batch-ea\n"
      << "  --group-batch-size <int> Block size for batch-ea-blocked\n"
      << "  --candidate-batch-budget <int> Candidate budget per CUDA blocked batch\n"
      << "  --overflow-group-gmax <int> Max group size before local refinement (0 disables)\n"
      << "  --overflow-refine-rounds <int> Max local refinement rounds for overflow groups\n"
      << "  --ea-use-threshold      Enable local gain ratio threshold for batch-ea\n"
      << "  --iterations <int>      Number of iterations\n"
      << "  --print-offset <int>    Log divide stats every K iterations\n"
      << "  --error-bound <double>  Lossy drop ratio; 0.0 keeps lossless mode\n"
      << "  --out <dir>             Output directory\n"
      << "  --seed <int>            RNG seed\n";
}

static bool IsFlag(const std::string& arg) {
  return !arg.empty() && arg[0] == '-';
}

static MergeMode ParseMergeMode(const std::string& value) {
  if (value == "sequential") {
    return MergeMode::kSequential;
  }
  if (value == "batch-sj") {
    return MergeMode::kBatchSuperJaccard;
  }
  if (value == "batch-ea") {
    return MergeMode::kBatchEncodingAware;
  }
  if (value == "batch-ea-frozen") {
    return MergeMode::kBatchEncodingAwareFrozen;
  }
  if (value == "batch-ea-blocked") {
    return MergeMode::kBatchEncodingAwareBlocked;
  }
  throw std::runtime_error("Unsupported merge mode: " + value);
}

static ScoringBackend ParseScoringBackend(const std::string& value) {
  if (value == "cpu") {
    return ScoringBackend::kCpu;
  }
  if (value == "cuda") {
    return ScoringBackend::kCuda;
  }
  throw std::runtime_error("Unsupported scoring backend: " + value);
}

static std::string DetectDatasetName(const CliOptions& opts) {
  if (!opts.dataset.empty()) {
    return opts.dataset;
  }
  return std::filesystem::path(opts.input).stem().string();
}

static int64_t CountUndirectedEdges(const CSRGraph& graph) {
  int64_t count = 0;
  for (int u = 0; u < graph.n; ++u) {
    auto neighbors = graph.neighbors(u);
    for (const int* it = neighbors.first; it != neighbors.second; ++it) {
      const int v = *it;
      if (u == v) {
        ++count;
      } else if (u < v && graph.HasArc(v, u)) {
        ++count;
      }
    }
  }
  return count;
}

static std::string FormatDouble(double value) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6) << value;
  return oss.str();
}

static void PrintMetric(const std::string& key, const std::string& value) {
  std::cout << "metric." << key << "=" << value << "\n";
}

static void AppendResultsCsv(const std::string& path,
                             const std::vector<std::string>& columns,
                             const std::vector<std::string>& values) {
  const bool exists = std::filesystem::exists(path);
  if (!exists) {
    std::ofstream out(path, std::ios::app);
    if (!out) {
      throw std::runtime_error("Failed to open results CSV: " + path);
    }
    for (size_t i = 0; i < columns.size(); ++i) {
      if (i > 0) {
        out << ',';
      }
      out << columns[i];
    }
    out << '\n';
  } else {
    std::ifstream in(path);
    if (!in) {
      throw std::runtime_error("Failed to read existing results CSV: " + path);
    }
    std::string header;
    std::getline(in, header);
    if (!header.empty() && header.back() == '\r') {
      header.pop_back();
    }
    std::ostringstream expected;
    for (size_t i = 0; i < columns.size(); ++i) {
      if (i > 0) {
        expected << ',';
      }
      expected << columns[i];
    }
    if (header != expected.str()) {
      throw std::runtime_error(
          "Existing results CSV header does not match current schema: " + path);
    }
  }

  std::ofstream out(path, std::ios::app);
  if (!out) {
    throw std::runtime_error("Failed to open results CSV: " + path);
  }
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      out << ',';
    }
    out << values[i];
  }
  out << '\n';
}

static CliOptions ParseArgs(int argc, char** argv) {
  CliOptions opts;
  if (argc >= 5 && !IsFlag(argv[1])) {
    opts.input = argv[1];
    opts.iterations = std::stoi(argv[3]);
    opts.print_offset = std::stoi(argv[4]);
    return opts;
  }

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      opts.show_help = true;
      return opts;
    }
    if (arg == "--input" && i + 1 < argc) {
      opts.input = argv[++i];
    } else if (arg == "--dataset" && i + 1 < argc) {
      opts.dataset = argv[++i];
    } else if (arg == "--results-csv" && i + 1 < argc) {
      opts.results_csv = argv[++i];
    } else if ((arg == "--format" || arg == "--input-format") && i + 1 < argc) {
      opts.format = argv[++i];
    } else if (arg == "--has-header") {
      opts.has_header = true;
    } else if (arg == "--no-header") {
      opts.has_header = false;
    } else if (arg == "--undirected") {
      opts.make_undirected = true;
    } else if (arg == "--directed") {
      opts.make_undirected = false;
    } else if (arg == "--merge-mode" && i + 1 < argc) {
      opts.merge_mode = ParseMergeMode(argv[++i]);
    } else if (arg == "--scoring-backend" && i + 1 < argc) {
      opts.scoring_backend = ParseScoringBackend(argv[++i]);
    } else if (arg == "--verify-cuda-gain") {
      opts.verify_cuda_gain = true;
    } else if (arg == "--top-k" && i + 1 < argc) {
      opts.top_k = std::stoi(argv[++i]);
    } else if (arg == "--group-batch-size" && i + 1 < argc) {
      opts.group_batch_size = std::stoi(argv[++i]);
    } else if (arg == "--candidate-batch-budget" && i + 1 < argc) {
      opts.candidate_batch_budget = std::stoi(argv[++i]);
    } else if (arg == "--overflow-group-gmax" && i + 1 < argc) {
      opts.overflow_group_gmax = std::stoi(argv[++i]);
    } else if (arg == "--overflow-refine-rounds" && i + 1 < argc) {
      opts.overflow_refine_rounds = std::stoi(argv[++i]);
    } else if (arg == "--ea-use-threshold") {
      opts.ea_use_threshold = true;
    } else if (arg == "--iterations" && i + 1 < argc) {
      opts.iterations = std::stoi(argv[++i]);
    } else if (arg == "--print-offset" && i + 1 < argc) {
      opts.print_offset = std::stoi(argv[++i]);
    } else if (arg == "--error-bound" && i + 1 < argc) {
      opts.error_bound = std::stod(argv[++i]);
    } else if (arg == "--out" && i + 1 < argc) {
      opts.out_dir = argv[++i];
    } else if (arg == "--seed" && i + 1 < argc) {
      opts.seed = static_cast<uint64_t>(std::stoull(argv[++i]));
    } else {
      std::cerr << "Unknown or incomplete argument: " << arg << "\n";
      opts.show_help = true;
      return opts;
    }
  }
  return opts;
}

int main(int argc, char** argv) {
  try {
    CliOptions opts = ParseArgs(argc, argv);
    if (opts.show_help || opts.input.empty()) {
      PrintUsage();
      return opts.input.empty() ? 1 : 0;
    }

    CSRGraph graph;
    if (opts.format == "edgelist") {
      graph = LoadEdgeList(opts.input, opts.has_header, opts.make_undirected);
    } else if (opts.format == "csrbin") {
      graph = LoadCsrBinary(opts.input);
    } else {
      std::cerr << "Unsupported format: " << opts.format << "\n";
      return 1;
    }

    std::filesystem::create_directories(opts.out_dir);

    const std::string dataset = DetectDatasetName(opts);
    const int64_t input_edges_raw = graph.input_edges_raw;
    const int64_t graph_arcs = graph.m;
    const int64_t original_undirected_edges =
        opts.make_undirected ? input_edges_raw : CountUndirectedEdges(graph);
    const int64_t original_edges_eval = original_undirected_edges;

    std::cout << "Loaded graph: n=" << graph.n << " m=" << graph.m << "\n";
    std::cout << "Output directory: " << opts.out_dir << "\n";
    std::cout << "Seed: " << opts.seed << "\n";
    std::cout << "Merge mode: " << MergeModeToString(opts.merge_mode) << "\n";
    std::cout << "Scoring backend: "
              << ScoringBackendToString(opts.scoring_backend) << "\n";
    std::cout << "CUDA scoring build: " << CudaScoringBuildMode() << "\n";

    Sweg sweg(graph, opts.merge_mode, opts.top_k, opts.ea_use_threshold,
              opts.seed, opts.scoring_backend, opts.verify_cuda_gain,
              opts.group_batch_size, opts.candidate_batch_budget,
              opts.overflow_group_gmax, opts.overflow_refine_rounds);
    sweg.Run(opts.iterations, opts.print_offset);

    const auto encode_start = std::chrono::steady_clock::now();
    EncodingResult result = sweg.Encode();
    const auto encode_end = std::chrono::steady_clock::now();
    RuntimeStats stats = sweg.runtime_stats();
    stats.runtime_encode_ms = std::chrono::duration<double, std::milli>(
                                  encode_end - encode_start)
                                  .count();

    sweg.Drop(opts.error_bound, result);

    const auto output_start = std::chrono::steady_clock::now();
    sweg.WriteOutput(opts.out_dir, result);
    const auto output_end = std::chrono::steady_clock::now();
    stats.runtime_output_ms =
        std::chrono::duration<double, std::milli>(output_end - output_start)
            .count();
    stats.runtime_total_ms += stats.runtime_encode_ms + stats.runtime_output_ms;

    const size_t encoded_edges =
        result.P.size() + result.Cp.size() + result.Cm.size();
    const double cost_ratio =
        original_edges_eval > 0
            ? static_cast<double>(encoded_edges) /
                  static_cast<double>(original_edges_eval)
            : 0.0;
    const double compression_gain = 1.0 - cost_ratio;
    const double compression_java_style = compression_gain;
    const std::string merge_mode = MergeModeToString(opts.merge_mode);
    const std::string scoring_backend =
        ScoringBackendToString(opts.scoring_backend);

    PrintMetric("dataset", dataset);
    PrintMetric("n", std::to_string(graph.n));
    PrintMetric("input_edges_raw", std::to_string(input_edges_raw));
    PrintMetric("graph_arcs", std::to_string(graph_arcs));
    PrintMetric("original_undirected_edges",
                std::to_string(original_undirected_edges));
    PrintMetric("original_edges_eval", std::to_string(original_edges_eval));
    PrintMetric("iterations", std::to_string(opts.iterations));
    PrintMetric("seed", std::to_string(opts.seed));
    PrintMetric("merge_mode", merge_mode);
    PrintMetric("scoring_backend", scoring_backend);
    PrintMetric("top_k", std::to_string(opts.top_k));
    PrintMetric("group_batch_size", std::to_string(opts.group_batch_size));
    PrintMetric("candidate_batch_budget",
                std::to_string(opts.candidate_batch_budget));
    PrintMetric("overflow_group_gmax",
                std::to_string(opts.overflow_group_gmax));
    PrintMetric("overflow_refine_rounds",
                std::to_string(opts.overflow_refine_rounds));
    PrintMetric("error_bound", FormatDouble(opts.error_bound));
    PrintMetric("supernode_count", std::to_string(result.supernodes.size()));
    PrintMetric("P", std::to_string(result.P.size()));
    PrintMetric("Cp", std::to_string(result.Cp.size()));
    PrintMetric("Cm", std::to_string(result.Cm.size()));
    PrintMetric("encoding_cost", std::to_string(encoded_edges));
    PrintMetric("cost_ratio", FormatDouble(cost_ratio));
    PrintMetric("compression_gain", FormatDouble(compression_gain));
    PrintMetric("compression_java_style", FormatDouble(compression_java_style));
    PrintMetric("runtime_total_ms", FormatDouble(stats.runtime_total_ms));
    PrintMetric("runtime_divide_ms", FormatDouble(stats.runtime_divide_ms));
    PrintMetric("runtime_merge_ms", FormatDouble(stats.runtime_merge_ms));
    PrintMetric("runtime_encode_ms", FormatDouble(stats.runtime_encode_ms));
    PrintMetric("runtime_output_ms", FormatDouble(stats.runtime_output_ms));
    PrintMetric("merge_create_w_ms", FormatDouble(stats.merge_create_w_ms));
    PrintMetric("merge_scoring_ms", FormatDouble(stats.merge_scoring_ms));
    PrintMetric("merge_candidate_gen_ms",
                FormatDouble(stats.merge_candidate_gen_ms));
    PrintMetric("merge_gain_scoring_ms",
                FormatDouble(stats.merge_gain_scoring_ms));
    PrintMetric("merge_selection_ms", FormatDouble(stats.merge_selection_ms));
    PrintMetric("merge_update_ms", FormatDouble(stats.merge_update_ms));
    PrintMetric("merge_candidate_pairs",
                std::to_string(stats.merge_candidate_pairs));
    PrintMetric("merge_selected_pairs",
                std::to_string(stats.merge_selected_pairs));
    PrintMetric("merge_total_local_gain",
                std::to_string(stats.merge_total_local_gain));
    const double avg_local_gain =
        stats.merge_selected_pairs > 0
            ? static_cast<double>(stats.merge_total_local_gain) /
                  static_cast<double>(stats.merge_selected_pairs)
            : 0.0;
    PrintMetric("merge_avg_local_gain", FormatDouble(avg_local_gain));
    PrintMetric("cuda_h2d_ms", FormatDouble(stats.cuda_h2d_ms));
    PrintMetric("cuda_kernel_ms", FormatDouble(stats.cuda_kernel_ms));
    PrintMetric("cuda_d2h_ms", FormatDouble(stats.cuda_d2h_ms));
    PrintMetric("cuda_total_ms", FormatDouble(stats.cuda_total_ms));
    PrintMetric("cuda_num_calls", std::to_string(stats.cuda_num_calls));
    PrintMetric("overflow_groups_seen",
                std::to_string(stats.overflow_groups_seen));
    PrintMetric("overflow_refined_subgroups",
                std::to_string(stats.overflow_refined_subgroups));
    PrintMetric("overflow_forced_chunks",
                std::to_string(stats.overflow_forced_chunks));
    PrintMetric("overflow_max_group_before",
                std::to_string(stats.overflow_max_group_before));
    PrintMetric("overflow_max_group_after",
                std::to_string(stats.overflow_max_group_after));
    PrintMetric("overflow_refine_ms",
                FormatDouble(stats.overflow_refine_ms));
    PrintMetric("encode_relabel_ms", FormatDouble(stats.encode_relabel_ms));
    PrintMetric("encode_count_edges_ms",
                FormatDouble(stats.encode_count_edges_ms));
    PrintMetric("encode_decision_ms", FormatDouble(stats.encode_decision_ms));
    PrintMetric("encode_correction_generation_ms",
                FormatDouble(stats.encode_correction_generation_ms));
    PrintMetric("reconstruction_pass", "unknown");

    if (!opts.results_csv.empty()) {
      const std::vector<std::string> columns = {
          "dataset",
          "n",
          "input_edges_raw",
          "graph_arcs",
          "original_undirected_edges",
          "original_edges_eval",
          "iterations",
          "seed",
          "merge_mode",
          "scoring_backend",
          "top_k",
          "group_batch_size",
          "candidate_batch_budget",
          "overflow_group_gmax",
          "overflow_refine_rounds",
          "error_bound",
          "supernode_count",
          "P",
          "Cp",
          "Cm",
          "encoding_cost",
          "cost_ratio",
          "compression_gain",
          "compression_java_style",
          "runtime_total_ms",
          "runtime_divide_ms",
          "runtime_merge_ms",
          "runtime_encode_ms",
          "runtime_output_ms",
          "merge_create_w_ms",
          "merge_scoring_ms",
          "merge_candidate_gen_ms",
          "merge_gain_scoring_ms",
          "merge_selection_ms",
          "merge_update_ms",
          "merge_candidate_pairs",
          "merge_selected_pairs",
          "merge_total_local_gain",
          "merge_avg_local_gain",
          "cuda_h2d_ms",
          "cuda_kernel_ms",
          "cuda_d2h_ms",
          "cuda_total_ms",
          "cuda_num_calls",
          "overflow_groups_seen",
          "overflow_refined_subgroups",
          "overflow_forced_chunks",
          "overflow_max_group_before",
          "overflow_max_group_after",
          "overflow_refine_ms",
          "encode_relabel_ms",
          "encode_count_edges_ms",
          "encode_decision_ms",
          "encode_correction_generation_ms",
          "reconstruction_pass",
      };
      const std::vector<std::string> values = {
          dataset,
          std::to_string(graph.n),
          std::to_string(input_edges_raw),
          std::to_string(graph_arcs),
          std::to_string(original_undirected_edges),
          std::to_string(original_edges_eval),
          std::to_string(opts.iterations),
          std::to_string(opts.seed),
          merge_mode,
          scoring_backend,
          std::to_string(opts.top_k),
          std::to_string(opts.group_batch_size),
          std::to_string(opts.candidate_batch_budget),
          std::to_string(opts.overflow_group_gmax),
          std::to_string(opts.overflow_refine_rounds),
          FormatDouble(opts.error_bound),
          std::to_string(result.supernodes.size()),
          std::to_string(result.P.size()),
          std::to_string(result.Cp.size()),
          std::to_string(result.Cm.size()),
          std::to_string(encoded_edges),
          FormatDouble(cost_ratio),
          FormatDouble(compression_gain),
          FormatDouble(compression_java_style),
          FormatDouble(stats.runtime_total_ms),
          FormatDouble(stats.runtime_divide_ms),
          FormatDouble(stats.runtime_merge_ms),
          FormatDouble(stats.runtime_encode_ms),
          FormatDouble(stats.runtime_output_ms),
          FormatDouble(stats.merge_create_w_ms),
          FormatDouble(stats.merge_scoring_ms),
          FormatDouble(stats.merge_candidate_gen_ms),
          FormatDouble(stats.merge_gain_scoring_ms),
          FormatDouble(stats.merge_selection_ms),
          FormatDouble(stats.merge_update_ms),
          std::to_string(stats.merge_candidate_pairs),
          std::to_string(stats.merge_selected_pairs),
          std::to_string(stats.merge_total_local_gain),
          FormatDouble(avg_local_gain),
          FormatDouble(stats.cuda_h2d_ms),
          FormatDouble(stats.cuda_kernel_ms),
          FormatDouble(stats.cuda_d2h_ms),
          FormatDouble(stats.cuda_total_ms),
          std::to_string(stats.cuda_num_calls),
          std::to_string(stats.overflow_groups_seen),
          std::to_string(stats.overflow_refined_subgroups),
          std::to_string(stats.overflow_forced_chunks),
          std::to_string(stats.overflow_max_group_before),
          std::to_string(stats.overflow_max_group_after),
          FormatDouble(stats.overflow_refine_ms),
          FormatDouble(stats.encode_relabel_ms),
          FormatDouble(stats.encode_count_edges_ms),
          FormatDouble(stats.encode_decision_ms),
          FormatDouble(stats.encode_correction_generation_ms),
          "unknown",
      };
      AppendResultsCsv(opts.results_csv, columns, values);
    }
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return 1;
  }
}
