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
  std::string results_csv;
  MergeMode merge_mode = MergeMode::kBatchEncodingAware;
  ScoringBackend scoring_backend = ScoringBackend::kCpu;
  ThresholdPolicy threshold_policy = ThresholdPolicy::kReciprocal;
  int top_k = 16;
  int group_batch_size = 64;
  int candidate_batch_budget = 0;
  int cuda_slice_memory_mb = 0;
  int overflow_group_gmax = 0;
  int overflow_refine_rounds = 0;
  bool ea_use_threshold = false;
  int iterations = 20;
  int print_offset = 1;
  int divide_hash_dims = 16;
  int divide_max_group = 512;
  double error_bound = 0.0;
  double threshold_high = 0.5;
  double threshold_low = 0.005;
  double threshold_min_low = 0.005;
  double adaptive_q_high = 0.85;
  double adaptive_q_low = 0.15;
  int adaptive_sample_limit = 4096;
  double acceptance_target = 0.15;
  std::string out_dir;
  uint64_t seed = 1;
  bool write_output = false;
  bool verify_cuda_gain = false;
  bool show_help = false;
};

static void PrintUsage() {
  std::cout
      << "Usage:\n"
      << "  beam --input <undirected-single-edge-list> [options]\n"
      << "\n"
      << "Input format:\n"
      << "  Plain text undirected edge list.\n"
      << "  Each non-comment line contains: u v\n"
      << "  Each undirected edge must appear once.\n"
      << "  The program automatically adds reverse arcs internally.\n"
      << "\n"
      << "Options:\n"
      << "  --input <path>           Input graph\n"
      << "  --results-csv <path>    Append one structured metrics row\n"
      << "  --merge-mode <batch-ea|batch-ea-blocked>\n"
      << "  --scoring-backend <cpu|cuda>\n"
      << "  --verify-cuda-gain      Debug hook for CPU/GPU gain checks\n"
      << "  --top-k <int>           Top-k EA-proxy candidates per supernode for batch-ea modes\n"
      << "  --group-batch-size <int> Block size for batch-ea-blocked\n"
      << "  --candidate-batch-budget <int> Candidate budget per CUDA blocked batch\n"
      << "  --cuda-slice-memory-mb <int> Max estimated CUDA memory per blocked slice in MiB (0=auto)\n"
      << "  --overflow-group-gmax <int> Max group size before local refinement (0 disables)\n"
      << "  --overflow-refine-rounds <int> Max local refinement rounds for overflow groups\n"
      << "  --ea-use-threshold      Enable local gain ratio threshold for batch-ea modes\n"
      << "  --threshold-policy <reciprocal|mags-geom|adaptive>\n"
      << "  --threshold-high <double> Default 0.5\n"
      << "  --threshold-low <double> Default 0.005\n"
      << "  --threshold-min-low <double> Default 0.005\n"
      << "  --adaptive-q-high <double> Default 0.85\n"
      << "  --adaptive-q-low <double> Default 0.15\n"
      << "  --adaptive-sample-limit <int> Default 4096\n"
      << "  --acceptance-target <double> Default 0.15\n"
      << "  --iterations <int>      Number of iterations\n"
      << "  --print-offset <int>    Log divide stats every K iterations\n"
      << "  --divide-hash-dims <int> Number of hash dimensions for adaptive divide\n"
      << "  --divide-max-group <int> Maximum target group size for adaptive divide\n"
      << "  --error-bound <double>  Lossy drop ratio; 0.0 keeps lossless mode\n"
      << "  --write-output          Write G.txt / P.txt / Cp.txt / Cm.txt\n"
      << "  --out <dir>             Output directory; required with --write-output\n"
      << "  --seed <int>            RNG seed\n";
}

static MergeMode ParseMergeMode(const std::string& value) {
  if (value == "batch-ea") {
    return MergeMode::kBatchEncodingAware;
  }
  if (value == "batch-ea-blocked") {
    return MergeMode::kBatchEncodingAwareBlocked;
  }
  throw std::runtime_error(
      "Unsupported merge mode: " + value +
      ". Supported modes: batch-ea, batch-ea-blocked");
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

static ThresholdPolicy ParseThresholdPolicy(const std::string& value) {
  if (value == "reciprocal") {
    return ThresholdPolicy::kReciprocal;
  }
  if (value == "mags-geom") {
    return ThresholdPolicy::kMagsGeom;
  }
  if (value == "adaptive") {
    return ThresholdPolicy::kAdaptive;
  }
  throw std::runtime_error("Unsupported threshold policy: " + value);
}

static std::string FormatDouble(double value) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6) << value;
  return oss.str();
}

static double SafeMetricRatio(uint64_t numerator, uint64_t denominator) {
  return denominator > 0
             ? static_cast<double>(numerator) /
                   static_cast<double>(denominator)
             : 0.0;
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
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      opts.show_help = true;
      return opts;
    }
    if (arg == "--input" && i + 1 < argc) {
      opts.input = argv[++i];
    } else if (arg == "--results-csv" && i + 1 < argc) {
      opts.results_csv = argv[++i];
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
    } else if (arg == "--cuda-slice-memory-mb" && i + 1 < argc) {
      opts.cuda_slice_memory_mb = std::stoi(argv[++i]);
    } else if (arg == "--overflow-group-gmax" && i + 1 < argc) {
      opts.overflow_group_gmax = std::stoi(argv[++i]);
    } else if (arg == "--overflow-refine-rounds" && i + 1 < argc) {
      opts.overflow_refine_rounds = std::stoi(argv[++i]);
    } else if (arg == "--ea-use-threshold") {
      opts.ea_use_threshold = true;
    } else if (arg == "--threshold-policy" && i + 1 < argc) {
      opts.threshold_policy = ParseThresholdPolicy(argv[++i]);
    } else if (arg == "--threshold-high" && i + 1 < argc) {
      opts.threshold_high = std::stod(argv[++i]);
    } else if (arg == "--threshold-low" && i + 1 < argc) {
      opts.threshold_low = std::stod(argv[++i]);
    } else if (arg == "--threshold-min-low" && i + 1 < argc) {
      opts.threshold_min_low = std::stod(argv[++i]);
    } else if (arg == "--adaptive-q-high" && i + 1 < argc) {
      opts.adaptive_q_high = std::stod(argv[++i]);
    } else if (arg == "--adaptive-q-low" && i + 1 < argc) {
      opts.adaptive_q_low = std::stod(argv[++i]);
    } else if (arg == "--adaptive-sample-limit" && i + 1 < argc) {
      opts.adaptive_sample_limit = std::stoi(argv[++i]);
    } else if (arg == "--acceptance-target" && i + 1 < argc) {
      opts.acceptance_target = std::stod(argv[++i]);
    } else if (arg == "--iterations" && i + 1 < argc) {
      opts.iterations = std::stoi(argv[++i]);
    } else if (arg == "--print-offset" && i + 1 < argc) {
      opts.print_offset = std::stoi(argv[++i]);
    } else if (arg == "--divide-hash-dims" && i + 1 < argc) {
      opts.divide_hash_dims = std::stoi(argv[++i]);
    } else if (arg == "--divide-max-group" && i + 1 < argc) {
      opts.divide_max_group = std::stoi(argv[++i]);
    } else if (arg == "--error-bound" && i + 1 < argc) {
      opts.error_bound = std::stod(argv[++i]);
    } else if (arg == "--write-output") {
      opts.write_output = true;
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
    if (opts.write_output && opts.out_dir.empty()) {
      throw std::runtime_error(
          "--write-output requires --out <dir>.");
    }

    CSRGraph graph = LoadUndirectedEdgeList(opts.input);
    if (graph.m != graph.input_edges_raw * 2) {
      throw std::runtime_error(
          "Input graph invariant violated: graph_arcs must equal 2 * input_edges_raw.");
    }

    const int64_t input_edges_raw = graph.input_edges_raw;
    const int64_t graph_arcs = graph.m;
    const int64_t original_undirected_edges = input_edges_raw;
    const int64_t original_edges_eval = input_edges_raw;
    if (opts.divide_hash_dims <= 0) {
      std::cerr << "Warning: invalid --divide-hash-dims=" << opts.divide_hash_dims
                << ", fallback to 16\n";
      opts.divide_hash_dims = 16;
    }
    if (opts.divide_max_group <= 0) {
      std::cerr << "Warning: invalid --divide-max-group=" << opts.divide_max_group
                << ", fallback to 512\n";
      opts.divide_max_group = 512;
    }
    if (opts.adaptive_sample_limit <= 0) {
      std::cerr << "Warning: invalid --adaptive-sample-limit="
                << opts.adaptive_sample_limit << ", fallback to 4096\n";
      opts.adaptive_sample_limit = 4096;
    }

    std::cout << "Loaded graph: n=" << graph.n << " m=" << graph.m << "\n";
    std::cout << "Seed: " << opts.seed << "\n";
    std::cout << "Divide hash dims: " << opts.divide_hash_dims << "\n";
    std::cout << "Divide max group: " << opts.divide_max_group << "\n";
    std::cout << "Merge mode: " << MergeModeToString(opts.merge_mode) << "\n";
    std::cout << "Scoring backend: "
              << ScoringBackendToString(opts.scoring_backend) << "\n";
    std::cout << "Threshold policy: "
              << ThresholdPolicyToString(opts.threshold_policy) << "\n";
    std::cout << "CUDA scoring build: " << CudaScoringBuildMode() << "\n";
    if (opts.write_output) {
      std::cout << "Output directory: " << opts.out_dir << "\n";
    } else {
      std::cout << "Output writing: disabled\n";
    }

    ThresholdConfig threshold_config;
    threshold_config.policy = opts.threshold_policy;
    threshold_config.high = opts.threshold_high;
    threshold_config.low = opts.threshold_low;
    threshold_config.min_low = opts.threshold_min_low;
    threshold_config.q_high = opts.adaptive_q_high;
    threshold_config.q_low = opts.adaptive_q_low;
    threshold_config.sample_limit = opts.adaptive_sample_limit;
    threshold_config.acceptance_target = opts.acceptance_target;

    Sweg sweg(graph, opts.merge_mode, opts.top_k, opts.ea_use_threshold,
              opts.seed, opts.scoring_backend, opts.verify_cuda_gain,
              opts.group_batch_size, opts.candidate_batch_budget,
              opts.cuda_slice_memory_mb,
              opts.overflow_group_gmax, opts.overflow_refine_rounds,
              opts.divide_hash_dims, opts.divide_max_group, threshold_config);
    sweg.Run(opts.iterations, opts.print_offset);

    const auto encode_start = std::chrono::steady_clock::now();
    EncodingResult result = sweg.Encode();
    const auto encode_end = std::chrono::steady_clock::now();
    RuntimeStats stats = sweg.runtime_stats();
    stats.runtime_encode_ms = std::chrono::duration<double, std::milli>(
                                  encode_end - encode_start)
                                  .count();

    sweg.Drop(opts.error_bound, result);

    if (opts.write_output) {
      std::filesystem::create_directories(opts.out_dir);
      const auto output_start = std::chrono::steady_clock::now();
      sweg.WriteOutput(opts.out_dir, result);
      const auto output_end = std::chrono::steady_clock::now();
      stats.runtime_output_ms =
          std::chrono::duration<double, std::milli>(output_end - output_start)
              .count();
    }

    const double runtime_algorithm_ms =
        stats.runtime_run_ms + stats.runtime_encode_ms;
    const double runtime_end_to_end_ms =
        runtime_algorithm_ms + stats.runtime_output_ms;

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
    const std::string threshold_policy =
        ThresholdPolicyToString(opts.threshold_policy);

    PrintMetric("n", std::to_string(graph.n));
    PrintMetric("input_edges_raw", std::to_string(input_edges_raw));
    PrintMetric("graph_arcs", std::to_string(graph_arcs));
    PrintMetric("original_undirected_edges",
                std::to_string(original_undirected_edges));
    PrintMetric("original_edges_eval", std::to_string(original_edges_eval));
    PrintMetric("iterations", std::to_string(opts.iterations));
    PrintMetric("seed", std::to_string(opts.seed));
    PrintMetric("divide_hash_dims", std::to_string(opts.divide_hash_dims));
    PrintMetric("divide_max_group", std::to_string(opts.divide_max_group));
    PrintMetric("merge_mode", merge_mode);
    PrintMetric("scoring_backend", scoring_backend);
    PrintMetric("threshold_policy", threshold_policy);
    PrintMetric("top_k", std::to_string(opts.top_k));
    PrintMetric("group_batch_size", std::to_string(opts.group_batch_size));
    PrintMetric("candidate_batch_budget",
                std::to_string(opts.candidate_batch_budget));
    PrintMetric("cuda_slice_memory_mb",
                std::to_string(opts.cuda_slice_memory_mb));
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
    PrintMetric("runtime_run_ms", FormatDouble(stats.runtime_run_ms));
    PrintMetric("runtime_divide_ms", FormatDouble(stats.runtime_divide_ms));
    PrintMetric("runtime_merge_ms", FormatDouble(stats.runtime_merge_ms));
    PrintMetric("divide_max_group_size",
                std::to_string(stats.divide_max_group_size));
    PrintMetric("divide_fallback_splits",
                std::to_string(stats.divide_fallback_splits));
    PrintMetric("runtime_encode_ms", FormatDouble(stats.runtime_encode_ms));
    PrintMetric("runtime_output_ms", FormatDouble(stats.runtime_output_ms));
    PrintMetric("runtime_algorithm_ms", FormatDouble(runtime_algorithm_ms));
    PrintMetric("runtime_end_to_end_ms", FormatDouble(runtime_end_to_end_ms));
    PrintMetric("merge_prepare_wall_ms",
                FormatDouble(stats.merge_prepare_wall_ms));
    PrintMetric("merge_prepare_task_sum_ms",
                FormatDouble(stats.merge_prepare_task_sum_ms));
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
    stats.merge_exact_gain_calls_per_selected =
        SafeMetricRatio(stats.merge_exact_gain_calls,
                        std::max<uint64_t>(1, stats.merge_selected_pairs));
    stats.merge_positive_gain_ratio =
        SafeMetricRatio(stats.merge_positive_gain_pairs,
                        std::max<uint64_t>(1, stats.merge_exact_gain_calls));
    PrintMetric("merge_avg_local_gain", FormatDouble(avg_local_gain));
    PrintMetric("merge_group_count", std::to_string(stats.merge_group_count));
    PrintMetric("merge_group_max_size",
                std::to_string(stats.merge_group_max_size));
    PrintMetric("merge_raw_pair_count",
                std::to_string(stats.merge_raw_pair_count));
    PrintMetric("merge_candidate_pairs_after_prune",
                std::to_string(stats.merge_candidate_pairs_after_prune));
    PrintMetric("merge_exact_gain_calls",
                std::to_string(stats.merge_exact_gain_calls));
    PrintMetric("merge_positive_gain_pairs",
                std::to_string(stats.merge_positive_gain_pairs));
    PrintMetric("merge_rejected_by_overlap",
                std::to_string(stats.merge_rejected_by_overlap));
    PrintMetric("merge_rejected_by_threshold",
                std::to_string(stats.merge_rejected_by_threshold));
    PrintMetric("merge_exact_gain_calls_per_selected",
                FormatDouble(stats.merge_exact_gain_calls_per_selected));
    PrintMetric("merge_positive_gain_ratio",
                FormatDouble(stats.merge_positive_gain_ratio));
    PrintMetric("threshold_last", FormatDouble(stats.threshold_last));
    PrintMetric("threshold_geom_last", FormatDouble(stats.threshold_geom_last));
    PrintMetric("threshold_adaptive_last",
                FormatDouble(stats.threshold_adaptive_last));
    PrintMetric("threshold_acceptance_scale",
                FormatDouble(stats.threshold_acceptance_scale));
    PrintMetric("threshold_acceptance_rate_last",
                FormatDouble(stats.threshold_acceptance_rate_last));
    PrintMetric("threshold_sample_count_last",
                std::to_string(stats.threshold_sample_count_last));
    PrintMetric("cuda_init_ms", FormatDouble(stats.cuda_init_ms));
    PrintMetric("cuda_h2d_ms", FormatDouble(stats.cuda_h2d_ms));
    PrintMetric("cuda_row_h2d_ms", FormatDouble(stats.cuda_row_h2d_ms));
    PrintMetric("cuda_pair_h2d_ms", FormatDouble(stats.cuda_pair_h2d_ms));
    PrintMetric("cuda_kernel_ms", FormatDouble(stats.cuda_kernel_ms));
    PrintMetric("cuda_d2h_ms", FormatDouble(stats.cuda_d2h_ms));
    PrintMetric("cuda_total_ms", FormatDouble(stats.cuda_total_ms));
    PrintMetric("cuda_packed_h2d_ms",
                FormatDouble(stats.cuda_packed_h2d_ms));
    PrintMetric("cuda_packed_d2h_ms",
                FormatDouble(stats.cuda_packed_d2h_ms));
    PrintMetric("cuda_num_calls", std::to_string(stats.cuda_num_calls));
    PrintMetric("cuda_row_uploads", std::to_string(stats.cuda_row_uploads));
    PrintMetric("cuda_kernel_launches",
                std::to_string(stats.cuda_kernel_launches));
    PrintMetric("cuda_h2d_bytes", std::to_string(stats.cuda_h2d_bytes));
    PrintMetric("cuda_d2h_bytes", std::to_string(stats.cuda_d2h_bytes));
    PrintMetric("cuda_max_candidates_per_launch",
                std::to_string(stats.cuda_max_candidates_per_launch));
    PrintMetric("cuda_packed_h2d_calls",
                std::to_string(stats.cuda_packed_h2d_calls));
    PrintMetric("cuda_packed_d2h_calls",
                std::to_string(stats.cuda_packed_d2h_calls));
    PrintMetric("cuda_packed_input_bytes",
                std::to_string(stats.cuda_packed_input_bytes));
    PrintMetric("cuda_packed_output_bytes",
                std::to_string(stats.cuda_packed_output_bytes));
    PrintMetric("cuda_slice_count",
                std::to_string(stats.cuda_slice_count));
    PrintMetric("cuda_blocks_single_slice",
                std::to_string(stats.cuda_blocks_single_slice));
    PrintMetric("cuda_blocks_multi_slice",
                std::to_string(stats.cuda_blocks_multi_slice));
    PrintMetric("cuda_max_slice_rows",
                std::to_string(stats.cuda_max_slice_rows));
    PrintMetric("cuda_max_slice_nnz",
                std::to_string(stats.cuda_max_slice_nnz));
    PrintMetric("cuda_max_slice_candidates",
                std::to_string(stats.cuda_max_slice_candidates));
    PrintMetric("cuda_slice_memory_budget_bytes",
                std::to_string(stats.cuda_slice_memory_budget_bytes));
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
          "n",
          "input_edges_raw",
          "graph_arcs",
          "original_undirected_edges",
          "original_edges_eval",
          "iterations",
          "seed",
          "divide_hash_dims",
          "divide_max_group",
          "merge_mode",
          "scoring_backend",
          "threshold_policy",
          "top_k",
          "group_batch_size",
          "candidate_batch_budget",
          "cuda_slice_memory_mb",
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
          "runtime_run_ms",
          "runtime_divide_ms",
          "runtime_merge_ms",
          "divide_max_group_size",
          "divide_fallback_splits",
          "runtime_encode_ms",
          "runtime_output_ms",
          "runtime_algorithm_ms",
          "runtime_end_to_end_ms",
          "merge_prepare_wall_ms",
          "merge_prepare_task_sum_ms",
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
          "merge_group_count",
          "merge_group_max_size",
          "merge_raw_pair_count",
          "merge_candidate_pairs_after_prune",
          "merge_exact_gain_calls",
          "merge_positive_gain_pairs",
          "merge_rejected_by_overlap",
          "merge_rejected_by_threshold",
          "merge_exact_gain_calls_per_selected",
          "merge_positive_gain_ratio",
          "threshold_last",
          "threshold_geom_last",
          "threshold_adaptive_last",
          "threshold_acceptance_scale",
          "threshold_acceptance_rate_last",
          "threshold_sample_count_last",
          "cuda_init_ms",
          "cuda_h2d_ms",
          "cuda_row_h2d_ms",
          "cuda_pair_h2d_ms",
          "cuda_kernel_ms",
          "cuda_d2h_ms",
          "cuda_total_ms",
          "cuda_packed_h2d_ms",
          "cuda_packed_d2h_ms",
          "cuda_num_calls",
          "cuda_row_uploads",
          "cuda_kernel_launches",
          "cuda_h2d_bytes",
          "cuda_d2h_bytes",
          "cuda_max_candidates_per_launch",
          "cuda_packed_h2d_calls",
          "cuda_packed_d2h_calls",
          "cuda_packed_input_bytes",
          "cuda_packed_output_bytes",
          "cuda_slice_count",
          "cuda_blocks_single_slice",
          "cuda_blocks_multi_slice",
          "cuda_max_slice_rows",
          "cuda_max_slice_nnz",
          "cuda_max_slice_candidates",
          "cuda_slice_memory_budget_bytes",
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
          std::to_string(graph.n),
          std::to_string(input_edges_raw),
          std::to_string(graph_arcs),
          std::to_string(original_undirected_edges),
          std::to_string(original_edges_eval),
          std::to_string(opts.iterations),
          std::to_string(opts.seed),
          std::to_string(opts.divide_hash_dims),
          std::to_string(opts.divide_max_group),
          merge_mode,
          scoring_backend,
          threshold_policy,
          std::to_string(opts.top_k),
          std::to_string(opts.group_batch_size),
          std::to_string(opts.candidate_batch_budget),
          std::to_string(opts.cuda_slice_memory_mb),
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
          FormatDouble(stats.runtime_run_ms),
          FormatDouble(stats.runtime_divide_ms),
          FormatDouble(stats.runtime_merge_ms),
          std::to_string(stats.divide_max_group_size),
          std::to_string(stats.divide_fallback_splits),
          FormatDouble(stats.runtime_encode_ms),
          FormatDouble(stats.runtime_output_ms),
          FormatDouble(runtime_algorithm_ms),
          FormatDouble(runtime_end_to_end_ms),
          FormatDouble(stats.merge_prepare_wall_ms),
          FormatDouble(stats.merge_prepare_task_sum_ms),
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
          std::to_string(stats.merge_group_count),
          std::to_string(stats.merge_group_max_size),
          std::to_string(stats.merge_raw_pair_count),
          std::to_string(stats.merge_candidate_pairs_after_prune),
          std::to_string(stats.merge_exact_gain_calls),
          std::to_string(stats.merge_positive_gain_pairs),
          std::to_string(stats.merge_rejected_by_overlap),
          std::to_string(stats.merge_rejected_by_threshold),
          FormatDouble(stats.merge_exact_gain_calls_per_selected),
          FormatDouble(stats.merge_positive_gain_ratio),
          FormatDouble(stats.threshold_last),
          FormatDouble(stats.threshold_geom_last),
          FormatDouble(stats.threshold_adaptive_last),
          FormatDouble(stats.threshold_acceptance_scale),
          FormatDouble(stats.threshold_acceptance_rate_last),
          std::to_string(stats.threshold_sample_count_last),
          FormatDouble(stats.cuda_init_ms),
          FormatDouble(stats.cuda_h2d_ms),
          FormatDouble(stats.cuda_row_h2d_ms),
          FormatDouble(stats.cuda_pair_h2d_ms),
          FormatDouble(stats.cuda_kernel_ms),
          FormatDouble(stats.cuda_d2h_ms),
          FormatDouble(stats.cuda_total_ms),
          FormatDouble(stats.cuda_packed_h2d_ms),
          FormatDouble(stats.cuda_packed_d2h_ms),
          std::to_string(stats.cuda_num_calls),
          std::to_string(stats.cuda_row_uploads),
          std::to_string(stats.cuda_kernel_launches),
          std::to_string(stats.cuda_h2d_bytes),
          std::to_string(stats.cuda_d2h_bytes),
          std::to_string(stats.cuda_max_candidates_per_launch),
          std::to_string(stats.cuda_packed_h2d_calls),
          std::to_string(stats.cuda_packed_d2h_calls),
          std::to_string(stats.cuda_packed_input_bytes),
          std::to_string(stats.cuda_packed_output_bytes),
          std::to_string(stats.cuda_slice_count),
          std::to_string(stats.cuda_blocks_single_slice),
          std::to_string(stats.cuda_blocks_multi_slice),
          std::to_string(stats.cuda_max_slice_rows),
          std::to_string(stats.cuda_max_slice_nnz),
          std::to_string(stats.cuda_max_slice_candidates),
          std::to_string(stats.cuda_slice_memory_budget_bytes),
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
