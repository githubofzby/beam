#include "graph_io/graph_io.hpp"
#include "compression_metrics.hpp"
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
  ProfilingMode profiling_mode = ProfilingMode::kOff;
  CostObjective cost_objective = CostObjective::kLegacy;
  StateBackend state_backend = StateBackend::kLegacy;
  bool validate_quotient = false;
  QuotientUpdateMode quotient_update_mode = QuotientUpdateMode::kIncremental;
  CandidateIndexMode candidate_index_mode = CandidateIndexMode::kLegacy;
  int candidate_budget = 8;
  CertificationMode certification_mode = CertificationMode::kOff;
  CommitPolicy commit_policy = CommitPolicy::kGreedyFull;
  bool commit_audit = false;
  std::string commit_audit_csv;
  std::string profile_csv;
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
      << "  --cost-objective <legacy|mags-compatible> Exact merge and encoding objective (default: legacy)\n"
      << "  --state-backend <legacy|persistent> Supernode adjacency state (default: legacy)\n"
      << "  --validate-quotient     Rebuild quotient state after every merge (debug only)\n"
      << "  --quotient-update <incremental|bulk_rebuild|auto> Persistent update strategy (default: incremental)\n"
      << "  --candidate-index <legacy|quotient-neighbor|residual-signature> Candidate proposal backend (default: legacy)\n"
      << "  --candidate-budget <int> Per-source proposal budget (default: 8)\n"
      << "  --certification <off|safe> Safe persistent CPU exact certification (default: off)\n"
      << "  --commit-policy <g0|s1|t2|t4|t8|m4> Fixed-candidate commit policy (default: g0)\n"
      << "  --commit-audit      Verify and record exact-cost commit trajectory\n"
      << "  --commit-audit-csv <path> Write long-form commit trajectory CSV\n"
      << "  --profiling <off|summary|rounds> Collect Stage 0 runtime profile (default: off)\n"
      << "  --profile-csv <path>    Write long-form per-iteration profile CSV\n"
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

static ProfilingMode ParseProfilingMode(const std::string& value) {
  if (value == "off") {
    return ProfilingMode::kOff;
  }
  if (value == "summary") {
    return ProfilingMode::kSummary;
  }
  if (value == "rounds") {
    return ProfilingMode::kRounds;
  }
  throw std::runtime_error("Unsupported profiling mode: " + value);
}

static CertificationMode ParseCertificationMode(const std::string& value) {
  if (value == "off") {
    return CertificationMode::kOff;
  }
  if (value == "safe") {
    return CertificationMode::kSafe;
  }
  throw std::runtime_error("Unsupported certification mode: " + value);
}

static CommitPolicy ParseCommitPolicy(const std::string& value) {
  if (value == "g0") return CommitPolicy::kGreedyFull;
  if (value == "s1") return CommitPolicy::kSequentialOne;
  if (value == "t2") return CommitPolicy::kTransactional2;
  if (value == "t4") return CommitPolicy::kTransactional4;
  if (value == "t8") return CommitPolicy::kTransactional8;
  if (value == "m4") return CommitPolicy::kMutualBest4;
  throw std::runtime_error("Unsupported commit policy: " + value);
}

static const char* CertificationModeToString(CertificationMode mode) {
  return mode == CertificationMode::kSafe ? "safe" : "off";
}

static CostObjective ParseCostObjective(const std::string& value) {
  if (value == "legacy") {
    return CostObjective::kLegacy;
  }
  if (value == "mags-compatible") {
    return CostObjective::kMagsCompatible;
  }
  throw std::runtime_error("Unsupported cost objective: " + value);
}

static const char* CostObjectiveToString(CostObjective objective) {
  return objective == CostObjective::kMagsCompatible ? "mags-compatible"
                                                     : "legacy";
}

static StateBackend ParseStateBackend(const std::string& value) {
  if (value == "legacy") {
    return StateBackend::kLegacy;
  }
  if (value == "persistent") {
    return StateBackend::kPersistent;
  }
  throw std::runtime_error("Unsupported state backend: " + value);
}

static const char* StateBackendToString(StateBackend backend) {
  return backend == StateBackend::kPersistent ? "persistent" : "legacy";
}

static QuotientUpdateMode ParseQuotientUpdateMode(const std::string& value) {
  if (value == "incremental") return QuotientUpdateMode::kIncremental;
  if (value == "bulk_rebuild") return QuotientUpdateMode::kBulkRebuild;
  if (value == "auto") return QuotientUpdateMode::kAuto;
  throw std::runtime_error("Unsupported quotient update mode: " + value);
}

static const char* QuotientUpdateModeToString(QuotientUpdateMode mode) {
  switch (mode) {
    case QuotientUpdateMode::kIncremental: return "incremental";
    case QuotientUpdateMode::kBulkRebuild: return "bulk_rebuild";
    case QuotientUpdateMode::kAuto: return "auto";
  }
  return "unknown";
}

static CandidateIndexMode ParseCandidateIndexMode(const std::string& value) {
  if (value == "legacy") return CandidateIndexMode::kLegacy;
  if (value == "quotient-neighbor") {
    return CandidateIndexMode::kQuotientNeighbor;
  }
  if (value == "residual-signature") {
    return CandidateIndexMode::kResidualSignature;
  }
  throw std::runtime_error("Unsupported candidate index: " + value);
}

static const char* ProfilingModeToString(ProfilingMode mode) {
  switch (mode) {
    case ProfilingMode::kOff:
      return "off";
    case ProfilingMode::kSummary:
      return "summary";
    case ProfilingMode::kRounds:
      return "rounds";
  }
  return "unknown";
}

static std::string DatasetLabelFromPath(const std::string& input) {
  std::filesystem::path path(input);
  std::string label = path.parent_path().filename().string();
  return label.empty() ? path.stem().string() : label;
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
    } else if (arg == "--cost-objective" && i + 1 < argc) {
      opts.cost_objective = ParseCostObjective(argv[++i]);
    } else if (arg == "--state-backend" && i + 1 < argc) {
      opts.state_backend = ParseStateBackend(argv[++i]);
    } else if (arg == "--validate-quotient") {
      opts.validate_quotient = true;
    } else if (arg == "--quotient-update" && i + 1 < argc) {
      opts.quotient_update_mode = ParseQuotientUpdateMode(argv[++i]);
    } else if (arg == "--candidate-index" && i + 1 < argc) {
      opts.candidate_index_mode = ParseCandidateIndexMode(argv[++i]);
    } else if (arg == "--candidate-budget" && i + 1 < argc) {
      opts.candidate_budget = std::stoi(argv[++i]);
    } else if (arg == "--certification" && i + 1 < argc) {
      opts.certification_mode = ParseCertificationMode(argv[++i]);
    } else if (arg == "--commit-policy" && i + 1 < argc) {
      opts.commit_policy = ParseCommitPolicy(argv[++i]);
    } else if (arg == "--commit-audit") {
      opts.commit_audit = true;
    } else if (arg == "--commit-audit-csv" && i + 1 < argc) {
      opts.commit_audit_csv = argv[++i];
      opts.commit_audit = true;
    } else if (arg == "--profiling" && i + 1 < argc) {
      opts.profiling_mode = ParseProfilingMode(argv[++i]);
    } else if (arg == "--profile-csv" && i + 1 < argc) {
      opts.profile_csv = argv[++i];
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
    if (opts.cost_objective == CostObjective::kMagsCompatible &&
        opts.scoring_backend == ScoringBackend::kCuda) {
      throw std::runtime_error(
          "--cost-objective mags-compatible currently requires --scoring-backend cpu");
    }
    if (opts.validate_quotient && opts.state_backend != StateBackend::kPersistent) {
      throw std::runtime_error(
          "--validate-quotient requires --state-backend persistent");
    }
    if (opts.quotient_update_mode != QuotientUpdateMode::kIncremental &&
        opts.state_backend != StateBackend::kPersistent) {
      throw std::runtime_error(
          "non-incremental --quotient-update requires --state-backend persistent");
    }
    if (opts.candidate_budget <= 0) {
      throw std::runtime_error("--candidate-budget must be positive");
    }
    if (opts.certification_mode == CertificationMode::kSafe &&
        (opts.state_backend != StateBackend::kPersistent ||
         opts.scoring_backend != ScoringBackend::kCpu ||
         opts.cost_objective != CostObjective::kMagsCompatible ||
         opts.candidate_index_mode != CandidateIndexMode::kLegacy)) {
      throw std::runtime_error(
          "--certification safe requires persistent state, CPU scoring, "
          "MAGS-compatible cost, and the legacy candidate index");
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
    std::cout << "Cost objective: "
              << CostObjectiveToString(opts.cost_objective) << "\n";
    std::cout << "State backend: " << StateBackendToString(opts.state_backend)
              << "\n";
    std::cout << "Quotient update: "
              << QuotientUpdateModeToString(opts.quotient_update_mode) << "\n";
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
              opts.divide_hash_dims, opts.divide_max_group, threshold_config,
              opts.profiling_mode, opts.cost_objective, opts.state_backend,
              opts.validate_quotient, opts.quotient_update_mode,
              opts.candidate_index_mode, opts.candidate_budget,
              opts.certification_mode, opts.commit_policy, opts.commit_audit);
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

    const CompressionMetrics compression_metrics = ComputeCompressionMetrics(
        result, input_edges_raw, opts.cost_objective);
    const EncodingCost exact_partition_cost = sweg.ExactCurrentPartitionCost();
    if (opts.cost_objective == CostObjective::kMagsCompatible &&
        opts.error_bound == 0.0 &&
        compression_metrics.encoding_cost_standard != exact_partition_cost) {
      throw std::runtime_error(
          "MAGS-compatible payload cost differs from exact partition cost");
    }
    const double compression_gain =
        1.0 - compression_metrics.cost_ratio_standard;
    const double compression_java_style = compression_gain;
    const std::string merge_mode = MergeModeToString(opts.merge_mode);
    const std::string scoring_backend =
        ScoringBackendToString(opts.scoring_backend);
    const std::string threshold_policy =
        ThresholdPolicyToString(opts.threshold_policy);
    const std::string profiling_mode = ProfilingModeToString(opts.profiling_mode);

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
    PrintMetric("profiling_mode", profiling_mode);
    PrintMetric("cost_objective", CostObjectiveToString(opts.cost_objective));
    PrintMetric("state_backend", StateBackendToString(opts.state_backend));
    PrintMetric("certification",
                CertificationModeToString(opts.certification_mode));
    PrintMetric("commit_policy", CommitPolicyToString(opts.commit_policy));
    PrintMetric("partition_hash", std::to_string(sweg.PartitionHash()));
    PrintMetric("validate_quotient", opts.validate_quotient ? "true" : "false");
    PrintMetric("quotient_update",
                QuotientUpdateModeToString(opts.quotient_update_mode));
    PrintMetric("exact_partition_cost", std::to_string(exact_partition_cost));
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
    // Legacy names retain the original BEAM standard metric.
    PrintMetric("encoding_cost",
                std::to_string(compression_metrics.encoding_cost_standard));
    PrintMetric("cost_ratio", FormatDouble(compression_metrics.cost_ratio_standard));
    PrintMetric("num_superedges_nonloop",
                std::to_string(compression_metrics.num_superedges_nonloop));
    PrintMetric("num_superedges_loop",
                std::to_string(compression_metrics.num_superedges_loop));
    PrintMetric("num_positive_corrections",
                std::to_string(compression_metrics.num_positive_corrections));
    PrintMetric("num_negative_corrections",
                std::to_string(compression_metrics.num_negative_corrections));
    PrintMetric("encoding_cost_standard",
                std::to_string(compression_metrics.encoding_cost_standard));
    PrintMetric("cost_ratio_standard",
                FormatDouble(compression_metrics.cost_ratio_standard));
    PrintMetric("encoding_cost_mags_x2",
                std::to_string(compression_metrics.encoding_cost_mags_x2));
    PrintMetric("encoding_cost_mags_compatible",
                FormatDouble(compression_metrics.encoding_cost_mags_compatible));
    PrintMetric("cost_ratio_mags_compatible",
                FormatDouble(compression_metrics.cost_ratio_mags_compatible));
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
    PrintMetric("certification_candidates_seen",
                std::to_string(stats.certification_candidates_seen));
    PrintMetric("upper_bound_pruned",
                std::to_string(stats.upper_bound_pruned));
    PrintMetric("upper_bound_passed",
                std::to_string(stats.upper_bound_passed));
    PrintMetric("upper_bound_prune_rate",
                FormatDouble(SafeMetricRatio(
                    stats.upper_bound_pruned,
                    std::max<uint64_t>(1, stats.certification_candidates_seen))));
    PrintMetric("early_abort_count", std::to_string(stats.early_abort_count));
    PrintMetric("early_abort_rate",
                FormatDouble(SafeMetricRatio(
                    stats.early_abort_count,
                    std::max<uint64_t>(1, stats.upper_bound_passed))));
    PrintMetric("exact_full_scan_count",
                std::to_string(stats.exact_full_scan_count));
    PrintMetric("exact_entries_available",
                std::to_string(stats.exact_entries_available));
    PrintMetric("exact_entries_scanned",
                std::to_string(stats.exact_entries_scanned));
    PrintMetric("exact_entries_skipped",
                std::to_string(stats.exact_entries_skipped));
    PrintMetric("upper_bound_ms", FormatDouble(stats.upper_bound_ms));
    PrintMetric("early_abort_exact_ms",
                FormatDouble(stats.early_abort_exact_ms));
    stats.original_exact_calls = stats.merge_exact_gain_calls;
    PrintMetric("isolated_gain_sum", std::to_string(stats.isolated_gain_sum));
    PrintMetric("realized_marginal_gain_sum",
                std::to_string(stats.realized_marginal_gain_sum));
    PrintMetric("actual_batch_cost_reduction",
                std::to_string(stats.actual_batch_cost_reduction));
    PrintMetric("interaction_delta", std::to_string(stats.interaction_delta));
    PrintMetric("gain_decay_ratio",
                FormatDouble(static_cast<double>(stats.isolated_gain_sum -
                                                 stats.realized_marginal_gain_sum) /
                             std::max<int64_t>(1, stats.isolated_gain_sum)));
    PrintMetric("selected_merges_for_validation",
                std::to_string(stats.selected_merges_for_validation));
    PrintMetric("accepted_merges_after_validation",
                std::to_string(stats.accepted_merges_after_validation));
    PrintMetric("rejected_nonpositive",
                std::to_string(stats.rejected_nonpositive));
    PrintMetric("stale_endpoint", std::to_string(stats.stale_endpoint));
    PrintMetric("gain_decreased", std::to_string(stats.gain_decreased));
    PrintMetric("gain_increased", std::to_string(stats.gain_increased));
    PrintMetric("negative_marginal", std::to_string(stats.negative_marginal));
    PrintMetric("candidate_refresh_count",
                std::to_string(stats.candidate_refresh_count));
    PrintMetric("original_exact_calls",
                std::to_string(stats.original_exact_calls));
    PrintMetric("validation_exact_calls",
                std::to_string(stats.validation_exact_calls));
    PrintMetric("total_exact_calls",
                std::to_string(stats.original_exact_calls +
                               stats.validation_exact_calls));
    PrintMetric("validation_exact_row_entry_work",
                std::to_string(stats.validation_exact_row_entry_work));
    PrintMetric("commit_validation_ms",
                FormatDouble(stats.commit_validation_ms));
    PrintMetric("audit_oracle_ms", FormatDouble(stats.audit_oracle_ms));
    PrintMetric("update_touched_quotient_entries",
                std::to_string(stats.update_touched_quotient_entries));
    PrintMetric("quotient_nnz_final", std::to_string(stats.quotient_nnz_final));
    PrintMetric("quotient_incremental_batch_count",
                std::to_string(stats.quotient_incremental_batch_count));
    PrintMetric("quotient_bulk_rebuild_count",
                std::to_string(stats.quotient_bulk_rebuild_count));
    PrintMetric("prepare_row_entries_copied",
                std::to_string(stats.prepare_row_entries_copied));
    PrintMetric("prepare_row_copy_bytes",
                std::to_string(stats.prepare_row_copy_bytes));
    PrintMetric("prepare_row_views_created",
                std::to_string(stats.prepare_row_views_created));
    PrintMetric("prepare_unique_representatives",
                std::to_string(stats.prepare_unique_representatives));
    PrintMetric("prepare_row_acquisition_requests",
                std::to_string(stats.prepare_row_acquisition_requests));
    PrintMetric("prepare_row_acquisition_hits",
                std::to_string(stats.prepare_row_acquisition_hits));
    PrintMetric("prepare_row_acquisition_misses",
                std::to_string(stats.prepare_row_acquisition_misses));
    PrintMetric("prepare_duplicate_acquisitions_avoided",
                std::to_string(stats.prepare_duplicate_acquisitions_avoided));
    PrintMetric("prepare_row_registry_entries",
                std::to_string(stats.prepare_row_registry_entries));
    PrintMetric("prepare_row_registry_bytes",
                std::to_string(stats.prepare_row_registry_bytes));
    PrintMetric("exact_persistent_pairs",
                std::to_string(stats.exact_persistent_pairs));
    PrintMetric("exact_raw_entries_a",
                std::to_string(stats.exact_raw_entries_a));
    PrintMetric("exact_raw_entries_b",
                std::to_string(stats.exact_raw_entries_b));
    PrintMetric("exact_union_neighbors",
                std::to_string(stats.exact_union_neighbors));
    PrintMetric("exact_overlap_neighbors",
                std::to_string(stats.exact_overlap_neighbors));
    PrintMetric("exact_single_sided_neighbors",
                std::to_string(stats.exact_single_sided_neighbors));
    PrintMetric("exact_internal_block_terms",
                std::to_string(stats.exact_internal_block_terms));
    PrintMetric("exact_block_cost_evaluations",
                std::to_string(stats.exact_block_cost_evaluations));
    PrintMetric("exact_capacity_multiplications",
                std::to_string(stats.exact_capacity_multiplications));
    PrintMetric("candidate_index_build_ms",
                FormatDouble(stats.candidate_index_build_ms));
    PrintMetric("candidate_index_refresh_ms",
                FormatDouble(stats.candidate_index_refresh_ms));
    PrintMetric("candidate_proposal_ms",
                FormatDouble(stats.candidate_proposal_ms));
    PrintMetric("candidate_proposals_raw",
                std::to_string(stats.candidate_proposals_raw));
    PrintMetric("candidate_proposals_unique",
                std::to_string(stats.candidate_proposals_unique));
    PrintMetric("candidate_duplicates_removed",
                std::to_string(stats.candidate_duplicates_removed));
    PrintMetric("candidate_budget_exhausted_nodes",
                std::to_string(stats.candidate_budget_exhausted_nodes));
    PrintMetric("candidate_nodes_with_zero_proposals",
                std::to_string(stats.candidate_nodes_with_zero_proposals));
    PrintMetric("candidate_direct_neighbor_count",
                std::to_string(stats.candidate_direct_neighbor_count));
    PrintMetric("candidate_shared_neighbor_count",
                std::to_string(stats.candidate_shared_neighbor_count));
    PrintMetric("candidate_exploration_count",
                std::to_string(stats.candidate_exploration_count));
    PrintMetric("residual_signature_build_ms",
                FormatDouble(stats.residual_signature_build_ms));
    PrintMetric("residual_signature_refresh_ms",
                FormatDouble(stats.residual_signature_refresh_ms));
    PrintMetric("residual_signature_rows_scanned",
                std::to_string(stats.residual_signature_rows_scanned));
    PrintMetric("residual_signature_features_created",
                std::to_string(stats.residual_signature_features_created));
    PrintMetric("residual_signature_cache_hits",
                std::to_string(stats.residual_signature_cache_hits));
    PrintMetric("residual_signature_cache_misses",
                std::to_string(stats.residual_signature_cache_misses));
    PrintMetric("residual_bucket_count",
                std::to_string(stats.residual_bucket_count));
    PrintMetric("residual_bucket_max_size",
                std::to_string(stats.residual_bucket_max_size));
    PrintMetric("residual_bucket_candidates_considered",
                std::to_string(stats.residual_bucket_candidates_considered));
    PrintMetric("residual_bucket_candidates_dropped_by_cap",
                std::to_string(stats.residual_bucket_candidates_dropped_by_cap));
    PrintMetric("residual_alignment_score_sum",
                std::to_string(stats.residual_alignment_score_sum));
    PrintMetric("residual_conflict_penalty_sum",
                std::to_string(stats.residual_conflict_penalty_sum));
    PrintMetric("residual_direct_score_sum",
                std::to_string(stats.residual_direct_score_sum));
    PrintMetric("residual_size_compatibility_sum",
                std::to_string(stats.residual_size_compatibility_sum));
    PrintMetric("quotient_row_lookup_ms",
                FormatDouble(stats.quotient_row_lookup_ms));
    PrintMetric("quotient_row_copy_ms",
                FormatDouble(stats.quotient_row_copy_ms));
    PrintMetric("quotient_exact_gain_ms",
                FormatDouble(stats.quotient_exact_gain_ms));
    PrintMetric("quotient_incremental_update_ms",
                FormatDouble(stats.quotient_incremental_update_ms));
    PrintMetric("quotient_reciprocal_update_ms",
                FormatDouble(stats.quotient_reciprocal_update_ms));
    PrintMetric("quotient_memory_allocation_ms",
                FormatDouble(stats.quotient_memory_allocation_ms));
    PrintMetric("quotient_sort_or_merge_ms",
                FormatDouble(stats.quotient_sort_or_merge_ms));
    PrintMetric("quotient_rebuild_ms", FormatDouble(stats.quotient_rebuild_ms));
    PrintMetric("quotient_rows_updated",
                std::to_string(stats.quotient_rows_updated));
    PrintMetric("quotient_entries_inserted",
                std::to_string(stats.quotient_entries_inserted));
    PrintMetric("quotient_entries_removed",
                std::to_string(stats.quotient_entries_removed));
    PrintMetric("quotient_entries_shifted",
                std::to_string(stats.quotient_entries_shifted));
    PrintMetric("quotient_high_degree_rows_touched",
                std::to_string(stats.quotient_high_degree_rows_touched));
    PrintMetric("quotient_max_row_degree",
                std::to_string(stats.quotient_max_row_degree));
    PrintMetric("quotient_allocated_bytes",
                std::to_string(stats.quotient_allocated_bytes));
    PrintMetric("quotient_peak_nnz",
                std::to_string(stats.quotient_peak_nnz));
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

    const RuntimeProfile& runtime_profile = sweg.runtime_profile();
    if (opts.profiling_mode != ProfilingMode::kOff) {
      PrintMetric("profile_iterations_observed",
                  std::to_string(runtime_profile.iterations_observed));
      PrintMetric("profile_active_supernodes_last",
                  std::to_string(runtime_profile.active_supernodes_last));
      PrintMetric("profile_group_count_last",
                  std::to_string(runtime_profile.group_count_last));
      PrintMetric("profile_sum_group_sizes_last",
                  std::to_string(runtime_profile.sum_group_sizes_last));
      const IterationProfile& total_profile = runtime_profile.total;
      PrintMetric("candidate_proxy_pairs_examined",
                  std::to_string(total_profile.candidate_proxy_pairs_examined));
      PrintMetric("candidate_pairs_after_topk",
                  std::to_string(total_profile.candidate_pairs_after_topk));
      PrintMetric("candidate_pairs_submitted_for_scoring",
                  std::to_string(total_profile.candidate_pairs_submitted_for_scoring));
      PrintMetric("candidate_pairs_scored",
                  std::to_string(total_profile.candidate_pairs_scored));
      PrintMetric("exact_gain_calls_profiled",
                  std::to_string(total_profile.exact_gain_calls));
      PrintMetric("exact_gain_positive_count",
                  std::to_string(total_profile.exact_gain_positive_count));
      PrintMetric("above_threshold_count",
                  std::to_string(total_profile.above_threshold_count));
      PrintMetric("matching_selected_count",
                  std::to_string(total_profile.matching_selected_count));
      PrintMetric("actual_merge_count",
                  std::to_string(total_profile.actual_merge_count));
      PrintMetric("prepare_original_edges_scanned",
                  std::to_string(total_profile.prepare_original_edges_scanned));
      PrintMetric("prepare_aggregated_nnz",
                  std::to_string(total_profile.prepare_aggregated_nnz));
      PrintMetric("exact_gain_input_nnz",
                  std::to_string(total_profile.exact_gain_input_nnz));
      PrintMetric("update_partition_nodes_touched",
                  std::to_string(total_profile.update_partition_nodes_touched));
      PrintMetric("prepare_row_entries_copied_profiled",
                  std::to_string(total_profile.prepare_row_entries_copied));
      PrintMetric("prepare_row_copy_bytes_profiled",
                  std::to_string(total_profile.prepare_row_copy_bytes));
      PrintMetric("prepare_row_views_created_profiled",
                  std::to_string(total_profile.prepare_row_views_created));
      PrintMetric("prepare_unique_representatives_profiled",
                  std::to_string(total_profile.prepare_unique_representatives));
      PrintMetric("prepare_row_acquisition_requests_profiled",
                  std::to_string(
                      total_profile.prepare_row_acquisition_requests));
      PrintMetric("prepare_row_acquisition_hits_profiled",
                  std::to_string(total_profile.prepare_row_acquisition_hits));
      PrintMetric("prepare_row_acquisition_misses_profiled",
                  std::to_string(total_profile.prepare_row_acquisition_misses));
      PrintMetric("prepare_duplicate_acquisitions_avoided_profiled",
                  std::to_string(
                      total_profile.prepare_duplicate_acquisitions_avoided));
      PrintMetric("prepare_row_registry_entries_profiled",
                  std::to_string(total_profile.prepare_row_registry_entries));
      PrintMetric("prepare_row_registry_bytes_profiled",
                  std::to_string(total_profile.prepare_row_registry_bytes));
      PrintMetric("exact_persistent_pairs_profiled",
                  std::to_string(total_profile.exact_persistent_pairs));
      PrintMetric("exact_raw_entries_a_profiled",
                  std::to_string(total_profile.exact_raw_entries_a));
      PrintMetric("exact_raw_entries_b_profiled",
                  std::to_string(total_profile.exact_raw_entries_b));
      PrintMetric("exact_union_neighbors_profiled",
                  std::to_string(total_profile.exact_union_neighbors));
      PrintMetric("exact_overlap_neighbors_profiled",
                  std::to_string(total_profile.exact_overlap_neighbors));
      PrintMetric("exact_single_sided_neighbors_profiled",
                  std::to_string(total_profile.exact_single_sided_neighbors));
      PrintMetric("exact_internal_block_terms_profiled",
                  std::to_string(total_profile.exact_internal_block_terms));
      PrintMetric("exact_block_cost_evaluations_profiled",
                  std::to_string(total_profile.exact_block_cost_evaluations));
      PrintMetric("exact_capacity_multiplications_profiled",
                  std::to_string(total_profile.exact_capacity_multiplications));
      PrintMetric("candidate_index_build_ms_profiled",
                  FormatDouble(total_profile.candidate_index_build_ms));
      PrintMetric("candidate_index_refresh_ms_profiled",
                  FormatDouble(total_profile.candidate_index_refresh_ms));
      PrintMetric("candidate_proposal_ms_profiled",
                  FormatDouble(total_profile.candidate_proposal_ms));
      PrintMetric("candidate_proposals_raw_profiled",
                  std::to_string(total_profile.candidate_proposals_raw));
      PrintMetric("candidate_proposals_unique_profiled",
                  std::to_string(total_profile.candidate_proposals_unique));
      PrintMetric("candidate_duplicates_removed_profiled",
                  std::to_string(total_profile.candidate_duplicates_removed));
      PrintMetric("profiling_divide_ms",
                  FormatDouble(runtime_profile.profiling_divide_ms));
      PrintMetric("profiling_prepare_ms",
                  FormatDouble(runtime_profile.profiling_prepare_ms));
      PrintMetric("profiling_candidate_discovery_task_sum_ms",
                  FormatDouble(
                      runtime_profile.profiling_candidate_discovery_task_sum_ms));
      PrintMetric("profiling_exact_gain_ms",
                  FormatDouble(runtime_profile.profiling_exact_gain_ms));
      PrintMetric("profiling_matching_ms",
                  FormatDouble(runtime_profile.profiling_matching_ms));
      PrintMetric("profiling_update_ms",
                  FormatDouble(runtime_profile.profiling_update_ms));
    }
    if (!opts.profile_csv.empty()) {
      if (opts.profiling_mode != ProfilingMode::kRounds) {
        throw std::runtime_error(
            "--profile-csv requires --profiling rounds so every row has an iteration.");
      }
      const std::vector<std::string> profile_columns = {
          "dataset", "iteration", "algorithm", "profiling_mode",
          "active_supernodes", "group_count", "sum_group_sizes",
          "candidate_proxy_pairs_examined", "candidate_pairs_after_topk",
          "candidate_pairs_submitted_for_scoring", "candidate_pairs_scored",
          "exact_gain_calls", "exact_gain_positive_count",
          "above_threshold_count", "matching_selected_count",
          "actual_merge_count", "prepare_original_edges_scanned",
          "prepare_aggregated_nnz", "exact_gain_input_nnz",
          "update_partition_nodes_touched", "update_touched_quotient_entries",
          "prepare_row_entries_copied", "prepare_row_copy_bytes",
          "prepare_row_views_created", "prepare_unique_representatives",
          "prepare_row_acquisition_requests", "prepare_row_acquisition_hits",
          "prepare_row_acquisition_misses",
          "prepare_duplicate_acquisitions_avoided",
          "prepare_row_registry_entries", "prepare_row_registry_bytes",
          "exact_persistent_pairs", "exact_raw_entries_a",
          "exact_raw_entries_b", "exact_union_neighbors",
          "exact_overlap_neighbors", "exact_single_sided_neighbors",
          "exact_internal_block_terms", "exact_block_cost_evaluations",
          "exact_capacity_multiplications",
          "certification_candidates_seen", "upper_bound_pruned",
          "upper_bound_passed", "early_abort_count",
          "exact_full_scan_count", "exact_entries_available",
          "exact_entries_scanned", "exact_entries_skipped",
          "upper_bound_ms", "early_abort_exact_ms",
          "candidate_index_build_ms", "candidate_index_refresh_ms",
          "candidate_proposal_ms", "candidate_proposals_raw",
          "candidate_proposals_unique", "candidate_duplicates_removed",
          "candidate_budget_exhausted_nodes",
          "candidate_nodes_with_zero_proposals",
          "candidate_direct_neighbor_count", "candidate_shared_neighbor_count",
          "candidate_exploration_count",
          "quotient_row_lookup_ms",
          "quotient_row_copy_ms", "quotient_exact_gain_ms",
          "divide_ms", "prepare_ms",
          "candidate_discovery_task_sum_ms", "exact_gain_ms", "matching_ms",
          "update_ms"};
      const std::string dataset = DatasetLabelFromPath(opts.input);
      for (const IterationProfile& row : runtime_profile.rounds) {
        AppendResultsCsv(opts.profile_csv, profile_columns,
                         {dataset, std::to_string(row.iteration),
                          opts.cost_objective == CostObjective::kLegacy
                              ? "legacy_beam"
                              : "beam_cost_oracle_reference",
                          profiling_mode,
                          std::to_string(row.active_supernodes),
                          std::to_string(row.group_count),
                          std::to_string(row.sum_group_sizes),
                          std::to_string(row.candidate_proxy_pairs_examined),
                          std::to_string(row.candidate_pairs_after_topk),
                          std::to_string(row.candidate_pairs_submitted_for_scoring),
                          std::to_string(row.candidate_pairs_scored),
                          std::to_string(row.exact_gain_calls),
                          std::to_string(row.exact_gain_positive_count),
                          std::to_string(row.above_threshold_count),
                          std::to_string(row.matching_selected_count),
                          std::to_string(row.actual_merge_count),
                          std::to_string(row.prepare_original_edges_scanned),
                          std::to_string(row.prepare_aggregated_nnz),
                          std::to_string(row.exact_gain_input_nnz),
                          std::to_string(row.update_partition_nodes_touched),
                          std::to_string(row.update_touched_quotient_entries),
                          std::to_string(row.prepare_row_entries_copied),
                          std::to_string(row.prepare_row_copy_bytes),
                          std::to_string(row.prepare_row_views_created),
                          std::to_string(row.prepare_unique_representatives),
                          std::to_string(row.prepare_row_acquisition_requests),
                          std::to_string(row.prepare_row_acquisition_hits),
                          std::to_string(row.prepare_row_acquisition_misses),
                          std::to_string(
                              row.prepare_duplicate_acquisitions_avoided),
                          std::to_string(row.prepare_row_registry_entries),
                          std::to_string(row.prepare_row_registry_bytes),
                          std::to_string(row.exact_persistent_pairs),
                          std::to_string(row.exact_raw_entries_a),
                          std::to_string(row.exact_raw_entries_b),
                          std::to_string(row.exact_union_neighbors),
                          std::to_string(row.exact_overlap_neighbors),
                          std::to_string(row.exact_single_sided_neighbors),
                          std::to_string(row.exact_internal_block_terms),
                          std::to_string(row.exact_block_cost_evaluations),
                          std::to_string(row.exact_capacity_multiplications),
                          std::to_string(row.certification_candidates_seen),
                          std::to_string(row.upper_bound_pruned),
                          std::to_string(row.upper_bound_passed),
                          std::to_string(row.early_abort_count),
                          std::to_string(row.exact_full_scan_count),
                          std::to_string(row.exact_entries_available),
                          std::to_string(row.exact_entries_scanned),
                          std::to_string(row.exact_entries_skipped),
                          FormatDouble(row.upper_bound_ms),
                          FormatDouble(row.early_abort_exact_ms),
                          FormatDouble(row.candidate_index_build_ms),
                          FormatDouble(row.candidate_index_refresh_ms),
                          FormatDouble(row.candidate_proposal_ms),
                          std::to_string(row.candidate_proposals_raw),
                          std::to_string(row.candidate_proposals_unique),
                          std::to_string(row.candidate_duplicates_removed),
                          std::to_string(row.candidate_budget_exhausted_nodes),
                          std::to_string(row.candidate_nodes_with_zero_proposals),
                          std::to_string(row.candidate_direct_neighbor_count),
                          std::to_string(row.candidate_shared_neighbor_count),
                          std::to_string(row.candidate_exploration_count),
                          FormatDouble(row.quotient_row_lookup_ms),
                          FormatDouble(row.quotient_row_copy_ms),
                          FormatDouble(row.quotient_exact_gain_ms),
                          FormatDouble(row.divide_ms), FormatDouble(row.prepare_ms),
                          FormatDouble(row.candidate_discovery_task_sum_ms),
                          FormatDouble(row.exact_gain_ms),
                          FormatDouble(row.matching_ms), FormatDouble(row.update_ms)});
      }
    }

    if (!opts.commit_audit_csv.empty()) {
      const std::vector<std::string> columns = {
          "dataset", "commit_policy", "batch_id", "batch_size",
          "partition_exact_cost", "cumulative_realized_gain",
          "rejected_pair_count"};
      const std::string dataset = DatasetLabelFromPath(opts.input);
      for (const CommitAuditRow& row : sweg.commit_audit_rows()) {
        AppendResultsCsv(
            opts.commit_audit_csv, columns,
            {dataset, CommitPolicyToString(opts.commit_policy),
             std::to_string(row.batch_id), std::to_string(row.batch_size),
             std::to_string(row.partition_exact_cost),
             std::to_string(row.cumulative_realized_gain),
             std::to_string(row.rejected_pair_count)});
      }
    }

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
          "cost_objective",
          "state_backend",
          "candidate_index",
          "candidate_budget",
          "certification",
          "commit_policy",
          "validate_quotient",
          "quotient_update",
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
          "num_superedges_nonloop",
          "num_superedges_loop",
          "num_positive_corrections",
          "num_negative_corrections",
          "encoding_cost_standard",
          "cost_ratio_standard",
          "encoding_cost_mags_x2",
          "encoding_cost_mags_compatible",
          "cost_ratio_mags_compatible",
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
          "partition_hash",
          "isolated_gain_sum", "realized_marginal_gain_sum",
          "actual_batch_cost_reduction", "interaction_delta",
          "selected_merges_for_validation",
          "accepted_merges_after_validation", "rejected_nonpositive",
          "stale_endpoint", "gain_decreased", "gain_increased",
          "negative_marginal", "candidate_refresh_count",
          "original_exact_calls", "validation_exact_calls",
          "validation_exact_row_entry_work", "commit_validation_ms",
          "audit_oracle_ms",
          "certification_candidates_seen",
          "upper_bound_pruned",
          "upper_bound_passed",
          "early_abort_count",
          "exact_full_scan_count",
          "exact_entries_available",
          "exact_entries_scanned",
          "exact_entries_skipped",
          "upper_bound_ms",
          "early_abort_exact_ms",
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
          CostObjectiveToString(opts.cost_objective),
          StateBackendToString(opts.state_backend),
          CandidateIndexModeToString(opts.candidate_index_mode),
          std::to_string(opts.candidate_budget),
          CertificationModeToString(opts.certification_mode),
          CommitPolicyToString(opts.commit_policy),
          opts.validate_quotient ? "true" : "false",
          QuotientUpdateModeToString(opts.quotient_update_mode),
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
          std::to_string(compression_metrics.encoding_cost_standard),
          FormatDouble(compression_metrics.cost_ratio_standard),
          std::to_string(compression_metrics.num_superedges_nonloop),
          std::to_string(compression_metrics.num_superedges_loop),
          std::to_string(compression_metrics.num_positive_corrections),
          std::to_string(compression_metrics.num_negative_corrections),
          std::to_string(compression_metrics.encoding_cost_standard),
          FormatDouble(compression_metrics.cost_ratio_standard),
          std::to_string(compression_metrics.encoding_cost_mags_x2),
          FormatDouble(compression_metrics.encoding_cost_mags_compatible),
          FormatDouble(compression_metrics.cost_ratio_mags_compatible),
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
          std::to_string(sweg.PartitionHash()),
          std::to_string(stats.isolated_gain_sum),
          std::to_string(stats.realized_marginal_gain_sum),
          std::to_string(stats.actual_batch_cost_reduction),
          std::to_string(stats.interaction_delta),
          std::to_string(stats.selected_merges_for_validation),
          std::to_string(stats.accepted_merges_after_validation),
          std::to_string(stats.rejected_nonpositive),
          std::to_string(stats.stale_endpoint),
          std::to_string(stats.gain_decreased),
          std::to_string(stats.gain_increased),
          std::to_string(stats.negative_marginal),
          std::to_string(stats.candidate_refresh_count),
          std::to_string(stats.original_exact_calls),
          std::to_string(stats.validation_exact_calls),
          std::to_string(stats.validation_exact_row_entry_work),
          FormatDouble(stats.commit_validation_ms),
          FormatDouble(stats.audit_oracle_ms),
          std::to_string(stats.certification_candidates_seen),
          std::to_string(stats.upper_bound_pruned),
          std::to_string(stats.upper_bound_passed),
          std::to_string(stats.early_abort_count),
          std::to_string(stats.exact_full_scan_count),
          std::to_string(stats.exact_entries_available),
          std::to_string(stats.exact_entries_scanned),
          std::to_string(stats.exact_entries_skipped),
          FormatDouble(stats.upper_bound_ms),
          FormatDouble(stats.early_abort_exact_ms),
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
