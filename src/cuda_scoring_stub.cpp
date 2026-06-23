#include "cuda_scoring.hpp"
#include "sweg.hpp"

#include <stdexcept>

struct Sweg::CudaScoringCache {};

const char* CudaScoringBuildMode() { return "stub"; }

bool CudaScoringCompiled() { return false; }

Sweg::~Sweg() = default;

std::vector<Sweg::LocalGainResult> Sweg::ScoreCandidatesCuda(
    const std::vector<std::pair<int, int>>& candidate_pairs,
    const FlatAggCSR& flat_agg, size_t candidate_chunk_size) {
  (void)candidate_pairs;
  (void)flat_agg;
  (void)candidate_chunk_size;
  (void)verify_cuda_gain_;
  throw std::runtime_error(
      "CUDA scoring backend is not compiled in this build. "
      "Current source is the stub backend. "
      "Use --scoring-backend cpu.");
}
