#include "cuda_scoring.hpp"
#include "sweg.hpp"

#include <stdexcept>

struct Sweg::CudaScoringCache {};

const char* CudaScoringBuildMode() { return "stub"; }

bool CudaScoringCompiled() { return false; }

Sweg::~Sweg() = default;

void Sweg::EnsureCudaScoringCache() {
  throw std::runtime_error(
      "CUDA scoring backend is not compiled in this build. "
      "Current source is the stub backend. "
      "Use --scoring-backend cpu.");
}

size_t Sweg::QueryCudaSliceMemoryBudgetBytes() {
  return 0;
}

size_t Sweg::GetCudaSliceMemoryBudgetBytes() {
  if (cuda_slice_memory_mb_ > 0) {
    return static_cast<size_t>(cuda_slice_memory_mb_) * 1024ULL * 1024ULL;
  }
  return QueryCudaSliceMemoryBudgetBytes();
}

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
