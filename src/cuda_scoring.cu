#include "cuda_scoring.hpp"
#include "sweg.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

inline double ElapsedMs(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

void CheckCuda(cudaError_t status, const char* context) {
  if (status == cudaSuccess) {
    return;
  }
  std::ostringstream oss;
  oss << context << " failed: " << cudaGetErrorString(status);
  throw std::runtime_error(oss.str());
}

__device__ inline int64_t EncodeCostDevice(int row_rep, int64_t row_size,
                                           int neighbor_rep,
                                           int64_t neighbor_size,
                                           int64_t edges) {
  int64_t capacity = 0;
  if (row_rep == neighbor_rep) {
    capacity = row_size * (row_size - 1) / 2;
  } else {
    capacity = row_size * neighbor_size;
  }
  const int64_t complement_cost = 1 + capacity - edges;
  return edges < complement_cost ? edges : complement_cost;
}

struct DeviceLocalGainResult {
  int64_t gain;
  int64_t before_cost;
};

template <typename T>
void EnsureDeviceCapacity(T** ptr, size_t* capacity, size_t required,
                          const char* label) {
  if (*capacity >= required) {
    return;
  }
  CheckCuda(cudaFree(*ptr), label);
  *ptr = nullptr;
  *capacity = 0;
  if (required == 0) {
    return;
  }
  CheckCuda(cudaMalloc(reinterpret_cast<void**>(ptr), sizeof(T) * required),
            label);
  *capacity = required;
}

inline Sweg::LocalGainResult ComputeLocalGainHostFromFlat(
    const Sweg::FlatAggCSR& flat_agg, int a_idx, int b_idx) {
  const int rep_a = flat_agg.row_rep_ids[static_cast<size_t>(a_idx)];
  const int rep_b = flat_agg.row_rep_ids[static_cast<size_t>(b_idx)];
  const int64_t size_a = flat_agg.row_sizes[static_cast<size_t>(a_idx)];
  const int64_t size_b = flat_agg.row_sizes[static_cast<size_t>(b_idx)];
  const int64_t self_loops_a =
      flat_agg.row_self_loops[static_cast<size_t>(a_idx)];
  const int64_t self_loops_b =
      flat_agg.row_self_loops[static_cast<size_t>(b_idx)];
  const int64_t size_m = size_a + size_b;

  const int a_begin = flat_agg.offsets[static_cast<size_t>(a_idx)];
  const int a_end = flat_agg.offsets[static_cast<size_t>(a_idx + 1)];
  const int b_begin = flat_agg.offsets[static_cast<size_t>(b_idx)];
  const int b_end = flat_agg.offsets[static_cast<size_t>(b_idx + 1)];

  int ia = a_begin;
  int ib = b_begin;
  int64_t before_total = 0;
  int64_t after_total = 0;
  bool seen_rep_a = false;
  bool seen_rep_b = false;

  auto encode_cost = [](int row_rep, int64_t row_size, int neighbor_rep,
                        int64_t neighbor_size, int64_t edges) {
    int64_t capacity = 0;
    if (row_rep == neighbor_rep) {
      capacity = row_size * (row_size - 1) / 2;
    } else {
      capacity = row_size * neighbor_size;
    }
    const int64_t complement_cost = 1 + capacity - edges;
    return std::min(edges, complement_cost);
  };

  auto accumulate_rep = [&](int rep_x, int64_t size_x, int64_t edges_a_raw,
                            int64_t edges_b_raw) {
    if (rep_x == rep_a) {
      seen_rep_a = true;
    }
    if (rep_x == rep_b) {
      seen_rep_b = true;
    }
    const int64_t edges_a =
        (rep_x == rep_a) ? std::max<int64_t>(0, edges_a_raw - self_loops_a)
                         : edges_a_raw;
    const int64_t edges_b =
        (rep_x == rep_b) ? std::max<int64_t>(0, edges_b_raw - self_loops_b)
                         : edges_b_raw;
    const int64_t edges_m = edges_a + edges_b;

    before_total += encode_cost(rep_a, size_a, rep_x, size_x, edges_a) +
                    encode_cost(rep_b, size_b, rep_x, size_x, edges_b);

    const int merged_rep = (rep_x == rep_b) ? rep_a : rep_x;
    const int64_t merged_size_x = (rep_x == rep_b) ? size_m : size_x;
    after_total +=
        encode_cost(rep_a, size_m, merged_rep, merged_size_x, edges_m);
  };

  while (ia < a_end || ib < b_end) {
    int rep_x = -1;
    int64_t size_x = 0;
    int64_t edges_a_raw = 0;
    int64_t edges_b_raw = 0;

    if (ib >= b_end ||
        (ia < a_end && flat_agg.neighbors[static_cast<size_t>(ia)] <
                           flat_agg.neighbors[static_cast<size_t>(ib)])) {
      rep_x = flat_agg.neighbors[static_cast<size_t>(ia)];
      size_x = flat_agg.neighbor_sizes[static_cast<size_t>(ia)];
      edges_a_raw = flat_agg.weights[static_cast<size_t>(ia)];
      ++ia;
    } else if (ia >= a_end ||
               flat_agg.neighbors[static_cast<size_t>(ib)] <
                   flat_agg.neighbors[static_cast<size_t>(ia)]) {
      rep_x = flat_agg.neighbors[static_cast<size_t>(ib)];
      size_x = flat_agg.neighbor_sizes[static_cast<size_t>(ib)];
      edges_b_raw = flat_agg.weights[static_cast<size_t>(ib)];
      ++ib;
    } else {
      rep_x = flat_agg.neighbors[static_cast<size_t>(ia)];
      size_x = flat_agg.neighbor_sizes[static_cast<size_t>(ia)];
      edges_a_raw = flat_agg.weights[static_cast<size_t>(ia)];
      edges_b_raw = flat_agg.weights[static_cast<size_t>(ib)];
      ++ia;
      ++ib;
    }

    accumulate_rep(rep_x, size_x, edges_a_raw, edges_b_raw);
  }

  if (!seen_rep_a) {
    accumulate_rep(rep_a, size_a, 0, 0);
  }
  if (rep_b != rep_a && !seen_rep_b) {
    accumulate_rep(rep_b, size_b, 0, 0);
  }

  return Sweg::LocalGainResult{before_total - after_total, before_total};
}

__device__ inline int64_t ClampSubNonNegative(int64_t lhs, int64_t rhs) {
  const int64_t value = lhs - rhs;
  return value > 0 ? value : 0;
}

__device__ inline void AccumulateRepDevice(
    int rep_a, int rep_b, int64_t size_a, int64_t size_b, int64_t size_m,
    int64_t self_loops_a, int64_t self_loops_b, int rep_x, int64_t size_x,
    int64_t edges_a_raw, int64_t edges_b_raw, bool* seen_rep_a,
    bool* seen_rep_b, int64_t* before_total, int64_t* after_total) {
  if (rep_x == rep_a) {
    *seen_rep_a = true;
  }
  if (rep_x == rep_b) {
    *seen_rep_b = true;
  }
  const int64_t edges_a =
      (rep_x == rep_a) ? ClampSubNonNegative(edges_a_raw, self_loops_a)
                       : edges_a_raw;
  const int64_t edges_b =
      (rep_x == rep_b) ? ClampSubNonNegative(edges_b_raw, self_loops_b)
                       : edges_b_raw;
  const int64_t edges_m = edges_a + edges_b;

  *before_total += EncodeCostDevice(rep_a, size_a, rep_x, size_x, edges_a) +
                   EncodeCostDevice(rep_b, size_b, rep_x, size_x, edges_b);

  const int merged_rep = (rep_x == rep_b) ? rep_a : rep_x;
  const int64_t merged_size_x = (rep_x == rep_b) ? size_m : size_x;
  *after_total +=
      EncodeCostDevice(rep_a, size_m, merged_rep, merged_size_x, edges_m);
}

__global__ void ComputeLocalGainKernel(
    const int2* candidate_pairs, int num_candidates, const int* offsets,
    const int* row_rep_ids, const int* neighbors, const int64_t* weights,
    const int64_t* neighbor_sizes, const int64_t* row_sizes,
    const int64_t* row_self_loops, DeviceLocalGainResult* out_results) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_candidates) {
    return;
  }

  const int a_idx = candidate_pairs[idx].x;
  const int b_idx = candidate_pairs[idx].y;
  const int rep_a = row_rep_ids[a_idx];
  const int rep_b = row_rep_ids[b_idx];
  const int64_t size_a = row_sizes[a_idx];
  const int64_t size_b = row_sizes[b_idx];
  const int64_t self_loops_a = row_self_loops[a_idx];
  const int64_t self_loops_b = row_self_loops[b_idx];
  const int64_t size_m = size_a + size_b;

  const int a_begin = offsets[a_idx];
  const int a_end = offsets[a_idx + 1];
  const int b_begin = offsets[b_idx];
  const int b_end = offsets[b_idx + 1];

  int ia = a_begin;
  int ib = b_begin;
  int64_t before_total = 0;
  int64_t after_total = 0;
  bool seen_rep_a = false;
  bool seen_rep_b = false;

  while (ia < a_end || ib < b_end) {
    int rep_x = -1;
    int64_t size_x = 0;
    int64_t edges_a_raw = 0;
    int64_t edges_b_raw = 0;

    if (ib >= b_end || (ia < a_end && neighbors[ia] < neighbors[ib])) {
      rep_x = neighbors[ia];
      size_x = neighbor_sizes[ia];
      edges_a_raw = weights[ia];
      ++ia;
    } else if (ia >= a_end || neighbors[ib] < neighbors[ia]) {
      rep_x = neighbors[ib];
      size_x = neighbor_sizes[ib];
      edges_b_raw = weights[ib];
      ++ib;
    } else {
      rep_x = neighbors[ia];
      size_x = neighbor_sizes[ia];
      edges_a_raw = weights[ia];
      edges_b_raw = weights[ib];
      ++ia;
      ++ib;
    }

    AccumulateRepDevice(rep_a, rep_b, size_a, size_b, size_m, self_loops_a,
                        self_loops_b, rep_x, size_x, edges_a_raw, edges_b_raw,
                        &seen_rep_a, &seen_rep_b, &before_total, &after_total);
  }

  if (!seen_rep_a) {
    AccumulateRepDevice(rep_a, rep_b, size_a, size_b, size_m, self_loops_a,
                        self_loops_b, rep_a, size_a, 0, 0, &seen_rep_a,
                        &seen_rep_b, &before_total, &after_total);
  }
  if (rep_b != rep_a && !seen_rep_b) {
    AccumulateRepDevice(rep_a, rep_b, size_a, size_b, size_m, self_loops_a,
                        self_loops_b, rep_b, size_b, 0, 0, &seen_rep_a,
                        &seen_rep_b, &before_total, &after_total);
  }

  out_results[idx].gain = before_total - after_total;
  out_results[idx].before_cost = before_total;
}

}  // namespace

struct Sweg::CudaScoringCache {
  int2* d_pairs = nullptr;
  int* d_offsets = nullptr;
  int* d_row_rep_ids = nullptr;
  int* d_neighbors = nullptr;
  int64_t* d_weights = nullptr;
  int64_t* d_neighbor_sizes = nullptr;
  int64_t* d_row_sizes = nullptr;
  int64_t* d_row_self_loops = nullptr;
  DeviceLocalGainResult* d_results = nullptr;

  size_t pair_capacity = 0;
  size_t offset_capacity = 0;
  size_t row_rep_capacity = 0;
  size_t neighbor_capacity = 0;
  size_t weight_capacity = 0;
  size_t neighbor_size_capacity = 0;
  size_t row_size_capacity = 0;
  size_t row_self_loop_capacity = 0;
  size_t result_capacity = 0;

  ~CudaScoringCache() {
    cudaFree(d_pairs);
    cudaFree(d_offsets);
    cudaFree(d_row_rep_ids);
    cudaFree(d_neighbors);
    cudaFree(d_weights);
    cudaFree(d_neighbor_sizes);
    cudaFree(d_row_sizes);
    cudaFree(d_row_self_loops);
    cudaFree(d_results);
  }
};

const char* CudaScoringBuildMode() { return "cuda"; }

bool CudaScoringCompiled() { return true; }

Sweg::~Sweg() = default;

std::vector<Sweg::LocalGainResult> Sweg::ScoreCandidatesCuda(
    const std::vector<std::pair<int, int>>& candidate_pairs,
    const FlatAggCSR& flat_agg) {
  if (candidate_pairs.empty()) {
    return {};
  }

  const auto cuda_total_start = Clock::now();
  stats_.cuda_num_calls += 1;

  std::vector<int2> host_pairs(candidate_pairs.size());
  for (size_t i = 0; i < candidate_pairs.size(); ++i) {
    host_pairs[i] =
        make_int2(candidate_pairs[i].first, candidate_pairs[i].second);
  }

  if (!cuda_cache_) {
    cuda_cache_ = std::make_unique<CudaScoringCache>();
  }
  CudaScoringCache& cache = *cuda_cache_;

  try {
    const auto h2d_start = Clock::now();
    EnsureDeviceCapacity(&cache.d_pairs, &cache.pair_capacity, host_pairs.size(),
                         "cudaMalloc(d_pairs)");
    EnsureDeviceCapacity(&cache.d_offsets, &cache.offset_capacity,
                         flat_agg.offsets.size(), "cudaMalloc(d_offsets)");
    EnsureDeviceCapacity(&cache.d_row_rep_ids, &cache.row_rep_capacity,
                         flat_agg.row_rep_ids.size(),
                         "cudaMalloc(d_row_rep_ids)");
    EnsureDeviceCapacity(&cache.d_neighbors, &cache.neighbor_capacity,
                         flat_agg.neighbors.size(), "cudaMalloc(d_neighbors)");
    EnsureDeviceCapacity(&cache.d_weights, &cache.weight_capacity,
                         flat_agg.weights.size(), "cudaMalloc(d_weights)");
    EnsureDeviceCapacity(&cache.d_neighbor_sizes,
                         &cache.neighbor_size_capacity,
                         flat_agg.neighbor_sizes.size(),
                         "cudaMalloc(d_neighbor_sizes)");
    EnsureDeviceCapacity(&cache.d_row_sizes, &cache.row_size_capacity,
                         flat_agg.row_sizes.size(), "cudaMalloc(d_row_sizes)");
    EnsureDeviceCapacity(&cache.d_row_self_loops,
                         &cache.row_self_loop_capacity,
                         flat_agg.row_self_loops.size(),
                         "cudaMalloc(d_row_self_loops)");
    EnsureDeviceCapacity(&cache.d_results, &cache.result_capacity,
                         host_pairs.size(), "cudaMalloc(d_results)");

    CheckCuda(cudaMemcpy(cache.d_pairs, host_pairs.data(),
                         sizeof(int2) * host_pairs.size(),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy(d_pairs)");
    CheckCuda(cudaMemcpy(cache.d_offsets, flat_agg.offsets.data(),
                         sizeof(int) * flat_agg.offsets.size(),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy(d_offsets)");
    CheckCuda(cudaMemcpy(cache.d_row_rep_ids, flat_agg.row_rep_ids.data(),
                         sizeof(int) * flat_agg.row_rep_ids.size(),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy(d_row_rep_ids)");
    CheckCuda(cudaMemcpy(cache.d_neighbors, flat_agg.neighbors.data(),
                         sizeof(int) * flat_agg.neighbors.size(),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy(d_neighbors)");
    CheckCuda(cudaMemcpy(cache.d_weights, flat_agg.weights.data(),
                         sizeof(int64_t) * flat_agg.weights.size(),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy(d_weights)");
    CheckCuda(cudaMemcpy(cache.d_neighbor_sizes, flat_agg.neighbor_sizes.data(),
                         sizeof(int64_t) * flat_agg.neighbor_sizes.size(),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy(d_neighbor_sizes)");
    CheckCuda(cudaMemcpy(cache.d_row_sizes, flat_agg.row_sizes.data(),
                         sizeof(int64_t) * flat_agg.row_sizes.size(),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy(d_row_sizes)");
    CheckCuda(cudaMemcpy(cache.d_row_self_loops, flat_agg.row_self_loops.data(),
                         sizeof(int64_t) * flat_agg.row_self_loops.size(),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy(d_row_self_loops)");
    const auto h2d_end = Clock::now();
    stats_.cuda_h2d_ms += ElapsedMs(h2d_start, h2d_end);

    const auto kernel_start = Clock::now();
    constexpr int kBlockSize = 256;
    const int blocks =
        static_cast<int>((host_pairs.size() + kBlockSize - 1) / kBlockSize);
    ComputeLocalGainKernel<<<blocks, kBlockSize>>>(
        cache.d_pairs, static_cast<int>(host_pairs.size()), cache.d_offsets,
        cache.d_row_rep_ids, cache.d_neighbors, cache.d_weights,
        cache.d_neighbor_sizes, cache.d_row_sizes, cache.d_row_self_loops,
        cache.d_results);
    CheckCuda(cudaGetLastError(), "ComputeLocalGainKernel launch");
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    const auto kernel_end = Clock::now();
    stats_.cuda_kernel_ms += ElapsedMs(kernel_start, kernel_end);

    const auto d2h_start = Clock::now();
    std::vector<DeviceLocalGainResult> host_results(host_pairs.size());
    CheckCuda(cudaMemcpy(host_results.data(), cache.d_results,
                         sizeof(DeviceLocalGainResult) * host_results.size(),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy(d_results)");
    const auto d2h_end = Clock::now();
    stats_.cuda_d2h_ms += ElapsedMs(d2h_start, d2h_end);

    std::vector<LocalGainResult> results(host_results.size());
    for (size_t i = 0; i < host_results.size(); ++i) {
      results[i].gain = host_results[i].gain;
      results[i].before_cost = host_results[i].before_cost;
    }

    if (verify_cuda_gain_ && cuda_verify_samples_remaining_ > 0) {
      std::mt19937_64 verify_rng(1);
      const size_t samples = std::min(
          cuda_verify_samples_remaining_,
          std::min<size_t>(64, candidate_pairs.size()));
      for (size_t s = 0; s < samples; ++s) {
        const size_t idx = verify_rng() % candidate_pairs.size();
        const int a_idx = candidate_pairs[idx].first;
        const int b_idx = candidate_pairs[idx].second;
        const LocalGainResult cpu_result = ComputeLocalGainHostFromFlat(
            flat_agg, a_idx, b_idx);
        if (cpu_result.gain != results[idx].gain ||
            cpu_result.before_cost != results[idx].before_cost) {
          std::ostringstream oss;
          oss << "CUDA gain verification failed at candidate " << idx
              << " (a_idx=" << a_idx << ", b_idx=" << b_idx
              << "): cpu_gain=" << cpu_result.gain
              << " gpu_gain=" << results[idx].gain
              << " cpu_before=" << cpu_result.before_cost
              << " gpu_before=" << results[idx].before_cost;
          throw std::runtime_error(oss.str());
        }
      }
      cuda_verify_samples_remaining_ -= samples;
    }

    const auto cuda_total_end = Clock::now();
    stats_.cuda_total_ms += ElapsedMs(cuda_total_start, cuda_total_end);
    return results;
  } catch (...) {
    throw;
  }
}
