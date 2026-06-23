#include "cuda_scoring.hpp"
#include "sweg.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <limits>
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

void DestroyEventNoThrow(cudaEvent_t event) noexcept {
  if (event != nullptr) {
    cudaEventDestroy(event);
  }
}

void DestroyStreamNoThrow(cudaStream_t stream) noexcept {
  if (stream != nullptr) {
    cudaStreamDestroy(stream);
  }
}

template <typename T>
void FreeDeviceNoThrow(T* ptr) noexcept {
  if (ptr != nullptr) {
    cudaFree(ptr);
  }
}

template <typename T>
void FreeHostNoThrow(T* ptr) noexcept {
  if (ptr != nullptr) {
    cudaFreeHost(ptr);
  }
}

size_t GrowCapacity(size_t current, size_t required) {
  if (current >= required) {
    return current;
  }
  size_t next = std::max<size_t>(current, 1024);
  while (next < required) {
    const size_t grown = next + next / 2;
    if (grown <= next) {
      return required;
    }
    next = grown;
  }
  return next;
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

struct PackedCudaInputLayout {
  size_t total_bytes = 0;
  size_t offsets_offset = 0;
  size_t row_rep_ids_offset = 0;
  size_t neighbors_offset = 0;
  size_t weights_offset = 0;
  size_t neighbor_sizes_offset = 0;
  size_t row_sizes_offset = 0;
  size_t row_self_loops_offset = 0;
  size_t pairs_offset = 0;
};

size_t AlignUp(size_t value, size_t alignment) {
  if (alignment <= 1) {
    return value;
  }
  const size_t remainder = value % alignment;
  return remainder == 0 ? value : (value + alignment - remainder);
}

template <typename T>
size_t AppendPackedSection(size_t offset, size_t count, size_t* section_offset) {
  offset = AlignUp(offset, alignof(T));
  if (section_offset != nullptr) {
    *section_offset = offset;
  }
  return offset + sizeof(T) * count;
}

PackedCudaInputLayout BuildPackedCudaInputLayout(
    const Sweg::FlatAggCSR& flat_agg, size_t pair_count) {
  PackedCudaInputLayout layout;
  size_t offset = 0;
  offset = AppendPackedSection<int>(offset, flat_agg.offsets.size(),
                                    &layout.offsets_offset);
  offset = AppendPackedSection<int>(offset, flat_agg.row_rep_ids.size(),
                                    &layout.row_rep_ids_offset);
  offset = AppendPackedSection<int>(offset, flat_agg.neighbors.size(),
                                    &layout.neighbors_offset);
  offset = AppendPackedSection<int64_t>(offset, flat_agg.weights.size(),
                                        &layout.weights_offset);
  offset = AppendPackedSection<int64_t>(offset, flat_agg.neighbor_sizes.size(),
                                        &layout.neighbor_sizes_offset);
  offset = AppendPackedSection<int64_t>(offset, flat_agg.row_sizes.size(),
                                        &layout.row_sizes_offset);
  offset = AppendPackedSection<int64_t>(offset, flat_agg.row_self_loops.size(),
                                        &layout.row_self_loops_offset);
  offset = AppendPackedSection<int2>(offset, pair_count, &layout.pairs_offset);
  layout.total_bytes = offset;
  return layout;
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

namespace {

template <typename T>
void EnsureDeviceCapacity(T** ptr, size_t* capacity, size_t required,
                          const char* label);

}  // namespace

struct Sweg::CudaScoringCache {
  size_t slice_memory_budget_bytes = 0;
  bool slice_memory_budget_initialized = false;

  void* h_input_blob = nullptr;
  void* d_input_blob = nullptr;
  DeviceLocalGainResult* h_results = nullptr;
  DeviceLocalGainResult* d_results = nullptr;
  size_t input_capacity = 0;
  size_t output_capacity = 0;

  cudaStream_t transfer_stream = nullptr;
  cudaStream_t compute_streams[2] = {nullptr, nullptr};
  cudaEvent_t input_ready = nullptr;
  cudaEvent_t compute_done[2] = {nullptr, nullptr};
  cudaEvent_t result_ready = nullptr;
  cudaEvent_t packed_h2d_start = nullptr;
  cudaEvent_t packed_h2d_end = nullptr;
  cudaEvent_t packed_d2h_start = nullptr;
  cudaEvent_t packed_d2h_end = nullptr;
  std::vector<cudaEvent_t> kernel_start_events;
  std::vector<cudaEvent_t> kernel_end_events;

  CudaScoringCache() {
    try {
      CheckCuda(
          cudaStreamCreateWithFlags(&transfer_stream, cudaStreamNonBlocking),
          "cudaStreamCreateWithFlags(transfer)");
      for (int i = 0; i < 2; ++i) {
        CheckCuda(cudaStreamCreateWithFlags(&compute_streams[i],
                                            cudaStreamNonBlocking),
                  "cudaStreamCreateWithFlags(compute)");
      }
      CheckCuda(cudaEventCreate(&input_ready), "cudaEventCreate(input_ready)");
      CheckCuda(cudaEventCreate(&compute_done[0]),
                "cudaEventCreate(compute_done0)");
      CheckCuda(cudaEventCreate(&compute_done[1]),
                "cudaEventCreate(compute_done1)");
      CheckCuda(cudaEventCreate(&result_ready), "cudaEventCreate(result_ready)");
      CheckCuda(cudaEventCreate(&packed_h2d_start),
                "cudaEventCreate(packed_h2d_start)");
      CheckCuda(cudaEventCreate(&packed_h2d_end),
                "cudaEventCreate(packed_h2d_end)");
      CheckCuda(cudaEventCreate(&packed_d2h_start),
                "cudaEventCreate(packed_d2h_start)");
      CheckCuda(cudaEventCreate(&packed_d2h_end),
                "cudaEventCreate(packed_d2h_end)");
    } catch (...) {
      FreeHostNoThrow(h_input_blob);
      FreeDeviceNoThrow(d_input_blob);
      FreeHostNoThrow(h_results);
      FreeDeviceNoThrow(d_results);
      DestroyEventNoThrow(input_ready);
      DestroyEventNoThrow(compute_done[0]);
      DestroyEventNoThrow(compute_done[1]);
      DestroyEventNoThrow(result_ready);
      DestroyEventNoThrow(packed_h2d_start);
      DestroyEventNoThrow(packed_h2d_end);
      DestroyEventNoThrow(packed_d2h_start);
      DestroyEventNoThrow(packed_d2h_end);
      for (cudaEvent_t event : kernel_start_events) {
        DestroyEventNoThrow(event);
      }
      for (cudaEvent_t event : kernel_end_events) {
        DestroyEventNoThrow(event);
      }
      DestroyStreamNoThrow(transfer_stream);
      for (int i = 0; i < 2; ++i) {
        DestroyStreamNoThrow(compute_streams[i]);
      }
      throw;
    }
  }

  ~CudaScoringCache() {
    if (transfer_stream != nullptr) {
      cudaStreamSynchronize(transfer_stream);
    }
    for (int i = 0; i < 2; ++i) {
      if (compute_streams[i] != nullptr) {
        cudaStreamSynchronize(compute_streams[i]);
      }
    }
    FreeHostNoThrow(h_input_blob);
    FreeDeviceNoThrow(d_input_blob);
    FreeHostNoThrow(h_results);
    FreeDeviceNoThrow(d_results);
    DestroyEventNoThrow(input_ready);
    DestroyEventNoThrow(compute_done[0]);
    DestroyEventNoThrow(compute_done[1]);
    DestroyEventNoThrow(result_ready);
    DestroyEventNoThrow(packed_h2d_start);
    DestroyEventNoThrow(packed_h2d_end);
    DestroyEventNoThrow(packed_d2h_start);
    DestroyEventNoThrow(packed_d2h_end);
    for (cudaEvent_t event : kernel_start_events) {
      DestroyEventNoThrow(event);
    }
    for (cudaEvent_t event : kernel_end_events) {
      DestroyEventNoThrow(event);
    }
    DestroyStreamNoThrow(transfer_stream);
    for (int i = 0; i < 2; ++i) {
      DestroyStreamNoThrow(compute_streams[i]);
    }
  }

  void EnsureInputCapacity(size_t required_bytes) {
    if (input_capacity >= required_bytes) {
      return;
    }
    const size_t new_capacity = GrowCapacity(input_capacity, required_bytes);
    FreeHostNoThrow(h_input_blob);
    FreeDeviceNoThrow(d_input_blob);
    h_input_blob = nullptr;
    d_input_blob = nullptr;
    if (new_capacity == 0) {
      input_capacity = 0;
      return;
    }
    CheckCuda(cudaHostAlloc(&h_input_blob, new_capacity, cudaHostAllocDefault),
              "cudaHostAlloc h_input_blob");
    CheckCuda(cudaMalloc(&d_input_blob, new_capacity),
              "cudaMalloc d_input_blob");
    input_capacity = new_capacity;
  }

  void EnsureOutputCapacity(size_t required_results) {
    if (output_capacity >= required_results) {
      return;
    }
    const size_t new_capacity = GrowCapacity(output_capacity, required_results);
    FreeHostNoThrow(h_results);
    FreeDeviceNoThrow(d_results);
    h_results = nullptr;
    d_results = nullptr;
    if (new_capacity == 0) {
      output_capacity = 0;
      return;
    }
    CheckCuda(cudaHostAlloc(reinterpret_cast<void**>(&h_results),
                            sizeof(DeviceLocalGainResult) * new_capacity,
                            cudaHostAllocDefault),
              "cudaHostAlloc h_results");
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_results),
                         sizeof(DeviceLocalGainResult) * new_capacity),
              "cudaMalloc d_results");
    output_capacity = new_capacity;
  }

  void EnsureKernelEventCapacity(size_t required) {
    if (kernel_start_events.size() >= required) {
      return;
    }
    const size_t old_size = kernel_start_events.size();
    kernel_start_events.resize(required, nullptr);
    kernel_end_events.resize(required, nullptr);
    for (size_t i = old_size; i < required; ++i) {
      CheckCuda(cudaEventCreate(&kernel_start_events[i]),
                "cudaEventCreate(kernel_start)");
      CheckCuda(cudaEventCreate(&kernel_end_events[i]),
                "cudaEventCreate(kernel_end)");
    }
  }
};

namespace {

template <typename T>
void EnsureDeviceCapacity(T** ptr, size_t* capacity, size_t required,
                          const char* label) {
  if (*capacity >= required) {
    return;
  }
  const size_t new_capacity = GrowCapacity(*capacity, required);
  FreeDeviceNoThrow(*ptr);
  *ptr = nullptr;
  *capacity = 0;
  if (new_capacity == 0) {
    return;
  }
  CheckCuda(cudaMalloc(reinterpret_cast<void**>(ptr), sizeof(T) * new_capacity),
            label);
  *capacity = new_capacity;
}

std::vector<size_t> BuildVerificationIndices(size_t candidate_count,
                                             size_t chunk_size,
                                             uint64_t call_counter,
                                             size_t remaining_budget) {
  std::vector<size_t> indices;
  if (candidate_count == 0 || remaining_budget == 0) {
    return indices;
  }

  auto add = [&](size_t idx) {
    if (idx >= candidate_count) {
      return;
    }
    indices.push_back(idx);
  };

  add(0);
  add(candidate_count - 1);

  if (chunk_size == 0) {
    chunk_size = candidate_count;
  }
  if (chunk_size > 0) {
    for (size_t start = 0; start < candidate_count; start += chunk_size) {
      add(start);
      add(std::min(candidate_count - 1, start + chunk_size - 1));
    }
    if (chunk_size > 0) {
      add(chunk_size - 1);
    }
    add(chunk_size);
    add(chunk_size + 1);
  }

  std::mt19937_64 rng(0x9e3779b97f4a7c15ULL ^
                      (call_counter * 0xbf58476d1ce4e5b9ULL) ^
                      static_cast<uint64_t>(candidate_count));
  const size_t random_target =
      std::min<size_t>(16, std::max<size_t>(0, remaining_budget));
  for (size_t i = 0; i < random_target; ++i) {
    add(static_cast<size_t>(rng() % candidate_count));
  }

  std::sort(indices.begin(), indices.end());
  indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
  if (indices.size() > remaining_budget) {
    indices.resize(remaining_budget);
  }
  return indices;
}

void VerifyCudaResultsIfEnabled(const std::vector<std::pair<int, int>>& candidate_pairs,
                                const Sweg::FlatAggCSR& flat_agg,
                                const std::vector<Sweg::LocalGainResult>& results,
                                size_t chunk_size, bool verify_enabled,
                                size_t* remaining_budget,
                                uint64_t* call_counter) {
  if (!verify_enabled || remaining_budget == nullptr || *remaining_budget == 0) {
    return;
  }

  const size_t effective_chunk_size =
      chunk_size == 0 ? candidate_pairs.size() : chunk_size;
  const size_t budget = candidate_pairs.size() <= 10000
                            ? candidate_pairs.size()
                            : *remaining_budget;
  const uint64_t call_id = call_counter != nullptr ? *call_counter : 0;
  std::vector<size_t> indices =
      candidate_pairs.size() <= 10000
          ? [&]() {
              std::vector<size_t> all(candidate_pairs.size());
              for (size_t i = 0; i < candidate_pairs.size(); ++i) {
                all[i] = i;
              }
              return all;
            }()
          : BuildVerificationIndices(candidate_pairs.size(), effective_chunk_size,
                                     call_id, budget);

  for (size_t idx : indices) {
    const int a_idx = candidate_pairs[idx].first;
    const int b_idx = candidate_pairs[idx].second;
    const Sweg::LocalGainResult cpu_result =
        ComputeLocalGainHostFromFlat(flat_agg, a_idx, b_idx);
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

  if (candidate_pairs.size() <= 10000) {
    *remaining_budget =
        (*remaining_budget > candidate_pairs.size())
            ? (*remaining_budget - candidate_pairs.size())
            : 0;
  } else if (*remaining_budget > indices.size()) {
    *remaining_budget -= indices.size();
  } else {
    *remaining_budget = 0;
  }

  if (call_counter != nullptr) {
    *call_counter += 1;
  }
}

}  // namespace

const char* CudaScoringBuildMode() { return "cuda"; }

bool CudaScoringCompiled() { return true; }

Sweg::~Sweg() = default;

void Sweg::EnsureCudaScoringCache() {
  if (cuda_cache_) {
    return;
  }
  const auto init_start = Clock::now();
  cuda_cache_ = std::make_shared<CudaScoringCache>();
  stats_.cuda_init_ms += ElapsedMs(init_start, Clock::now());
}

size_t Sweg::QueryCudaSliceMemoryBudgetBytes() {
  EnsureCudaScoringCache();
  CudaScoringCache& cache = *cuda_cache_;
  if (cache.slice_memory_budget_initialized) {
    return cache.slice_memory_budget_bytes;
  }

  size_t free_bytes = 0;
  size_t total_bytes = 0;
  CheckCuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
  const size_t reserve_bytes = free_bytes / 5;
  const size_t usable_after_reserve =
      free_bytes > reserve_bytes ? (free_bytes - reserve_bytes) : 0;
  const size_t quarter_free = free_bytes / 4;
  const size_t auto_budget =
      std::min(static_cast<size_t>(512ULL * 1024ULL * 1024ULL),
               std::min(usable_after_reserve, quarter_free));
  cache.slice_memory_budget_bytes = auto_budget;
  cache.slice_memory_budget_initialized = true;
  return cache.slice_memory_budget_bytes;
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
  if (candidate_pairs.empty()) {
    return {};
  }

  EnsureCudaScoringCache();
  CudaScoringCache& cache = *cuda_cache_;

  const auto cuda_total_start = Clock::now();
  stats_.cuda_num_calls += 1;

  std::vector<LocalGainResult> results(candidate_pairs.size());
  const size_t chunk_size =
      candidate_chunk_size == 0
          ? candidate_pairs.size()
          : std::max<size_t>(1, std::min(candidate_chunk_size,
                                         candidate_pairs.size()));

  const PackedCudaInputLayout layout =
      BuildPackedCudaInputLayout(flat_agg, candidate_pairs.size());
  cache.EnsureInputCapacity(layout.total_bytes);
  cache.EnsureOutputCapacity(candidate_pairs.size());

  auto copy_vec = [&](size_t offset, const auto& vec) {
    using T = typename std::decay_t<decltype(vec)>::value_type;
    if (!vec.empty()) {
      std::memcpy(static_cast<char*>(cache.h_input_blob) + offset, vec.data(),
                  sizeof(T) * vec.size());
    }
  };

  copy_vec(layout.offsets_offset, flat_agg.offsets);
  copy_vec(layout.row_rep_ids_offset, flat_agg.row_rep_ids);
  copy_vec(layout.neighbors_offset, flat_agg.neighbors);
  copy_vec(layout.weights_offset, flat_agg.weights);
  copy_vec(layout.neighbor_sizes_offset, flat_agg.neighbor_sizes);
  copy_vec(layout.row_sizes_offset, flat_agg.row_sizes);
  copy_vec(layout.row_self_loops_offset, flat_agg.row_self_loops);
  int2* h_pairs = reinterpret_cast<int2*>(
      static_cast<char*>(cache.h_input_blob) + layout.pairs_offset);
  for (size_t i = 0; i < candidate_pairs.size(); ++i) {
    h_pairs[i] = make_int2(candidate_pairs[i].first, candidate_pairs[i].second);
  }

  CheckCuda(cudaEventRecord(cache.packed_h2d_start, cache.transfer_stream),
            "cudaEventRecord(packed_h2d_start)");
  CheckCuda(cudaMemcpyAsync(cache.d_input_blob, cache.h_input_blob,
                            layout.total_bytes, cudaMemcpyHostToDevice,
                            cache.transfer_stream),
            "cudaMemcpyAsync packed input");
  CheckCuda(cudaEventRecord(cache.packed_h2d_end, cache.transfer_stream),
            "cudaEventRecord(packed_h2d_end)");
  CheckCuda(cudaEventRecord(cache.input_ready, cache.transfer_stream),
            "cudaEventRecord(input_ready)");

  const int* d_offsets = reinterpret_cast<const int*>(
      static_cast<const char*>(cache.d_input_blob) + layout.offsets_offset);
  const int* d_row_rep_ids = reinterpret_cast<const int*>(
      static_cast<const char*>(cache.d_input_blob) + layout.row_rep_ids_offset);
  const int* d_neighbors = reinterpret_cast<const int*>(
      static_cast<const char*>(cache.d_input_blob) + layout.neighbors_offset);
  const int64_t* d_weights = reinterpret_cast<const int64_t*>(
      static_cast<const char*>(cache.d_input_blob) + layout.weights_offset);
  const int64_t* d_neighbor_sizes = reinterpret_cast<const int64_t*>(
      static_cast<const char*>(cache.d_input_blob) + layout.neighbor_sizes_offset);
  const int64_t* d_row_sizes = reinterpret_cast<const int64_t*>(
      static_cast<const char*>(cache.d_input_blob) + layout.row_sizes_offset);
  const int64_t* d_row_self_loops = reinterpret_cast<const int64_t*>(
      static_cast<const char*>(cache.d_input_blob) + layout.row_self_loops_offset);
  const int2* d_pairs = reinterpret_cast<const int2*>(
      static_cast<const char*>(cache.d_input_blob) + layout.pairs_offset);

  const size_t num_chunks =
      chunk_size == 0 ? 0 : (candidate_pairs.size() + chunk_size - 1) / chunk_size;
  cache.EnsureKernelEventCapacity(num_chunks);

  for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
    cudaStream_t compute_stream = cache.compute_streams[chunk_idx % 2];
    CheckCuda(cudaStreamWaitEvent(compute_stream, cache.input_ready, 0),
              "cudaStreamWaitEvent(input_ready)");
    const size_t offset = chunk_idx * chunk_size;
    const size_t count = std::min(chunk_size, candidate_pairs.size() - offset);
    CheckCuda(cudaEventRecord(cache.kernel_start_events[chunk_idx], compute_stream),
              "cudaEventRecord(kernel_start)");
    constexpr int kBlockSize = 256;
    const int blocks =
        static_cast<int>((count + kBlockSize - 1) / kBlockSize);
    ComputeLocalGainKernel<<<blocks, kBlockSize, 0, compute_stream>>>(
        d_pairs + offset, static_cast<int>(count), d_offsets, d_row_rep_ids,
        d_neighbors, d_weights, d_neighbor_sizes, d_row_sizes,
        d_row_self_loops, cache.d_results + offset);
    CheckCuda(cudaGetLastError(), "ComputeLocalGainKernel launch");
    CheckCuda(cudaEventRecord(cache.kernel_end_events[chunk_idx], compute_stream),
              "cudaEventRecord(kernel_end)");
    CheckCuda(cudaEventRecord(cache.compute_done[chunk_idx % 2], compute_stream),
              "cudaEventRecord(compute_done)");

    stats_.cuda_kernel_launches += 1;
    stats_.cuda_max_candidates_per_launch =
        std::max<int64_t>(stats_.cuda_max_candidates_per_launch,
                          static_cast<int64_t>(count));
  }

  if (num_chunks == 0) {
    CheckCuda(cudaEventRecord(cache.compute_done[0], cache.transfer_stream),
              "cudaEventRecord(empty_compute_done0)");
    CheckCuda(cudaEventRecord(cache.compute_done[1], cache.transfer_stream),
              "cudaEventRecord(empty_compute_done1)");
  }

  CheckCuda(cudaStreamWaitEvent(cache.transfer_stream, cache.compute_done[0], 0),
            "cudaStreamWaitEvent(compute_done0)");
  CheckCuda(cudaStreamWaitEvent(cache.transfer_stream, cache.compute_done[1], 0),
            "cudaStreamWaitEvent(compute_done1)");
  CheckCuda(cudaEventRecord(cache.packed_d2h_start, cache.transfer_stream),
            "cudaEventRecord(packed_d2h_start)");
  CheckCuda(cudaMemcpyAsync(cache.h_results, cache.d_results,
                            sizeof(DeviceLocalGainResult) * candidate_pairs.size(),
                            cudaMemcpyDeviceToHost, cache.transfer_stream),
            "cudaMemcpyAsync packed results");
  CheckCuda(cudaEventRecord(cache.packed_d2h_end, cache.transfer_stream),
            "cudaEventRecord(packed_d2h_end)");
  CheckCuda(cudaEventRecord(cache.result_ready, cache.transfer_stream),
            "cudaEventRecord(result_ready)");
  CheckCuda(cudaEventSynchronize(cache.result_ready), "cudaEventSynchronize(result_ready)");

  for (size_t i = 0; i < candidate_pairs.size(); ++i) {
    results[i].gain = cache.h_results[i].gain;
    results[i].before_cost = cache.h_results[i].before_cost;
  }

  float packed_h2d_ms = 0.0f;
  float packed_d2h_ms = 0.0f;
  CheckCuda(cudaEventElapsedTime(&packed_h2d_ms, cache.packed_h2d_start,
                                 cache.packed_h2d_end),
            "cudaEventElapsedTime(packed_h2d)");
  CheckCuda(cudaEventElapsedTime(&packed_d2h_ms, cache.packed_d2h_start,
                                 cache.packed_d2h_end),
            "cudaEventElapsedTime(packed_d2h)");
  stats_.cuda_packed_h2d_ms += static_cast<double>(packed_h2d_ms);
  stats_.cuda_packed_d2h_ms += static_cast<double>(packed_d2h_ms);
  stats_.cuda_h2d_ms += static_cast<double>(packed_h2d_ms);
  stats_.cuda_d2h_ms += static_cast<double>(packed_d2h_ms);
  stats_.cuda_packed_h2d_calls += 1;
  stats_.cuda_packed_d2h_calls += 1;
  stats_.cuda_packed_input_bytes += static_cast<int64_t>(layout.total_bytes);
  stats_.cuda_packed_output_bytes +=
      static_cast<int64_t>(sizeof(DeviceLocalGainResult) * candidate_pairs.size());
  stats_.cuda_h2d_bytes += static_cast<int64_t>(layout.total_bytes);
  stats_.cuda_d2h_bytes +=
      static_cast<int64_t>(sizeof(DeviceLocalGainResult) * candidate_pairs.size());
  stats_.cuda_row_uploads += 1;

  for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
    float kernel_ms = 0.0f;
    CheckCuda(cudaEventElapsedTime(&kernel_ms, cache.kernel_start_events[chunk_idx],
                                   cache.kernel_end_events[chunk_idx]),
              "cudaEventElapsedTime(kernel)");
    stats_.cuda_kernel_ms += static_cast<double>(kernel_ms);
  }

  VerifyCudaResultsIfEnabled(candidate_pairs, flat_agg, results, chunk_size,
                             verify_cuda_gain_,
                             &cuda_verify_samples_remaining_,
                             &cuda_verify_call_counter_);

  const auto cuda_total_end = Clock::now();
  stats_.cuda_total_ms += ElapsedMs(cuda_total_start, cuda_total_end);
  return results;
}
