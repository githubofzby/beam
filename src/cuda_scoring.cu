#include "cuda_scoring.hpp"
#include "sweg.hpp"

#include <cuda_runtime.h>

#include <algorithm>
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

  int* d_offsets = nullptr;
  int* d_row_rep_ids = nullptr;
  int* d_neighbors = nullptr;
  int64_t* d_weights = nullptr;
  int64_t* d_neighbor_sizes = nullptr;
  int64_t* d_row_sizes = nullptr;
  int64_t* d_row_self_loops = nullptr;

  size_t offset_capacity = 0;
  size_t row_rep_capacity = 0;
  size_t neighbor_capacity = 0;
  size_t weight_capacity = 0;
  size_t neighbor_size_capacity = 0;
  size_t row_size_capacity = 0;
  size_t row_self_loop_capacity = 0;

  struct Slot {
    int2* d_pairs = nullptr;
    DeviceLocalGainResult* d_results = nullptr;
    int2* h_pairs = nullptr;
    DeviceLocalGainResult* h_results = nullptr;
    size_t capacity = 0;
    cudaStream_t stream = nullptr;
    cudaEvent_t h2d_start = nullptr;
    cudaEvent_t h2d_end = nullptr;
    cudaEvent_t kernel_end = nullptr;
    cudaEvent_t d2h_end = nullptr;
    bool in_flight = false;
    size_t result_offset = 0;
    size_t result_count = 0;
  };

  static constexpr int kSlotCount = 2;
  Slot slots[kSlotCount];

  CudaScoringCache() {
    try {
      for (Slot& slot : slots) {
        CheckCuda(
            cudaStreamCreateWithFlags(&slot.stream, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags");
        CheckCuda(cudaEventCreate(&slot.h2d_start), "cudaEventCreate(h2d)");
        CheckCuda(cudaEventCreate(&slot.h2d_end), "cudaEventCreate(h2d_end)");
        CheckCuda(cudaEventCreate(&slot.kernel_end),
                  "cudaEventCreate(kernel_end)");
        CheckCuda(cudaEventCreate(&slot.d2h_end), "cudaEventCreate(d2h_end)");
      }
    } catch (...) {
      for (Slot& slot : slots) {
        FreeDeviceNoThrow(slot.d_pairs);
        FreeDeviceNoThrow(slot.d_results);
        FreeHostNoThrow(slot.h_pairs);
        FreeHostNoThrow(slot.h_results);
        DestroyEventNoThrow(slot.h2d_start);
        DestroyEventNoThrow(slot.h2d_end);
        DestroyEventNoThrow(slot.kernel_end);
        DestroyEventNoThrow(slot.d2h_end);
        DestroyStreamNoThrow(slot.stream);
        slot = Slot{};
      }
      throw;
    }
  }

  ~CudaScoringCache() {
    for (Slot& slot : slots) {
      if (slot.in_flight && slot.d2h_end != nullptr) {
        cudaEventSynchronize(slot.d2h_end);
      }
      FreeDeviceNoThrow(slot.d_pairs);
      FreeDeviceNoThrow(slot.d_results);
      FreeHostNoThrow(slot.h_pairs);
      FreeHostNoThrow(slot.h_results);
      DestroyEventNoThrow(slot.h2d_start);
      DestroyEventNoThrow(slot.h2d_end);
      DestroyEventNoThrow(slot.kernel_end);
      DestroyEventNoThrow(slot.d2h_end);
      DestroyStreamNoThrow(slot.stream);
      slot = Slot{};
    }

    FreeDeviceNoThrow(d_offsets);
    FreeDeviceNoThrow(d_row_rep_ids);
    FreeDeviceNoThrow(d_neighbors);
    FreeDeviceNoThrow(d_weights);
    FreeDeviceNoThrow(d_neighbor_sizes);
    FreeDeviceNoThrow(d_row_sizes);
    FreeDeviceNoThrow(d_row_self_loops);
  }

  void EnsureSlotCapacity(Slot* slot, size_t required) {
    if (slot == nullptr || slot->capacity >= required) {
      return;
    }

    if (slot->in_flight) {
      CheckCuda(cudaEventSynchronize(slot->d2h_end), "wait slot before resize");
      slot->in_flight = false;
    }

    const size_t new_capacity = GrowCapacity(slot->capacity, required);

    FreeDeviceNoThrow(slot->d_pairs);
    FreeDeviceNoThrow(slot->d_results);
    FreeHostNoThrow(slot->h_pairs);
    FreeHostNoThrow(slot->h_results);
    slot->d_pairs = nullptr;
    slot->d_results = nullptr;
    slot->h_pairs = nullptr;
    slot->h_results = nullptr;

    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&slot->d_pairs),
                         sizeof(int2) * new_capacity),
              "cudaMalloc slot d_pairs");
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&slot->d_results),
                         sizeof(DeviceLocalGainResult) * new_capacity),
              "cudaMalloc slot d_results");
    CheckCuda(cudaHostAlloc(reinterpret_cast<void**>(&slot->h_pairs),
                            sizeof(int2) * new_capacity, cudaHostAllocDefault),
              "cudaHostAlloc slot h_pairs");
    CheckCuda(cudaHostAlloc(reinterpret_cast<void**>(&slot->h_results),
                            sizeof(DeviceLocalGainResult) * new_capacity,
                            cudaHostAllocDefault),
              "cudaHostAlloc slot h_results");

    slot->capacity = new_capacity;
  }

  void AccumulateSlotTiming(Slot& slot, RuntimeStats* stats) const {
    if (stats == nullptr || slot.result_count == 0) {
      return;
    }

    float h2d_ms = 0.0f;
    float kernel_ms = 0.0f;
    float d2h_ms = 0.0f;
    CheckCuda(cudaEventElapsedTime(&h2d_ms, slot.h2d_start, slot.h2d_end),
              "cudaEventElapsedTime(h2d)");
    CheckCuda(cudaEventElapsedTime(&kernel_ms, slot.h2d_end, slot.kernel_end),
              "cudaEventElapsedTime(kernel)");
    CheckCuda(cudaEventElapsedTime(&d2h_ms, slot.kernel_end, slot.d2h_end),
              "cudaEventElapsedTime(d2h)");
    stats->cuda_pair_h2d_ms += static_cast<double>(h2d_ms);
    stats->cuda_h2d_ms += static_cast<double>(h2d_ms);
    stats->cuda_kernel_ms += static_cast<double>(kernel_ms);
    stats->cuda_d2h_ms += static_cast<double>(d2h_ms);
  }

  void UploadResidentRows(const Sweg::FlatAggCSR& flat,
                          RuntimeStats* stats) {
    EnsureDeviceCapacity(&d_offsets, &offset_capacity, flat.offsets.size(),
                         "cudaMalloc d_offsets");
    EnsureDeviceCapacity(&d_row_rep_ids, &row_rep_capacity,
                         flat.row_rep_ids.size(), "cudaMalloc d_row_rep_ids");
    EnsureDeviceCapacity(&d_neighbors, &neighbor_capacity,
                         flat.neighbors.size(), "cudaMalloc d_neighbors");
    EnsureDeviceCapacity(&d_weights, &weight_capacity, flat.weights.size(),
                         "cudaMalloc d_weights");
    EnsureDeviceCapacity(&d_neighbor_sizes, &neighbor_size_capacity,
                         flat.neighbor_sizes.size(),
                         "cudaMalloc d_neighbor_sizes");
    EnsureDeviceCapacity(&d_row_sizes, &row_size_capacity,
                         flat.row_sizes.size(), "cudaMalloc d_row_sizes");
    EnsureDeviceCapacity(&d_row_self_loops, &row_self_loop_capacity,
                         flat.row_self_loops.size(),
                         "cudaMalloc d_row_self_loops");

    const auto upload_start = Clock::now();
    CheckCuda(cudaMemcpy(d_offsets, flat.offsets.data(),
                         sizeof(int) * flat.offsets.size(),
                         cudaMemcpyHostToDevice),
              "upload offsets");
    CheckCuda(cudaMemcpy(d_row_rep_ids, flat.row_rep_ids.data(),
                         sizeof(int) * flat.row_rep_ids.size(),
                         cudaMemcpyHostToDevice),
              "upload row_rep_ids");
    CheckCuda(cudaMemcpy(d_neighbors, flat.neighbors.data(),
                         sizeof(int) * flat.neighbors.size(),
                         cudaMemcpyHostToDevice),
              "upload neighbors");
    CheckCuda(cudaMemcpy(d_weights, flat.weights.data(),
                         sizeof(int64_t) * flat.weights.size(),
                         cudaMemcpyHostToDevice),
              "upload weights");
    CheckCuda(cudaMemcpy(d_neighbor_sizes, flat.neighbor_sizes.data(),
                         sizeof(int64_t) * flat.neighbor_sizes.size(),
                         cudaMemcpyHostToDevice),
              "upload neighbor_sizes");
    CheckCuda(cudaMemcpy(d_row_sizes, flat.row_sizes.data(),
                         sizeof(int64_t) * flat.row_sizes.size(),
                         cudaMemcpyHostToDevice),
              "upload row_sizes");
    CheckCuda(cudaMemcpy(d_row_self_loops, flat.row_self_loops.data(),
                         sizeof(int64_t) * flat.row_self_loops.size(),
                         cudaMemcpyHostToDevice),
              "upload row_self_loops");
    const auto upload_end = Clock::now();

    if (stats != nullptr) {
      const uint64_t row_bytes =
          static_cast<uint64_t>(sizeof(int) * flat.offsets.size()) +
          static_cast<uint64_t>(sizeof(int) * flat.row_rep_ids.size()) +
          static_cast<uint64_t>(sizeof(int) * flat.neighbors.size()) +
          static_cast<uint64_t>(sizeof(int64_t) * flat.weights.size()) +
          static_cast<uint64_t>(sizeof(int64_t) * flat.neighbor_sizes.size()) +
          static_cast<uint64_t>(sizeof(int64_t) * flat.row_sizes.size()) +
          static_cast<uint64_t>(sizeof(int64_t) *
                                flat.row_self_loops.size());
      const double elapsed = ElapsedMs(upload_start, upload_end);
      stats->cuda_row_h2d_ms += elapsed;
      stats->cuda_h2d_ms += elapsed;
      stats->cuda_row_uploads += 1;
      stats->cuda_h2d_bytes += static_cast<int64_t>(row_bytes);
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

  cache.UploadResidentRows(flat_agg, &stats_);

  std::vector<LocalGainResult> results(candidate_pairs.size());
  const size_t chunk_size =
      candidate_chunk_size == 0
          ? candidate_pairs.size()
          : std::max<size_t>(1, std::min(candidate_chunk_size,
                                         candidate_pairs.size()));

  auto collect_slot = [&](CudaScoringCache::Slot& slot) {
    if (!slot.in_flight) {
      return;
    }

    CheckCuda(cudaEventSynchronize(slot.d2h_end), "wait scoring slot");

    for (size_t i = 0; i < slot.result_count; ++i) {
      const DeviceLocalGainResult& src = slot.h_results[i];
      LocalGainResult& dst = results[slot.result_offset + i];
      dst.gain = src.gain;
      dst.before_cost = src.before_cost;
    }

    cache.AccumulateSlotTiming(slot, &stats_);
    slot.in_flight = false;
  };

  size_t offset = 0;
  size_t launch_index = 0;
  while (offset < candidate_pairs.size()) {
    CudaScoringCache::Slot& slot =
        cache.slots[launch_index % CudaScoringCache::kSlotCount];
    collect_slot(slot);

    const size_t count =
        std::min(chunk_size, candidate_pairs.size() - offset);
    cache.EnsureSlotCapacity(&slot, count);

    for (size_t i = 0; i < count; ++i) {
      const auto& pair = candidate_pairs[offset + i];
      slot.h_pairs[i] = make_int2(pair.first, pair.second);
    }

    slot.result_offset = offset;
    slot.result_count = count;

    CheckCuda(cudaEventRecord(slot.h2d_start, slot.stream),
              "cudaEventRecord(h2d_start)");
    CheckCuda(cudaMemcpyAsync(slot.d_pairs, slot.h_pairs, sizeof(int2) * count,
                              cudaMemcpyHostToDevice, slot.stream),
              "async upload candidate pairs");
    CheckCuda(cudaEventRecord(slot.h2d_end, slot.stream),
              "cudaEventRecord(h2d_end)");

    constexpr int kBlockSize = 256;
    const int blocks =
        static_cast<int>((count + kBlockSize - 1) / kBlockSize);
    ComputeLocalGainKernel<<<blocks, kBlockSize, 0, slot.stream>>>(
        slot.d_pairs, static_cast<int>(count), cache.d_offsets,
        cache.d_row_rep_ids, cache.d_neighbors, cache.d_weights,
        cache.d_neighbor_sizes, cache.d_row_sizes, cache.d_row_self_loops,
        slot.d_results);
    CheckCuda(cudaGetLastError(), "ComputeLocalGainKernel launch");
    CheckCuda(cudaEventRecord(slot.kernel_end, slot.stream),
              "cudaEventRecord(kernel_end)");

    CheckCuda(cudaMemcpyAsync(slot.h_results, slot.d_results,
                              sizeof(DeviceLocalGainResult) * count,
                              cudaMemcpyDeviceToHost, slot.stream),
              "async download scoring results");
    CheckCuda(cudaEventRecord(slot.d2h_end, slot.stream),
              "cudaEventRecord(d2h_end)");

    slot.in_flight = true;
    stats_.cuda_kernel_launches += 1;
    stats_.cuda_h2d_bytes +=
        static_cast<int64_t>(sizeof(int2) * count);
    stats_.cuda_d2h_bytes +=
        static_cast<int64_t>(sizeof(DeviceLocalGainResult) * count);
    stats_.cuda_max_candidates_per_launch =
        std::max<int64_t>(stats_.cuda_max_candidates_per_launch,
                          static_cast<int64_t>(count));

    offset += count;
    ++launch_index;
  }

  for (CudaScoringCache::Slot& slot : cache.slots) {
    collect_slot(slot);
  }

  VerifyCudaResultsIfEnabled(candidate_pairs, flat_agg, results, chunk_size,
                             verify_cuda_gain_,
                             &cuda_verify_samples_remaining_,
                             &cuda_verify_call_counter_);

  const auto cuda_total_end = Clock::now();
  stats_.cuda_total_ms += ElapsedMs(cuda_total_start, cuda_total_end);
  return results;
}
