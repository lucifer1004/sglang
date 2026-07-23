#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>

#include <cstdint>

namespace {

struct CopyTile {
  int32_t src_start;
  int32_t dst_start;
  int32_t row_count;
};

template <uint32_t kNumGPU>
struct SourcePushParams {
  void* peer_bases[kNumGPU];
  const void* key;
  const void* value;
  const CopyTile* tiles;
  uint32_t* completion;
  uint64_t k_offset_bytes;
  uint64_t v_offset_bytes;
  uint64_t signal_offset_bytes;
  uint64_t signal_stride_bytes;
  uint32_t destination_mask;
  uint32_t source_rank;
  uint64_t epoch;
  uint32_t num_tiles;
  bool publish_signal;
};

template <uint32_t kNumGPU>
struct IndexedSourcePushParams {
  void* peer_bases[kNumGPU];
  const void* key;
  const void* value;
  const int32_t* source_rows;
  const int32_t* destination_rows;
  uint32_t* completion;
  uint64_t k_offset_bytes;
  uint64_t v_offset_bytes;
  uint64_t signal_offset_bytes;
  uint64_t signal_stride_bytes;
  uint32_t destination_mask;
  uint32_t source_rank;
  uint64_t epoch;
  uint32_t num_rows;
  uint32_t rows_per_block;
  uint32_t num_blocks;
  bool publish_signal;
};

template <uint32_t kNumGPU>
struct PublishEpochParams {
  void* peer_bases[kNumGPU];
  uint64_t signal_offset_bytes;
  uint64_t signal_stride_bytes;
  uint32_t destination_mask;
  uint32_t publisher_rank;
  uint64_t epoch;
};

template <typename T>
SGL_DEVICE T* byte_offset(void* base, uint64_t offset) {
  return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(base) + offset);
}

template <typename T>
SGL_DEVICE const T* byte_offset(const void* base, uint64_t offset) {
  return reinterpret_cast<const T*>(reinterpret_cast<const uint8_t*>(base) + offset);
}

SGL_DEVICE uint4 load_16b(const void* address) {
  const auto* pointer = static_cast<const uint4*>(address);
  uint4 value;
  asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
               : "l"(pointer));
  return value;
}

SGL_DEVICE void store_volatile_16b(void* address, const uint4& value) {
  auto* pointer = static_cast<uint4*>(address);
  asm volatile("st.volatile.global.v4.b32 [%4], {%0, %1, %2, %3};"
               :
               : "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w), "l"(pointer)
               : "memory");
}

SGL_DEVICE void store_release_system(uint64_t* address, uint64_t value) {
  asm volatile("st.release.sys.global.u64 [%0], %1;" : : "l"(address), "l"(value) : "memory");
}

SGL_DEVICE uint64_t load_acquire_system(const uint64_t* address) {
  uint64_t value;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(value) : "l"(address) : "memory");
  return value;
}

template <typename DType, uint32_t kNumGPU, uint32_t kRowWidth>
__global__ void source_push_kernel(const SourcePushParams<kNumGPU> params) {
  constexpr uint32_t kVectorBytes = 16;
  constexpr uint32_t kRowBytes = kRowWidth * sizeof(DType);
  constexpr uint32_t kVectorsPerRow = kRowBytes / kVectorBytes;
  static_assert(kRowBytes % kVectorBytes == 0);

  const CopyTile tile = params.tiles[blockIdx.x];
  const uint32_t vector_count = static_cast<uint32_t>(tile.row_count) * kVectorsPerRow;
  for (uint32_t index = threadIdx.x; index < vector_count; index += blockDim.x) {
    const uint32_t row = index / kVectorsPerRow;
    const uint32_t vector = index % kVectorsPerRow;
    const uint64_t src_offset =
        (static_cast<uint64_t>(tile.src_start) + row) * kRowBytes + vector * kVectorBytes;
    const uint64_t dst_offset =
        (static_cast<uint64_t>(tile.dst_start) + row) * kRowBytes + vector * kVectorBytes;
    const uint4 key = load_16b(byte_offset<void>(params.key, src_offset));
    const uint4 value = load_16b(byte_offset<void>(params.value, src_offset));

#pragma unroll
    for (uint32_t destination = 0; destination < kNumGPU; ++destination) {
      if ((params.destination_mask & (1u << destination)) == 0) continue;
      store_volatile_16b(
          byte_offset<void>(
              params.peer_bases[destination], params.k_offset_bytes + dst_offset),
          key);
      store_volatile_16b(
          byte_offset<void>(
              params.peer_bases[destination], params.v_offset_bytes + dst_offset),
          value);
    }
  }

  // Every writing thread must make its own peer stores system-visible before
  // this block contributes to the completion counter.
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x != 0) return;

  const uint32_t previous = atomicInc(params.completion, params.num_tiles - 1);
  if (previous != params.num_tiles - 1 || !params.publish_signal) return;

#pragma unroll
  for (uint32_t destination = 0; destination < kNumGPU; ++destination) {
    if ((params.destination_mask & (1u << destination)) == 0) continue;
    auto* signal = byte_offset<uint64_t>(
        params.peer_bases[destination],
        params.signal_offset_bytes +
            static_cast<uint64_t>(params.source_rank) * params.signal_stride_bytes);
    store_release_system(signal, params.epoch);
  }
}

template <typename DType, uint32_t kNumGPU, uint32_t kRowWidth>
__global__ void indexed_source_push_kernel(
    const IndexedSourcePushParams<kNumGPU> params) {
  constexpr uint32_t kVectorBytes = 16;
  constexpr uint32_t kWarpSize = 32;
  constexpr uint32_t kRowBytes = kRowWidth * sizeof(DType);
  constexpr uint32_t kVectorsPerRow = kRowBytes / kVectorBytes;
  static_assert(kRowBytes % kVectorBytes == 0);
  static_assert(
      kVectorsPerRow == kWarpSize,
      "indexed source-push assigns one warp to one K/V row");

  const uint32_t warp = threadIdx.x / kWarpSize;
  const uint32_t lane = threadIdx.x % kWarpSize;
  const uint32_t warps_per_block = blockDim.x / kWarpSize;
  const uint32_t block_row_start = blockIdx.x * params.rows_per_block;

  for (uint32_t row_in_block = warp; row_in_block < params.rows_per_block;
       row_in_block += warps_per_block) {
    const uint32_t mapping_index = block_row_start + row_in_block;
    if (mapping_index >= params.num_rows) break;

    const int32_t source_row = params.source_rows[mapping_index];
    const int32_t destination_row = params.destination_rows[mapping_index];
    const uint64_t src_offset =
        static_cast<uint64_t>(source_row) * kRowBytes + lane * kVectorBytes;
    const uint64_t dst_offset =
        static_cast<uint64_t>(destination_row) * kRowBytes + lane * kVectorBytes;
    const uint4 key = load_16b(byte_offset<void>(params.key, src_offset));
    const uint4 value = load_16b(byte_offset<void>(params.value, src_offset));

#pragma unroll
    for (uint32_t destination = 0; destination < kNumGPU; ++destination) {
      if ((params.destination_mask & (1u << destination)) == 0) continue;
      store_volatile_16b(
          byte_offset<void>(
              params.peer_bases[destination], params.k_offset_bytes + dst_offset),
          key);
      store_volatile_16b(
          byte_offset<void>(
              params.peer_bases[destination], params.v_offset_bytes + dst_offset),
          value);
    }
  }

  __threadfence_system();
  __syncthreads();
  if (threadIdx.x != 0) return;

  const uint32_t previous = atomicInc(params.completion, params.num_blocks - 1);
  if (previous != params.num_blocks - 1 || !params.publish_signal) return;

#pragma unroll
  for (uint32_t destination = 0; destination < kNumGPU; ++destination) {
    if ((params.destination_mask & (1u << destination)) == 0) continue;
    auto* signal = byte_offset<uint64_t>(
        params.peer_bases[destination],
        params.signal_offset_bytes +
            static_cast<uint64_t>(params.source_rank) * params.signal_stride_bytes);
    store_release_system(signal, params.epoch);
  }
}

template <uint32_t kNumGPU>
__global__ void wait_ready_kernel(
    const void* local_arena_base,
    uint32_t source_mask,
    uint64_t signal_offset_bytes,
    uint64_t signal_stride_bytes,
    uint64_t epoch) {
  const uint32_t source = threadIdx.x;
  if (source >= kNumGPU || (source_mask & (1u << source)) == 0) return;
  const auto* signal = byte_offset<uint64_t>(
      local_arena_base, signal_offset_bytes + static_cast<uint64_t>(source) * signal_stride_bytes);
  while (load_acquire_system(signal) != epoch) {
    __nanosleep(64);
  }
}

template <uint32_t kNumGPU>
__global__ void publish_epoch_kernel(
    const PublishEpochParams<kNumGPU> params) {
  const uint32_t destination = threadIdx.x;
  if (destination >= kNumGPU ||
      (params.destination_mask & (1u << destination)) == 0) {
    return;
  }
  auto* signal = byte_offset<uint64_t>(
      params.peer_bases[destination],
      params.signal_offset_bytes +
          static_cast<uint64_t>(params.publisher_rank) * params.signal_stride_bytes);
  store_release_system(signal, params.epoch);
}

template <typename DType, uint32_t kNumGPU, uint32_t kRowWidth>
void prefill_cp_kv_source_push(
    const tvm::ffi::TensorView key,
    const tvm::ffi::TensorView value,
    const tvm::ffi::TensorView tiles,
    const tvm::ffi::TensorView peer_bases,
    int64_t destination_mask,
    int64_t k_offset_bytes,
    int64_t v_offset_bytes,
    int64_t signal_offset_bytes,
    int64_t signal_stride_bytes,
    const tvm::ffi::TensorView completion,
    int64_t source_rank,
    int64_t epoch,
    bool publish_signal,
    int64_t num_threads) {
  using namespace host;

  auto rows = SymbolicSize{"rows"};
  auto num_tiles = SymbolicSize{"num_tiles"};
  auto device_ = SymbolicDevice{};
  device_.set_options<kDLCUDA>();
  TensorMatcher({rows, kRowWidth})
      .with_dtype<DType>()
      .with_device(device_)
      .verify(key)
      .verify(value);
  TensorMatcher({num_tiles, 3}).with_dtype<int32_t>().with_device(device_).verify(tiles);
  TensorMatcher({kNumGPU}).with_dtype<int64_t>().with_device<kDLCPU>().verify(peer_bases);
  TensorMatcher({1}).with_dtype<int32_t>().with_device(device_).verify(completion);

  const auto num_tiles_i64 = num_tiles.unwrap();
  RuntimeCheck(num_tiles_i64 > 0, "num_tiles must be positive");
  RuntimeCheck(num_tiles_i64 <= UINT32_MAX, "num_tiles exceeds uint32 range");
  RuntimeCheck(
      destination_mask > 0 && destination_mask < (1ll << kNumGPU),
      "invalid destination mask");
  RuntimeCheck(source_rank >= 0 && source_rank < kNumGPU, "invalid source rank");
  RuntimeCheck(epoch > 0, "invalid epoch");
  RuntimeCheck(
      num_threads == 64 || num_threads == 128 || num_threads == 256 || num_threads == 512,
      "invalid thread count");
  RuntimeCheck(k_offset_bytes >= 0 && k_offset_bytes % 16 == 0, "invalid K offset");
  RuntimeCheck(v_offset_bytes >= 0 && v_offset_bytes % 16 == 0, "invalid V offset");
  RuntimeCheck(
      signal_offset_bytes >= 0 && signal_offset_bytes % sizeof(uint64_t) == 0,
      "invalid signal offset");
  RuntimeCheck(
      signal_stride_bytes >= sizeof(uint64_t) && signal_stride_bytes % sizeof(uint64_t) == 0,
      "invalid signal stride");

  SourcePushParams<kNumGPU> params{};
  const auto* bases = static_cast<const int64_t*>(peer_bases.data_ptr());
  for (uint32_t rank = 0; rank < kNumGPU; ++rank) {
    RuntimeCheck(bases[rank] > 0, "peer base must be non-null");
    params.peer_bases[rank] = reinterpret_cast<void*>(static_cast<uintptr_t>(bases[rank]));
  }
  params.key = key.data_ptr();
  params.value = value.data_ptr();
  params.tiles = static_cast<const CopyTile*>(tiles.data_ptr());
  params.completion = static_cast<uint32_t*>(completion.data_ptr());
  params.k_offset_bytes = static_cast<uint64_t>(k_offset_bytes);
  params.v_offset_bytes = static_cast<uint64_t>(v_offset_bytes);
  params.signal_offset_bytes = static_cast<uint64_t>(signal_offset_bytes);
  params.signal_stride_bytes = static_cast<uint64_t>(signal_stride_bytes);
  params.destination_mask = static_cast<uint32_t>(destination_mask);
  params.source_rank = static_cast<uint32_t>(source_rank);
  params.epoch = static_cast<uint64_t>(epoch);
  params.num_tiles = static_cast<uint32_t>(num_tiles_i64);
  params.publish_signal = publish_signal;

  LaunchKernel(params.num_tiles, static_cast<uint32_t>(num_threads), device_.unwrap())(
      source_push_kernel<DType, kNumGPU, kRowWidth>, params);
}

template <typename DType, uint32_t kNumGPU, uint32_t kRowWidth>
void prefill_cp_kv_source_push_indexed(
    const tvm::ffi::TensorView key,
    const tvm::ffi::TensorView value,
    const tvm::ffi::TensorView source_rows,
    const tvm::ffi::TensorView destination_rows,
    const tvm::ffi::TensorView peer_bases,
    int64_t destination_mask,
    int64_t k_offset_bytes,
    int64_t v_offset_bytes,
    int64_t signal_offset_bytes,
    int64_t signal_stride_bytes,
    const tvm::ffi::TensorView completion,
    int64_t source_rank,
    int64_t epoch,
    bool publish_signal,
    int64_t rows_per_block,
    int64_t num_threads) {
  using namespace host;

  auto rows = SymbolicSize{"rows"};
  auto source_capacity = SymbolicSize{"source_capacity"};
  auto device_ = SymbolicDevice{};
  device_.set_options<kDLCUDA>();
  TensorMatcher({source_capacity, kRowWidth})
      .with_dtype<DType>()
      .with_device(device_)
      .verify(key)
      .verify(value);
  TensorMatcher({rows}).with_dtype<int32_t>().with_device(device_).verify(source_rows);
  TensorMatcher({rows})
      .with_dtype<int32_t>()
      .with_device(device_)
      .verify(destination_rows);
  TensorMatcher({kNumGPU}).with_dtype<int64_t>().with_device<kDLCPU>().verify(peer_bases);
  TensorMatcher({1}).with_dtype<int32_t>().with_device(device_).verify(completion);

  const auto rows_i64 = rows.unwrap();
  RuntimeCheck(rows_i64 > 0, "indexed source-push rows must be positive");
  RuntimeCheck(rows_i64 <= UINT32_MAX, "indexed source-push rows exceed uint32 range");
  RuntimeCheck(
      destination_mask > 0 && destination_mask < (1ll << kNumGPU),
      "invalid destination mask");
  RuntimeCheck(source_rank >= 0 && source_rank < kNumGPU, "invalid source rank");
  RuntimeCheck(epoch > 0, "invalid epoch");
  RuntimeCheck(rows_per_block > 0 && rows_per_block <= UINT32_MAX, "invalid rows per block");
  RuntimeCheck(
      num_threads == 64 || num_threads == 128 || num_threads == 256 || num_threads == 512,
      "invalid thread count");
  RuntimeCheck(k_offset_bytes >= 0 && k_offset_bytes % 16 == 0, "invalid K offset");
  RuntimeCheck(v_offset_bytes >= 0 && v_offset_bytes % 16 == 0, "invalid V offset");
  RuntimeCheck(
      signal_offset_bytes >= 0 && signal_offset_bytes % sizeof(uint64_t) == 0,
      "invalid signal offset");
  RuntimeCheck(
      signal_stride_bytes >= sizeof(uint64_t) && signal_stride_bytes % sizeof(uint64_t) == 0,
      "invalid signal stride");

  IndexedSourcePushParams<kNumGPU> params{};
  const auto* bases = static_cast<const int64_t*>(peer_bases.data_ptr());
  for (uint32_t rank = 0; rank < kNumGPU; ++rank) {
    RuntimeCheck(bases[rank] > 0, "peer base must be non-null");
    params.peer_bases[rank] = reinterpret_cast<void*>(static_cast<uintptr_t>(bases[rank]));
  }
  params.key = key.data_ptr();
  params.value = value.data_ptr();
  params.source_rows = static_cast<const int32_t*>(source_rows.data_ptr());
  params.destination_rows = static_cast<const int32_t*>(destination_rows.data_ptr());
  params.completion = static_cast<uint32_t*>(completion.data_ptr());
  params.k_offset_bytes = static_cast<uint64_t>(k_offset_bytes);
  params.v_offset_bytes = static_cast<uint64_t>(v_offset_bytes);
  params.signal_offset_bytes = static_cast<uint64_t>(signal_offset_bytes);
  params.signal_stride_bytes = static_cast<uint64_t>(signal_stride_bytes);
  params.destination_mask = static_cast<uint32_t>(destination_mask);
  params.source_rank = static_cast<uint32_t>(source_rank);
  params.epoch = static_cast<uint64_t>(epoch);
  params.num_rows = static_cast<uint32_t>(rows_i64);
  params.rows_per_block = static_cast<uint32_t>(rows_per_block);
  params.num_blocks =
      (params.num_rows + params.rows_per_block - 1) / params.rows_per_block;
  params.publish_signal = publish_signal;

  LaunchKernel(params.num_blocks, static_cast<uint32_t>(num_threads), device_.unwrap())(
      indexed_source_push_kernel<DType, kNumGPU, kRowWidth>, params);
}

template <uint32_t kNumGPU>
void prefill_cp_kv_wait_ready(
    int64_t local_arena_base,
    int64_t source_mask,
    int64_t signal_offset_bytes,
    int64_t signal_stride_bytes,
    int64_t epoch,
    int64_t device_id) {
  using namespace host;
  RuntimeCheck(local_arena_base > 0 && local_arena_base % 16 == 0, "invalid local arena base");
  RuntimeCheck(source_mask > 0 && source_mask < (1ll << kNumGPU), "invalid source mask");
  RuntimeCheck(
      signal_offset_bytes >= 0 && signal_offset_bytes % sizeof(uint64_t) == 0,
      "invalid signal offset");
  RuntimeCheck(
      signal_stride_bytes >= sizeof(uint64_t) && signal_stride_bytes % sizeof(uint64_t) == 0,
      "invalid signal stride");
  RuntimeCheck(epoch > 0, "invalid epoch");
  RuntimeCheck(device_id >= 0, "invalid CUDA device id");

  const DLDevice device{kDLCUDA, static_cast<int32_t>(device_id)};
  LaunchKernel(1, 32, device)(
      wait_ready_kernel<kNumGPU>,
      reinterpret_cast<const void*>(static_cast<uintptr_t>(local_arena_base)),
      static_cast<uint32_t>(source_mask),
      static_cast<uint64_t>(signal_offset_bytes),
      static_cast<uint64_t>(signal_stride_bytes),
      static_cast<uint64_t>(epoch));
}

template <uint32_t kNumGPU>
void prefill_cp_kv_publish_epoch(
    const tvm::ffi::TensorView peer_bases,
    int64_t destination_mask,
    int64_t signal_offset_bytes,
    int64_t signal_stride_bytes,
    int64_t publisher_rank,
    int64_t epoch,
    int64_t device_id) {
  using namespace host;
  TensorMatcher({kNumGPU}).with_dtype<int64_t>().with_device<kDLCPU>().verify(peer_bases);
  RuntimeCheck(
      destination_mask > 0 && destination_mask < (1ll << kNumGPU),
      "invalid destination mask");
  RuntimeCheck(publisher_rank >= 0 && publisher_rank < kNumGPU, "invalid publisher rank");
  RuntimeCheck(epoch > 0, "invalid epoch");
  RuntimeCheck(
      signal_offset_bytes >= 0 && signal_offset_bytes % sizeof(uint64_t) == 0,
      "invalid signal offset");
  RuntimeCheck(
      signal_stride_bytes >= sizeof(uint64_t) && signal_stride_bytes % sizeof(uint64_t) == 0,
      "invalid signal stride");
  RuntimeCheck(device_id >= 0, "invalid CUDA device id");

  PublishEpochParams<kNumGPU> params{};
  const auto* raw_bases = static_cast<const int64_t*>(peer_bases.data_ptr());
  for (uint32_t rank = 0; rank < kNumGPU; ++rank) {
    RuntimeCheck(raw_bases[rank] > 0, "peer base must be non-null");
    params.peer_bases[rank] =
        reinterpret_cast<void*>(static_cast<uintptr_t>(raw_bases[rank]));
  }
  params.destination_mask = static_cast<uint32_t>(destination_mask);
  params.signal_offset_bytes = static_cast<uint64_t>(signal_offset_bytes);
  params.signal_stride_bytes = static_cast<uint64_t>(signal_stride_bytes);
  params.publisher_rank = static_cast<uint32_t>(publisher_rank);
  params.epoch = static_cast<uint64_t>(epoch);

  const DLDevice device{kDLCUDA, static_cast<int32_t>(device_id)};
  LaunchKernel(1, 32, device)(
      publish_epoch_kernel<kNumGPU>, params);
}

}  // namespace
