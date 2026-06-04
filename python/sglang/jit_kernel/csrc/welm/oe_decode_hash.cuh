#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstddef>
#include <cstdint>

namespace {

constexpr size_t kBlockSize = 256;
constexpr size_t kMaxBranches = 8;

template <size_t MaxPrefixes>
struct WelmOeHashParams {
  int32_t num_segments;
  int32_t num_branches;
  uint32_t vocab_size;
  int32_t prefixes[MaxPrefixes];
  int32_t oe_grams[kMaxBranches];
  int32_t oe_vocab_sizes[kMaxBranches];
};

template <typename InputIdT, size_t MaxPrefixes>
__global__ void welm_oe_hash_decode_from_prefixes_kernel(
    const InputIdT* __restrict__ input_ids,
    int64_t* __restrict__ hashed_out,
    int64_t hashed_out_branch_stride,
    WelmOeHashParams<MaxPrefixes> params) {
  const size_t token_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (token_idx >= static_cast<size_t>(params.num_segments)) return;

  const uint32_t input = static_cast<uint32_t>(input_ids[token_idx]);
  const size_t num_segments = static_cast<size_t>(params.num_segments);
  for (int32_t branch_idx = 0; branch_idx < params.num_branches; ++branch_idx) {
    const int32_t gram = params.oe_grams[branch_idx];
    const uint32_t oe_vocab_size =
        static_cast<uint32_t>(params.oe_vocab_sizes[branch_idx]);
    uint32_t running_ids = input;
    uint32_t vocab_power = params.vocab_size;
    for (int32_t lag = 1; lag < gram; ++lag) {
      const size_t prefix_idx =
          static_cast<size_t>(lag - 1) * num_segments + token_idx;
      const uint32_t prev = static_cast<uint32_t>(params.prefixes[prefix_idx]);
      running_ids += prev * vocab_power;
      vocab_power *= params.vocab_size;
    }
    const uint32_t hashed = running_ids * 2654435761u;
    hashed_out[static_cast<size_t>(branch_idx) * hashed_out_branch_stride +
               token_idx] = static_cast<int64_t>(hashed % oe_vocab_size);
  }
}

template <typename InputIdT, size_t MaxPrefixes>
__global__ void welm_oe_hash_segments_from_prefixes_kernel(
    const InputIdT* __restrict__ input_ids,
    const int32_t* __restrict__ extend_start_loc,
    const int32_t* __restrict__ extend_seq_lens,
    int64_t* __restrict__ hashed_out,
    int64_t hashed_out_branch_stride,
    WelmOeHashParams<MaxPrefixes> params) {
  const int32_t segment_idx = static_cast<int32_t>(blockIdx.x);
  if (segment_idx >= params.num_segments) return;

  const int32_t segment_start = extend_start_loc[segment_idx];
  const int32_t segment_len = extend_seq_lens[segment_idx];
  const size_t num_segments = static_cast<size_t>(params.num_segments);

  for (int32_t local_pos = static_cast<int32_t>(threadIdx.x);
       local_pos < segment_len;
       local_pos += static_cast<int32_t>(blockDim.x)) {
    const int32_t token_idx = segment_start + local_pos;
    const uint32_t input = static_cast<uint32_t>(input_ids[token_idx]);

    for (int32_t branch_idx = 0; branch_idx < params.num_branches; ++branch_idx) {
      const int32_t gram = params.oe_grams[branch_idx];
      const uint32_t oe_vocab_size =
          static_cast<uint32_t>(params.oe_vocab_sizes[branch_idx]);
      uint32_t running_ids = input;
      uint32_t vocab_power = params.vocab_size;

      for (int32_t lag = 1; lag < gram; ++lag) {
        uint32_t prev = 0;
        if (local_pos >= lag) {
          prev = static_cast<uint32_t>(input_ids[token_idx - lag]);
        } else {
          const int32_t prefix_lag = lag - local_pos - 1;
          const size_t prefix_idx =
              static_cast<size_t>(prefix_lag) * num_segments + segment_idx;
          prev = static_cast<uint32_t>(params.prefixes[prefix_idx]);
        }
        running_ids += prev * vocab_power;
        vocab_power *= params.vocab_size;
      }

      const uint32_t hashed = running_ids * 2654435761u;
      hashed_out[static_cast<size_t>(branch_idx) * hashed_out_branch_stride +
                 token_idx] = static_cast<int64_t>(hashed % oe_vocab_size);
    }
  }
}

template <size_t MaxPrefixes>
WelmOeHashParams<MaxPrefixes> make_params(
    const tvm::ffi::Shape& prefixes,
    const tvm::ffi::Shape& oe_grams,
    const tvm::ffi::Shape& oe_vocab_sizes,
    size_t num_segments,
    uint32_t vocab_size) {
  WelmOeHashParams<MaxPrefixes> params{};
  params.num_segments = static_cast<int32_t>(num_segments);
  params.num_branches = static_cast<int32_t>(oe_grams.size());
  params.vocab_size = vocab_size;
  for (size_t i = 0; i < prefixes.size(); ++i) {
    params.prefixes[i] = static_cast<int32_t>(prefixes[i]);
  }
  for (size_t i = 0; i < oe_grams.size(); ++i) {
    params.oe_grams[i] = static_cast<int32_t>(oe_grams[i]);
    params.oe_vocab_sizes[i] = static_cast<int32_t>(oe_vocab_sizes[i]);
  }
  return params;
}

template <typename InputIdT, size_t MaxPrefixes>
void launch_welm_oe_hash_decode_from_prefixes(
    const InputIdT* input_ids,
    int64_t* hashed_out,
    int64_t hashed_out_branch_stride,
    const tvm::ffi::Shape& prefixes,
    const tvm::ffi::Shape& oe_grams,
    const tvm::ffi::Shape& oe_vocab_sizes,
    size_t num_tokens,
    uint32_t vocab_size,
    DLDevice dl_device) {
  using namespace host;

  auto params = make_params<MaxPrefixes>(
      prefixes, oe_grams, oe_vocab_sizes, num_tokens, vocab_size);
  const size_t grid_size = div_ceil(num_tokens, kBlockSize);
  LaunchKernel(grid_size, kBlockSize, dl_device)(
      welm_oe_hash_decode_from_prefixes_kernel<InputIdT, MaxPrefixes>,
      input_ids,
      hashed_out,
      hashed_out_branch_stride,
      params);
}

template <typename InputIdT, size_t MaxPrefixes>
void launch_welm_oe_hash_segments_from_prefixes(
    const InputIdT* input_ids,
    const int32_t* extend_start_loc,
    const int32_t* extend_seq_lens,
    int64_t* hashed_out,
    int64_t hashed_out_branch_stride,
    const tvm::ffi::Shape& prefixes,
    const tvm::ffi::Shape& oe_grams,
    const tvm::ffi::Shape& oe_vocab_sizes,
    size_t num_segments,
    uint32_t vocab_size,
    DLDevice dl_device) {
  using namespace host;

  auto params = make_params<MaxPrefixes>(
      prefixes, oe_grams, oe_vocab_sizes, num_segments, vocab_size);
  LaunchKernel(num_segments, kBlockSize, dl_device)(
      welm_oe_hash_segments_from_prefixes_kernel<InputIdT, MaxPrefixes>,
      input_ids,
      extend_start_loc,
      extend_seq_lens,
      hashed_out,
      hashed_out_branch_stride,
      params);
}

void check_hash_inputs(
    const tvm::ffi::Shape& prefixes,
    const tvm::ffi::Shape& oe_grams,
    const tvm::ffi::Shape& oe_vocab_sizes,
    size_t num_segments,
    size_t num_branches,
    int64_t vocab_size) {
  using namespace host;

  RuntimeCheck(num_segments > 0, "num_segments must be positive");
  RuntimeCheck(num_branches > 0, "num_branches must be positive");
  RuntimeCheck(
      oe_grams.size() == num_branches,
      "oe_grams size must match hashed_out branch count");
  RuntimeCheck(
      oe_vocab_sizes.size() == num_branches,
      "oe_vocab_sizes size must match hashed_out branch count");
  RuntimeCheck(
      prefixes.size() % num_segments == 0,
      "prefix count must be divisible by num_segments");
  const size_t history_width = prefixes.size() / num_segments;
  RuntimeCheck(history_width > 0, "history_width must be positive");
  RuntimeCheck(vocab_size > 0, "vocab_size must be positive");
  RuntimeCheck(num_branches <= kMaxBranches, "num_branches exceeds max branches");

  for (size_t i = 0; i < num_branches; ++i) {
    const int64_t gram = oe_grams[i];
    RuntimeCheck(gram >= 2, "oe gram must be at least 2");
    RuntimeCheck(
        static_cast<size_t>(gram - 1) <= history_width,
        "history_width is too small for oe gram");
    RuntimeCheck(oe_vocab_sizes[i] > 0, "oe vocab size must be positive");
  }
}

struct WelmOeHashDecodeFromPrefixes {
  static void run(
      tvm::ffi::TensorView input_ids,
      tvm::ffi::Shape prefixes,
      tvm::ffi::Shape oe_grams,
      tvm::ffi::Shape oe_vocab_sizes,
      tvm::ffi::TensorView hashed_out,
      int64_t vocab_size) {
    using namespace host;

    SymbolicSize N = {"num_tokens"};
    SymbolicSize B = {"num_branches"};
    SymbolicSize SO = {"hashed_out_branch_stride"};
    SymbolicDevice device;
    SymbolicDType input_dtype;
    device.set_options<kDLCUDA, kDLROCM>();

    TensorMatcher({N})
        .with_dtype<int32_t, int64_t>(input_dtype)
        .with_device(device)
        .verify(input_ids);
    TensorMatcher({B, N})
        .with_strides({SO, 1})
        .with_dtype<int64_t>()
        .with_device(device)
        .verify(hashed_out);

    const size_t num_tokens = N.unwrap();
    const size_t num_branches = B.unwrap();
    if (num_tokens == 0) return;

    check_hash_inputs(
        prefixes, oe_grams, oe_vocab_sizes, num_tokens, num_branches, vocab_size);

    const uint32_t vocab_size_u32 = static_cast<uint32_t>(vocab_size);
    const DLDevice dl_device = device.unwrap();
    const bool input_is_int32 = input_dtype.unwrap().bits == 32;

#define DISPATCH_WELM_OE_HASH(MAX_PREFIXES)                                   \
    if (prefixes.size() <= (MAX_PREFIXES)) {                                  \
      if (input_is_int32) {                                                    \
        launch_welm_oe_hash_decode_from_prefixes<int32_t, (MAX_PREFIXES)>(     \
            static_cast<const int32_t*>(input_ids.data_ptr()),                 \
            static_cast<int64_t*>(hashed_out.data_ptr()),                      \
            SO.unwrap(),                                                       \
            prefixes,                                                          \
            oe_grams,                                                          \
            oe_vocab_sizes,                                                    \
            num_tokens,                                                        \
            vocab_size_u32,                                                    \
            dl_device);                                                        \
      } else {                                                                 \
        launch_welm_oe_hash_decode_from_prefixes<int64_t, (MAX_PREFIXES)>(     \
            static_cast<const int64_t*>(input_ids.data_ptr()),                 \
            static_cast<int64_t*>(hashed_out.data_ptr()),                      \
            SO.unwrap(),                                                       \
            prefixes,                                                          \
            oe_grams,                                                          \
            oe_vocab_sizes,                                                    \
            num_tokens,                                                        \
            vocab_size_u32,                                                    \
            dl_device);                                                        \
      }                                                                        \
      return;                                                                  \
    }

    DISPATCH_WELM_OE_HASH(128);
    DISPATCH_WELM_OE_HASH(256);
    DISPATCH_WELM_OE_HASH(512);
    DISPATCH_WELM_OE_HASH(1024);
    DISPATCH_WELM_OE_HASH(2048);
    DISPATCH_WELM_OE_HASH(4096);
#undef DISPATCH_WELM_OE_HASH

    RuntimeCheck(false, "too many WeLM OE hash prefixes");
  }
};

struct WelmOeHashSegmentsFromPrefixes {
  static void run(
      tvm::ffi::TensorView input_ids,
      tvm::ffi::TensorView extend_start_loc,
      tvm::ffi::TensorView extend_seq_lens,
      tvm::ffi::Shape prefixes,
      tvm::ffi::Shape oe_grams,
      tvm::ffi::Shape oe_vocab_sizes,
      tvm::ffi::TensorView hashed_out,
      int64_t vocab_size) {
    using namespace host;

    SymbolicSize N = {"num_tokens"};
    SymbolicSize S = {"num_segments"};
    SymbolicSize B = {"num_branches"};
    SymbolicSize SO = {"hashed_out_branch_stride"};
    SymbolicDevice device;
    SymbolicDType input_dtype;
    device.set_options<kDLCUDA, kDLROCM>();

    TensorMatcher({N})
        .with_dtype<int32_t, int64_t>(input_dtype)
        .with_device(device)
        .verify(input_ids);
    TensorMatcher({S}).with_dtype<int32_t>().with_device(device).verify(extend_start_loc);
    TensorMatcher({S}).with_dtype<int32_t>().with_device(device).verify(extend_seq_lens);
    TensorMatcher({B, N})
        .with_strides({SO, 1})
        .with_dtype<int64_t>()
        .with_device(device)
        .verify(hashed_out);

    const size_t num_tokens = N.unwrap();
    const size_t num_segments = S.unwrap();
    const size_t num_branches = B.unwrap();
    if (num_tokens == 0) return;

    check_hash_inputs(
        prefixes, oe_grams, oe_vocab_sizes, num_segments, num_branches, vocab_size);

    const uint32_t vocab_size_u32 = static_cast<uint32_t>(vocab_size);
    const DLDevice dl_device = device.unwrap();
    const bool input_is_int32 = input_dtype.unwrap().bits == 32;

#define DISPATCH_WELM_OE_HASH(MAX_PREFIXES)                                   \
    if (prefixes.size() <= (MAX_PREFIXES)) {                                  \
      if (input_is_int32) {                                                    \
        launch_welm_oe_hash_segments_from_prefixes<int32_t, (MAX_PREFIXES)>(   \
            static_cast<const int32_t*>(input_ids.data_ptr()),                 \
            static_cast<const int32_t*>(extend_start_loc.data_ptr()),          \
            static_cast<const int32_t*>(extend_seq_lens.data_ptr()),           \
            static_cast<int64_t*>(hashed_out.data_ptr()),                      \
            SO.unwrap(),                                                       \
            prefixes,                                                          \
            oe_grams,                                                          \
            oe_vocab_sizes,                                                    \
            num_segments,                                                      \
            vocab_size_u32,                                                    \
            dl_device);                                                        \
      } else {                                                                 \
        launch_welm_oe_hash_segments_from_prefixes<int64_t, (MAX_PREFIXES)>(   \
            static_cast<const int64_t*>(input_ids.data_ptr()),                 \
            static_cast<const int32_t*>(extend_start_loc.data_ptr()),          \
            static_cast<const int32_t*>(extend_seq_lens.data_ptr()),           \
            static_cast<int64_t*>(hashed_out.data_ptr()),                      \
            SO.unwrap(),                                                       \
            prefixes,                                                          \
            oe_grams,                                                          \
            oe_vocab_sizes,                                                    \
            num_segments,                                                      \
            vocab_size_u32,                                                    \
            dl_device);                                                        \
      }                                                                        \
      return;                                                                  \
    }

    DISPATCH_WELM_OE_HASH(128);
    DISPATCH_WELM_OE_HASH(256);
    DISPATCH_WELM_OE_HASH(512);
    DISPATCH_WELM_OE_HASH(1024);
    DISPATCH_WELM_OE_HASH(2048);
    DISPATCH_WELM_OE_HASH(4096);
#undef DISPATCH_WELM_OE_HASH

    RuntimeCheck(false, "too many WeLM OE hash prefixes");
  }
};

}  // namespace
