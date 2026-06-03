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
struct WelmOeDecodeHashParams {
  int32_t num_samples;
  int32_t num_branches;
  uint32_t vocab_size;
  int32_t prefixes[MaxPrefixes];
  int32_t oe_grams[kMaxBranches];
  int32_t oe_vocab_sizes[kMaxBranches];
};

template <size_t MaxPrefixes>
__global__ void welm_oe_decode_hash_from_prefixes_kernel(
    const int64_t* __restrict__ input_ids,
    int64_t* __restrict__ hashed_out,
    int64_t hashed_out_branch_stride,
    WelmOeDecodeHashParams<MaxPrefixes> params) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= static_cast<size_t>(params.num_samples)) return;

  const uint32_t input = static_cast<uint32_t>(input_ids[idx]);
  const size_t num_samples = static_cast<size_t>(params.num_samples);
  for (int32_t branch_idx = 0; branch_idx < params.num_branches; ++branch_idx) {
    const int32_t gram = params.oe_grams[branch_idx];
    const uint32_t oe_vocab_size =
        static_cast<uint32_t>(params.oe_vocab_sizes[branch_idx]);
    uint32_t running_ids = input;
    uint32_t vocab_power = params.vocab_size;
    for (int32_t lag = 1; lag < gram; ++lag) {
      const size_t prefix_idx =
          static_cast<size_t>(lag - 1) * num_samples + idx;
      const uint32_t prev = static_cast<uint32_t>(params.prefixes[prefix_idx]);
      running_ids += prev * vocab_power;
      vocab_power *= params.vocab_size;
    }
    const uint32_t hashed = running_ids * 2654435761u;
    hashed_out[static_cast<size_t>(branch_idx) * hashed_out_branch_stride + idx] =
        static_cast<int64_t>(hashed % oe_vocab_size);
  }
}

template <size_t MaxPrefixes>
void launch_welm_oe_decode_hash_from_prefixes(
    const int64_t* input_ids,
    int64_t* hashed_out,
    int64_t hashed_out_branch_stride,
    const tvm::ffi::Shape& prefixes,
    const tvm::ffi::Shape& oe_grams,
    const tvm::ffi::Shape& oe_vocab_sizes,
    size_t num_samples,
    uint32_t vocab_size,
    DLDevice dl_device) {
  using namespace host;

  WelmOeDecodeHashParams<MaxPrefixes> params{};
  params.num_samples = static_cast<int32_t>(num_samples);
  params.num_branches = static_cast<int32_t>(oe_grams.size());
  params.vocab_size = vocab_size;
  for (size_t i = 0; i < prefixes.size(); ++i) {
    params.prefixes[i] = static_cast<int32_t>(prefixes[i]);
  }
  for (size_t i = 0; i < oe_grams.size(); ++i) {
    params.oe_grams[i] = static_cast<int32_t>(oe_grams[i]);
    params.oe_vocab_sizes[i] = static_cast<int32_t>(oe_vocab_sizes[i]);
  }

  const size_t grid_size = div_ceil(num_samples, kBlockSize);
  LaunchKernel(grid_size, kBlockSize, dl_device)(
      welm_oe_decode_hash_from_prefixes_kernel<MaxPrefixes>,
      input_ids,
      hashed_out,
      hashed_out_branch_stride,
      params);
}

struct WelmOeDecodeHashFromPrefixes {
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
    device.set_options<kDLCUDA, kDLROCM>();

    TensorMatcher({N}).with_dtype<int64_t>().with_device(device).verify(input_ids);
    TensorMatcher({B, N})
        .with_strides({SO, 1})
        .with_dtype<int64_t>()
        .with_device(device)
        .verify(hashed_out);

    const size_t num_tokens = N.unwrap();
    const size_t num_branches = B.unwrap();
    if (num_tokens == 0) return;

    RuntimeCheck(num_branches > 0, "num_branches must be positive");
    RuntimeCheck(
        oe_grams.size() == num_branches,
        "oe_grams size must match hashed_out branch count");
    RuntimeCheck(
        oe_vocab_sizes.size() == num_branches,
        "oe_vocab_sizes size must match hashed_out branch count");
    RuntimeCheck(
        prefixes.size() % num_tokens == 0,
        "prefix count must be divisible by num_tokens");
    const size_t history_width = prefixes.size() / num_tokens;
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

    const uint32_t vocab_size_u32 = static_cast<uint32_t>(vocab_size);
    const DLDevice dl_device = device.unwrap();

#define DISPATCH_WELM_OE_DECODE_HASH(MAX_PREFIXES)                       \
    if (prefixes.size() <= (MAX_PREFIXES)) {                              \
	      launch_welm_oe_decode_hash_from_prefixes<(MAX_PREFIXES)>(           \
	          static_cast<const int64_t*>(input_ids.data_ptr()),               \
	          static_cast<int64_t*>(hashed_out.data_ptr()),                    \
	          SO.unwrap(),                                                     \
	          prefixes, oe_grams, oe_vocab_sizes, num_tokens, vocab_size_u32,  \
	          dl_device);                                                      \
      return;                                                             \
    }

    DISPATCH_WELM_OE_DECODE_HASH(128);
    DISPATCH_WELM_OE_DECODE_HASH(256);
    DISPATCH_WELM_OE_DECODE_HASH(512);
    DISPATCH_WELM_OE_DECODE_HASH(1024);
    DISPATCH_WELM_OE_DECODE_HASH(2048);
    DISPATCH_WELM_OE_DECODE_HASH(4096);
#undef DISPATCH_WELM_OE_DECODE_HASH

    RuntimeCheck(false, "too many WeLM OE decode prefixes");
  }
};

}  // namespace
