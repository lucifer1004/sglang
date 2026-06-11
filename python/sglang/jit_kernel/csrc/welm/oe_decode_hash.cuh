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

struct WelmOeHashRuntimeParams {
  int32_t num_segments;
  int32_t num_branches;
  int32_t history_width;
  uint32_t vocab_size;
  int32_t oe_grams[kMaxBranches];
  int32_t oe_vocab_sizes[kMaxBranches];
};

template <typename InputIdT, size_t MaxPrefixes>
__global__ void welm_oe_hash_mtp_init_history_from_prefixes_kernel(
    const InputIdT* __restrict__ first_token_ids,
    int64_t* __restrict__ history_out,
    int32_t has_first_token,
    WelmOeHashParams<MaxPrefixes> params) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t history_width = static_cast<size_t>(params.num_branches);
  const size_t num_segments = static_cast<size_t>(params.num_segments);
  if (idx >= num_segments * history_width) return;

  const size_t segment_idx = idx / history_width;
  const size_t history_col = idx % history_width;
  if (has_first_token && history_col + 1 == history_width) {
    history_out[idx] = static_cast<int64_t>(first_token_ids[segment_idx]);
    return;
  }

  const size_t lag = history_width - history_col - (has_first_token ? 1 : 0);
  const size_t prefix_idx = (lag - 1) * num_segments + segment_idx;
  history_out[idx] = static_cast<int64_t>(params.prefixes[prefix_idx]);
}

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

__device__ uint32_t get_history_state_token(
    const int64_t* __restrict__ history_state,
    int32_t row,
    int32_t history_width,
    int32_t lag) {
  const int32_t col = history_width - lag;
  if (col < 0 || col >= history_width) return 0;
  return static_cast<uint32_t>(
      history_state[static_cast<size_t>(row) * history_width + col]);
}

__device__ uint32_t get_history_state_token_before_current(
    const int64_t* __restrict__ history_state,
    int32_t row,
    int32_t history_width,
    int32_t lag) {
  const int32_t col = history_width - lag - 1;
  if (col < 0 || col >= history_width) return 0;
  return static_cast<uint32_t>(
      history_state[static_cast<size_t>(row) * history_width + col]);
}

template <typename InputIdT, typename SeqLenT>
__device__ uint32_t get_target_verify_history_token_from_history(
    const InputIdT* __restrict__ draft_token_ids,
    const bool* __restrict__ tree_mask,
    const SeqLenT* __restrict__ seq_lens,
    const int64_t* __restrict__ history_state,
    int32_t seq_id,
    int32_t local_token_idx,
    int32_t draft_token_num,
    int32_t history_width,
    int32_t history_rank) {
  int64_t mask_offset = 0;
  for (int32_t i = 0; i < seq_id; ++i) {
    const int64_t mask_len =
        static_cast<int64_t>(seq_lens[i]) + draft_token_num;
    mask_offset += static_cast<int64_t>(draft_token_num) * mask_len;
  }

  const int64_t seq_len = static_cast<int64_t>(seq_lens[seq_id]);
  const int64_t mask_len = seq_len + draft_token_num;
  mask_offset += static_cast<int64_t>(local_token_idx) * mask_len;

  int32_t target_gram = history_rank + 1;
  for (int64_t pos = seq_len + draft_token_num - 1; pos >= seq_len; --pos) {
    if (tree_mask[mask_offset + pos]) {
      --target_gram;
      if (target_gram == 0) {
        const int64_t draft_idx =
            static_cast<int64_t>(seq_id) * draft_token_num + pos - seq_len;
        return static_cast<uint32_t>(draft_token_ids[draft_idx]);
      }
    }
  }

  return get_history_state_token_before_current(
      history_state, seq_id, history_width, target_gram);
}

template <typename InputIdT, typename SeqLenT>
__global__ void welm_oe_hash_mtp_target_verify_from_history_kernel(
    const InputIdT* __restrict__ draft_token_ids,
    const bool* __restrict__ tree_mask,
    const SeqLenT* __restrict__ seq_lens,
    const int64_t* __restrict__ history_state,
    int64_t* __restrict__ hashed_out,
    int64_t hashed_out_branch_stride,
    int32_t draft_token_num,
    WelmOeHashRuntimeParams params) {
  const size_t token_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t num_tokens =
      static_cast<size_t>(params.num_segments) * draft_token_num;
  if (token_idx >= num_tokens) return;

  const int32_t seq_id = static_cast<int32_t>(token_idx / draft_token_num);
  const int32_t local_token_idx =
      static_cast<int32_t>(token_idx % draft_token_num);
  const uint32_t input = static_cast<uint32_t>(draft_token_ids[token_idx]);

  for (int32_t branch_idx = 0; branch_idx < params.num_branches; ++branch_idx) {
    const int32_t gram = params.oe_grams[branch_idx];
    const uint32_t oe_vocab_size =
        static_cast<uint32_t>(params.oe_vocab_sizes[branch_idx]);
    uint32_t running_ids = input;
    uint32_t vocab_power = params.vocab_size;

    for (int32_t lag = 1; lag < gram; ++lag) {
      const uint32_t prev = get_target_verify_history_token_from_history(
          draft_token_ids,
          tree_mask,
          seq_lens,
          history_state,
          seq_id,
          local_token_idx,
          draft_token_num,
          params.history_width,
          lag);
      running_ids += prev * vocab_power;
      vocab_power *= params.vocab_size;
    }

    const uint32_t hashed = running_ids * 2654435761u;
    hashed_out[static_cast<size_t>(branch_idx) * hashed_out_branch_stride +
               token_idx] = static_cast<int64_t>(hashed % oe_vocab_size);
  }
}

template <typename AcceptedT, typename AcceptLenT>
__device__ uint32_t get_after_accept_history_token(
    const int64_t* __restrict__ entry_history,
    const AcceptedT* __restrict__ accepted_draft_token_ids,
    const AcceptLenT* __restrict__ accept_lens,
    int32_t seq_id,
    int32_t history_width,
    int32_t accepted_width,
    int32_t prefix_lag) {
  int32_t extra_lens = static_cast<int32_t>(accept_lens[seq_id]) - 1;
  if (extra_lens < 0) extra_lens = 0;
  if (extra_lens > accepted_width - 1) extra_lens = accepted_width - 1;

  bool skipped_current = false;
  if (prefix_lag <= extra_lens) {
    const int32_t target_tail_idx = extra_lens - prefix_lag;
    int32_t tail_idx = 0;
    for (int32_t i = 0; i < accepted_width; ++i) {
      const AcceptedT token =
          accepted_draft_token_ids[static_cast<size_t>(seq_id) *
                                       accepted_width +
                                   i];
      if (token < 0) continue;
      if (!skipped_current) {
        skipped_current = true;
        continue;
      }
      if (tail_idx == target_tail_idx) {
        return static_cast<uint32_t>(token);
      }
      ++tail_idx;
    }
  }

  return get_history_state_token(
      entry_history, seq_id, history_width, prefix_lag - extra_lens);
}

template <typename InputIdT, typename AcceptedT, typename AcceptLenT>
__global__ void welm_oe_hash_mtp_draft_extend_after_verify_from_history_kernel(
    const InputIdT* __restrict__ input_ids,
    const AcceptedT* __restrict__ accepted_draft_token_ids,
    const AcceptLenT* __restrict__ accept_lens,
    const int64_t* __restrict__ entry_history,
    int64_t* __restrict__ hashed_out,
    int64_t hashed_out_branch_stride,
    int64_t* __restrict__ next_history_state,
    int32_t draft_token_num,
    int32_t accepted_width,
    int32_t use_entry_history_for_extend_hash_prefix,
    WelmOeHashRuntimeParams params) {
  const int32_t seq_id = static_cast<int32_t>(blockIdx.x);
  if (seq_id >= params.num_segments) return;

  const int32_t history_width = params.history_width;
  if (threadIdx.x < static_cast<uint32_t>(history_width)) {
    const int32_t col = static_cast<int32_t>(threadIdx.x);
    int64_t value = 0;
    if (col + 1 == history_width) {
      int32_t selected_pos = static_cast<int32_t>(accept_lens[seq_id]) - 1;
      if (selected_pos < 0) selected_pos = 0;
      if (selected_pos >= draft_token_num) selected_pos = draft_token_num - 1;
      value = static_cast<int64_t>(
          input_ids[static_cast<size_t>(seq_id) * draft_token_num +
                    selected_pos]);
    } else {
      const int32_t prefix_lag = history_width - col - 1;
      value = static_cast<int64_t>(get_after_accept_history_token(
          entry_history,
          accepted_draft_token_ids,
          accept_lens,
          seq_id,
          history_width,
          accepted_width,
          prefix_lag));
    }
    next_history_state[static_cast<size_t>(seq_id) * history_width + col] =
        value;
  }

  for (int32_t local_pos = static_cast<int32_t>(threadIdx.x);
       local_pos < draft_token_num;
       local_pos += static_cast<int32_t>(blockDim.x)) {
    const int32_t token_idx = seq_id * draft_token_num + local_pos;
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
          if (use_entry_history_for_extend_hash_prefix) {
            prev = get_history_state_token(
                entry_history, seq_id, history_width, lag - local_pos);
          } else {
            prev = get_after_accept_history_token(
                entry_history,
                accepted_draft_token_ids,
                accept_lens,
                seq_id,
                history_width,
                accepted_width,
                lag - local_pos);
          }
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

template <typename InputIdT>
__global__ void welm_oe_hash_mtp_draft_decode_from_history_kernel(
    const InputIdT* __restrict__ input_ids,
    const int64_t* __restrict__ history_state,
    const int64_t* __restrict__ parent_indices,
    int64_t* __restrict__ hashed_out,
    int64_t hashed_out_branch_stride,
    int64_t* __restrict__ next_history_state,
    int32_t base_query_count,
    int32_t use_parent,
    WelmOeHashRuntimeParams params) {
  const size_t token_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t num_tokens = static_cast<size_t>(params.num_segments);
  if (token_idx >= num_tokens) return;

  int32_t source_row = 0;
  if (use_parent) {
    source_row = static_cast<int32_t>(parent_indices[token_idx]);
  } else {
    const int32_t safe_base_query_count =
        base_query_count > 0 ? base_query_count : 1;
    int32_t repeat = static_cast<int32_t>(num_tokens) / safe_base_query_count;
    if (repeat < 1) repeat = 1;
    source_row = static_cast<int32_t>(token_idx) / repeat;
  }

  const int32_t history_width = params.history_width;
  const uint32_t input = static_cast<uint32_t>(input_ids[token_idx]);
  for (int32_t col = 0; col < history_width; ++col) {
    int64_t value = 0;
    if (col + 1 == history_width) {
      value = static_cast<int64_t>(input_ids[token_idx]);
    } else {
      value = history_state[static_cast<size_t>(source_row) * history_width +
                            col + 1];
    }
    next_history_state[token_idx * static_cast<size_t>(history_width) + col] =
        value;
  }

  for (int32_t branch_idx = 0; branch_idx < params.num_branches; ++branch_idx) {
    const int32_t gram = params.oe_grams[branch_idx];
    const uint32_t oe_vocab_size =
        static_cast<uint32_t>(params.oe_vocab_sizes[branch_idx]);
    uint32_t running_ids = input;
    uint32_t vocab_power = params.vocab_size;

    for (int32_t lag = 1; lag < gram; ++lag) {
      const uint32_t prev = get_history_state_token(
          history_state, source_row, history_width, lag);
      running_ids += prev * vocab_power;
      vocab_power *= params.vocab_size;
    }

    const uint32_t hashed = running_ids * 2654435761u;
    hashed_out[static_cast<size_t>(branch_idx) * hashed_out_branch_stride +
               token_idx] = static_cast<int64_t>(hashed % oe_vocab_size);
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

WelmOeHashRuntimeParams make_runtime_params(
    const tvm::ffi::Shape& oe_grams,
    const tvm::ffi::Shape& oe_vocab_sizes,
    size_t num_segments,
    size_t history_width,
    uint32_t vocab_size) {
  WelmOeHashRuntimeParams params{};
  params.num_segments = static_cast<int32_t>(num_segments);
  params.num_branches = static_cast<int32_t>(oe_grams.size());
  params.history_width = static_cast<int32_t>(history_width);
  params.vocab_size = vocab_size;
  for (size_t i = 0; i < oe_grams.size(); ++i) {
    params.oe_grams[i] = static_cast<int32_t>(oe_grams[i]);
    params.oe_vocab_sizes[i] = static_cast<int32_t>(oe_vocab_sizes[i]);
  }
  return params;
}

template <typename InputIdT, size_t MaxPrefixes>
void launch_welm_oe_hash_mtp_init_history_from_prefixes(
    const InputIdT* first_token_ids,
    int64_t* history_out,
    const tvm::ffi::Shape& prefixes,
    size_t num_segments,
    size_t history_width,
    bool has_first_token,
    DLDevice dl_device) {
  using namespace host;

  WelmOeHashParams<MaxPrefixes> params{};
  params.num_segments = static_cast<int32_t>(num_segments);
  params.num_branches = static_cast<int32_t>(history_width);
  for (size_t i = 0; i < prefixes.size(); ++i) {
    params.prefixes[i] = static_cast<int32_t>(prefixes[i]);
  }
  params.num_branches = static_cast<int32_t>(history_width);
  const size_t numel = num_segments * history_width;
  const size_t grid_size = div_ceil(numel, kBlockSize);
  LaunchKernel(grid_size, kBlockSize, dl_device)(
      welm_oe_hash_mtp_init_history_from_prefixes_kernel<InputIdT, MaxPrefixes>,
      first_token_ids,
      history_out,
      static_cast<int32_t>(has_first_token),
      params);
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

template <typename InputIdT, typename SeqLenT>
void launch_welm_oe_hash_mtp_target_verify_from_history(
    const InputIdT* draft_token_ids,
    const bool* tree_mask,
    const SeqLenT* seq_lens,
    const int64_t* history_state,
    int64_t* hashed_out,
    int64_t hashed_out_branch_stride,
    int32_t draft_token_num,
    const tvm::ffi::Shape& oe_grams,
    const tvm::ffi::Shape& oe_vocab_sizes,
    size_t batch_size,
    size_t history_width,
    uint32_t vocab_size,
    DLDevice dl_device) {
  using namespace host;

  auto params = make_runtime_params(
      oe_grams, oe_vocab_sizes, batch_size, history_width, vocab_size);
  const size_t num_tokens = batch_size * static_cast<size_t>(draft_token_num);
  const size_t grid_size = div_ceil(num_tokens, kBlockSize);
  LaunchKernel(grid_size, kBlockSize, dl_device)(
      welm_oe_hash_mtp_target_verify_from_history_kernel<InputIdT, SeqLenT>,
      draft_token_ids,
      tree_mask,
      seq_lens,
      history_state,
      hashed_out,
      hashed_out_branch_stride,
      draft_token_num,
      params);
}

template <typename InputIdT, typename AcceptedT, typename AcceptLenT>
void launch_welm_oe_hash_mtp_draft_extend_after_verify_from_history(
    const InputIdT* input_ids,
    const AcceptedT* accepted_draft_token_ids,
    const AcceptLenT* accept_lens,
    const int64_t* entry_history,
    int64_t* hashed_out,
    int64_t hashed_out_branch_stride,
    int64_t* next_history_state,
    int32_t draft_token_num,
    int32_t accepted_width,
    int32_t use_entry_history_for_extend_hash_prefix,
    const tvm::ffi::Shape& oe_grams,
    const tvm::ffi::Shape& oe_vocab_sizes,
    size_t batch_size,
    size_t history_width,
    uint32_t vocab_size,
    DLDevice dl_device) {
  using namespace host;

  auto params = make_runtime_params(
      oe_grams, oe_vocab_sizes, batch_size, history_width, vocab_size);
  LaunchKernel(batch_size, kBlockSize, dl_device)(
      welm_oe_hash_mtp_draft_extend_after_verify_from_history_kernel<
          InputIdT,
          AcceptedT,
          AcceptLenT>,
      input_ids,
      accepted_draft_token_ids,
      accept_lens,
      entry_history,
      hashed_out,
      hashed_out_branch_stride,
      next_history_state,
      draft_token_num,
      accepted_width,
      use_entry_history_for_extend_hash_prefix,
      params);
}

template <typename InputIdT>
void launch_welm_oe_hash_mtp_draft_decode_from_history(
    const InputIdT* input_ids,
    const int64_t* history_state,
    const int64_t* parent_indices,
    int64_t* hashed_out,
    int64_t hashed_out_branch_stride,
    int64_t* next_history_state,
    int32_t base_query_count,
    int32_t use_parent,
    const tvm::ffi::Shape& oe_grams,
    const tvm::ffi::Shape& oe_vocab_sizes,
    size_t num_tokens,
    size_t history_width,
    uint32_t vocab_size,
    DLDevice dl_device) {
  using namespace host;

  auto params = make_runtime_params(
      oe_grams, oe_vocab_sizes, num_tokens, history_width, vocab_size);
  const size_t grid_size = div_ceil(num_tokens, kBlockSize);
  LaunchKernel(grid_size, kBlockSize, dl_device)(
      welm_oe_hash_mtp_draft_decode_from_history_kernel<InputIdT>,
      input_ids,
      history_state,
      parent_indices,
      hashed_out,
      hashed_out_branch_stride,
      next_history_state,
      base_query_count,
      use_parent,
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

void check_runtime_hash_inputs(
    const tvm::ffi::Shape& oe_grams,
    const tvm::ffi::Shape& oe_vocab_sizes,
    size_t num_segments,
    size_t num_branches,
    size_t history_width,
    int64_t vocab_size) {
  using namespace host;

  RuntimeCheck(num_segments > 0, "num_segments must be positive");
  RuntimeCheck(history_width > 0, "history_width must be positive");
  RuntimeCheck(num_branches > 0, "num_branches must be positive");
  RuntimeCheck(vocab_size > 0, "vocab_size must be positive");
  RuntimeCheck(
      oe_grams.size() == num_branches,
      "oe_grams size must match hashed_out branch count");
  RuntimeCheck(
      oe_vocab_sizes.size() == num_branches,
      "oe_vocab_sizes size must match hashed_out branch count");
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

void check_prefix_history_inputs(
    const tvm::ffi::Shape& prefixes,
    size_t num_segments,
    size_t history_width) {
  using namespace host;

  RuntimeCheck(num_segments > 0, "num_segments must be positive");
  RuntimeCheck(history_width > 0, "history_width must be positive");
  RuntimeCheck(
      prefixes.size() == num_segments * history_width,
      "prefix count must equal num_segments * history_width");
}

struct WelmOeHashMtpInitHistoryFromPrefixes {
  static void run(
      tvm::ffi::TensorView first_token_ids,
      tvm::ffi::Shape prefixes,
      tvm::ffi::TensorView history_out,
      int64_t has_first_token) {
    using namespace host;

    SymbolicSize S = {"num_segments"};
    SymbolicSize H = {"history_width"};
    SymbolicDevice device;
    SymbolicDType input_dtype;
    device.set_options<kDLCUDA, kDLROCM>();

    TensorMatcher({S, H})
        .with_strides({H, 1})
        .with_dtype<int64_t>()
        .with_device(device)
        .verify(history_out);
    if (has_first_token) {
      TensorMatcher({S})
          .with_dtype<int32_t, int64_t>(input_dtype)
          .with_device(device)
          .verify(first_token_ids);
    }

    const size_t num_segments = S.unwrap();
    const size_t history_width = H.unwrap();
    check_prefix_history_inputs(prefixes, num_segments, history_width);
    const DLDevice dl_device = device.unwrap();
    const bool input_is_int32 =
        has_first_token && input_dtype.unwrap().bits == 32;

#define DISPATCH_WELM_OE_INIT_HISTORY(MAX_PREFIXES)                           \
    if (prefixes.size() <= (MAX_PREFIXES)) {                                  \
      if (input_is_int32) {                                                   \
        launch_welm_oe_hash_mtp_init_history_from_prefixes<                       \
            int32_t,                                                          \
            (MAX_PREFIXES)>(                                                  \
            static_cast<const int32_t*>(first_token_ids.data_ptr()),          \
            static_cast<int64_t*>(history_out.data_ptr()),                    \
            prefixes,                                                         \
            num_segments,                                                     \
            history_width,                                                    \
            true,                                                             \
            dl_device);                                                       \
      } else {                                                                \
        launch_welm_oe_hash_mtp_init_history_from_prefixes<                       \
            int64_t,                                                          \
            (MAX_PREFIXES)>(                                                  \
            has_first_token                                                   \
                ? static_cast<const int64_t*>(first_token_ids.data_ptr())     \
                : static_cast<const int64_t*>(history_out.data_ptr()),        \
            static_cast<int64_t*>(history_out.data_ptr()),                    \
            prefixes,                                                         \
            num_segments,                                                     \
            history_width,                                                    \
            has_first_token != 0,                                             \
            dl_device);                                                       \
      }                                                                       \
      return;                                                                 \
    }

    DISPATCH_WELM_OE_INIT_HISTORY(128);
    DISPATCH_WELM_OE_INIT_HISTORY(256);
    DISPATCH_WELM_OE_INIT_HISTORY(512);
    DISPATCH_WELM_OE_INIT_HISTORY(1024);
    DISPATCH_WELM_OE_INIT_HISTORY(2048);
    DISPATCH_WELM_OE_INIT_HISTORY(4096);
#undef DISPATCH_WELM_OE_INIT_HISTORY

    RuntimeCheck(false, "too many WeLM OE history prefixes");
  }
};

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

struct WelmOeHashMtpTargetVerifyFromHistory {
  static void run(
      tvm::ffi::TensorView draft_token_ids,
      tvm::ffi::TensorView tree_mask,
      tvm::ffi::TensorView seq_lens,
      tvm::ffi::TensorView history_state,
      tvm::ffi::Shape oe_grams,
      tvm::ffi::Shape oe_vocab_sizes,
      tvm::ffi::TensorView hashed_out,
      int64_t vocab_size,
      int64_t draft_token_num) {
    using namespace host;

    SymbolicSize N = {"num_tokens"};
    SymbolicSize M = {"tree_mask_numel"};
    SymbolicSize S = {"batch_size"};
    SymbolicSize H = {"history_width"};
    SymbolicSize B = {"num_branches"};
    SymbolicSize SO = {"hashed_out_branch_stride"};
    SymbolicDevice device;
    SymbolicDType input_dtype;
    SymbolicDType seq_lens_dtype;
    device.set_options<kDLCUDA, kDLROCM>();

    TensorMatcher({N})
        .with_dtype<int32_t, int64_t>(input_dtype)
        .with_device(device)
        .verify(draft_token_ids);
    TensorMatcher({M}).with_device(device).verify(tree_mask);
    TensorMatcher({S})
        .with_dtype<int32_t, int64_t>(seq_lens_dtype)
        .with_device(device)
        .verify(seq_lens);
    TensorMatcher({S, H})
        .with_strides({H, 1})
        .with_dtype<int64_t>()
        .with_device(device)
        .verify(history_state);
    TensorMatcher({B, N})
        .with_strides({SO, 1})
        .with_dtype<int64_t>()
        .with_device(device)
        .verify(hashed_out);

    const size_t num_tokens = N.unwrap();
    const size_t batch_size = S.unwrap();
    const size_t history_width = H.unwrap();
    const size_t num_branches = B.unwrap();
    RuntimeCheck(draft_token_num > 0, "draft_token_num must be positive");
    RuntimeCheck(
        num_tokens == batch_size * static_cast<size_t>(draft_token_num),
        "target-verify token count must equal batch_size * draft_token_num");
    if (num_tokens == 0) return;

    check_runtime_hash_inputs(
        oe_grams,
        oe_vocab_sizes,
        batch_size,
        num_branches,
        history_width,
        vocab_size);

    const uint32_t vocab_size_u32 = static_cast<uint32_t>(vocab_size);
    const DLDevice dl_device = device.unwrap();
    const bool input_is_int32 = input_dtype.unwrap().bits == 32;
    const bool seq_lens_is_int32 = seq_lens_dtype.unwrap().bits == 32;

#define DISPATCH_WELM_OE_HASH_TARGET_VERIFY_HISTORY(INPUT_T, SEQ_T)           \
    launch_welm_oe_hash_mtp_target_verify_from_history<INPUT_T, SEQ_T>(           \
        static_cast<const INPUT_T*>(draft_token_ids.data_ptr()),              \
        static_cast<const bool*>(tree_mask.data_ptr()),                       \
        static_cast<const SEQ_T*>(seq_lens.data_ptr()),                       \
        static_cast<const int64_t*>(history_state.data_ptr()),                \
        static_cast<int64_t*>(hashed_out.data_ptr()),                         \
        SO.unwrap(),                                                          \
        static_cast<int32_t>(draft_token_num),                                \
        oe_grams,                                                             \
        oe_vocab_sizes,                                                       \
        batch_size,                                                           \
        history_width,                                                        \
        vocab_size_u32,                                                       \
        dl_device)

    if (input_is_int32) {
      if (seq_lens_is_int32) {
        DISPATCH_WELM_OE_HASH_TARGET_VERIFY_HISTORY(int32_t, int32_t);
      } else {
        DISPATCH_WELM_OE_HASH_TARGET_VERIFY_HISTORY(int32_t, int64_t);
      }
    } else {
      if (seq_lens_is_int32) {
        DISPATCH_WELM_OE_HASH_TARGET_VERIFY_HISTORY(int64_t, int32_t);
      } else {
        DISPATCH_WELM_OE_HASH_TARGET_VERIFY_HISTORY(int64_t, int64_t);
      }
    }
#undef DISPATCH_WELM_OE_HASH_TARGET_VERIFY_HISTORY
  }
};

struct WelmOeHashMtpDraftExtendAfterVerifyFromHistory {
  static void run(
      tvm::ffi::TensorView input_ids,
      tvm::ffi::TensorView accepted_draft_token_ids,
      tvm::ffi::TensorView accept_lens,
      tvm::ffi::TensorView entry_history,
      tvm::ffi::Shape oe_grams,
      tvm::ffi::Shape oe_vocab_sizes,
      tvm::ffi::TensorView hashed_out,
      tvm::ffi::TensorView next_history_state,
      int64_t vocab_size,
      int64_t draft_token_num,
      int64_t use_entry_history_for_extend_hash_prefix) {
    using namespace host;

    SymbolicSize N = {"num_tokens"};
    SymbolicSize S = {"batch_size"};
    SymbolicSize A = {"accepted_width"};
    SymbolicSize H = {"history_width"};
    SymbolicSize B = {"num_branches"};
    SymbolicSize SO = {"hashed_out_branch_stride"};
    SymbolicDevice device;
    SymbolicDType input_dtype;
    SymbolicDType accepted_dtype;
    SymbolicDType accept_lens_dtype;
    device.set_options<kDLCUDA, kDLROCM>();

    TensorMatcher({N})
        .with_dtype<int32_t, int64_t>(input_dtype)
        .with_device(device)
        .verify(input_ids);
    TensorMatcher({S, A})
        .with_dtype<int32_t, int64_t>(accepted_dtype)
        .with_device(device)
        .verify(accepted_draft_token_ids);
    TensorMatcher({S})
        .with_dtype<int32_t, int64_t>(accept_lens_dtype)
        .with_device(device)
        .verify(accept_lens);
    TensorMatcher({S, H})
        .with_strides({H, 1})
        .with_dtype<int64_t>()
        .with_device(device)
        .verify(entry_history);
    TensorMatcher({B, N})
        .with_strides({SO, 1})
        .with_dtype<int64_t>()
        .with_device(device)
        .verify(hashed_out);
    TensorMatcher({S, H})
        .with_strides({H, 1})
        .with_dtype<int64_t>()
        .with_device(device)
        .verify(next_history_state);

    const size_t num_tokens = N.unwrap();
    const size_t batch_size = S.unwrap();
    const size_t accepted_width = A.unwrap();
    const size_t history_width = H.unwrap();
    const size_t num_branches = B.unwrap();
    RuntimeCheck(draft_token_num > 0, "draft_token_num must be positive");
    RuntimeCheck(
        num_tokens == batch_size * static_cast<size_t>(draft_token_num),
        "draft-extend token count must equal batch_size * draft_token_num");
    if (num_tokens == 0) return;

    check_runtime_hash_inputs(
        oe_grams,
        oe_vocab_sizes,
        batch_size,
        num_branches,
        history_width,
        vocab_size);

    const uint32_t vocab_size_u32 = static_cast<uint32_t>(vocab_size);
    const DLDevice dl_device = device.unwrap();
    const bool input_is_int32 = input_dtype.unwrap().bits == 32;
    const bool accepted_is_int32 = accepted_dtype.unwrap().bits == 32;
    const bool accept_lens_is_int32 = accept_lens_dtype.unwrap().bits == 32;

#define DISPATCH_WELM_OE_HASH_DRAFT_EXTEND_HISTORY(                          \
    INPUT_T, ACCEPTED_T, ACCEPT_LEN_T)                                       \
    launch_welm_oe_hash_mtp_draft_extend_after_verify_from_history<              \
        INPUT_T,                                                             \
        ACCEPTED_T,                                                          \
        ACCEPT_LEN_T>(                                                       \
        static_cast<const INPUT_T*>(input_ids.data_ptr()),                   \
        static_cast<const ACCEPTED_T*>(accepted_draft_token_ids.data_ptr()), \
        static_cast<const ACCEPT_LEN_T*>(accept_lens.data_ptr()),            \
        static_cast<const int64_t*>(entry_history.data_ptr()),               \
        static_cast<int64_t*>(hashed_out.data_ptr()),                        \
        SO.unwrap(),                                                         \
        static_cast<int64_t*>(next_history_state.data_ptr()),                \
        static_cast<int32_t>(draft_token_num),                               \
        static_cast<int32_t>(accepted_width),                                \
        static_cast<int32_t>(use_entry_history_for_extend_hash_prefix),       \
        oe_grams,                                                            \
        oe_vocab_sizes,                                                      \
        batch_size,                                                          \
        history_width,                                                       \
        vocab_size_u32,                                                      \
        dl_device)

#define DISPATCH_WELM_OE_HASH_DRAFT_EXTEND_HISTORY_ACCEPT(                   \
    INPUT_T, ACCEPT_LEN_T)                                                   \
    if (accepted_is_int32) {                                                 \
      DISPATCH_WELM_OE_HASH_DRAFT_EXTEND_HISTORY(                            \
          INPUT_T, int32_t, ACCEPT_LEN_T);                                   \
    } else {                                                                 \
      DISPATCH_WELM_OE_HASH_DRAFT_EXTEND_HISTORY(                            \
          INPUT_T, int64_t, ACCEPT_LEN_T);                                   \
    }

    if (input_is_int32) {
      if (accept_lens_is_int32) {
        DISPATCH_WELM_OE_HASH_DRAFT_EXTEND_HISTORY_ACCEPT(int32_t, int32_t);
      } else {
        DISPATCH_WELM_OE_HASH_DRAFT_EXTEND_HISTORY_ACCEPT(int32_t, int64_t);
      }
    } else {
      if (accept_lens_is_int32) {
        DISPATCH_WELM_OE_HASH_DRAFT_EXTEND_HISTORY_ACCEPT(int64_t, int32_t);
      } else {
        DISPATCH_WELM_OE_HASH_DRAFT_EXTEND_HISTORY_ACCEPT(int64_t, int64_t);
      }
    }

#undef DISPATCH_WELM_OE_HASH_DRAFT_EXTEND_HISTORY_ACCEPT
#undef DISPATCH_WELM_OE_HASH_DRAFT_EXTEND_HISTORY
  }
};

struct WelmOeHashMtpDraftDecodeFromHistory {
  static void run(
      tvm::ffi::TensorView input_ids,
      tvm::ffi::TensorView history_state,
      tvm::ffi::TensorView parent_indices,
      tvm::ffi::Shape oe_grams,
      tvm::ffi::Shape oe_vocab_sizes,
      tvm::ffi::TensorView hashed_out,
      tvm::ffi::TensorView next_history_state,
      int64_t vocab_size,
      int64_t base_query_count,
      int64_t use_parent) {
    using namespace host;

    SymbolicSize N = {"num_tokens"};
    SymbolicSize P = {"parent_index_count"};
    SymbolicSize S = {"history_rows"};
    SymbolicSize H = {"history_width"};
    SymbolicSize B = {"num_branches"};
    SymbolicSize SO = {"hashed_out_branch_stride"};
    SymbolicDevice device;
    SymbolicDType input_dtype;
    device.set_options<kDLCUDA, kDLROCM>();

    TensorMatcher({N})
        .with_dtype<int32_t, int64_t>(input_dtype)
        .with_device(device)
        .verify(input_ids);
    TensorMatcher({S, H})
        .with_strides({H, 1})
        .with_dtype<int64_t>()
        .with_device(device)
        .verify(history_state);
    TensorMatcher({P}).with_dtype<int64_t>().with_device(device).verify(parent_indices);
    TensorMatcher({B, N})
        .with_strides({SO, 1})
        .with_dtype<int64_t>()
        .with_device(device)
        .verify(hashed_out);
    TensorMatcher({N, H})
        .with_strides({H, 1})
        .with_dtype<int64_t>()
        .with_device(device)
        .verify(next_history_state);

    const size_t num_tokens = N.unwrap();
    const size_t history_width = H.unwrap();
    const size_t num_branches = B.unwrap();
    if (num_tokens == 0) return;
    RuntimeCheck(
        !use_parent || P.unwrap() >= num_tokens,
        "draft-decode parent_indices is shorter than input_ids");
    RuntimeCheck(
        use_parent || static_cast<size_t>(base_query_count) <= S.unwrap(),
        "draft-decode base_query_count exceeds history rows");

    check_runtime_hash_inputs(
        oe_grams,
        oe_vocab_sizes,
        num_tokens,
        num_branches,
        history_width,
        vocab_size);

    const uint32_t vocab_size_u32 = static_cast<uint32_t>(vocab_size);
    const DLDevice dl_device = device.unwrap();
    const bool input_is_int32 = input_dtype.unwrap().bits == 32;
    if (input_is_int32) {
      launch_welm_oe_hash_mtp_draft_decode_from_history<int32_t>(
          static_cast<const int32_t*>(input_ids.data_ptr()),
          static_cast<const int64_t*>(history_state.data_ptr()),
          static_cast<const int64_t*>(parent_indices.data_ptr()),
          static_cast<int64_t*>(hashed_out.data_ptr()),
          SO.unwrap(),
          static_cast<int64_t*>(next_history_state.data_ptr()),
          static_cast<int32_t>(base_query_count),
          static_cast<int32_t>(use_parent),
          oe_grams,
          oe_vocab_sizes,
          num_tokens,
          history_width,
          vocab_size_u32,
          dl_device);
    } else {
      launch_welm_oe_hash_mtp_draft_decode_from_history<int64_t>(
          static_cast<const int64_t*>(input_ids.data_ptr()),
          static_cast<const int64_t*>(history_state.data_ptr()),
          static_cast<const int64_t*>(parent_indices.data_ptr()),
          static_cast<int64_t*>(hashed_out.data_ptr()),
          SO.unwrap(),
          static_cast<int64_t*>(next_history_state.data_ptr()),
          static_cast<int32_t>(base_query_count),
          static_cast<int32_t>(use_parent),
          oe_grams,
          oe_vocab_sizes,
          num_tokens,
          history_width,
          vocab_size_u32,
          dl_device);
    }
  }
};

}  // namespace
