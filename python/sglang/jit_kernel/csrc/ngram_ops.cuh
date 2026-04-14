// ngram_ops.cuh — JIT-compiled CUDA kernels for n-gram operations
// Used by WeLM v4 speculative decoding (over-encoding + MTP)
//
// Migrated from custom_ops/csrc/overencoding_mtp.cu to eliminate the
// external prc_custom_ops dependency.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// ---------------------------------------------------------------------------
// Kernel: build_ngram_with_tree
// ---------------------------------------------------------------------------

__global__ void build_ngram_with_tree_kernel(
    int64_t* ngram_input_ids,
    int64_t* parent_list,
    int64_t* token_list,
    int64_t* current_parrent_list,
    int64_t* buffer,
    int topk,
    int gram_n,
    int buffer_size,
    int i,
    int parent_list_stride,
    int token_list_stride)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    if (tid >= topk) return;

    int64_t current_pos = current_parrent_list[bid * topk + tid];
    int gram = gram_n - 1 - i;
    if (gram > 0) {
        ngram_input_ids[bid * topk + tid] = buffer[(bid + 1) * buffer_size - gram];
        return;
    }

    int64_t parent_token;
    for (int gram_ids = 0; gram_ids < gram_n - 1; gram_ids++) {
        int pre_layer_num_node = topk + topk * topk * (i - 1);
        int cur_layer_pos = current_pos - pre_layer_num_node;
        int parent_layer_pos = cur_layer_pos / topk;
        int parent_offset = 1 + topk * (i - 1);
        int parent_pos = parent_layer_pos + parent_offset;
        parent_pos = parent_list[bid * parent_list_stride + parent_pos];
        parent_token = token_list[bid * token_list_stride + parent_pos];
        current_pos = parent_pos;
        i--;
    }
    ngram_input_ids[bid * topk + tid] = parent_token;
}

// ---------------------------------------------------------------------------
// Kernel: build_target_verify_ngram
// ---------------------------------------------------------------------------

__global__ void build_target_verify_ngram_kernel(
    int64_t* ngram_input_ids,
    int64_t* buffer,
    int64_t* draft_token_ids,
    bool* tree_mask,
    int64_t* positions,
    int64_t* seq_lens,
    int gram_n,
    int draft_token_num,
    int buffer_size)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    if (tid != 0) return;

    int seq_id = bid / draft_token_num;
    int64_t mask_offset = 0;
    for (int i = 0; i < seq_id; i++) {
        int64_t mask_len = seq_lens[i] + draft_token_num;
        mask_offset += draft_token_num * mask_len;
    }
    int64_t seq_len = seq_lens[seq_id];
    int64_t mask_len = seq_len + draft_token_num;
    mask_offset += (bid % draft_token_num) * mask_len;

    int target_gram = gram_n;
    int64_t res = 0;
    for (int64_t i = seq_len + draft_token_num - 1; i >= seq_len; i--) {
        if (tree_mask[mask_offset + i]) {
            target_gram--;
            if (target_gram == 0) {
                res = draft_token_ids[seq_id * draft_token_num + i - seq_len];
                break;
            }
        }
    }
    if (target_gram != 0) {
        res = buffer[(seq_id + 1) * buffer_size - target_gram - 1];
    }
    ngram_input_ids[bid] = res;
}

// ---------------------------------------------------------------------------
// Kernel: assign_ngram_input_ids_draft_extend_after_decode
// ---------------------------------------------------------------------------

__global__ void assign_ngram_input_ids_draft_extend_after_decode_kernel(
    int64_t* input_ids,
    int64_t* buffer,
    int64_t* input_ids_gram,
    int32_t* accept_length,
    int gram_n,
    int buffer_size,
    bool update_buffer)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    int gram = gram_n - 1;
    int accum_accept_len = 0;
    for (int i = 0; i < bid; i++) {
        accum_accept_len += accept_length[i];
    }
    int curr_accept_len = accept_length[bid];
    if (tid < curr_accept_len) {
        if (tid >= gram) {
            input_ids_gram[accum_accept_len + tid] = input_ids[accum_accept_len + tid - gram];
        } else {
            input_ids_gram[accum_accept_len + tid] = buffer[bid * buffer_size + buffer_size - (gram - tid)];
        }
    }
    // Note: buffer update path is disabled in original implementation
    // (early return at line 116-118 of overencoding_mtp.cu)
}

// ---------------------------------------------------------------------------
// Host wrapper functions (called by TVM FFI)
// ---------------------------------------------------------------------------

inline void build_ngram_with_tree(
    torch::Tensor ngram_input_ids,
    torch::Tensor parent_list,
    torch::Tensor token_list,
    torch::Tensor current_parrent_list,
    torch::Tensor buffer,
    int buffer_size,
    int gram_n,
    int topk,
    int i)
{
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    int bs = parent_list.size(0);
    int parent_list_stride = parent_list.stride(0);
    int token_list_stride = token_list.stride(0);

    build_ngram_with_tree_kernel<<<bs, 32, 0, stream>>>(
        ngram_input_ids.data_ptr<int64_t>(),
        parent_list.data_ptr<int64_t>(),
        token_list.data_ptr<int64_t>(),
        current_parrent_list.data_ptr<int64_t>(),
        buffer.data_ptr<int64_t>(),
        topk, gram_n, buffer_size, i,
        parent_list_stride, token_list_stride);
}

inline void build_ngram_with_target_verify(
    torch::Tensor ngram_input_ids,
    torch::Tensor buffer,
    torch::Tensor draft_token_ids,
    torch::Tensor tree_mask,
    torch::Tensor positions,
    torch::Tensor seq_lens,
    int gram_n,
    int draft_token_num,
    int buffer_size)
{
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    int bs = seq_lens.size(0);
    build_target_verify_ngram_kernel<<<bs * draft_token_num, 32, 0, stream>>>(
        ngram_input_ids.data_ptr<int64_t>(),
        buffer.data_ptr<int64_t>(),
        draft_token_ids.data_ptr<int64_t>(),
        tree_mask.data_ptr<bool>(),
        positions.data_ptr<int64_t>(),
        seq_lens.data_ptr<int64_t>(),
        gram_n, draft_token_num, buffer_size);
}

inline void assign_ngram_input_ids_draft_extend_after_decode(
    torch::Tensor input_ids,
    torch::Tensor buffer,
    torch::Tensor input_ids_gram,
    torch::Tensor accept_length,
    int gram_n,
    int buffer_size,
    int update_buffer)
{
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    int bs = accept_length.numel();
    assign_ngram_input_ids_draft_extend_after_decode_kernel<<<bs, 32, 0, stream>>>(
        input_ids.data_ptr<int64_t>(),
        buffer.data_ptr<int64_t>(),
        input_ids_gram.data_ptr<int64_t>(),
        accept_length.data_ptr<int32_t>(),
        gram_n, buffer_size, static_cast<bool>(update_buffer));
}
