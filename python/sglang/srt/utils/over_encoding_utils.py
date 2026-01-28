from typing import List

import cutex
import torch
import triton
import triton.language as tl


@triton.jit
def assign_ngram_input_ids_kernel(
    input_ids: torch.Tensor,
    input_ids_gram: torch.Tensor,
    gram_n: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GRAM_BLOCK_SIZE: tl.constexpr,
    extend_len,
):
    data_offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    gram_offset = gram_n - 1
    mask = data_offset < extend_len
    data = tl.load(input_ids + data_offset, mask)
    tl.store(
        input_ids_gram + data_offset + gram_offset,
        data,
        data_offset + gram_offset < extend_len,
    )
    prefix_offset = tl.arange(0, GRAM_BLOCK_SIZE)
    tl.store(
        input_ids_gram + prefix_offset,
        0,
        prefix_offset < tl.minimum(gram_offset, extend_len),
    )


def assign_ngram_input_ids_draft_extend(
    input_ids: torch.Tensor,
    input_ids_gram: torch.Tensor,
    extend_lens: List[int],
    gram_n: int,
):
    pt = 0
    GRAM_BLOCK_SIZE = triton.next_power_of_2(gram_n - 1)
    for i, extend_len in enumerate(extend_lens):
        grid = triton.cdiv(extend_len, 128)
        assign_ngram_input_ids_kernel[(grid,)](
            input_ids[pt : pt + extend_len],
            input_ids_gram[pt : pt + extend_len],
            gram_n,
            128,
            GRAM_BLOCK_SIZE,
            extend_len,
        )
        pt += extend_len


@triton.jit
def assign_ngram_input_ids_draft_decode_first_token_kernel(
    input_ids_buffer: torch.Tensor,
    input_ids_gram_decode: torch.Tensor,
    buffer_size: tl.constexpr,
    gram_n: tl.constexpr,
    topk: tl.constexpr,
    topk_block: tl.constexpr,
    bs_block: tl.constexpr,
):
    bid = tl.program_id(0)

    gram_offset = bid * buffer_size + buffer_size - gram_n + 1
    data = tl.load(input_ids_buffer + gram_offset)

    repeat_data = tl.full((topk_block,), data, dtype=input_ids_gram_decode.dtype)
    save_offset = tl.arange(0, topk_block)
    tl.store(
        input_ids_gram_decode + bid * topk + save_offset,
        repeat_data,
        save_offset < topk,
    )


def assign_ngram_input_ids_draft_decode_first_token(
    input_ids_buffer: torch.Tensor,
    input_ids_gram_decode: torch.Tensor,
    seq_lens: torch.Tensor,
    gram_n: int,
    topk: int,
    buffer_size: int,
):
    bs = seq_lens.numel()
    bs_block = triton.next_power_of_2(bs)
    topk_block = triton.next_power_of_2(topk)
    assign_ngram_input_ids_draft_decode_first_token_kernel[(bs,)](
        input_ids_buffer,
        input_ids_gram_decode,
        buffer_size,
        gram_n,
        topk,
        topk_block,
        bs_block,
    )


@triton.jit
def assign_ngram_buffer_kernel(
    input_ids: torch.Tensor,
    buffer: torch.Tensor,
    seq_lens: torch.Tensor,
    buffer_size: tl.constexpr,
    bs_block: tl.constexpr,
    buffer_size_block: tl.constexpr,
):
    bid = tl.program_id(0)
    bs_offset = tl.arange(0, bs_block)
    cu_seq_len = tl.sum(tl.load(seq_lens + bs_offset, bs_offset <= bid))
    data_offset = tl.arange(0, buffer_size_block)
    load_offset = cu_seq_len - buffer_size + data_offset
    mask = (data_offset < buffer_size) & (load_offset >= 0)
    data = tl.load(input_ids + load_offset, mask, other=0)
    tl.store(buffer + bid * buffer_size + data_offset, data, data_offset < buffer_size)


def assign_ngram_buffer(
    input_ids: torch.Tensor,
    buffer: torch.Tensor,
    seq_lens: torch.Tensor,
    buffer_size: int,
):
    bs = seq_lens.numel()
    bs_block = triton.next_power_of_2(bs)
    buffer_size_block = triton.next_power_of_2(buffer_size)
    assign_ngram_buffer_kernel[(bs,)](
        input_ids, buffer, seq_lens, buffer_size, bs_block, buffer_size_block
    )


kernels = cutex.SourceModule(
    """
__global__ void build_ngram_with_tree_kernel(Tensor<long, 1> ngram_input_ids, Tensor<long, 2> parent_list, Tensor<long, 2> token_list, Tensor<long, 2> current_parrent_list,
                           Tensor<long, 1> buffer, int topk, int gram_n, int buffer_size, int i) {
        int bid = blockIdx.x;
        int tid = threadIdx.x;
        if(tid >= topk){
            return;
        }
        //long *ngram_input_ids_ptr = ngram_input_ids + bid * topk;
        long current_pos = current_parrent_list[bid][tid];
        int gram = gram_n - 1 - i;
        if(gram > 0){
            ngram_input_ids[bid * topk + tid] = buffer[(bid+1) * buffer_size - gram];
            return;
        }
        long parent_token;
        for(int gram_ids=0; gram_ids<gram_n-1; gram_ids++){
            int pre_layer_num_node = topk + topk*topk*(i-1); // exclude root node
            int cur_layer_pos = current_pos - pre_layer_num_node;
            int parent_layer_pos = cur_layer_pos / topk;
            int parent_offset = 1 + topk * (i-1);
            int parent_pos = parent_layer_pos + parent_offset;
            parent_pos = parent_list[bid][parent_pos];
            parent_token = token_list[bid][parent_pos];
            current_pos = parent_pos;
            i--;
        }
        ngram_input_ids[bid * topk + tid] = parent_token;
        return;
}

__global__ void build_target_verify_ngram_kernel(
    Tensor<long, 1> ngram_input_ids,
    Tensor<long, 1> buffer,
    Tensor<long, 1> draft_token_ids,
    Tensor<bool, 1> tree_mask,
    Tensor<long, 1> positions,
    Tensor<long, 1> seq_lens,
    int gram_n,
    int draft_token_num,
    int buffer_size)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    if(tid != 0){
        return;
    }
    int seq_id = bid / draft_token_num;
    long seq_len, mask_len, mask_offset;
    mask_offset = 0;
    for(int i=0; i<seq_id; i++){
        mask_len = seq_lens[i] + draft_token_num;
        mask_offset += draft_token_num * mask_len;
    }
    seq_len = seq_lens[seq_id];
    mask_len = seq_len + draft_token_num;
    mask_offset += (bid % draft_token_num) * mask_len;

    int target_gram = gram_n;
    long res;
    for(int i=seq_len + draft_token_num - 1; i>=seq_len; i--){
        if(tree_mask[mask_offset + i]){
            target_gram--;
            if(target_gram == 0){
                res = draft_token_ids[seq_id * draft_token_num + i - seq_len];
                //printf("== %d %d %ld %d %d %ld %ld == ", bid, seq_id, mask_offset, seq_id * draft_token_num, i, seq_len, res);
                break;
            }
        }
    }
    if(target_gram != 0){
        res = buffer[(seq_id+1) * buffer_size - target_gram - 1];
    }
    ngram_input_ids[bid] = res;
    return;
}

__global__ void assign_ngram_input_ids_draft_extend_after_decode(
    Tensor<long, 1> input_ids,
    Tensor<long, 1> buffer,
    Tensor<long, 1> input_ids_gram,
    Tensor<int, 1> accept_length,
    int gram_n,
    int buffer_size,
    bool update_buffer
){
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    int gram = gram_n - 1;
    int accum_accept_len = 0, curr_accept_len;
    for(int i=0; i<bid; i++){
        accum_accept_len += accept_length[i];
    }
    curr_accept_len = accept_length[bid];
    if(tid < curr_accept_len){
        if(tid >= gram){
            input_ids_gram[accum_accept_len + tid] = int64_t(input_ids[accum_accept_len + tid - gram]);
        }else{
            input_ids_gram[accum_accept_len + tid] = buffer[bid * buffer_size + buffer_size - (gram-tid)];
        }
    }
    if(not update_buffer){
        return;
    }

    if(tid >= buffer_size){
        return;
    }
    long new_buffer[10];
    int remained_size = buffer_size - curr_accept_len;
    if(tid < remained_size){
        new_buffer[tid] = buffer[bid * buffer_size + buffer_size - remained_size + tid];
    } else{
        new_buffer[tid] = int64_t(input_ids[accum_accept_len + tid - remained_size]);
    }
    buffer[bid * buffer_size + tid] = new_buffer[tid];
    return;
}

""",
    float_bits=16,  # change to 16 to use half precision as `float` type in the above source code.
    boundscheck=True,  # turning on for debug and off for performance (to use full threads of a block), default is on.
)


def build_ngram_with_tree(
    ngram_input_ids: torch.Tensor,
    parent_list: torch.Tensor,
    token_list: torch.Tensor,
    current_parrent_list: torch.Tensor,
    buffer: torch.Tensor,
    buffer_size: int,
    gram_n: int,
    topk: int,
    i: int,
):
    bs = parent_list.shape[0]
    kernels.build_ngram_with_tree_kernel(
        ngram_input_ids,
        parent_list,
        token_list,
        current_parrent_list,
        buffer,
        topk,
        gram_n,
        buffer_size,
        i,
        grid=(bs, 1, 1),
        block=(32, 1, 1),
    )


def build_ngram_with_target_verify(
    n_gram_input_ids: torch.Tensor,
    buffer: torch.Tensor,
    draft_token_ids: torch.Tensor,
    tree_mask: torch.Tensor,
    positions: torch.Tensor,
    seq_lens: torch.Tensor,
    gram_n: int,
    draft_token_num: int,
    buffer_size: int,
):
    bs = seq_lens.numel()
    kernels.build_target_verify_ngram_kernel(
        n_gram_input_ids,
        buffer,
        draft_token_ids,
        tree_mask,
        positions,
        seq_lens,
        gram_n,
        draft_token_num,
        buffer_size,
        grid=(bs * draft_token_num, 1, 1),
        block=(32, 1, 1),
    )


# will also update the buffer if update_buffer is True
def assign_ngram_input_ids_draft_extend_after_decode(
    input_ids: torch.Tensor,
    buffer: torch.Tensor,
    input_ids_gram: torch.Tensor,
    accept_length: torch.Tensor,
    gram_n: int,
    buffer_size: int,
    update_buffer: bool = False,
):
    bs = accept_length.numel()
    assert buffer_size < 10, "buffer_size should be less than 10"
    kernels.assign_ngram_input_ids_draft_extend_after_decode(
        input_ids,
        buffer,
        input_ids_gram,
        accept_length,
        gram_n,
        buffer_size,
        update_buffer,
        grid=(bs, 1, 1),
        block=(32, 1, 1),
    )


@triton.jit
def assign_buffer_kernel(
    buffer: torch.Tensor,
    new_buffer: torch.Tensor,
    keep_indices: torch.Tensor,
    buffer_size: tl.constexpr,
    buffer_block: tl.constexpr,
):
    bid = tl.program_id(0)
    offset = tl.load(keep_indices + bid)
    buffer_offset = tl.arange(0, buffer_block)
    data = tl.load(
        buffer + offset * buffer_size + buffer_offset, buffer_offset < buffer_size
    )
    tl.store(
        new_buffer + bid * buffer_size + buffer_offset,
        data,
        buffer_offset < buffer_size,
    )


def filter_buffer(buffer: torch.Tensor, keep_indices: torch.Tensor, buffer_size: int):
    size = keep_indices.numel()
    new_buffer = torch.empty(
        (size * buffer_size), device=buffer.device, dtype=buffer.dtype
    )
    buffer_block = triton.next_power_of_2(buffer_size)
    assign_buffer_kernel[(size,)](
        buffer, new_buffer, keep_indices, buffer_size, buffer_block
    )
    return new_buffer
