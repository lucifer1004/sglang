import torch
import triton
import triton.language as tl
import cutex

from typing import List

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
    tl.store(input_ids_gram + data_offset + gram_offset, data, data_offset+gram_offset < extend_len)
    prefix_offset = tl.arange(0, GRAM_BLOCK_SIZE)
    tl.store(input_ids_gram + prefix_offset, 0, prefix_offset < tl.minimum(gram_offset, extend_len))


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
        assign_ngram_input_ids_kernel[(grid, )](input_ids[pt : pt + extend_len], input_ids_gram[pt : pt + extend_len], gram_n, 
                                                128, GRAM_BLOCK_SIZE, extend_len)
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
    save_offset =  tl.arange(0, topk_block)
    tl.store(input_ids_gram_decode + bid * topk + save_offset, repeat_data, save_offset < topk)


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
    assign_ngram_input_ids_draft_decode_first_token_kernel[(bs, )](input_ids_buffer, input_ids_gram_decode, buffer_size, gram_n, topk, topk_block, bs_block)
    
    
    
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
    assign_ngram_buffer_kernel[(bs, )](input_ids, buffer, seq_lens, buffer_size, bs_block, buffer_size_block)

    


kernels = cutex.SourceModule(
    """
//cuda
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
            ngram_input_ids[bid * topk + tid] = buffer[(bid+1) * buffer_size - gram - 1];
            return;
        }
        long parrent_token;
        for(int gram_ids=0; gram_ids<gram_n-1; gram_ids++){
            int pre_layer_num_node = topk + topk*topk*(i-1);
            int cur_layer_pos = current_pos - pre_layer_num_node;
            int parent_layer_pos = cur_layer_pos / topk;
            int parrent_pre_layer_num_node = i == 1 ? topk : pre_layer_num_node - topk*topk;
            long parrent_layer_pos = parent_layer_pos + parrent_pre_layer_num_node + 1;
            parrent_token = token_list[bid][parrent_layer_pos];
            i--;
        }
        ngram_input_ids[bid * topk + tid] = parrent_token;
        return;
}
//!cuda
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
    i: int
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

