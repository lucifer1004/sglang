# FA3 vs FlashInfer Decode Attention Microbench

Date: 2026-06-08

## Scope

This benchmark measures single decode attention backend op latency only. It does not run the SGLang server or the full model layer stack.

- Model config: `/home/josephyu/models/config.json`
- Model: `WeLMV4MoeForCausalLM`, `model_type=welmv4_moe`
- Dtype: `bfloat16`
- Total attention heads: `24`
- Total KV heads: `2`
- Head dim: `256`
- Batch size: `1`
- Decode tokens per request: `1`
- KV lengths: `1024` to `131072`, doubling each step
- Page size: `1`, matching SGLang's default CUDA page size
- Timing: CUDA events around the backend op loop. Metadata, planning, allocation, random initialization, and synchronization outside the op loop are excluded.
- Raw result JSON: `docs/microbenchs/attn/fa3_flashinfer_decode_results.json`

TP is simulated as the per-rank attention shape, without launching multi-process TP or NCCL:

| TP | Q heads per rank | KV heads per rank |
|---:|---:|---:|
| 1 | 24 | 2 |
| 2 | 12 | 1 |
| 3 | 8 | 1 |
| 4 | 6 | 1 |

The measured ops are:

- FA3: `sgl_kernel.flash_attn.flash_attn_with_kvcache(...)`
- FlashInfer: `flashinfer.BatchDecodeWithPagedKVCacheWrapper.forward(...)`

Command:

```bash
PYTHONPATH=/home/josephyu/sgl-attn-speed/python \
gpu-lease run --count 1 --wait -- \
  /home/josephyu/sglang/.venv/bin/python \
  benchmark/kernels/attention/bench_fa3_flashinfer_decode.py \
  --output docs/microbenchs/attn/fa3_flashinfer_decode_results.json
```

Environment:

- GPU: `NVIDIA H20`
- CUDA: `12.8`
- PyTorch: `2.11.0+cu128`
- FlashInfer: `0.6.8.post1`
- sgl-kernel: `0.4.2.post2`
- Warmup: `25`
- Iterations per trial: `200`
- Trials: `5`

## Results

Latency values are median single-op latency in microseconds. `FI speedup` is `FA3 median / FlashInfer median`.

| TP | KV len | FA3 us | FlashInfer us | FI speedup |
|---:|---:|---:|---:|---:|
| 1 | 1024 | 26.186 | 18.563 | 1.41x |
| 1 | 2048 | 23.342 | 18.701 | 1.25x |
| 1 | 4096 | 23.282 | 18.493 | 1.26x |
| 1 | 8192 | 33.979 | 18.739 | 1.81x |
| 1 | 16384 | 51.874 | 23.093 | 2.25x |
| 1 | 32768 | 89.081 | 38.332 | 2.32x |
| 1 | 65536 | 158.341 | 58.968 | 2.69x |
| 1 | 131072 | 297.216 | 97.771 | 3.04x |
| 2 | 1024 | 23.163 | 18.982 | 1.22x |
| 2 | 2048 | 22.998 | 19.077 | 1.21x |
| 2 | 4096 | 23.603 | 18.486 | 1.28x |
| 2 | 8192 | 26.539 | 18.646 | 1.42x |
| 2 | 16384 | 38.393 | 19.882 | 1.93x |
| 2 | 32768 | 56.054 | 26.771 | 2.09x |
| 2 | 65536 | 92.782 | 41.176 | 2.25x |
| 2 | 131072 | 162.186 | 61.530 | 2.64x |
| 3 | 1024 | 22.886 | 18.514 | 1.24x |
| 3 | 2048 | 22.890 | 18.638 | 1.23x |
| 3 | 4096 | 23.609 | 18.907 | 1.25x |
| 3 | 8192 | 23.854 | 18.736 | 1.27x |
| 3 | 16384 | 35.583 | 19.567 | 1.82x |
| 3 | 32768 | 53.550 | 26.049 | 2.06x |
| 3 | 65536 | 91.254 | 40.444 | 2.26x |
| 3 | 131072 | 160.490 | 61.207 | 2.62x |
| 4 | 1024 | 22.823 | 18.316 | 1.25x |
| 4 | 2048 | 22.950 | 18.507 | 1.24x |
| 4 | 4096 | 23.152 | 19.306 | 1.20x |
| 4 | 8192 | 26.187 | 19.393 | 1.35x |
| 4 | 16384 | 37.956 | 19.505 | 1.95x |
| 4 | 32768 | 55.495 | 25.621 | 2.17x |
| 4 | 65536 | 92.667 | 40.367 | 2.30x |
| 4 | 131072 | 161.999 | 61.239 | 2.65x |

## Conclusion

FlashInfer is faster than FA3 for every measured single-token decode attention shape in this WeLM config.

For short KV cache lengths, `1K` to `4K`, FlashInfer is still consistently ahead, but the margin is modest: about `1.20x` to `1.41x`. This range is close to the fixed per-op overhead floor, so the absolute gap is usually only `4` to `8 us`.

The difference becomes material from `16K` onward. At `16K`, FlashInfer is already about `1.82x` to `2.25x` faster. At `128K`, FlashInfer is about `2.62x` to `3.04x` faster.

TP changes the per-rank head shape. TP=1 keeps two local KV heads and shows the largest long-context gap: `297.216 us` for FA3 vs `97.771 us` for FlashInfer at `128K`. TP=2/3/4 all use one local KV head in this model config and converge to similar long-context behavior: FA3 is about `160` to `162 us`, while FlashInfer is about `61 us` at `128K`.

For this model's decode path, when the goal is single-token long-context attention kernel latency, `flashinfer` should be preferred over `fa3`.
