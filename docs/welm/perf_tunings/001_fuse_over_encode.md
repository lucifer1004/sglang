# 001 fuse over encode

## Vision

减少 over-encoding 路径里的 `all_reduce` 次数，并把热点局部计算 fuse 成更少、更大的 kernel。

## Background

Welm 的 over-encoding（OE）路径原始实现有两个明显问题：

1. **通信次数偏多**：每个 OE branch 各自做 embedding lookup，再分别进入后续计算，天然不利于把 TP 通信收敛成一次。
2. **局部小 kernel 太碎**：n-gram 构造、hash/mod、localize、embedding lookup、concat 分散在多个 Python / PyTorch / Triton 步骤里，短 kernel 的 launch 开销和中间张量读写会被放大。

这个优化的目标不是改模型语义，而是把 **同样的 OE 数学逻辑** 组织成更适合 GPU 和 TP 的执行方式。

## Scope

当前只对**线上主形状**做 specialized 优化：

- `oe_grams = [2, 2, 3, 3]`
- `oe_dim = 512`
- `oe_vocab_sizes = [16000008, 16000016, 16000024, 16000032]`

非 `2233` 形状仍然保留 generic fallback。

## Implementation Summary

### 1. TP fused path：先本地拼接，再一次 all-reduce

核心思路是把 OE 路径改成：

1. 每个 TP rank 本地计算自己 shard 上的 4 个 OE branch 的 embedding contribution
2. 将 4 个 branch 本地 **concat** 成一个 `(num_tokens, 4 * oe_dim)` 张量
3. 对这个 concat 后的张量做 **一次** `all_reduce`
4. 再做一次 `oe_proj`

这样替代了更碎的 branch-by-branch 路径。

### 2. exact-shape specialized Triton kernel

对线上主形状 `2233`，加入 specialized Triton kernel：

- 输入：
  - `input_ids`
  - `gram2`
  - `gram3`
  - 4 路 embedding weight
- kernel 内完成：
  - `gram=2/3` 的 running ids 构造
  - hash / modulo
  - shard range check
  - 4 路 embedding gather
  - concat write 到 `(num_tokens, 2048)` 输出

也就是把原来分散的：

- n-gram 取值
- hash/mod
- local branch lookup
- concat

收敛到单个局部热点 kernel。

### 3. generic helper 回退到最平凡实现

为了减少维护复杂度，generic helper `_compute_welm_oe_hashed_inputs_fused` 最终回退成与 `main_v056` 同风格的最平凡实现：

1. 先按 gram 深度构造完整 `ngram_inputs`
2. 再按 branch 选择对应 gram 并做 `hash_and_localize`

这样 generic 路径更直接，可读性也更高。复杂优化只留在 `2233 + embedding lookup` 这条明确有收益的 specialized 路径上。

## Runtime Conditions for Specialized Path

specialized path 只在以下条件同时满足时启用：

- `oe_grams == [2, 2, 3, 3]`
- CUDA tensor
- 4 个 OE branch
- unquantized contiguous embedding weights
- no vocab padding / no added vocab padded elements

否则回退到 generic fallback。

## Test Data and Measurement Method

### End-to-end serving benchmark

服务端吞吐 benchmark 使用：

- dataset: **MMLU**
- prompts: `100`
- concurrency: `32`
- max decode tokens: `128`

这个配置的目的不是做长上下文极限压测，而是用更接近真实请求分布的 prompt 长度来观察：

- prefill throughput
- total throughput
- TTFT
- E2E latency

### Runtime dump for kernel-level benchmarking

为了测 lookup fused kernel 的真实工作负载，运行时会 dump 线上实际输入。当前观察到的典型 shape：

- `welm_oe_inputs_rank0_512.pt`
  - input shape: `(512,)`
  - output shape: `(512, 2048)`
- `welm_oe_inputs_rank0_2581.pt`
  - input shape: `(2581,)`
  - output shape: `(2581, 2048)`

这些 dump 用来做局部 kernel 的 replay benchmark，而不是只依赖合成输入。

### Short-kernel timing methodology

短 kernel 很容易被 launch overhead 干扰，因此 perf test 做了两件事：

1. **先用若干个 `4096 x 4096` bf16 matmul 预热 GPU**
   - 目的是把 GPU 拉到稳定工作状态，避免把冷启动状态当成 kernel 本身的时间
2. **用 CUDA graph replay + CUDA event 计时**
   - 先 capture 一组真实 shape 的 workload
   - replay 多次后再计算平均时间
   - 尽量减少 Python launch 和短 kernel 调度噪声

## Kernel-level Result

在线上热 shape `welm_oe_inputs_rank0_2581.pt` 上，当前 specialized lookup kernel 的 perf test 结果：

- `avg_ms = 0.2240`
- `read_tbps = 1.5104`
- `effective_hbm_tbps = 3.0207`

这里：

- `read_tbps` 只统计 4 路 embedding weight read
- `effective_hbm_tbps` 统计 **读 + 写** 的总 HBM traffic

对当前这个 kernel 来说，`effective_hbm_tbps` 更接近真实 roofline，因为它不是纯读 kernel，还必须把 `(num_tokens, 2048)` 输出写回 HBM。

## End-to-end Result

### main_v056 plain / no tp_fuse

基线是 `main_v056` 的 plain over-encoding 实现：

- Total time: `7.61 s`
- Request throughput: `13.13 req/s`
- Prefill throughput: `1837.68 tok/s`
- Decode throughput: `1681.24 tok/s`
- Total throughput: `3518.92 tok/s`
- Mean TTFT: `368.25 ms`
- Mean E2E: `1997.62 ms`

### current branch tp_fused + Triton

当前优化版：

- Total time: `6.77 s`
- Request throughput: `14.78 req/s`
- Prefill throughput: `2067.79 tok/s`
- Decode throughput: `1891.76 tok/s`
- Total throughput: `3959.55 tok/s`
- Mean TTFT: `208.93 ms`
- Mean E2E: `1831.85 ms`

### Diff

当前分支相对 `main_v056` plain：

- Request throughput: **+12.6%**
- Prefill throughput: **+12.5%**
- Decode throughput: **+12.5%**
- Total throughput: **+12.5%**
- Mean TTFT: **-43.3%**
- Mean E2E: **-8.3%**

## Why the Gain Is Not “Kernel-level Speedup”

局部 kernel 的收益远大于端到端收益，这是预期内的。

原因很简单：

- kernel 级测试只看 OE 局部热点
- 服务端 benchmark 还包含：
  - prefill / decode 主干
  - scheduler / batching
  - 其他 layer 的 compute 和通信

因此最终端到端提升约 `12.5%`，而不是局部 kernel 的数量级加速。

## Design Choice

这个版本最终采取的是**收敛而不是扩张**的策略：

- 默认开启 `2233 + embedding lookup fusion`
- 删掉其他 specialized fuse 分支
- generic helper 保持最平凡实现

这样做的原因是：

1. **只有固定 shape 才能做足够充分的 fuse**。
   `2233` 把 gram 组合、hash/mod、shard check、4 路 embedding gather、concat write 都压进了同一条明确的执行路径里，kernel 足够“厚”，速度收益才稳定。
2. generic shape 很难做到同样程度的 fuse。
   branch/gram 组合一旦变成动态，控制流、地址计算和 launch 组织都会变复杂，kernel 更容易变碎，收益也更不稳定。
3. 收益已经由线上主路径拿到，继续保留多个实验性 specialized 分支只会增加维护成本。
4. 对未来新 shape，generic fallback 仍然可用。

## Current Status

当前默认策略：

- `2233`：走 specialized lookup fusion
- 其他 shape：走 generic fallback

这个版本的目标不是覆盖所有 shape，而是先把**线上最主要 shape** 做到收益明确、结构收敛、可维护。
