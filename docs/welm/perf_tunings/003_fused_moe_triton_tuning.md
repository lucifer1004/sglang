# 003 fused moe triton tuning

## Insight

```insight
当前 SGLang 在推理 WeLMV4 时，会出现如下日志：
[2026-04-22 16:35:28 TP7 EP3] Using default MoE kernel config. Performance might be sub-optimal! Config file not found at /envs/train/lib/python3.11/site-packages/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=32,N=512,device_name=NVIDIA_H20.json, you can create them with https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton
[2026-04-22 16:35:28 TP7 EP3] Using MoE kernel config with down_moe=False. Performance might be sub-optimal! Config file not found at /envs/train/lib/python3.11/site-packages/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=32,N=512,device_name=NVIDIA_H20_down.json, you can create them with https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton

发现是在 TP=32, EP=16 的部署下，在 H20 上没有 tuned config 导致，使用 benchmark/kernels/fused_moe_triton 中的 tuning 工具生成最佳的 kernel config。

当前使用已有的 tuning_fused_moe_triton.py 时，在H20机器上会阻塞在 ray worker 上，导致tune过程无法进行。使用 tune_welm_moe.py 绕过ray调度机制，直接对 welm 600B 的模型进行 tuning。
```

## Vision

为 WeLM 在 H20 上的 fused MoE Triton kernel 做一次完整的 per-shape tuning，并把
产出的配置文件按 SGLang 启动期查找路径归档，让线上部署一启动就能命中专门调过的
`BLOCK_SIZE_{M,N,K}` / `GROUP_SIZE_M` / `num_warps` / `num_stages`，而不是走默认
fallback。

这轮的目标不是改 kernel 逻辑，而是把**已有 Triton MoE kernel 的 autotune 过程**
做得更工程化：

- 让 tuning 在单机多卡上并行，而不是串行跑满 batch 扫描
- 让新模型只需要在 registry 里填 HF config 字段即可复用同一套流程
- 把 H20 上 welm_600b 的 tuned config 直接 checkin 到 SGLang 的 config 目录

## Background

SGLang 的 fused MoE Triton 路径在启动时会去
`python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_<ver>/`
目录里查 `E=<num_experts>,N=<shard_intermediate_size>,device_name=<gpu>.json`。
如果命中，就用里面按 batch size 索引的 best config；命中不到则走 Triton 自身的
默认 heuristic。

对 welm_600b 在 H20 + `tp=32, ep=16` 部署下，启动期查找的形状是
`E=32, N=512`，原本这条路径在 H20 上没有 tuned config，跑的就是 fallback。这次
tuning 的出发点就是把这两个形状（gate/up 和 down）补齐。

已有的 `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` 本身具备
autotune 能力，但存在几个工程问题：

1. **模型识别不完整**：`common_utils.get_model_config` 没有 `WeLMV4MoeForCausalLM`
   分支，无法从 HF config 直接推出 `E / N / topk / dtype`。
2. **全局 server args 没有初始化**：Ray worker 里调用 `get_moe_configs()` 会去
   读 `enable_deterministic_inference`，但默认 ServerArgs 没有被注册，worker 启动
   就报错。
3. **没有快速 smoke 入口**：原脚本没有限制 search space 的开关，改一次逻辑就得
   扫完整个配置空间才能验证，迭代成本偏高。
4. **Ray-based 调度对单机场景过重**：只想在一台机 8 卡上 tune 的时候，Ray 的启动、
   调度和 tmpdir 管理都是额外负担；且 batch 分配策略是 round-robin，对"大 batch
   耗时远大于小 batch"这种实际负载并不是最优。

这轮优化就是绕着这四点做收敛。

## Scope

当前这一轮只覆盖：

- **模型**：welm_600b（即 `WeLMV4MoeForCausalLM`）
- **硬件**：NVIDIA H20
- **并行**：`tp=32, ep=16`（对应 MoE shape `E=32, N=512`）
- **精度**：bfloat16，无 block-wise quant
- **产出**：gate/up 投影和 down 投影共享同一份 tuned config

其他模型 / 其他硬件 / 量化路径都不在这轮范围内，但 tuner 设计成了“填 registry 就
能用”的形式，便于后续扩展。

## Implementation Summary

### 1. 新增单机多卡 tuner `tune_welm_moe.py`

核心思路：**不用 Ray，直接用 `multiprocessing.spawn` 起 N 个子进程，每个子进程
绑定一张 GPU**。每个子进程：

1. 在 `import torch` 之前设置 `CUDA_VISIBLE_DEVICES=<gpu_index>`，保证后续一切
   torch 操作都发生在目标设备上。
2. 注册默认 `ServerArgs` 到全局，让下游 `get_moe_configs()` 能正常读开关。
3. 按分配到的 batch 子集，对每个 batch 遍历 search space 找 best config。
4. 把 best config 以 JSON partial 的形式写到 workdir。

主进程在所有 worker 退出后 merge partials，生成最终的 `E=..,N=..,device_name=..json`。

### 2. Longest-Processing-Time 调度

观察到 `batch=1` 和 `batch=4096` 单条 config 的耗时差异是量级的，所以 batch 分配
不能用 round-robin。脚本改成 LPT 贪心：

- 按 batch 从大到小排序
- 依次放进当前负载最轻的 shard

这样能把"大 batch 扎堆"的尾部时间压下来，8 卡的整体墙钟更接近理想并行。

### 3. Model registry：构造期只依赖 HF config 字段

`ModelSpec` 只记录模型**内在**字段：

```python
total_experts / topk / hidden_size / moe_intermediate_size / torch_dtype / block_shape
```

并行相关的 `tp_size / ep_size` 作为 CLI 参数传入。kernel 真正需要的 `E / N`
由一个显式公式推出：

```
E = total_experts / ep_size
N = 2 * moe_intermediate_size / (tp_size / ep_size)
```

这样新加一个 WeLM 变体只需要往 `MODELS` 字典里追加一项，不用动执行逻辑。

### 4. 完善已有 `tuning_fused_moe_triton.py`

在不改变原 Ray-based 入口行为的前提下做了三处收敛：

- `common_utils.get_model_config` 增加 `WeLMV4MoeForCausalLM` 分支，让 HF config
  能被正确解析成 `E / topk` 等字段。
- Ray worker 启动时调用 `set_global_server_args_for_scheduler(ServerArgs(...))`，
  确保 `get_moe_configs()` 能读到 `enable_deterministic_inference`。
- 加 `--max-configs N` 作为**开发/调试工具**：把 search space 截到前 N 条，用于
  快速验证链路是否跑通。正式 tuning 不使用这个 flag。

### 5. 产出的 config 文件

这轮 tuning 最终 checkin 的两个文件：

```
python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/
├── E=32,N=512,device_name=NVIDIA_H20.json 
```

命名和 SGLang 启动期查找路径保持一致，部署时无需额外配置。

## Current Status

当前默认策略：

- welm_600b 在 H20 + `tp=32, ep=16` 部署下：走 checked-in 的 tuned config
- 其他 shape / 其他硬件：继续走 Triton 默认 heuristic（fallback）
- 新模型接入：在 `tune_welm_moe.py` 的 `MODELS` 注册表里追加一项，指定 tp/ep 后
  直接跑 tuner

这一轮交付的边界很明确：**只补齐 H20 上 welm_600b 的 MoE config，并把 tuning 流
程本身沉淀成可复用的工具**，不触碰 kernel 实现。

### 当前的吞吐提升
- 32 卡 H20：welm_600b 从 15.414 tok/seq/sec 提升到 15.495 tok/seq/sec
