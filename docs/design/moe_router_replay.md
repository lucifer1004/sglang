# MoE Router Replay Design

## 背景与目标

当前 `return_routed_experts` 已经可以把每个 MoE layer 的 router top-k expert ids 返回给请求方。这个设计要补上反向路径：请求方把一次请求的 routed experts 再传回 `/generate`，服务端在后续模型 forward 中跳过真实 router 选择结果，强制使用请求给定的 expert ids 做 replay。

第一版目标：

- 只覆盖原生 `/generate` 和 Engine generate 路径。
- replay 是显式 opt-in 功能，服务端必须开启 `--enable-moe-router-replay` 才接受请求中的 `routed_experts`。
- 只 replay expert ids，不 replay top-k weights。weights 仍基于当前 router logits 对 forced ids 重新 gather 和归一化。
- 支持普通 prefill、chunked prefill、decode，以及 decode CUDA graph。
- 不支持混合 batch：同一个 `ForwardBatch` 不能同时包含 replay 请求和非 replay 请求；同一个 batch generate 请求也不能部分 item 有 `routed_experts`、部分没有。
- 不支持 speculative decode、TBO、以及 chunked prefill 与 running decode 混合成 `ForwardMode.MIXED` 的 replay 路径。replay 请求遇到这些路径应禁用混合或 fail fast。

## Public API 类型

在 `GenerateReqInput` 中增加请求字段：

```python
routed_experts: Optional[
    Union[
        List[List[List[int]]],        # 单请求: [router_seq_len, num_layers, top_k]
        List[List[List[List[int]]]],  # batch 请求: [batch, router_seq_len, num_layers, top_k]
    ]
] = None
```

字段语义：

- 第一维 `router_seq_len` 是本请求中实际经过 MoE router 的 token 序列长度，包含 prefill token 和 decode token。
- 第二维 `num_layers` 是 MoE router 层数，第一版要求等于 `model_config.hf_text_config.num_hidden_layers` 中实际会捕获 routed experts 的层数。
- 第三维 `top_k` 是每个 token 每层选择的 routed expert 数量，要求等于 `model_config.hf_text_config.num_experts_per_tok`。
- 这里的 expert id 是逻辑 routed expert id，不包含 fused shared expert id。已有 `_post_process_topk_ids()` 仍负责后续 logical-to-physical 映射和 shared expert 处理。

不要把请求字段设计成 base64 tensor。base64 可以继续作为当前返回格式的一种优化，但 replay 请求的 public schema 使用结构化三维数组，便于校验、调试和跨语言客户端构造。

## 内部数据类型

不要复用 `Req.routed_experts`，它当前是返回值。新增 replay 专用字段：

```python
class Req:
    router_replay_experts: Optional[torch.Tensor]
    # CPU int32, shape [router_seq_len, num_layers, top_k]
```

`TokenizerManager` 只做轻量结构归一化；真正依赖模型配置的校验放在 Scheduler 创建 `Req` 时执行：

- `dtype` 转为 `torch.int32`。
- 必须是 rank-3 tensor。
- `shape[1] == num_layers`。
- `shape[2] == num_experts_per_tok`。
- expert id 必须在 `[0, num_logical_routed_experts)`。
- batch 请求中要么所有 item 都有 `routed_experts`，要么都没有；不允许部分有、部分没有。

如果 replay trace 比实际 forward 需要的 token 短，不在请求入口一次性强行判断完整生成长度，而是在构建每个 forward batch 时按当前 token offset 检查。这样 early stop 不会要求请求方提供多余 decode trace。

## ForwardBatch 表示

`ForwardBatch` 不保存 request 级完整 trace，而只保存已经和当前 forward rows 对齐的 tensor：

```python
class ScheduleBatch:
    router_replay_topk_ids: Optional[torch.Tensor]
    # GPU int32, shape [num_forward_tokens, num_layers, top_k]

    router_replay_mask: Optional[torch.Tensor]
    # GPU bool, shape [num_forward_tokens]

class ModelWorkerBatch:
    router_replay_topk_ids: Optional[torch.Tensor]
    router_replay_mask: Optional[torch.Tensor]

class ForwardBatch:
    router_replay_topk_ids: Optional[torch.Tensor]
    router_replay_mask: Optional[torch.Tensor]
```

设计原则：

- offset 只在 Scheduler / `ScheduleBatch` 层计算。
- MoE `TopK` 层只看当前 row 对齐后的 `ForwardBatch.router_replay_topk_ids[:, layer_id, :]`。
- 由于 v1 不支持混合 batch，如果 `router_replay_topk_ids is not None`，同一个 `ForwardBatch` 中所有真实 rows 都应有 replay ids，mask 主要用于 CUDA graph padding rows。

## Offset 构建规则

### Extend / Prefill / Chunked Prefill

`prepare_for_extend()` 已经有当前 forward 的逻辑 token 区间：

```python
scale = self._get_scale_seq_factor()
logical_prefix_len = len(req.prefix_indices) // scale
extend_len = req.extend_input_len

start = logical_prefix_len
end = logical_prefix_len + extend_len
```

构建 replay tensor：

```python
slice = req.router_replay_experts[start:end]
# shape [extend_len, num_layers, top_k]
```

然后按 `prepare_for_extend()` 中 `input_ids = [r.fill_ids[lpl:] ...]` 的同样顺序 `torch.cat`。这天然支持：

- 无 prefix cache：`start = 0`。
- prefix cache hit：跳过已命中的 prefix，只 replay 当前未 forward 的 token。
- chunked prefill：每个 chunk 的 `prefix_indices`/`extend_input_len` 会推进，因此 slice 自动对应当前 chunk。

如果 `end > req.router_replay_experts.shape[0]`，直接拒绝请求并返回清晰错误，说明 replay trace 不足。

### Decode

`prepare_for_decode()` 中当前 forward 的输入 token 是当前完整序列最后一个 token：

```python
trace_pos = req.seqlen - 1
slice = req.router_replay_experts[trace_pos:trace_pos + 1]
```

这里 `req.seqlen = len(req.origin_input_ids) + len(req.output_ids)`。prefill 后第一个 decode forward 会使用第一个 generated token，因此 `trace_pos` 正好是该 token 在完整序列里的位置。

如果 `trace_pos >= router_seq_len`，说明请求方提供的 replay trace 不足，应 fail fast。

### Mixed Batch 限制

v1 不支持 replay/non-replay 混合，也不支持 prefill replay 与普通 running decode 通过 `mix_with_running()` 合成一个 `ForwardMode.MIXED`。

调度侧需要增加约束：

- waiting queue 组 prefill batch 时，`adder.can_run_list` 必须按 `req.router_replay_experts is None` 分组。
- replay 请求不能和非 replay running batch 进入 `mix_with_running()`。
- 如果当前路径会把 replay 请求放进 `ForwardMode.MIXED`，第一版应禁用 mixed chunk 或直接返回不支持错误。

## MoE TopK Override

在 `python/sglang/srt/layers/moe/topk.py` 的 `select_experts()` 中，正常计算 top-k 后，进入 `_post_process_topk_ids()` 前执行 replay override。

逻辑：

```python
if forward_batch.router_replay_topk_ids is not None:
    forced_ids = forward_batch.router_replay_topk_ids[:, layer_id, :]
    mask = forward_batch.router_replay_mask

    forced_weights = gather_weights_from_current_router_logits(
        router_logits=router_logits,
        forced_ids=forced_ids,
        topk_config=topk_config,
    )

    topk_ids = torch.where(mask[:, None], forced_ids, topk_ids)
    topk_weights = torch.where(mask[:, None], forced_weights, topk_weights)
```

weights 计算要求：

- softmax router：先按当前逻辑得到 scores，再 `scores.gather(1, forced_ids)`。
- sigmoid router：同样从 sigmoid scores gather。
- raw-logits routed backend：按当前 backend 需要的 logits/softmax 规则保持一致。
- 如果 `topk_config.renormalize` 为 true，forced weights 按现有规则重新归一化。

之后继续调用 `_post_process_topk_ids()`，保证：

- `return_routed_experts` 捕获到 replay 后的 ids。
- EPLB logical-to-physical 映射仍在统一位置发生。
- padding region mask 和 shared expert append 仍复用现有逻辑。

对于只能走 `BypassedTopKOutput` 或 `TritonKernelTopKOutput` 且无法接收 forced ids 的 backend，v1 应在请求进入时或 top-k 层 fail fast，错误信息说明该 MoE backend 不支持 router replay。支持 replay 的路径应强制使用 `StandardTopKOutput`。

## CUDA Graph 设计

decode CUDA graph 需要静态输入 buffer。新增 buffer：

```python
class DecodeInputBuffers:
    router_replay_topk_ids: torch.Tensor
    # [max_num_tokens, num_layers, top_k], int32

    router_replay_mask: torch.Tensor
    # [max_num_tokens], bool
```

capture 阶段：

- `ForwardBatch.router_replay_topk_ids` 指向静态 buffer。
- `ForwardBatch.router_replay_mask` 指向静态 mask buffer。
- mask 默认全 false，因此 capture 时仍等价于正常 router。

replay 阶段：

```python
buffers.router_replay_topk_ids[:raw_num_token].copy_(
    forward_batch.router_replay_topk_ids
)
buffers.router_replay_mask[:raw_num_token].copy_(
    forward_batch.router_replay_mask
)
buffers.router_replay_mask[raw_num_token:static_num_token].fill_(False)
```

`CudaGraphRunner.replay_prepare()` 返回的静态 `ForwardBatch` 必须引用这些 buffer。这样图内只执行固定形状的 tensor ops：gather、renormalize、`torch.where`。

piecewise CUDA graph 的 prefill replay 也采用同样模式：在 piecewise runner 的 static buffers 中增加这两个字段，并在 `replay_prepare()` 中 copy 当前 replay slice。

## 分阶段验证计划

### 1. 数据类型与校验单测

覆盖：

- 单请求 rank-3 list 正常转为 CPU int32 tensor。
- batch 请求必须全有或全无 `routed_experts`。
- rank 错误、`num_layers` 错误、`top_k` 错误、expert id 越界。
- replay trace 过短时，在构建对应 forward batch 时返回明确错误。

### 2. Offset 构建单测

用 fake `Req.router_replay_experts` 构造可识别值，例如：

```python
value = token_pos * 10000 + layer_id * 100 + topk_idx
```

分别验证：

- 普通 prefill: slice `[0:prompt_len]`。
- prefix cache hit: slice `[prefix_len:prompt_len]`。
- chunked prefill: 多个 chunk 的 slice 连续且不重叠。
- decode: 每步使用 `req.seqlen - 1`。
- padding rows 的 `router_replay_mask=False`。

### 3. TopK Override 单测

构造小 tensor：

- `router_logits: [num_tokens, num_experts]`
- `forced_ids: [num_tokens, top_k]`

验证：

- replay mask 为 true 的 rows 输出 ids 等于 forced ids。
- weights 来自当前 router logits 对 forced ids 的 gather。
- renormalize 后每 row 权重符合现有 top-k 规则。
- mask 为 false 的 rows 保持原 top-k 结果。

### 4. 非 CUDA Graph E2E

流程：

1. 启动 MoE 模型，打开 `--enable-return-routed-experts`。
2. 第一次 `/generate` 设置 `return_routed_experts=True`，拿到返回 routed experts。
3. 第二次 `/generate` 传入结构化 `routed_experts`，同时设置 `return_routed_experts=True`。
4. 断言第二次返回的 routed experts 与请求传入值逐元素相等。

### 5. Chunked Prefill E2E

启动时设置较小 `--chunked-prefill-size`，使用长 prompt：

- 第一次生成并记录 routed experts。
- 第二次 replay。
- 断言每个 chunk 处理后最终返回 routed experts 完全一致。

### 6. CUDA Graph E2E

不加 `--disable-cuda-graph`，并确保 decode token 数足够进入 CUDA graph replay：

- 先 capture routed experts。
- 再 replay routed experts。
- 断言返回 routed experts 完全一致。
- 增加一次 trace 长度不足的请求，确认 decode 中途 fail fast，而不是 silently fallback。

### 7. 不支持路径测试

覆盖：

- 同一个 batch generate 请求中部分 item 带 replay、部分不带 replay，预期拒绝。
- replay 请求尝试进入 replay/non-replay 混合 `ForwardBatch`，预期调度侧隔离或拒绝。
- replay 请求进入 `ForwardMode.MIXED`，第一版预期拒绝或禁用该 mixed 路径。
- 不支持 `StandardTopKOutput` replay 的 MoE backend，预期返回明确错误。
