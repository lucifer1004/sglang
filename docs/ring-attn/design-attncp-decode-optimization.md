# AttnCP Decode ITL 优化设计

**状态**: Draft for review  
**日期**: 2026-06-24  
**范围**: TP4-CP2 sharded-KV AttnCP decode 阶段 ITL 优化  
**非目标**: 本文不改变 sharded-KV cache residency 语义,不优化 prefill 主路径,不改变输出精度标准

---

## 1. 背景

当前 TP4-CP2 sharded-KV AttnCP 已经验证:

| 场景                                         |       NaiveTP4 |        TP4-CP2 | 结论           |
| -------------------------------------------- | -------------: | -------------: | -------------- |
| 32k input / 2k output 最大常驻并发           |             23 |             45 | 容量 `1.96x`   |
| 32k input / 2k output 最佳 output throughput | `448.54 tok/s` | `651.20 tok/s` | 吞吐 `1.45x`   |
| 8k input / 512 output / c4 mean ITL          |      `8.78 ms` |     `10.61 ms` | ITL 慢约 `21%` |

容量收益符合预期,但 decode ITL gap 仍然偏大。后续优化目标是:

- 保持 persistent KV cache sharded,不回退到 replicated KV。
- 缩短 TP4-CP2 相比 NaiveTP4 的 decode ITL gap。
- 优先优化 decode 阶段的通信量、通信等待和 merge/copy 开销。
- 不先动 prefill correctness path。

---

## 2. 当前 Decode 路径

当前 sharded-KV decode 不重建 full KV,每个 CP rank 只读本 rank owned KV shard。

以 TP4-CP2 为例:

```text
global TP ranks: 0, 1, 2, 3
CP groups:
  group A: rank0, rank1
  group B: rank2, rank3

每个 rank:
  q_local: 当前 TP shard 的 Q heads
  local KV cache: 只包含本 CP rank owner 的 sequence chunks
```

decode 单层 attention 当前流程:

```text
q_local
  -> CP all-gather on head dim
  -> q_full

q_full + local KV shard
  -> flash_attn_with_kvcache(return_softmax_lse=True)
  -> local_o_full, local_lse_full

local_o_full / local_lse_full
  -> CP all-gather
  -> gathered partial states from every CP rank
  -> merge_state_v2 across CP shards
  -> slice back to this rank's local Q heads
  -> o_proj / TP all-reduce / following blocks
```

当前路径的关键性质:

- KV 不移动: decode 阶段省显存的核心语义成立。
- Q 需要在 CP group 内 all-gather,因为每个 CP rank 要为 full Q heads 计算本地 KV shard 的 partial attention。
- O/LSE 也需要跨 CP rank 交换,因为每个 Q head 的最终 softmax 结果来自所有 KV shards 的 partial state merge。
- merge 使用 online softmax state:

```text
state_i = (o_i, lse_i)
final = merge_state_v2(state_0, state_1, ...)
```

当前主要开销点:

1. `q_allgather`
2. `local_flash_attn`
3. `o_lse_allgather`
4. `merge_state_v2`
5. head slice / copy / flatten
6. 后续 `o_proj` 和 TP all-reduce

其中 `q_allgather` 数据量较小;`o_lse_allgather + merge/copy` 更值得优先关注。

---

## 3. 为什么不先优化 Prefill

prefill 当前走 correctness-first 路径:

```text
sharded persistent KV
  -> temporary dense full KV reconstruction
  -> normal flash_attn_with_kvcache
```

这条路径的缺点是 TTFT 开销较大,但优势是:

- 复用普通 FA causal mask / varlen metadata。
- chunked prefill、kv mirror、attention sink、over encoding 的语义更稳。
- 精度更接近 NaiveTP。

如果 prefill 改成 decode 类似的 local partial attention + merge,需要处理每个 query token 对本地 KV shard 的 causal visibility:

```text
visible(q_pos, local_kv_pos) = local_kv_pos <= q_pos
```

这会把复杂度转移到:

- per-owner-chunk causal mask
- chunked prefill 当前 chunk 与 cached prefix 的边界
- batched requests 不同 seq lens
- attention sink 只注入一次
- SWA/local window
- kv mirror / over encoding
- partial LSE merge 的数值顺序

因此本文先只优化 decode ITL。

---

## 4. 优化路径总览

| 优先级 | 方案                                         | 目标                                               | 风险 | 预期收益                                       |
| -----: | -------------------------------------------- | -------------------------------------------------- | ---- | ---------------------------------------------- |
|     P0 | decode profiling 分解                        | 确认瓶颈来源                                       | 低   | 指导后续优化                                   |
|     P1 | O/LSE full all-gather 改 head-slice exchange | 减少 O/LSE 通信量和 workspace                      | 中   | CP2 约减少一半 O/LSE 通信                      |
|     P1 | CP2 专用 fast path                           | 去掉通用 all-gather/view/loop 开销                 | 中   | 当前主场景最快落地                             |
|     P2 | merge_state + slice + flatten fusion         | 减少小 kernel 和 copy                              | 中   | 降低 merge/copy 时间                           |
|     P2 | Q rotation pipeline overlap                  | 通信/计算 overlap                                  | 中高 | 降低 Q 通信等待和 full tensor materialization  |
| P2-exp | Triton Q+FA fusion                           | 单 kernel 内加载本地/peer Q 并单次扫描 resident KV | 高   | 理论上减少 Q materialization 和 FA launch/读写 |
|     P3 | attention output / gate / o_proj fusion      | 减少大 tensor 读写                                 | 高   | 进一步压低 ITL                                 |
|     P3 | metadata 维护优化                            | 降低每步 metadata 构造成本                         | 中   | 长上下文 batch 下收益更明显                    |

### 4.1 当前 Fusion 决策

当前已经有一个 CP2 Triton Q+FA fusion 原型,但只能作为实验路径:

```bash
--attn-cp-decode-fused-q-fa
```

用户侧只暴露一个 fused 开关。kernel 的最小 graph seq cap 目前固定为
`16384`;split workspace 由 CUDA graph sequence cap 按 4096 local-KV tokens
粒度自动推导,不再暴露 `max_seq_cap` 或 `max_splits` 调试参数。

该 kernel 的调度粒度是 `(batch, kv_head, kv_split)`,不是 Q head,因此满足
decode 阶段的核心省显存语义:

```text
每个 resident KV split 只被 owner rank 扫一次,
同一 K/V tile 在 SRAM 内复用给 local Q heads 和 peer Q heads。
```

这个 invariant 是 fused decode path 的硬约束,不是性能建议:

- 主 attention kernel 的 launch grid 只能包含 `batch`, `kv_head`,以及可选的
  `kv_split`;不能增加 `q_head` grid 维度。
- 对同一个 `kv_head` group,`full_q_heads_per_kv` 必须能放进同一个 program 的
  `BLOCK_H`;当前实现上限是 `16`。超过上限时必须 fallback,不能把 Q heads
  拆成多个 attention kernel。
- `kv_split` 只能切 sequence range。不同 split 读取互不重叠的 KV range,后续
  merge 只读 partial `O/LSE`,不再读取 K/V。
- remote Q 可以通过 P2P/all-gather/peer pointer 获得,但不能通过重复跑
  `FA(local_q, KV)` 和 `FA(peer_q, KV)` 来实现,否则同一份 resident KV shard
  会被读两遍。

当前代码在 `attncp_cp2_fused_q_fa_decode()` 入口显式检查
`attncp_cp2_fused_q_fa_supports_shape()`。不满足 KV-stationary shape 时会报错或
由服务层 fallback 到 FA3 exact path,避免未来为了兼容新模型误引入 Q-head split。

split 策略必须避免 CUDA graph seq-cap 改变前缀 KV 的归约边界。当前实现优先使用
固定 `4096` local-KV token split size;服务启动时根据 CUDA graph sequence cap
自动分配 split workspace。`32k input / 2k output` 使用 `40960` cap 时,内部会
分配足够的 split 槽位来完整覆盖该 cap,同时前 `32768` token 的 split 边界保持在
`4096` 粒度。split 之间的 KV range 仍然互不重叠,merge 阶段只读 partial O/LSE,
不再读取 K/V。

2026-06-26 复测显示,固定 split 后 `40960` cap 的 32k hot probe 能真实命中
`hit_split`,输出 token 和 exact TP4+CP2 一致,但 strict output-logprob 仍有
`2.847e-02` max diff。因此该策略只降低 drift,不能作为 fused path 已经数值等价
FA3 的证据。

后续把 fused kernel 的 online softmax 改成 FA3 风格 `exp2/log2` 后,同一
`40960` cap hot probe 的 exact-vs-fused strict max diff 降到
`8.621e-03`,但仍未达到 `1e-5` strict gate。split partial merge 按 FA3
combine 的源码路径保留自然底 `exp/log`,因为 split LSE 本身已经是 natural
logsumexp。该实现可以作为当前更好的实验版本保留,但不能声称已经与 FA3 数值等价。

本轮还验证了几个看似可行但无效的收敛方向:

- 只 fused 前半层 `0-23`: exact-vs-fused strict max diff `1.260e-01`。
- 只 fused 后半层 `24-47`: exact-vs-fused strict max diff `2.273e-01`。
- 单 split: exact-vs-fused strict max diff `6.964e-02`。
- graph cap 从 `40960` 收紧到 `34816`: strict max diff 仍是 `8.621e-03`。
- 强制 FA3 `num_splits=4`: exact-vs-fused strict max diff `1.030e-01`。
- 倒序扫描 KV block 以模仿 FA3 mainloop: exact-vs-fused strict max diff
  `1.243e-02`,比当前最佳更差。
- attention sink 改成 FA3-style finalize: exact-vs-fused strict max diff
  `3.721e-01`。
- raw QK score 域做 row max: exact-vs-fused strict max diff `9.257e-02`。
- Triton QK dot 显式 `input_precision="ieee"`:本地 WeLM 形状单测通过,但
  16k local shard O mean diff 没有改善,已回退。

这些结果说明当前 fused path 的误差存在层间抵消;简单 layer fallback、单 split、
强制 split 数、调整 KV 访问顺序、attention sink finalize 位置、raw/scaled score
域切换、QK dot precision hint、或继续收紧 graph cap 都不能达成 strict parity。
本地 CP0/CP1 双 shard 诊断显示,merged O diff 基本是对称的一 BF16 ulp 级误差
(`max ~= 2.44e-04`,mean 约 `2e-05`),LSE diff 约 `1e-06`;这不像 CP shard
merge 语义错误或可用全局 bias 修正的系统误差。后续如果必须满足 `1e-5`
strict logprob,更可靠的方向是把 Q exchange/remote-Q load 融进 FA3/CUDA
实现,或者复刻 FA3 内部更完整的 mainloop 和 rounding 路径,而不是继续微调
外部 Triton attention 近似实现。

本轮优化收敛为纯 Triton Q+FA fusion,不改 `sgl-kernel`,也不引入
`sgl-attn`/FA3 internal Q-provider。原因是当前目标是先把 decode 中
Q all-gather + local FA3 的替换边界做干净,验证独立 Triton kernel 的速度收益。

当前实现边界:

- `_attncp_try_fused_q_fa_decode(...)` 是 SGLang decode AttnCP 路径里的唯一替换点。
- 该函数内部先交换 peer Q,再调用
  `attncp_cp2_fused_q_fa_decode(...)`。
- `attncp_cp2_fused_q_fa_decode(...)` 位于
  `python/sglang/srt/layers/attention/attncp_fused_ops.py`,由 Triton JIT 编译。
- 不需要修改或重新编译 `sgl-kernel`。
- 不向 `python/sglang/jit_kernel/flash_attention*.py` 暴露新 wrapper。

这条 Triton 路径继续保持 KV-stationary 约束:

- non-split launch grid 是 `(batch, kv_head)`。
- split launch grid 是 `(batch, kv_head, split)`。
- grid 中没有 Q-head 维度,不能通过拆 Q head 来绕过寄存器压力。
- 每个 resident KV tile 只在对应 `(batch, kv_head[, split])` program 中读取一次,
  同一个 program 内复用到该 KV head 对应的所有 CP Q heads。

因此,如果 shape guard 不满足,服务端必须 fallback 到 FA3 exact path,不能自动拆成
多个 Q-head program。当前 guard 固定为:

```text
cp_world_size == 2
next_power_of_2(q_heads_per_kv) <= 16
page_size == 1
v_head_dim == head_dim
decode q_len == 1
```

strict logprob parity 暂不作为本轮 Triton fusion 的通过条件。当前 Triton fused path
的目标是 token-level correctness 和吞吐/ITL 收益;如果后续必须达到 `1e-5`
strict logprob,需要另开 FA3/CUDA internal Q-provider 或完整复刻 FA3 mainloop,
但那是另一条需要 `sgl-kernel`/`sgl-attn` 编译的路线,不放入本轮代码。

2026-06-26 纯 Triton focused speed probe:

```bash
PYTHONPATH=python CUDA_VISIBLE_DEVICES=0,1 .venv/bin/torchrun \
  --standalone --nproc_per_node=2 \
  benchmark/kernels/attention/bench_attncp_decode_paths.py \
  --batch-size 16 \
  --kv-lens 32768 \
  --tp-size 4 \
  --cp-size 2 \
  --cp-kv-chunk-size 1024 \
  --warmup 3 \
  --iters 20 \
  --trials 2 \
  --fused-max-splits 8 \
  --cuda-graph
```

| path                        | median us | fused/fullq speedup | max output diff |
| --------------------------- | --------: | ------------------: | --------------: |
| `target_sharded_fullq`      | `527.027` |            baseline |      `2.44e-04` |
| `target_sharded_fused_q_fa` | `175.360` |             `3.01x` |      `2.44e-04` |

非 CUDA graph 小 probe 显示收益有边界:

| batch | kv_len | fullq median us | fused median us | fused/fullq speedup | max output diff |
| ----: | -----: | --------------: | --------------: | ------------------: | --------------: |
|     4 |   8192 |       `254.514` |       `238.434` |             `1.07x` |      `2.44e-04` |
|     4 |  32768 |       `246.251` |       `260.723` |             `0.94x` |      `2.44e-04` |
|    16 |   8192 |       `261.642` |       `246.818` |             `1.06x` |      `4.88e-04` |
|    16 |  32768 |       `545.013` |       `268.408` |             `2.03x` |      `2.44e-04` |

结论: Triton fusion 在较大 batch/长上下文 decode bucket 中有明确收益;小 batch
或短 local KV 下不保证更快,因此必须保留 shape/seq-cap guard 和 FA3 fallback。

当前服务级精度状态:

| 路径                              | 覆盖范围                                          | MMLU/C-Eval token mismatch |           max logprob diff | 结论          |
| --------------------------------- | ------------------------------------------------- | -------------------------: | -------------------------: | ------------- |
| TP4+CP2 fallback local-merge FA3  | full regression                                   |                  `0 / 100` |                 `0.00e+00` | 通过          |
| TP4+CP2 Q/O-LSE P2P + FA3         | full regression                                   |                  `0 / 100` |                 `0.00e+00` | 通过          |
| TP4+CP2 guarded Triton Q+FA fused | full regression, short prompt 主要走 FA3 fallback |                  `0 / 100` |                 `0.00e+00` | 通过当前回归  |
| TP4+CP2 hot Triton Q+FA fused     | 32k controlled long-context probe                 |                 token 一致 | `8.621e-03` exact-vs-fused | strict 未达标 |

因此当前默认路径必须继续保留 FA3 attention math 或 guarded fused fallback。
P2P Q/O-LSE exchange 是当前精度安全的优化基线;Triton Q+FA fusion 可以用于
长上下文吞吐实验,但在 strict logprob gate 下仍不能声明与 FA3 完全等价。如果要让
fusion 成为默认无 guard 路径,需要直接对齐 FA3 的数值实现,或者在 FA3/CUDA
kernel 内部完成 Q exchange / remote Q load,而不是用独立 Triton attention
近似替换 FA3。

当前 fused path 额外使用 seq-cap guard:

```text
只在 cp_local_page_table.shape[1] * page_size >= 16384 时启用 fused。
```

这个内部 guard 只依赖静态 CUDA graph bucket shape,不读取 runtime tensor 值,
因此不会破坏 CUDA graph capture。需要注意,这里的 `seq_cap` 是
**local CP KV shard** 的 page-table capacity,不是全局 prompt 长度。对于
TP4-CP2 sharded-KV:

```text
global prompt ~= 32k
local CP shard cap ~= 16k
```

因此当前 `16384` 内部阈值可以覆盖全局 `32k`/CP2 的 hot bucket;短 prompt /
MMLU-C-Eval strict regression 仍会因 graph bucket 小于阈值继续走 FA3 exact
path。这个 guard 的目的不是证明 fused path 数值完全等价,而是避免短上下文下
引入不必要的 Triton fused math 风险。

补充诊断:

- 单 split 仍然失败,说明当前主要问题不是 split-KV merge 顺序。
- 服务语义单测已覆盖 CP2 local merge:
  - CP0 local shard 注入 full attention sinks。
  - CP1 local shard 使用 disabled sinks (`-inf`)。
  - empty local KV rows 显式归一为 `O=0,LSE=sink-or--inf`。
  - 两个 shard 使用 `merge_state_v2` 合并。
- 该单测通过,说明 sink / empty-local / O-LSE merge 语义没有暴露错误。
- 严格单步统计仍显示独立 Triton attention math 和 FA3 不 bitwise-equivalent:
  - merged O max diff 约 `4.88e-04`。
  - merged LSE max diff 约 `9.54e-07`。
- WeLM v4 regression 是 48 层、32 decode tokens 的 strict logprob gate,
  上述 BF16 级别差异会在确定性 decode 中累积并导致 token/logprob drift。

当前 guarded integration 结果:

| 设置                               | controlled compare |                  MMLU/C-Eval | 结论                       |
| ---------------------------------- | -----------------: | ---------------------------: | -------------------------- |
| fused enabled, internal min seq cap `16384` |       `max_diff=0` | `0/100 mismatch, max diff=0` | 通过当前短 prompt 精度回测 |

解释:

- 这个结果依赖 seq-cap guard: 短 prompt regression 继续走 FA3 exact path。
- 它说明 guarded integration 不破坏现有精度测试,但不能证明 Triton fused
  对所有长上下文请求都精度等价。
- 如果要把 Triton fused path 作为默认 decode attention,仍然需要新的长上下文
  precision gate,或者让 fused kernel 在数值上直接对齐 FA3。

当前端到端 sweep 结果:

| fused min seq cap | concurrency | output TPS | mean ITL ms | 结论                      |
| ----------------: | ----------: | ---------: | ----------: | ------------------------- |
|           `32768` |          44 |   `646.07` |     `49.83` | 和 P2P exact 基本持平     |
|           `32768` |          45 |   `665.67` |     `49.77` | 和 P2P exact 基本持平     |
|           `32768` |          46 |   `525.32` |     `49.26` | 超容量点更差              |
|           `16384` |          45 |   `653.58` |     `49.97` | fused 命中更多但更慢      |
|           `16384` |          44 |   `648.52` |     `49.69` | 2026-06-26 fused-hot 复测 |
|           `16384` |          45 |   `664.77` |     `49.94` | 2026-06-26 fused-hot 复测 |
|           `16384` |          46 |   `526.60` |     `49.22` | 超容量点, peak running=45 |

结论:

- 当前 Triton Q+FA fusion 还不能作为缩短 TP4-CP2 ITL gap 的完成方案。
- internal min seq cap `32768` 的历史 sweep 不能证明 fused 对全局 `32k` workload 有收益,
  因为该阈值按 local CP shard 判断,多数 `32k` bucket 会 fallback 到 FA3。
- internal min seq cap `16384` 可以覆盖 global `32k`/CP2 的 hot bucket,但已有诊断显示
  端到端吞吐没有提升,且 unguarded fused math 仍未通过 strict logprob gate。
- 2026-06-26 复测中,internal min seq cap `16384` 的 c44/c45/c46 结果分别为
  `648.52/664.77/526.60 tok/s`,和 FA3/P2P exact 路径相比没有稳定收益。
- isolated attention microbench 仍显示 fused kernel 本身更快:
  - global KV `32768` / local CP KV `16384` / batch `45`:
    `target_sharded_slice_a2a=1251.97 us`,
    `target_sharded_fused_slice_a2a=397.16 us`。
  - local KV `1024` / batch `45`:
    `target_sharded_slice_a2a=139.01 us`,
    `target_sharded_fused_slice_a2a=81.12 us`。
- 因此当前问题不是 fused kernel 完全没有局部收益,而是该收益没有转化成服务级
  output TPS / ITL。后续必须依赖端到端 profile 判断 attention 在完整 decode
  step 中的占比,以及 fused path 在 full-window/SWA/kv-mirror 层上的实际命中分布。
- 32k 长上下文 precision probe 显示:
  - TP4、TP4-CP2 exact、TP4-CP2 fused-hot 的 4 个输出 token 完全一致。
  - TP4 vs TP4-CP2 exact 的 output-logprob max diff 为 `3.153e-02`。
  - TP4 vs TP4-CP2 fused-hot 的 output-logprob max diff 为 `3.538e-01`。
  - TP4-CP2 exact vs fused-hot 的 output-logprob max diff 为 `3.854e-01`。
  - fused-hot 日志确认 `seq_cap=16384 hit_split=3`,说明这是实际 hot path
    命中后的结果,不是 fallback 伪通过。
- 因此当前 fused Q+FA 只能视为 token-level experimental path,不能作为
  strict-logprob 等价路径。
- 该 probe 已固化为:

```bash
cd /home/fhkong/wxwork/attncp_precision_regression
./run_hot_fused_precision_probe.sh
```

  默认要求三路 token-level 一致,并记录 strict logprob diff。验证 kernel fusion
  是否保持原 TP4-CP2 AttnCP exact 路径时,使用 `--require-fused-strict` 将
  `exact_vs_fused` strict equality 作为硬门槛。`--require-strict` 还会要求
  TP4 vs TP4-CP2 strict equality,会命中当前 local-merge reduction-order 的已知差异。
- 继续优化时不应只看 synthetic attention microbench;必须以服务级
  `32k input / 2k output / c45` 的 output TPS 和 ITL 为准。
- 当前可合入/默认启用的仍应是 FA3 attention math + CP2 P2P exchange 的
  precision-safe 路径。

2026-06-26 追加验证:

- 使用 `--attn-cp-decode-cuda-graph-max-seq-len 40960` 和
  `--attn-cp-decode-fused-q-fa` 重新运行 hot probe。
- `exact_vs_fused_token` 通过,但 `exact_vs_fused_strict` 仍失败:
  max diff `8.621e-03`,mean diff `2.549e-03`。
- fused log 确认不是 fallback:
  - graph capture: `batch_size=16 seq_cap=40960 hit_split=1`
  - request decode: `batch_size=1 seq_cap=16384 hit_split=3`
- 新增单测覆盖 `page_table_cap=40960` 且 `cache_seqlens=16384` 的局部
  WeLM shape,仍然通过 FA3 对齐阈值。因此 strict drift 不是简单的
  graph bucket cap 大于 runtime local KV length 导致的局部 attention bug。
- 当前判断保持不变:Triton Q+FA 是 token-level experimental path;strict
  logprob parity 需要 FA3 internal Q-provider 或完整复刻 FA3 mainloop/rounding。

2026-06-26 clean pure-Triton 复测:

- 代码范围已收敛为纯 Triton fused Q+FA:
  - 不修改 `sgl-kernel`;
  - 不依赖本地 patched `sgl-attn`;
  - 不暴露 FA3 provider wrapper;
  - `_attncp_try_fused_q_fa_decode(...)` 只在 shape/seq-cap guard 命中时进入
    `attncp_cp2_fused_q_fa_decode(...)`,否则回退 FA3 exact path。
- full precision regression 仍通过:
  - artifact: `/tmp/welmv4_attncp_precision/20260626_104151`
  - controlled compare: token-level 通过,`max_logprob_diff=0`;
  - MMLU/C-Eval 100 samples: token mismatch `0/100`,
    max/mean logprob diff 均为 `0.00e+00`。
- hot fused 32k probe 仍只满足 token-level:
  - artifact: `/tmp/attncp_hot_precision_probe/20260626_104638`
  - TP4、TP4-CP2 exact、TP4-CP2 fused-hot 输出 token 均为
    `[78, 70, 79, 257]`;
  - `exact_vs_fused_token` 通过;
  - `exact_vs_fused_strict` 失败,max diff `2.128e-01`,
    mean diff `5.949e-02`。

同一代码上重新跑服务级 `32k input / 2k output` sweep:

| config                       | concurrency | output TPS | mean TTFT ms | mean ITL ms | cuda graph |
| ---------------------------- | ----------: | ---------: | -----------: | ----------: | ---------- |
| TP4                          |          22 |   `430.65` |   `17947.19` |     `42.34` | yes        |
| TP4                          |          23 |   `449.54` |   `18445.21` |     `42.17` | yes        |
| TP4                          |          24 |   `403.97` |   `21688.49` |     `41.01` | yes        |
| TP4-CP2 FA3/P2P exact        |          45 |   `658.85` |   `37898.35` |     `49.83` | yes        |
| TP4-CP2 Triton fused enabled |          44 |   `632.21` |   `37479.67` |     `51.33` | yes        |
| TP4-CP2 Triton fused enabled |          45 |   `650.49` |   `36187.07` |     `51.51` | yes        |
| TP4-CP2 Triton fused enabled |          46 |   `555.01` |   `38840.91` |     `50.74` | yes        |

本轮 artifacts:

```text
/tmp/welmv4_attncp_manual_sweep/20260626_105331_tp4_s32768_o2048/summary.tsv
/tmp/welmv4_attncp_manual_sweep/20260626_110156_tp4_cp2_s32768_o2048/summary.tsv
/tmp/welmv4_attncp_manual_sweep/20260626_111851_tp4_cp2_exact_s32768_o2048/summary.tsv
```

TP4-CP2 服务日志确认四个 rank 均启用 experimental Triton fused Q+FA decode。
当前 clean Triton path 的端到端结论是:

- CP2 仍然把最佳常驻并发从 TP4 的 `23` 提升到 `45`,容量接近 `1.96x`。
- TP4-CP2 FA3/P2P exact c45 的 output TPS 是 `658.85 tok/s`,相对 TP4
  best `449.54 tok/s` 提升 `46.6%`。
- TP4-CP2 Triton fused-enabled c45 的 output TPS 是 `650.49 tok/s`,相对 TP4
  best 提升 `44.7%`,但低于同代码 fused-off exact c45。
- 当前端到端 `~46%` 提升主要来自 AttnCP sharded-KV 容量提升和 CP2 P2P
  exchange,不能归因于 Triton Q+FA fusion。
- TP4-CP2 exact mean ITL 为 `49.83 ms`,比 TP4 best `42.17 ms`
  慢约 `18.2%`; fused-enabled mean ITL 为 `51.51 ms`,更慢。
- Triton fused Q+FA 在 focused attention microbench 中有明显收益,但在当前
  service sweep 下没有把 ITL gap 消掉;继续优化必须以服务级 profile 为准。
- 由于 hot fused strict logprob 未对齐,该路径仍应保持 experimental/guarded,
  不能替代默认 precision-safe FA3 exact attention math。

### 4.2 当前精度安全性能基线

2026-06-26 使用 P2P exact 路径重新 sweep `32k input / 2k output`:

| config            | best concurrency | best output TPS | mean ITL at best | mean TTFT at best |
| ----------------- | ---------------: | --------------: | ---------------: | ----------------: |
| TP4               |             `23` |  `449.48 tok/s` |       `42.39 ms` |     `18024.11 ms` |
| TP4+CP2 P2P exact |             `45` |  `665.58 tok/s` |       `49.96 ms` |     `36202.06 ms` |

结论:

- TP4+CP2 最大有效常驻并发约为 TP4 的 `1.96x`。
- 最佳 output throughput 从 `449.48 tok/s` 提升到 `665.58 tok/s`,约 `1.48x`。
- mean ITL 仍慢约 `17.9%`,说明 decode 单 token latency 还有优化空间。
- TTFT 在该长输入高并发场景明显更高,这和 CP2 prefill / cuda graph /
  高并发排队共同相关;本文仍只把 decode ITL 作为优化重点。

---

## 5. P0: Decode Profiling 分解

实现优化前先补 profiling,否则容易把复杂度加在非瓶颈上。这里记录的是
2026-06-24 的临时 Python/CUDA event profiler 结果;对应 env-gated profiler 已在
后续 polish 中移除,当前代码不再支持这些临时调试开关。

注意:

- Python 层 profiler 会跳过 CUDA graph capture,因此用于拆分 eager decode region 耗时。
- 如果要看真实线上 ITL,仍然需要用开启 CUDA graph 的 server 跑端到端 benchmark。
- 做 region profiling 时建议临时加 `--disable-cuda-graph`,否则 graph replay 不会逐步进入 Python attention 函数。

建议对 sharded-KV decode local-merge 路径打点:

```text
attncp_decode_q_gather
attncp_decode_local_fa
attncp_decode_o_lse_exchange
attncp_decode_merge
attncp_decode_out_copy_flatten
attncp_decode_total
```

当前实际日志 region:

| region           | 含义                                                 |
| ---------------- | ---------------------------------------------------- |
| `q_gather`       | CP group 内 gather full Q heads                      |
| `sink_gather`    | attention sink gather / sink masking                 |
| `local_fa`       | 本 rank local KV shard 上的 FA3 decode               |
| `post_fa`        | LSE transpose/contiguous 和 empty local KV 修正      |
| `o_lse_exchange` | local O/LSE partial state 跨 CP rank exchange        |
| `merge`          | online softmax state merge 并 slice 回 local Q heads |
| `out_copy`       | 写回 caller-provided output buffer                   |

对照场景:

| 场景                                       | 用途                      |
| ------------------------------------------ | ------------------------- |
| 8k input / 512 output / c4                 | 同并发 ITL gap            |
| 32k input / 2k output / TP4 c23 vs CP2 c45 | 最大常驻并发下 ITL / 吞吐 |
| 32k input / 2k output / CP2 c44/c45/c46    | 未满、刚满、超容量排队    |

需要记录:

- `mean_itl_ms`, `p95_itl_ms`, `output_tps`
- server `#running-req`, `#queue-req`, `token usage`
- decode cuda graph 是否命中
- 每个 profiling region 的平均耗时和占比

### 5.1 当前 profiling 样本

本轮 profile 目标是拆解 AttnCP decode attention 内部耗时,不是给出正式线上吞吐结论。因为当时的临时 profiler 在 Python 层使用 CUDA event 打点,所以本轮临时关闭 CUDA graph。

实验配置:

- model: `80a3_v4d5_256k_merge_thinking_kimi_k25_0502_20260503_032335/epoch_003_step_0002610`
- topology: `TP4-CP2`
- attention backend: FA3
- features: `welm-kv-mirror-opt`, attention sink, over encoding, chunked prefill enabled
- profiling mode: `--disable-cuda-graph`, layer 0 only, skip first 8 samples, every sample recorded
- workload: random ids, `input=2048`, `output=64`, `num_prompts=4`, `max_concurrency=4`
- artifact: `/tmp/welmv4_attncp_decode_profile_20260624_174435/profile_s2k_o64_c4_warmup8.jsonl`

bench_serving 端到端结果:

| metric                 |       value |
| ---------------------- | ----------: |
| successful requests    |           4 |
| total input tokens     |        8192 |
| total generated tokens |         256 |
| request throughput     |  0.27 req/s |
| output throughput      | 17.01 tok/s |
| mean TTFT              |  8300.12 ms |
| mean ITL               |   107.89 ms |
| median ITL             |    87.93 ms |
| p95 ITL                |    90.22 ms |
| p99 ITL                |    90.56 ms |

说明: 这里的端到端 ITL 明显受到 `--disable-cuda-graph` 和 profiling 同步开销影响,只能作为本轮 profiler smoke test 的背景数据,不能和线上 CUDA graph benchmark 直接比较。

最后一组累计 samples=56 的平均耗时如下:

| rank    |  total | q_gather | sink_gather | local_fa | post_fa | o_lse_exchange |  merge | out_copy |
| ------- | -----: | -------: | ----------: | -------: | ------: | -------------: | -----: | -------: |
| CP0-TP0 | 0.7019 |   0.1689 |      0.1108 |   0.0637 |  0.0841 |         0.2211 | 0.0393 |   0.0072 |
| CP0-TP1 | 0.7147 |   0.1683 |      0.1197 |   0.0696 |  0.0867 |         0.2034 | 0.0510 |   0.0081 |
| CP1-TP2 | 0.9342 |   0.4302 |      0.1236 |   0.0223 |  0.0420 |         0.3044 | 0.0057 |   0.0030 |
| CP1-TP3 | 0.6808 |   0.1420 |      0.1167 |   0.0752 |  0.0879 |         0.1870 | 0.0551 |   0.0087 |
| avg     | 0.7579 |   0.2274 |      0.1177 |   0.0577 |  0.0752 |         0.2290 | 0.0378 |   0.0068 |

按 `avg total=0.7579 ms` 估算占比:

| region           | avg ms | approx share |
| ---------------- | -----: | -----------: |
| `q_gather`       | 0.2274 |        30.0% |
| `o_lse_exchange` | 0.2290 |        30.2% |
| `sink_gather`    | 0.1177 |        15.5% |
| `post_fa`        | 0.0752 |         9.9% |
| `local_fa`       | 0.0577 |         7.6% |
| `merge`          | 0.0378 |         5.0% |
| `out_copy`       | 0.0068 |         0.9% |

初步结论:

- layer 0 单层 local-merge attention eager profile 中,`q_gather` 和 `o_lse_exchange` 是主要显式通信项,平均各约 0.23 ms。
- 两项通信加起来约占 layer 0 local-merge attention profile 的 60%,说明后续优化应优先围绕通信量和通信等待。
- `local_fa` 本身很小,单看 layer 0 不是 decode attention 的主要瓶颈;继续优化 FA kernel 本身优先级低于通信路径。
- `sink_gather` 约 0.12 ms,attention sink 的通信/处理成本不可忽略。
- `post_fa + merge + out_copy` 合计约 0.12 ms,可以作为 P2 fusion 的候选,但不是第一瓶颈。
- 后续 P1 优化仍应优先做 `O/LSE head-slice exchange`;同时评估 `q_gather` 是否能通过 overlap、CP2 fast path 或 CUDA graph 内部化降低等待。
- 需要再补一轮更接近真实线上场景的 Nsight/CUDA graph profile,用来确认 graph replay 下 attention region 是否仍呈现同样瓶颈分布。

---

## 6. P1: O/LSE Head-Slice Exchange

### 6.1 当前问题

当前每个 CP rank 计算 `local_o_full / local_lse_full`,然后 all-gather 到所有 CP ranks。

```text
local_o_full shape: [B, full_q_heads, D]
local_lse_full shape: [B, full_q_heads]

CP all-gather:
  every rank receives every other rank's full_q_heads partial states

merge:
  every rank only keeps its own local_q_heads slice
```

这意味着每个 rank 接收了大量最终不会使用的 head slices。

### 6.2 优化目标

把 full all-gather 改成按 Q head owner 分发:

```text
for each CP rank src:
  src computes local_o_full, local_lse_full
  src sends head slice for dst to dst

for each CP rank dst:
  dst receives only its local_q_heads slice from every src
  dst merges these partial states
```

通信量变化:

```text
当前:  O/LSE all-gather full_q_heads
目标:  O/LSE exchange local_q_heads only
```

对于 CP2:

```text
full_q_heads = 2 * local_q_heads
O/LSE exchange bytes 约减少 50%
```

对于 CP4:

```text
full_q_heads = 4 * local_q_heads
O/LSE exchange bytes 约减少 75%
```

### 6.3 实现选择

候选实现:

1. 基于 `pynccl` send/recv 做 CP2 pair exchange。
2. 扩展 `GroupCoordinator` 支持 graph-safe all-to-all/coalesced all-to-all。
3. 用 `all_gather` 保持 primitive 不变,但先 slice 后 gather;只能在某些 layout 下简化,不是真正 all-to-all。

首版建议做 **CP2 fast path**,避免一开始实现泛化 CPN all-to-all。

### 6.4 精度语义

数学语义不变:

```text
same local_o/local_lse
same merge_state_v2 order
only communication layout changes
```

因此精度风险低于 Q rotation pipeline。

---

## 7. P1/P2: CP2 Fast Path

当前主要目标部署是 TP4-CP2,所以可以优先写 CP2 专用路径。

### 7.1 当前通用路径

```text
q_full = all_gather(q_local)
local_o_full, local_lse_full = FA(q_full, local_kv)
gathered_o, gathered_lse = all_gather(local_o_full, local_lse_full)
merged = merge loop over cp_world_size
out = merged[:, local_head_slice]
```

### 7.2 CP2 fast path

```text
remote_q = exchange(q_local)
q_full = concat(q_local, remote_q) or use two views

local_o_full, local_lse_full = FA(q_full, local_kv)

local_head_state  = local_o_full[:, local_head_slice],  local_lse_full[:, local_head_slice]
remote_head_state = local_o_full[:, remote_head_slice], local_lse_full[:, remote_head_slice]

exchange only remote_head_state with peer

merged = merge_state_v2(local_partial_from_self, partial_from_peer)
write final local output
```

如果先不做 Q pipeline,CP2 fast path 仍然可以:

- 保持一次 FA。
- 减少 O/LSE exchange bytes。
- 减少 gather buffer 和 view/loop 开销。
- merge 固定为 two-state merge。

---

## 8. P2: Merge + Slice + Flatten Fusion

### 8.1 当前问题

当前 merge 后还有若干中间动作:

```text
gathered_o / gathered_lse
  -> copy local head slice to merge buffers
  -> merge_state_v2
  -> copy to out
  -> view/flatten for o_proj
```

这些操作单次不大,但 decode 每 token 每层都会执行。

### 8.2 CP2 fused merge kernel

CP2 merge 可以写成一个专用 kernel:

```text
input:
  o0, lse0
  o1, lse1

compute:
  m  = max(lse0, lse1)
  w0 = exp(lse0 - m)
  w1 = exp(lse1 - m)
  o  = (o0 * w0 + o1 * w1) / (w0 + w1)

output:
  final local attn output in flat layout
```

目标输出直接匹配后续 `o_proj` 输入:

```text
[B, local_q_heads, head_dim] -> [B, local_q_heads * head_dim]
```

收益:

- 减少 merge 临时 tensor。
- 减少 copy/slice kernel。
- 更好服务 decode cuda graph 固定 workspace。

风险:

- 需要确认 dtype: `o` bf16,`lse` fp32。
- attention sink empty local KV 语义必须保持。
- 对比 `merge_state_v2` 做数值回归。

### 8.3 已落地结果: CP2 Fused Local-Head Merge

2026-06-24 已实现并验证 CP2 fused local-head merge:

```bash
SGLANG_ATTNCP_DECODE_CP2_FUSED_MERGE=1   # default
```

实现边界:

- 只替换 CP2 allgather 路径后的 local-head merge。
- 不改变 `q_allgather`。
- 不改变 O/LSE `all_gather_coalesced`。
- 不改变 sharded-KV cache residency 和 owner 语义。
- O/LSE P2P 仍默认关闭。

组件 benchmark:

| component                       |      median |
| ------------------------------- | ----------: |
| current copy + `merge_state_v2` | `36.907 us` |
| fused Triton local-head merge   |  `8.652 us` |

服务级 benchmark:

- artifact: `/tmp/welmv4_attncp_perf/20260624_224525_p4_fused_merge`
- workload: `input_len=8192`, `output_len=512`, `concurrency=4`,
  `num_prompts=16`, CUDA graph enabled, FA3 prefill/decode,
  `welm-kv-mirror-opt` enabled.

| config                |       output TPS |      mean TTFT |     mean ITL |
| --------------------- | ---------------: | -------------: | -----------: |
| TP4                   | `336.0555 tok/s` | `1620.4334 ms` |  `8.7661 ms` |
| TP4+CP2 fused merge   | `290.6197 tok/s` | `1832.3900 ms` | `10.2214 ms` |
| TP4+CP2 unfused merge | `282.4439 tok/s` | `1828.8627 ms` | `10.6283 ms` |

结论:

- fused merge 相比 unfused merge,ITL 降低 `3.8%`,output TPS 提升 `2.9%`。
- TP4-CP2 相比 TP4 的 ITL gap 从本轮 unfused 的 `+21.2%` 降到 `+16.6%`。
- 精度回测通过: `/tmp/welmv4_attncp_precision/20260624_224011`,
  controlled token diff 为 0,100 条 MMLU/C-Eval token mismatch 为 0,
  max/mean logprob diff 为 0。
- CUDA graph 正常命中,fused CP2 decode log 中 `cuda graph: True`。

P2P follow-up:

- artifact: `/tmp/welmv4_attncp_perf/20260624_225548_p4_p2p_fused_probe`
- 在 O/LSE P2P 路径中增加 fused pack 和 fused local/remote merge 后,
  P2P 仍比 allgather fused 慢:

| config                  |       output TPS |      mean TTFT |     mean ITL |
| ----------------------- | ---------------: | -------------: | -----------: |
| TP4+CP2 allgather fused | `290.2648 tok/s` | `1850.5601 ms` | `10.2030 ms` |
| TP4+CP2 P2P fused       | `282.3482 tok/s` | `1871.4723 ms` | `10.5499 ms` |

因此 O/LSE P2P 继续保持默认关闭。当前 Pynccl send/recv graph 开销仍然
抵消 payload 缩减收益。

---

## 9. P2: Q Rotation Pipeline Overlap

这是后续最有潜力的通信/计算 overlap 方向。

### 9.1 当前串行路径

```text
wait q_allgather
run local FA on q_full
wait o_lse_allgather
merge
```

### 9.2 目标路径

把 Q shard 作为 pipeline 数据流,不要先 materialize full Q:

```text
double buffer:
  q_buf_local
  q_buf_remote

step 0:
  compute local_q_heads against local KV
  async recv remote_q

step 1:
  compute remote_q_heads against local KV
  async send local_q partial O/LSE back to owner

finish:
  receive peer partial O/LSE for local_q_heads
  merge local_q_heads partial states
```

CP2 简化版本:

```text
rank0:
  FA(q0, KV0) while recv q1
  FA(q1, KV0) while send state(q0, KV0) to rank1
  send state(q1, KV0) to rank1 or directly route to owner

rank1:
  symmetric

owner rank:
  merge state(q_local, KV_local) + state(q_local, KV_remote)
```

### 9.3 可能收益

- Q 通信可以和 first FA overlap。
- 不需要完整 `q_full` workspace。
- 可以天然结合 head-slice O/LSE exchange。

### 9.4 主要风险

- FA launch 次数可能从 1 次变成 CP 次。CP2 是 2 次,需要确认 launch overhead 是否抵消 overlap 收益。
- CUDA graph 捕获复杂度升高:需要固定 stream/event 和通信 launch 顺序。
- `flash_attn_with_kvcache` 是否支持小 head slice 形状下保持高效,需要 benchmark。
- attention sink 仍必须只被一个 CP shard 注入,不能重复计入 denominator。

### 9.5 推荐推进方式

不要一开始直接替换默认路径。建议加一个实验路径:

```text
ATTNCP_DECODE_Q_ROTATION=1
```

先在 eager / cuda graph off 下验证:

1. token-level 精度。
2. strict logprob diff 范围。
3. 每层 timing 是否真的 overlap。
4. 是否比 CP2 head-slice exchange 更快。

验证后再适配 cuda graph。

---

## 10. P3: Attention Output / Gate / OProj Fusion

参考 MMQ fused mixer 中的思路,可以把 attention output 后处理和 out projection 结合。

候选融合范围:

```text
attn_out
  -> optional gate/head scale
  -> flatten
  -> o_proj GEMM
```

当前优先级较低,原因:

- 要碰模型层和 linear/GEMM 路径,侵入比 attention backend 更深。
- 需要确认 WeLM 当前 gate/head-scale 的实际形态。
- TP all-reduce 仍在 o_proj 后,融合后还要保证 TP group 语义不变。

适合作为 head-slice exchange 和 merge fusion 完成后的进一步优化。

---

## 11. P3: Decode Metadata 优化

当前每个 decode step 需要从 full page table 构造 CP-local compact page table:

```text
page_table
  -> local_valid
  -> compact_cols
  -> local_page_table
  -> local_cache_seqlens
```

长上下文和大 batch 下,这部分和 `batch_size * max_seq_len` 相关。

后续可以考虑:

- allocator/scheduler 维护 CP-local page table。
- decode batch 构建时直接生成 local metadata。
- 避免每步在 attention backend 内重新 compact。

这不改变 attention 数学,但会改 scheduler / metadata 生命周期,建议后置。

---

## 12. 推荐阶段

### Phase 1: 可观测性和低风险通信优化

目标:

- 补 decode profiling。
- 实现 CP2 O/LSE head-slice exchange。
- 保持当前 Q full gather 和一次 FA 不变。

验收:

- precision regression PASS。
- 8k/512/c4 ITL gap 明显下降。
- 32k/2k/c45 output throughput 提升。
- cuda graph 仍命中。

### Phase 2: Merge fusion

目标:

- CP2 merge + slice + flatten fused kernel。
- 减少小 kernel / copy。

验收:

- 对比 `merge_state_v2` token-level 对齐。
- strict logprob diff 不扩大。
- profiling 中 merge/copy region 降低。

### Phase 3: Q rotation pipeline overlap

目标:

- 实验 Q shard rotation。
- 通信/FA overlap。
- 减少 q_full/local_o_full workspace。

验收:

- eager path 正确。
- cuda graph path 正确。
- FA launch 增加带来的开销小于 overlap 收益。
- 在 CP2 主场景明显优于 Phase 1/2。

---

## 13. 测试计划

### 精度测试

使用现有脚本:

```bash
cd /home/fhkong/wxwork/attncp_precision_regression
./run_full_precision.sh
```

必须覆盖:

- TP4 vs TP4-CP2 controlled output token/text。
- MMLU/C-Eval 100 samples。
- WeLM kv mirror opt enabled。
- over encoding enabled。
- chunked prefill enabled。
- ordinary cuda graph enabled。
- attention sink 模型单独回归。

### 性能测试

同并发 ITL:

```text
8k input / 512 output / c4
```

容量边界吞吐:

```text
32k input / 2k output
TP4: c22/c23/c24
TP4-CP2: c44/c45/c46
```

重点指标:

- `mean_itl_ms`
- `p95_itl_ms`
- `output_tps`
- `mean_ttft_ms`
- `peak_server_running`
- `max_token_usage`
- `cuda_graph_seen`

### 回归判断

优化成功的最低目标:

```text
8k/512/c4:
  TP4-CP2 mean ITL gap 从当前约 +21% 降到 +10%~15%

32k/2k/c45:
  TP4-CP2 最佳 output throughput 从当前 1.30x 提升到 1.45x 左右
```

---

## 14. Open Questions

1. CP2 fast path 是否只服务 `attn_cp_size=2`,还是同时设计成 CPN 泛化接口?
2. O/LSE head-slice exchange 应基于 pynccl send/recv,还是先补 graph-safe all-to-all primitive?
3. Q rotation pipeline 是否值得承担 FA launch 次数增加?
4. merge fusion 是否直接输出 flat buffer 给 o_proj,还是先保持 `[B, H, D]` layout?
5. attention sink 在 Q rotation 下是否继续固定 CP rank 0 注入,还是按 owner rank 注入并额外校正 denominator?
6. 是否需要保留 dense decode correctness fallback 用于 strict logprob 对齐?

---

## 15. KV 一次读取约束

decode fused Q+FA 的核心不变量是 KV-stationary:

```text
program grid = (batch, kv_head, optional_kv_split)
```

这是硬约束,不是普通性能建议。作用域是单层、单 decode step、单 batch item、
单 resident `kv_head`:每个本地 K/V token 只能被 attention mainloop 读取一次。
如果启用 split-KV,则约束作用在每个不重叠 split range 内;所有 split 的 range
合起来覆盖本地 KV shard,但彼此不能重复。

每个 program 负责一个 batch、一个 resident `kv_head`、以及一个不重叠的本地 KV
range。它需要在同一次 K/V tile 驻留期间计算这个 `kv_head` 对应的所有 logical
CP Q heads:

```text
q_heads_per_kv = full_cp_q_heads / local_kv_heads
```

因此不能把 fused 路径拆成:

```text
local Q -> FA3
remote Q -> FA3
```

也不能按 Q head group 启动多次 attention kernel。否则同一个 resident KV shard
会被重复读取,和 AttnCP decode 降低 KV 带宽/显存压力的目标冲突。

同理,如果后续做 producer/consumer 或 TMA pipeline,producer 只能预取下一段
K/V tile,consumer 必须在这段 K/V tile 仍驻留时完成 local/peer Q heads 的计算。
不能先用 local Q 消费一次 K/V,再为 peer Q 重新 issue 同一段 K/V load。

允许的 split 只能沿本地 KV sequence 切分:

```text
split0: local KV [0, N0)
split1: local KV [N0, N1)
...
```

split 之间的 KV range 必须互不重叠。后续只能 merge partial O/LSE,不能再次读取
K/V。换句话说,允许重复读取的是 Q 或 partial O/LSE;不允许重复读取的是 resident
K/V cache。

当前 Triton prototype 的形态:

- `attncp_cp2_fused_q_fa_decode(...)` 是替换 decode 中 Q exchange + local FA3 的函数边界。
- Triton kernel grid 是 `(batch, kv_head)` 或 `(batch, kv_head, split)`。
- 支持 CP2、`page_size=1`、attention sink、SWA window、split KV。
- 仍然是实验路径,需要通过环境变量显式开启。
- 如果后续因为寄存器压力无法一次覆盖所有 `q_heads_per_kv`,首选策略是 fallback
  到 FA3 exact path,而不是按 Q head group 拆分 fused kernel。按 Q head group
  拆分会让同一段 resident KV 被重复读取,违反本节的硬约束。
- 当前实现用 shape guard 固定这个约束:只有
  `next_power_of_2(q_heads_per_kv) <= 16` 的形状可以进入 Triton fused path。
  超出该范围时服务端必须回退 FA3 exact path,不能通过增加 Q-head grid 维度来
  继续跑 fused kernel。

2026-06-26 新增本地严格单测覆盖 WeLM decode 形状:

```text
local_q_heads = 6
full_cp_q_heads = 12
local_kv_heads = 1
head_dim = 256
local_kv_len = 16384
attention sinks enabled
split cases = 1 and 8
```

当前单测结果:

```text
python/sglang/jit_kernel/tests/test_attncp_fused_ops.py: 36 passed
```

其中新增 `test_attncp_cp2_fused_q_fa_decode_launch_grid_is_kv_stationary`
会 monkeypatch Triton kernel 并截获 launch grid,确保 non-split 路径只用
`(batch, kv_head)`,split 路径只用 `(batch, kv_head, split)`。这个测试的目的
是防止后续优化误引入 Q-head program 维度,导致同一 resident KV shard 被重复读取。

另外新增的 shape/source guard 测试会检查:

- 不支持的 `q_heads_per_kv` 形状会直接 rejected/fallback,而不是拆 Q-head
  program。
- fused attention kernel 源码中没有 `q_head_idx = tl.program_id(...)`。
- non-split 和 split attention kernel 各自只保留一处 K/V tile load 逻辑。

需要注意: 这个测试只证明 fused local attention 和 FA3 的局部 O/LSE 差异在严格阈值内,不能替代完整服务级长上下文 logprob 回归。当前 32k hot-fused 服务 probe 仍然显示 fused path 的 output token 一致,但 logprob drift 明显大于 exact FA3/P2P 路径。

### CUDA graph seq-cap 约束

hot-fused decode 对 CUDA graph 的 sequence cap 很敏感。当前 32k controlled probe 的结果:

| graph / split 配置                                                      | exact vs fused token | exact vs fused strict max diff |
| ----------------------------------------------------------------------- | -------------------: | -----------------------------: |
| fixed 4096 local-KV split, `graph_cap=32768`                            |                 pass |                    `0.000e+00` |
| fixed 4096 local-KV split, `graph_cap=40960`, exp2 online softmax       |                 pass |                    `8.621e-03` |
| fixed 4096 local-KV split, `graph_cap=34816`, exp2 online softmax       |                 pass |                    `8.621e-03` |
| single split, `graph_cap=40960`                                         |                 pass |                    `6.964e-02` |
| force FA3 `num_splits=4` for comparison                                 |                 pass |                    `1.030e-01` |
| reverse KV block traversal                                              |                 pass |                    `1.243e-02` |

结论:

- 固定 split 和 FA-style `exp2/log2` online softmax 已经把 hot fused drift
  降到当前最好 `8.621e-03`,但仍未达到 `1e-5` strict gate。
- `graph_cap=32768` 的受限 bucket 当前可以做到 exact AttnCP vs fused-hot
  strict 一致,但这只覆盖 prompt 长度接近 32k 且 decode 不超过该 capture cap 的
  场景。TP4 vs TP4-CP2 exact 仍有已知 local-merge reduction-order logprob diff。
- 2026-06-26 使用 `--require-fused-strict` 复测通过:
  `/tmp/attncp_hot_precision_probe/20260626_075425`。该历史 run 证明的是
  `seq_cap=32768` 受限 bucket 内 fused-vs-exact strict 一致,不是长输出全程
  fused strict 一致。
- 单 split、强行匹配 FA3 split 数、反向扫描 KV block、调整 sink finalize 或
  score scale 域都没有解决 strict drift。
- `sgl-attn` FA3 源码显示 split-KV partial O 使用 fp32 `ElementPartial`,
  split combine 对 LSE 使用自然底 `expf/logf`;这说明后续若要 strict 对齐,
  需要更完整地复用/改造 FA3 内部路径,而不是只在外层 Triton 近似重写。
- 对 32k prompt 的 fused-vs-exact strict 验证,当前建议显式设置:

```bash
--attn-cp-decode-cuda-graph-max-seq-len 32768
--attn-cp-decode-fused-q-fa
```

对 `32k input / 2k output` 性能实验,如果要真实命中 `40960` bucket 的 fused
path,建议设置 `--attn-cp-decode-cuda-graph-max-seq-len 40960` 并开启
`--attn-cp-decode-fused-q-fa`;当前仍不能声明 strict logprob 等价。

风险:

- `32k input / 2k output` 的 hot fused 路径当前只证明 token-level 稳定和吞吐收益。
  后续需要继续优化 fused kernel 数值行为,或超过 strict bucket 后 fallback 到 exact
  FA3,否则不能声称长输出 strict logprob 对齐。

### 2026-06-26 local page-table cap 更新

之前 fused service sweep 没有体现出 isolated attention microbench 的收益,主要原因是
decode CUDA graph bucket 使用的是全局 seq cap,而 sharded-KV local attention 只需要
扫描本 CP rank owned 的 compact KV cap。例如 global graph cap `40960`,CP2 且
`cp_kv_chunk_size=1024` 时,每个 CP rank 的 local compact cap 应为 `20480`,不是
`40960`。如果 fused split 按未裁剪的 page-table cap 调度,会产生大量空 split /
空循环开销,抵消 kernel 本身收益。

当前代码将 local compact page table 的 capacity 改为按 chunk owner distribution
计算:

```text
local_cap = owned_full_chunks * cp_kv_chunk_size + optional_tail
```

并新增单测覆盖 `global seq cap -> local cap` 的 chunk round-robin 分配。当前单测:

```text
PYTHONPATH=python .venv/bin/python -m pytest python/sglang/jit_kernel/tests/test_attncp_fused_ops.py -q
46 passed
```

focused attention microbench 显示空 split 开销被显著压低:

| page-table cap | target_sharded_fused_slice_a2a median |
| -------------: | ------------------------------------: |
|        `40960` |                          `733.331 us` |
|        `20480` |                          `435.117 us` |

同一代码重新跑完整服务 sweep:

```text
Model:
  /home/fhkong/models/80a3_v4d5_256k_merge_thinking_kimi_k25_0502_20260503_032335/epoch_003_step_0002610

Workload:
  random-ids, input_len=32768, output_len=2048,
  num_prompts=max_concurrency=concurrency, request_rate=inf

TP4-CP2 fused launch:
  --attn-cp-decode-fused-q-fa
```

`--attn-cp-decode-fused-q-fa` 会默认启用 CP2 Q P2P 和 O/LSE P2P fast path;
不再要求用户额外设置性能 env。

结果:

| config                         | concurrency | output TPS | mean TTFT ms | mean ITL ms | cuda graph |
| ------------------------------ | ----------: | ---------: | -----------: | ----------: | ---------- |
| TP4 best                       |          23 |   `449.54` |   `18445.21` |     `42.17` | yes        |
| TP4-CP2 FA3/P2P exact          |          45 |   `656.77` |   `38342.85` |     `49.82` | yes        |
| TP4-CP2 Triton fused/local-cap |          44 |   `792.44` |   `37349.47` |     `37.30` | yes        |
| TP4-CP2 Triton fused/local-cap |          45 |   `816.60` |   `36245.02` |     `37.42` | yes        |
| TP4-CP2 Triton fused/local-cap |          46 |   `669.45` |   `37892.30` |     `37.03` | yes        |

Artifacts:

```text
/tmp/welmv4_attncp_manual_sweep/20260626_105331_tp4_s32768_o2048/summary.tsv
/tmp/welmv4_attncp_manual_sweep/20260626_114531_tp4_cp2_exact_localcap_s32768_o2048/summary.tsv
/tmp/welmv4_attncp_manual_sweep/20260626_115340_tp4_cp2_fused_localcap_full_s32768_o2048/summary.tsv
```

更新后的性能结论:

- precision-safe FA3/P2P exact c45 相对 TP4 best 的端到端 output TPS 提升
  `46.1%`。
- experimental Triton fused/local-cap c45 相对 TP4 best 的端到端 output TPS 提升
  `81.7%`。
- experimental Triton fused/local-cap c45 相对 FA3/P2P exact c45 提升 `24.3%`。
- fused/local-cap c45 mean ITL `37.42 ms`,比 TP4 best `42.17 ms` 低 `11.3%`。
- c46 的 peak running 只有 `45`,说明该点已经越过当前稳定常驻容量边界,吞吐回落
  是预期现象。

精度状态保持不变:

- full precision regression 在 precision-safe exact path 上通过:
  `/tmp/welmv4_attncp_precision/20260626_120255`,
  MMLU/C-Eval token mismatch `0/100`,max/mean logprob diff `0.00e+00`。
- hot fused probe 输出 token 一致:
  `/tmp/attncp_hot_precision_probe/20260626_120726`,
  但 `exact_vs_fused_strict` 仍失败,max diff `8.621e-03`。
- hot fused profile 确认实际请求命中 fused split,不是 fallback 伪通过:
  batch size `1` 下 `seq_cap=131072/65536/32768/16384` 均有 `hit_split`,
  `seq_cap=8192` 按内部 min seq cap `16384` fallback。
- 因此 fused/local-cap 可以作为长上下文吞吐实验路径,但仍不能声明 strict FA3
  logprob parity。默认 precision-safe 路径仍应保留 FA3 exact attention math。

2026-06-26 继续追 strict drift 的诊断结果:

- `tl.dot(p, V)` 保持 `p` 为 fp32 不可直接编译,Triton 要求 dot 两个 operand dtype
  一致。
- 将 `V` cast 到 fp32 并使用 `input_precision="ieee"` 后,WeLM shape 下需要把
  `BLOCK_N` 从 `128` 降到 `64` 才能避开 shared memory 超限;局部 O max diff
  仍是 `2.44e-04`,mean 只从 `2.75e-05` 降到 `2.25e-05`,不值得牺牲当前性能。
- 仅把 `BLOCK_N` 降到 `64` 且保持 bf16 `P @ V` 时,局部 O max diff 仍是
  `2.44e-04`,mean 变为 `2.79e-05`,没有改善。
- 8-layer block sweep 显示 strict drift 有明显层间抵消:

| fused layers | output ids          | strict max diff |
| ------------ | ------------------- | --------------: |
| `0-7`        | `[78, 70, 79, 257]` |  `7.954895e-02` |
| `8-15`       | `[78, 70, 79, 257]` |  `2.038772e-01` |
| `16-23`      | `[78, 70, 79, 257]` |  `7.937133e-02` |
| `24-31`      | `[78, 70, 79, 257]` |  `4.506576e-02` |
| `32-39`      | `[78, 70, 79, 257]` |  `3.147352e-02` |
| `40-47`      | `[78, 70, 79, 257]` |  `1.834488e-02` |
| all layers   | `[78, 70, 79, 257]` |     `8.621e-03` |

Artifact:

```text
/tmp/attncp_fused_layer_probe/20260626_122126/summary.tsv
```

该结果说明 strict drift 不是单个层段的语义 bug,而是纯 Triton attention math 与
FA3 mainloop/rounding 的细小差异在层间非线性传播并部分抵消。继续用简单 layer
fallback 很难同时保留 fused 性能和证明 strict parity。若 strict logprob parity 是
硬门槛,更合理的技术路线仍然是 FA3/CUDA internal Q-provider,而不是继续在外部
Triton kernel 中近似复刻 FA3。

### 2026-06-26 fused logprob graph guard 更新

当前 pure Triton fused Q+FA 仍不能声明 strict FA3 logprob parity。为了同时保留
非 logprob 长上下文热路径性能和 logprob 回归精度,实现上采用 guarded fallback:

- 非 logprob decode:
  - 满足 CP2、`page_size=1`、KV-stationary shape、seq bucket 在
    内部 fused min seq cap 以上时,走 Triton fused Q+FA。
  - 不满足条件时 fallback 到原 FA3 exact local attention。
- `return_logprob=True` decode:
  - 如果 selected CUDA graph seq bucket 会捕获 fused 分支,则禁用普通 CUDA graph,
    走 eager FA3 exact fallback。
  - 如果 selected CUDA graph seq bucket 小于 fused min seq cap,则保留原 CUDA graph
    exact path。短 prompt MMLU/C-Eval top-logprob 回归依赖这个行为。

这个 guard 不能简单地对所有 fused+logprob 请求禁用 CUDA graph。实测短 prompt 下
AttnCP eager decode 与 CUDA-graph exact path 在 top-logprobs 上并不 strict-identical;
过度禁用 graph 会导致 MMLU/C-Eval 回归失败。

对应失败与修复证据:

| case                                | artifact                                          | result                                                     |
| ----------------------------------- | ------------------------------------------------- | ---------------------------------------------------------- |
| unconditional logprob graph-disable | `/tmp/welmv4_attncp_precision/20260626_124758`    | FAIL, token mismatch `27/100`, max logprob diff `2.69e+01` |
| exact/no-fused first10              | same baseline first10                             | PASS, max/mean diff `0.00e+00`                             |
| fused first10 after bucket guard    | same baseline first10                             | PASS, max/mean diff `0.00e+00`                             |
| hot fused probe after bucket guard  | `/tmp/attncp_hot_precision_probe/20260626_130644` | PASS for `exact_vs_fused_strict`, max/mean diff `0.00e+00` |
| full precision after bucket guard   | `/tmp/welmv4_attncp_precision/20260626_131150`    | PASS, token mismatch `0/100`, max/mean diff `0.00e+00`     |

最新三组同代码 sweep:

```text
Model:
  /home/fhkong/models/80a3_v4d5_256k_merge_thinking_kimi_k25_0502_20260503_032335/epoch_003_step_0002610

Workload:
  random-ids, input_len=32768, output_len=2048,
  num_prompts=max_concurrency=concurrency, request_rate=inf

TP4-CP2 non-fusion env:
  SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_Q_P2P=1
  SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_OLSE_P2P=1

TP4-CP2 fused launch:
  --attn-cp-decode-fused-q-fa
```

| config             | concurrency | peak running | output TPS | mean TTFT ms | mean ITL ms | cuda graph |
| ------------------ | ----------: | -----------: | ---------: | -----------: | ----------: | ---------- |
| Naive TP4          |          22 |           22 |   `431.36` |   `17383.74` |     `42.53` | yes        |
| Naive TP4          |          23 |           23 |   `448.13` |   `18067.73` |     `42.52` | yes        |
| Naive TP4          |          24 |           23 |   `403.00` |   `22153.64` |     `40.92` | yes        |
| TP4-CP2 non-fusion |          22 |           22 |   `554.75` |   `19548.00` |     `30.13` | yes        |
| TP4-CP2 non-fusion |          23 |           23 |   `587.31` |   `18547.37` |     `30.12` | yes        |
| TP4-CP2 non-fusion |          24 |           24 |   `692.98` |   `19096.72` |     `25.32` | yes        |
| TP4-CP2 non-fusion |          44 |           44 |   `657.32` |   `35549.50` |     `49.61` | yes        |
| TP4-CP2 non-fusion |          45 |           45 |   `664.33` |   `36690.94` |     `49.85` | yes        |
| TP4-CP2 non-fusion |          46 |           45 |   `590.46` |   `38927.42` |     `49.31` | yes        |
| TP4-CP2 fused      |          22 |           22 |   `653.69` |   `19831.79` |     `23.98` | yes        |
| TP4-CP2 fused      |          23 |           23 |   `685.93` |   `18974.66` |     `24.28` | yes        |
| TP4-CP2 fused      |          24 |           24 |   `712.59` |   `19163.62` |     `24.33` | yes        |
| TP4-CP2 fused      |          44 |           44 |   `801.91` |   `35866.20` |     `37.37` | yes        |
| TP4-CP2 fused      |          45 |           45 |   `812.37` |   `36537.16` |     `37.57` | yes        |
| TP4-CP2 fused      |          46 |           45 |   `668.67` |   `38079.50` |     `37.07` | yes        |

Artifacts:

```text
/tmp/welmv4_attncp_manual_sweep/20260626_132740_tp4_s32768_o2048/summary.tsv
/tmp/welmv4_attncp_manual_sweep/20260626_142827_tp4_cp2_s32768_o2048/summary.tsv
/tmp/welmv4_attncp_manual_sweep/20260626_144337_tp4_cp2_s32768_o2048/summary.tsv
```

结论:

- 当前同代码最佳 TP4 是 c23,output TPS `448.13`。
- 当前同代码最佳 TP4-CP2 non-fusion 是 c24,output TPS `692.98`。
- 当前同代码最佳 TP4-CP2 fused 是 c45,output TPS `812.37`。
- 端到端 output TPS 相比 TP4 分别提升:non-fusion `+54.6%`,fused `+81.3%`。
- c46 peak running 仍只有 `45`,说明 c45 是该 32k/2k 场景下当前稳定容量 knee。
- strict logprob parity 依赖 guarded fallback;pure Triton fused kernel 本身仍是
  token-level/throughput 优化路径。

#### ITL 对比

同等并发对比取三组都覆盖的 `c22/c23/c24`。注意 TP4 在 `c24` 已经出现
peak running 回落到 `23`,所以 `c24` 是 TP4 的边界点,不是稳定可继续扩展点。

| concurrency | TP4 ITL ms |  TP4 TPS | CP2 non-fusion ITL ms | CP2 non-fusion TPS | non-fusion vs TP4 ITL | CP2 fused ITL ms | CP2 fused TPS | fused vs TP4 ITL | fused vs non-fusion ITL |
| ----------: | ---------: | -------: | --------------------: | -----------------: | --------------------: | ---------------: | ------------: | ---------------: | ----------------------: |
|          22 |    `42.53` | `431.36` |               `30.13` |           `554.75` |              `-29.2%` |          `23.98` |      `653.69` |         `-43.6%` |                `-20.4%` |
|          23 |    `42.52` | `448.13` |               `30.12` |           `587.31` |              `-29.1%` |          `24.28` |      `685.93` |         `-42.9%` |                `-19.4%` |
|          24 |    `40.92` | `403.00` |               `25.32` |           `692.98` |              `-38.1%` |          `24.33` |      `712.59` |         `-40.5%` |                 `-3.9%` |

最佳吞吐点使用三组各自 sweep 中的最高 output TPS:

| config             | best concurrency | peak running | max token usage | output TPS | mean ITL ms | TPS vs TP4 best | ITL vs TP4 best |
| ------------------ | ---------------: | -----------: | --------------: | ---------: | ----------: | --------------: | --------------: |
| TP4                |               23 |           23 |          `0.98` |   `448.13` |     `42.52` |        baseline |        baseline |
| TP4-CP2 non-fusion |               24 |           24 |          `0.52` |   `692.98` |     `25.32` |        `+54.6%` |        `-40.4%` |
| TP4-CP2 fused Q+FA |               45 |           45 |          `0.98` |   `812.37` |     `37.57` |        `+81.3%` |        `-11.6%` |

TP4 best 与 CP2 容量点 `c45` 统一对比:

TP4 无法稳定常驻 `c45`,因此 TP4 行使用自身最佳吞吐点 `c23`;CP2 两行使用
容量点 `c45`,用于对比 non-fusion 与 fused 在高 residency 下的差异。

| config             | comparison point | peak running | max token usage | output TPS | mean ITL ms | TPS vs TP4 best | ITL vs TP4 best |
| ------------------ | ---------------: | -----------: | --------------: | ---------: | ----------: | --------------: | --------------: |
| TP4                |         best c23 |           23 |          `0.98` |   `448.13` |     `42.52` |        baseline |        baseline |
| TP4-CP2 non-fusion |              c45 |           45 |          `0.98` |   `664.33` |     `49.85` |        `+48.2%` |        `+17.2%` |
| TP4-CP2 fused Q+FA |              c45 |           45 |          `0.98` |   `812.37` |     `37.57` |        `+81.3%` |        `-11.6%` |

关键现象:

- CP2 non-fusion 的最佳吞吐在 `c24`,不是 `c45`。它在低并发下受益于 shard KV
  后每个 rank 读取更短 KV,但高并发 `c44/c45` 被 Q all-gather 和 O/LSE gather
  通信拖慢,ITL 升到约 `49.6-49.8 ms`。
- fused Q+FA 在同等并发 `c22/c23` 相比 non-fusion 继续降低 ITL 约 `19-20%`。
  在容量点 `c45`,output TPS 从 non-fusion `664.33` 提升到 fused `812.37`,
  提升约 `22.3%`;ITL 从 `49.85 ms` 降到 `37.57 ms`,降低约 `24.6%`。
- 因此当前融合优化的主要价值不是让低并发 ITL 更低,而是把 CP2 多出来的
  KV residency 转化为高并发可用吞吐。
