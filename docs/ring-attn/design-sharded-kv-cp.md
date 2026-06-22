# Sharded-KV Context Parallel 设计

```insight
在 SGLang 上做 Attention Context Parallel,目标是把 KV cache 沿 sequence 切片到各
rank,让长上下文的 KV 显存随 cp 倍线性下降。

核心思路(以 TP=4, attn_cp=2 为例,attn_tp = TP / cp = 2):
  Q  : 按完整 TP 在 head 维度切分,每 rank 持 num_q_heads/TP 个 q-head;
       sequence 不切,全 S 跑过 attention。
  KV: KV head 按 attn_tp 切 (num_kv_heads/attn_tp 个 head/rank),
       KV sequence 按 attn_cp 切(sequence/attn_cp 个 token/rank)。
       KV cache 物理上每 rank 只占 1/cp slot。
  Hidden states: 全 S 不切; MLP/MoE 沿用普通 TP 行为。

每层 attention 内部 (prefill 和 decode 走同一条 path，后续可以针对通信量选择再优化 pass-Q/pass-KV 的方式):
  1. sharded_kv_cp_group 内 Q allgather on head 维: 每 rank 临时拿到 H/attn_tp 个 head 的 Q,
     仍然全 S
  2. 每 rank 用 allgather 后的 Q × 本地 1/cp KV 做 segment loop FA
     (per-owner-chunk mask + 本地 LSE merge)→ partial (out, lse)
  3. sharded_kv_cp_group 内 cp_lse_ag_out_rs(LSE allgather + 跨 rank 合并 + reduce_scatter
     on heads):每 rank 收回到 H/TP 个 head 的 final attention output
  4. o_proj 走纯 TP RowParallel + 全 TP group all_reduce,无需 sharded_kv_attn_tp_group 特殊路径

⚠️ sharded-KV mode 使用专用 logical topology:
   tp_rank = sharded_attn_tp_rank * cp_size + sharded_cp_rank(cp 最快)。
   这不是要求全局改 SGLang default group 语义;projection / o_proj / MLP 仍使用
   global TP rank/group。attention 内部单独构造 sharded_kv_cp_group,避免影响现有
   in-seq-split CP、DP attention、MoE CP。
```

**状态**:Design Review v2(去掉 attn_tp Q shard,改为纯 TP Q shard + Q allgather)
**作者**:fhkong · **日期**:2026-06-16
**关联文档**:[plan.md](./plan.md)(原 8 阶段计划)

---

## 1. TL;DR

在 sgl 上落地 **sharded KV cache** 的 attention context parallel(AttnCP),让 GQA
模型在长上下文下 **KV 显存随 cp 维线性下降** + **decode KV 读带宽 ÷ cp**。

**核心设计**(对照 `tp_size = dp_size × attn_tp_size × attn_cp_size`):

- **Q heads 按完整 `tp_size` 切**(纯 TP shard,每 rank 持 `H/TP` 个 q-head),**Q sequence 不切**
- **KV heads 按 `attn_tp = tp_size / attn_cp_size` 切**(每 rank 持 `K/attn_tp` 个 kv-head)
- **KV sequence 按 `attn_cp` 切**(每 rank 持 `S/cp` 个 token)
- **KV pool 真切片**(每 rank 物理上只占 `1/cp` slot)
- **Attention compute(prefill / decode 共用)**:
  1. Q allgather on head 维(sharded_kv_cp_group 内,4 个 rank 各 6 heads → 临时 12 heads)
  2. segment loop FA(allgather 后 Q × 本地 1/cp KV,per-owner-chunk mask + 本地 LSE merge)→ partial
  3. `cp_lse_ag_out_rs`(LSE allgather + heads reduce_scatter)→ 每 rank 收回 `H/TP` heads
- **o_proj 走纯 TP**(RowParallel + 全 TP all_reduce),无 sharded_kv_attn_tp_group 特殊路径
- **sharded-KV logical topology**:`tp_rank = sharded_attn_tp_rank * cp_size + sharded_cp_rank`(cp 最快)

预期收益(WeLM v4 80BA3B,`H=24, K=2, tp_size=4, attn_cp=2`,attn_tp=2,256k 上下文):

| 指标 | 现状 V1 | 本方案目标态 |
|---|---|---|
| KV 显存 / rank | replicated,**不省** | **÷ 2**(随 cp 线性扩展)|
| 集群 KV 容量上限 | 固定 | **随 cp 线性扩展** |
| Decode KV 读带宽 / rank | 全长读 | **÷ 2** |
| Prefill attention 算力 / rank | full S² | **S × S/cp,真分摊** |
| Q 通信(per layer) | 无 | head 维 allgather(sharded_kv_cp_group 内,size = `S × H/TP × D`) |
| Output 通信(per layer) | 无(走 attn_tp all_reduce) | LSE allgather + heads reduce_scatter |
| MLP/MoE 算力 ÷ cp | ✓(V1 cp_split hidden) | ✗(代价,首版求稳)|
| 实现复杂度 | 已落地 ~530 行 | **核心 ~1500-1900 行,PD transfer 另计 ~250-400 行**,核心难点在 logical topology + 切片 pool + runtime 特性兼容 |

核心实现工作量:**~6-8 周**,3 阶段独立可发布、可回滚(详见 §6)。其中
eager correctness 只是开发里程碑,最终交付必须支持 WeLM kv mirror / over encoding /
chunked prefill / cuda graph / overlap schedule。PD 分离作为后置 Phase 4 实现。

PD 分离(disaggregation)是后续必须支持项,但不能只靠把 prefill/decode attention backend
都切到 FA3 解决。sharded-KV 下 PD 还需要改 KV transfer 协议、rank mapping 和
DUMMY slot 过滤。当前设计先限定同拓扑 P/D,详见 §3.9。

### 1.1 当前实现状态(2026-06-18)

当前代码已经落地的是 **correctness-first sharded-KV** 路径:
- Persistent KV cache 已按 CP owner 切片,非 owner token 使用 DUMMY slot 0;常驻 KV
  显存随 `attn_cp_size` 下降。
- Attention compute 当前为了精度对齐,先在 sharded CP group 内把本地 sharded KV
  临时 all-reduce/gather 成 dense full-KV table,再调用 FA3。也就是说,当前实现先保证
  "KV cache residency sharded",但还不是本文目标态的
  "Q allgather + per-owner-chunk segment FA + LSE merge + output reduce-scatter"。
- 因此,当前已经验证的是 **省常驻 KV 显存 + TP4 精度严格对齐**;prefill 算力分摊、
  decode KV 读带宽下降、长上下文临时 buffer 峰值等性能收益,仍需要后续 ring/LSE
  目标态实现和专项 benchmark 验证。

**已验证可用功能**:

| 功能 | 当前状态 | 验证说明 |
|---|---|---|
| 基础启动参数 | 支持 | `--attn-cp-size 2 --attn-cp-mode sharded-kv --attn-cp-kv-chunk-size 1024` |
| 拓扑 | 支持 `TP4 + CP2` | `tp_rank = sharded_attn_tp_rank * cp + sharded_cp_rank`,projection/o_proj 仍走 global TP |
| Persistent sharded KV | 支持 | TP4 K/V 各约 18.72GB/rank;TP4+CP2 K/V 各约 9.13GB/rank(WeLM v4.5 attention-sink checkpoint) |
| Q/hidden 全 sequence | 支持 | sharded-KV 下禁用旧 in-seq/zigzag hidden split;QKV projection 仍全 sequence 计算 |
| Fused QKV layout | 支持受限 | 仅支持 `attn_tp_size == total_num_kv_heads` 且 GQA 边界对齐的模型 |
| Prefill / decode | 支持 | 两阶段都走 sharded-KV residency + FA3 dense-gather correctness path |
| Chunked prefill | 支持 | 回归脚本使用 `chunked_prefill_size=1024`;controlled case 覆盖 64/512/1024/2048 prompt |
| Batched request / batched decode | 支持基础场景 | controlled `/generate` 一次发送 4 个不同长度 prompt 并生成 token |
| WeLM kv mirror opt | 支持 | 回归脚本保持 `--enable-welm-kv-mirror-opt` |
| Over encoding | 支持 | 回归脚本保持 `--enable-over-encoding` |
| Ordinary decode cuda graph | 支持 | 回归脚本保持 cuda graph on,`--cuda-graph-max-bs 16` |
| WeLM SWA | 支持当前 dense-gather path | FA3 看到 dense full-KV table,窗口语义与 TP4 对齐 |
| Attention sink | 支持当前 dense-gather path | `/home/fhkong/models/WeLM-v4.5-80B-A3B-Instruct-for-debug-0529` 48 层 sink 全开,TP4 vs TP4+CP2 回归 0 diff |

**精度回归记录**:

| 模型 | 配置 | 结果 |
|---|---|---|
| `/home/fhkong/models/80a3_v4d5_256k_merge_thinking_kimi_k25_0502_20260503_032335/epoch_003_step_0002610` | TP4 baseline vs TP4+CP2 sharded-KV,FA3,chunked prefill,kv mirror,over encoding,cuda graph | 100/100 samples,token mismatch 0,max/mean logprob diff 0 |
| `/home/fhkong/models/WeLM-v4.5-80B-A3B-Instruct-for-debug-0529` | 同上,且 48 层 attention sink 全开 | controlled compare `max_diff=0`;MMLU/C-Eval 100/100 samples,token mismatch 0,max/mean logprob diff 0;artifact `/tmp/welmv4_attncp_precision/20260618_165056` |

**当前不支持 / 不应声明支持**:

| 功能 | 当前状态 | 原因 / 后续要求 |
|---|---|---|
| PD disaggregation | 不支持 | KV transfer 仍需 sharded owner mapping、DUMMY 过滤、bootstrap metadata 校验;详见 §3.9 |
| `page_size > 1` | 启动期拒绝 | 当前 allocator / req_to_token_pool / metadata 都是 token-slot 语义;需要 page-level owner/DUMMY |
| DP attention / AttnDP | 启动期拒绝 | 当前只支持 `dp_size=1`,`enable_dp_attention=False` |
| PP | 未验证,不要声明支持 | 需要补 PP proxy tensor、metadata 生命周期、o_proj/kv mirror/cuda graph 专项回归 |
| 非 FA3 backend | 启动期拒绝 | 当前 correctness path 依赖 FA3 `flash_attn_with_kvcache` |
| `kv_cache_dtype != auto` / FP8/FP4 KV cache | 启动期拒绝或运行期不支持 | 当前 sharded dense-gather path 不支持 KV descale tensor |
| MLA / cross attention / encoder-decoder | 不支持 | sharded-KV path 针对 WeLM/GQA MHA 首发,MLA 与 cross-attn 需要独立设计 |
| Radix cache / chunked prefix cache hit | 自动禁用 | DUMMY slot 与跨 CP prefix hit 语义尚未适配 |
| CPU KV offload / hierarchical cache | 启动期拒绝 | offload/load/free 路径必须识别 DUMMY slot,当前未实现 |
| Spec decode / cascade attention / MTP topk>1 | 运行期拒绝 | sharded-KV path 当前对 cascade attention `NotImplementedError` |
| local iROPE attention | decode 路径拒绝 | local page table/window 语义还未适配 sharded-KV |
| 异构 head layout | 启动期拒绝 | `attn_tp_size != total_num_kv_heads` 需要拆/定制 KV projection |
| Piecewise cuda graph | 未验证 | 当前精度回归显式 `--disable-piecewise-cuda-graph`;后续需要 graph bucket/static buffer 专项 |
| 高并发/abort/retract/rollback 压测 | 未完成 | allocator 单测覆盖基础 DUMMY 行为,但线上压力路径仍需专项 |

**当前主要风险**:

1. **文档目标态与当前实现形态不同**:当前 dense-gather path 保证精度和常驻 KV 省显存,
   但不代表最终 ring/LSE path 的数值、峰值显存和性能。后续切到 segment/LSE 时需要重新
   跑完整精度回归,尤其是 attention sink、SWA、over encoding、kv mirror。
2. **临时 full-KV buffer 峰值**:TP4+CP2 常驻 KV 已减半,但 attention 内部会临时构造
   dense full KV。长上下文、cuda graph capture bucket 或大 batch 下可能重新抬高峰值显存。
3. **Attention sink 后续改造风险**:dense-gather path 只调用一次 FA,天然不会重复注入
   sink;真正 segment/LSE path 必须保证 sink 只由一个 CP rank 注入一次,否则 softmax
   denominator 会重复计数。
4. **DUMMY slot 旁路风险**:任何扫描 `req_to_token_pool` 的新路径(cache write、debug dump、
   offload、PD transfer、abort/retract)如果漏过滤 slot 0,可能出现误释放、误传输或读空 KV。
5. **`cp_kv_chunk_size` 工作负载敏感**:默认 1024 有利于连续 KV segment,但短请求负载可能
   偏向低 CP rank;正确性测试建议用 128/256 覆盖 owner 边界,性能基线覆盖 512/1024/2048。
6. **组合特性缺口**:当前回归覆盖单机 TP4/CP2、kv mirror、over encoding、chunked prefill、
   ordinary cuda graph、attention sink;尚未覆盖 PD、PP、DP attention、piecewise graph、
   spec decode、高并发 abort/retract 等组合。

---

## 2. 动机

### 2.1 现状

WeLM v4 长上下文(thinking + 工具调用 256k)serving 受限于 **KV 容量**:

| 路径 | KV 冗余系数 | 备注 |
|---|---|---|
| 纯 TP | `tp / num_kv_heads` | GQA 下 K 小,attn_tp > K 后 KV head 复制 |
| 现有 allgather CP(上游 #18233) | `cp_size` | 每层 K/V allgather 全量写池,**没省一寸** |
| **V1 ring-pass-kv**(旧实验) | **`cp_size`** | 数学正确,但 KV pool 仍 allgather 写满 |
| **本方案 sharded-KV CP** | **1** | 真正切片,集群容量随卡线性扩展 |

V1 解决了 "ring 数学正确性 + MLP 算力分摊";本方案解决 "KV 内存"。两者是不同问题。

### 2.2 业界先例

| 项目 | Q 切分 | KV 切分 | Q allgather | Output merge |
|---|---|---|---|---|
| vLLM DCP(2026)| TP × DCP(细)| token | head 维 | `cp_lse_ag_out_rs` |
| NVIDIA Helix | 类似 vLLM DCP | token | head 维 | flash-decoding |
| Mimikyu(训练)| n/a | sequence(zigzag)| n/a | ring + merge_state |
| **本方案** | **TP(纯 TP shard)** | attn_tp 切 head + cp 切 sequence | **head 维(sharded_kv_cp_group 内)** | `cp_lse_ag_out_rs` |

本方案与 vLLM DCP **数学路径基本一致**,差异是:
- vLLM 把 Q 用 `TP × DCP` 双层切;本方案直接用 SGLang 的 `tp_size` 切,DCP 作为
  attention 内部的 cp 维,与 Q proj weight shard 解耦
- sharded-KV logical topology 使用 `(dp, attn_tp, cp)`,使得
  sharded_kv_cp_group 内 Q allgather 拼出来正好是本 KV head shard 对应的 GQA group

---

## 3. 设计

### 3.1 Rank 拓扑(以 `H=24, K=2, TP=4, attn_cp=2` 为例)

`tp_size = dp_size × attn_tp_size × attn_cp_size` → `4 = 1 × 2 × 2`

**sharded-KV logical topology**:本设计在 sharded-KV mode 内部解释
`tp_rank = sharded_attn_tp_rank * cp_size + sharded_cp_rank`(cp 是 fastest-changing axis,与 SGLang 默认 attention CP API 相反)。
物理 worker rank、global `tp_group`、projection/o_proj/MLP/MoE 的 TP 语义不变。

```
tp_rank → (sharded_attn_tp_rank, sharded_cp_rank):
  tp 0 → sharded_attn_tp=0, sharded_cp=0
  tp 1 → sharded_attn_tp=0, sharded_cp=1
  tp 2 → sharded_attn_tp=1, sharded_cp=0
  tp 3 → sharded_attn_tp=1, sharded_cp=1
```

各 rank 持的切片(GQA group = `H/K = 12`,kv-head 0 服务 q `{0..11}`,kv-head 1 服务 q `{12..23}`):

| GPU | tp_rank | sharded_attn_tp_rank | sharded_cp_rank | Q heads(TP=4 切)| KV heads(attn_tp=2 切)| KV tokens(cp=2 切)|
|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | `{0..5}` | `{0}` | owner chunks with `owner=0` |
| 1 | 1 | 0 | 1 | `{6..11}` | `{0}` | owner chunks with `owner=1` |
| 2 | 2 | 1 | 0 | `{12..17}` | `{1}` | owner chunks with `owner=0` |
| 3 | 3 | 1 | 1 | `{18..23}` | `{1}` | owner chunks with `owner=1` |

(token owner 按 `owner_cp_rank = (global_token_id // cp_kv_chunk_size) % cp_size` 决定,见 §3.7。)

**GQA 兼容性**:
- sharded_kv_cp_group `{0,1}`(sharded_attn_tp=0)合并持 q `{0..11}` ⊂ kv-head 0 的 GQA group ✓
- sharded_kv_cp_group `{2,3}`(sharded_attn_tp=1)合并持 q `{12..23}` ⊂ kv-head 1 的 GQA group ✓

**关键 group**:
- **sharded_kv_cp_group**(同 sharded_attn_tp_rank,不同 sharded_cp_rank):`{0,1}` 与 `{2,3}` — q-head 不同(`{0..5}` vs `{6..11}`)但合起来是 `H/attn_tp` 个 head;KV head 相同;token 互补。**Q allgather on head 维与 cp_lse_ag_out_rs 都在这里**。
- **sharded_kv_attn_tp_group(概念分组)**(同 sharded_cp_rank,不同 sharded_attn_tp_rank):`{0,2}` 与 `{1,3}` — token 相同,kv-head 不同。当前实现不构造该 group,`o_proj` 走完整 tp_group。
- **tp_group**(全部 4 个 rank):全 TP 范围,`o_proj` / MLP / MoE 的 all_reduce 用这个。

**首版约束**:
- **强限制**:首版 fused `QKVParallelLinear(tp_size=tp_size)` 只支持
  `attn_tp_size == num_kv_heads`。此时 standard QKV 的 KV replica factor 正好等于
  `cp_size`,同一个 sharded_kv_cp_group 内 rank 复制同一个 KV head shard,只在
  sequence owner 上互补。若 `num_kv_heads > attn_tp_size` 或
  `num_kv_heads < attn_tp_size`,不能直接复用 fused QKV,必须拆/定制 KV projection,
  让 Q 按完整 TP 切、KV 按 attn_tp 切。
- 必须满足 `num_q_heads % tp_size == 0` 且 `(num_q_heads / tp_size) × cp_size == num_q_heads / attn_tp_size`(sharded_kv_cp_group 内 Q allgather 拼出一个 KV head shard 对应的 GQA group)。
- 必须满足 q-head GQA 边界对齐 sharded attention TP shard:`num_q_heads / attn_tp_size` 必须是 `num_q_heads / num_kv_heads` 的整数倍。
- 首版只支持单 DP(`dp_size=1`,`enable_dp_attention=False`)、FA3 backend、
  `kv_cache_dtype=auto`、`page_size=1`;radix cache、CPU KV offload、FP8/FP4 KV
  cache 可先拒绝或 fallback。WeLM kv mirror / over encoding / chunked prefill /
  cuda graph / overlap schedule 是 GA 必须支持项,不能作为最终上线条件禁用。
- PD 分离当前先不作为基础 correctness 里程碑的一部分。后续实现时,首版只支持
  prefill/decode 同拓扑的 sharded-KV PD:两边 `tp_size`、`attn_cp_size`、
  `attn_tp_size`、`pp_size`、`page_size`、`attn_cp_kv_chunk_size`、
  `kv_cache_dtype`、FA3 backend 和模型 head layout 必须一致;不支持异构 P/D
  拓扑、decode CP=1 拉全量 KV、page_size > 1、decode radix cache、CPU KV offload。

### 3.2 Storage 切分(prefill / decode 共用)

| 张量 | 形状 / rank | 永久 vs 临时 |
|---|---|---|
| `hidden_states` | `[seq_len, hidden_dim]`(全 rank 相同,不切 sequence) | 永久 |
| `Q`(QKV proj 输出) | `[seq_len, H/TP, D]`(prefill: seq_len=S; decode: B) | 永久(weight loader 等都按这个 shape 写) |
| `K_new` / `V_new`(本步算) | `[seq_len, K/attn_tp, D]` | 临时(写池后丢) |
| Q allgather 后(临时) | `[seq_len, H/attn_tp, D]` | **临时,仅 attention 内部** |
| KV pool(物理) | `[max_total_tokens / cp_size, K/attn_tp, D]` | **永久 sharded** |

**核心不变量**:
- Q proj 输出 / o_proj 输入用 `H/TP` heads(SGLang 现有 weight 布局)
- Attention compute 内部临时升级到 `H/attn_tp` heads(为了配 GQA group)
- 升降都靠 sharded_kv_cp_group collective(Q allgather + cp_lse_ag_out_rs)
- projection / o_proj / gate_proj 使用 global tensor TP rank/size;attention collective 使用
  sharded-KV logical rank/group。不要把 `get_attention_tp_rank()` 混用于 projection shard。

**QKV proj 实现**:

首版**沿用现有 `QKVParallelLinear(tp_rank=global_tp_rank, tp_size=tp_size)`**,但只在
`attn_tp_size == num_kv_heads` 时启用:
- Q 部分:`H/TP` 个 head per rank ✓
- K, V 部分:以 WeLM `K=2,TP=4,CP=2,attn_tp=2` 为例,standard QKV
  触发 KV head replica,每 rank 1 个 kv-head,且 sharded_kv_cp_group 内 2 个
  rank 复制同一个 kv-head ✓
- 物理上的"K replica"在我们的语义里就是"attn_tp head shard,sharded_kv_cp_group 内复制" — 正好和 KV cache 按 cp owner 切片对应

若 `attn_tp_size != num_kv_heads`,standard fused QKV 的 KV replica / shard 规则不再等价
于 "KV 按 attn_tp 切 head + CP 内复制",首版直接拒绝启动。后续要支持该类模型时,
需要拆 `q_proj`/`kv_proj` 或定制 `QKVParallelLinear` 的 KV shard mapping:
Q 仍按完整 TP 切 head,KV 必须按 attn_tp 切 head。

代价是每 rank 都把全 sequence 的 K, V 算了一遍(其中 1/cp 是自己 owner,其余在写池时丢弃)。换来不需要拆 q_proj/kv_proj,所有权重加载、量化、presharded checkpoint 都不用动。后续优化再考虑拆开。

### 3.3 Attention 数据流(prefill / decode 共用,rank 0 视角)

为了说清楚,先看 **prefill,S = 8000,cp_kv_chunk_size = 1024**。decode 只是 `seq_len=B` 的特例。

```
Step 0: hidden_states 入口
   每个 rank 都持 hidden[8000, hidden_dim]    ← 不切 sequence

Step 1: QKV proj(纯 TP fused)
   QKVParallelLinear(tp_rank=tp_rank, tp_size=4)

   rank 0 出: Q[8000, H/TP=6, D]                     (q-heads {0..5})
             K_full[8000, K/attn_tp=1, D]            (kv-head {0},sharded_kv_cp_group 内复制)
             V_full[8000, 1, D]

   rank 1 出: Q[8000, 6 (q-heads {6..11}), D]
             K_full[8000, 1, D]                      (同样的 kv-head {0})
             V_full[同]

   rank 2 出: Q[8000, 6 (q-heads {12..17}), D]
             K_full[8000, 1 (kv-head {1}), D]
             V_full[同]

   rank 3 出: Q[8000, 6 (q-heads {18..23}), D]
             K_full[8000, 1, D]                      (kv-head {1})

Step 2: 按 CP owner 过滤 K, V,写入本地 sharded pool
   cp_kv_chunk_size = 1024:
     chunk 0: tokens [0, 1024),    owner_cp = 0 % 2 = 0
     chunk 1: tokens [1024, 2048), owner_cp = 1 % 2 = 1
     chunk 2: tokens [2048, 3072), owner_cp = 0
     ...

   rank 0 (cp=0): 写 token [0,1024), [2048,3072), [4096,5120), [6144,7168)
   rank 1 (cp=1): 写 token [1024,2048), [3072,4096), [5120,6144), [7168,8000)
   rank 2 (cp=0): 同 rank 0 的 token 范围,但写 kv-head {1}
   rank 3 (cp=1): 同 rank 1 的 token 范围,写 kv-head {1}

   ★ 各 rank 写完后丢掉非自己 owner 的 K_full/V_full row。
   ★ 当前层 attention 仍用本 rank owned K, V(从 owner-filtered 出来,不读池),
     ring-style sharded layout 与池读完全一致。

Step 3: sharded_kv_cp_group 内 Q allgather on head 维
   sharded_kv_cp_group {0,1} 内做 all_gather(query, dim=1):
       rank 0 持 Q[8000, 6 (heads {0..5}), D]
       rank 1 持 Q[8000, 6 (heads {6..11}), D]

       allgather 后:
       rank 0: Q_global[8000, 12 (heads {0..11}), D]
       rank 1: Q_global[8000, 12 (heads {0..11}), D]   ← 同样的内容

   sharded_kv_cp_group {2,3} 同理,allgather 后两 rank 都持 q-heads {12..23}。

   ★ 通信量(每 rank 收): (cp_size-1) × S × (H/TP) × D × 2 byte
                       = 1 × 8000 × 6 × 128 × 2 = 12 MB
   ★ allgather 后每 rank 持的 12 heads 与本 rank KV head shard 0
     的 GQA group 完全对应。

Step 4: 本地 segment loop FA(全 Q_global × 本 rank 1/cp KV)
   rank 0 输入:
       Q_global[8000, 12, D]                          ← 全 S × allgather 后 head
       K_owned[4000, 1, D]                            ← 自己 owned token 的 K
       V_owned[4000, 1, D]
       per-owner-chunk segment mask                   ← 见 §3.4

   for each owned segment:
     FA(Q_global visible rows, K_segment, V_segment, segment mask)
     merge_state_v2(local_partial, segment_partial)
     → partial_out[8000, 12, D]
       partial_lse[8000, 12]

   ★ 这个 partial 是 "q-heads {0..11} 全 8000 query × 本 rank 4000 owner KV" 的
     部分 attention,正确归一化要等 cp 邻居的 partial 合并。

Step 5: cp_lse_ag_out_rs(sharded_kv_cp_group {0,1} 内)
   ① allgather partial_lse on sharded_kv_cp_group(dim=cp 维):
      rank 0 和 rank 1 都拿到 [lse_rank0[8000,12], lse_rank1[8000,12]]
   ② correct_attn_out(triton kernel):用 logsumexp 算合并权重,
      把每 rank 自己的 partial_out scale 到全局 softmax 分母下
   ③ reduce_scatter on heads dim(sharded_kv_cp_group {0,1} 内):
      合并后的 [8000, 12, D] reduce_scatter → 每 rank 拿回 6 heads
      rank 0: attn_out[8000, 6 (heads {0..5}), D]    ← 完整 attention output
      rank 1: attn_out[8000, 6 (heads {6..11}), D]

   通信量:
     allgather lse:  1 × 8000 × 12 × 4 byte (fp32 LSE) = 384 KB
     reduce_scatter out: 1 × 8000 × 12 × 128 × 2 byte = 24 MB → 实际 reduce_scatter 后接收 12 MB

Step 6: o_proj + 全 TP all_reduce
   attn_out[8000, 6, D] · W_o (RowParallel tp=4, tp_rank shard)
     → [8000, hidden_dim] (partial,等 all_reduce)
   tp_group all_reduce(全 4 个 rank)
     → hidden[8000, hidden_dim] (每 rank 相同)

   ★ 这里走的就是 SGLang 标准 RowParallelLinear,不需要 sharded_kv_attn_tp_group 特殊路径。
   ★ 数学上正确:每 rank 计算 (本地 H/TP heads × W_o 的对应行 slice)的 partial,
     all_reduce 后等于 (full heads × full W_o) 的结果。CP rank 之间的 attention
     output 是 head 互补的,reduce_scatter 后正好填满 TP 的 head shard,所以 all_reduce
     不会重复相加。

Step 7: 后续 MLP / MoE
   每 rank 处理全 8000 tokens(继续按 full tp=4 切 intermediate)
   → hidden[8000, hidden_dim](下一层输入)
```

**Decode 的差异**:
- `seq_len = B`(batch_size)而不是 `S`
- Step 2 写池只追加新 token(B 个,只有 owner sharded_cp_rank 写;不是 owner 的 sharded CP rank skip)
- Step 4 的 K, V 来源是本 rank pool 历史 + 本 rank owned new K, V(如果新 token 属本 rank owner);如果新 token 不属本 rank owner,本 rank 只有历史 KV
- Batched decode:每个 request 独立计算 `owner_cp_rank` / `local_page_table` / `local_cache_seqlens`,某些 rank 对某些 request 可能 local KV 长度为 0,显式构造 `partial_out=0, partial_lse=-inf` 而不是调 FA

**Per-layer 通信账(prefill,S=8000,K=2)**:
- Q allgather(head 维,sharded_kv_cp_group 内):每 rank 接收 12 MB
- LSE allgather(sharded_kv_cp_group 内):几百 KB
- Output reduce_scatter(sharded_kv_cp_group 内):接收 12 MB
- o_proj all_reduce(全 TP=4):约 32 MB(`S × hidden_dim × 2 byte`)
- **无 KV ring P2P,无 KV 写池 allgather**

### 3.4 因果 mask schedule

每 rank 的本地 KV 是若干个非连续 owner chunk(rank 0 持 `[0,1024)`、`[2048,3072)` 等)。FA 调用要按 chunk 维度处理:

设 `cp_kv_chunk_size = C`,owner chunk `j` 覆盖全局 token `[j·C, min((j+1)·C, S))`,本 rank 持的 chunk 集合为 `{j : (j % cp_size) == sharded_cp_rank}`。

对每个本 rank owned chunk `[k0, k1)`,Q allgather 后的全 S query 分三段处理:

| Q 范围 | 与 chunk 的关系 | 处理 |
|---|---|---|
| `[0, k0)` | Q 早于 chunk | 全不可见(causal),partial = 0,LSE = `-inf` |
| `[k0, k1)` | Q 与 chunk 重叠 | local causal:`q_pos < k_pos` mask 掉 |
| `[k1, S)` | Q 晚于 chunk | chunk 全可见,no causal mask |

具体调用方式(portable 路径,Phase 1 用):

1. 对 `[k0, k1)` 调一次 causal FA,输出 scatter 到全 S partial 的对应行
2. 对 `[k1, S)` 调一次 non-causal FA,输出 scatter 到全 S partial 的后续行
3. `[0, k0)` 不调 FA,partial 行直接置 0 + LSE 置 `-inf`
4. 对本 rank 持有的所有 chunk 各做一遍上面三段,在本地 `merge_state_v2` 合并

后续性能优化考虑 FA3 `mask_mod` / block mask 把多段合并成一次 FA。

**SWA**(WeLM 奇数层 sliding window):mask 在上面基础上额外要求 `k_pos >= q_pos - sliding_window`,在 segment FA 调用前按全局 position 显式 mask;不能直接用 FA 的 `window_size` 参数(因为本地 KV 是 chunked 非连续,FA 的 window 参数假设 K 是连续序列)。

**Attention sink**(WeLM):sink 是全局 softmax denominator 里的单个 `+exp(sink)` 项,**只能由一个 sharded CP rank 注入**。约定 `sharded_cp_rank=0` 注入:
- 如果 `sharded_cp_rank=0` 对该 request 本地有 owned KV,正常 FA + 在 LSE 末尾 merge 一个 `(out=0, lse=sink)` partial
- 如果 `sharded_cp_rank=0` 本地 KV 为空,显式构造 `(out=0, lse=sink)` 作为它的 partial,参与 cp_lse_ag 合并

### 3.5 Headwise 参数切片规则

本方案有两种 head 视角,实现时必须分清:

| 阶段 | head 数 / rank | 说明 |
|---|---|---|
| QKV projection 输出 | `H/TP` | global TP shard,用于权重加载和本 rank 持久张量 |
| attention 内部 | `H/attn_tp = (H/TP) × cp` | sharded_kv_cp_group 内按 head 维 allgather 后的临时视图 |
| attention 输出 | `H/TP` | `cp_lse_ag_out_rs` reduce_scatter 后回到 global TP shard |
| o_proj / gate_proj 输入 | `H/TP` | 继续使用标准 TP row/column shard |

**Q head order**:
- local Q head range:`[tp_rank × H/TP, (tp_rank + 1) × H/TP)`
- sharded_kv_cp_group 内按 `sharded_cp_rank` 升序 concat,得到
  `[sharded_attn_tp_rank × H/attn_tp, (sharded_attn_tp_rank + 1) × H/attn_tp)`
- 这个范围必须完整落在一个或多个完整 GQA group 内;首版用 §3.1 的 assert 保证。

**WeLM attention sink**:
- `attn_sink` 是 per-q-head 参数,projection 侧仍按 `H/TP` shard 加载。
- attention 内部需要和 Q 一样在 sharded_kv_cp_group 内 allgather 成
  `[H/attn_tp]` 的 `sink_global`。
- sink 只能注入一次,否则 cp_lse_ag 会把 softmax denominator 重复计数。约定
  `sharded_cp_rank == 0` 注入 `sink_global`,其他 CP rank 不传 sink。
- 如果 `sharded_cp_rank == 0` 对某个 request 本地 KV 为空,也要构造
  `(partial_out=0, partial_lse=sink_global)` 参与合并。

**WeLM gate / o_norm**:
- `gate_proj` 仍按 global TP 切 `H/TP` heads,必须在 `cp_lse_ag_out_rs`
  reduce_scatter 之后作用到本地 attention output;不要对 allgather 后的
  `H/attn_tp` 临时 head 视图做 gate。
- `o_proj` 仍是 `RowParallelLinear(tp_size=tp_size)` + full `tp_group`
  all_reduce。`o_norm` 作用在 o_proj 后的 hidden 维输出,不参与 sharded_kv_cp_group。

### 3.6 Chunked prefill 与 prefix cache

注意 "chunked" 这里只指 **prefill chunk**(scheduler 把长 prompt 分多个 chunk 喂模型);decode 阶段没有 prefix/self 双 partial 问题。

Chunked prefill 下,本次 chunk 的 Q 要 attend 到:
- (a) 历史 cached prefix(在 pool 中,sharded 已存)
- (b) 本 chunk 自己的 K/V(刚算出,owner 已写池)

由于 prefill / decode 都走同一条 attention path(Q allgather + segment loop FA + cp_lse_ag_out_rs),chunked prefill 不需要拆 prefix/self 双 partial。统一从本 rank pool 读 owned KV(包括历史 prefix + 本 chunk 刚写的 owner),按 owner segment loop 做本地 partial:

```
本次 chunk 进来:
  - chunk Q[chunk_size, H/TP, D] per rank
  - chunk K/V 算完按 owner 写入本 rank sharded pool
  - 此时本 rank pool 里有:历史 prefix 中本 rank owner 的部分 + 本 chunk 中本 rank owner 的部分

Attention compute(同 §3.3):
  - sharded_kv_cp_group 内 Q allgather → Q_global[chunk_size, H/attn_tp, D]
  - 本 rank 遍历 owned KV segments:
      FA(Q_global visible rows × K/V segment,带 per-segment mask)
      merge_state_v2 合并本地 segment partial
  - cp_lse_ag_out_rs → 收回 H/TP heads
```

**关键正确性要求**:
- `cp_kv_chunk_size` 是 owner 映射的一部分,必须是 request/cache 级稳定常量。一旦 request 启动时决定了 owner 切分边界,之后所有 prefill chunk + decode step 都按这个边界写池
- 不能为了不同 chunk size 动态改 owner 边界,否则历史 KV 在 pool 的物理位置会乱
- decode 阶段没有 chunked prefill 的 prefix/self 语义,但 decode 新 token 仍然沿用同一个
  `cp_kv_chunk_size` owner 规则追加 KV。

**为什么需要 `cp_kv_chunk_size`**:
- 如果按 token round-robin(`owner = pos % cp`)切,负载最均匀,但每 rank 的 KV 是
  stride token,segment 长度接近 1,portable FA 要退化成大量小 kernel,cache write 也不连续。
- 如果按大 chunk 切,每 rank 的 KV segment 连续,FA/mask/metadata 都简单,但短请求或
  一个 scheduler prefill chunk 内可能只落到少数 CP rank,负载更倾斜。
- 因此 `cp_kv_chunk_size` 是 **FA segment 粒度与负载均衡之间的固定折中**,不是运行中
  可以随意改变的调参。

**参数策略**:
- 外部只暴露三项 AttnCP 参数:
  `--attn-cp-size C --attn-cp-mode sharded-kv --attn-cp-kv-chunk-size N`。
  `sharded-kv` 同时打开 prefill 的 sharded-KV 写池和 decode 的 sharded-KV 读池。
- `--attn-cp-kv-chunk-size` 默认 `1024` logical tokens。
- 该值在 server 启动时固定并写入日志;不要按 request / batch / chunk 动态变化。
- 集成和 correctness 测试建议传 `128` 或 `256`,让短 prompt 也能跨 CP owner
  边界,覆盖多 rank KV ownership、batched decode 和 chunked prefill。
- 性能基线至少覆盖 `512 / 1024 / 2048`,根据目标 workload 的 prompt length 分布决定
  生产默认值是否保持 1024。
- 如果未来支持 `page_size > 1`,要求 `cp_kv_chunk_size % page_size == 0`,
  owner 边界必须 page 对齐。
- 当前 sharded-KV AttnCP **不支持 `page_size > 1`**。`cp_kv_chunk_size % page_size == 0`
  只是未来支持的必要条件,不是充分条件。原因是当前 owner / DUMMY / free / transfer
  语义都是 token-level:非 owner token 在 `req_to_token_pool` 中写 DUMMY slot 0,
  而 `page_size > 1` 后物理分配、page table、cache write、PD transfer 都变成 page
  粒度。强行放开会导致一个 page 内混入不同 owner token、DUMMY 被当作真实 page
  传输/释放,或者 attention metadata 把 page id 当 token slot 使用。

### 3.7 KV pool 物理布局

每 rank 的物理 slot 数:

```
slot_count_per_rank = max_total_tokens / cp_size × num_kv_heads_local × head_dim
                    = max_total_tokens / cp × (K / attn_tp) × D
```

**Slot owner 规则(首版固定)**:
- `page_size = 1`。`page_size > 1` 启动时必须拒绝或 fallback 到非 sharded-KV 路径,
  不能 silent 进入 sharded-KV。
- chunk-granular owner:`owner_cp_rank = (global_token_id // cp_kv_chunk_size) % cp_size`
- 同一个 owner chunk 内 token 归同一个 cp rank,prefill / decode 都按这个规则

**为什么首版限制 `page_size=1`**:
- 当前 `CPShardedKVAllocator` 以 token slot 为最小所有权单位;`page_size > 1` 需要整
  page 分配/释放,不能继续让一个 page 内同时存在 owner token 和 non-owner DUMMY token。
- 当前 `req_to_token_pool` 用 slot 0 表示 non-owner token。`page_size > 1` 时需要
  DUMMY page 或 page-level invalid bitmap,否则 release/rollback/cache write 无法区分
  真实 page 与 DUMMY page。
- 当前 attention metadata compact 直接过滤 `slot != 0` 并产出 local token positions。
  `page_size > 1` 时 page table 需要展开到 token positions,并正确处理最后一个
  partial page;FA3 路径也不能把 page id 当 token slot 使用。
- 当前 PD transfer / staging 里很多路径按 page indices 搬运。sharded-KV 下必须按
  logical owner page 过滤,并保证 source/destination page 对齐;否则会把其他 CP owner
  的 token 一起传过去。

**未来支持 `page_size > 1` 的最低要求**:
- `cp_kv_chunk_size % page_size == 0`,owner chunk 边界必须 page 对齐。
- allocator 改为 page-level ownership:owner page 分配真实 page,non-owner page 写
  DUMMY page 或显式 invalid page bitmap。
- metadata compact 能从 page table 展开 token positions,处理 partial last page,
  并在 SWA/causal/over-encoding mask 中使用 expanded logical position。
- cache write/free/rollback/abort/SWA eviction/PD transfer 全部按 page-level DUMMY
  过滤,不能释放或传输 DUMMY page。
- PD transfer 需要按 owner page 过滤,empty transfer 必须是合法 success。

**Block table**:每 rank 的 page table 仅指向自己的 slot;DUMMY slot 占位非 owner token(读 0,写 no-op)。

**Radix cache**:首版 disable(prefix 命中跨 cp 是后续工作)。

新加 `CPShardedKVAllocator`:
- Slot 分配:owner rank 分配真 slot,其他 rank 写入 DUMMY slot id
- 真 slot 从 1 开始分配,slot 0 永久保留为 DUMMY,不进入 freelist、不释放、不计容量
- `set_kv_buffer(layer, slots, k, v)`:只对 `slots != 0` 写真值,DUMMY no-op
- `get_kv_buffer(layer)`:返回本 rank 的 owned token 切片
- Metadata 为每个 request 生成 `local_page_table` / `local_cache_seqlens` / `local_kv_positions`(batched 下不同 request 的 owned token 数可能不同)

**DUMMY slot 生命周期规则**:
- `req_to_token_pool` 仍保存完整 logical sequence 长度;非 owner token 的 entry 写 0。
- attention metadata compact 时必须过滤 `slot != 0`,并同步产出对应的 global
  token position,供 causal/SWA segment mask 使用。
- free / retract / abort / request finish / SWA eviction 只能释放 `slot != 0` 的唯一真 slot;
  DUMMY slot 0 必须忽略。
- cache write、CPU offload、prefix cache、debug dump 等扫描 `req_to_token_pool` 的路径,
  在 sharded-KV mode 下都必须显式过滤 DUMMY。首版 radix cache / CPU KV offload 禁用,
  但 allocator 单测仍要覆盖这些 release/rollback 行为。
- alloc 失败 rollback 只回滚本 rank 已分配真 slot;DUMMY 不参与 rollback。

### 3.8 必须支持的 SGLang runtime 特性

这些特性不是可选兼容项。Phase 2 可以先用 eager path 对拍,但最终交付不能要求用户关闭它们。

**Chunked prefill**:
- `extend_prefix_lens` / `extend_seq_lens` 仍按 SGLang 现有调度语义维护;本方案只改变
  KV residency,不改变 scheduler 的 chunk 切分。
- 当前 chunk 的 K/V 先按 owner 规则写入 sharded pool;attention 从本 rank pool compact 出
  `prefix + current chunk` 的 owned KV,走同一套 segment loop FA + `cp_lse_ag_out_rs`。
- 不允许为了 chunked prefill 回退到 full-KV allgather;否则 KV 带宽和显存收益会在长
  prefix 下消失。

**WeLM kv mirror opt**:
- `enable_welm_kv_mirror_opt` 必须支持。mirror contraction 改变的是参与 attention 的 Q
  row 集合,但 KV 写池仍要按原 logical token position/owner 完成,不能因为 Q contraction
  漏写 prefix/current chunk 的 K/V。
- `custom_last_index`、`kv_mirror_active_batch_indices`、`kv_mirror_output_size`、
  `welm_kv_mirror_last_q_indices` 必须贯穿 Q allgather、segment attention、
  `cp_lse_ag_out_rs` 和 o_proj scatter/rebuild。Q allgather 的 head 维顺序不能改变这些
  row-level indices 的语义。
- PP proxy tensor / cuda graph kv-mirror static buffer 仍按现有 WeLM 语义传递 mirror
  states;sharded-KV 只改变每层 KV cache residency,不改变跨层 mirror state 的 key/value
  内容和生命周期。

**Over encoding**:
- `enable_over_encoding` / `oe_context` / `scale_seq_factor` 必须支持。owner 计算、DUMMY
  写入、`local_kv_positions` 和 segment mask 都必须使用 over-encoding 后的 logical
  position,不能混用原始 prompt token index。
- `req_to_token_pool` 的 logical length 与 `positions` / `out_cache_loc` 的 expanded
  row 数必须一致;释放、rollback 和 chunked prefill metadata compact 都按 expanded
  logical sequence 过滤 DUMMY。

**Cuda graph / piecewise cuda graph**:
- Decode cuda graph 和 piecewise prefill graph 都必须支持。eager fallback 只用于开发调试,
  不能作为 GA 交付条件。
- Q allgather buffer、partial output/LSE buffer、`cp_lse_ag_out_rs` workspace、
  compacted local page table/positions buffer 都需要静态预分配或按 capture size
  bucket 化;graph replay 期间不能在 Python 侧重建 shape-dependent tensor。
- Segment loop 的最大 segment 数由 `capture_num_tokens`、`cp_kv_chunk_size` 和
  `attn_cp_size` 推导;capture bucket 内用 padding/valid mask 固定 kernel launch 结构。

**Overlap schedule / TBO**:
- Scheduler overlap 不能因为 sharded-KV 自动关闭。`ForwardBatch` 中新增的
  sharded-KV metadata 必须支持 ping-pong/overlap 生命周期,不能在 GPU forward 仍在使用时
  被下一批 CPU 调度覆盖。
- WeLM 的 `model_forward_maybe_tbo` 当前会在 kv mirror contraction 时绕开;sharded-KV
  需要保持这个语义,并保证非 mirror contraction 场景下 TBO 能继续运行。
- 所有 sharded_kv_cp_group collective 必须在模型 forward stream 上按现有 overlap
  同步规则排队,避免和 KV cache 写入、metadata compact 发生跨 stream 读写竞争。

### 3.9 PD 分离(disaggregation)当前限制与后续实现边界

PD 分离下 prefill server 负责算 prompt KV,decode server 负责后续 decode。对于
sharded-KV AttnCP,attention backend 统一用 FA3 只是必要条件之一;真正需要适配的是
KV cache transfer 必须保持 "KV 按 CP owner 永久切片" 的语义。

**当前限制**:
- 在 sharded-KV PD 实现完成前,`--attn-cp-mode sharded-kv` 与
  `--disaggregation-mode prefill/decode` 应该显式拒绝或 guarded fallback,避免 silent
  走成 replicated KV。
- 首版 PD 只支持同拓扑 P/D:
  - prefill 和 decode 都启用 `--attn-cp-mode sharded-kv`
  - `tp_size`、`attn_cp_size`、派生 `attn_tp_size`、`pp_size`、`page_size=1`、
    `attn_cp_kv_chunk_size`、`kv_cache_dtype`、FA3 backend、模型 head layout 全部一致
  - `dp_size=1`,`enable_dp_attention=False`
  - decode 侧也必须 `attn_cp_size > 1`,不能用 decode CP=1 去接收全量 KV
- 首版不支持异构 P/D 拓扑:
  - prefill CP size 与 decode CP size 不一致
  - prefill TP/PP 与 decode TP/PP 不一致
  - decode rank 从多个 prefill CP rank 重组 owner chunk
  - page_size > 1、decode radix cache、CPU KV offload、FP8/FP4 KV cache

**Rank mapping 要求**:
- 当前 PD 公共逻辑假设 decode `attn_cp_size == 1`。sharded-KV PD 需要新增分支:
  当 P/D 同拓扑时,decode `(attn_tp_rank, sharded_cp_rank, pp_rank)` 只从 prefill
  同 `(attn_tp_rank, sharded_cp_rank, pp_rank)` 拉取 KV。
- 所有 prefill CP rank 都必须参与 transfer。不能沿用 "只让 cp0 发送,其他 CP rank
  dummy success" 的逻辑,因为每个 CP rank 都拥有不同 owner chunk。
- `required_prefill_response_num` 必须按新的 TP/CP/PP 映射重新计算;即使某个 rank
  对某个 request 没有 owner token,也要发送空 transfer 的 success,避免 decode 等待卡死。

**Bootstrap metadata 要求**:
- Prefill 注册到 bootstrap server 时需要带上并在 decode 侧校验:
  `attn_cp_mode`、`attn_cp_kv_chunk_size`、`page_size`、`kv_cache_dtype`、
  `attn_tp_size`、`attn_cp_size`、`pp_size`、owner policy/version。
- owner policy 固定为:
  `owner_cp_rank = (logical_token_pos // attn_cp_kv_chunk_size) % attn_cp_size`。
  P/D 两侧必须使用同一套 logical position,包括 over encoding 后的 expanded position。

**KV transfer 语义**:
- Decode 侧预分配必须使用 sharded allocator:owner token 分配真实 slot,非 owner token
  在 `req_to_token_pool` 中写 DUMMY slot 0。不能在 PD transfer 时为 decode 侧展开成
  dense/full KV cache。
- Transfer 只传输 `src_slot != 0 && dst_slot != 0` 的 KV。DUMMY slot 不能注册 RDMA
  读写,也不能参与 release/rollback。
- 现有 CP transfer 的连续 page-range filter 不适用于 sharded-KV。sharded-KV 必须按
  logical token position 的 owner 规则过滤,而不是按 `page_indices` 的连续范围切。
- Transfer chunk 要么在发送前按 `attn_cp_kv_chunk_size` / owner 边界拆分,保证
  `index_slice` 仍能表达连续映射;要么扩展 transfer message,显式携带 filtered
  positions/mask。不能把跨 owner chunk 的非连续过滤结果强行塞进单个连续 slice。
- Batched decode 下每个 request 独立过滤 owner/DUMMY;某些 CP rank 对某个 request
  可能为空 transfer,这必须是合法成功状态。

**Attention / cuda graph 关系**:
- PD transfer 完成后,decode 侧 attention 路径应与非 PD sharded-KV decode 相同:
  本 rank 持久化 owner KV,attention 内部临时 Q allgather + segment FA +
  `cp_lse_ag_out_rs`。
- PD 分离不应该改变 sharded-KV 的省显存语义。临时 dense/gather buffer 可以存在,
  但 persistent KV cache 仍然只能保存本 CP owner 的 KV。
- Cuda graph、kv mirror、over encoding、chunked prefill、overlap schedule 在 PD 下仍是
  最终必须支持项;开发阶段可以先用 eager 或单项关闭定位问题,但不能作为上线限制。

**验收建议**:
- 最小 E2E:FA3 + `TP4,CP2` prefill server + `TP4,CP2` decode server,对比非 PD
  `TP4,CP2` 或 `TP4` baseline,tokens 一致,logprobs 在已知容差内。
- 必测场景:single request、batched decode、chunked prefill、`attn_cp_kv_chunk_size=128`
  的短 prompt owner 覆盖、WeLM kv mirror on、over encoding on、cuda graph on、overlap on。

---

## 4. 与 V1 / vLLM DCP 对比

| 维度 | V1(已落地)| **本方案** | vLLM DCP |
|---|---|---|---|
| Q heads 切分 | `H/attn_tp` | **`H/TP`(纯 TP shard)** | `H/(TP×DCP)` |
| Q sequence 切分 | `S/cp` | **不切(全 S)** | 不切(全 S) |
| Hidden cp_split | ✓ | **✗(全 S)** | ✗ |
| KV heads 切分 | `K/attn_tp` | `K/attn_tp` | `K/TP` 或复制 |
| KV sequence 切分 | `S/cp`(in-flight)| **`S/cp`(pool 永久切片)** | `S/cp`(pool 切片) |
| KV pool 物理 | replicated 写满 | **sharded** | sharded |
| Prefill / Decode 路径 | 不同(prefill ring,decode 无 CP)| **同一条 path** | 同一条 path |
| Q allgather | 无 | **head 维(sharded_kv_cp_group)** | head 维(DCP group) |
| Output merge | merge_state_v2 本地 | **`cp_lse_ag_out_rs`(LSE allgather + heads RS)** | `cp_lse_ag_out_rs` |
| o_proj 通信域 | `attn_tp_group` all_reduce | **全 `tp_group` all_reduce(纯 TP)** | 类似 |
| Rank 拓扑 | `(dp, cp, attn_tp)`(SGLang 默认)| **sharded-KV 专用 logical `(dp, attn_tp, cp)`** | DCP 独立 group |
| KV 内存 ÷ cp | ✗ | **✓** | ✓ |
| MLP/MoE 算力 ÷ cp | ✓ | ✗ | ✗ |
| Prefill 算力 ÷ cp | ✓(token cp 切)| **✓(KV cp 切)** | ✓ |
| 实现复杂度 | 已落地 ~530 行 | **~1500-1900 行**(logical topology + 切片 pool + runtime 特性兼容)| 重 |

**本方案与 vLLM DCP 的核心差异**:
- vLLM 在 Q proj weight 上做 `TP × DCP` 双层切,本方案直接用 `tp_size` 切,DCP 维只在 attention 内部存在,Q proj weight 永远是纯 TP shard。
- 这让 weight loader、presharded checkpoint、量化、o_proj 都能直接复用 SGLang 标准 TP 实现,不需要为 sharded-KV mode 特殊适配。

---

## 5. 改动清单

### 5.1 必改

| 模块 | 改动 | 估计代码量 |
|---|---|---|
| `parallel_state.py` | 新建 sharded-KV 专用 CP group:`tp_rank = sharded_attn_tp_rank * cp + sharded_cp_rank`;只构造 attention 里实际使用的 `sharded_kv_cp_group`;现有 default `_ATTN_TP/_ATTN_CP` 不改 | ~120 行 |
| 模型 attention 层(`welmv4.py` 首发)| QKV proj 使用 global `QKVParallelLinear(tp_size=tp_size)`;强 assert `attn_tp_size == num_kv_heads`;attention forward 入口加 sharded_kv_cp_group 内 Q/headwise 参数 allgather + 出口 cp_lse_ag_out_rs;保留 WeLM kv mirror / over encoding row contraction 语义 | ~260 行 |
| `o_proj` | 保持现有 `RowParallelLinear(tp_size=tp_size)` + 全 tp_group all_reduce,不引入 sharded_kv_attn_tp_group 特殊路径 | 0 行(直接复用)|
| `cp_utils.py` | 1) 新加 `cp_q_allgather_head` 原语(sharded_kv_cp_group 内 head 维 allgather);2) 新加 `cp_lse_ag_out_rs`(LSE allgather + 跨 cp merge + heads reduce_scatter);3) 删 V1 的 ring-pass-kv 相关 helper | +250 行,-200 行 |
| 新建 `mem_cache/cp_sharded_allocator.py` + attention metadata | 切片 slot 分配器、per-rank compact page table、local cache_seqlens、local_kv_positions、owner/DUMMY slot 映射与 release/rollback/overlap 生命周期过滤 | ~550-750 行 |
| `attention/flashattention_backend.py` | 新增 sharded-KV path:Q allgather → per-owner-chunk segment loop FA + local LSE merge → cp_lse_ag_out_rs。prefill 和 decode 走同一个函数,差异只在写池规则;支持 chunked prefill、SWA、sink、kv mirror contraction、over encoding positions | ~420 行 |
| 模型层入口/出口 | 入口 cp_split / 出口 cp_gather 接线**移除**(本方案 hidden 不切);首版仅 WeLM fused QKV 约束内落地 | -50 行(净减少)|
| `cuda_graph_runner.py` / `piecewise_cuda_graph_runner.py` | Q allgather + LSE allgather + reduce_scatter + compact metadata buffer 在 graph 内 capture;kv mirror static buffer 继续兼容 | ~300 行(Phase 3)|
| overlap schedule / `ForwardBatch` metadata | sharded-KV metadata 支持 ping-pong batch 生命周期,TBO 非 mirror contraction 场景继续可用 | ~120 行 |
| PD disaggregation KV transfer(后续) | 同拓扑 P/D sharded-KV rank mapping;bootstrap metadata 校验;decode sharded 预分配;按 logical owner 过滤 transfer;DUMMY slot 过滤;空 transfer success;Mooncake/NIXL/Mori backend 逐一适配 | ~250-400 行 |
| `server_args.py` | `--attn-cp-size C`、`--attn-cp-mode sharded-kv`、`--attn-cp-kv-chunk-size N` 与启动约束;内部派生 prefill sharded-KV 和 decode sharded-KV residency;不自动禁用 kv mirror / over encoding / cuda graph / overlap schedule | ~40 行 |
| **总计** | | **核心 Phase 1-3 ~1700 行,净增 ~1500;PD transfer 另计 ~250-400 行** |

### 5.2 不改 / 移除

- 模型层入口/出口的 `cp_split_and_rebuild_data` / `cp_all_gather_rerange_output` 调用 — **移除**(本方案 hidden 不切 sequence)
- DP attention(首版禁用;后续兼容时复核 sharded-KV logical group 与 DP group 组合)
- Tokenizer / detokenizer 协议

### 5.3 兼容性

- **Fallback**:`--attn-cp-mode none` 为默认路径,行为与普通 TP 一致。sharded-KV 稳定前不替换默认路径。
- **logical topology 影响范围**:sharded-KV mode 用专用 group/getter,其他 mode(纯 TP、in-seq-split CP、V1 ring-pass-kv 等)沿用 SGLang 默认 topology。两套 group 构造逻辑可以共存,但代码要清晰隔离。
- **必须保持开启兼容**:WeLM kv mirror、over encoding、chunked prefill、cuda graph /
  piecewise cuda graph、overlap schedule 均为 GA 必须支持项。开发阶段可以临时用 eager
  或关闭单项做问题定位,但不能作为最终上线要求。

---

## 6. Roadmap(3 阶段,~6-8 周)

每阶段独立可发布、可回滚。

### Phase 1:基础设施 + sharded-KV logical topology(2 周)

**目标**:把 rank/head mapping 校验、sharded pool 元数据、logical topology 搭好;
attention 还走 replicated KV(shadow path),仅校验 group 构造 + slot 分配。

| 子任务 | 验证 |
|---|---|
| 1.1 sharded-KV 专用 logical rank getter + `sharded_kv_cp_group` / `sharded_kv_attn_tp_group` 构造 | 单元测试覆盖 group ranks 计算;V1 / 纯 TP 路径不受影响 |
| 1.2 GQA / head mapping assert(`H % TP == 0`、`attn_tp_size == K`、sharded_kv_cp_group Q 拼接覆盖一个 GQA group)| 不满足时拒绝启动,error message 清晰 |
| 1.3 `CPShardedKVAllocator` + compact page table + DUMMY slot + `local_kv_positions` | 单元测试 + 内存占用对比(每 rank ÷ cp)|
| 1.4 `cp_q_allgather_head` + `cp_lse_ag_out_rs` 原语 | 单元测试(数学对拍 torch reference)|
| 1.5 `--attn-cp-kv-chunk-size N` server-start 固定 | 默认 1024;测试可设 128/256;启动日志打印 C;request 期间不变 |

**交付**:基础设施落地;default 路径业务行为不变,sharded-KV 路径仅在新 flag 下启用。

### Phase 2:Attention 路径打通(2 周)

**目标**:实现 sharded-KV attention(prefill + decode 同一条 path),验证 correctness。

| 子任务 | 验证 |
|---|---|
| 2.1 sharded-KV attention forward:Q allgather → per-chunk segment FA → cp_lse_ag_out_rs | 单测对拍 replicated 路径 |
| 2.2 Per-owner-chunk segment causal mask(拆 FA + LSE merge 的 portable 路径) | mask 单测 + 边界 case(chunk 边界、SWA window 边界)|
| 2.3 KV pool 切片写入(只 owner 写,其他 DUMMY)| 内存占用 ÷ cp |
| 2.4 Chunked prefill:Q allgather + 本 rank pool owned KV segment loop(prefix + self 统一)| 长 prompt 对拍 baseline |
| 2.5 Decode 同 path(seq_len=B,KV 来源 pool)| Decode 数学对拍 |
| 2.6 WeLM SWA(显式按全局 position mask)| WeLM 模型对拍 |
| 2.7 WeLM attention sink(只由 sharded_cp_rank=0 注入)| Sink 注入正确性单测 |
| 2.8 Batched decode(每 request 独立 `local_page_table`,空 KV partial 处理)| Batched decode 不崩 |
| 2.9 WeLM kv mirror opt:Q row contraction + full KV 写池 + o_proj scatter/rebuild | `enable_welm_kv_mirror_opt` on/off 对拍;mirror layer / imitated layer 覆盖 |
| 2.10 Over encoding:`oe_context` / `scale_seq_factor` 下 owner、positions、DUMMY 过滤一致 | over-encoding prompt 对拍;chunked prefill + decode 不崩 |

**交付**:WeLM v4 长 prompt prefill / 长 context decode 对拍 baseline `mean|Δ| < 0.01`,
KV 内存实测 ÷ cp;chunked prefill、kv mirror、over encoding 在 eager path 下全部跑通。
其他模型只有在满足 `attn_tp_size == num_kv_heads` 时纳入首版;否则等 Phase 4 定制 KV projection。

### Phase 3:Cuda graph + overlap schedule + 性能 + 录基线(2-3 周)

**目标**:进入 cuda graph / piecewise cuda graph,恢复 overlap schedule,达到性能目标。

| 子任务 | 验证 |
|---|---|
| 3.1 Decode cuda graph:Q allgather + LSE allgather + reduce_scatter + static metadata buffer capture | Decode TPOT 稳定;kv mirror buffer 兼容 |
| 3.2 Piecewise cuda graph:prefill/chunked prefill capture bucket 固定 segment launch 结构 | 8k/64k chunked prefill graph replay |
| 3.3 Overlap schedule / TBO:sharded metadata ping-pong 生命周期 + collective stream 顺序 | overlap on/off 对拍;无跨 batch metadata overwrite |
| 3.4 Segment FA call 优化(FA3 mask_mod / block mask 合并多段) | 单层 FA latency 对比 |
| 3.5 Bench 录基线到 `docs/ring-attn/baselines.md`:WeLM v4 8k/64k/256k,mirror/over-encoding/overlap/graph 全场景 | 完整数据入库 |

**交付**:WeLM v4 256k 长上下文 serving 跑通,且不要求关闭 kv mirror、over encoding、
chunked prefill、cuda graph 或 overlap schedule;V1 评估 retire。

### Phase 4(后置):PD 分离与兼容性扩展

- PD disaggregation KV transfer 适配(先同拓扑 P/D:相同 TP/CP/PP/page/chunk/backend)
- 异构 P/D 拓扑支持(可选):prefill/decode CP/TP/PP 不一致时的 owner remap / scatter
- Radix cache 跨 cp prefix hit
- DP attention 兼容
- FP8/FP4 KV cache 兼容
- CPU KV offload

---

## 7. 风险与缓解

### 7.1 高风险

1. **logical topology 误用风险**:sharded-KV 需要 `(dp, attn_tp, cp)` 视角,但 SGLang default attention CP API 仍是 `(dp, cp, attn_tp)` 视角。
   - 缓解:Phase 1 新增 sharded-KV 专用 getter/group,**不复用 default `get_attention_tp_rank()` 做 projection 或 sharded-KV rank 判断**;default 路径全量回归(V1 / in-seq-split CP / 纯 TP / DP attention)。

2. **Cuda graph 与 sharded_kv_cp_group collective 兼容**:NCCL allgather / reduce_scatter 在 graph 内 capture 有历史坑(stream 同步、内存别名)
   - 缓解:Phase 3 单独立项;先 eager 跑通正确性,但 GA 必须 graph on 通过。若 NCCL
     collective capture 不稳定,优先切到预分配 workspace + graph-safe group API,而不是要求
     用户关闭 cuda graph。

3. **GQA 切分对齐**:Q 按 TP 切 + KV 按 attn_tp 切要求 GQA group 边界与 attn_tp 边界对齐
   - 缓解:启动 assert(详见 §3.1 首版约束),首版 fused QKV 强制 `attn_tp_size == num_kv_heads`,不对齐拒绝启动

4. **Per-chunk segment mask 正确性**:每 rank 持非连续 owner chunks,FA 调用要按 chunk 拆;SWA 不能直接用 FA `window_size`,必须按全局 position 显式 mask
   - 缓解:Phase 2 portable 路径(拆 FA + LSE merge),mask 单测覆盖 chunk 边界、SWA 边界、attention sink

5. **DUMMY slot 泄漏 / 误释放**:非 owner token 在 `req_to_token_pool` 中用 slot 0 占位,释放或 metadata compact 漏过滤会导致重复 free、读到空 KV 或 capacity 统计错误。
   - 缓解:slot 0 永久保留且不进 freelist;allocator / release / compact / write path 单测覆盖 DUMMY;首版禁用 radix cache、CPU KV offload。

6. **WeLM kv mirror 与 sharded KV 写池语义交叉**:mirror contraction 会把 attention Q rows
   收缩到 last-token rows,但 KV cache 仍必须写完整 logical K/V。若把 Q contraction 误用于
   KV 写池,会导致 prefix/current chunk KV 丢失。
   - 缓解:把 "KV fill rows" 与 "attention Q rows" 在 metadata 中分开;mirror 相关
     `custom_last_index` / scatter/rebuild 单测覆盖 prefill、chunked prefill、decode、PP proxy
     和 cuda graph。

7. **Over encoding position 语义混用**:`scale_seq_factor` 下 logical positions、pool slot、
   DUMMY entry、segment mask 如果有一处按原始 token index 计算,会出现 owner 错位或 causal
   mask 错。
   - 缓解:owner / local_kv_positions / segment mask 统一只吃 expanded logical position;
     over-encoding case 加到 allocator、metadata 和端到端对拍。

### 7.2 中风险

8. **Q allgather + reduce_scatter 在 cuda graph 下的内存别名**
   - 缓解:Phase 3 评估 vLLM 的 `CPTritonContext` 是否直接 port

9. **`mask_mod` 在 sgl-kernel FA3 是否可用**
   - 缓解:Phase 2 用 portable "拆 FA + LSE merge",Phase 3 优化时再尝试 mask_mod

10. **Page table compact 在 batched decode 下的开销**
   - 缓解:每 request `local_page_table` 在 metadata 生成阶段就准备好,attention 路径只读不算

11. **`cp_kv_chunk_size` 负载倾斜**:固定大 chunk 对短请求和短 prefill chunk 不友好,可能长期只使用低 sharded_cp_rank 的 KV。
   - 缓解:默认 1024;集成/正确性测试显式传 128 或 256 覆盖多 CP owner;性能基线覆盖
     512/1024/2048。启动日志打印 C;request 期间禁止动态变化。

12. **Overlap schedule metadata 生命周期**:CPU scheduler 可能在 GPU forward 未完成时复用或改写
   sharded-KV metadata,导致 graph/overlap 下读到下一批的 page table 或 positions。
   - 缓解:sharded metadata 纳入现有 overlap ping-pong buffer;forward stream 上 record event,
     释放/复用前等待;overlap on/off 对拍作为 Phase 3 gate。

13. **PD disaggregation transfer 语义误用**:现有 PD 代码倾向于 decode CP=1、cp0 发送
   或按连续 page range 切 KV;这些假设会把 sharded-KV 退化成 replicated KV,或者导致
   DUMMY slot 被错误传输。
   - 缓解:sharded-KV PD 单独分支,首版强制 P/D 同拓扑;bootstrap 校验
     `attn_cp_mode` / `attn_cp_kv_chunk_size` / owner policy;transfer 按 logical position
     owner 过滤,并为 empty transfer 明确返回 success。

### 7.3 兼容边界与后置项

- **必须支持**:`enable_welm_kv_mirror_opt`、`enable_over_encoding`、chunked prefill、
  cuda graph / piecewise cuda graph、overlap schedule。
- **后续必须支持但可分阶段**:PD disaggregation。第一阶段只支持 P/D 同拓扑 sharded-KV,
  异构拓扑、跨 CP radix cache、CPU offload 等后置。
- **可后置或拒绝**:Spec decode、CPU KV offload、非 FA3 backend、DP attention、
  FP8/FP4 KV cache、radix cache 跨 CP prefix hit。

---

## 8. 决策点(需 review 确认)

1. **拓扑怎么搞**:
   - 决策:不全局翻转 SGLang default topology。sharded-KV mode 新增专用 logical
     mapping:`sharded_attn_tp_rank = tp_rank // cp_size`,
     `sharded_cp_rank = tp_rank % cp_size`;projection/o_proj/MLP 继续用 global
     TP rank/group。

2. **Q proj 用 fused `QKVParallelLinear(tp=tp_size)`** 还是拆 `q_proj` + `kv_proj`?
   - 决策:首版 fused,但强限制 `attn_tp_size == num_kv_heads`。不满足时拒绝启动,
     后续再做定制 KV projection。

3. **`cp_kv_chunk_size` 默认值**:
   - 决策:显式暴露 `--attn-cp-kv-chunk-size N`,默认 1024。集成/正确性测试可传
     128/256,保证短 prompt 也跨 CP owner 边界;性能基线覆盖 512/1024/2048。禁止
     request 级动态变化。

4. **WeLM kv-mirror-opt 是否可以首版禁用?**
   - 决策:不可以。`enable_welm_kv_mirror_opt` 是 sharded-KV GA 必须支持项;开发阶段
     可以用 mirror-off 定位问题,但最终对拍和性能基线必须覆盖 mirror-on。

5. **Cuda graph 上线策略**:
   - 决策:eager 可以作为 Phase 2 correctness 里程碑;Phase 3/GA 必须支持 decode cuda
     graph 和 piecewise prefill graph。不能要求用户通过 `--disable-cuda-graph` 使用本方案。

6. **Overlap schedule / over encoding 是否可后置?**
   - 决策:不可后置。`enable_over_encoding`、chunked prefill、overlap schedule 是 WeLM
     serving 必需路径,必须进入 Phase 2/3 验收。

7. **PD 分离首版范围**:
   - 决策:必须支持 PD,但首版限定同拓扑 P/D,且 prefill/decode 都使用 FA3 和
     sharded-KV AttnCP。KV transfer 保持 owner-sharded 语义,不能让 decode CP=1
     接全量 KV。异构 P/D 拓扑后置。

8. **V1 retire 时机**:本方案 Phase 2/3 跑通后,V1 是否同步删除?
   - 推荐:V1 deprecated 1 个版本,稳定后再删。

9. **logical topology 影响 sgl 其他用户**:是否需要为现有 in-seq-split CP / DP attention 用户做迁移指引?
   - 决策:default topology/API 不动,只 sharded-KV mode 使用新 getter/group;现有用户无感。

---

## 9. 附录

### 9.1 参考实现

- vLLM `_forward_with_dcp`:`vllm/v1/attention/backends/flash_attn.py:930` — Q allgather + cp_lse_ag 标准范式
- vLLM `cp_lse_ag_out_rs`:`vllm/v1/attention/ops/common.py:212` — 跨 rank LSE 合并 + heads reduce_scatter triton kernel
- 本仓库 sgl-kernel `merge_state_v2`:`sgl-kernel/python/sgl_kernel/attention.py:6` — 本地 LSE 合并

### 9.2 关联文档

- 长期 8 阶段计划(原):[plan.md](./plan.md)
- V1 落地报告(旧实验):`/home/fhkong/wxwork/80a_attncp_dev/welm_attncp_ring_v1_status.md`
- KV mirror 设计梳理:`/home/fhkong/wxwork/80a_attncp_dev/kv_mirror_design.md`

### 9.3 术语

| 词 | 含义 |
|---|---|
| `tp` / `tp_size` | tensor parallel 总规模(Q proj / o_proj / MLP / MoE 都用这个)|
| `attn_tp` / `attn_tp_size` | `tp / cp`,attention 里 KV head 切分粒度 |
| `attn_cp` / `attn_cp_size` | KV sequence 切分粒度,本方案核心 |
| `sharded_attn_tp_rank` | sharded-KV 专用 logical rank:`tp_rank // attn_cp_size` |
| `sharded_cp_rank` | sharded-KV 专用 logical rank:`tp_rank % attn_cp_size` |
| GQA | Grouped Query Attention,`H >> K`,group 大小 = `H/K` |
| LSE | log-sum-exp,FA 内 softmax 归一化项 |
| sharded_kv_cp_group | 同 `sharded_attn_tp_rank`、不同 `sharded_cp_rank` 的 ranks;Q allgather + cp_lse_ag_out_rs 的通信域 |
| sharded_kv_attn_tp_group | 概念上是同 `sharded_cp_rank`、不同 `sharded_attn_tp_rank` 的 ranks;当前实现不构造该 group,`o_proj` 直接走完整 `tp_group` |
| tp_group | 全部 tp_size 个 rank;`o_proj` all_reduce / MLP / MoE 的通信域 |

---

**评审人请关注**:§3.1 拓扑表 + logical topology 决策、§3.3 attention 数据流、§3.4 mask schedule、§3.5 headwise 参数、§3.7 DUMMY slot、§3.8 runtime 必须支持项、§3.9 PD 分离限制、§5 改动清单、§8 决策点。
