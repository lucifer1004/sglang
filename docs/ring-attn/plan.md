# Sharded-KV Context Parallel(Ring Attention 风格)执行计划

> **本文档是唯一事实源**,合并了原 ring prefill 计划与 sharded-KV 深度设计。
> 面向长期执行者(AI 或工程师):每个交付点(DP)有明确边界、改动清单与
> 可执行的验证条件。按 DP 顺序执行,**每个 DP 一个 PR、独立可合入、独立可回滚**。

---

## 0. 执行须知(先读)

- **工作分支**:以 `perf/welm-v4-optimization` 为基线拉特性分支(不要从
  `main_v056` 出发)。注意两点:
  (a) 该分支的 CP 路径已损坏(见 §2.3),因此 **DP0 是硬前置,必须最先完成**;
  (b) 该分支的 `flashattention_backend.py` 携带大量 WeLM v4 定制路径
  (cascade、KV mirror、custom-last-q、MTP SWA compact 等),本计划所有触及
  该文件的改动都必须保证这些路径不回退(回归手段见 §7)。
- **目标模型双轨**:正确性主力仍用 Qwen3-30B-A3B(GQA 标准形态、有 CI 先例);
  但本分支的业务目标是 WeLM v4(SWA/全注意混合架构),因此 DP7b(混合 SWA
  驻留)优先级提升,应在 DP5 完成后优先排期。
- **GPU 环境**:本机跑 GPU 任务须通过 `gpu-lease run` 包裹(见 gpu-lease skill)。
- **默认路径零影响**:所有新功能挂在 `--cp-kv-residency sharded`(默认
  `replicated`)之后;flag 关闭时任何行为/性能不得变化。
- **主力验证模型**:`Qwen/Qwen3-30B-A3B-FP8`(GQA:32 Q heads / 4 KV heads,
  48 层,CI 有先例);最小正确性配置 2 卡(`--tp-size 2 --attn-cp-size 2`,
  即 attn_tp=1 × cp=2),完整配置 4 卡(attn_tp=2 × cp=2)。
- **代码规范**:每 PR 过 `SKIP=no-commit-to-branch pre-commit run --all-files`;
  新测试按 `test/README.md` 与 `register_cuda_ci` 约定注册。

---

## 1. 背景与目标

### 1.1 问题

GQA 模型(如 Qwen3:4 个 KV heads)长上下文 serving 时:

1. **纯 TP**:`attn_tp_size > num_kv_heads` 后 KV head 被复制
   (`linear.py::QKVParallelLinear.num_kv_head_replicas`,
   `model_config.py::get_num_kv_heads = max(1, total // tp)`),
   集群 KV 冗余系数 = `tp / num_kv_heads`,显存容量不随卡数扩展。
2. **现有 allgather CP**(`--enable-prefill-context-parallel`,上游 #18233):
   prefill 按 zigzag 切 token,但每层把 K/V allgather 后**全量写入每个 CP
   rank 的池**——冗余系数 = `cp_size`,与纯 TP 一样没省内存;
   allgather 阻塞在关键路径;decode 完全不做 CP(CP 组内重复计算)。

**结论:现有 CP 无法解决 KV 容量问题,需要存储级的序列分片。**

### 1.2 目标(最终态)

1. **KV 分片驻留**:attn_tp 切 head × attn_cp 切序列,每 rank 只存
   `1/cp_size` 的 KV page,全集群 (head, token) 不相交,冗余系数 1。
2. **Prefill = striped ring attention**:P2P 环形流水,通信藏进计算,
   KV 只写本地分片。
3. **Decode 原生 CP**:分布式 flash-decoding——各 rank 对本地分片算
   partial `(out, lse)`,CP 组内 allgather KB 级结果后合并;
   长上下文 decode 的每 rank KV 读带宽同样 ↓ cp 倍。
4. 调度器/模型代码改动最小;CUDA graph 可用;支持 chunked prefill 与
   任意 decode batch。

**业界先例**(路线可行性背书):vLLM DCP(interleaved KV 分片 + decode
partial-merge)、NVIDIA Helix Parallelism、本仓库 NSA `round-robin-split`
CP 模式(token 级 round-robin 的现成工具链)。

### 1.3 方案选型(为什么是 ring + stripe)

| 方案 | 通信 | KV 内存/rank | 重叠 | 致命约束 |
|---|---|---|---|---|
| allgather CP(现状) | 每层阻塞 allgather | 全量复制 | 无 | 内存不省 |
| Ulysses(all-to-all heads) | 每层 2×a2a | 1/cp | 部分 | 并行度 ≤ num_kv_heads(GQA 仅 4) |
| **ring + stripe(本计划)** | P2P 流水 | **1/cp** | 完全 | 无 |

GQA 是 ring 的最佳场景:环上只传 KV(∝ h_kv,小),计算 ∝ h_q(大),
通信几乎可完全隐藏。

---

## 2. 现状:代码地图与已知问题

### 2.1 可复用的基础设施(全部已存在)

| 组件 | 位置 | 说明 |
|---|---|---|
| CP 进程组 | `distributed/parallel_state.py`、`layers/dp_attention.py`(`get_attention_cp_group/rank/size`;`attn_tp_size = tp_size // dp_size // attn_cp_size`) | `GroupCoordinator.send/recv` 与 pynccl P2P 已具备 |
| Server args | `server_args.py`:`attn_cp_size`、`enable_prefill_context_parallel`、`prefill_cp_mode`、`_handle_context_parallelism()` | 开关与校验骨架 |
| (out,lse) 合并核 | `layers/attention/merge_state.py`(`merge_state_v2` + triton fallback) | ring/decode 合并的核心算子;FA backend cascade 路径(`merge_state_v2_wrapper`、`lse.T.contiguous()`)有现成用法 |
| round-robin 切分工具 | `layers/attention/nsa/utils.py::nsa_cp_round_robin_split_data`;`layers/utils/cp_utils.py::cp_all_gather_rerange_output` 的 round-robin 分支 | token 级 round-robin 切分/聚合,DP3 推广为 stripe 级 |
| 模型侧钩子 | `models/qwen3_moe.py`(metadata 准备)、`models/qwen2_moe.py`(输入切分 / 输出聚合) | DP3 等量替换为 stripe 版 |
| 调度门控 | `cp_utils.py::can_cp_split`、`forward_batch_info.py::is_context_parallel_extend`、`schedule_policy.py::PrefillAdder`(CP 时 bs=1)、`schedule_batch.py`(decode 前清 CP metadata) | 复用 |
| 分配器接口 | `mem_cache/allocator.py`(`alloc / alloc_extend / alloc_decode / free / available_size`) | DP2 wrapper 的注入点 |
| SWA 层分组 | `mem_cache/swa_memory_pool.py::SWAKVPool`(`swa/full_attention_layer_ids`) | DP7b 混合驻留的天然挂点 |
| FA3 能力 | `flash_attn_with_kvcache` / `flash_attn_varlen_func` 均支持 `return_softmax_lse=True`、per-element `cache_seqlens`、paged KV | partial attention 的全部所需 |
| MoE 协同 | `layers/communicator.py`(`MOE_FULL`、moe_cp allgather/scatter,带 padding) | CP 与 MoE DP/EP 的 token 重组,复用 |

### 2.2 现有 allgather CP 的行为(对照基线)

每 rank 全量 embedding → zigzag 切 hidden_states(rank r 取块 `{r, 2c-1-r}`)
→ 每层:本地算 K/V → CP 组 allgather → **全量写本 rank 池** → 本地 q 分
prev/next 两段、以 `kv_len_prev/next` 为因果边界调 FA → 模型尾部 allgather
hidden_states 出 logits。decode:metadata 已清,各 rank 持全量 KV 各自冗余计算。
启动参数见 `test/registered/4-gpu-models/test_qwen3_30b.py`。

### 2.3 已知损坏(必须知晓)

在 `perf/welm-v4-optimization` 分支上,`layers/attention/flashattention_backend.py`
的 GQA CP 路径已被 WeLM squash(commit `68605a84c0`)破坏:

1. 调用 `cp_attn_forward_extend`(forward_extend 的 CP 分支)但**文件中无对应
   import** → 启用 CP 即 `NameError`;
2. 原 `is_cp_mode` 下的 `cp_allgather_and_save_kv_cache(...)` 调用丢失 →
   即使补 import,池中无人写全量 KV,attention 读到未初始化槽位。

原始正确实现参考 commit `bb737d7a82`(#18233)的 diff。

---

## 3. 目标架构:核心设计决策(D1–D8)

执行时不得擅自更改以下决策;若实现中发现决策不可行,停下并升级讨论。

**D1|所有权函数 = stripe round-robin(弃用 zigzag 作存储布局)**

```
stripe_size = cp_stripe_size              # server arg;page_size 整数倍;默认 max(page_size, 256)
block(pos)  = pos // stripe_size          # 每请求独立按位置计
owner(pos)  = block(pos) % cp_size
# rank r 在序列长 S 时持有的 token 数(O(1)):
local_len(r, S) = (S // (stripe*c)) * stripe + clamp(S % (stripe*c) - r*stripe, 0, stripe)
# 全局位置 → owner rank 上的本地序号:
local_idx(pos) = (block(pos) // c) * stripe + pos % stripe
```

理由:zigzag 边界依赖单次 prefill 总长,序列增长后不稳定,不能当存储布局;
stripe round-robin 增量稳定、任意前缀长度下负载均衡(striped attention),
且对齐 vLLM DCP 的 interleaved 设计。约束 `stripe_size % page_size == 0`
保证 pool page 不跨 owner。

**D2|铁律:对称调度、非对称存储**

所有调度决策(admission/retract/eviction/chunk 切分)只依赖全局逻辑量
(token 数、global position),禁止依赖任何本地物理量(本地空闲 slot、
本地分片长度)。这是 CP 组内 rank 不分叉(集体通信不挂死)的唯一保障。
debug 模式提供跨 rank 决策 hash 校验。

**D3|逻辑/物理两级分配器**

新增 `CPShardedAllocator` 包装现有 allocator 接口:逻辑层按全局 token 记账
(逻辑容量 = 物理容量 × c × (1 - margin),margin 覆盖 stripe 余数偏差,
偏差上界 = running_reqs × stripe_size),所有 rank 决策必然一致;
物理层只为 `owner == self` 的位置分配真实 slot,其余返回 `DUMMY_SLOT`。

**D4|req_to_token 保持全局位置索引 + DUMMY 哨兵**

owned 位置存本地 slot,非 owned 位置存 DUMMY_SLOT(池保留 slot 0 作 scratch,
写入无害、永不被读;debug 模式填 NaN + assert)。scheduler/retract/radix 等
按位置切片的代码不感知 CP。attention backend 在 `init_forward_metadata` 用 D1
公式向量化压缩出本地紧凑 page_table 与 `local_cache_seqlens`。

**D5|Prefill = striped ring**

Q 切分 = KV 所有权切分(同一 stripe round-robin),K/V 直写本地池
(与非 CP 路径相同的 `set_kv_buffer`,无跨 rank 写)。ring 共 c 步,
第 s 步用来自 rank `(r-s) mod c` 的分片(staging 双缓冲、独立 comm stream、
event 同步、奇偶配对 send/recv 防死锁):

- s=0(本地):每 q-stripe 作为 varlen batch 一个元素,
  `cache_seqlens[i] = 本地可见长度`,`causal=True`(FA3 右对齐语义匹配对角);
- s>0(来访):**无对角块**(块号 mod c 不同),每对 (q-stripe, kv-stripe)
  全可见或全跳过,`causal=False`、`cache_seqlens[i] = 来访分片中块号 < i 的
  token 数`;page_table 指向 staging(视作小分页池)。

每步每层 1 次 kernel;partial 用 `merge_state_v2` 累加(FA 返回 lse 为
`[heads, tokens]`,merge 前转置,照抄 cascade 写法)。

chunked prefill / 前缀复用统一处理:每步发送
`[本地池 gather 出的本 rank 前缀分片 ‖ 本 chunk 新算分片]`。

**D6|Decode = 分布式 flash-decoding,封装在 backend 内,模型层零改动**

每层:全 batch k/v 照常算好,`out_cache_loc` 中非 owned 槽位即 DUMMY
(由 D3 产生),一次 `set_kv_buffer` 零分支完成 owner-only 写入;
本地紧凑页表上 `flash_attn_with_kvcache(..., return_softmax_lse=True)`
得 partial → CP 组 allgather `(out, lse)`(bs=64、h=16、d=128 时每 rank
~256 KB)→ `merge_state_v2` 链(c-1 次,顺序固定保证决定论)→ 返回,
o_proj 及之后照旧。数学上与池内 split-KV(`num_splits>1`)同一套,无精度损失。

**D7|混合 SWA 驻留(为 WeLM v4 类混合模型预留,DP7b)**

全注意层 sharded;SWA 层(窗口小)保持复制驻留、decode 本地完成不参与
merge。按 `layer_id ∈ full_attention_layer_ids` 分支。

**D8|短请求策略(v2,DP7c)**

decode merge 有每层固定通信延迟,短上下文(≲8k)亏(见附录 A.3)。
v2 按请求长度混合驻留(`< L_min` 复制、不参与 merge)。v1 不做,
但所有权按请求独立计算,结构上已兼容。

---

## 4. 全局不变式与 v1 支持矩阵

**不变式(每个 DP 的验证都必须包含)**

1. `--cp-kv-residency replicated`(默认)时,行为与改动前逐 bit 一致
   (现有测试不回退);
2. D2 铁律:集体通信调用点必须所有 rank 无条件到达(禁止 data-dependent
   分支包裹 collective);
3. lse 用 fp32;merge 顺序固定;
4. DUMMY slot 只写不读;
5. 每 DP 一个 PR,可独立 revert。

**v1 支持矩阵(超界即 assert 拒绝,禁止隐式降级)**

| 维度 | v1 支持 | 备注 |
|---|---|---|
| Attention backend | FlashAttention(fa_impl_ver==3)、CUDA | 其余报错 |
| 模型 | Qwen2-MoE / Qwen3-MoE(GQA) | 钩子等量替换 |
| prefill batch | 1(沿用现有 CP 限制) | multi-batch 列入后续 |
| decode batch | 任意 | 本计划核心能力 |
| kv_cache_dtype | auto(bf16) | FP8 列入 DP6 之后 |
| radix cache | 关闭(sharded 时强制) | DP7a 解禁 |
| spec decode / MTP / TBO / PD 分离 / 跨机 CP | 不支持 | 互斥 assert |
| CUDA graph | DP5 起支持 decode graph | 之前用 `--disable-cuda-graph` |

---

## 5. 交付点总览

```
DP0 ──(硬前置,必须最先:本分支 CP 已损坏)── 修复 legacy CP + 建立基线
DP1 → DP2 → DP3 → DP4 → DP5 → DP6        (主线,严格顺序)
                          └→ DP7b(优先,WeLM v4)/ DP7a / DP7c
                                          (扩展轨道,DP5 后并行)
```

| DP | 一句话 | GPU 需求 | 预估规模 |
|---|---|---|---|
| DP0 | 修复损坏的 allgather CP,跑通基线 | 4 卡 | ~10 行修复 + 基线报告 |
| DP1 | stripe 所有权纯函数库 + 穷举单测 | 无 | ~200 行 |
| DP2 | 两级分配器 + DUMMY + server args(默认关) | 无 | ~400 行 |
| DP3 | striped ring prefill + sharded 写入(eager) | 2–4 卡 | ~600 行 |
| DP4 | decode 分布式 flash-decoding(eager) | 2–4 卡 | ~400 行 |
| DP5 | decode CUDA graph + chunked prefill + 长稳 | 4 卡 | ~300 行 |
| DP6 | 性能:one-shot merge、stripe 调参、重叠率与达标 | 4–8 卡 | ~300 行 |
| DP7 | a: CP-aware radix;b: 混合 SWA;c: 短请求混合驻留 | 4 卡 | 各独立 |

---

## 6. 交付点详细定义

### DP0|修复 legacy CP 路径,建立对照基线(硬前置)

**目标**:让现有 allgather CP 在 `perf/welm-v4-optimization` 上可运行,
产出正确性与性能基线。后续所有 DP 的对拍与性能对照都依赖本 DP 的产物,
不得跳过。

**范围内**
- `flashattention_backend.py`:恢复 `cp_attn_forward_extend` /
  `cp_allgather_and_save_kv_cache` 的 import 与 `is_cp_mode` 下的 KV 写入调用
  (以 `git show bb737d7a82` 为准,等量恢复,不做任何"顺手优化")。

**范围外**:一切新设计;zigzag 逻辑修改;性能调优。

**验证条件**
1. 4 卡启动不再 NameError:
   `--tp-size 4 --attn-cp-size 2 --moe-dp-size 2 --ep-size 2 --enable-prefill-context-parallel --disable-piecewise-cuda-graph`;
2. GSM8K(`sglang.test.run_eval`,200 题,参数照
   `test/registered/4-gpu-models/test_qwen3_30b.py`)score ≥ 0.85;
3. logprob 对拍:同一 8k-token prompt,greedy,`/generate` 带
   `return_logprob=True, logprob_start_len=0`,对比 TP4(无 CP)服务:
   prompt logprobs 平均绝对差 < 2e-2、最大 < 2e-1;
4. 基线报告(写入 `docs/ring-attn/baselines.md`):TP4 与 TP4+CP2 在
   input 32k/128k、output 1 下的 TTFT(`bench_serving` 或 `bench_one_batch`),
   以及 128k 上下文、output 256 的 decode TPOT。

**完成定义**:上述 4 项全过;PR 合入;基线数据入库。

---

### DP1|stripe 所有权纯函数库

**目标**:D1 全部公式落为纯函数,穷举级单测,后续所有 DP 的地基。

**范围内**
- 新文件 `python/sglang/srt/layers/attention/ring/stripe.py`:
  `owner(pos)`、`local_len(rank, S)`、`local_idx(pos)`、
  `owned_positions(rank, start, end)`(向量化 torch 版 + 标量参考版)、
  `build_local_page_table(req_to_token_row, seq_len, rank, ...)` 的纯函数核心;
- 新文件 `test/srt/cp/test_cp_stripe.py`(注册 unit suite)。

**范围外**:任何运行时接线;GPU 代码。

**验证条件**
1. `python -m pytest test/srt/cp/test_cp_stripe.py -v` 全绿;
2. 单测必须覆盖:对 `c ∈ {2,3,4,8}`、`stripe ∈ {1,4,256}`、
   `S ∈ [0, 4·c·stripe]` 全量穷举,向量化版与暴力参考版逐点相等;
   恒等式 `Σ_r local_len(r,S) == S`;`local_idx` 在 owner 内单调且无碰撞;
3. pre-commit 全过。

**完成定义**:函数库零依赖(仅 torch)、文档字符串含公式;单测绿。

---

### DP2|逻辑/物理两级分配器 + server args(默认关闭,无行为变化)

**目标**:落地 D2/D3/D4 的存储与记账层,可在无 GPU 环境完整测试。

**范围内**
- 新文件 `python/sglang/srt/mem_cache/cp_sharded_allocator.py`:
  实现 `allocator.py` 同接口(`alloc/alloc_extend/alloc_decode/free/
  available_size` 等);逻辑全局记账 + 物理 owner-only 分配 + DUMMY 填充;
  按 req 记录本地 slot 集合供 free;暴露 debug 接口
  `get_shard_stats()`(本地物理占用、逻辑占用)供后续 DP 验证;
- `mem_cache/memory_pool.py`:预留 DUMMY_SLOT(容量 +1;debug 模式 NaN 填充);
- `server_args.py`:`--cp-kv-residency {replicated,sharded}`(默认
  replicated)、`--cp-stripe-size`;sharded 时的互斥校验(§4 支持矩阵:
  FA3、禁 radix、禁 spec/TBO/PD、kv dtype auto);
- `managers/schedule_batch.py`:allocator 注入点(sharded 时换 wrapper),
  flag 关闭时不构造 wrapper;
- 对称性校验钩子:env `SGLANG_CP_CHECK_SYMMETRY=1` 时,每次
  alloc/free 后跨 CP 组 allreduce 比对(逻辑占用, 决策序号)hash,不一致即
  crash-fast(实现可放 wrapper 内)。

**范围外**:attention/模型/调度策略改动;任何 kernel。

**验证条件**
1. 新单测 `test/srt/cp/test_cp_sharded_allocator.py`:单进程实例化
   c 个 wrapper(模拟 c 个 rank),回放同一随机 alloc/free 序列
   (extend/decode 混合,含 free 乱序):
   (a) 各实例逻辑 `available_size()` 序列逐步相等;
   (b) 物理占用差 ≤ 活跃 req 数 × stripe_size;
   (c) DUMMY 槽位从未出现在任何 free 列表;
2. 回归:flag 默认关时,
   `python -m pytest python/sglang/test/attention/test_flashattn_backend.py -v`
   及 `test/srt/test_srt_engine.py` 不回退;
3. sharded + 不满足支持矩阵的组合启动必须报清晰错误(逐项试)。

**完成定义**:单测绿;默认路径零 diff(代码审查确认 flag-off 时无新对象构造)。

---

### DP3|Striped ring prefill + sharded 存储(eager,正确性优先)

**目标**:prefill 端到端跑通 D5;显存真实分片;`max_new_tokens=1` 可服务。

**范围内**
- 新文件 `layers/attention/ring/ring_exchange.py`:P2P 双缓冲
  (staging `[max_local_len, h_kv_local, d] ×2`,K/V 各一组;独立 comm
  stream;event 同步;奇偶配对 send/recv;buffer 逐层复用);
- 新文件 `layers/attention/ring/ring_prefill.py`:c 步环形 driver,
  backend 以闭包注入 `partial_attn_fn`;`merge_state_v2` 累加;
- `layers/utils/cp_utils.py`:stripe 级切分/聚合
  (`cp_stripe_split_data/position`、`cp_stripe_gather_output`,推广现有
  round-robin 工具);stripe 版 CP metadata;
- `flashattention_backend.py::forward_extend`:
  `cp_kv_residency == sharded` 分支 → striped ring(s=0 causal 调用、
  s>0 无对角调用、staging 页表构造);K/V 写本地 `out_cache_loc`
  (走 DP2 wrapper 产出的本地 slot);
- `models/qwen3_moe.py` / `qwen2_moe.py`:CP 钩子在 sharded 模式下调
  stripe 版切分/聚合(等量替换,zigzag 路径保留);
- 门控:sharded 时 prefill bs=1(复用 PrefillAdder 现有限制)、
  radix 强制关闭、`--disable-cuda-graph` 要求(本 DP 阶段)。

**范围外**:decode CP(本 DP 后 decode 不可用,见验证方式)、chunked
prefill、prefix 命中、CUDA graph、性能调优。chunked prefill 在本 DP 用
`--chunked-prefill-size -1` 显式关闭并 assert。

**验证条件**
1. **首 token 对拍**(decode 未支持,用 `max_new_tokens=1`,采样关闭):
   2 卡(`--tp-size 2 --attn-cp-size 2 --cp-kv-residency sharded`)vs
   1 卡 TP1 参考,prompt 长度覆盖
   `{c·stripe-1, c·stripe, c·stripe+1, 8k, 32k(非整除值)}`:
   prompt logprobs 平均 |Δ| < 2e-2、最大 < 2e-1;greedy 首 token 一致;
2. **显存分片验证**:发送 64k prompt 后查各 rank
   `get_shard_stats()`(或 scheduler 日志 token 占用):每 rank 物理占用
   ∈ [64k/c − stripe, 64k/c + stripe];
3. **对称性**:`SGLANG_CP_CHECK_SYMMETRY=1` 下跑 20 条随机长度请求不触发
   crash-fast;
4. 4 卡配置(attn_tp=2×cp=2)重复验证 1、2;
5. 回归:replicated 模式 + DP0 用例不回退。

**完成定义**:上述全过;`ring/` 模块有 README 段落说明数据流。

---

### DP4|Decode 分布式 flash-decoding(eager)

**目标**:落地 D6,端到端多 token 生成正确;长上下文 decode 可用。

**范围内**
- 新文件 `layers/attention/ring/decode_merge.py`:
  partial → CP allgather(pynccl)→ `merge_state_v2` 链;
- `flashattention_backend.py`:
  `init_forward_metadata` 在 sharded 模式构造本地紧凑 page_table 与
  `local_cache_seqlens`(用 DP1 函数,向量化,预分配 buffer);
  `forward_decode` 增加 partial+merge 分支(`return_softmax_lse=True`);
- decode 新 token 写入:无需新代码——`out_cache_loc` 的 DUMMY 哨兵由
  DP2 wrapper 在 `alloc_decode` 时产生,本 DP 验证其端到端行为;
- `distributed/parallel_state.py`:attn_cp 组注册进 `graph_capture()`
  上下文(为 DP5 铺垫,本 DP 仍 eager)。

**范围外**:CUDA graph(仍 `--disable-cuda-graph`)、chunked prefill、
性能优化(one-shot merge 等)、radix。

**验证条件**
1. **多 token 对拍**:2 卡 sharded vs 1 卡 TP1,greedy,
   prompt 8k / 输出 64 token:输出 token 序列完全一致,或在出现分叉的
   位置验证 top-2 logprob 差 < 5e-2(bf16 容差判据,二者满足其一);
   输出 64 token 的逐 token logprob 平均 |Δ| < 2e-2;
2. **stripe 轮转边界**:构造 prompt 长度使 decode 恰好跨越 stripe 边界
   (如 `S0 = c·stripe − 8`,生成 32 token),验证生成连贯、
   `get_shard_stats()` 显示新增 slot 落在公式预测的 owner rank;
3. **长上下文**:4 卡、128k prompt、生成 256 token 成功;每 rank KV 物理
   占用 ≈ (128k+256)/c ± stripe;
4. **GSM8K**:4 卡 sharded 配置 score ≥ 0.85;
5. **多请求 decode batch**:并发 16 条不同长度请求(混合 1k–32k),
   全部完成且对称性校验通过;
6. 回归:replicated 路径与 DP0 基线不回退。

**完成定义**:上述全过;decode 数据流文档段落更新。

---

### DP5|Decode CUDA graph + chunked prefill + 长稳

**目标**:把 sharded 模式带到可压测状态。

**范围内**
- decode CUDA graph:捕获本地 kernel + CP allgather + merge;固定 shape
  buffer(`[max_bs, max_local_pages]` 页表等,值为 graph 输入);
  cuda_graph_runner 的 sharded 适配;
- chunked prefill(D5 末段):ring 每步发送
  `[池内本 rank 前缀分片(本地 gather 到 staging)‖ 本 chunk 新分片]`;
  解除 DP3 的 chunked 互斥;同机制顺带覆盖"多轮对话续推"(无 radix 下的
  re-extend);
- 长稳与故障路径:retract(decode OOM)、abort、max_total_tokens 逼近。

**范围外**:性能调优(DP6);radix(DP7a);prefill 进 graph(prefill 本就不捕获)。

**验证条件**
1. graph on(默认)vs `--disable-cuda-graph`:同 prompt greedy 64 token
   输出一致;GSM8K ≥ 0.85;
2. chunked 对拍:`--chunked-prefill-size 8192`、输入 64k:
   首 token logprob 与不分 chunk(DP3 路径)差 < 2e-2;
3. TPOT 初测:4 卡、128k 上下文、bs=1:记录 graph on 的 TPOT,
   与 DP0 基线(replicated 全量 KV decode)对比并写入 baselines.md
   (本 DP 只记录,不设达标线——达标在 DP6);
4. 长稳:bs=16 混合长度(1k–64k)连续 1 小时,无 hang、无 OOM 崩溃、
   对称性校验 0 触发;retract 注入测试(压小 `--max-total-tokens`)后
   请求最终全部完成;
5. 回归:replicated + 默认路径全量不回退。

**完成定义**:sharded 模式在文档标注为 experimental 可用;CI 注册
4 卡 sharded GSM8K 用例。

---

### DP6|性能达标:one-shot merge、stripe 调参、重叠验证

**目标**:兑现性能模型(附录 A),给出默认参数。

**范围内**
- decode merge 通信优化:symmetric-memory one-shot allgather
  (`use_symmetric_memory` + ld/st kernel,参考 flashinfer trtllm
  allreduce one-shot 路径),目标每层 < 5 µs;可选 gather+merge 融合 kernel;
- prefill ring 重叠调优:nsys 确认 comm stream 与 compute 重叠;
  staging gather(chunked 前缀)与上一步计算重叠;
- `cp_stripe_size` microbench(256/512/1024)定默认值;
- FP8 KV cache 支持评估(per-chunk descale),可做可立项后移。

**范围外**:新功能;DP7 内容。

**验证条件**(对照 `docs/ring-attn/baselines.md`,4 卡 Qwen3-30B)
1. **TTFT**:input 128k,sharded ring ≤ DP0 的 allgather CP 基线
   (预期更优:省掉关键路径 allgather 与全量池写);
2. **TPOT**:128k 上下文 bs=1,sharded ≤ replicated 基线 × 0.8
   (附录 A.3 预期 ~2 ms → ~1 ms 量级;若未达标,提交 nsys 分析报告
   说明瓶颈与后续项);
3. **容量**:同 `--mem-fraction-static` 下,sharded 可同时驻留的 64k 请求
   数 ≥ replicated 的 (c−0.5) 倍;
4. nsys 报告:prefill ring P2P 与 attention kernel 重叠率 > 80%
   (定义:P2P 时间中与 compute kernel 并行的占比);
5. 报告 stripe 默认值的依据数据。

**完成定义**:性能报告入 `docs/ring-attn/baselines.md`;默认参数固化;
sharded 标注从 experimental 升级或列出阻塞项。

---

### DP7|扩展轨道(DP5 后可并行,各自独立 PR)

**DP7a|CP-aware radix cache**
- 设计:树结构/匹配(按 token id)不变;节点 value 改
  「global_len + 本地 slot 数组」;evictable/protected 记账与逐出决策
  全用 global_len(D2 铁律);命中后 req_to_token 前缀段 = 本地 slot
  (owned 位)+ DUMMY。
- 验证:开 radix 后,同 prompt 二次请求 TTFT 显著下降且 logprob 与
  首次一致;eviction 压力测试下对称性校验 0 触发;
  关 radix 的 DP5 用例不回退。

**DP7b|混合 SWA 驻留(WeLM v4,扩展轨道中优先)**

> 本分支业务目标模型即 WeLM v4,DP5 完成后应最先做本项。
- 设计:`full_attention_layer_ids` 分片、SWA 层复制(decode 本地、
  不 merge;prefill 时 SWA 层 KV 从环上流转数据中取窗口写本地);
  挂点 `SWAKVPool` 层分组。
- 验证:混合模型(或构造 config)端到端对拍;SWA 层池占用为全量、
  full 层为 1/c。

**DP7c|短请求混合驻留(D8)**
- 设计:`< L_min` 请求 replicated(所有 rank 写、decode 本地不 merge),
  batch 内 sharded/replicated 两路分别计算;`L_min` 由 DP6 crossover
  数据定。
- 验证:短请求(1k)TPOT 回到 replicated 水平;混合 batch 正确性对拍。

---

## 7. 每 PR 全局验收门槛(所有 DP 通用)

1. `SKIP=no-commit-to-branch pre-commit run --all-files` 通过;
2. flag 默认关时,以下不回退:
   `python/sglang/test/attention/test_flashattn_backend.py`、
   `test/srt/test_srt_engine.py`、DP0 注册的 allgather CP 用例;
3. **WeLM 回归(本分支特有)**:凡触及 `flashattention_backend.py`、
   `schedule_batch.py`、`memory_pool.py`/allocator、`cuda_graph_runner.py`
   的 PR,须跑 WeLM v4 80A3 回归矩阵(`~/scripts`,来自
   perf_optimize_scripts.git)且精度/吞吐不回退;
4. 新增断言路径(支持矩阵外组合)有对应的负向测试;
5. 文档:本文件对应 DP 状态行更新(添加"状态:已交付 @commit"批注);
6. PR 描述包含:对应 DP 编号、验证条件逐项勾选结果、回滚方式。

---

## 附录 A|性能与代价模型(验收阈值的依据)

**A.1 内存**:每 rank KV = `S · h_kv_local · d · 2(K,V) · 2B / c`。
冗余系数:纯 TP 超额 = tp/h_kv;allgather CP = c;**本方案 = 1**。

**A.2 prefill 通信**(每层每 rank):ring 总量 `2·S·h_kv·d·(c−1)/c·2B`,
与 allgather 同量但流水化可重叠。例:Qwen3-30B,attn_tp=2(每 rank 2 KV
head)、S=128k、c=4:每步分片 ≈ 33 MB,NVLink ~0.2 ms/步,
被同步的 attention 计算(毫秒级)覆盖。

**A.3 decode TPOT**(Qwen3-30B,48 层,c=4,bs=1):

| 上下文 | 全量 KV 读/step | 分片读/step | merge 通信(pynccl → one-shot) | 净效果 |
|---|---|---|---|---|
| 128k | ~6 GB ≈ 2 ms | 0.5 ms | +0.7 → +0.2 ms | **赚 ~1–1.3 ms** |
| 32k | ~0.5 ms | 0.13 ms | 同上 | 约平 → 小赚 |
| 8k | ~0.13 ms | 0.03 ms | 同上 | 亏(→ DP7c) |

merge payload(每层每 rank)= `bs·h_q_local·d·2B + bs·h_q_local·4B`
(bs=64、h=16、d=128 → ~256 KB)。

**A.4 与上下文无关的收益**:容量 ×c(同显存服务 c 倍 token)——
这是本计划的第一目标。

## 附录 B|术语

- **c / cp_size**:`--attn-cp-size`,CP 组大小;attn_tp_size = tp/(dp·cp)。
- **stripe**:所有权粒度(`--cp-stripe-size`),page_size 整数倍。
- **DUMMY_SLOT**:池 slot 0,非 owned 位置的写入目标,只写不读。
- **replicated / sharded**:`--cp-kv-residency`,KV 驻留模式;
  replicated = 现状 allgather CP 行为。
- **partial (out, lse)**:某 KV 子集上的 attention 输出与 log-sum-exp,
  可经 `merge_state_v2` 无损合并。

## 附录 C|参考

- 原始 allgather CP 实现:commit `bb737d7a82`(#18233);损坏点见 §2.3。
- NSA round-robin CP:`server_args.py::nsa_prefill_cp_mode`、
  `layers/attention/nsa/utils.py`。
- 业界:vLLM Decode Context Parallel(interleaved 分片)、NVIDIA Helix
  Parallelism、Ring/Striped Attention 论文。
- 历史设计讨论稿已合并进本文(原 `phase2-sharded-kv.md` 已删除)。
