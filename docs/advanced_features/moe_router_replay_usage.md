# MoE Router Replay 使用指南

让 dummy-weight 模型在 forward 中**强制使用一份预先捕获的 router top-k expert ids**，跳过真实 router 选择。用途：

1. **吞吐 benchmark**：在没有真实权重时，仍然得到与真权重一致的 MoE 负载分布，量化 attention/MoE forward 的吞吐上限。
2. **离线分析 / 调试**：固定 routing 模式，便于复现性能/正确性问题。
3. **能力评估**：dummy + replay 可以在拿到全权重之前先测部署架构。

`return_routed_experts`（已有功能）是**捕获**侧；本文档介绍**回放**侧（v1 + v6 改动）。整体设计见 `docs/design/moe_router_replay.md` 与 `docs/design/routed_experts_remote_store.md`。

---

## 限制

- 必须在启动时设置 `--enable-moe-router-replay` 才接受 `routed_experts` 字段。
- 不支持混合 batch：同一个 ForwardBatch 不能同时有 replay 与非 replay 请求。
- 不支持 `--speculative-algorithm=*` 任何形式的 speculative decode。
- 不支持 `--enable-two-batch-overlap`。
- 不支持 `--moe-runner-backend` 中的 triton_kernels / flashinfer_trtllm / flashinfer_mxfp4。
- 启用此 feature 不影响默认 `/generate` 行为：请求体不带 `routed_experts` 字段时走原有 router。
- replay 仅 force expert ids，weights 仍然基于当前 router_logits 在 forced ids 上重新 gather + 归一化。

---

## 三种 trace 投递格式

| `format` | 客户端发什么 | 服务端做什么 | 适用场景 |
|---|---|---|---|
| 列表 / 张量（无 `format` 字段） | `[seq_len, num_layers, top_k]` 嵌套 int 列表 | tokenizer_manager 直接收下，转成 tensor 走 ZMQ | 单条小 trace 调试（< 1 MB） |
| `format: "remote"` | `{"format":"remote","backend":"redis","key":"...","shape":[...],"dtype":"int32","byte_size":N}` | 服务端 `RoutedExpertsStore` 异步从 Redis/Mooncake 拉，写到 `/dev/shm`，DP 内 mmap 共享 | 64 MB+ 单 trace、批量 sweep（v6 推荐）|
| `format: "shm"` | `{"format":"shm","path":"/dev/shm/sglang_replay_<hash>","shape":[...],"dtype":"int32"}` | 服务端直接 `os.open + mmap`，零拷贝零 IPC | 客户端已经把 trace 提前部署到节点本地 `/dev/shm` 的最快路径 |

`shm` 是 `remote` 路径的产物的"提前阶段"——`remote` 的 trace 也最终被 server 写到 `/dev/shm` 由 followers mmap。`shm` 跳过 Redis 拉取阶段，让客户端自己负责把文件铺到节点。

后两种格式对吞吐至关重要：单条 600B trace 是 130 MB，纯走 ZMQ + gloo 的 inline 路径会让 scheduler 主循环阻塞数分钟。

---

## v6 patch 改动概要与必要性

| 模块 | 改动 | 必要性 |
|---|---|---|
| `tokenizer_manager.py` `_resolve_router_replay_experts` | 不再在 tokenizer 端 fetch；将 `format=remote/shm` 的 ref dict 透传给 scheduler | 130 MB tensor pickle + ZMQ 跨进程一次要 ~270 ms，会卡死 admission 主循环 |
| `routed_experts_store.py` | 新增 `is_remote_routed_experts_ref()`、`decode_remote_routed_experts_tensor()` | 让 scheduler 端可以独立从 store 拉 trace |
| `scheduler.py` `__init__` | 持有 `RoutedExpertsStore` 句柄；初始化异步 fetch 线程池 + `/dev/shm` 跟踪表 | 让 leader 异步拉，不阻塞主循环 |
| `scheduler.py` `_enqueue_routed_replay_fetches` | 收到 `format=remote` 的 req → 提交到线程池 → 暂存 | leader-only；非 leader rank 看到的是已经 broadcast 好的 shm dict |
| `scheduler.py` `_drain_routed_replay_fetches` | 拉完之后写 `/dev/shm/sglang_replay_<uuid>`，把 ref dict 改成 `format=shm` | 后续 broadcast 只走 ~100 B 的 path，不走 64 MB 的 pickle |
| `scheduler.py` `_validate_router_replay_request` | 增加 `format=shm` 与 `format=remote` 两个分支：shm 走 mmap，remote 走 store fetch | 每个 rank 独立 mmap 同一文件，零拷贝零 broadcast |
| `scheduler.py` `_cleanup_routed_replay_shm` | absent counter 阈值清理（避免 follower 还没 mmap 就被 unlink） | 多 rank 间没有 fan-out 同步信号；用计数器 + grace period 替代 |
| `schedule_batch.py` `build_router_replay_{extend,decode}_batch` | 把 `mask & ge(0).all()` 与 sentinel→0 的清理从 per-layer (在 `topk.py`) 提到 per-step | 每 decode step 减少 ~5 × num_layers 次 kernel launch；600B 93 层每 step 省约 465 次 launch |
| `schedule_batch.py` `build_router_replay_decode_batch` | 不再用 `torch.tensor([True]*N, device=cuda)`，改成 `torch.ones(N, bool, cuda)` | 原写法触发 ~6-14 ms/step 的 H→D sync；TP=8 DP=4 80B c=32 占总耗时 7-8% |
| `topk.py` `_apply_router_replay_from_forward_batch` | 移除已经被 hoist 到 per-step 的 mask 清理代码 | 跟上面 schedule_batch 那个改动配套；保证调用契约不变（forced_ids 已 ≥ 0，mask 已包含全 layer 有效性）|
| `cuda_graph_runner.py` + `forward_batch_info.py` | 把 replay 张量的 DP gather 从 `dp_gather_partial` 改成 `dp_gather_replicate` | **正确性 fix**：DP-attn 下 replay 张量在 attn-TP 内是 replicated 的；用 partial（all-reduce sum）会把同一 expert id 加倍（510+510=1020），下游 `scores.gather(1, forced_ids)` 越界崩溃 |
| `io_struct.py` `TokenizedGenerateReqInput` | `router_replay_experts` 类型放宽到 `Union[List, torch.Tensor, Dict]` | 三种投递格式都能装，不影响默认值 None |

---

## 服务启动

```bash
python -m sglang.launch_server \
    --model-path /path/to/welmv4-600b \
    --load-format dummy \                     # router replay 通常配 dummy 权重
    --enable-moe-router-replay \              # 必须
    --routed-experts-store-dsn \              # 仅当客户端会发 format=remote 时才需要
        'redis://10.0.0.99:6380/0?prefix=sglang:routed_experts_600b' \
    --tp-size 32 --dp-size 4 --moe-dp-size 1 \
    --enable-dp-attention --enable-dp-lm-head \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend fake \
    --disable-radix-cache \
    --disable-hybrid-swa-memory \             # SWA 模型必加，否则启动 assert
    --decode-attention-backend fa3 \
    --page-size 16 \
    --enable-metrics
```

启动后服务能接受三种格式的 `routed_experts`。`--routed-experts-store-dsn` 在你只发 `format=shm` 时**可选**——server 不需要去 Redis 拉。

---

## 三种投递格式的请求 body

### 1. inline list（< 1 MB 调试）

```json
{
    "input_ids": [...],
    "sampling_params": {"max_new_tokens": 8, "temperature": 0.0},
    "routed_experts": [
        [[5, 12, 67, 209, 410, 88, 32, 156, 290, 401], ...],
        ...
    ]
}
```

### 2. `format=remote`（让 server 自己去 store 拉）

```json
{
    "input_ids": [...],
    "routed_experts": {
        "format": "remote",
        "backend": "redis",
        "key": "sglang:routed_experts_600b:abc123...",
        "encoding": "raw_tensor_bytes",
        "shape": [34815, 93, 10],
        "dtype": "int32",
        "byte_size": 129511800
    }
}
```

服务启动时必须配 `--routed-experts-store-dsn` 指向同一个 Redis（且 `prefix` 一致）。

### 3. `format=shm`（最快路径，客户端预铺到节点）

```json
{
    "input_ids": [...],
    "routed_experts": {
        "format": "shm",
        "path": "/dev/shm/sglang_replay_<hash>",
        "shape": [34815, 93, 10],
        "dtype": "int32"
    }
}
```

每个 server 节点的 `/dev/shm` 必须已经存在该文件（多节点部署时每个节点都要铺）。Redis 完全不参与推理 hot path。

---

## WeLM v4.5 600B 端到端流程

需要的：
- 一份**真权重** 600B 模型（用来 capture）
- 一份**任意架构** 600B（或层数兼容的）模型（用来 replay）—— 通常用 `--load-format dummy`
- 一台 Redis（96 GB+ 内存，至少 256 MB `proto-max-bulk-len`）
- 4 个推理节点（TP=32 DP=4 EP=32 跨 4 节点的典型部署）

### Phase 0: 准备 Redis + 应用 patch（一次性）

```bash
# Redis 上调好 maxmemory + proto-max-bulk-len（单 600B trace 130 MB 远超 64 MB 默认上限）
cat > redis.conf <<'EOF'
bind 0.0.0.0
port 6380
daemonize yes
save ""
maxmemory 96gb
maxmemory-policy noeviction
proto-max-bulk-len 256mb
client-output-buffer-limit normal 1gb 512mb 60
EOF
redis-server redis.conf

# 在每个推理节点 git apply 本 patch
cd /path/to/sglang-perf-v4
git apply <patch>
```

### Phase 1: 准备数据集

```python
# 一个固定的 prompt 池 — 每条 (input_ids, max_new_tokens, bootstrap_room)
# 关键约束：seed 固定，便于 capture/replay 用同一份输入
import json, random
N = 512; INPUT_LEN = 32768; OUTPUT_LEN = 2048; SEED = 20260604
rng = random.Random(SEED)
ds = {
    "name": "welm600b_n512",
    "input_len": INPUT_LEN, "output_len": OUTPUT_LEN,
    "bootstrap_host": "2.2.2.2",
    "requests": [
        {"id": i,
         "input_ids": [rng.randint(0, 155647) for _ in range(INPUT_LEN)],
         "max_new_tokens": OUTPUT_LEN,
         "bootstrap_room": 1_000_000 + i}
        for i in range(N)
    ],
}
json.dump(ds, open("dataset.json", "w"))
```

### Phase 2: Capture 真权重 trace

```bash
# 启动 capture server: 真权重 + --enable-return-routed-experts
python -m sglang.launch_server \
    --model-path /path/to/welmv4-600b-real \
    --load-format auto \
    --enable-return-routed-experts \
    --disable-cuda-graph \                  # capture 必须关 cuda graph
    --tp-size 32 --dp-size 4 --moe-dp-size 1 \
    --enable-dp-attention --enable-dp-lm-head \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend fake \
    --disable-radix-cache \
    --disable-hybrid-swa-memory \
    --port 30002 ...
```

每条 prompt POST 一次，把响应里的 `meta_info.routed_experts`（base64 或 remote ref）落盘成 `req_NNN.npz`。注意：`--routed-experts-start-len` 指向 input_len，response 只返回 decode 部分。

### Phase 3: Push 到 Redis

```python
# 每条 trace 写成 RESP "SET <prefix>:<hash> <raw_int32_bytes>"
# 单条 130 MB（uncompressed），512 条共 65 GB
# emits redis_refs.json: 一份 manifest 记录 request_idx → redis_key 的映射
```

### Phase 4: Prefetch 到节点 `/dev/shm`（推荐路径）

在每个推理节点跑一次：

```python
# 用 redis_refs.json 作为索引，从 Redis pull 全部 trace 写到本地 /dev/shm
# 每节点产出 shm_refs.json (manifest 内 path 指向 /dev/shm/sglang_replay_<hash>)
```

如果客户端不想做 prefetch（嫌麻烦），跳过这一步直接用 `format=remote`，让 v6 patch 的异步 fetch pool 替你做。结果一样，仅多 ~5 min 的 sweep 启动开销（512 × 130 MB / 200 MB/s ≈ 5 min 在 leader 节点 pull）。

### Phase 5: 启动 replay server + bench

```bash
# replay server: dummy 权重 + --enable-moe-router-replay (cuda graph ON)
python -m sglang.launch_server \
    --model-path /path/to/welmv4-600b-dummy \
    --load-format dummy \
    --enable-moe-router-replay \
    --routed-experts-store-dsn 'redis://10.0.0.99:6380/0?prefix=sglang:routed_experts_600b' \
    --tp-size 32 --dp-size 4 --moe-dp-size 1 \
    --enable-dp-attention --enable-dp-lm-head \
    --disaggregation-mode decode --disaggregation-transfer-backend fake \
    --disable-radix-cache --disable-hybrid-swa-memory \
    --port 30000 ...
```

bench 客户端发 `format=shm` ref（如果 phase 4 跑过）或 `format=remote`：

```python
import aiohttp
body = {
    "input_ids": prompt_ids,
    "sampling_params": {"max_new_tokens": 2048, "temperature": 0.0, "ignore_eos": True},
    "bootstrap_host": "2.2.2.2",
    "bootstrap_room": 1_000_000 + i,
    "stream": False,
    "routed_experts": {                 # 用 phase 4 manifest 里的 ref
        "format": "shm",
        "path": "/dev/shm/sglang_replay_abc123...",
        "shape": [34815, 93, 10],
        "dtype": "int32",
    },
}
# 注意：用 force_close=True 的 TCPConnector 防 VIP/proxy 长连接死锁
```

---

## 不同模型架构间的 trace 重用

trace 张量 shape 是 `(num_tokens, num_hidden_layers, num_experts_per_tok)`。要换替换 replay 模型时：

| 字段 | 必须与 capture 一致 | 不一致时 |
|---|---|---|
| `num_experts` | ✅ | trace 里有 OOR id，crashes downstream gather |
| `num_experts_per_tok` (top_k) | ✅ | shape 不匹配 |
| `routing_type` / `score_func` | ✅ | weight 计算路径不一致 |
| `num_hidden_layers` | ❌ | 把 trace 沿 axis=1 切片到目标层数即可 |
| `hidden_size`、`num_attention_heads`、`sliding_window` 等 | ❌ | trace 完全不存这些 |

切片示例（93 层 → 77 层）：

```python
arr = np.fromfile(shm_path, dtype=np.int32).reshape(N_TOKENS, 93, 10)
np.ascontiguousarray(arr[:, :77, :]).tofile(shm_path_77L)
```

WeLM v4.5 系列里：捕获用 `welmv4-600b-real`（93 layers），replay 通常用 `model_32k_sw128_*` 或 `*_1p3*` 这些 dummy 架构（77 layers），需要切片。

---

## 故障排查

| 现象 | 根因 | 解决 |
|---|---|---|
| 启动报 `Memory pool size is too small` | SWA 模型上 hybrid_swa_memory 检测分配 0-token aux pool | 启动加 `--disable-hybrid-swa-memory` |
| Redis push 失败 `proto-max-bulk-len exceeded` | 默认 64 MB 上限 < 单 trace 130 MB | `proto-max-bulk-len 256mb` |
| Bench 期间 Redis 被 OOM kill | 大量并发 GET 撑爆 client output buffer，`maxmemory` 不限制此项 | (a) 换 `format=shm` 旁路 Redis；(b) `client-output-buffer-limit normal 1gb 512mb 60` |
| Server log 静默无请求活动，bench HTTP 长时间 ESTABLISHED | aiohttp 默认 keep-alive 通过 VIP/proxy 异常 | 客户端用 `aiohttp.TCPConnector(force_close=True)` |
| Bench 看似 OK 但 `per_req_tps` 异常高（如 in=32k out=4 用时 0.4s）| 请求字段名错（`routed_experts_ref` vs `routed_experts`），未知字段被丢弃，replay 静默跳过 | 字段名必须为 `routed_experts` |
| Server 报 `Unsupported routed_experts payload format: 'shm'` | leader 节点的 tokenizer_manager 没应用本 patch | 重新 apply、重启 |
| 同一架构两次 sweep 数字差异 > 1% | 通常是 cuda graph capture 状态没暖、或并发上限触发 micro-batch 串行 | 检查 `--cuda-graph-bs` 列表覆盖你的 c；c 离开 peak 后 `per_req_tps` 退化是正常 |

---

## 验证 replay 真的生效（gold-standard echo test）

可以临时启动一个验证 server（cuda graph OFF + return-routed-experts ON + replay ON），发已知 trace，比较返回的 topk 是否按 cell 等于 trace。设计见 `docs/design/moe_router_replay.md`。

由于 `apply_router_replay_topk_override()` 是 `topk.py` 中的无条件 `torch.where(mask[:,None], forced_ids, topk_ids)`：只要 `forward_batch.router_replay_topk_ids != None` 且 `mask=True`，那一行的 expert ids 就**完全**被 trace 覆盖（数学上无 partial override 的可能）。所以验证退化为"`router_replay_topk_ids` 是否被设置 + mask 是否 True"——这两个条件在每个 forward step 都能 grep server log 实测确认。

---

## 兼容性

| 场景 | 是否受本 patch 影响 |
|---|---|
| 不传 `routed_experts` 的请求（默认行为） | **完全不变**：所有新代码路径都被 `routed_experts is None` / `router_replay_topk_ids is None` 提前 return 防住 |
| `--enable-moe-router-replay=False`（默认）| **完全不变**：scheduler 构造时新增的 5 个属性（fetch_pool=None、pending=[]、shm={}…）零开销，每个 method 入口都有空检查 |
| 仅 `return_routed_experts=True`（capture 路径，无 replay）| **完全不变**：只走 tokenizer_manager 的 `_decode_remote_routed_experts_tensor` 老路径 |
| Replay + DP-attn | DP gather 从 partial 改成 replicate，**修了之前会 crash 的 bug** |
| Replay + cuda graph | 兼容，graph 内 forward 路径正常走 override |
| Replay + speculative decode | 不支持（旧约束，未变）|
| Replay + TBO | 不支持（旧约束，未变）|

---

## 性能数据（welm v4.5 600B, 4-node TP=32 DP=4, in=32k out=2k, dummy weights）

| c | n | wall (s) | out_tps | per_req_tps | p99/p50 | Redis GETs |
|---|---|---|---|---|---|---|
| 32 | 64 | 244.7 | 535.7 | 16.8 | 1.03 | 0 |
| 64 | 128 | 265.9 | 985.8 | 15.5 | 1.01 | 0 |
| 96 | 192 | 282.2 | 1393.39 | 13.9 | 1.01 | 0 |
| 112 | 224 | 318.09 | 1442.20 | 12.1 | 1.02 | 0 |
| 128 | 256 | 503.4 | 1041.4 (over-saturated) | 10.5 | 1.03 | 0 |

完全走 `format=shm` 路径时，Redis GET 计数全程 0 增长。`p99/p50 ≈ 1.01-1.03`，无长尾。

---

## 相关代码与文档

- 设计：`docs/design/moe_router_replay.md`、`docs/design/routed_experts_remote_store.md`
- 服务端核心：`python/sglang/srt/managers/scheduler.py`、`schedule_batch.py`、`tokenizer_manager.py`、`routed_experts_store.py`
- forward 路径：`python/sglang/srt/layers/moe/topk.py` (`_apply_router_replay_from_forward_batch`、`apply_router_replay_topk_override`)
- DP-attn 适配：`python/sglang/srt/model_executor/cuda_graph_runner.py`、`forward_batch_info.py`
