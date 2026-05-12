# Routed Experts Remote Store Design

## 背景与目标

当前 `return_routed_experts` 会在请求结束后把每个 token、每个 MoE layer 的 top-k expert ids 作为 int32 tensor 捕获，并在 `TokenizerManager` 中编码为 base64 字符串返回：

```python
meta_info["routed_experts"] = base64(tensor.numpy().tobytes())
```

这个格式对小请求方便，但对长 prompt、长 decode、较多 MoE 层或较大 batch 不合适：

- HTTP/SSE payload 会随 `[num_tokens, num_layers, top_k]` 线性膨胀。
- base64 有约 33% 编码膨胀，并增加 CPU copy/encode 成本。
- replay 或离线分析场景通常只需要一个稳定引用，不需要把二进制数据放进每个响应体。

目标：

- 抽象 routed experts 存储接口，支持把二进制 routed experts payload 写入远端存储。
- 第一阶段支持 `inline`、`mooncake`、`redis` 三类 backend。
- 默认接口完全兼容旧行为，`routed_experts` 继续返回 base64 string。
- 只有启动时通过 DSN 配置 remote backend 时，响应中才返回可解析的 key/ref。
- 保持兼容：默认行为仍是当前 base64 inline 返回；只有显式配置 remote store 时才改变返回格式。
- 为后续 router replay 支持 “从 key 读取 routed experts” 留出协议与接口。

非目标：

- 本设计阶段不实现 replay 从远端 store 读取。
- 不改变 `return_routed_experts` 的捕获逻辑和 scheduler 内部 tensor 表示。
- 不要求 OpenAI 官方 schema 兼容；remote ref 放在 SGLang extension 字段里。

## 当前数据流

现有路径：

1. `Req.return_routed_experts=True`。
2. Scheduler 在请求结束或输出阶段调用 `get_global_experts_capturer().get_routed_experts(...)`。
3. `BatchTokenIDOutput.routed_experts` 携带 `List[Optional[torch.Tensor]]` 到 Detokenizer。
4. `BatchStrOutput.routed_experts` 继续传给 TokenizerManager。
5. `TokenizerManager` 把每个 tensor 转 bytes 后 base64，写入 `meta_info["routed_experts"]`。
6. OpenAI 层从 `meta_info` 取出字符串，放入 `sglext.routed_experts`。

基础版 remote store 可以只替换第 5 步，避免影响 GPU/Scheduler 热路径。Mooncake device-direct 是进一步优化路径，会把 store 写入前移到 capturer export 阶段。

## Capturer vs Store Boundary

不建议为不同 backend 实现不同的 `experts_capturer`。

`experts_capturer` 属于模型执行和 scheduler 热路径，它应该只负责捕获 routed experts tensor，并保持 backend-agnostic。不同 backend 的差异应封装在 `RoutedExpertsStore`，在 `TokenizerManager` 收到最终 tensor 后完成序列化、写入和 response value 生成。

这样做有几个好处：

- routed experts 的语义只有一份，不会因为 Redis/Mooncake/inline 出现不同捕获实现。
- remote IO 不进入 GPU/scheduler 路径，降低对 decode 延迟的影响。
- 默认 inline 路径可以继续复用当前 base64 行为，兼容旧客户端和老版本部署。
- 后续如果 Mooncake 需要更少 copy，可以在 store 内部引入 staging buffer 或 allocator，而不是拆分 capturer。

这个边界对 `inline` 和 `redis` 成立，因为它们最终都需要 host bytes。对 `mooncake` 可以进一步优化：仍然不要复制出多套业务语义不同的 capturer，但需要给 capturer 增加 backend capability 感知的导出路径，例如 `HostTensorExport` 和 `DeviceDirectExport`。

- `inline` / `redis`: `capture -> device cache -> D2H host cache -> per-request tensor -> response/store`。
- `mooncake`: `capture -> device cache -> compact device staging buffer -> Mooncake batch_put_from(device_ptr)`，尽量不经过 D2H。

也就是说，不同 backend 不应该实现不同 routed experts 语义；但 `RoutedExpertsStore` 可以声明能力，capturer 根据能力选择 host export 或 device-direct export。

## Performance Tradeoff

remote store 只优化后段传输和序列化成本，不会自动消除 routed experts capture 本身的开销。

当前 capturer 的主要成本在前段：

- MoE top-k 后每层调用 `capture(layer_id, topk_ids)`，把 top-k ids 写入 device cache。
- 每次 forward 结束后把本轮 token 的所有 MoE layer routed experts 从 device cache 同步到 pinned host cache。
- 请求完成时再根据 `req_to_token_pool` 从 host cache gather 出该请求的 `[num_tokens, num_layers, top_k]` tensor。
- 默认 inline 返回还会在 `TokenizerManager` 中做 `.numpy().tobytes()` 和 base64 encode。

因此 remote backend 的收益主要是：

- 避免把大体积 base64 字符串塞进 HTTP/SSE response。
- 避免 response JSON 膨胀和客户端接收/解析大字符串。
- 对长 prompt、长 decode、批量离线分析场景更友好。

但如果目标是降低开启 `return_routed_experts` 后的端到端推理开销，只做 remote store 不够。remote store 是必要但不充分的一步，下一阶段应优化 capture 路径：

- 按 batch/request gating：只有当前 batch 中存在 `return_routed_experts=True` 或 router replay 相关请求时才启用 capture 和 D2H sync。
- 缩小同步范围：只同步需要返回 routed experts 的 token 区间，避免对无关请求或 padding token 做 D2H。
- Mooncake device-direct：对 Mooncake backend 不做 D2H，直接把 device staging buffer 注册到 Mooncake 并 `batch_put_from`。
- 异步化 D2H：对仍需 host bytes 的 backend，用 pinned host buffer 和 CUDA stream/event，把 D2H 从 scheduler critical path 中移出，完成后再进入 detokenizer/tokenizer 输出路径。
- 延迟 materialize：保留 compact host buffer 或 per-request view，直到请求完成且确实要输出时再组装最终 tensor。
- 增加指标：分别统计 capture write、D2H sync、request gather、base64 encode、remote put、response size，避免只看到 remote store 的局部收益。

实现优先级建议：先做 DSN remote store 和协议兼容，解决大 response；随后做 per-batch capture gating 和分段耗时指标，再决定是否需要更激进的 async D2H/Mooncake staging 优化。

## Dummy Backend Benchmark

在实现 Redis/Mooncake 前，应先支持 `dummy://` backend，用来回答一个关键问题：`TokenizerManager` 进程拿到 routed experts tensor 之前和之后，到底有多少 overhead，会不会影响吞吐。

`DummyStore` 行为：

- 接口上仍认为请求打开了 `return_routed_experts`。
- Scheduler、Detokenizer、TokenizerManager 仍走 routed experts 数据流。
- TokenizerManager 收到 `routed_experts_tensor` 后不做 `.numpy().tobytes()`、不做 base64、不写远端 store。
- response 中只返回一个很小的 marker，例如：

```json
{
  "format": "dummy",
  "backend": "dummy",
  "dropped": true,
  "shape": [1024, 48, 8],
  "byte_size": 1572864
}
```

这个实验可以拆出几段成本：

- baseline: server 不启用 routed experts，请求不传 `return_routed_experts`。
- allocation-only: server 启用 routed experts，但请求不传 `return_routed_experts`。用于观察 capturer 初始化和全局 capture 是否已经影响普通流量。
- dummy: server 启用 `--routed-experts-store-dsn dummy://`，请求传 `return_routed_experts=true`。用于测 capture、D2H、request gather、跨进程传 tensor、TokenizerManager 收到 tensor后的最小处理成本。
- inline: 同样请求，但使用默认 inline base64。`inline - dummy` 近似代表 `.numpy().tobytes()`、base64、response JSON 和网络传输成本。
- remote host-export: Redis/Mooncake host-export。`remote - dummy` 近似代表 host bytes 序列化和 store put 成本。

需要关注的指标：

- request throughput、TTFT、ITL、E2E latency p50/p90/p99。
- Scheduler forward end 到 output processing 的耗时。
- D2H sync 耗时。
- `maybe_collect_routed_experts()` gather 耗时。
- Detokenizer/TokenizerManager queue latency 和 tensor 传递耗时。
- TokenizerManager routed experts 处理耗时。
- response body bytes。

判断标准：

- 如果 `dummy` 相比 baseline 已经明显掉吞吐，瓶颈主要在 capturer/D2H/跨进程传递，remote store 的收益有限，应优先做 per-batch gating、device-direct 或 async D2H。
- 如果 `dummy` 接近 baseline，但 `inline` 明显变慢，说明主要瓶颈在 base64/response 传输，remote store 会有直接收益。
- 如果 allocation-only 已经变慢，说明当前 capturer 的全局启用策略不合理，需要先做 request-level gating。

## Mooncake Device-Direct Export

Mooncake store 的底层接口可以从任意已注册 buffer 指针读取：

```python
store.register_buffer(ptr, size)
store.batch_put_from(keys, ptrs, sizes)
```

因此 Mooncake backend 不必沿用当前 host cache。建议引入一个可选导出接口：

```python
class RoutedExpertsStore(Protocol):
    supports_device_direct: bool

    def put(self, payload: RoutedExpertsPayload) -> RoutedExpertsRef:
        ...

    def put_device_parts(self, parts: list[RoutedExpertsDevicePart]) -> RoutedExpertsRef:
        ...
```

其中 `RoutedExpertsDevicePart` 描述 device buffer 的一个连续片段：

```python
@dataclass
class RoutedExpertsDevicePart:
    key: str
    ptr: int
    byte_size: int
    dtype: str
    shape: tuple[int, int, int]
    token_start: int
    token_count: int
    layer_start: int
    layer_count: int
    tp_rank: int
    ep_rank: int
    pp_rank: int
```

Mooncake v1 推荐使用 compact device staging buffer，而不是直接暴露当前 device cache：

- 当前 device cache 按 forward batch 写入，request 完成时需要通过 `req_to_token_pool` gather，目标 token 通常不是一个连续切片。
- 把单个 request 的 local layer/token routed experts 先做 device-to-device compact，形成连续 `[num_req_tokens, local_layers, top_k]` staging tensor。
- staging tensor 注册到 Mooncake 后直接 `batch_put_from`。
- staging tensor 生命周期必须覆盖 Mooncake put 完成，可用 CUDA event 或同步结果管理。

这样可以避免 D2H，同时保持外部读取时仍看到稳定的逻辑 shape。

## Parallel Aggregation

Mooncake device-direct 会把“返回一个完整 routed experts tensor”的问题变成“多个 rank 写出若干 part 后聚合成一个 manifest”。

目标逻辑 shape 仍是：

```text
[num_tokens, global_num_moe_layers, top_k]
```

聚合原则：

- TP：router top-k logical ids 通常在 TP rank 间重复，默认只选一个 writer rank，建议 `moe_tp_rank == 0` 或等价的 owner rank，避免重复写同一份 routed experts。
- EP：capture 点必须在 logical-to-physical expert id 转换之前。若 EP rank 按 token 分片，只写本 rank 拥有的 token rows；若 top-k 在 EP rank 间重复，也只选 owner rank。
- PP：每个 PP stage 只拥有自己的 layer range，因此每个 PP stage 写 `[num_tokens, local_layers, top_k]` part。
- DP：请求属于某个 DP worker，本设计只聚合同一个请求所在 DP replica 的 parts，不跨 DP replica 合并。

response 不应直接返回所有 part，避免 response 又变大。remote ref 返回一个 manifest key：

```json
{
  "format": "remote_ref",
  "backend": "mooncake",
  "dsn_scheme": "mooncake",
  "layout": "partitioned_manifest",
  "key": "sglang:routed_experts:v1:{rid}:manifest",
  "dtype": "int32",
  "shape": [1024, 48, 8],
  "encoding": "raw",
  "byte_size": 1572864
}
```

manifest 内容存放在 store 中，描述每个 part 的 key、rank、token range、layer range、shape 和 byte size：

```json
{
  "version": 1,
  "shape": [1024, 48, 8],
  "parts": [
    {
      "key": "sglang:routed_experts:v1:{rid}:pp0:tp0:ep0",
      "pp_rank": 0,
      "tp_rank": 0,
      "ep_rank": 0,
      "token_start": 0,
      "token_count": 1024,
      "layer_start": 0,
      "layer_count": 12,
      "shape": [1024, 12, 8],
      "byte_size": 393216
    }
  ]
}
```

协调方式建议：

- 每个 producing scheduler rank 在请求完成或本 PP stage 完成时写自己的 part。
- final/output rank 收集 part metadata，确认覆盖完整 `[token, layer]` 后写 manifest。
- 如果现有 PP 输出链路不能自然携带 part metadata，先在 v1 限制 Mooncake device-direct 只支持 `pp_size == 1`，同时保留 host-export fallback；随后再补 PP metadata 汇聚。

失败语义：

- 任一 required part 写失败，manifest 不写入，请求按 `fallback` 策略处理。
- manifest 写入必须发生在所有 part 可读之后，避免客户端拿到半成品 key。
- manifest 应包含 `writer_world_size`、`tp_size`、`ep_size`、`pp_size`、`dp_rank`，方便读取侧校验并行拓扑是否匹配。

## Public Response Schema

保留现有字段名 `routed_experts`，并明确双形态：

- 默认未配置 remote DSN，或显式使用 `inline://`：`routed_experts` 仍是 base64 string，和旧接口一致。
- 配置 `redis://...`、`mooncake://...` 等 remote DSN：`routed_experts` 返回 remote ref object。
- 配置 `dummy://`：只用于 benchmark，返回小 object，不返回真实 routed experts 数据。

```json
{
  "sglext": {
    "routed_experts": {
      "format": "remote_ref",
      "backend": "mooncake",
      "dsn_scheme": "mooncake",
      "key": "sglang:routed_experts:v1:...",
      "dtype": "int32",
      "shape": [1024, 48, 8],
      "encoding": "raw",
      "byte_size": 1572864,
      "expires_at": 1778563600
    }
  }
}
```

兼容策略：

- `inline` backend 保持现状：`routed_experts` 是 base64 string，不包 object。
- `remote` backend 返回 object。客户端可通过 JSON 类型或 `format == "remote_ref"` 判断。
- 不按 payload 大小自动切换 string/object，避免同一启动配置下 response schema 不稳定。

remote ref 公共字段：

- `format`: 固定 `"remote_ref"`；`dummy://` benchmark 使用 `"dummy"`。
- `backend`: `"dummy" | "mooncake" | "redis"`。
- `dsn_scheme`: remote DSN scheme，例如 `"redis"` 或 `"mooncake"`。
- `layout`: `"single_blob"` 或 `"partitioned_manifest"`，默认 `"single_blob"`。
- `key`: 远端存储 key。
- `dtype`: 初始固定 `"int32"`。
- `shape`: `[num_tokens, num_layers, top_k]`。
- `encoding`: 初始固定 `"raw"`，未来可扩展 `"zstd"`。
- `byte_size`: 原始 payload 字节数。
- `expires_at`: Unix timestamp，可选。无 TTL 时为 `null`。
- `checksum`: 可选，建议 `xxh64` 或 `sha256`，用于跨服务读取校验。

## Storage Interface

新增模块建议：

```text
python/sglang/srt/routed_experts_store/
  __init__.py
  base.py
  dummy_store.py
  inline_store.py
  redis_store.py
  mooncake_store.py
  factory.py
  serde.py
```

核心接口：

```python
@dataclass
class RoutedExpertsPayload:
    data: memoryview
    dtype: str
    shape: tuple[int, int, int]
    byte_size: int
    request_id: str
    created_at: float
    ttl_s: Optional[int]


@dataclass
class RoutedExpertsRef:
    format: Literal["base64", "remote_ref"]
    backend: str
    dsn_scheme: Optional[str]
    layout: Literal["single_blob", "partitioned_manifest"]
    key: Optional[str]
    dtype: str
    shape: tuple[int, int, int]
    encoding: Literal["raw", "zstd"]
    byte_size: int
    expires_at: Optional[int]
    checksum: Optional[str] = None
    inline_data: Optional[str] = None


class RoutedExpertsStore(Protocol):
    backend_name: str
    ttl_s: Optional[int]

    def put(self, payload: RoutedExpertsPayload) -> RoutedExpertsRef:
        ...

    def get(self, ref: RoutedExpertsRef) -> bytes:
        ...

    def delete(self, ref: RoutedExpertsRef) -> None:
        ...

    def close(self) -> None:
        ...
```

说明：

- `put()` 必须是同步接口，便于第一版接入现有 TokenizerManager 代码。后续可增加 async/batch API。
- `get()` 是为 replay 和调试预留，第一版可以只在单测中覆盖。
- `delete()` 对 TTL backend 可以是 no-op。
- `RoutedExpertsRef` 负责生成 response value。`InlineStore` 返回旧的 base64 string；remote store 返回 ref object，不把 backend 客户端对象泄漏到 API 层。

## Serialization

初始序列化固定为：

- Tensor dtype: `torch.int32`。
- CPU contiguous。
- Byte order: native little-endian。
- Data bytes: `tensor.numpy().tobytes(order="C")`。
- Shape metadata 单独放在 ref 中，不写入 raw bytes。

`serde.py` 提供：

```python
def tensor_to_payload(tensor: torch.Tensor, request_id: str, ttl_s: Optional[int]) -> RoutedExpertsPayload
def payload_to_tensor(data: bytes, dtype: str, shape: tuple[int, int, int]) -> torch.Tensor
```

第一版不压缩。后续如果 payload 很大，可以在 store 配置里启用 `encoding=zstd`，但必须把压缩前后的 `byte_size` 和 `encoding` 写入 ref。

## Key Design

key 必须可跨进程唯一，建议格式：

```text
sglang:routed_experts:v1:{served_model_name}:{request_id}:{uuid}
```

如果需要便于多租户隔离，可以加入 namespace：

```text
{namespace}:routed_experts:v1:{served_model_name}:{request_id}:{uuid}
```

key 不应直接包含 prompt hash 或用户输入，避免泄漏请求内容。

## Backend Design

### DummyStore

用途：性能隔离实验，不用于生产返回 routed experts。

建议启动配置：

```text
--routed-experts-store-dsn dummy://
```

行为：

- `put()` 不读取 tensor bytes，不做 base64，不写远端。
- 返回一个小 object，包含 `shape`、`byte_size` 和 `dropped=true`。
- 不支持 `get()`。

注意：

- `dummy://` 仍应保留前面的 routed experts 数据流，不能在 scheduler 侧短路，否则测不到 capture、D2H、request gather 和跨进程传递成本。
- `dummy://` 结果不能用于 replay，只用于性能对比。

### InlineStore

用途：保持默认兼容。

行为：

- `put()` 直接返回 `RoutedExpertsRef(format="base64", inline_data=...)`。
- 不需要 `get()` 支持，或只在 ref 带 `inline_data` 时 decode。

### RedisRoutedExpertsStore

用途：部署简单，适合中小 payload、调试和非 RDMA 环境。

建议启动配置：

```text
--routed-experts-store-dsn redis://host:6379/0?ttl=3600&namespace=sglang&fallback=inline
```

实现：

- 使用 `redis.Redis.from_url(url)`，不要复用现有 `connector.RedisConnector` 的 tensor serde；这里存的是 raw bytes，不是 safetensors。
- `SET key value EX ttl`。
- `GET key` 返回 bytes。
- `DEL key` 删除。

注意：

- Redis 单 value 较大时会带来内存压力，建议服务启动时打印 warning，例如 payload 超过 64MB 时建议 Mooncake。
- 如果 Redis 不可用，默认 fail request；可选 DSN 参数 `fallback=inline` 用于降级。

### MooncakeRoutedExpertsStore

用途：高吞吐、低 copy、适合大 payload 和同机/跨机高性能读取。

复用现有 Mooncake 配置加载逻辑：

- `MooncakeBaseStore._load_config()`
- `MooncakeDistributedStore`
- `register_buffer()` / `batch_put_from()` / `batch_get_into()` 思路参考 `MooncakeEmbeddingStore`

第一版实现建议：

- 在 TokenizerManager 进程初始化一个 pinned CPU staging buffer pool。
- 将 routed experts tensor 拷贝到 staging buffer。
- 调用 `store.batch_put_from([key], [ptr], [size])`。
- 返回 `remote_ref`。

如果 payload 可能超过固定 staging buffer：

- 先做 simple path：按需分配 CPU contiguous `torch.empty(size, dtype=torch.uint8, pin_memory=True)` 并 `register_buffer()`。
- 后续再引入 allocator/pool，参考 `EmbeddingCacheController.ContiguousMemoryAllocator`。

TTL：

- Mooncake store 本身如果没有 key-level TTL，`expires_at` 可以为 `null`。
- 可选实现后台 janitor 线程维护 `{key: expire_time}` 并定期 delete。
- 第一版可以只提供 best-effort delete，不承诺严格 TTL。

建议启动配置：

```text
--routed-experts-store-dsn mooncake://localhost:17913?metadata_server=http://127.0.0.1:8080/metadata&master_server=127.0.0.1:50051&global_segment_size=4gb&local_buffer_size=16mb&protocol=tcp&device=&prefix=sglang:routed_experts&replica_num=1
```

也可以通过环境变量提供 Mooncake 初始化所需字段，并在 DSN 中只覆盖需要变化的部分：

```text
MOONCAKE_MASTER=127.0.0.1:50051
MOONCAKE_TE_META_DATA_SERVER=http://127.0.0.1:8080/metadata
MOONCAKE_GLOBAL_SEGMENT_SIZE=4gb
MOONCAKE_PROTOCOL=tcp
MOONCAKE_DEVICE=
--routed-experts-store-dsn mooncake://localhost:17913?prefix=sglang:routed_experts
```

当前 simple path 支持的 Mooncake query keys：`local_hostname`/`client_hostname`、`metadata_server`/`metadata`、`master_server`/`master_server_addr`/`master_server_address`、`global_segment_size`、`local_buffer_size`、`protocol`、`device`/`device_name`/`rdma_devices`、`prefix`、`replica_num`、`enable_ssd_offload`、`ssd_offload_path`。DSN host 会在未显式设置 `local_hostname` 时作为 Mooncake local hostname。

## Server Arguments

新增一个启动参数：

```text
--routed-experts-store-dsn DSN
```

默认值：

```python
routed_experts_store_dsn = None
```

行为：

- `None` 或空字符串：使用 `InlineStore`，当前 base64 行为。
- `dummy://`：创建 `DummyStore`，用于 benchmark，丢弃 routed experts payload 并返回小 marker object。
- `inline://`：显式使用 `InlineStore`，当前 base64 行为。
- `redis://...`：创建 `RedisRoutedExpertsStore`，返回 remote ref object。
- `mooncake://...`：创建 `MooncakeRoutedExpertsStore`，返回 remote ref object。
- 未知 scheme：server 启动失败，避免请求期才发现配置错误。

DSN query 参数建议：

- `namespace`: key 前缀，默认 `sglang`。
- `ttl`: key TTL 秒数；不传表示 backend 默认策略。
- `fallback`: `none` 或 `inline`，默认 `none`。
- `encoding`: `raw` 或未来的 `zstd`，默认 `raw`。
- `timeout_ms`: remote put/get 超时。

示例 factory：

```python
def create_routed_experts_store(dsn: Optional[str]) -> RoutedExpertsStore:
    if not dsn:
        return InlineStore()

    parsed = urllib.parse.urlparse(dsn)
    if parsed.scheme == "dummy":
        return DummyStore.from_dsn(parsed)
    if parsed.scheme == "inline":
        return InlineStore.from_dsn(parsed)
    if parsed.scheme == "redis":
        return RedisRoutedExpertsStore.from_dsn(parsed)
    if parsed.scheme == "mooncake":
        return MooncakeRoutedExpertsStore.from_dsn(parsed)
    raise ValueError(f"Unsupported routed experts store DSN scheme: {parsed.scheme}")
```

## Integration Points

### TokenizerManager

初始化：

- 在 `TokenizerManager.__init__` 中根据 `server_args.routed_experts_store_dsn` 创建 `self.routed_experts_store`。
- 对 multi-tokenizer worker，每个 worker 都可以持有 store client；key 全局唯一。

输出：

把现有逻辑：

```python
meta_info["routed_experts"] = base64(tensor.numpy().tobytes())
```

替换为：

```python
ref = self.routed_experts_store.put(
    tensor_to_payload(
        routed_experts_tensor,
        request_id=recv_obj.rids[i],
        ttl_s=self.routed_experts_store.ttl_s,
    )
)
meta_info["routed_experts"] = ref.to_response()
```

错误处理：

- store 写失败且 fallback 为 `none`: 当前请求返回 error，避免响应里出现不可用 key。
- store 写失败且 fallback 为 `inline`: 记录 warning，并返回 base64。

### OpenAI EntryPoints

`process_routed_experts_from_ret()` 不需要关心 inline/remote，只透传 `meta_info["routed_experts"]`。

协议类型需要从 `Optional[str]` 改为：

```python
RoutedExpertsResponse = Union[str, RoutedExpertsRefResponse]
```

涉及：

- `SglExt.routed_experts`
- streaming chunk `sglext.routed_experts`
- completions/chat 两条路径

### Native `/generate`

原生返回里的 `meta_info["routed_experts"]` 同样是 string 或 object。默认启动配置下仍是 string，remote DSN 下才是 object。文档需要说明客户端应通过类型判断。

### Router Replay 后续扩展

未来请求侧支持：

```json
{
  "routed_experts": {
    "format": "remote_ref",
    "backend": "redis",
    "key": "...",
    "dtype": "int32",
    "shape": [1024, 48, 8]
  }
}
```

TokenizerManager 或 Scheduler 在校验 replay 前调用 store `get()`，反序列化为 CPU int32 tensor，复用现有 replay 校验逻辑。

## Streaming Semantics

当前 streaming 模式最终会额外发送一个 `sglext.routed_experts` chunk。remote store 仍保持这个语义：

- 不在每个 token chunk 返回 routed experts。
- 请求结束并完成 store `put()` 后，发送一个 extension chunk。
- 如果 store 写入失败，按 fallback 或 error 处理。

这避免客户端收到 key 后远端数据仍不可读。

## Failure Modes

| 场景 | 默认行为 |
| --- | --- |
| 未配置 DSN | 使用 inline，返回 base64 string |
| remote backend 初始化失败 | server 启动失败 |
| put 超时/失败 | 请求失败 |
| DSN `fallback=inline` 且 put 失败 | 返回 base64 string，记录 warning |
| payload 为空 | 返回空 shape ref 或不返回字段；建议不返回字段 |
| key 冲突 | 使用 uuid 避免；如果 SET NX 失败则重试 |
| TTL 到期后客户端读取 | store `get()` 返回 not found，客户端需要重新请求 |
| Mooncake 无 TTL | `expires_at=null`，由外部 store 策略清理 |

## Security and Isolation

- key 不包含 prompt 文本、用户 ID 或其他敏感内容。
- namespace 必须可配置，便于多服务共享 Redis/Mooncake 时隔离。
- 如果后续对外暴露 read API，必须加鉴权；第一版只返回 key，不新增 HTTP read endpoint。
- Redis URL / Mooncake 配置不应出现在响应体。

## Metrics and Logs

建议新增指标：

- `sglang:routed_experts_store_put_total{backend,status}`
- `sglang:routed_experts_store_put_latency_seconds{backend}`
- `sglang:routed_experts_store_payload_bytes{backend}`
- `sglang:routed_experts_store_fallback_total{backend}`

日志：

- 初始化时打印 DSN scheme、backend、namespace、ttl，不打印完整 DSN 中的密码。
- payload 超过阈值时 debug 或 warning。
- put 失败时 warning/error，包含 key、byte_size、backend，不打印数据内容。

## Test Plan

### Unit Tests

- `InlineStore.put()` 返回现有 base64，decode 后 bytes 与 tensor 一致。
- `create_routed_experts_store(None)` 和 `create_routed_experts_store("inline://")` 都返回 `InlineStore`。
- `DummyStore.put()` 不读取 tensor bytes，返回小 marker object。
- factory 根据 DSN scheme 创建 Dummy/Redis/Mooncake backend，并解析 `namespace`、`ttl`、`fallback`。
- `RedisRoutedExpertsStore` 使用 fake redis client 验证 `SET EX`、`GET`、`DEL`。
- `MooncakeRoutedExpertsStore` 使用 fake Mooncake client 验证 `batch_put_from` 参数：key、ptr、size。
- Mooncake device-direct 使用 fake Mooncake client 验证 device part manifest、part key、ptr、size 和 rank metadata。
- `tensor_to_payload()` 校验 dtype、shape、contiguous、byte_size。
- `RoutedExpertsRef.to_response()` 对 inline 返回 string，对 remote 返回 object。

### Integration Tests

- 启动 server 默认参数，`return_routed_experts=True` 仍返回 base64 string。
- 启动 `--routed-experts-store-dsn dummy://`，`return_routed_experts=True` 返回小 marker object，响应体大小不随 routed experts payload 增长。
- 启动 `--routed-experts-store-dsn inline://`，`return_routed_experts=True` 仍返回 base64 string。
- 启动 `--routed-experts-store-dsn redis://host:6379/0?ttl=3600&namespace=sglang`，返回 remote ref object，Redis 中 key 存在，bytes 可还原为同 shape int32 tensor。
- 启动 Mooncake device-direct 且 `pp_size == 1`，返回 `partitioned_manifest`，manifest parts 覆盖完整 token/layer 范围。
- streaming 模式最后一个 extension chunk 返回 remote ref，且 chunk 到达时 key 已可读。
- store 写失败时：
  - fallback none: 请求失败。
  - fallback inline: 请求成功且返回 base64。

### Performance Checks

- 先跑 dummy backend 对照实验：
  - baseline: server 不启用 routed experts，请求不传 `return_routed_experts`。
  - allocation-only: server 启用 routed experts，请求不传 `return_routed_experts`。
  - dummy: server 使用 `dummy://`，请求传 `return_routed_experts=true`。
  - inline: 默认 inline base64，请求传 `return_routed_experts=true`。
- 比较 baseline、allocation-only、dummy、inline 的吞吐、TTFT、ITL、E2E latency 和 TokenizerManager queue latency。
- 对长 prompt 比较 inline base64 与 remote ref 的响应体大小。
- 统计 TokenizerManager CPU 时间，确认 base64 encode 成本下降。
- Redis backend 压测 payload 大小阈值，给出 Mooncake 推荐阈值。

## Rollout Plan

1. 引入接口、InlineStore、serde 和单测，不改变默认行为。
2. 引入 DSN factory、`DummyStore` 和 `--routed-experts-store-dsn`，仍默认 inline。
3. 接入 TokenizerManager，仍默认返回 base64 string。
4. 先跑 dummy backend benchmark，判断主要瓶颈在 capture/D2H/跨进程传递还是 base64/response。
5. 实现 Redis backend 和 fake-client 单测。
6. 实现 Mooncake host-export backend，优先复用现有 Mooncake config。
7. 实现 Mooncake device-direct export，v1 可先限制 `pp_size == 1` 并返回 `partitioned_manifest`。
8. 补 PP metadata 汇聚，支持多 PP stage 的 manifest 聚合。
9. 更新 OpenAI protocol 类型和 docs。
10. 后续 replay 支持 remote ref 输入。

## Open Questions

- Mooncake 是否需要严格 TTL；如果需要，应该由 Mooncake 服务端支持还是 SGLang 本地 janitor 维护。
- 是否需要 checksum；如果 replay 使用 remote ref，建议加 checksum。
- PP metadata 应复用现有 PP 输出链路还是引入专门的 rank-local manifest aggregator。
- 是否需要跨 DP rank 读取同一个 key；如果需要，Mooncake/Redis client 初始化应在所有 tokenizer workers 上保持一致 namespace。
