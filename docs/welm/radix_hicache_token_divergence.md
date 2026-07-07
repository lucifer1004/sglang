# WeLMv4 + RadixCache + HiCache Token 分叉问题记录

时间：2026-07-07

## 现象

在 WeLMv4 TP4 开启 `SGLANG_ENABLE_UNIFIED_RADIX_TREE=1` 和 CPU HiCache 后，同一条 prompt 在 cold/device-hit/host-hit 三种路径下，host-hit 会出现 token-level 分叉。

这个问题不是 AttnCP 特有问题；普通 TP4 + radix + HiCache 也能复现。Qwen 模型在相同验证方式下没有观察到 token 分叉。

## 关键背景

SGLang 的 prefix cache 不一定命中完整 prompt。以 `input_len=8192, page-size=16` 为例：

```text
max prefix hit = input_len - 1 = 8191
page aligned prefix = floor(8191 / 16) * 16 = 8176

cached prefix = token[0:8176]
recomputed suffix = token[8176:8192]
```

也就是说，本次 forward 只跳过前 8176 个 token，后 16 个 prompt token 仍然需要走 extend/prefill 计算，用来得到最后 prompt token 的 logits。

## 根因

bug 不在“需要重算 suffix”本身，而在重算后的 radix overlap insert 语义不一致。

第一次 cold 请求会把完整 prompt 的 KV 写进 radix tree。后续请求虽然 prefix-hit 只用到 page 对齐前缀，但 radix tree 里仍然可能已经存在 prompt 尾部那段 token 的旧 KV。

重算后插入 radix tree 时：

```text
device-hit 路径：
  发现 overlap 节点已经在 GPU 上，丢弃本次重算出来的重复 KV，继续复用旧 cache KV。

host-hit 路径：
  overlap 节点只在 host 上，旧逻辑用本次重算 KV 恢复这个 evicted node。
```

于是同一段 token 在两条路径下使用了不同来源的 KV：

```text
device-hit -> old cache KV wins
host-hit   -> fresh recomputed KV wins
```

这两个 KV 在数学语义上等价，但 GPU 实际执行路径不同，不保证 bitwise 一致：

```text
cold full prefill:
  长序列 FA prefill kernel，一次处理完整 prompt。

cache-hit extend:
  前缀 KV 来自 cache/host load-back，只计算 suffix query，paged layout、kernel 分块和 reduction order 都不同。
```

WeLMv4 还有 attention sink、hybrid SWA、OE/over-encoding 等路径，这类微小差异更容易在 decode 中放大并改变 greedy top-1 token。Qwen 没复现，主要是对这个差异不敏感，不代表旧 radix/HiCache 语义是正确的。

## 修复原则

radix cache 的一致语义应该是：

```text
如果 radix tree 里这段 token 已经存在：
  无论它当前在 device 还是 host，都应该 old cache KV wins。

只有真正没有命中的 suffix：
  才使用本次新算 KV。
```

因此修复为：host-backed evicted node 在 overlap insert 时，把 host 上已有的 canonical KV copy 回当前请求已经分配好的 fresh device slots，再让 radix node 持有这些 slots。这样当前请求的 `req_to_token` 映射不变，但 KV 内容与 cold/device-hit 路径一致。

对于 page-size=16 + hybrid SWA，还必须同步恢复 SWA sidecar KV。只恢复 FULL KV 不够，因为 SWA layer 会继续使用本次重算出来的 SWA KV，仍然可能分叉。

## 代码改动

- `python/sglang/srt/mem_cache/unified_radix_cache.py`
  - 新增 host-backed overlap insert 恢复逻辑。
  - FULL KV 从 host copy 到 fresh device slots。
  - hybrid SWA 场景下，SWA host_value 也 copy 到 fresh SWA slots。

- `python/sglang/srt/mem_cache/unified_cache_components/full_component.py`
  - 只有 host backup 已完成的 node 才允许作为 host match。
  - host-hit token 数从 `last_host_node` 到 `last_device_node` 统计，避免把未完成或不在 host 上的路径算进去。

- `python/sglang/srt/managers/scheduler.py`
  - 在 overlap scheduler 中，带 prefix-cache hit 的 extend 结果需要先处理，再调度下一轮 decode。
  - 该结果处理可能通过 `cache_unfinished_req` 重写 `req_to_token_pool` 和 radix lock；如果延后处理，下一轮 decode metadata 可能观察到 stale cache mapping。

## 验证结果

以下验证均为真实 host-hit，且 cold/device-hit/host-hit token 完全一致：

```text
page-size=1 + pure Full:
  input=8192, output=128
  host-hit: device=0, host=8191
  passed=true

page-size=16 + hybrid SWA + cuda graph:
  input=8192, output=128
  host-hit: device=0, host=8176
  passed=true

page-size=16 + hybrid SWA + cuda graph:
  input=8192, output=512
  host-hit: device=0, host=8176
  passed=true
```
