# 004 catch up with sglang

## Insight

```insight
这次需要把 WeLM 侧依赖的 SGLang 版本追到更新的 upstream 版本，核心原因是新版本
已经支持通过命令行参数开启 MFU 相关指标上报：

--enable-mfu-metrics

配合已有的 --enable-metrics 后，SGLang 会把每张 GPU 的估算 FLOPs、读带宽、写带宽
暴露成 metrics。这样我们不再只能看 tokens/sec 或端到端 latency，而是可以直接用同
一套服务启动参数去观察 prefill 和 decode 阶段各自的模型计算利用率变化。
```

## Vision

把版本更新本身当成一次性能观测能力的补齐：目标不是为了追新特性而升级，而是让
WeLM 的后续性能调优可以多一个统一、可复现的指标入口。

升级后，我们希望能做到：

- 在服务启动时用标准参数打开 MFU 相关 metrics
- 用相同模型、相同并行配置分别压测 prefill-dominated 与 decode-dominated workload
- 在每轮 kernel / attention / MoE / scheduler 调优后，用 MFU 辅助判断瓶颈是在算力、
  带宽，还是调度与空泡
- 把 tokens/sec、latency、显存、MFU 相关指标放在一起看，而不是只靠单一吞吐数字做
  判断

## Background

之前的性能文档主要围绕具体代码路径做优化，例如 QKV projection cleanup、fused MoE
Triton tuning 等。这些优化最终都会体现在吞吐上，但单看吞吐很难回答几个问题：

1. **prefill 是否已经接近算力上限**：长 prompt 场景通常更偏矩阵计算，如果 MFU 偏低，
   需要继续检查 kernel 选择、shape、batch packing 或并行配置。
2. **decode 是否被小 batch / memory traffic / scheduler overhead 限制**：decode 每步
   token 数少，tokens/sec 下降不一定代表算力没有用满，也可能是访存或调度成为瓶颈。
3. **一次优化到底提升了什么**：同样的 tokens/sec 提升，可能来自更高的 GPU FLOPs，
   也可能来自更低的调度空泡或更好的 cache 命中。

SGLang 新版本加入的 `--enable-mfu-metrics` 正好补齐了这个观测缺口。它会在 metrics
中暴露 per-GPU 的估算 FLOPs 和 memory bytes counter，后续可以在 Prometheus/Grafana
中用 `rate(...)` 计算一段时间窗口内的 TFLOPS、带宽，并进一步和硬件理论峰值对比得
到 MFU 相关判断。

## Scope

这次 catch up 的主要范围是：

- **更新 SGLang 版本**：追到包含 `--enable-mfu-metrics` 的版本。
- **保留 WeLM 现有能力**：确保 WeLM 模型加载、推理、speculative / MTP、MoE tuning
  等已有路径不因为版本更新回退。
- **启用性能观测入口**：服务启动参数中可以同时打开 `--enable-metrics` 与
  `--enable-mfu-metrics`。
- **支撑 prefill / decode MFU 测试**：后续压测脚本或手工 benchmark 可以用同一套
  metrics 分别观察两个阶段。

不在本轮范围内的事情：

- 不把 MFU 作为唯一性能目标；最终仍然需要结合业务 workload 下的吞吐和 latency。
- 不在这次更新里重写性能测试框架；本轮先补齐 SGLang runtime 上报能力。
- 不把 SGLang 的估算指标等同于 Nsight / GPU profiler 的硬件 counter；它更适合做
  线上趋势分析和版本间对比。

## Runtime Behavior

升级后，启动服务时可以显式打开 metrics：

```bash
python -m sglang.launch_server \
  --model-path <welm_model_path> \
  --tp-size <tp_size> \
  --enable-metrics \
  --enable-mfu-metrics
```

其中：

- `--enable-metrics`：打开 SGLang metrics 暴露。
- `--enable-mfu-metrics`：额外打开 MFU 相关估算 counter。

开启后可以关注以下指标：

- `sglang:estimated_flops_per_gpu_total`
- `sglang:estimated_read_bytes_per_gpu_total`
- `sglang:estimated_write_bytes_per_gpu_total`

这些指标是累计 counter，使用时应按时间窗口取 rate，例如：

```promql
rate(sglang:estimated_flops_per_gpu_total[1m]) / 1e12
```

得到的是每张 GPU 的估算 TFLOPS。再除以对应硬件的理论峰值，就可以得到一个用于
对比的 MFU 近似值。

## How to Use It for Prefill / Decode

### 1. Prefill-dominated 测试

目标是让大部分计算发生在 prompt prefill 阶段。常见做法：

- 使用较长 input length
- output length 设得很短，例如 1 或少量 token
- 保持 batch / concurrency 与线上目标场景一致

此时观察 MFU 可以帮助判断：长上下文输入下，attention / GEMM / MoE 是否能把 GPU
算力吃起来。

### 2. Decode-dominated 测试

目标是让大部分时间花在逐 token decode 阶段。常见做法：

- 使用较短 input length
- output length 设得较长
- 控制并发，让 decode batch 形态接近线上服务

此时 MFU 和 memory bytes rate 一起看更有价值：decode 可能不是纯算力瓶颈，低 MFU
不一定意味着 kernel 算得慢，也可能说明调度、KV cache 访存、小 batch shape 或通信
成为主要限制。

## Why This Matters

这次更新 SGLang 版本的直接收益是：**WeLM 性能调优从“只看结果指标”前进到“能看
资源利用率指标”**。

后续每次做类似优化时，都可以多回答一个问题：

- tokens/sec 提升了，MFU 是否也提升？
- latency 降了，是因为 GPU 算得更满，还是调度空泡减少？
- prefill 和 decode 的瓶颈是不是同一个？
- MoE / attention / speculative decoding 的改动是否真的改善了目标阶段？

因此，这次 catch up 不是一次单纯的依赖升级，而是给后续 WeLM 性能优化建立统一的
观测基线。

## Current Status

当前结论：需要更新到支持 `--enable-mfu-metrics` 的 SGLang 版本，并在 WeLM 性能压测
中默认保留打开该参数的能力。

推荐后续性能报告至少同时记录：

- prefill / decode 的 tokens/sec
- TTFT / TPOT 或对应 latency 指标
- per-GPU estimated TFLOPS
- estimated read / write bandwidth
- 基于硬件理论峰值换算出的 MFU 近似值

这样才能更稳定地区分“吞吐变了”和“GPU 利用方式变了”。
