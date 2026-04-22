# 002 welm qkv projection cleanup

## Insight

```insight
WeLM 的 QKV projection 不是一个“运行时 if/else 决定行为”的单一抽象，
而是几种数据布局和消费语义不同的投影类型:

1. StandardProjection
2. ImitateQkvKvProjection
3. MirrorQProjection
4. NextnMirrorQProjection(For MTP)

把它们拆成显式类型之后，可以：
- 避免参数重复存储
- 避免 finalize / 全局变量绑定

同时 speculative decoding 通过 request-scope state 读取所需 mirror KV。

```

## Vision

把 WeLM attention 里的 QKV projection 从“共享一套抽象 + 运行时分支”收敛成“构造期分型 + 加载期定型”的实现。

## Background

旧实现里有几个典型问题：

1. **Projection 语义被压在同一个抽象里**：标准 QKV、imitate QKVKV、mirror Q-only 的差异主要靠 forward 里的条件分支表达。
2. **参数容易重复存储**：为支持 mirror / imitate 行为，会额外缓存 self/mirror projection 结果或中间权重。
3. **全局绑定过强**：一些 mirror / imitate 关系需要通过全局 manager 或 post-load finalize 才能成立。
4. **speculative decoding 状态链路不清晰**：draft model 需要读取 target model 产生的 mirror KV，但这份状态原本没有被明确定义成 request-scope 载体。

## Scope

这一轮只处理 WeLM v4 / WeLM v4 NextN 路径中的 QKV projection 重构，不改模型数学语义。

## Implementation Summary

### 1. 将不同 projection 特化成显式类型

当前分支把 WeLM projection 收敛成几类明确类型：

- `StandardQkvProjection`
- `ImitateQkvKvProjection`
- `MirrorQProjection`
- `NextnMirrorQProjection`

这样可以在构造 attention 时直接选择正确类型，而不是在同一个 projection 内维持越来越多的运行时分支。

### 2. 直接加载各自需要的参数布局

`ImitateQkvKvProjection` 改成直接使用 `q / k1 / v1 / k2 / v2` 这种布局加载参数，而不是在 forward 中反复拼接 mirror 权重。

目标是：

- projection 的数据布局在 load_weights 时就固定下来
- forward 只负责执行本类型对应的投影语义

### 3. 避免参数存储两次

重构后尽量去掉了额外的 self/mirror projection 缓存副本。

典型例子：

- `MirrorQProjection` 只保留自己的 `q weight / q bias`
- 不再额外维护一套“从通用 qkv 切片出来的缓存参数”

### 4. 避免全局变量 / 全局 finalize

mirror / imitate 关系不再依赖全局 manager 做后处理绑定，而是改成：

- 构造期确定 projection 类型
- 加载期根据类型与配对关系填充所需 shard
- request-scope 的运行时状态通过 `forward_batch.model_specific_states` 携带

### 5. 让 speculative decoding 能读取 model_specific_states

为了让 draft model 读取 target model 生成的 mirror KV，当前分支把这类状态放进 request-scope 的 `model_specific_states`，并在 WeLM 路径中按需消费。

进一步地，这份 state 已被收缩成 **first draft step only** 的 one-shot 信息：

- target model 写入
- draft model 第一步消费
- 后续 draft steps 不再依赖它

## Runtime Behavior

当前分支还收紧了 projection 的 gating：

- 当 `enable_welm_kv_mirror_opt = False` 时，projection 统一退化到 `StandardQkvProjection`
- 只有在开启 mirror 优化时，才会构造 mirror / imitate projection

## Test Coverage

这轮开发里主要依赖以下验证：

- base smoke test
- base regression / MMLU 精度测试
- RL / online parameter update 测试
- MTP / NextN smoke 与最小参数更新测试

这些测试的目标是确认：

- 重构没有改变 base 模型输出精度
- online weight update 仍然工作
- speculative draft model 仍然能正常启动和服务

## Current Status

当前分支的 QKV projection 已经从“共享抽象 + 运行时分支”收敛到“显式类型 + request-scope state”的实现。

这轮的核心收益不是改变模型算法，而是：

- 降低 projection 路径里的隐藏耦合
- 收缩参数与状态的重复存储
- 让 draft / speculative 链路读取 mirror KV 的方式更明确
