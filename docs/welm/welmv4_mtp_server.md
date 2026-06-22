# WeLMV4 MTP Server 使用文档

本文说明如何用 SGLang server 启动带 WeLMV4 MTP 的模型。

## 适用模型

MTP serving 需要使用真正带 MTP/NextN 权重的 WeLMV4 checkpoint。模型配置需要满足：

- 主模型是 WeLMV4 MoE 模型。
- `config.json` 中 `num_nextn_predict_layers > 0`。

## MTP 参数

| 参数 | 含义 |
| --- | --- |
| `--speculative-algorithm NEXTN` | 开启 WeLMV4 MTP/NextN 推理。固定使用 `NEXTN` |
| `--speculative-draft-model-path` | MTP draft 权重路径。通常和 `--model` 使用同一个带 MTP 权重的 checkpoint。建议同 `--model` |
| `--speculative-num-steps` | draft 向前预测的深度。 |
| `--speculative-eagle-topk` | 每一步保留的候选分支数。`1` 是主推荐配置；`2` 可用于更宽的树，但吞吐不一定更高。 |
| `--speculative-num-draft-tokens` | 每轮 verify 的 draft token 数量上限。 |

## 相关 server 参数

| 参数 | 含义 |
| --- | --- |
| `--sampling-backend` | 选择采样后端，可选 `flashinfer`、`pytorch`、`ascend`。不设置时会自动选择：如果 FlashInfer 可用则使用 `flashinfer`，否则使用 `pytorch`。CUDA 上建议显式设置 `--sampling-backend flashinfer`；排障或 FlashInfer 不可用时再切到 `pytorch`。 |
| `--cuda-graph-max-bs` | CUDA graph 捕获的最大 batch size。未设置 `--cuda-graph-bs` 时，SGLang 会根据该值自动生成一组 capture batch sizes。值需要覆盖预期 decode 并发；越大显存占用越高。80A3 MTP c20 压测中使用 `20`。 |
| `--cuda-graph-bs` | 手动指定 CUDA graph 捕获的 batch size 列表，例如 `--cuda-graph-bs 1 2 4 8 16 20`。设置后会覆盖自动生成逻辑，并且 `cuda_graph_max_bs` 会取列表最大值。只在需要精确控制捕获列表、减少 graph 显存或对齐固定并发时使用。 |

## MTP 相关环境变量

| 环境变量 | 默认值 | 含义和限制 |
| --- | --- | --- |
| `SGLANG_ENABLE_SPEC_V2` | `1` | WeLMV4 MTP 必须使用 Spec V2/overlap schedule。建议启动命令里显式设置为 `1`；不要设为 `0`，也不要传 `--disable-overlap-schedule`。 |
| `SGLANG_WELM_MTP_SAMPLE_DRAFT` | `0` | 开启 draft sampling。只支持 `--speculative-eagle-topk 1`；`topk > 1` 时设置为 `1` 会直接启动失败。|
| `SGLANG_WELM_MTP_DRAFT_FIXED_TEMPERATURE` | 未设置 | draft sampling 使用固定 temperature。只在 `SGLANG_WELM_MTP_SAMPLE_DRAFT=1` 时有意义；值必须大于 `0`。 |
| `SGLANG_WELM_MTP_DRAFT_FIXED_TOP_P` | 未设置 | draft sampling 使用固定 top-p。只在 `SGLANG_WELM_MTP_SAMPLE_DRAFT=1` 时有意义；值必须在 `(0, 1]`。 |
| `SGLANG_WELM_MTP_DRAFT_SAMPLING_TOPK` | `0` | 限制 draft sampling 只在前 K 个 token 内采样。`<=0` 表示关闭；值必须小于 vocab size，否则会被忽略。只建议和 `topk=1` 的 draft sampling 一起使用。 |

## 参数限制

必须满足这些限制，否则 server 会拒绝启动或结果没有经过验证：

- 必须启用 Spec V2：设置 `SGLANG_ENABLE_SPEC_V2=1`，并且不要传
  `--disable-overlap-schedule`。
- `--speculative-draft-model-path` 必须指向带 MTP 权重的 WeLMV4 checkpoint。
- 模型 ckpt config 中的 `num_nextn_predict_layers` 必须等于 `1` 或等于 `--speculative-num-steps`。
- 当 `--speculative-eagle-topk 1` 时，
  `--speculative-num-draft-tokens` 必须等于 `--speculative-num-steps + 1`。
  例如 `steps=3` 时必须设置 `draft_tokens=4`。
- 当 `--speculative-eagle-topk > 1` 时，
  `--speculative-num-draft-tokens` 不能超过
  `1 + topk + (steps - 1) * topk * topk`。例如 `steps=3, topk=2`
  时最大是 `11`。
- `topk > 1` 时不支持draft sampling，即不能设置 `SGLANG_WELM_MTP_SAMPLE_DRAFT=1`。
- Attention Backend请固定使用 `fa3`。

## 示例：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
SGLANG_ENABLE_SPEC_V2=1 \
python -m sglang.launch_server \
  --host 127.0.0.1 \
  --port 30002 \
  --model /home/jiayifeng/ckpts/80a3_mtp_3_sft_jfs \
  --served-model-name welmv4 \
  --trust-remote-code \
  --tp 4 \
  --mem-fraction-static 0.72 \
  --attention-backend fa3 \
  --prefill-attention-backend fa3 \
  --decode-attention-backend fa3 \
  --enable-over-encoding \
  --enable-welm-kv-mirror-opt \
  --sampling-defaults openai \
  --sampling-backend flashinfer \
  --disable-radix-cache \
  --cuda-graph-max-bs 20 \
  --speculative-algorithm NEXTN \
  --speculative-draft-model-path /home/jiayifeng/ckpts/80a3_mtp_3_sft_jfs \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4
```

## 请求示例

server 启动后可用 OpenAI compatible API 访问：

```bash
curl http://127.0.0.1:30002/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "welmv4",
    "messages": [
      {"role": "user", "content": "写一段关于多 token prediction 的说明。"}
    ],
    "max_tokens": 256,
    "temperature": 0.7
  }'
```
