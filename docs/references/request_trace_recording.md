# Request Trace Recording 使用说明

该功能用于把一次外部 HTTP generation 请求记录为一行 JSONL，便于判断异常内容来自模型输出 token，还是来自接口层后处理。

默认关闭。只有设置 `--request-trace-record-dir` 后才会记录。

## 支持的接口

- `/generate`
- `/v1/completions`
- `/v1/chat/completions`
- `/v1/responses`
- `/v1/messages`：Anthropic / Claude-compatible Messages API，Claude Code 使用该接口。

不记录 tokenize、detokenize、embedding、score、count_tokens、管理类接口。

## 启动参数

```bash
python -m sglang.launch_server \
  --model-path /path/to/model \
  --request-trace-record-dir /tmp/sglang_request_traces \
  --request-trace-max-bytes 1073741824 \
  --request-trace-backup-count 5
```

参数含义：

- `--request-trace-record-dir`：JSONL 记录目录。不设置则关闭该功能。
- `--request-trace-max-bytes`：单个压缩 JSONL 文件达到该大小后轮转，默认 `1073741824`。
- `--request-trace-backup-count`：每个进程保留的轮转文件数量，默认 `5`。

文件名格式：

```text
request_trace_<hostname>_<rank>_<pid>.jsonl.gz
request_trace_<hostname>_<rank>_<pid>.jsonl.gz.1
```

## 记录格式

每个外部 HTTP 请求写一行 JSON，文件使用 gzip 压缩。可以用 `gzip.open`、`zcat`、`gunzip -c` 等方式直接读取。

非 streaming 请求：

```json
{
  "schema_version": 1,
  "request_id": "req-xxx",
  "endpoint": "/v1/chat/completions",
  "stream": false,
  "status": "ok",
  "created_at": 1782124680.0,
  "finished_at": 1782124681.0,
  "http_request": {
    "method": "POST",
    "path": "/v1/chat/completions",
    "headers": {
      "authorization": "<redacted>"
    },
    "query": {},
    "body": {}
  },
  "http_response": {},
  "chunks": [],
  "generations": [
    {
      "rid": "gen-xxx",
      "prompt_token_ids": [1, 2, 3],
      "output_token_ids": [4, 5, 6],
      "meta_info": {},
      "finished": true
    }
  ],
  "error": null,
  "model_path": "/path/to/model",
  "tokenizer_path": "/path/to/tokenizer"
}
```

Streaming 请求：

```json
{
  "schema_version": 1,
  "request_id": "req-xxx",
  "endpoint": "/generate",
  "stream": true,
  "status": "ok",
  "http_request": {},
  "http_response": null,
  "chunks": [
    {
      "chunk_index": 0,
      "chunk": "data: {...}\\n\\n"
    },
    {
      "chunk_index": 1,
      "chunk": "data: [DONE]\\n\\n"
    }
  ],
  "generations": [
    {
      "rid": "gen-xxx",
      "prompt_token_ids": [1, 2, 3],
      "output_token_ids": [4, 5, 6],
      "meta_info": {},
      "finished": true
    }
  ],
  "error": null
}
```

## 字段说明

- `schema_version`：记录格式版本。
- `request_id`：trace 记录内部生成的请求 ID。
- `endpoint`：HTTP endpoint。
- `stream`：是否为 streaming 请求。
- `status`：`ok` 或 `error`。
- `created_at` / `finished_at`：记录开始和结束时间，Unix timestamp。
- `http_request`：原始 HTTP 请求信息。敏感 header 会被替换为 `<redacted>`。
- `http_response`：非 streaming 时实际返回给 client 的 HTTP response。
- `chunks`：streaming 时实际返回给 client 的 chunk 列表；非 streaming 为空列表。
- `generations`：该 HTTP 请求内部触发的模型 generation 列表。
- `rid`：SGLang generation request id。
- `prompt_token_ids`：实际送入模型的 prompt token ids。
- `output_token_ids`：SGLang 已 stop-trim 后、交给 serving 层使用的 output token ids。
- `meta_info`：SGLang native output 的 `meta_info`。
- `finished`：该 generation 是否结束。
- `error`：请求异常信息；成功时为 `null`。
- `model_path` / `tokenizer_path`：server 启动时使用的模型和 tokenizer 路径。

## 注意事项

- `output_token_ids` 不是 stop-trim 前的原始 token ids。
- streaming 请求会在 stream 结束后写出一整条 JSONL，`chunks` 中保存实际发送给 client 的每个 chunk。
- 如果请求 body 中设置 `"no_logs": true`，该请求不会被记录。
- 该功能和 OpenTelemetry tracing 不同：这里记录的是 request/response/token ids，用于排查模型输出和接口后处理差异。
