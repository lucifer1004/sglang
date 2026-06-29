---
name: sglang-pd-benchmark
description: Run SGLang PD disaggregation benchmarks without real cross-server transfer for prefill-only and decode-only performance with CacheHit=None and no MTP. Use when Codex needs to benchmark TTFT, TPOT, decode first-token latency, tokens/s/gpu, offline MFU, and SGLang-estimated MFU for a model under a user-specified TP/EP parallel configuration and produce the standard CSV result table.
---

# SGLang PD Benchmark

Use this skill to run SGLang PD performance benchmarks without real cross-server transfer and produce the standard result CSV table. Keep the scope narrow:

- Benchmark PD prefill-only with `--disaggregation-mode prefill` and `--disaggregation-transfer-backend mooncake`. Current SGLang prefill servers do not support the `fake` transfer backend.
- Benchmark PD decode-only with `--disaggregation-mode decode` and `--disaggregation-transfer-backend fake`.
- Add dummy bootstrap information in EvalScope `--extra-args`, such as `"bootstrap_host":"2.2.2.2"` and `"bootstrap_room":0`, so prefill-only mooncake runs do not trigger real PD transfer.
- Benchmark prefill-only and decode-only separately.
- Use CacheHit `None` only: disable radix cache on the server and set evalscope prefix length to `0`.
- Do not enable MTP. Treat MTP as a separate future benchmark.
- Use evalscope random dataset.
- Produce one CSV table with the header in this skill.
- Always compute both MFU columns. Do not publish a final table with blank MFU columns.

## Inputs

Require only:

- `model_path`
- Parallel configuration, such as `TP4`, `TP4 EP4`, or `TP2`

Accept optional overrides:

- Prompt lengths
- Request concurrencies
- Output CSV path
- Extra server args
- Single-GPU peak TFLOPS for MFU
- Served model name
- Python path, SGLang repo path, or GPU list

Do not ask upfront for served model name, Python path, SGLang repo path, GPU allocation, common server args, or peak TFLOPS. Try defaults and auto-detection first, and ask only if detection fails.

## Defaults

Use these defaults unless the user specifies otherwise:

```text
dataset = random
number = 200
min_tokens = max_tokens = 100
temperature = 0.7
prefix_length = 0
tokenize_prompt = true
extra_args = {"ignore_eos":true,"bootstrap_host":"2.2.2.2","bootstrap_room":0}
CacheHit = None
MTP配置 = none
attention_backend = fa3
enable_mfu_metrics = true
```

Default prompt lengths:

```text
[1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
```

Default prefill concurrencies:

```text
For 1k/2k/4k: [4, 8, 16, 32, 48, 64, 96, 128]
For 8k/16k/32k/64k/128k: [1, 2, 3, 4, 8, 16, 32]
```

Default decode concurrencies:

```text
[4, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 144, 176, 192]
```

If the user supplies prompt lengths or concurrencies, run exactly those values unless the user asks for automatic exploration.

For automatic exploration, stop adding larger concurrencies for a prompt length when throughput reaches at least 95% of the current maximum and the next larger concurrency improves throughput by less than 5%. If KV cache usage is high and first-token latency rises sharply, record the point but do not treat queueing-heavy points as a better platform point.

## Auto-Detection

Use conservative auto-detection before asking the user.

Python:

1. Use the current Python if it can import `sglang`.
2. Else try `python`.
3. If neither can import SGLang or run `python -m sglang.launch_server --help`, ask the user for a Python or venv path.

SGLang repo:

1. Use the SGLang installed in the selected Python environment by default.
2. Do not set `PYTHONPATH` by default.
3. If installed SGLang is missing or incompatible, ask the user for an SGLang repo path.
4. If a repo is provided, set `PYTHONPATH=${SGLANG_REPO}/python`.

Served model name:

1. Use the user-provided value if present.
2. Else use the basename of `model_path`.
3. If the model has a known local convention in the conversation, use that convention.

GPU allocation:

1. Use `nvidia-smi` to list GPUs and compute processes.
2. Prefer all GPUs without existing compute processes.
3. Convert the user parallel configuration to GPUs per server:
   - `TP4` -> 4 GPUs
   - `TP4 EP4` -> 4 GPUs
   - `TP2` -> 2 GPUs
4. Split available GPUs into as many same-size server groups as possible.
5. Run multiple servers in parallel to cover cases faster.
6. If there are not enough GPUs for one server, ask the user.
7. Do not kill unrelated existing processes unless explicitly instructed.

Peak TFLOPS for MFU:

1. Use the user-provided `peak_tflops_per_gpu` if present.
2. Else use `SGLANG_MFU_PEAK_TFLOPS_PER_GPU` if set.
3. Else inspect `nvidia-smi --query-gpu=name --format=csv,noheader`.
4. If the GPU model and precision peak are known locally, use that value and record it in the run notes.
5. If peak TFLOPS is unknown, ask the user before running the benchmark. MFU cannot be computed without this denominator.

Ports:

1. Probe local ports before use.
2. Assign one port per server.
3. Prefer `127.0.0.1`.

Parallel args:

```text
TP<N>       -> --tp-size <N>
TP<N> EP<M> -> --tp-size <N> --ep-size <M>
```

## Server Commands

Unset proxy variables for local serving and requests.

Prefill-only server template:

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
CUDA_VISIBLE_DEVICES=${GPU_LIST} \
PYTHONPATH=${SGLANG_REPO}/python \
${PYTHON} -m sglang.launch_server \
  --model-path ${MODEL_PATH} \
  --served-model-name ${SERVED_MODEL_NAME} \
  --trust-remote-code \
  ${PARALLEL_ARGS} \
  --attention-backend fa3 \
  --prefill-attention-backend fa3 \
  --decode-attention-backend fa3 \
  --enable-mfu-metrics \
  --disable-radix-cache \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --host 127.0.0.1 \
  --port ${PORT} \
  ${EXTRA_SERVER_ARGS}
```

Decode-only server template:

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
CUDA_VISIBLE_DEVICES=${GPU_LIST} \
PYTHONPATH=${SGLANG_REPO}/python \
${PYTHON} -m sglang.launch_server \
  --model-path ${MODEL_PATH} \
  --served-model-name ${SERVED_MODEL_NAME} \
  --trust-remote-code \
  ${PARALLEL_ARGS} \
  --attention-backend fa3 \
  --prefill-attention-backend fa3 \
  --decode-attention-backend fa3 \
  --enable-mfu-metrics \
  --disable-radix-cache \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend fake \
  --cuda-graph-bs ${CUDA_GRAPH_BS} \
  --host 127.0.0.1 \
  --port ${PORT} \
  ${EXTRA_SERVER_ARGS}
```

If no source repo is used, omit `PYTHONPATH=${SGLANG_REPO}/python`.

For decode, choose `CUDA_GRAPH_BS` so it covers every planned decode concurrency for that server. If a run shows decode batch sizes outside the captured set, restart the decode server with a larger `--cuda-graph-bs` list and rerun affected cases.

## EvalScope Command

Install or verify evalscope before benchmarking:

```bash
${PYTHON} -m pip show evalscope || ${PYTHON} -m pip install evalscope==1.6.1
```

Use this command template for every case:

```bash
evalscope perf \
  --url http://127.0.0.1:${PORT}/v1/completions \
  --model ${SERVED_MODEL_NAME} \
  --api openai \
  --dataset random \
  --parallel ${CONCURRENCY} \
  --number 200 \
  --max-prompt-length ${PROMPT_LEN} \
  --min-prompt-length ${PROMPT_LEN} \
  --max-tokens 100 \
  --min-tokens 100 \
  --prefix-length 0 \
  --tokenize-prompt \
  --temperature 0.7 \
  --extra-args '{"ignore_eos":true,"bootstrap_host":"2.2.2.2","bootstrap_room":0}'
```

Always use `--tokenize-prompt` for formal prefill and decode benchmark runs. This makes EvalScope send tokenized prompts/input IDs, avoids counting server-side tokenizer cost in TTFT, and keeps prompt length exact. Without it, random text prompts can tokenize to lengths different from `PROMPT_LEN` and make TTFT/input throughput incomparable.

Use `curl --noproxy '*'` for local health checks and cache operations.

## MFU Calculation

Always compute MFU using both paths:

- Offline hand calculation: run `${SGLANG_REPO}/benchmark/compute_welmv4_mfu.py` on the result CSV.
- SGLang estimate: start servers with `--enable-mfu-metrics` and parse SGLang server logs.

Offline command after writing or appending benchmark rows:

```bash
${PYTHON} ${SGLANG_REPO}/benchmark/compute_welmv4_mfu.py \
  --csv ${OUTPUT_CSV} \
  --inplace \
  --model-path ${MODEL_PATH} \
  --peak-tflops-per-gpu ${PEAK_TFLOPS_PER_GPU} \
  --decode-output-len 100
```

If the server args include `--enable-welm-kv-mirror-opt`, add `--enable-kv-mirror-opt` to the offline command. If decode output length is not `100`, pass the actual generated token count with `--decode-output-len`.

For SGLang MFU, parse log lines emitted by `--enable-mfu-metrics`:

```text
est. prefill TFLOPS/s (per GPU): <value>
est. decode TFLOPS/s (per GPU): <value>
```

For each benchmark case, first restrict log parsing to the timestamps inside that case's evalscope run. Do not take a blind mean over every matching line in the time window, because warmup, health/probe requests, first-fill batches, and tail drain batches can produce non-representative or inflated instantaneous TFLOPS.

Use these filtering rules before calculating `MFU(%) (SGLang给出)`:

- Parse only positive TFLOPS values from scheduler batch lines.
- For prefill rows, parse `#new-token` from `Prefill batch` lines and keep only full-work batches:

```text
prefill_target_new_tokens = min(chunked_prefill_size, prompt_len * concurrency)
keep line if #new-token >= prefill_target_new_tokens
```

- For decode rows, parse `#running-req` from `Decode batch` lines and keep only steady-state batches:

```text
keep line if #running-req >= request_concurrency
```

- After the stage-specific filter, compute the median TFLOPS. Drop obvious one-off outliers outside `[0.5 * median, 1.5 * median]`. If this leaves fewer than 3 samples, fall back to the stage-filtered samples and note the low sample count.
- If the stage-specific filter captures no samples, fall back to all positive in-window samples only as a last resort, record that fallback in the run notes, and inspect the server log for warmup/tail contamination.
- Save or report the SGLang MFU sample count, mean TFLOPS/s per GPU, and filter/fallback rule used for each row so the MFU value can be audited.

Convert the filtered mean to percent:

```text
MFU(%) (SGLang给出) = 100 * mean(filtered_case_tflops_per_gpu) / PEAK_TFLOPS_PER_GPU
```

If no SGLang MFU line is captured for a case, rerun that case with `--enable-mfu-metrics` and a dedicated server log. Do not leave `MFU(%) (SGLang给出)` empty in the final table.

## Case Workflow

For each benchmark phase:

1. Start enough servers to use the available GPUs efficiently.
2. Wait until each server is ready. `/health` and `/model_info` should respond.
3. Assign pending cases to ready servers.
4. Before each benchmark case, clear the server cache:

```bash
curl --noproxy '*' -sS -X POST http://127.0.0.1:${PORT}/flush_cache
```

5. Run evalscope for the case.
6. Parse TTFT, TPOT, decode first-token latency if available, total throughput, and token counts.
7. Convert throughput to `tokens/s/gpu` if evalscope reports total throughput.
8. Parse and filter SGLang-estimated TFLOPS/s from the server log for this case and fill `MFU(%) (SGLang给出)`.
9. Append one row to the CSV.
10. Re-run `benchmark/compute_welmv4_mfu.py --inplace` so `MFU(%) (离线手算)` is filled for every row.
11. On completion, stop all servers and confirm no SGLang process or GPU compute process remains.

Do not clear cache before every request. Clear cache once before each benchmark case.

## Output CSV

Use this exact header:

```csv
模型ckpt,并行方式,MTP配置,Prompt长度,请求并发度,CacheHit None/L1/L2/L3,Prefill/Decode,Chunked Prefill Size,TTFT_or_TPOT_mean,TTFT_or_TPOT_P50,TTFT_or_TPOT_P90,TTFT_or_TPOT_P99,DecodeFirstTokenLatency_mean,DecodeFirstTokenLatency_P50,DecodeFirstTokenLatency_P90,DecodeFirstTokenLatency_P99,tokens/s/gpu,MFU(%) (离线手算),MFU(%) (SGLang给出)
```

Column rules:

- `模型ckpt`: model path.
- `并行方式`: normalized user parallel configuration, such as `TP4` or `TP4 EP4`.
- `MTP配置`: always `none`.
- `CacheHit None/L1/L2/L3`: always `None`.
- `Prefill/Decode`: `Prefill` for prefill-only rows, `Decode` for decode-only rows.
- `Chunked Prefill Size`: use the server value, usually `8192` unless overridden or disabled.
- `TTFT_or_TPOT_*`:
  - Prefill rows: TTFT mean/P50/P90/P99.
  - Decode rows: TPOT mean/P50/P90/P99.
- `DecodeFirstTokenLatency_*`:
  - Decode rows: first-token latency mean/P50/P90/P99 if available.
  - Prefill rows: leave empty.
- `tokens/s/gpu`: single-GPU throughput, not aggregate throughput.
- `MFU(%) (离线手算)`: always fill using `benchmark/compute_welmv4_mfu.py`.
- `MFU(%) (SGLang给出)`: always fill from SGLang `--enable-mfu-metrics` logs.
- Leave unavailable non-MFU values empty. Do not use `0` for missing metrics.
- If either MFU column cannot be filled, pause and report the blocker instead of producing the final CSV.

## Throughput Normalization

If evalscope reports aggregate generation throughput, compute:

```text
tokens/s/gpu = aggregate_tokens_per_second / GPUs_per_server
```

For prefill, use prompt/input token throughput when reporting prefill throughput. For decode, use generated/output token throughput. Keep the chosen source consistent across all rows and state it in the final summary.

## Failure Handling

Try simple fixes only:

- Local HTTP returns 403: unset proxy variables and use `curl --noproxy '*'`.
- Server is slow to start: inspect `ps` for `nvcc`, `cicc`, or `ptxas`; JIT compile can take minutes.
- Decode performance is unexpectedly low: check whether cuda graph covered the observed batch sizes.
- Prefix cache unexpectedly hits: confirm server has `--disable-radix-cache`, evalscope has `--prefix-length 0`, and `/flush_cache` was called before the case.
- OOM: record the failed case and report it. Do not silently change model path, parallel configuration, prompt length, or concurrency unless the user allows it.
- KV cache queueing: if server logs show high token usage or queueing and first-token latency rises sharply, record the result and call out queueing in the summary.

If a failure cannot be resolved with a small fix, pause and report:

- case parameters
- server command
- evalscope command
- relevant log excerpt
- attempted fix

## Sanity Checks

Before finalizing, inspect the table:

- Prefill TTFT should generally increase with prompt length.
- Prefill long prompts often reach throughput platform at low concurrency; do not assume larger concurrency is always better.
- Decode TPOT should improve or stay near-flat as concurrency increases until the server saturates.
- Decode first-token latency rising sharply while TPOT stays flat usually indicates KV cache pressure or queueing.
- `tokens/s/gpu` must be per GPU; do not mix aggregate and per-GPU throughput in the same table.
- SGLang MFU should come from filtered steady-state batch log lines. Recheck rows where SGLang MFU differs sharply from offline MFU, where sample count is very low, or where a single outlier dominates the mean.
- CacheHit must remain `None` for every row.
- Both MFU columns must be populated; if either is blank, fix the calculation or report the blocker.

In the final response, report the output CSV path, server modes tested, any failed cases, peak TFLOPS used for MFU, and the MFU calculation status.
