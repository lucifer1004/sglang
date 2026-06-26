---
name: sglang-mtp-decode-benchmark
description: Run SGLang serving benchmarks that compare decode TPOT with WeLMv4 MTP enabled versus disabled. Use when Codex needs to quantify MTP/no-MTP impact on TPOT, throughput, TTFT, latency, speculative accept length, and accept rate by warming prompts into prefix cache/HiCache and then running a single-wave decode benchmark.
---

# SGLang MTP Decode Benchmark

Use this skill to compare decode performance with MTP enabled versus disabled under the same model, prompts, parallelism, and request concurrency. Keep the scope narrow:

- Use normal SGLang serving, not PD disaggregation.
- Measure decode behavior after prompts are warmed into prefix cache/HiCache.
- Compare paired `No MTP` and `WeLMv4 MTP` runs.
- Optimize for TPOT comparison; TTFT, latency, throughput, queueing, and speculative metrics are supporting evidence.
- Do not use random prompts unless the user explicitly asks; MTP accept rate is prompt/content sensitive.

## Hard Rules

- Keep EOS enabled. Do not pass `ignore_eos`, do not force EOS to be ignored, and do not set `min_tokens`.
- Set `number == parallel` for every formal benchmark case.
- Run exactly one formal wave per case. Do not run extra benchmark repetitions to hide JIT effects.
- Use the same prompt sample IDs for MTP and no-MTP at the same concurrency.
- Use the same warmup prompts and formal prompts for a case.
- Do not publish a TPOT comparison if formal requests did not mostly hit prefix cache/HiCache.
- Do not clear cache between warmup and the formal run.

The reason for `number == parallel` is to avoid a second request wave whose prefill can be scheduled while the first wave is decoding, which would drag down decode TPOT and make the MTP/no-MTP comparison ambiguous.

## Inputs

Require:

- `model_path`
- Parallel configuration, such as `TP4` or `TP4 EP4`
- A real prompt dataset or trace-derived prompt source

Accept optional overrides:

- Prompt length, default `16384`
- Decode concurrencies, default `[32, 40, 48, 56, 64, 80]`
- `max_tokens`, default `1000`
- Output directory or CSV paths
- GPU list
- Extra server args
- Served model name
- Python path
- HiCache ratio

Use the current Python environment by default. Use the SGLang installed in that Python environment by default; do not set `PYTHONPATH` unless the user provides a source repo path.

## Dataset Preparation

Prepare one evalscope custom dataset that can provide at least `max(concurrency)` prompts at the target length.

Preferred path:

1. Load the real prompt source.
2. Tokenize with the target model tokenizer.
3. Keep samples with token length at least `prompt_len`.
4. Truncate each selected prompt to exactly `prompt_len` tokens.
5. Do not pad short prompts.
6. Write a custom evalscope dataset that can be sent with `--dataset custom --dataset-path ${DATASET_PATH}`.
7. Use tokenized prompts via `--tokenize-prompt --tokenizer-path ${MODEL_PATH}` when supported, to reduce tokenizer overhead in the measured run.

For each concurrency `P`, select exactly `P` prompts. Keep a stable selection order across MTP and no-MTP. If there are fewer than `P` valid prompts, report the blocker instead of padding.

## Server Setup

Unset proxy variables for local serving and requests. Probe ports before use.

Common server template:

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
CUDA_VISIBLE_DEVICES=${GPU_LIST} \
${PYTHON} -m sglang.launch_server \
  --model-path ${MODEL_PATH} \
  --served-model-name ${SERVED_MODEL_NAME} \
  --trust-remote-code \
  ${PARALLEL_ARGS} \
  --attention-backend fa3 \
  --prefill-attention-backend fa3 \
  --decode-attention-backend fa3 \
  --enable-hierarchical-cache \
  --hicache-ratio ${HICACHE_RATIO} \
  --hicache-size 0 \
  --hicache-io-backend kernel \
  --hicache-mem-layout layer_first \
  --hicache-write-policy write_through \
  --cuda-graph-bs ${CUDA_GRAPH_BS} \
  --host 127.0.0.1 \
  --port ${PORT} \
  ${EXTRA_SERVER_ARGS}
```

For WeLMv4, include model-specific args if they are required by the target checkpoint:

```text
--enable-over-encoding --sampling-defaults openai --enable-welm-kv-mirror-opt
```

Do not set `--disable-radix-cache`; the benchmark depends on prompt cache hits. Choose `CUDA_GRAPH_BS` to cover the initial concurrency and lower batch sizes caused by requests finishing at EOS. A dense list such as `1 2 4 8 12 16 24 32 40 48 56 64 72 80 96 ...` is appropriate when those batch sizes may appear.

MTP server extras:

```bash
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_WELM_MTP_SAMPLE_DRAFT=1
export SGLANG_WELM_MTP_DRAFT_FIXED_TEMPERATURE=1.0
export SGLANG_WELM_MTP_DRAFT_FIXED_TOP_P=0.95
export SGLANG_WELM_MTP_DRAFT_SAMPLING_TOPK=8
```

```text
--speculative-algorithm EAGLE
--speculative-draft-model-path ${MODEL_PATH}
--speculative-num-steps 3
--speculative-eagle-topk 1
--speculative-num-draft-tokens 4
--sampling-backend flashinfer
```

No-MTP server uses only the common server args.

## EvalScope Commands

Verify or install evalscope:

```bash
${PYTHON} -m pip show evalscope || ${PYTHON} -m pip install evalscope==1.6.1
```

Warmup command for each case:

```bash
evalscope perf \
  --url http://127.0.0.1:${PORT}/v1/completions \
  --model ${SERVED_MODEL_NAME} \
  --api openai \
  --dataset custom \
  --dataset-path ${CASE_DATASET_PATH} \
  --tokenize-prompt \
  --tokenizer-path ${MODEL_PATH} \
  --no-test-connection \
  --parallel ${CONCURRENCY} \
  --number ${CONCURRENCY} \
  --max-tokens 1
```

Formal command for each case:

```bash
evalscope perf \
  --url http://127.0.0.1:${PORT}/v1/completions \
  --model ${SERVED_MODEL_NAME} \
  --api openai \
  --dataset custom \
  --dataset-path ${CASE_DATASET_PATH} \
  --tokenize-prompt \
  --tokenizer-path ${MODEL_PATH} \
  --no-test-connection \
  --parallel ${CONCURRENCY} \
  --number ${CONCURRENCY} \
  --max-tokens ${MAX_TOKENS}
```

Do not add `--min-tokens`. Do not add any flag or request parameter that ignores EOS. If the evalscope version defaults to ignoring EOS, find and pass the option that keeps EOS active before benchmarking.

## Case Workflow

For each `mode in [No MTP, WeLMv4 MTP]` and each concurrency:

1. Start the matching server and wait for `/health` and `/model_info`.
2. Call `/flush_cache`.
3. Build `CASE_DATASET_PATH` with exactly `CONCURRENCY` prompts.
4. Run the warmup command with `max_tokens=1`.
5. Do not record warmup performance.
6. Run the formal command once with `max_tokens=${MAX_TOKENS}`.
7. Parse evalscope summary metrics.
8. Parse server logs for cache hit evidence, cuda graph coverage, queueing, running batch size, `accept len`, and `accept rate`.
9. Stop the server when all assigned cases are complete.

Use multiple servers only when enough GPUs are available and each server still reaches its own requested concurrency. Never treat aggregate concurrency across servers as the per-server concurrency.

## Required Validation

For every formal case, verify:

- `number == parallel == CONCURRENCY`.
- EOS was active: no `ignore_eos`, no `min_tokens`, no forced fixed-length decode.
- The maximum formal `#running-req` reaches `CONCURRENCY` near the start of decode.
- Observed decode batch sizes are covered by cuda graph. If not, restart with more `--cuda-graph-bs` values and rerun the case.
- Prompt cache hit is high after warmup. If not, increase `--hicache-ratio`, confirm radix cache is enabled, and rerun the case.
- Queueing is recorded. If queueing is heavy, keep the row but flag it in `notes`.

If validation fails and cannot be fixed with a small retry, pause and report the failed mode, concurrency, server command, evalscope command, and relevant log excerpt.

## Metrics To Extract

From evalscope:

- Total requests, success, failed, success rate
- Average output tokens and P99 output tokens
- Output token throughput
- Request throughput
- Latency mean/P50/P90/P99
- TTFT mean/P50/P90/P99
- TPOT mean/P50/P90/P99
- Approx speculative decoding acceptance rate, when present

From SGLang server logs:

- Max formal `#running-req`
- Max formal `#queue-req`
- Whether cuda graph was used
- Prefix/cache hit evidence
- MTP `accept len`
- MTP `accept rate`

Compute:

```text
tokens/s/gpu = output_token_throughput / GPUs_per_server
TPOT_improvement_ratio = NoMTP_TPOT_mean / MTP_TPOT_mean
TPOT_reduction = (NoMTP_TPOT_mean - MTP_TPOT_mean) / NoMTP_TPOT_mean
throughput_ratio = MTP_tokens/s/gpu / NoMTP_tokens/s/gpu
```

## Output Files

Produce a raw CSV with one row per `(mode, concurrency)`:

```csv
模型ckpt,并行方式,MTP配置,Prompt长度,请求并发度,max_tokens,number,success,failed,success_rate,avg_output_tokens,p99_output_tokens,output_tokens/s,tokens/s/gpu,request_throughput(req/s),latency_mean_s,latency_p50_s,latency_p90_s,latency_p99_s,TTFT_mean_s,TTFT_p50_s,TTFT_p90_s,TTFT_p99_s,TPOT_mean_s,TPOT_p50_s,TPOT_p90_s,TPOT_p99_s,decoded_tokens_per_iter,spec_accept_rate,max_formal_queue_req,max_formal_running_req,cuda_graph_covered,cache_hit_verified,notes
```

Produce a comparison CSV with one row per concurrency:

```csv
模型ckpt,并行方式,Prompt长度,请求并发度,max_tokens,NoMTP_TPOT_mean_s,MTP_TPOT_mean_s,TPOT改善倍数,TPOT耗时降低比例,NoMTP_tokens/s/gpu,MTP_tokens/s/gpu,吞吐提升倍数,NoMTP_TTFT_mean_s,MTP_TTFT_mean_s,NoMTP_latency_mean_s,MTP_latency_mean_s,MTP_decoded_tokens_per_iter,MTP_spec_accept_rate,NoMTP_max_queue,MTP_max_queue,notes
```

Also write a short Markdown summary with:

- Benchmark setup
- Dataset and prompt length
- Confirmation that EOS was active
- Confirmation that `number == parallel`
- Main comparison table
- Validation warnings, especially queueing, cache misses, or cuda graph misses

## Interpretation

- If TPOT improves but TTFT increases, inspect HiCache transfer/cache lookup and queueing before blaming MTP.
- If MTP accept length is close to `1`, small TPOT gains are expected.
- If accept rate is measured on forced post-EOS tokens, discard the case; EOS must remain active.
- If throughput and TPOT move in opposite directions, check output length distribution, request failures, queueing, and batch size coverage.
- If high concurrency triggers queueing, the row is still useful but should not be used as the cleanest TPOT comparison point.
