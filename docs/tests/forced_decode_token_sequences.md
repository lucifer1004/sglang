# Forced Decode Token Sequences

This test validates the `/generate` HTTP API field `forced_decode_token_ids` for
WeLM v4 80A3 benchmarking.

## Scope

- Model: WeLM v4 80A3
- Parallelism: TP=4, DP=1, EP=1
- API: native `/generate`
- Cache: radix/KV cache enabled, with cache reporting enabled for validation
- Workload: 100 MMLU prompts, concurrency 64, decode 5 tokens

The feature forces the token IDs appended after each forward/sampling step. It
does not skip prefill or decode execution.

## Server Command

```bash
gpu-lease run --count 4 --wait -- bash -lc '
cd /home/josephyu/sglang-returns-router-result
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="localhost,127.0.0.1,::1"
export SGLANG_WELMV4_MMQ_SCORE_ON_SWIGLU=true
export SGLANG_WELMV4_MMQ_MOE_COMBINE=true
export SGLANG_MOE_PADDING=0
exec .venv/bin/python -m sglang.launch_server \
  --model-path /home/josephyu/models \
  --served-model-name welmv4 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --mem-fraction-static 0.8 \
  --prefill-attention-backend fa3 \
  --decode-attention-backend flashinfer \
  --enable-flashinfer-allreduce-fusion \
  --sampling-defaults openai \
  --skip-server-warmup \
  --enable-cache-report \
  --host 127.0.0.1 \
  --port 32200
'
```

Expected server-side conditions:

- `tp_size=4`
- `dp_size=1`
- `ep_size=1`
- `disable_radix_cache=False`
- `enable_cache_report=True`
- `enable_welm_kv_mirror_opt=False`

## Test Command

```bash
cd /home/josephyu/sglang-returns-router-result
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="localhost,127.0.0.1,::1"
export SGLANG_MMLU_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
.venv/bin/python /home/josephyu/scripts/forced_decode_tokens_test.py \
  --server-url http://127.0.0.1:32200 \
  --model-path /home/josephyu/models \
  --num-prompts 100 \
  --concurrency 64 \
  --max-tokens 5 \
  --compare-unforced \
  --timeout 300
```

The script checks:

- every forced response returns exactly the requested `output_ids`
- the second forced pass is cache-hot (`cached_tokens > 0`)
- the throughput comparison uses cache-hot prompts
- `--compare-unforced` first captures per-prompt output IDs from an unforced
  cache-hot round, then forces those same per-prompt token sequences in
  interleaved forced/unforced comparison rounds
- median forced request throughput must be at least `0.98` of median unforced
  request throughput

## Result

Latest local validation:

```text
forced_decode_token_ids=[6006, 17896, 26698, 10903, 305]
requests=100
concurrency=64
decode_tokens=5
forced-pass-2-cache-hot.cached_token_sum=12491
unforced-capture-cache-hot.cached_token_sum=12491
forced_request_throughput_median=195.38124997670874 req/s
unforced_request_throughput_median=194.23374966573 req/s
forced_to_unforced_request_throughput_ratio=1.0059078317385808
```

This passes the default `0.98` throughput-ratio threshold.
