---
name: welmv4-regression
description: Run the local WeLM v4 80A3 regression matrix using ~/scripts from perf_optimize_scripts.git.
---

# WeLM v4 Regression

Use this skill when validating WeLM v4 changes on the local 80A3 model. The
test scripts live in `/home/josephyu/scripts`, cloned from
`https://git.woa.com/wxg-odyssey/serving/perf_optimize_scripts.git`.

## Prerequisites

- Run from `/home/josephyu/sglang` on the target branch.
- Use `/home/josephyu/sglang/.venv`; it should be an editable install of
  `/home/josephyu/sglang/python`.
- Use model path `/home/josephyu/models`.
- Export the dataset proxy when loading MMLU:
  `http://mmsprwelmposttrainhttpproxy.polaris:11113`.
- Keep localhost traffic off proxies with
  `NO_PROXY=127.0.0.1,localhost` and `no_proxy=127.0.0.1,localhost`.
- Stop any old servers before starting:
  `bash ~/scripts/stop_serve.sh || true`,
  `bash ~/scripts/stop_welmv4_80a3_attndp_ep.sh || true`,
  `bash ~/scripts/stop_welmv4_pp.sh || true`.

## Matrix

Run all four rows before declaring the branch fully regressed:

1. `start_serve.sh`, TP=4
   - smoke test
   - dirty-prefix test
   - regression test

2. AttnDP=4 + TP=4
   - smoke test
   - dirty-prefix test
   - regression test

3. TP=4 + PP=2
   - smoke test
   - dirty-prefix test
   - regression test

4. TP=4 high-concurrency return routed experts
   - enable KV-mirror optimization
   - capture base64 `return_routed_experts`
   - capture Redis `return_routed_experts`
   - require Redis bytes to match base64 bytes exactly

## Commands

### TP=4 Base

Hold a 4-GPU lease for the whole server lifetime:

```bash
gpu-lease run --count 4 --wait -- bash -lc '
  set -euo pipefail
  cd /home/josephyu/scripts
  export SGLANG_DIR=/home/josephyu/sglang
  export MODEL_PATH=/home/josephyu/models
  export HOST=127.0.0.1
  export PORT=18081
  export TP_SIZE=4
  export EXTRA_SERVER_ARGS="--enforce-disable-flashinfer-allreduce-fusion --skip-server-warmup"
  export LOG_FILE=/tmp/welmv4_tp4_serve.log
  export PID_FILE=/tmp/welmv4_tp4_serve.pid
  export NO_PROXY=127.0.0.1,localhost
  export no_proxy=127.0.0.1,localhost
  export http_proxy=http://mmsprwelmposttrainhttpproxy.polaris:11113
  export https_proxy=http://mmsprwelmposttrainhttpproxy.polaris:11113
  trap "PID_FILE=/tmp/welmv4_tp4_serve.pid bash /home/josephyu/scripts/stop_serve.sh || true" EXIT
  bash ./start_serve.sh
  until curl -fsS --noproxy "*" "http://${HOST}:${PORT}/v1/models" >/dev/null; do sleep 2; done
  bash ./smoke_test.sh "${HOST}" "${PORT}"
  /home/josephyu/.local/bin/uv run --active -p /home/josephyu/sglang/.venv \
    python ./dirty_prefix_test.py --server-url "http://${HOST}:${PORT}" --model welmv4 --use-known-bad-seeds
  /home/josephyu/.local/bin/uv run --active -p /home/josephyu/sglang/.venv \
    python ./regression_test.py test --server-url "http://${HOST}:${PORT}" --model welmv4
'
```

### AttnDP=4 + TP=4

```bash
gpu-lease run --count 4 --wait -- bash -lc '
  set -euo pipefail
  cd /home/josephyu/scripts
  export SGLANG_DIR=/home/josephyu/sglang
  export MODEL_PATH=/home/josephyu/models
  export HOST=127.0.0.1
  export PORT=18083
  export TP_SIZE=4
  export DP_SIZE=4
  export EP_SIZE=4
  export REGRESSION_TOLERANCE=1e-5
  export EXTRA_SERVER_ARGS="--enforce-disable-flashinfer-allreduce-fusion --cuda-graph-max-bs 16"
  export NO_PROXY=127.0.0.1,localhost
  export no_proxy=127.0.0.1,localhost
  export http_proxy=http://mmsprwelmposttrainhttpproxy.polaris:11113
  export https_proxy=http://mmsprwelmposttrainhttpproxy.polaris:11113
  trap "bash /home/josephyu/scripts/stop_welmv4_80a3_attndp_ep.sh || true" EXIT
  bash ./start_welmv4_80a3_attndp_ep.sh
  until curl -fsS --noproxy "*" "http://${HOST}:${PORT}/v1/models" >/dev/null; do sleep 2; done
  bash ./smoke_test.sh "${HOST}" "${PORT}"
  /home/josephyu/.local/bin/uv run --active -p /home/josephyu/sglang/.venv \
    python ./dirty_prefix_test.py --server-url "http://${HOST}:${PORT}" --model welmv4 --use-known-bad-seeds
  WELMV4_REGRESSION_BASELINE_PREFIX=regression_baseline_attndp_ep \
    /home/josephyu/.local/bin/uv run --active -p /home/josephyu/sglang/.venv \
    python ./regression_test.py test --server-url "http://${HOST}:${PORT}" --model welmv4 \
    --tolerance "${REGRESSION_TOLERANCE}" --routed-dp-size "${DP_SIZE}"
'
```

Do not set `DISABLE_CUDA_GRAPH=1` for this row. Keep ordinary CUDA graph
enabled; only piecewise CUDA graph is disabled by `start_welmv4_80a3_attndp_ep.sh`.
Use `--routed-dp-size "${DP_SIZE}"` for regression so each baseline sample is
sent to the same DP rank sequence regardless of earlier smoke or dirty-prefix
requests consuming the server's round-robin counter.
Do not use `run_welmv4_80a3_attndp_ep_tests.sh all` for this matrix row,
because `all` also enables routed-experts replay coverage.

### TP=4 + PP=2

This row uses 8 GPUs, so run it alone:

```bash
gpu-lease run --count 8 --wait -- bash -lc '
  set -euo pipefail
  cd /home/josephyu/scripts
  export SGLANG_DIR=/home/josephyu/sglang
  export MODEL_PATH=/home/josephyu/models
  export HOST=127.0.0.1
  export PORT=18082
  export TP_SIZE=4
  export PP_SIZE=2
  export EXTRA_SERVER_ARGS="--enforce-disable-flashinfer-allreduce-fusion --cuda-graph-max-bs 16"
  export LOG_FILE=/tmp/welmv4_pp_serve.log
  export PID_FILE=/tmp/welmv4_pp_serve.pid
  export NO_PROXY=127.0.0.1,localhost
  export no_proxy=127.0.0.1,localhost
  export http_proxy=http://mmsprwelmposttrainhttpproxy.polaris:11113
  export https_proxy=http://mmsprwelmposttrainhttpproxy.polaris:11113
  trap "PID_FILE=/tmp/welmv4_pp_serve.pid bash /home/josephyu/scripts/stop_welmv4_pp.sh || true" EXIT
  bash ./start_welmv4_pp.sh
  until curl -fsS --noproxy "*" "http://${HOST}:${PORT}/v1/models" >/dev/null; do sleep 2; done
  bash ./smoke_test.sh "${HOST}" "${PORT}"
  /home/josephyu/.local/bin/uv run --active -p /home/josephyu/sglang/.venv \
    python ./dirty_prefix_test.py --server-url "http://${HOST}:${PORT}" --model welmv4 --use-known-bad-seeds
  /home/josephyu/.local/bin/uv run --active -p /home/josephyu/sglang/.venv \
    python ./regression_test.py test --server-url "http://${HOST}:${PORT}" --model welmv4
'
```

`run_welmv4_pp_tests.sh` does not include dirty-prefix coverage, so keep the
server alive and run smoke, dirty-prefix, and regression manually.
Do not set `DISABLE_CUDA_GRAPH=1`; `start_welmv4_pp.sh` keeps ordinary CUDA
graph enabled and disables only piecewise CUDA graph by default.

### Routed Experts Base64 vs Redis

This script starts one Redis-backed server and one base64 server sequentially,
then checks byte equality:

```bash
env NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  /home/josephyu/sglang/.venv/bin/python /home/josephyu/scripts/test_routed_experts_redis_backend.py \
  --sglang-dir /home/josephyu/sglang \
  --model-path /home/josephyu/models \
  --model welmv4 \
  --host 127.0.0.1 \
  --redis-server-port 18731 \
  --base64-server-port 18732 \
  --redis-port 19453 \
  --gpu-ids 4,5,6,7 \
  --tp-size 4 \
  --dp-size 1 \
  --ep-size 1 \
  --enable-kv-mirror \
  --cuda-graph-max-bs 16 \
  --extra-server-arg=--enforce-disable-flashinfer-allreduce-fusion \
  --num-requests 16 \
  --concurrency 8 \
  --max-new-tokens 3
```

Expected success line:

```text
PASS: Redis backend bytes match base64 routed_experts bytes exactly for 16 requests at concurrency=8.
```

## Completion Evidence

Record the exact command output for each row:

- smoke: `passed` response from `smoke_test.sh`
- dirty-prefix: no dirty prefix failures
- regression: `Regression test PASSED`
- routed experts: Redis/base64 byte equality PASS line
- final git state: `git status --short --branch`
