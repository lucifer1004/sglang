"""Compare server outputs with the mk fused decode OE GEMM path on vs off.

This is a manual equivalence test: it boots a sglang server twice (once with
``SGLANG_WELM_OE_FUSED_DECODE_GEMM=0`` and once with ``=1``), issues the same
deterministic completions request to each, and asserts that the per-token
top-1 ids and top-1 logprobs match within a bf16 tolerance.

The test is intentionally not registered with the CI suites: it requires
``~/models`` (WeLM v4 80B-A3B, hidden=2048 / oe_dim=512), TP=4, and the
``mk`` Python package installed. Invoke directly:

    python test/manual/welmv4/test_fused_decode_oe_gemm_equivalence.py

Pass ``--skip-launch`` to point at an already-running server (useful while
iterating on the kernel).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from contextlib import closing
from pathlib import Path
from typing import Optional

# Two short prompts; the second is long enough that we get many decode steps,
# which is the regime the fused kernel actually targets.
PROMPTS = [
    "1+1=",
    "请用一句话介绍一下大型语言模型的训练流程。",
]
MAX_NEW_TOKENS = 32
TOP_LOGPROBS = 1


def _free_port() -> int:
    # sglang derives a gRPC port = http_port + 10000, so keep us under 55535.
    for _ in range(64):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            sock.bind(("", 0))
            port = int(sock.getsockname()[1])
        if port < 55000:
            return port
    raise RuntimeError("could not find a free port < 55000 after 64 tries")


def _wait_for_health(host: str, port: int, timeout_s: float = 600.0) -> None:
    url = f"http://{host}:{port}/health_generate"
    deadline = time.time() + timeout_s
    last_err: Optional[BaseException] = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return
        except Exception as exc:  # pragma: no cover - polling loop
            last_err = exc
        time.sleep(2)
    raise RuntimeError(f"server at {url} never became healthy: {last_err!r}")


def _post_json(url: str, payload: dict, timeout_s: float = 120.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _completions(host: str, port: int, prompt: str) -> dict:
    return _post_json(
        f"http://{host}:{port}/v1/completions",
        {
            "model": "welmv4",
            "prompt": prompt,
            # Greedy: temperature=0 + top_k=1 makes the sampling step
            # effectively argmax, so any non-trivial logits drift between
            # fused/unfused will surface as a token mismatch.
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "max_tokens": MAX_NEW_TOKENS,
            "logprobs": TOP_LOGPROBS,
            "seed": 0,
        },
    )


def _launch_server(
    *,
    host: str,
    port: int,
    model: str,
    tp: int,
    fused_env: str,
    log_path: Path,
    extra_env: Optional[dict] = None,
) -> subprocess.Popen:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    # The variable under test:
    env["SGLANG_WELM_OE_FUSED_DECODE_GEMM"] = fused_env
    # When fused is on, also enable the one-shot fused-vs-unfused numerical
    # probe so the server log captures the max-abs diff for the first decode
    # forward — that's the cleanest signal of whether mk's kernel matches.
    if fused_env == "1":
        env["SGLANG_WELM_OE_FUSED_DECODE_GEMM_PROBE"] = "1"
        env["SGLANG_WELM_OE_FUSED_DECODE_GEMM_DUMP_DIR"] = "/tmp/welmv4_fused_oe_dump"
    # Make the server pick up the in-tree sglang regardless of how the user
    # invoked the script.
    env.setdefault("PYTHONPATH", str(Path(__file__).resolve().parents[3] / "python"))
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--host",
        host,
        "--port",
        str(port),
        "--model",
        model,
        "--served-model-name",
        "welmv4",
        "--trust-remote-code",
        "--tp",
        str(tp),
        "--mem-fraction-static",
        "0.8",
        "--enable-over-encoding",
        "--disable-radix-cache",
        # cuda graph is enabled here on purpose — the integration runs the
        # mk fused decode kernel inside the captured graph via
        # PreparedFusedDecodeNGramHashEmbeddingGemmAllReduce. With cuda graph
        # off the eager fused_decode_*_all_reduce path is exercised instead;
        # both modes should produce equivalent decode tokens within bf16
        # noise.
        "--sampling-defaults",
        "openai",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", buffering=1)
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return proc


def _stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=30)


def _collect(host: str, port: int) -> list[dict]:
    return [_completions(host, port, p) for p in PROMPTS]


def _compare(
    a: list[dict], b: list[dict], *, atol: float, rtol: float
) -> tuple[int, int, float]:
    """Compare two completions runs token-by-token.

    Returns ``(token_mismatches, total_tokens, max_logprob_diff)``. A
    token-id mismatch dominates: it always means the fused kernel diverged.
    Top-1 logprobs are checked as a softer signal.

    Once decode diverges at step ``k``, every token from step ``k+1`` onwards
    is generated against a different running context, so post-divergence
    logprobs aren't meaningful comparisons. We therefore only count the
    *pre-divergence* logprob delta toward ``max_logprob_diff``; this surfaces
    real numeric drift introduced by the fused kernel without being
    swamped by the autoregressive cascade after the first id mismatch.
    """
    mismatches = 0
    total = 0
    max_diff = 0.0
    assert len(a) == len(b), "different prompt counts"
    for prompt_idx, (ra, rb) in enumerate(zip(a, b)):
        ca = ra["choices"][0]
        cb = rb["choices"][0]
        ta = ca["logprobs"]["tokens"]
        tb = cb["logprobs"]["tokens"]
        la = ca["logprobs"]["token_logprobs"]
        lb = cb["logprobs"]["token_logprobs"]
        n = min(len(ta), len(tb))
        diverged = False
        first_div = None
        for i in range(n):
            total += 1
            ids_match = ta[i] == tb[i]
            if not ids_match:
                mismatches += 1
                if not diverged:
                    diverged = True
                    first_div = i
            if la[i] is None or lb[i] is None:
                continue
            if not diverged:
                # Pre-divergence: same context, same logits modulo numerical
                # drift — this is the bf16-precision check we actually care
                # about.
                diff = abs(float(la[i]) - float(lb[i]))
                if diff > max_diff:
                    max_diff = diff
        if first_div is not None:
            print(
                f"[diverge] prompt #{prompt_idx} first id mismatch at step {first_div}: "
                f"off='{ta[first_div]}' (lp={la[first_div]}) "
                f"on='{tb[first_div]}' (lp={lb[first_div]})",
                flush=True,
            )
    return mismatches, total, max_diff


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/josephyu/models")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument(
        "--port-off",
        type=int,
        default=0,
        help="port for the env-off server (0 = pick a free port)",
    )
    parser.add_argument(
        "--port-on",
        type=int,
        default=0,
        help="port for the env-on server (0 = pick a free port)",
    )
    parser.add_argument(
        "--skip-launch",
        action="store_true",
        help="if set, --port-off / --port-on must point at running servers",
    )
    parser.add_argument("--log-dir", default="/tmp/welmv4_fused_oe_eq_test")
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    port_off = args.port_off or _free_port()
    port_on = args.port_on or _free_port()

    procs: list[subprocess.Popen] = []
    try:
        if not args.skip_launch:
            print(f"[launch] off -> :{port_off}, log={log_dir}/off.log", flush=True)
            procs.append(
                _launch_server(
                    host=args.host,
                    port=port_off,
                    model=args.model,
                    tp=args.tp,
                    fused_env="0",
                    log_path=log_dir / "off.log",
                )
            )
            _wait_for_health(args.host, port_off)
            print(f"[launch] off ready on :{port_off}", flush=True)

        print(f"[query] off :{port_off}", flush=True)
        results_off = _collect(args.host, port_off)

        if not args.skip_launch:
            # Stop the env-off server before booting the env-on one to keep TP
            # GPUs available.
            _stop_server(procs.pop())

            print(f"[launch] on  -> :{port_on}, log={log_dir}/on.log", flush=True)
            procs.append(
                _launch_server(
                    host=args.host,
                    port=port_on,
                    model=args.model,
                    tp=args.tp,
                    fused_env="1",
                    log_path=log_dir / "on.log",
                )
            )
            _wait_for_health(args.host, port_on)
            print(f"[launch] on  ready on :{port_on}", flush=True)

        print(f"[query] on  :{port_on}", flush=True)
        results_on = _collect(args.host, port_on)
    finally:
        for proc in procs:
            _stop_server(proc)

    mismatches, total, max_diff = _compare(
        results_off, results_on, atol=args.atol, rtol=args.rtol
    )
    print(
        f"[result] tokens={total} mismatched_ids={mismatches} "
        f"max_top1_logprob_diff={max_diff:.4g}",
        flush=True,
    )
    if mismatches > 0:
        for prompt, off, on in zip(PROMPTS, results_off, results_on):
            text_off = off["choices"][0]["text"]
            text_on = on["choices"][0]["text"]
            if text_off != text_on:
                print(f"[diff] prompt={prompt!r}", flush=True)
                print(f"[diff] off={text_off!r}", flush=True)
                print(f"[diff] on ={text_on!r}", flush=True)
        return 1
    if max_diff > args.atol * 4:
        # bf16 logprob jitter > 0.2 nats is suspicious; surface it as a soft
        # warning via non-zero exit so CI-style invocations notice.
        print(
            f"[warn] max top-1 logprob diff {max_diff:.4g} exceeds soft "
            f"threshold {args.atol * 4:.4g}",
            flush=True,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
