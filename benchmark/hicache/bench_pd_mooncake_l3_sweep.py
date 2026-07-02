import json
import math
import os
import signal
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Optional

import requests
from transformers import AutoConfig, AutoTokenizer

PY = os.environ.get("PY", "/envs/venv/bin/python")
HOST = os.environ.get("HOST", "127.0.0.1")
MODEL_PATH = os.environ.get(
    "MODEL_PATH", "/mnt/wfs/mmshanghaiwfssh/llmmodels/openai/gpt-oss-20b"
)
SERVED_NAME = os.environ.get("SERVED_NAME", "gpt-oss-20b")
X_LEN = int(os.environ.get("X_LEN", "2048"))
Y_FACTORS = [
    int(item)
    for item in os.environ.get("Y_FACTORS", "1,2,4,8,16,32").split(",")
    if item.strip()
]
Y_BASE = int(os.environ.get("Y_BASE", "1024"))
REPLAY_MAX_NEW_TOKENS = int(os.environ.get("REPLAY_MAX_NEW_TOKENS", "128"))
CONTEXT_LENGTH = int(os.environ.get("CONTEXT_LENGTH", "65536"))
PAGE_SIZE = int(os.environ.get("PAGE_SIZE", "64"))
MEM_FRACTION_STATIC = os.environ.get("MEM_FRACTION_STATIC", "0.25")
SERVER_TP = int(os.environ.get("SERVER_TP", "1"))
PREFILL_BASE_GPU_ID = int(os.environ.get("PREFILL_BASE_GPU_ID", "0"))
DECODE_BASE_GPU_ID = int(os.environ.get("DECODE_BASE_GPU_ID", str(SERVER_TP)))
CUDA_VISIBLE_DEVICES = os.environ.get(
    "CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in range(max(2, SERVER_TP * 2)))
)
HICACHE_SIZE = os.environ.get("HICACHE_SIZE", "24")
HICACHE_RATIO = os.environ.get("HICACHE_RATIO", "2")
HICACHE_STORAGE_BATCH_SIZE = os.environ.get("HICACHE_STORAGE_BATCH_SIZE")
PD_TRANSFER_BACKEND = os.environ.get("PD_TRANSFER_BACKEND", "mooncake")
PD_IB_DEVICE = os.environ.get("PD_IB_DEVICE", "mlx5_bond_1")
MOONCAKE_PROTOCOL = os.environ.get("MOONCAKE_PROTOCOL", "tcp")
MOONCAKE_DEVICE = os.environ.get("MOONCAKE_DEVICE", "")
MOONCAKE_GLOBAL_SEGMENT_SIZE = os.environ.get(
    "MOONCAKE_GLOBAL_SEGMENT_SIZE", "4294967296"
)
MC_MS_AUTO_DISC = os.environ.get("MC_MS_AUTO_DISC", "0")
MC_TCP_ENABLE_CONNECTION_POOL = os.environ.get("MC_TCP_ENABLE_CONNECTION_POOL", "true")
EXTRA_SERVER_ARGS = shlex.split(os.environ.get("EXTRA_SERVER_ARGS", ""))
PREFILL_EXTRA_SERVER_ARGS = shlex.split(os.environ.get("PREFILL_EXTRA_SERVER_ARGS", ""))
DECODE_EXTRA_SERVER_ARGS = shlex.split(os.environ.get("DECODE_EXTRA_SERVER_ARGS", ""))
DECODE_RESERVED_TOKENS = os.environ.get(
    "DECODE_RESERVED_TOKENS", str(max(Y_FACTORS) * Y_BASE + REPLAY_MAX_NEW_TOKENS)
)
RUN_MODES = [
    item.strip()
    for item in os.environ.get("RUN_MODES", "off,on").split(",")
    if item.strip()
]
CASE_REPEATS = int(os.environ.get("CASE_REPEATS", "1"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "info")


def ensure_local_no_proxy() -> None:
    for key in ("NO_PROXY", "no_proxy"):
        parts = [item.strip() for item in os.environ.get(key, "").split(",")]
        parts = [item for item in parts if item]
        for local in ("127.0.0.1", "localhost"):
            if local not in parts:
                parts.append(local)
        os.environ[key] = ",".join(parts)


ensure_local_no_proxy()


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False))
    tmp.replace(path)


def free_port(start: int) -> int:
    port = start
    while port < start + 1000:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((HOST, port))
                return port
            except OSError:
                port += 1
    raise RuntimeError(f"no free port from {start}")


def bench_port(offset: int) -> int:
    return int(os.environ.get("BENCH_PORT_BASE", "44000")) + offset


def percentile(values: list[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil((pct / 100.0) * len(ordered)) - 1))
    return ordered[idx]


class PDMooncakeL3Sweep:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "main.log"
        self.procs: dict[str, subprocess.Popen] = {}
        self.ports = {
            "lb": free_port(bench_port(300)),
            "prefill": free_port(bench_port(400)),
            "decode": free_port(bench_port(500)),
            "boot": free_port(bench_port(600)),
            "mooncake_master": free_port(bench_port(700)),
            "mooncake_meta": free_port(bench_port(800)),
            "mooncake_master_metrics": free_port(bench_port(900)),
        }
        atomic_json(self.run_dir / "ports.json", self.ports)
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, trust_remote_code=True
        )

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%F %T')}] {msg}"
        print(line, flush=True)
        with self.log_path.open("a") as f:
            f.write(line + "\n")

    def env(self, extra: Optional[dict[str, Optional[str]]] = None) -> dict[str, str]:
        env = os.environ.copy()
        env.update(
            {
                "PATH": "/envs/venv/bin:" + env.get("PATH", ""),
                "PYTHONPATH": "/tmp/sglang-dev/python:" + env.get("PYTHONPATH", ""),
                "CUDA_VISIBLE_DEVICES": CUDA_VISIBLE_DEVICES,
                "TOKENIZERS_PARALLELISM": "false",
                "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
                "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1",
                "SGLANG_HICACHE_SWA_STORAGE_ENABLE": "1",
                "SGLANG_HICACHE_SWA_DECODE_STORAGE_ENABLE": "1",
                "SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE": str(PAGE_SIZE),
                "MC_TCP_ENABLE_CONNECTION_POOL": MC_TCP_ENABLE_CONNECTION_POOL,
                "MOONCAKE_MASTER": f"127.0.0.1:{self.ports['mooncake_master']}",
                "MOONCAKE_MASTER_METRICS_PORT": str(
                    self.ports["mooncake_master_metrics"]
                ),
                "MOONCAKE_PROTOCOL": MOONCAKE_PROTOCOL,
                "MC_MS_AUTO_DISC": MC_MS_AUTO_DISC,
                "MOONCAKE_DEVICE": MOONCAKE_DEVICE,
                "MOONCAKE_TE_META_DATA_SERVER": (
                    f"http://127.0.0.1:{self.ports['mooncake_meta']}/metadata"
                ),
                "MOONCAKE_GLOBAL_SEGMENT_SIZE": MOONCAKE_GLOBAL_SEGMENT_SIZE,
            }
        )
        if extra:
            for key, value in extra.items():
                if value is None:
                    env.pop(key, None)
                else:
                    env[key] = value
        return env

    def start_proc(
        self,
        name: str,
        cmd: list[str],
        log_name: str,
        extra_env: Optional[dict[str, Optional[str]]] = None,
    ) -> None:
        self.stop_proc(name)
        log_path = self.run_dir / log_name
        self.log(f"starting {name}: {' '.join(cmd)} log={log_path}")
        stream = open(log_path, "ab", buffering=0)
        proc = subprocess.Popen(
            cmd,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=self.env(extra_env),
            stdin=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )
        self.procs[name] = proc

    def stop_proc(self, name: str) -> None:
        proc = self.procs.pop(name, None)
        if proc is None or proc.poll() is not None:
            return
        self.log(f"stopping {name} pid={proc.pid}")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.time() + 30
        while time.time() < deadline and proc.poll() is None:
            time.sleep(0.5)
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass

    def cleanup_servers(self) -> None:
        for name in ("router", "decode", "prefill"):
            self.stop_proc(name)

    def cleanup_all(self) -> None:
        self.cleanup_servers()
        for name in ("mooncake_master", "mooncake_meta"):
            self.stop_proc(name)

    def wait_http(self, url: str, name: str, timeout: int, log_name: str) -> None:
        deadline = time.time() + timeout
        log_path = self.run_dir / log_name
        while time.time() < deadline:
            try:
                if requests.get(url, timeout=2).status_code == 200:
                    self.log(f"{name} ready at {url}")
                    return
            except Exception:
                pass
            proc = self.procs.get(name)
            if proc is not None and proc.poll() is not None:
                tail = ""
                if log_path.exists():
                    tail = "".join(
                        log_path.read_text(errors="replace").splitlines(True)[-160:]
                    )
                raise RuntimeError(f"{name} exited early; tail:\n{tail}")
            time.sleep(1)
        raise TimeoutError(f"timeout waiting for {name} at {url}")

    def wait_mooncake_master_ready(
        self,
        log_path: Path,
        start_offset: int,
        attempt_timeout: int,
    ) -> bool:
        deadline = time.time() + attempt_timeout
        while time.time() < deadline:
            proc = self.procs.get("mooncake_master")
            if proc is not None and proc.poll() is not None:
                return False

            if log_path.exists():
                with log_path.open("rb") as f:
                    f.seek(start_offset)
                    text = f.read().decode(errors="replace")
                if text.count("role=leader, state=serving, service_ready=true") >= 2:
                    return True

            time.sleep(1)
        return False

    def start_mooncake(self) -> None:
        self.start_proc(
            "mooncake_meta",
            [
                PY,
                "-m",
                "mooncake.http_metadata_server",
                "--port",
                str(self.ports["mooncake_meta"]),
                "--host",
                HOST,
            ],
            "mooncake_meta.log",
        )
        self.start_proc(
            "mooncake_master",
            [
                "mooncake_master",
                "--port",
                str(self.ports["mooncake_master"]),
                "--metrics_port",
                str(self.ports["mooncake_master_metrics"]),
            ],
            "mooncake_master.log",
            extra_env={"MOONCAKE_DEVICE": None},
        )
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                requests.get(
                    f"http://{HOST}:{self.ports['mooncake_meta']}/metadata",
                    timeout=2,
                )
                self.log("mooncake_meta ready")
                break
            except Exception:
                time.sleep(1)
        else:
            raise TimeoutError("timeout waiting for mooncake_meta")

        master_log_path = self.run_dir / "mooncake_master.log"
        for attempt in range(1, 6):
            start_offset = (
                master_log_path.stat().st_size if master_log_path.exists() else 0
            )
            if attempt > 1:
                self.start_proc(
                    "mooncake_master",
                    [
                        "mooncake_master",
                        "--port",
                        str(self.ports["mooncake_master"]),
                        "--metrics_port",
                        str(self.ports["mooncake_master_metrics"]),
                    ],
                    "mooncake_master.log",
                    extra_env={"MOONCAKE_DEVICE": None},
                )
            if self.wait_mooncake_master_ready(
                master_log_path,
                start_offset,
                attempt_timeout=25,
            ):
                break

            proc = self.procs.get("mooncake_master")
            state = (
                "exited"
                if proc is not None and proc.poll() is not None
                else "not ready"
            )
            self.log(f"mooncake_master attempt {attempt} {state}; retrying")
            self.stop_proc("mooncake_master")
        else:
            tail = ""
            if master_log_path.exists():
                tail = "".join(
                    master_log_path.read_text(errors="replace").splitlines(True)[-160:]
                )
            raise RuntimeError(f"mooncake_master not ready; tail:\n{tail}")
        self.log(
            "mooncake_master stable "
            f"rpc={self.ports['mooncake_master']} "
            f"metrics={self.ports['mooncake_master_metrics']}"
        )

    def server_base(self) -> list[str]:
        cmd = [
            PY,
            "-m",
            "sglang.launch_server",
            "--model-path",
            MODEL_PATH,
            "--trust-remote-code",
            "--served-model-name",
            SERVED_NAME,
            "--tp",
            str(SERVER_TP),
            "--host",
            HOST,
            "--disaggregation-bootstrap-port",
            str(self.ports["boot"]),
            "--disaggregation-transfer-backend",
            PD_TRANSFER_BACKEND,
        ]
        if PD_IB_DEVICE:
            cmd += ["--disaggregation-ib-device", PD_IB_DEVICE]
        return cmd

    def common_args(self) -> list[str]:
        return [
            "--mem-fraction-static",
            MEM_FRACTION_STATIC,
            "--page-size",
            str(PAGE_SIZE),
            "--enable-cache-report",
            "--context-length",
            str(CONTEXT_LENGTH),
            "--disable-cuda-graph",
            "--log-level",
            LOG_LEVEL,
        ]

    def hicache_args(self) -> list[str]:
        storage_extra_config = {"hicache_storage_pass_prefix_keys": True}
        if HICACHE_STORAGE_BATCH_SIZE:
            storage_extra_config["storage_batch_size"] = int(
                HICACHE_STORAGE_BATCH_SIZE
            )
        return [
            "--hicache-size",
            HICACHE_SIZE,
            "--hicache-ratio",
            HICACHE_RATIO,
            "--hicache-storage-prefetch-policy",
            "wait_complete",
            "--hicache-write-policy",
            "write_through",
            "--hicache-storage-backend",
            "mooncake",
            "--hicache-storage-backend-extra-config",
            json.dumps(storage_extra_config),
        ]

    def start_stack(self, mode: str) -> None:
        l3_enabled = mode == "on"
        base = self.server_base()
        prefill_log = f"{mode}.prefill.log"
        decode_log = f"{mode}.decode.log"
        router_log = f"{mode}.router.log"

        prefill_cmd = (
            base
            + [
                "--port",
                str(self.ports["prefill"]),
                "--base-gpu-id",
                str(PREFILL_BASE_GPU_ID),
                "--disaggregation-mode",
                "prefill",
            ]
            + self.common_args()
            + EXTRA_SERVER_ARGS
            + PREFILL_EXTRA_SERVER_ARGS
        )
        decode_cmd = (
            base
            + [
                "--port",
                str(self.ports["decode"]),
                "--base-gpu-id",
                str(DECODE_BASE_GPU_ID),
                "--disaggregation-mode",
                "decode",
            ]
            + self.common_args()
            + EXTRA_SERVER_ARGS
            + DECODE_EXTRA_SERVER_ARGS
        )
        if l3_enabled:
            prefill_cmd += ["--enable-hierarchical-cache"] + self.hicache_args()
            decode_cmd += [
                "--disaggregation-decode-enable-offload-kvcache",
                "--num-reserved-decode-tokens",
                DECODE_RESERVED_TOKENS,
            ] + self.hicache_args()

        self.start_proc("prefill", prefill_cmd, prefill_log)
        self.wait_http(
            f"http://{HOST}:{self.ports['prefill']}/health",
            "prefill",
            1800,
            prefill_log,
        )

        self.start_proc("decode", decode_cmd, decode_log)
        self.wait_http(
            f"http://{HOST}:{self.ports['decode']}/health",
            "decode",
            1800,
            decode_log,
        )

        self.start_proc(
            "router",
            [
                PY,
                "-m",
                "sglang_router.launch_router",
                "--pd-disaggregation",
                "--mini-lb",
                "--prefill",
                f"http://{HOST}:{self.ports['prefill']}",
                "--decode",
                f"http://{HOST}:{self.ports['decode']}",
                "--host",
                HOST,
                "--port",
                str(self.ports["lb"]),
            ],
            router_log,
        )
        self.wait_http(
            f"http://{HOST}:{self.ports['lb']}/health", "router", 300, router_log
        )

    def make_ids(self, length: int, salt: str) -> list[int]:
        seed = (
            f"PD Mooncake HiCache L3 benchmark case {salt}. "
            "This deterministic synthetic prompt intentionally avoids overlap "
            "between cases while remaining plain text. "
        )
        text = seed * ((length // 24) + 300)
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < length:
            raise RuntimeError(f"tokenizer produced {len(ids)} ids, wanted {length}")
        return ids[:length]

    def generate(self, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
        start = time.perf_counter()
        resp = requests.post(
            f"http://{HOST}:{self.ports['lb']}/generate",
            json=payload,
            timeout=timeout,
        )
        elapsed = time.perf_counter() - start
        if resp.status_code != 200:
            raise RuntimeError(
                f"generate failed {resp.status_code}: {resp.text[:3000]}"
            )
        obj = resp.json()
        obj["_client_elapsed_s"] = elapsed
        return obj

    def stream_generate(self, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
        stream_payload = dict(payload)
        stream_payload["stream"] = True
        start = time.perf_counter()
        first_token_s: Optional[float] = None
        last_token_time: Optional[float] = None
        last_completion_tokens = 0
        itl_samples: list[float] = []
        last_obj: dict[str, Any] = {}
        chunk_count = 0
        with requests.post(
            f"http://{HOST}:{self.ports['lb']}/generate",
            json=stream_payload,
            stream=True,
            timeout=timeout,
        ) as resp:
            if resp.status_code != 200:
                raise RuntimeError(
                    f"stream generate failed {resp.status_code}: {resp.text[:3000]}"
                )
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.strip()
                if line.startswith("data:"):
                    line = line[len("data:") :].strip()
                if line == "[DONE]":
                    break
                now = time.perf_counter()
                data = json.loads(line)
                last_obj = data
                text = data.get("text") or ""
                meta = data.get("meta_info") or {}
                completion_tokens = int(
                    meta.get("completion_tokens", last_completion_tokens) or 0
                )
                if text and first_token_s is None:
                    first_token_s = now - start
                    last_token_time = now
                if completion_tokens > last_completion_tokens:
                    if last_token_time is not None and completion_tokens > 1:
                        delta_tokens = completion_tokens - last_completion_tokens
                        gap = now - last_token_time
                        itl_samples.extend([gap / delta_tokens] * delta_tokens)
                    last_token_time = now
                    last_completion_tokens = completion_tokens
                chunk_count += 1
        elapsed = time.perf_counter() - start
        last_obj["_client_elapsed_s"] = elapsed
        last_obj["_ttft_s"] = first_token_s
        last_obj["_itl_samples_s"] = itl_samples
        last_obj["_stream_chunks"] = chunk_count
        return last_obj

    def flush_side(self, side: str, port: int) -> dict[str, Any]:
        start = time.perf_counter()
        try:
            resp = requests.post(
                f"http://{HOST}:{port}/flush_cache?timeout=600", timeout=660
            )
            return {
                "status": resp.status_code,
                "body": resp.text[:500],
                "elapsed_s": time.perf_counter() - start,
            }
        except Exception as exc:
            return {"error": repr(exc), "elapsed_s": time.perf_counter() - start}

    def flush_all(self) -> dict[str, Any]:
        return {
            "prefill": self.flush_side("prefill", self.ports["prefill"]),
            "decode": self.flush_side("decode", self.ports["decode"]),
        }

    def meta_summary(self, obj: dict[str, Any]) -> dict[str, Any]:
        meta = obj.get("meta_info") or {}
        details = meta.get("cached_tokens_details") or {}
        if isinstance(details, list):
            details = details[0] if details else {}
        if not isinstance(details, dict):
            details = {}
        return {
            "prompt_tokens": int(meta.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(meta.get("completion_tokens", 0) or 0),
            "cached_tokens": int(meta.get("cached_tokens", 0) or 0),
            "cached_tokens_details": details,
            "storage_cached_tokens": int(details.get("storage", 0) or 0),
            "host_cached_tokens": int(details.get("host", 0) or 0),
            "device_cached_tokens": int(details.get("device", 0) or 0),
            "server_e2e_latency_s": meta.get("e2e_latency"),
            "client_elapsed_s": obj.get("_client_elapsed_s"),
        }

    def config_info(self) -> dict[str, Any]:
        from sglang.srt.configs.model_config import is_hybrid_swa_model

        cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
        arch = list(getattr(cfg, "architectures", []) or [])
        return {
            "architectures": arch,
            "model_type": getattr(cfg, "model_type", None),
            "sliding_window": getattr(cfg, "sliding_window", None),
            "hybrid_swa_allowlisted": bool(is_hybrid_swa_model(arch)),
        }

    def row_from_case(
        self,
        mode: str,
        repeat_index: int,
        y_target: int,
        x_ids: list[int],
        y_ids: list[int],
        populate: dict[str, Any],
        replay: dict[str, Any],
        flush: dict[str, Any],
        replay_flush: dict[str, Any],
    ) -> dict[str, Any]:
        replay_meta = self.meta_summary(replay)
        populate_meta = self.meta_summary(populate)
        ttft_s = replay.get("_ttft_s")
        e2e_s = float(replay.get("_client_elapsed_s") or 0.0)
        output_tokens = replay_meta["completion_tokens"]
        decode_duration_s = max(e2e_s - (ttft_s or 0.0), 1e-9)
        input_len = len(x_ids) + len(y_ids)
        itl_samples = list(replay.get("_itl_samples_s") or [])
        return {
            "config": f"hicache_l3_{mode}",
            "l3_enabled": mode == "on",
            "model": MODEL_PATH,
            "topology": f"pd_prefill_tp{SERVER_TP}_decode_tp{SERVER_TP}",
            "router_policy": "mini_lb",
            "concurrency": 1,
            "sample_count": 1,
            "repeat_index": repeat_index,
            "x_len": len(x_ids),
            "target_y_len": y_target,
            "actual_y_len": len(y_ids),
            "input_len_x_plus_y": input_len,
            "page_size": PAGE_SIZE,
            "expected_page_aligned_x_plus_y": (input_len // PAGE_SIZE) * PAGE_SIZE,
            "max_decode_tokens": REPLAY_MAX_NEW_TOKENS,
            "successful_requests": 1,
            "failed_requests": 0,
            "empty_predict_count": 1 if output_tokens == 0 else 0,
            "duration_sec": e2e_s,
            "prefill_tok_s": None if not ttft_s else input_len / ttft_s,
            "prefill_tok_s_per_gpu": (
                None if not ttft_s else input_len / ttft_s / SERVER_TP
            ),
            "decode_tok_s": output_tokens / decode_duration_s,
            "decode_tok_s_per_gpu": output_tokens / decode_duration_s / SERVER_TP,
            "decode_tok_s_per_request": output_tokens / decode_duration_s,
            "request_rate_per_s": 1.0 / e2e_s if e2e_s > 0 else None,
            "total_prompt_tokens": input_len,
            "total_completion_tokens": output_tokens,
            "completion_tokens_avg": output_tokens,
            "overall_tok_s": (input_len + output_tokens) / e2e_s if e2e_s > 0 else None,
            "populate_duration_sec": populate.get("_client_elapsed_s"),
            "populate_decode_tok_s": (
                len(y_ids) / populate["_client_elapsed_s"]
                if populate.get("_client_elapsed_s")
                else None
            ),
            "ttft_avg_ms": None if ttft_s is None else ttft_s * 1000.0,
            "ttft_p50_ms": None if ttft_s is None else ttft_s * 1000.0,
            "ttft_p99_ms": None if ttft_s is None else ttft_s * 1000.0,
            "e2e_avg_ms": e2e_s * 1000.0,
            "itl_avg_ms": None if not itl_samples else mean(itl_samples) * 1000.0,
            "itl_p50_ms": (
                None if not itl_samples else percentile(itl_samples, 50) * 1000.0
            ),
            "itl_p99_ms": (
                None if not itl_samples else percentile(itl_samples, 99) * 1000.0
            ),
            "replay_meta": replay_meta,
            "populate_meta": populate_meta,
            "flush": flush,
            "replay_flush": replay_flush,
        }

    def collect_evidence(self, mode: str) -> dict[str, list[str]]:
        keywords = [
            "SWAKVPool",
            "Attached hybrid pool",
            "Finished SWA backup",
            "Finished backup",
            "#cached-token",
            "HiCache prefetch success",
            "HICACHE_TIMING",
            "Cache flushed",
            "Decode HiCache",
            "SWA HiCache",
            "Disable hybrid SWA",
            "RuntimeError",
            "Traceback",
        ]
        out: dict[str, list[str]] = {}
        for path in sorted(self.run_dir.glob(f"{mode}.*.log")):
            lines = []
            for line in path.read_text(errors="replace").splitlines():
                if any(k in line for k in keywords):
                    lines.append(line)
            out[path.name] = lines[-200:]
        return out

    def run_mode(self, mode: str) -> list[dict[str, Any]]:
        self.log(f"MODE_START {mode}")
        mode_dir = self.run_dir / mode
        mode_dir.mkdir(exist_ok=True)
        self.start_mooncake()
        self.start_stack(mode)
        rows = []
        try:
            for factor in Y_FACTORS:
                y_target = Y_BASE * factor
                for repeat_index in range(CASE_REPEATS):
                    case_name = f"{mode}_x{X_LEN}_y{y_target}_r{repeat_index}"
                    case_dir = mode_dir / case_name
                    case_dir.mkdir(parents=True, exist_ok=True)
                    self.log(f"CASE_START {case_name}")
                    x_ids = self.make_ids(X_LEN, case_name)
                    atomic_json(case_dir / "x_ids.json", x_ids)
                    populate = self.generate(
                        {
                            "input_ids": x_ids,
                            "sampling_params": {
                                "temperature": 0.0,
                                "max_new_tokens": y_target,
                                "ignore_eos": True,
                            },
                        },
                        timeout=max(1800, y_target * 3),
                    )
                    y_ids = populate.get("output_ids") or []
                    atomic_json(case_dir / "populate.response.json", populate)
                    atomic_json(case_dir / "y_ids.json", y_ids)
                    flush = self.flush_all()
                    atomic_json(case_dir / "populate.flush.json", flush)
                    time.sleep(10)
                    replay_ids = x_ids + y_ids
                    atomic_json(case_dir / "replay_input_ids.json", replay_ids)
                    replay = self.stream_generate(
                        {
                            "input_ids": replay_ids,
                            "sampling_params": {
                                "temperature": 0.0,
                                "max_new_tokens": REPLAY_MAX_NEW_TOKENS,
                                "ignore_eos": True,
                            },
                        },
                        timeout=max(1800, len(replay_ids) * 2),
                    )
                    atomic_json(case_dir / "replay.response.json", replay)
                    replay_flush = self.flush_all()
                    atomic_json(case_dir / "replay.flush.json", replay_flush)
                    row = self.row_from_case(
                        mode,
                        repeat_index,
                        y_target,
                        x_ids,
                        y_ids,
                        populate,
                        replay,
                        flush,
                        replay_flush,
                    )
                    atomic_json(case_dir / "metrics.json", row)
                    rows.append(row)
                    atomic_json(mode_dir / "results.partial.json", rows)
                    self.log(
                        "CASE_DONE "
                        + json.dumps(
                            {
                                "mode": mode,
                                "repeat": repeat_index,
                                "target_y": y_target,
                                "actual_y": len(y_ids),
                                "ttft_ms": row["ttft_avg_ms"],
                                "prefill_tok_s": row["prefill_tok_s"],
                                "storage_cached": row["replay_meta"][
                                    "storage_cached_tokens"
                                ],
                                "cached": row["replay_meta"]["cached_tokens"],
                            },
                            sort_keys=True,
                        )
                    )
        finally:
            atomic_json(mode_dir / "evidence.json", self.collect_evidence(mode))
            self.cleanup_all()
            time.sleep(5)
        self.log(f"MODE_DONE {mode}")
        return rows

    def write_report(self, rows: list[dict[str, Any]]) -> None:
        lines = [
            "# PD Mooncake HiCache L3 Sweep",
            "",
            f"- model: `{MODEL_PATH}`",
            (
                "- topology: "
                f"PD separation, prefill TP{SERVER_TP} base GPU {PREFILL_BASE_GPU_ID}, "
                f"decode TP{SERVER_TP} base GPU {DECODE_BASE_GPU_ID}"
            ),
            f"- PD transfer backend: `{PD_TRANSFER_BACKEND}`, ib device: `{PD_IB_DEVICE}`",
            f"- x_len: `{X_LEN}`, y: `1024 * {Y_FACTORS}`",
            f"- case_repeats: `{CASE_REPEATS}`",
            f"- context_length: `{CONTEXT_LENGTH}`, page_size: `{PAGE_SIZE}`",
            f"- replay max_new_tokens: `{REPLAY_MAX_NEW_TOKENS}`",
            "",
            "| L3 | repeat | y_target | actual_y | storage_cached | TTFT ms | E2E ms | effective prefill tok/s | decode tok/s | overall tok/s |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                "| {l3} | {repeat} | {target} | {actual} | {storage} | {ttft:.2f} | {e2e:.2f} | {prefill:.2f} | {decode:.2f} | {overall:.2f} |".format(
                    l3="on" if row["l3_enabled"] else "off",
                    repeat=row["repeat_index"],
                    target=row["target_y_len"],
                    actual=row["actual_y_len"],
                    storage=row["replay_meta"]["storage_cached_tokens"],
                    ttft=row["ttft_avg_ms"] or 0.0,
                    e2e=row["e2e_avg_ms"],
                    prefill=row["prefill_tok_s"] or 0.0,
                    decode=row["decode_tok_s"] or 0.0,
                    overall=row["overall_tok_s"] or 0.0,
                )
            )
        (self.run_dir / "report.md").write_text("\n".join(lines) + "\n")

    def run(self) -> None:
        self.log(f"run_dir={self.run_dir}")
        self.log(f"sglang_import={self._sglang_import_path()}")
        self.log(f"config_info={json.dumps(self.config_info(), sort_keys=True)}")
        self.log(f"run_modes={','.join(RUN_MODES)}")
        rows: list[dict[str, Any]] = []
        for mode in RUN_MODES:
            if mode not in {"on", "off"}:
                raise ValueError(f"unknown mode {mode!r}; expected on/off")
            rows.extend(self.run_mode(mode))
            atomic_json(self.run_dir / "results.partial.json", rows)
        atomic_json(
            self.run_dir / "summary.json",
            {
                "config": {
                    "model": MODEL_PATH,
                    "served_name": SERVED_NAME,
                    "x_len": X_LEN,
                    "y_base": Y_BASE,
                    "y_factors": Y_FACTORS,
                    "replay_max_new_tokens": REPLAY_MAX_NEW_TOKENS,
                    "context_length": CONTEXT_LENGTH,
                    "page_size": PAGE_SIZE,
                    "mem_fraction_static": MEM_FRACTION_STATIC,
                    "server_tp": SERVER_TP,
                    "prefill_base_gpu_id": PREFILL_BASE_GPU_ID,
                    "decode_base_gpu_id": DECODE_BASE_GPU_ID,
                    "cuda_visible_devices": CUDA_VISIBLE_DEVICES,
                    "extra_server_args": EXTRA_SERVER_ARGS,
                    "prefill_extra_server_args": PREFILL_EXTRA_SERVER_ARGS,
                    "decode_extra_server_args": DECODE_EXTRA_SERVER_ARGS,
                    "pd_transfer_backend": PD_TRANSFER_BACKEND,
                    "pd_ib_device": PD_IB_DEVICE,
                    "decode_reserved_tokens": DECODE_RESERVED_TOKENS,
                    "case_repeats": CASE_REPEATS,
                },
                "model_config": self.config_info(),
                "results": rows,
            },
        )
        atomic_json(self.run_dir / "results.json", rows)
        self.write_report(rows)
        self.log("SWEEP_DONE")

    def _sglang_import_path(self) -> str:
        import pathlib

        import sglang

        return str(pathlib.Path(sglang.__file__).resolve())


def main() -> None:
    run_dir = Path(os.environ.get("RUN_DIR", "/tmp/swa-hicache-l3-perf-sweep"))
    bench = PDMooncakeL3Sweep(run_dir)
    try:
        bench.run()
        (run_dir / "SUCCESS").write_text(time.strftime("%F %T") + "\n")
    except Exception as exc:
        (run_dir / "FAILED").write_text(f"{type(exc).__name__}: {exc}\n")
        raise
    finally:
        bench.cleanup_all()


if __name__ == "__main__":
    sys.exit(main())
