import json
import math
import os
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Optional

import requests
from transformers import AutoConfig

from bench_pd_mooncake_l3_sweep import (
    CONTEXT_LENGTH,
    HOST,
    MC_MS_AUTO_DISC,
    MODEL_PATH,
    MOONCAKE_DEVICE,
    MOONCAKE_GLOBAL_SEGMENT_SIZE,
    MOONCAKE_PROTOCOL,
    PAGE_SIZE,
    PDMooncakeL3Sweep,
    PD_IB_DEVICE,
    PD_TRANSFER_BACKEND,
    SERVED_NAME,
    SERVER_TP,
    atomic_json,
)

M_LEN = int(os.environ.get("M_LEN", "2048"))
N_BASE = int(os.environ.get("N_BASE", "1024"))
if os.environ.get("N_LIST"):
    N_LIST = [int(item) for item in os.environ["N_LIST"].split(",") if item.strip()]
else:
    N_FACTORS = [
        int(item)
        for item in os.environ.get("N_FACTORS", "1,2,4").split(",")
        if item.strip()
    ]
    N_LIST = [N_BASE * factor for factor in N_FACTORS]

CASE_REPEATS = int(os.environ.get("CASE_REPEATS", "2"))
MEASURE_MAX_NEW_TOKENS = int(os.environ.get("MEASURE_MAX_NEW_TOKENS", "1"))
POPULATE_MAX_NEW_TOKENS = int(os.environ.get("POPULATE_MAX_NEW_TOKENS", "1"))
POST_POPULATE_SLEEP_SEC = float(os.environ.get("POST_POPULATE_SLEEP_SEC", "10"))
POST_FLUSH_SLEEP_SEC = float(os.environ.get("POST_FLUSH_SLEEP_SEC", "0.2"))
WARMUP_LEN = int(os.environ.get("WARMUP_LEN", "256"))
KV_BYTES_PER_ELEM = int(os.environ.get("KV_BYTES_PER_ELEM", "2"))


def percentile(values: list[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def summarize_values(values: list[float]) -> dict[str, Optional[float]]:
    if not values:
        return {"avg": None, "p50": None, "p90": None, "min": None, "max": None}
    return {
        "avg": mean(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "min": min(values),
        "max": max(values),
    }


def parse_hicache_timing_line(line: str) -> Optional[dict[str, Any]]:
    marker = "HICACHE_TIMING "
    pos = line.find(marker)
    if pos < 0:
        return None
    payload = line[pos + len(marker) :].strip()
    try:
        event = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return event if isinstance(event, dict) else None


def event_num(event: dict[str, Any], *names: str) -> Optional[float]:
    for name in names:
        value = event.get(name)
        if isinstance(value, (int, float)):
            return float(value)
    return None


class PDMooncakeL3EffectiveLoad(PDMooncakeL3Sweep):
    def env(self, extra: Optional[dict[str, str]] = None) -> dict[str, str]:
        env = super().env(extra)
        env.setdefault("SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE", "0")
        return env

    def collect_hicache_timing_events(self) -> list[dict[str, Any]]:
        events = []
        for path in sorted(self.run_dir.glob("*.log")):
            for line_no, line in enumerate(
                path.read_text(errors="replace").splitlines(), 1
            ):
                event = parse_hicache_timing_line(line)
                if event is None:
                    continue
                event["source_log"] = path.name
                event["line_no"] = line_no
                events.append(event)
        return events

    def summarize_hicache_timing_events(
        self, events: list[dict[str, Any]]
    ) -> dict[str, Any]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for event in events:
            key = "|".join(
                [
                    str(event.get("event", "")),
                    str(event.get("interface", "")),
                    str(event.get("pool", "")),
                ]
            )
            groups.setdefault(key, []).append(event)

        group_summaries = {}
        for key, group_events in groups.items():
            elapsed = [
                value
                for event in group_events
                if (value := event_num(event, "elapsed_ms", "transfer_elapsed_ms"))
                is not None
            ]
            request_e2e = [
                value
                for event in group_events
                if (value := event_num(event, "request_e2e_ms")) is not None
            ]
            bandwidth = [
                value
                for event in group_events
                if (
                    value := event_num(
                        event,
                        "gib_per_s",
                        "transfer_gib_per_s",
                        "request_e2e_gib_per_s",
                    )
                )
                is not None
            ]
            bytes_values = [
                value
                for event in group_events
                if (
                    value := event_num(
                        event,
                        "transferred_bytes",
                        "total_bytes",
                        "requested_bytes",
                        "transfer_total_bytes",
                    )
                )
                is not None
            ]
            group_summaries[key] = {
                "count": len(group_events),
                "bytes_sum": int(sum(bytes_values)),
                "bytes": summarize_values(bytes_values),
                "elapsed_ms": summarize_values(elapsed),
                "request_e2e_ms": summarize_values(request_e2e),
                "gib_per_s": summarize_values(bandwidth),
            }

        m_prefetch_events = [
            event
            for event in events
            if event.get("event") == "hicache_prefetch_request"
            and int(event.get("kv_completed_tokens") or 0) == M_LEN
        ]
        m_transfer_ms = [
            value
            for event in m_prefetch_events
            if (value := event_num(event, "transfer_elapsed_ms")) is not None
        ]
        m_e2e_ms = [
            value
            for event in m_prefetch_events
            if (value := event_num(event, "request_e2e_ms")) is not None
        ]
        m_bytes = [
            value
            for event in m_prefetch_events
            if (value := event_num(event, "total_bytes")) is not None
        ]
        m_transfer_gib_s = [
            value
            for event in m_prefetch_events
            if (value := event_num(event, "transfer_gib_per_s")) is not None
        ]
        m_e2e_gib_s = [
            value
            for event in m_prefetch_events
            if (value := event_num(event, "request_e2e_gib_per_s")) is not None
        ]
        m_load_events = [
            event
            for event in events
            if event.get("event") == "hicache_load_to_device"
            and int(event.get("kv_tokens") or 0) == M_LEN
        ]
        m_load_ms = [
            value
            for event in m_load_events
            if (value := event_num(event, "elapsed_ms")) is not None
        ]
        m_load_bytes = [
            value
            for event in m_load_events
            if (value := event_num(event, "total_bytes")) is not None
        ]
        m_load_gib_s = [
            value
            for event in m_load_events
            if (value := event_num(event, "gib_per_s")) is not None
        ]

        return {
            "event_count": len(events),
            "groups": group_summaries,
            "m_len_prefetch_request_count": len(m_prefetch_events),
            "m_len_prefetch_request": {
                "total_bytes": summarize_values(m_bytes),
                "transfer_elapsed_ms": summarize_values(m_transfer_ms),
                "request_e2e_ms": summarize_values(m_e2e_ms),
                "transfer_gib_per_s": summarize_values(m_transfer_gib_s),
                "request_e2e_gib_per_s": summarize_values(m_e2e_gib_s),
            },
            "m_len_load_to_device_count": len(m_load_events),
            "m_len_load_to_device": {
                "total_bytes": summarize_values(m_load_bytes),
                "elapsed_ms": summarize_values(m_load_ms),
                "gib_per_s": summarize_values(m_load_gib_s),
            },
        }

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
                meta = data.get("meta_info") or {}
                completion_tokens = int(
                    meta.get("completion_tokens", last_completion_tokens) or 0
                )
                if completion_tokens > last_completion_tokens:
                    if first_token_s is None:
                        first_token_s = now - start
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

    def request_payload(self, input_ids: list[int]) -> dict[str, Any]:
        return {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": MEASURE_MAX_NEW_TOKENS,
                "ignore_eos": True,
            },
        }

    def populate_payload(self, input_ids: list[int]) -> dict[str, Any]:
        return {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": POPULATE_MAX_NEW_TOKENS,
                "ignore_eos": True,
            },
        }

    def measured_request(
        self, input_ids: list[int], case_dir: Path, name: str
    ) -> dict[str, Any]:
        timeout = max(1800, len(input_ids) * 2)
        resp = self.stream_generate(self.request_payload(input_ids), timeout=timeout)
        atomic_json(case_dir / f"{name}.response.json", resp)
        meta = self.meta_summary(resp)
        ttft_s = resp.get("_ttft_s")
        return {
            "name": name,
            "input_len": len(input_ids),
            "ttft_ms": None if ttft_s is None else ttft_s * 1000.0,
            "e2e_ms": float(resp.get("_client_elapsed_s") or 0.0) * 1000.0,
            "meta": meta,
            "response_path": str(case_dir / f"{name}.response.json"),
        }

    def flush_and_pause(self) -> dict[str, Any]:
        result = self.flush_all()
        if POST_FLUSH_SLEEP_SEC > 0:
            time.sleep(POST_FLUSH_SLEEP_SEC)
        return result

    def warmup(self, mode_dir: Path, l3_enabled: bool) -> None:
        warmup_dir = mode_dir / "warmup"
        warmup_dir.mkdir(parents=True, exist_ok=True)
        lengths = sorted({WARMUP_LEN, M_LEN, *(M_LEN + n for n in N_LIST)})
        shape_rows = []
        for length in lengths:
            ids = self.make_ids(length, f"effective_load_shape_warmup_{length}")
            resp = self.stream_generate(self.request_payload(ids), timeout=1800)
            shape_rows.append(
                {
                    "length": length,
                    "ttft_ms": (
                        None if resp.get("_ttft_s") is None else resp["_ttft_s"] * 1000
                    ),
                    "meta": self.meta_summary(resp),
                }
            )
            atomic_json(
                warmup_dir / f"shape_{length}.flush.json", self.flush_and_pause()
            )

        l3_row = None
        if l3_enabled:
            max_n = max(N_LIST)
            ids = self.make_ids(M_LEN + max_n, "effective_load_l3_warmup")
            populate = self.generate(
                self.populate_payload(ids[:M_LEN]), timeout=max(1800, M_LEN * 2)
            )
            atomic_json(warmup_dir / "l3_populate.response.json", populate)
            atomic_json(warmup_dir / "l3_populate.flush.json", self.flush_and_pause())
            time.sleep(POST_POPULATE_SLEEP_SEC)
            replay = self.stream_generate(
                self.request_payload(ids), timeout=max(1800, len(ids) * 2)
            )
            l3_row = {
                "length": len(ids),
                "ttft_ms": (
                    None if replay.get("_ttft_s") is None else replay["_ttft_s"] * 1000
                ),
                "meta": self.meta_summary(replay),
            }
            atomic_json(warmup_dir / "l3_replay.response.json", replay)
            atomic_json(warmup_dir / "l3_replay.flush.json", self.flush_and_pause())

        atomic_json(
            warmup_dir / "summary.json",
            {"shape_warmups": shape_rows, "l3_warmup": l3_row},
        )

    def case_ids(
        self, repeat_index: int, n_len: int
    ) -> tuple[str, list[int], list[int]]:
        case_name = f"m{M_LEN}_n{n_len}_r{repeat_index}"
        ids = self.make_ids(M_LEN + n_len, f"effective_load_{case_name}")
        return case_name, ids[:M_LEN], ids[M_LEN:]

    def run_nocache_mode(self) -> list[dict[str, Any]]:
        self.log("MODE_START nocache")
        mode_dir = self.run_dir / "nocache"
        mode_dir.mkdir(exist_ok=True)
        self.start_mooncake()
        self.start_stack("off")
        rows: list[dict[str, Any]] = []
        try:
            self.warmup(mode_dir, l3_enabled=False)
            for n_len in N_LIST:
                for repeat_index in range(CASE_REPEATS):
                    case_name, m_ids, n_ids = self.case_ids(repeat_index, n_len)
                    case_dir = mode_dir / case_name
                    case_dir.mkdir(parents=True, exist_ok=True)
                    self.log(f"NO_CACHE_CASE_START {case_name}")
                    prompt_mn = m_ids + n_ids

                    atomic_json(case_dir / "input_m_ids.json", m_ids)
                    atomic_json(case_dir / "input_n_ids.json", n_ids)

                    atomic_json(
                        case_dir / "before_t1.flush.json", self.flush_and_pause()
                    )
                    t1 = self.measured_request(prompt_mn, case_dir, "t1_m_plus_n")
                    atomic_json(
                        case_dir / "after_t1.flush.json", self.flush_and_pause()
                    )

                    atomic_json(
                        case_dir / "before_t2.flush.json", self.flush_and_pause()
                    )
                    t2 = self.measured_request(m_ids, case_dir, "t2_m_only")
                    atomic_json(
                        case_dir / "after_t2.flush.json", self.flush_and_pause()
                    )

                    row = {
                        "case": case_name,
                        "repeat_index": repeat_index,
                        "m_len": M_LEN,
                        "n_len": n_len,
                        "t1_no_cache_m_plus_n": t1,
                        "t2_no_cache_m": t2,
                    }
                    rows.append(row)
                    atomic_json(case_dir / "metrics.json", row)
                    atomic_json(mode_dir / "results.partial.json", rows)
                    self.log(
                        "NO_CACHE_CASE_DONE "
                        + json.dumps(
                            {
                                "case": case_name,
                                "t1_ttft_ms": t1["ttft_ms"],
                                "t2_ttft_ms": t2["ttft_ms"],
                            },
                            sort_keys=True,
                        )
                    )
        finally:
            atomic_json(mode_dir / "evidence.json", self.collect_evidence("off"))
            self.cleanup_all()
            time.sleep(5)
        self.log("MODE_DONE nocache")
        return rows

    def run_cached_mode(self) -> list[dict[str, Any]]:
        self.log("MODE_START cached_l3")
        mode_dir = self.run_dir / "cached_l3"
        mode_dir.mkdir(exist_ok=True)
        self.start_mooncake()
        self.start_stack("on")
        rows: list[dict[str, Any]] = []
        try:
            self.warmup(mode_dir, l3_enabled=True)
            for n_len in N_LIST:
                for repeat_index in range(CASE_REPEATS):
                    case_name, m_ids, n_ids = self.case_ids(repeat_index, n_len)
                    case_dir = mode_dir / case_name
                    case_dir.mkdir(parents=True, exist_ok=True)
                    self.log(f"CACHED_CASE_START {case_name}")
                    prompt_mn = m_ids + n_ids

                    atomic_json(case_dir / "input_m_ids.json", m_ids)
                    atomic_json(case_dir / "input_n_ids.json", n_ids)

                    populate = self.generate(
                        self.populate_payload(m_ids), timeout=max(1800, M_LEN * 2)
                    )
                    atomic_json(case_dir / "populate.response.json", populate)
                    populate_meta = self.meta_summary(populate)
                    atomic_json(
                        case_dir / "populate.flush.json", self.flush_and_pause()
                    )
                    time.sleep(POST_POPULATE_SLEEP_SEC)

                    t3 = self.measured_request(
                        prompt_mn, case_dir, "t3_cached_m_plus_n"
                    )
                    atomic_json(
                        case_dir / "after_t3.flush.json", self.flush_and_pause()
                    )
                    row = {
                        "case": case_name,
                        "repeat_index": repeat_index,
                        "m_len": M_LEN,
                        "n_len": n_len,
                        "populate_meta": populate_meta,
                        "t3_cached_m_plus_n": t3,
                    }
                    rows.append(row)
                    atomic_json(case_dir / "metrics.json", row)
                    atomic_json(mode_dir / "results.partial.json", rows)
                    self.log(
                        "CACHED_CASE_DONE "
                        + json.dumps(
                            {
                                "case": case_name,
                                "t3_ttft_ms": t3["ttft_ms"],
                                "storage_cached": t3["meta"]["storage_cached_tokens"],
                                "cached": t3["meta"]["cached_tokens"],
                            },
                            sort_keys=True,
                        )
                    )
        finally:
            atomic_json(mode_dir / "evidence.json", self.collect_evidence("on"))
            self.cleanup_all()
            time.sleep(5)
        self.log("MODE_DONE cached_l3")
        return rows

    def model_transfer_info(self) -> dict[str, Any]:
        cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
        text_cfg = getattr(cfg, "text_config", cfg)
        arch = list(getattr(cfg, "architectures", []) or [])
        if not arch:
            arch = list(getattr(text_cfg, "architectures", []) or [])

        try:
            from sglang.srt.configs.model_config import get_hybrid_layer_ids

            swa_layers, full_layers = get_hybrid_layer_ids(
                arch, text_cfg, context_len=CONTEXT_LENGTH
            )
        except Exception as exc:
            swa_layers, full_layers = None, None
            layer_error = repr(exc)
        else:
            layer_error = None

        num_heads = int(getattr(text_cfg, "num_attention_heads", 0) or 0)
        num_kv_heads = int(
            getattr(text_cfg, "num_key_value_heads", None)
            or getattr(text_cfg, "swa_num_key_value_heads", None)
            or num_heads
        )
        hidden_size = int(getattr(text_cfg, "hidden_size", 0) or 0)
        head_dim = int(getattr(text_cfg, "head_dim", 0) or 0)
        if head_dim <= 0 and num_heads > 0:
            head_dim = hidden_size // num_heads

        windows = [
            int(x)
            for x in (getattr(text_cfg, "sliding_window_size_layerwise", []) or [])
            if x is not None and int(x) > 0 and int(x) < CONTEXT_LENGTH
        ]
        swa_window = (
            min(windows) if windows else getattr(text_cfg, "sliding_window", None)
        )
        if swa_window is not None:
            swa_window = int(swa_window)

        return {
            "architectures": arch,
            "model_type": getattr(cfg, "model_type", None),
            "num_hidden_layers": int(getattr(text_cfg, "num_hidden_layers", 0) or 0),
            "swa_layer_count": None if swa_layers is None else len(swa_layers),
            "full_layer_count": None if full_layers is None else len(full_layers),
            "swa_layers": swa_layers,
            "full_layers": full_layers,
            "layer_detection_error": layer_error,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
            "head_dim": head_dim,
            "kv_bytes_per_elem": KV_BYTES_PER_ELEM,
            "kv_bytes_per_token_per_layer_total": (
                2 * num_kv_heads * head_dim * KV_BYTES_PER_ELEM
                if num_kv_heads > 0 and head_dim > 0
                else None
            ),
            "swa_window": swa_window,
        }

    def estimate_loaded_bytes(
        self, storage_cached_tokens: int, model_info: dict[str, Any]
    ) -> dict[str, Any]:
        bytes_per_token_layer = model_info.get("kv_bytes_per_token_per_layer_total")
        full_layer_count = model_info.get("full_layer_count")
        swa_layer_count = model_info.get("swa_layer_count")
        if (
            bytes_per_token_layer is None
            or full_layer_count is None
            or swa_layer_count is None
        ):
            return {
                "hybrid_estimated_bytes": None,
                "all_layers_if_not_swa_aware_bytes": None,
                "swa_loaded_tokens_est": None,
            }

        swa_window = model_info.get("swa_window")
        if swa_window is None:
            swa_loaded_tokens_est = storage_cached_tokens
        else:
            # SWA fetch should be bounded by the attention window plus one page
            # of alignment/guard overhead, while full layers fetch all cached M.
            swa_loaded_tokens_est = min(
                storage_cached_tokens,
                math.ceil((int(swa_window) + PAGE_SIZE) / PAGE_SIZE) * PAGE_SIZE,
            )
        hybrid_units = (
            storage_cached_tokens * full_layer_count
            + swa_loaded_tokens_est * swa_layer_count
        )
        all_layer_units = storage_cached_tokens * (full_layer_count + swa_layer_count)
        return {
            "hybrid_estimated_bytes": hybrid_units * bytes_per_token_layer,
            "all_layers_if_not_swa_aware_bytes": all_layer_units
            * bytes_per_token_layer,
            "swa_loaded_tokens_est": swa_loaded_tokens_est,
        }

    def merge_results(
        self,
        nocache_rows: list[dict[str, Any]],
        cached_rows: list[dict[str, Any]],
        model_info: dict[str, Any],
    ) -> list[dict[str, Any]]:
        nocache_by_case = {row["case"]: row for row in nocache_rows}
        merged: list[dict[str, Any]] = []
        for cached in cached_rows:
            case = cached["case"]
            base = nocache_by_case[case]
            t1_ms = base["t1_no_cache_m_plus_n"]["ttft_ms"]
            t2_ms = base["t2_no_cache_m"]["ttft_ms"]
            t3_ms = cached["t3_cached_m_plus_n"]["ttft_ms"]
            if t1_ms is None or t2_ms is None or t3_ms is None:
                compute_n_ms = None
                effective_load_ms = None
            else:
                compute_n_ms = t1_ms - t2_ms
                effective_load_ms = t3_ms - compute_n_ms

            storage_cached_tokens = cached["t3_cached_m_plus_n"]["meta"][
                "storage_cached_tokens"
            ]
            byte_estimate = self.estimate_loaded_bytes(
                storage_cached_tokens, model_info
            )
            if effective_load_ms is not None and effective_load_ms > 0:
                token_load_throughput = storage_cached_tokens / (
                    effective_load_ms / 1000.0
                )
                hybrid_bytes = byte_estimate["hybrid_estimated_bytes"]
                all_layer_bytes = byte_estimate["all_layers_if_not_swa_aware_bytes"]
                hybrid_gib_s = (
                    None
                    if hybrid_bytes is None
                    else hybrid_bytes / (1024**3) / (effective_load_ms / 1000.0)
                )
                all_layer_gib_s = (
                    None
                    if all_layer_bytes is None
                    else all_layer_bytes / (1024**3) / (effective_load_ms / 1000.0)
                )
            else:
                token_load_throughput = None
                hybrid_gib_s = None
                all_layer_gib_s = None

            merged.append(
                {
                    "case": case,
                    "repeat_index": cached["repeat_index"],
                    "m_len": cached["m_len"],
                    "n_len": cached["n_len"],
                    "t1_no_cache_m_plus_n_ttft_ms": t1_ms,
                    "t2_no_cache_m_ttft_ms": t2_ms,
                    "t3_cached_m_plus_n_ttft_ms": t3_ms,
                    "compute_n_net_ttft_ms": compute_n_ms,
                    "effective_l3_load_m_ms": effective_load_ms,
                    "effective_load_positive": (
                        effective_load_ms is not None and effective_load_ms > 0
                    ),
                    "storage_cached_tokens": storage_cached_tokens,
                    "total_cached_tokens": cached["t3_cached_m_plus_n"]["meta"][
                        "cached_tokens"
                    ],
                    "token_load_throughput_tok_s": token_load_throughput,
                    "hybrid_estimated_load_gib_s": hybrid_gib_s,
                    "all_layers_if_not_swa_aware_gib_s": all_layer_gib_s,
                    **byte_estimate,
                    "nocache": base,
                    "cached": cached,
                }
            )
        return merged

    def write_effective_report(
        self,
        rows: list[dict[str, Any]],
        model_info: dict[str, Any],
        timing_summary: Optional[dict[str, Any]] = None,
    ) -> None:
        lines = [
            "# PD Mooncake HiCache L3 Effective Load Benchmark",
            "",
            f"- model: `{MODEL_PATH}`",
            f"- served name: `{SERVED_NAME}`",
            f"- topology: PD separation, prefill TP{SERVER_TP}, decode TP{SERVER_TP}",
            f"- M: `{M_LEN}`",
            f"- N list: `{N_LIST}`",
            f"- repeats: `{CASE_REPEATS}`",
            f"- context_length: `{CONTEXT_LENGTH}`, page_size: `{PAGE_SIZE}`",
            f"- measure max_new_tokens: `{MEASURE_MAX_NEW_TOKENS}`",
            f"- populate max_new_tokens: `{POPULATE_MAX_NEW_TOKENS}`",
            f"- PD transfer backend: `{PD_TRANSFER_BACKEND}`, ib device: `{PD_IB_DEVICE}`",
            (
                "- HiCache storage Mooncake: "
                f"protocol=`{MOONCAKE_PROTOCOL}`, device=`{MOONCAKE_DEVICE}`, "
                f"MC_MS_AUTO_DISC=`{MC_MS_AUTO_DISC}`, "
                f"MC_NUM_QP_PER_EP=`{os.environ.get('MC_NUM_QP_PER_EP', '')}`, "
                f"segment_size=`{MOONCAKE_GLOBAL_SEGMENT_SIZE}`, "
                f"storage_batch_size=`{os.environ.get('HICACHE_STORAGE_BATCH_SIZE', '128')}`"
            ),
            (
                "- model layers: "
                f"full={model_info.get('full_layer_count')}, "
                f"swa={model_info.get('swa_layer_count')}, "
                f"swa_window={model_info.get('swa_window')}"
            ),
            "",
            "Formula: `compute_N = t1 - t2`; `ttft_residual_load_estimate = t3 - compute_N`.",
            "TTFT is measured by streaming first-token latency with one generated token.",
            "",
            "| repeat | N | t1 no-cache M+N TTFT ms | t2 no-cache M TTFT ms | t3 cached M+N TTFT ms | compute N ms | TTFT residual load est ms | storage cached | token load tok/s | hybrid GiB/s est | all-layer GiB/s est |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                "| {repeat} | {n} | {t1:.2f} | {t2:.2f} | {t3:.2f} | {compute:.2f} | {load:.2f} | {storage} | {tok_s} | {hybrid} | {all_layer} |".format(
                    repeat=row["repeat_index"],
                    n=row["n_len"],
                    t1=row["t1_no_cache_m_plus_n_ttft_ms"] or 0.0,
                    t2=row["t2_no_cache_m_ttft_ms"] or 0.0,
                    t3=row["t3_cached_m_plus_n_ttft_ms"] or 0.0,
                    compute=row["compute_n_net_ttft_ms"] or 0.0,
                    load=row["effective_l3_load_m_ms"] or 0.0,
                    storage=row["storage_cached_tokens"],
                    tok_s=(
                        "n/a"
                        if row["token_load_throughput_tok_s"] is None
                        else f"{row['token_load_throughput_tok_s']:.2f}"
                    ),
                    hybrid=(
                        "n/a"
                        if row["hybrid_estimated_load_gib_s"] is None
                        else f"{row['hybrid_estimated_load_gib_s']:.2f}"
                    ),
                    all_layer=(
                        "n/a"
                        if row["all_layers_if_not_swa_aware_gib_s"] is None
                        else f"{row['all_layers_if_not_swa_aware_gib_s']:.2f}"
                    ),
                )
            )

        lines.extend(["", "## Aggregates", ""])
        for n_len in N_LIST:
            group = [row for row in rows if row["n_len"] == n_len]
            load_ms = [
                row["effective_l3_load_m_ms"]
                for row in group
                if row["effective_l3_load_m_ms"] is not None
            ]
            tok_s = [
                row["token_load_throughput_tok_s"]
                for row in group
                if row["token_load_throughput_tok_s"] is not None
            ]
            hybrid = [
                row["hybrid_estimated_load_gib_s"]
                for row in group
                if row["hybrid_estimated_load_gib_s"] is not None
            ]
            lines.append(
                "- N={n}: load_ms={load}, token_load_tok_s={tok}, hybrid_GiB_s_est={hybrid}".format(
                    n=n_len,
                    load=json.dumps(summarize_values(load_ms), sort_keys=True),
                    tok=json.dumps(summarize_values(tok_s), sort_keys=True),
                    hybrid=json.dumps(summarize_values(hybrid), sort_keys=True),
                )
            )

        if timing_summary is not None:
            lines.extend(["", "## HiCache Timing Instrumentation", ""])
            lines.append(
                f"- parsed timing events: `{timing_summary.get('event_count', 0)}`"
            )
            lines.append(
                "- `mooncake_batch_get` measures Mooncake `batch_get_into` host-to-host transfer time."
            )
            lines.append(
                "- `hicache_prefetch_request` measures request-level prefetch from operation enqueue to transfer completion."
            )
            lines.append(
                "- `hicache_load_to_device` measures local host-memory to GPU-memory load time with CUDA timing events."
            )
            lines.append(
                "- `pd_prefill_forward`, `pd_prefill_kv_transfer`, `pd_decode_prealloc`, `pd_decode_kv_arrival`, `pd_decode_prebuilt`, and `pd_decode_forward` break down the PD path after cache load."
            )
            m_summary = timing_summary.get("m_len_prefetch_request") or {}
            lines.append(
                f"- M-length prefetch request events: `{timing_summary.get('m_len_prefetch_request_count', 0)}`"
            )
            lines.append(
                "- M-length request timing: "
                f"transfer_ms={json.dumps(m_summary.get('transfer_elapsed_ms'), sort_keys=True)}, "
                f"request_e2e_ms={json.dumps(m_summary.get('request_e2e_ms'), sort_keys=True)}, "
                f"total_bytes={json.dumps(m_summary.get('total_bytes'), sort_keys=True)}, "
                f"transfer_GiB_s={json.dumps(m_summary.get('transfer_gib_per_s'), sort_keys=True)}, "
                f"request_e2e_GiB_s={json.dumps(m_summary.get('request_e2e_gib_per_s'), sort_keys=True)}"
            )
            m_load_summary = timing_summary.get("m_len_load_to_device") or {}
            lines.append(
                f"- M-length load-to-device events: `{timing_summary.get('m_len_load_to_device_count', 0)}`"
            )
            lines.append(
                "- M-length host-to-device timing: "
                f"elapsed_ms={json.dumps(m_load_summary.get('elapsed_ms'), sort_keys=True)}, "
                f"total_bytes={json.dumps(m_load_summary.get('total_bytes'), sort_keys=True)}, "
                f"GiB_s={json.dumps(m_load_summary.get('gib_per_s'), sort_keys=True)}"
            )
            lines.extend(
                [
                    "",
                    "| event/interface/pool | count | bytes sum | elapsed/transfer ms avg | GiB/s avg |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for key, group_summary in sorted(
                (timing_summary.get("groups") or {}).items()
            ):
                elapsed_avg = (group_summary.get("elapsed_ms") or {}).get("avg")
                gib_avg = (group_summary.get("gib_per_s") or {}).get("avg")
                lines.append(
                    "| {key} | {count} | {bytes_sum} | {elapsed} | {gib} |".format(
                        key=key,
                        count=group_summary.get("count"),
                        bytes_sum=group_summary.get("bytes_sum"),
                        elapsed="n/a" if elapsed_avg is None else f"{elapsed_avg:.3f}",
                        gib="n/a" if gib_avg is None else f"{gib_avg:.3f}",
                    )
                )

        lines.extend(
            [
                "",
                "## Notes",
                "",
                "- `ttft_residual_load_estimate` is a TTFT residual estimate, not a pure Mooncake bandwidth counter.",
                "- For pure cache IO, use the direct `hicache_prefetch_request` and `hicache_load_to_device` events, preferably max across TP ranks for the same request.",
                "- Hybrid GiB/s uses full-layer KV for all cached M tokens and SWA-layer KV bounded by `swa_window + page_size`.",
                "- The all-layer estimate shows the numerator that would be used if SWA layers fetched the whole cached prefix.",
            ]
        )
        (self.run_dir / "report.md").write_text("\n".join(lines) + "\n")

    def run(self) -> None:
        self.log(f"run_dir={self.run_dir}")
        self.log(f"sglang_import={self._sglang_import_path()}")
        self.log(f"config_info={json.dumps(self.config_info(), sort_keys=True)}")
        model_info = self.model_transfer_info()
        atomic_json(self.run_dir / "model_transfer_info.json", model_info)

        nocache_rows = self.run_nocache_mode()
        atomic_json(self.run_dir / "nocache_results.json", nocache_rows)

        cached_rows = self.run_cached_mode()
        atomic_json(self.run_dir / "cached_l3_results.json", cached_rows)

        rows = self.merge_results(nocache_rows, cached_rows, model_info)
        timing_events = self.collect_hicache_timing_events()
        timing_summary = self.summarize_hicache_timing_events(timing_events)
        atomic_json(self.run_dir / "hicache_timing_events.json", timing_events)
        atomic_json(self.run_dir / "hicache_timing_summary.json", timing_summary)
        summary = {
            "config": {
                "model": MODEL_PATH,
                "served_name": SERVED_NAME,
                "m_len": M_LEN,
                "n_list": N_LIST,
                "case_repeats": CASE_REPEATS,
                "measure_max_new_tokens": MEASURE_MAX_NEW_TOKENS,
                "populate_max_new_tokens": POPULATE_MAX_NEW_TOKENS,
                "context_length": CONTEXT_LENGTH,
                "page_size": PAGE_SIZE,
                "server_tp": SERVER_TP,
                "host": HOST,
                "pd_transfer_backend": PD_TRANSFER_BACKEND,
                "pd_ib_device": PD_IB_DEVICE,
                "mooncake_protocol": MOONCAKE_PROTOCOL,
                "mooncake_device": MOONCAKE_DEVICE,
                "mc_ms_auto_disc": MC_MS_AUTO_DISC,
                "mc_num_qp_per_ep": os.environ.get("MC_NUM_QP_PER_EP", ""),
                "mooncake_global_segment_size": MOONCAKE_GLOBAL_SEGMENT_SIZE,
                "hicache_storage_batch_size": os.environ.get(
                    "HICACHE_STORAGE_BATCH_SIZE", "128"
                ),
            },
            "model_transfer_info": model_info,
            "hicache_timing_summary": timing_summary,
            "results": rows,
        }
        atomic_json(self.run_dir / "summary.json", summary)
        atomic_json(self.run_dir / "results.json", rows)
        self.write_effective_report(rows, model_info, timing_summary)
        self.log("EFFECTIVE_LOAD_DONE")


def main() -> None:
    run_dir = Path(
        os.environ.get("RUN_DIR", "/tmp/welmv45-pd-mooncake-l3-effective-load")
    )
    bench = PDMooncakeL3EffectiveLoad(run_dir)
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
