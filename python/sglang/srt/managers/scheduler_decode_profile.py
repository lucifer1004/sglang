from __future__ import annotations

import atexit
import json
import os
import time
from typing import Any

_PATH = os.environ.get("SGLANG_DECODE_SCHEDULER_TIMING_PATH")
_FILE: str | None = None
_RECORDS: list[dict[str, Any]] = []


def decode_scheduler_profile_enabled() -> bool:
    return bool(_PATH)


def _resolve_file() -> str:
    global _FILE
    if _FILE is not None:
        return _FILE

    assert _PATH is not None
    path = _PATH
    if path.endswith(os.sep) or os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, f"decode_scheduler_timing_{os.getpid()}.jsonl")
    else:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        root, ext = os.path.splitext(path)
        path = f"{root}_{os.getpid()}{ext or '.jsonl'}"
    _FILE = path
    return path


def _flush() -> None:
    if not _RECORDS or not _PATH:
        return
    path = _resolve_file()
    with open(path, "a", encoding="utf-8") as f:
        for record in _RECORDS:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
    _RECORDS.clear()


if _PATH:
    atexit.register(_flush)


def _batch_summary(batch: Any) -> dict[str, Any]:
    if batch is None:
        return {}
    reqs = getattr(batch, "reqs", None) or []
    seq_lens: list[int] = []
    output_lens: list[int] = []
    for req in reqs:
        try:
            seq_lens.append(int(req.seqlen))
        except Exception:  # noqa: BLE001
            pass
        try:
            output_lens.append(len(req.output_ids))
        except Exception:  # noqa: BLE001
            pass

    summary: dict[str, Any] = {
        "batch_id": id(batch),
        "batch_size": len(reqs),
        "forward_iter": getattr(batch, "forward_iter", None),
    }
    forward_mode = getattr(batch, "forward_mode", None)
    if forward_mode is not None:
        summary["forward_mode"] = str(forward_mode)
    if seq_lens:
        summary["seq_min"] = min(seq_lens)
        summary["seq_max"] = max(seq_lens)
    if output_lens:
        summary["output_len_min"] = min(output_lens)
        summary["output_len_max"] = max(output_lens)
    return summary


def record_decode_scheduler_event(
    event: str,
    scheduler: Any | None = None,
    *,
    batch: Any | None = None,
    start_ns: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    if not _PATH:
        return

    now_ns = time.perf_counter_ns()
    record: dict[str, Any] = {
        "event": event,
        "pid": os.getpid(),
        "monotonic_ns": now_ns,
    }
    if start_ns is not None:
        record["duration_ms"] = (now_ns - start_ns) / 1.0e6
    if scheduler is not None:
        for name in (
            "tp_rank",
            "attn_tp_rank",
            "attn_cp_rank",
            "dp_rank",
            "gpu_id",
            "forward_ct",
            "forward_ct_decode",
        ):
            if hasattr(scheduler, name):
                value = getattr(scheduler, name)
                if isinstance(value, (int, float, str, bool)) or value is None:
                    record[name] = value
        for name, attr in (
            ("waiting_queue_len", "waiting_queue"),
            ("result_queue_len", "result_queue"),
        ):
            if hasattr(scheduler, attr):
                try:
                    record[name] = len(getattr(scheduler, attr))
                except Exception:  # noqa: BLE001
                    pass
        running_batch = getattr(scheduler, "running_batch", None)
        if running_batch is not None:
            try:
                record["running_batch_size"] = running_batch.batch_size()
            except Exception:  # noqa: BLE001
                pass
    record.update(_batch_summary(batch))
    if extra:
        record.update(extra)

    _RECORDS.append(record)
    if len(_RECORDS) >= 64:
        _flush()
