import argparse
import ctypes
import hashlib
import http.server
import json
import logging
import mmap
import os
import re
import statistics
import sys
import time
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("bench_mooncake_raw_transfer")

HICACHE_PAGE_KV_SPLIT_LAYOUT = "hicache_page_kv_split"


@dataclass
class PageLayout:
    model_path: str
    config_path: str
    config_sha256: str
    tp_size: int
    page_size: int
    num_hidden_layers: int
    main_hidden_layer_count: int
    num_nextn_predict_layers: int
    include_nextn_kv: bool
    effective_kv_layer_count: int
    total_kv_heads: int
    local_kv_heads: int
    head_dim: int
    dtype: str
    dtype_bytes: int
    objects_per_page: int
    k_object_bytes: int
    v_object_bytes: int
    logical_page_bytes: int
    kv_bytes_per_token: int
    shape_per_k_object: List[int]
    shape_per_v_object: List[int]
    object_layout: str = "page_kv_split_all_layers"
    architectures: List[str] = field(default_factory=list)
    layer_types: List[str] = field(default_factory=list)
    sliding_window_size_layerwise: List[int] = field(default_factory=list)
    layer_windows_used: List[int] = field(default_factory=list)
    layer_window_summary: Dict[str, int] = field(default_factory=dict)
    is_hybrid_swa: bool = False
    sliding_window_size: Optional[int] = None
    full_attention_layer_ids: List[int] = field(default_factory=list)
    swa_attention_layer_ids: List[int] = field(default_factory=list)
    full_layer_count: int = 0
    swa_layer_count: int = 0
    v_head_dim: Optional[int] = None
    swa_total_kv_heads: Optional[int] = None
    swa_local_kv_heads: Optional[int] = None
    swa_head_dim: Optional[int] = None
    swa_v_head_dim: Optional[int] = None
    full_object_bytes: int = 0
    swa_object_bytes: int = 0
    full_bytes_per_token_per_layer: int = 0
    swa_bytes_per_token_per_layer: int = 0
    shape_per_full_layer_object: List[int] = field(default_factory=list)
    shape_per_swa_layer_object: List[int] = field(default_factory=list)
    layout_warning: str = ""


@dataclass
class BenchResult:
    direction: str
    prompt_tokens: int
    page_size: int
    repeat: int
    num_pages: int
    object_count: int
    object_bytes: int
    k_object_bytes: int
    v_object_bytes: int
    logical_page_bytes: int
    total_payload_bytes: int
    latency_ms: float
    bandwidth_gib_s: float
    ok: bool
    result_count: int
    success_count: int
    failure_count: int
    short_count: int
    min_result: int
    max_result: int
    chunks: int
    raw_results: Any
    result_counts: Dict[str, int]
    full_layer_count: int = 0
    swa_layer_count: int = 0
    swa_pages: int = 0
    per_layer_page_count: Dict[str, int] = field(default_factory=dict)
    per_layer_window: Dict[str, int] = field(default_factory=dict)
    object_bytes_by_layer_type: Dict[str, int] = field(default_factory=dict)
    useful_payload_bytes: int = 0
    cuda_sample_ok: Optional[bool] = None
    cuda_sample_count: int = 0
    cuda_sample_bad: int = 0
    cuda_sample_bytes: int = 0
    cuda_sample_offsets: List[int] = field(default_factory=list)
    full_object_count: int = 0
    object_slice_index: int = 0
    object_slice_count: int = 1
    object_slice_start: int = 0
    object_slice_end: int = 0
    call_count: int = 0
    call_breakdown: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ObjectSpec:
    key: str
    offset: int
    size: int
    layer_id: int = -1
    layer_type: str = ""
    page_id: int = -1


class RawBuffer:
    def __init__(self, backing, ptr: int, size_bytes: int, allocator: str):
        self.backing = backing
        self.ptr = ptr
        self.size_bytes = size_bytes
        self.allocator = allocator

    def data_ptr(self) -> int:
        return self.ptr


class ReadyHandler(http.server.BaseHTTPRequestHandler):
    payload: Dict[str, Any] = {}

    def do_GET(self):
        if self.path != "/ready":
            self.send_error(404)
            return
        body = json.dumps(self.payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        logger.info("http: " + fmt, *args)


def parse_ks(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def normalize_dtype_name(name: Optional[str]) -> str:
    if not name:
        return "bfloat16"
    name = str(name).replace("torch.", "").lower()
    aliases = {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "float16": "float16",
        "half": "float16",
        "fp32": "float32",
        "float32": "float32",
        "uint8": "uint8",
    }
    if name not in aliases:
        raise ValueError(f"unsupported dtype from model config: {name}")
    return aliases[name]


def dtype_from_name(name: str):
    import torch

    name = normalize_dtype_name(name)
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    if name == "uint8":
        return torch.uint8
    raise ValueError(f"unsupported dtype: {name}")


def dtype_nbytes(name: str) -> int:
    name = normalize_dtype_name(name)
    nbytes = {
        "bfloat16": 2,
        "float16": 2,
        "float32": 4,
        "uint8": 1,
    }
    return nbytes[name]


def resolve_config_path(model_path: str) -> Path:
    path = Path(model_path)
    config_path = path if path.is_file() else path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"model config not found: {config_path}")
    return config_path


def load_model_config(model_path: str) -> Dict[str, Any]:
    config_path = resolve_config_path(model_path)
    return json.loads(config_path.read_text())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_get(config: Dict[str, Any], name: str, default: Optional[Any] = None) -> Any:
    if name in config:
        return config[name]
    text_config = config.get("text_config")
    if isinstance(text_config, dict) and name in text_config:
        return text_config[name]
    return default


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def as_int_list(value: Any) -> List[int]:
    return [int(item) for item in as_list(value)]


def first_config_value(config: Dict[str, Any], names: List[str]) -> Any:
    for name in names:
        value = config_get(config, name)
        if value is not None:
            return value
    return None


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def derive_attention_layer_ids(
    config: Dict[str, Any], num_layers: int
) -> Tuple[List[int], List[int], List[str], str]:
    explicit_full = first_config_value(
        config,
        [
            "full_attention_layer_ids",
            "full_attn_layer_ids",
            "full_layers",
        ],
    )
    explicit_swa = first_config_value(
        config,
        [
            "swa_attention_layer_ids",
            "sliding_window_layer_ids",
            "sliding_attention_layer_ids",
            "swa_layers",
        ],
    )
    if explicit_full is not None or explicit_swa is not None:
        swa_ids = sorted(set(as_int_list(explicit_swa)))
        if explicit_full is None:
            full_ids = [idx for idx in range(num_layers) if idx not in set(swa_ids)]
        else:
            full_ids = sorted(set(as_int_list(explicit_full)))
        return full_ids, swa_ids, [], ""

    raw_layer_types = as_list(config_get(config, "layer_types"))
    layer_types = [str(item) for item in raw_layer_types]
    if len(layer_types) == num_layers:
        swa_names = {
            "sliding_attention",
            "sliding_window_attention",
            "swa",
            "swa_attention",
        }
        full_names = {
            "full_attention",
            "full",
            "global_attention",
            "attention",
            "mha",
        }
        full_ids = []
        swa_ids = []
        unknown = []
        for idx, layer_type in enumerate(layer_types):
            normalized = layer_type.lower()
            if normalized in swa_names:
                swa_ids.append(idx)
            elif normalized in full_names:
                full_ids.append(idx)
            else:
                unknown.append(layer_type)
        if unknown:
            warning = (
                "unrecognized layer_types values: "
                + ",".join(sorted(set(unknown)))
                + "; treated as full attention"
            )
            for idx, layer_type in enumerate(layer_types):
                if idx not in set(swa_ids):
                    full_ids.append(idx)
            full_ids = sorted(set(full_ids))
            return full_ids, swa_ids, layer_types, warning
        return full_ids, swa_ids, layer_types, ""

    hybrid_pattern = as_list(config_get(config, "hybrid_layer_pattern"))
    if len(hybrid_pattern) == num_layers:
        # SGLang's SWA helpers use 1 for sliding-window layers in this pattern.
        swa_ids = [idx for idx, value in enumerate(hybrid_pattern) if int(value) == 1]
        full_ids = [idx for idx in range(num_layers) if idx not in set(swa_ids)]
        layer_types = [
            "sliding_attention" if idx in set(swa_ids) else "full_attention"
            for idx in range(num_layers)
        ]
        return full_ids, swa_ids, layer_types, ""

    return list(range(num_layers)), [], layer_types, ""


def summarize_ints(values: List[int]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for value in values:
        key = str(int(value))
        summary[key] = summary.get(key, 0) + 1
    return dict(sorted(summary.items(), key=lambda item: int(item[0])))


def derive_layout(args: argparse.Namespace) -> PageLayout:
    config_path = resolve_config_path(args.model_path)
    config = json.loads(config_path.read_text())
    config_sha = file_sha256(config_path)
    num_layers = int(config_get(config, "num_hidden_layers"))
    num_nextn_predict_layers = int(
        config_get(config, "num_nextn_predict_layers", 0) or 0
    )
    effective_kv_layer_count = num_layers + (
        num_nextn_predict_layers if args.include_nextn_kv else 0
    )
    num_attention_heads = int(config_get(config, "num_attention_heads"))
    total_kv_heads = int(config_get(config, "num_key_value_heads", num_attention_heads))
    head_dim = config_get(config, "head_dim")
    if head_dim is None:
        hidden_size = int(config_get(config, "hidden_size"))
        head_dim = hidden_size // num_attention_heads
    head_dim = int(head_dim)
    dtype = normalize_dtype_name(
        args.dtype or config_get(config, "torch_dtype") or config_get(config, "dtype")
    )
    dtype_bytes = dtype_nbytes(dtype)

    # Match SGLang ModelConfig.get_num_kv_heads(): KV heads are replicated when
    # total KV heads are fewer than the attention TP size.
    local_kv_heads = max(1, total_kv_heads // args.tp_size)
    v_head_dim = int(config_get(config, "v_head_dim", head_dim))
    k_object_bytes = (
        num_layers * local_kv_heads * head_dim * dtype_bytes * args.page_size
    )
    v_object_bytes = (
        num_layers * local_kv_heads * v_head_dim * dtype_bytes * args.page_size
    )
    logical_page_bytes = k_object_bytes + v_object_bytes
    kv_bytes_per_token = logical_page_bytes // args.page_size

    architectures = [str(item) for item in as_list(config_get(config, "architectures"))]
    is_hybrid_swa = bool(config_get(config, "is_hybrid_swa", False))
    sliding_window_value = first_config_value(
        config,
        ["sliding_window_size", "sliding_window", "window_size"],
    )
    sliding_window_size = (
        int(sliding_window_value) if sliding_window_value is not None else None
    )
    full_ids: List[int]
    swa_ids: List[int]
    layer_types: List[str]
    layout_warning = ""
    sliding_window_size_layerwise = as_int_list(
        config_get(config, "sliding_window_size_layerwise")
    )
    layer_windows_used: List[int] = []
    uses_layer_window_layout = args.object_layout in (
        "layer_page_kv_combined",
        HICACHE_PAGE_KV_SPLIT_LAYOUT,
    )
    if uses_layer_window_layout and sliding_window_size_layerwise:
        if len(sliding_window_size_layerwise) < effective_kv_layer_count:
            raise ValueError(
                "sliding_window_size_layerwise has "
                f"{len(sliding_window_size_layerwise)} entries, but "
                f"{effective_kv_layer_count} KV layers are requested"
            )
        layer_windows_used = sliding_window_size_layerwise[:effective_kv_layer_count]
        max_window = max(layer_windows_used)
        full_ids = [
            idx for idx, window in enumerate(layer_windows_used) if window == max_window
        ]
        swa_ids = [
            idx for idx, window in enumerate(layer_windows_used) if window != max_window
        ]
        layer_types = [
            "full_window" if window == max_window else f"window_{window}"
            for window in layer_windows_used
        ]
        non_full_windows = [
            window for window in layer_windows_used if window != max_window
        ]
        if non_full_windows:
            sliding_window_size = min(non_full_windows)
        else:
            sliding_window_size = None
        if (
            not args.include_nextn_kv
            and num_nextn_predict_layers
            and len(sliding_window_size_layerwise)
            >= num_layers + num_nextn_predict_layers
        ):
            layout_warning = (
                f"excluded {num_nextn_predict_layers} nextn layer window entries "
                f"from {args.object_layout} layout"
            )
    else:
        full_ids, swa_ids, layer_types, layout_warning = derive_attention_layer_ids(
            config, num_layers
        )
        layer_windows_used = []
        if (
            sliding_window_size is not None
            and not swa_ids
            and not layer_types
            and not layout_warning
        ):
            layout_warning = (
                "sliding_window is present but no layer_types or SWA layer ids were "
                "found; treating all layers as full attention"
            )
        if swa_ids and sliding_window_size is None:
            raise ValueError(
                "SWA layers were detected, but no sliding_window_size/sliding_window/window_size "
                "field exists in the model config."
            )

    swa_total_kv_heads_value = config_get(config, "swa_num_key_value_heads")
    swa_total_kv_heads = (
        int(swa_total_kv_heads_value)
        if swa_total_kv_heads_value is not None
        else total_kv_heads
    )
    swa_local_kv_heads = max(1, swa_total_kv_heads // args.tp_size)
    swa_head_dim = int(config_get(config, "swa_head_dim", head_dim))
    swa_v_head_dim = int(config_get(config, "swa_v_head_dim", v_head_dim))
    full_object_bytes = (
        local_kv_heads * (head_dim + v_head_dim) * dtype_bytes * args.page_size
    )
    swa_object_bytes = (
        swa_local_kv_heads
        * (swa_head_dim + swa_v_head_dim)
        * dtype_bytes
        * args.page_size
    )
    page_k_object_bytes = k_object_bytes
    page_v_object_bytes = v_object_bytes
    if args.object_layout == HICACHE_PAGE_KV_SPLIT_LAYOUT:
        page_k_object_bytes = (
            len(full_ids) * local_kv_heads * head_dim * dtype_bytes * args.page_size
        )
        page_v_object_bytes = (
            len(full_ids) * local_kv_heads * v_head_dim * dtype_bytes * args.page_size
        )
        logical_page_bytes = page_k_object_bytes + page_v_object_bytes
        kv_bytes_per_token = logical_page_bytes // args.page_size

    return PageLayout(
        model_path=args.model_path,
        config_path=str(config_path),
        config_sha256=config_sha,
        tp_size=args.tp_size,
        page_size=args.page_size,
        num_hidden_layers=num_layers,
        main_hidden_layer_count=num_layers,
        num_nextn_predict_layers=num_nextn_predict_layers,
        include_nextn_kv=args.include_nextn_kv,
        effective_kv_layer_count=effective_kv_layer_count
        if uses_layer_window_layout
        else num_layers,
        total_kv_heads=total_kv_heads,
        local_kv_heads=local_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        dtype_bytes=dtype_bytes,
        objects_per_page=2,
        k_object_bytes=page_k_object_bytes,
        v_object_bytes=page_v_object_bytes,
        logical_page_bytes=logical_page_bytes,
        kv_bytes_per_token=kv_bytes_per_token,
        shape_per_k_object=[
            len(full_ids)
            if args.object_layout == HICACHE_PAGE_KV_SPLIT_LAYOUT
            else num_layers,
            args.page_size,
            local_kv_heads,
            head_dim,
        ],
        shape_per_v_object=[
            len(full_ids)
            if args.object_layout == HICACHE_PAGE_KV_SPLIT_LAYOUT
            else num_layers,
            args.page_size,
            local_kv_heads,
            v_head_dim,
        ],
        object_layout=args.object_layout,
        architectures=architectures,
        layer_types=layer_types,
        sliding_window_size_layerwise=sliding_window_size_layerwise,
        layer_windows_used=layer_windows_used,
        layer_window_summary=summarize_ints(layer_windows_used),
        is_hybrid_swa=is_hybrid_swa,
        sliding_window_size=sliding_window_size,
        full_attention_layer_ids=full_ids,
        swa_attention_layer_ids=swa_ids,
        full_layer_count=len(full_ids),
        swa_layer_count=len(swa_ids),
        v_head_dim=v_head_dim,
        swa_total_kv_heads=swa_total_kv_heads,
        swa_local_kv_heads=swa_local_kv_heads,
        swa_head_dim=swa_head_dim,
        swa_v_head_dim=swa_v_head_dim,
        full_object_bytes=full_object_bytes,
        swa_object_bytes=swa_object_bytes,
        full_bytes_per_token_per_layer=full_object_bytes // args.page_size,
        swa_bytes_per_token_per_layer=swa_object_bytes // args.page_size,
        shape_per_full_layer_object=[2, args.page_size, local_kv_heads, head_dim],
        shape_per_swa_layer_object=[
            2,
            args.page_size,
            swa_local_kv_heads,
            swa_head_dim,
        ],
        layout_warning=layout_warning,
    )


def effective_device_name(args: argparse.Namespace) -> str:
    return args.device_name or os.environ.get("MOONCAKE_DEVICE", "")


def rail_count(args: argparse.Namespace) -> int:
    device_name = effective_device_name(args)
    if not device_name:
        return 0
    return len([item for item in device_name.split(",") if item.strip()])


def validate_environment(args: argparse.Namespace) -> None:
    if args.fail_on_auto_disc and os.environ.get("MC_MS_AUTO_DISC") == "1":
        raise RuntimeError(
            "MC_MS_AUTO_DISC=1 would override explicit Mooncake devices; unset it."
        )
    if args.require_explicit_device and not effective_device_name(args):
        raise RuntimeError(
            "--require-explicit-device needs --device-name or MOONCAKE_DEVICE"
        )


def num_pages_for_tokens(layout: PageLayout, token_count: int) -> int:
    if token_count % layout.page_size != 0:
        raise ValueError(
            f"K={token_count} must be divisible by page_size={layout.page_size}"
        )
    return token_count // layout.page_size


def layer_type_for_id(layout: PageLayout, layer_id: int) -> str:
    if layer_id in set(layout.full_attention_layer_ids):
        return "full"
    if layer_id in set(layout.swa_attention_layer_ids):
        return "swa"
    if 0 <= layer_id < len(layout.layer_windows_used):
        return f"window_{layout.layer_windows_used[layer_id]}"
    return "layer"


def object_bytes_for_layer(layout: PageLayout, layer_id: int) -> int:
    if layer_id in set(layout.swa_attention_layer_ids):
        return layout.swa_object_bytes
    return layout.full_object_bytes


def bytes_per_token_for_layer(layout: PageLayout, layer_id: int) -> int:
    if layer_id in set(layout.swa_attention_layer_ids):
        return layout.swa_bytes_per_token_per_layer
    return layout.full_bytes_per_token_per_layer


def hicache_main_k_object_bytes(layout: PageLayout) -> int:
    return (
        layout.full_layer_count
        * layout.local_kv_heads
        * layout.head_dim
        * layout.dtype_bytes
        * layout.page_size
    )


def hicache_main_v_object_bytes(layout: PageLayout) -> int:
    return (
        layout.full_layer_count
        * layout.local_kv_heads
        * int(layout.v_head_dim or layout.head_dim)
        * layout.dtype_bytes
        * layout.page_size
    )


def hicache_swa_k_object_bytes(layout: PageLayout) -> int:
    return (
        layout.swa_layer_count
        * int(layout.swa_local_kv_heads or layout.local_kv_heads)
        * int(layout.swa_head_dim or layout.head_dim)
        * layout.dtype_bytes
        * layout.page_size
    )


def hicache_swa_v_object_bytes(layout: PageLayout) -> int:
    return (
        layout.swa_layer_count
        * int(layout.swa_local_kv_heads or layout.local_kv_heads)
        * int(layout.swa_v_head_dim or layout.v_head_dim or layout.head_dim)
        * layout.dtype_bytes
        * layout.page_size
    )


def hicache_swa_pages_for_tokens(layout: PageLayout, token_count: int) -> int:
    if layout.swa_layer_count <= 0:
        return 0
    if layout.sliding_window_size is None:
        raise ValueError("hicache_page_kv_split with SWA layers needs sliding window")
    return min(
        num_pages_for_tokens(layout, token_count),
        ceil_div(layout.sliding_window_size, layout.page_size),
    )


def layer_pages_for_tokens(layout: PageLayout, token_count: int) -> Dict[int, int]:
    num_pages = num_pages_for_tokens(layout, token_count)
    if layout.object_layout != "layer_page_kv_combined":
        return {}

    if layout.layer_windows_used:
        return {
            layer_id: min(num_pages, ceil_div(max(0, int(window)), layout.page_size))
            for layer_id, window in enumerate(layout.layer_windows_used)
        }

    pages: Dict[int, int] = {
        layer_id: num_pages for layer_id in layout.full_attention_layer_ids
    }
    if layout.swa_layer_count:
        if layout.sliding_window_size is None:
            raise ValueError("SWA layer layout needs sliding_window_size")
        swa_pages = min(
            num_pages, ceil_div(layout.sliding_window_size, layout.page_size)
        )
        for layer_id in layout.swa_attention_layer_ids:
            pages[layer_id] = swa_pages
    return pages


def layer_windows_for_payload(layout: PageLayout, token_count: int) -> Dict[int, int]:
    if layout.layer_windows_used:
        return {
            layer_id: int(window)
            for layer_id, window in enumerate(layout.layer_windows_used)
        }
    windows = {layer_id: token_count for layer_id in layout.full_attention_layer_ids}
    if layout.swa_layer_count and layout.sliding_window_size is not None:
        for layer_id in layout.swa_attention_layer_ids:
            windows[layer_id] = layout.sliding_window_size
    return windows


def transfer_info_for_tokens(layout: PageLayout, token_count: int) -> Dict[str, Any]:
    num_pages = num_pages_for_tokens(layout, token_count)
    if layout.object_layout == "page_kv_split_all_layers":
        object_count = num_pages * layout.objects_per_page
        total_payload_bytes = num_pages * layout.logical_page_bytes
        return {
            "num_pages": num_pages,
            "swa_pages": 0,
            "object_count": object_count,
            "object_bytes": layout.k_object_bytes
            if layout.k_object_bytes == layout.v_object_bytes
            else 0,
            "object_bytes_by_layer_type": {
                "k": layout.k_object_bytes,
                "v": layout.v_object_bytes,
            },
            "total_transfer_payload_bytes": total_payload_bytes,
            "useful_payload_bytes": total_payload_bytes,
        }

    if layout.object_layout == HICACHE_PAGE_KV_SPLIT_LAYOUT:
        main_k_bytes = hicache_main_k_object_bytes(layout)
        main_v_bytes = hicache_main_v_object_bytes(layout)
        swa_k_bytes = hicache_swa_k_object_bytes(layout)
        swa_v_bytes = hicache_swa_v_object_bytes(layout)
        swa_pages = hicache_swa_pages_for_tokens(layout, token_count)
        main_payload = num_pages * (main_k_bytes + main_v_bytes)
        swa_payload = swa_pages * (swa_k_bytes + swa_v_bytes)
        object_sizes = {main_k_bytes, main_v_bytes}
        object_bytes_by_layer_type = {
            "kv_k": main_k_bytes,
            "kv_v": main_v_bytes,
        }
        if swa_pages:
            object_sizes.update({swa_k_bytes, swa_v_bytes})
            object_bytes_by_layer_type.update(
                {
                    "swa_k": swa_k_bytes,
                    "swa_v": swa_v_bytes,
                }
            )
        total_payload_bytes = main_payload + swa_payload
        return {
            "num_pages": num_pages,
            "swa_pages": swa_pages,
            "main_pages": num_pages,
            "main_object_count": num_pages * 2,
            "swa_object_count": swa_pages * 2,
            "object_count": num_pages * 2 + swa_pages * 2,
            "object_bytes": object_sizes.pop() if len(object_sizes) == 1 else 0,
            "object_bytes_by_layer_type": object_bytes_by_layer_type,
            "main_payload_bytes": main_payload,
            "swa_payload_bytes": swa_payload,
            "total_transfer_payload_bytes": total_payload_bytes,
            "useful_payload_bytes": total_payload_bytes,
        }

    if layout.object_layout != "layer_page_kv_combined":
        raise ValueError(f"unsupported object_layout: {layout.object_layout}")

    per_layer_pages = layer_pages_for_tokens(layout, token_count)
    per_layer_windows = layer_windows_for_payload(layout, token_count)
    total_payload = 0
    useful_payload = 0
    object_count = 0
    object_bytes_by_layer_type: Dict[str, int] = {}
    for layer_id in sorted(per_layer_pages):
        pages = int(per_layer_pages[layer_id])
        object_bytes = object_bytes_for_layer(layout, layer_id)
        layer_type = layer_type_for_id(layout, layer_id)
        object_count += pages
        total_payload += pages * object_bytes
        useful_tokens = min(
            token_count, int(per_layer_windows.get(layer_id, token_count))
        )
        useful_payload += useful_tokens * bytes_per_token_for_layer(layout, layer_id)
        object_bytes_by_layer_type[layer_type] = object_bytes
    swa_pages = max(
        [per_layer_pages[layer_id] for layer_id in layout.swa_attention_layer_ids]
        or [0]
    )
    object_bytes_values = set(object_bytes_by_layer_type.values())
    return {
        "num_pages": num_pages,
        "swa_pages": swa_pages,
        "per_layer_page_count": {str(k): v for k, v in per_layer_pages.items()},
        "per_layer_window": {str(k): v for k, v in per_layer_windows.items()},
        "object_count": object_count,
        "object_bytes": object_bytes_values.pop()
        if len(object_bytes_values) == 1
        else 0,
        "object_bytes_by_layer_type": object_bytes_by_layer_type,
        "total_transfer_payload_bytes": total_payload,
        "useful_payload_bytes": useful_payload,
    }


def object_count_for_tokens(layout: PageLayout, token_count: int) -> int:
    return int(transfer_info_for_tokens(layout, token_count)["object_count"])


def object_slice_bounds(
    layout: PageLayout, token_count: int, args: argparse.Namespace
) -> Tuple[int, int, int]:
    full_count = object_count_for_tokens(layout, token_count)
    slice_count = max(1, int(args.object_slice_count))
    slice_index = int(args.object_slice_index)
    if slice_index < 0 or slice_index >= slice_count:
        raise ValueError(
            f"object_slice_index={slice_index} must be in [0, {slice_count})"
        )
    start = full_count * slice_index // slice_count
    end = full_count * (slice_index + 1) // slice_count
    return start, end, full_count


def select_specs_for_slice(
    specs: List[ObjectSpec], seen_before: int, slice_start: int, slice_end: int
) -> List[ObjectSpec]:
    if slice_start == 0 and slice_end >= seen_before + len(specs):
        return specs
    selected = []
    for local_idx, spec in enumerate(specs):
        global_idx = seen_before + local_idx
        if slice_start <= global_idx < slice_end:
            selected.append(spec)
    return selected


def safe_barrier_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", value)


def wait_barrier(args: argparse.Namespace, label: str) -> None:
    if not args.barrier_dir or args.barrier_participants <= 1:
        return
    barrier_dir = Path(args.barrier_dir)
    barrier_dir.mkdir(parents=True, exist_ok=True)
    label = safe_barrier_label(f"{args.barrier_prefix}_{label}")
    ready = barrier_dir / f"{label}.rank{args.barrier_rank}.ready"
    release = barrier_dir / f"{label}.release"
    ready.write_text(f"{os.getpid()} {time.time():.9f}\n")
    deadline = time.time() + args.barrier_timeout
    glob_pattern = f"{label}.rank*.ready"
    while time.time() < deadline:
        if len(list(barrier_dir.glob(glob_pattern))) >= args.barrier_participants:
            break
        time.sleep(0.001)
    else:
        raise TimeoutError(
            f"barrier timeout label={label} participants={args.barrier_participants}"
        )
    release_at = None
    try:
        fd = os.open(str(release), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        release_at = time.time() + args.barrier_release_delay
        with os.fdopen(fd, "w") as f:
            f.write(f"{release_at:.9f}\n")
    except FileExistsError:
        pass
    while release_at is None and time.time() < deadline:
        try:
            release_at = float(release.read_text().strip())
        except Exception:
            time.sleep(0.001)
    if release_at is None:
        raise TimeoutError(f"barrier release timeout label={label}")
    while time.time() < release_at:
        time.sleep(min(0.001, release_at - time.time()))


def validate_layout(layout: PageLayout, ks: List[int]) -> None:
    if (
        layout.object_layout == "page_kv_split_all_layers"
        and layout.k_object_bytes != layout.v_object_bytes
    ):
        raise ValueError("this raw MHA benchmark expects equal K and V object sizes")
    if layout.object_layout == "layer_page_kv_combined":
        if not layout.full_attention_layer_ids and not layout.swa_attention_layer_ids:
            raise ValueError("layer_page_kv_combined produced no layer objects")
        if layout.swa_layer_count and layout.sliding_window_size is None:
            raise ValueError("SWA layers require sliding_window_size")
    if layout.object_layout == HICACHE_PAGE_KV_SPLIT_LAYOUT:
        if not layout.full_attention_layer_ids:
            raise ValueError("hicache_page_kv_split produced no main KV layers")
        if layout.swa_layer_count and layout.sliding_window_size is None:
            raise ValueError("hicache_page_kv_split SWA layers require sliding window")
    for k in ks:
        transfer_info_for_tokens(layout, k)


def buffer_size_for_tokens(layout: PageLayout, max_tokens: int) -> int:
    return int(
        transfer_info_for_tokens(layout, max_tokens)["total_transfer_payload_bytes"]
    )


def allocate_buffer(
    layout: PageLayout,
    max_tokens: int,
    touch_buffer: bool,
    allocator: str,
    buffer_device: str,
    cuda_device: int,
    fill_byte: int,
):
    info = transfer_info_for_tokens(layout, max_tokens)
    total_bytes = int(info["total_transfer_payload_bytes"])
    logger.info(
        "allocating buffer: allocator=%s device=%s pages=%d objects=%d object_bytes=%s dtype=%s total=%.3f GiB",
        allocator,
        buffer_device,
        info["num_pages"],
        info["object_count"],
        info["object_bytes_by_layer_type"],
        layout.dtype,
        total_bytes / (1 << 30),
    )
    if buffer_device == "cuda":
        import torch

        if allocator != "torch":
            raise ValueError("CUDA buffer allocation requires --allocator torch")
        torch.cuda.set_device(cuda_device)
        tensor = torch.empty(
            total_bytes, dtype=torch.uint8, device=f"cuda:{cuda_device}"
        )
        if touch_buffer:
            tensor.fill_(int(fill_byte) & 0xFF)
            torch.cuda.synchronize()
        return tensor

    if allocator == "torch":
        import torch

        tensor = torch.empty(total_bytes, dtype=torch.uint8)
        if touch_buffer:
            tensor.fill_(int(fill_byte) & 0xFF)
        return tensor
    if allocator != "mmap":
        raise ValueError(f"unsupported allocator: {allocator}")
    backing = mmap.mmap(-1, total_bytes, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    ptr = ctypes.addressof(ctypes.c_char.from_buffer(backing))
    if touch_buffer:
        logger.info("touching mmap buffer bytes=%d", total_bytes)
        ctypes.memset(ptr, int(fill_byte) & 0xFF, total_bytes)
    return RawBuffer(backing, ptr, total_bytes, allocator)


def setup_store(args: argparse.Namespace):
    from mooncake.store import MooncakeDistributedStore

    store = MooncakeDistributedStore()
    setup_args = [
        args.local_hostname,
        args.metadata_server,
        args.global_segment_size,
        args.local_buffer_size,
        args.protocol,
        effective_device_name(args),
        args.master_address,
    ]
    try:
        ret = store.setup(*setup_args, None)
    except TypeError:
        ret = store.setup(*setup_args)
    if ret:
        raise RuntimeError(f"Mooncake setup failed: {ret}")
    return store


def make_page_component_keys(
    prefix: str,
    prompt_tokens: int,
    repeat: int,
    start_page: int,
    count_pages: int,
) -> List[str]:
    keys = []
    base = f"{prefix}_k{prompt_tokens}_r{repeat}"
    for page_id in range(start_page, start_page + count_pages):
        keys.append(f"{base}_p{page_id}_k")
        keys.append(f"{base}_p{page_id}_v")
    return keys


def make_page_component_ptrs(
    base_ptr: int, start_page: int, count_pages: int, object_bytes: int
) -> List[int]:
    ptrs = []
    for page_id in range(start_page, start_page + count_pages):
        object_id = page_id * 2
        ptrs.append(base_ptr + object_id * object_bytes)
        ptrs.append(base_ptr + (object_id + 1) * object_bytes)
    return ptrs


def make_page_component_specs(
    prefix: str,
    prompt_tokens: int,
    repeat: int,
    start_page: int,
    count_pages: int,
    layout: PageLayout,
) -> List[ObjectSpec]:
    specs = []
    base = f"{prefix}_k{prompt_tokens}_r{repeat}"
    for page_id in range(start_page, start_page + count_pages):
        page_offset = page_id * layout.logical_page_bytes
        specs.append(
            ObjectSpec(
                key=f"{base}_p{page_id}_k",
                offset=page_offset,
                size=layout.k_object_bytes,
                page_id=page_id,
                layer_type="k",
            )
        )
        specs.append(
            ObjectSpec(
                key=f"{base}_p{page_id}_v",
                offset=page_offset + layout.k_object_bytes,
                size=layout.v_object_bytes,
                page_id=page_id,
                layer_type="v",
            )
        )
    return specs


def page_prefix_bytes(layout: PageLayout, num_pages: int, page_id: int) -> int:
    per_layer_pages = layer_pages_for_tokens(layout, num_pages * layout.page_size)
    total = 0
    for layer_id, pages in per_layer_pages.items():
        tail_start = num_pages - int(pages)
        pages_before = min(max(page_id - tail_start, 0), int(pages))
        total += pages_before * object_bytes_for_layer(layout, layer_id)
    return total


def make_layer_page_kv_specs(
    prefix: str,
    prompt_tokens: int,
    repeat: int,
    start_page: int,
    count_pages: int,
    layout: PageLayout,
) -> List[ObjectSpec]:
    num_pages = num_pages_for_tokens(layout, prompt_tokens)
    per_layer_pages = layer_pages_for_tokens(layout, prompt_tokens)
    specs: List[ObjectSpec] = []
    base = f"{prefix}_k{prompt_tokens}_r{repeat}"
    for page_id in range(start_page, start_page + count_pages):
        offset = page_prefix_bytes(layout, num_pages, page_id)
        for layer_id in sorted(per_layer_pages):
            layer_pages = int(per_layer_pages[layer_id])
            tail_start = num_pages - layer_pages
            if page_id >= tail_start and layer_pages > 0:
                layer_type = layer_type_for_id(layout, layer_id)
                object_bytes = object_bytes_for_layer(layout, layer_id)
                window = (
                    layout.layer_windows_used[layer_id]
                    if 0 <= layer_id < len(layout.layer_windows_used)
                    else 0
                )
                specs.append(
                    ObjectSpec(
                        key=f"{base}_l{layer_id}_w{window}_{layer_type}_p{page_id}_kv",
                        offset=offset,
                        size=object_bytes,
                        layer_id=layer_id,
                        layer_type=layer_type,
                        page_id=page_id,
                    )
                )
                offset += object_bytes
    return specs


def make_hicache_main_kv_specs(
    prefix: str,
    prompt_tokens: int,
    repeat: int,
    start_page: int,
    count_pages: int,
    layout: PageLayout,
) -> List[ObjectSpec]:
    specs = []
    base = f"{prefix}_k{prompt_tokens}_r{repeat}"
    k_bytes = hicache_main_k_object_bytes(layout)
    v_bytes = hicache_main_v_object_bytes(layout)
    logical_page_bytes = k_bytes + v_bytes
    for page_id in range(start_page, start_page + count_pages):
        page_offset = page_id * logical_page_bytes
        page_key = f"{base}_p{page_id}"
        specs.append(
            ObjectSpec(
                key=f"{page_key}_mha_k",
                offset=page_offset,
                size=k_bytes,
                page_id=page_id,
                layer_type="kv_k",
            )
        )
        specs.append(
            ObjectSpec(
                key=f"{page_key}_mha_v",
                offset=page_offset + k_bytes,
                size=v_bytes,
                page_id=page_id,
                layer_type="kv_v",
            )
        )
    return specs


def make_hicache_swa_specs(
    prefix: str,
    prompt_tokens: int,
    repeat: int,
    start_page: int,
    count_pages: int,
    layout: PageLayout,
) -> List[ObjectSpec]:
    swa_pages = hicache_swa_pages_for_tokens(layout, prompt_tokens)
    if swa_pages <= 0:
        return []
    num_pages = num_pages_for_tokens(layout, prompt_tokens)
    tail_start = num_pages - swa_pages
    start = max(start_page, tail_start)
    end = min(start_page + count_pages, num_pages)
    if start >= end:
        return []

    specs = []
    base = f"{prefix}_k{prompt_tokens}_r{repeat}"
    main_bytes = num_pages * (
        hicache_main_k_object_bytes(layout) + hicache_main_v_object_bytes(layout)
    )
    k_bytes = hicache_swa_k_object_bytes(layout)
    v_bytes = hicache_swa_v_object_bytes(layout)
    logical_page_bytes = k_bytes + v_bytes
    for page_id in range(start, end):
        swa_page_id = page_id - tail_start
        page_offset = main_bytes + swa_page_id * logical_page_bytes
        page_key = f"{base}_p{page_id}"
        specs.append(
            ObjectSpec(
                key=f"{page_key}_mha_swa_k",
                offset=page_offset,
                size=k_bytes,
                page_id=page_id,
                layer_type="swa_k",
            )
        )
        specs.append(
            ObjectSpec(
                key=f"{page_key}_mha_swa_v",
                offset=page_offset + k_bytes,
                size=v_bytes,
                page_id=page_id,
                layer_type="swa_v",
            )
        )
    return specs


def make_specs(
    prefix: str,
    prompt_tokens: int,
    repeat: int,
    start_page: int,
    count_pages: int,
    layout: PageLayout,
) -> List[ObjectSpec]:
    if layout.object_layout == "page_kv_split_all_layers":
        return make_page_component_specs(
            prefix, prompt_tokens, repeat, start_page, count_pages, layout
        )
    if layout.object_layout == "layer_page_kv_combined":
        return make_layer_page_kv_specs(
            prefix, prompt_tokens, repeat, start_page, count_pages, layout
        )
    if layout.object_layout == HICACHE_PAGE_KV_SPLIT_LAYOUT:
        specs = make_hicache_swa_specs(
            prefix, prompt_tokens, repeat, start_page, count_pages, layout
        )
        specs.extend(
            make_hicache_main_kv_specs(
                prefix, prompt_tokens, repeat, start_page, count_pages, layout
            )
        )
        return specs
    raise ValueError(f"unsupported object_layout: {layout.object_layout}")


def make_spec_groups(
    prefix: str,
    prompt_tokens: int,
    repeat: int,
    start_page: int,
    count_pages: int,
    layout: PageLayout,
) -> List[Tuple[str, List[ObjectSpec]]]:
    if layout.object_layout == HICACHE_PAGE_KV_SPLIT_LAYOUT:
        groups = []
        swa_specs = make_hicache_swa_specs(
            prefix, prompt_tokens, repeat, start_page, count_pages, layout
        )
        if swa_specs:
            groups.append(("swa", swa_specs))
        kv_specs = make_hicache_main_kv_specs(
            prefix, prompt_tokens, repeat, start_page, count_pages, layout
        )
        if kv_specs:
            groups.append(("kv", kv_specs))
        return groups
    return [
        (
            layout.object_layout,
            make_specs(prefix, prompt_tokens, repeat, start_page, count_pages, layout),
        )
    ]


def sample_specs_for_case(
    layout: PageLayout, prompt_tokens: int, repeat: int, args: argparse.Namespace
) -> List[ObjectSpec]:
    slice_start, slice_end, object_count = object_slice_bounds(
        layout, prompt_tokens, args
    )
    sliced_count = slice_end - slice_start
    if sliced_count <= 0 or args.cuda_sample_objects <= 0:
        return []
    indices = {0, sliced_count // 2, sliced_count - 1}
    if args.cuda_sample_objects > 3:
        step = max(1, sliced_count // args.cuda_sample_objects)
        indices.update(range(0, sliced_count, step))
    wanted = sorted(indices)[: args.cuda_sample_objects]
    result = []
    seen = 0
    sliced_seen = 0
    num_pages = num_pages_for_tokens(layout, prompt_tokens)
    for start_page in range(0, num_pages, args.chunk_pages):
        count_pages = min(args.chunk_pages, num_pages - start_page)
        spec_groups = make_spec_groups(
            args.key_prefix, prompt_tokens, repeat, start_page, count_pages, layout
        )
        for _, specs in spec_groups:
            selected = select_specs_for_slice(specs, seen, slice_start, slice_end)
            seen += len(specs)
            for spec in selected:
                if sliced_seen in wanted:
                    result.append(spec)
                sliced_seen += 1
                if len(result) == len(wanted):
                    return result
    return result


def specs_to_batch(
    base_ptr: int, specs: List[ObjectSpec]
) -> Tuple[List[str], List[int], List[int]]:
    keys = [spec.key for spec in specs]
    ptrs = [base_ptr + spec.offset for spec in specs]
    sizes = [spec.size for spec in specs]
    return keys, ptrs, sizes


def wait_url(url: str, timeout: int):
    deadline = time.time() + timeout
    last_exc = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            last_exc = exc
            time.sleep(1)
    raise TimeoutError(f"timed out waiting for {url}") from last_exc


def result_counts(results: List[int]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for result in results:
        key = str(result)
        counts[key] = counts.get(key, 0) + 1
    return counts


def compact_results(results: List[int], limit: int) -> Any:
    if limit <= 0 or len(results) <= limit:
        return results
    half = max(1, limit // 2)
    return {
        "head": results[:half],
        "tail": results[-half:],
        "count": len(results),
        "truncated": True,
    }


def make_row(
    direction: str,
    layout: PageLayout,
    prompt_tokens: int,
    repeat: int,
    elapsed: float,
    results: List[int],
    sizes: List[int],
    chunks: int,
    args: argparse.Namespace,
    cuda_sample: Optional[Dict[str, Any]] = None,
    call_breakdown: Optional[List[Dict[str, Any]]] = None,
) -> BenchResult:
    info = transfer_info_for_tokens(layout, prompt_tokens)
    slice_start, slice_end, full_object_count = object_slice_bounds(
        layout, prompt_tokens, args
    )
    num_pages = int(info["num_pages"])
    object_count = slice_end - slice_start
    object_bytes = int(info["object_bytes"])
    if direction == "put":
        success_count = sum(1 for result in results if result == 0)
        short_count = 0
        bytes_done = sum(size for result, size in zip(results, sizes) if result == 0)
        ok = len(results) == object_count and success_count == object_count
    else:
        success_count = sum(1 for result, size in zip(results, sizes) if result == size)
        short_count = sum(
            1 for result, size in zip(results, sizes) if 0 < result < size
        )
        bytes_done = sum(result for result in results if result > 0)
        ok = len(results) == object_count and success_count == object_count
    if cuda_sample and cuda_sample.get("enabled"):
        ok = ok and bool(cuda_sample.get("ok"))
    failure_count = len(results) - success_count - short_count
    total_payload_bytes = sum(sizes)
    cuda_sample = cuda_sample or {}
    return BenchResult(
        direction=direction,
        prompt_tokens=prompt_tokens,
        page_size=layout.page_size,
        repeat=repeat,
        num_pages=num_pages,
        object_count=object_count,
        object_bytes=object_bytes,
        k_object_bytes=layout.k_object_bytes,
        v_object_bytes=layout.v_object_bytes,
        logical_page_bytes=layout.logical_page_bytes,
        total_payload_bytes=total_payload_bytes,
        latency_ms=elapsed * 1000,
        bandwidth_gib_s=(bytes_done / (1 << 30)) / elapsed if elapsed > 0 else 0,
        ok=ok,
        result_count=len(results),
        success_count=success_count,
        failure_count=failure_count,
        short_count=short_count,
        min_result=min(results) if results else 0,
        max_result=max(results) if results else 0,
        chunks=chunks,
        raw_results=compact_results(results, args.raw_results_limit),
        result_counts=result_counts(results),
        full_layer_count=layout.full_layer_count,
        swa_layer_count=layout.swa_layer_count,
        swa_pages=int(info["swa_pages"]),
        per_layer_page_count=dict(info.get("per_layer_page_count", {})),
        per_layer_window=dict(info.get("per_layer_window", {})),
        object_bytes_by_layer_type=dict(info["object_bytes_by_layer_type"]),
        useful_payload_bytes=total_payload_bytes,
        cuda_sample_ok=cuda_sample.get("ok"),
        cuda_sample_count=int(cuda_sample.get("count", 0)),
        cuda_sample_bad=int(cuda_sample.get("bad", 0)),
        cuda_sample_bytes=int(cuda_sample.get("bytes", 0)),
        cuda_sample_offsets=list(cuda_sample.get("offsets", [])),
        full_object_count=full_object_count,
        object_slice_index=int(args.object_slice_index),
        object_slice_count=max(1, int(args.object_slice_count)),
        object_slice_start=slice_start,
        object_slice_end=slice_end,
        call_count=len(call_breakdown or []),
        call_breakdown=list(call_breakdown or []),
    )


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_rows(rows: List[BenchResult]) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[BenchResult]] = {}
    for row in rows:
        groups.setdefault((row.direction, row.page_size, row.prompt_tokens), []).append(
            row
        )
    summary = []
    for (direction, page_size, prompt_tokens), group in sorted(groups.items()):
        bandwidths = [row.bandwidth_gib_s for row in group if row.ok]
        latencies = [row.latency_ms for row in group]
        summary.append(
            {
                "direction": direction,
                "page_size": page_size,
                "prompt_tokens": prompt_tokens,
                "repeats": len(group),
                "ok_repeats": sum(1 for row in group if row.ok),
                "best_gib_s": max(bandwidths) if bandwidths else 0.0,
                "median_gib_s": statistics.median(bandwidths) if bandwidths else 0.0,
                "latency_p50_ms": percentile(latencies, 50),
                "latency_p90_ms": percentile(latencies, 90),
                "latency_p99_ms": percentile(latencies, 99),
            }
        )
    return summary


def prepare_cuda_samples(
    tensor,
    specs: List[ObjectSpec],
    fill_byte: int,
    sample_bytes: int,
) -> None:
    if not specs:
        return
    import torch

    for spec in specs:
        count = min(spec.size, sample_bytes)
        tensor[spec.offset : spec.offset + count].fill_(int(fill_byte) & 0xFF)
    torch.cuda.synchronize()


def verify_cuda_samples(
    tensor,
    specs: List[ObjectSpec],
    expected_byte: int,
    sample_bytes: int,
) -> Dict[str, Any]:
    if not specs:
        return {
            "enabled": False,
            "ok": None,
            "count": 0,
            "bad": 0,
            "bytes": 0,
            "offsets": [],
        }
    import torch

    torch.cuda.synchronize()
    bad = 0
    total_bytes = 0
    offsets = []
    expected = int(expected_byte) & 0xFF
    for spec in specs:
        count = min(spec.size, sample_bytes)
        host = tensor[spec.offset : spec.offset + count].detach().cpu()
        if not bool(torch.all(host == expected).item()):
            bad += 1
        offsets.append(spec.offset)
        total_bytes += count
    return {
        "enabled": True,
        "ok": bad == 0,
        "count": len(specs),
        "bad": bad,
        "bytes": total_bytes,
        "offsets": offsets,
    }


def run_transfer_case(
    store,
    layout: PageLayout,
    buffer_obj,
    base_ptr: int,
    args: argparse.Namespace,
    direction: str,
    prompt_tokens: int,
    repeat: int,
) -> BenchResult:
    num_pages = num_pages_for_tokens(layout, prompt_tokens)
    slice_start, slice_end, _ = object_slice_bounds(layout, prompt_tokens, args)
    all_results = []
    all_sizes = []
    chunks = 0
    sample_specs: List[ObjectSpec] = []
    if (
        direction == "get"
        and args.reader_buffer_device == "cuda"
        and args.sync_cuda_for_correctness
    ):
        sample_specs = sample_specs_for_case(layout, prompt_tokens, repeat, args)
        prepare_cuda_samples(
            buffer_obj, sample_specs, args.dest_fill_byte, args.cuda_sample_bytes
        )

    begin = time.perf_counter()
    seen = 0
    call_breakdown: List[Dict[str, Any]] = []
    for start_page in range(0, num_pages, args.chunk_pages):
        count_pages = min(args.chunk_pages, num_pages - start_page)
        spec_groups = make_spec_groups(
            args.key_prefix, prompt_tokens, repeat, start_page, count_pages, layout
        )
        for group_name, all_specs in spec_groups:
            specs = select_specs_for_slice(all_specs, seen, slice_start, slice_end)
            seen += len(all_specs)
            if not specs:
                continue
            keys, ptrs, sizes = specs_to_batch(base_ptr, specs)
            call_begin = time.perf_counter()
            if direction == "put":
                results = store.batch_put_from(keys, ptrs, sizes)
            else:
                results = store.batch_get_into(keys, ptrs, sizes)
            call_elapsed = time.perf_counter() - call_begin
            if direction == "put":
                bytes_done = sum(
                    size for result, size in zip(results, sizes) if result == 0
                )
            else:
                bytes_done = sum(result for result in results if result > 0)
            call_breakdown.append(
                {
                    "group": group_name,
                    "chunk_start_page": start_page,
                    "chunk_pages": count_pages,
                    "object_count": len(keys),
                    "bytes": sum(sizes),
                    "latency_ms": call_elapsed * 1000,
                    "bandwidth_gib_s": (bytes_done / (1 << 30)) / call_elapsed
                    if call_elapsed > 0
                    else 0,
                }
            )
            all_results.extend(results)
            all_sizes.extend(sizes)
            chunks += 1
    elapsed = time.perf_counter() - begin
    cuda_sample = None
    if sample_specs:
        cuda_sample = verify_cuda_samples(
            buffer_obj,
            sample_specs,
            args.source_fill_byte,
            args.cuda_sample_bytes,
        )
    row = make_row(
        direction,
        layout,
        prompt_tokens,
        repeat,
        elapsed,
        all_results,
        all_sizes,
        chunks,
        args,
        cuda_sample,
        call_breakdown,
    )
    log_row = asdict(row)
    logger.info("%s result %s", direction, log_row)
    if not row.ok and args.fail_fast:
        if direction == "put":
            bad = [(i, result) for i, result in enumerate(all_results) if result != 0]
        else:
            bad = [
                (i, result, size)
                for i, (result, size) in enumerate(zip(all_results, all_sizes))
                if result != size
            ]
        raise RuntimeError(
            f"{direction} failed for k={prompt_tokens} repeat={repeat}: bad={bad[:16]}"
        )
    return row


def output_payload(
    args: argparse.Namespace,
    layout: PageLayout,
    rows: List[BenchResult],
    ready_payload: Optional[Dict[str, Any]] = None,
    writer_segment: str = "",
    register_buffer_ret: Optional[int] = None,
) -> Dict[str, Any]:
    dest_memory = (
        f"cuda:{args.cuda_device}"
        if args.role == "reader" and args.reader_buffer_device == "cuda"
        else "cpu"
    )
    return {
        "run_id": args.run_id,
        "role": args.role,
        "key_prefix": args.key_prefix,
        "source_memory": "cpu",
        "dest_memory": dest_memory,
        "reader_buffer_device": args.reader_buffer_device,
        "cuda_device": args.cuda_device,
        "master_address": args.master_address,
        "local_hostname": args.local_hostname,
        "metadata_server": args.metadata_server,
        "protocol": args.protocol,
        "device_name": effective_device_name(args),
        "rail_count": rail_count(args),
        "env_mooncake_device": os.environ.get("MOONCAKE_DEVICE", ""),
        "env_mc_ms_auto_disc": os.environ.get("MC_MS_AUTO_DISC", ""),
        "env_mc_num_qp_per_ep": os.environ.get("MC_NUM_QP_PER_EP", ""),
        "MC_NUM_QP_PER_EP": os.environ.get("MC_NUM_QP_PER_EP", ""),
        "global_segment_size": args.global_segment_size,
        "local_buffer_size": args.local_buffer_size,
        "register_buffer_ret": register_buffer_ret,
        "layout": asdict(layout),
        "ks": parse_ks(args.ks),
        "repeats": args.repeats,
        "chunk_pages": args.chunk_pages,
        "ready_payload": ready_payload or {},
        "writer_segment": writer_segment,
        "object_slice_index": int(args.object_slice_index),
        "object_slice_count": max(1, int(args.object_slice_count)),
        "barrier_dir": args.barrier_dir,
        "barrier_participants": args.barrier_participants,
        "barrier_rank": args.barrier_rank,
        "barrier_prefix": args.barrier_prefix,
        "summary": summarize_rows(rows),
        "results": [asdict(row) for row in rows],
    }


def run_writer(args: argparse.Namespace):
    validate_environment(args)
    ks = parse_ks(args.ks)
    layout = derive_layout(args)
    validate_layout(layout, ks)
    max_tokens = max(max(ks), args.max_tokens)
    if max_tokens % layout.page_size != 0:
        raise ValueError(
            f"max_tokens={max_tokens} must be divisible by page_size={layout.page_size}"
        )
    tensor = allocate_buffer(
        layout,
        max_tokens,
        args.touch_buffer,
        args.allocator,
        "cpu",
        args.cuda_device,
        args.source_fill_byte,
    )
    store = setup_store(args)
    base_ptr = tensor.data_ptr()
    total_bytes = buffer_size_for_tokens(layout, max_tokens)
    ret = store.register_buffer(base_ptr, total_bytes)
    if ret:
        raise RuntimeError(f"register_buffer failed: {ret}")

    writer_segment = store.get_hostname()
    logger.info("writer segment hostname=%s", writer_segment)
    rows: List[BenchResult] = []
    for repeat in range(args.repeats):
        for k in ks:
            rows.append(
                run_transfer_case(
                    store, layout, tensor, base_ptr, args, "put", k, repeat
                )
            )
            if args.put_sleep > 0:
                time.sleep(args.put_sleep)

    payload = output_payload(
        args,
        layout,
        rows,
        writer_segment=writer_segment,
        register_buffer_ret=ret,
    )
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    ready_payload = {
        "ready": True,
        "role": "writer",
        "run_id": args.run_id,
        "key_prefix": args.key_prefix,
        "ks": ks,
        "repeats": args.repeats,
        "layout": asdict(layout),
        "device_name": effective_device_name(args),
        "rail_count": rail_count(args),
        "output": args.output,
        "writer_segment": writer_segment,
        "register_buffer_ret": ret,
        "summary": payload["summary"],
    }
    ReadyHandler.payload = ready_payload
    logger.info("writer ready payload=%s", ready_payload)
    server = http.server.ThreadingHTTPServer(
        (args.http_host, args.http_port), ReadyHandler
    )
    server.serve_forever()


def run_reader(args: argparse.Namespace):
    validate_environment(args)
    ready_payload = wait_url(args.wait_url, args.wait_timeout) if args.wait_url else {}
    logger.info("ready payload=%s", ready_payload)

    ks = parse_ks(args.ks)
    layout = derive_layout(args)
    validate_layout(layout, ks)
    max_tokens = max(max(ks), args.max_tokens)
    if max_tokens % layout.page_size != 0:
        raise ValueError(
            f"max_tokens={max_tokens} must be divisible by page_size={layout.page_size}"
        )
    tensor = allocate_buffer(
        layout,
        max_tokens,
        args.touch_buffer,
        args.allocator,
        args.reader_buffer_device,
        args.cuda_device,
        args.dest_fill_byte,
    )
    store = setup_store(args)
    base_ptr = tensor.data_ptr()
    total_bytes = buffer_size_for_tokens(layout, max_tokens)
    ret = store.register_buffer(base_ptr, total_bytes)
    if ret:
        raise RuntimeError(f"register_buffer failed: {ret}")

    rows: List[BenchResult] = []
    for repeat in range(args.repeats):
        for k in ks:
            if args.warmup:
                slice_start, slice_end, _ = object_slice_bounds(layout, k, args)
                warm_pages = min(num_pages_for_tokens(layout, k), args.chunk_pages)
                seen = 0
                warm_objects = 0
                warm_ok = 0
                spec_groups = make_spec_groups(
                    args.key_prefix, k, repeat, 0, warm_pages, layout
                )
                for _, all_specs in spec_groups:
                    specs = select_specs_for_slice(
                        all_specs, seen, slice_start, slice_end
                    )
                    seen += len(all_specs)
                    if not specs:
                        continue
                    keys, ptrs, sizes = specs_to_batch(base_ptr, specs)
                    warm_results = store.batch_get_into(keys, ptrs, sizes)
                    warm_objects += len(keys)
                    warm_ok += sum(
                        1
                        for result, size in zip(warm_results, sizes)
                        if result == size
                    )
                logger.info(
                    "warmup k=%d repeat=%d objects=%d ok=%d/%d",
                    k,
                    repeat,
                    warm_objects,
                    warm_ok,
                    warm_objects,
                )
            wait_barrier(args, f"get_r{repeat}_k{k}")
            rows.append(
                run_transfer_case(
                    store, layout, tensor, base_ptr, args, "get", k, repeat
                )
            )

    payload = output_payload(
        args,
        layout,
        rows,
        ready_payload=ready_payload,
        register_buffer_ret=ret,
    )
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    logger.info("wrote %s", args.output)


def run_layout(args: argparse.Namespace):
    ks = parse_ks(args.ks)
    layout = derive_layout(args)
    validate_layout(layout, ks)
    rows = []
    for k in ks:
        info = transfer_info_for_tokens(layout, k)
        rows.append(
            {
                "prompt_tokens": k,
                "page_size": layout.page_size,
                "num_pages": info["num_pages"],
                "swa_pages": info["swa_pages"],
                "object_count": info["object_count"],
                "object_bytes": info["object_bytes"],
                "object_bytes_by_layer_type": info["object_bytes_by_layer_type"],
                "per_layer_page_count": info.get("per_layer_page_count", {}),
                "per_layer_window": info.get("per_layer_window", {}),
                "full_layer_count": layout.full_layer_count,
                "swa_layer_count": layout.swa_layer_count,
                "logical_page_bytes": layout.logical_page_bytes,
                "total_transfer_payload_bytes": info["total_transfer_payload_bytes"],
                "useful_payload_bytes": info["useful_payload_bytes"],
            }
        )
    payload = {
        "role": "layout",
        "layout": asdict(layout),
        "ks": ks,
        "cases": rows,
    }
    print(json.dumps(payload, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["writer", "reader", "layout"], default="")
    parser.add_argument("--emit-layout-only", action="store_true")
    parser.add_argument("--run-id", default=str(uuid.uuid4()))
    parser.add_argument("--key-prefix", default="")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--page-size", type=int, required=True)
    parser.add_argument(
        "--object-layout",
        choices=[
            "page_kv_split_all_layers",
            "layer_page_kv_combined",
            HICACHE_PAGE_KV_SPLIT_LAYOUT,
        ],
        default="page_kv_split_all_layers",
    )
    parser.add_argument(
        "--layer-window-policy",
        choices=["full_and_swa_tail"],
        default="full_and_swa_tail",
    )
    parser.add_argument(
        "--include-nextn-kv",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="include num_nextn_predict_layers entries from sliding_window_size_layerwise",
    )
    parser.add_argument(
        "--ks",
        default="2048,4096,8192,16384,32768,65536,131072",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--dtype", default="")
    parser.add_argument("--chunk-pages", type=int, default=256)
    parser.add_argument(
        "--touch-buffer", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--allocator", choices=["mmap", "torch"], default="mmap")
    parser.add_argument(
        "--reader-buffer-device", choices=["cpu", "cuda"], default="cpu"
    )
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--sync-cuda-for-correctness", action="store_true")
    parser.add_argument("--source-fill-byte", type=int, default=90)
    parser.add_argument("--dest-fill-byte", type=int, default=165)
    parser.add_argument("--cuda-sample-bytes", type=int, default=4096)
    parser.add_argument("--cuda-sample-objects", type=int, default=3)
    parser.add_argument("--raw-results-limit", type=int, default=4096)
    parser.add_argument("--object-slice-index", type=int, default=0)
    parser.add_argument("--object-slice-count", type=int, default=1)
    parser.add_argument("--barrier-dir", default="")
    parser.add_argument("--barrier-participants", type=int, default=1)
    parser.add_argument("--barrier-rank", type=int, default=0)
    parser.add_argument("--barrier-prefix", default="transfer")
    parser.add_argument("--barrier-timeout", type=float, default=300.0)
    parser.add_argument("--barrier-release-delay", type=float, default=0.2)
    parser.add_argument("--master-address", default="")
    parser.add_argument("--local-hostname", default="")
    parser.add_argument("--metadata-server", default="P2PHANDSHAKE")
    parser.add_argument("--protocol", default="rdma")
    parser.add_argument("--device-name", default="")
    parser.add_argument(
        "--global-segment-size", default=128 * 1024 * 1024 * 1024, type=int
    )
    parser.add_argument("--local-buffer-size", default=16 * 1024 * 1024, type=int)
    parser.add_argument("--http-host", default="0.0.0.0")
    parser.add_argument("--http-port", type=int, default=18080)
    parser.add_argument("--wait-url", default="")
    parser.add_argument("--wait-timeout", type=int, default=1800)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--fail-fast", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--fail-on-auto-disc", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--require-explicit-device", action="store_true")
    parser.add_argument("--put-sleep", type=float, default=0.0)
    parser.add_argument("--output", default="/tmp/mooncake_raw_transfer_result.json")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    if args.emit_layout_only:
        args.role = "layout"
    if not args.role:
        parser.error("--role is required unless --emit-layout-only is set")
    if args.role == "reader" and args.reader_buffer_device == "cuda":
        args.allocator = "torch"

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger.info(
        "started role=%s allocator=%s pid=%d python=%s",
        args.role,
        args.allocator,
        os.getpid(),
        sys.executable,
    )

    if args.role != "layout":
        if not args.key_prefix:
            raise ValueError("--key-prefix is required for writer/reader")
        if not args.master_address:
            raise ValueError("--master-address is required for writer/reader")
        if not args.local_hostname:
            raise ValueError("--local-hostname is required for writer/reader")

    if args.role == "writer":
        run_writer(args)
    elif args.role == "reader":
        run_reader(args)
    else:
        run_layout(args)


if __name__ == "__main__":
    main()
