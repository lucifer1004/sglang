#!/usr/bin/env python3
"""Compute WeLMv4 main-model FLOPs/token and MFU for benchmark CSVs.

The estimator matches the WeLMv4 benchmark setup:
  - main model only, no MTP / nextn draft FLOPs
  - sliding-window attention is included
  - WeLM KV mirror prefill contraction is included only when explicitly enabled
  - rank-local FLOPs are estimated with SGLang-style TP sharding
  - CSV throughput is expected to be tokens/s/gpu, so rank-local FLOPs/token
    are multiplied by TP size before dividing by single-GPU peak TFLOPS
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MODEL_PATH_COLUMN = "模型ckpt"
PARALLEL_COLUMN = "并行方式"
PROMPT_LEN_COLUMN = "Prompt长度"
STAGE_COLUMN = "Prefill/Decode"
CHUNKED_PREFILL_COLUMN = "Chunked Prefill Size"
THROUGHPUT_COLUMN = "tokens/s/gpu"
OFFLINE_MFU_COLUMN = "MFU(%) (离线手算)"
LEGACY_OFFLINE_MFU_COLUMNS = ("MFU(%)(离线手算)", "MFU(%)")
SGLANG_MFU_COLUMN = "MFU(%) (SGLang给出)"
LEGACY_SGLANG_MFU_COLUMNS = ("MFU(%)(SGLang给出)",)


@dataclass(frozen=True)
class ModelSpec:
    hidden_size: int
    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    has_shared_expert_gate: bool
    num_experts: int
    vocab_size: int
    max_position_embeddings: int
    sliding_window_size_layerwise: tuple[int, ...]
    kv_mirror_layers: tuple[int, ...]
    kv_mirror_imitated_layers: tuple[int, ...]

    @classmethod
    def from_config(cls, path: Path) -> "ModelSpec":
        with path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)

        hidden_size = int(cfg["hidden_size"])
        num_q_heads = int(cfg["num_attention_heads"])
        head_dim = int(cfg.get("head_dim", hidden_size // num_q_heads))
        num_layers = int(cfg.get("num_hidden_layers", cfg.get("num_attention_layers")))

        sliding = tuple(
            int(x) for x in cfg.get("sliding_window_size_layerwise", [])[:num_layers]
        )
        if len(sliding) < num_layers:
            scalar_window = cfg.get("sliding_window_size", cfg.get("sliding_window", -1))
            sliding = sliding + tuple(
                int(scalar_window) for _ in range(num_layers - len(sliding))
            )

        return cls(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_q_heads=num_q_heads,
            num_kv_heads=int(cfg["num_key_value_heads"]),
            head_dim=head_dim,
            num_experts_per_tok=int(cfg.get("num_experts_per_tok", 0)),
            moe_intermediate_size=int(cfg.get("moe_intermediate_size", 0)),
            shared_expert_intermediate_size=int(
                cfg.get("shared_expert_intermediate_size", 0)
            ),
            has_shared_expert_gate=bool(cfg.get("has_shared_expert_gate", True)),
            num_experts=int(cfg.get("num_experts", 0)),
            vocab_size=int(cfg["vocab_size"]),
            max_position_embeddings=int(cfg.get("max_position_embeddings", 10**18)),
            sliding_window_size_layerwise=sliding,
            kv_mirror_layers=tuple(int(x) for x in cfg.get("kv_mirror_layers", [])),
            kv_mirror_imitated_layers=tuple(
                int(x) for x in cfg.get("kv_mirror_imitated_layers", [])
            ),
        )


@dataclass(frozen=True)
class FlopComponents:
    q: int
    kv: int
    o: int
    attn_gate: int
    mlp: int
    lm_head: int
    attn_ctx: int

    @property
    def q_o_gate(self) -> int:
        return self.q + self.o + self.attn_gate

    @property
    def linear_per_layer(self) -> int:
        return self.q + self.kv + self.o + self.attn_gate + self.mlp


def build_components(spec: ModelSpec, *, tp_size: int) -> FlopComponents:
    h = spec.hidden_size
    nq = max(1, spec.num_q_heads // tp_size)
    nkv = max(1, spec.num_kv_heads // tp_size)
    d = spec.head_dim

    f_q = 2 * h * nq * d
    f_kv = 2 * h * (2 * nkv * d)
    f_o = 2 * (nq * d) * h
    f_attn_gate = 2 * h * nq

    if (
        spec.num_experts
        and spec.num_experts_per_tok
        and spec.moe_intermediate_size
        and spec.shared_expert_intermediate_size
    ):
        # The WeLM MoE router is replicated on every TP rank, while routed and
        # shared expert GEMMs are tensor-parallel sharded.
        f_router = 2 * h * spec.num_experts
        f_routed_moe = (
            6 * h * spec.num_experts_per_tok * spec.moe_intermediate_size
        ) / tp_size
        f_shared_moe = 6 * h * spec.shared_expert_intermediate_size / tp_size
        f_shared_gate = 2 * h if spec.has_shared_expert_gate else 0
        f_mlp = f_router + f_routed_moe + f_shared_moe + f_shared_gate
    else:
        f_mlp = 0

    # ParallelLMHead is vocab-parallel in the benchmark setup.
    f_lm_head = 2 * h * spec.vocab_size / tp_size
    f_attn_ctx = 4 * nq * d

    return FlopComponents(
        q=int(f_q),
        kv=int(f_kv),
        o=int(f_o),
        attn_gate=int(f_attn_gate),
        mlp=int(f_mlp),
        lm_head=int(f_lm_head),
        attn_ctx=int(f_attn_ctx),
    )


def valid_main_mirror_layers(spec: ModelSpec) -> set[int]:
    valid: set[int] = set()
    for mirror, imitated in zip(spec.kv_mirror_layers, spec.kv_mirror_imitated_layers):
        if 0 <= mirror < spec.num_layers and 0 <= imitated < spec.num_layers:
            valid.add(mirror)
    return valid


def layer_window_plus_current(spec: ModelSpec, layer_idx: int) -> int | None:
    """Return max attended tokens for this layer, or None for full attention."""
    window = spec.sliding_window_size_layerwise[layer_idx]
    if window <= 0:
        return None
    if window >= spec.max_position_embeddings:
        return None
    return window + 1


def prefill_context_sum(
    prefix_len: int, extend_len: int, window_plus_current: int | None
) -> float:
    if extend_len <= 0:
        return 0.0
    if window_plus_current is None:
        return extend_len * prefix_len + extend_len * (extend_len + 1) / 2.0

    uncapped = min(extend_len, max(0, window_plus_current - prefix_len))
    return (
        uncapped * prefix_len
        + uncapped * (uncapped + 1) / 2.0
        + (extend_len - uncapped) * window_plus_current
    )


def prefill_last_context(
    prefix_len: int, extend_len: int, window_plus_current: int | None
) -> float:
    if extend_len <= 0:
        return 0.0
    context = prefix_len + extend_len
    if window_plus_current is not None:
        context = min(context, window_plus_current)
    return float(context)


def decode_avg_context(
    prompt_len: int, output_len: int, window_plus_current: int | None
) -> float:
    if window_plus_current is None:
        return prompt_len + (output_len - 1) / 2
    return sum(
        min(prompt_len + t, window_plus_current) for t in range(output_len)
    ) / output_len


def prefill_flops_for_extend(
    spec: ModelSpec,
    comp: FlopComponents,
    *,
    prefix_len: int,
    extend_len: int,
    enable_kv_mirror_opt: bool,
) -> float:
    mirror_layers = valid_main_mirror_layers(spec) if enable_kv_mirror_opt else set()
    first_mirror = min(mirror_layers) if mirror_layers else None

    total = extend_len * spec.num_layers * comp.kv
    active_seqs = 1

    for layer_idx in range(spec.num_layers):
        window = layer_window_plus_current(spec, layer_idx)

        q_rows = extend_len
        context_sum = prefill_context_sum(prefix_len, extend_len, window)
        if first_mirror is not None and layer_idx > first_mirror:
            q_rows = active_seqs
            context_sum = prefill_last_context(prefix_len, extend_len, window)

        total += q_rows * comp.q_o_gate
        total += context_sum * comp.attn_ctx

        mlp_rows = extend_len
        if first_mirror is not None and layer_idx >= first_mirror:
            mlp_rows = active_seqs
        total += mlp_rows * comp.mlp

    total += active_seqs * comp.lm_head
    return total


def prefill_flops_total(
    spec: ModelSpec,
    comp: FlopComponents,
    prompt_len: int,
    *,
    enable_kv_mirror_opt: bool,
    chunked_prefill_size: int,
) -> float:
    """Total FLOPs for one prompt request's prefill pass."""
    if prompt_len <= 0:
        return 0.0
    if chunked_prefill_size <= 0 or chunked_prefill_size >= prompt_len:
        return prefill_flops_for_extend(
            spec,
            comp,
            prefix_len=0,
            extend_len=prompt_len,
            enable_kv_mirror_opt=enable_kv_mirror_opt,
        )

    total = 0.0
    prefix_len = 0
    while prefix_len < prompt_len:
        extend_len = min(chunked_prefill_size, prompt_len - prefix_len)
        total += prefill_flops_for_extend(
            spec,
            comp,
            prefix_len=prefix_len,
            extend_len=extend_len,
            enable_kv_mirror_opt=enable_kv_mirror_opt,
        )
        prefix_len += extend_len
    return total


def prefill_flops_per_input_token(
    spec: ModelSpec,
    comp: FlopComponents,
    prompt_len: int,
    *,
    enable_kv_mirror_opt: bool,
    chunked_prefill_size: int,
) -> float:
    return prefill_flops_total(
        spec,
        comp,
        prompt_len,
        enable_kv_mirror_opt=enable_kv_mirror_opt,
        chunked_prefill_size=chunked_prefill_size,
    ) / prompt_len


def decode_flops_per_output_token(
    spec: ModelSpec,
    comp: FlopComponents,
    prompt_len: int,
    *,
    output_len: int,
) -> float:
    """Average FLOPs per generated token for a request.

    Normal decode has one query row per layer, so WeLM KV mirror contraction does
    not reduce the main-model layer count here. SWA is still included.
    """
    total = spec.num_layers * comp.linear_per_layer
    for layer_idx in range(spec.num_layers):
        window = layer_window_plus_current(spec, layer_idx)
        total += comp.attn_ctx * decode_avg_context(prompt_len, output_len, window)
    total += comp.lm_head
    return total


def standard_prompt_lengths() -> tuple[int, ...]:
    return (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)


def print_table(
    spec: ModelSpec,
    *,
    tp_size: int,
    output_len: int,
    enable_kv_mirror_opt: bool,
    chunked_prefill_size: int,
) -> None:
    comp = build_components(spec, tp_size=tp_size)
    print("prompt_len,prefill_gflops_per_input_token,decode_gflops_per_output_token")
    for prompt_len in standard_prompt_lengths():
        prefill = prefill_flops_per_input_token(
            spec,
            comp,
            prompt_len,
            enable_kv_mirror_opt=enable_kv_mirror_opt,
            chunked_prefill_size=chunked_prefill_size,
        ) * tp_size
        decode = decode_flops_per_output_token(
            spec,
            comp,
            prompt_len,
            output_len=output_len,
        ) * tp_size
        print(f"{prompt_len},{prefill / 1e9:.6f},{decode / 1e9:.6f}")


def parse_length(value: str) -> int:
    value = str(value).strip().lower().replace(",", "")
    if value in {"", "none", "nan", "-"}:
        return 0
    if value.endswith("k"):
        return int(float(value[:-1]) * 1024)
    return int(float(value))


def parse_tp_size(value: str) -> int:
    match = re.search(r"\bTP\s*(\d+)\b", str(value), flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse TP size from {value!r}")
    return int(match.group(1))


def choose_flops_for_row(
    row: dict[str, str],
    spec: ModelSpec,
    comp: FlopComponents,
    *,
    output_len: int,
    enable_kv_mirror_opt: bool,
    default_chunked_prefill_size: int,
) -> float:
    prompt_len = parse_length(row[PROMPT_LEN_COLUMN])
    stage = row[STAGE_COLUMN].strip().lower()
    chunked_prefill_size = parse_length(
        row.get(CHUNKED_PREFILL_COLUMN, default_chunked_prefill_size)
    )
    if chunked_prefill_size <= 0:
        chunked_prefill_size = default_chunked_prefill_size

    if stage == "prefill":
        return prefill_flops_per_input_token(
            spec,
            comp,
            prompt_len,
            enable_kv_mirror_opt=enable_kv_mirror_opt,
            chunked_prefill_size=chunked_prefill_size,
        )
    if stage == "decode":
        return decode_flops_per_output_token(
            spec,
            comp,
            prompt_len,
            output_len=output_len,
        )
    raise ValueError(f"Unknown {STAGE_COLUMN} value: {row[STAGE_COLUMN]!r}")


def first_existing_column(fieldnames: list[str], names: Iterable[str]) -> str | None:
    existing = set(fieldnames)
    for name in names:
        if name in existing:
            return name
    return None


def ensure_column(
    fieldnames: list[str], canonical_name: str, legacy_names: Iterable[str] = ()
) -> None:
    if canonical_name in fieldnames:
        return
    legacy = first_existing_column(fieldnames, legacy_names)
    if legacy is None:
        fieldnames.append(canonical_name)
    else:
        fieldnames[fieldnames.index(legacy)] = canonical_name


def add_mfu_columns(
    rows: Iterable[dict[str, str]],
    fieldnames: list[str],
    spec: ModelSpec,
    *,
    tp_size_override: int | None,
    output_len: int,
    enable_kv_mirror_opt: bool,
    default_chunked_prefill_size: int,
    peak_tflops_per_gpu: float,
    include_flops_columns: bool,
) -> tuple[list[dict[str, str]], list[str]]:
    required = [PROMPT_LEN_COLUMN, STAGE_COLUMN, THROUGHPUT_COLUMN]
    missing = [name for name in required if name not in fieldnames]
    if missing:
        raise ValueError(f"CSV is missing required column(s): {', '.join(missing)}")
    if tp_size_override is None and PARALLEL_COLUMN not in fieldnames:
        raise ValueError(f"CSV is missing {PARALLEL_COLUMN!r}; pass --tp-size")

    new_fields = list(fieldnames)
    ensure_column(new_fields, OFFLINE_MFU_COLUMN, LEGACY_OFFLINE_MFU_COLUMNS)
    ensure_column(new_fields, SGLANG_MFU_COLUMN, LEGACY_SGLANG_MFU_COLUMNS)
    if include_flops_columns:
        for name in ("FLOPs/token", "GFLOPs/token"):
            if name not in new_fields:
                new_fields.append(name)

    out_rows: list[dict[str, str]] = []
    component_cache: dict[int, FlopComponents] = {}
    for row in rows:
        tp_size = (
            tp_size_override
            if tp_size_override is not None
            else parse_tp_size(row[PARALLEL_COLUMN])
        )
        comp = component_cache.setdefault(
            tp_size, build_components(spec, tp_size=tp_size)
        )
        rank_local_flops = choose_flops_for_row(
            row,
            spec,
            comp,
            output_len=output_len,
            enable_kv_mirror_opt=enable_kv_mirror_opt,
            default_chunked_prefill_size=default_chunked_prefill_size,
        )
        flops_for_normalized_throughput = rank_local_flops * tp_size

        row = dict(row)
        for legacy_name in LEGACY_OFFLINE_MFU_COLUMNS:
            if legacy_name in row and legacy_name != OFFLINE_MFU_COLUMN:
                row.pop(legacy_name)
        for legacy_name in LEGACY_SGLANG_MFU_COLUMNS:
            if legacy_name in row and SGLANG_MFU_COLUMN not in row:
                row[SGLANG_MFU_COLUMN] = row.pop(legacy_name)
            else:
                row.pop(legacy_name, None)
        row.setdefault(SGLANG_MFU_COLUMN, "")
        if include_flops_columns:
            row["FLOPs/token"] = f"{flops_for_normalized_throughput:.0f}"
            row["GFLOPs/token"] = f"{flops_for_normalized_throughput / 1e9:.6f}"

        tokens_per_sec_per_gpu = float(row[THROUGHPUT_COLUMN])
        mfu = (
            tokens_per_sec_per_gpu
            * flops_for_normalized_throughput
            / (peak_tflops_per_gpu * 1e12)
        )
        row[OFFLINE_MFU_COLUMN] = f"{mfu * 100:.3f}"
        out_rows.append(row)

    return out_rows, new_fields


def resolve_config_path(
    *, config: Path | None, model_path: Path | None, csv_path: Path | None
) -> Path:
    candidates: list[Path] = []
    if config is not None:
        candidates.append(config)
    if model_path is not None:
        candidates.append(model_path / "config.json")
    if csv_path is not None and csv_path.exists():
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            first_row = next(reader, None)
            if first_row and first_row.get(MODEL_PATH_COLUMN):
                candidates.append(Path(first_row[MODEL_PATH_COLUMN]) / "config.json")

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    tried = ", ".join(str(path) for path in candidates) or "none"
    raise FileNotFoundError(f"Cannot locate model config.json; tried: {tried}")


def process_csv(
    csv_path: Path,
    output_path: Path,
    spec: ModelSpec,
    *,
    tp_size: int | None,
    output_len: int,
    enable_kv_mirror_opt: bool,
    default_chunked_prefill_size: int,
    peak_tflops_per_gpu: float,
    include_flops_columns: bool,
) -> None:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        rows, fieldnames = add_mfu_columns(
            reader,
            reader.fieldnames,
            spec,
            tp_size_override=tp_size,
            output_len=output_len,
            enable_kv_mirror_opt=enable_kv_mirror_opt,
            default_chunked_prefill_size=default_chunked_prefill_size,
            peak_tflops_per_gpu=peak_tflops_per_gpu,
            include_flops_columns=include_flops_columns,
        )

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Model config.json path.")
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Model directory. Used to find config.json when --config is omitted.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional benchmark CSV to annotate with offline MFU.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV path. Required when --csv is set unless --inplace is used.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite --csv with annotated output.",
    )
    parser.add_argument(
        "--peak-tflops-per-gpu",
        type=float,
        help="Single-GPU dense peak TFLOPS. Required when --csv is set.",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        help="Tensor parallel size. If omitted, parse TP<N> from each CSV row.",
    )
    parser.add_argument(
        "--decode-output-len",
        type=int,
        default=100,
        help="Generated token count used for average decode context. Default: 100.",
    )
    parser.add_argument(
        "--default-chunked-prefill-size",
        type=int,
        default=8192,
        help="Chunked prefill size used when the CSV row has no value. Default: 8192.",
    )
    parser.add_argument(
        "--enable-kv-mirror-opt",
        action="store_true",
        help="Include WeLM KV mirror prefill contraction in FLOPs calculation.",
    )
    parser.add_argument(
        "--include-flops-columns",
        action="store_true",
        help="Also append FLOPs/token and GFLOPs/token columns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_config_path(
        config=args.config, model_path=args.model_path, csv_path=args.csv
    )
    spec = ModelSpec.from_config(config_path)

    if args.csv is None:
        tp_size = args.tp_size or 1
        print_table(
            spec,
            tp_size=tp_size,
            output_len=args.decode_output_len,
            enable_kv_mirror_opt=args.enable_kv_mirror_opt,
            chunked_prefill_size=args.default_chunked_prefill_size,
        )
        return

    if args.peak_tflops_per_gpu is None:
        raise ValueError("--peak-tflops-per-gpu is required when --csv is set")
    if args.inplace:
        output_path = args.csv
    elif args.output is not None:
        output_path = args.output
    else:
        output_path = args.csv.with_name(f"{args.csv.stem}_with_mfu{args.csv.suffix}")

    process_csv(
        args.csv,
        output_path,
        spec,
        tp_size=args.tp_size,
        output_len=args.decode_output_len,
        enable_kv_mirror_opt=args.enable_kv_mirror_opt,
        default_chunked_prefill_size=args.default_chunked_prefill_size,
        peak_tflops_per_gpu=args.peak_tflops_per_gpu,
        include_flops_columns=args.include_flops_columns,
    )
    print(output_path)


if __name__ == "__main__":
    main()
