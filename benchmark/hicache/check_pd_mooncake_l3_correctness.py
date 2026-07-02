import json
import os
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

from bench_pd_mooncake_l3_sweep import (
    HOST,
    MODEL_PATH,
    PAGE_SIZE,
    REPLAY_MAX_NEW_TOKENS,
    X_LEN,
    Y_BASE,
    PDMooncakeL3Sweep,
    atomic_json,
)

NUM_CASES = int(os.environ.get("NUM_CASES", "3"))
LOGPROB_ATOL = float(os.environ.get("LOGPROB_ATOL", "1e-4"))
TOP_LOGPROBS_NUM = int(os.environ.get("TOP_LOGPROBS_NUM", "0"))


def output_ids(resp: dict[str, Any]) -> list[int]:
    ids = resp.get("output_ids") or []
    return [int(x) for x in ids]


def output_logprobs(resp: dict[str, Any]) -> tuple[list[float], list[int]]:
    meta = resp.get("meta_info") or {}
    entries = meta.get("output_token_logprobs") or []
    vals: list[float] = []
    ids: list[int] = []
    for item in entries:
        if isinstance(item, dict):
            vals.append(float(item["logprob"]))
            ids.append(int(item["token_id"]))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            vals.append(float(item[0]))
            ids.append(int(item[1]))
        else:
            raise TypeError(f"unexpected output_token_logprobs item: {item!r}")
    return vals, ids


def first_mismatch(left: list[int], right: list[int]) -> dict[str, Any] | None:
    limit = min(len(left), len(right))
    for idx in range(limit):
        if left[idx] != right[idx]:
            return {"index": idx, "left": left[idx], "right": right[idx]}
    if len(left) != len(right):
        return {"index": limit, "left_len": len(left), "right_len": len(right)}
    return None


def compare_responses(
    case_name: str,
    replay_input_len: int,
    on_storage_cached_tokens: int,
    on_resp: dict[str, Any],
    off_resp: dict[str, Any],
) -> dict[str, Any]:
    on_ids = output_ids(on_resp)
    off_ids = output_ids(off_resp)
    on_lp, on_lp_ids = output_logprobs(on_resp)
    off_lp, off_lp_ids = output_logprobs(off_resp)

    token_mismatch = first_mismatch(on_ids, off_ids)
    logprob_token_mismatch = first_mismatch(on_lp_ids, off_lp_ids)
    logprob_len_equal = len(on_lp) == len(off_lp)
    lp_diffs = [abs(a - b) for a, b in zip(on_lp, off_lp)]
    max_abs = max(lp_diffs) if lp_diffs else None
    mean_abs = mean(lp_diffs) if lp_diffs else None
    expected_storage_floor = max(0, replay_input_len - PAGE_SIZE)

    pass_status = (
        token_mismatch is None
        and logprob_token_mismatch is None
        and logprob_len_equal
        and len(on_lp) == len(on_ids)
        and len(off_lp) == len(off_ids)
        and max_abs is not None
        and max_abs <= LOGPROB_ATOL
        and on_storage_cached_tokens >= expected_storage_floor
    )

    return {
        "case": case_name,
        "pass": pass_status,
        "replay_input_len": replay_input_len,
        "expected_storage_floor": expected_storage_floor,
        "on_storage_cached_tokens": on_storage_cached_tokens,
        "output_tokens_on": len(on_ids),
        "output_tokens_off": len(off_ids),
        "token_ids_equal": token_mismatch is None,
        "token_mismatch": token_mismatch,
        "logprob_count_on": len(on_lp),
        "logprob_count_off": len(off_lp),
        "logprob_token_ids_equal": logprob_token_mismatch is None,
        "logprob_token_mismatch": logprob_token_mismatch,
        "logprob_len_equal": logprob_len_equal,
        "max_abs_logprob_diff": max_abs,
        "mean_abs_logprob_diff": mean_abs,
        "logprob_atol": LOGPROB_ATOL,
    }


class PDMooncakeL3Correctness(PDMooncakeL3Sweep):
    def replay_payload(self, input_ids: list[int]) -> dict[str, Any]:
        return {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": REPLAY_MAX_NEW_TOKENS,
                "ignore_eos": True,
            },
            "return_logprob": True,
            "logprob_start_len": -1,
            "top_logprobs_num": TOP_LOGPROBS_NUM,
        }

    def populate_payload(self, x_ids: list[int]) -> dict[str, Any]:
        return {
            "input_ids": x_ids,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": Y_BASE,
                "ignore_eos": True,
            },
        }

    def run_on_mode(self) -> list[dict[str, Any]]:
        self.log("MODE_START on")
        mode_dir = self.run_dir / "on"
        mode_dir.mkdir(exist_ok=True)
        self.start_mooncake()
        self.start_stack("on")
        cases = []
        try:
            for idx in range(NUM_CASES):
                case_name = f"case{idx}_x{X_LEN}_y{Y_BASE}"
                case_dir = mode_dir / case_name
                case_dir.mkdir(parents=True, exist_ok=True)
                self.log(f"CASE_ON_START {case_name}")

                x_ids = self.make_ids(X_LEN, f"correctness_{idx}")
                populate = self.generate(
                    self.populate_payload(x_ids),
                    timeout=max(1800, Y_BASE * 3),
                )
                y_ids = output_ids(populate)
                if len(y_ids) != Y_BASE:
                    raise RuntimeError(
                        f"{case_name}: expected y_len={Y_BASE}, got {len(y_ids)}"
                    )

                flush = self.flush_all()
                time.sleep(10)
                replay_ids = x_ids + y_ids
                on_replay = self.generate(
                    self.replay_payload(replay_ids),
                    timeout=max(1800, len(replay_ids) * 2),
                )
                replay_flush = self.flush_all()

                case = {
                    "case": case_name,
                    "x_ids": x_ids,
                    "y_ids": y_ids,
                    "replay_input_ids": replay_ids,
                    "populate_meta": self.meta_summary(populate),
                    "on_replay_meta": self.meta_summary(on_replay),
                    "on_replay_response": on_replay,
                    "populate_flush": flush,
                    "on_replay_flush": replay_flush,
                }
                atomic_json(case_dir / "case.json", case)
                cases.append(case)
                atomic_json(mode_dir / "cases.partial.json", cases)
                self.log(
                    "CASE_ON_DONE "
                    + json.dumps(
                        {
                            "case": case_name,
                            "replay_input_len": len(replay_ids),
                            "storage_cached": case["on_replay_meta"][
                                "storage_cached_tokens"
                            ],
                            "cached": case["on_replay_meta"]["cached_tokens"],
                        },
                        sort_keys=True,
                    )
                )
        finally:
            atomic_json(mode_dir / "evidence.json", self.collect_evidence("on"))
            self.cleanup_all()
            time.sleep(5)
        self.log("MODE_DONE on")
        return cases

    def run_off_mode(self, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.log("MODE_START off")
        mode_dir = self.run_dir / "off"
        mode_dir.mkdir(exist_ok=True)
        self.start_mooncake()
        self.start_stack("off")
        compared = []
        try:
            for case in cases:
                case_name = case["case"]
                case_dir = mode_dir / case_name
                case_dir.mkdir(parents=True, exist_ok=True)
                self.log(f"CASE_OFF_START {case_name}")
                replay_ids = [int(x) for x in case["replay_input_ids"]]
                off_replay = self.generate(
                    self.replay_payload(replay_ids),
                    timeout=max(1800, len(replay_ids) * 2),
                )
                replay_flush = self.flush_all()
                on_replay = case["on_replay_response"]
                comparison = compare_responses(
                    case_name=case_name,
                    replay_input_len=len(replay_ids),
                    on_storage_cached_tokens=case["on_replay_meta"][
                        "storage_cached_tokens"
                    ],
                    on_resp=on_replay,
                    off_resp=off_replay,
                )
                row = {
                    **comparison,
                    "on_replay_meta": case["on_replay_meta"],
                    "off_replay_meta": self.meta_summary(off_replay),
                    "off_replay_flush": replay_flush,
                }
                atomic_json(case_dir / "off_replay.response.json", off_replay)
                atomic_json(case_dir / "comparison.json", row)
                compared.append(row)
                atomic_json(mode_dir / "comparisons.partial.json", compared)
                self.log("CASE_OFF_DONE " + json.dumps(comparison, sort_keys=True))
        finally:
            atomic_json(mode_dir / "evidence.json", self.collect_evidence("off"))
            self.cleanup_all()
            time.sleep(5)
        self.log("MODE_DONE off")
        return compared

    def write_correctness_report(self, comparisons: list[dict[str, Any]]) -> None:
        passed = sum(1 for row in comparisons if row["pass"])
        lines = [
            "# PD Mooncake HiCache L3 Correctness",
            "",
            f"- model: `{MODEL_PATH}`",
            "- topology: PD separation, prefill TP1 on GPU0, decode TP1 on GPU1",
            f"- x_len: `{X_LEN}`",
            f"- y_len: `{Y_BASE}` generated by the L3-on populate request",
            f"- replay max_new_tokens: `{REPLAY_MAX_NEW_TOKENS}`",
            f"- cases: `{NUM_CASES}`",
            f"- page_size: `{PAGE_SIZE}`",
            f"- logprob_atol: `{LOGPROB_ATOL}`",
            f"- result: `{passed}/{len(comparisons)}` cases passed",
            "",
            "| case | input_len | L3 storage cached | token ids equal | logprob ids equal | max abs logprob diff | mean abs logprob diff | status |",
            "|---|---:|---:|---|---|---:|---:|---|",
        ]
        for row in comparisons:
            max_abs = row["max_abs_logprob_diff"]
            mean_abs = row["mean_abs_logprob_diff"]
            lines.append(
                "| {case} | {input_len} | {storage} | {tok} | {lp_tok} | {max_abs} | {mean_abs} | {status} |".format(
                    case=row["case"],
                    input_len=row["replay_input_len"],
                    storage=row["on_storage_cached_tokens"],
                    tok="yes" if row["token_ids_equal"] else "no",
                    lp_tok="yes" if row["logprob_token_ids_equal"] else "no",
                    max_abs="n/a" if max_abs is None else f"{max_abs:.8g}",
                    mean_abs="n/a" if mean_abs is None else f"{mean_abs:.8g}",
                    status="PASS" if row["pass"] else "FAIL",
                )
            )
        failures = [row for row in comparisons if not row["pass"]]
        if failures:
            lines.extend(["", "## Failures", ""])
            for row in failures:
                lines.append(
                    "- `{case}` token_mismatch={token_mismatch} "
                    "logprob_token_mismatch={logprob_token_mismatch} "
                    "max_abs_logprob_diff={max_abs_logprob_diff} "
                    "storage={on_storage_cached_tokens}/{expected_storage_floor}".format(
                        **row
                    )
                )
        (self.run_dir / "report.md").write_text("\n".join(lines) + "\n")

    def run(self) -> None:
        self.log(f"run_dir={self.run_dir}")
        self.log(f"sglang_import={self._sglang_import_path()}")
        self.log(f"config_info={json.dumps(self.config_info(), sort_keys=True)}")
        cases = self.run_on_mode()
        comparisons = self.run_off_mode(cases)
        summary = {
            "config": {
                "model": MODEL_PATH,
                "x_len": X_LEN,
                "y_len": Y_BASE,
                "replay_max_new_tokens": REPLAY_MAX_NEW_TOKENS,
                "num_cases": NUM_CASES,
                "page_size": PAGE_SIZE,
                "logprob_atol": LOGPROB_ATOL,
                "host": HOST,
            },
            "model_config": self.config_info(),
            "comparisons": comparisons,
            "pass": all(row["pass"] for row in comparisons),
        }
        atomic_json(self.run_dir / "summary.json", summary)
        self.write_correctness_report(comparisons)
        if not summary["pass"]:
            raise RuntimeError("one or more correctness comparisons failed")
        self.log("CORRECTNESS_DONE")


def main() -> None:
    run_dir = Path(os.environ.get("RUN_DIR", "/tmp/swa-hicache-l3-correctness"))
    bench = PDMooncakeL3Correctness(run_dir)
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
