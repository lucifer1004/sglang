---
name: sglang-new-eval-accuracy
description: Run the fixed WeLMv4/SGLang 80b_v4d5 new_eval accuracy regression against a running OpenAI-compatible SGLang endpoint, with optional dataset selection. Use when Codex is asked to run new_eval, run the 80b_v4d5 precision/accuracy backtest for all or selected datasets, compare against the TP4-DP16 baseline, or produce a pass/fail accuracy report with per-dataset counts.
---

# SGLang New Eval Accuracy

Run the fixed `80b_v4d5` `new_eval` accuracy regression and produce a required `PASS`/`FAIL` conclusion. The user should only need to provide the SGLang serving URL, optional concurrency, and optional dataset selection.

## Fixed Contract

Required input:

- SGLang OpenAI-compatible endpoint, for example `http://127.0.0.1:8000`.

Optional input:

- Concurrency. Default to `20` if omitted. Use the user's value for all selected jobs and the judge concurrency.
- Dataset selection. Default to all supported datasets if omitted. If the user specifies a dataset, run only the requested dataset(s), for example `数据集用 gpqa-diamond`.

Do not ask for the binary URL, supported dataset list, output directory, judge endpoint, or baseline. Use these fixed values:

```text
new_eval binary = https://mirrors.tencent.com/repository/generic/welm/new_eval/bin/20260702/new_eval-linux-amd64-36dedda2
baseline_dir    = /home/fhkong/wxwork/new_eval/80b_v4d5/20260607/TP4-DP16
example_config  = /home/fhkong/wxwork/new_eval/configs/eval.example.5tasks.yaml
run_root        = /tmp/new_eval/80b_v4d5
score_tolerance = 0.01
```

Interpret `score_tolerance = 0.01` as an absolute 1 percentage point drop:

```text
task passes score gate iff actual_score >= baseline_score - 0.01
```

Do not hardcode baseline scores in the skill. Read `score` and `sample_count` from `${baseline_dir}/*_metrics.json` at run time so baseline updates are picked up automatically.

## Dataset Selection

Supported datasets are fixed. Run all of them by default, in this order:

| job name | task type | copied metrics file |
|---|---|---|
| `suite_gary_math` | `gary-math` | `gary_math_metrics.json` |
| `suite_aime_2025` | `aime-2025` | `aime_2025_metrics.json` |
| `suite_gpqa_diamond` | `gpqa-diamond` | `gpqa_diamond_metrics.json` |
| `suite_aa_lcr` | `aa-lcr` | `aa_lcr_metrics.json` |
| `suite_aa_omniscience` | `aa-omniscience-public` | `aa_omniscience_metrics.json` |

If the user requests one or more datasets, set `SELECTED_TASKS_RAW` to only those task types and generate jobs only for the matching datasets. Preserve the fixed order above. Accept comma-separated, Chinese-comma-separated, or whitespace-separated values. If any requested dataset is unsupported, do not run `new_eval`; report the supported task types.

## Run Directory

Create one timestamped run directory per invocation:

```bash
RUN_ID="$(date -u '+%Y%m%d_%H%M%S')_pid$$"
RUN_DIR="/tmp/new_eval/80b_v4d5/${RUN_ID}"
mkdir -p "${RUN_DIR}/binary" "${RUN_DIR}/outputs" "${RUN_DIR}/metrics" "${RUN_DIR}/baseline"
```

Use this layout:

```text
${RUN_DIR}/
  config.yaml
  command.txt
  run.log
  new_eval.exitcode
  models.json
  selected_tasks.json
  binary/
    new_eval
    new_eval.sha256
    new_eval.help.txt
  baseline/
    *_metrics.json
  outputs/
  metrics/
    *_metrics.json
  summary.json
  summary.csv
  summary.md
```

## Workflow

1. Normalize the user URL to an OpenAI API base URL. If the user gives `http://host:port`, use `http://host:port/v1`. If the URL already ends in `/v1`, keep it.
2. Probe `${API_BASE}/models` with `curl --noproxy '*'`. Save the response to `${RUN_DIR}/models.json`. Use the first model id if available; otherwise use `welmv4`.
3. Download the fixed `new_eval` binary into `${RUN_DIR}/binary/new_eval`, `chmod +x` it, save `sha256sum`, and save `new_eval --help`.
4. Generate `${RUN_DIR}/config.yaml` and `${RUN_DIR}/selected_tasks.json`.
5. Run `new_eval run --config "${RUN_DIR}/config.yaml"` with proxy variables unset and tee all output to `${RUN_DIR}/run.log`.
6. Copy each task's produced `metrics.json` into `${RUN_DIR}/metrics/<copied metrics file>`.
7. Generate `summary.json`, `summary.csv`, and `summary.md`.
8. In the final answer, show the first line conclusion from `summary.md`, the summary table, and the run directory path.

## Command Template

Use this as the concrete execution shape. Replace `BASE_URL_RAW`, `CONCURRENCY`, and optionally `SELECTED_TASKS_RAW` from the user request.

```bash
set -euo pipefail

BASE_URL_RAW="http://127.0.0.1:8000"
CONCURRENCY="24"
SELECTED_TASKS_RAW=""  # Empty means all datasets. Example: "gpqa-diamond".

NEW_EVAL_URL="https://mirrors.tencent.com/repository/generic/welm/new_eval/bin/20260702/new_eval-linux-amd64-36dedda2"
BASELINE_DIR="/home/fhkong/wxwork/new_eval/80b_v4d5/20260607/TP4-DP16"
EXAMPLE_CONFIG="/home/fhkong/wxwork/new_eval/configs/eval.example.5tasks.yaml"
RUN_ID="$(date -u '+%Y%m%d_%H%M%S')_pid$$"
RUN_DIR="/tmp/new_eval/80b_v4d5/${RUN_ID}"
BIN_DIR="${RUN_DIR}/binary"
BIN="${BIN_DIR}/new_eval"

mkdir -p "${BIN_DIR}" "${RUN_DIR}/outputs" "${RUN_DIR}/metrics" "${RUN_DIR}/baseline"

BASE_URL="${BASE_URL_RAW%/}"
case "${BASE_URL}" in
  */v1) API_BASE="${BASE_URL}" ;;
  *) API_BASE="${BASE_URL}/v1" ;;
esac

cp "${BASELINE_DIR}"/*_metrics.json "${RUN_DIR}/baseline/"

if curl --noproxy '*' -fsS "${API_BASE}/models" > "${RUN_DIR}/models.json"; then
  MODEL_ID="$(python3 -c 'import json,sys; d=json.load(sys.stdin); print((d.get("data") or [{}])[0].get("id",""))' < "${RUN_DIR}/models.json" || true)"
else
  printf '{}\n' > "${RUN_DIR}/models.json"
  MODEL_ID=""
fi
MODEL_ID="${MODEL_ID:-welmv4}"

curl -L --fail --retry 3 --noproxy '*' -o "${BIN}" "${NEW_EVAL_URL}"
chmod +x "${BIN}"
sha256sum "${BIN}" | tee "${BIN_DIR}/new_eval.sha256"
"${BIN}" --help > "${BIN_DIR}/new_eval.help.txt"
```

Generate `config.yaml` without printing the judge API key. Preserve the judge key from `example_config`; do not copy the secret into final responses.

```bash
export RUN_DIR API_BASE MODEL_ID CONCURRENCY EXAMPLE_CONFIG SELECTED_TASKS_RAW
python3 - <<'PY'
import json
import os
import re
from pathlib import Path

TASKS = [
    {"task": "gary-math", "job": "suite_gary_math", "metrics": "gary_math_metrics.json", "aliases": ["gary_math"]},
    {"task": "aime-2025", "job": "suite_aime_2025", "metrics": "aime_2025_metrics.json", "aliases": ["aime_2025"]},
    {"task": "gpqa-diamond", "job": "suite_gpqa_diamond", "metrics": "gpqa_diamond_metrics.json", "aliases": ["gpqa_diamond"]},
    {"task": "aa-lcr", "job": "suite_aa_lcr", "metrics": "aa_lcr_metrics.json", "aliases": ["aa_lcr"]},
    {"task": "aa-omniscience-public", "job": "suite_aa_omniscience", "metrics": "aa_omniscience_metrics.json", "aliases": ["aa-omniscience", "aa_omniscience"]},
]

ALIASES = {}
for item in TASKS:
    names = [item["task"], item["job"], item["metrics"], item["metrics"].replace("_metrics.json", "")]
    names.extend(item.get("aliases", []))
    for name in names:
        ALIASES[name.lower()] = item["task"]
        ALIASES[name.replace("_", "-").lower()] = item["task"]

def parse_selected(raw):
    names = [part.strip() for part in re.split(r"[\s,，]+", raw) if part.strip()]
    if not names:
        return TASKS
    selected = []
    seen = set()
    unknown = []
    for name in names:
        key = name.lower()
        task = ALIASES.get(key) or ALIASES.get(key.replace("_", "-"))
        if not task:
            unknown.append(name)
            continue
        if task in seen:
            continue
        seen.add(task)
        selected.extend(item for item in TASKS if item["task"] == task)
    if unknown:
        supported = ", ".join(item["task"] for item in TASKS)
        raise SystemExit(f"unsupported dataset(s): {', '.join(unknown)}; supported: {supported}")
    return selected

selected_tasks = [
    {"task": item["task"], "job": item["job"], "metrics": item["metrics"]}
    for item in parse_selected(os.environ.get("SELECTED_TASKS_RAW", ""))
]
Path(os.environ["RUN_DIR"], "selected_tasks.json").write_text(json.dumps(selected_tasks, indent=2, ensure_ascii=False))

example = Path(os.environ["EXAMPLE_CONFIG"])
text = example.read_text()

m = re.search(r"(?ms)^judge:\n(?P<body>(?:^[ \t]+.*\n?)+)", text)
if not m:
    raise SystemExit(f"missing judge block in {example}")
judge = m.group("body")

def field(name, default=None):
    mm = re.search(rf"(?m)^[ \t]+{re.escape(name)}:[ \t]*(.*?)[ \t]*(?:#.*)?$", judge)
    if not mm:
        if default is None:
            raise SystemExit(f"missing judge.{name} in {example}")
        return default
    value = mm.group(1).strip()
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        value = value[1:-1]
    return value

def y(value):
    return json.dumps(str(value), ensure_ascii=False)

judge_base_url = field("base_url", "http://welmgateway.production.polaris:30000/v1")
judge_api_key = field("api_key")
judge_model = field("model", "gpt-oss-120b-eval")
judge_rtx = field("rtx", "wenhanli_eval")
judge_max_retries = field("max_retries", "5")
job_lines = []
for item in selected_tasks:
    job_lines.append(f"""  - name: {item['job']}
    type: {item['task']}
    concurrency: {int(os.environ["CONCURRENCY"])}
    rollout_max_retries: 2
""")

config = f"""version: v1

defaults:
  base_url: {y(os.environ["API_BASE"])}
  api_key: "dummy"
  model: {y(os.environ["MODEL_ID"])}
  rtx: ""
  output_root: {y(str(Path(os.environ["RUN_DIR"]) / "outputs"))}

judge:
  base_url: {y(judge_base_url)}
  api_key: {y(judge_api_key)}
  model: {y(judge_model)}
  rtx: {y(judge_rtx)}
  concurrency: {int(os.environ["CONCURRENCY"])}
  max_retries: {int(judge_max_retries)}

jobs:
{''.join(job_lines)}"""

Path(os.environ["RUN_DIR"], "config.yaml").write_text(config)
PY
```

Record and run the command:

```bash
RUN_ARGS=(run --config "${RUN_DIR}/config.yaml")

printf 'env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY NO_PROXY=* no_proxy=* stdbuf -oL %q' "${BIN}" > "${RUN_DIR}/command.txt"
printf ' %q' "${RUN_ARGS[@]}" >> "${RUN_DIR}/command.txt"
printf '\n' >> "${RUN_DIR}/command.txt"

set +e
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
  NO_PROXY='*' no_proxy='*' \
  stdbuf -oL "${BIN}" "${RUN_ARGS[@]}" 2>&1 | tee "${RUN_DIR}/run.log"
NEW_EVAL_EXIT="${PIPESTATUS[0]}"
set -e
printf '%s\n' "${NEW_EVAL_EXIT}" > "${RUN_DIR}/new_eval.exitcode"
```

Copy metrics into the stable `metrics/` names:

```bash
export RUN_DIR
python3 - <<'PY'
import json
import os
import shutil
from pathlib import Path

RUN_DIR = Path(os.environ["RUN_DIR"])
for item in json.loads((RUN_DIR / "selected_tasks.json").read_text()):
    src = RUN_DIR / "outputs" / item["job"] / "output" / item["task"] / "metrics.json"
    dst = RUN_DIR / "metrics" / item["metrics"]
    if src.exists():
        shutil.copy2(src, dst)
PY
```

## Summary Generation

Generate the summary even when `new_eval` exits non-zero. Missing metrics or non-zero exit code must become `FAIL`.

```bash
export RUN_DIR BASELINE_DIR API_BASE MODEL_ID CONCURRENCY
python3 - <<'PY'
import csv
import json
import os
from pathlib import Path

RUN_DIR = Path(os.environ["RUN_DIR"])
BASELINE_DIR = Path(os.environ["BASELINE_DIR"])
TOL = 0.01

TASKS = [
    (item["task"], item["job"], item["metrics"])
    for item in json.loads((RUN_DIR / "selected_tasks.json").read_text())
]

def load_metrics(path):
    with path.open() as f:
        data = json.load(f)
    return data.get("metrics", data)

def as_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default

def as_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default

def count_correct_from_output(job, task, sample_count, score):
    output_path = RUN_DIR / "outputs" / job / "output" / task / f"{task}_output.json"
    if not output_path.exists():
        return round(score * sample_count)
    try:
        data = json.loads(output_path.read_text())
    except Exception:
        return round(score * sample_count)
    samples = data.get("batch_samples") or data.get("samples") or []
    correct = 0
    seen = 0
    for sample in samples:
        metric = ((sample.get("meta") or {}).get("metric") or {})
        if "correct" in metric:
            seen += 1
            correct += bool(metric.get("correct"))
        elif "score" in sample:
            seen += 1
            correct += as_float(sample.get("score")) >= 1.0
    if seen == sample_count:
        return int(correct)
    return round(score * sample_count)

exit_path = RUN_DIR / "new_eval.exitcode"
try:
    exit_code = int(exit_path.read_text().strip())
except Exception:
    exit_code = 1

rows = []
fail_reasons = []
if exit_code != 0:
    fail_reasons.append(f"new_eval exited with code {exit_code}")

for task, job, metrics_name in TASKS:
    task_reasons = []
    baseline_path = BASELINE_DIR / metrics_name
    actual_path = RUN_DIR / "metrics" / metrics_name

    if not baseline_path.exists():
        task_reasons.append(f"missing baseline metrics: {baseline_path}")
        baseline = {}
    else:
        baseline = load_metrics(baseline_path)

    if not actual_path.exists():
        task_reasons.append(f"missing actual metrics: {actual_path}")
        actual = {}
    else:
        actual = load_metrics(actual_path)

    baseline_score = as_float(baseline.get("score"))
    baseline_samples = as_int(baseline.get("sample_count"))
    score = as_float(actual.get("score"))
    sample_count = as_int(actual.get("sample_count"))
    empty = as_int(actual.get("empty_predict_count"))
    judge_fail = as_int(actual.get("judge_parse_failed_count"))
    failed = empty + judge_fail
    expected_samples = baseline_samples

    if expected_samples and sample_count != expected_samples:
        task_reasons.append(f"sample_count {sample_count} != expected {expected_samples}")
    if empty > 0:
        task_reasons.append(f"empty_predict_count={empty}")
    if judge_fail > 0:
        task_reasons.append(f"judge_parse_failed_count={judge_fail}")
    if actual_path.exists() and baseline_path.exists() and score < baseline_score - TOL:
        task_reasons.append(f"score drop {score - baseline_score:+.4f} exceeds -{TOL:.4f}")

    correct = count_correct_from_output(job, task, sample_count, score) if actual_path.exists() else 0
    wrong = max(sample_count - correct - failed, 0)
    status = "PASS" if not task_reasons else "FAIL"
    if task_reasons:
        fail_reasons.extend([f"{task}: {reason}" for reason in task_reasons])

    rows.append({
        "task": task,
        "status": status,
        "score": score,
        "baseline_score": baseline_score,
        "delta": score - baseline_score,
        "threshold": baseline_score - TOL,
        "sample_count": sample_count,
        "expected_sample_count": expected_samples,
        "baseline_sample_count": baseline_samples,
        "correct_count": correct,
        "wrong_count": wrong,
        "failed_count": failed,
        "empty_predict_count": empty,
        "judge_parse_failed_count": judge_fail,
        "reasons": task_reasons,
    })

overall = "PASS" if not fail_reasons else "FAIL"
summary = {
    "result": overall,
    "run_dir": str(RUN_DIR),
    "api_base": os.environ.get("API_BASE", ""),
    "model": os.environ.get("MODEL_ID", ""),
    "concurrency": as_int(os.environ.get("CONCURRENCY", 0)),
    "selected_tasks": [row["task"] for row in rows],
    "baseline_dir": str(BASELINE_DIR),
    "score_tolerance": TOL,
    "new_eval_exit_code": exit_code,
    "fail_reasons": fail_reasons,
    "tasks": rows,
}

(RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

with (RUN_DIR / "summary.csv").open("w", newline="") as f:
    fields = [
        "task", "status", "score", "baseline_score", "delta", "threshold",
        "sample_count", "expected_sample_count", "baseline_sample_count", "correct_count", "wrong_count",
        "failed_count", "empty_predict_count", "judge_parse_failed_count", "reasons",
    ]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        out = dict(row)
        out["reasons"] = "; ".join(row["reasons"])
        writer.writerow(out)

lines = [
    f"# Result: {overall}",
    "",
    f"Run dir: `{RUN_DIR}`",
    f"API base: `{os.environ.get('API_BASE', '')}`",
    f"Model: `{os.environ.get('MODEL_ID', '')}`",
    f"Concurrency: `{os.environ.get('CONCURRENCY', '')}`",
    f"Selected tasks: `{', '.join(row['task'] for row in rows)}`",
    f"Baseline dir: `{BASELINE_DIR}`",
    f"Score tolerance: `{TOL:.4f}` absolute",
    "",
]
if fail_reasons:
    lines += ["## Fail Reasons", ""]
    lines += [f"- {reason}" for reason in fail_reasons]
    lines += [""]

lines += [
    "## Scores",
    "",
    "| task | status | score | baseline | delta | threshold | total | expected | correct | wrong | failed | empty | judge_fail |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['task']} | {row['status']} | {row['score']:.4f} | "
        f"{row['baseline_score']:.4f} | {row['delta']:+.4f} | "
        f"{row['threshold']:.4f} | {row['sample_count']} | "
        f"{row['expected_sample_count']} | "
        f"{row['correct_count']} | {row['wrong_count']} | {row['failed_count']} | "
        f"{row['empty_predict_count']} | {row['judge_parse_failed_count']} |"
    )

(RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n")
print((RUN_DIR / "summary.md").read_text())
PY
```

## Pass/Fail Rules

Overall `PASS` requires all conditions:

- `new_eval` exit code is `0`.
- All selected actual metrics files exist.
- All selected baseline metrics files exist in `/home/fhkong/wxwork/new_eval/80b_v4d5/20260607/TP4-DP16`.
- For every selected task, `actual.sample_count == baseline.sample_count`.
- For every selected task, `empty_predict_count == 0`.
- For every selected task, `judge_parse_failed_count == 0`.
- For every selected task, `actual.score >= baseline.score - 0.01`.

Any missing file, non-zero exit code, sample count mismatch, failed sample, or score drop beyond 1 percentage point is `FAIL`.

## Final Response

Always include:

- The overall `PASS`/`FAIL` conclusion.
- The path to `${RUN_DIR}`.
- The selected task list.
- The score table from `summary.md`.
- A short list of fail reasons when the result is `FAIL`.

Do not print the judge API key or the full generated `config.yaml` in the final response.
