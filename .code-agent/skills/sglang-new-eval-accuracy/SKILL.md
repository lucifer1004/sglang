---
name: sglang-new-eval-accuracy
description: Run the fixed WeLMv4/SGLang 80b_v4d5 new_eval accuracy regression against a running OpenAI-compatible SGLang endpoint, with optional dataset selection. Use when Codex is asked to run new_eval, run the 80b_v4d5 precision/accuracy backtest for all or selected datasets, compare against the bundled TP4-DP16 baseline, or produce a pass/fail accuracy report with per-dataset counts.
---

# SGLang New Eval Accuracy

Run the fixed `80b_v4d5` `new_eval` accuracy regression and produce a required `PASS`/`FAIL` conclusion. Prefer the bundled script over retyping shell/Python snippets.

The expected user request is a single sentence like: `SGLang 推理服务运行在 http://127.0.0.1:8000，使用 new_eval 跑 80b_v4d5 精度回测，并发设置为 36，数据集用 aa-lcr。`

## Fixed Contract

Required input:

- SGLang OpenAI-compatible endpoint, for example `http://127.0.0.1:8000`.

Optional input:

- Concurrency. Default to `20` if omitted. Use the user's value for all selected jobs and the judge concurrency.
- Dataset selection. Default to all supported datasets if omitted. If the user specifies a dataset, run only the requested dataset(s), for example `数据集用 gpqa-diamond`.

Extract only these user-controlled values:

- Serving URL, for example `http://127.0.0.1:8000`.
- Concurrency, for example `36`.
- Dataset selection, for example `aa-lcr`. If omitted, run all supported datasets.

Do not ask for the binary URL, supported dataset list, output directory, judge config, judge API key, or baseline. Use these fixed values:

```text
new_eval binary = https://mirrors.tencent.com/repository/generic/welm/new_eval/bin/20260702/new_eval-linux-amd64-36dedda2
baseline_dir    = <this skill>/assets/baselines/80b_v4d5/20260607/TP4-DP16
judge_config    = <this skill>/assets/configs/eval.example.5tasks.yaml
run_root        = /tmp/new_eval/80b_v4d5
score_tolerance = 0.01
```

Do not use host-specific absolute home paths. The baseline metrics and judge config are bundled inside this skill and must be read from the skill directory containing this `SKILL.md`.

The judge API key is read from the bundled judge config. Do not print the key, dump the full generated `config.yaml`, or include secrets in the final response.

Interpret `score_tolerance = 0.01` as an absolute 1 percentage point drop:

```text
task passes score gate iff actual_score >= baseline_score - 0.01
```

Read `score` and `sample_count` from the bundled `*_metrics.json` files at run time. To update the baseline, replace the JSON files under `assets/baselines/80b_v4d5/20260607/TP4-DP16/`.

## Dataset Selection

Supported datasets are fixed. Run all of them by default, in this order:

| job name | task type | copied metrics file |
|---|---|---|
| `suite_gary_math` | `gary-math` | `gary_math_metrics.json` |
| `suite_aime_2025` | `aime-2025` | `aime_2025_metrics.json` |
| `suite_gpqa_diamond` | `gpqa-diamond` | `gpqa_diamond_metrics.json` |
| `suite_aa_lcr` | `aa-lcr` | `aa_lcr_metrics.json` |
| `suite_aa_omniscience` | `aa-omniscience-public` | `aa_omniscience_metrics.json` |

If the user requests one or more datasets, pass only those task types to `--tasks`. Preserve the fixed order above. Accept comma-separated, Chinese-comma-separated, or whitespace-separated values. If any requested dataset is unsupported, do not run `new_eval`; report the supported task types.

## Run Command

Locate this skill folder from the path of the `SKILL.md` you loaded, then run its bundled script:

```bash
/path/to/sglang-new-eval-accuracy/scripts/run_new_eval_accuracy.sh \
  --base-url "http://127.0.0.1:8000" \
  --concurrency 20 \
  --tasks "gpqa-diamond"
```

Omit `--tasks` to run all datasets. Replace `/path/to/sglang-new-eval-accuracy` with the actual skill directory; the script then resolves `assets/` relative to itself. Do not substitute a fixed user home path.

For local skill validation only, use `--prepare-only` to generate config and selected task files without downloading or running `new_eval`.

## Run Directory

The script creates one timestamped run directory per invocation:

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

The script copies bundled baseline JSONs into `${RUN_DIR}/baseline/` so each run is self-contained.

## Workflow

1. Normalize the user URL to an OpenAI API base URL. If the user gives `http://host:port`, the script uses `http://host:port/v1`. If the URL already ends in `/v1`, it keeps it.
2. Probe `${API_BASE}/models` with `curl --noproxy '*'`. The script saves the response to `${RUN_DIR}/models.json` and uses the first model id if available; otherwise it uses `welmv4`.
3. Download the fixed `new_eval` binary into `${RUN_DIR}/binary/new_eval`, `chmod +x` it, save `sha256sum`, and save `new_eval --help`.
4. Generate `${RUN_DIR}/config.yaml` and `${RUN_DIR}/selected_tasks.json` from the bundled judge config without printing the judge API key.
5. Run `new_eval run --config "${RUN_DIR}/config.yaml"` with proxy variables unset and tee all output to `${RUN_DIR}/run.log`.
6. Copy each task's produced `metrics.json` into `${RUN_DIR}/metrics/<copied metrics file>`.
7. Generate `summary.json`, `summary.csv`, and `summary.md`, even when `new_eval` exits non-zero.
8. In the final answer, show the first line conclusion from `summary.md`, the summary table, and the run directory path.

## Pass/Fail Rules

Overall `PASS` requires all conditions:

- `new_eval` exit code is `0`.
- All selected actual metrics files exist.
- All selected baseline metrics files exist in `${RUN_DIR}/baseline/`, copied from this skill's bundled baseline assets.
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
