# AttnCP Optimization Log

## 2026-06-26

### Update: KV-Stationary Precision Diagnostics

Additional local diagnostics were run for the CP2 fused Q+FA path under the
hard invariant that each resident KV tile may only be read once.

Shape:

- `batch=2/4`, `local_q_heads=6`, `full_q_heads=12`, `num_kv_heads=1`.
- `head_dim=256`, local KV length `16384`.
- CP0 uses full attention sinks; CP1 uses disabled sinks, matching service
  local-merge semantics.
- Fused CP0/CP1 partial states are merged with `merge_state_v2` and compared
  with the FA3 exact local partial path.

Findings:

- Single-shard and merged local O differences are about one BF16 ulp:
  max `2.441e-04`, mean around `2e-05`.
- LSE differences stay around `9.537e-07`.
- The signed O diff is nearly balanced:
  positive and negative fractions are both about `24%` to `27%`, with signed
  mean around `1e-07` to `4e-07`.
- Therefore the remaining strict service drift does not look like a CP shard
  merge semantic bug or a correctable global rounding bias. It is consistent
  with small per-layer BF16 output differences accumulating through decode.

Negative local experiment:

- Tried forcing the Triton QK dot to use `input_precision="ieee"`.
- Local WeLM-shape tests still passed, but local O mean diff did not improve
  and was slightly worse in the 16k diagnostic.
- Reverted. This does not close the strict gap and should not be kept as a
  cleanup change.

Current implication:

- The Triton fused path remains useful for token-level long-context throughput
  experiments, while strict logprob parity still requires either:
  1. moving Q exchange / remote-Q load into the FA3/CUDA implementation, or
  2. reimplementing FA3's internal mainloop and rounding behavior more exactly.
- Code boundary cleanup:
  `_attncp_try_fused_q_fa_decode(...)` now explicitly documents that the
  current provider is experimental Triton and that a strict-parity provider
  should preserve the call shape while moving Q selection into FA3/CUDA.
  Server startup now logs a warning when
  `SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_FUSED_Q_FA=1` is enabled, so benchmark
  runs cannot silently treat the path as strict-logprob equivalent.

### Update: FA3 Internal Fusion Feasibility Check

The local `sgl-kernel/CMakeLists.txt` pulls FA3 from `sgl-project/sgl-attn` at
commit `bcf72ccc6816b36a5fae2c5a3c027604629785e0` through CMake
`FetchContent`. The checked-out source used by a previous build was found at:

```text
/tmp/tmpel96dzd1/build/_deps/repo-flash-attention-src
```

Relevant code points:

- `hopper/flash.h`: `Flash_fwd_params` currently has one `q_ptr` plus
  `q_batch_stride`, `q_row_stride`, and `q_head_stride`.
- `hopper/flash_api.cpp:set_params_fprop`: fills those Q fields directly from
  the single `q` tensor.
- `hopper/flash_fwd_launch_template.h`: creates
  `CollectiveMainloop::Arguments` from the single Q pointer/stride.
- `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp` and
  `hopper/mainloop_fwd_sm80.hpp`: build `mQ` from `params.ptr_Q` and load it
  into shared memory before the QK mainloop.

Implication:

- A strict-parity implementation that avoids materializing `q_full` should add
  an FA3-side CP2 Q provider: two Q base pointers (`q_local`, `q_peer`) plus
  `cp_rank/local_q_heads`, and select the correct pointer for each logical Q
  head inside the existing FA3 Q load path.
- This keeps FA3's QK MMA, online softmax, sink handling, split combine, and
  output rounding unchanged, so it is the most credible path to `1e-5` strict
  logprob parity.
- It is not a small Python-only change. The actual source is external
  `sgl-attn` fetched by CMake, so the clean integration path is either a pinned
  `sgl-attn` fork/tag or a local patch applied by `sgl-kernel` build.

Attention-only microbench after this check:

```bash
CUDA_VISIBLE_DEVICES=4,5 PYTHONPATH=python .venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  benchmark/kernels/attention/bench_attncp_decode_paths.py \
  --batch-size 45 --kv-lens 32768 --warmup 1 --iters 3 --trials 1 \
  --cuda-graph --target-kv-layout logical --fused-max-splits 10 \
  --output /tmp/attncp_decode_paths_b45_32k_goal_continuation.json
```

Result, local CP KV `16384`, batch `45`:

| path | median |
|---|---:|
| `target_sharded_fullq` | `1274.517 us` |
| `target_sharded_slice_a2a` | `1287.968 us` |
| `target_sharded_fused_q_fa` | `427.211 us` |
| `target_sharded_fused_slice_a2a` | `439.872 us` |

Diff summary:

- Fused/local paths are within BF16 ulp level for local O:
  `fused_q_fa_diff_max=0.000244`.
- `fused_vs_fullq_diff_max=0.000000` in this benchmark output means the final
  merged outputs match at the benchmark's sampled BF16 granularity, but it does
  not contradict the service-level strict logprob drift seen over multi-token
  decode.

### Update: Full Precision Regression and 32k/2k Throughput Sweep

Precision regression:

- Script:
  `/home/fhkong/wxwork/attncp_precision_regression/run_full_precision.sh`
- Script fix before running:
  pass the already documented AttnCP args
  `--attn-cp-kv-chunk-size 1024` and
  `--attn-cp-decode-cuda-graph-max-seq-len 8704`.
- Artifact:
  `/tmp/welmv4_attncp_precision/20260626_064928`
- Controlled TP4 vs TP4+CP2:
  token-level pass, max diff `0.000e+00`.
- MMLU/C-Eval:
  `100 / 100` samples tested, `0 / 100` token mismatches,
  max logprob diff `0.00e+00`, mean logprob diff `0.00e+00`.

32k input / 2k output throughput sweep:

- Workload:
  random-ids, `input_len=32768`, `output_len=2048`,
  `request_rate=inf`, CUDA graph enabled, FA3 prefill/decode.
- TP4 artifact:
  `/tmp/welmv4_attncp_manual_sweep/20260626_065556_tp4_s32768_o2048/summary.tsv`
- TP4+CP2 fused artifact:
  `/tmp/welmv4_attncp_manual_sweep/20260626_070425_tp4_cp2_s32768_o2048/summary.tsv`
- TP4+CP2 settings:
  `--attn-cp-size 2`, `--attn-cp-mode sharded-kv`,
  `--attn-cp-kv-chunk-size 1024`,
  `--attn-cp-decode-cuda-graph-max-seq-len 40960`,
  CP2 Q/O-LSE P2P enabled, fused Q+FA enabled,
  `FUSED_MIN_SEQ_CAP=16384`, `FUSED_MAX_SPLITS=10`.

| path | concurrency | output TPS | mean TTFT ms | mean ITL ms | peak running | max token usage |
|---|---:|---:|---:|---:|---:|---:|
| TP4 | 22 | `433.7903` | `17546.1761` | `42.1645` | 22 | `0.94` |
| TP4 | 23 | `448.5448` | `18204.3237` | `42.4055` | 23 | `0.98` |
| TP4 | 24 | `403.7412` | `21603.6503` | `41.0777` | 23 | `0.98` |
| TP4+CP2 fused | 44 | `633.9916` | `37032.9167` | `51.3287` | 44 | `0.96` |
| TP4+CP2 fused | 45 | `651.2040` | `36279.4609` | `51.4191` | 45 | `0.98` |
| TP4+CP2 fused | 46 | `554.0776` | `38973.2071` | `50.8423` | 45 | `0.98` |

Conclusion:

- Best resident concurrency remains TP4 `23` vs TP4+CP2 `45`, about `1.96x`.
- Best output TPS is TP4 `448.5448` vs TP4+CP2 fused `651.2040`, about `1.45x`.
- TP4+CP2 still has higher per-request TTFT/ITL in this long-context workload;
  the throughput gain comes from larger KV residency capacity.
- Long-context fused-hot strict logprob parity remains unresolved. The current
  full precision regression passes because it exercises the stable guarded
  path, while the 32k hot fused probe still records strict drift.

### Update: Negative Precision Experiments After KV-Stationary Constraint

The decode fused Q+FA path must remain KV-stationary: each local K/V tile is
read once for the resident `(batch, kv_head, split)` program and used to
evaluate all logical CP Q heads mapped to that KV head. The following
experiments preserved that invariant but did not improve strict service-level
logprob parity.

Experiment 1: fp32 split O workspace.

- Change tried:
  allocate `fused_q_fa_split_o` as `torch.float32` instead of model dtype
  (`bf16`) so split merge consumes less-rounded partial O.
- Unit tests:
  `PYTHONPATH=python .venv/bin/python -m pytest python/sglang/jit_kernel/tests/test_attncp_fused_ops.py -q`
  passed (`30 passed`).
- Hot precision artifact:
  `/tmp/attncp_hot_precision_probe/20260626_060508`
- Result:
  token-level pass, but exact-vs-fused strict max diff was `1.144e-02`,
  mean diff `5.693e-03`, worse than the previous best `8.621e-03`.
- Decision:
  reverted. The existing bf16 split O workspace is closer to the current FA3
  service behavior.

Experiment 2: force FA3 `num_splits=4` to match fused 4096-token split at
`seq_cap=16384`.

- Command:
  `SGLANG_WELMV4_FLASH_ATTENTION_NUM_SPLITS=4 ATTN_CP_DECODE_CUDA_GRAPH_MAX_SEQ_LEN=40960 FUSED_MAX_SPLITS=4 /home/fhkong/wxwork/attncp_precision_regression/run_hot_fused_precision_probe.sh`
- Hot precision artifact:
  `/tmp/attncp_hot_precision_probe/20260626_061100`
- Result:
  token-level pass, but exact-vs-fused strict max diff worsened to
  `1.030e-01`, mean diff `3.561e-02`.
- Decision:
  do not force FA3 split count for this path. FA3's default heuristic remains
  closer to the fused Triton path than the forced 4-split setting.

Experiment 3: scan KV blocks in reverse order to match FA3 mainloop order.

- Change tried:
  make both the single-kernel and split fused kernels traverse KV blocks from
  high position to low position. This preserved the KV-stationary invariant and
  still read each KV block once.
- Unit tests:
  after fixing the negative-position mask for tiny test sequences,
  `PYTHONPATH=python .venv/bin/python -m pytest python/sglang/jit_kernel/tests/test_attncp_fused_ops.py -q`
  passed (`30 passed`).
- Hot precision artifact:
  `/tmp/attncp_hot_precision_probe/20260626_062653`
- Result:
  token-level pass, but exact-vs-fused strict max diff worsened to
  `1.243e-02`, mean diff `4.349e-03`, compared with the previous best
  `8.621e-03`.
- Decision:
  reverted. Matching the apparent FA3 `n_block` traversal direction alone is
  not sufficient; the current forward scan is closer at service level.

Experiment 4: handle attention sink in a FA3-style finalize step.

- Change tried:
  do not initialize online softmax state with the sink. Instead, scan only real
  KV tokens and add the sink as an extra denominator term after the KV loop,
  matching the apparent FA3 `Softmax::finalize` structure.
- Unit tests:
  `PYTHONPATH=python .venv/bin/python -m pytest python/sglang/jit_kernel/tests/test_attncp_fused_ops.py -q`
  passed (`30 passed`).
- Hot precision artifact:
  `/tmp/attncp_hot_precision_probe/20260626_063421`
- Result:
  token-level pass, but exact-vs-fused strict max diff worsened to
  `3.721e-01`, mean diff `9.658e-02`.
- Decision:
  reverted. Although the local math is close enough for unit thresholds, the
  full WeLM service path is much closer with the original sink-as-initial-state
  formulation.

Experiment 5: keep row max in raw QK score domain.

- Change tried:
  for non-softcap attention, keep QK scores unscaled for row max and multiply
  `softmax_scale * log2(e)` only in the `exp2` update, matching the visible FA3
  softmax structure. The sink-as-initial-state value was converted into the raw
  score domain to preserve the previous sink semantics.
- Unit tests:
  `PYTHONPATH=python .venv/bin/python -m pytest python/sglang/jit_kernel/tests/test_attncp_fused_ops.py -q`
  passed (`30 passed`).
- Hot precision artifact:
  `/tmp/attncp_hot_precision_probe/20260626_064151`
- Result:
  token-level pass, but exact-vs-fused strict max diff worsened to
  `9.257e-02`, mean diff `2.491e-02`.
- Decision:
  reverted. The scaled-score domain used by the previous fused kernel remains
  closer to service-level FA3 behavior.

### Update: Fixed Split Boundary for Fused Decode

Change:

- The experimental CP2 fused Q+FA kernel now prefers a fixed local-KV split
  size of `4096` tokens.
- Default `SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_FUSED_Q_FA_MAX_SPLITS` is
  raised from `8` to `10`.

Reason:

- The previous split size was derived from `seq_cap / max_splits`. Raising the
  CUDA graph cap from `32768` to `40960` changed split boundaries for the first
  32k tokens and could introduce service-level logprob drift.
- With fixed `4096`-token splits and `max_splits=10`, a `40960` graph cap
  covers `32k input / 2k output` while keeping the prefix split boundaries
  stable.
- The KV-stationary invariant is unchanged: each split owns a disjoint local
  KV range, and split merge only reads partial O/LSE, not K/V.

Hot precision probe after the change:

- Command:
  `ATTN_CP_DECODE_CUDA_GRAPH_MAX_SEQ_LEN=40960 /home/fhkong/wxwork/attncp_precision_regression/run_hot_fused_precision_probe.sh`
- Artifact:
  `/tmp/attncp_hot_precision_probe/20260626_041459`
- Fused profile:
  all captured buckets `batch_size in {1,2,4,8,12,16}` and
  `seq_cap in {16384,32768,40960}` reported `hit_split`.
- Token-level compare:
  TP4, TP4+CP2 exact, and TP4+CP2 fused-hot all generated `[78, 70, 79, 257]`.
- Strict logprob:
  exact-vs-fused max diff `2.847e-02`, mean diff `7.871e-03`.

Interpretation:

- Fixed 4096-token split boundaries reduce the previous `40960` cap drift
  (`3.742e-02` -> `2.847e-02`) but do not make the Triton fused attention math
  strict-logprob equivalent to FA3.
- The fused hot path is still experimental. Token-level behavior is stable on
  this probe, but strict long-context logprob parity is not achieved.

Negative experiment: fp32 `p @ V` accumulation with smaller block.

- Change tried:
  for `head_dim >= 256`, reduce `block_n` from `128` to `64` and compute
  `tl.dot(p, V.to(fp32), input_precision="ieee")`.
- Unit tests:
  `python/sglang/jit_kernel/tests/test_attncp_fused_ops.py -q` still passed.
- Single-layer diagnostic:
  mean O diff improved slightly for some long-context cases.
- Hot precision artifact:
  `/tmp/attncp_hot_precision_probe/20260626_045045`
- Result:
  token-level pass, but exact-vs-fused strict max diff worsened to
  `1.499e-01` from `2.847e-02`.
- Decision:
  reverted. Smaller local O mean diff is not a reliable predictor of
  service-level deterministic decode parity.

Experiment: FA-style exp2 online softmax in the fused Q+FA kernel.

- Change:
  replace `exp/log` with `exp2/log2` in the fused attention online softmax.
  Split partial merge now follows FA3 combine and uses natural `exp/log` over
  already-natural LSE values.
- Unit tests:
  `python/sglang/jit_kernel/tests/test_attncp_fused_ops.py -q` passed.
- Hot precision artifact:
  `/tmp/attncp_hot_precision_probe/20260626_050342`
- Result:
  token-level pass; exact-vs-fused strict max diff improved to `8.621e-03`,
  mean diff `2.549e-03`.
- Previous fixed-split BF16-dot result:
  exact-vs-fused strict max diff `2.847e-02`, mean diff `7.871e-03`.
- Interpretation:
  matching FA-style exp2 softmax reduces service-level drift by about `3.3x`,
  but still does not meet the `1e-5` strict logprob gate.

Attention-only microbench with FA-style exp2 online softmax:

- Artifact:
  `/tmp/attncp_decode_paths_fused_b45_graph_exp2_20260626.json`

| path | median |
|---|---:|
| `target_sharded_fullq` | `1248.115 us` |
| `target_sharded_fused_q_fa` | `453.299 us` |
| `target_sharded_slice_a2a` | `1254.061 us` |
| `target_sharded_fused_slice_a2a` | `435.168 us` |

Interpretation:

- FA-style exp2 online softmax improves hot-probe strict drift and keeps the
  attention-only fused path fast.
- The remaining gap is still too large for strict deterministic parity.

FA3 source check:

- CMake fetches FA3 from `sgl-project/sgl-attn` at
  `bcf72ccc6816b36a5fae2c5a3c027604629785e0`.
- The SM90 mainloop uses `Softmax::scale_apply_exp2` / `exp2f` for online
  attention softmax.
- Split-KV partial O is stored as `ElementPartial = float`, and the combine
  kernel merges split LSE with natural `expf/logf`.
- A service probe with natural split merge and exp2 online softmax preserved
  the current best exact-vs-fused strict diff:
  artifact `/tmp/attncp_hot_precision_probe/20260626_061804`, max diff
  `8.621e-03`, mean diff `2.549e-03`.

Full precision regression with exp2/log2:

- Command:
  `SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_Q_P2P=1 SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_OLSE_P2P=1 SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_FUSED_Q_FA=1 SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_FUSED_Q_FA_MIN_SEQ_CAP=32768 SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_FUSED_Q_FA_MAX_SPLITS=10 /home/fhkong/wxwork/attncp_precision_regression/run_full_precision.sh`
- Artifact:
  `/tmp/welmv4_attncp_precision/20260626_050930`
- Controlled compare:
  token-level pass, max diff `0.000e+00`.
- MMLU/C-Eval regression:
  `0 / 100` token mismatches, max logprob diff `0.00e+00`, mean logprob diff
  `0.00e+00`.

End-to-end TP4+CP2 sweep with exp2/log2:

- Artifact:
  `/tmp/welmv4_attncp_manual_sweep/20260626_051704_tp4_cp2_s32768_o2048/summary.tsv`
- Same launch shape as the previous TP4+CP2 sweep:
  Q P2P, O/LSE P2P, fused Q+FA enabled, `MIN_SEQ_CAP=16384`,
  `MAX_SPLITS=10`, `--attn-cp-decode-cuda-graph-max-seq-len 40960`.

| path | concurrency | output TPS | mean TTFT ms | mean ITL ms | peak running | max token usage |
|---|---:|---:|---:|---:|---:|---:|
| TP4+CP2 exp2 | 44 | `631.7961` | `37174.8972` | `51.5214` | 44 | `0.96` |
| TP4+CP2 exp2 | 45 | `651.2384` | `36279.7221` | `51.4153` | 45 | `0.98` |
| TP4+CP2 exp2 | 46 | `556.4455` | `38685.7945` | `50.6158` | 45 | `0.98` |

Compared with the previous fixed-split BF16-dot service sweep:

- c45 output TPS improved from `625.8572` to `651.2384` (`+4.1%`).
- c45 mean ITL improved from `54.2935 ms` to `51.4153 ms`.

Compared with the TP4 baseline from
`/tmp/welmv4_attncp_manual_sweep/20260626_042832_tp4_s32768_o2048/summary.tsv`:

- Best resident concurrency remains TP4 `23` vs TP4+CP2 `45` (`1.96x`).
- Best output TPS is TP4 `446.9412` vs TP4+CP2 exp2 `651.2384` (`1.46x`).

Selective fused diagnostics:

- Added experimental layer allowlist:
  `SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_FUSED_Q_FA_LAYERS`, format
  `0,1,8-15`.
- `FUSED_LAYERS=0-23`, graph cap `40960`:
  artifact `/tmp/attncp_hot_precision_probe/20260626_052905`,
  exact-vs-fused strict max diff `1.260e-01`.
- `FUSED_LAYERS=24-47`, graph cap `40960`:
  artifact `/tmp/attncp_hot_precision_probe/20260626_053352`,
  exact-vs-fused strict max diff `2.273e-01`.
- Interpretation:
  drift does not come from a simple bad half of layers. Full-layer fused
  execution has substantial cancellation and is more stable than either half
  alone, so selective layer fallback is not a promising strict-parity path.

Split/cap diagnostics:

- `FUSED_MAX_SPLITS=1`, graph cap `40960`:
  artifact `/tmp/attncp_hot_precision_probe/20260626_053844`,
  exact-vs-fused strict max diff `6.964e-02`.
- `FUSED_MAX_SPLITS=10`, graph cap `34816`:
  artifact `/tmp/attncp_hot_precision_probe/20260626_054344`,
  exact-vs-fused strict max diff `8.621e-03`.
- Interpretation:
  single split is worse than fixed 4096-token split ranges, and tightening the
  graph cap from `40960` to `34816` does not improve the current exp2 fused
  drift. The remaining strict mismatch likely requires matching FA3 internals
  more closely or moving the Q exchange into a FA3/CUDA implementation.

Negative experiment: scale Q before QK dot.

- Change tried:
  compute `tl.dot(q * softmax_scale, K^T)` instead of
  `tl.dot(q, K^T) * softmax_scale`.
- Unit tests:
  `python/sglang/jit_kernel/tests/test_attncp_fused_ops.py -q` passed.
- Hot precision artifact:
  `/tmp/attncp_hot_precision_probe/20260626_055335`.
- Result:
  token-level pass, but exact-vs-fused strict max diff worsened to
  `2.107e-01`.
- Decision:
  reverted. Dot-after-scale rounding does not match FA3 behavior for this
  service path.

Attention-only microbench after the change:

- Command:
  `PYTHONPATH=python .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 benchmark/kernels/attention/bench_attncp_decode_paths.py --batch-size 45 --kv-lens 32768 --warmup 2 --iters 5 --trials 1 --cuda-graph --target-kv-layout logical --fused-max-splits 10 --output /tmp/attncp_decode_paths_fused_b45_graph_fixed_split_20260626.json`
- Result for global KV `32768`, local CP KV `16384`, batch `45`:

| path | median |
|---|---:|
| `target_sharded_fullq` | `1247.776 us` |
| `target_sharded_fused_q_fa` | `473.715 us` |
| `target_sharded_slice_a2a` | `1271.802 us` |
| `target_sharded_fused_slice_a2a` | `454.400 us` |

Interpretation:

- The fixed-split fused kernel still has a clear attention-only speedup.
- This remains insufficient proof of service-level TPS/ITL improvement because
  the full decode step includes graph replay behavior, MoE/MLP, o_proj, TP
  communication, scheduler, sampling, and O/LSE exchange/merge.

Full precision regression after the change:

- Command:
  `SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_Q_P2P=1 SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_OLSE_P2P=1 SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_FUSED_Q_FA=1 SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_FUSED_Q_FA_MIN_SEQ_CAP=32768 SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_FUSED_Q_FA_MAX_SPLITS=10 /home/fhkong/wxwork/attncp_precision_regression/run_full_precision.sh`
- Artifact:
  `/tmp/welmv4_attncp_precision/20260626_042134`
- Controlled compare:
  token-level pass, max diff `0.000e+00`.
- MMLU/C-Eval regression:
  `0 / 100` token mismatches, max logprob diff `0.00e+00`, mean logprob diff
  `0.00e+00`.

Interpretation:

- The guarded integration still preserves existing TP4 vs TP4+CP2 precision.
- This full regression mostly exercises the FA3 exact fallback because the
  prompt lengths are below the fused `MIN_SEQ_CAP=32768` guard. Long-context
  fused-hot precision must be judged by the hot probe above.

End-to-end sweep after the change:

- Scenario:
  random ids, `input_len=32768`, `output_len=2048`, `warmup_requests=0`,
  `page_size=1`, `chunked_prefill_size=8192`, `cuda_graph_max_bs=128`.
- TP4 artifact:
  `/tmp/welmv4_attncp_manual_sweep/20260626_042832_tp4_s32768_o2048/summary.tsv`
- TP4+CP2 artifact:
  `/tmp/welmv4_attncp_manual_sweep/20260626_043704_tp4_cp2_s32768_o2048/summary.tsv`
- TP4+CP2 launch:
  Q P2P, O/LSE P2P, fused Q+FA enabled, `MIN_SEQ_CAP=16384`,
  `MAX_SPLITS=10`, `--attn-cp-decode-cuda-graph-max-seq-len 40960`.

| path | concurrency | output TPS | mean TTFT ms | mean ITL ms | peak running | max token usage |
|---|---:|---:|---:|---:|---:|---:|
| TP4 | 22 | `432.4188` | `17217.4634` | `42.4866` | 22 | `0.94` |
| TP4 | 23 | `446.9412` | `18793.6893` | `42.3017` | 23 | `0.98` |
| TP4 | 24 | `402.3543` | `22001.0175` | `41.0889` | 23 | `0.98` |
| TP4+CP2 | 44 | `608.4245` | `37205.4707` | `54.1859` | 44 | `0.96` |
| TP4+CP2 | 45 | `625.8572` | `36129.4827` | `54.2935` | 45 | `0.98` |
| TP4+CP2 | 46 | `533.4415` | `38681.2756` | `53.6681` | 45 | `0.98` |

Summary:

- Best resident concurrency: TP4 `23`, TP4+CP2 `45` (`1.96x`).
- Best output TPS: TP4 `446.9412`, TP4+CP2 `625.8572` (`1.40x`).
- TP4+CP2 still has higher per-request TTFT/ITL at the same long-output
  workload shape; the throughput gain comes from larger resident concurrency,
  not from lower per-token latency.

### Experiment: Decode Q+FA Triton Fusion Prototype

Goal:

- Add a replaceable function for the AttnCP decode path that can replace
  `q_allgather + flash_attn_with_kvcache`.
- Prototype a CP2 Triton implementation that keeps KV resident and reads each
  resident KV sequence split once per `(batch, kv_head, split)`.
- Validate precision before using the fused path for service benchmark.

Implementation state:

- Added `attncp_cp2_fused_q_fa_decode(...)` in
  `python/sglang/srt/layers/attention/attncp_fused_ops.py`.
- Added an env-gated service hook:
  `SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_FUSED_Q_FA=1`.
- The fused kernel is experimental and default-off.
- Added a CUDA-graph-safe seq-cap guard:
  `SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_FUSED_Q_FA_MIN_SEQ_CAP`.
  Default is `32768`, so short-prompt strict regression keeps FA3 exact
  attention math while long-context decode buckets can opt into the fused
  prototype.
- Focused CUDA unit test:

```bash
PYTHONPATH=python python -m pytest \
  python/sglang/jit_kernel/tests/test_attncp_fused_ops.py -q
```

Result:

- `26 passed`.

Important kernel invariant:

- Program granularity is `(batch, kv_head, split)`, not Q head.
- Within one split, the kernel loads the resident K/V tile once and reuses it
  across all CP Q heads mapped to the same KV head.
- Split ranges are non-overlapping. The split kernel masks both
  `kv_pos < seq_len` and `kv_pos < split_end` to avoid double-reading / double
  counting boundary KV.

Single-layer FA3 comparison on WeLM-like decode shapes:

| shape | max O diff | mean O diff | max LSE diff |
|---|---:|---:|---:|
| `seq=4096, window=512, sinks=True, splits=1` | `1.953e-03` | `8.50e-05` | `4.77e-07` |
| `seq=4608, window=512, sinks=True, splits=8` | `9.77e-04` | `1.12e-04` | `4.77e-07` |
| `seq=32768, window=512, sinks=True, splits=8` | `9.77e-04` | `1.37e-04` | `9.54e-07` |
| `seq=32768, full window, sinks=False, splits=8` | `2.44e-04` | `1.95e-05` | `9.54e-07` |

Full precision regression with fused path enabled:

```bash
env \
  SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_FUSED_Q_FA=1 \
  SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_FUSED_Q_FA_MAX_SPLITS=8 \
  SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_Q_P2P=1 \
  SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_OLSE_P2P=1 \
  /home/fhkong/wxwork/attncp_precision_regression/run_full_precision.sh
```

Artifact:

- `/tmp/welmv4_attncp_precision/20260626_000415`

Result:

- Controlled token-level compare: pass.
- MMLU/C-Eval strict regression: fail.
- Token mismatches: `32 / 100`.
- Max logprob diff: `2.75e+01`.
- Mean logprob diff: `4.28e-01`.

Diagnostic:

- Reusing the same TP4 baseline, TP4+CP2 without Triton fused passed strict
  regression:
  - Token mismatches: `0 / 100`.
  - Max logprob diff: `0.00e+00`.
- Reusing the same TP4 baseline, TP4+CP2 Triton fused with
  `SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_FUSED_Q_FA_MAX_SPLITS=1` still failed
  with the same scale:
  - Token mismatches: `32 / 100`.
  - Max logprob diff: `2.75e+01`.
  - Mean logprob diff: `4.28e-01`.
- Therefore the failure is not primarily caused by split-KV merge. It is
  specific to replacing FA3 attention math with the Triton fused attention
  kernel, not an AttnCP sharded-KV semantic issue.
- Added a focused service-shape unit test that simulates CP2 local-merge
  semantics: CP0 injects full attention sinks, CP1 uses disabled sinks, empty
  local KV rows are normalized to `O=0/LSE=sink-or--inf`, then both shards are
  merged with `merge_state_v2`.
  - Command:
    `PYTHONPATH=python .venv/bin/python -m pytest python/sglang/jit_kernel/tests/test_attncp_fused_ops.py::test_attncp_cp2_fused_q_fa_decode_matches_service_merge -q`
  - Result: `2 passed`.
- Strict single-step comparison on WeLM-like service shapes still shows the
  independent Triton attention math is not bitwise-equivalent to FA3:
  - merged O max diff: `4.88e-04`.
  - merged O mean diff: `2.76e-05`.
  - merged LSE max diff: `9.54e-07`.
  - This is small for one layer, but enough to drift deterministic decode over
    48 layers and 32 generated tokens under the strict logprob regression gate.

Tried local fp32 `p @ V` accumulation:

- Focused tests still passed.
- Single-layer O diff improved slightly.
- Full MMLU/C-Eval still failed:
  - Token mismatches: `25 / 100`.
  - Max logprob diff: `2.69e+01`.
  - Mean logprob diff: `3.94e-01`.
- Service-level runtime became slower.
- Reverted this change.

Conclusion:

- The Triton fused Q+FA prototype is useful for kernel-shape exploration, but
  it is not precision-safe under the current WeLM strict `1e-5` logprob gate.
- Do not enable it by default.
- The next precision-safe optimization should keep FA3 for attention math and
  optimize communication around it, or modify FA3/CUDA implementation itself so
  the fused path preserves FA3 numerical behavior.

Seq-cap guarded service precision:

```bash
env \
  SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_FUSED_Q_FA=1 \
  SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_FUSED_Q_FA_MIN_SEQ_CAP=32768 \
  SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_FUSED_Q_FA_MAX_SPLITS=8 \
  SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_Q_P2P=1 \
  SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_OLSE_P2P=1 \
  /home/fhkong/wxwork/attncp_precision_regression/run_full_precision.sh
```

Artifact:

- `/tmp/welmv4_attncp_precision/20260626_011740`

Result:

- Controlled token-level compare: pass, `max_diff=0.00e+00`.
- MMLU/C-Eval strict regression: pass.
- Token mismatches: `0 / 100`.
- Max logprob diff: `0.00e+00`.
- Mean logprob diff: `0.00e+00`.

Interpretation:

- The guard keeps short-prompt regression on FA3 exact attention math.
- This proves the guarded integration does not regress current precision tests.
- It does not prove Triton Q+FA is precision-safe for all long-context requests.

Service sweep with guarded fused path:

Scenario:

- Workload: `input_len=32768`, `output_len=2048`.
- Server: TP4+CP2 sharded-KV, FA3 backend, CUDA graph enabled.
- Env: Q P2P, O/LSE P2P, Triton fused Q+FA enabled.

Artifacts:

- `MIN_SEQ_CAP=32768`:
  `/tmp/welmv4_attncp_manual_sweep/20260626_012507_tp4_cp2_s32768_o2048/summary.tsv`
- `MIN_SEQ_CAP=16384` c45 diagnostic:
  `/tmp/welmv4_attncp_manual_sweep/20260626_013611_tp4_cp2_s32768_o2048/summary.tsv`

Results:

| fused min seq cap | concurrency | output TPS | mean ITL ms | note |
|---:|---:|---:|---:|---|
| `32768` | 44 | `646.07` | `49.83` | similar to P2P exact |
| `32768` | 45 | `665.67` | `49.77` | similar to P2P exact |
| `32768` | 46 | `525.32` | `49.26` | worse over-capacity point |
| `16384` | 45 | `653.58` | `49.97` | more fused hits, slower |

Conclusion:

- The current Triton fused Q+FA prototype can be much faster than the FA3
  local-attention component in a synthetic microbench, but it does not improve
  end-to-end service throughput/ITL in the 32k/2k c45 scenario.
- Lowering the seq-cap guard increases fused hits during CUDA graph capture
  (`hit_split=3664/4096` vs `1216/4096` in the sampled profile) but worsens
  end-to-end throughput.
- Therefore the current Triton prototype is not a completed performance
  optimization. Keep the precision-safe P2P exact path as the production
  baseline.

Additional diagnostics:

- CUDA graph microbench still shows the standalone fused layer path is faster:

| path | B | KV | CUDA graph median |
|---|---:|---:|---:|
| `target_sharded_fullq` | 45 | 32768 | `1245.28 us` |
| `target_sharded_fused_q_fa` | 45 | 32768 | `384.47 us` |

Artifact:

- `/tmp/attncp_decode_paths_fused_b45_graph_20260626.json`

Interpretation:

- CUDA graph replay itself is not the reason fused fails to improve service
  sweep.
- The remaining gap is between the synthetic one-layer attention path and the
  full service decode step. The likely causes are:
  - the synthetic full-Q path overestimates the exact service baseline because
    service uses CP2 Q/O-LSE P2P exchange;
  - full service ITL includes MoE/MLP, o_proj, TP all-reduce, scheduler and
    sampling work that the attention-only microbench does not cover;
  - graph bucket / layer coverage / kv-mirror layer behavior may make the
    fused path affect fewer hot regions than the synthetic benchmark assumes.

Profiling hook:

- Added an env-gated AttnCP decode region profiler:
  - `SGLANG_ATTNCP_DECODE_PROFILE=1`
  - `SGLANG_ATTNCP_DECODE_PROFILE_LAYERS=...`
  - `SGLANG_ATTNCP_DECODE_PROFILE_INTERVAL=...`
- It records `sink_setup`, `q_exchange`, `attention`, `empty_fix`,
  `o_lse_exchange`, `merge`, `out_copy`, and `total` for the workspace local
  merge path.
- Attempted eager profiling with `--disable-cuda-graph`, but current eager
  decode did not emit workspace profile lines in the tested run. Treat this as
  a profiling-tooling gap, not as performance evidence.

### Experiment: Exact FA3 Path With CP2 P2P Exchange

Goal:

- Keep FA3 local attention unchanged.
- Replace CP2 Q and O/LSE all-gather-style exchange with P2P exchange where
  possible.
- Verify this communication optimization does not change precision.

Command shape:

```bash
env \
  SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_Q_P2P=1 \
  SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_OLSE_P2P=1 \
  python -m sglang.launch_server ... \
    --tp 4 \
    --attn-cp-size 2 \
    --attn-cp-mode sharded-kv \
    --decode-attention-backend fa3 \
    --enable-welm-kv-mirror-opt \
    --cuda-graph-max-bs 128
```

Artifact:

- `/tmp/welmv4_attncp_precision/20260626_000415/regression_attncp_p2p_exact_test.log`
- Full script rerun:
  `/tmp/welmv4_attncp_precision/20260626_004353`

Result:

- Full script controlled token-level compare: pass.
- MMLU/C-Eval strict regression: pass.
- Token mismatches: `0 / 100`.
- Max logprob diff: `0.00e+00`.
- Mean logprob diff: `0.00e+00`.

Conclusion:

- Q/O-LSE communication can be optimized while preserving exact FA3 precision.
- This is the safe baseline for the next sweep.
- Triton Q+FA fusion should remain behind the explicit experimental env until
  it can match FA3 service-level precision.

### Sweep: TP4 vs TP4+CP2 P2P Exact

Scenario:

- Date: 2026-06-26 UTC.
- Model:
  `/home/fhkong/models/80a3_v4d5_256k_merge_thinking_kimi_k25_0502_20260503_032335/epoch_003_step_0002610`
- Workload: random ids, `input_len=32768`, `output_len=2048`,
  `request_rate=inf`, `warmup_requests=0`.
- Server args common:
  - `--tp 4`
  - `--page-size 1`
  - `--chunked-prefill-size 8192`
  - `--prefill-attention-backend fa3`
  - `--decode-attention-backend fa3`
  - `--enable-over-encoding`
  - `--enable-welm-kv-mirror-opt`
  - `--disable-radix-cache`
  - `--cuda-graph-max-bs 128`
- TP4+CP2 extra args/env:
  - `--attn-cp-size 2`
  - `--attn-cp-mode sharded-kv`
  - `SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_Q_P2P=1`
  - `SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_OLSE_P2P=1`
  - Triton Q+FA fusion disabled.

Artifacts:

- TP4:
  `/tmp/welmv4_attncp_manual_sweep/20260626_002558_tp4_s32768_o2048/summary.tsv`
- TP4+CP2:
  `/tmp/welmv4_attncp_manual_sweep/20260626_003428_tp4_cp2_s32768_o2048/summary.tsv`

Results:

| config | concurrency | completed | peak running | max queue | output TPS | mean TTFT ms | mean ITL ms | cuda graph |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| TP4 | 22 | 22 | 22 | 21 | `432.36` | `17594.41` | `42.31` | yes |
| TP4 | 23 | 23 | 23 | 22 | `449.48` | `18024.11` | `42.39` | yes |
| TP4 | 24 | 24 | 23 | 23 | `403.45` | `21328.25` | `41.26` | yes |
| TP4+CP2 P2P exact | 44 | 44 | 44 | 43 | `646.27` | `37614.18` | `49.75` | yes |
| TP4+CP2 P2P exact | 45 | 45 | 45 | 43 | `665.58` | `36202.06` | `49.96` | yes |
| TP4+CP2 P2P exact | 46 | 46 | 45 | 45 | `590.34` | `39038.13` | `49.28` | yes |

Best-point comparison:

| metric | TP4 best | TP4+CP2 P2P exact best | ratio / delta |
|---|---:|---:|---:|
| best concurrency | `23` | `45` | `1.96x` |
| best output TPS | `449.48` | `665.58` | `1.48x` |
| mean ITL at best TPS | `42.39 ms` | `49.96 ms` | `+17.9%` |
| mean TTFT at best TPS | `18024.11 ms` | `36202.06 ms` | `+100.9%` |

Conclusion:

- CP2 P2P exact gives a much better throughput result than the earlier
  `~490 tok/s` CP2 measurements while preserving strict precision.
- Capacity scaling is close to `2x`; best throughput improves by `1.48x`.
- Mean ITL is still slower than TP4 by about `18%`, so further decode kernel
  work can still help, but the next fusion attempt must preserve FA3 numerical
  behavior.

## 2026-06-23

### Goal

Align AttnCP precision with NaiveTP and reduce the TTFT/ITL gap, following
`task.md` and `docs/ring-attn/design-sharded-kv-cp.md`.

### Current Working Tree

- Branch: `perf/welm-v4-optimization`
- Current correctness path reconstructs dense full KV for sharded-KV AttnCP.
- Dense path is precision-safe but slow in decode, especially long output.
- Design target is Q-head all-gather + local KV attention + LSE merge, without
  reconstructing full KV.

### Known Evidence Before This Round

- Clean perf AttnCP fast path had decode token drift and failed precision.
- Dense-gather correctness path passed controlled and MMLU/C-Eval precision
  with zero logprob diff.
- Dense-gather 8k input / 512 output / concurrency 4 was much slower:
  TP4 mean ITL around 8.75 ms, TP4+CP2 dense mean ITL around 24.43 ms.

### Experiment 1: Decode Local-Merge For Full-Attention Layers

Hypothesis:

- Use the existing local-merge path only for non-SWA decode layers.
- Keep SWA layers on dense-gather path to avoid changing FA-visible window/page
  metadata.
- This is the smallest experiment that moves decode toward the design target.

Command planned:

```bash
SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE=1 \
  /home/fhkong/wxwork/attncp_precision_regression/run_full_precision.sh
```

Result: pending.

#### 1a. Full-Q Local Merge

Command:

```bash
SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE=1 \
  /home/fhkong/wxwork/attncp_precision_regression/run_full_precision.sh
```

Artifact:

- `/tmp/welmv4_attncp_precision/20260623_203954`

Result:

- Failed strict controlled compare.
- `max_logprob_diff=8.699742317199707`
- `mean_logprob_diff=0.0016671322988645472`
- Output token drift appeared for 1024/2048 prompt lengths.

Conclusion:

- The existing one-shot all-gathered-Q local-merge path is not safe to enable.

#### 1b. Eager Full-Q Local Merge

Artifact:

- `/tmp/welmv4_attncp_precision/eager_local_merge_20260623_204553`

Result:

- Failed in the same way with `--disable-cuda-graph`.
- This rules out CUDA graph replay as the primary cause.

#### 1c. Empty Local-Shard Identity Patch

Artifact:

- `/tmp/welmv4_attncp_precision/eager_local_merge_emptyfix_20260623_204949`

Result:

- No measurable change from 1b.

Conclusion:

- Empty local shard LSE identity is not the dominant issue.

#### 1d. Q-Shard Local Merge

Change:

- Keep Q all-gather, but run FA per original TP Q-head shard instead of one
  all-gathered GQA call.
- Merge each Q shard's `(out, lse)` across CP ranks and return the local shard.

Artifact:

- `/tmp/welmv4_attncp_precision/eager_local_merge_qshard_20260623_205513`

Result:

- Controlled output tokens matched for 64/512/1024/2048 prompt lengths.
- Strict logprob compare still failed:
  - `max_logprob_diff=0.999237060546875`
  - `mean_logprob_diff=0.0004740602194260793`
  - `issue_count=272`

Conclusion:

- Q-shard local merge removes token drift in the controlled test and is a
  reasonable performance candidate under the task's non-bitwise criterion.
- It is not strict logprob-equivalent to dense TP4.

#### 1e. Q-Shard Local Merge With CUDA Graph

Artifacts:

- `/tmp/welmv4_attncp_perf/20260623_205905`
- `/tmp/welmv4_attncp_precision/graph_local_merge_qshard_concat_20260623_210712`

Result:

- TP4 benchmark completed for `s8k_o512_c4`.
- TP4+CP2 q-shard local-merge server did not reach ready state with cuda graph.
- CUDA graph capture consistently stalled around the second capture bucket
  (`bs=12`) with GPU utilization at 0%.
- Reworking q-shard local merge from per-shard O/LSE collectives to local concat
  plus one O/LSE all-gather did not fix the stall.

Conclusion:

- Q-shard local merge is not currently compatible with the required cuda graph
  path.
- Do not enable it by default until graph capture/replay is fixed.

#### 1f. Q-Shard Local Merge Eager Benchmark

Artifact:

- `/tmp/welmv4_attncp_perf/eager_qshard_20260623_211016`

Scenario:

- `input_len=8192`
- `output_len=512`
- `concurrency=4`
- `num_prompts=16`
- Both TP4 and TP4+CP2 ran with `--disable-cuda-graph`.

Result:

| Config | Completed | Request throughput | Output throughput | Mean TTFT | Mean ITL |
|---|---:|---:|---:|---:|---:|
| TP4 eager | 16 | 0.1243 req/s | 63.66 tok/s | 1724.86 ms | 59.70 ms |
| TP4+CP2 q-shard eager | 16 | 0.0859 req/s | 43.97 tok/s | 2119.94 ms | 87.19 ms |

Conclusion:

- Q-shard local merge is slower than TP4 even without cuda graph.
- Combined with the graph capture stall, this is not a good optimization path.
- The q-shard local-merge code was reverted after this experiment.

### Experiment 2: Current Default Dense/SWA-Window Path Benchmark

Artifact:

- `/tmp/welmv4_attncp_perf/20260623_212116`

Scenario:

- `input_len=8192`
- `output_len=512`
- `concurrency=4`
- `num_prompts=16`
- CUDA graph enabled.
- TP4 baseline and TP4+CP2 sharded-KV ran on the same dirty working tree.
- Local-merge env was not set, so CP2 used the default dense-gather path.

Result:

| Config | Completed | Request throughput | Output throughput | Mean TTFT | Mean ITL | Peak GPU memory |
|---|---:|---:|---:|---:|---:|---:|
| TP4 | 16 | 0.6421 req/s | 328.73 tok/s | 1762.04 ms | 8.75 ms | 81665 MiB |
| TP4+CP2 | 16 | 0.2629 req/s | 134.58 tok/s | 1986.49 ms | 25.94 ms | 63681 MiB |

Comparison:

- TTFT: CP2 is about `+12.7%` slower.
- ITL: CP2 is about `+196.3%` slower.
- Peak memory: CP2 saves about `17.98 GiB` per measured GPU versus TP4.

Conclusion:

- The current SWA decode-window dense gather did not materially reduce the ITL
  bottleneck. It is slightly worse than the earlier exact-dense benchmark
  (`mean_itl_ms=24.43` in `/tmp/welmv4_attncp_perf/20260623_201055`).
- The next optimization should target the full-attention-layer dense KV
  reconstruction/all-reduce path, not q-shard local merge.

### Experiment 3: Full-Q Local Merge With `pack_gqa=False`

Artifact:

- `/tmp/welmv4_attncp_precision/eager_fullq_packgqa_false_20260623_212751`

Scenario:

- TP4 baseline reused from
  `/tmp/welmv4_attncp_precision/eager_local_merge_qshard_20260623_205513/controlled_tp4.json`.
- CP2 ran eager with `SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE=1`.
- The local-merge FA call was temporarily forced to `pack_gqa=False`.

Result:

- Same failure pattern as the original full-Q local merge.
- `max_logprob_diff=8.699742317199707`
- `mean_logprob_diff=0.001666...`
- Output tokens matched for 64/512 prompt lengths, but drifted for 1024/2048.

Conclusion:

- Full-Q local-merge drift is not fixed by disabling FA3 GQA packing.
- The temporary `pack_gqa=False` code change was reverted.

### Experiment 4: Remove SWA Window-Only Dense Gather

Artifact:

- `/tmp/welmv4_attncp_perf/20260623_213031`

Scenario:

- Temporarily removed the SWA decode-window dense gather branch.
- `input_len=8192`, `output_len=512`, `concurrency=4`, `num_prompts=16`.
- CUDA graph enabled.

Result:

| Config | Completed | Request throughput | Output throughput | Mean TTFT | Mean ITL | Peak GPU memory |
|---|---:|---:|---:|---:|---:|---:|
| TP4 | 16 | 0.6576 req/s | 336.69 tok/s | 1607.63 ms | 8.77 ms | 81665 MiB |
| TP4+CP2 without SWA window gather | 16 | 0.2030 req/s | 103.95 tok/s | 2005.68 ms | 34.70 ms | 63665 MiB |

Conclusion:

- Removing SWA window-only dense gather makes CP2 ITL much worse.
- The SWA window-only dense gather branch is beneficial and was restored.
- The remaining bottleneck is still the full-attention layers and/or graph cap,
  not the SWA window branch.

### Experiment 5: Dense Decode CUDA Graph Seq Cap

Artifact:

- `/tmp/welmv4_attncp_perf/20260623_213720`

Scenario:

- `input_len=8192`
- `output_len=512`
- `concurrency=4`
- `num_prompts=16`
- CUDA graph enabled.
- TP4 baseline and TP4+CP2 sharded-KV ran on the same dirty working tree.
- CP2 used `SGLANG_ATTNCP_DENSE_DECODE_GRAPH_MAX_SEQ_LEN=8704`.

Result:

| Config | Completed | Request throughput | Output throughput | Mean TTFT | Mean ITL | Peak GPU memory |
|---|---:|---:|---:|---:|---:|---:|
| TP4 | 16 | 0.6572 req/s | 336.48 tok/s | 1611.18 ms | 8.77 ms | 81665 MiB |
| TP4+CP2, graph cap 8704 | 16 | 0.3225 req/s | 165.13 tok/s | 2008.80 ms | 20.38 ms | 63501 MiB |

Comparison:

- Versus TP4 in the same run:
  - TTFT: CP2 is about `+24.7%` slower.
  - ITL: CP2 is about `+132.4%` slower.
  - Peak memory: CP2 saves about `17.7 GiB` per measured GPU.
- Versus Experiment 2 CP2 default graph cap:
  - Mean ITL improves from `25.94 ms` to `20.38 ms`.
  - Output throughput improves from `134.58 tok/s` to `165.13 tok/s`.

Conclusion:

- The CUDA graph decode seq cap is a real performance factor for the dense
  gather correctness path.
- The improvement is meaningful but not enough: CP2 decode is still over 2x
  slower than TP4 in this 8k/512/c4 scenario.
- This cap must pass precision regression before being considered a usable
  optimization knob or default.

Precision validation:

- Artifact: `/tmp/welmv4_attncp_precision/20260623_214448`
- Env: `SGLANG_ATTNCP_DENSE_DECODE_GRAPH_MAX_SEQ_LEN=8704`
- Controlled generation: PASS
  - `max_logprob_diff=0.0`
  - `mean_logprob_diff=0.0`
  - `issue_count=0`
- MMLU/C-Eval regression: PASS
  - `samples=100/100`
  - `token_mismatches=0`
  - `max_logprob_diff=0.00e+00`
  - `mean_logprob_diff=0.00e+00`

Updated conclusion:

- For the current 8k/512/c4 validation scenario, graph cap 8704 improves CP2
  decode performance without changing TP4 vs TP4+CP2 precision.
- The remaining open question is how to choose or expose this cap safely for
  other input/output length distributions.

### Experiment 6: Q-Granular Local Merge

Artifact:

- `/tmp/welmv4_attncp_precision/local_merge_qgranular_20260623_215913`

Scenario:

- Temporarily changed experimental local-merge to keep each FA call at the
  original TP Q-head shard granularity.
- Reused the TP4 controlled baseline from
  `/tmp/welmv4_attncp_precision/20260623_214448/controlled_tp4.json`.
- CP2 ran eager with
  `SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE=1` and
  `--disable-cuda-graph`.

Result:

- Controlled output token ids matched for prompt lengths `64/512/1024/2048`.
- Strict logprob compare failed:
  - `max_logprob_diff=0.999237060546875`
  - `mean_logprob_diff=0.000500941446027028`
  - `issue_count=283`

Conclusion:

- The full-Q local-merge token drift is caused by changing the FA call from
  original TP Q-head shard granularity to a gathered full-GQA call.
- Keeping original Q-head granularity removes token drift for this controlled
  case, but it still does not match strict logprobs.
- This path is not a usable default optimization. It also matches the earlier
  q-shard experiment pattern, which was slower in eager mode and unsuitable for
  cuda graph without further redesign.
- The temporary q-granular code was reverted after the experiment.

### Experiment 7: Explicit Decode CUDA Graph Cap CLI

Code change:

- Added `--attn-cp-decode-cuda-graph-max-seq-len`.
- The sharded-KV AttnCP dense decode correctness path now uses this explicit
  CLI cap before falling back to the legacy env var or `max_prefill_tokens`.
- Regression and benchmark scripts pass `--attn-cp-decode-cuda-graph-max-seq-len 8704`.

Precision validation:

- Artifact: `/tmp/welmv4_attncp_precision/20260623_220555`
- Controlled generation: PASS
  - `max_logprob_diff=0.0`
  - `mean_logprob_diff=0.0`
  - `issue_count=0`
- MMLU/C-Eval regression: PASS
  - `samples=100/100`
  - `token_mismatches=0`
  - `max_logprob_diff=0.00e+00`
  - `mean_logprob_diff=0.00e+00`

Performance validation:

- Artifact: `/tmp/welmv4_attncp_perf/20260623_221052`
- Scenario: 8k input / 512 output / concurrency 4 / 16 prompts.

| Config | Completed | Request throughput | Output throughput | Mean TTFT | Mean ITL | Peak GPU memory |
|---|---:|---:|---:|---:|---:|---:|
| TP4 | 16 | 0.6556 req/s | 335.64 tok/s | 1626.94 ms | 8.77 ms | 81665 MiB |
| TP4+CP2, graph cap 8704 | 16 | 0.3227 req/s | 165.20 tok/s | 2010.88 ms | 20.36 ms | 63501 MiB |

Comparison:

- TTFT: CP2 is `+23.6%` slower than TP4.
- ITL: CP2 is `+132.3%` slower than TP4.
- Output throughput: CP2 is `50.8%` lower than TP4.
- Peak memory: CP2 saves `18164 MiB` / `17.7 GiB` / `22.2%` per measured GPU.

Conclusion:

- The explicit CLI parameter is cleaner than relying on an env-only cap and
  preserves the current TP4 vs TP4+CP2 precision result.
- It does not solve the main performance issue. CP2 still pays a large decode
  overhead in the dense Q/KV gather correctness path.
- Do not keep optimizing blindly from this point. The next useful step is to
  attribute the decode overhead with targeted profiling around the all-gather,
  dense FA call, cuda graph replay shape, and scheduler/cache update path.

### Experiment 8: Existing Local-Merge Switch As Attribution Probe

Performance artifact:

- `/tmp/welmv4_attncp_perf/20260623_221744`
- Scenario: 8k input / 512 output / concurrency 4 / 16 prompts.
- Env: `SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE=1`

| Config | Completed | Request throughput | Output throughput | Mean TTFT | Mean ITL | Peak GPU memory |
|---|---:|---:|---:|---:|---:|---:|
| TP4 | 16 | 0.6555 req/s | 335.60 tok/s | 1630.51 ms | 8.76 ms | 81665 MiB |
| TP4+CP2 local-merge probe | 16 | 0.4552 req/s | 233.05 tok/s | 2074.33 ms | 13.16 ms | 63449 MiB |

Precision probe:

- Artifact: `/tmp/welmv4_attncp_precision/local_merge_probe_20260623_222239`
- Controlled compare: FAIL
  - `max_logprob_diff=8.699742317199707`
  - `mean_logprob_diff=0.0016671322988645472`
  - `issue_count=191`
  - Output token ids drift for longer controlled prompts.

Conclusion:

- Local-merge proves that the dense K/V reconstruction path is a major ITL
  contributor: CP2 ITL improves from about `20.36 ms` to `13.16 ms`.
- The current local-merge implementation is not usable: it changes generated
  tokens, not just low-order logprobs.
- The next real optimization must fix the `Q allgather + LSE merge` math and
  topology/head mapping first. Enabling this switch by default would trade
  correctness for speed.

### Experiment 9: SWA Compact Decode Window Attempt

Attempted code change:

- For SWA decode, reconstruct only the visible 513-token dense K/V window and
  call FA with a compact page table instead of scattering the window back into a
  `[batch, max_seq_len]` temporary dense buffer.

Precision probe:

- Artifact: `/tmp/welmv4_attncp_precision/swa_compact_probe_20260623_222900`
- Controlled compare: FAIL
  - `max_logprob_diff=1.293`
  - `mean_logprob_diff=9.906e-04`
  - `issue_count=570`
  - First mismatch: `req[0].text mismatch 'mti0.01s.' vs 'mti0.01a0'`

Conclusion:

- Compacting the SWA decode window changes FA semantics in this model. The
  likely reason is that the original FA `window_size` path is not equivalent to
  renumbering the visible K window to a compact local sequence when attention
  sink and WeLM decode metadata are involved.
- The attempted code change was reverted. Current default remains the strict
  dense/window correctness path.

### Source Finding: Current Head Layout Matches Design Target

Relevant code:

- `python/sglang/srt/models/welmv4.py`
  - `Qwen2MoeAttention` uses `get_tensor_model_parallel_rank()` /
    `get_tensor_model_parallel_world_size()` when `is_cp_kv_sharded()` is true.
  - `qkv_proj` and `o_proj` are therefore constructed with global TP rank/size
    in sharded-KV mode.
- `docs/ring-attn/design-sharded-kv-cp.md`
  - Target design says Q projection should remain global TP sharded
    (`H/TP` heads per rank), while attention internally all-gathers Q within
    `sharded_kv_cp_group` to `H/attn_tp`.

Implication:

- The current source already matches the design target for QKV/o_proj head
  layout in sharded-KV mode.
- The existing local-merge switch failing precision must therefore be caused by
  the local partial attention math/metadata path instead of by QKV/o_proj shard
  layout. The next candidates are LSE orientation, sink injection, local page
  table/cache-seqlens construction, and FA metadata differences versus the
  dense correctness path.

### Experiment 10: Local-Merge Failure Attribution

Artifacts:

- `/tmp/welmv4_attncp_precision/local_merge_eager_probe_20260623_223513`
- `/tmp/welmv4_attncp_precision/local_merge_emptyfix_eager_probe_20260623_223829`
- `/tmp/welmv4_attncp_precision/local_merge_no_kvmirror_probe_20260623_224025`
- `/tmp/welmv4_attncp_precision/local_merge_no_sink_probe_20260623_224354`
- `/tmp/welmv4_attncp_precision/local_merge_debug_compare_20260623_224801`

Checks:

- Disabled CUDA graph: local-merge failed with the same drift pattern.
- Patched empty local KV rows to use the softmax identity state: no measurable
  change.
- Disabled `welm-kv-mirror-opt`: local-merge still failed.
- Disabled attention sink through model override: local-merge still failed.
- Compared local-merge output against dense full-KV output inside the same CP2
  decode forward.

Result:

- The failure is not primarily caused by CUDA graph replay, kv mirror, attention
  sink, or empty local KV rows.
- Full-attention layers show BF16-scale output differences versus dense FA even
  within the same forward, typically with max absolute differences around
  `7.8e-03` to `3.1e-02` on affected layers.
- SWA layers are not part of this probe because local-merge is disabled for
  sliding-window decode layers.

Updated conclusion:

- The most likely cause is FA call-shape non-equivalence: full-Q local-merge
  changes the decode FA invocation from the original TP-local Q-head shard shape
  to a gathered full-GQA shape. The math is intended to be equivalent, but the
  FA kernel path and reduction order differ enough in BF16 to move WeLM logits
  and sometimes generated tokens.
- Q-granular local-merge reduced token drift in the controlled case, but it was
  not strict-logprob equivalent and was not a viable CUDA graph/performance path.
- Do not turn on local-merge as a correctness-preserving optimization. The
  current safe optimization remains the dense correctness path plus the decode
  CUDA graph sequence cap.

### Next Profiling Target

The remaining CP2 ITL gap should be attributed before more algorithm changes:

- full-attention-layer dense K/V reconstruction and `all_reduce_coalesced`
- dense FA call after reconstructing K/V
- CUDA graph eager fallback caused by sequence length cap
- scheduler/cache metadata work around decode

Only optimizations that preserve the TP4 vs TP4+CP2 precision regression should
be promoted out of experiment flags.

### Experiment 11: Skip Decode Local-CP Metadata On Dense Path

Profile artifact:

- `/tmp/welmv4_attncp_profile/20260623_225607`

Profile scenario:

- CP2 sharded-KV only
- `input_len=8192`
- `output_len=64`
- `concurrency=4`
- `num_prompts=8`
- profile stage: decode, 10 steps

Finding:

- The default dense correctness path does not use `cp_local_page_table` or
  `cp_local_cache_seqlens_int32`.
- CUDA graph capture/replay still built this local metadata every decode step.
- The trace showed `_set_cuda_graph_sharded_kv_decode_metadata` /
  `_set_sharded_kv_decode_metadata` around `20-22 ms` per 10 profiled decode
  steps per TP rank, with visible `nonzero`, gather, scatter, and elementwise
  kernels.

Code change:

- Cache `enable_attn_cp_decode_local_merge` from
  `SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE`.
- Only allocate and construct local CP decode metadata when the experimental
  local-merge path is enabled.
- Default dense correctness path keeps using full page tables and dense K/V
  reconstruction, so sharded-KV semantics and precision should be unchanged.

Benchmark artifact:

- `/tmp/welmv4_attncp_perf/cp2_metadata_guard_20260623_230100`

Benchmark scenario:

- CP2 sharded-KV only
- `input_len=8192`
- `output_len=512`
- `concurrency=4`
- `num_prompts=16`

Result:

| Config | Completed | Output throughput | Mean TTFT | Mean ITL | Peak GPU memory |
|---|---:|---:|---:|---:|---:|
| TP4+CP2 before guard | 16 | 165.20 tok/s | 2010.88 ms | 20.36 ms | 63501 MiB |
| TP4+CP2 metadata guard | 16 | 167.23 tok/s | 1997.60 ms | 20.10 ms | 63499 MiB |

Conclusion:

- This is a safe but small improvement: about `1.3%` lower mean ITL in the
  measured 8k/512/c4 CP2 case.
- The optimization removes a confirmed unused default-path cost, but it does not
  change the main conclusion: dense K/V reconstruction and NCCL all-reduce
  remain the dominant decode bottleneck.

Precision validation:

- Artifact: `/tmp/welmv4_attncp_precision/20260623_230400`
- Controlled generation: PASS
  - `max_logprob_diff=0.0`
  - `mean_logprob_diff=0.0`
  - `issue_count=0`
- MMLU/C-Eval regression: PASS
  - `samples=100/100`
  - `token_mismatches=0`
  - `max_logprob_diff=0.00e+00`
  - `mean_logprob_diff=0.00e+00`

### Experiment 12: TP4 vs CP2 Decode Profile Attribution

Artifacts:

- TP4 profile: `/tmp/welmv4_attncp_profile/tp4_20260623_231004`
- CP2 post-guard profile:
  `/tmp/welmv4_attncp_profile/cp2_post_guard_20260623_231309`

Profile scenario:

- `input_len=8192`
- `output_len=64`
- `concurrency=4`
- `num_prompts=8`
- profile stage: decode, 10 steps

Important note:

- Torch profiler inflates the end-to-end benchmark latencies, so the serving
  metrics in these profile runs should not be used as performance numbers.
- The useful signal is the relative trace breakdown.

Average per-rank trace totals over the profiled decode window:

| Item | TP4 | CP2 post-guard | CP2 - TP4 |
|---|---:|---:|---:|
| `replay_metadata` | 2.214 ms | 2.153 ms | -0.061 ms |
| local CP metadata | 0.000 ms | 0.000 ms | 0.000 ms |
| `ncclDevKernel_AllReduce` | 0.000 ms | 48.762 ms | +48.762 ms |
| FA kernels | 22.921 ms | 23.106 ms | +0.185 ms |
| K/V gather kernels | 0.000 ms | 12.864 ms | +12.864 ms |
| K/V scatter kernels | 0.000 ms | 3.645 ms | +3.645 ms |
| elementwise kernels | 5.109 ms | 47.457 ms | +42.348 ms |
| direct-copy kernels | 2.003 ms | 7.721 ms | +5.719 ms |

Conclusion:

- The metadata guard worked: local CP metadata is gone from the default dense
  path, and replay metadata is back to TP4-level cost.
- FA itself is no longer the differentiator in the post-guard profile.
- The remaining CP2-specific decode cost is the dense K/V reconstruction path:
  cache gather, mask/zero/copy elementwise kernels, scatter for SWA windows, and
  especially NCCL all-reduce of reconstructed dense K/V.
- The next meaningful optimization would need to reduce or replace dense K/V
  reconstruction while keeping TP4 vs CP2 precision stable. A compact
  all-gather plus scatter-back-to-logical-order path is a possible experiment,
  but it must reuse the allocator's owner rule:
  `floor(position / cp_kv_chunk_size) % cp_size`. It also needs fixed maximum
  compact shapes to remain CUDA-graph compatible.

### Experiment 13: Compact Dense K/V Reconstruction Probe

Attempted code change:

- Added an env-gated experimental path
  `SGLANG_ATTNCP_EXPERIMENTAL_DENSE_COMPACT_GATHER=1`.
- For non-SWA decode, compacted this rank's real CP-owned K/V positions into a
  fixed-width buffer, all-gathered compact K/V across CP ranks, then scattered
  back to dense logical order before calling the same dense FA path.
- The owner mapping reused the allocator rule:
  `floor(position / cp_kv_chunk_size) % cp_size`.

Precision artifacts:

- Compact eager:
  `/tmp/welmv4_attncp_precision/compact_gather_eager_20260623_232049`
- Dense eager control:
  `/tmp/welmv4_attncp_precision/dense_eager_probe_20260623_232301`
- Compact-vs-dense eager compare:
  `/tmp/welmv4_attncp_precision/compact_vs_dense_eager_compare_20260623_232301`
- Compact graph:
  `/tmp/welmv4_attncp_precision/compact_gather_graph_20260623_232501`

Precision result:

- Compact eager vs dense eager: PASS, `max_logprob_diff=0.0`.
- Compact graph vs TP4 graph controlled baseline: PASS,
  `max_logprob_diff=0.0`, `mean_logprob_diff=0.0`, `issue_count=0`.
- A dense eager vs graph baseline comparison has the same non-zero logprob diff
  pattern as compact eager, so that diff is from eager/graph execution mode, not
  compact reconstruction.

Performance artifact:

- `/tmp/welmv4_attncp_perf/compact_gather_20260623_232713`

Benchmark scenario:

- CP2 sharded-KV only
- `input_len=8192`
- `output_len=512`
- `concurrency=4`
- `num_prompts=16`

Result:

| Config | Completed | Output throughput | Mean TTFT | Mean ITL | Peak GPU memory |
|---|---:|---:|---:|---:|---:|
| TP4+CP2 metadata guard | 16 | 167.23 tok/s | 1997.60 ms | 20.10 ms | 63499 MiB |
| TP4+CP2 compact gather | 16 | 80.21 tok/s | 2012.63 ms | 46.13 ms | 64163 MiB |

Conclusion:

- The compact gather idea is precision-safe in this controlled test, including
  CUDA graph.
- It is not a useful optimization in the current Python/Torch implementation:
  additional cumsum/scatter/gather/loop work dominates the saved NCCL traffic.
- The attempted experimental code was removed after this result.
- A future version would need a fused CUDA/Triton reconstruction kernel or a FA
  backend that can consume compact CP-owned K/V directly. Rebuilding dense K/V
  with generic PyTorch ops is too expensive.

### Experiment 14: Zero Dummy Slot Before Dense Reconstruction

Hypothesis:

- In CP sharded-KV mode, non-owner tokens map to dummy slot `0`.
- The generic KV store path can overwrite slot `0`, so the dense
  reconstruction path previously had to gather K/V from `safe_slots` and then
  multiply the whole dense tensor by `local_valid`.
- If we explicitly zero `key_cache[0]` and `value_cache[0]` immediately before
  dense reconstruction, invalid/non-owner reads from slot `0` are already zero.
  This lets the path skip the large BF16 mask-multiply kernels.

Code change:

- For sharded-KV AttnCP dense reconstruction, zero dummy slot `0` before reading
  `key_cache[safe_slots]` / `value_cache[safe_slots]`.
- Skip the large `local_k *= local_mask` and `local_v *= local_mask` operations.
- Apply the same rule to both full dense decode and SWA decode-window dense
  reconstruction.
- This only mutates dummy slot `0`, which is not allocated for real KV and is
  already filtered out by allocator/free paths.

Precision artifacts:

- Controlled graph probe:
  `/tmp/welmv4_attncp_precision/zero_dummy_graph_20260623_233435`
- Full precision regression:
  `/tmp/welmv4_attncp_precision/20260623_233926`
- Full precision regression after promoting to default:
  `/tmp/welmv4_attncp_precision/20260623_234705`

Precision result:

- Controlled graph: PASS
  - `max_logprob_diff=0.0`
  - `mean_logprob_diff=0.0`
  - `issue_count=0`
- MMLU/C-Eval regression: PASS
  - `samples=100/100`
  - `token_mismatches=0`
  - `max_logprob_diff=0.00e+00`
  - `mean_logprob_diff=0.00e+00`
- Default no-env regression after promotion: PASS
  - `samples=100/100`
  - `token_mismatches=0`
  - `max_logprob_diff=0.00e+00`
  - `mean_logprob_diff=0.00e+00`

Performance artifact:

- `/tmp/welmv4_attncp_perf/zero_dummy_20260623_233642`

Benchmark scenario:

- CP2 sharded-KV only
- `input_len=8192`
- `output_len=512`
- `concurrency=4`
- `num_prompts=16`

Result:

| Config | Completed | Output throughput | Mean TTFT | Mean ITL | Peak GPU memory |
|---|---:|---:|---:|---:|---:|
| TP4+CP2 metadata guard | 16 | 167.23 tok/s | 1997.60 ms | 20.10 ms | 63499 MiB |
| TP4+CP2 zero dummy slot | 16 | 185.71 tok/s | 1984.34 ms | 17.73 ms | 63429 MiB |

Conclusion:

- This is a correctness-preserving optimization for the current dense
  reconstruction path.
- It improves the measured CP2 ITL by about `11.8%` versus the metadata-guard
  baseline in the 8k/512/c4 scenario.
- The remaining dominant cost is still NCCL all-reduce of dense K/V and
  gather/scatter around dense reconstruction, but the large BF16 mask multiply
  is no longer on the hot path.
- After validation, this optimization was promoted from the env-gated experiment
  to the default sharded-KV AttnCP dense path.

### Experiment 15: Current Default TP4 vs TP4+CP2 Fair Benchmark

Purpose:

- Re-run the same benchmark case on the current default code path after the
  zero-dummy-slot optimization was promoted, without adding new experiments.
- Keep this as the current performance baseline before deciding whether more
  optimization work is justified.

Artifact:

- `/tmp/welmv4_attncp_perf/20260623_235254`

Benchmark scenario:

- Branch: `perf/welm-v4-optimization`
- GPUs: `4,5,6,7`
- Model:
  `/home/fhkong/models/80a3_v4d5_256k_merge_thinking_kimi_k25_0502_20260503_032335/epoch_003_step_0002610`
- `input_len=8192`
- `output_len=512`
- `concurrency=4`
- `num_prompts=16`
- TP4 command uses normal TP.
- TP4+CP2 command adds:
  `--attn-cp-size 2 --attn-cp-mode sharded-kv --attn-cp-kv-chunk-size 1024 --attn-cp-decode-cuda-graph-max-seq-len 8704`

Result:

| Config | Completed | Output throughput | Mean TTFT | Mean ITL | Mean E2E latency | Peak GPU memory |
|---|---:|---:|---:|---:|---:|---:|
| TP4 | 16 | 335.20 tok/s | 1635.61 ms | 8.77 ms | 6105.58 ms | 81665 MiB |
| TP4+CP2 sharded-KV | 16 | 185.40 tok/s | 2000.93 ms | 17.73 ms | 11043.28 ms | 63429 MiB |

Delta:

- TP4+CP2 peak GPU memory is lower by `18236 MiB` (`22.3%`).
- TP4+CP2 mean TTFT is higher by `365.32 ms` (`22.3%`).
- TP4+CP2 mean ITL is higher by `8.97 ms` (`102.3%`).
- TP4+CP2 output throughput is lower by `44.7%`.

Conclusion:

- Current AttnCP sharded-KV is correctness-validated and does reduce peak GPU
  memory substantially in this case.
- Performance is still not acceptable for long-output decode: TTFT is about
  `22%` slower and ITL is about `2.0x` slower than TP4.
- Based on previous profiling, the remaining decode bottleneck is not local CP
  metadata or FlashAttention itself. The hot path is dense K/V reconstruction:
  NCCL all-reduce of reconstructed dense K/V plus gather/scatter work around
  that reconstruction.
- Do not blindly add more Python/Torch-side reshaping experiments. The next
  meaningful optimization should target the reconstruction cost directly, for
  example with a fused reconstruction path or an attention backend that consumes
  CP-sharded/compact K/V without rebuilding full dense K/V every decode step.

### Experiment 16: Int32 Page Table Indexing Probe

Hypothesis:

- `_gather_sharded_kv_dense` and `_gather_sharded_kv_dense_decode_window`
  convert `page_table` from int32 to int64 before indexing KV cache.
- PyTorch CUDA tensor indexing accepts int32 indices, so avoiding this
  conversion might reduce elementwise/memory traffic in dense reconstruction.

Temporary code change:

- Keep `page_table` as int32 in the two AttnCP dense reconstruction helpers.
- Use int32 `logical_pos` for the length-validity mask.

Precision artifact:

- `/tmp/welmv4_attncp_precision/20260624_000042`

Precision result:

- Controlled compare: PASS
  - `max_logprob_diff=0.0`
  - `mean_logprob_diff=0.0`
  - `issue_count=0`
- MMLU/C-Eval regression: PASS
  - `samples=100/100`
  - `token_mismatches=0`
  - `max_logprob_diff=0.00e+00`
  - `mean_logprob_diff=0.00e+00`

Performance artifact:

- `/tmp/welmv4_attncp_perf/20260624_000532`

Benchmark scenario:

- `input_len=8192`
- `output_len=512`
- `concurrency=4`
- `num_prompts=16`

Result:

| Config | Completed | Output throughput | Mean TTFT | Mean ITL | Peak GPU memory |
|---|---:|---:|---:|---:|---:|
| TP4 | 16 | 335.69 tok/s | 1625.48 ms | 8.77 ms | 81665 MiB |
| TP4+CP2 int32 page table probe | 16 | 185.33 tok/s | 1965.62 ms | 17.81 ms | 63427 MiB |
| TP4+CP2 previous default | 16 | 185.40 tok/s | 2000.93 ms | 17.73 ms | 63429 MiB |

Conclusion:

- The change is precision-safe, but it does not produce a measurable decode
  performance win in the target long-output benchmark.
- Mean ITL is slightly worse than the previous default run (`17.81 ms` vs
  `17.73 ms`), and the small TTFT improvement is not enough evidence to keep a
  code change on the hot path.
- The temporary code change was reverted. This reinforces that the remaining
  gap is not likely to be solved by small dtype/metadata cleanup; the next
  optimization should reduce dense K/V reconstruction communication or replace
  the reconstruction path.

### Experiment 17: Current Post-Zero-Dummy Decode Profile

Purpose:

- Refresh the decode profile after zero dummy slot was promoted to default.
- The older profile in Experiment 12 was taken before the mask-multiply removal,
  so its elementwise attribution was stale.

Artifact:

- Successful profile:
  `/tmp/welmv4_attncp_profile/cp2_zero_dummy_20260624_001516`
- A previous attempt at
  `/tmp/welmv4_attncp_profile/cp2_zero_dummy_20260624_001131` failed because
  the benchmark client did not inherit `NO_PROXY/no_proxy` and did not reach the
  localhost server. No performance conclusion was taken from that failed run.

Profile scenario:

- TP4+CP2 sharded-KV only
- `input_len=8192`
- `output_len=64`
- `concurrency=4`
- `num_prompts=8`
- Torch profiler: decode stage, 10 steps

Serving metric from the profile run:

| Config | Output throughput | Mean TTFT | Mean ITL | P95 ITL |
|---|---:|---:|---:|---:|
| TP4+CP2 current default | 18.13 tok/s | 11750.75 ms | 38.20 ms | 17.74 ms |

The mean ITL is inflated by profiler outliers; P95 ITL matches the normal
8k/512 benchmark and is the useful sanity check here.

GPU-kernel totals from DECODE traces, averaged per rank over the profiled
window:

| Category | TP4 old profile | CP2 post-guard old | CP2 current zero-dummy |
|---|---:|---:|---:|
| Total GPU kernels | 170.39 ms | 192.44 ms | 169.20 ms |
| NCCL all-reduce | 0.14 ms | 48.91 ms | 48.48 ms |
| gather/index | 0.15 ms | 19.14 ms | 19.03 ms |
| scatter | included in other | 3.65 ms | 3.63 ms |
| elementwise/fill | 1.58 ms | 31.80 ms | 11.01 ms |
| FlashAttention | 18.93 ms | 19.17 ms | 19.25 ms |

Top current CP2 GPU kernels:

| Kernel group | Avg time per rank | Count per rank |
|---|---:|---:|
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 48.34 ms | 480 |
| KV cache gather kernel | 12.78 ms | 960 |
| SWA scatter-gather kernel | 3.63 ms | 460 |
| BF16 fill kernels | 3.88 ms | 1420 |
| FlashAttention main/combine kernels | 19.25 ms | 480 total approx |

Additional observation:

- The broader CPU/runtime trace still shows CP2-specific
  `aten::item` / `aten::_local_scalar_dense` / `cudaStreamSynchronize`
  totaling about `130 ms` per rank in the profiled window.
- This is not yet safely attributable to one source line. `seq_lens_cpu` in
  CUDA graph buffers is a true CPU tensor, so the obvious
  `seq_lens_cpu.max().item()` path should not itself force a GPU sync.
- Do not optimize this blindly. A follow-up needs stack-enabled profiling or
  targeted NVTX/instrumentation around CUDA graph replay metadata, scheduler
  metadata, and WeLM OE decode hash paths.

Conclusion:

- Zero dummy slot did exactly what it was supposed to do: the elementwise/fill
  part dropped from about `31.8 ms` to `11.0 ms` per rank in this profile.
- The remaining GPU-side AttnCP-specific reconstruction cost is dominated by:
  1. dense K/V NCCL all-reduce, about `48.5 ms` per rank per 10 profiled steps;
  2. dense KV gather/index, about `19.0 ms`;
  3. SWA scatter, about `3.6 ms`.
- The next implementation-level optimization should either reduce the dense
  K/V all-reduce volume or avoid full dense reconstruction. Small metadata or
  dtype cleanups are unlikely to close the gap.
- Separately, investigate the CP2-only sync path before assuming all remaining
  ITL gap is NCCL. That investigation should be attribution-only first.

### Experiment 18: Temporary Replay Instrumentation For CP2 Sync Attribution

Purpose:

- Determine whether the CP2-only `aten::item` / `cudaStreamSynchronize`
  attribution from Experiment 17 happens in replay metadata preparation or
  inside the captured CUDA graph replay.
- This was attribution-only. The temporary `torch.profiler.record_function`
  markers were removed after the profile.

Temporary markers:

- `attncp.replay_prepare.populate`
- `attncp.replay_prepare.oe_hash`
- `attncp.replay_prepare.metadata`
- `attncp.graph_replay`

Artifact:

- `/tmp/welmv4_attncp_profile/cp2_instrumented_20260624_002242`

Profile scenario:

- TP4+CP2 sharded-KV only
- `input_len=8192`
- `output_len=64`
- `concurrency=4`
- `num_prompts=8`
- Torch profiler: decode stage, 10 steps

Serving sanity:

| Config | Output throughput | Mean TTFT | Mean ITL | P95 ITL |
|---|---:|---:|---:|---:|
| TP4+CP2 instrumented | 18.35 tok/s | 11556.30 ms | 38.61 ms | 18.10 ms |

Attribution result from TP0 DECODE trace:

| Marker | Count | Total CPU-scope time |
|---|---:|---:|
| `attncp.replay_prepare.populate` | 20 | 1.49 ms |
| `attncp.replay_prepare.oe_hash` | 20 | 1.03 ms |
| `attncp.replay_prepare.metadata` | 20 | 2.84 ms |
| `attncp.graph_replay` | 490 | 195.95 ms |

`aten::item` attribution:

| Enclosing marker | Count | Total `aten::item` time | Max single item |
|---|---:|---:|---:|
| `attncp.graph_replay` | 36 | 123.69 ms | 13.15 ms |
| `attncp.replay_prepare.metadata` | 3 | 0.004 ms | 0.002 ms |
| none | 1 | 0.005 ms | 0.005 ms |

Conclusion:

- The expensive sync is not in `replay_prepare` metadata or WeLM OE hash buffer
  update. It is inside the captured model graph replay.
- Therefore, removing a Python-side metadata `.item()` in replay prepare would
  not address this profile signal.
- The next attribution step, if needed, should add temporary markers inside the
  captured forward body, especially around:
  1. AttnCP dense reconstruction / FA call;
  2. WeLM OE / routed-experts capture path;
  3. sampler/logits post-processing inside the captured graph.
- No code from this experiment was kept.

### Experiment 19: Temporary Captured-Forward Marker Attempt

Purpose:

- Try to split the expensive CP2 `aten::item` / sync attribution inside
  captured CUDA graph replay into finer candidate regions:
  1. AttnCP dense reconstruction;
  2. AttnCP dense K/V all-reduce;
  3. AttnCP dense FA call;
  4. routed-experts row selection / device capture.

Temporary markers:

- `attncp.decode.prepare_dense_kv`
- `attncp.decode.reconstruct_full`
- `attncp.decode.reconstruct_window`
- `attncp.decode.all_reduce_full`
- `attncp.decode.all_reduce_window`
- `attncp.decode.flash_attn_dense`
- `attncp.routed_experts.select_rows`
- `attncp.routed_experts.device_capture`

Artifact:

- `/tmp/welmv4_attncp_profile/cp2_forward_markers_20260624_003002`

Profile scenario:

- TP4+CP2 sharded-KV only
- `input_len=8192`
- `output_len=64`
- `concurrency=4`
- `num_prompts=8`
- Torch profiler: decode stage, 10 steps

Serving sanity:

| Config | Output throughput | Mean TTFT | Mean ITL | P95 ITL |
|---|---:|---:|---:|---:|
| TP4+CP2 forward markers | 17.04 tok/s | 12158.96 ms | 46.13 ms | 20.57 ms |

Result:

- The temporary markers did not appear in the DECODE replay traces.
- `aten::item` was still reported under no enclosing marker:
  - about `47.5` item events per rank;
  - about `128.9 ms` total item time per rank in the profiled window;
  - largest item events around `13 ms`.
- NCCL kernel time was also outside those markers in the trace, even though the
  source markers wrapped the all-reduce calls.

Interpretation:

- Python-side `torch.profiler.record_function` markers inside code that is
  captured into a CUDA graph are not preserved in a useful way for replay-stage
  attribution here.
- This experiment cannot identify the source line for the expensive item/sync
  events, and it should not drive a code change.
- The trace has `with_stack=1`, but the exported Chrome trace still does not
  expose a usable Python call stack for these `aten::item` events.

Conclusion:

- No optimization was kept.
- Do not repeat this marker approach for captured graph replay attribution.
- The actionable performance evidence remains Experiment 17: dense K/V
  reconstruction has large, directly measured GPU-side costs from NCCL
  all-reduce plus gather/scatter. Further performance work should target that
  path directly or use lower-level profiling that can attribute CUDA graph
  replay internals.

### Experiment 20: Compact SWA Decode KV Without Changing Logical Positions

Purpose:

- Reduce the SWA decode reconstruction overhead identified in Experiment 17.
- The old SWA decode window path already communicated only the active window
  tokens, but then scattered the result into a full `[batch, max_seq_len]`
  dense K/V buffer before calling FA.

Attempt 1:

- Returned compact `[batch, window_tokens]` K/V and also clamped
  `cache_seqlens` to the compact window length.
- Precision failed, so this variant was not kept.

Failed precision artifact:

- `/tmp/welmv4_attncp_precision/20260624_003927`

Failure:

| Check | Result |
|---|---:|
| Controlled compare | FAIL |
| Max logprob diff | `1.293e+00` |
| Mean logprob diff | `9.906e-04` |
| Issues | `570` |
| First mismatch | `req[0].text mismatch 'mti0.01s.' vs 'mti0.01a0'` |

Interpretation:

- FA/SWA decode still depends on the original logical positions and full
  `cache_seqlens` when applying causal/window semantics.
- Compacting both physical K/V and logical lengths is not equivalent.

Kept change:

- Keep physical K/V compact at `[batch, window_tokens, heads, dim]`.
- Preserve full logical positions by building a full logical `dense_page_table`
  whose active window columns point into the compact K/V buffer.
- Preserve the original `cache_seqlens`.
- This removes the large full-K/V zero-fill and BF16 scatter while leaving the
  FA logical mask semantics unchanged.

Precision artifact:

- `/tmp/welmv4_attncp_precision/20260624_004358`

Precision result:

| Check | Result |
|---|---:|
| Controlled compare | PASS |
| Controlled max diff | `0.000e+00` |
| Controlled mean diff | `0.000e+00` |
| MMLU/C-Eval samples | `100 / 100` |
| Token mismatches | `0 / 100` |
| Max logprob diff | `0.00e+00` |
| Mean logprob diff | `0.00e+00` |

Benchmark artifact:

- `/tmp/welmv4_attncp_perf/20260624_004835`

Benchmark scenario:

- Same branch and dirty worktree for both configs
- `input_len=8192`
- `output_len=512`
- `concurrency=4`
- `num_prompts=16`
- FA3 prefill/decode, CUDA graph enabled, piecewise CUDA graph disabled
- WeLM KV mirror enabled

Benchmark result:

| Config | Output throughput | Mean TTFT | Mean ITL | P95 ITL | Peak memory |
|---|---:|---:|---:|---:|---:|
| TP4 | `336.42 tok/s` | `1618.95 ms` | `8.76 ms` | `8.67 ms` | `81665 MiB` |
| TP4+CP2 | `192.11 tok/s` | `1975.17 ms` | `17.03 ms` | `16.88 ms` | `63425 MiB` |

Delta in the current benchmark:

| Metric | TP4+CP2 vs TP4 |
|---|---:|
| Output throughput | `-42.9%` |
| Mean TTFT | `+22.0%` |
| Mean ITL | `+94.5%` |
| Peak memory | `-18240 MiB` / `-22.3%` |

Compared with the previous default benchmark
`/tmp/welmv4_attncp_perf/20260623_235254`, CP2 improved modestly:

| Metric | Previous CP2 | Current CP2 |
|---|---:|---:|
| Output throughput | `185.40 tok/s` | `192.11 tok/s` |
| Mean TTFT | `2000.93 ms` | `1975.17 ms` |
| Mean ITL | `17.73 ms` | `17.03 ms` |
| Peak memory | `63429 MiB` | `63425 MiB` |

Conclusion:

- This is a valid but small optimization. It removes unnecessary full-K/V
  scatter/fill in SWA decode while preserving precision.
- It does not solve the main performance gap. CP2 is still about `2x` slower
  than TP4 in ITL for this long-output scenario.
- The remaining large bottleneck is still the dense reconstruction design:
  per-step K/V gather plus NCCL all-reduce before FA.

### Experiment 21: Decode CP KV Owner-Mask Allocation Sync

Purpose:

- Re-profile the current compact-SWA version and verify what remains before
  attempting a larger dense-KV reconstruction rewrite.

Profile artifact before this experiment's code change:

- `/tmp/welmv4_attncp_profile/cp2_compact_swa_20260624_005655`

Profile scenario:

- TP4+CP2 sharded-KV only
- `input_len=8192`
- `output_len=64`
- `concurrency=4`
- `num_prompts=8`
- Torch profiler: decode stage, 10 steps

Finding:

- The trace attributed about `120 ms` per 10 profiled decode steps to
  `CPShardedKVPoolAllocator.alloc_for_positions` on every rank.
- The expensive child was `aten::item` / Tensor `.item()`, caused by computing
  `owner_mask.sum().item()` on a GPU owner mask during CP sharded decode KV
  allocation.
- This is a real synchronization hazard, but it is scheduler/allocation side,
  not attention math.

Code change:

- `CPShardedKVPoolAllocator.alloc_for_positions` now accepts optional
  `positions_cpu`.
- CP sharded prefill/decode allocation passes CPU positions for owner-count
  calculation.
- The returned `out_cache_loc` remains a device tensor; sharded-KV semantics are
  unchanged.

Precision artifact:

- `/tmp/welmv4_attncp_precision/20260624_010101`

Precision result:

| Check | Result |
|---|---:|
| Controlled compare | PASS |
| Controlled max diff | `0.000e+00` |
| Controlled mean diff | `0.000e+00` |
| MMLU/C-Eval samples | `100 / 100` |
| Token mismatches | `0 / 100` |
| Max logprob diff | `0.00e+00` |
| Mean logprob diff | `0.00e+00` |

Benchmark artifact:

- `/tmp/welmv4_attncp_perf/20260624_010538`

Benchmark result:

| Config | Output throughput | Mean TTFT | Mean ITL | P95 ITL | Peak memory |
|---|---:|---:|---:|---:|---:|
| TP4 | `335.02 tok/s` | `1633.94 ms` | `8.78 ms` | `8.68 ms` | `81665 MiB` |
| TP4+CP2 | `189.33 tok/s` | `2099.17 ms` | `17.09 ms` | `16.94 ms` | `63425 MiB` |

Conclusion:

- The CPU owner-count change preserves correctness, but the benchmark does not
  show a stable ITL win.
- The original large `.item()` attribution was partly profiler-visible
  synchronization, not the main steady-state throughput bottleneck.

### Experiment 22: CP KV Owner All-Local Fast Path

Purpose:

- Explain why Experiment 21 still showed `alloc_for_positions` taking about
  `129 ms` on CP rank 0 after removing GPU `.item()`.

Finding:

- In the profiled decode window, all new positions are in chunk `8192..`,
  owned by CP rank 0 for `cp_kv_chunk_size=1024` and `cp_size=2`.
- Rank 0 had `local_count == positions.numel()` but still constructed a dummy
  output tensor and used GPU boolean indexing to scatter the allocated slots.
- Rank 1 had `local_count == 0` and returned immediately, which is why rank 1
  was already cheap.

Code change:

- If all positions are owned by the current CP rank, return the allocated slots
  directly, reshaped to `positions`.
- If only some positions are local, use CPU-computed owner indices rather than a
  GPU boolean mask for the final scatter.

Precision artifact:

- `/tmp/welmv4_attncp_precision/20260624_011347`

Precision result:

| Check | Result |
|---|---:|
| Controlled compare | PASS |
| Controlled max diff | `0.000e+00` |
| Controlled mean diff | `0.000e+00` |
| MMLU/C-Eval samples | `100 / 100` |
| Token mismatches | `0 / 100` |
| Max logprob diff | `0.00e+00` |
| Mean logprob diff | `0.00e+00` |

Benchmark artifact:

- `/tmp/welmv4_attncp_perf/20260624_011827`

Benchmark result:

| Config | Output throughput | Mean TTFT | Mean ITL | P95 ITL | Peak memory |
|---|---:|---:|---:|---:|---:|
| TP4 | `334.61 tok/s` | `1642.89 ms` | `8.77 ms` | `8.67 ms` | `81665 MiB` |
| TP4+CP2 | `191.44 tok/s` | `1973.97 ms` | `17.10 ms` | `16.96 ms` | `63425 MiB` |

Post-change profile artifact:

- `/tmp/welmv4_attncp_profile/cp2_owner_fastpath_20260624_012314`

Profile result:

| Trace item | Before fast path | After fast path |
|---|---:|---:|
| `alloc_for_positions`, CP rank 0 | `~128.8 ms / 10 steps` | `~1.0 ms / 10 steps` |
| `alloc_for_positions`, CP rank 1 | `~0.85 ms / 10 steps` | `~0.85 ms / 10 steps` |
| `aten::item` total | `~0.1 ms` after Experiment 21 | `~0.1 ms` |
| NCCL dense K/V all-reduce | `~48.4 ms / rank` | `~48.4 ms / rank` |
| KV gather kernel | `~12.8 ms / rank` | `~12.8 ms / rank` |

Conclusion:

- This is worth keeping as cleanup: it removes an avoidable CPU/GPU allocation
  path cost and makes profiler attribution cleaner.
- It still does not reduce the user-visible long-output ITL, because the
  remaining direct GPU-side bottleneck is unchanged:
  dense K/V all-reduce plus dense K/V gather before FA.
- The next meaningful optimization would need to change the dense KV
  reconstruction algorithm, e.g. compact cross-CP K/V all-gather with logical
  page-table reconstruction, or a true sharded-KV attention path. That is a
  larger design change and should not be attempted as a blind patch.

### Experiment 23: Env-Gated Compact Cross-CP KV All-Gather Prototype

Purpose:

- Test whether replacing full dense K/V all-reduce with compact owner-KV
  all-gather can reduce the remaining decode bottleneck.
- Keep the prototype behind
  `SGLANG_ATTNCP_EXPERIMENTAL_DENSE_DECODE_COMPACT_AG=1` during testing.

Prototype shape:

- Only touched decode correctness path.
- Did not change prefill.
- Did not change true 512-token SWA window layers.
- Built compact local owner-KV buffers, all-gathered them across the CP group,
  then reconstructed a full logical page table for FA.

Attempt 1:

- Only treated `window_left < 0` as full attention.
- Precision passed, but benchmark did not improve.

Artifacts:

- Precision: `/tmp/welmv4_attncp_precision/20260624_012932`
- Benchmark: `/tmp/welmv4_attncp_perf/20260624_013410`
- Profile: `/tmp/welmv4_attncp_profile/cp2_compact_ag_20260624_013901`

Result:

| Config | Output throughput | Mean TTFT | Mean ITL | P95 ITL | Peak memory |
|---|---:|---:|---:|---:|---:|
| TP4 | `336.49 tok/s` | `1613.25 ms` | `8.76 ms` | `8.67 ms` | `81665 MiB` |
| TP4+CP2 compact-AG attempt 1 | `191.61 tok/s` | `1967.67 ms` | `17.10 ms` | `16.94 ms` | `63425 MiB` |

Profile finding:

- NCCL all-gather appeared only as `~0.14 ms / rank / 10 steps`.
- NCCL all-reduce stayed at `~48.4 ms / rank / 10 steps`.
- Root cause: many WeLM "full" layers are encoded as
  `sliding_window_size_layerwise=262144`, not `-1`, so they did not hit the
  first full-attention condition.

Attempt 2:

- Also treated `window_left >= max_context_len` as full attention.
- Precision still passed, but performance became much worse.

Artifacts:

- Precision: `/tmp/welmv4_attncp_precision/20260624_014233`
- Benchmark: `/tmp/welmv4_attncp_perf/20260624_014750`

Precision result:

| Check | Result |
|---|---:|
| Controlled compare | PASS |
| Controlled max diff | `0.000e+00` |
| Controlled mean diff | `0.000e+00` |
| MMLU/C-Eval samples | `100 / 100` |
| Token mismatches | `0 / 100` |
| Max logprob diff | `0.00e+00` |
| Mean logprob diff | `0.00e+00` |

Benchmark result:

| Config | Output throughput | Mean TTFT | Mean ITL | P95 ITL | Peak memory |
|---|---:|---:|---:|---:|---:|
| TP4 | `333.70 tok/s` | `1657.84 ms` | `8.78 ms` | `8.68 ms` | `81665 MiB` |
| TP4+CP2 compact-AG attempt 2 | `147.61 tok/s` | `1973.54 ms` | `23.34 ms` | `23.18 ms` | `63697 MiB` |

Conclusion:

- The prototype is correctness-safe for the tested cases but not performance
  useful.
- The extra compact-buffer scatter, page-table reconstruction, and larger
  temporary compact workspace more than offset any communication reduction.
- The env-gated prototype code was removed after the experiment.
- Do not pursue Python-level compact all-gather reconstruction further. A
  useful version would need a fused kernel or a true sharded-KV attention
  backend that consumes owner-sharded KV directly instead of reconstructing a
  FA-compatible page table every layer.

### Experiment 24: Training Overlap Code Review

Purpose:

- Review the training AttnCP/sequence-parallel code referenced by `task.md`
  before making more inference-side changes.
- Decide whether the overlap logic can directly reduce the current decode ITL
  gap.

Source reviewed:

- `/home/fhkong/wxwork/mimikyu/mmq/mmq/modules/block_v2/memory_optimizer/qkv_proj_and_post_processing/overlap.py`
- `/home/fhkong/wxwork/mimikyu/mmq/mmq/modules/block_v2/submodules/dist_attn.py`
- `/home/fhkong/wxwork/mimikyu/mmq/mmq/modules/block_v2/memory_optimizer/qkv_attn/ring_attn_varlen_attn.py`
- `/home/fhkong/wxwork/mimikyu/mmq/mmq/modules/attention/allgather_ring_flash_attn.py`
- `/home/fhkong/wxwork/mimikyu/mmq/mmq/modules/attention/ring_attn_v3.py`

Finding:

- `OverlapQKVProjAndPostProcessing` overlaps Q all-to-all with K/V/gate
  projection and later waits for Q/K/V communication. This hides QKV projection
  communication in the training Ulysses path.
- `DistributedAttention` is sequence all-to-all based: it scatters heads and
  gathers sequence for dense training tensors. This does not operate on
  SGLang's paged decode KV cache.
- The training ring attention code has the right high-level math pattern:
  compute partial attention, keep `softmax_lse`, and merge states. However the
  available implementations are varlen training kernels over dense contiguous
  Q/K/V tensors, not FA3 paged-kvcache decode kernels.
- The current inference bottleneck is already after QKV projection: every
  decode layer reconstructs dense FA-compatible K/V from owner-sharded paged KV
  before calling `flash_attn_with_kvcache`. Profile evidence after Experiment
  22 shows about `48.4 ms / rank / 10 steps` in dense K/V NCCL all-reduce plus
  about `12.8 ms / rank / 10 steps` in K/V gather kernels.

Conclusion:

- The training overlap code is useful as a design reference but is not a
  drop-in optimization for the current inference decode path.
- Simple stream overlap around the current dense reconstruction is unlikely to
  close the ITL gap, because `flash_attn_with_kvcache` cannot start until the
  reconstructed dense K/V and page table are ready.
- The remaining meaningful optimization is algorithmic: implement a true
  sharded-KV attention backend for paged decode/prefill, or a fused segment/LSE
  path that consumes owner-sharded KV directly. That path must preserve:
  cuda graph, batched decode, chunked prefill, attention sink, KV mirror, SWA,
  and the current sharded KV residency semantics.
- Do not blindly port the training overlap code into inference. It would add
  complexity without removing the measured dense K/V reconstruction bottleneck.

### Source Finding: CUDA Graph Shape Fixes Dense Decode Work

Purpose:

- Check whether the dense decode path is doing extra work because
  `metadata.page_table` is wider than the current replay step's true
  `metadata.max_seq_len_k`.
- Identify whether a simple page-table slice can reduce the measured 8k/512
  ITL gap without changing attention math.

Relevant source:

- `flashattention_backend.py`
  - `_gather_sharded_kv_dense` uses `page_table.shape[1]` as `max_seq_len`.
  - `get_cuda_graph_seq_len_fill_value()` returns
    `self.cuda_graph_max_seq_len` for AttnCP sharded-KV.
- `cuda_graph_runner.py`
  - capture uses one `seq_len_fill_value` for the graph.
  - replay only checks `seq_lens_cpu.max() <= cuda_graph_max_seq_len`; it does
    not select a smaller sequence-length graph bucket.
- `attncp_performance_benchmark/run_attncp_perf_benchmark.sh`
  - current 8k/512 scenario sets
    `--attn-cp-decode-cuda-graph-max-seq-len 8704`.

Finding:

- In non-CUDA-graph decode, `metadata.page_table` is already sliced to the
  current `metadata.max_seq_len_k`.
- In CUDA graph decode, the captured graph shape is fixed by
  `attn_cp_decode_cuda_graph_max_seq_len`. During replay the metadata tensors
  are updated with current lengths, but the captured dense K/V gather and NCCL
  all-reduce shapes remain the capture shapes.
- Therefore a Python-side slice such as
  `metadata.page_table[:, :metadata.max_seq_len_k]` is not expected to reduce
  the already-captured dense K/V communication shape in the current graph path.

Conclusion:

- The 8k/512 CP2 dense decode path is effectively paying dense reconstruction
  over the configured graph cap (`8704`) for every decode step. This is only
  about 6% larger than the longest sequence in that benchmark, so it cannot
  explain the roughly 2x ITL gap by itself.
- More sequence-length graph buckets could help mixed/short decode workloads,
  but would add capture memory/startup cost and still would not remove the
  full-KV reconstruction cost at long lengths.
- For the current target gap, the useful directions remain:
  1. a true paged sharded-KV attention path with Q all-gather and LSE merge; or
  2. a fused segment/LSE backend that avoids materializing dense full K/V.
- Do not implement a page-table slicing patch as a performance fix for the
  current benchmark; it would mostly be a no-op under CUDA graph replay.

### Source Finding: Minimal Decode Prototype Boundary

Purpose:

- Decide whether there is a small runtime patch between the current dense
  correctness path and a full custom sharded-KV attention backend.
- Re-check the existing local-merge implementation, prior q-granular attempt,
  and reusable in-repo merge primitives before writing more code.

Source reviewed:

- `flashattention_backend.py`
  - `_set_sharded_kv_decode_metadata`
  - `_flash_attn_sharded_kv_local_merge`
  - `_forward_decode_sharded_kv`
- `layers/attention/merge_state.py`
- `layers/attention/triton_ops/merge_state.py`
- `models/deepseek_common/attention_forward_methods/forward_mha.py`
- Prior q-granular graph artifact:
  `/tmp/welmv4_attncp_precision/graph_local_merge_qshard_concat_20260623_210712/tp4_cp2_server.log`

Finding:

- For single-token decode on full-attention layers, the current experimental
  local-merge path is already the smallest "true sharded-KV" shape available
  without a new kernel:
  Q head all-gather -> local paged-KV FA -> LSE merge across CP ranks -> return
  local Q-head slice.
- This path avoids dense K/V reconstruction and is graph-friendly, but it uses
  one full-Q FA call with `H/attn_tp` Q heads. That FA call shape is not
  numerically equivalent to the TP4 baseline for WeLM and causes token drift.
- The q-granular variant restores the original TP-local Q-head FA shape and
  reduced/removed controlled token drift, but it requires multiple FA calls per
  layer per rank. In eager mode it was slower than TP4, and the graph-capture
  artifact stops after capturing `bs=16` and entering `bs=12` with no Python
  exception. This points to a CUDA graph capture/collective/kernel interaction,
  not an ordinary fixable Python error.
- Existing in-repo segmented attention examples (`forward_mha.py`) use the same
  `merge_state_v2` math, but they operate on local dense tensors and do not
  provide a paged sharded-KV decode primitive or a graph-safe CP
  allgather/reduce-scatter output primitive.

Conclusion:

- There is no evidence-backed low-risk Python patch that both preserves WeLM
  output tokens and removes the dense reconstruction bottleneck.
- Re-adding q-granular local-merge as another env-gated experiment would repeat
  a known slower/stalled path unless the implementation is moved into a fused
  graph-safe kernel/primitive.
- The next runtime implementation should start below the current Python
  attention path:
  1. define a fixed-shape graph-safe workspace for full-attention decode;
  2. fuse local paged-KV partial attention and LSE state production at the
     original TP Q-head granularity; and
  3. merge CP partial states with a graph-safe output primitive that returns
     only the local TP Q-head slice.
- Until that primitive exists, keep the dense correctness path as the default
  and avoid further dense-path micro-optimizations.

### Source Finding: sgl-kernel Primitive Landing Point

Purpose:

- Identify where a real full-attention decode sharded-KV primitive should live.
- Check whether existing `sgl-kernel` APIs already provide enough building
  blocks.

Source reviewed:

- `sgl-kernel/python/sgl_kernel/flash_attn.py`
- `sgl-kernel/csrc/flash_extension.cc`
- `sgl-kernel/csrc/attention/merge_attn_states.cu`
- `sgl-kernel/csrc/common_extension.cc`
- `python/sglang/srt/distributed/parallel_state.py`
- `python/sglang/srt/distributed/device_communicators/pynccl.py`

Finding:

- `flash_attn_with_kvcache` already supports paged KV, preallocated `out`,
  scheduler metadata, attention sink, and `return_softmax_lse=True`. It is a
  usable single-rank partial-attention primitive.
- `merge_state_v2` is implemented in
  `sgl-kernel/csrc/attention/merge_attn_states.cu` and expects contiguous
  `[num_tokens, num_heads, head_dim]` outputs and `[num_tokens, num_heads]`
  LSE tensors. It is useful for local or post-communication state merge.
- The FA3 binding is registered as `torch.ops.sgl_kernel.fwd` in
  `sgl-kernel/csrc/flash_extension.cc`; extending the FA call shape itself
  means modifying the FA extension path, not only SGLang Python code.
- `GroupCoordinator` / `PyNcclCommunicator` already have graph-aware
  all-gather / reduce-scatter building blocks, but they are bare collectives.
  They do not solve the fixed-workspace CP partial merge/output-slice problem.

Conclusion:

- The practical landing point is a new graph-safe primitive or wrapper around
  the FA3 partial attention path plus `merge_state_v2`-style state merge, with
  static workspaces owned by the attention backend.
- The primitive should not be a Python loop over Q-head shards plus multiple
  collectives. That shape already failed the performance/graph evidence.
- Minimum interface for the next prototype:
  - input: local TP Q shard, local CP compact page table/cache lengths,
    local KV cache, optional full gathered sinks, CP rank/world metadata;
  - workspace: gathered Q buffer, partial O/LSE buffer per CP rank, merge O/LSE
    scratch, final local-head output;
  - output: `[batch, local_tp_q_heads, head_dim]`, matching current `o_proj`
    contract.
- Documented this landing point in
  `docs/ring-attn/design-sharded-kv-cp.md` under the required change list and
  Phase 2 task list.

### Experiment 25: Decode Component Benchmark Harness

Purpose:

- Avoid blind runtime optimization by separating the decode cost into local FA,
  dense-KV materialization, merge-state, and estimated communication payload.
- Keep the measurement outside the serving path so it cannot change AttnCP
  correctness or sharded-KV residency semantics.

Change:

- Added `benchmark/kernels/attention/bench_attncp_decode_components.py`.
- The script measures:
  - current dense correctness FA after temporary K/V reconstruction;
  - local dense materialization before CP all-reduce;
  - target local full-Q FA with returned LSE over the CP-owned KV shard;
  - `merge_state_v2` chain cost for CP partial states.
- The script also records per-layer/per-decode-step payload estimates. It does
  not time NCCL collectives in the single-process mode.

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmark/kernels/attention/bench_attncp_decode_components.py \
  --kv-lens 1024 --warmup 2 --iters 5 --trials 2 \
  --output /tmp/attncp_decode_components_smoke.json
```

Smoke result:

- Passed FA3 and `merge_state_v2` invocation.
- Output: `/tmp/attncp_decode_components_smoke.json`.

Short component command:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmark/kernels/attention/bench_attncp_decode_components.py \
  --kv-lens 8192,8704 --batch-size 4 --tp-size 4 --cp-size 2 \
  --cp-rank 0 --cp-kv-chunk-size 1024 \
  --warmup 10 --iters 50 --trials 3 \
  --output /tmp/attncp_decode_components_8k.json
```

Result:

| kv_len | local_kv_len | component | median_us |
|---:|---:|---|---:|
| 8192 | 4096 | dense_tp_fa_after_reconstruct | 49.492 |
| 8192 | 4096 | dense_local_materialize_before_allreduce | 49.596 |
| 8192 | 4096 | target_local_fullq_fa_with_lse | 29.810 |
| 8192 | 4096 | target_merge_state_chain | 6.179 |
| 8704 | 4608 | dense_tp_fa_after_reconstruct | 55.144 |
| 8704 | 4608 | dense_local_materialize_before_allreduce | 52.765 |
| 8704 | 4608 | target_local_fullq_fa_with_lse | 35.240 |
| 8704 | 4608 | target_merge_state_chain | 6.105 |

Payload estimate for `batch=4, TP4, CP2, kv_len=8192`:

- Current dense K/V all-reduce tensor: about `32 MiB` per layer per decode
  step.
- Current local dense materialization read: about `32 MiB` per layer per decode
  step.
- Target local CP-owned K/V read: about `16 MiB`.
- Target Q all-gather receive: about `24 KiB`.
- Target O/LSE state all-gather receive: about `24 KiB`.

Conclusion:

- This is consistent with the service-level profile: the expensive part is not
  Q all-gather or LSE merge by itself; it is the current correctness path's
  dense full-KV materialization plus dense K/V all-reduce.
- A useful optimization still needs to remove dense K/V reconstruction in
  decode. More Python-side tweaks around allocator/page-table slicing are
  unlikely to close the ITL gap.

### Experiment 26: Decode Collectives Microbenchmark

Purpose:

- Measure real NCCL latency for the communication shapes implied by the current
  dense correctness path and the target sharded-KV decode path.
- Validate whether Q/O-LSE communication is actually small enough to justify
  focusing on dense K/V reconstruction removal.

Change:

- Added `benchmark/kernels/attention/bench_attncp_decode_collectives.py`.
- Launch mode: `torchrun`; step time is the max rank CUDA-event time for each
  trial.
- Measured components:
  - target Q all-gather;
  - target partial O all-gather;
  - target partial LSE all-gather;
  - target O+LSE all-gather pair;
  - current dense K all-reduce;
  - current dense K+V all-reduce pair;
  - attention sink all-gather.

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone \
  --nproc_per_node=2 benchmark/kernels/attention/bench_attncp_decode_collectives.py \
  --kv-lens 1024 --warmup 2 --iters 5 --trials 2 \
  --output /tmp/attncp_decode_collectives_smoke.json
```

Smoke result:

- Passed.
- Output: `/tmp/attncp_decode_collectives_smoke.json`.

Target command:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone \
  --nproc_per_node=2 benchmark/kernels/attention/bench_attncp_decode_collectives.py \
  --kv-lens 8192,8704 --batch-size 4 --tp-size 4 --cp-size 2 \
  --warmup 20 --iters 100 --trials 5 \
  --output /tmp/attncp_decode_collectives_cp2_8k.json
```

Result:

| kv_len | component | input bytes | median_us |
|---:|---|---:|---:|
| 8192 | target_q_allgather | 12 KiB | 18.203 |
| 8192 | target_o_lse_allgather_pair | 24.2 KiB | 34.394 |
| 8192 | current_dense_k_allreduce | 16 MiB | 75.297 |
| 8192 | current_dense_kv_allreduce_pair | 32 MiB | 152.493 |
| 8192 | target_sinks_allgather | 12 B | 17.163 |
| 8704 | target_q_allgather | 12 KiB | 19.409 |
| 8704 | target_o_lse_allgather_pair | 24.2 KiB | 33.761 |
| 8704 | current_dense_k_allreduce | 17 MiB | 79.932 |
| 8704 | current_dense_kv_allreduce_pair | 34 MiB | 163.107 |
| 8704 | target_sinks_allgather | 12 B | 17.192 |

Interpretation:

- Current dense correctness path pays roughly `150-163 us/layer/decode-step`
  only for the dense K+V all-reduce pair at the 8k/8.5k graph bucket shapes.
- Target sharded decode communication would pay roughly `18-19 us` for Q
  all-gather plus `34 us` for O/LSE all-gather, about `52-53 us/layer` before
  local FA and merge work.
- On a 48-layer WeLM model, removing dense K/V all-reduce can save about
  `4.8-5.3 ms/token` before accounting for reduced materialization and local
  FA changes. This is directionally consistent with the observed CP2 ITL gap.
- Attention sink communication is latency-bound but tiny; it is not the main
  bottleneck.

Conclusion:

- The real communication data supports the same conclusion as Experiment 25:
  the next useful optimization is a true sharded-KV decode path that avoids
  dense K/V all-reduce and dense temporary K/V materialization.
- Do not spend more time on dense-path page-table or allocator micro-tuning
  unless a new profile contradicts this result.

### Experiment 27: End-to-End Decode Path Prototype

Purpose:

- Measure whether the obvious Python-level true-sharded decode path is enough:
  Q all-gather -> local CP-owned KV FA -> O/LSE all-gather -> `merge_state_v2`
  -> local TP-head slice.
- Compare it against the current dense correctness path in the same torchrun
  process, with the same random Q/K/V and attention sink settings.
- Also measure a q-shard loop variant that keeps the original TP Q-head FA
  shape but runs one FA call per CP Q shard.

Change:

- Added `benchmark/kernels/attention/bench_attncp_decode_paths.py`.
- The script runs:
  - `current_dense`: materialize local dense K/V, all-reduce K and V, then FA
    on the local TP Q heads;
  - `target_sharded_fullq`: all-gather Q, run one full-Q FA over local CP KV,
    all-gather O/LSE, merge, then return the local TP head slice;
  - `target_sharded_qloop`: all-gather Q, run per-CP-Q-shard FA calls, gather
    and merge O/LSE, then return the local TP head slice.

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone \
  --nproc_per_node=2 benchmark/kernels/attention/bench_attncp_decode_paths.py \
  --kv-lens 2048 --batch-size 2 --tp-size 4 --cp-size 2 \
  --cp-kv-chunk-size 1024 --warmup 1 --iters 2 --trials 1 \
  --output /tmp/attncp_decode_paths_smoke_qloop.json
```

Smoke result:

- Passed.
- `fullq_diff_max=0.000977`, `qloop_diff_max=0.000977`.
- Output: `/tmp/attncp_decode_paths_smoke_qloop.json`.

Target command:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone \
  --nproc_per_node=2 benchmark/kernels/attention/bench_attncp_decode_paths.py \
  --kv-lens 8192,8704 --batch-size 4 --tp-size 4 --cp-size 2 \
  --cp-kv-chunk-size 1024 --warmup 10 --iters 50 --trials 3 \
  --output /tmp/attncp_decode_paths_cp2_8k_qloop.json
```

Result:

| kv_len | local_kv_len(rank0) | path | median_us/layer | max_abs_diff |
|---:|---:|---|---:|---:|
| 8192 | 4096 | current_dense | 258.935 | 0.000244 |
| 8192 | 4096 | target_sharded_fullq | 244.531 | 0.000244 |
| 8192 | 4096 | target_sharded_qloop | 334.580 | 0.000244 |
| 8704 | 4608 | current_dense | 274.345 | 0.000488 |
| 8704 | 4608 | target_sharded_fullq | 239.581 | 0.000488 |
| 8704 | 4608 | target_sharded_qloop | 350.029 | 0.000488 |

Interpretation:

- The simple full-Q true-sharded Python path is only `14-35 us/layer` faster
  than the dense path in this standalone prototype, roughly `0.7-1.7 ms/token`
  over 48 layers. This is useful but not enough to close the observed service
  ITL gap by itself.
- The q-shard loop keeps the original TP Q-head FA shape but is much slower in
  Python because it launches multiple FA calls and still gathers/merges full
  O/LSE state. This matches the earlier service-level q-granular experiment.
- The prototype's random-tensor max diff is small, but this does not override
  the real WeLM service finding that full-Q local merge can cause long-prompt
  token drift. Model-level controlled precision remains the required gate.

Conclusion:

- Do not enable the existing Python local-merge path as the main optimization.
- A real runtime optimization needs to move below this Python composition:
  reduce full-Q FA overhead, avoid full O/LSE all-gather when only the local
  TP-head slice is needed, and make the CP partial merge graph-safe.

### Experiment 28: Head-Slice O/LSE All-to-All Prototype

Purpose:

- Test whether replacing full O/LSE all-gather with head-slice all-to-all is a
  useful intermediate step.
- Each CP rank computes partial O/LSE for all Q heads over its local KV shard,
  but the final output rank only needs its own TP Q-head slice. In principle,
  all-to-all can send slice `j` to CP rank `j` and avoid communicating full
  heads to every rank.

Change:

- Extended `benchmark/kernels/attention/bench_attncp_decode_paths.py` with
  `target_sharded_slice_a2a`.
- Extended `benchmark/kernels/attention/bench_attncp_decode_collectives.py` with
  `target_o_lse_alltoall_slice_pair`.

Path command:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone \
  --nproc_per_node=2 benchmark/kernels/attention/bench_attncp_decode_paths.py \
  --kv-lens 8192,8704 --batch-size 4 --tp-size 4 --cp-size 2 \
  --cp-kv-chunk-size 1024 --warmup 10 --iters 50 --trials 3 \
  --output /tmp/attncp_decode_paths_cp2_8k_a2a.json
```

Path result:

| kv_len | path | median_us/layer | max_abs_diff |
|---:|---|---:|---:|
| 8192 | current_dense | 258.109 | 0.000244 |
| 8192 | target_sharded_fullq | 244.073 | 0.000244 |
| 8192 | target_sharded_slice_a2a | 342.950 | 0.000244 |
| 8192 | target_sharded_qloop | 341.368 | 0.000244 |
| 8704 | current_dense | 272.915 | 0.000488 |
| 8704 | target_sharded_fullq | 233.914 | 0.000488 |
| 8704 | target_sharded_slice_a2a | 254.868 | 0.000488 |
| 8704 | target_sharded_qloop | 339.328 | 0.000488 |

Collectives command:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone \
  --nproc_per_node=2 benchmark/kernels/attention/bench_attncp_decode_collectives.py \
  --kv-lens 8192,8704 --batch-size 4 --tp-size 4 --cp-size 2 \
  --warmup 20 --iters 100 --trials 5 \
  --output /tmp/attncp_decode_collectives_cp2_8k_a2a.json
```

Collectives result:

| kv_len | component | median_us |
|---:|---|---:|
| 8192 | target_o_lse_allgather_pair | 34.702 |
| 8192 | target_o_lse_alltoall_slice_pair | 36.790 |
| 8704 | target_o_lse_allgather_pair | 36.261 |
| 8704 | target_o_lse_alltoall_slice_pair | 37.338 |

Interpretation:

- The slice all-to-all path is numerically correct in the standalone prototype,
  but it is not faster.
- O/LSE state is only tens of KiB at CP2. The latency of two all-to-all calls
  plus slice packing/copying outweighs the reduced remote payload.
- This optimization may become relevant at larger CP sizes or larger Q-head
  groups, but it is not the right next step for the current TP4/CP2 WeLM case.

Conclusion:

- Keep the simpler full O/LSE all-gather in Python prototypes.
- If optimizing O/LSE exchange later, do it as part of a fused graph-safe
  primitive rather than as a Python all-to-all composition.

### Experiment 29: CUDA Graph Decode Path Prototype

Purpose:

- Re-run the standalone decode path prototype under CUDA graph capture/replay.
- The serving decode path relies on CUDA graph, so eager-only conclusions can
  be misleading when the candidate path has multiple small kernels and
  collectives.

Change:

- Extended `benchmark/kernels/attention/bench_attncp_decode_paths.py` with
  `--cuda-graph`.
- The script now captures each path once with `torch.cuda.CUDAGraph` and times
  graph replay with CUDA events. Capture failures are reported per path.

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone \
  --nproc_per_node=2 benchmark/kernels/attention/bench_attncp_decode_paths.py \
  --cuda-graph --kv-lens 2048 --batch-size 2 --tp-size 4 --cp-size 2 \
  --cp-kv-chunk-size 1024 --warmup 1 --iters 2 --trials 1 \
  --output /tmp/attncp_decode_paths_graph_smoke.json
```

Smoke result:

- All paths captured and replayed.
- Output: `/tmp/attncp_decode_paths_graph_smoke.json`.

Target command:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone \
  --nproc_per_node=2 benchmark/kernels/attention/bench_attncp_decode_paths.py \
  --cuda-graph --kv-lens 8192,8704 --batch-size 4 --tp-size 4 --cp-size 2 \
  --cp-kv-chunk-size 1024 --warmup 5 --iters 50 --trials 3 \
  --output /tmp/attncp_decode_paths_graph_cp2_8k.json
```

Result:

| kv_len | path | graph median_us/layer | max_abs_diff |
|---:|---|---:|---:|
| 8192 | current_dense | 254.303 | 0.000244 |
| 8192 | target_sharded_fullq | 61.443 | 0.000244 |
| 8192 | target_sharded_slice_a2a | 67.411 | 0.000244 |
| 8192 | target_sharded_qloop | 100.040 | 0.000244 |
| 8704 | current_dense | 277.640 | 0.000488 |
| 8704 | target_sharded_fullq | 69.729 | 0.000488 |
| 8704 | target_sharded_slice_a2a | 75.673 | 0.000488 |
| 8704 | target_sharded_qloop | 112.120 | 0.000488 |

Interpretation:

- CUDA graph replay changes the conclusion from the eager-only prototype.
  Multiple small operations in the sharded path become much cheaper when their
  launch overhead is captured.
- `target_sharded_fullq` is about `190-208 us/layer` faster than the dense path,
  which is large enough to matter for 48 layers.
- `target_sharded_qloop` is also much faster than dense under graph replay,
  despite being slower in eager. This is important because q-loop preserves the
  original TP Q-head FA shape better than full-Q.
- The standalone random-tensor diff remains small, but this does **not** prove
  model-level correctness. Prior WeLM service experiments showed full-Q
  local-merge can cause long-prompt token drift, and q-granular graph capture
  previously stalled in the SGLang integration.

Conclusion:

- Do not use eager microbench numbers alone to reject the sharded decode path.
- The next useful runtime experiment is a graph-focused service integration
  behind an explicit flag:
  1. first try a graph-stable q-loop/full-Q local-merge implementation with
     fixed workspaces;
  2. run the full TP4 vs TP4+CP2 precision suite, not just random tensor diff;
  3. if q-loop preserves tokens, benchmark 8k/512/c4 under normal CUDA graph;
  4. only then decide whether a fused `sgl-kernel` primitive is still needed.

### Experiment 30: Service Q-Loop Local-Merge With Fixed Workspaces

Purpose:

- Move the CUDA graph prototype into the SGLang decode path behind an explicit
  experiment flag.
- Check whether q-loop local merge can capture/replay in the real server and
  whether it improves the 8k/512/c4 ITL gap without changing sharded-KV
  residency.

Change:

- Extended `flashattention_backend.py` with an env-gated workspace path:
  - `SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE=1`
  - `SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE_MODE=fullq|qloop`
- The workspace path preallocates graph buffers for:
  Q all-gather, full-Q assembly, q-loop Q/O/LSE shards, full local O/LSE,
  gathered O/LSE, local-head merge scratch, and attention-sink state.
- Default behavior is unchanged. Without the env flag, decode still uses the
  dense correctness path.

Precision command:

```bash
SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE=1 \
SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE_MODE=qloop \
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
bash /home/fhkong/wxwork/attncp_precision_regression/run_full_precision.sh
```

Artifact:

- `/tmp/welmv4_attncp_precision/20260624_024352`

Precision result:

- TP4 baseline completed.
- TP4+CP2 q-loop server started and completed CUDA graph capture for
  `bs=[16,12,8,4,2,1]`; it did **not** stall at `bs=12`.
- Controlled output token/text matched for all 4 prompt lengths
  (`64,512,1024,2048`).
- Strict logprob compare failed:
  - `max_logprob_diff=0.999237060546875`
  - `mean_logprob_diff=0.000500941446027028`
  - `issue_count=283`
- Because the controlled strict compare failed, the script stopped before
  MMLU/C-Eval regression.

Benchmark command:

```bash
CASE_SET=decode512 ALLOW_DIRTY=1 \
SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE=1 \
SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE_MODE=qloop \
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
bash /home/fhkong/wxwork/attncp_performance_benchmark/run_attncp_perf_benchmark.sh
```

Artifact:

- `/tmp/welmv4_attncp_perf/20260624_024845`

Benchmark result:

| Config | Completed | Output throughput | Mean TTFT | Mean ITL | Peak GPU memory |
|---|---:|---:|---:|---:|---:|
| TP4 | 16 | 334.6899 tok/s | 1640.2553 ms | 8.7756 ms | 81665 MiB |
| TP4+CP2 q-loop local-merge | 16 | 222.7499 tok/s | 1959.2789 ms | 14.1835 ms | 63353 MiB |

Comparison:

- Versus the dense CP2 baseline from Experiment 2 / later baseline
  (`mean_itl_ms` around `17.10-25.94` depending on revision), q-loop improves
  decode ITL materially.
- Versus TP4 in the same run, CP2 q-loop is still about `+61.6%` slower in ITL
  and `+19.5%` slower in TTFT.
- Peak memory saving remains about `17.9 GiB/rank`.

Conclusion:

- The fixed-workspace q-loop path proves the previous graph stall is fixable in
  the real server.
- It is a strong performance direction, but it is not a drop-in replacement for
  the dense correctness path because strict logprobs still differ.
- The next decision is product/serving policy:
  - if token stability is the target, continue validating q-loop with
    MMLU/C-Eval and longer/batched prompts, then consider an opt-in runtime
    flag;
  - if strict logprob equivalence is required, q-loop cannot replace dense
    correctness and we need a lower-level numerically closer primitive.

### Experiment 31: Service Full-Q Local-Merge With Fixed Workspaces

Purpose:

- Compare the fixed-workspace `fullq` path against `qloop`.
- Check whether the previous full-Q token drift still exists after the
  graph-focused workspace implementation.

Precision command:

```bash
SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE=1 \
SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE_MODE=fullq \
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
bash /home/fhkong/wxwork/attncp_precision_regression/run_full_precision.sh
```

Artifact:

- `/tmp/welmv4_attncp_precision/20260624_025508`

Precision result:

- TP4+CP2 full-Q server completed CUDA graph capture for
  `bs=[16,12,8,4,2,1]`.
- Controlled output token/text matched for all 4 prompt lengths
  (`64,512,1024,2048`).
- Strict controlled logprob compare still failed with the same profile as
  q-loop:
  - `max_logprob_diff=0.999237060546875`
  - `mean_logprob_diff=0.000500941446027028`
  - `issue_count=283`

Additional token regression:

```bash
# Reused /tmp/welmv4_attncp_precision/20260624_025508/tp4_regression_baseline.pkl
# and tested CP2 full-Q with a large tolerance to isolate token/text mismatch.
python /home/fhkong/wxwork/perf_optimize_scripts/regression_test.py test \
  --server-url http://127.0.0.1:18192 \
  --model welmv4 \
  --baseline-path /tmp/welmv4_attncp_precision/20260624_025508/tp4_regression_baseline.pkl \
  --tolerance 1000
```

Artifact:

- `/tmp/welmv4_attncp_precision/20260624_025508/fullq_token_regression_tolerance1000.log`

Result:

- `Samples tested: 100 / 100`
- `Token mismatches: 0 / 100 PASS`
- `Request errors: 0`

Benchmark command:

```bash
CASE_SET=decode512 ALLOW_DIRTY=1 \
SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE=1 \
SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE_MODE=fullq \
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
bash /home/fhkong/wxwork/attncp_performance_benchmark/run_attncp_perf_benchmark.sh
```

Artifact:

- `/tmp/welmv4_attncp_perf/20260624_025915`

Benchmark result:

| Config | Completed | Output throughput | Mean TTFT | Mean ITL | Peak GPU memory |
|---|---:|---:|---:|---:|---:|
| TP4 | 16 | 335.9090 tok/s | 1622.4135 ms | 8.7672 ms | 81665 MiB |
| TP4+CP2 full-Q local-merge | 16 | 240.4510 tok/s | 1957.1886 ms | 12.8602 ms | 63331 MiB |

Comparison:

- Full-Q is faster than q-loop in this run:
  - q-loop CP2 ITL: `14.1835 ms`
  - full-Q CP2 ITL: `12.8602 ms`
- Full-Q still trails TP4 by about `+46.7%` ITL and `+20.6%` TTFT.
- Peak memory saving remains about `17.9 GiB/rank`.

Conclusion:

- The fixed-workspace full-Q path is currently the best opt-in performance
  candidate for token-stable serving.
- It is still not strict-logprob equivalent to the dense correctness path, so
  it must remain behind an explicit experimental flag unless product policy
  accepts token-level equivalence.
- The next optimization target is the remaining `~4.1 ms/token` ITL gap versus
  TP4. Candidate sources include full-Q FA shape overhead, O/LSE gather/merge,
  and non-attention side effects from CP sharded-KV metadata/replay.

### Experiment 32: Gap Attribution From Existing Logs and Small Microbenches

Purpose:

- Avoid blind optimization after the full-Q local-merge experiment.
- Quantify where the remaining TP4 vs TP4+CP2 gap is likely coming from.

Artifacts read:

- Dense correctness/perf baseline:
  `/tmp/welmv4_attncp_perf/20260624_011827`
- q-loop local-merge:
  `/tmp/welmv4_attncp_perf/20260624_024845`
- full-Q local-merge:
  `/tmp/welmv4_attncp_perf/20260624_025915`

Server log summary for 8k input / 512 output / concurrency 4:

| Run | Config | Mean TTFT | Mean ITL | Bench output throughput | Server steady decode | Server prefill median |
|---|---|---:|---:|---:|---:|---:|
| dense default | TP4 | 1642.89 ms | 8.77 ms | 334.61 tok/s | 472.05 tok/s | 18762.70 tok/s |
| dense default | TP4+CP2 | 1973.97 ms | 17.10 ms | 191.44 tok/s | 239.25 tok/s | 15822.59 tok/s |
| q-loop local-merge | TP4 | 1640.26 ms | 8.78 ms | 334.69 tok/s | 471.84 tok/s | 18740.77 tok/s |
| q-loop local-merge | TP4+CP2 | 1959.28 ms | 14.18 ms | 222.75 tok/s | 289.68 tok/s | 15996.51 tok/s |
| full-Q local-merge | TP4 | 1622.41 ms | 8.77 ms | 335.91 tok/s | 471.99 tok/s | 19105.54 tok/s |
| full-Q local-merge | TP4+CP2 | 1957.19 ms | 12.86 ms | 240.45 tok/s | 320.35 tok/s | 16006.01 tok/s |

Observations:

- full-Q local-merge improves CP2 steady decode from `239.25` to
  `320.35 tok/s`, but TP4 is still `471.99 tok/s`.
- full-Q local-merge improves CP2 ITL from `17.10 ms` to `12.86 ms`, but TP4
  is still `8.77 ms`.
- CP2 prefill median is about `16.0k tok/s`, while TP4 is about
  `19.1k tok/s`, so TTFT also has a real prefill-side gap.

Layer mix:

- The serving model has `48` hidden layers, but the layerwise sliding-window
  list has `49` entries because of the next-token prediction layer.
- Window counts from the model config:
  - `262144`: `25`
  - `512`: `24`
- All `49` listed attention layers enable attention sink.

Existing full-window decode path microbench:

- Artifact: `/tmp/attncp_decode_paths_graph_cp2_8k.json`
- CUDA graph, CP2, bs=4:

| KV len | Path | Median |
|---:|---|---:|
| 8192 | current dense CP gather | 254.30 us/layer |
| 8192 | target full-Q local-merge | 61.44 us/layer |
| 8192 | target q-loop local-merge | 100.04 us/layer |
| 8704 | current dense CP gather | 277.64 us/layer |
| 8704 | target full-Q local-merge | 69.73 us/layer |
| 8704 | target q-loop local-merge | 112.12 us/layer |

SWA-sized decode microbench:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  benchmark/kernels/attention/bench_attncp_decode_paths.py \
  --model-config /home/fhkong/models/80a3_v4d5_256k_merge_thinking_kimi_k25_0502_20260503_032335/epoch_003_step_0002610/config.json \
  --output /tmp/attncp_decode_paths_graph_cp2_swa512.json \
  --kv-lens 512,513,1024 \
  --batch-size 4 \
  --tp-size 4 \
  --cp-size 2 \
  --cp-kv-chunk-size 1024 \
  --warmup 5 \
  --iters 50 \
  --trials 3 \
  --cuda-graph
```

Result:

| KV len | Path | Median |
|---:|---|---:|
| 512 | current dense CP gather | 56.68 us/layer |
| 512 | target full-Q local-merge | 51.54 us/layer |
| 513 | current dense CP gather | 61.12 us/layer |
| 513 | target full-Q local-merge | 55.59 us/layer |
| 1024 | current dense CP gather | 89.62 us/layer |
| 1024 | target full-Q local-merge | 55.83 us/layer |

Interpretation:

- The current implementation only enables local-merge for full-window layers:
  `window_left < 0 or window_left >= max_context_len`.
- SWA `512` layers still use dense-window K/V reconstruction.
- However, SWA-sized dense-window cost is only about `56-61 us/layer`; a
  local-merge SWA path would save only about `5-6 us/layer` at `512/513`.
- Therefore SWA fallback is a contributor, but it is not large enough to
  explain the full `~4 ms/decode-step` steady gap versus TP4.

Attention sink check:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  benchmark/kernels/attention/bench_attncp_decode_paths.py \
  --model-config /home/fhkong/models/80a3_v4d5_256k_merge_thinking_kimi_k25_0502_20260503_032335/epoch_003_step_0002610/config.json \
  --output /tmp/attncp_decode_paths_graph_cp2_8k_no_sinks.json \
  --kv-lens 8192,8704 \
  --batch-size 4 \
  --tp-size 4 \
  --cp-size 2 \
  --cp-kv-chunk-size 1024 \
  --warmup 5 \
  --iters 50 \
  --trials 3 \
  --cuda-graph \
  --disable-sinks
```

Result:

- `8192` full-Q local-merge with sinks: `61.44 us/layer`
- `8192` full-Q local-merge without sinks: `55.59 us/layer`
- `8704` full-Q local-merge with sinks: `69.73 us/layer`
- `8704` full-Q local-merge without sinks: `64.20 us/layer`

Interpretation:

- Per-layer sink gather costs around `5-6 us` in the graph path.
- This is worth optimizing later, but it only accounts for roughly
  `0.15 ms/decode-step` across the `25` full-window layers.

Current attribution:

- full-Q local-merge removes the large dense full-KV reconstruction cost for
  full-window layers and the observed server improvement matches that direction.
- The remaining server gap is larger than the isolated attention-path
  microbench predicts. That points to whole-model effects:
  - extra CP collectives per attention layer interleaving with the model's
    existing TP all-reduces;
  - graph replay/collective launch serialization across many layers;
  - prefill sharded-KV dense reconstruction still being slower than TP4;
  - smaller contributors from SWA dense-window fallback and attention sink
    gather.

Do not blindly optimize next:

- Optimizing attention sink gather or SWA local-merge alone cannot close the
  remaining gap.
- The next useful step should be profiling/measurement that covers a full
  decode step, not just isolated attention kernels.
- The most plausible performance direction is overlap or fusion of the CP
  collectives with existing per-layer compute/TP communication, similar to the
  training AttnCP overlap design referenced in `task.md`.

## Experiment 33: formal 8k/512/c4 benchmark after decode metadata sync removal

Change under test:

- Replaced `_set_sharded_kv_decode_metadata` boolean compaction based on
  `local_valid.nonzero()` with a GPU-side `cumsum + scatter_reduce_` path.
- Purpose: remove the CPU-visible CUDA stream synchronization caused by
  `aten::nonzero` during decode CUDA graph replay metadata preparation.

Validation before formal benchmark:

- `python -m py_compile python/sglang/srt/layers/attention/flashattention_backend.py`
  passed.
- Random CPU/CUDA equivalence test between the old compaction logic and the new
  `scatter_reduce_` logic passed.
- Profile run confirmed the sync was removed:
  - before: `set_cp_metadata` median `6.669 ms`, `aten::nonzero` total
    `80.851 ms` over 9 calls, `cudaStreamSynchronize` total `80.178 ms`;
  - after: `set_cp_metadata` median `0.423 ms`, `aten::nonzero` `0`,
    `cudaStreamSynchronize` `0`.

Formal benchmark:

```bash
CASE_SET=decode512 ALLOW_DIRTY=1 \
SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE=1 \
SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE_MODE=fullq \
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
bash /home/fhkong/wxwork/attncp_performance_benchmark/run_attncp_perf_benchmark.sh
```

Artifacts:

- `/tmp/welmv4_attncp_perf/20260624_032910`

Result:

| Config | Status | TTFT | ITL | Output throughput | Peak GPU memory |
|---|---|---:|---:|---:|---:|
| TP4 | PASS | 1693.43 ms | 8.7567 ms | 332.33 tok/s | 81665 MiB |
| TP4+CP2 | PASS | 1975.20 ms | 12.6513 ms | 242.98 tok/s | 63331 MiB |

Compared with the previous full-Q local-merge run
`/tmp/welmv4_attncp_perf/20260624_025915`:

- TP4+CP2 ITL improved from `12.8602 ms` to `12.6513 ms`.
- TP4+CP2 TTFT changed from `1957.19 ms` to `1975.20 ms`.
- TP4+CP2 output throughput improved from `240.45 tok/s` to `242.98 tok/s`.
- Peak memory stayed at `63331 MiB`.

Interpretation:

- The metadata sync fix is real and should be kept because it removes a
  correctness/performance hazard from the CUDA graph replay path.
- However, the formal benchmark only improved ITL by about `1.6%` relative to
  the previous full-Q local-merge run.
- Current 8k input / 512 output / concurrency 4 gap remains large:
  TP4+CP2 is about `16.6%` slower on TTFT and about `44.5%` slower on ITL than
  TP4, while saving about `18 GiB` peak GPU memory.
- Therefore the remaining bottleneck is not explained by decode metadata
  compaction alone. Do not continue with speculative micro-optimizations until
  the next bottleneck is shown by a full-step decode profile or trace.

## Experiment 34: precision check after decode metadata sync removal

Command:

```bash
SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE=1 \
SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE_MODE=fullq \
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
bash /home/fhkong/wxwork/attncp_precision_regression/run_full_precision.sh
```

Artifacts:

- `/tmp/welmv4_attncp_precision/20260624_033518`

Result:

- TP4 baseline generation succeeded:
  - controlled outputs saved to `controlled_tp4.json`;
  - MMLU/C-Eval baseline generated for `100` samples with `0` errors.
- TP4+CP2 controlled generation succeeded:
  - controlled outputs saved to `controlled_tp4_cp2.json`.
- The wrapper script exited before MMLU/C-Eval because controlled logprob
  comparison uses a strict `1e-5` tolerance:
  - `passed=false`;
  - `max_logprob_diff=9.992e-01`;
  - `mean_logprob_diff=5.009e-04`;
  - `issue_count=283`;
  - first issue:
    `req[2].output_token_logprobs[1].logprob` diff `9.168e-03`.

Controlled token check:

- All `4/4` controlled requests produced identical `output_ids` and identical
  text between TP4 and TP4+CP2.
- Generated-token logprob diffs, excluding top-logprob alternatives:
  - max: `3.403e-02`;
  - mean: `2.872e-03`;
  - p95: `1.403e-02`.

Manual MMLU/C-Eval rerun using the same TP4 baseline and the same TP4+CP2
local-merge full-Q server path:

```bash
python /home/fhkong/wxwork/perf_optimize_scripts/regression_test.py test \
  --server-url http://127.0.0.1:18192 \
  --model welmv4 \
  --baseline-path /tmp/welmv4_attncp_precision/20260624_033518/tp4_regression_baseline.pkl \
  --tolerance 1e-5
```

Result:

- `Samples tested: 100 / 100`
- `Token mismatches: 0 / 100 PASS`
- `Max logprob diff: 0.00e+00 PASS`
- `Mean logprob diff: 0.00e+00 PASS`
- `Result: PASS`

Interpretation:

- The metadata sync removal did not introduce token drift in the covered
  controlled or MMLU/C-Eval cases.
- The current full-Q local-merge path still has non-bitwise logprob differences
  on long controlled `/generate` prompts. This is consistent with the design
  requirement that AttnCP does not need bitwise/logprob equality with NaiveTP,
  but the current wrapper script treats those differences as failures.
- For reporting, use both facts: token-level correctness passes on the covered
  cases, while strict controlled top-logprob equality does not pass.

## Experiment 35: full decode-step trace attribution

Artifacts parsed:

- TP4:
  `/tmp/welmv4_attncp_profile/20260624_032211_tp4_8k128/profile/1782242617.9296858/*TP-*-DECODE.trace.json.gz`
- TP4+CP2 full-Q local-merge after metadata `scatter_reduce_` fix:
  `/tmp/welmv4_attncp_profile/20260624_032647_cp2_fullq_8k128_scatter_reduce/profile/1782242901.9441447/*TP-*-DECODE.trace.json.gz`

Scenario:

- 8k input / 128 output / concurrency 4 profile run.
- Profile traces are not used as formal throughput numbers because profiling
  adds overhead and the first captured decode step can include compile/profile
  noise.
- For attribution, the first decode step was skipped and the two stable decode
  steps across all 4 TP ranks were aggregated.

Stable decode-step summary:

| Config | CPU step median | GPU graph wall median | Kernel sum median |
|---|---:|---:|---:|
| TP4 | 1.619 ms | 8.485 ms | 8.071 ms |
| TP4+CP2 | 3.986 ms | 12.418 ms | 11.005 ms |

Kernel category mean per stable decode step:

| Category | TP4 | TP4+CP2 | Delta |
|---|---:|---:|---:|
| MoE | 3.017 ms | 3.044 ms | +0.027 ms |
| FlashAttention | 1.885 ms | 1.505 ms | -0.380 ms |
| GEMM | 1.288 ms | 1.291 ms | +0.003 ms |
| TP cross-device reduce | 0.582 ms | 0.535 ms | -0.047 ms |
| NCCL CP collectives | 0.014 ms | 1.577 ms | +1.563 ms |
| elementwise/copy | 0.622 ms | 2.038 ms | +1.416 ms |
| index/scatter | 0.011 ms | 0.138 ms | +0.127 ms |
| other | 0.371 ms | 0.546 ms | +0.175 ms |

NCCL structure in one stable CP2 decode step on TP rank 0:

- `124` NCCL kernels total.
- `101` `ncclDevKernel_AllGather_RING_LL`.
- `23` `ncclDevKernel_AllReduce_Sum_bf16_RING_LL`.

Interpretation:

- The remaining decode gap is not from MoE or GEMM; those are essentially flat.
- FlashAttention itself is faster in CP2 because each rank attends to a local
  KV shard.
- The overhead comes from the communication and glue around the sharded-KV
  attention:
  - CP collectives add about `1.56 ms` kernel time per stable decode step;
  - workspace copies/scatters add about `1.5 ms`;
  - graph wall time has about another `~1.0 ms` gap beyond summed kernel deltas,
    likely launch/stream/collective serialization inside graph replay.
- The `101` AllGather count matches the current local-merge algorithm:
  full-window layers do Q all-gather, sink all-gather, O all-gather, and LSE
  all-gather. The model has attention sink enabled, so sink gather is on the
  decode hot path.
- The `23` AllReduce count comes from the remaining SWA dense-window fallback,
  which reconstructs sharded K/V for sliding-window layers.

Optimization implications:

- Do not focus on MoE, GEMM, or FA kernel speed for the current gap.
- Attention sink is static per layer, so per-step sink all-gather is
  structurally wasteful. However, removing it is only valid if full sinks are
  materialized before CUDA graph capture; otherwise the first gather would still
  be captured into the graph.
- The bigger structural issue is that current full-Q local-merge communicates
  full `O` and `LSE` tensors back to every CP rank, then each rank keeps only
  its own head slice. A destination-specific exchange/all-to-all style merge
  could reduce output-side bandwidth, but it is a larger communication semantic
  change and needs isolated correctness/perf validation.
- SWA local-merge can remove the `23` AllReduce kernels, but earlier SWA-sized
  microbench showed only `~5-6 us/layer` saving at 512-token windows, so it is
  not the first lever.
- Small copy cleanup is possible in the current implementation, but it should be
  measured against the `~4 ms` decode wall gap; it is unlikely to close the
  main difference by itself.

## Experiment 36: cache full attention sinks before CUDA graph capture

Change:

- Added a per-layer AttnCP full-sink cache in the FA local-merge workspace path.
- The first non-captured warmup forward all-gathers full sinks and stores both
  bf16 and fp32 copies.
- CUDA graph capture and replay then reuse the cached full sinks, so the
  per-step sink all-gather is not recorded in the decode graph.
- This does not change Q all-gather, sharded-KV residency, local FA compute, or
  O/LSE merge semantics.

Compile check:

```bash
python -m py_compile python/sglang/srt/layers/attention/flashattention_backend.py
```

Result: passed.

Trace validation:

New artifact:

- `/tmp/welmv4_attncp_profile/20260624_035023_cp2_fullq_sink_cache_8k128`

Comparison against Experiment 35 CP2 trace:

| Metric | Before sink cache | After sink cache |
|---|---:|---:|
| AllGather kernels / stable decode step | 101 | 76 |
| AllReduce kernels / stable decode step | 23 | 23 |
| Total NCCL kernels / stable decode step | 124 | 99 |
| CPU step median | 3.986 ms | 3.558 ms |
| GPU graph wall median | 12.418 ms | 12.141 ms |
| Kernel sum median | 11.005 ms | 10.840 ms |
| NCCL category mean | 1.577 ms | 1.449 ms |

Interpretation:

- The `25` removed AllGather kernels match the number of full-window layers
  that used to gather attention sinks every decode step.
- The graph-wall improvement is real but small: about `0.28 ms` per stable
  decode step in this profile.
- This confirms attention sink gather was a measurable contributor, but not the
  dominant remaining bottleneck.

Formal 8k input / 512 output / concurrency 4 benchmark:

Command:

```bash
CASE_SET=decode512 ALLOW_DIRTY=1 \
SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE=1 \
SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE_MODE=fullq \
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
bash /home/fhkong/wxwork/attncp_performance_benchmark/run_attncp_perf_benchmark.sh
```

Artifact:

- `/tmp/welmv4_attncp_perf/20260624_035304`

Result:

| Config | Status | TTFT | ITL | Output throughput | Peak GPU memory |
|---|---|---:|---:|---:|---:|
| TP4 | PASS | 1620.35 ms | 8.7640 ms | 336.11 tok/s | 81665 MiB |
| TP4+CP2 | PASS | 1971.33 ms | 12.4052 ms | 246.77 tok/s | 63305 MiB |

Compared with Experiment 33 (`/tmp/welmv4_attncp_perf/20260624_032910`):

- TP4+CP2 ITL improved from `12.6513 ms` to `12.4052 ms`
  (`~1.9%` relative improvement).
- TP4+CP2 output throughput improved from `242.98 tok/s` to `246.77 tok/s`.
- TP4+CP2 TTFT stayed roughly flat (`1975.20 ms` to `1971.33 ms`).
- Peak memory stayed roughly flat (`63331 MiB` to `63305 MiB`).

Current remaining gap in this formal run:

- TP4+CP2 TTFT is about `21.7%` slower than TP4.
- TP4+CP2 ITL is about `41.5%` slower than TP4.
- TP4+CP2 saves about `18.4 GiB` peak GPU memory.

Precision validation:

Command:

```bash
SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE=1 \
SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE_MODE=fullq \
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
bash /home/fhkong/wxwork/attncp_precision_regression/run_full_precision.sh
```

Artifact:

- `/tmp/welmv4_attncp_precision/20260624_035730`

Result:

- TP4 baseline generated for `100` MMLU/C-Eval samples with `0` errors.
- Controlled compare still fails the strict `1e-5` logprob threshold:
  - `max_logprob_diff=9.992e-01`;
  - `mean_logprob_diff=5.009e-04`;
  - first generated-token logprob diff `9.168e-03`.
- Controlled output token/text check:
  - `4/4` requests have identical `output_ids`;
  - generated-token logprob max diff `3.403e-02`;
  - generated-token logprob mean diff `2.872e-03`.
- Manual MMLU/C-Eval rerun using the same TP4 baseline:
  - `Samples tested: 100 / 100`;
  - `Token mismatches: 0 / 100 PASS`;
  - `Max logprob diff: 0.00e+00 PASS`;
  - `Mean logprob diff: 0.00e+00 PASS`;
  - `Result: PASS`.

Decision:

- Keep the sink cache optimization. It is small, localized, and validated by
  trace, benchmark, and token-level regression.
- Do not expect this to close the main gap. The next meaningful optimization
  target remains output-side communication and workspace copies in full-Q
  local-merge.

## Experiment 37: runtime output-slice exchange attempt, rejected

Purpose:

- Test the service runtime version of the output-side idea from Experiment 28.
- The goal was to avoid full `O`/`LSE` all-gather after each CP rank computes
  partial attention for all Q heads. Each CP rank only needs its local TP Q-head
  slice after LSE merge, so a destination-specific output exchange should
  reduce communicated payload in principle.

Temporary change tested:

- Added a graph-captured grouped P2P output exchange path behind an experimental
  env flag.
- The path replaced the per-layer full `O`/`LSE` all-gather with head-slice
  send/receive before the local LSE merge.
- This was tested only as a performance experiment and has been reverted from
  the current source tree.

Current source check:

```bash
rg "OUTPUT_EXCHANGE|all_to_all_coalesced|o_exchange" python/sglang/srt || true
```

Result: no matches.

Artifacts:

- Sink-cache baseline:
  `/tmp/welmv4_attncp_profile/20260624_035023_cp2_fullq_sink_cache_8k128`
- Output-exchange attempt:
  `/tmp/welmv4_attncp_profile/20260624_040803_cp2_fullq_output_exchange_8k128`

Trace structure, parsed from all TP-rank DECODE traces:

| Metric | Sink-cache all-gather path | Output-exchange path |
|---|---:|---:|
| Stable CPU step median | 3.558 ms | 3.717 ms |
| AllGather kernels / decode step / rank | 76 | 26 |
| SendRecv kernels / decode step / rank | 0 | 25 |
| AllReduce kernels / decode step / rank | 23 | 23 |
| Total NCCL kernels / decode step / rank | 99 | 74 |
| AllGather kernel time / step / rank | 0.438 ms | 0.172 ms |
| SendRecv kernel time / step / rank | 0.000 ms | 0.298 ms |
| AllReduce kernel time / step / rank | 1.012 ms | 1.006 ms |

Previous trace summary for the same artifacts:

| Metric | Sink-cache all-gather path | Output-exchange path |
|---|---:|---:|
| GPU graph wall median | 12.141 ms | 12.347 ms |
| Kernel sum median | 10.840 ms | 11.047 ms |
| NCCL category mean | 1.449 ms | 1.477 ms |

Interpretation:

- The exchange path does reduce full-output AllGather count and payload.
- At CP=2, the saved `O`/`LSE` all-gather payload is small enough that grouped
  P2P SendRecv latency, packing/slicing, and CUDA graph overhead outweigh the
  bandwidth saving.
- The service trace also showed higher graph/capture memory for this path, so
  it is not a good tradeoff for the current TP4+CP2 target.

Decision:

- Do not keep the runtime output-exchange implementation.
- Keep the current full `O`/`LSE` all-gather path until there is a lower-level
  fused/overlapped communication primitive. A Python/PyNccl composition that
  merely swaps AllGather for P2P is not sufficient.
- The next useful optimization should target overlap/fusion around Q gather,
  `O`/`LSE` merge, and workspace copies, with profile evidence before code
  changes.

## Experiment 38: source audit before the next optimization

Purpose:

- Avoid blind optimization after the output-exchange attempt regressed runtime.
- Re-read the current AttnCP decode hot path and identify only changes that are
  directly tied to the profile signal.

Current full-Q workspace path:

- `_forward_decode_sharded_kv` selects local-merge only when:
  - sharded-KV AttnCP is enabled;
  - the explicit experimental local-merge env is on;
  - the layer is a full-window layer;
  - compact local CP page-table metadata exists.
- `_flash_attn_sharded_kv_local_merge_workspace` then:
  - all-gathers local Q into full-Q layout;
  - runs FA over the local KV shard;
  - all-gathers local partial `O` and `LSE`;
  - merges only the current CP rank's TP-head slice.
- SWA/local-window layers still use the dense reconstruct fallback, which
  explains the remaining `23` AllReduce kernels per decode step.

Concrete cleanup candidate:

- `_attncp_gather_full_q()` always copies gathered Q into both:
  - `q_full`, used by the default `fullq` path;
  - `q_shards`, used only by the slower `qloop` path.
- In the current measured benchmark path, `qloop` is not used, so the `q_shards`
  copy is unnecessary copy/glue work on every full-window layer.
- This is a low-risk cleanup candidate because it does not change communication
  semantics, KV residency, FA inputs for `fullq`, or LSE merge math.

Why this is not enough by itself:

- The copied Q tensor is small compared with the full step. Removing the extra
  `q_shards` write can reduce copy kernels and workspace traffic, but it cannot
  explain the remaining `~3.6 ms` ITL gap alone.
- The larger remaining bottleneck is still the combination of CP collectives,
  output/LSE merge, and graph/stream serialization.

Next gate before editing:

- If making this cleanup, keep it scoped to the `fullq` path and run at least:
  - `python -m py_compile python/sglang/srt/layers/attention/flashattention_backend.py`;
  - the AttnCP precision script;
  - a focused 8k/512/c4 benchmark or decode profile to confirm the change is
    not noise.

## Experiment 39: skip qloop-only Q shard copies in full-Q mode

Change:

- `_attncp_gather_full_q()` now takes `copy_q_shards: bool = False`.
- The default `fullq` decode local-merge path gathers Q and materializes only
  `q_full`.
- The slower `qloop` mode still passes `copy_q_shards=True` and keeps the
  previous `q_shards` behavior.
- This does not change Q all-gather, KV shard residency, FA inputs in `fullq`,
  `O`/`LSE` communication, or LSE merge math.

Compile check:

```bash
python -m py_compile python/sglang/srt/layers/attention/flashattention_backend.py
```

Result: passed.

Precision validation:

Command:

```bash
SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE=1 \
SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE_MODE=fullq \
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
bash /home/fhkong/wxwork/attncp_precision_regression/run_full_precision.sh
```

Artifact:

- `/tmp/welmv4_attncp_precision/20260624_041830`

Result:

- TP4 baseline generated for `100` MMLU/C-Eval samples with `0` errors.
- The wrapper exited at the known strict controlled logprob compare:
  - `passed=false`;
  - `max_logprob_diff=9.992e-01`;
  - `mean_logprob_diff=5.009e-04`;
  - first generated-token logprob diff `9.168e-03`.
- Controlled output token/text check:
  - `4/4` requests have identical `output_ids`;
  - `4/4` requests have identical text;
  - generated-token logprob max diff `3.403e-02`;
  - generated-token logprob mean diff `2.872e-03`.
- Manual MMLU/C-Eval rerun against the same TP4 baseline:
  - `Samples tested: 100 / 100`;
  - `Token mismatches: 0 / 100 PASS`;
  - `Max logprob diff: 0.00e+00 PASS`;
  - `Mean logprob diff: 0.00e+00 PASS`;
  - `Result: PASS`.

Focused 8k input / 512 output / concurrency 4 benchmark:

Command:

```bash
CASE_SET=decode512 ALLOW_DIRTY=1 \
SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE=1 \
SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE_MODE=fullq \
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
bash /home/fhkong/wxwork/attncp_performance_benchmark/run_attncp_perf_benchmark.sh
```

Artifact:

- `/tmp/welmv4_attncp_perf/20260624_042559`

Result:

| Config | Status | TTFT | ITL | Output throughput | Peak GPU memory |
|---|---|---:|---:|---:|---:|
| TP4 | PASS | 1629.7710 ms | 8.7700 ms | 335.4253 tok/s | 81665 MiB |
| TP4+CP2 | PASS | 1967.2327 ms | 12.3620 ms | 247.5477 tok/s | 63303 MiB |

Compared with Experiment 36 formal benchmark
(`/tmp/welmv4_attncp_perf/20260624_035304`):

- TP4+CP2 TTFT improved from `1971.33 ms` to `1967.23 ms`
  (`~0.2%`).
- TP4+CP2 ITL improved from `12.4052 ms` to `12.3620 ms`
  (`~0.35%`).
- TP4+CP2 output throughput improved from `246.77 tok/s` to
  `247.55 tok/s` (`~0.3%`).

Current remaining gap in this run:

- TP4+CP2 TTFT is about `20.7%` slower than TP4.
- TP4+CP2 ITL is about `41.0%` slower than TP4.
- TP4+CP2 saves about `17.9 GiB` peak GPU memory.

Interpretation:

- The cleanup is correctness-safe and slightly positive, but the measured
  improvement is tiny and could be close to benchmark noise.
- Keep it because it removes unnecessary full-window decode copy work in the
  measured `fullq` path.
- Do not spend more time on small Python-level copy cleanups unless a new trace
  shows a larger copy hotspot. The remaining gap still points to CP collectives,
  output/LSE merge, and graph/stream serialization.

## Experiment 40: coalesce full-window decode O/LSE all-gather

Motivation:

- In the `fullq` local-merge decode path each full-window layer gathers:
  - full Q before local FA;
  - local O after local FA;
  - local LSE after local FA.
- O and LSE all-gathers are independent and were launched sequentially.
- The dense K/V fallback already benefits from coalescing K/V all-reduce, so
  coalescing O/LSE all-gather is a small, scoped way to test whether launch and
  NCCL grouping overhead is part of the remaining decode gap.

Change:

- Added `PyNcclCommunicator.all_gather_coalesced()`, implemented with
  `ncclGroupStart()` / `ncclGroupEnd()` around multiple `ncclAllGather` calls.
- Added `GroupCoordinator.all_gather_coalesced()` with a sequential
  `all_gather_into_tensor()` fallback.
- Replaced the two full-window local-merge O/LSE all-gathers with one
  coalesced call:
  - `o_gather <- local_o_full`;
  - `lse_gather <- local_lse_full`.
- This does not change Q all-gather, KV shard residency, FA inputs, empty-KV
  sink handling, or LSE merge math.

Compile check:

```bash
python -m py_compile \
  python/sglang/srt/layers/attention/flashattention_backend.py \
  python/sglang/srt/distributed/parallel_state.py \
  python/sglang/srt/distributed/device_communicators/pynccl.py
```

Result: passed.

Precision validation:

Command:

```bash
SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE=1 \
SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE_MODE=fullq \
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
bash /home/fhkong/wxwork/attncp_precision_regression/run_full_precision.sh
```

Artifact:

- `/tmp/welmv4_attncp_precision/20260624_044031`

Result:

- The wrapper exited at the known strict controlled logprob compare:
  - `passed=false`;
  - `max_logprob_diff=9.992e-01`;
  - `mean_logprob_diff=5.009e-04`;
  - first generated-token logprob diff `9.168e-03`.
- Controlled output token/text check:
  - `4/4` requests have identical `output_ids`;
  - `4/4` requests have identical text;
  - generated-token logprob max diff `3.403e-02`;
  - generated-token logprob mean diff `2.872e-03`.
- Manual MMLU/C-Eval rerun against the same TP4 baseline:
  - `Samples tested: 100 / 100`;
  - `Token mismatches: 0 / 100 PASS`;
  - `Max logprob diff: 0.00e+00 PASS`;
  - `Mean logprob diff: 0.00e+00 PASS`;
  - `Result: PASS`.

Focused 8k input / 512 output / concurrency 4 benchmark:

Command:

```bash
CASE_SET=decode512 ALLOW_DIRTY=1 \
SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE=1 \
SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE_MODE=fullq \
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
bash /home/fhkong/wxwork/attncp_performance_benchmark/run_attncp_perf_benchmark.sh
```

Artifact:

- `/tmp/welmv4_attncp_perf/20260624_044814`

Result:

| Config | Status | TTFT | ITL | Output throughput | Peak GPU memory |
|---|---|---:|---:|---:|---:|
| TP4 | PASS | 1721.0381 ms | 8.7713 ms | 330.4585 tok/s | 81665 MiB |
| TP4+CP2 | PASS | 1954.8032 ms | 12.1581 ms | 251.0770 tok/s | 63303 MiB |

Compared with Experiment 39
(`/tmp/welmv4_attncp_perf/20260624_042559`):

- TP4+CP2 TTFT improved from `1967.2327 ms` to `1954.8032 ms`
  (`~0.6%`).
- TP4+CP2 ITL improved from `12.3620 ms` to `12.1581 ms`
  (`~1.6%`).
- TP4+CP2 output throughput improved from `247.5477 tok/s` to
  `251.0770 tok/s` (`~1.4%`).

Current remaining gap in this run:

- TP4+CP2 TTFT is about `13.6%` slower than TP4.
- TP4+CP2 ITL is about `38.6%` slower than TP4.
- TP4+CP2 saves about `17.9 GiB` peak GPU memory.

Interpretation:

- The coalesced O/LSE all-gather change is correctness-safe under the tested
  CUDA graph + KV mirror + attention sink path.
- The endpoint improvement is positive but small. It is reasonable to keep as a
  scoped communication cleanup, but it does not explain or close the remaining
  ITL gap.
- Do not add more communication micro-optimizations without a decode trace that
  shows kernel count, launch serialization, or NCCL time is still the dominant
  cost after this change.

## Experiment 41: current decode trace and SWA local-merge probe

Goal:

- Stop guessing about the remaining TP4 vs TP4+CP2 ITL gap.
- Profile the current code first, then only test an optimization if the trace
  points to a specific source of overhead.

Decode profile setup:

- Workload: 8k input / 512 output / concurrency 4.
- SGLang profile: `--profile --profile-by-stage --profile-stages decode`.
- TP4 artifact:
  - `/tmp/welmv4_attncp_profile/20260624_current_tp4_decode`
  - Profiled mean ITL `10.06 ms`, median ITL `8.39 ms`.
- TP4+CP2 artifact:
  - `/tmp/welmv4_attncp_profile/20260624_current_cp2_decode`
  - Profiled mean ITL `14.03 ms`, median ITL `11.79 ms`.

Trace finding:

- TP4+CP2 has per decode step, normalized by rank and profiled step:
  - about `+23` NCCL all-reduce kernels, about `+1013 us/step`;
  - about `+50` extra NCCL all-gather kernels, about `+343 us/step`;
  - many extra tiny metadata/copy kernels, including direct-copy,
    unrolled-copy, vectorized-gather, scatter-gather, index, where/fill/arange.
- The `+23` all-reduce count matches the number of SWA layers still going
  through the dense K/V fallback path.

Interpretation:

- The remaining gap is not just the full-window local-merge O/LSE communication.
- SWA decode currently reconstructs dense window K/V for sharded KV by doing
  K/V all-reduce and several metadata/copy kernels per SWA layer.
- This is a concrete bottleneck candidate; it is not yet a proven safe
  optimization target.

Probe change:

- Added a local experimental change so SWA decode can build a compact
  per-CP-rank local SWA page table and reuse the local-merge attention path.
- The intended semantic is unchanged sharded KV residency:
  - Q is still all-gathered;
  - each CP rank attends only to its local KV shard;
  - O/LSE are gathered and merged with the same softmax-state merge;
  - SWA windowing is encoded in the compact local page table.

Compile check:

```bash
python -m py_compile python/sglang/srt/layers/attention/flashattention_backend.py
```

Result: passed.

Precision validation:

Command:

```bash
SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE=1 \
SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE_MODE=fullq \
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
bash /home/fhkong/wxwork/attncp_precision_regression/run_full_precision.sh
```

Artifact:

- `/tmp/welmv4_attncp_precision/20260624_050722`

Result:

- The wrapper exited at strict controlled logprob compare:
  - `passed=false`;
  - `max_logprob_diff=8.751e-01`;
  - `mean_logprob_diff=6.246e-04`.
- Controlled output token/text check:
  - `4/4` requests have identical `output_ids`;
  - `4/4` requests have identical text.
- The script did not continue to the full manual MMLU/C-Eval comparison because
  the strict controlled compare failed first.

Conclusion:

- The SWA local-merge probe is not validated yet.
- Do not benchmark or stack more optimizations on top of it before isolating
  whether the larger top-logprob differences are expected numeric noise or a
  correctness bug in the SWA local page-table construction/merge path.

Follow-up guard:

- Added `SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE_SWA=1` as a separate
  opt-in guard for the SWA local-merge probe.
- Default `SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE=1` now keeps SWA
  decode on the previous dense-window fallback path.
- This avoids accidentally benchmarking or serving with an unvalidated SWA
  local-merge path while preserving the probe code for isolated debugging.

Validation after adding the guard:

- Artifact: `/tmp/welmv4_attncp_precision/20260624_051557`
- Command:

```bash
SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE=1 \
SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE_MODE=fullq \
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
bash /home/fhkong/wxwork/attncp_precision_regression/run_full_precision.sh
```

- Controlled strict compare still fails, as expected for the local-merge
  opt-in path:
  - `max_logprob_diff=8.751e-01`;
  - `mean_logprob_diff=6.246e-04`;
  - `issue_count=280`.
- Controlled output token/text check:
  - `4/4` requests have identical `output_ids`;
  - `4/4` requests have identical text.
- Manual MMLU/C-Eval token-level regression against the same TP4 baseline:
  - `Samples tested: 100 / 100`;
  - `Token mismatches: 0 / 100 PASS`;
  - `Max logprob diff: 0.00e+00 PASS`;
  - `Mean logprob diff: 0.00e+00 PASS`;
  - `Result: PASS`.

Training overlap reference check:

- Reference path from `task.md`:
  `/home/fhkong/wxwork/mimikyu/mmq/mmq/modules/block_v2/memory_optimizer/qkv_proj_and_post_processing/overlap.py`.
- The training implementation launches async AP all-to-all for Q first, then
  computes KV projection and gate projection before waiting in
  `post_all2all()`.
- This pattern hides communication under substantial projection compute.
- Serving decode is different:
  - QKV projection is already complete before attention backend decode;
  - the remaining CP collectives are inside attention: Q all-gather, O/LSE
    gather, and SWA dense K/V all-reduce;
  - there is no equivalent large KV/gate projection window inside
    `flashattention_backend.py` to hide these collectives.
- Therefore the next overlap attempt should not blindly port the training
  qkv path. It should first target a serving-specific schedule, e.g. moving
  per-layer CP collectives onto a dedicated communication stream and proving
  overlap with adjacent layer compute in a decode trace.

## Experiment 42: SWA dense-window static metadata cleanup

Motivation:

- Experiment 41 showed the remaining decode gap is dominated by SWA dense K/V
  reconstruction and CP collectives.
- Replacing SWA dense fallback with SWA local-merge is not yet correctness
  validated, so the safe direction is to keep the dense fallback semantics and
  remove only obviously redundant metadata work.
- The trace showed repeated `arange_cuda`, row-index, and compact-slot creation
  around `_gather_sharded_kv_dense_decode_window()`.

Change:

- Preallocate static SWA dense-window tensors in CUDA graph metadata:
  - `attncp_dense_window_offsets`;
  - `attncp_dense_window_rows`;
  - `attncp_dense_window_compact_slots`.
- Reuse these tensors in `_gather_sharded_kv_dense_decode_window()` instead of
  rebuilding them for every SWA layer.
- Added a hard guard in `_forward_decode_sharded_kv()` so SWA window layers use
  local-merge only when
  `SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE_SWA=1` is explicitly set.
  This prevents any ambiguous page-table path from accidentally enabling the
  unvalidated SWA local-merge probe.

Compile check:

```bash
python -m py_compile python/sglang/srt/layers/attention/flashattention_backend.py
```

Result: passed.

Precision validation:

Artifact:

- `/tmp/welmv4_attncp_precision/20260624_054831`

Result:

- Controlled strict compare still exits at the known local-merge logprob
  threshold:
  - `max_logprob_diff=9.992e-01`;
  - `mean_logprob_diff=5.009e-04`;
  - `issue_count=283`.
- Controlled output token/text check:
  - `4/4` requests have identical `output_ids`;
  - `4/4` requests have identical text.
- Manual MMLU/C-Eval token-level regression against the same TP4 baseline:
  - `Samples tested: 100 / 100`;
  - `Token mismatches: 0 / 100 PASS`;
  - `Max logprob diff: 0.00e+00 PASS`;
  - `Mean logprob diff: 0.00e+00 PASS`;
  - `Result: PASS`.

Decode profile validation:

- Artifact:
  `/tmp/welmv4_attncp_profile/20260624_054528_cp2_dense_guard_verify`
- Scenario: 8k input / 512 output / concurrency 4 / 8 prompts,
  `--profile-by-stage --profile-stages decode`.

Key per-rank, per-profiled-step trace deltas versus Experiment 41 CP2 trace:

| Kernel family | Before | After | Delta |
|---|---:|---:|---:|
| NCCL AllReduce | `23.00 / 1013.4 us` | `23.00 / 1009.6 us` | unchanged |
| NCCL AllGather | `51.00 / 357.4 us` | `51.00 / 354.8 us` | unchanged |
| `arange_cuda` | `70.88 / 68.6 us` | `1.88 / 2.2 us` | `-69.00 / -66.4 us` |
| `vectorized_gather` | `46.00 / 125.2 us` | `46.00 / 125.2 us` | unchanged |
| `index_elementwise` | `24.00 / 109.6 us` | `24.00 / 109.6 us` | unchanged |

Endpoint profile metrics:

- Old CP2 profile artifact:
  `/tmp/welmv4_attncp_profile/20260624_current_cp2_decode`
  - mean ITL `14.03 ms`;
  - median ITL `11.79 ms`.
- New CP2 profile artifact:
  `/tmp/welmv4_attncp_profile/20260624_054528_cp2_dense_guard_verify`
  - mean ITL `14.59 ms`;
  - median ITL `11.73 ms`.

Conclusion:

- This is a correctness-safe cleanup, but it is intentionally small.
- It removes about `66 us` of repeated `arange_cuda` metadata kernels per
  profiled decode step while preserving the dense SWA fallback, all-reduce
  count, all-gather count, and sharded-KV semantics.
- It does not materially close the TP4 vs TP4+CP2 ITL gap. The next useful
  optimization must target the actual heavy pieces: SWA K/V all-reduce, dense
  window gather/scatter, or a serving-specific overlap schedule.

## Experiment 43: SWA local-merge validation

Motivation:

- Experiment 42 showed that preserving SWA dense fallback keeps correctness
  but leaves the main decode gap: per-step SWA K/V reconstruction plus CP
  collectives.
- The task accepts non-bitwise numeric differences as long as output tokens
  remain aligned, so SWA local-merge should be evaluated directly instead of
  rejected only by strict logprob tolerance.

Precision probe:

- Artifact:
  `/tmp/welmv4_attncp_precision/20260624_060156_cp2_swa_local_merge_probe`
- Server env:
  - `SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE=1`;
  - `SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE_MODE=fullq`;
  - `SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE_SWA=1`.

Result:

- Controlled strict logprob compare still fails:
  - `max_logprob_diff=8.751e-01`;
  - `mean_logprob_diff=6.246e-04`;
  - `issue_count=280`.
- Controlled output ids/text stay aligned.
- Manual MMLU/C-Eval token-level regression against the same TP4 baseline:
  - `Samples tested: 100 / 100`;
  - `Token mismatches: 0 / 100 PASS`;
  - `Max logprob diff: 0.00e+00 PASS`;
  - `Mean logprob diff: 0.00e+00 PASS`;
  - `Result: PASS`.

Conclusion:

- SWA local-merge has the same acceptable token-level behavior as the
  existing non-SWA local-merge path under the current task criterion.
- Strict top-logprob/chosen-logprob equality is still not achieved and should
  not be claimed.

## Experiment 44: 8k/512/c4 decode benchmark split

Artifact:

- `/tmp/welmv4_attncp_perf/20260624_060703_current_swa_local_merge_probe`

Scenario:

- Model:
  `/home/fhkong/models/80a3_v4d5_256k_merge_thinking_kimi_k25_0502_20260503_032335/epoch_003_step_0002610`
- 8k random input / 512 output / concurrency 4 / 16 prompts.
- Common server args:
  - `--tp 4`;
  - `--page-size 1`;
  - `--chunked-prefill-size 1024`;
  - `--enable-welm-kv-mirror-opt`;
  - `--disable-radix-cache`;
  - `--cuda-graph-max-bs 16`;
  - `--disable-piecewise-cuda-graph`;
  - FA3 prefill/decode.

Results:

| Config | Mean TTFT | Mean ITL | Output TPS | Notes |
|---|---:|---:|---:|---|
| TP4 | `1634.07 ms` | `8.76 ms` | `335.51 tok/s` | baseline |
| CP2, no local-merge env | `1984.65 ms` | `17.03 ms` | `191.91 tok/s` | falls back to dense materialization |
| CP2, local-merge + SWA dense fallback | `1973.97 ms` | `12.11 ms` | `251.20 tok/s` | previous safe path |
| CP2, local-merge + SWA local-merge | `1962.59 ms` | `10.88 ms` | `272.56 tok/s` | fastest validated CP2 path |

Interpretation:

- Requiring an env var for local-merge is a performance footgun: with only the
  public AttnCP CLI flags, CP2 falls to `17.03 ms` mean ITL.
- Ordinary local-merge is the biggest decode fix in this comparison:
  `17.03 ms -> 12.11 ms`.
- SWA local-merge gives a further decode improvement:
  `12.11 ms -> 10.88 ms`.
- The remaining ITL gap versus TP4 is still about `+24%`
  (`10.88 ms` vs `8.76 ms`).
- TTFT remains about `+20%`, and the SWA decode change does not materially
  affect it.

## Experiment 45: make sharded-KV decode local-merge the default

Change:

- In `flashattention_backend.py`, sharded-KV decode now enables local-merge by
  default when `--attn-cp-mode sharded-kv` is active.
- SWA local-merge is also enabled by default.
- The existing env names remain as opt-out/debug switches:
  - set `SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE=0` to disable all decode
    local-merge;
  - set `SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE_SWA=0` to keep SWA on
    dense fallback while preserving non-SWA local-merge.

Compile check:

```bash
python -m py_compile python/sglang/srt/layers/attention/flashattention_backend.py
```

Result: passed.

Default-path validation:

- Artifact:
  `/tmp/welmv4_attncp_verify/20260624_062025_default_local_merge`
- Server command used only the public AttnCP CLI flags:
  - `--attn-cp-size 2`;
  - `--attn-cp-mode sharded-kv`;
  - `--attn-cp-kv-chunk-size 1024`;
  - `--attn-cp-decode-cuda-graph-max-seq-len 8704`.
- No local-merge env vars were set.

Precision result:

- Controlled strict logprob compare:
  - `passed=false`;
  - `max_logprob_diff=8.751e-01`;
  - `mean_logprob_diff=6.246e-04`;
  - `issue_count=280`.
- Manual MMLU/C-Eval token-level regression:
  - `Samples tested: 100 / 100`;
  - `Token mismatches: 0 / 100 PASS`;
  - `Max logprob diff: 0.00e+00 PASS`;
  - `Mean logprob diff: 0.00e+00 PASS`;
  - `Result: PASS`.

Benchmark result:

- Same 8k/512/c4 scenario on the same no-env CP2 server:
  - mean TTFT `2112.80 ms`;
  - mean ITL `10.86 ms`;
  - output throughput `267.69 tok/s`.

Conclusion:

- The public AttnCP CLI path now reaches the fast decode implementation by
  default; users no longer need hidden local-merge env vars for the intended
  sharded-KV decode behavior.
- The verified no-env ITL matches the explicit SWA local-merge benchmark
  (`10.86 ms` vs `10.88 ms`).
- TTFT remains the next unresolved gap.

## TTFT / extend trace notes

Existing stage profile artifacts:

- TP4: `/tmp/welmv4_attncp_profile/20260624_current_tp4_decode`
- CP2: `/tmp/welmv4_attncp_profile/20260624_current_cp2_decode`

EXTEND trace comparison:

- TP4 per-rank median `step[EXTEND bs=1 toks=1024]`:
  about `89.8-91.0 ms`.
- CP2 per-rank median `step[EXTEND bs=1 toks=1024]`:
  about `107.6-109.2 ms`.
- CPU op aggregate per extend step:
  - TP4: about `77.6 ms`, `766` `cudaLaunchKernel` events;
  - CP2: about `98.7 ms`, `1255` `cudaLaunchKernel` events.
- CP2 has additional CPU/runtime overhead in:
  - `sglang::outplace_all_reduce`;
  - `sgl_kernel::all_reduce`;
  - `aten::item` / `aten::_local_scalar_dense`;
  - `aten::arange`;
  - `aten::index`;
  - `cudaStreamSynchronize`.

TTFT conclusion:

- The remaining TTFT gap is not explained by SWA decode.
- It appears in chunked prefill/extend as extra launch/sync/metadata overhead.
- The next optimization should focus on reducing CP sharded-KV prefill metadata
  and synchronization work before touching more decode code.

## Experiment 46: remove hot-path prefill metadata `.item()` sync

Motivation:

- The EXTEND trace showed CP2 spends extra time in
  `aten::item` / `aten::_local_scalar_dense` and `cudaStreamSynchronize`.
- `_forward_extend_sharded_kv()` had a metadata sanity check:
  `cu_seqlens_q[-1].item()`.
- `cu_seqlens_q` is a CUDA tensor in this path, so the check can force a GPU to
  CPU synchronization on every layer during chunked prefill.

Change:

- Added `SGLANG_ATTNCP_DEBUG_METADATA_CHECKS`.
- The metadata check is now opt-in debug behavior for sharded-KV AttnCP.
- Default serving path skips the `.item()` synchronization.
- This does not change attention computation; it only removes a hot-path
  runtime assertion.

Compile check:

```bash
python -m py_compile python/sglang/srt/layers/attention/flashattention_backend.py
```

Result: passed.

Benchmark:

- Artifact:
  `/tmp/welmv4_attncp_perf/20260624_062610_cp2_no_item_check`
- Scenario: same 8k input / 512 output / concurrency 4 / 16 prompts.
- Server used only public AttnCP CLI flags, no local-merge env vars.

Result:

- Mean TTFT: `1871.20 ms`;
- Mean ITL: `10.85 ms`;
- Output throughput: `276.48 tok/s`.

Interpretation:

- ITL is unchanged versus the default local-merge validation (`10.86 ms`),
  as expected.
- TTFT improved versus the previous no-env validation run
  (`2112.80 ms -> 1871.20 ms`) and is also better than the clean
  SWA-local-merge split run (`1962.59 ms`).
- Because TTFT has benchmark noise, the safe claim is that the sync removal is
  directionally positive and should be kept, but a fresh EXTEND profile is
  still needed before claiming the entire TTFT gap is solved.
- Remaining TTFT gap versus TP4 from the same clean split baseline is about
  `+14.5%` (`1871.20 ms` vs `1634.07 ms`).

## Experiment 47: cache dense-gather static metadata

Motivation:

- After removing the `.item()` sync, the EXTEND profile still showed repeated
  metadata construction overhead around dense gather, especially `aten::arange`.
- This metadata is shape-dependent and does not need to be rebuilt for every
  layer.

Change:

- Added a small shape-keyed cache for dense-gather static tensors in the FA
  backend.
- Reused cached `logical_pos` and `dense_page_table` tensors across layers.
- This only changes metadata construction; it does not change sharded-KV
  residency or attention math.

Compile check:

```bash
python -m py_compile python/sglang/srt/layers/attention/flashattention_backend.py
```

Result: passed.

Benchmark:

- Artifact:
  `/tmp/welmv4_attncp_perf/20260624_063544_cp2_dense_gather_static`
- Scenario: same 8k input / 512 output / concurrency 4 / 16 prompts.
- Server used only public AttnCP CLI flags, no local-merge env vars.

Result:

- Mean TTFT: `1861.08 ms`;
- Mean ITL: `10.86 ms`;
- Output throughput: `276.66 tok/s`.

Precision check:

- Artifact:
  `/tmp/welmv4_attncp_verify/20260624_063836_after_dense_gather_static`
- Baseline:
  `/tmp/welmv4_attncp_precision/20260624_054831`

Controlled strict logprob compare:

- `passed=false`;
- `max_logprob_diff=8.751e-01`;
- `mean_logprob_diff=6.246e-04`;
- `issue_count=280`.

Manual MMLU/C-Eval token-level regression:

- `Samples tested: 100 / 100`;
- `Token mismatches: 0 / 100 PASS`;
- `Max logprob diff: 0.00e+00 PASS`;
- `Mean logprob diff: 0.00e+00 PASS`;
- `Result: PASS`.

Interpretation:

- The static metadata cache has only a small TTFT effect in the end-to-end
  benchmark (`1871.20 ms -> 1861.08 ms`) and does not change ITL.
- It does not introduce an obvious precision regression: token-level regression
  remains passing, while strict controlled logprob comparison remains failing at
  the same level as before.
- The remaining performance gap should not be optimized blindly; the next step
  should be a focused EXTEND profile that proves where the residual TTFT
  overhead comes from.

## Experiment 48: focused EXTEND profile after dense-gather cache

Motivation:

- The latest benchmark still shows TP4+CP2 TTFT about `13.9%` slower than TP4
  in the 8k input / 512 output / concurrency 4 scenario.
- Before making more changes, verify whether the residual gap still comes from
  chunked prefill/EXTEND and identify the concrete code path.

Profile command:

- Artifact:
  `/tmp/welmv4_attncp_profile/20260624_064308_cp2_extend_after_dense_gather_static`
- Server:
  public AttnCP CLI only, `--attn-cp-size 2 --attn-cp-mode sharded-kv
  --attn-cp-kv-chunk-size 1024
  --attn-cp-decode-cuda-graph-max-seq-len 8704`.
- Benchmark/profile:
  random ids, `input_len=8192`, `output_len=512`, `num_prompts=8`,
  `concurrency=4`, `--profile --profile-by-stage --profile-stages extend
  --profile-num-steps 8`.

Trace comparison:

| Profile | EXTEND step median | Notes |
| --- | ---: | --- |
| TP4 baseline | `90.80 ms` | Existing artifact `/tmp/welmv4_attncp_profile/20260624_current_tp4_decode` |
| CP2 after `.item()` removal | `104.82 ms` | Existing artifact `/tmp/welmv4_attncp_profile/20260624_063017_cp2_extend_after_noitem` |
| CP2 after dense-gather cache | `103.25 ms` | New artifact above |

Selected trace totals across all profiled EXTEND step windows:

| Op / path | TP4 | CP2 after dense-gather cache |
| --- | ---: | ---: |
| `cudaLaunchKernel` | `24395` launches / `118.20 ms` | `36840` launches / `163.33 ms` |
| `sglang::outplace_all_reduce` | `2716` / `117.52 ms` | `4768` / `191.33 ms` |
| `aten::arange` | `64` / `0.92 ms` | `192` / `2.48 ms` |
| `aten::index` | `164` / `5.26 ms` | `2272` / `40.71 ms` |
| `_forward_extend_sharded_kv` | none | `1056` / `475.74 ms` |
| `_flash_attn_sharded_kv_dense` | none | `1056` / `440.05 ms` |
| `_gather_sharded_kv_dense` | none | `1056` / `320.92 ms` |

Interpretation:

- The dense-gather cache removed the previous metadata outliers:
  - old CP2 `aten::arange`: `4528` / `59.50 ms`;
  - new CP2 `aten::arange`: `192` / `2.48 ms`;
  - old CP2 `aten::index`: `2396` / `888.96 ms`;
  - new CP2 `aten::index`: `2272` / `40.71 ms`.
- The remaining TTFT gap is now explained by the implementation strategy, not
  a small metadata bug:
  - `_forward_extend_sharded_kv()` is a correctness-first dense path;
  - every sharded-KV attention layer calls `_flash_attn_sharded_kv_dense()`;
  - `_flash_attn_sharded_kv_dense()` calls `_gather_sharded_kv_dense()`;
  - `_gather_sharded_kv_dense()` materializes temporary full K/V via CP-group
    `all_reduce_coalesced([local_k, local_v])`.
- This differs from the design target in
  `docs/ring-attn/design-sharded-kv-cp.md`, where chunked prefill should keep
  K/V local, all-gather Q heads, compute partial attention on owned KV
  segments, then merge LSE/output across the sharded-KV CP group.

Important boundary:

- The decode local-merge code cannot be blindly reused for prefill.
- Decode has one Q row per sequence, so compact local KV does not need to
  preserve full causal positions.
- Prefill has multi-token Q chunks. Compacting local KV loses global logical
  positions, so correctness requires segment-level causal handling:
  - past owner segments can be attended with non-causal/full visibility;
  - the current overlapping owner segment needs the correct causal mask;
  - future/non-written segments must not participate.

Conclusion:

- More metadata caching is unlikely to close the TTFT gap.
- The next real optimization should implement a prefill-specific sharded-KV
  local-merge path that preserves global causal semantics without reconstructing
  full K/V. It should be introduced behind a guard and validated against:
  controlled token output, MMLU/C-Eval token regression, and 8k/512/c4
  benchmark before replacing the dense correctness path.

Training-code reference:

- Reviewed:
  `/home/fhkong/wxwork/mimikyu/mmq/mmq/modules/block_v2/memory_optimizer/qkv_proj_and_post_processing/overlap.py`
  and ring attention wrappers under
  `/home/fhkong/wxwork/mimikyu/mmq/mmq/modules/attention/`.
- The qkv overlap path overlaps Q/K/V distribution after projection, but the
  part relevant to this inference TTFT gap is the attention merge structure.
- `ring_attn_v3.py` computes the self block with `causal=True`, computes other
  eligible ring blocks with `causal=False`, and merges partial outputs/LSE via
  `update_out_and_lse`.
- `zigzag_*` variants make the same distinction more explicitly: self/document
  block uses causal mask, remote/front/back blocks use full-attention masks.
- This supports the next implementation direction: a prefill-specific
  sharded-KV segment loop should not compact all local KV and call one causal FA
  kernel. It needs segment-aware causal/full handling and LSE merge, similar in
  spirit to the training ring attention code, while preserving SGLang's
  sharded-KV cache residency.

## Experiment 49: opt-in prefill local-merge prototype sanity check

Date: 2026-06-24

Change under test:

- Added an opt-in experimental prefill local-merge path guarded by
  `SGLANG_ATTNCP_EXPERIMENTAL_PREFILL_LOCAL_MERGE=1`.
- The default path remains unchanged when the env flag is not set.

Validation:

- Artifact:
  `/tmp/welmv4_attncp_verify/20260624_065749_prefill_local_merge_proto`
- Controlled compare against TP4 reference:
  - passed: `False`
  - max diff: `8.751e-01`
  - mean diff: `6.246e-04`
  - issues: `280`
  - first mismatch:
    `req[2].output_token_logprobs[1].logprob`, diff `6.981e-03`

Interpretation:

- The prototype did not improve precision versus the dense-gather path.
- No benchmark should be trusted for this prototype until correctness is fixed.
- Keep the prototype disabled by default; do not replace the dense correctness
  path blindly.

Follow-up:

- Removed this failed prototype from `flashattention_backend.py` to keep the
  active code path explainable.
- Verified no `SGLANG_ATTNCP_EXPERIMENTAL_PREFILL_LOCAL_MERGE` /
  `prefill_local_merge` symbols remain in the attention backend.
- `python -m py_compile python/sglang/srt/layers/attention/flashattention_backend.py`
  passed after cleanup.

## Current source boundary after cleanup

Date: 2026-06-24

Relevant code paths:

- Prefill sharded-KV:
  - `_forward_extend_sharded_kv()` is explicitly a temporary full-KV
    correctness path.
  - It always calls `_flash_attn_sharded_kv_dense()`.
  - `_flash_attn_sharded_kv_dense()` calls `_gather_sharded_kv_dense()` for
    prefill, which materializes dense K/V via
    `cp_group.all_reduce_coalesced([local_k, local_v])`.
- Decode sharded-KV:
  - `_forward_decode_sharded_kv()` can use
    `_flash_attn_sharded_kv_local_merge()`.
  - The local-merge path does Q head all-gather, local KV FA, O/LSE all-gather
    and `merge_state_v2`, then slices back to local TP heads.
  - Dense fallback remains for unsupported translated page-table/SWA cases.

Implication:

- The remaining TTFT gap is primarily a prefill implementation strategy gap:
  current code still reconstructs temporary full K/V instead of running
  per-owner-chunk local KV attention.
- The remaining ITL gap is no longer the old full-KV decode path in the common
  case, but there is still per-layer Q/O/LSE collective and Python-level merge
  overhead. The next decode optimization should be graph/fused workspace
  focused, not another dense-gather metadata tweak.
- Do not continue micro-optimizing the dense correctness path unless a new
  profile identifies a concrete regression. It cannot remove the main data
  movement that explains the current performance gap.

## Experiment 50: kv-mirror last-Q prefill local-merge

Date: 2026-06-24

Motivation:

- The failed prefill local-merge prototype tried to handle general multi-token
  prefill and did not cover WeLM kv-mirror contraction.
- In kv-mirror contracted prefill, attention Q has one last-query row per
  active request. This is decode-like: each Q row attends all cache tokens for
  that request, so the already validated sharded-KV local-merge path can be
  reused without implementing the full multi-token segment causal loop.

Change:

- Let `_flash_attn_sharded_kv_local_merge[_workspace]()` accept explicit
  `cu_seqlens_q` and `max_seqlen_q`.
- In `_forward_extend_sharded_kv()`, when `use_welm_custom_last_q` is true,
  Q rows match active request count, and the layer is full-window, build compact
  local CP page-table metadata and call the local-merge path instead of
  temporary dense full-KV reconstruction.
- Ordinary multi-token prefill still uses the dense correctness path.

Pre-code semantic check:

- Ran a synthetic CUDA single-layer test comparing dense causal FA against
  owner-chunk segment FA + `merge_state_v2`.
- Covered full prefill, chunked prefill with prefix, and attention sink injected
  exactly once.
- Result: output max diff `<= 0.0078125` in bf16, LSE max diff
  `<= 4.768e-07`, consistent with merge-order numeric noise.

Compile check:

```bash
python -m py_compile python/sglang/srt/layers/attention/flashattention_backend.py
```

Result: passed.

Precision validation:

- Artifact:
  `/tmp/welmv4_attncp_precision/20260624_071013`
- Controlled strict compare remains at the known local-merge logprob level:
  - `passed=false`;
  - `max_logprob_diff=8.751e-01`;
  - `mean_logprob_diff=6.246e-04`;
  - `issue_count=280`.
- Controlled output token/text check:
  - `4/4` requests have identical `output_ids`;
  - `4/4` requests have identical text.
- Manual MMLU/C-Eval token-level regression against the TP4 baseline generated
  in the same artifact:
  - `Samples tested: 100 / 100`;
  - `Token mismatches: 0 / 100 PASS`;
  - `Max logprob diff: 0.00e+00 PASS`;
  - `Mean logprob diff: 0.00e+00 PASS`;
  - `Result: PASS`.

Next validation:

- Run the standard 8k input / 512 output / concurrency 4 benchmark to determine
  whether this targeted prefill change actually reduces TTFT.

Benchmark validation:

1. Initial last-Q local-merge condition:
   - Artifact: `/tmp/welmv4_attncp_perf/20260624_071723`
   - TP4: TTFT `1613.7901 ms`, ITL `8.7665 ms`, output TPS `336.4022`
   - TP4+CP2: TTFT `1870.1798 ms`, ITL `10.8761 ms`, output TPS `276.0660`
   - Compared with Experiment 47 CP2 (`1861.08 ms` TTFT, `10.86 ms` ITL),
     this did not improve TTFT.

2. Full-window-for-current-batch condition fix:
   - Artifact: `/tmp/welmv4_attncp_precision/20260624_072200`
   - Precision:
     - controlled strict compare still at known local-merge diff level;
     - controlled output token/text `4/4` identical;
     - MMLU/C-Eval token-level regression `100/100 PASS`.
   - Benchmark artifact: `/tmp/welmv4_attncp_perf/20260624_072824`
   - TP4: TTFT `1630.0087 ms`, ITL `8.7634 ms`, output TPS `335.5919`
   - TP4+CP2: TTFT `1887.8211 ms`, ITL `10.8238 ms`, output TPS `276.4270`

Decision:

- Reverted the kv-mirror last-Q prefill local-merge experiment from the active
  source tree.
- It was precision-safe at token level, but it did not reduce TTFT in the target
  8k/512/c4 benchmark and added attention-path complexity.
- Do not retry this narrow optimization without a trace proving the branch is
  both hit and material to EXTEND latency.

## Experiment 51: standalone prefill owner-chunk probe

Date: 2026-06-24

Motivation:

- Before changing the service path again, isolate the proposed prefill
  sharded-KV algorithm outside SGLang serving.
- Check whether owner-chunk local attention plus LSE merge has acceptable
  numerics and whether the compute shape is likely to recover the observed
  TTFT gap.

Harness:

- Added standalone script:
  `benchmark/kernels/attention/bench_attncp_prefill_segments.py`.
- It builds WeLM TP4+CP2-style local shapes, page-size 1 KV cache, optional
  attention sinks, dense FA reference, and owner-chunk segment FA partials.
- Segment rules:
  - past owner chunks use full/non-causal attention;
  - the current overlap chunk uses causal attention;
  - future chunks are skipped;
  - attention sink is injected exactly once;
  - partial O/LSE values are merged with `merge_state_v2`.
- This script is a microbench only. It is not wired into the serving path.

Checks:

```bash
python -m py_compile benchmark/kernels/attention/bench_attncp_prefill_segments.py

CUDA_VISIBLE_DEVICES=0 python benchmark/kernels/attention/bench_attncp_prefill_segments.py \
  --kv-len 512 --q-len 128 --q-starts 0,128,384 \
  --cp-kv-chunk-size 128 --include-sinks \
  --warmup 1 --iters 2 --trials 2 \
  --output /tmp/attncp_prefill_segments_smoke.json

CUDA_VISIBLE_DEVICES=0 python benchmark/kernels/attention/bench_attncp_prefill_segments.py \
  --kv-len 8192 --q-len 1024 \
  --q-starts 0,1024,2048,3072,4096,5120,6144,7168 \
  --cp-kv-chunk-size 1024 --include-sinks \
  --warmup 1 --iters 2 --trials 2 \
  --output /tmp/attncp_prefill_segments_8k_all_chunks.json
```

Result:

- Correctness is acceptable for bf16:
  - max output diff up to `0.015625`;
  - LSE diff up to `2.86102e-06`.
- For full 8k prefill split into eight 1024-token chunks:
  - dense total: `5200.376 us`;
  - serial segment total: `8588.768 us`, `1.652x` dense;
  - ideal CP-rank-parallel segment total: `4986.808 us`, `0.959x` dense.
- Per-chunk ideal segment speedup only appears in later chunks:
  - `q_start=0`: ideal/dense `1.399`;
  - `q_start=1024`: `1.134`;
  - `q_start=2048`: `1.120`;
  - `q_start=3072`: `0.943`;
  - `q_start=4096`: `0.985`;
  - `q_start=5120`: `0.886`;
  - `q_start=6144`: `0.925`;
  - `q_start=7168`: `0.869`.

Communication estimate:

- WeLM GQA has much larger Q/O traffic than KV traffic in this topology.
- A smoke estimate for the current design shows:
  - dense K/V all-reduce input bytes per rank: `262144`;
  - current Q/O/LSE collective input bytes per rank: `592896`;
  - ideal reduced Q/O/LSE input bytes per rank: `396288`.

Decision:

- Do not integrate a naive prefill owner-chunk loop into the serving path.
- The math is viable, but the service implementation would need fused or
  graph-friendly collectives and reduced O/LSE traffic to be worthwhile.
- The next prefill optimization should start from a trace-backed design around
  `cp_lse_ag_out_rs`-style fused communication or overlap, not another Python
  control-flow branch in `flashattention_backend.py`.

## Experiment 52: short decode microbench on current source

Date: 2026-06-24

Motivation:

- Re-check the current decode cost model before making any more code changes.
- The target 8k/512/c4 service benchmark shows TP4+CP2 ITL is still about 24%
  slower than TP4. This experiment isolates one attention layer at the same
  8k KV length and batch size 4.

Commands:

```bash
python -m py_compile \
  benchmark/kernels/attention/bench_attncp_decode_collectives.py \
  benchmark/kernels/attention/bench_attncp_decode_components.py \
  benchmark/kernels/attention/bench_attncp_decode_paths.py \
  benchmark/kernels/attention/bench_attncp_prefill_segments.py

NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
CUDA_VISIBLE_DEVICES=0 \
python benchmark/kernels/attention/bench_attncp_decode_components.py \
  --kv-lens 8192 --batch-size 4 --warmup 5 --iters 30 --trials 3 \
  --output /tmp/attncp_decode_components_current_8k.json

NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
CUDA_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nproc_per_node=2 \
  benchmark/kernels/attention/bench_attncp_decode_collectives.py \
  --kv-lens 8192 --batch-size 4 --warmup 5 --iters 30 --trials 3 \
  --output /tmp/attncp_decode_collectives_current_8k.json

NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
CUDA_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nproc_per_node=2 \
  benchmark/kernels/attention/bench_attncp_decode_paths.py \
  --kv-lens 8192 --batch-size 4 --warmup 5 --iters 30 --trials 3 \
  --output /tmp/attncp_decode_paths_current_8k.json

NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
CUDA_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nproc_per_node=2 \
  benchmark/kernels/attention/bench_attncp_decode_paths.py \
  --kv-lens 8192 --batch-size 4 --warmup 5 --iters 30 --trials 3 \
  --cuda-graph \
  --output /tmp/attncp_decode_paths_current_8k_cudagraph.json
```

Model shape:

- WeLM debug model has 48 layers, 24 Q heads, 2 KV heads, head dim 256.
- Attention sink is enabled layer-wise.

Results:

- Component probe:
  - `dense_tp_fa_after_reconstruct`: `49.708 us`;
  - `dense_local_materialize_before_allreduce`: `50.690 us`;
  - `target_local_fullq_fa_with_lse`: `30.135 us`;
  - `target_merge_state_chain`: `6.711 us`.
- Collective probe:
  - `target_q_allgather`: `18.510 us`;
  - `target_o_lse_allgather_pair`: `35.487 us`;
  - `target_o_lse_alltoall_slice_pair`: `37.648 us`;
  - `current_dense_kv_allreduce_pair`: `153.670 us`.
- Full path probe without CUDA graph:
  - `current_dense`: `259.126 us`;
  - `target_sharded_fullq`: `254.478 us`;
  - `target_sharded_slice_a2a`: `269.549 us`;
  - `target_sharded_qloop`: `351.525 us`.
- Full path probe with CUDA graph:
  - `current_dense`: `256.765 us`;
  - `target_sharded_fullq`: `61.617 us`;
  - `target_sharded_slice_a2a`: `67.777 us`;
  - `target_sharded_qloop`: `100.288 us`.
- Numeric diff vs dense reference in path probe:
  - max abs diff `0.000244` for full-Q, slice-a2a, and qloop paths.

Interpretation:

- The local-merge decode math is numerically stable at the one-layer level.
- Under CUDA graph, the current full-Q local-merge shape is the best tested
  decode variant. Slice/all-to-all and qloop are slower, so simply changing the
  O/LSE exchange primitive is not an obvious win.
- Compared with TP4's local attention component (`~49.7 us`), the sharded-KV
  graph replay path (`~61.6 us`) carries roughly `12 us/layer` extra attention
  cost at 8k/batch4. With 48 layers this explains about `0.6 ms/token` of ITL
  gap before service-level scheduler and non-attention effects.
- The remaining service ITL gap should be profiled at serving level. Blindly
  adding another decode communication variant is unlikely to close it.

## Experiment 53: decode sink A/B probe

Date: 2026-06-24

Motivation:

- The debug WeLM model enables attention sink on all 48 layers.
- In sharded-KV local merge, sink is a per-Q-head softmax denominator term and
  has to be included on exactly one CP rank before LSE merge. Check whether
  attention sink handling is the dominant ITL overhead.

Command:

```bash
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
CUDA_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nproc_per_node=2 \
  benchmark/kernels/attention/bench_attncp_decode_paths.py \
  --kv-lens 8192 --batch-size 4 --warmup 5 --iters 30 --trials 3 \
  --cuda-graph --disable-sinks \
  --output /tmp/attncp_decode_paths_current_8k_cudagraph_no_sinks.json
```

Results:

- With sinks enabled:
  - `target_sharded_fullq`: `61.617 us`;
  - `target_sharded_slice_a2a`: `67.777 us`;
  - `target_sharded_qloop`: `100.288 us`.
- With sinks disabled:
  - `target_sharded_fullq`: `55.530 us`;
  - `target_sharded_slice_a2a`: `61.968 us`;
  - `target_sharded_qloop`: `92.618 us`.

Interpretation:

- Attention sink contributes about `6 us/layer` in this one-layer CUDA graph
  probe, or roughly `0.3 ms/token` for 48 layers.
- This is a real cost, but not enough to explain the full service ITL gap by
  itself.
- The existing path already has a gathered-full-sink cache. CUDA graph capture
  runs two eager warmup forwards before capture, so the sink cache should be
  populated before graph recording in the normal path.
- Therefore this is probably FA sink-term cost, not a repeated sink all-gather
  bug. Do not optimize sink handling unless a service-level trace contradicts
  this.

## Experiment 54: service-level torch profiler A/B

Date: 2026-06-24

Motivation:

- Microbench results explain part of the ITL gap, but not the whole service
  difference.
- Use SGLang's existing `/start_profile` path through `bench_serving --profile`
  instead of adding custom timers.

Commands:

- Started TP4+CP2 and TP4 servers on the current source with:
  - `--tp 4`;
  - `--page-size 1`;
  - `--chunked-prefill-size 1024`;
  - FA3 prefill/decode;
  - `--enable-over-encoding`;
  - `--enable-welm-kv-mirror-opt`;
  - `--cuda-graph-max-bs 16`;
  - `--disable-piecewise-cuda-graph`;
  - CP2 additionally used `--attn-cp-size 2 --attn-cp-mode sharded-kv
    --attn-cp-kv-chunk-size 1024
    --attn-cp-decode-cuda-graph-max-seq-len 8704`.
- Bench/profile workload:
  - random input len `8192`;
  - output len `16`;
  - prompts `8`;
  - concurrency `4`;
  - `--profile --profile-by-stage --profile-num-steps 1`.

Artifacts:

- CP2:
  `/tmp/welmv4_attncp_profile_service/20260624_074814_cp2_short`
- TP4:
  `/tmp/welmv4_attncp_profile_service/20260624_075122_tp4_short`

Notes:

- The benchmark latency numbers in this experiment are profiler-contaminated and
  should not be used as normal TTFT/ITL results.
- The trace files are still useful to compare kernel categories.

Decode trace median across TP ranks:

| category | CP2 | TP4 | CP2 - TP4 |
|---|---:|---:|---:|
| GPU step annotation | `18.927 ms` | `14.402 ms` | `+4.525 ms` |
| CPU step annotation | `9.487 ms` | `5.303 ms` | `+4.184 ms` |
| cudaGraphLaunch CPU | `3.467 ms` | `1.970 ms` | `+1.496 ms` |
| NCCL kernel | `0.671 ms` | `0.015 ms` | `+0.656 ms` |
| flash-attn kernels | `1.347 ms` | `1.741 ms` | `-0.395 ms` |
| attention LSE merge kernel | `0.076 ms` | `0.000 ms` | `+0.076 ms` |
| TP custom all-reduce kernels | `3.813 ms` | `2.109 ms` | `+1.704 ms` |
| MoE kernels | `1.267 ms` | `1.239 ms` | `+0.028 ms` |
| router kernels | `1.172 ms` | `1.169 ms` | `+0.003 ms` |
| GEMM kernels | `1.287 ms` | `1.286 ms` | `+0.001 ms` |
| other kernels | `3.115 ms` | `2.000 ms` | `+1.115 ms` |
| CPU ops | `4.363 ms` | `1.682 ms` | `+2.681 ms` |
| other CUDA runtime | `9.129 ms` | `7.134 ms` | `+1.995 ms` |

Cross-device reduce detail:

- Both CP2 and TP4 traces show `98` custom all-reduce kernel events on TP0.
- CP2 does not add another all-reduce call count in this trace; the same
  full-TP custom all-reduce kernels are slower in the profiled decode step.
- This may be partly first-profiled-step noise, so it needs a later-start
  profile before changing code.

Source check:

- In `welmv4.py`, when `is_cp_kv_sharded()` is true, QKV and `o_proj` are
  intentionally configured with the full tensor parallel rank/size:
  `get_tensor_model_parallel_rank()` and
  `get_tensor_model_parallel_world_size()`.
- This matches the design document: sharded-KV CP only exists inside attention
  KV residency/read/merge; projection, `o_proj`, MLP and MoE remain normal TP.
- Therefore `o_proj` / layer communicator full-TP all-reduce is semantically
  intentional and cannot simply be changed to `attn_tp_group`.

Interpretation:

- The decode gap is not dominated by the sharded attention NCCL all-gather
  alone. The trace shows additional cost in CUDA graph launch/runtime, custom
  full-TP all-reduce duration, and miscellaneous kernels/CPU ops.
- Do not optimize by blindly swapping Q/O/LSE collective primitives.
- A useful next trace should start after several decode steps instead of
  profiling the first decode step, then compare whether the full-TP custom
  all-reduce delta persists. If it persists, the optimization direction is
  likely around fusing or avoiding the extra synchronization around full-TP
  reductions in decode, not around the FA partial itself.

Follow-up attempt:

- Artifact:
  `/tmp/welmv4_attncp_profile_service/20260624_075554_cp2_late_decode`
- Command shape:
  - CP2;
  - random input len `8192`;
  - output len `64`;
  - prompts `4`;
  - concurrency `4`;
  - `--profile --profile-start-step 50 --profile-steps 2`.
- Result:
  - Trace was generated, but it captured two EXTEND steps, not DECODE.
  - Server log showed prefill was still running when profiling started.
- Decision:
  - Do not use this artifact for decode optimization conclusions.
  - If later-start decode profiling is needed, use a higher start step or a
    manual profile trigger after confirming the server has entered decode.

## Experiment 55: manual decode profile and q-layout prototype

Date: 2026-06-24

Motivation:

- `profile_start_step=50` still captured EXTEND. Use a manual trigger instead:
  start the profiler only after the server log reports `Decode batch`.
- Then test one concrete micro-optimization idea before touching the service
  path: avoid reshaping gathered Q into `[batch, full_q_heads, head_dim]` by
  treating gathered Q shards as `cp_size * batch` decode rows.

Manual profile commands:

- Start TP4+CP2 and TP4 servers with the same benchmark flags as previous
  service profiles.
- Run `bench_serving` with:
  - random input len `8192`;
  - output len `128`;
  - prompts `4`;
  - concurrency `4`;
  - no `bench_serving --profile`.
- A shell monitor watches `server.log`; after the first `Decode batch` line it
  calls:

```bash
curl -fsS --noproxy '*' -X POST "http://${HOST}:${PORT}/start_profile" \
  -H 'Content-Type: application/json' \
  -d '{"output_dir":".../profile","activities":["CPU","GPU"],"with_stack":false,"record_shapes":false,"profile_prefix":"..."}'
sleep 3
curl -fsS --noproxy '*' -X POST "http://${HOST}:${PORT}/stop_profile"
```

Artifacts:

- CP2:
  `/tmp/welmv4_attncp_profile_service/20260624_075944_cp2_manual_decode`
- TP4:
  `/tmp/welmv4_attncp_profile_service/20260624_080215_tp4_manual_decode`

Benchmark sanity:

- CP2:
  - mean TTFT `1734.681 ms`;
  - mean ITL `18.939 ms`;
  - median ITL `10.495 ms`;
  - p95 ITL `12.156 ms`.
- TP4:
  - mean TTFT `1522.862 ms`;
  - mean ITL `12.045 ms`;
  - median ITL `8.352 ms`;
  - p95 ITL `8.984 ms`.
- These are still profiler-window runs, but the median/p95 ITL values are close
  enough to normal behavior to use the trace structure.

Normalized decode trace, median per captured decode step:

| category | CP2 ms/step | TP4 ms/step | delta | CP2 calls/step | TP4 calls/step |
|---|---:|---:|---:|---:|---:|
| CPU step annotation | `3.220` | `1.826` | `+1.394` | `1.00` | `1.00` |
| cudaGraphLaunch CPU | `2.537` | `1.447` | `+1.090` | `1.00` | `1.00` |
| NCCL kernel | `0.664` | `0.014` | `+0.650` | `97.00` | `1.00` |
| flash-attn kernel | `1.358` | `1.745` | `-0.387` | `96.00` | `96.00` |
| attention LSE merge kernel | `0.076` | `0.000` | `+0.076` | `48.00` | `0.00` |
| TP custom all-reduce kernel | `0.568` | `0.547` | `+0.020` | `98.00` | `98.00` |
| fused MoE kernel | `1.250` | `1.238` | `+0.011` | `96.00` | `96.00` |
| router kernel | `1.180` | `1.175` | `+0.005` | `48.00` | `48.00` |
| GEMM kernel | `1.288` | `1.288` | `+0.000` | `290.00` | `290.00` |
| other kernel | `3.097` | `1.998` | `+1.100` | `1517.91` | `877.05` |
| CPU op | `5.700` | `7.399` | `-1.699` | `2811.28` | `4178.32` |
| other CUDA runtime | `6.384` | `5.699` | `+0.685` | `53.92` | `34.14` |

Interpretation:

- The stable decode trace confirms the main AttnCP-specific attention cost:
  CP2 launches about `97` NCCL kernels per decode step, approximately
  `2 * num_layers + 1`, matching one Q all-gather and one O/LSE all-gather per
  layer plus a small extra collective.
- TP custom all-reduce is not a persistent normalized gap in this trace:
  `+0.020 ms/step`, not the `+1.7 ms` seen in the first-decode-step trace.
- FA itself is faster under CP2 (`-0.387 ms/step`) because each rank attends to
  half the KV. The win is outweighed by per-layer NCCL plus graph/runtime/copy
  overhead.
- The optimization target is now clearer: reduce or overlap per-layer CP
  collectives and the associated copy/merge kernels. Full-TP projection and
  MoE all-reduce are not the first target.

Q layout prototype:

- Added a microbench-only path to
  `benchmark/kernels/attention/bench_attncp_decode_paths.py`:
  `target_sharded_batched_qshard`.
- Instead of materializing full-Q layout `[B, C * H, D]`, it runs FA once on
  gathered Q as `[C * B, H, D]` with repeated page-table metadata.
- Attention sink is represented as a separate softmax state partial
  (`O=0`, `LSE=sink`) and merged with `merge_state_v2`, so correctness still
  covers sink-enabled WeLM.

Validation command:

```bash
python -m py_compile benchmark/kernels/attention/bench_attncp_decode_paths.py

NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
CUDA_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nproc_per_node=2 \
  benchmark/kernels/attention/bench_attncp_decode_paths.py \
  --kv-lens 8192 --batch-size 4 --warmup 5 --iters 30 --trials 3 \
  --cuda-graph \
  --output /tmp/attncp_decode_paths_batched_qshard_8k_cudagraph.json
```

Result:

| path | median |
|---|---:|
| `current_dense` | `255.708 us` |
| `target_sharded_fullq` | `61.468 us` |
| `target_sharded_batched_qshard` | `80.268 us` |
| `target_sharded_slice_a2a` | `68.802 us` |
| `target_sharded_qloop` | `103.127 us` |

- `target_sharded_batched_qshard` max diff vs dense: `0.000244`.
- It is numerically valid but slower than the current full-Q local-merge path.

Decision:

- Do not implement `batched_qshard` in the serving path.
- The current full-Q local-merge remains the best tested decode variant.
- Further ITL work should focus on fused/overlapped CP communication, not
  another Q layout variant.

## 2026-06-24 - Experiment 56: no-blind-change decode bottleneck audit

Goal:

- Re-check the current implementation and the training overlap reference before
  making another serving-path optimization.
- Avoid changing the attention path without evidence that the change targets the
  measured gap.

Code-path audit:

- Current decode sharded-KV path is still:
  1. gather Q inside `sharded_kv_cp_group`;
  2. run FA over local KV shard with full gathered Q heads;
  3. gather local partial O/LSE across CP ranks;
  4. merge softmax states and slice the local TP head shard.
- Current prefill sharded-KV path is still correctness-first dense gather:
  `_forward_extend_sharded_kv()` calls `_flash_attn_sharded_kv_dense()`, which
  reconstructs temporary dense KV before FA.
- This means decode really uses sharded KV, but prefill is not yet the ideal
  segment-loop design from the doc.

Training overlap reference:

- The training code under
  `/home/fhkong/wxwork/mimikyu/mmq/mmq/modules/block_v2/memory_optimizer/qkv_proj_and_post_processing`
  mainly overlaps Ulysses all-to-all by splitting communication into
  `pre_all2all()`, `all2all()`, and `post_all2all()` waiter phases.
- It starts Q/K/V communication early and hides it behind later projection,
  gate, and norm/rope work.
- This pattern is not directly portable to serving decode because the AttnCP
  communication is inside each attention layer:
  - Q all-gather is required before local FA;
  - O/LSE communication and softmax-state merge are required before `o_proj`;
  - MLP/router work is after attention and cannot hide the same layer's Q
    all-gather.

Existing trace re-analysis:

- Artifact:
  `/tmp/welmv4_attncp_profile_service/20260624_075944_cp2_manual_decode/profile/cp2_manual_decode-1782259282.9819431-TP-0.trace.json.gz`
- CP2 trace has 64 captured decode steps and 48 layers.
- NCCL kernel breakdown:

| NCCL grid | calls | avg |
|---|---:|---:|
| `(3, 1, 1)` | `3072` | `6.245 us` |
| `(7, 1, 1)` | `3072` | `7.255 us` |
| `(24, 1, 1)` | `64` | `14.356 us` |

- `3072 = 64 steps * 48 layers`, so the per-layer pattern is exactly two CP
  collectives: one Q all-gather and one O/LSE gather/coalesced collective.
- The extra `64` NCCL kernels are one per decode step and are not the per-layer
  AttnCP bottleneck.

Potential overlap audit:

- `gate_proj(hidden_states)` in WeLM attention depends only on pre-attention
  hidden states, so it is theoretically movable before attention.
- However, the trace shows the large `mmq_style_router_linear_kernel`
  (`~24 us/layer/step`) is the MLP router, which depends on the post-attention
  hidden state and cannot hide the same layer's Q all-gather.
- The attention headwise gate projection is much smaller and is unlikely to
  hide the full CP communication cost.
- Therefore, moving gate computation or adding async Q gather should not be the
  first serving-path change unless a targeted prototype proves otherwise.

Current best next target:

- The design doc's ideal decode communication is closer to `cp_lse_ag_out_rs`:
  communicate only the destination head slice, merge LSE/state, and return the
  local TP shard.
- Current implementation gathers full O/LSE to every CP rank and then slices
  local heads. This is correct and simple, but it pays extra communication and
  copy/merge overhead.
- Existing Python-level `all_to_all_single` slice prototype was slower than the
  full-Q path, so the next useful optimization likely needs a lower-level
  fused/direct communication path:
  - direct CP send/recv or all-to-all of contiguous head-slice workspaces;
  - fused O/LSE pack plus merge for the local destination heads;
  - CUDA-graph-safe implementation.

Decision:

- Do not change the serving decode path in this step.
- Keep the current full-Q local-merge path as the correctness baseline.
- Next optimization should prototype a lower-level `cp_lse_ag_out_rs`-like
  output exchange/merge, not another Q layout variant or a model-level reorder.

## 2026-06-24 - Experiment 57: `cp_lse_ag_out_rs`-style reduce-scatter prototype

Goal:

- Test a decode communication path closer to the design doc's
  `cp_lse_ag_out_rs` target without changing serving code.
- Avoid full O all-gather by:
  1. all-gathering only LSE;
  2. computing global softmax LSE;
  3. scaling local partial O by `exp(local_lse - global_lse)`;
  4. reduce-scattering weighted O over CP head chunks.

Implementation:

- Added microbench-only path
  `target_sharded_lse_ag_o_rs` in
  `benchmark/kernels/attention/bench_attncp_decode_paths.py`.
- The path preallocates CUDA graph workspaces for:
  - global LSE;
  - softmax weights;
  - weighted O;
  - reduce-scatter send/output buffers.
- No serving-path logic was changed.

Validation command:

```bash
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
CUDA_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nproc_per_node=2 \
  benchmark/kernels/attention/bench_attncp_decode_paths.py \
  --kv-lens 8192 --batch-size 4 --warmup 5 --iters 30 --trials 3 \
  --cuda-graph \
  --output /tmp/attncp_decode_paths_lse_ag_o_rs_8k_cudagraph.json
```

Result:

| path | median | CUDA graph | max diff vs dense |
|---|---:|---:|---:|
| `current_dense` | `254.667 us` | yes | n/a |
| `target_sharded_fullq` | `61.675 us` | yes | `0.000244` |
| `target_sharded_lse_ag_o_rs` | `80.865 us` | yes | `0.000488` |
| `target_sharded_batched_qshard` | `82.591 us` | yes | `0.000244` |
| `target_sharded_slice_a2a` | `72.403 us` | yes | `0.000244` |
| `target_sharded_qloop` | `103.236 us` | yes | `0.000244` |

Interpretation:

- The reduce-scatter formulation is numerically valid at the expected bf16
  tolerance and CUDA-graph capturable.
- It is still slower than the current full-Q local-merge path in this
  Python/torch prototype.
- The likely reason is not the algorithmic communication volume alone; the
  unfused path adds:
  - `logsumexp`;
  - `sub/exp/mul` over full O;
  - head-chunk packing before reduce-scatter.
- Therefore a serving-path change to this exact PyTorch sequence would regress
  ITL.

Decision:

- Do not port this reduce-scatter prototype to serving as-is.
- A useful `cp_lse_ag_out_rs` optimization needs a lower-level fused
  implementation that combines LSE merge, O scaling/packing, and
  reduce-scatter/direct exchange.
- Until such a fused path exists, the current full-Q local-merge remains the
  best tested decode path.

## 2026-06-24 - Experiment 58: Triton-fused `cp_lse_ag_out_rs` prototype

Goal:

- Determine whether the slow `target_sharded_lse_ag_o_rs` result came from the
  algorithm or from unfused PyTorch elementwise/packing overhead.
- Keep the experiment microbench-only; do not change the serving path.

Component breakdown:

Command:

```bash
CUDA_VISIBLE_DEVICES=0 \
python benchmark/kernels/attention/bench_attncp_decode_components.py \
  --kv-lens 8192 --batch-size 4 --warmup 10 --iters 50 --trials 3 \
  --output /tmp/attncp_decode_components_lse_pack_triton_8k_c4.json
```

Result:

| component | median |
|---|---:|
| `target_local_fullq_fa_with_lse` | `29.781 us` |
| `target_merge_state_chain` | `6.212 us` |
| `target_lse_global_reduce_torch` | `33.146 us` |
| `target_lse_global_reduce_triton` | `8.079 us` |
| `target_o_scale_pack_torch` | `19.423 us` |
| `target_o_scale_pack_triton` | `8.863 us` |

Interpretation:

- The previous PyTorch `LSE allgather + O reduce_scatter` prototype was slow
  mostly because `torch.logsumexp` and unfused `sub/exp/mul/pack` were very
  expensive for this small decode shape.
- Simple Triton kernels reduce those two pieces from about `52.6 us` to about
  `16.9 us`, so a fused implementation is plausible.

Distributed path prototype:

- Added microbench-only path `target_sharded_lse_ag_o_rs_triton` in
  `benchmark/kernels/attention/bench_attncp_decode_paths.py`.
- It uses Triton kernels for:
  - global LSE reduction across gathered CP LSE;
  - O scaling and CP-head packing before reduce-scatter.
- It still uses regular NCCL collectives for Q all-gather, LSE all-gather, and
  O reduce-scatter.

8k / batch 4 command:

```bash
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
CUDA_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nproc_per_node=2 \
  benchmark/kernels/attention/bench_attncp_decode_paths.py \
  --kv-lens 8192 --batch-size 4 --warmup 5 --iters 30 --trials 3 \
  --cuda-graph \
  --output /tmp/attncp_decode_paths_lse_ag_o_rs_triton_8k_cudagraph.json
```

8k / batch 4 result:

| path | median | max diff vs dense |
|---|---:|---:|
| `target_sharded_fullq` | `61.632 us` | `0.000244` |
| `target_sharded_lse_ag_o_rs` | `80.895 us` | `0.000488` |
| `target_sharded_lse_ag_o_rs_triton` | `62.176 us` | `0.000488` |
| `target_sharded_slice_a2a` | `71.814 us` | `0.000244` |

8k / batch 16 result:

| path | median | max diff vs dense |
|---|---:|---:|
| `target_sharded_fullq` | `159.797 us` | `0.000488` |
| `target_sharded_lse_ag_o_rs` | `185.843 us` | `0.000488` |
| `target_sharded_lse_ag_o_rs_triton` | `163.205 us` | `0.000488` |
| `target_sharded_slice_a2a` | `174.149 us` | `0.000488` |

Decision:

- The fused Triton prototype is much better than the PyTorch reduce-scatter
  version, but still does not beat the current full-Q local-merge path.
- Do not port this path to serving now.
- To make `cp_lse_ag_out_rs` worthwhile, the communication and merge need to be
  fused more deeply than "Triton pack + NCCL reduce-scatter"; likely a custom
  CUDA/NCCL or mscclpp-style path that avoids an extra collective/kernel
  boundary.
- For near-term optimization, keep decode on current full-Q local-merge and
  shift attention back to TTFT/prefill dense-gather overhead, where the service
  profile still shows a clear gap.

## 2026-06-24 - Experiment 59: distributed prefill path prototypes

Goal:

- Re-check TTFT/prefill before changing serving code.
- Compare the current correctness-first prefill path against two plausible
  alternatives in a 2-GPU CP microbench:
  1. `current_dense_reconstruct`: owned-KV dense reconstruction with K/V
     all-reduce, then dense FA on local TP Q heads;
  2. `current_dense_compact_ag`: compact owned-KV all-gather + scatter
     reconstruction, then the same dense FA;
  3. `target_segment_local_merge`: Q all-gather, owner-segment FA, O/LSE
     all-gather, softmax-state merge, and local head slice.

Implementation:

- Added microbench-only script
  `benchmark/kernels/attention/bench_attncp_prefill_paths.py`.
- The compact all-gather prototype initially used advanced-index `copy_`, which
  does not write back to the base tensor. Fixed it with `index_copy_`.
- After the fix, compact reconstruction matches dense reconstruction exactly:
  `compact_ag_k_max_abs_diff=0`, `compact_ag_v_max_abs_diff=0`,
  `compact_ag_max_abs_diff=0`.

Command:

```bash
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
CUDA_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nproc_per_node=2 \
  benchmark/kernels/attention/bench_attncp_prefill_paths.py \
  --q-starts 0,1024,4096,7168 --q-len 1024 \
  --warmup 1 --iters 2 --trials 3 \
  --output /tmp/attncp_prefill_paths_compact_ag_fixed_8k_q1024.json
```

Result:

| q_start | total_len | current dense | compact AG dense | segment local merge |
|---:|---:|---:|---:|---:|
| `0` | `1024` | `276.000 us` | `329.328 us` | `456.944 us` |
| `1024` | `2048` | `286.720 us` | `378.944 us` | `465.904 us` |
| `4096` | `5120` | `535.488 us` | `583.280 us` | `930.160 us` |
| `7168` | `8192` | `770.176 us` | `824.080 us` | `1179.472 us` |

Interpretation:

- The segment prefill algorithm is mathematically viable, but for 1024-token
  chunked prefill it is slower because it breaks attention into multiple FA
  launches and merge steps. This matches Experiment 51's single-process result.
- Compact all-gather reconstruction preserves dense FA semantics, but the
  `all_gather + index_copy_ scatter` prototype is slower than the current dense
  all-reduce reconstruction at these shapes.
- Current dense reconstruction remains the fastest tested prefill path, despite
  being conceptually wasteful.

Decision:

- Do not replace prefill with segment-local-merge in serving.
- Do not replace dense K/V all-reduce with compact all-gather + scatter as-is.
- A useful prefill optimization likely needs a fused dense reconstruction
  kernel/collective, or a FA backend that can consume owner-segmented KV without
  many small FA launches. Pure Python/Torch composition is not enough.

## 2026-06-24 - Experiment 60: default workspace cleanup

Goal:

- Check whether the current serving implementation has obvious default-path
  waste after the decode/prefill performance experiments.
- Keep any edit scoped and semantics-preserving.

Observation:

- `_init_attn_cp_local_merge_cuda_graph_state()` allocated qloop-only buffers
  even when the default and fastest tested decode mode is `fullq`:
  - `q_shards`;
  - `qloop_o_shards`;
  - `qloop_lse_shards`.
- These tensors are only read when
  `SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE_MODE=qloop`.
- Default mode `fullq` passes `copy_q_shards=False` and does not enter the
  qloop branch.

Change:

- Lazily allocate the qloop-only buffers only when
  `self.attn_cp_decode_local_merge_mode == "qloop"`.
- No attention math or communication path changed for the default `fullq` mode.

Validation:

```bash
python -m py_compile python/sglang/srt/layers/attention/flashattention_backend.py
```

Result:

- Syntax check passed.
- Estimated default WeLM TP4+CP2, `cuda_graph_max_bs=16` workspace reduction is
  only about `0.09 MiB`, so this should be treated as cleanup, not a meaningful
  TTFT/ITL optimization.

Decision:

- Keep the cleanup unless code review prefers zero service-path edits during
  performance investigation.
- Do not claim performance improvement from this change.

## 2026-06-24 - Experiment 61: no-blind-optimization status audit

Goal:

- Stop adding runtime variants without evidence.
- Reconcile the design document with the current implementation and measured
  bottleneck before choosing the next optimization.

Current implementation summary:

- Persistent KV cache is sharded by CP owner and still provides the intended
  resident KV memory saving.
- Decode now defaults to the full-Q local-merge path:
  Q head all-gather, local FA3 over local owned KV, O/LSE all-gather, local
  `merge_state_v2`, and local TP-head slice output.
- Prefill/chunked prefill remains on dense reconstruction because the tested
  segment-local-merge and compact-all-gather prototypes were slower at the
  current 1024-token chunked prefill shapes.
- The current correctness claim is token-level alignment, not strict logprob or
  bitwise equivalence.

Evidence to carry forward:

| scenario | TP4 | TP4+CP2 | gap |
|---|---:|---:|---:|
| 8k input / 512 output / c4 TTFT | `1634.07 ms` | `1861.08 ms` | `+13.9%` |
| 8k input / 512 output / c4 ITL | `8.76 ms` | `10.86 ms` | `+24.0%` |
| output throughput | `335.51 tok/s` | `276.66 tok/s` | `-17.5%` |

Service-level decode profile:

- CP2 has about `97` NCCL kernels per decode step, matching
  `2 * num_layers + 1`.
- CP2 NCCL time is about `0.664 ms/step`; TP4 is about `0.014 ms/step`.
- FA itself is faster under sharded KV, but per-layer CP collectives and
  merge/copy overhead dominate the remaining gap.

Audit result:

- Updated `docs/ring-attn/design-sharded-kv-cp.md` so it no longer describes
  the current code as only a dense correctness path.
- Document now states:
  - decode full-Q local-merge is the default measured path;
  - prefill is still dense correctness path;
  - strict logprob is not claimed;
  - q-loop, Python reduce-scatter, compact all-gather, and prefill segment loop
    are not recommended next steps.
- No new runtime optimization was made in this step.

Next guarded optimization direction:

- Decode: only pursue a lower-level fused/graph-safe LSE merge + O exchange
  primitive if a one-layer microbench beats current full-Q local-merge.
- Prefill/TTFT: only pursue a fused reconstruction or FA backend that can
  consume owner-segmented KV efficiently; the Python segment-loop path is not
  worth porting.

## 2026-06-24 - Experiment 62: remove q-loop runtime branch

Goal:

- Reduce serving-path complexity without changing the default full-Q local-merge
  decode semantics.
- Remove a runtime variant that has already been measured slower than the
  current default path.

Evidence:

- q-loop was only reachable through the private
  `SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE_MODE=qloop` environment
  variable.
- It was not used by the public AttnCP CLI contract:
  `--attn-cp-size`, `--attn-cp-mode`, `--attn-cp-kv-chunk-size`.
- Previous benchmark evidence showed q-loop was slower than full-Q local-merge
  in the service path and in the one-layer CUDA graph microbench.
- The benchmark-only q-loop implementation remains in
  `benchmark/kernels/attention/bench_attncp_decode_paths.py` for historical
  comparison.

Change:

- Removed `SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE_MODE` handling from
  `flashattention_backend.py`.
- Removed q-loop-only CUDA graph workspace buffers:
  `q_shards`, `qloop_o_shards`, and `qloop_lse_shards`.
- Removed the q-loop FA loop from `_flash_attn_sharded_kv_local_merge_workspace`.
- Simplified Q all-gather by removing the q-loop-only Q shard copy option.

Validation:

```bash
rg -n "attn_cp_decode_local_merge_mode|EXPERIMENTAL_DECODE_LOCAL_MERGE_MODE|qloop|q_shards|qloop_o_shards|qloop_lse_shards|copy_q_shards" \
  python/sglang/srt/layers/attention/flashattention_backend.py || true

python -m py_compile \
  python/sglang/srt/layers/attention/flashattention_backend.py
```

Result:

- No runtime q-loop references remain in `flashattention_backend.py`.
- Syntax check passed.

Decision:

- Keep the cleanup.
- Do not claim a material TTFT/ITL improvement from this change; it is a review
  and maintenance cleanup that preserves the current default full-Q local-merge
  path.

## 2026-06-24 - Experiment 63: avoid local-merge accumulator copy-back

Goal:

- Reduce per-layer default full-Q local-merge overhead without changing
  communication, merge order, or attention semantics.

Observation:

- `_attncp_merge_local_head_slice()` copied gathered shard 0 into
  `merge_current_*`, copied the next shard into `merge_next_*`, merged into
  `merge_tmp_*`, and then copied `merge_tmp_*` back into `merge_current_*`.
- For CP=2 this final copy-back is unnecessary. For larger CP sizes, the
  accumulator can alternate between `merge_current_*` and `merge_tmp_*`.
- The input slices from `o_gather`/`lse_gather` still need to be copied into
  contiguous workspace because `sgl_kernel.merge_state_v2` requires contiguous
  inputs.

Change:

- Keep the existing contiguous input copies.
- After each `merge_state_v2`, switch the accumulator reference to the output
  buffer instead of copying it back to `merge_current_*`.
- Return the final accumulator buffer.

Validation:

```bash
python -m py_compile \
  python/sglang/srt/layers/attention/flashattention_backend.py

git diff --check -- \
  python/sglang/srt/layers/attention/flashattention_backend.py
```

Microbench:

- Shape: CP=2, batch=4, local Q heads=6, full Q heads=12, head_dim=256,
  dtype=bf16.
- Operation: service-style local head slice merge, including the necessary
  contiguous copies from gathered O/LSE.

| path | median |
|---|---:|
| old copy-back chain | `43.199 us` |
| alternating accumulator | `34.144 us` |

Correctness:

- `max_o_diff=0.0`
- `max_lse_diff=0.0`

Decision:

- Keep the cleanup.
- This is still a small per-layer optimization. The remaining major ITL gap is
  dominated by per-layer CP collectives and cannot be closed by copy cleanup
  alone.

## 2026-06-24 - Experiment 64: precision regression after merge cleanup

Goal:

- Verify that Experiment 62 and Experiment 63 did not break TP4 vs TP4+CP2
  AttnCP token-level precision.

Command:

```bash
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  /home/fhkong/wxwork/attncp_precision_regression/run_full_precision.sh
```

Artifact:

- `/tmp/welmv4_attncp_precision/20260624_085359`

Result:

- TP4 baseline generated successfully:
  - MMLU/C-Eval baseline samples: `100`
  - baseline errors: `0`
- Controlled TP4 vs TP4+CP2:
  - output ids/text match for all `4/4` controlled requests;
  - strict logprob compare still fails with the known local-merge numerical
    difference:
    - `max_logprob_diff=0.8751058578491211`
    - `mean_logprob_diff=0.0006246352326386641`
    - `issue_count=280`
  - the failure is logprob-only; no output token drift was observed.
- Because the strict controlled compare exits nonzero, the script did not reach
  MMLU/C-Eval test in the first run. Reused the generated TP4 baseline and
  restarted only TP4+CP2 to run:

```bash
/home/fhkong/wxwork/perf_optimize_scripts/regression_test.py test \
  --server-url http://127.0.0.1:18192 \
  --model welmv4 \
  --baseline-path /tmp/welmv4_attncp_precision/20260624_085359/tp4_regression_baseline.pkl \
  --tolerance 1e-5
```

MMLU/C-Eval TP4 vs TP4+CP2 result:

- Samples tested: `100 / 100`
- Errors: `0`
- Token mismatches: `0 / 100 PASS`
- Max logprob diff: `0.00e+00 PASS`
- Mean logprob diff: `0.00e+00 PASS`
- Result: `PASS`

Decision:

- Keep the runtime cleanup.
- The full precision script still needs a token-level mode for local-merge
  validation; its current strict controlled logprob gate is intentionally
  stronger than the task's accepted correctness criterion.

## 2026-06-24 - Experiment 65: align precision script with token-level criterion

Goal:

- Make `/home/fhkong/wxwork/attncp_precision_regression/run_full_precision.sh`
  usable for the current AttnCP local-merge correctness criterion.
- Avoid failing the whole script on the known controlled strict-logprob diff
  when generated output tokens/text are stable.

Change:

- Added `--token-level-only` to
  `/home/fhkong/wxwork/attncp_precision_regression/controlled_generate.py compare`.
- In token-level mode, controlled compare still checks:
  - generated `text`;
  - generated `output_ids`;
  - `prompt_tokens`, `completion_tokens`, and `reasoning_tokens`.
- It skips strict comparison of:
  - chosen input/output token logprobs;
  - input/output top-logprobs.
- Updated `run_full_precision.sh` to use token-level controlled compare by
  default via `CONTROLLED_TOKEN_LEVEL_ONLY=1`.
- Updated the precision regression README to document this policy.

Validation:

```bash
python -m py_compile \
  /home/fhkong/wxwork/attncp_precision_regression/controlled_generate.py

bash -n \
  /home/fhkong/wxwork/attncp_precision_regression/run_full_precision.sh

NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  /home/fhkong/wxwork/sglang-perf-v4/.venv/bin/python \
  /home/fhkong/wxwork/attncp_precision_regression/controlled_generate.py compare \
  --baseline-json /tmp/welmv4_attncp_precision/20260624_085359/controlled_tp4.json \
  --candidate-json /tmp/welmv4_attncp_precision/20260624_085359/controlled_tp4_cp2.json \
  --output-dir /tmp/welmv4_attncp_precision/20260624_085359 \
  --tolerance 1e-5 \
  --top-logprobs 20 \
  --top-logprobs-min-overlap 18 \
  --token-level-only
```

Result:

- Controlled compare now reports:
  - `mode=token_level`
  - `passed=True`
  - `issues=0`
- Full single-entry precision script was rerun after the script change:
  - artifact: `/tmp/welmv4_attncp_precision/20260624_090330`
  - controlled compare: `mode=token_level`, `passed=True`, `issues=0`
  - MMLU/C-Eval samples tested: `100 / 100`
  - token mismatches: `0 / 100 PASS`
  - max logprob diff: `0.00e+00 PASS`
  - mean logprob diff: `0.00e+00 PASS`
  - final result: `PASS: TP4 vs TP4+CP2 sharded-KV precision regression passed.`

Decision:

- Keep this script change. It matches `task.md`: bitwise/logprob equality is not
  required, output token stability is the intended correctness gate.

## 2026-06-24 - Experiment 66: 8k/512/c4 benchmark after merge cleanup

Goal:

- Quantify the current end-to-end TTFT/ITL gap after removing q-loop runtime
  code and eliminating local-merge accumulator copy-back.

Command:

```bash
ALLOW_DIRTY=1 CASE_SET=decode512 \
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  /home/fhkong/wxwork/attncp_performance_benchmark/run_attncp_perf_benchmark.sh
```

Artifact:

- `/tmp/welmv4_attncp_perf/20260624_090840`

Scenario:

- Model:
  `/home/fhkong/models/80a3_v4d5_256k_merge_thinking_kimi_k25_0502_20260503_032335/epoch_003_step_0002610`
- 8k input / 512 output / concurrency 4 / 16 prompts
- FA3 prefill/decode
- CUDA graph max bs 16
- WeLM kv mirror enabled
- Over encoding enabled
- Chunked prefill size 1024

Result:

| config | output TPS | mean TTFT | mean ITL | peak GPU memory |
|---|---:|---:|---:|---:|
| TP4 | `336.6365 tok/s` | `1614.7824 ms` | `8.7562 ms` | `81665 MiB` |
| TP4+CP2 | `280.0401 tok/s` | `1835.6070 ms` | `10.7373 ms` | `63303 MiB` |

Gap:

- TTFT: `+13.7%`
- ITL: `+22.6%`
- Output throughput: `-16.8%`
- Peak memory: `-18.36 GiB/rank`

Interpretation:

- The accumulator cleanup is directionally consistent with a small ITL
  improvement, but the service-level gap remains dominated by per-layer CP
  collectives and merge/copy overhead.
- Current benchmark baseline is slightly better than the previous recorded
  `10.86 ms` CP2 ITL, but still not enough to make AttnCP performance-neutral.
- Continue avoiding Python-level communication variants unless a one-layer
  graph microbench beats current full-Q local-merge first.

## 2026-06-24 - Experiment 67: direct final merge into attention output buffer

Goal:

- Remove one more default decode local-merge copy when CUDA graph provides
  `forward_batch._attn_output` as the attention output buffer.

Observation:

- `forward_decode()` passes `_fa_out =
  forward_batch._attn_output.view(-1, local_q_heads, head_dim)` into the
  sharded-KV decode path.
- After Experiment 63, `_attncp_merge_local_head_slice()` returns the final
  merged local-head tensor, and the caller still copies it into `out` when
  `out is not None`.
- The final `merge_state_v2` can write directly into `out` if it is contiguous
  and has shape `(batch_size, local_q_heads, head_dim)`.

Change:

- `_attncp_merge_local_head_slice()` now accepts optional `final_o`.
- On the last CP merge step, it uses `final_o` as `v_merged` when shape and
  contiguity match.
- The caller keeps the old fallback copy when `final_o` cannot be used.

Validation:

```bash
python -m py_compile \
  python/sglang/srt/layers/attention/flashattention_backend.py

git diff --check -- \
  python/sglang/srt/layers/attention/flashattention_backend.py
```

Microbench:

- Shape: CP=2, batch=4, local Q heads=6, full Q heads=12, head_dim=256,
  dtype=bf16.
- Operation: service-style local head slice merge plus final copy into output
  buffer versus direct final merge into output buffer.

| path | median |
|---|---:|
| merge then copy to output | `39.039 us` |
| final merge writes output | `34.505 us` |

Correctness:

- `max_o_diff=0.0`
- Full single-entry precision regression:
  - artifact: `/tmp/welmv4_attncp_precision/20260624_091632`
  - controlled compare: `mode=token_level`, `passed=True`, `issues=0`
  - MMLU/C-Eval samples tested: `100 / 100`
  - token mismatches: `0 / 100 PASS`
  - max logprob diff: `0.00e+00 PASS`
  - mean logprob diff: `0.00e+00 PASS`
  - final result: `PASS: TP4 vs TP4+CP2 sharded-KV precision regression passed.`

Decision:

- Keep the guarded direct-output path.
- This is another small copy cleanup. It does not change the conclusion that
  the remaining service ITL gap is mainly collective launch/communication cost.

End-to-end benchmark after this change:

```bash
ALLOW_DIRTY=1 CASE_SET=decode512 \
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  /home/fhkong/wxwork/attncp_performance_benchmark/run_attncp_perf_benchmark.sh
```

Artifact:

- `/tmp/welmv4_attncp_perf/20260624_092117`

| config | output TPS | mean TTFT | mean ITL | peak GPU memory |
|---|---:|---:|---:|---:|
| TP4 | `333.6666 tok/s` | `1666.7093 ms` | `8.7609 ms` | `81665 MiB` |
| TP4+CP2 | `280.7516 tok/s` | `1828.6179 ms` | `10.7143 ms` | `63303 MiB` |

Gap:

- TTFT: `+9.7%`
- ITL: `+22.3%`
- Output throughput: `-15.9%`
- Peak memory: `-18.36 GiB/rank`

Interpretation:

- The latest direct-output cleanup moved CP2 ITL slightly from the previous
  `10.7373 ms` to `10.7143 ms`, which is directionally correct but small.
- The remaining optimization target is still the per-layer CP collective and
  merge boundary, not another local Python copy cleanup.

## 2026-06-24 - Experiment 68: decode bottleneck attribution, no runtime change

Goal:

- Stop making blind runtime changes and re-check where the remaining
  TP4-vs-TP4+CP2 decode ITL gap comes from.

Inputs:

- Latest end-to-end 8k input / 512 output / concurrency 4 benchmark:
  `/tmp/welmv4_attncp_perf/20260624_092117/summary.csv`.
- Decode component microbench:
  `/tmp/attncp_decode_components_20260624_092827.json`.
- Decode CP collective microbench:
  `/tmp/attncp_decode_collectives_20260624_092846.json`.
- Decode path microbench with CUDA graph:
  `/tmp/attncp_decode_paths_20260624_092914.json`.
- Decode path microbench with CUDA graph and attention sink disabled:
  `/tmp/attncp_decode_paths_nosink_20260624_093052.json`.

End-to-end baseline:

| config | mean TTFT | mean ITL | output TPS | peak memory |
|---|---:|---:|---:|---:|
| TP4 | `1666.7093 ms` | `8.7609 ms` | `333.6666 tok/s` | `81665 MiB` |
| TP4+CP2 | `1828.6179 ms` | `10.7143 ms` | `280.7516 tok/s` | `63303 MiB` |

Gap:

- TTFT: `+9.7%`
- ITL: `+22.3%`
- Output throughput: `-15.9%`
- Memory: `-18.36 GiB/rank`

Model shape used for the microbench:

- `num_hidden_layers=48`
- `num_attention_heads=24`
- `num_key_value_heads=2`
- `head_dim=256`
- `dtype=bf16`

Component results at `batch=4`, `kv_len=8704`, `TP4+CP2`:

| component | median |
|---|---:|
| TP-style dense FA after reconstruction | `55.538 us` |
| AttnCP local full-Q FA with LSE | `35.522 us` |
| local merge_state chain | `6.782 us` |
| Q allgather | `17.941 us` |
| O allgather | `18.336 us` |
| LSE allgather | `17.960 us` |
| O+LSE allgather pair | `36.127 us` |
| sink allgather | `19.533 us` |

CUDA-graph path results at `batch=4`, `kv_len=8704`, `TP4+CP2`:

| path | median |
|---|---:|
| current dense reconstruct path | `271.005 us` |
| target sharded full-Q local-merge | `65.543 us` |
| target LSE allgather + O reduce-scatter prototype | `85.071 us` |
| target LSE allgather + O reduce-scatter triton prototype | `66.044 us` |
| target batched-qshard prototype | `91.959 us` |
| target slice all-to-all prototype | `75.494 us` |
| target qloop prototype | `112.690 us` |

Attention sink check:

| path | with sink | sink disabled | delta |
|---|---:|---:|---:|
| target sharded full-Q local-merge | `65.543 us` | `60.213 us` | `5.330 us/layer` |

Interpretation:

- Sharded-KV local FA compute is not the bottleneck. It is faster than the
  TP-style dense FA microbench because each CP rank attends only its owned KV
  shard.
- The service ITL gap is `10.7143 - 8.7609 = 1.9534 ms/token`. With 48
  layers this is about `40.7 us/layer`, which matches the launch/communication
  scale of the required Q and O/LSE CP collectives.
- Existing experimental alternatives are not better than the current default:
  qloop, batched-qshard, and slice all-to-all are all slower in the path
  microbench. Do not switch runtime to those paths.
- Attention sink has visible cost but is not the dominant source. Disabling it
  saves about `5.3 us/layer` in the path microbench, roughly `0.25 ms/token`
  across 48 layers. This is worth a targeted follow-up only after verifying
  whether sink allgather is still captured/replayed per decode step in the real
  CUDA graph.

Decision:

- No runtime change in this experiment.
- The next optimization should target the collective/merge boundary with
  stronger profiling evidence, not more Python-level copy cleanup.
- Candidate follow-ups:
  1. Use a service-level trace or lightweight counters to confirm per-token
     NCCL launch counts in CUDA graph replay.
  2. Verify whether attention sink allgather is captured per layer; if yes,
     pre-gather full sinks before graph capture or cache per-layer full sinks in
     a graph-safe workspace.
  3. Prototype a true fused `cp_lse_ag_out_rs` only if it reduces the number of
     CP collective launches or removes the O/LSE gather + local slice copies in
     the actual runtime path.

## 2026-06-24 - Experiment 69: service-level decode trace, no runtime change

Goal:

- Verify the Experiment 68 microbench attribution in the actual SGLang serving
  decode path.
- Compare TP4 and TP4+CP2 with the same short profile workload and identify
  which kernels are AttnCP-specific.
- Re-check the training overlap code referenced by `task.md` before deciding
  on the next optimization direction.

Trace workload:

- Model:
  `/home/fhkong/models/80a3_v4d5_256k_merge_thinking_kimi_k25_0502_20260503_032335/epoch_003_step_0002610`
- GPUs: `CUDA_VISIBLE_DEVICES=4,5,6,7`
- Serving args: FA3 prefill/decode, `page_size=1`, `chunked_prefill_size=1024`,
  `--enable-over-encoding`, `--enable-welm-kv-mirror-opt`, CUDA graph on,
  piecewise CUDA graph disabled.
- Request workload: random ids, input 8192, output 64, concurrency 4,
  8 prompts, profile-by-stage decode, 6 profiled decode graph launches.

Artifacts:

- TP4: `/tmp/welmv4_attncp_trace/20260624_093722_tp4_decode`
- TP4+CP2: `/tmp/welmv4_attncp_trace/20260624_093414_cp2_decode`

Profiler caveat:

- Torch profiler adds large overhead, so these trace benchmark numbers are not
  used as final performance numbers.
- The relative median ITL shape still matches the unprofiled benchmark:
  TP4 median ITL `8.33 ms`, TP4+CP2 median ITL `10.25 ms`.

Rank-0 decode trace summary:

| config | graph launches | kernels / step | kernel time / step | NCCL kernels / step | NCCL time / step |
|---|---:|---:|---:|---:|---:|
| TP4 | `6` | `1506.0` | `8.199 ms` | `1.0` | `0.014 ms` |
| TP4+CP2 | `6` | `2194.0` | `10.729 ms` | `97.0` | `0.662 ms` |

Kernel category comparison:

| category | TP4 | TP4+CP2 | delta |
|---|---:|---:|---:|
| NCCL kernels / step | `1.0` | `97.0` | `+96.0` |
| merge kernels / step | `0.0` | `48.0` | `+48.0` |
| copy / elementwise kernels / step | `291.0` | `833.0` | `+542.0` |
| FA kernels / step | `145.0` | `145.0` | `0.0` |
| TP custom reduce kernels / step | `98.0` | `98.0` | `0.0` |
| TP custom reduce time / step | `0.694 ms` | `1.906 ms` | `+1.212 ms` |

Interpretation:

- Service trace confirms the actual AttnCP decode CUDA graph does not replay a
  per-layer attention-sink allgather. The dominant AttnCP-specific NCCL count
  is `48 layers × 2 CP allgather kernels/layer = 96 kernels/step`, plus one
  extra non-attention gather.
- The current implementation has the intended true-sharded-KV compute shape:
  FA kernel count per step is effectively unchanged, and local FA time is not
  the source of the gap.
- CP2 adds one merge kernel per layer and many copy/elementwise kernels from Q
  head layout materialization, LSE normalization/copy, empty-local-KV masking,
  and local-head merge staging.
- The same TP custom reduce kernels exist in TP4 and CP2, but their measured
  time is higher in CP2. This is consistent with the added CP NCCL collectives
  and copy/merge kernels increasing graph length and communication contention.
- This makes the remaining ITL gap a graph-level collective/merge issue, not a
  single slow FA call or a missed sink cache.

Training overlap review:

- Reviewed:
  `/home/fhkong/wxwork/mimikyu/mmq/mmq/modules/block_v2/memory_optimizer/qkv_proj_and_post_processing/overlap.py`.
- The training overlap path launches Q all-to-all asynchronously right after
  `q_proj`, then computes K/V/gate projections while Q communication is in
  flight, and only waits before QK norm / RoPE.
- This does not directly transfer to serving decode local-merge:
  serving decode has already finished QKV projection before attention, and
  `flash_attn_with_kvcache` cannot run until full-Q is available. The O/LSE
  CP gather depends on local FA output, so it also cannot be overlapped with
  the same layer's FA.

Decision:

- No runtime change in this experiment.
- Do not pursue a direct port of the training QKV overlap code for decode.
- The next meaningful optimization needs to reduce the number or cost of the
  per-layer CP boundary, for example:
  1. a lower-level fused `cp_lse_ag_out_rs`/head-slice primitive that avoids
     full-head O/LSE gather staging and merge copies;
  2. graph-level scheduling that overlaps a layer's CP output communication
     with independent work from another layer, if correctness and CUDA graph
     ordering can be proven;
  3. a fused Q gather layout path that writes the FA-compatible full-Q layout
     without per-layer slice-copy kernels.

## 2026-06-24 - Experiment 70: single-kernel full-Q layout pack

Goal:

- Reduce one source of the extra AttnCP copy/elementwise kernels identified by
  Experiment 69 without changing sharded-KV semantics.

Change:

- In `_attncp_gather_full_q()`, replace per-CP slice copies:

```python
for cp_idx in range(cp_world_size):
    q_full[:, cp_idx * local_q_heads : (cp_idx + 1) * local_q_heads, :].copy_(
        q_view[cp_idx]
    )
```

- With one strided copy into a 4D view:

```python
q_full.view(batch_size, cp_world_size, local_q_heads, head_dim).copy_(
    q_view.permute(1, 0, 2, 3)
)
```

Correctness:

- Random tensor layout check:
  - `cp=2`, `batch=1/4/7`: `max_diff=0`
  - `cp=4`, `batch=1/4/7`: `max_diff=0`
- Full precision regression:
  - artifact: `/tmp/welmv4_attncp_precision/20260624_094421`
  - controlled compare: `mode=token_level`, `passed=True`, `issues=0`
  - MMLU/C-Eval samples tested: `100 / 100`
  - token mismatches: `0 / 100 PASS`
  - max logprob diff: `0.00e+00 PASS`
  - mean logprob diff: `0.00e+00 PASS`
  - final result: `PASS: TP4 vs TP4+CP2 sharded-KV precision regression passed.`

Local microbench:

| shape | old slice-copy pack | new 4D strided-copy pack |
|---|---:|---:|
| `CP2, batch=4, local_q_heads=6, head_dim=256` | `16.239 us` | `6.913 us` |
| `CP4, batch=4, local_q_heads=6, head_dim=256` | `32.443 us` | `6.901 us` |

End-to-end benchmark:

```bash
ALLOW_DIRTY=1 CASE_SET=decode512 \
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  /home/fhkong/wxwork/attncp_performance_benchmark/run_attncp_perf_benchmark.sh
```

Artifact:

- `/tmp/welmv4_attncp_perf/20260624_094840`

| config | output TPS | mean TTFT | mean ITL | peak GPU memory |
|---|---:|---:|---:|---:|
| TP4 | `336.0596 tok/s` | `1623.1546 ms` | `8.7606 ms` | `81665 MiB` |
| TP4+CP2 | `281.9817 tok/s` | `1837.9362 ms` | `10.6340 ms` | `63299 MiB` |

Gap:

- TTFT: `+13.2%`
- ITL: `+21.4%`
- Output throughput: `-16.1%`
- Peak memory: `-17.94 GiB/rank`

Interpretation:

- The Q layout pack cleanup is correctness-safe and directionally improves CP2
  ITL versus the previous `10.7143 ms` run, but the service-level gain is small.
- This matches the trace: Q pack copies are only one part of the extra CP2
  copy/elementwise kernels. The larger remaining issue is still the per-layer
  CP collective/merge boundary and the interference it creates with existing TP
  communication.

## 2026-06-24 - Experiment 71: LSE allgather + O reduce-scatter prototype

Goal:

- Test whether replacing the workspace decode merge tail:
  `O/LSE allgather -> merge_state_v2`
  with:
  `LSE allgather -> global LSE reduce -> scale/pack O -> reduce_scatter`
  can reduce the CP2 decode ITL gap.

Change tested:

- Added a temporary runtime path guarded by
  `SGLANG_ATTNCP_EXPERIMENTAL_DECODE_REDUCE_SCATTER_MERGE=1`.
- The path kept sharded-KV semantics unchanged:
  - full-Q allgather before FA3 stayed the same;
  - local FA3 over local KV shard stayed the same;
  - attention sink and empty-local-KV correction stayed the same;
  - only the post-FA CP merge path changed.

Correctness:

- Full precision regression with the experimental path enabled:
  - artifact: `/tmp/welmv4_attncp_precision/20260624_100330`
  - controlled compare: `mode=token_level`, `passed=True`, `issues=0`
  - controlled max/mean diff: `0.000e+00` / `0.000e+00`
  - MMLU/C-Eval samples tested: `100 / 100`
  - token mismatches: `0 / 100 PASS`
  - max logprob diff: `0.00e+00 PASS`
  - mean logprob diff: `0.00e+00 PASS`

End-to-end benchmark:

```bash
SGLANG_ATTNCP_EXPERIMENTAL_DECODE_REDUCE_SCATTER_MERGE=1 \
ALLOW_DIRTY=1 CASE_SET=decode512 \
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  /home/fhkong/wxwork/attncp_performance_benchmark/run_attncp_perf_benchmark.sh
```

Artifact:

- `/tmp/welmv4_attncp_perf/20260624_100804`

| config | output TPS | mean TTFT | mean ITL | peak GPU memory |
|---|---:|---:|---:|---:|
| TP4 | `335.4569 tok/s` | `1629.9008 ms` | `8.7687 ms` | `81665 MiB` |
| TP4+CP2 | `281.5893 tok/s` | `1849.2736 ms` | `10.6318 ms` | `63321 MiB` |

Gap:

- TTFT: `+13.5%`
- ITL: `+21.2%`
- Output throughput: `-16.1%`
- Peak memory: `-17.92 GiB/rank`

Decision:

- The experimental path is correctness-safe in this run, but it does not
  improve end-to-end service performance versus the default local-merge path
  (`10.6340 ms` -> `10.6318 ms` ITL is noise-level).
- Do not keep this as a runtime branch. The temporary runtime code was removed.
- Keep the standalone benchmark prototype as evidence for future lower-level
  fused collective work.

## 2026-06-24 - Experiment 72: CP2 local-head slice merge microbench

Goal:

- Measure whether the local post-gather merge staging in decode local-merge has
  enough standalone cost to justify a future fused kernel. This is a
  microbench-only experiment; no runtime path was added.

Change:

- Extended `benchmark/kernels/attention/bench_attncp_decode_components.py` with
  a CP2-only prototype kernel that directly merges the local head slice from
  gathered O/LSE buffers:
  `gathered_o/gathered_lse -> local H/TP output`.
- The prototype bypasses the current staging sequence:
  slice copy to `current/next` buffers + `merge_state_v2`.

Command:

```bash
CUDA_VISIBLE_DEVICES=0 \
python benchmark/kernels/attention/bench_attncp_decode_components.py \
  --batch-size 4 \
  --kv-lens 8192 \
  --tp-size 4 \
  --cp-size 2 \
  --cp-rank 0 \
  --cp-kv-chunk-size 1024 \
  --warmup 10 \
  --iters 50 \
  --trials 3 \
  --output /tmp/attncp_decode_components_cp2_slice_merge_20260624_101819.json
```

Results:

| component | median |
|---|---:|
| `target_cp2_local_slice_merge_current` | `38.301 us` |
| `target_cp2_local_slice_merge_fused` | `9.005 us` |

Correctness in the microbench:

- `max_diff=0.0`
- `mean_diff=0.0`

Interpretation:

- The local slice-copy + merge staging is a real standalone component cost in
  this synthetic shape, and a fused local merge kernel can remove most of it.
- This does not yet prove service-level improvement. The earlier service trace
  still showed the larger issue is per-layer CP collective/graph boundary
  overhead, so a runtime change needs a full decode trace + precision
  regression before it should be considered.
- Stop here for manual review before any runtime integration.

## 2026-06-24 - P0: Decode sink cache only

Goal:

- Validate the first incremental optimization step before enabling CP2 P2P
  exchange paths.
- P0 keeps Q gather and O/LSE exchange on the existing collective path and only
  changes fallback/eager sink handling to reuse the full-sink cache.

Runtime flags:

```bash
SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_Q_P2P=0
SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_OLSE_P2P=0
```

Correctness:

- Artifact: `/tmp/welmv4_attncp_precision/20260624_195801`
- Controlled compare: `mode=token_level`, `passed=True`, `issues=0`
- Controlled max/mean diff: `0.000e+00` / `0.000e+00`
- MMLU/C-Eval samples tested: `100 / 100`
- Token mismatches: `0 / 100 PASS`
- Max logprob diff: `0.00e+00 PASS`
- Mean logprob diff: `0.00e+00 PASS`

Benchmark:

- Artifact: `/tmp/welmv4_attncp_perf/20260624_200311_p0_only`
- Scenario: `input_len=8192`, `output_len=512`, `concurrency=4`,
  `num_prompts=16`, CUDA graph enabled, FA3 prefill/decode,
  `welm-kv-mirror-opt` enabled.

| config | completed | output TPS | mean TTFT | mean ITL |
|---|---:|---:|---:|---:|
| TP4 | 16 | `336.2735 tok/s` | `1619.2296 ms` | `8.7607 ms` |
| TP4+CP2 P0 | 16 | `281.8596 tok/s` | `1845.5468 ms` | `10.6254 ms` |

Gap:

- TTFT: CP2 is `+14.0%` slower.
- ITL: CP2 is `+21.3%` slower.
- Output throughput: CP2 is `-16.2%` lower.

Decision:

- P0 is correctness-safe.
- P0 by itself does not materially reduce the existing TP4 vs TP4+CP2 ITL gap.

## 2026-06-24 - P1: CP2 O/LSE head-slice P2P exchange

Goal:

- Replace the workspace decode tail:
  `full O/LSE allgather_coalesced -> local head slice merge`
  with a CP2-only P2P exchange of only the peer-owned local-head slice.
- Keep Q on the existing allgather path to isolate the post-FA exchange change.

Runtime flags:

```bash
SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_Q_P2P=0
SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_OLSE_P2P=1
```

Correctness:

- Artifact: `/tmp/welmv4_attncp_precision/20260624_200744`
- Controlled compare: `mode=token_level`, `passed=True`, `issues=0`
- Controlled max/mean diff: `0.000e+00` / `0.000e+00`
- MMLU/C-Eval samples tested: `100 / 100`
- Token mismatches: `0 / 100 PASS`
- Max logprob diff: `0.00e+00 PASS`
- Mean logprob diff: `0.00e+00 PASS`

Benchmark:

- Artifact: `/tmp/welmv4_attncp_perf/20260624_201245_p1_olse_p2p`
- Scenario: `input_len=8192`, `output_len=512`, `concurrency=4`,
  `num_prompts=16`, CUDA graph enabled, FA3 prefill/decode,
  `welm-kv-mirror-opt` enabled.

| config | completed | output TPS | mean TTFT | mean ITL |
|---|---:|---:|---:|---:|
| TP4 | 16 | `335.8331 tok/s` | `1629.3735 ms` | `8.7564 ms` |
| TP4+CP2 P1 | 16 | `276.5146 tok/s` | `1866.9824 ms` | `10.8589 ms` |

Gap:

- TTFT: CP2 is `+14.6%` slower.
- ITL: CP2 is `+24.0%` slower.
- Output throughput: CP2 is `-17.7%` lower.

Observation:

- P1 is correctness-safe, but slower than P0 in this service benchmark
  (`10.6254 ms -> 10.8589 ms` mean ITL).
- CUDA graph capture memory for CP2 increased in the server log
  (`~0.27 GB` in P0 vs `~0.68 GB` in P1), consistent with extra staging
  buffers/graph nodes.

Decision:

- Do not keep the current Python-level CP2 O/LSE P2P exchange enabled by
  default. It needs a lower-level fused pack/exchange/merge implementation
  before it can beat the current coalesced allgather path.

## 2026-06-24 - P2: CP2 Q P2P exchange

Goal:

- Replace decode Q `all_gather_into_tensor` with a CP2-only P2P exchange that
  directly fills the existing `q_full` workspace layout.
- Keep O/LSE on the existing coalesced allgather path to isolate Q exchange.

Runtime flags:

```bash
SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_Q_P2P=1
SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_OLSE_P2P=0
```

Correctness:

- Artifact: `/tmp/welmv4_attncp_precision/20260624_201729`
- Controlled compare: `mode=token_level`, `passed=True`, `issues=0`
- Controlled max/mean diff: `0.000e+00` / `0.000e+00`
- MMLU/C-Eval samples tested: `100 / 100`
- Token mismatches: `0 / 100 PASS`
- Max logprob diff: `0.00e+00 PASS`
- Mean logprob diff: `0.00e+00 PASS`

Benchmark:

- Artifact: `/tmp/welmv4_attncp_perf/20260624_202221_p2_q_p2p`
- Scenario: `input_len=8192`, `output_len=512`, `concurrency=4`,
  `num_prompts=16`, CUDA graph enabled, FA3 prefill/decode,
  `welm-kv-mirror-opt` enabled.

| config | completed | output TPS | mean TTFT | mean ITL |
|---|---:|---:|---:|---:|
| TP4 | 16 | `335.3294 tok/s` | `1634.7214 ms` | `8.7638 ms` |
| TP4+CP2 P2 | 16 | `277.9886 tok/s` | `1853.7970 ms` | `10.8076 ms` |

Gap:

- TTFT: CP2 is `+13.4%` slower.
- ITL: CP2 is `+23.3%` slower.
- Output throughput: CP2 is `-17.1%` lower.

Observation:

- P2 is correctness-safe, but slower than the P0 collective path in this
  service benchmark (`10.6254 ms -> 10.8076 ms` mean ITL).
- The Python-level P2P path still launches communication and staging work as
  graph-visible operations, so it does not remove the dominant per-layer decode
  overhead.

Decision:

- Keep CP2 Q P2P as an internal experimental path only.
- The default decode path should remain the stable P0 path until Q exchange is
  fused/lowered enough to reduce graph node and launch overhead.

## 2026-06-24 - P3: Final default CUDA graph coverage

Goal:

- Validate the default runtime path after disabling the experimental CP2 Q P2P
  and O/LSE P2P paths.
- Confirm the default path still captures decode CUDA graph and preserves TP4
  precision.

Runtime flags:

```bash
# no experimental AttnCP decode P2P env vars
SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_Q_P2P=0      # default
SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_OLSE_P2P=0   # default
```

Correctness:

- Artifact: `/tmp/welmv4_attncp_precision/20260624_203857`
- Controlled compare: `mode=token_level`, `passed=True`, `issues=0`
- Controlled max/mean diff: `0.000e+00` / `0.000e+00`
- MMLU/C-Eval samples tested: `100 / 100`
- Token mismatches: `0 / 100 PASS`
- Max logprob diff: `0.00e+00 PASS`
- Mean logprob diff: `0.00e+00 PASS`

Benchmark:

- Artifact: `/tmp/welmv4_attncp_perf/20260624_204356_p3_default_final`
- Scenario: `input_len=8192`, `output_len=512`, `concurrency=4`,
  `num_prompts=16`, CUDA graph enabled, FA3 prefill/decode,
  `welm-kv-mirror-opt` enabled.

| config | completed | output TPS | mean TTFT | mean ITL |
|---|---:|---:|---:|---:|
| TP4 | 16 | `335.5000 tok/s` | `1627.9340 ms` | `8.7710 ms` |
| TP4+CP2 default | 16 | `281.9686 tok/s` | `1852.6652 ms` | `10.6063 ms` |

Gap:

- TTFT: CP2 is `+13.8%` slower.
- ITL: CP2 is `+20.9%` slower.
- Output throughput: CP2 is `-16.0%` lower.

CUDA graph coverage:

- TP4 log shows graph capture end on all TP ranks and decode batches with
  `cuda graph: True`.
- TP4+CP2 log shows graph capture end on all `(ATTN_CP, TP)` ranks and decode
  batches with `cuda graph: True`.
- CP2 capture memory is back to the P0 level (`~0.27 GB`) instead of the P1
  P2P path (`~0.68 GB`).

Decision:

- Keep the P0 sink-cache-only decode cleanup as the default.
- Keep CP2 Q P2P and O/LSE P2P behind internal env flags, disabled by default.
- The next useful performance step is not more Python-level collectives; it
  should be a fused/lowered decode path that removes per-layer Q gather and
  O/LSE gather overhead from the CUDA graph critical path.

## 2026-06-24 - P4: CP2 fused local-head merge kernel

Goal:

- Replace the CP2 local-head merge tail:
  `slice copy from gathered O/LSE -> merge_state_v2`
  with one Triton kernel that reads the gathered tensors directly and writes
  the final local-head output.
- Keep Q allgather, O/LSE `all_gather_coalesced`, KV shard ownership, and
  attention semantics unchanged.

Runtime flag:

```bash
SGLANG_ATTNCP_DECODE_CP2_FUSED_MERGE=1   # default
```

Component benchmark:

- Artifact: `/tmp/attncp_decode_components_cp2_merge_pre.json`
- Shape: `batch_size=4`, `kv_len=8192`, `TP=4`, `CP=2`, `cp_rank=0`,
  `local_q_heads=8`, `head_dim=128`, model dtype `bfloat16`.

| component | median |
|---|---:|
| current copy + `merge_state_v2` | `36.907 us` |
| fused Triton local-head merge | `8.652 us` |

Micro correctness:

- Direct helper comparison against `merge_state_v2`.
- Dtypes: `bfloat16`, `float16`.
- Batch sizes: `1`, `4`, `16`.
- CP ranks: `0`, `1`.
- Max output diff: `0.0`.

Correctness:

- Artifact: `/tmp/welmv4_attncp_precision/20260624_224011`
- Controlled compare: `mode=token_level`, `passed=True`, `issues=0`
- Controlled max/mean diff: `0.000e+00` / `0.000e+00`
- MMLU/C-Eval samples tested: `100 / 100`
- Token mismatches: `0 / 100 PASS`
- Max logprob diff: `0.00e+00 PASS`
- Mean logprob diff: `0.00e+00 PASS`

Benchmark:

- Artifact: `/tmp/welmv4_attncp_perf/20260624_224525_p4_fused_merge`
- Scenario: `input_len=8192`, `output_len=512`, `concurrency=4`,
  `num_prompts=16`, CUDA graph enabled, FA3 prefill/decode,
  `welm-kv-mirror-opt` enabled.

| config | completed | output TPS | mean TTFT | mean ITL |
|---|---:|---:|---:|---:|
| TP4 | 16 | `336.0555 tok/s` | `1620.4334 ms` | `8.7661 ms` |
| TP4+CP2 fused merge | 16 | `290.6197 tok/s` | `1832.3900 ms` | `10.2214 ms` |
| TP4+CP2 unfused merge | 16 | `282.4439 tok/s` | `1828.8627 ms` | `10.6283 ms` |

Gap:

- TP4+CP2 fused vs TP4 ITL: `+16.6%`.
- TP4+CP2 unfused vs TP4 ITL: `+21.2%`.
- Fused merge vs unfused merge ITL: `-3.8%`.
- Fused merge vs unfused merge output TPS: `+2.9%`.

CUDA graph coverage:

- TP4+CP2 fused log shows graph capture end on all `(ATTN_CP, TP)` ranks.
- TP4+CP2 fused decode batches show `cuda graph: True`.
- Fused capture memory is `~0.25 GB`, slightly lower than unfused `~0.27 GB`
  in this run.

Decision:

- Keep fused CP2 local-head merge enabled by default.
- This closes part of the ITL gap without changing communication or sharded-KV
  semantics.
- Remaining ITL gap is still dominated by Q allgather and O/LSE gather. The next
  kernel-fusion step needs to target communication-adjacent work, not another
  local-only merge.

Follow-up P2P probe:

- Artifact: `/tmp/welmv4_attncp_perf/20260624_225548_p4_p2p_fused_probe`
- Change: keep fused local merge enabled and additionally enable
  `SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_OLSE_P2P=1`.
- Extra fused kernels added for the P2P path:
  `local_o_full/local_lse_full -> peer send buffer` pack and
  `local slice + remote slice -> final output` merge.

| config | completed | output TPS | mean TTFT | mean ITL |
|---|---:|---:|---:|---:|
| TP4+CP2 allgather fused | 16 | `290.2648 tok/s` | `1850.5601 ms` | `10.2030 ms` |
| TP4+CP2 P2P fused | 16 | `282.3482 tok/s` | `1871.4723 ms` | `10.5499 ms` |

Result:

- P2P fused is still `+3.4%` slower in ITL than allgather fused.
- Output throughput is `-2.7%` lower.
- Keep O/LSE P2P disabled by default. The Pynccl send/recv graph overhead still
  outweighs the reduced payload at this shape.

## 2026-06-25

### Experiment 5: CP2 Fused Q + Local-FA Prototype

Goal:

- Introduce a replaceable function boundary for the AttnCP decode region:
  `Q gather + local FA3 partial attention`.
- Prototype a Triton implementation that consumes `q_local + q_peer` directly
  and computes local-shard partial `O/LSE` for all CP Q heads.
- Preserve sharded KV residency: each CP rank still only reads its local KV
  shard.

Code shape:

- New env-gated backend path:
  `SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_FUSED_Q_FA=1`.
- Default remains off, so existing precision/perf path is unchanged unless the
  env is explicitly enabled.
- Current prototype uses pynccl send/recv to populate a `q_peer` workspace; the
  remote peer-memory load/CUDA IPC part is not implemented yet.
- The Triton kernel is KV-tile driven: one program handles `(batch, kv_head)`
  and reuses each loaded local KV tile across the CP2 full Q-head group.

Supported prototype constraints:

- CP size = 2.
- `page_size = 1`.
- decode only, `max_seq_len_q = 1`.
- `head_dim == v_head_dim`.
- no local window inside the kernel; SWA must already be translated to local
  metadata with `(-1, -1)` attention window.
- optional attention sink supported by initializing the online softmax state
  with sink mass.

Focused correctness:

```bash
PYTHONPATH=python python -m pytest \
  python/sglang/jit_kernel/tests/test_attncp_fused_ops.py -q
```

Result:

- `12 passed`
- Coverage:
  - CP rank `0/1`
  - attention sink on/off
  - softcap `0.0/15.0`
  - empty local KV row
  - paged local KV with `page_size=1`
  - direct comparison against current FA3 `flash_attn_with_kvcache` local partial
    path for sink on/off

Microbenchmark:

- Single GPU, compares local partial attention only.
- FA3 path uses already-materialized `q_full`.
- Triton path uses `q_local + q_peer`.
- Q communication is not included in either number.
- dtype `bf16`, `local_q_heads=6`, `full_q_heads=12`, `kv_heads=1`,
  `head_dim=128`, `batch=24`.

| local KV len | Triton prototype | FA3 local partial | ratio |
|---:|---:|---:|---:|
| 512 | `0.0187 ms` | `0.0226 ms` | `0.83x` |
| 2048 | `0.0364 ms` | `0.0321 ms` | `1.14x` |
| 8192 | `0.1348 ms` | `0.0739 ms` | `1.82x` |

Decision:

- Keep this path disabled by default.
- The function boundary and numerical prototype are useful, but the current
  single-program-per-`(batch, kv_head)` design underutilizes the GPU for long
  local KV.
- Before enabling end-to-end server regression/sweep, the next step should add
  split-KV partial states plus a split merge kernel, so long-context decode can
  recover FA3-like parallelism while still avoiding Q all-gather materialization.

### Experiment 6: Split-KV Fused Q + Local-FA Prototype

Change:

- Add a split-KV variant for the fused Q+local-FA prototype.
- Each split owns a disjoint KV position range, so each local KV token is still
  read by exactly one split.
- A second Triton kernel merges split partial `(O, LSE)` states. This merge only
  reads split partial outputs, not KV.
- The backend allocates split workspace only when
  `SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_FUSED_Q_FA=1`.
- Default split cap:
  `SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_FUSED_Q_FA_MAX_SPLITS=8`.
- Heuristic: do not split when the local page-table capacity is `<=4096`
  tokens; split longer local KV with target split size around 1024 tokens.

Focused correctness:

```bash
PYTHONPATH=python python -m pytest \
  python/sglang/jit_kernel/tests/test_attncp_fused_ops.py -q
```

Result:

- `16 passed`
- Added split-KV comparison against current FA3 for CP rank `0/1` and sink
  on/off with `max_seq_len=4608`.

Microbenchmark:

- Same setup as Experiment 5.
- Fused path uses split workspace with `max_splits=8`.

| local KV len | fused Q+FA prototype | FA3 local partial | ratio |
|---:|---:|---:|---:|
| 512 | `0.0202 ms` | `0.0228 ms` | `0.88x` |
| 2048 | `0.0362 ms` | `0.0304 ms` | `1.19x` |
| 8192 | `0.0616 ms` | `0.0736 ms` | `0.84x` |
| 16384 | `0.1078 ms` | `0.1342 ms` | `0.80x` |

Decision:

- Split-KV fixes the long-local-KV parallelism issue and makes the prototype
  locally faster than FA3 for 8k/16k local shard lengths.
- The prototype is still not ready to enable by default because remote-Q
  acquisition is still pynccl send/recv rather than direct peer-memory load, and
  true end-to-end TP4-CP2 CUDA graph correctness/perf has not been validated.

Server smoke:

- Artifact: `/tmp/welmv4_attncp_fused_qfa_smoke_20260625_223108`
- Config: TP4-CP2 sharded-KV, FA3 prefill/decode, CUDA graph max bs 16,
  `SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_FUSED_Q_FA=1`,
  `SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_FUSED_Q_FA_MAX_SPLITS=8`,
  `--attn-cp-decode-cuda-graph-max-seq-len 8704`.
- Result:
  - server reached ready state,
  - CUDA graph capture completed for bs `[1, 2, 4, 8, 12, 16]`,
  - one controlled request with `prompt_len=64`, `max_new_tokens=4` completed,
  - no traceback/error in server log.

Full precision regression:

```bash
cd /home/fhkong/wxwork/attncp_precision_regression
env \
  SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_FUSED_Q_FA=1 \
  SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_FUSED_Q_FA_MAX_SPLITS=8 \
  SGLANG_ATTNCP_DENSE_DECODE_GRAPH_MAX_SEQ_LEN=8704 \
  ./run_full_precision.sh
```

Artifact:

- `/tmp/welmv4_attncp_precision/20260625_223327`

Result:

- Controlled compare: `mode=token_level`, `passed=True`, `issues=0`,
  `max_logprob_diff=0.0`, `mean_logprob_diff=0.0`.
- MMLU/C-Eval regression: `100 / 100` samples, `0` errors.
- Token mismatches: `0 / 100 PASS`.
- Max logprob diff: `0.00e+00 PASS`.
- Mean logprob diff: `0.00e+00 PASS`.
- CP2 server captured CUDA graph for bs
  `[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128]`
  with AttnCP seq cap `8704`.
- CP2 decode log showed `cuda graph: True`.

Throughput sweep:

- Scenario: `input_len=32768`, `output_len=2048`, random ids,
  `request-rate=inf`, CUDA graph enabled, FA3 prefill/decode, page size 1,
  `welm-kv-mirror-opt` enabled.
- TP4 artifact:
  `/tmp/welmv4_attncp_manual_sweep/20260625_224030_tp4_s32768_o2048/summary.tsv`
- TP4-CP2 fused artifact:
  `/tmp/welmv4_attncp_manual_sweep/20260625_224851_tp4_cp2_s32768_o2048/summary.tsv`
- TP4-CP2 server log:
  `/tmp/welmv4_attncp_manual_sweep/20260625_224654_tp4_cp2/server.log`
- TP4-CP2 fused env:
  `SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_FUSED_Q_FA=1`,
  `SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_FUSED_Q_FA_MAX_SPLITS=8`,
  `SGLANG_ATTNCP_DENSE_DECODE_GRAPH_MAX_SEQ_LEN=36864`.

| config | concurrency | output TPS | mean TTFT | mean ITL | peak running | max token usage |
|---|---:|---:|---:|---:|---:|---:|
| TP4 | 22 | `433.6718` | `17030.5992 ms` | `42.4303 ms` | 22 | 0.94 |
| TP4 | 23 | `445.6913` | `18490.6186 ms` | `42.5941 ms` | 23 | 0.98 |
| TP4 | 24 | `403.1272` | `21555.5237 ms` | `41.1964 ms` | 23 | 0.98 |
| TP4-CP2 fused-QFA | 44 | `646.0761` | `38093.9664 ms` | `49.5316 ms` | 44 | 0.96 |
| TP4-CP2 fused-QFA | 45 | `669.1198` | `36068.7343 ms` | `49.6412 ms` | 45 | 0.98 |
| TP4-CP2 fused-QFA | 46 | `597.0761` | `38809.8019 ms` | `48.8172 ms` | 45 | 0.98 |

Interpretation:

- Capacity result remains good: CP2 supports about 45 resident 32k/2k requests
  versus TP4 about 23.
- Best measured throughput in this sweep: TP4 c23 `445.69 tok/s`, TP4-CP2
  fused c45 `669.12 tok/s`, about `1.50x`.
- Relative to the earlier default CP2 local-merge sweep, fused-QFA does not show
  a clear end-to-end improvement at c44/c45/c46. The likely reason is that the
  current prototype replaces NCCL Q all-gather with pynccl send/recv plus custom
  FA kernels, not true peer-memory remote load inside the attention kernel.
- Keep fused-QFA disabled by default. The next step should verify path-hit
  counters/profile and then replace peer-Q acquisition with direct peer-memory
  load or a lower-overhead communication primitive before another full sweep.

### 2026-06-26 Follow-up: KV-Stationary Constraint and Fused Hit Diagnosis

Kernel invariant:

- The fused Triton prototype is intentionally KV-stationary.
- Program granularity is `(batch, kv_head, split)`.
- Inside one program, a resident K/V tile is loaded once and reused for every
  CP2 Q head mapped to that KV head.
- Split ranges are non-overlapping, so split-KV parallelism does not re-read the
  same KV token range in a different split. The split merge reads only partial
  O/LSE state, not K/V.

Service diagnosis:

- Added env-gated fused-path hit profiling that can log during CUDA graph
  capture:

```bash
SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_FUSED_Q_FA_PROFILE=1
SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_FUSED_Q_FA_PROFILE_INTERVAL=1
```

- A TP4-CP2 startup probe with fused enabled reached ready state:
  `/tmp/welmv4_attncp_manual_sweep/20260626_020657_tp4_cp2/server.log`.
- The log confirmed CUDA graph capture can hit the fused path, for example
  `hit_split` was present during graph capture.
- A small runtime probe also completed:
  `/tmp/welmv4_attncp_manual_sweep/20260626_020657_tp4_cp2_probe/result.jsonl`.
  Scenario: `32k input / 64 output / c4`, result `completed=4`, server decode
  log showed `cuda graph: True`.

Important correction:

- `SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_FUSED_Q_FA_MIN_SEQ_CAP` is checked
  against the **local CP KV shard page-table capacity**, not the global prompt
  length.
- For global `32k` prompt with CP2 sharded KV, the local shard cap is about
  `16k`. Therefore a default `MIN_SEQ_CAP=32768` usually keeps the `32k`
  workload on the FA3 fallback path.
- This explains why a guarded fused run can look identical to the exact FA3
  P2P path in the `32k/2k` sweep: the hot bucket may never use the Triton
  fused attention kernel.
- Lowering the guard to `16384` is required to exercise fused Q+FA for a global
  `32k` CP2 workload, but the earlier `MIN_SEQ_CAP=16384` service sweep was
  slightly slower and the unguarded fused math still failed strict logprob
  regression. This remains experimental only.

Current status:

- Unit coverage remains green:

```bash
PYTHONPATH=python .venv/bin/python -m pytest \
  python/sglang/jit_kernel/tests/test_attncp_fused_ops.py -q
```

Result: `28 passed`.

- The fused kernel satisfies the intended "read resident KV once per split"
  structure.
- It is not yet production-ready because independent Triton attention math is
  not strict-regression equivalent to FA3 across the full WeLM decode stack.

Guarded full precision regression after the profiling changes:

```bash
env \
  SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_FUSED_Q_FA=1 \
  SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_FUSED_Q_FA_MIN_SEQ_CAP=32768 \
  SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_FUSED_Q_FA_MAX_SPLITS=8 \
  SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_Q_P2P=1 \
  SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_OLSE_P2P=1 \
  /home/fhkong/wxwork/attncp_precision_regression/run_full_precision.sh
```

Artifact:

- `/tmp/welmv4_attncp_precision/20260626_021558`

Result:

- Controlled token-level compare: PASS, `max_diff=0.000e+00`.
- MMLU/C-Eval strict regression: PASS.
- Samples tested: `100 / 100`, errors `0`.
- Token mismatches: `0 / 100`.
- Max logprob diff: `0.00e+00`.
- Mean logprob diff: `0.00e+00`.

Interpretation:

- This validates the guarded integration and P2P exact fallback path.
- It still does not prove hot fused-QFA attention math is strict-equivalent,
  because the short-prompt precision suite stays below the local-shard
  `MIN_SEQ_CAP=32768` threshold and therefore uses FA3.

Fused-hot CP2 sweep for global `32k/2k`:

```bash
env \
  SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_FUSED_Q_FA=1 \
  SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_FUSED_Q_FA_MIN_SEQ_CAP=16384 \
  SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_FUSED_Q_FA_MAX_SPLITS=8 \
  SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_Q_P2P=1 \
  SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_OLSE_P2P=1 \
  /home/fhkong/wxwork/attncp_tp4_cp2_sweep/start_tp4_cp2.sh

/home/fhkong/wxwork/attncp_tp4_cp2_sweep/sweep_concurrency.sh tp4_cp2 44 45 46
```

Artifacts:

- Server: `/tmp/welmv4_attncp_manual_sweep/20260626_022144_tp4_cp2/server.log`
- Summary:
  `/tmp/welmv4_attncp_manual_sweep/20260626_022340_tp4_cp2_s32768_o2048/summary.tsv`

Results:

| config | concurrency | output TPS | mean TTFT | mean ITL | peak running | max queue | max token usage | cuda graph |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| TP4-CP2 fused-hot | 44 | `648.52` | `37247.03 ms` | `49.69 ms` | 44 | 43 | 0.96 | yes |
| TP4-CP2 fused-hot | 45 | `664.77` | `36414.24 ms` | `49.94 ms` | 45 | 43 | 0.98 | yes |
| TP4-CP2 fused-hot | 46 | `526.60` | `38579.20 ms` | `49.22 ms` | 45 | 44 | 0.98 | yes |

Conclusion:

- Lowering the local-shard guard to `16384` makes the global `32k` workload
  eligible for fused Q+FA, but the end-to-end numbers remain effectively the
  same as the exact FA3/P2P path.
- The current Triton fused prototype therefore does not close the service-level
  ITL/TPS gap, even though it can be faster in synthetic attention-only
  microbenchmarks.
- The likely next meaningful optimization is not more Python-side routing, but
  a FA3/CUDA-level implementation that keeps FA3 numerical behavior while
  eliminating Q materialization/communication overhead, or a lower-level
  remote-Q load path with better register/occupancy behavior.

Additional attention-only microbenchmarks:

```bash
CUDA_VISIBLE_DEVICES=4,5 PYTHONPATH=python .venv/bin/torchrun \
  --standalone --nproc_per_node=2 \
  benchmark/kernels/attention/bench_attncp_decode_paths.py \
  --batch-size 45 --kv-lens 32768 --tp-size 4 --cp-size 2 \
  --cp-kv-chunk-size 1024 --warmup 5 --iters 20 --trials 3 \
  --fa3-num-splits 0 --fused-max-splits 8 \
  --target-kv-layout logical --cuda-graph \
  --output /tmp/attncp_decode_paths_b45_global32768_logical_20260626.json
```

Result for global KV `32768`, local CP KV `16384`, batch `45`:

| path | median |
|---|---:|
| `target_sharded_fullq` | `1238.47 us` |
| `target_sharded_slice_a2a` | `1251.97 us` |
| `target_sharded_fused_q_fa` | `383.72 us` |
| `target_sharded_fused_slice_a2a` | `397.16 us` |

Small-KV probe:

```bash
CUDA_VISIBLE_DEVICES=4,5 PYTHONPATH=python .venv/bin/torchrun \
  --standalone --nproc_per_node=2 \
  benchmark/kernels/attention/bench_attncp_decode_paths.py \
  --batch-size 45 --kv-lens 1024 --tp-size 4 --cp-size 2 \
  --cp-kv-chunk-size 1024 --warmup 5 --iters 50 --trials 3 \
  --fa3-num-splits 0 --fused-max-splits 8 \
  --target-kv-layout logical --cuda-graph \
  --output /tmp/attncp_decode_paths_b45_global1024_logical_20260626.json
```

Result for local KV `1024`, batch `45`:

| path | median |
|---|---:|
| `target_sharded_fullq` | `126.08 us` |
| `target_sharded_slice_a2a` | `139.01 us` |
| `target_sharded_fused_q_fa` | `70.63 us` |
| `target_sharded_fused_slice_a2a` | `81.12 us` |

Interpretation:

- The fused kernel is locally faster in the attention-only benchmark.
- End-to-end service still does not improve, so the bottleneck is either outside
  this isolated attention region, or the service hit distribution/layer mix
  reduces the effective benefit.
- WeLM v4 has 25 full-window layers and 23 `512`-window layers in the tested
  48-layer model. With `MIN_SEQ_CAP=16384`, only the full-window local-shard
  buckets are expected to hit fused Q+FA; the SWA buckets remain FA3 fallback.

Long-context precision probe:

- Artifact: `/tmp/attncp_hot_precision_probe_20260626_024009`
- Prompt: one deterministic `32768`-token controlled prompt.
- Decode: `max_new_tokens=4`, `temperature=0.0`, `ignore_eos=true`.
- Logprob capture: output token logprobs only
  (`logprob_start_len=32768`, `top_logprobs=0`).
- Server configs:
  - TP4 baseline: FA3 prefill/decode, WeLM kv-mirror opt, over-encoding,
    CUDA graph max bs 16.
  - TP4-CP2 exact: sharded-KV, FA3/P2P Q and O-LSE exchange, no Triton fused
    Q+FA.
  - TP4-CP2 fused-hot: same as exact plus
    `SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_FUSED_Q_FA=1` and
    `MIN_SEQ_CAP=16384`.

Output tokens:

| path | output ids | text |
|---|---|---|
| TP4 | `[78, 70, 79, 257]` | `ogp\\n\\n` |
| TP4-CP2 exact | `[78, 70, 79, 257]` | `ogp\\n\\n` |
| TP4-CP2 fused-hot | `[78, 70, 79, 257]` | `ogp\\n\\n` |

Strict output-logprob comparison:

| compare | max diff | mean diff | issues |
|---|---:|---:|---:|
| TP4 vs TP4-CP2 exact | `3.153e-02` | `1.100e-02` | 3 |
| TP4 vs TP4-CP2 fused-hot | `3.538e-01` | `9.186e-02` | 3 |
| TP4-CP2 exact vs fused-hot | `3.854e-01` | `9.671e-02` | 3 |

Notes:

- The long-context exact FA3/P2P sharded-KV path is token-identical to TP4 but
  not strict-logprob-identical at `1e-5` on this 32k prompt.
- The fused-hot path preserves output tokens for this probe, but increases
  output-logprob drift substantially.
- The fused service log confirms real hot-path hits, e.g. `seq_cap=16384
  hit_split=3` for decode layers during the 4-token generation.
- This is stronger evidence than the short-prompt full regression: current
  Triton fused Q+FA should not be treated as precision-equivalent for long
  context.

The probe has been fixed as a reusable script:

```bash
cd /home/fhkong/wxwork/attncp_precision_regression
./run_hot_fused_precision_probe.sh
```

Self-check artifact:

- `/tmp/attncp_hot_precision_probe/20260626_025138`

Self-check result:

- Token-level comparisons passed for:
  - TP4 vs TP4-CP2 exact.
  - TP4 vs TP4-CP2 fused-hot.
  - TP4-CP2 exact vs TP4-CP2 fused-hot.
- Strict logprob comparisons reproduced the current drift:
  - TP4 vs exact: max diff `3.153e-02`.
  - TP4 vs fused-hot: max diff `3.538e-01`.
  - exact vs fused-hot: max diff `3.854e-01`.
- Use `./run_hot_fused_precision_probe.sh --require-strict` after kernel fixes
  to make strict long-context logprob equality a hard gate.

Superseded numerical experiment: exp2 softmax path

- Hypothesis: FA-style kernels often use base-2 exponentiation internally, so
  replacing Triton `exp` with `exp2(score * log2(e))` might reduce FA3 diff.
- Implemented as a temporary local wrapper option and measured against FA3 on
  WeLM-like decode shapes. The change was removed after the experiment.

Results:

| shape | exp mode | time | O max diff | O mean diff | LSE max diff |
|---|---|---:|---:|---:|---:|
| local KV `16384`, full window, batch 4 | `exp` | `77.54 us` | `2.44e-04` | `2.67e-05` | `9.54e-07` |
| local KV `16384`, full window, batch 4 | `exp2` | `76.48 us` | `2.44e-04` | `2.67e-05` | `9.54e-07` |
| local KV `16384`, SWA `512`, batch 4 | `exp` | `62.94 us` | `1.95e-03` | `1.24e-04` | `4.77e-07` |
| local KV `16384`, SWA `512`, batch 4 | `exp2` | `56.19 us` | `1.95e-03` | `1.24e-04` | `3.39e-05` |

Original conclusion:

- `exp2` does not reduce O diff versus FA3.
- It makes SWA LSE diff worse.
- Do not add an exp2 mode to production code.

Superseded by later service-level hot probes:

- The early conclusion only used isolated local O/LSE comparisons and did not
  predict deterministic decode drift.
- Later 32k service hot probes showed exp2/log2 reduced exact-vs-fused strict
  output-logprob max diff from `2.847e-02` to `8.621e-03` while keeping
  token-level output identical and improving the c45 sweep throughput.
- Current experimental fused code therefore uses exp2/log2, but still does not
  satisfy the `1e-5` strict parity gate.

## 2026-06-26: KV-stationary invariant and stricter local precision gate

Added an explicit wrapper comment for `attncp_cp2_fused_q_fa_decode` documenting
the hard invariant for the fused decode path:

- The kernel must be KV-stationary.
- A program owns `(batch, kv_head[, split])`.
- All logical CP Q heads mapped to that KV head are evaluated while the K/V
  tile is resident.
- Future changes must not split this path by Q head, because that would reread
  the same resident KV shard.

Added a stricter CUDA unit test for the WeLM decode local shape:

```text
local_q_heads = 6
full_cp_q_heads = 12
local_kv_heads = 1
head_dim = 256
local_kv_len = 16384
attention sinks enabled
max_splits = 1 and 8
```

The new test compares fused local Q+FA output against FA3 directly and enforces:

```text
O max diff <= 5e-4
O mean diff <= 6e-5
LSE max diff <= 2e-5
```

Validation:

```bash
PYTHONPATH=python .venv/bin/python -m pytest \
  python/sglang/jit_kernel/tests/test_attncp_fused_ops.py::test_attncp_cp2_fused_q_fa_decode_welm_shape_strict_local -q

PYTHONPATH=python .venv/bin/python -m pytest \
  python/sglang/jit_kernel/tests/test_attncp_fused_ops.py -q
```

Result:

- strict WeLM local test: `2 passed`
- full fused-op unit suite: `30 passed`

Attention-only probe:

```bash
PYTHONPATH=python .venv/bin/python -m torch.distributed.run --standalone \
  --nproc_per_node=2 benchmark/kernels/attention/bench_attncp_decode_paths.py \
  --batch-size 4 --kv-lens 32768 --warmup 2 --iters 5 --trials 1 \
  --cuda-graph --target-kv-layout logical \
  --output /tmp/attncp_decode_paths_probe.json
```

Rank-0 result for global KV 32768 / local KV 16384:

| path | median |
|---|---:|
| target_sharded_fullq | `126.02 us` |
| target_sharded_fused_q_fa | `106.52 us` |
| target_sharded_fused_slice_a2a | `122.04 us` |

The probe still reports `fused_vs_fullq_diff_max=0.0` for output tensors. This
is only an attention-only local-path result; the long-context service logprob
probe remains the stronger end-to-end precision gate and is not fixed yet.

## 2026-06-26: CUDA graph seq-cap sensitivity

The previous graph hot-fused drift was narrowed down to CUDA graph sequence cap
sensitivity rather than a general eager-path fused math mismatch.

Probe changes:

- `/home/fhkong/wxwork/attncp_precision_regression/run_hot_fused_precision_probe.sh`
  now supports environment overrides:
  - `DISABLE_CUDA_GRAPH=1`
  - `ATTN_CP_DECODE_CUDA_GRAPH_MAX_SEQ_LEN=<N>`
  - `FUSED_MAX_SPLITS=<N>`
  - existing prompt/output length knobs.

Key results:

| config | exact vs fused token | exact vs fused strict max diff | mean diff | artifact |
|---|---:|---:|---:|---|
| `DISABLE_CUDA_GRAPH=1` | pass | `0.000e+00` | `0.000e+00` | `/tmp/attncp_hot_precision_probe/20260626_031857` |
| default graph auto cap (`seq_cap=122791`) | pass | `3.854e-01` | `9.671e-02` | `/tmp/attncp_hot_precision_probe/20260626_032452` |
| default graph, `FUSED_MAX_SPLITS=1` | pass | `1.000e-01` | `3.451e-02` | `/tmp/attncp_hot_precision_probe/20260626_033036` |
| graph cap `32768` | pass | `0.000e+00` | `0.000e+00` | `/tmp/attncp_hot_precision_probe/20260626_033816` |
| graph cap `40960` | pass | `3.742e-02` | `1.243e-02` | `/tmp/attncp_hot_precision_probe/20260626_034758` |
| graph cap `65536` | pass | `3.854e-01` | `9.671e-02` | `/tmp/attncp_hot_precision_probe/20260626_034310` |

Interpretation:

- Eager TP4-CP2 exact and fused-hot are strict identical on the 32k/4-token
  controlled probe.
- CUDA graph exact and fused-hot are strict identical when graph seq cap is
  close to the actual prompt length (`32768`).
- Larger graph caps change the fused Q+FA numerical path enough that per-layer
  BF16-level differences accumulate into visible output-logprob drift.
- `40960` is much better than auto/`65536`, but still not strict at `1e-5`.
- For strict hot-fused validation at 32k prompt, use:

```bash
ATTN_CP_DECODE_CUDA_GRAPH_MAX_SEQ_LEN=32768 \
  /home/fhkong/wxwork/attncp_precision_regression/run_hot_fused_precision_probe.sh \
  --require-strict
```

Open follow-up:

- For `32k input / 2k output`, graph cap `32768` may be too small for later
  decode steps. Need either a tighter bucket above 32768 that stays within the
  acceptable drift range, or an explicit fallback to eager/exact FA3 when the
  request grows past the strict fused graph bucket.

## 2026-06-26: Launch-grid invariant test

Added `test_attncp_cp2_fused_q_fa_decode_launch_grid_is_kv_stationary` to pin
the KV-stationary launch contract for the fused decode wrapper:

- non-split path must launch as `(batch, kv_head)`;
- split path must launch as `(batch, kv_head, split)`;
- merge can launch over `(batch, q_head)` because it only reads partial O/LSE,
  not resident K/V cache.

This prevents a future optimization from adding a Q-head program dimension to
the Q+FA kernel and accidentally rereading the same resident KV shard.

Validation:

```bash
PYTHONPATH=python .venv/bin/python -m py_compile \
  python/sglang/srt/layers/attention/attncp_fused_ops.py \
  python/sglang/srt/layers/attention/flashattention_backend.py \
  python/sglang/jit_kernel/tests/test_attncp_fused_ops.py

PYTHONPATH=python .venv/bin/python -m pytest \
  python/sglang/jit_kernel/tests/test_attncp_fused_ops.py -q
```

Result:

- `31 passed, 5 warnings`

Follow-up:

- Added `attncp_cp2_fused_q_fa_supports_shape(...)` so the fused provider only
  accepts shapes where all Q heads mapped to a KV head can be evaluated in one
  KV-stationary program.
- Current limit: `next_power_of_2(q_heads_per_kv) <= 16`.
- Unsupported shapes must fall back to the FA3 exact path; they must not add a
  Q-head program dimension, because that would reread resident K/V.
- Added source/shape tests that reject unsupported Q-head split shapes and
  verify the fused attention kernels do not use `q_head_idx = tl.program_id`.

Validation after this follow-up:

```bash
PYTHONPATH=python .venv/bin/python -m py_compile \
  python/sglang/srt/layers/attention/attncp_fused_ops.py \
  python/sglang/srt/layers/attention/flashattention_backend.py \
  python/sglang/jit_kernel/tests/test_attncp_fused_ops.py

PYTHONPATH=python .venv/bin/python -m pytest \
  python/sglang/jit_kernel/tests/test_attncp_fused_ops.py -q

git diff --check
```

Result:

- `34 passed, 5 warnings`
- `py_compile` passed.
- `git diff --check` passed.

Attention-only speed sanity after the shape guard:

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=python .venv/bin/python \
  -m torch.distributed.run --standalone --nproc_per_node=2 \
  benchmark/kernels/attention/bench_attncp_decode_paths.py \
  --batch-size 45 --kv-lens 32768 --warmup 1 --iters 3 --trials 1 \
  --cuda-graph --target-kv-layout logical --fused-max-splits 10 \
  --output /tmp/attncp_decode_paths_shape_guard_20260626.json
```

Result:

| path | median |
|---|---:|
| `target_sharded_fullq` | `1286.987 us` |
| `target_sharded_fused_q_fa` | `458.176 us` |
| `target_sharded_slice_a2a` | `1262.059 us` |
| `target_sharded_fused_slice_a2a` | `437.120 us` |

Diff summary:

- `fused_vs_fullq_diff_max=0.000000` in the benchmark's final merged BF16
  output sample.
- Local partial path max differences remain BF16-ulp level
  (`fused_q_fa_diff_max=0.000244`).

## 2026-06-26: Current attention-only speed check after invariant test

Re-ran the CP2 decode-path microbenchmark after adding the launch-grid
invariant test. This did not change runtime code; the point was to refresh the
local speed evidence for the current worktree.

Command:

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=python .venv/bin/python \
  -m torch.distributed.run --standalone --nproc_per_node=2 \
  benchmark/kernels/attention/bench_attncp_decode_paths.py \
  --batch-size 45 --kv-lens 32768 --warmup 1 --iters 3 --trials 1 \
  --cuda-graph --target-kv-layout logical --fused-max-splits 10 \
  --output /tmp/attncp_decode_paths_current_goal.json
```

Shape:

- WeLM v4 H20 attention-only benchmark.
- TP4-CP2 logical target, CP world size 2.
- Global KV len `32768`, local CP KV len `16384`.
- Batch size `45`, local Q heads `6`, full CP Q heads `12`, head dim `256`.
- Attention sinks enabled, CUDA graph enabled.

Result:

| path | median |
|---|---:|
| `target_sharded_fullq` | `1252.757 us` |
| `target_sharded_fused_q_fa` | `422.837 us` |
| `target_sharded_slice_a2a` | `1263.595 us` |
| `target_sharded_fused_slice_a2a` | `434.507 us` |

Diffs:

- `fused_q_fa_vs_fullq_max_abs_diff = 0.0`
- `fused_q_fa_max_abs_diff = 0.000244140625`

Interpretation:

- The Triton fused Q+FA prototype still has a large attention-only speed
  advantage over full-Q + FA3 local attention for this shape (`~2.96x` for the
  local attention replacement boundary).
- This does not prove service-level strict logprob parity or end-to-end TPS
  improvement. The known fused-hot service strict gap remains unresolved.

## 2026-06-26: Fused max seq-cap guard and strict bucket probe

Added an optional runtime guard:

```text
SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_FUSED_Q_FA_MAX_SEQ_CAP
```

When this value is greater than zero, the experimental fused Q+FA path only
hits when the CUDA graph/local page-table seq cap is not larger than the guard.
Wider buckets fall back to the exact FA3 local-merge path. This does not make
the Triton provider globally strict-equivalent, but it gives precision tests a
way to constrain fused-hot execution to buckets already verified against the
exact AttnCP path.

Also updated `/home/fhkong/wxwork/attncp_precision_regression/run_hot_fused_precision_probe.sh`:

- added `FUSED_MAX_SEQ_CAP`;
- added `--require-fused-strict`, which only gates `TP4-CP2 exact` vs
  `TP4-CP2 fused-hot`;
- kept `--require-strict` as the stronger all-pairs gate, including the known
  TP4 vs TP4-CP2 local-merge reduction-order logprob difference.

Probe:

```bash
ATTN_CP_DECODE_CUDA_GRAPH_MAX_SEQ_LEN=32768 \
FUSED_MAX_SEQ_CAP=32768 \
  /home/fhkong/wxwork/attncp_precision_regression/run_hot_fused_precision_probe.sh \
  --require-strict
```

Artifact:

```text
/tmp/attncp_hot_precision_probe/20260626_074611
```

Result:

| compare | token | strict max diff | strict result |
|---|---:|---:|---|
| `exact_vs_fused` | pass | `0.000e+00` | pass |
| `tp4_vs_exact` | pass | `3.153e-02` | fail |
| `tp4_vs_fused` | pass | `3.153e-02` | fail |

Fused-hot log confirmed real fused execution in the constrained bucket:

```text
seq_cap=32768 hit_split=3
```

Interpretation:

- For this `32768` graph bucket, the Triton fused Q+FA path is strict-identical
  to the exact TP4-CP2 AttnCP path on the 32k/4-token controlled probe.
- The script failed only because `--require-strict` still requires TP4 vs
  TP4-CP2 strict equality, which is a known non-goal for the current
  local-merge AttnCP path.
- Future kernel-fusion validation should use `--require-fused-strict` when the
  question is whether fused Q+FA preserves the original TP4-CP2 AttnCP path.

Follow-up gate run with the corrected script:

```bash
ATTN_CP_DECODE_CUDA_GRAPH_MAX_SEQ_LEN=32768 \
FUSED_MAX_SEQ_CAP=32768 \
  /home/fhkong/wxwork/attncp_precision_regression/run_hot_fused_precision_probe.sh \
  --require-fused-strict
```

Artifact:

```text
/tmp/attncp_hot_precision_probe/20260626_075425
```

Exit status: `0`.

Summary:

| compare | token | strict max diff | strict result |
|---|---:|---:|---|
| `exact_vs_fused` | pass | `0.000e+00` | pass |
| `tp4_vs_exact` | pass | `3.153e-02` | expected strict fail |
| `tp4_vs_fused` | pass | `3.153e-02` | expected strict fail |

Fused profile evidence:

- `hit_split` lines: `6912`
- `fallback:seq_cap_high` lines: `768`
- Initial CUDA graph buckets at `seq_cap=32768` hit fused Q+FA.
- Later decode steps reached `seq_cap=32772` and correctly fell back to exact
  FA3 due to `FUSED_MAX_SEQ_CAP=32768`.

Interpretation:

- The corrected `--require-fused-strict` gate now passes for the constrained
  bucket.
- This verifies that the experimental Triton Q+FA provider can replace Q
  exchange + local FA3 without changing the exact TP4-CP2 AttnCP path in the
  covered bucket.
- It does not verify long-output performance buckets that must keep hitting
  fused beyond 32768; those still require either strict math fixes or accepting
  the experimental token-level-only risk.

## 2026-06-26: Full precision regression and 32k/2k sweep

Full TP4 vs TP4-CP2 sharded-KV precision regression:

```bash
/home/fhkong/wxwork/attncp_precision_regression/run_full_precision.sh
```

Artifact:

```text
/tmp/welmv4_attncp_precision/20260626_080119
```

Result:

- controlled TP4 vs TP4-CP2 token-level compare: pass, max diff `0.000e+00`;
- MMLU/C-Eval regression: `100 / 100` samples, `0 / 100` token mismatches;
- max logprob diff `0.00e+00`, mean logprob diff `0.00e+00`;
- final result: PASS.

This full regression uses the precision-safe default path. It verifies that the
experimental fused Q+FA code and integration guards do not break the default
AttnCP sharded-KV precision path.

32k input / 2k output sweep:

```bash
cd /home/fhkong/wxwork/attncp_tp4_cp2_sweep
bash ./start_naive_tp4.sh
bash ./sweep_concurrency.sh tp4 22 23 24
bash ./start_tp4_cp2.sh
bash ./sweep_concurrency.sh tp4_cp2 44 45 46
```

Artifacts:

```text
TP4:     /tmp/welmv4_attncp_manual_sweep/20260626_080823_tp4_s32768_o2048/summary.tsv
TP4-CP2: /tmp/welmv4_attncp_manual_sweep/20260626_081638_tp4_cp2_s32768_o2048/summary.tsv
```

TP4:

| concurrency | output TPS | mean TTFT ms | mean ITL ms | peak running | max token usage |
|---:|---:|---:|---:|---:|---:|
| 22 | `433.3889` | `17380.6572` | `42.2927` | 22 | `0.94` |
| 23 | `449.5380` | `17724.4797` | `42.5263` | 23 | `0.98` |
| 24 | `403.6989` | `21708.6728` | `41.0311` | 23 | `0.98` |

TP4-CP2 fused-QFA experimental performance path:

| concurrency | output TPS | mean TTFT ms | mean ITL ms | peak running | max token usage |
|---:|---:|---:|---:|---:|---:|
| 44 | `634.1892` | `36894.9660` | `51.3955` | 44 | `0.96` |
| 45 | `650.2253` | `36158.3472` | `51.5817` | 45 | `0.98` |
| 46 | `555.4520` | `38638.5180` | `50.7680` | 45 | `0.98` |

Conclusion:

- Best resident concurrency improved from TP4 `23` to TP4-CP2 `45`, about
  `1.96x`.
- Best output TPS improved from TP4 `449.5380` to TP4-CP2 `650.2253`, about
  `1.45x`.
- TP4-CP2 c46 accepted 46 prompts but peak server running stayed at `45`,
  showing the expected capacity boundary.
- The TP4-CP2 sweep used `FUSED_MAX_SEQ_CAP=0` and graph cap `40960`, so it is
  the experimental long-output fused performance path. It is not the
  strict-safe constrained bucket from the `--require-fused-strict` probe.

## 2026-06-26: Current Worktree Precision and Throughput Refresh

This refresh was run after adding the KV-stationary shape guard. The code path
for WeLM's `q_heads_per_kv=12` remains supported, while unsupported larger
GQA shapes now fall back to the FA3 exact path instead of splitting by Q head.

Focused unit/static checks:

```bash
PYTHONPATH=python .venv/bin/python -m pytest \
  python/sglang/jit_kernel/tests/test_attncp_fused_ops.py -q

PYTHONPATH=python .venv/bin/python -m py_compile \
  python/sglang/srt/layers/attention/attncp_fused_ops.py \
  python/sglang/srt/layers/attention/flashattention_backend.py \
  python/sglang/jit_kernel/tests/test_attncp_fused_ops.py

git diff --check
```

Result:

- `34 passed, 5 warnings`
- `py_compile` passed.
- `git diff --check` passed.

Full TP4 vs TP4-CP2 sharded-KV precision regression:

```bash
/home/fhkong/wxwork/attncp_precision_regression/run_full_precision.sh
```

Artifact:

```text
/tmp/welmv4_attncp_precision/20260626_083732
```

Result:

- controlled TP4 vs TP4-CP2 token-level compare: pass, max diff `0.000e+00`;
- MMLU/C-Eval regression: `100 / 100` samples, `0 / 100` token mismatches;
- max logprob diff `0.00e+00`, mean logprob diff `0.00e+00`;
- final result: PASS.

Constrained fused-hot strict probe:

```bash
ATTN_CP_DECODE_CUDA_GRAPH_MAX_SEQ_LEN=32768 \
FUSED_MAX_SEQ_CAP=32768 \
  /home/fhkong/wxwork/attncp_precision_regression/run_hot_fused_precision_probe.sh \
  --require-fused-strict
```

Artifact:

```text
/tmp/attncp_hot_precision_probe/20260626_084227
```

Result:

- outputs all matched: `[78, 70, 79, 257]`, text `'ogp\n\n'`;
- `exact_vs_fused_token`: pass, max diff `0.000e+00`;
- `exact_vs_fused_strict`: pass, max diff `0.000e+00`;
- `tp4_vs_exact_strict` and `tp4_vs_fused_strict`: fail with the known
  local-merge reduction-order diff, max diff `3.153e-02`;
- fused profile in `tp4_cp2_fused_hot.log`: `hit_split=6912`,
  `fallback:seq_cap_high=768`.

Interpretation:

- The constrained bucket proves that the Triton fused provider can replace
  Q exchange + local FA for the covered `32768` seq-cap bucket without changing
  the exact TP4-CP2 AttnCP result.
- Later decode buckets beyond `32768` still fall back to exact FA3 in this
  strict-safe configuration.

32k input / 2k output throughput sweep on the same current worktree:

```bash
cd /home/fhkong/wxwork/attncp_tp4_cp2_sweep
./start_naive_tp4.sh
./sweep_concurrency.sh tp4 22 23 24
./start_tp4_cp2.sh
./sweep_concurrency.sh tp4_cp2 44 45 46
```

Artifacts:

```text
TP4:     /tmp/welmv4_attncp_manual_sweep/20260626_084914_tp4_s32768_o2048/summary.tsv
TP4-CP2: /tmp/welmv4_attncp_manual_sweep/20260626_085745_tp4_cp2_s32768_o2048/summary.tsv
```

TP4:

| concurrency | output TPS | mean TTFT ms | mean ITL ms | peak running | max token usage |
|---:|---:|---:|---:|---:|---:|
| 22 | `431.3825` | `17347.4139` | `42.5457` | 22 | `0.94` |
| 23 | `447.3388` | `18066.7170` | `42.6090` | 23 | `0.98` |
| 24 | `403.7897` | `22043.8594` | `40.8576` | 23 | `0.98` |

TP4-CP2 fused-QFA experimental performance path:

| concurrency | output TPS | mean TTFT ms | mean ITL ms | peak running | max token usage |
|---:|---:|---:|---:|---:|---:|
| 44 | `631.1834` | `37932.0732` | `51.2187` | 44 | `0.96` |
| 45 | `648.0209` | `36656.2451` | `51.5752` | 45 | `0.98` |
| 46 | `554.7051` | `38804.5494` | `50.8630` | 45 | `0.98` |

Conclusion:

- Best resident concurrency: TP4 `23` vs TP4-CP2 `45`, about `1.96x`.
- Best output TPS: TP4 `447.3388` vs TP4-CP2 `648.0209`, about `1.45x`.
- Best-point ITL: TP4 `42.6090 ms` vs TP4-CP2 `51.5752 ms`, about `1.21x`
  slower per token.
- Best-point TTFT: TP4 `18066.7170 ms` vs TP4-CP2 `36656.2451 ms`, about
  `2.03x` slower per request in this full 32k prefill workload.
- TP4-CP2 c46 accepted 46 prompts but peak server running stayed at `45`,
  showing the expected capacity boundary.
- The TP4-CP2 sweep used `FUSED_MAX_SEQ_CAP=0` and graph cap `40960`, so it is
  still the experimental long-output fused performance path, not the
  strict-safe constrained bucket.

Strict-safe long-output throughput check:

```bash
cd /home/fhkong/wxwork/attncp_tp4_cp2_sweep
ATTN_CP_DECODE_CUDA_GRAPH_MAX_SEQ_LEN=32768 \
FUSED_MAX_SEQ_CAP=32768 \
  ./start_tp4_cp2.sh
./sweep_concurrency.sh tp4_cp2 44 45 46
```

Artifact:

```text
/tmp/welmv4_attncp_manual_sweep/20260626_091032_tp4_cp2_s32768_o2048/summary.tsv
```

Result:

| concurrency | output TPS | mean TTFT ms | mean ITL ms | peak running | cuda graph seen |
|---:|---:|---:|---:|---:|---|
| 44 | `447.6849` | `37549.7235` | `80.0105` | 44 | no |
| 45 | `458.9742` | `36518.0913` | `80.2765` | 45 | no |
| 46 | `262.0226` | `40384.2232` | `80.3800` | 45 | no |

Comparison of current best points:

| path | strict status | best concurrency | output TPS | mean ITL ms | cuda graph seen |
|---|---|---:|---:|---:|---|
| TP4 baseline | exact baseline | 23 | `447.3388` | `42.6090` | yes |
| TP4-CP2 constrained fused | strict-safe vs exact CP2 | 45 | `458.9742` | `80.2765` | no |
| TP4-CP2 long-output fused | experimental token-level path | 45 | `648.0209` | `51.5752` | yes |

Interpretation:

- The constrained fused path preserves the strict fused-vs-exact CP2 result in
  the covered bucket, but it is not a viable long-output performance answer:
  with `32768` cap, the 32k/2k sweep falls out of the captured decode graph and
  regresses to about TP4-level output throughput with much worse ITL.
- The performance result therefore still depends on the long-output fused path
  (`FUSED_MAX_SEQ_CAP=0`, graph cap `40960`), which is token-level correct in
  the tested workload but not strict-logprob equivalent to FA3 exact.
- This closes the simple guard-based route. To make the fused path both
  strict-parity and high-throughput for long output, the next credible
  implementation target is the FA3/CUDA internal Q-provider: keep FA3's
  mainloop/softmax/sink/split/output math intact, but source logical Q heads
  from local/peer Q pointers instead of materializing full Q outside the kernel.

## 2026-06-26: FA3 Internal Q-Provider Python Scaffold

Inspected the current `sgl-kernel` integration:

- `sgl-kernel/CMakeLists.txt` fetches FA3 from `sgl-project/sgl-attn` at
  `bcf72ccc6816b36a5fae2c5a3c027604629785e0` via CMake `FetchContent`.
- The local repo does not track the FA3 Hopper mainloop sources directly.
- `sgl-kernel/csrc/flash_extension.cc` registers one generic
  `sgl_kernel::fwd` op. Its schema has a single `q` tensor and no peer-Q
  provider fields.
- The Python wrappers route through `sgl-kernel/python/sgl_kernel/flash_attn.py`
  and `python/sglang/jit_kernel/flash_attention_v3.py`.

Added a repo-local Python API scaffold for the future strict-parity provider:

- `sgl-kernel/python/sgl_kernel/flash_attn.py`
  - `has_flash_attn_with_kvcache_cp2_q_provider()`
  - `flash_attn_with_kvcache_cp2_q_provider(...)`
- `python/sglang/jit_kernel/flash_attention_v3.py`
  - same availability probe and wrapper
- `python/sglang/jit_kernel/flash_attention.py`
  - public forwarding helpers with `ver=3`

The planned op name is:

```text
sgl_kernel.fwd_attncp_cp2_q_provider
```

The wrapper accepts `q_local` and `q_peer` directly, plus normal paged KV cache
inputs, `cp_rank`, scheduler metadata, sinks, split settings, and optional
preallocated output. Its intended output contract is the same local partial
state that the AttnCP decode local-merge path consumes today.

Important status:

- The C++/CUDA op is not registered by the current `sgl-attn` pin.
- Therefore `has_flash_attn_with_kvcache_cp2_q_provider()` currently returns
  false in a built environment, and calling the wrapper raises
  `NotImplementedError`.
- No service path uses this scaffold yet; current precision/performance
  behavior is unchanged.

Validation:

```bash
PYTHONPATH=python .venv/bin/python -m py_compile \
  sgl-kernel/python/sgl_kernel/flash_attn.py \
  python/sglang/jit_kernel/flash_attention_v3.py \
  python/sglang/jit_kernel/flash_attention.py \
  python/sglang/srt/layers/attention/attncp_fused_ops.py \
  python/sglang/srt/layers/attention/flashattention_backend.py \
  python/sglang/jit_kernel/tests/test_attncp_fused_ops.py

PYTHONPATH=python .venv/bin/python - <<'PY'
from sglang.jit_kernel.flash_attention import has_flash_attn_with_kvcache_cp2_q_provider
print(has_flash_attn_with_kvcache_cp2_q_provider())
PY
```

Result:

- `py_compile` passed.
- provider availability probe returned `False`, as expected for the current
  unpatched `sgl-attn` build.
- direct wrapper call raises `NotImplementedError` before any CUDA launch.

Remaining implementation work to make this usable:

1. Add a new C++ op schema in `sgl-kernel/csrc/flash_extension.cc` or extend
   `sgl_kernel::fwd` with optional Q-provider fields.
2. Patch/fork `sgl-attn` so `Flash_fwd_params` carries `q_peer_ptr`,
   `local_q_heads`, and `cp_rank`.
3. In the FA3 Q-load path, choose local or peer Q pointer by logical Q head,
   while leaving K/V page-table traversal, TMA/cp.async, QK MMA, softmax,
   attention sink, split combine, and output rounding unchanged.
4. Rebuild `sgl-kernel` against the patched `sgl-attn`.
5. Switch `_attncp_try_fused_q_fa_decode(...)` to prefer this provider when
   available, falling back to the current Triton prototype or exact FA3 path.

## 2026-06-26: KV-Stationary Guard Tightening

User requirement: the fused decode path must guarantee resident K/V is read
only once per owned KV tile.

Code updates:

- `attncp_cp2_fused_q_fa_decode()` now uses explicit runtime guards instead of
  relying on `assert` for the KV-stationary shape contract.
- Unsupported `full_q_heads_per_kv` layouts now raise a `ValueError` explaining
  that the fused path must fallback instead of splitting by Q head.
- The existing source-level test still checks that both fused attention kernels:
  - do not derive work from a Q-head program id.
  - contain one textual `tl.load(key_cache...)` and one `tl.load(value_cache...)`.
- The launch-grid test still requires:
  - non-split grid: `(batch, kv_head)`.
  - split grid: `(batch, kv_head, split)`.
  - merge grid may use `(batch, q_head)` because it reads only partial `O/LSE`,
    not K/V.

Validation:

```bash
PYTHONPATH=python .venv/bin/python -m py_compile \
  python/sglang/srt/layers/attention/attncp_fused_ops.py \
  python/sglang/jit_kernel/tests/test_attncp_fused_ops.py

PYTHONPATH=python .venv/bin/python -m pytest \
  python/sglang/jit_kernel/tests/test_attncp_fused_ops.py -q
```

Result:

- `py_compile` passed.
- `34 passed, 5 warnings in 31.63s`.

FA3 source inspection update:

- Current `sgl-attn` Hopper path has two Q load modes:
  - TMA-Q for non-packed Q.
  - `PackGQAManager::load_Q()` cp.async path for PackGQA.
- Decode/GQA/split configurations commonly enable `PackGQA`; in that path Q row
  pointers are already computed per logical GQA row before the copy into smem.
- Therefore the most practical strict-parity provider hook is to extend the
  PackGQA Q loader to select between `q_local` and `q_peer` by logical CP head.
  K/V TMA/page-table traversal and the rest of FA3 mainloop remain unchanged.
- TMA-Q provider support is harder because a TMA descriptor describes one
  contiguous Q tensor. For the first strict-parity implementation, the safer
  constraint is to force the CP2 provider path into PackGQA/cp.async Q loading.

## 2026-06-26: 40960 Hot Probe and Capacity-Mismatch Unit Check

Re-ran the hot precision probe with the long-output graph cap and the current
recommended split count:

```bash
ATTN_CP_DECODE_CUDA_GRAPH_MAX_SEQ_LEN=40960 \
FUSED_MAX_SEQ_CAP=40960 \
FUSED_MAX_SPLITS=10 \
  /home/fhkong/wxwork/attncp_precision_regression/run_hot_fused_precision_probe.sh \
  --require-fused-strict
```

Artifact:

```text
/tmp/attncp_hot_precision_probe/20260626_094247
```

Result:

| compare | mode | passed | max diff | mean diff | issues |
|---|---|---:|---:|---:|---:|
| exact_vs_fused_token | token_level | true | `0.000e+00` | `0.000e+00` | 0 |
| exact_vs_fused_strict | strict_logprob | false | `8.621e-03` | `2.549e-03` | 3 |

The fused log confirms this is a real fused-path result, not fallback:

- graph capture: `batch_size=16 seq_cap=40960 hit_split=1`
- request decode: `batch_size=1 seq_cap=16384 hit_split=3`

Conclusion:

- The strict gap is not caused by the old `FUSED_MAX_SPLITS=8` setting.
- The current `40960`/`10` split setup still has the same strict
  exact-vs-fused drift.
- Token-level output remains identical for this probe.

Added a focused unit-test variant for the suspected bucket-cap mismatch:

- `page_table_cap=16384`, `cache_seqlens=16384`
- `page_table_cap=40960`, `cache_seqlens=16384`

Command:

```bash
PYTHONPATH=python .venv/bin/python -m pytest \
  python/sglang/jit_kernel/tests/test_attncp_fused_ops.py::test_attncp_cp2_fused_q_fa_decode_welm_shape_strict_local -q
```

Result:

- `4 passed, 5 warnings in 12.93s`.

Interpretation:

- A padded page table / graph bucket capacity larger than runtime local KV
  length does not by itself reproduce a local FA3-vs-Triton correctness bug.
- The remaining service-level strict drift is consistent with per-layer
  BF16-level Triton-vs-FA3 attention math differences accumulating through the
  full model.
- Continue treating the Triton Q+FA path as token-level experimental only. For
  strict logprob parity, the next implementation target should be the FA3
  internal Q-provider path.

Added an opt-in `sgl-kernel` CMake source override to make that next step
buildable from this repo without editing transient `FetchContent` directories:

```bash
cmake ... -DSGL_KERNEL_FLASH_ATTENTION_SOURCE_DIR=/path/to/patched/sgl-attn
```

Default behavior is unchanged: without this cache variable, `sgl-kernel` still
fetches `sgl-project/sgl-attn` at
`bcf72ccc6816b36a5fae2c5a3c027604629785e0`. With the override set, CMake checks
that the provided tree contains `hopper/flash_api.cpp` and uses it for FA3
sources. This gives the future CP2 Q-provider patch a reproducible build entry
instead of relying on `/tmp/.../repo-flash-attention-src` edits.

Also wired the SGLang decode replacement boundary to prefer the future FA3
provider when it is available:

- `FlashAttentionBackend.__init__` caches
  `has_flash_attn_with_kvcache_cp2_q_provider()`.
- `_attncp_try_fused_q_fa_decode(...)` now first calls
  `flash_attn_with_kvcache_cp2_q_provider(...)` when the op is registered.
- The provider call reuses the same replacement boundary as the Triton
  prototype: `q_local`, exchanged `q_peer`, resident local KV cache, local page
  table, local cache lengths, attention sinks, and output workspace.
- It passes `pack_gqa=True` deliberately; the strict provider is expected to
  hook the FA3 PackGQA Q loader so logical CP Q heads choose local or peer Q
  without materializing full Q outside FA3.
- If the op is not registered, current behavior is unchanged: the path falls
  through to the existing KV-stationary Triton fused Q+FA prototype.

This means a future patched `sgl-attn` build can be validated without another
Python/SGLang integration change.

## 2026-06-26: C++ Op Registration Shell for FA3 Provider

Added a default-off sgl-kernel build switch:

```bash
-DSGL_KERNEL_ENABLE_ATTNCP_CP2_Q_PROVIDER=ON
```

When this switch is off, the current build remains unchanged and
`has_flash_attn_with_kvcache_cp2_q_provider()` returns false.

When the switch is on, `flash_ops` compiles with
`SGL_KERNEL_ENABLE_ATTNCP_CP2_Q_PROVIDER` and registers:

```text
sgl_kernel::fwd_attncp_cp2_q_provider
```

The switch must be paired with
`-DSGL_KERNEL_FLASH_ATTENTION_SOURCE_DIR=/path/to/patched/sgl-attn`; otherwise
CMake fails at configure time. This avoids a later link-time unresolved symbol
against the unpatched pinned `sgl-attn` source.

Files touched:

- `sgl-kernel/CMakeLists.txt`
  - adds `SGL_KERNEL_ENABLE_ATTNCP_CP2_Q_PROVIDER`.
  - forwards the compile definition to `flash_ops` only when enabled.
- `sgl-kernel/include/sgl_flash_kernel_ops.h`
  - declares `mha_fwd_attncp_cp2_q_provider(...)` behind the same macro.
- `sgl-kernel/csrc/flash_extension.cc`
  - registers the torch op schema and CUDA impl behind the same macro.

Important: enabling this switch requires the local/patched `sgl-attn` source to
provide `mha_fwd_attncp_cp2_q_provider(...)`. If the patched source does not
define that function, the build should fail clearly instead of exposing a fake
provider.

Added a lightweight Python contract test:

- current unpatched build: provider availability probe is false;
- direct wrapper call raises `NotImplementedError`;
- if a future build registers the provider, this absence test skips.

## 2026-06-26: Scope Reset to Pure Triton Q+FA Fusion

The FA3 internal Q-provider exploration above is not part of the current code
path. The current implementation has been reset to the pure Triton target:

- no `sgl-kernel` source changes;
- no `sgl-attn` local source override;
- no `flash_attn_with_kvcache_cp2_q_provider(...)` Python wrapper;
- `_attncp_try_fused_q_fa_decode(...)` only chooses between the Triton fused
  Q+FA path and the existing FA3 fallback path.

This matches the implementation goal for this round: keep a clean replacement
boundary for decode-side Q exchange + local attention, implement the fused
kernel in Triton, and validate speed before considering any CUDA/FA3 provider.

Validation after reset:

```bash
PYTHONPATH=python .venv/bin/python -m py_compile \
  python/sglang/srt/layers/attention/flashattention_backend.py \
  python/sglang/srt/layers/attention/attncp_fused_ops.py \
  python/sglang/jit_kernel/tests/test_attncp_fused_ops.py \
  python/sglang/jit_kernel/flash_attention.py \
  python/sglang/jit_kernel/flash_attention_v3.py \
  sgl-kernel/python/sgl_kernel/flash_attn.py

git diff --check

PYTHONPATH=python .venv/bin/python -m pytest \
  python/sglang/jit_kernel/tests/test_attncp_fused_ops.py -q
```

Result:

```text
36 passed, 5 warnings in 13.19s
```

Focused speed probe:

```bash
PYTHONPATH=python CUDA_VISIBLE_DEVICES=0,1 .venv/bin/torchrun \
  --standalone --nproc_per_node=2 \
  benchmark/kernels/attention/bench_attncp_decode_paths.py \
  --output /tmp/attncp_decode_paths_triton_probe_b16_graph.json \
  --batch-size 16 \
  --kv-lens 32768 \
  --tp-size 4 \
  --cp-size 2 \
  --cp-kv-chunk-size 1024 \
  --warmup 3 \
  --iters 20 \
  --trials 2 \
  --fused-max-splits 8 \
  --cuda-graph
```

Key result for `batch_size=16, kv_len=32768, local_kv_len=16384`:

| path | median us | fused/fullq speedup | max diff |
|---|---:|---:|---:|
| `target_sharded_fullq` | `527.027` | baseline | `2.44e-04` |
| `target_sharded_fused_q_fa` | `175.360` | `3.01x` | `2.44e-04` |

The same non-CUDA-graph probe showed:

| batch | kv_len | fullq median us | fused median us | fused/fullq speedup | max diff |
|---:|---:|---:|---:|---:|---:|
| 4 | 8192 | `254.514` | `238.434` | `1.07x` | `2.44e-04` |
| 4 | 32768 | `246.251` | `260.723` | `0.94x` | `2.44e-04` |
| 16 | 8192 | `261.642` | `246.818` | `1.06x` | `4.88e-04` |
| 16 | 32768 | `545.013` | `268.408` | `2.03x` | `2.44e-04` |

Interpretation:

- Triton fusion is useful when batch/sequence is large enough for Q exchange
  and launch overhead to matter.
- It is not uniformly faster for small batch or short local KV; the service
  integration must keep the existing seq-cap/shape guard and fallback.
- The current validation target is token-level / attention-output tolerance.
  Strict long-output logprob parity remains a separate risk and is not solved by
  this pure Triton path.

## 2026-06-26: Clean Triton Fusion Service Validation

After the pure Triton scope reset, I reran the end-to-end validation on the
current `perf/welm-v4-optimization` workspace.

Precision:

```text
Full precision regression artifact:
  /tmp/welmv4_attncp_precision/20260626_104151

Controlled compare:
  token-level PASS
  max_logprob_diff = 0.0
  mean_logprob_diff = 0.0

MMLU/C-Eval 100-sample regression:
  token mismatches = 0 / 100
  max logprob diff = 0.00e+00
  mean logprob diff = 0.00e+00
  result = PASS
```

Hot fused 32k probe:

```text
Artifact:
  /tmp/attncp_hot_precision_probe/20260626_104638

Output tokens:
  TP4                 [78, 70, 79, 257]
  TP4-CP2 exact       [78, 70, 79, 257]
  TP4-CP2 fused-hot   [78, 70, 79, 257]

exact_vs_fused_token:
  PASS

exact_vs_fused_strict:
  FAIL
  max diff  = 2.128e-01
  mean diff = 5.949e-02
```

This confirms the current Triton fused path is still token-level experimental,
not a strict-logprob-equivalent replacement for FA3 exact attention math.

Service sweep setup:

```text
Model:
  /home/fhkong/models/80a3_v4d5_256k_merge_thinking_kimi_k25_0502_20260503_032335/epoch_003_step_0002610

GPUs:
  CUDA_VISIBLE_DEVICES=4,5,6,7

Common args:
  --tp 4
  --mem-fraction-static 0.8
  --page-size 1
  --chunked-prefill-size 8192
  --prefill-attention-backend fa3
  --decode-attention-backend fa3
  --enable-over-encoding
  --enable-welm-kv-mirror-opt
  --disable-radix-cache
  --cuda-graph-max-bs 128

TP4-CP2 extra args/env:
  --attn-cp-size 2
  --attn-cp-mode sharded-kv
  --attn-cp-decode-cuda-graph-max-seq-len 40960
  --attn-cp-kv-chunk-size 1024
  SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_Q_P2P=1
  SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_OLSE_P2P=1
  SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_FUSED_Q_FA=1
  SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_FUSED_Q_FA_MIN_SEQ_CAP=16384
  SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_FUSED_Q_FA_MAX_SEQ_CAP=0
  SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_FUSED_Q_FA_MAX_SPLITS=10

Benchmark:
  random-ids, input_len=32768, output_len=2048, request_rate=inf,
  num_prompts=max_concurrency=concurrency, warmup_requests=0
```

Service sweep results:

| config | concurrency | output TPS | mean TTFT ms | mean ITL ms | cuda graph |
|---|---:|---:|---:|---:|---|
| TP4 | 22 | `430.65` | `17947.19` | `42.34` | yes |
| TP4 | 23 | `449.54` | `18445.21` | `42.17` | yes |
| TP4 | 24 | `403.97` | `21688.49` | `41.01` | yes |
| TP4-CP2 exact FA3/P2P | 45 | `658.85` | `37898.35` | `49.83` | yes |
| TP4-CP2 fused enabled | 44 | `632.21` | `37479.67` | `51.33` | yes |
| TP4-CP2 fused enabled | 45 | `650.49` | `36187.07` | `51.51` | yes |
| TP4-CP2 fused enabled | 46 | `555.01` | `38840.91` | `50.74` | yes |

Artifacts:

```text
/tmp/welmv4_attncp_manual_sweep/20260626_105331_tp4_s32768_o2048/summary.tsv
/tmp/welmv4_attncp_manual_sweep/20260626_110156_tp4_cp2_s32768_o2048/summary.tsv
/tmp/welmv4_attncp_manual_sweep/20260626_111851_tp4_cp2_exact_s32768_o2048/summary.tsv
/tmp/welmv4_attncp_manual_sweep/20260626_105955_tp4_cp2/server.log
```

The TP4-CP2 server log printed the experimental Triton fused Q+FA warning on
all four ranks, so the fused path was enabled for this run.

Conclusion:

- Best resident concurrency remains `23` for TP4 and `45` for TP4-CP2.
- TP4-CP2 exact FA3/P2P c45 improves output throughput from `449.54 tok/s`
  to `658.85 tok/s`, about `1.466x` or `+46.6%`.
- TP4-CP2 fused-enabled c45 reaches `650.49 tok/s`, about `1.447x` or
  `+44.7%` over TP4, but is slower than the same-code exact FA3/P2P c45.
- TP4-CP2 exact c45 ITL is `49.83 ms`, about `18.2%` slower than TP4 best
  `42.17 ms`; fused-enabled c45 ITL is `51.51 ms`.
- The current end-to-end `~46%` throughput gain should be attributed to
  sharded-KV capacity plus CP2 P2P exchange, not to Triton Q+FA fusion.
- The clean Triton fusion has strong isolated-kernel speedup, but does not yet
  close the service-level ITL gap.
- Because hot fused strict logprob still fails, keep Triton fused Q+FA
  experimental and guarded. The default precision-safe path should remain FA3
  exact attention math plus CP2 P2P exchange.

## 2026-06-26 local page-table cap follow-up

Change:

- Extracted `attncp_sharded_kv_local_cap(...)` and used it when building
  sharded-KV decode metadata.
- The local compact page table is now capped by CP-owned chunk distribution
  instead of the global CUDA graph sequence bucket.
- Example: global graph cap `40960`,CP2,`cp_kv_chunk_size=1024` now gives
  local cap `20480` per CP rank.
- Added CPU unit coverage for chunk owner distribution and tail handling.

Why it matters:

- The fused Q+FA kernel splits work by local page-table capacity.
- Keeping the global cap made the fused path spend time on empty split ranges.
- Focused benchmark confirmed the overhead:

| page-table cap | `target_sharded_fused_slice_a2a` median |
|---:|---:|
| `40960` | `733.331 us` |
| `20480` | `435.117 us` |

Validation after the change:

```text
PYTHONPATH=python .venv/bin/python -m py_compile \
  python/sglang/srt/layers/attention/flashattention_backend.py \
  python/sglang/srt/layers/attention/attncp_fused_ops.py \
  python/sglang/jit_kernel/tests/test_attncp_fused_ops.py

PYTHONPATH=python .venv/bin/python -m pytest \
  python/sglang/jit_kernel/tests/test_attncp_fused_ops.py -q
46 passed
```

Precision:

| test | artifact | result |
|---|---|---|
| full precision regression | `/tmp/welmv4_attncp_precision/20260626_120255` | PASS, MMLU/C-Eval mismatch `0/100`, max/mean logprob diff `0.00e+00` |
| hot fused probe | `/tmp/attncp_hot_precision_probe/20260626_120726` | token PASS, `exact_vs_fused_strict` FAIL, max diff `8.621e-03` |

Hot fused profile evidence:

```text
batch_size=1 seq_cap=131072 hit_split
batch_size=1 seq_cap=65536 hit_split
batch_size=1 seq_cap=32768 hit_split
batch_size=1 seq_cap=16384 hit_split
batch_size=1 seq_cap=8192 fallback:seq_cap
```

The `8192` fallback is expected because `MIN_SEQ_CAP=16384`; the larger request
buckets prove the probe did execute the fused path.

Service sweep:

| config | concurrency | output TPS | mean TTFT ms | mean ITL ms | cuda graph |
|---|---:|---:|---:|---:|---|
| TP4 best | 23 | `449.54` | `18445.21` | `42.17` | yes |
| TP4-CP2 FA3/P2P exact local-cap | 45 | `656.77` | `38342.85` | `49.82` | yes |
| TP4-CP2 Triton fused/local-cap | 44 | `792.44` | `37349.47` | `37.30` | yes |
| TP4-CP2 Triton fused/local-cap | 45 | `816.60` | `36245.02` | `37.42` | yes |
| TP4-CP2 Triton fused/local-cap | 46 | `669.45` | `37892.30` | `37.03` | yes |

Artifacts:

```text
/tmp/welmv4_attncp_manual_sweep/20260626_105331_tp4_s32768_o2048/summary.tsv
/tmp/welmv4_attncp_manual_sweep/20260626_114531_tp4_cp2_exact_localcap_s32768_o2048/summary.tsv
/tmp/welmv4_attncp_manual_sweep/20260626_115340_tp4_cp2_fused_localcap_full_s32768_o2048/summary.tsv
```

Current conclusion:

- Precision-safe exact/P2P c45: `656.77 / 449.54 = 1.461x`, or `+46.1%`
  output throughput over TP4 best.
- Experimental fused/local-cap c45: `816.60 / 449.54 = 1.817x`, or `+81.7%`
  output throughput over TP4 best.
- Fused/local-cap c45 is `+24.3%` over exact/P2P c45.
- Fused/local-cap c45 mean ITL is `37.42 ms`, `11.3%` lower than TP4 best
  `42.17 ms`.
- c46 peak running is `45`, so it is already beyond the stable resident
  capacity point and the throughput drop is expected.
- This is the first service-level result where the pure Triton fused path
  converts isolated attention speedup into end-to-end throughput/ITL gain.
- The strict-logprob limitation remains: fused/local-cap is still
  experimental and should not replace the default precision-safe FA3 exact path
  unless the required correctness gate is token-level rather than strict
  logprob parity.

## 2026-06-26 strict drift follow-up

Tried math/order variants:

| variant | result |
|---|---|
| keep softmax probability `p` as fp32 in `tl.dot(p, V)` | Triton compile failed: dot operands must have the same dtype (`fp32` vs `bf16`) |
| cast `V` to fp32 and use `tl.dot(p, V_fp32, input_precision="ieee")` | compiles only after reducing `BLOCK_N` for `head_dim=256`; local WeLM shape O max diff stayed `2.44e-04`, mean improved only from `2.75e-05` to `2.25e-05`; shared memory pressure/perf risk too high |
| reduce `BLOCK_N` from `128` to `64` while keeping bf16 `P @ V` | local WeLM shape O max diff stayed `2.44e-04`, mean slightly worsened to `2.79e-05` |

Conclusion: the current `8.621e-03` hot fused strict-logprob drift is not fixed
by only changing `P @ V` precision or the Triton block size. The local attention
output remains at about one BF16 ulp max diff versus FA3, and that small hidden
state difference can still amplify into selected-token logprob drift.

Layer-range probe:

```text
Artifact: /tmp/attncp_fused_layer_probe/20260626_122126/summary.tsv
Baseline: /tmp/attncp_hot_precision_probe/20260626_120726/tp4_cp2_exact_32k.json
Workload: prompt_len=32768, max_new_tokens=4, compare against TP4-CP2 exact
```

| fused layers | token ids | strict passed | max diff | mean diff |
|---|---|---:|---:|---:|
| `0-7` | `[78, 70, 79, 257]` | false | `7.954895e-02` | `2.048671e-02` |
| `8-15` | `[78, 70, 79, 257]` | false | `2.038772e-01` | `5.575072e-02` |
| `16-23` | `[78, 70, 79, 257]` | false | `7.937133e-02` | `2.240794e-02` |
| `24-31` | `[78, 70, 79, 257]` | false | `4.506576e-02` | `1.192995e-02` |
| `32-39` | `[78, 70, 79, 257]` | false | `3.147352e-02` | `9.024611e-03` |
| `40-47` | `[78, 70, 79, 257]` | false | `1.834488e-02` | `6.025991e-03` |
| all layers | `[78, 70, 79, 257]` | false | `8.621e-03` | `2.549e-03` |

Interpretation:

- Every 8-layer block keeps generated tokens identical but fails strict logprob.
- Several partial layer ranges have larger strict drift than all-layer fused.
- This means drift is not localized to one bad layer block; there is substantial
  layer-to-layer cancellation.
- A simple layer fallback policy is unlikely to prove strict parity unless it
  falls back almost entirely to FA3, which would remove the fused speedup.
- The remaining strict-parity path still points toward reusing/modifying FA3's
  internal CUDA mainloop with a CP Q-provider, or accepting the current pure
  Triton implementation as a token-level experimental throughput path.

## 2026-06-26 logprob graph guard fix and latest validation

Problem found during fused-env full precision regression:

- Enabling `SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_FUSED_Q_FA=1` while forcing
  all `return_logprob=True` batches out of CUDA graph made short-prompt
  MMLU/C-Eval regression fail:
  `/tmp/welmv4_attncp_precision/20260626_124758`, token mismatch `27/100`,
  max logprob diff `2.69e+01`.
- Exact/no-fused first10 using the same TP4 baseline passed with max/mean diff
  `0.00e+00`.
- Fused-env first10 reproduced the failure when the unconditional graph guard
  disabled decode CUDA graph.
- The fused profile showed request-time `fallback:logprob` only, so the failure
  was not caused by the Triton fused kernel being hit. The real issue was that
  short-prompt AttnCP eager decode is not strict-identical to the existing
  CUDA-graph exact path for top-logprob regression.

Fix:

- CUDA graph is now disabled for fused+logprob only when the selected graph
  seq bucket can capture the experimental fused branch:
  `seq_len_bucket >= FUSED_MIN_SEQ_CAP` and within `FUSED_MAX_SEQ_CAP`.
- Short graph buckets below the fused min seq cap keep the original graph exact
  path. This preserves MMLU/C-Eval strict parity.
- Long graph buckets that could have captured fused still bypass graph for
  logprob requests, so hot fused strict comparison falls back to exact FA3.

Validation after the fix:

```text
PYTHONPATH=python .venv/bin/python -m py_compile \
  python/sglang/srt/model_executor/cuda_graph_runner.py \
  python/sglang/srt/layers/attention/flashattention_backend.py \
  python/sglang/srt/layers/attention/attncp_fused_ops.py

git diff --check

PYTHONPATH=python .venv/bin/python -m pytest \
  python/sglang/jit_kernel/tests/test_attncp_fused_ops.py -q
46 passed
```

Precision artifacts:

| test | artifact | result |
|---|---|---|
| fused-env first10 after guard fix | `/tmp/welmv4_attncp_precision/20260626_124758/tp4_regression_baseline_first10.pkl` against restarted fused server | PASS, token mismatch `0/10`, max/mean diff `0.00e+00` |
| hot fused probe | `/tmp/attncp_hot_precision_probe/20260626_130644` | PASS for token-level and `exact_vs_fused_strict`, max/mean exact-vs-fused diff `0.00e+00` |
| full precision regression | `/tmp/welmv4_attncp_precision/20260626_131150` | PASS, MMLU/C-Eval token mismatch `0/100`, max/mean logprob diff `0.00e+00` |

Latest same-code service sweep, workload `random-ids`, input `32768`, output
`2048`, `num_prompts=max_concurrency`, `request_rate=inf`:

| config | concurrency | successful | peak running | output TPS | mean TTFT ms | mean ITL ms | cuda graph |
|---|---:|---:|---:|---:|---:|---:|---|
| Naive TP4 | 22 | 22 | 22 | `431.36` | `17383.74` | `42.53` | yes |
| Naive TP4 | 23 | 23 | 23 | `448.13` | `18067.73` | `42.52` | yes |
| Naive TP4 | 24 | 24 | 23 | `403.00` | `22153.64` | `40.92` | yes |
| TP4-CP2 fused | 44 | 44 | 44 | `790.93` | `37328.10` | `37.36` | yes |
| TP4-CP2 fused | 45 | 45 | 45 | `813.53` | `36718.27` | `37.40` | yes |
| TP4-CP2 fused | 46 | 46 | 45 | `667.34` | `38619.65` | `36.92` | yes |

Artifacts:

```text
/tmp/welmv4_attncp_manual_sweep/20260626_131858_tp4_cp2_s32768_o2048/summary.tsv
/tmp/welmv4_attncp_manual_sweep/20260626_132740_tp4_s32768_o2048/summary.tsv
```

Latest conclusion:

- Best same-code Naive TP4 point: c23, output TPS `448.13`.
- Best same-code TP4-CP2 fused point: c45, output TPS `813.53`.
- End-to-end output throughput ratio: `813.53 / 448.13 = 1.815x`, or
  `+81.5%`.
- TP4-CP2 c46 peak running stays at `45`, confirming c45 is still the practical
  resident-capacity knee for this 32k/2k workload.
- Strict logprob parity is achieved by guarded fallback, not by the pure Triton
  fused math itself. Non-logprob long-context decode remains the intended fused
  hot path.
