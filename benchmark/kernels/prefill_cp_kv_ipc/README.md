# Prefill CP K/V IPC Source-Push Microbenchmark

This directory contains the independent CP4 benchmark for the CUDA IPC
source-push kernel. The indexed variant is also consumed by the automatic
WeLM V4 CP4 Prefill CP production transport; benchmark execution remains
separate from serving.

## Scope

The first experiment intentionally supports only:

- one node with four mutually peer-accessible NVIDIA GPUs;
- CP size 4;
- BF16 K/V rows with width 256;
- Dense delivery to all four CP ranks or Mirror delivery to one CP rank;
- a cold 32K-token layout and a 31.5K Prefix + 512 Extend layout;
- rotated zigzag token ownership and fragmented Prefix physical slots.

Unsupported shapes and configurations fail explicitly in the Python wrapper.

## Compared Paths

The PyNccl baseline reproduces the current data movement in four stages:

1. materialize fragmented Prefix rows with `index_select`;
2. exchange rank-packed K/V with grouped AllGather for Dense or grouped
   Send/Recv for Mirror;
3. restore Prefix and Extend from rank-packed order into logical order;
4. copy both segments into the final ready-to-consume K/V buffers.

The IPC path benchmarks both contiguous-run descriptors and the indexed row
mapping used by the production integration. Each source rank reads a local K/V
vector once and writes it directly to the final logical row in each selected
destination arena. A system-scope release/acquire ready epoch makes payload
completion visible to the destination stream. A separate consumed epoch tests
safe single-slot arena reuse after FA has finished reading.

## Run

Run the CPU descriptor contract tests:

```bash
/envs/train/bin/python -m pytest -q \
  benchmark/kernels/prefill_cp_kv_ipc/test_source_push_contract.py
```

Compile the JIT kernel and validate every default layout and mode on four GPUs:

```bash
PATH=/opt/conda/bin:$PATH \
  /envs/train/bin/torchrun --standalone --nproc-per-node=4 \
  benchmark/kernels/prefill_cp_kv_ipc/benchmark_source_push.py \
  --correctness-only
```

The correctness run also instantiates the production PyTorch-storage IPC arena
and executes consecutive Dense/Mirror transfers, covering ready/consumed reuse
and the exact handle-opening path used by serving.

Sweep tile and CTA sizes:

```bash
PATH=/opt/conda/bin:$PATH \
  /envs/train/bin/torchrun --standalone --nproc-per-node=4 \
  benchmark/kernels/prefill_cp_kv_ipc/benchmark_source_push.py \
  --warmup 20 --iterations 50 \
  --tile-rows 4 8 16 32 64 \
  --threads 64 128 256
```

The script prints a compact `RESULT` line per configuration followed by the
complete rank-0 JSON result.

## Metric Definitions

- `nccl_transfer`: grouped PyNccl communication only.
- `reorder_only`: logical Prefix/Extend restore plus final segment copies.
- `baseline.ready_to_consume`: Prefix materialization, communication, reorder,
  and final copies measured together.
- `payload_only`: IPC source-push payload kernels without destination waits.
- `ready_to_consume`: source-push plus destination acquire waits.
- `ready_and_consumed`: source-push, destination ready wait, reverse consumed
  publish, and source consumed wait. This is a synchronization round trip; it
  is reported separately because production overlaps the consumed interval
  with post-attention compute before the next layer reuses the arena.
- `logical_bytes`: all destination-visible K/V bytes, including local copies.
- `remote_bytes`: the subset expected to cross a peer link.
- `effective_gbps`: logical bytes divided by critical-path GPU latency; it is
  not physical NVLink throughput.

Each timing is a CUDA-event interval. The reported sample is the maximum rank
latency for that iteration, collected over Gloo after GPU synchronization.
Persistent buffers, descriptors, and Python layout planning are outside the
timed region. The baseline Prefix `index_select`, including its output
materialization, is intentionally inside `baseline.ready_to_consume`, as are
the data-movement kernels needed by each path. Twenty warmups are used by
default because shorter warmup runs can retain a cold GPU clock-state artifact.

## Current Boundary

The benchmark remains independently runnable. Production integration is
automatic for WeLM V4 full-attention Dense/Mirror CP4 sharded-KV prefill on one
node; there is no separate CLI switch. Compact SWA keeps its existing
all-to-allv transport. Once the automatic route is selected, unsupported
conditions fail instead of falling back to the NCCL gather path.
