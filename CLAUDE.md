# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SGLang is a high-performance serving framework for large language models (LLMs) and vision-language models (VLMs). It provides low-latency, high-throughput inference from single GPU to large distributed clusters. The project consists of three sub-packages: the main Python package (`python/`), a CUDA/C++ kernel library (`sgl-kernel/`), and a Rust-based router/gateway (`sgl-router/`).

## Build and Install

```bash
# Install the main package in dev mode (from repo root)
pip install -e "python[dev]"

# Install sgl-kernel (CUDA required)
cd sgl-kernel && pip install -e .

# Install sgl-router (Rust required)
cd sgl-router && pip install -e .
```

Key pinned dependencies (check `python/pyproject.toml` for current versions): `torch`, `transformers`, `flashinfer_python`, `sgl-kernel`, `xgrammar`, `grpcio`. These must stay aligned.

## Linting and Formatting

```bash
# Run all pre-commit checks (this is what CI runs)
SKIP=no-commit-to-branch pre-commit run --all-files --show-diff-on-failure

# Format only modified Python files
make format

# Individual tools used:
#   isort (profile=black, known_first_party=sglang)
#   black
#   ruff (rules: F401 unused imports, F821 undefined names)
#   clang-format (for C++/CUDA in sgl-kernel)
#   codespell
```

Ruff only checks files under `benchmark/`, `docs/`, `examples/`, `python/sglang/`, `sgl-router/py_*`, `test/`. Generated protobuf files (`*_pb2.py`, `*_pb2_grpc.py`) are excluded from all formatters.

**CI lint check**: Proto files must stay in sync: `python/sglang/srt/grpc/sglang_scheduler.proto` ↔ `sgl-router/src/proto/sglang_scheduler.proto`.

## Testing

Tests live under `test/srt/` and use Python `unittest` (with pytest runner, `asyncio_mode = auto`).

```bash
# Run a single test file
python -m pytest test/srt/test_srt_engine.py -v

# Run a specific test class or method
python -m pytest test/srt/test_srt_engine.py::TestClass::test_method -v

# Run the full per-commit 1-GPU test suite (CI)
python test/srt/run_suite.py --suite per-commit-1-gpu

# Run the full per-commit 2-GPU test suite
python test/srt/run_suite.py --suite per-commit-2-gpu
```

Test suites are defined in `test/srt/run_suite.py` with timeout values per file:
- **per-commit-1-gpu**: ~100 test files (models, OpenAI server, quantization, LoRA, speculative decoding, constrained decoding, etc.)
- **per-commit-2-gpu**: multi-GPU tests (expert parallelism, data parallelism, disaggregation, tensor parallelism)
- **per-commit-4-gpu / 8-gpu**: larger scale tests (pipeline parallelism, DeepSeek models)

Most tests launch a server process internally and test against it. The test infrastructure is in `python/sglang/test/`.

## Starting a Server

```bash
# Launch HTTP server
python -m sglang.launch_server --model-path <model> --port 30000

# Launch gRPC server
python -m sglang.launch_server --model-path <model> --grpc-mode

# Use as a Python engine (no server process)
from sglang.srt.entrypoints.engine import Engine
engine = Engine(model_path="<model>")
```

## Architecture

### Three Sub-packages

| Package | Language | Purpose |
|---------|----------|---------|
| `python/sglang/` | Python | Main serving framework: runtime, models, scheduling, API |
| `sgl-kernel/` | C++/CUDA | High-performance GPU kernels (attention, MoE, quantization) |
| `sgl-router/` | Rust | Multi-instance load balancer and model gateway |

### Runtime Architecture (`python/sglang/srt/`)

"SRT" = SGLang Runtime. This is the core serving engine. The system runs as multiple processes communicating via ZeroMQ (zmq):

```
Client Request
    ↓
HTTP/gRPC Server (entrypoints/)
    ↓  (zmq)
TokenizerManager (managers/tokenizer_manager.py)
    - Tokenizes inputs, manages chat templates
    - Routes to scheduler(s)
    ↓  (zmq)
Scheduler (managers/scheduler.py)
    - Central orchestrator (~4000 lines, heavily mixin-based)
    - Manages request queues, batching, memory allocation
    - Drives the forward loop: prefill → decode cycles
    - Mixins: scheduler_*_mixin.py for DP attention, PP, profiling, output processing, etc.
    ↓
ModelRunner (model_executor/model_runner.py)
    - Runs model forward passes
    - Manages CUDA graphs (cuda_graph_runner.py)
    ↓
TpWorker (managers/tp_worker.py)
    - Tensor-parallel worker that owns the model runner and memory pool
    ↓
DetokenizerManager (managers/detokenizer_manager.py)
    - Converts token IDs back to text
    - Streams responses back to client
```

### Key Subsystems

- **Memory & Caching** (`mem_cache/`): RadixAttention prefix caching (`radix_cache.py`), paged KV-cache memory pools (`memory_pool.py`), hierarchical caching (`hiradix_cache.py`), eviction policies
- **Scheduling** (`managers/schedule_batch.py`, `schedule_policy.py`): Continuous batching, chunked prefill, priority scheduling, request retraction on OOM
- **Model Support** (`models/`): ~130 model implementations. Each model file registers via `EntryClass` metadata. Models use layers from `layers/` (attention, linear, MoE, quantization, embeddings)
- **Constrained Decoding** (`constrained/`): Grammar-guided generation via xgrammar, outlines, or llguidance backends
- **Speculative Decoding** (`speculative/`): EAGLE draft models, n-gram speculation, standalone speculative workers
- **Parallelism** (`distributed/`): Tensor parallelism (TP), pipeline parallelism (PP), expert parallelism (EP), data parallelism (DP), prefill-decode disaggregation (`disaggregation/`)
- **LoRA** (`lora/`): Multi-LoRA batching with runtime adapter loading/unloading
- **Quantization** (`layers/quantization/`): FP4, FP8, INT4, INT8, AWQ, GPTQ, compressed tensors
- **Structured Outputs** (`constrained/`): JSON mode, regex, EBNF grammars
- **Overlap Scheduling** (`batch_overlap/`, `managers/overlap_utils.py`): Overlapping compute and communication

### Frontend Language (`python/sglang/lang/`)

A programming interface for LLM applications with chained generation, control flow, and parallelism. Separate from the runtime—uses the runtime as a backend.

### Entrypoints (`python/sglang/srt/entrypoints/`)

- `engine.py`: Python `Engine` class — programmatic access without a server
- `http_server.py`: FastAPI/uvicorn HTTP server with OpenAI-compatible API
- `grpc_server.py`: gRPC server alternative
- `openai/`: OpenAI API protocol implementations (chat, completions, embeddings, etc.)

### Server Configuration

Server args are defined in `python/sglang/srt/server_args.py` (100+ arguments). Key categories: model loading, parallelism (TP/PP/EP/DP), memory management, scheduling policy, quantization, speculative decoding, disaggregation.

## Adding a New Model

Model files go in `python/sglang/srt/models/`. Follow the pattern of existing models — each file defines model classes and registers them with `EntryClass` in the module metadata. The model uses layers from `python/sglang/srt/layers/` (attention via `radix_attention.py`, linear layers, MoE, etc.). Model loading happens through `python/sglang/srt/model_loader/`.
