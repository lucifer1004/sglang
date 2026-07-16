"""Eight-GPU correctness for full-vocabulary WeLM embeddings under CP."""

from __future__ import annotations

import gc
import json
import multiprocessing as mp
import os
import socket
import tempfile
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci


register_cuda_ci(est_time=120, stage="stage-c", runner_config="8-gpu-h20")

WORLD_SIZE = 8
TOPOLOGIES = ((1, 8), (2, 4), (4, 2))


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _deterministic_weight(
    rows: int, width: int, device: int | None, offset: int
):
    values = torch.arange(rows * width, dtype=torch.float32).reshape(rows, width)
    weight = ((values + offset) / 128).to(torch.bfloat16)
    return weight if device is None else weight.to(device)


def _write_checkpoint(root: Path) -> Path:
    from safetensors.torch import save_file

    root.mkdir()
    tensors = {
        "model.embed_tokens.weight": _deterministic_weight(17, 8, None, 1)
    }
    for index, rows in enumerate((11, 13, 17, 19)):
        tensors[f"model.oe_embed.{index}.weight"] = _deterministic_weight(
            rows, 4, None, 100 * (index + 1)
        )
    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, root / shard)
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "total_size": sum(
                        tensor.numel() * tensor.element_size()
                        for tensor in tensors.values()
                    )
                },
                "weight_map": {key: shard for key in tensors},
            }
        )
    )
    return root


def _run_rank(
    rank: int, port: int, cp_size: int, manifest_path: str, result_queue
) -> None:
    registry = None
    base_module = None
    oe_modules = None
    result = None
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(WORLD_SIZE)
        os.environ["LOCAL_RANK"] = str(rank)
        torch.cuda.set_device(rank)

        from sglang.srt.distributed.parallel_state import (
            get_attn_cp_group,
            get_attn_tp_group,
            init_distributed_environment,
            initialize_model_parallel,
        )

        init_distributed_environment(
            world_size=WORLD_SIZE,
            rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            local_rank=rank,
            backend="nccl",
            timeout=120,
        )
        initialize_model_parallel(
            tensor_model_parallel_size=WORLD_SIZE,
            attention_context_model_parallel_size=cp_size,
            attention_data_parallel_size=1,
            backend="nccl",
        )

        from sglang.srt.layers.full_vocab_shared_embedding import (
            FullVocabSharedEmbedding,
        )
        from sglang.srt.model_loader.welm_shared_embedding_weights import (
            WeLMSharedEmbeddingRegistry,
        )
        from sglang.srt.models import welm_perf_opt
        from sglang.srt.models.welm_perf_opt import compute_welm_oe_embedding

        registry = WeLMSharedEmbeddingRegistry.from_manifest(
            manifest_path,
            gpu_id=rank,
            gpu_numa_node=0 if rank < WORLD_SIZE // 2 else 1,
        )

        hidden_size = 8
        oe_dim = 4
        vocab_size = 17
        oe_vocab_sizes = (11, 13, 17, 19)
        weights = {
            "model.embed_tokens.weight": _deterministic_weight(
                vocab_size, hidden_size, rank, 1
            )
        }
        for index, rows in enumerate(oe_vocab_sizes):
            weights[f"model.oe_embed.{index}.weight"] = _deterministic_weight(
                rows, oe_dim, rank, 100 * (index + 1)
            )
        base_module = FullVocabSharedEmbedding(
            key="model.embed_tokens.weight",
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            registry=registry,
        )
        oe_modules = [
            FullVocabSharedEmbedding(
                key=f"model.oe_embed.{index}.weight",
                num_embeddings=rows,
                embedding_dim=oe_dim,
                registry=registry,
            )
            for index, rows in enumerate(oe_vocab_sizes)
        ]
        projection = torch.nn.Linear(
            len(oe_vocab_sizes) * oe_dim,
            hidden_size,
            bias=False,
            dtype=torch.bfloat16,
            device=f"cuda:{rank}",
        )
        with torch.no_grad():
            projection.weight.copy_(
                _deterministic_weight(
                    hidden_size,
                    len(oe_vocab_sizes) * oe_dim,
                    rank,
                    1000,
                )
            )

        input_ids = torch.tensor([0, 5, 16, 3], device=rank, dtype=torch.int64)
        hashed_inputs = torch.tensor(
            [
                [0, 3, 10, 7],
                [1, 5, 12, 8],
                [2, 7, 16, 9],
                [3, 9, 18, 11],
            ],
            device=rank,
            dtype=torch.int64,
        )
        forward_batch = SimpleNamespace(
            oe_context=SimpleNamespace(hash_prefixes=None),
            welm_oe_decode_hashed_inputs=hashed_inputs,
        )

        collective_calls = 0

        def reject_collective(_tensor):
            nonlocal collective_calls
            collective_calls += 1
            raise AssertionError("full-vocabulary embedding invoked all-reduce")

        with mock.patch.object(
            welm_perf_opt,
            "tensor_model_parallel_all_reduce",
            reject_collective,
        ), mock.patch.object(
            welm_perf_opt,
            "attn_tp_all_reduce",
            reject_collective,
        ), mock.patch(
            "torch.distributed.all_reduce",
            side_effect=reject_collective,
        ):
            base_actual = base_module(input_ids)
            base_expected = torch.nn.functional.embedding(
                input_ids, weights["model.embed_tokens.weight"]
            )
            torch.testing.assert_close(base_actual, base_expected, rtol=0, atol=0)

            oe_expected_parts = []
            for index, module in enumerate(oe_modules):
                branch_actual = module(hashed_inputs[index])
                branch_expected = torch.nn.functional.embedding(
                    hashed_inputs[index],
                    weights[f"model.oe_embed.{index}.weight"],
                )
                torch.testing.assert_close(
                    branch_actual, branch_expected, rtol=0, atol=0
                )
                oe_expected_parts.append(branch_expected)

            actual = compute_welm_oe_embedding(
                input_ids=input_ids,
                forward_batch=forward_batch,
                base_hidden_states=base_actual,
                oe_grams=(2, 2, 3, 3),
                oe_vocab_sizes=oe_vocab_sizes,
                vocab_size=vocab_size,
                oe_embed_modules=oe_modules,
                oe_proj_module=projection,
                use_triton_preprocess=False,
            )
            expected = (
                base_expected
                + torch.nn.functional.linear(
                    torch.cat(oe_expected_parts, dim=-1), projection.weight
                )
            ) / 2
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert collective_calls == 0

        result = {
            "rank": rank,
            "status": "ok",
            "attn_cp_size": get_attn_cp_group().world_size,
            "attn_tp_size": get_attn_tp_group().world_size,
            "attn_cp_ranks": tuple(get_attn_cp_group().ranks),
            "attn_tp_ranks": tuple(get_attn_tp_group().ranks),
            "checksum": float(actual.float().sum().item()),
            "collective_calls": collective_calls,
        }
    except BaseException as exc:
        result = {"rank": rank, "status": "error", "error": repr(exc)}
    finally:
        cleanup_errors = []
        base_module = None
        oe_modules = None
        module = None
        gc.collect()
        if registry is not None:
            try:
                registry.close()
            except BaseException as exc:
                cleanup_errors.append(exc)
        try:
            from sglang.srt.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )

            destroy_model_parallel()
            destroy_distributed_environment()
        except BaseException as exc:
            cleanup_errors.append(exc)
        if cleanup_errors:
            result = {
                "rank": rank,
                "status": "error",
                "error": repr(
                    BaseExceptionGroup("worker cleanup failed", cleanup_errors)
                ),
            }
    result_queue.put(result)


def _run_topology(cp_size: int, manifest_path: str):
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    port = _get_free_port()
    processes = [
        context.Process(
            target=_run_rank,
            args=(rank, port, cp_size, manifest_path, result_queue),
        )
        for rank in range(WORLD_SIZE)
    ]
    started = []
    try:
        for process in processes:
            process.start()
            started.append(process)
        results = [result_queue.get(timeout=300) for _ in processes]
        for process in started:
            process.join(timeout=60)
            assert process.exitcode == 0, f"rank exited with {process.exitcode}"
    except BaseException as original_error:
        cleanup_errors = _stop_processes(started)
        if cleanup_errors:
            raise BaseExceptionGroup(
                "distributed test and cleanup failed",
                [original_error, *cleanup_errors],
            )
        raise
    finally:
        cleanup_errors = _stop_processes(started)
        if cleanup_errors:
            raise BaseExceptionGroup("distributed test cleanup failed", cleanup_errors)
    return sorted(results, key=lambda result: result["rank"])


def _stop_processes(processes) -> list[BaseException]:
    errors = []
    for process in processes:
        try:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=10)
            if process.is_alive():
                raise RuntimeError(f"rank process {process.pid} did not stop")
        except BaseException as exc:
            errors.append(exc)
    return errors


@pytest.mark.parametrize(("cp_size", "attn_tp_size"), TOPOLOGIES)
def test_full_vocab_shared_embedding_matches_reference_without_collectives(
    cp_size, attn_tp_size
):
    if not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE:
        pytest.skip(f"requires {WORLD_SIZE} CUDA devices")

    from sglang.srt.model_loader.welm_shared_embedding_weights import (
        WeLMSharedEmbeddingPolicy,
        launch_welm_embedding_arena_process,
        plan_welm_embedding_replicas,
    )

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = _write_checkpoint(Path(directory) / "checkpoint")
        arena_root = Path("/dev/shm") / f"sglang-welm-cp-test-{uuid.uuid4().hex}"
        plans = plan_welm_embedding_replicas(
            WeLMSharedEmbeddingPolicy.INTERLEAVE,
            (0, 0, 0, 0, 1, 1, 1, 1),
            bind_node=None,
        )
        handle = launch_welm_embedding_arena_process(
            checkpoint=checkpoint,
            root=arena_root,
            plans=plans,
            timeout=60,
        )
        try:
            results = _run_topology(cp_size, handle.manifest_path)
        finally:
            handle.close(timeout=30)

    errors = [result for result in results if result["status"] != "ok"]
    assert not errors, errors
    assert {result["attn_cp_size"] for result in results} == {cp_size}
    assert {result["attn_tp_size"] for result in results} == {attn_tp_size}
    assert {result["collective_calls"] for result in results} == {0}
    assert len({result["checksum"] for result in results}) == 1
    for result in results:
        rank = result["rank"]
        attn_tp_start = rank // attn_tp_size * attn_tp_size
        assert result["attn_tp_ranks"] == tuple(
            range(attn_tp_start, attn_tp_start + attn_tp_size)
        )
        attn_tp_lane = rank % attn_tp_size
        assert result["attn_cp_ranks"] == tuple(
            cp_rank * attn_tp_size + attn_tp_lane
            for cp_rank in range(cp_size)
        )
