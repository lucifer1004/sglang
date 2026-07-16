"""Multi-rank coverage for active-rank prefill CP K/V exchange."""

from __future__ import annotations

import multiprocessing as mp
import socket
from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.test.ci.ci_register import register_cuda_ci


register_cuda_ci(est_time=30, stage="stage-b", runner_config="4-gpu-h100")

_WORLD_SIZE = 4


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run_active_gatherv_rank(rank: int, port: int, result_queue) -> None:
    communicator = None
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=_WORLD_SIZE,
            timeout=timedelta(seconds=30),
        )
        communicator = PyNcclCommunicator(dist.group.WORLD, device=rank)
        coordinator = SimpleNamespace(
            world_size=_WORLD_SIZE,
            rank_in_group=rank,
            pynccl_comm=communicator,
            use_symmetric_memory=lambda *_args, **_kwargs: nullcontext(),
        )

        counts = {"send": 0, "recv": 0, "group_start": 0, "group_end": 0}
        original_send = communicator.send
        original_recv = communicator.recv
        original_group_start = communicator.group_start
        original_group_end = communicator.group_end

        def count_send(tensor, dst):
            counts["send"] += 1
            return original_send(tensor, dst)

        def count_recv(tensor, src):
            counts["recv"] += 1
            return original_recv(tensor, src)

        def count_group_start():
            counts["group_start"] += 1
            return original_group_start()

        def count_group_end():
            counts["group_end"] += 1
            return original_group_end()

        communicator.send = count_send
        communicator.recv = count_recv
        communicator.group_start = count_group_start
        communicator.group_end = count_group_end

        prefix_sizes = [1, 0, 1, 0]
        prefix_rows = prefix_sizes[rank]
        prefix_k = torch.full(
            (prefix_rows, 2), float(rank), device=f"cuda:{rank}"
        )
        prefix_v = prefix_k + 10
        prefix_outputs = GroupCoordinator.gatherv_to_ranks(
            coordinator,
            [prefix_k, prefix_v],
            sizes=prefix_sizes,
            dst_ranks=[0, 1],
        )
        if rank in (0, 1):
            expected = torch.tensor(
                [[0.0, 0.0], [2.0, 2.0]], device=f"cuda:{rank}"
            )
            torch.testing.assert_close(prefix_outputs[0], expected)
            torch.testing.assert_close(prefix_outputs[1], expected + 10)
        else:
            assert prefix_outputs is None

        zero_outputs = GroupCoordinator.gatherv_to_ranks(
            coordinator,
            [
                torch.empty((0, 2), device=f"cuda:{rank}"),
                torch.empty((0, 2), device=f"cuda:{rank}"),
            ],
            sizes=[0, 0, 0, 0],
            dst_ranks=[0, 1],
        )
        if rank in (0, 1):
            assert [tuple(output.shape) for output in zero_outputs] == [
                (0, 2),
                (0, 2),
            ]
        else:
            assert zero_outputs is None

        extend_sizes = [1, 1, 0, 0]
        extend_rows = extend_sizes[rank]
        extend_k = torch.full(
            (extend_rows, 2), float(rank), device=f"cuda:{rank}"
        )
        extend_v = extend_k + 20
        extend_outputs = GroupCoordinator.gatherv_to_ranks(
            coordinator,
            [extend_k, extend_v],
            sizes=extend_sizes,
            dst_ranks=[0, 1],
        )
        if rank in (0, 1):
            expected = torch.tensor(
                [[0.0, 0.0], [1.0, 1.0]], device=f"cuda:{rank}"
            )
            torch.testing.assert_close(extend_outputs[0], expected)
            torch.testing.assert_close(extend_outputs[1], expected + 20)
        else:
            assert extend_outputs is None

        all_rank_input = torch.full(
            (1, 2), float(rank), device=f"cuda:{rank}"
        )
        all_rank_output = GroupCoordinator.all_gatherv(
            coordinator,
            [all_rank_input],
            sizes=[1, 1, 1, 1],
        )[0]
        expected_all_rank = torch.arange(
            _WORLD_SIZE, dtype=torch.float32, device=f"cuda:{rank}"
        ).view(_WORLD_SIZE, 1).expand(-1, 2)
        torch.testing.assert_close(all_rank_output, expected_all_rank)

        send_matrix = (
            (1, 2, 0, 1),
            (0, 1, 1, 0),
            (2, 0, 1, 1),
            (0, 1, 2, 0),
        )
        send_sizes = list(send_matrix[rank])
        recv_sizes = [send_matrix[src][rank] for src in range(_WORLD_SIZE)]
        send_parts = [
            (
                torch.arange(count, dtype=torch.float32, device=f"cuda:{rank}")
                + rank * 100
                + dst * 10
            ).view(-1, 1).expand(-1, 2)
            for dst, count in enumerate(send_sizes)
            if count
        ]
        exchange_k = torch.cat(send_parts, dim=0)
        exchange_v = exchange_k + 1000
        exchange_outputs = GroupCoordinator.all_to_allv(
            coordinator,
            [exchange_k, exchange_v],
            send_sizes=send_sizes,
            recv_sizes=recv_sizes,
        )
        expected_parts = [
            (
                torch.arange(count, dtype=torch.float32, device=f"cuda:{rank}")
                + src * 100
                + rank * 10
            ).view(-1, 1).expand(-1, 2)
            for src, count in enumerate(recv_sizes)
            if count
        ]
        expected_exchange = torch.cat(expected_parts, dim=0)
        torch.testing.assert_close(exchange_outputs[0], expected_exchange)
        torch.testing.assert_close(exchange_outputs[1], expected_exchange + 1000)

        torch.cuda.synchronize(rank)
        dist.barrier()
        result_queue.put((rank, "ok", counts))
    except Exception as exc:  # pragma: no cover - subprocess diagnostics
        result_queue.put((rank, "error", repr(exc)))
    finally:
        communicator = None
        if dist.is_initialized():
            dist.destroy_process_group()


def test_active_gatherv_excludes_nonparticipants_without_deadlock():
    if not torch.cuda.is_available() or torch.cuda.device_count() < _WORLD_SIZE:
        pytest.skip(f"requires {_WORLD_SIZE} CUDA devices")

    context = mp.get_context("spawn")
    result_queue = context.Queue()
    port = _free_port()
    processes = [
        context.Process(
            target=_run_active_gatherv_rank,
            args=(rank, port, result_queue),
        )
        for rank in range(_WORLD_SIZE)
    ]
    for process in processes:
        process.start()

    try:
        results = {}
        for _ in range(_WORLD_SIZE):
            rank, status, payload = result_queue.get(timeout=60)
            assert status == "ok", f"rank {rank} failed: {payload}"
            results[rank] = payload

        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)

    assert results == {
        0: {"send": 8, "recv": 6, "group_start": 4, "group_end": 4},
        1: {"send": 4, "recv": 10, "group_start": 4, "group_end": 4},
        2: {"send": 8, "recv": 4, "group_start": 3, "group_end": 3},
        3: {"send": 4, "recv": 4, "group_start": 2, "group_end": 2},
    }
