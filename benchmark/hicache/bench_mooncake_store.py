import argparse
import http.server
import json
import logging
import socket
import time
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore

logger = logging.getLogger("bench_mooncake_store")


@dataclass
class BenchCase:
    case_id: int
    prompt_tokens: int
    repeat: int
    latency_ms: float
    ok: bool
    bytes_read: int
    gib_read: float
    bandwidth_gib_s: float
    result_min: int
    result_max: int


class MockHostKVCache:
    def __init__(self, buffer: torch.Tensor, entries_per_page: int, page_elements: int):
        self.kv_buffer = buffer
        self.layout = "page_first"
        self.page_size = 1
        self.entries_per_page = entries_per_page
        self.page_elements = page_elements

    def get_page_buffer_meta(self, indices):
        ptr_list = []
        element_size_list = []
        item_bytes = self.page_elements * self.kv_buffer.element_size()
        for idx in indices:
            page_idx = int(idx)
            page_offset = page_idx * self.entries_per_page * self.page_elements
            for entry_idx in range(self.entries_per_page):
                offset = page_offset + entry_idx * self.page_elements
                ptr_list.append(self.kv_buffer[offset:].data_ptr())
                element_size_list.append(item_bytes)
        return ptr_list, element_size_list

    def get_ksize_per_token(self):
        return (
            self.entries_per_page * self.page_elements * self.kv_buffer.element_size()
        )


def parse_ks(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def dtype_from_name(name: str):
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    if name == "uint8":
        return torch.uint8
    raise ValueError(f"unsupported dtype: {name}")


def make_storage_config(args: argparse.Namespace) -> HiCacheStorageConfig:
    extra_config = {
        "local_hostname": args.local_hostname,
        "metadata_server": args.metadata_server,
        "global_segment_size": args.global_segment_size,
        "protocol": args.protocol,
        "device_name": args.device_name,
        "master_server_address": args.master_address,
        "check_server": args.check_server,
        "enable_ssd_offload": args.enable_ssd_offload,
    }
    if args.ssd_path:
        extra_config["ssd_offload_path"] = args.ssd_path

    return HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name=None,
        extra_config=extra_config,
    )


def make_keys(prefix: str, prompt_tokens: int, repeat: int) -> List[str]:
    base = f"{prefix}_k{prompt_tokens}_r{repeat}"
    return [f"{base}_{i}" for i in range(prompt_tokens)]


def make_case_keys(
    args: argparse.Namespace, prompt_tokens: int, repeat: int
) -> List[str]:
    key_token_count = args.key_token_count or prompt_tokens
    key_start_index = args.key_start_index
    if key_start_index < 0:
        raise ValueError(f"key_start_index={key_start_index} must be >= 0")
    if key_token_count < key_start_index + prompt_tokens:
        raise ValueError(
            f"key_token_count={key_token_count} must be >= "
            f"key_start_index + prompt_tokens={key_start_index + prompt_tokens}"
        )
    return make_keys(args.key_prefix, key_token_count, repeat)[
        key_start_index : key_start_index + prompt_tokens
    ]


def allocate_pool(args: argparse.Namespace, max_pages: int):
    dtype = dtype_from_name(args.dtype)
    entries_per_page = 2
    if (
        args.bytes_per_token
        % (entries_per_page * torch.tensor([], dtype=dtype).element_size())
        != 0
    ):
        raise ValueError(
            "bytes_per_token must divide into two KV entries for the dtype"
        )
    page_elements = args.bytes_per_token // (
        entries_per_page * torch.tensor([], dtype=dtype).element_size()
    )
    total_elements = max_pages * entries_per_page * page_elements
    logger.info(
        "allocating host buffer: pages=%d entries_per_page=%d page_elements=%d dtype=%s bytes=%.3f GiB",
        max_pages,
        entries_per_page,
        page_elements,
        args.dtype,
        total_elements * torch.tensor([], dtype=dtype).element_size() / (1 << 30),
    )
    buffer = torch.empty(total_elements, dtype=dtype)
    buffer.zero_()
    return MockHostKVCache(buffer, entries_per_page, page_elements), buffer


class ReadyHandler(http.server.BaseHTTPRequestHandler):
    payload = {}

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


def run_writer(args: argparse.Namespace) -> None:
    ks = parse_ks(args.ks)
    max_k = max(ks)
    pool, _ = allocate_pool(args, max_k)
    store = MooncakeStore(make_storage_config(args))
    store.register_mem_pool_host(pool)
    try:
        writer_segment = store.store.get_hostname()
    except Exception:
        writer_segment = ""

    for repeat in range(args.repeats):
        for case_id, k in enumerate(ks):
            keys = make_case_keys(args, k, repeat)
            indices = torch.arange(k, dtype=torch.int64)
            start = time.perf_counter()
            if args.set_chunk_pages > 0:
                results = []
                for chunk_id, chunk_start in enumerate(
                    range(0, k, args.set_chunk_pages)
                ):
                    chunk_end = min(chunk_start + args.set_chunk_pages, k)
                    chunk_results = store.batch_set_v1(
                        keys[chunk_start:chunk_end], indices[chunk_start:chunk_end]
                    )
                    results.extend(chunk_results)
                    if args.set_chunk_log_every > 0 and (
                        chunk_id == 0
                        or chunk_end == k
                        or chunk_id % args.set_chunk_log_every == 0
                    ):
                        chunk_ok = sum(1 for result in chunk_results if result)
                        logger.info(
                            "writer put chunk case=%d k=%d repeat=%d chunk=%d pages=%d-%d ok=%d/%d",
                            case_id,
                            k,
                            repeat,
                            chunk_id,
                            chunk_start,
                            chunk_end,
                            chunk_ok,
                            len(chunk_results),
                        )
                    if args.set_chunk_sleep > 0 and chunk_end < k:
                        time.sleep(args.set_chunk_sleep)
            else:
                results = store.batch_set_v1(keys, indices)
            duration = time.perf_counter() - start
            if not all(results):
                ok_count = sum(1 for result in results if result)
                raise RuntimeError(
                    f"batch_set_v1 failed for k={k} repeat={repeat}; ok={ok_count}/{len(results)}"
                )
            logger.info(
                "writer put complete case=%d k=%d repeat=%d duration=%.3fs",
                case_id,
                k,
                repeat,
                duration,
            )
            if args.post_case_sleep > 0:
                logger.info(
                    "sleeping %.3fs after writer case=%d k=%d repeat=%d",
                    args.post_case_sleep,
                    case_id,
                    k,
                    repeat,
                )
                time.sleep(args.post_case_sleep)

    if args.post_write_sleep > 0:
        logger.info("sleeping %.3fs before ready", args.post_write_sleep)
        time.sleep(args.post_write_sleep)

    writer_clear_count = 0
    if args.clear_own_replica_before_ready:
        if not writer_segment:
            raise RuntimeError("cannot clear own replicas without writer segment")
        for repeat in range(args.repeats):
            for case_id, k in enumerate(ks):
                keys = make_case_keys(args, k, repeat)
                indices = torch.arange(k, dtype=torch.int64)
                key_strs, _, _ = store._batch_preprocess(store._tag_keys(keys), indices)
                cleared = store.store.batch_replica_clear(key_strs, writer_segment)
                writer_clear_count += len(cleared)
                logger.info(
                    "writer cleared stale replicas case=%d k=%d repeat=%d segment=%s count=%d",
                    case_id,
                    k,
                    repeat,
                    writer_segment,
                    len(cleared),
                )

    ReadyHandler.payload = {
        "ready": True,
        "key_prefix": args.key_prefix,
        "ks": ks,
        "repeats": args.repeats,
        "bytes_per_token": args.bytes_per_token,
        "mode": args.mode,
        "set_chunk_pages": args.set_chunk_pages,
        "set_chunk_sleep": args.set_chunk_sleep,
        "post_case_sleep": args.post_case_sleep,
        "writer_segment": writer_segment,
        "writer_clear_count": writer_clear_count,
    }
    server = http.server.ThreadingHTTPServer(
        (args.http_host, args.http_port), ReadyHandler
    )
    logger.info("writer ready on http://%s:%d/ready", args.http_host, args.http_port)
    server.serve_forever()


def run_reader(args: argparse.Namespace) -> None:
    ks = parse_ks(args.ks)
    max_k = max(ks)
    pool, _ = allocate_pool(args, max_k)
    store = MooncakeStore(make_storage_config(args))
    store.register_mem_pool_host(pool)

    if args.wait_url:
        deadline = time.time() + args.wait_url_timeout
        ready_payload = {}
        while True:
            try:
                with urllib.request.urlopen(args.wait_url, timeout=3) as resp:
                    if resp.status == 200:
                        ready_payload = json.loads(resp.read().decode("utf-8"))
                        logger.info("wait url ready: %s", args.wait_url)
                        break
            except Exception as exc:
                if time.time() > deadline:
                    raise TimeoutError(
                        f"timed out waiting for {args.wait_url}"
                    ) from exc
                time.sleep(1)
    else:
        ready_payload = {}

    clear_replica_segment = args.clear_replica_segment
    if clear_replica_segment == "auto":
        clear_replica_segment = ready_payload.get("writer_segment", "")
    if clear_replica_segment:
        logger.info("will clear stale replicas from segment: %s", clear_replica_segment)

    rows: List[BenchCase] = []
    for repeat in range(args.repeats):
        for case_id, k in enumerate(ks):
            keys = make_case_keys(args, k, repeat)
            if args.skip_exists_check:
                logger.info(
                    "skipping batch_exists check case=%d k=%d repeat=%d",
                    case_id,
                    k,
                    repeat,
                )
            else:
                deadline = time.time() + args.exists_timeout
                while True:
                    exists = store.batch_exists(keys)
                    if exists == k:
                        break
                    if time.time() > deadline:
                        raise TimeoutError(
                            f"timed out waiting for keys k={k} repeat={repeat}; exists={exists}/{k}"
                        )
                    time.sleep(1)

            indices = torch.arange(k, dtype=torch.int64)
            chunk_pages = args.get_chunk_pages if args.get_chunk_pages > 0 else k
            results = []
            latency = 0.0
            for chunk_start in range(0, k, chunk_pages):
                chunk_end = min(chunk_start + chunk_pages, k)
                chunk_keys = keys[chunk_start:chunk_end]
                chunk_indices = indices[chunk_start:chunk_end]
                key_strs, ptrs, sizes = store._batch_preprocess(
                    store._tag_keys(chunk_keys), chunk_indices
                )
                if clear_replica_segment:
                    clear_results = store.store.batch_replica_clear(
                        key_strs, clear_replica_segment
                    )
                    logger.info(
                        "cleared stale replicas case=%d k=%d repeat=%d segment=%s count=%d",
                        case_id,
                        k,
                        repeat,
                        clear_replica_segment,
                        len(clear_results),
                    )
                start = time.perf_counter()
                chunk_results = store._get_batch_zero_copy_impl(key_strs, ptrs, sizes)
                latency += time.perf_counter() - start
                results.extend(chunk_results)
            ok = all(res > 0 for res in results)
            bytes_read = sum(res for res in results if res > 0)
            gib_read = bytes_read / (1 << 30)
            row = BenchCase(
                case_id=case_id,
                prompt_tokens=k,
                repeat=repeat,
                latency_ms=latency * 1000.0,
                ok=ok,
                bytes_read=bytes_read,
                gib_read=gib_read,
                bandwidth_gib_s=gib_read / latency if latency > 0 else 0.0,
                result_min=min(results) if results else 0,
                result_max=max(results) if results else 0,
            )
            rows.append(row)
            logger.info("reader result %s", asdict(row))
            if not ok:
                raise RuntimeError(f"batch_get_into failed for k={k} repeat={repeat}")

    output = {
        "run_id": args.run_id,
        "mode": args.mode,
        "key_prefix": args.key_prefix,
        "key_token_count": args.key_token_count,
        "key_start_index": args.key_start_index,
        "master_address": args.master_address,
        "local_hostname": args.local_hostname,
        "metadata_server": args.metadata_server,
        "protocol": args.protocol,
        "device_name": args.device_name,
        "global_segment_size": args.global_segment_size,
        "enable_ssd_offload": args.enable_ssd_offload,
        "ssd_path": args.ssd_path,
        "bytes_per_token": args.bytes_per_token,
        "dtype": args.dtype,
        "ks": ks,
        "repeats": args.repeats,
        "set_chunk_pages": args.set_chunk_pages,
        "set_chunk_sleep": args.set_chunk_sleep,
        "post_case_sleep": args.post_case_sleep,
        "get_chunk_pages": args.get_chunk_pages,
        "skip_exists_check": args.skip_exists_check,
        "clear_replica_segment": clear_replica_segment,
        "results": [asdict(row) for row in rows],
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    logger.info("wrote %s", args.output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["writer", "reader"], required=True)
    parser.add_argument("--mode", choices=["memory", "disk", "ssd"], required=True)
    parser.add_argument("--run-id", default=str(uuid.uuid4()))
    parser.add_argument("--key-prefix", required=True)
    parser.add_argument(
        "--key-token-count",
        type=int,
        default=0,
        help=(
            "override the token count embedded in generated keys; useful for reading "
            "a prefix of a larger writer case"
        ),
    )
    parser.add_argument(
        "--key-start-index",
        type=int,
        default=0,
        help="start index into the generated key list for this case",
    )
    parser.add_argument("--ks", default="1024,2048,4096,8192,16384,32768,65536,131072")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--bytes-per-token", type=int, required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--master-address", required=True)
    parser.add_argument("--local-hostname", required=True)
    parser.add_argument("--metadata-server", default="P2PHANDSHAKE")
    parser.add_argument("--protocol", default="rdma")
    parser.add_argument("--device-name", default="")
    parser.add_argument("--global-segment-size", default="32gb")
    parser.add_argument("--check-server", action="store_true")
    parser.add_argument("--enable-ssd-offload", action="store_true")
    parser.add_argument("--ssd-path", default="")
    parser.add_argument("--set-chunk-pages", type=int, default=0)
    parser.add_argument("--set-chunk-sleep", type=float, default=0.0)
    parser.add_argument("--set-chunk-log-every", type=int, default=16)
    parser.add_argument("--post-case-sleep", type=float, default=0.0)
    parser.add_argument("--get-chunk-pages", type=int, default=0)
    parser.add_argument("--post-write-sleep", type=float, default=0.0)
    parser.add_argument("--clear-own-replica-before-ready", action="store_true")
    parser.add_argument("--http-host", default="0.0.0.0")
    parser.add_argument("--http-port", type=int, default=18080)
    parser.add_argument("--exists-timeout", type=int, default=900)
    parser.add_argument("--skip-exists-check", action="store_true")
    parser.add_argument("--wait-url", default="")
    parser.add_argument("--wait-url-timeout", type=int, default=1800)
    parser.add_argument(
        "--clear-replica-segment",
        default="",
        help="segment whose expired replicas should be cleared before each get; use 'auto' with --wait-url",
    )
    parser.add_argument("--output", default="/tmp/mooncake_store_bench_result.json")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.role == "writer":
        run_writer(args)
    else:
        run_reader(args)


if __name__ == "__main__":
    main()
