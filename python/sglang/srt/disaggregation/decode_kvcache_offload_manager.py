from __future__ import annotations

import json
import logging
import threading
import time
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from sglang.srt.disaggregation.kv_events import OffloadedState
from sglang.srt.environ import envs
from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageExtraInfo,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.memory_pool_host import (
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
)
from sglang.srt.mem_cache.swa_memory_pool import (
    SWAKVPool,
    SWATokenToKVPoolAllocator,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.common import ceil_align

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


class _SubPoolHiCacheController(HiCacheController):
    """HiCacheController whose ``mem_pool_device`` is a sub-pool of an
    outer :class:`SWAKVPool`.

    The base controller derives ``mem_pool_device`` from
    ``allocator.get_kvcache()``, which for SWA models returns the outer
    ``SWAKVPool`` — a wrapper over two real K/V pools (full-attention and
    sliding-window). The pool object the host pool's
    ``backup_from_device_all_layer`` actually wants to read is one of
    those inner ``MHATokenToKVPool`` sub-pools, not the wrapper.

    This subclass overrides ``mem_pool_device`` (and the layer count
    derived from it) to a specific sub-pool after the base ``__init__``
    has finished setting up queues, threads, and storage attach. The
    duplicate ``register_layer_transfer_counter`` against the outer pool
    that the base ``__init__`` performed is harmless: the decode-side
    backup-only path never reads ``layer_done_counter``.
    """

    def __init__(self, *, override_mem_pool_device, storage_pool_name=None, **kwargs):
        super().__init__(**kwargs)
        self.mem_pool_device = override_mem_pool_device
        self.layer_num = override_mem_pool_device.layer_num
        self.storage_pool_name = storage_pool_name
        if self.storage_pool_name is not None:
            self.storage_backend.register_mem_host_pool_v2(
                self.mem_pool_host, self.storage_pool_name
            )

    def _page_backup(self, operation):
        if self.storage_pool_name is None:
            return super()._page_backup(operation)

        prefix_keys = operation.prefix_keys
        for i in range(0, len(operation.hash_value), self.storage_batch_size):
            batch_hashes = operation.hash_value[i : i + self.storage_batch_size]
            batch_host_indices = operation.host_indices[
                i * self.page_size : (i + len(batch_hashes)) * self.page_size
            ]
            transfer = PoolTransfer(
                name=self.storage_pool_name,
                host_indices=batch_host_indices,
                keys=batch_hashes,
            )
            extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys)
            results = self.storage_backend.batch_set_v2([transfer], extra_info)
            pool_results = results.get(self.storage_pool_name, [])
            success = len(pool_results) == len(batch_hashes) and all(pool_results)
            if not success:
                logger.warning(
                    "Write %s page to storage: %d pages failed.",
                    self.storage_pool_name,
                    len(batch_hashes),
                )
                break

            if prefix_keys and len(prefix_keys) > 0:
                prefix_keys += batch_hashes
            operation.completed_tokens += self.page_size * len(batch_hashes)


class DecodeKVCacheOffloadManager:
    """Manage decode-side KV cache offloading lifecycle and operations."""

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        tp_group: torch.distributed.ProcessGroup,
        tree_cache: BasePrefixCache,
        server_args: ServerArgs,
    ) -> None:
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.page_size = server_args.page_size
        self.server_args = server_args
        self.request_counter = 0
        self.tree_cache = tree_cache
        env_stride = envs.SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE.get()
        if env_stride is None or env_stride <= 0:
            self.offload_stride = self.page_size
        else:
            self.offload_stride = max(
                self.page_size, (env_stride // self.page_size) * self.page_size
            )
        kv_cache = self.token_to_kv_pool_allocator.get_kvcache()
        # Whether the device pool is hybrid SWA (full + swa sub-pools). When
        # True, the decode side must mirror BOTH sub-pools to host so the
        # SWA window-tail of output tokens can also reach L3.
        self.is_swa = isinstance(kv_cache, SWAKVPool)

        if self.is_swa:
            # Opt-in only: if the flag is off, fall through to the legacy
            # raise so existing deployments don't silently start mirroring
            # a second host pool / spawning a second backup pipeline.
            if not envs.SGLANG_HICACHE_SWA_DECODE_STORAGE_ENABLE.get():
                raise ValueError(
                    "Decode-side SWA offload is opt-in. Set "
                    "SGLANG_HICACHE_SWA_DECODE_STORAGE_ENABLE=1 to enable "
                    "the dual-pool backup path (SWA payload roughly doubles "
                    "L3 traffic, so it is gated per deployment)."
                )
            # Spec decoding + SWA-decode-L3 has unverified rollback semantics
            # (a rejected spec token's SWA slot must NOT be backed up; that
            # path is wired up in the spec commit). Until then, fail fast.
            if server_args.speculative_algorithm is not None:
                raise ValueError(
                    "SWA decode-side L3 offload is not yet compatible with "
                    "speculative decoding (algorithm="
                    f"{server_args.speculative_algorithm}). This will be "
                    "enabled in a follow-up commit that wires "
                    "committed_seqlen tracking through the spec verify path."
                )
            assert isinstance(
                self.token_to_kv_pool_allocator, SWATokenToKVPoolAllocator
            ), (
                "SWA decode offload requires SWATokenToKVPoolAllocator, got "
                f"{type(self.token_to_kv_pool_allocator)}"
            )
            # Mirror the FULL sub-pool with a regular MHA host pool — this is
            # the "main KV channel" host that holds full-attention layers.
            self.decode_host_mem_pool = MHATokenToKVPoolHost(
                kv_cache.full_kv_pool,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
            )
            # Mirror the SWA sub-pool with its own MHA host pool — sliding-
            # window-attention layers live here. Sized independently so SWA's
            # smaller per-token footprint doesn't crowd out full-attention.
            self.swa_host_mem_pool = MHATokenToKVPoolHost(
                kv_cache.swa_kv_pool,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
            )
        elif isinstance(kv_cache, MHATokenToKVPool):
            self.decode_host_mem_pool = MHATokenToKVPoolHost(
                kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
            )
            self.swa_host_mem_pool = None
        elif isinstance(kv_cache, MLATokenToKVPool):
            self.decode_host_mem_pool = MLATokenToKVPoolHost(
                kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
            )
            self.swa_host_mem_pool = None
        else:
            raise ValueError("Unsupported KV cache type for decode offload")

        self.tp_group = tp_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)

        hicache_storage_backend_extra_config = {}
        if server_args.hicache_storage_backend_extra_config:
            try:
                hicache_storage_backend_extra_config = json.loads(
                    server_args.hicache_storage_backend_extra_config
                )
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid hicache storage backend extra config JSON: {e}"
                )

        # FULL controller (or the only controller for non-SWA models).
        self.cache_controller = HiCacheController(
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            mem_pool_host=self.decode_host_mem_pool,
            page_size=self.page_size,
            tp_group=tp_group,
            io_backend=server_args.hicache_io_backend,
            load_cache_event=threading.Event(),
            storage_backend=server_args.hicache_storage_backend,
            model_name=server_args.served_model_name,
            storage_backend_extra_config=hicache_storage_backend_extra_config,
        )
        # When the device pool is SWA, the base init left mem_pool_device set
        # to the outer SWAKVPool wrapper, but the host pool's
        # backup_from_device_all_layer needs the inner full_kv_pool to match
        # its head/layer layout.
        if self.is_swa:
            self.cache_controller.mem_pool_device = kv_cache.full_kv_pool
            self.cache_controller.layer_num = kv_cache.full_kv_pool.layer_num

        # SWA controller: a second pipeline (D->H + H->L3 threads) that
        # mirrors only the SWA sub-pool. Independent ack queues let the FULL
        # and SWA paths complete at their own pace without one stalling the
        # other.
        self.swa_cache_controller: Optional[HiCacheController] = None
        if self.is_swa:
            self.swa_cache_controller = _SubPoolHiCacheController(
                override_mem_pool_device=kv_cache.swa_kv_pool,
                storage_pool_name=PoolName.SWA,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                mem_pool_host=self.swa_host_mem_pool,
                page_size=self.page_size,
                tp_group=tp_group,
                io_backend=server_args.hicache_io_backend,
                load_cache_event=threading.Event(),
                storage_backend=server_args.hicache_storage_backend,
                model_name=server_args.served_model_name,
                storage_backend_extra_config=hicache_storage_backend_extra_config,
            )

        self.ongoing_offload = {}
        self.ongoing_backup = {}
        # SWA-side tracking — separate maps so progress polling is symmetric
        # with the FULL side's queue draining.
        self.ongoing_swa_offload: dict = {}
        self.ongoing_swa_backup: dict = {}
        self.offloaded_state = {}
        self.offload_inflight = {}
        logger.info(
            "Enable offload kv cache for decode side (is_swa=%s)", self.is_swa
        )

    def pending_work_count(self) -> int:
        count = (
            len(self.ongoing_offload)
            + len(self.ongoing_backup)
            + len(self.ongoing_swa_offload)
            + len(self.ongoing_swa_backup)
            + len(self.offload_inflight)
        )
        controllers = [self.cache_controller, self.swa_cache_controller]
        for controller in controllers:
            if controller is None:
                continue
            count += len(controller.ack_write_queue)
            if hasattr(controller, "ack_backup_queue"):
                count += controller.ack_backup_queue.qsize()
        return count

    def has_pending_work(self) -> bool:
        return self.pending_work_count() > 0

    def _mark_offload_started(self, rid):
        self.offload_inflight[rid] = self.offload_inflight.get(rid, 0) + 1

    def _mark_offload_finished(self, rid):
        count = self.offload_inflight.get(rid, 0)
        if count <= 1:
            self.offload_inflight.pop(rid, None)
        else:
            self.offload_inflight[rid] = count - 1

    def _has_inflight_offload(self, rid):
        return self.offload_inflight.get(rid, 0) > 0

    def offload_kv_cache(self, req) -> bool:
        """Offload incremental KV cache for decode side."""

        if self.cache_controller is None or self.decode_host_mem_pool is None:
            return False

        if req.req_pool_idx == -1 or len(req.output_ids) == 0:
            return False

        token_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx]
        if token_indices.dim() == 0 or token_indices.numel() == 0:
            return False

        # Prefill side offloads page-aligned origin_input_ids, decode side offloads the incremental part
        all_tokens = req.origin_input_ids + req.output_ids[:-1]
        prefill_offloaded_len = (
            len(req.origin_input_ids) // self.page_size * self.page_size
        )
        state = self.offloaded_state.get(req.rid)
        if state is None:
            prefill_hashes = self._compute_prefix_hash(
                req.origin_input_ids[:prefill_offloaded_len]
            )
            last_prefill_hash = (
                prefill_hashes[-1] if prefill_offloaded_len > 0 else None
            )
            state = OffloadedState(
                prefill_len=prefill_offloaded_len,
                inc_len=0,
                last_hash=last_prefill_hash,
            )
            self.offloaded_state[req.rid] = state
        incremental_total = len(all_tokens) - state.prefill_len
        incremental_new = incremental_total - state.inc_len
        incremental_aligned_len = (
            incremental_new // self.offload_stride * self.offload_stride
        )

        if incremental_aligned_len == 0:
            return False

        # Extract incremental tokens and indices for the newly available chunk
        start = state.prefill_len + state.inc_len
        end = start + incremental_aligned_len
        incremental_tokens = all_tokens[start:end]
        incremental_indices = token_indices[start:end]
        page_hashes = self._compute_prefix_hash(incremental_tokens, state.last_hash)
        next_hash = page_hashes[-1] if page_hashes else state.last_hash

        # Prefill-aligned GPU slots are freed at request finish in
        # _release_finished_req, NOT here. The decoding request continues to
        # attend to those slots via req_to_token; freeing them mid-decode races
        # with concurrent admission, which can reuse the slots and produce
        # cross-pollinated KV reads.

        # Asynchronously offload incremental KV cache from device to host
        self.request_counter += 1
        ack_id = self.request_counter
        host_indices = self.cache_controller.write(
            device_indices=incremental_indices.long(),
            node_id=ack_id,
        )
        if host_indices is None:
            logger.error(f"Not enough host memory for request {req.rid}")
            return False

        self._mark_offload_started(req.rid)
        self.ongoing_offload[ack_id] = (
            req,
            host_indices,
            incremental_tokens,
            page_hashes,
            time.time(),
            start,
            end,
        )

        # SWA mirror: when the device pool is hybrid SWA, drive a parallel
        # D->H copy on the SWA sub-pool's controller. Each side independently
        # progresses to L3, so the FULL completion is not gated on SWA's.
        if self.is_swa and self.swa_cache_controller is not None:
            self._enqueue_swa_offload(
                req,
                ack_id,
                incremental_indices,
                incremental_tokens,
                page_hashes,
            )

        state.inc_len += incremental_aligned_len
        state.last_hash = next_hash
        return True

    def _enqueue_swa_offload(
        self,
        req,
        ack_id: int,
        incremental_indices: torch.Tensor,
        incremental_tokens: List[int],
        page_hashes: List[str],
    ) -> None:
        """Translate full pool indices to SWA pool indices and enqueue a SWA D->H copy.

        Pins the SWA slots with ``add_backup_pending`` so a concurrent
        ``maybe_evict_swa`` cannot recycle them while the backup is in flight.
        ``dec_backup_pending`` is paired in ``_check_swa_backup_progress`` once
        the L3 write returns.

        Failures here are non-fatal: a SWA-side dropout means the
        window-tail of this batch will not reach L3, but the FULL prefix
        backup proceeds and the request continues normally.
        """
        full_loc = incremental_indices.long()
        # translate_loc_from_full_to_swa returns int32, but the JIT kernel below
        # is bound to the host indices dtype (int64). Promote to keep the
        # cache_controller / hicache kernel signature consistent.
        swa_loc = self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
            full_loc
        ).long()
        # alloc_decode populates a SWA pair for every newly-allocated slot,
        # so the only realistic source of 0 entries here is a torn rollback
        # mid-batch. Skip the SWA path if any are unmapped.
        if not bool((swa_loc > 0).all().item()):
            logger.warning(
                "rid=%s: %d SWA slots unmapped — skipping SWA offload for ack %d",
                req.rid,
                int((swa_loc <= 0).sum().item()),
                ack_id,
            )
            return

        swa_host_indices = self.swa_cache_controller.write(
            device_indices=swa_loc, node_id=ack_id
        )
        if swa_host_indices is None:
            logger.warning(
                "rid=%s: SWA host pool exhausted; window-tail not backed up "
                "for ack %d",
                req.rid,
                ack_id,
            )
            return

        # Pin the SWA device slots until the L3 write returns.
        self.token_to_kv_pool_allocator.add_backup_pending(swa_loc)
        self._mark_offload_started(req.rid)
        self.ongoing_swa_offload[ack_id] = (
            req,
            swa_host_indices,
            swa_loc,
            incremental_tokens,
            page_hashes,
            time.time(),
        )

    def check_offload_progress(self):
        """Check the progress of offload from device to host and backup from host to storage.

        For SWA models we drain TWO independent pipelines:
        - the FULL controller (existing main KV channel)
        - the SWA controller (window-tail KV)
        We take the TP-wide MIN of all four queue sizes so every rank
        processes the same number of completions per tick.
        """
        cc = self.cache_controller
        swa_cc = self.swa_cache_controller

        if swa_cc is not None:
            qsizes = torch.tensor(
                [
                    len(cc.ack_write_queue),
                    cc.ack_backup_queue.qsize(),
                    len(swa_cc.ack_write_queue),
                    swa_cc.ack_backup_queue.qsize(),
                ],
                dtype=torch.int,
            )
        else:
            qsizes = torch.tensor(
                [
                    len(cc.ack_write_queue),
                    cc.ack_backup_queue.qsize(),
                ],
                dtype=torch.int,
            )
        if self.tp_world_size > 1:
            torch.distributed.all_reduce(
                qsizes, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
            )

        if swa_cc is not None:
            n_write, n_backup, n_swa_write, n_swa_backup = map(int, qsizes.tolist())
        else:
            n_write, n_backup = map(int, qsizes.tolist())
            n_swa_write = n_swa_backup = 0
        self._check_offload_progress(n_write)
        self._check_backup_progress(n_backup)
        if swa_cc is not None:
            self._check_swa_offload_progress(n_swa_write)
            self._check_swa_backup_progress(n_swa_backup)

    def _check_offload_progress(self, finish_count):
        """Check the progress of offload from device to host."""
        while finish_count > 0:
            _, finish_event, ack_list = self.cache_controller.ack_write_queue.pop(0)
            finish_event.synchronize()
            for ack_id in ack_list:
                (
                    req,
                    host_indices,
                    incremental_tokens,
                    page_hashes,
                    start_time,
                    start,
                    end,
                ) = self.ongoing_offload.pop(ack_id)

                self._mark_offload_finished(req.rid)
                self._trigger_backup(
                    req, host_indices, incremental_tokens, page_hashes, start_time
                )

                if req.finished() and not self._has_inflight_offload(req.rid):
                    state = self.offloaded_state.get(req.rid)
                    start_offset = state.prefill_len if state is not None else start
                    self._release_finished_req(req, start_offset)
            finish_count -= 1

    def _release_finished_req(self, req: Req, start_offset: int):
        # Defensive guard: ReqToTokenPool.free sets req_pool_idx to None, so a
        # previously released request must be skipped here to avoid non-idempotent
        # side effects such as protected_size_ double-decrement.
        if req.req_pool_idx is None or req.req_pool_idx == -1:
            return

        kv_committed_len = req.pop_committed_kv_cache()

        # Free the prefill-aligned slots here, after the request is guaranteed
        # to no longer attend to them.
        state = self.offloaded_state.get(req.rid)
        if state is not None and state.prefill_len > 0:
            prefill_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : state.prefill_len
            ]
            self.token_to_kv_pool_allocator.free(prefill_indices)

        start = start_offset
        end = kv_committed_len
        # Free the incremental part of the request (NSA-aware)
        kv_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx, start:end]
        self.token_to_kv_pool_allocator.free(kv_indices)

        # Free over-allocated KV cache slots (e.g. from speculative decoding v2).
        # Without spec v2, start_p == end_p so this is a no-op.
        start_p, end_p = req.pop_overallocated_kv_cache()
        if self.page_size > 1:
            start_p = ceil_align(start_p, self.page_size)
        if start_p < end_p:
            overalloc_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, start_p:end_p
            ]
            self.token_to_kv_pool_allocator.free(overalloc_indices)

        self.req_to_token_pool.free(req)
        self.tree_cache.protected_size_ -= len(req.prefix_indices)
        if req.rid in self.offloaded_state:
            del self.offloaded_state[req.rid]

    def _check_backup_progress(self, finish_count):
        """Check the progress of backup from host to storage."""
        for _ in range(finish_count):
            storage_operation = self.cache_controller.ack_backup_queue.get()
            ack_id = storage_operation.id
            req_id, host_indices, start_time = self.ongoing_backup.pop(ack_id)

            # Release host memory
            self.decode_host_mem_pool.free(host_indices)

            logger.debug(
                f"Finished backup request {req_id}, free host memory, len:{len(host_indices)}, cost time:{time.time() - start_time:.2f} seconds."
            )

    def _check_swa_offload_progress(self, finish_count):
        """Drain SWA D->H acks; trigger SWA L3 backup for each completed batch.

        Mirror of :meth:`_check_offload_progress` for the SWA sub-pool.
        Each ack groups several offload ack_ids together (HiCacheAck merges
        ops within one stream batch); for every ack_id we trigger the L3
        write so the window-tail bytes reach mooncake before the SWA host
        slot is freed in :meth:`_check_swa_backup_progress`.

        When SWA happens to be the last inflight piece for a finished req
        (FULL D->H drained first), this is also where the request gets
        released. Otherwise the Full progress loop releases.
        """
        while finish_count > 0:
            _, finish_event, ack_list = self.swa_cache_controller.ack_write_queue.pop(0)
            finish_event.synchronize()
            for ack_id in ack_list:
                if ack_id not in self.ongoing_swa_offload:
                    # Could be a stale ack from a request whose tracking we
                    # cleared (e.g., abort). Nothing to do.
                    continue
                (
                    req,
                    swa_host_indices,
                    swa_loc,
                    incremental_tokens,
                    page_hashes,
                    start_time,
                ) = self.ongoing_swa_offload.pop(ack_id)
                self._mark_offload_finished(req.rid)
                # Same per-page hash chain as FULL — token IDs drive both.
                state = self.offloaded_state.get(req.rid)
                self._trigger_swa_backup(
                    req,
                    swa_host_indices,
                    swa_loc,
                    incremental_tokens,
                    page_hashes,
                    start_time,
                )
                # If FULL already drained while SWA was still in flight, the
                # FULL release path skipped because _has_inflight_offload was
                # still True. Re-check here now that we've decremented.
                if req.finished() and not self._has_inflight_offload(req.rid):
                    start_offset = state.prefill_len if state is not None else 0
                    self._release_finished_req(req, start_offset)
            finish_count -= 1

    def _check_swa_backup_progress(self, finish_count):
        """Drain SWA L3 acks; release SWA host slots and clear backup-pending refs.

        Once the SWA L3 write returns, the host slot is no longer needed
        (reads come from L3 next time) and the device-side
        ``backup_pending_ref`` can drop, freeing any slots that
        ``maybe_evict_swa`` had deferred.
        """
        for _ in range(finish_count):
            storage_operation = self.swa_cache_controller.ack_backup_queue.get()
            ack_id = storage_operation.id
            if ack_id not in self.ongoing_swa_backup:
                continue
            req_id, swa_host_indices, swa_loc, start_time = self.ongoing_swa_backup.pop(
                ack_id
            )
            # 1) Drop the device-side pin so any pending free can proceed.
            self.token_to_kv_pool_allocator.dec_backup_pending(swa_loc)
            # 2) Release the SWA host slot itself.
            self.swa_host_mem_pool.free(swa_host_indices)
            logger.debug(
                "Finished SWA backup rid=%s len=%d cost=%.2fs",
                req_id,
                len(swa_host_indices),
                time.time() - start_time,
            )

    def _trigger_backup(
        self, req, host_indices, incremental_tokens, page_hashes, start_time
    ):
        """Trigger async backup from host to storage."""
        ack_id = self.cache_controller.write_storage(
            host_indices,
            incremental_tokens,
            hash_value=page_hashes,
        )
        self.ongoing_backup[ack_id] = (req.rid, host_indices, start_time)

    def _trigger_swa_backup(
        self,
        req,
        swa_host_indices,
        swa_loc,
        incremental_tokens,
        page_hashes,
        start_time,
    ):
        """Trigger async SWA host->L3 backup, paired with dec_backup_pending later.

        Page hashes are computed from the same token-id chain as the FULL
        path so a prefill node prefetching by token sequence will hit the
        SWA pages this decode side wrote.
        """
        ack_id = self.swa_cache_controller.write_storage(
            swa_host_indices,
            incremental_tokens,
            hash_value=page_hashes,
        )
        self.ongoing_swa_backup[ack_id] = (
            req.rid,
            swa_host_indices,
            swa_loc,
            start_time,
        )

    def _compute_prefix_hash(self, tokens, prior_hash=""):
        page_hashes = []
        last_hash = prior_hash
        for offset in range(0, len(tokens), self.page_size):
            page_tokens = tokens[offset : offset + self.page_size]
            last_hash = self.cache_controller.get_hash_str(page_tokens, last_hash)
            page_hashes.append(last_hash)
        return page_hashes

    def finalize_release_on_finish(self, req: Req):
        """Free any remaining tail KV that was not offloaded due to non-aligned length."""
        if req.req_pool_idx == -1:
            return

        if self._has_inflight_offload(req.rid):
            # The final tail is not worth a new offload, but earlier aligned
            # chunks are still being copied D->H. Release the whole request from
            # the progress callback after those slots are no longer being read.
            logger.debug(
                "Finalize release deferred for req %s because offload is still in flight.",
                req.rid,
            )
            return

        state = self.offloaded_state.get(req.rid)
        if state is None:
            start_offset = 0
        else:
            # Previous aligned chunks have already finished their D->H copy.
            # Free them together with the final unaligned tail now.
            start_offset = state.prefill_len
        self._release_finished_req(req, start_offset)
