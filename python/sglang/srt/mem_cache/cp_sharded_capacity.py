from __future__ import annotations

from collections.abc import Sequence

from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, EvictParams
from sglang.srt.mem_cache.cp_sharded_allocator import CPShardedKVPoolAllocator


def _capacity_deficits(
    available_tokens: Sequence[int], demand_tokens: Sequence[int]
) -> tuple[int, ...]:
    if len(available_tokens) != len(demand_tokens):
        raise ValueError("CP capacity and demand vectors must have the same length")
    if any(int(demand) < 0 for demand in demand_tokens):
        raise ValueError("CP capacity demand must be non-negative")
    return tuple(
        max(0, int(demand) - int(available))
        for available, demand in zip(available_tokens, demand_tokens)
    )


def ensure_cp_sharded_kv_capacity(
    *,
    allocator: CPShardedKVPoolAllocator,
    tree_cache: BasePrefixCache,
    demand_tokens: Sequence[int],
) -> bool:
    """Make enough room for one replicated CP-sharded allocation decision.

    Every CP scheduler executes the same logical radix operations and maintains
    the same owner ledger. Therefore the demand, deficits, and eviction request
    are deterministic on every rank and need no distributed synchronization.
    The logical demand is the sum of the per-owner physical demand vector.

    Radix eviction accepts only a logical token count, not an owner mask. Under
    pressure we evict one balanced estimate, inspect the resulting owner load,
    and continue only if the deficient owner still lacks room. The no-pressure
    path does not touch the radix cache.
    """

    demand = tuple(int(value) for value in demand_tokens)
    if len(demand) != allocator.cp_size:
        raise ValueError(
            f"CP capacity demand must have {allocator.cp_size} entries, "
            f"got {len(demand)}"
        )
    logical_demand = sum(demand)

    while True:
        snapshot = allocator.physical_load_snapshot()
        deficits = _capacity_deficits(snapshot.available_tokens, demand)
        max_deficit = max(deficits, default=0)
        logical_available = int(allocator.available_size())
        logical_deficit = max(0, logical_demand - logical_available)
        if max_deficit == 0 and logical_deficit == 0:
            return True

        eviction_tokens = max(
            max_deficit * allocator.cp_size,
            logical_deficit,
        )
        page_size = max(1, int(allocator.page_size))
        eviction_tokens = (
            (eviction_tokens + page_size - 1) // page_size * page_size
        )

        owns_free_group = allocator.is_not_in_free_group
        if owns_free_group:
            allocator.free_group_begin()
        try:
            result = tree_cache.evict(EvictParams(num_tokens=eviction_tokens))
        finally:
            if owns_free_group:
                allocator.free_group_end()
        if result.num_tokens_evicted <= 0:
            return False

        next_snapshot = allocator.physical_load_snapshot()
        if (
            next_snapshot.allocated_tokens == snapshot.allocated_tokens
            and int(allocator.available_size()) == logical_available
        ):
            return False
