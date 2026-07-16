from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class CPShardedKVLoadSnapshot:
    capacity_tokens: tuple[int, ...]
    allocated_tokens: tuple[int, ...]
    available_tokens_override: tuple[int, ...] | None = None

    @property
    def available_tokens(self) -> tuple[int, ...]:
        if self.available_tokens_override is not None:
            return self.available_tokens_override
        return tuple(
            capacity - allocated
            for capacity, allocated in zip(
                self.capacity_tokens, self.allocated_tokens
            )
        )


@dataclass(frozen=True)
class CPLogicalOwnerPlan:
    owner_ranks: torch.Tensor
    per_rank_counts: tuple[int, ...]
    rank_packed_to_logical: torch.Tensor


class CPShardedKVResidencyLedger:
    """Replicated logical-owner accounting without distributed collectives.

    The map and counters stay on the allocator device so logical allocate/free
    operations do not introduce a device-to-host synchronization. Admission
    reads only the ``cp_size`` counters when it needs a capacity decision.
    """

    def __init__(
        self,
        *,
        cp_size: int,
        logical_unit_capacity: int,
        allocation_unit_tokens: int,
        physical_capacity_tokens_per_rank: int,
        device: torch.device | str,
    ):
        if cp_size <= 0:
            raise ValueError("cp_size must be positive")
        if logical_unit_capacity < 0:
            raise ValueError("logical_unit_capacity must be non-negative")
        if allocation_unit_tokens <= 0:
            raise ValueError("allocation_unit_tokens must be positive")
        if physical_capacity_tokens_per_rank < 0:
            raise ValueError("physical capacity must be non-negative")

        self.cp_size = int(cp_size)
        self.allocation_unit_tokens = int(allocation_unit_tokens)
        self.device = torch.device(device)
        self._capacity_tokens = tuple(
            int(physical_capacity_tokens_per_rank) for _ in range(self.cp_size)
        )
        self._owner_by_logical_unit = torch.full(
            (int(logical_unit_capacity) + 1,),
            -1,
            dtype=torch.int16,
            device=self.device,
        )
        self._allocated_units = torch.zeros(
            (self.cp_size,), dtype=torch.int64, device=self.device
        )
        self._snapshot_cache: CPShardedKVLoadSnapshot | None = (
            CPShardedKVLoadSnapshot(
                capacity_tokens=self._capacity_tokens,
                allocated_tokens=(0,) * self.cp_size,
            )
        )

    def _to_device_long(
        self, values: torch.Tensor | Iterable[int]
    ) -> torch.Tensor:
        if isinstance(values, torch.Tensor):
            return values.detach().reshape(-1).to(
                device=self.device, dtype=torch.int64
            )
        return torch.tensor(list(values), dtype=torch.int64, device=self.device)

    def assign(
        self,
        logical_units: torch.Tensor | Iterable[int],
        owners: torch.Tensor | Iterable[int],
    ) -> None:
        units = self._to_device_long(logical_units)
        owner_tensor = self._to_device_long(owners)
        if units.numel() != owner_tensor.numel():
            raise ValueError("logical_units and owners must have the same length")
        if units.numel() == 0:
            return
        self._owner_by_logical_unit[units] = owner_tensor.to(torch.int16)
        self._allocated_units.scatter_add_(
            0,
            owner_tensor,
            torch.ones_like(owner_tensor, dtype=torch.int64),
        )
        self._snapshot_cache = None

    def release(
        self, logical_units: torch.Tensor | Iterable[int]
    ) -> torch.Tensor:
        # The allocator canonicalizes token IDs to unique logical units before
        # calling this method. Avoid another unique/sort on the cache free path.
        units = self._to_device_long(logical_units)
        units = units[units > 0]
        if units.numel() == 0:
            return units

        owners = self._owner_by_logical_unit[units].to(torch.int64)
        live_mask = owners >= 0
        self._allocated_units.scatter_add_(
            0,
            torch.clamp_min(owners, 0),
            -live_mask.to(dtype=torch.int64),
        )
        self._owner_by_logical_unit[units] = -1
        self._snapshot_cache = None
        return units[live_mask]

    def snapshot(self) -> CPShardedKVLoadSnapshot:
        if self._snapshot_cache is None:
            self._snapshot_cache = CPShardedKVLoadSnapshot(
                capacity_tokens=self._capacity_tokens,
                allocated_tokens=tuple(
                    int(units) * self.allocation_unit_tokens
                    for units in self._allocated_units.detach().cpu().tolist()
                ),
            )
        return self._snapshot_cache

    def owner_ranks_for_logical_units(
        self, logical_units: torch.Tensor | Iterable[int]
    ) -> torch.Tensor:
        units = self._to_device_long(logical_units)
        if units.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=self.device)

        valid_units = (units > 0) & (
            units < self._owner_by_logical_unit.numel()
        )
        safe_units = torch.where(valid_units, units, torch.zeros_like(units))
        owner_ranks = self._owner_by_logical_unit[safe_units].to(torch.int64)
        return torch.where(
            valid_units,
            owner_ranks,
            torch.full_like(owner_ranks, -1),
        )

    def owner_plan_for_logical_units(
        self, logical_units: torch.Tensor | Iterable[int]
    ) -> CPLogicalOwnerPlan:
        units = self._to_device_long(logical_units)
        if units.numel() == 0:
            return CPLogicalOwnerPlan(
                owner_ranks=torch.empty(
                    (0,), dtype=torch.int64, device=self.device
                ),
                per_rank_counts=(0,) * self.cp_size,
                rank_packed_to_logical=torch.empty(
                    (0,), dtype=torch.int64, device=self.device
                ),
            )

        owner_ranks = self.owner_ranks_for_logical_units(units)
        valid_owner = (owner_ranks >= 0) & (owner_ranks < self.cp_size)
        count_buckets = torch.where(
            valid_owner,
            owner_ranks,
            torch.full_like(owner_ranks, self.cp_size),
        )
        counts_with_invalid = torch.bincount(
            count_buckets, minlength=self.cp_size + 1
        )
        counts_cpu = counts_with_invalid.detach().cpu().tolist()
        if counts_cpu[self.cp_size] != 0:
            raise ValueError("logical owner query contains an invalid logical unit")

        return CPLogicalOwnerPlan(
            owner_ranks=owner_ranks,
            per_rank_counts=tuple(int(count) for count in counts_cpu[: self.cp_size]),
            rank_packed_to_logical=torch.argsort(owner_ranks, stable=True),
        )

    def backup_state(self):
        return (
            self._owner_by_logical_unit.clone(),
            self._allocated_units.clone(),
        )

    def restore_state(self, state) -> None:
        owners, allocated_units = state
        if len(allocated_units) != self.cp_size:
            raise ValueError("invalid CP residency ledger state")
        self._owner_by_logical_unit = owners.clone().to(device=self.device)
        self._allocated_units = torch.as_tensor(
            allocated_units, dtype=torch.int64, device=self.device
        ).clone()
        self._snapshot_cache = None
