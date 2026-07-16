"""Pure CPU planning for context-parallel prefill token ownership."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CPBlock:
    logical_start: int
    token_count: int
    owner_rank: int


@dataclass(frozen=True)
class CPPrefillSplitSpec:
    extend_start: int
    extend_len: int
    owner_rotation: int
    blocks: tuple[CPBlock, ...]
    per_rank_tokens: tuple[int, ...]

    def local_blocks(self, cp_rank: int) -> tuple[CPBlock, ...]:
        if cp_rank < 0 or cp_rank >= len(self.per_rank_tokens):
            raise ValueError(
                f"cp_rank must be in [0, {len(self.per_rank_tokens)}), got {cp_rank}"
            )
        return tuple(block for block in self.blocks if block.owner_rank == cp_rank)

    def page_demand(self, page_size: int) -> tuple[int, ...]:
        cp_size = len(self.per_rank_tokens)
        self.validate(cp_size=cp_size, page_size=page_size)

        demand = [0] * cp_size
        for block_index, block in enumerate(self.blocks):
            block_end = block.logical_start + block.token_count
            first_page = block.logical_start // page_size
            last_page = (block_end - 1) // page_size
            if block_index == 0 and block.logical_start % page_size:
                # This page already exists and is represented by its recorded owner.
                first_page += 1
            demand[block.owner_rank] += max(0, last_page - first_page + 1)
        return tuple(demand)

    def validate(self, cp_size: int, page_size: int) -> None:
        if cp_size <= 0:
            raise ValueError("cp_size must be positive")
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        if self.extend_start < 0:
            raise ValueError("extend_start must be non-negative")
        if self.extend_len < 0:
            raise ValueError("extend_len must be non-negative")
        if len(self.per_rank_tokens) != cp_size:
            raise ValueError("per_rank_tokens must have one entry per CP rank")
        if self.owner_rotation < 0 or self.owner_rotation >= cp_size:
            raise ValueError("owner_rotation must be in [0, cp_size)")

        expected_start = self.extend_start
        counted_tokens = [0] * cp_size
        new_block_index = 0
        for block_index, block in enumerate(self.blocks):
            if block.token_count <= 0:
                raise ValueError("CP blocks must have positive token_count")
            if block.owner_rank < 0 or block.owner_rank >= cp_size:
                raise ValueError("CP block owner is outside the CP group")
            if block.logical_start != expected_start:
                raise ValueError("CP blocks must cover the extend interval without gaps")

            block_end = block.logical_start + block.token_count
            is_last_block = block_index == len(self.blocks) - 1
            if block_index and block.logical_start % page_size:
                raise ValueError("interior CP block starts must be page aligned")
            if not is_last_block and block_end % page_size:
                raise ValueError("interior CP block ends must be page aligned")

            is_leading_resident_fragment = (
                block_index == 0 and block.logical_start % page_size
            )
            if is_leading_resident_fragment:
                resident_fragment_len = min(
                    page_size - (self.extend_start % page_size), self.extend_len
                )
                if block.token_count != resident_fragment_len:
                    raise ValueError(
                        "leading resident fragment must cover exactly its resident page"
                    )
            if not is_leading_resident_fragment:
                if new_block_index >= 2 * cp_size:
                    raise ValueError(
                        "CP blocks may not exceed 2 * cp_size newly allocated "
                        "zigzag blocks"
                    )
                expected_owner = _rotated_zigzag_owner(
                    new_block_index, cp_size, self.owner_rotation
                )
                if block.owner_rank != expected_owner:
                    raise ValueError(
                        "CP block owner must match the rotated zigzag order"
                    )
                new_block_index += 1

            counted_tokens[block.owner_rank] += block.token_count
            expected_start = block_end

        if expected_start != self.extend_start + self.extend_len:
            raise ValueError("CP blocks must cover exactly extend_len tokens")
        if tuple(counted_tokens) != self.per_rank_tokens:
            raise ValueError("per_rank_tokens must match the scheduled blocks")


def build_cp_prefill_split_spec(
    *,
    extend_start: int,
    extend_len: int,
    cp_size: int,
    page_size: int,
    owner_rotation: int,
    leading_page_owner: int | None = None,
) -> CPPrefillSplitSpec:
    """Build a page-aligned, rotated-zigzag CP ownership plan for one extend."""
    if cp_size <= 0:
        raise ValueError("cp_size must be positive")
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if extend_start < 0:
        raise ValueError("extend_start must be non-negative")
    if extend_len < 0:
        raise ValueError("extend_len must be non-negative")
    if leading_page_owner is not None and not 0 <= leading_page_owner < cp_size:
        raise ValueError("leading_page_owner is outside the CP group")

    blocks: list[CPBlock] = []
    logical_end = extend_start + extend_len
    next_start = extend_start

    if extend_len and extend_start % page_size:
        if leading_page_owner is None:
            raise ValueError("leading_page_owner is required for a partial leading page")
        leading_len = min(page_size - (extend_start % page_size), extend_len)
        blocks.append(CPBlock(extend_start, leading_len, leading_page_owner))
        next_start += leading_len

    new_page_count = (logical_end - next_start + page_size - 1) // page_size
    block_count = min(2 * cp_size, new_page_count)
    if block_count:
        pages_per_block, extra_pages = divmod(new_page_count, block_count)
        for block_index in range(block_count):
            page_count = pages_per_block + (block_index < extra_pages)
            token_count = min(page_count * page_size, logical_end - next_start)
            owner = _rotated_zigzag_owner(block_index, cp_size, owner_rotation)
            blocks.append(CPBlock(next_start, token_count, owner))
            next_start += token_count

    per_rank_tokens = tuple(
        sum(block.token_count for block in blocks if block.owner_rank == cp_rank)
        for cp_rank in range(cp_size)
    )
    spec = CPPrefillSplitSpec(
        extend_start=extend_start,
        extend_len=extend_len,
        owner_rotation=owner_rotation,
        blocks=tuple(blocks),
        per_rank_tokens=per_rank_tokens,
    )
    spec.validate(cp_size=cp_size, page_size=page_size)
    return spec


def _rotated_zigzag_owner(
    block_index: int, cp_size: int, owner_rotation: int
) -> int:
    if block_index < cp_size:
        owner = block_index
    else:
        owner = 2 * cp_size - block_index - 1
    return (owner + owner_rotation) % cp_size
