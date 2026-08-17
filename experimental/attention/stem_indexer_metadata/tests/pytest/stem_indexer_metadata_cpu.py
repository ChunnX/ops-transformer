"""Pure-CPU reference for the A6 StemIndexer metadata scheduler.

This module mirrors the StemIndexer-specific configuration of SectionStreamK:
64 AIC/64 AIV, 96x256 basic blocks, 96 MiB L2 sectioning, and FD disabled.
The generated metadata uses the same uint32 field layout as the AICPU producer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Union


AIC_CORE_NUM = 64
AIV_CORE_NUM = 64
HEAD_METADATA_STRIDE = 16
FA_METADATA_STRIDE = 16
AIV_RESERVED_METADATA_STRIDE = 16
METADATA_ALIGN_SIZE = 4096

M_BASE_SIZE = 96
S2_BASE_SIZE = 256
L2_BYTES = 96 * 1024 * 1024
FA_TOLERANCE_RATIO = 2
UINT32_MAX = (1 << 32) - 1

Position = Tuple[int, int, int]


@dataclass(frozen=True)
class CoreSchedule:
    """Metadata range assigned to one logical AIC."""

    start: Position
    end: Position
    cost: int


@dataclass(frozen=True)
class SectionSchedule:
    """One L2 section and the AIC ranges selected for it."""

    bn2_end: int
    cores: Tuple[CoreSchedule, ...]

    @property
    def used_core_num(self) -> int:
        return len(self.cores)


@dataclass(frozen=True)
class StemIndexerSchedule:
    """Debug-friendly result of the A6 metadata scheduler."""

    sections: Tuple[SectionSchedule, ...]
    q_block_lens: Tuple[int, ...]
    kv_block_lens: Tuple[int, ...]
    m_base_size: int = M_BASE_SIZE
    s2_base_size: int = S2_BASE_SIZE

    @property
    def section_num(self) -> int:
        return len(self.sections)


@dataclass(frozen=True)
class _RowCost:
    s2_start: int
    s2_end: int
    cost: int
    last_block_cost: int
    block_num: int


@dataclass(frozen=True)
class _BatchCost:
    rows: Tuple[_RowCost, ...]
    total_cost: int
    total_blocks: int
    last_block_cost: int
    s1_size: int
    s2_size: int


def _ceil_div(value: int, divisor: int) -> int:
    if divisor == 0:
        return 0
    return value // divisor + (1 if value % divisor else 0)


def _within_tolerance(limit: int, tolerance: int, value: int) -> bool:
    return limit + tolerance >= value


def _to_int_list(values: Union[Sequence[int], Iterable[int]]) -> List[int]:
    if hasattr(values, "detach"):
        values = values.detach().cpu().reshape(-1).tolist()
    return [int(value) for value in values]


def _validate_inputs(
    q_seq_lens: Sequence[int],
    kv_seq_lens: Sequence[int],
    q_heads: int,
    kv_heads: int,
    stem_block_size: int,
    dim_qkflat: int,
    window_size: int,
) -> None:
    if not q_seq_lens:
        raise ValueError("q_seq_lens must contain at least one element")
    if len(q_seq_lens) != len(kv_seq_lens):
        raise ValueError("q_seq_lens and kv_seq_lens must have the same length")
    if any(value < 0 for value in q_seq_lens) or any(value < 0 for value in kv_seq_lens):
        raise ValueError("sequence lengths must be non-negative")
    if q_heads <= 0 or kv_heads <= 0:
        raise ValueError("q_heads and kv_heads must be positive")
    if q_heads % kv_heads != 0:
        raise ValueError("q_heads must be divisible by kv_heads")
    if stem_block_size != 128:
        raise ValueError("stem_block_size must be 128")
    if dim_qkflat <= 0:
        raise ValueError("dim_qkflat must be positive")
    if window_size < 0:
        raise ValueError("window_size must be non-negative")


def _calc_cost(m_size: int, s2_size: int) -> int:
    return 6 * _ceil_div(m_size, 16) + 10 * _ceil_div(s2_size, 64)


def _calc_s2_range(
    s1g_idx: int,
    s1_size: int,
    s2_size: int,
    group_size: int,
    causal: bool,
    window_size: int,
) -> Tuple[int, int]:
    if s1_size == 0 or s2_size == 0:
        return 0, 0
    if not causal:
        return 0, _ceil_div(s2_size, S2_BASE_SIZE)

    s1g_first_token = s1g_idx * M_BASE_SIZE
    s1g_last_token = min(s1g_first_token + M_BASE_SIZE, s1_size * group_size) - 1

    # StemIndexer uses Layout::BUTT, so S1G is interpreted as grouped rows.
    if s1g_first_token // s1_size == s1g_last_token // s1_size:
        s1_first_token = s1g_first_token % s1_size
        s1_last_token = s1g_last_token % s1_size
    else:
        s1_first_token = 0
        # Keep the C++ SectionStreamK behavior at a group boundary.
        s1_last_token = s1_size

    pre_token_left_up = UINT32_MAX
    next_token_left_up = s2_size - s1_size - window_size
    s2_first_token = s1_first_token - pre_token_left_up
    s2_last_token = s1_last_token + next_token_left_up

    if s2_first_token >= s2_size or s2_last_token < 0 or s2_last_token < s2_first_token:
        return 0, 0

    s2_first_token = min(max(s2_first_token, 0), s2_size - 1)
    s2_last_token = min(max(s2_last_token, 0), s2_size - 1)
    return s2_first_token // S2_BASE_SIZE, s2_last_token // S2_BASE_SIZE + 1


def _build_batch_cost(
    s1_size: int,
    s2_size: int,
    group_size: int,
    causal: bool,
    window_size: int,
) -> _BatchCost:
    m_extent = s1_size * group_size
    m_base_num = _ceil_div(m_extent, M_BASE_SIZE)
    m_tail_size = m_extent % M_BASE_SIZE
    s2_base_num = _ceil_div(s2_size, S2_BASE_SIZE)
    s2_tail_size = s2_size % S2_BASE_SIZE

    normal_normal = _calc_cost(M_BASE_SIZE, S2_BASE_SIZE)
    tail_normal = _calc_cost(m_tail_size, S2_BASE_SIZE) if m_tail_size else 0
    normal_tail = _calc_cost(M_BASE_SIZE, s2_tail_size) if s2_tail_size else 0
    tail_tail = _calc_cost(m_tail_size, s2_tail_size) if m_tail_size and s2_tail_size else 0

    rows: List[_RowCost] = []
    total_cost = 0
    total_blocks = 0
    last_nonempty_block_cost = 0
    for s1g_idx in range(m_base_num):
        s2_start, s2_end = _calc_s2_range(
            s1g_idx, s1_size, s2_size, group_size, causal, window_size
        )
        block_num = max(s2_end - s2_start, 0)
        if block_num == 0:
            row = _RowCost(s2_start, s2_end, 0, 0, 0)
            rows.append(row)
            continue

        has_s2_tail = s2_tail_size != 0 and s2_end == s2_base_num
        normal_s2_num = block_num - (1 if has_s2_tail else 0)
        is_m_tail = s1g_idx == m_base_num - 1 and m_tail_size != 0
        if is_m_tail:
            normal_block_cost = tail_normal
            last_block_cost = tail_tail if has_s2_tail else tail_normal
        else:
            normal_block_cost = normal_normal
            last_block_cost = normal_tail if has_s2_tail else normal_normal

        row_cost = normal_block_cost * normal_s2_num + (last_block_cost if has_s2_tail else 0)
        row = _RowCost(s2_start, s2_end, row_cost, last_block_cost, block_num)
        rows.append(row)
        total_cost += row_cost
        total_blocks += block_num
        last_nonempty_block_cost = last_block_cost

    return _BatchCost(
        tuple(rows), total_cost, total_blocks, last_nonempty_block_cost, s1_size, s2_size
    )


def _calc_section_bn2_ends(
    batch_costs: Sequence[_BatchCost], kv_heads: int, group_size: int, dim_qkflat: int
) -> List[int]:
    section_ends: List[int] = []
    bn2_idx = 0
    section_bytes = 0
    max_grouped_s1 = 0
    max_single_head_bytes = 0

    for batch in batch_costs:
        # Q/O and K/V are each counted as two FP16 tensors, matching SectionStreamK.
        single_head_bytes = 4 * dim_qkflat * (batch.s1_size + batch.s2_size)
        max_single_head_bytes = max(max_single_head_bytes, single_head_bytes)
        max_grouped_s1 = max(max_grouped_s1, group_size * batch.s1_size)
        for _ in range(kv_heads):
            if section_bytes and section_bytes + single_head_bytes > L2_BYTES:
                section_ends.append(bn2_idx)
                section_bytes = 0
            section_bytes += single_head_bytes
            bn2_idx += 1

    section_ends.append(len(batch_costs) * kv_heads)
    if max_grouped_s1 <= M_BASE_SIZE or max_single_head_bytes <= L2_BYTES // AIC_CORE_NUM:
        return [len(batch_costs) * kv_heads]
    return section_ends


def _row_at(batch: _BatchCost, m_idx: int, group_size: int, causal: bool, window_size: int) -> _RowCost:
    if m_idx < len(batch.rows):
        return batch.rows[m_idx]
    s2_start, s2_end = _calc_s2_range(
        m_idx, batch.s1_size, batch.s2_size, group_size, causal, window_size
    )
    return _RowCost(s2_start, s2_end, 0, 0, 0)


def _schedule_section(
    section_start: int,
    section_end: int,
    batch_costs: Sequence[_BatchCost],
    kv_heads: int,
    group_size: int,
    causal: bool,
    window_size: int,
    previous_end: Position,
) -> Tuple[SectionSchedule, Position]:
    section_cost = sum(batch_costs[bn2 // kv_heads].total_cost for bn2 in range(section_start, section_end))
    section_blocks = sum(batch_costs[bn2 // kv_heads].total_blocks for bn2 in range(section_start, section_end))

    if section_blocks == 0:
        end = (section_end, 0, 0)
        core = CoreSchedule(previous_end, end, 0)
        return SectionSchedule(section_end, (core,)), end

    core_num = min(AIC_CORE_NUM, section_blocks)
    cur_bn2 = section_start
    cur_m = 0
    batch = batch_costs[cur_bn2 // kv_heads]
    remaining_head_cost = batch.total_cost
    remaining_head_blocks = batch.total_blocks
    remaining_section_cost = section_cost
    finished = False
    core_ends: List[Tuple[Position, int]] = []

    for core_idx in range(core_num):
        if finished or remaining_section_cost <= 0:
            break

        row = _row_at(batch, cur_m, group_size, causal, window_size)
        core_cost = 0
        cost_limit = remaining_section_cost // (core_num - core_idx)
        cost_limit = max(cost_limit, row.cost)

        while remaining_head_cost == 0 or _within_tolerance(
            cost_limit,
            batch.last_block_cost // FA_TOLERANCE_RATIO,
            core_cost + remaining_head_cost,
        ):
            core_cost += remaining_head_cost
            cur_bn2 += 1
            if cur_bn2 == section_end:
                cur_m = 0
                finished = True
                break

            next_batch = batch_costs[cur_bn2 // kv_heads]
            batch = next_batch
            remaining_head_cost = batch.total_cost
            remaining_head_blocks = batch.total_blocks
            cur_m = 0
            row = _row_at(batch, cur_m, group_size, causal, window_size)

        if not finished:
            while cur_m < len(batch.rows):
                row = batch.rows[cur_m]
                if not _within_tolerance(
                    cost_limit,
                    row.last_block_cost // FA_TOLERANCE_RATIO,
                    core_cost + row.cost,
                ):
                    break
                core_cost += row.cost
                remaining_head_cost = max(remaining_head_cost - row.cost, 0)
                remaining_head_blocks = max(remaining_head_blocks - row.block_num, 0)
                cur_m += 1
                while cur_m < len(batch.rows) and batch.rows[cur_m].block_num == 0:
                    cur_m += 1

        row = _row_at(batch, cur_m, group_size, causal, window_size) if not finished else None
        end = (cur_bn2, cur_m, 0 if row is None else row.s2_start)
        core_ends.append((end, core_cost))
        remaining_section_cost -= core_cost

    cores: List[CoreSchedule] = []
    start = previous_end
    for end, cost in core_ends:
        cores.append(CoreSchedule(start, end, cost))
        start = end
    return SectionSchedule(section_end, tuple(cores)), start


def build_stem_indexer_schedule(
    q_seq_lens: Union[Sequence[int], Iterable[int]],
    kv_seq_lens: Union[Sequence[int], Iterable[int]],
    q_heads: int,
    kv_heads: int,
    causal: bool = True,
    stem_block_size: int = 128,
    dim_qkflat: int = 2048,
    window_size: int = 4,
) -> StemIndexerSchedule:
    """Build the A6 SectionStreamK schedule without requiring PyTorch or an NPU."""

    q_values = _to_int_list(q_seq_lens)
    kv_values = _to_int_list(kv_seq_lens)
    _validate_inputs(q_values, kv_values, q_heads, kv_heads, stem_block_size, dim_qkflat, window_size)

    q_blocks = tuple(_ceil_div(value, stem_block_size) for value in q_values)
    kv_blocks = tuple(_ceil_div(value, stem_block_size) for value in kv_values)
    group_size = q_heads // kv_heads
    batch_costs = tuple(
        _build_batch_cost(s1, s2, group_size, causal, window_size)
        for s1, s2 in zip(q_blocks, kv_blocks)
    )
    section_ends = _calc_section_bn2_ends(batch_costs, kv_heads, group_size, dim_qkflat)

    sections: List[SectionSchedule] = []
    section_start = 0
    previous_end: Position = (0, 0, 0)
    for section_end in section_ends:
        section, previous_end = _schedule_section(
            section_start,
            section_end,
            batch_costs,
            kv_heads,
            group_size,
            causal,
            window_size,
            previous_end,
        )
        sections.append(section)
        section_start = section_end

    return StemIndexerSchedule(tuple(sections), q_blocks, kv_blocks)


def _metadata_capacity(batch_size: int, kv_heads: int) -> int:
    max_section_num = batch_size * kv_heads
    required = HEAD_METADATA_STRIDE + max_section_num * (
        AIC_CORE_NUM * FA_METADATA_STRIDE + AIV_CORE_NUM * AIV_RESERVED_METADATA_STRIDE
    )
    return _ceil_div(required, METADATA_ALIGN_SIZE) * METADATA_ALIGN_SIZE


def generate_stem_indexer_metadata_values(
    q_seq_lens: Union[Sequence[int], Iterable[int]],
    kv_seq_lens: Union[Sequence[int], Iterable[int]],
    q_heads: int,
    kv_heads: int,
    causal: bool = True,
    stem_block_size: int = 128,
    dim_qkflat: int = 2048,
    window_size: int = 4,
    return_schedule: bool = False,
):
    """Return metadata as Python integers, optionally with its decoded schedule."""

    q_values = _to_int_list(q_seq_lens)
    kv_values = _to_int_list(kv_seq_lens)
    schedule = build_stem_indexer_schedule(
        q_values,
        kv_values,
        q_heads,
        kv_heads,
        causal,
        stem_block_size,
        dim_qkflat,
        window_size,
    )
    metadata = [0] * _metadata_capacity(len(q_values), kv_heads)
    metadata[0] = schedule.section_num

    for section_idx, section in enumerate(schedule.sections):
        section_base = HEAD_METADATA_STRIDE + section_idx * AIC_CORE_NUM * FA_METADATA_STRIDE
        for core_idx, core in enumerate(section.cores):
            core_base = section_base + core_idx * FA_METADATA_STRIDE
            metadata[core_base : core_base + 6] = [*core.start, *core.end]

    if return_schedule:
        return metadata, schedule
    return metadata


def generate_stem_indexer_metadata_cpu(
    q_seq_lens,
    kv_seq_lens,
    q_heads: int,
    kv_heads: int,
    causal: bool = True,
    stem_block_size: int = 128,
    dim_qkflat: int = 2048,
    window_size: int = 4,
    return_schedule: bool = False,
):
    """Return a CPU torch.int32 tensor compatible with StemIndexer metadata input."""

    values, schedule = generate_stem_indexer_metadata_values(
        q_seq_lens,
        kv_seq_lens,
        q_heads,
        kv_heads,
        causal,
        stem_block_size,
        dim_qkflat,
        window_size,
        return_schedule=True,
    )
    try:
        import torch
    except ImportError as error:  # pragma: no cover - depends on the caller environment
        raise RuntimeError("PyTorch is required for tensor output; use generate_stem_indexer_metadata_values instead") from error

    metadata = torch.tensor(values, dtype=torch.int32, device="cpu")
    if return_schedule:
        return metadata, schedule
    return metadata


__all__ = [
    "AIC_CORE_NUM",
    "AIV_CORE_NUM",
    "HEAD_METADATA_STRIDE",
    "M_BASE_SIZE",
    "S2_BASE_SIZE",
    "CoreSchedule",
    "SectionSchedule",
    "StemIndexerSchedule",
    "build_stem_indexer_schedule",
    "generate_stem_indexer_metadata_values",
    "generate_stem_indexer_metadata_cpu",
]
