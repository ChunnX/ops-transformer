"""Metadata-v8 simulator for GenericBlockSparseAttention ATK tests.

This module mirrors the AICPU scheduler in
``generic_block_sparse_attention_metadata_scheduler.h``.  It deliberately contains
no attention math: the serialized metadata drives the NPU kernel and the
structured plan is consumed by the Python FD golden.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple

import torch


ARCH35_AIC_NUM = 28
METADATA_TOTAL_SIZE = 1024
METADATA_MAGIC = 0x5341534D
METADATA_VERSION = 8
METADATA_USED_SIZE = 272

FD_SCHEDULE_ENABLED = 1 << 0
FD_ACTUAL_BLOCK_PREFIX = 1 << 1
MAX_AIC_CORE_NUM = 36
MAX_DECODE_CORE_NUM = 32
MAX_COMBINE_TASK_NUM = 32
MAX_FD_ACTIVE_CORE_NUM = MAX_DECODE_CORE_NUM
MAX_FD_PARTIAL_PER_BASE_TASK = MAX_DECODE_CORE_NUM
MAX_SPARSE_BLOCK_CAPACITY = 256

FD_COST_M16 = 125
FD_COST_N = 740
FD_COST_M16_N = 35
FD_LAUNCH_COST = 278
FLOATS_PER_BLOCK = 8

MAGIC_INDEX = 0
VERSION_INDEX = 1
METADATA_USED_SIZE_INDEX = 2
SA_USED_CORE_NUM_INDEX = 3
SA_TOTAL_TASK_NUM_INDEX = 4
FD_ACTIVE_CORE_NUM_INDEX = 5
DECODE_PER_CORE_TASK_NUM_INDEX = 6
COMBINE_TASK_NUM_INDEX = 7
FD_SCHEDULE_FLAGS_INDEX = 8
FD_TOTAL_FLAT_TASK_NUM_INDEX = 9
FD_PARTIAL_TASK_NUM_INDEX = 10
CONFIG_SIGNATURE_INDEX = 11
HEADER_SIZE = 16

DECODE_SCHEDULE_OFFSET = HEADER_SIZE
DECODE_SCHEDULE_FIELD_NUM = 4
COMBINE_SCHEDULE_OFFSET = (
    DECODE_SCHEDULE_OFFSET + MAX_DECODE_CORE_NUM * DECODE_SCHEDULE_FIELD_NUM
)
COMBINE_SCHEDULE_FIELD_NUM = 4


def _ceil_div(value: int, divisor: int) -> int:
    if divisor <= 0:
        raise ValueError(f"divisor must be positive, got {divisor}")
    return (value + divisor - 1) // divisor if value > 0 else 0


def _round_up(value: int, alignment: int) -> int:
    return _ceil_div(value, alignment) * alignment


def _int_values(value: Any, name: str) -> list[int]:
    if value is None:
        raise ValueError(f"{name} must not be None")
    if isinstance(value, torch.Tensor):
        return [int(item) for item in value.detach().cpu().view(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    raise TypeError(f"{name} must be a tensor or an integer sequence")


def _prefix_sum(values: Iterable[int]) -> list[int]:
    result = [0]
    for value in values:
        result.append(result[-1] + int(value))
    return result


def _signed_int32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value if value < 0x80000000 else value - 0x100000000


def _hash_config_value(hash_value: int, value: int) -> int:
    value &= 0xFFFFFFFF
    for shift in range(0, 32, 8):
        hash_value ^= (value >> shift) & 0xFF
        hash_value = (hash_value * 16777619) & 0xFFFFFFFF
    return hash_value


def calculate_config_signature(
    *,
    batch: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_shape_x: int,
    block_shape_y: int,
    block_index_stride: int,
    q_block_storage_num: int,
    is_packed_gqa: int,
) -> int:
    """Mirror both AICPU and Kernel FNV-1a config hashing."""

    hash_value = 2166136261
    for value in (
        batch,
        num_q_heads,
        num_kv_heads,
        head_dim,
        block_shape_x,
        block_shape_y,
        block_index_stride,
        q_block_storage_num,
        is_packed_gqa,
    ):
        hash_value = _hash_config_value(hash_value, value)
    return _signed_int32(hash_value)


@dataclass(frozen=True)
class FdTaskSlice:
    base_task: int
    raw_begin: int
    raw_end: int
    partial_task_id: Optional[int]


@dataclass(frozen=True)
class FdCoreRange:
    core_idx: int
    flat_start: int
    flat_end: int
    base_task_start: int
    base_task_end: int
    first_block_start: int
    last_block_end: int
    task_slices: Tuple[FdTaskSlice, ...]


@dataclass(frozen=True)
class FdPlan:
    enabled: bool
    metadata_fd_enabled: bool
    reason: str
    physical_aic_num: int
    normal_used_core_num: int
    total_q_tokens: int
    total_base_tasks: int
    total_split_tasks: int
    top_k: int
    group_size: int
    total_cost: int
    best_core_num: int
    per_core_task_num: int
    active_core_num: int
    fd_lse_sub_stride: int
    partial_task_num: int
    combine_base_tasks: Tuple[int, ...]
    partial_start_by_base: Tuple[int, ...]
    partial_count_by_base: Tuple[int, ...]
    block_prefix: Tuple[int, ...]
    core_ranges: Tuple[FdCoreRange, ...]
    fd_schedule_flags: int
    config_signature: int
    metadata_version: int = METADATA_VERSION
    metadata_used_size: int = METADATA_USED_SIZE

    @property
    def pipeline_mode(self) -> str:
        return "fd" if self.enabled else "normal"

    @property
    def combine_task_num(self) -> int:
        return len(self.combine_base_tasks)


def _decode_lengths(
    cu_seq_lengths_q: Any,
    cu_seq_lengths_kv: Any,
    seqused_q: Any,
    seqused_kv: Any,
) -> tuple[list[int], list[int], list[int], list[int], list[int], list[int]]:
    cu_q = _int_values(cu_seq_lengths_q, "cuSeqLengthsQ")
    cu_kv = _int_values(cu_seq_lengths_kv, "cuSeqLengthsKv")
    if len(cu_q) < 2 or len(cu_kv) != len(cu_q):
        raise ValueError("cuSeqLengthsQ/Kv must have the same batch+1 length")
    if cu_q[0] != 0 or cu_kv[0] != 0:
        raise ValueError("cuSeqLengthsQ/Kv must start at zero")
    storage_q = [cu_q[i + 1] - cu_q[i] for i in range(len(cu_q) - 1)]
    storage_kv = [cu_kv[i + 1] - cu_kv[i] for i in range(len(cu_kv) - 1)]
    if any(value < 0 for value in storage_q + storage_kv):
        raise ValueError("cuSeqLengthsQ/Kv must be nondecreasing")
    actual_q = storage_q if seqused_q is None else _int_values(seqused_q, "sequsedQ")
    actual_kv = storage_kv if seqused_kv is None else _int_values(seqused_kv, "sequsedKv")
    if len(actual_q) != len(storage_q) or len(actual_kv) != len(storage_kv):
        raise ValueError("sequsedQ/Kv must have one value per batch")
    if any(actual < 0 or actual > storage for actual, storage in zip(actual_q, storage_q)):
        raise ValueError("sequsedQ is outside Q storage bounds")
    if any(actual < 0 or actual > storage for actual, storage in zip(actual_kv, storage_kv)):
        raise ValueError("sequsedKv is outside KV storage bounds")
    return cu_q, cu_kv, storage_q, storage_kv, actual_q, actual_kv


def _valid_block_nums(
    sparse_block_count: torch.Tensor,
    *,
    actual_q: Sequence[int],
    storage_q: Sequence[int],
    sparse_head_num: int,
    block_shape_x: int,
) -> list[int]:
    count = torch.as_tensor(sparse_block_count).detach().cpu().to(torch.int64)
    if count.ndim != 2 or count.shape[0] != sparse_head_num:
        raise ValueError(
            "sparseBlockCount must be [sparseHeadNum, qBlockStorageNum]"
        )
    q_block_starts = [0]
    for storage_len in storage_q:
        q_block_starts.append(
            q_block_starts[-1] + _ceil_div(int(storage_len), block_shape_x)
        )
    if q_block_starts[-1] != count.shape[1]:
        raise ValueError(
            "sparseBlockCount Q-block dimension does not match cuSeqLengthsQ storage"
        )
    result: list[int] = []
    for batch_idx, q_len in enumerate(actual_q):
        for q_token in range(int(q_len)):
            q_block = q_block_starts[batch_idx] + q_token // block_shape_x
            for sparse_head in range(sparse_head_num):
                result.append(int(count[sparse_head, q_block].item()))
    return result


def _runtime_reason(
    metadata_fd_enabled: bool,
    *,
    metadata_reason: str,
    softmax_precision: int,
    return_softmax_lse: int,
) -> tuple[bool, str]:
    if not metadata_fd_enabled:
        return False, metadata_reason
    if softmax_precision != 1:
        return False, "Arch35 requires softmaxPrecision=1"
    if return_softmax_lse != 0:
        return False, "returnSoftmaxLse=1 keeps the main kernel on normal fallback"
    return True, "metadata v8 FD schedule enabled"


def simulate_aicpu_metadata_v8(
    *,
    sparse_block_idx: torch.Tensor,
    sparse_block_count: torch.Tensor,
    cu_seq_lengths_q: Any,
    cu_seq_lengths_kv: Any,
    seqused_q: Any = None,
    seqused_kv: Any = None,
    max_q_seq_len: Optional[int] = None,
    max_kv_seq_len: Optional[int] = None,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_shape: Sequence[int],
    is_packed_gqa: int = 1,
    layout_q: str = "TND",
    layout_kv: str = "PAGED_BBND",
    softmax_precision: int = 1,
    return_softmax_lse: int = 0,
    physical_aic_num: int = ARCH35_AIC_NUM,
) -> tuple[torch.Tensor, FdPlan]:
    """Build, serialize and validate one Metadata-v8 schedule."""

    sparse_idx = torch.as_tensor(sparse_block_idx).detach().cpu()
    sparse_count = torch.as_tensor(sparse_block_count).detach().cpu()
    if sparse_idx.ndim != 3 or sparse_count.shape != sparse_idx.shape[:2]:
        raise ValueError(
            "sparseBlockIdx must be rank 3 and sparseBlockCount must match its first two dimensions"
        )
    if len(block_shape) != 2:
        raise ValueError("blockShape must contain [blockShapeX, blockShapeY]")
    block_shape_x, block_shape_y = (int(block_shape[0]), int(block_shape[1]))
    top_k = int(sparse_idx.shape[2])
    q_block_storage_num = int(sparse_idx.shape[1])
    if physical_aic_num <= 0 or physical_aic_num > MAX_AIC_CORE_NUM:
        raise ValueError(
            f"physicalAicNum must be in [1, {MAX_AIC_CORE_NUM}], got {physical_aic_num}"
        )
    if str(layout_q).upper() != "TND" or str(layout_kv).upper() != "PAGED_BBND":
        raise ValueError("ATK Metadata-v8 simulator supports TND + PAGED_BBND only")
    if num_q_heads <= 0 or num_kv_heads <= 0 or num_q_heads % num_kv_heads != 0:
        raise ValueError("numQHeads must be positive and divisible by numKvHeads")
    if is_packed_gqa not in (0, 1):
        raise ValueError("isPackedGQA must be 0 or 1")
    if block_shape_x != 1 or block_shape_y <= 0 or top_k <= 0:
        raise ValueError("invalid blockShape or sparse block capacity")

    _, _, storage_q, storage_kv, actual_q, actual_kv = _decode_lengths(
        cu_seq_lengths_q,
        cu_seq_lengths_kv,
        seqused_q,
        seqused_kv,
    )
    del storage_kv, actual_kv, max_kv_seq_len  # Not consumed by the C++ FD scheduler.
    batch = len(actual_q)
    sparse_head_num = num_kv_heads if is_packed_gqa == 1 else num_q_heads
    group_size = num_q_heads // num_kv_heads
    valid_block_nums = _valid_block_nums(
        sparse_count,
        actual_q=actual_q,
        storage_q=storage_q,
        sparse_head_num=sparse_head_num,
        block_shape_x=block_shape_x,
    )
    if any(value < 0 or value > top_k for value in valid_block_nums):
        raise ValueError("sparseBlockCount is outside [0, topK]")

    total_q_tokens = sum(actual_q)
    total_base_tasks = total_q_tokens * sparse_head_num
    if len(valid_block_nums) != total_base_tasks:
        raise RuntimeError("valid block count list does not match base-task count")
    normal_used_core_num = min(total_base_tasks, physical_aic_num)
    block_prefix = _prefix_sum(valid_block_nums)
    total_split_tasks = block_prefix[-1]
    max_q_seq_len = (
        int(max_q_seq_len)
        if max_q_seq_len is not None
        else q_block_storage_num * block_shape_x
    )
    config_signature = calculate_config_signature(
        batch=batch,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_shape_x=block_shape_x,
        block_shape_y=block_shape_y,
        block_index_stride=top_k,
        q_block_storage_num=q_block_storage_num,
        is_packed_gqa=is_packed_gqa,
    )

    m16 = _ceil_div(group_size if is_packed_gqa == 1 else 1, 16)
    total_cost = sum(
        FD_COST_M16 * m16 + FD_COST_N * count + FD_COST_M16_N * m16 * count
        for count in valid_block_nums
        if count > 0
    )
    fd_shape_supported = (
        head_dim == 128
        and block_shape_y == 128
        and 12 <= top_k <= MAX_SPARSE_BLOCK_CAPACITY
        and 0 < group_size <= 128
        and max_q_seq_len >= total_q_tokens
        and total_base_tasks > 0
        and total_base_tasks * 10 < physical_aic_num * 3
    )
    metadata_fd_enabled = (
        fd_shape_supported
        and total_split_tasks > normal_used_core_num
        and total_cost > 0
        and total_split_tasks > 0
    )
    metadata_reason = "metadata FD gate is not satisfied"
    gate_checks = (
        (head_dim == 128, "headDim is not 128"),
        (block_shape_y == 128, "blockShapeY is not 128"),
        (top_k >= 12, "sparse block capacity is smaller than 12"),
        (
            top_k <= MAX_SPARSE_BLOCK_CAPACITY,
            f"sparse block capacity exceeds {MAX_SPARSE_BLOCK_CAPACITY}",
        ),
        (0 < group_size <= 128, "groupSize is outside [1, 128]"),
        (
            max_q_seq_len >= total_q_tokens,
            "maxQSeqLen is smaller than totalQTokenNum",
        ),
        (total_base_tasks > 0, "total base-task count is zero"),
        (
            total_base_tasks * 10 < physical_aic_num * 3,
            "base-task count does not satisfy totalTaskNum*10 < aicNum*3",
        ),
        (
            total_split_tasks > normal_used_core_num,
            "actual split-task count does not exceed normal used-core count",
        ),
        (
            total_cost > 0 and total_split_tasks > 0,
            "no valid KV block contributes FD cost",
        ),
    )
    for passed, failed_reason in gate_checks:
        if not passed:
            metadata_reason = failed_reason
            break
    if metadata_fd_enabled:
        metadata_reason = "metadata v8 FD schedule enabled"

    best_core_num = 0
    per_core_task_num = 0
    active_core_num = 0
    partial_task_num = 0
    combine_base_tasks: list[int] = []
    partial_start_by_base = [0] * total_base_tasks
    partial_count_by_base = [0] * total_base_tasks
    core_ranges: list[FdCoreRange] = []

    if metadata_fd_enabled:
        max_core = min(
            physical_aic_num,
            MAX_FD_ACTIVE_CORE_NUM,
            total_split_tasks,
        )
        ratio = total_cost / FD_LAUNCH_COST
        best_core_num = max(
            1,
            min(
                math.ceil((math.sqrt(1.0 + 4.0 * ratio) - 1.0) / 2.0),
                max_core,
            ),
        )
        per_core_task_num = _ceil_div(total_split_tasks, best_core_num)
        active_core_num = _ceil_div(total_split_tasks, per_core_task_num)

        for base_task in range(total_base_tasks):
            task_start = block_prefix[base_task]
            task_end = block_prefix[base_task + 1]
            if task_start == task_end:
                continue
            first_core = task_start // per_core_task_num
            last_core = (task_end - 1) // per_core_task_num
            partial_count = last_core - first_core + 1
            if partial_count <= 1:
                continue
            if (
                partial_count > active_core_num
                or partial_count > MAX_FD_PARTIAL_PER_BASE_TASK
                or first_core + partial_count > active_core_num
                or len(combine_base_tasks) >= MAX_COMBINE_TASK_NUM
            ):
                raise ValueError("FD combine schedule exceeds Metadata-v8 capacity")
            combine_base_tasks.append(base_task)
            partial_start_by_base[base_task] = partial_task_num
            partial_count_by_base[base_task] = partial_count
            partial_task_num += partial_count

        for core_idx in range(active_core_num):
            flat_start = min(core_idx * per_core_task_num, total_split_tasks)
            flat_end = min(flat_start + per_core_task_num, total_split_tasks)
            if flat_start == flat_end:
                raise RuntimeError("active FD core received an empty flat range")
            first_task = bisect.bisect_right(block_prefix, flat_start) - 1
            last_task = bisect.bisect_right(block_prefix, flat_end - 1) - 1
            task_slices: list[FdTaskSlice] = []
            for base_task in range(first_task, last_task + 1):
                task_start = block_prefix[base_task]
                task_end = block_prefix[base_task + 1]
                raw_begin = max(flat_start - task_start, 0)
                raw_end = min(flat_end - task_start, task_end - task_start)
                partial_id: Optional[int] = None
                if partial_count_by_base[base_task] > 1:
                    first_core = task_start // per_core_task_num
                    partial_id = (
                        partial_start_by_base[base_task] + core_idx - first_core
                    )
                task_slices.append(
                    FdTaskSlice(base_task, raw_begin, raw_end, partial_id)
                )
            core_ranges.append(
                FdCoreRange(
                    core_idx=core_idx,
                    flat_start=flat_start,
                    flat_end=flat_end,
                    base_task_start=first_task,
                    base_task_end=last_task + 1,
                    first_block_start=flat_start - block_prefix[first_task],
                    last_block_end=flat_end - block_prefix[last_task],
                    task_slices=tuple(task_slices),
                )
            )

    enabled, reason = _runtime_reason(
        metadata_fd_enabled,
        metadata_reason=metadata_reason,
        softmax_precision=int(softmax_precision),
        return_softmax_lse=int(return_softmax_lse),
    )
    flags = FD_ACTUAL_BLOCK_PREFIX
    if metadata_fd_enabled:
        flags |= FD_SCHEDULE_ENABLED
    plan = FdPlan(
        enabled=enabled,
        metadata_fd_enabled=metadata_fd_enabled,
        reason=reason,
        physical_aic_num=physical_aic_num,
        normal_used_core_num=normal_used_core_num,
        total_q_tokens=total_q_tokens,
        total_base_tasks=total_base_tasks,
        total_split_tasks=total_split_tasks,
        top_k=top_k,
        group_size=group_size,
        total_cost=total_cost,
        best_core_num=best_core_num,
        per_core_task_num=per_core_task_num,
        active_core_num=active_core_num,
        fd_lse_sub_stride=_round_up(_ceil_div(group_size, 2), FLOATS_PER_BLOCK),
        partial_task_num=partial_task_num,
        combine_base_tasks=tuple(combine_base_tasks),
        partial_start_by_base=tuple(partial_start_by_base),
        partial_count_by_base=tuple(partial_count_by_base),
        block_prefix=tuple(block_prefix),
        core_ranges=tuple(core_ranges),
        fd_schedule_flags=flags,
        config_signature=config_signature,
    )
    metadata = serialize_plan(plan)
    validate_metadata(metadata, plan)
    return metadata, plan


def serialize_plan(plan: FdPlan) -> torch.Tensor:
    metadata = torch.zeros(METADATA_TOTAL_SIZE, dtype=torch.int32)
    metadata[MAGIC_INDEX] = METADATA_MAGIC
    metadata[VERSION_INDEX] = METADATA_VERSION
    metadata[METADATA_USED_SIZE_INDEX] = METADATA_USED_SIZE
    metadata[SA_USED_CORE_NUM_INDEX] = plan.normal_used_core_num
    metadata[SA_TOTAL_TASK_NUM_INDEX] = plan.total_base_tasks
    metadata[FD_ACTIVE_CORE_NUM_INDEX] = plan.active_core_num
    metadata[DECODE_PER_CORE_TASK_NUM_INDEX] = plan.per_core_task_num
    metadata[COMBINE_TASK_NUM_INDEX] = plan.combine_task_num
    metadata[FD_SCHEDULE_FLAGS_INDEX] = plan.fd_schedule_flags
    metadata[FD_TOTAL_FLAT_TASK_NUM_INDEX] = plan.total_split_tasks
    metadata[FD_PARTIAL_TASK_NUM_INDEX] = plan.partial_task_num
    metadata[CONFIG_SIGNATURE_INDEX] = plan.config_signature

    for core in plan.core_ranges:
        offset = DECODE_SCHEDULE_OFFSET + core.core_idx * DECODE_SCHEDULE_FIELD_NUM
        metadata[offset:offset + DECODE_SCHEDULE_FIELD_NUM] = torch.tensor(
            (
                core.base_task_start,
                core.base_task_end,
                core.first_block_start,
                core.last_block_end,
            ),
            dtype=torch.int32,
        )
    for combine_idx, base_task in enumerate(plan.combine_base_tasks):
        offset = COMBINE_SCHEDULE_OFFSET + combine_idx * COMBINE_SCHEDULE_FIELD_NUM
        metadata[offset:offset + COMBINE_SCHEDULE_FIELD_NUM] = torch.tensor(
            (
                base_task,
                plan.block_prefix[base_task] // plan.per_core_task_num,
                plan.partial_start_by_base[base_task],
                plan.partial_count_by_base[base_task],
            ),
            dtype=torch.int32,
        )
    return metadata


def decode_metadata(metadata: torch.Tensor) -> dict[str, Any]:
    values = [
        int(value)
        for value in torch.as_tensor(metadata).detach().cpu().view(-1).tolist()
    ]
    if len(values) != METADATA_TOTAL_SIZE:
        raise ValueError(
            f"metadata must contain {METADATA_TOTAL_SIZE} INT32 values, got {len(values)}"
        )
    header = {
        "magic": values[MAGIC_INDEX],
        "version": values[VERSION_INDEX],
        "used_size": values[METADATA_USED_SIZE_INDEX],
        "sa_used_core_num": values[SA_USED_CORE_NUM_INDEX],
        "sa_total_task_num": values[SA_TOTAL_TASK_NUM_INDEX],
        "fd_active_core_num": values[FD_ACTIVE_CORE_NUM_INDEX],
        "decode_per_core_task_num": values[DECODE_PER_CORE_TASK_NUM_INDEX],
        "combine_task_num": values[COMBINE_TASK_NUM_INDEX],
        "fd_schedule_flags": values[FD_SCHEDULE_FLAGS_INDEX],
        "fd_total_flat_task_num": values[FD_TOTAL_FLAT_TASK_NUM_INDEX],
        "fd_partial_task_num": values[FD_PARTIAL_TASK_NUM_INDEX],
        "config_signature": values[CONFIG_SIGNATURE_INDEX],
    }
    decode_schedules = []
    for core_idx in range(max(0, header["fd_active_core_num"])):
        offset = DECODE_SCHEDULE_OFFSET + core_idx * DECODE_SCHEDULE_FIELD_NUM
        decode_schedules.append(tuple(values[offset:offset + DECODE_SCHEDULE_FIELD_NUM]))
    combine_schedules = []
    for combine_idx in range(max(0, header["combine_task_num"])):
        offset = COMBINE_SCHEDULE_OFFSET + combine_idx * COMBINE_SCHEDULE_FIELD_NUM
        combine_schedules.append(
            tuple(values[offset:offset + COMBINE_SCHEDULE_FIELD_NUM])
        )
    return {
        **header,
        "decode_schedules": decode_schedules,
        "combine_schedules": combine_schedules,
        "all_values": values,
    }


def expected_metadata(plan: FdPlan) -> dict[str, Any]:
    return {
        "magic": METADATA_MAGIC,
        "version": METADATA_VERSION,
        "used_size": METADATA_USED_SIZE,
        "sa_used_core_num": plan.normal_used_core_num,
        "sa_total_task_num": plan.total_base_tasks,
        "fd_active_core_num": plan.active_core_num,
        "decode_per_core_task_num": plan.per_core_task_num,
        "combine_task_num": plan.combine_task_num,
        "fd_schedule_flags": plan.fd_schedule_flags,
        "fd_total_flat_task_num": plan.total_split_tasks,
        "fd_partial_task_num": plan.partial_task_num,
        "config_signature": plan.config_signature,
        "decode_schedules": [
            (
                core.base_task_start,
                core.base_task_end,
                core.first_block_start,
                core.last_block_end,
            )
            for core in plan.core_ranges
        ],
        "combine_schedules": [
            (
                base_task,
                plan.block_prefix[base_task] // plan.per_core_task_num,
                plan.partial_start_by_base[base_task],
                plan.partial_count_by_base[base_task],
            )
            for base_task in plan.combine_base_tasks
        ],
    }


def validate_metadata(metadata: torch.Tensor, plan: FdPlan) -> dict[str, Any]:
    decoded = decode_metadata(metadata)
    expected = expected_metadata(plan)
    for name, expected_value in expected.items():
        if decoded[name] != expected_value:
            raise ValueError(
                f"metadata {name}={decoded[name]!r} does not match plan {expected_value!r}"
            )

    values = decoded["all_values"]
    if any(values[12:HEADER_SIZE]):
        raise ValueError("reserved Metadata-v8 header fields must be zero")
    decode_used_end = (
        DECODE_SCHEDULE_OFFSET
        + plan.active_core_num * DECODE_SCHEDULE_FIELD_NUM
    )
    if any(values[decode_used_end:COMBINE_SCHEDULE_OFFSET]):
        raise ValueError("inactive Decode Schedule entries must be zero")
    combine_used_end = (
        COMBINE_SCHEDULE_OFFSET
        + plan.combine_task_num * COMBINE_SCHEDULE_FIELD_NUM
    )
    if any(values[combine_used_end:METADATA_TOTAL_SIZE]):
        raise ValueError("inactive Combine Schedule entries and metadata tail must be zero")

    if (plan.fd_schedule_flags & FD_ACTUAL_BLOCK_PREFIX) == 0:
        raise ValueError("FD_ACTUAL_BLOCK_PREFIX must always be set")
    if plan.metadata_fd_enabled:
        if not (
            plan.active_core_num > 0
            and plan.per_core_task_num > 0
            and plan.total_split_tasks > 0
            and plan.combine_task_num > 0
            and plan.partial_task_num > 0
        ):
            raise ValueError("enabled FD plan has an incomplete Decode/Combine schedule")
    elif any((plan.active_core_num, plan.combine_task_num, plan.partial_task_num)):
        raise ValueError("normal plan must not contain FD Decode/Combine tasks")

    scheduled_partial_num = 0
    for combine_idx, base_task in enumerate(plan.combine_base_tasks):
        partial_start = plan.partial_start_by_base[base_task]
        partial_count = plan.partial_count_by_base[base_task]
        if partial_start != scheduled_partial_num or partial_count <= 1:
            raise ValueError(f"invalid partial range for combine task {combine_idx}")
        scheduled_partial_num += partial_count
    if scheduled_partial_num != plan.partial_task_num:
        raise ValueError("Combine Schedule partial ranges are not contiguous")
    return decoded


def print_fd_plan(plan: FdPlan, *, show_cores: bool = True) -> None:
    metadata_mode = "fd" if plan.metadata_fd_enabled else "normal"
    print(
        f"[ATK][FD] metadataVersion={plan.metadata_version}, "
        f"pipelineMode={plan.pipeline_mode}, metadataSchedule={metadata_mode}"
    )
    print(
        f"[ATK][FD] reason={plan.reason}, physicalAIC={plan.physical_aic_num}, "
        f"normalUsedAIC={plan.normal_used_core_num}, activeAIC={plan.active_core_num}, "
        f"baseTasks={plan.total_base_tasks}, actualBlocks={plan.total_split_tasks}, "
        f"partialTasks={plan.partial_task_num}, combineTasks={plan.combine_task_num}"
    )
    if show_cores and plan.metadata_fd_enabled:
        for core in plan.core_ranges:
            slices = ", ".join(
                f"base={item.base_task}:raw[{item.raw_begin},{item.raw_end})"
                + (
                    f"->partial={item.partial_task_id}"
                    if item.partial_task_id is not None
                    else "->direct"
                )
                for item in core.task_slices
            )
            print(
                f"[ATK][FD][AIC {core.core_idx:02d}] "
                f"flat=[{core.flat_start},{core.flat_end}) {slices}"
            )
