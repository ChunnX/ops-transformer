"""Deterministic Arch35 FD cases shared by the ATK generator and plugins."""

from __future__ import annotations


FD_CASE_SPECS = (
    {
        "name": "bf16_fd_topk12",
        "dtype": "bf16",
        "q_storage": 1,
        "q_actual": 1,
        "kv_heads": 1,
        "group_size": 16,
        "top_k": 12,
        "counts": (12,),
    },
    {
        "name": "bf16_fd_topk15",
        "dtype": "bf16",
        "q_storage": 1,
        "q_actual": 1,
        "kv_heads": 1,
        "group_size": 8,
        "top_k": 15,
        "counts": (15,),
    },
    {
        "name": "bf16_fd_topk16",
        "dtype": "bf16",
        "q_storage": 1,
        "q_actual": 1,
        "kv_heads": 1,
        "group_size": 32,
        "top_k": 16,
        "counts": (16,),
    },
    {
        "name": "fp16_fd_topk16",
        "dtype": "fp16",
        "q_storage": 1,
        "q_actual": 1,
        "kv_heads": 1,
        "group_size": 16,
        "top_k": 16,
        "counts": (16,),
    },
    {
        "name": "bf16_fd_topk256",
        "dtype": "bf16",
        "q_storage": 1,
        "q_actual": 1,
        "kv_heads": 1,
        "group_size": 8,
        "top_k": 256,
        "max_blocks": 256,
        "counts": (256,),
    },
    {
        "name": "bf16_fd_two_base_tasks",
        "dtype": "bf16",
        "q_storage": 2,
        "q_actual": 2,
        "kv_heads": 1,
        "group_size": 8,
        "top_k": 16,
        "counts": (16, 12),
    },
    {
        "name": "bf16_fd_eight_base_tasks",
        "dtype": "bf16",
        "q_storage": 8,
        "q_actual": 8,
        "kv_heads": 1,
        "group_size": 4,
        "top_k": 16,
        "counts": (16, 15, 14, 13, 12, 16, 15, 14),
    },
    {
        "name": "bf16_fd_empty_task",
        "dtype": "bf16",
        "q_storage": 2,
        "q_actual": 2,
        "kv_heads": 1,
        "group_size": 16,
        "top_k": 16,
        "counts": (16, 0),
    },
    {
        "name": "bf16_fd_seqused_q_padding",
        "dtype": "bf16",
        "q_storage": 4,
        "q_actual": 1,
        "kv_heads": 1,
        "group_size": 16,
        "top_k": 16,
        "counts": (16,),
    },
    {
        "name": "bf16_fd_noncontiguous_kv",
        "dtype": "bf16",
        "q_storage": 1,
        "q_actual": 1,
        "kv_heads": 1,
        "group_size": 8,
        "top_k": 16,
        "counts": (16,),
    },
)


def fd_case_spec_from_shapes(
    *,
    batch: int,
    max_blocks: int,
    num_physical: int,
):
    """Recover the deterministic spec from generator-only physical headroom."""

    if batch != 1:
        raise ValueError("deterministic FD ATK cases currently require batch=1")
    marker = int(num_physical) - int(batch) * int(max_blocks)
    if marker < 1 or marker > len(FD_CASE_SPECS):
        raise ValueError(
            f"cannot recover deterministic FD case marker from physical headroom {marker}"
        )
    return FD_CASE_SPECS[marker - 1]
