# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Stable Python facade for CatCCOS UDMA operators."""

from typing import Optional

import torch

from .runtime import get_udma_config


def all_to_allv_grouped_mat_mul(
    x: torch.Tensor,
    weight: torch.Tensor,
    local_tokens_per_expert: torch.Tensor,
    global_tokens_per_local_expert: torch.Tensor,
    *,
    ep_size: int,
    expert_num: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dispatch tokens with UDMA and apply the destination rank's grouped matmuls.

    ``x`` is ordered by global expert according to ``local_tokens_per_expert``.
    It is a regular contiguous BF16 NPU tensor; UDMA reads it directly as the
    local source of remote PUT operations, without symmetric-input staging.
    The output has padded capacity ``[world_size * m, n]``; valid rows are packed
    at the beginning and their count is the sum of the global route table.
    """
    config = get_udma_config()
    if ep_size != config.world_size:
        raise ValueError("TP=1 requires ep_size == UDMA world_size")
    arguments = (
        x,
        weight,
        local_tokens_per_expert,
        global_tokens_per_local_expert,
        ep_size,
        expert_num,
    )
    if out is None:
        return torch.ops.ops_transformer_catccos_udma.all_to_allv_grouped_mat_mul_udma(
            *arguments
        )
    return torch.ops.ops_transformer_catccos_udma.all_to_allv_grouped_mat_mul_udma_out(
        *arguments, out
    )


def grouped_mat_mul_all_to_allv(
    x: torch.Tensor,
    weight: torch.Tensor,
    local_tokens_per_expert: torch.Tensor,
    global_tokens_per_local_expert: torch.Tensor,
    *,
    ep_size: int,
    expert_num: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply local grouped matmuls and return tokens with UDMA AllToAllV.

    ``x`` is a regular contiguous BF16 NPU tensor grouped by this rank's local
    experts. The result has shape ``[m, n]`` and resides in regular NPU memory.
    UDMA GET writes directly to that local allocation, so a reusable ``out`` can
    be created with :func:`torch.empty` and no symmetric-output staging or D2D
    copy is required.
    """
    config = get_udma_config()
    if ep_size != config.world_size:
        raise ValueError("TP=1 requires ep_size == UDMA world_size")
    arguments = (
        x,
        weight,
        local_tokens_per_expert,
        global_tokens_per_local_expert,
        ep_size,
        expert_num,
    )
    if out is None:
        return torch.ops.ops_transformer_catccos_udma.grouped_mat_mul_all_to_allv_udma(
            *arguments
        )
    return torch.ops.ops_transformer_catccos_udma.grouped_mat_mul_all_to_allv_udma_out(
        *arguments, out
    )
