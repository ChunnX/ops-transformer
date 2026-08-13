# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Stable Python facade for CatCCOS UDMA operators."""

from typing import Optional, Tuple, Union

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
    output_all_to_allv: bool = False,
    all_to_allv_out: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Dispatch tokens with UDMA and apply the destination rank's grouped matmuls.

    ``x`` is ordered by global expert according to ``local_tokens_per_expert``.
    Both route tensors returned by :func:`build_rank_route_tensors` are flat and
    have shape ``[expert_num]``. The global table is laid out by source rank and
    then local expert.
    It is a regular contiguous BF16 NPU tensor; UDMA reads it directly as the
    local source of remote PUT operations, without symmetric-input staging.
    The output has padded capacity ``[world_size * m, n]``; valid rows are packed
    at the beginning and their count is the sum of the global route table.

    When ``output_all_to_allv=True``, the call returns ``(output,
    all_to_allv_output)``. The second tensor is the exact BF16 activation consumed
    by grouped matmul, ordered by local expert first and source rank second, with
    shape ``[sum(global_tokens_per_local_expert), k]``. The default path selects
    the original kernel variant and performs no export copy.
    """
    config = get_udma_config()
    if ep_size != config.world_size:
        raise ValueError("TP=1 requires ep_size == UDMA world_size")
    global_tokens_per_local_expert = global_tokens_per_local_expert.view(-1)
    arguments = (
        x,
        weight,
        local_tokens_per_expert,
        global_tokens_per_local_expert,
        ep_size,
        expert_num,
    )
    if all_to_allv_out is not None and not output_all_to_allv:
        raise ValueError("all_to_allv_out requires output_all_to_allv=True")
    if not output_all_to_allv:
        if out is None:
            return torch.ops.ops_transformer_catccos_udma.all_to_allv_grouped_mat_mul_udma(
                *arguments
            )
        return torch.ops.ops_transformer_catccos_udma.all_to_allv_grouped_mat_mul_udma_out(
            *arguments, out
        )

    received_rows = int(global_tokens_per_local_expert.sum().item())
    if all_to_allv_out is None:
        all_to_allv_out = torch.empty(
            (received_rows, x.shape[1]), dtype=x.dtype, device=x.device
        )
    elif tuple(all_to_allv_out.shape) != (received_rows, x.shape[1]):
        raise ValueError(
            "all_to_allv_out must have shape "
            "[sum(global_tokens_per_local_expert), x.shape[1]]"
        )
    if out is None:
        return torch.ops.ops_transformer_catccos_udma.all_to_allv_grouped_mat_mul_udma_export(
            *arguments, all_to_allv_out
        )
    return torch.ops.ops_transformer_catccos_udma.all_to_allv_grouped_mat_mul_udma_out_export(
        *arguments, out, all_to_allv_out
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
    experts and has ``sum(global_tokens_per_local_expert)`` rows. The result has
    shape ``[sum(local_tokens_per_expert), n]`` and resides in regular NPU memory.
    Both route tensors returned by :func:`build_rank_route_tensors` are flat and
    have shape ``[expert_num]``. The global table is laid out by source rank and
    then local expert.
    UDMA GET writes directly to that local allocation, so a reusable ``out`` can
    be created with :func:`torch.empty` and no symmetric-output staging or D2D
    copy is required.
    """
    config = get_udma_config()
    if ep_size != config.world_size:
        raise ValueError("TP=1 requires ep_size == UDMA world_size")
    global_tokens_per_local_expert = global_tokens_per_local_expert.view(-1)
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
