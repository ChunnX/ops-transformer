#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""TTK metadata-first adapter for the installed SparseAttnSharedkv API."""

from typing import Optional

import torch

try:
    import custom_ops  # noqa: F401  (registers torch.ops.custom.npu_sparse_attn_sharedkv)
except ImportError:
    pass


class SparseAttnSharedkvMetadataBuilder:
    """Derive scalar metadata parameters from operator inputs."""

    @staticmethod
    def max_seq(cu_seqlens, seqused, fallback=0):
        if seqused is not None and seqused.numel() > 0:
            return int(seqused.detach().cpu().max().item())
        if cu_seqlens is not None and cu_seqlens.numel() > 1:
            values = cu_seqlens.detach().cpu()
            return int((values[1:] - values[:-1]).max().item())
        return int(fallback or 0)

    @staticmethod
    def max_prefix_span(cu_seqlens, fallback=0):
        if cu_seqlens is not None and cu_seqlens.numel() > 1:
            values = cu_seqlens.detach().cpu()
            return int((values[1:] - values[:-1]).max().item())
        return int(fallback or 0)

    @staticmethod
    def infer_batch(cu_seqlens, seqused):
        if seqused is not None:
            return int(seqused.numel())
        if cu_seqlens is not None:
            return max(int(cu_seqlens.numel()) - 1, 0)
        return 0


def build_sparse_attn_sharedkv_metadata(
    q: torch.Tensor,
    *,
    ori_kv: Optional[torch.Tensor] = None,
    cmp_kv: Optional[torch.Tensor] = None,
    cmp_sparse_indices: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_ori_kv: Optional[torch.Tensor] = None,
    cu_seqlens_cmp_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    cmp_ratio: Optional[int] = None,
    ori_mask_mode: int = 4,
    cmp_mask_mode: int = 3,
    ori_win_left: int = 127,
    ori_win_right: int = 0,
    layout_q: str = "BSND",
    layout_kv: str = "BSND",
    has_ori_kv: Optional[bool] = None,
    has_cmp_kv: Optional[bool] = None,
    **extra_kwargs,
):
    """Build metadata before the main op or graph capture."""
    del extra_kwargs
    num_heads_q = int(q.shape[1] if layout_q == "TND" else q.shape[2])
    kv_ref = ori_kv if ori_kv is not None else cmp_kv
    num_heads_kv = int(kv_ref.shape[1] if layout_kv == "TND" else kv_ref.shape[2])
    head_dim = int(q.shape[-1])
    batch_size = (
        int(q.shape[0])
        if layout_q == "BSND"
        else SparseAttnSharedkvMetadataBuilder.infer_batch(cu_seqlens_q, seqused_q)
    )
    q_fallback = q.shape[1] if layout_q == "BSND" else q.shape[0]
    max_seqlen_q = (
        int(q_fallback)
        if layout_q == "BSND"
        else SparseAttnSharedkvMetadataBuilder.max_prefix_span(cu_seqlens_q, q_fallback)
    )
    ori_fallback = 0 if ori_kv is None else (ori_kv.shape[0] if layout_kv == "TND" else ori_kv.shape[1])
    max_seqlen_kv = SparseAttnSharedkvMetadataBuilder.max_seq(
        cu_seqlens_ori_kv, seqused_kv, ori_fallback
    )
    if has_ori_kv is None:
        has_ori_kv = ori_kv is not None
    if has_cmp_kv is None:
        has_cmp_kv = cmp_kv is not None

    metadata_kwargs = dict(
        num_heads_q=int(num_heads_q),
        num_heads_kv=int(num_heads_kv),
        head_dim=int(head_dim),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_ori_kv=cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
        seqused_q=seqused_q,
        seqused_kv=seqused_kv,
        batch_size=int(batch_size),
        max_seqlen_q=int(max_seqlen_q),
        max_seqlen_kv=int(max_seqlen_kv),
        ori_mask_mode=int(ori_mask_mode),
        cmp_mask_mode=int(cmp_mask_mode),
        ori_win_left=int(ori_win_left),
        ori_win_right=int(ori_win_right),
        layout_q=layout_q,
        layout_kv=layout_kv,
        has_ori_kv=bool(has_ori_kv),
        has_cmp_kv=bool(has_cmp_kv),
    )
    if cmp_sparse_indices is not None:
        metadata_kwargs["cmp_topk"] = int(cmp_sparse_indices.shape[-1])
    if cmp_ratio is not None:
        metadata_kwargs["cmp_ratio"] = int(cmp_ratio)
    metadata_kwargs["device"] = (
        str(q.device)
        if torch.is_tensor(q) and str(q.device).startswith("npu")
        else "npu:0"
    )
    metadata = torch.ops.custom.npu_sparse_attn_sharedkv_metadata(**metadata_kwargs)
    if (
        hasattr(metadata, "to")
        and torch.is_tensor(q)
        and str(q.device) != "meta"
        and metadata.device != q.device
    ):
        metadata = metadata.to(q.device)
    return metadata


def sparse_attn_sharedkv_ttk(
    q: torch.Tensor,
    *,
    ori_kv: Optional[torch.Tensor] = None,
    cmp_kv: Optional[torch.Tensor] = None,
    cmp_sparse_indices: Optional[torch.Tensor] = None,
    ori_block_table: Optional[torch.Tensor] = None,
    cmp_block_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_ori_kv: Optional[torch.Tensor] = None,
    cu_seqlens_cmp_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    softmax_scale: float = 1.0,
    cmp_ratio: Optional[int] = None,
    ori_mask_mode: int = 4,
    cmp_mask_mode: int = 3,
    ori_win_left: int = 127,
    ori_win_right: int = 0,
    layout_q: str = "BSND",
    layout_kv: str = "BSND",
    return_softmax_lse: bool = False,
    has_ori_kv: Optional[bool] = None,
    has_cmp_kv: Optional[bool] = None,
    **extra_kwargs,
):
    """Generate metadata in TTK, then call the extension op."""
    del extra_kwargs
    effective_cmp_ratio = 1 if cmp_ratio is None else int(cmp_ratio)
    metadata = build_sparse_attn_sharedkv_metadata(
        q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        cmp_sparse_indices=cmp_sparse_indices,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_ori_kv=cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
        seqused_q=seqused_q,
        seqused_kv=seqused_kv,
        cmp_ratio=cmp_ratio,
        ori_mask_mode=ori_mask_mode,
        cmp_mask_mode=cmp_mask_mode,
        ori_win_left=ori_win_left,
        ori_win_right=ori_win_right,
        layout_q=layout_q,
        layout_kv=layout_kv,
        has_ori_kv=has_ori_kv,
        has_cmp_kv=has_cmp_kv,
    )

    op_kwargs = dict(
        ori_kv=ori_kv,
        ori_sparse_indices=None,
        ori_block_table=ori_block_table,
        cu_seqlens_q=cu_seqlens_q if layout_q == "TND" else None,
        cu_seqlens_ori_kv=cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
        seqused_q=seqused_q,
        seqused_kv=seqused_kv,
        sinks=sinks,
        metadata=metadata,
        softmax_scale=float(softmax_scale),
        ori_mask_mode=int(ori_mask_mode),
        ori_win_left=int(ori_win_left),
        ori_win_right=int(ori_win_right),
        layout_q=layout_q,
        layout_kv=layout_kv,
        return_softmax_lse=bool(return_softmax_lse),
    )
    if cmp_ratio is not None:
        op_kwargs["cmp_ratio"] = effective_cmp_ratio
        op_kwargs["cmp_mask_mode"] = int(cmp_mask_mode)
    if cmp_kv is not None:
        op_kwargs["cmp_kv"] = cmp_kv
    if cmp_sparse_indices is not None:
        op_kwargs["cmp_sparse_indices"] = cmp_sparse_indices
    if cmp_block_table is not None:
        op_kwargs["cmp_block_table"] = cmp_block_table

    return torch.ops.custom.npu_sparse_attn_sharedkv(q, **op_kwargs)
