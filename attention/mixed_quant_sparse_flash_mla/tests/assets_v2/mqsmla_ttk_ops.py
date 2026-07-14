#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Thin TTK wrapper for MixedQuantSparseFlashMla with precomputed metadata."""

from typing import Optional

import torch

import cann_ops_transformer  # noqa: F401 - registers torch.ops namespace


def mixed_quant_sparse_flash_mla_ttk(
    q: torch.Tensor,
    *,
    ori_kv: Optional[torch.Tensor] = None,
    cmp_kv: Optional[torch.Tensor] = None,
    ori_sparse_indices: Optional[torch.Tensor] = None,
    cmp_sparse_indices: Optional[torch.Tensor] = None,
    ori_block_table: Optional[torch.Tensor] = None,
    cmp_block_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_ori_kv: Optional[torch.Tensor] = None,
    cu_seqlens_cmp_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_ori_kv: Optional[torch.Tensor] = None,
    seqused_cmp_kv: Optional[torch.Tensor] = None,
    cmp_residual_kv: Optional[torch.Tensor] = None,
    ori_topk_length: Optional[torch.Tensor] = None,
    cmp_topk_length: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    quant_mode: int = 1,
    rope_head_dim: int = 64,
    softmax_scale: float = 1.0,
    cmp_ratio: int = 1,
    ori_mask_mode: int = 4,
    cmp_mask_mode: int = 3,
    ori_win_left: int = 127,
    ori_win_right: int = 0,
    layout_q: str = "BSND",
    layout_kv: str = "BSND",
    topk_value_mode: int = 1,
    return_softmax_lse: bool = False,
    metadata_cmp_topk: Optional[int] = None,
    key_dtype: Optional[int] = None,
    value_dtype: Optional[int] = None,
):
    del metadata_cmp_topk
    return torch.ops.cann_ops_transformer.mixed_quant_sparse_flash_mla(
        q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        ori_sparse_indices=ori_sparse_indices,
        cmp_sparse_indices=cmp_sparse_indices,
        ori_block_table=ori_block_table,
        cmp_block_table=cmp_block_table,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_ori_kv=cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
        seqused_q=seqused_q,
        seqused_ori_kv=seqused_ori_kv,
        seqused_cmp_kv=seqused_cmp_kv,
        cmp_residual_kv=cmp_residual_kv,
        ori_topk_length=ori_topk_length,
        cmp_topk_length=cmp_topk_length,
        sinks=sinks,
        metadata=metadata,
        quant_mode=quant_mode,
        rope_head_dim=rope_head_dim,
        softmax_scale=softmax_scale,
        cmp_ratio=cmp_ratio,
        ori_mask_mode=ori_mask_mode,
        cmp_mask_mode=cmp_mask_mode,
        ori_win_left=ori_win_left,
        ori_win_right=ori_win_right,
        layout_q=layout_q,
        layout_kv=layout_kv,
        topk_value_mode=topk_value_mode,
        return_softmax_lse=return_softmax_lse,
        key_dtype=key_dtype,
        value_dtype=value_dtype,
    )
