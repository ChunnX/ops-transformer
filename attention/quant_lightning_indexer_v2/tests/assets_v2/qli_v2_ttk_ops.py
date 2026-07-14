#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Thin TTK wrapper for QuantLightningIndexer with precomputed metadata."""

from typing import Optional

import torch

import cann_ops_transformer  # noqa: F401 - registers torch.ops namespace


def quant_lightning_indexer_v2(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_dequant_scale: torch.Tensor,
    key_dequant_scale: torch.Tensor,
    *,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    cmp_residual_k: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    output_idx_offset: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    sparse_count: int = 0,
    quant_mode: int = 1,
    max_seqlen_q: int = -1,
    layout_query: str = "BSND",
    layout_key: str = "BSND",
    sparse_mode: int = 0,
    cmp_ratio: int = 1,
    return_value: int = 0,
):
    return torch.ops.cann_ops_transformer.quant_lightning_indexer(
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        sparse_count,
        quant_mode,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k,
        block_table=block_table,
        output_idx_offset=output_idx_offset,
        metadata=metadata,
        max_seqlen_q=max_seqlen_q,
        layout_q=layout_query,
        layout_k=layout_key,
        mask_mode=sparse_mode,
        cmp_ratio=cmp_ratio,
        return_value=return_value,
    )
