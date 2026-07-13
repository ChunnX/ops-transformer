#!/usr/bin/python
# -*- coding: utf-8 -*-
"""TTK metadata-first adapter for the installed QuantLightningIndexer API."""

from typing import Optional

import torch

import cann_ops_transformer  # noqa: F401 - registers torch.ops namespace


def _max_seq(cu_seqlens, seqused, fallback):
    if seqused is not None and seqused.numel() > 0:
        return int(seqused.detach().cpu().max().item())
    if cu_seqlens is not None and cu_seqlens.numel() > 1:
        diff = cu_seqlens.detach().cpu()[1:] - cu_seqlens.detach().cpu()[:-1]
        return int(diff.max().item())
    return int(fallback)


def _batch_size(query, layout_query, cu_seqlens_q, seqused_q):
    if layout_query == "BSND":
        return int(query.shape[0])
    if seqused_q is not None:
        return int(seqused_q.numel())
    if cu_seqlens_q is not None:
        return max(int(cu_seqlens_q.numel()) - 1, 0)
    return 0


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
    sparse_count: int = 0,
    quant_mode: int = 1,
    max_seqlen_q: int = -1,
    layout_query: str = "BSND",
    layout_key: str = "BSND",
    sparse_mode: int = 0,
    cmp_ratio: int = 1,
    return_value: int = 0,
):
    q_seq = int(query.shape[1]) if layout_query == "BSND" else int(query.shape[0])
    k_seq = int(key.shape[1]) if layout_key == "BSND" else int(key.shape[0])
    q_head_num = int(query.shape[2]) if layout_query == "BSND" else int(query.shape[1])
    k_head_num = int(key.shape[1]) if layout_key == "TND" else int(key.shape[2])
    head_dim = int(query.shape[-1])
    max_q = _max_seq(cu_seqlens_q, seqused_q, q_seq)
    max_k = _max_seq(cu_seqlens_k, seqused_k, k_seq)
    meta_batch = _batch_size(query, layout_query, cu_seqlens_q, seqused_q)
    metadata = torch.ops.cann_ops_transformer.quant_lightning_indexer_metadata(
        num_heads_q=int(q_head_num),
        num_heads_k=int(k_head_num),
        head_dim=int(head_dim),
        topk=int(sparse_count),
        quant_mode=int(quant_mode),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k,
        batch_size=meta_batch,
        max_seqlen_q=int(max_q),
        max_seqlen_k=int(max_k),
        layout_q=layout_query,
        layout_k=layout_key,
        mask_mode=int(sparse_mode),
        cmp_ratio=int(cmp_ratio),
    )
    if hasattr(metadata, "to") and hasattr(query, "device") and metadata.device != query.device:
        metadata = metadata.to(query.device)
    return torch.ops.cann_ops_transformer.quant_lightning_indexer(
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        int(sparse_count),
        int(quant_mode),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k,
        block_table=block_table,
        output_idx_offset=output_idx_offset,
        metadata=metadata,
        max_seqlen_q=int(max_seqlen_q),
        layout_q=layout_query,
        layout_k=layout_key,
        mask_mode=int(sparse_mode),
        cmp_ratio=int(cmp_ratio),
        return_value=int(return_value),
    )
