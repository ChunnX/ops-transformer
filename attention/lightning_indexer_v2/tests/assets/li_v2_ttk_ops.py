#!/usr/bin/python
# -*- coding: utf-8 -*-
"""TTK metadata-first adapter for the installed LightningIndexer API."""

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


def _batch_size(q, layout_q, cu_seqlens_q, seqused_q):
    if layout_q == "BSND":
        return int(q.shape[0])
    if seqused_q is not None:
        return int(seqused_q.numel())
    if cu_seqlens_q is not None:
        return max(int(cu_seqlens_q.numel()) - 1, 0)
    return 0


def lightning_indexer_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    *,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    cmp_residual_k: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    output_idx_offset: Optional[torch.Tensor] = None,
    topk: int = 0,
    max_seqlen_q: int = -1,
    layout_q: str = "BSND",
    layout_k: str = "BSND",
    mask_mode: int = 0,
    cmp_ratio: int = 1,
    return_value: int = 0,
):
    """Compute metadata first, then call cann_ops_transformer.lightning_indexer."""
    q_seq = int(q.shape[1]) if layout_q == "BSND" else int(q.shape[0])
    k_seq = int(k.shape[1]) if layout_k == "BSND" else int(k.shape[0])
    q_head_num = int(q.shape[2]) if layout_q == "BSND" else int(q.shape[1])
    k_head_num = int(k.shape[2]) if layout_k != "TND" else int(k.shape[1])
    head_dim = int(q.shape[-1])
    max_q = _max_seq(cu_seqlens_q, seqused_q, q_seq)
    max_k = _max_seq(cu_seqlens_k, seqused_k, k_seq)
    meta_batch = _batch_size(q, layout_q, cu_seqlens_q, seqused_q)
    metadata = torch.ops.cann_ops_transformer.lightning_indexer_metadata(
        num_heads_q=int(q_head_num),
        num_heads_k=int(k_head_num),
        head_dim=int(head_dim),
        topk=int(topk),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k,
        batch_size=meta_batch,
        max_seqlen_q=int(max_q),
        max_seqlen_k=int(max_k),
        layout_q=layout_q,
        layout_k=layout_k,
        mask_mode=int(mask_mode),
        cmp_ratio=int(cmp_ratio),
    )
    if hasattr(metadata, "to") and hasattr(q, "device") and metadata.device != q.device:
        metadata = metadata.to(q.device)
    return torch.ops.cann_ops_transformer.lightning_indexer(
        q,
        k,
        w,
        int(topk),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k,
        block_table=block_table,
        output_idx_offset=output_idx_offset,
        metadata=metadata,
        max_seqlen_q=int(max_seqlen_q),
        layout_q=layout_q,
        layout_k=layout_k,
        mask_mode=int(mask_mode),
        cmp_ratio=int(cmp_ratio),
        return_value=int(return_value),
    )
