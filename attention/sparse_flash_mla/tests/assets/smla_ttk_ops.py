#!/usr/bin/python
# -*- coding: utf-8 -*-
"""TTK metadata-first adapter for the installed SparseFlashMla API."""

from typing import Optional

import torch

import cann_ops_transformer  # noqa: F401 - registers torch.ops namespace


def _max_seq(cu_seqlens, seqused, fallback=0):
    if seqused is not None and seqused.numel() > 0:
        return int(seqused.detach().cpu().max().item())
    if cu_seqlens is not None and cu_seqlens.numel() > 1:
        diff = cu_seqlens.detach().cpu()[1:] - cu_seqlens.detach().cpu()[:-1]
        return int(diff.max().item())
    return int(fallback or 0)


def _max_prefix_span(cu_seqlens, fallback=0):
    if cu_seqlens is not None and cu_seqlens.numel() > 1:
        values = cu_seqlens.detach().cpu()
        return int((values[1:] - values[:-1]).max().item())
    return int(fallback or 0)


def sparse_flash_mla_ttk(
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
    has_ori_kv: Optional[bool] = None,
    has_cmp_kv: Optional[bool] = None,
):
    """Generate metadata in TTK, then call the extension op."""
    num_heads_q = int(q.shape[1] if layout_q == "TND" else q.shape[2])
    kv_ref = ori_kv if ori_kv is not None else cmp_kv
    num_heads_kv = int(kv_ref.shape[1] if layout_kv == "TND" else kv_ref.shape[2])
    head_dim = int(q.shape[-1])
    batch_size = int(q.shape[0]) if layout_q == "BSND" else _infer_batch(cu_seqlens_q, seqused_q)
    q_fallback = q.shape[1] if layout_q == "BSND" else q.shape[0]
    max_seqlen_q = (
        int(q_fallback)
        if layout_q == "BSND"
        else _max_prefix_span(cu_seqlens_q, q_fallback)
    )
    ori_fallback = 0 if ori_kv is None else (ori_kv.shape[0] if layout_kv == "TND" else ori_kv.shape[1])
    cmp_fallback = 0 if cmp_kv is None else (cmp_kv.shape[0] if layout_kv == "TND" else cmp_kv.shape[1])
    max_seqlen_ori_kv = _max_seq(cu_seqlens_ori_kv, seqused_ori_kv, ori_fallback)
    max_seqlen_cmp_kv = _max_seq(cu_seqlens_cmp_kv, seqused_cmp_kv, cmp_fallback)
    ori_topk = int(ori_sparse_indices.shape[-1]) if ori_sparse_indices is not None else 0
    cmp_topk = int(cmp_sparse_indices.shape[-1]) if cmp_sparse_indices is not None else 0
    if has_ori_kv is None:
        has_ori_kv = ori_kv is not None
    if has_cmp_kv is None:
        has_cmp_kv = cmp_kv is not None

    metadata = torch.ops.cann_ops_transformer.sparse_flash_mla_metadata(
        int(num_heads_q),
        int(num_heads_kv),
        int(head_dim),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_ori_kv=cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
        seqused_q=seqused_q,
        seqused_ori_kv=seqused_ori_kv,
        seqused_cmp_kv=seqused_cmp_kv,
        cmp_residual_kv=cmp_residual_kv,
        ori_topk_length=ori_topk_length,
        cmp_topk_length=cmp_topk_length,
        batch_size=int(batch_size),
        max_seqlen_q=int(max_seqlen_q),
        max_seqlen_ori_kv=int(max_seqlen_ori_kv),
        max_seqlen_cmp_kv=int(max_seqlen_cmp_kv),
        ori_topk=int(ori_topk),
        cmp_topk=int(cmp_topk),
        cmp_ratio=int(cmp_ratio),
        ori_mask_mode=int(ori_mask_mode),
        cmp_mask_mode=int(cmp_mask_mode),
        ori_win_left=int(ori_win_left),
        ori_win_right=int(ori_win_right),
        layout_q=layout_q,
        layout_kv=layout_kv,
        has_ori_kv=bool(has_ori_kv),
        has_cmp_kv=bool(has_cmp_kv),
    )
    if hasattr(metadata, "to") and metadata.device != q.device:
        metadata = metadata.to(q.device)

    return torch.ops.cann_ops_transformer.sparse_flash_mla(
        q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        ori_sparse_indices=ori_sparse_indices,
        cmp_sparse_indices=cmp_sparse_indices,
        ori_block_table=ori_block_table,
        cmp_block_table=cmp_block_table,
        cu_seqlens_q=cu_seqlens_q if layout_q == "TND" else None,
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
        softmax_scale=float(softmax_scale),
        cmp_ratio=int(cmp_ratio),
        ori_mask_mode=int(ori_mask_mode),
        cmp_mask_mode=int(cmp_mask_mode),
        ori_win_left=int(ori_win_left),
        ori_win_right=int(ori_win_right),
        layout_q=layout_q,
        layout_kv=layout_kv,
        topk_value_mode=int(topk_value_mode),
        return_softmax_lse=bool(return_softmax_lse),
    )


def _infer_batch(cu_seqlens, seqused):
    if seqused is not None:
        return int(seqused.numel())
    if cu_seqlens is not None:
        return max(int(cu_seqlens.numel()) - 1, 0)
    return 0
