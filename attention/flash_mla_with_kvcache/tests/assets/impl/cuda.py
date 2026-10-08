# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""DeepSeek FlashMLA dense decode adapter for TTK's CUDA third party.

Imports flash_mla only on execution. NPU metadata is never reused on CUDA.
Batch-local calls preserve right-aligned causal semantics for variable Q lengths.
"""

import torch


def _lengths(tensor, batch, name):
    if tensor is None or tensor.ndim != 1 or tensor.numel() != batch:
        raise ValueError(f"{name} must contain {batch} lengths")
    return [int(x) for x in tensor.detach().cpu().tolist()]


def _cache_bbnd(cache, layout):
    if layout == "PA_BBND" and cache.ndim == 4:
        result = cache
    elif layout == "PA_BNBD" and cache.ndim == 4:
        result = cache.permute(0, 2, 1, 3)
    elif layout == "PA_NZ" and cache.ndim == 5 and cache.shape[-1] == 16:
        nb, nk, d1, bs, d0 = cache.shape
        result = cache.permute(0, 3, 1, 2, 4).reshape(nb, bs, nk, d1 * d0)
    else:
        raise ValueError(
            f"Unsupported CUDA MLA KV layout/shape: {layout}, {cache.shape}"
        )
    if result.shape[2:] != (1, 576):
        raise ValueError("DeepSeek MLA requires one KV head and head_dim_qk=576")
    return result.contiguous()


class DeepSeekFlashMla:
    @torch.no_grad()
    def __call__(
        self,
        q,
        k_cache,
        block_table=None,
        cache_seqlens=None,
        cu_seqlens_q=None,
        seqused_q=None,
        attn_mask=None,
        metadata=None,
        head_dim_v=512,
        softmax_scale=1.0,
        mask_mode=0,
        max_seqlen_q=-1,
        max_seqlen_kv=-1,
        layout_q="BSND",
        layout_kv="PA_BBND",
        layout_out="BSND",
        return_softmax_lse=False,
        **unused,
    ):
        if q.device.type != "cuda" or k_cache.device != q.device:
            raise ValueError(
                "DeepSeek FlashMLA requires Q and KV on the same CUDA device"
            )
        if q.dtype not in (torch.float16, torch.bfloat16) or k_cache.dtype != q.dtype:
            raise ValueError("DeepSeek FlashMLA requires matching FP16/BF16 Q and KV")
        if q.shape[-1] != 576 or head_dim_v != 512:
            raise ValueError("DeepSeek FlashMLA requires D=576 and DV=512")
        if mask_mode not in (0, 3):
            raise ValueError("DeepSeek dense MLA adapter supports only mask_mode=0/3")
        if unused.get("sinks") is not None:
            raise ValueError("Sinks are not supported by this MLA adapter")
        if mask_mode == 0 and attn_mask is not None and bool(attn_mask.any()):
            raise ValueError(
                "DeepSeek dense MLA cannot represent an explicit nonzero mask"
            )
        from flash_mla import flash_mla_with_kvcache, get_mla_metadata

        return _run(
            q,
            k_cache,
            block_table,
            cache_seqlens,
            cu_seqlens_q,
            seqused_q,
            head_dim_v,
            softmax_scale,
            mask_mode,
            layout_q,
            layout_kv,
            layout_q if layout_out is None else layout_out,
            return_softmax_lse,
            flash_mla_with_kvcache,
            get_mla_metadata,
        )


def _run(
    q,
    k_cache,
    block_table,
    cache_seqlens,
    cu_seqlens_q,
    seqused_q,
    head_dim_v,
    softmax_scale,
    mask_mode,
    layout_q,
    layout_kv,
    layout_out,
    return_softmax_lse,
    flash_mla,
    get_metadata,
):
    if layout_q == "TND":
        if q.ndim != 3 or cu_seqlens_q is None:
            raise ValueError("TND requires a 3D Q and cu_seqlens_q")
        cu = cu_seqlens_q.detach().cpu().tolist()
        if (
            len(cu) < 2
            or cu[0] != 0
            or cu[-1] != q.shape[0]
            or any(a > b for a, b in zip(cu, cu[1:]))
        ):
            raise ValueError("cu_seqlens_q must monotonically cover Q")
        batch, heads = len(cu) - 1, q.shape[1]
        physical = [b - a for a, b in zip(cu, cu[1:])]
        out = torch.zeros((*q.shape[:-1], head_dim_v), dtype=q.dtype, device=q.device)
        lse = torch.full((heads, q.shape[0]), float("inf"), device=q.device)
    elif layout_q in ("BNSD", "BSND") and q.ndim == 4:
        batch = q.shape[0]
        heads = q.shape[1 if layout_q == "BNSD" else 2]
        physical = [q.shape[2 if layout_q == "BNSD" else 1]] * batch
        out = torch.zeros(
            (batch, physical[0], heads, head_dim_v), dtype=q.dtype, device=q.device
        )
        lse = torch.full((batch, heads, physical[0]), float("inf"), device=q.device)
    else:
        raise ValueError(f"Unsupported CUDA MLA Q layout/shape: {layout_q}, {q.shape}")
    qlens = physical if seqused_q is None else _lengths(seqused_q, batch, "seqused_q")
    klens = _lengths(cache_seqlens, batch, "cache_seqlens")
    if any(s < 0 or s > p for s, p in zip(qlens, physical)) or any(
        k < 0 for k in klens
    ):
        raise ValueError("Invalid Q/KV sequence lengths")
    if block_table is None or block_table.ndim != 2 or block_table.shape[0] != batch:
        raise ValueError("block_table must have one row per batch")
    cache = _cache_bbnd(k_cache, layout_kv)
    bs = cache.shape[1]
    # DeepSeek dense decode uses 64-token pages. Split physical pages and
    # translate only referenced page IDs, preserving sharing and unused -1s.
    if bs <= 0 or bs % 64:
        raise ValueError("DeepSeek MLA adapter requires a KV page size divisible by 64")
    factor = bs // 64
    cache = cache.reshape(-1, 64, 1, 576)
    for b, (sq, skv) in enumerate(zip(qlens, klens)):
        if sq == 0 or skv == 0:
            continue
        pages = (skv + bs - 1) // bs
        if pages > block_table.shape[1]:
            raise ValueError("block_table does not cover cache_seqlens")
        ids = block_table[b, :pages].to(device=q.device, dtype=torch.int64)
        if bool(((ids < 0) | (ids >= cache.shape[0] // factor)).any()):
            raise ValueError("Referenced block_table index is outside KV cache")
        ids = (ids[:, None] * factor + torch.arange(factor, device=q.device)).flatten()
        ids = ids[: (skv + 63) // 64].to(torch.int32).unsqueeze(0).contiguous()
        if layout_q == "TND":
            qb = q[cu[b] : cu[b] + sq].unsqueeze(0)
        elif layout_q == "BNSD":
            qb = q[b : b + 1, :, :sq].permute(0, 2, 1, 3)
        else:
            qb = q[b : b + 1, :sq]
        lens = torch.tensor([skv], dtype=torch.int32, device=q.device)
        scheduler, splits = get_metadata(lens, sq * heads, 1)
        ob, lb = flash_mla(
            qb.contiguous(),
            cache,
            ids,
            lens,
            head_dim_v,
            scheduler,
            splits,
            softmax_scale=softmax_scale,
            causal=mask_mode == 3,
        )
        if tuple(ob.shape) != (1, sq, heads, head_dim_v) or tuple(lb.shape) != (
            1,
            heads,
            sq,
        ):
            raise ValueError("Unexpected DeepSeek FlashMLA output/LSE shapes")
        # MLA contract uses +Inf for rows with no visible KV; CUDA versions
        # can return -Inf LSE for those rows. Normalize only geometric empty rows.
        if mask_mode == 3 and sq > skv:
            empty = sq - skv
            ob = ob.clone()
            lb = lb.clone()
            ob[:, :empty] = 0
            lb[:, :, :empty] = float("inf")
        if layout_q == "TND":
            out[cu[b] : cu[b] + sq] = ob[0]
            lse[:, cu[b] : cu[b] + sq] = lb[0]
        else:
            out[b, :sq] = ob[0]
            lse[b, :, :sq] = lb[0]
    if layout_q == "TND":
        if layout_out == "NTD":
            out = out.permute(1, 0, 2)
        elif layout_out != "TND":
            raise ValueError(f"Unsupported TND output layout: {layout_out}")
    elif layout_out == "BNSD":
        out = out.permute(0, 2, 1, 3)
    elif layout_out == "NBSD":
        out = out.permute(2, 0, 1, 3)
    elif layout_out != "BSND":
        raise ValueError(f"Unsupported output layout: {layout_out}")
    return out.contiguous(), lse.contiguous() if return_softmax_lse else None
