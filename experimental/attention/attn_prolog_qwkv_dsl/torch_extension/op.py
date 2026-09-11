# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Public host entry for ``attn_prolog_qwkv``.

Dispatch: NPU tensors go to the fused CANNBotDSL kernel in :mod:`.kernel`, CPU
tensors (and shapes the fused kernel does not cover) fall back to the CPU golden
in :mod:`.golden`.  Both paths honour the same frozen interface.

Roofline for the frozen case ``dim=5120, q_lora=1280, N=32, D=128, Dr=64``,
per row of ``T``:

* Q GEMM  ``qr @ wqb^T``  1280x4096  MXFP8 -> 5.24 MMAC
* W GEMM  ``x  @ ww^T``   5120x32    BF16  -> 0.16 MMAC (3% of the cube work)
* GM traffic  ``x`` 10 KB + ``qr`` 1.3 KB + ``q`` 2 KB + ``descale_q`` 128 B
  + ``w`` 128 B, plus ``wqb`` 5.24 MB amortised over the 128 rows of a tile
  (41 KB/row, served from L2 after the first AIC touches it)
* Vector epilogue 4096 elements -> ~1.4 k 64-lane instructions, i.e. ~700
  cycles per AIV pair

So the Q GEMM is 32x the W GEMM and the whole design is built around keeping it
fed: ``qr`` plus its MX scales are pinned in L1 for the whole 32-head loop,
``wqb`` streams in 80 KB windows, and the W path's ``x`` traffic is chopped into
32 windows that are retired one per head so MTE2 never stalls the cube.
"""

from __future__ import annotations

from typing import Optional

import torch

try:
    from .golden import AttnPrologInputs, attn_prolog_qwkv_golden, ceil_div
except ImportError:  # source-tree pytest loads these as loose modules
    from golden import AttnPrologInputs, attn_prolog_qwkv_golden, ceil_div

MX_GROUP_SIZE = 32


def _is_npu(*tensors: torch.Tensor) -> bool:
    return all(t.device.type == "npu" for t in tensors)


def attn_prolog_qwkv(
    x: torch.Tensor,
    qr: torch.Tensor,
    wqb: torch.Tensor,
    ww: torch.Tensor,
    descale_qr: torch.Tensor,
    descale_wqb: torch.Tensor,
    rope_sin: torch.Tensor,
    rope_cos: torch.Tensor,
    *,
    softmax_scale: float,
    q: Optional[torch.Tensor] = None,
    descale_q: Optional[torch.Tensor] = None,
    w: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused attention prologue: MXFP8 Q GEMM + RoPE + MXFP4 quant, and BF16 W GEMM.

    Returns ``(q, descale_q, w)``. ``q`` is packed uint8 e2m1 with shape
    ``(T, N, D/2)``; ``descale_q`` is e8m0; ``w`` is fp32.
    """
    if x.dim() != 2 or qr.dim() != 2 or wqb.dim() != 2 or ww.dim() != 2:
        raise ValueError("x, qr, wqb, ww must be rank-2")
    if x.shape[0] != qr.shape[0]:
        raise ValueError(f"T mismatch: x={x.shape[0]} qr={qr.shape[0]}")
    if rope_sin.shape != rope_cos.shape:
        raise ValueError("rope_sin and rope_cos shapes must match")
    if softmax_scale is None:
        raise ValueError("softmax_scale is required")

    if _is_npu(x, qr, wqb, ww, descale_qr, descale_wqb, rope_sin, rope_cos):
        return _run_npu(
            x,
            qr,
            wqb,
            ww,
            descale_qr,
            descale_wqb,
            rope_sin,
            rope_cos,
            softmax_scale=softmax_scale,
            q=q,
            descale_q=descale_q,
            w=w,
        )

    inputs = AttnPrologInputs(
        x=x.cpu(),
        qr=qr.cpu(),
        wqb=wqb.cpu(),
        ww=ww.cpu(),
        descale_qr=descale_qr.cpu(),
        descale_wqb=descale_wqb.cpu(),
        rope_sin=rope_sin.cpu(),
        rope_cos=rope_cos.cpu(),
        softmax_scale=float(softmax_scale),
    )
    out = attn_prolog_qwkv_golden(inputs)

    def _copy(dst: Optional[torch.Tensor], src: torch.Tensor) -> torch.Tensor:
        if dst is None:
            return src
        dst.copy_(src.to(dst.device))
        return dst

    return _copy(q, out.q), _copy(descale_q, out.descale_q), _copy(w, out.w)


def _run_npu(
    x: torch.Tensor,
    qr: torch.Tensor,
    wqb: torch.Tensor,
    ww: torch.Tensor,
    descale_qr: torch.Tensor,
    descale_wqb: torch.Tensor,
    rope_sin: torch.Tensor,
    rope_cos: torch.Tensor,
    *,
    softmax_scale: float,
    q: Optional[torch.Tensor],
    descale_q: Optional[torch.Tensor],
    w: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    try:
        from .kernel import AttnPrologQwkv
    except ImportError:
        from kernel import AttnPrologQwkv

    t, dim = x.shape
    q_lora = qr.shape[1]
    n_heads = ww.shape[0]
    d = wqb.shape[0] // n_heads
    dr = rope_sin.shape[-1]
    groups = ceil_div(d, MX_GROUP_SIZE)
    dev = x.device

    op = AttnPrologQwkv(t, dim, q_lora, n_heads, d, dr)
    lut = _lut_cache(dev)
    scale_gm = torch.tensor([float(softmax_scale)], dtype=torch.float32, device=dev)

    q_flat = torch.empty((t, n_heads * (d // 2)), dtype=torch.uint8, device=dev)
    dsc_flat = torch.empty((t, n_heads * groups), dtype=torch.uint8, device=dev)
    w_out = torch.empty((t, n_heads), dtype=torch.float32, device=dev)

    # The W path reduces x straight into L0C with no vector hop, so a short last
    # row tile makes x_l1's actual shape shorter than base_m and that propagates
    # into the matmul, corrupting the second half of the head axis.  Pad x's rows
    # to whole tiles so the W GEMM only ever sees full tiles; q's own tail is
    # handled in the vector epilogue and does not need this.
    x_rows = ceil_div(t, op.p.base_m) * op.p.base_m
    if x_rows != t:
        x_pad = torch.zeros((x_rows, dim), dtype=x.dtype, device=dev)
        x_pad[:t] = x
        x = x_pad

    op.run(
        x.view(torch.bfloat16),
        qr.view(torch.uint8),
        wqb.view(torch.uint8),
        ww.view(torch.bfloat16),
        descale_qr.reshape(t, -1, 2).view(torch.uint8),
        descale_wqb.reshape(n_heads * d, -1, 2).view(torch.uint8),
        rope_sin.float(),
        rope_cos.float(),
        lut,
        scale_gm,
        q_flat,
        dsc_flat,
        w_out,
    )

    q_out = q_flat.view(t, n_heads, d // 2)
    dsc_out = dsc_flat.view(t, n_heads, ceil_div(d, 64), 2)
    if q is not None:
        q.copy_(q_out)
        q_out = q
    if descale_q is not None:
        descale_q.copy_(dsc_out)
        dsc_out = descale_q
    if w is not None:
        w.copy_(w_out)
        w_out = w
    return q_out, dsc_out, w_out


_LUT_CACHE: dict[torch.device, torch.Tensor] = {}


def _lut_cache(device: torch.device) -> torch.Tensor:
    """E2M1 rounding table, uploaded once per device."""
    try:
        from .kernel import build_e2m1_lut
    except ImportError:
        from kernel import build_e2m1_lut

    cached = _LUT_CACHE.get(device)
    if cached is None:
        cached = build_e2m1_lut().to(device)
        _LUT_CACHE[device] = cached
    return cached
