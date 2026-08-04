#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CPU golden adapter for FlashAttn TTK cases.

Uses FlashAttnGolden from ops-transformer-perf.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Optional

import torch


_GOLDEN_PARENT_DIR = Path("/workspace/Code/perf/ops-transformer-perf")
_GOLDEN_MODULE = None


def _load_golden_module():
    global _GOLDEN_MODULE
    if _GOLDEN_MODULE is not None:
        return _GOLDEN_MODULE
    module_path = _GOLDEN_PARENT_DIR / "golden" / "flash_attn_golden.py"
    spec = importlib.util.spec_from_file_location(
        f"ttk_flash_attn_golden_{abs(hash(module_path))}", module_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _GOLDEN_MODULE = module
    return _GOLDEN_MODULE


def _to_cpu(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu()
    return value


def _derive_head_dims(q, k, layout_q, layout_kv):
    head_dim = int(q.shape[-1])
    if layout_q == "TND":
        num_heads_q = int(q.shape[1])
    elif layout_q == "BNSD":
        num_heads_q = int(q.shape[1])
    else:
        num_heads_q = int(q.shape[2])

    if layout_kv == "TND":
        num_heads_kv = int(k.shape[1])
    elif layout_kv == "BNSD":
        num_heads_kv = int(k.shape[1])
    elif layout_kv in ("PA_BBND", "PA_BNBD"):
        num_heads_kv = int(k.shape[2]) if layout_kv == "PA_BBND" else int(k.shape[1])
    else:
        num_heads_kv = int(k.shape[2])
    return num_heads_q, num_heads_kv, head_dim


def cpu_flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = None,
    sinks: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    softmax_scale: float = 1.0,
    mask_mode: int = 0,
    win_left: int = -1,
    win_right: int = -1,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_kv: Optional[int] = None,
    layout_q: str = "BSND",
    layout_kv: str = "BSND",
    layout_out: str = "BSND",
    return_softmax_lse: bool = False,
    **_unused,
):
    """Compute CPU golden reference using FlashAttnGolden."""
    golden_module = _load_golden_module()
    FlashAttnGolden = golden_module.FlashAttnGolden

    q_cpu = _to_cpu(q)
    k_cpu = _to_cpu(k)
    v_cpu = _to_cpu(v)
    bt_cpu = _to_cpu(block_table)
    cu_q_cpu = _to_cpu(cu_seqlens_q)
    cu_kv_cpu = _to_cpu(cu_seqlens_kv)
    sq_cpu = _to_cpu(seqused_q)
    skv_cpu = _to_cpu(seqused_kv)
    sinks_cpu = _to_cpu(sinks)
    attn_mask_cpu = _to_cpu(attn_mask)

    num_heads_q, num_heads_kv, head_dim = _derive_head_dims(
        q_cpu, k_cpu, layout_q, layout_kv
    )

    golden = FlashAttnGolden(
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        softmax_scale=float(softmax_scale),
    )

    out, lse = golden.forward_golden(
        q_cpu,
        k_cpu,
        v_cpu,
        block_table=bt_cpu,
        cu_seqlens_q=cu_q_cpu,
        cu_seqlens_kv=cu_kv_cpu,
        seqused_q=sq_cpu,
        seqused_kv=skv_cpu,
        sinks=sinks_cpu,
        attn_mask=attn_mask_cpu,
        softmax_scale=float(softmax_scale),
        mask_mode=int(mask_mode),
        win_left=int(win_left),
        win_right=int(win_right),
        max_seqlen_q=int(max_seqlen_q) if max_seqlen_q is not None else -1,
        max_seqlen_kv=int(max_seqlen_kv) if max_seqlen_kv is not None else -1,
        layout_q=layout_q,
        layout_kv=layout_kv,
        layout_out=layout_out,
        return_softmax_lse=bool(return_softmax_lse),
        device_mode="cpu",
        compute_mode="full",
    )

    if not return_softmax_lse:
        lse = None
    return out, lse
