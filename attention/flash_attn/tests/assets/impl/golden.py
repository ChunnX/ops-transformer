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

"""CPU golden adapter for FlashAttn TTK cases.

Uses FlashAttentionGolden from ops-transformer-perf.
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


def _to_int_list(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return [int(x) for x in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    return [int(value)]


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


def _build_sliding_window_mask(sq, skv, win_left, win_right, device):
    effective_win_right = skv if win_right == -1 else win_right
    effective_win_left = sq if win_left == -1 else win_left
    mask_u = torch.triu(
        torch.ones(sq, skv, device=device), diagonal=effective_win_right + 1 + skv - sq
    )
    mask_l = torch.tril(
        torch.ones(sq, skv, device=device), diagonal=-effective_win_left - 1 + skv - sq
    )
    return (mask_u + mask_l).bool()


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
    """Compute CPU golden reference using FlashAttentionGolden."""
    golden_module = _load_golden_module()
    FlashAttentionGolden = golden_module.FlashAttentionGolden

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

    golden = FlashAttentionGolden(
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        softmax_scale=float(softmax_scale),
    )

    golden_mask_mode = int(mask_mode)
    golden_attn_mask = attn_mask_cpu

    if mask_mode == 4 and attn_mask_cpu is None:
        if sq_cpu is not None:
            q_lens = _to_int_list(sq_cpu)
        elif cu_q_cpu is not None:
            cu_q_list = _to_int_list(cu_q_cpu)
            q_lens = [
                cu_q_list[i + 1] - cu_q_list[i] for i in range(len(cu_q_list) - 1)
            ]
        else:
            q_lens = None

        if skv_cpu is not None:
            kv_lens = _to_int_list(skv_cpu)
        elif cu_kv_cpu is not None:
            cu_kv_list = _to_int_list(cu_kv_cpu)
            kv_lens = [
                cu_kv_list[i + 1] - cu_kv_list[i] for i in range(len(cu_kv_list) - 1)
            ]
        else:
            kv_lens = None

        if q_lens and kv_lens:
            max_sq = max(q_lens)
            max_skv = max(kv_lens)
            golden_attn_mask = _build_sliding_window_mask(
                max_sq, max_skv, win_left, win_right, q_cpu.device
            )
        golden_mask_mode = 0

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
        softmax_scale=float(softmax_scale),
        mask_mode=golden_mask_mode,
        attn_mask=golden_attn_mask,
        layout_q=layout_q,
        layout_kv=layout_kv,
        layout_out=layout_out,
        return_softmax_lse=int(bool(return_softmax_lse)),
        device_mode="cpu",
        compute_mode="full",
    )

    if not return_softmax_lse:
        lse = None
    return out, lse
