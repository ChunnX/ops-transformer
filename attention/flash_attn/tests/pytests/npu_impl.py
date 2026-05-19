#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to License for details. You may not use this file except in compliance with License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import torch
import torch_npu
import math
import numpy as np
import random
from einops import rearrange
import npu_ops_transformer
from npu_ops_transformer.ops import npu_flash_attn
from npu_ops_transformer.ops import npu_flash_attn_metadata
from test_utils import trans_bnsd_to_layout
import torchair
from torchair.configs.compiler_config import CompilerConfig

def _is_small_1d_int_tensor(x):
    return torch.is_tensor(x) and x.dim() == 1 and x.numel() <= 64 and x.dtype in (torch.int64, torch.int32)

def _summ_val(x):
    if torch.is_tensor(x):
        if _is_small_1d_int_tensor(x):
            return x.cpu().tolist()
        return f"Tensor(dtype={str(x.dtype)}, shape={tuple(x.shape)}, device={str(x.device)})"
    return x

def _print_args(title, args_dict):
    print(f"\n[PyTest] {title}")
    for k in sorted(args_dict.keys()):
        print(f"  - {k}: {_summ_val(args_dict[k])}")


def print_metadata(metadata):
    """Decode and print all FA/FD fields from the AICPU-generated metadata buffer.

    Buffer layout (uint32 elements, matching flash_attn_metadata.h):
      [0]          sectionNum
      [1]          mBaseSize
      [2]          s2BaseSize
      [3..15]      reserved
      [16 + sectionIdx * AIC*16 + coreIdx*16 + fieldIdx]   FA fields (7 used)
      [16 + sectionNum*AIC*16 + sectionIdx*AIV*16 + coreIdx*16 + fieldIdx]  FD fields (6 used)
    """
    AIC_CORE_NUM = 36
    AIV_CORE_NUM = 72
    FA_META_SIZE = 16   # uint32 slots per AIC core
    FD_META_SIZE = 16   # uint32 slots per AIV core
    HEADER_SIZE  = 16   # uint32 slots for the header

    FA_FIELDS = ["BN2_START", "M_START",  "S2_START",
                 "BN2_END",   "M_END",    "S2_END",   "FIRST_FD_WS_IDX"]
    FD_FIELDS = ["BN2_IDX",  "M_IDX",    "WS_IDX",
                 "WS_NUM",   "M_START",  "M_NUM"]

    # Reinterpret buffer as flat uint32 array regardless of original dtype
    # arr = metadata.cpu().contiguous().view(torch.int32).numpy().view(np.uint32)
    arr = metadata.cpu().contiguous().to(torch.int32).numpy().astype(np.uint32)

    section_num  = int(arr[0])
    is_FD = int(arr[1])
    m_base_size  = int(arr[2])
    s2_base_size = int(arr[3])

    SEP  = "=" * 78
    DASH = "-" * 78

    print(f"\n{SEP}")
    print(f"  [Metadata Header]")
    print(f"  sectionNum  = {section_num}")
    print(f"  isFD   = {is_FD}")
    print(f"  mBaseSize   = {m_base_size}")
    print(f"  s2BaseSize  = {s2_base_size}")

    fa_base = HEADER_SIZE
    fd_base = HEADER_SIZE + section_num * AIC_CORE_NUM * FA_META_SIZE

    for sec in range(section_num):
        print(f"\n{DASH}")
        print(f"  [Section {sec}] FA Metadata — AIC cores  "
              f"(36 cores × 16 slots, 7 fields used)")
        print(DASH)
        cw = 15
        hdr = f"  {'Core':<7}" + "".join(f"{n:>{cw}}" for n in FA_FIELDS)
        print(hdr)
        print(f"  {'':<7}" + "".join(f"{'['+str(i)+']':>{cw}}" for i in range(len(FA_FIELDS))))
        print(f"  {'─'*7}" + "─" * (cw * len(FA_FIELDS)))
        for core in range(AIC_CORE_NUM):
            off = fa_base + sec * AIC_CORE_NUM * FA_META_SIZE + core * FA_META_SIZE
            vals = [int(arr[off + i]) for i in range(len(FA_FIELDS))]
            note = "  (inactive)" if all(v == 0 for v in vals) else ""
            print(f"  AIC{core:02d}  " + "".join(f"{v:>{cw}}" for v in vals) + note)

    for sec in range(section_num):
        print(f"\n{DASH}")
        print(f"  [Section {sec}] FD Metadata — active AIV cores only  "
              f"(M_NUM > 0 shown, 72 cores × 16 slots, 6 fields used)")
        print(DASH)
        cw = 12
        hdr = f"  {'Core':<7}" + "".join(f"{n:>{cw}}" for n in FD_FIELDS)
        print(hdr)
        print(f"  {'':<7}" + "".join(f"{'['+str(i)+']':>{cw}}" for i in range(len(FD_FIELDS))))
        print(f"  {'─'*7}" + "─" * (cw * len(FD_FIELDS)))
        active = 0
        for core in range(AIV_CORE_NUM):
            off = fd_base + sec * AIV_CORE_NUM * FD_META_SIZE + core * FD_META_SIZE
            vals = [int(arr[off + i]) for i in range(len(FD_FIELDS))]
            if vals[5] == 0:   # FD_M_NUM_INDEX = 5; skip inactive cores
                continue
            print(f"  AIV{core:02d} " + "".join(f"{v:>{cw}}" for v in vals))
            active += 1
        if active == 0:
            print("  (no active FD cores)")

    print(f"{SEP}\n")

device_id = 0
device = torch.device(f"npu:{device_id}")

def flash_attn_metadata_only(**kwargs):
    """Call only npu_flash_attn_metadata. No q/k/v tensors needed.
    All shape params are derived from kwargs (set by normalize_case + call_flash_attn TND preprocessing).
    """
    input_layout = kwargs.get("input_layout", "BNSD")
    layout_q   = kwargs.get("layout_q",   input_layout)
    layout_kv  = kwargs.get("layout_kv",  input_layout)
    layout_out = kwargs.get("layout_out", input_layout)
    n1          = kwargs.get("N1")
    n2          = kwargs.get("N2", n1)
    b           = kwargs.get("B", 1)
    d           = kwargs.get("D")
    sq          = kwargs.get("S1", None)
    skv         = kwargs.get("S2", sq)
    sparse_mode = kwargs.get("sparse_mode", 0)
    pre_tokens  = kwargs.get("pre_tokens",  2147483647)
    next_tokens = kwargs.get("next_tokens", 2147483647)
    if sparse_mode in (0, 3):
        pre_tokens = -1
        next_tokens = -1
    actual_seq_qlen  = kwargs.get("actual_seq_qlen",  None)
    actual_seq_kvlen = kwargs.get("actual_seq_kvlen", actual_seq_qlen)
    cu_q       = kwargs.get("cu_seqlens_q", None)
    cu_kv      = kwargs.get("cu_seqlens_kv", None)
    seqused_q  = kwargs.get("seqused_q", actual_seq_qlen)
    seqused_kv = kwargs.get("seqused_kv", actual_seq_kvlen)

    # Derive actual batch_size from cu_seqlens list (B+1 elements) when new 4-param style is used.
    # B in kwargs is set to 1 for TND tensor generation and must not be used here.
    if isinstance(cu_q, list):
        batch_size = len(cu_q) - 1
    elif isinstance(seqused_q, list):
        batch_size = len(seqused_q)
    else:
        batch_size = b

    def _to_u32_tensor(v):
        if v is None:
            return None
        if torch.is_tensor(v):
            return v.to(dtype=torch.int32, device=device).contiguous()
        return torch.tensor(v, dtype=torch.int32, device=device).contiguous()

    max_seqlen_q  = sq  if sq  is not None else 0
    max_seqlen_kv = skv if skv is not None else 0

    _print_args("Call npu_flash_attn_metadata (metadata_only)", {
        "cu_seqlens_q": cu_q,
        "cu_seqlens_kv": cu_kv,
        "seqused_q": seqused_q,
        "seqused_kv": seqused_kv,
        "num_heads_q": num_heads_q,
        "num_heads_kv": num_heads_kv,
        "head_dim": head_dim,
        "batch_size": batch_size,
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_kv": max_seqlen_kv,
        "mask_mode": sparse_mode,
        "win_left": pre_tokens,
        "win_right": next_tokens,
        "layout_q": layout_q,
        "layout_kv": layout_kv,
        "layout_out": layout_out,
    })

    metadata = torch.ops.npu_ops_transformer.npu_flash_attn_metadata(
        cu_seqlens_q  = cu_q_u32,
        cu_seqlens_kv = cu_kv_u32,
        seqused_q     = seqused_q_u32,
        seqused_kv    = seqused_kv_u32,
        num_heads_q   = num_heads_q,
        num_heads_kv  = num_heads_kv,
        head_dim      = head_dim,
        batch_size    = batch_size,
        max_seqlen_q  = max_seqlen_q,
        max_seqlen_kv = max_seqlen_kv,
        mask_mode     = sparse_mode,
        win_left      = pre_tokens,
        win_right     = next_tokens,
        layout_q      = layout_q,
        layout_kv     = layout_kv,
        layout_out    = layout_out,
    )
    print_metadata(metadata)
    return metadata


def flash_attn_npu(q, k, v, q_rope, k_rope, atten_mask, pse, **kwargs):
    input_layout        = kwargs.get("input_layout", "BNSD")
    layout_q            = kwargs.get("layout_q",   input_layout)
    layout_kv           = kwargs.get("layout_kv",  input_layout)
    layout_out          = kwargs.get("layout_out", input_layout)
    n1                  = kwargs.get("N1")
    n2                  = kwargs.get("N2", n1)
    b                   = kwargs.get("B",  1)
    d                   = kwargs.get("D")
    sq                  = kwargs.get("S1", None)
    skv                 = kwargs.get("S2", sq)
    d_rope              = kwargs.get("DRope", 0)
    sparse_mode         = kwargs.get("sparse_mode", 0)
    pre_tokens          = kwargs.get("pre_tokens", 2147483647)
    next_tokens         = kwargs.get("next_tokens", 2147483647)
    if sparse_mode in (0, 3):
        pre_tokens = -1
        next_tokens = -1
    actual_seq_qlen  = kwargs.get("actual_seq_qlen",  None)
    actual_seq_kvlen = kwargs.get("actual_seq_kvlen", actual_seq_qlen)
    cu_q       = kwargs.get("cu_seqlens_q", None)
    cu_kv      = kwargs.get("cu_seqlens_kv", None)
    seqused_q  = kwargs.get("seqused_q", actual_seq_qlen)
    seqused_kv = kwargs.get("seqused_kv", actual_seq_kvlen)
    scale = kwargs.get("scale", 1 / (d ** 0.5))
    keep_prob = kwargs.get("keep_prob", 1)
    seed = kwargs.get("seed", 0)
    lse_flag = kwargs.get("return_softmax_lse", 0)

    # q1 = trans_bnsd_to_layout(q, layout_q).contiguous().to(device)
    # k1 = trans_bnsd_to_layout(k, layout_kv).contiguous().to(device)
    # v1 = trans_bnsd_to_layout(v, layout_kv).contiguous().to(device)


    # actual_seq_qlen     = kwargs.get("actual_seq_qlen",  None)
    # actual_seq_kvlen    = kwargs.get("actual_seq_kvlen", actual_seq_qlen)
    # # New 4-param style: normalize_case sets _npu_cu_q/kv (full B+1 list) and _npu_seqused_q/kv (B individual).
    # # Legacy style: these are None, fall back to last element / cumulative list.
    # _cu_q               = kwargs.get("_npu_cu_q",       None)
    # _cu_kv              = kwargs.get("_npu_cu_kv",      None)
    # _seqused_q          = kwargs.get("_npu_seqused_q",  None)
    # _seqused_kv         = kwargs.get("_npu_seqused_kv", None)
    # scale               = kwargs.get("scale", 1 / (d ** 0.5))
    # keep_prob           = kwargs.get("keep_prob", 1)
    # seed                = kwargs.get("seed", 0)
    # lse_flag            = kwargs.get("return_softmax_lse", 0)
    block_table         = kwargs.get("block_table")

    q1 = trans_bnsd_to_layout(q, layout_q, **kwargs).contiguous().to(device)
    k1 = trans_bnsd_to_layout(k, layout_kv, **kwargs).contiguous().to(device)
    v1 = trans_bnsd_to_layout(v, layout_kv, **kwargs).contiguous().to(device)

    if d_rope != 0:
        query_rope = trans_bnsd_to_layout(q_rope, layout_q, **kwargs).contiguous().to(device)
        key_rope   = trans_bnsd_to_layout(k_rope, layout_kv, **kwargs).contiguous().to(device)
    else:
        query_rope = None
        key_rope   = None
    pse1        = pse.to(device)        if pse        is not None else None
    atten_mask1 = atten_mask.to(device) if atten_mask is not None else None

    torch.npu.manual_seed(seed)

    # cuSeqlensQ/Kv: for TND, normalize_case puts the full (B+1,) list; for others, None
    # The C++ operator signature is Optional[Tensor]; Python list cannot be auto-cast, convert explicitly.
    # kernel (fia_block_vec_noquant_gqa.h) reads seqlen arrays as (__gm__ uint64_t *), MUST be int64.
    def _to_u32_tensor(v):
        if v is None:
            return None
        if torch.is_tensor(v):
            return v.to(dtype=torch.int32, device=device).contiguous()
        return torch.tensor(v, dtype=torch.int32, device=device).contiguous()

    cu_q_meta       = _to_u32_tensor(cu_q)
    cu_kv_meta      = _to_u32_tensor(cu_kv)
    seqused_q_meta  = _to_u32_tensor(seqused_q)
    seqused_kv_meta = _to_u32_tensor(seqused_kv)

    # 从实际张量形状推导 npu_flash_attn_metadata 所需参数
    if layout_q == "BNSD":
        batch_size    = q1.shape[0]
        num_heads_q   = q1.shape[1]
        max_seqlen_q  = q1.shape[2]
        head_dim      = q1.shape[3]
        num_heads_kv  = k1.shape[1]
        max_seqlen_kv = k1.shape[2]
    elif layout_q == "BSND":
        batch_size    = q1.shape[0]
        num_heads_q   = q1.shape[2]
        max_seqlen_q  = q1.shape[1]
        head_dim      = q1.shape[3]
        num_heads_kv  = k1.shape[2]
        max_seqlen_kv = k1.shape[1]
    elif layout_q == "TND":
        # B in kwargs = 1 (for tensor generation); derive actual request count from cu_seqlens/seqused
        if cu_q_meta is not None and hasattr(cu_q_meta, 'shape'):
            batch_size = cu_q_meta.shape[0] - 1
        elif seqused_q_meta is not None and hasattr(seqused_q_meta, 'shape'):
            batch_size = seqused_q_meta.shape[0]
        else:
            batch_size = b
        num_heads_q   = q1.shape[1]
        head_dim      = q1.shape[2]
        num_heads_kv  = k1.shape[1]
        max_seqlen_q  = sq  if sq  is not None else q1.shape[0]
        max_seqlen_kv = skv if skv is not None else k1.shape[0]
    else:  # BSH
        batch_size    = q1.shape[0]
        num_heads_q   = n1
        num_heads_kv  = n2
        head_dim      = d
        max_seqlen_q  = q1.shape[1]
        max_seqlen_kv = k1.shape[1]
    
    block_table_ = None
    # PA
    if layout_kv in ["PA_BBND", "PA_BNBD", "PA_NZ"]:
        block_table_  = block_table.to(device)
        num_heads_kv  = kwargs.get("N2")
        head_dim      = d
        max_seqlen_kv = max(seqused_kv)

    _print_args("Call npu_flash_attn_metadata", {
        "cu_seqlens_q": cu_q_meta,
        "cu_seqlens_kv": cu_kv_meta,
        "seqused_q": seqused_q_meta,
        "seqused_kv": seqused_kv_meta,
        "num_heads_q": num_heads_q,
        "num_heads_kv": num_heads_kv,
        "head_dim": head_dim,
        "batch_size": batch_size,
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_kv": max_seqlen_kv,
        "mask_mode": sparse_mode,
        "win_left": pre_tokens,
        "win_right": next_tokens,
        "layout_q": layout_q,
        "layout_kv": layout_kv,
        "layout_out": layout_out,
    })

    metadata = torch.ops.npu_ops_transformer.npu_flash_attn_metadata(
        cu_seqlens_q  = cu_q_meta,
        cu_seqlens_kv = cu_kv_meta,
        seqused_q     = seqused_q_meta,
        seqused_kv    = seqused_kv_meta,
        num_heads_q   = num_heads_q,
        num_heads_kv  = num_heads_kv,
        head_dim      = head_dim,
        batch_size    = batch_size,
        max_seqlen_q  = max_seqlen_q,
        max_seqlen_kv = max_seqlen_kv,
        mask_mode     = sparse_mode,
        win_left      = pre_tokens,
        win_right     = next_tokens,
        layout_q      = layout_q,
        layout_kv     = layout_kv,
        layout_out    = layout_out,
    )
    print_metadata(metadata)

    max_seqlen_q  = -1
    max_seqlen_kv = -1
    atten_mask1 = atten_mask1 if sparse_mode != 0 else None
    if atten_mask1 is not None and sparse_mode in (3, 4):
        atten_mask1 = atten_mask1.to(dtype=torch.int8)

    _print_args("Call npu_flash_attn", {
        "q": q1,
        "k": k1,
        "v": v1,
        "block_table": block_table_,
        "attn_mask": atten_mask1,
        "metadata": metadata,
        "cu_seqlens_q": cu_q_meta,
        "cu_seqlens_kv": cu_kv_meta,
        "seqused_q": seqused_q_meta,
        "seqused_kv": seqused_kv_meta,
        "softmax_scale": scale,
        "mask_mode": sparse_mode,
        "win_left": pre_tokens,
        "win_right": next_tokens,
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_kv": max_seqlen_kv,
        "layout_q": layout_q,
        "layout_kv": layout_kv,
        "layout_out": layout_out,
    })

    out, lse_out = npu_flash_attn(
        q1, k1, v1,
        block_table   = block_table_,
        sinks         = None,
        attn_mask     = atten_mask1,
        metadata      = metadata,
        cu_seqlens_q  = cu_q_meta,
        cu_seqlens_kv = cu_kv_meta,
        seqused_q     = seqused_q_meta,
        seqused_kv    = seqused_kv_meta,
        softmax_scale = scale,
        mask_mode     = sparse_mode,
        win_left      = pre_tokens,
        win_right     = next_tokens,
        max_seqlen_q  = max_seqlen_q,
        max_seqlen_kv = max_seqlen_kv,
        layout_q      = layout_q,
        layout_kv     = layout_kv,
        layout_out    = layout_out,
        return_softmax_lse = lse_flag,
        deterministic = 0,
    )
    torch.npu.synchronize()
    return out.cpu(), lse_out.cpu()


class FlashAttnGraphNetwork(torch.nn.Module):
    def __init__(self):
        super(FlashAttnGraphNetwork, self).__init__()

    def forward(self, q, k, v, block_table, attn_mask, metadata,
                cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv,
                softmax_scale, mask_mode, win_left, win_right,
                max_seqlen_q, max_seqlen_kv, layout_q, layout_kv, layout_out,
                return_softmax_lse, deterministic):
        out, lse_out = npu_flash_attn(
            q, k, v,
            block_table   = block_table,
            sinks         = None,
            attn_mask     = attn_mask,
            metadata      = metadata,
            cu_seqlens_q  = cu_seqlens_q,
            cu_seqlens_kv = cu_seqlens_kv,
            seqused_q     = seqused_q,
            seqused_kv    = seqused_kv,
            softmax_scale = softmax_scale,
            mask_mode     = mask_mode,
            win_left      = win_left,
            win_right     = win_right,
            max_seqlen_q  = max_seqlen_q,
            max_seqlen_kv = max_seqlen_kv,
            layout_q      = layout_q,
            layout_kv     = layout_kv,
            layout_out    = layout_out,
            return_softmax_lse = return_softmax_lse,
            deterministic = deterministic,
        )
        return out, lse_out


def flash_attn_npu_graph(q, k, v, q_rope, k_rope, atten_mask, pse, **kwargs):
    input_layout        = kwargs.get("input_layout", "BNSD")
    layout_q            = kwargs.get("layout_q",   input_layout)
    layout_kv           = kwargs.get("layout_kv",  input_layout)
    layout_out          = kwargs.get("layout_out", input_layout)
    n1                  = kwargs.get("N1")
    n2                  = kwargs.get("N2", n1)
    b                   = kwargs.get("B",  1)
    d                   = kwargs.get("D")
    sq                  = kwargs.get("S1", None)
    skv                 = kwargs.get("S2", sq)
    d_rope              = kwargs.get("DRope", 0)
    sparse_mode         = kwargs.get("sparse_mode", 0)
    pre_tokens          = kwargs.get("pre_tokens", 2147483647)
    next_tokens         = kwargs.get("next_tokens", 2147483647)
    if sparse_mode in (0, 3):
        pre_tokens = -1
        next_tokens = -1
    actual_seq_qlen  = kwargs.get("actual_seq_qlen",  None)
    actual_seq_kvlen = kwargs.get("actual_seq_kvlen", actual_seq_qlen)
    cu_q       = kwargs.get("cu_seqlens_q", None)
    cu_kv      = kwargs.get("cu_seqlens_kv", None)
    seqused_q  = kwargs.get("seqused_q", actual_seq_qlen)
    seqused_kv = kwargs.get("seqused_kv", actual_seq_kvlen)
    scale = kwargs.get("scale", 1 / (d ** 0.5))
    keep_prob = kwargs.get("keep_prob", 1)
    seed = kwargs.get("seed", 0)
    lse_flag = kwargs.get("return_softmax_lse", 0)
    block_table         = kwargs.get("block_table")

    q1 = trans_bnsd_to_layout(q, layout_q, **kwargs).contiguous().to(device)
    k1 = trans_bnsd_to_layout(k, layout_kv, **kwargs).contiguous().to(device)
    v1 = trans_bnsd_to_layout(v, layout_kv, **kwargs).contiguous().to(device)

    if d_rope != 0:
        query_rope = trans_bnsd_to_layout(q_rope, layout_q, **kwargs).contiguous().to(device)
        key_rope   = trans_bnsd_to_layout(k_rope, layout_kv, **kwargs).contiguous().to(device)
    else:
        query_rope = None
        key_rope   = None
    pse1        = pse.to(device)        if pse        is not None else None
    atten_mask1 = atten_mask.to(device) if atten_mask is not None else None

    torch.npu.manual_seed(seed)

    def _to_u32_tensor(v):
        if v is None:
            return None
        if torch.is_tensor(v):
            return v.to(dtype=torch.int32, device=device).contiguous()
        return torch.tensor(v, dtype=torch.int32, device=device).contiguous()

    cu_q_meta       = _to_u32_tensor(cu_q)
    cu_kv_meta      = _to_u32_tensor(cu_kv)
    seqused_q_meta  = _to_u32_tensor(seqused_q)
    seqused_kv_meta = _to_u32_tensor(seqused_kv)

    if layout_q == "BNSD":
        batch_size    = q1.shape[0]
        num_heads_q   = q1.shape[1]
        max_seqlen_q  = q1.shape[2]
        head_dim      = q1.shape[3]
        num_heads_kv  = k1.shape[1]
        max_seqlen_kv = k1.shape[2]
    elif layout_q == "BSND":
        batch_size    = q1.shape[0]
        num_heads_q   = q1.shape[2]
        max_seqlen_q  = q1.shape[1]
        head_dim      = q1.shape[3]
        num_heads_kv  = k1.shape[2]
        max_seqlen_kv = k1.shape[1]
    elif layout_q == "TND":
        if cu_q_meta is not None and hasattr(cu_q_meta, 'shape'):
            batch_size = cu_q_meta.shape[0] - 1
        elif seqused_q_meta is not None and hasattr(seqused_q_meta, 'shape'):
            batch_size = seqused_q_meta.shape[0]
        else:
            batch_size = b
        num_heads_q   = q1.shape[1]
        head_dim      = q1.shape[2]
        num_heads_kv  = k1.shape[1]
        max_seqlen_q  = sq  if sq  is not None else q1.shape[0]
        max_seqlen_kv = skv if skv is not None else k1.shape[0]
    else:
        batch_size    = q1.shape[0]
        num_heads_q   = n1
        num_heads_kv  = n2
        head_dim      = d
        max_seqlen_q  = q1.shape[1]
        max_seqlen_kv = k1.shape[1]

    block_table_ = None
    if layout_kv in ["PA_BBND", "PA_BNBD", "PA_NZ"]:
        block_table_  = block_table.to(device)
        num_heads_kv  = kwargs.get("N2")
        head_dim      = d
        max_seqlen_kv = max(seqused_kv)

    _print_args("Call npu_flash_attn_metadata (graph mode)", {
        "cu_seqlens_q": cu_q_meta,
        "cu_seqlens_kv": cu_kv_meta,
        "seqused_q": seqused_q_meta,
        "seqused_kv": seqused_kv_meta,
        "num_heads_q": num_heads_q,
        "num_heads_kv": num_heads_kv,
        "head_dim": head_dim,
        "batch_size": batch_size,
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_kv": max_seqlen_kv,
        "mask_mode": sparse_mode,
        "win_left": pre_tokens,
        "win_right": next_tokens,
        "layout_q": layout_q,
        "layout_kv": layout_kv,
        "layout_out": layout_out,
    })

    metadata = torch.ops.npu_ops_transformer.npu_flash_attn_metadata(
        cu_seqlens_q  = cu_q_meta,
        cu_seqlens_kv = cu_kv_meta,
        seqused_q     = seqused_q_meta,
        seqused_kv    = seqused_kv_meta,
        num_heads_q   = num_heads_q,
        num_heads_kv  = num_heads_kv,
        head_dim      = head_dim,
        batch_size    = batch_size,
        max_seqlen_q  = max_seqlen_q,
        max_seqlen_kv = max_seqlen_kv,
        mask_mode     = sparse_mode,
        win_left      = pre_tokens,
        win_right     = next_tokens,
        layout_q      = layout_q,
        layout_kv     = layout_kv,
        layout_out    = layout_out,
    )
    print_metadata(metadata)

    max_seqlen_q  = -1
    max_seqlen_kv = -1
    atten_mask1 = atten_mask1 if sparse_mode != 0 else None
    if atten_mask1 is not None and sparse_mode in (3, 4):
        atten_mask1 = atten_mask1.to(dtype=torch.int8)

    _print_args("Call npu_flash_attn (graph mode)", {
        "q": q1,
        "k": k1,
        "v": v1,
        "block_table": block_table_,
        "attn_mask": atten_mask1,
        "metadata": metadata,
        "cu_seqlens_q": cu_q_meta,
        "cu_seqlens_kv": cu_kv_meta,
        "seqused_q": seqused_q_meta,
        "seqused_kv": seqused_kv_meta,
        "softmax_scale": scale,
        "mask_mode": sparse_mode,
        "win_left": pre_tokens,
        "win_right": next_tokens,
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_kv": max_seqlen_kv,
        "layout_q": layout_q,
        "layout_kv": layout_kv,
        "layout_out": layout_out,
    })

    torch._dynamo.reset()
    npu_mode = FlashAttnGraphNetwork().npu()
    config = CompilerConfig()
    config.mode = "reduce-overhead"
    config.experimental_config.aclgraph._aclnn_static_shape_kernel = True
    config.experimental_config.aclgraph._aclnn_static_shape_kernel_build_dir = "./"
    config.experimental_config.frozen_parameter = True
    config.experimental_config.tiling_schedule_optimize = True
    config.experimental_config.topology_sorting_strategy = "StableRDFS"
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    npu_mode = torch.compile(npu_mode, fullgraph=False, backend=npu_backend, dynamic=False)

    out, lse_out = npu_mode(
        q1, k1, v1,
        block_table_,
        atten_mask1,
        metadata,
        cu_q_meta,
        cu_kv_meta,
        seqused_q_meta,
        seqused_kv_meta,
        scale,
        sparse_mode,
        pre_tokens,
        next_tokens,
        max_seqlen_q,
        max_seqlen_kv,
        layout_q,
        layout_kv,
        layout_out,
        lse_flag,
        0,
    )

    torch.npu.synchronize()
    return out.cpu(), lse_out.cpu()