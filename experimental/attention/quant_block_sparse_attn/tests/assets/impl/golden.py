#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CPU golden adapter for quant_block_sparse_attn TTK cases."""

import importlib.util
import sys
from pathlib import Path

import torch

PYTEST_GOLDEN_MODULE = None


def load_pytest_golden_module():
    global PYTEST_GOLDEN_MODULE
    if PYTEST_GOLDEN_MODULE is not None:
        return PYTEST_GOLDEN_MODULE
    pytest_dir = Path(__file__).resolve().parents[2] / "pytest"
    module_path = pytest_dir / "quant_block_sparse_attn_golden.py"
    sys.path.insert(0, str(pytest_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            f"bsa_pytest_golden_{abs(hash(module_path))}", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(pytest_dir))
        except ValueError:
            pass
    PYTEST_GOLDEN_MODULE = module
    return module


def to_list(value):
    if value is None:
        return []
    if torch.is_tensor(value):
        return [int(x) for x in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    return [int(value)]


def to_cpu(value):
    return value.detach().cpu() if torch.is_tensor(value) else value


def lengths_from_prefix(value):
    vals = to_list(value)
    return [vals[i + 1] - vals[i] for i in range(len(vals) - 1)] if len(vals) > 1 else []


def infer_geometry(query, key, layout_q, cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv):
    if layout_q == "BSND":
        B, S1, N1, D = [int(x) for x in query.shape]
        N2 = key.shape[2] if key.dim() == 4 else key.shape[1]
        q_lengths = to_list(seqused_q) or [S1] * B
    else:
        T, N1, D = [int(x) for x in query.shape]
        # PA layout: key shape is (num_blocks, N2, block_size, D)
        # TND layout: key shape is (total_tokens, N2, D)
        N2 = key.shape[1]
        q_lengths = to_list(seqused_q) or lengths_from_prefix(cu_seqlens_q)
        B = len(q_lengths)
        S1 = max(q_lengths) if q_lengths else T

    kv_lengths = to_list(seqused_kv) or lengths_from_prefix(cu_seqlens_kv)
    if not kv_lengths:
        if key.dim() == 4:
            # PA layout: (num_blocks, N2, block_size, D) -> total_kv_len = num_blocks * block_size
            num_blocks = key.shape[0]
            block_size = key.shape[2]
            S2 = num_blocks * block_size
        else:
            # TND layout: (total_tokens, N2, D)
            S2 = key.shape[0]
        kv_lengths = [S2] * B if B > 0 else [S2]

    return B, S1, N1, N2, D, q_lengths, kv_lengths


def cpu_quant_block_sparse_attn(query, key, value, q_descale, k_descale, v_descale, p_scale,
                                  sparse_indices, sparse_seq_len, atten_mask,
                                  softmax_scale, sparse_q_block_size, sparse_kv_block_size,
                                  cu_seqlens_q=None, cu_seqlens_kv=None, seqused_q=None, seqused_kv=None,
                                  block_table=None, metadata=None, max_seqlen_q=0, max_seqlen_kv=0,
                                  pa_block_stride=0, layout_kv="PA_BNSD", layout_q="BSND",
                                  layout_sparse_indices="B_N_Qb_Kb", layout_out="TND",
                                  quant_mode=1, mask_mode=3, return_softmax_lse=False, **kwargs):
    """CPU reference implementation wrapping the existing pytest golden."""
    golden_module = load_pytest_golden_module()

    q_cpu = to_cpu(query)
    key_cpu = to_cpu(key)
    value_cpu = to_cpu(value)
    q_descale_cpu = to_cpu(q_descale)
    k_descale_cpu = to_cpu(k_descale)
    v_descale_cpu = to_cpu(v_descale)
    p_scale_cpu = to_cpu(p_scale)
    sparse_indices_cpu = to_cpu(sparse_indices)
    sparse_seq_len_cpu = to_cpu(sparse_seq_len)
    atten_mask_cpu = to_cpu(atten_mask)
    cu_seqlens_q_cpu = to_cpu(cu_seqlens_q) if cu_seqlens_q is not None else None
    cu_seqlens_kv_cpu = to_cpu(cu_seqlens_kv) if cu_seqlens_kv is not None else None
    seqused_q_cpu = to_cpu(seqused_q) if seqused_q is not None else None
    seqused_kv_cpu = to_cpu(seqused_kv) if seqused_kv is not None else None
    block_table_cpu = to_cpu(block_table) if block_table is not None else None

    B, S1, N1, N2, D, q_lengths, kv_lengths = infer_geometry(
        q_cpu, key_cpu, layout_q, cu_seqlens_q_cpu, cu_seqlens_kv_cpu, seqused_q_cpu, seqused_kv_cpu)

    sparse_q_block_size = int(sparse_q_block_size)
    sparse_kv_block_size = int(sparse_kv_block_size)
    max_seqlen_q = int(max_seqlen_q) if max_seqlen_q > 0 else S1
    max_seqlen_kv = int(max_seqlen_kv) if max_seqlen_kv > 0 else max(kv_lengths) if kv_lengths else S1

    if key_cpu.dim() == 4:
        dense_key = key_cpu.permute(0, 2, 1, 3).contiguous()
        dense_value = value_cpu.permute(0, 2, 1, 3).contiguous()
        cu_seqlens_q_inp = None
        cu_seqlens_kv_inp = None
    else:
        dense_key = key_cpu
        dense_value = value_cpu
        cu_seqlens_q_inp = cu_seqlens_q_cpu
        cu_seqlens_kv_inp = cu_seqlens_kv_cpu

    if k_descale_cpu.dim() == 1:
        k_scale = k_descale_cpu.unsqueeze(0).expand(B, -1, N2).contiguous()
    elif k_descale_cpu.dim() == 3:
        k_scale = k_descale_cpu
    else:
        k_scale = k_descale_cpu

    v_scale = v_descale_cpu if v_descale_cpu.dim() == 1 else v_descale_cpu.view(-1)

    case = {
        "B": B, "S1": S1, "S2": max(kv_lengths) if kv_lengths else S1,
        "N1": N1, "N2": N2, "D": D,
        "layout_q": layout_q,
        "softmax_scale": float(softmax_scale),
        "sparse_q_block_size": sparse_q_block_size,
        "sparse_kv_block_size": sparse_kv_block_size,
        "mask_mode": int(mask_mode),
        "output_dtype": torch.bfloat16,
    }

    attention_out, softmax_lse = golden_module._reference_attention(
        case, q_cpu, dense_key, dense_value,
        q_descale_cpu, k_scale, v_scale, p_scale_cpu,
        sparse_indices_cpu, sparse_seq_len_cpu,
        cu_seqlens_q_inp, cu_seqlens_kv_inp,
        q_lengths, kv_lengths)

    if not return_softmax_lse:
        softmax_lse = torch.empty(0, dtype=torch.float32)

    # TTK spec requires golden to return a list (one element per output)
    return [attention_out, softmax_lse]
