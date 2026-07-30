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

"""CPU golden adapter for MXFP8 quant_block_sparse_attn TTK cases."""

import importlib.util
import math
import sys
from pathlib import Path

import torch

PYTEST_GOLDEN_MODULE = None


def load_pytest_mxfp8_golden_module():
    global PYTEST_GOLDEN_MODULE
    if PYTEST_GOLDEN_MODULE is not None:
        return PYTEST_GOLDEN_MODULE
    pytest_dir = (
        Path(__file__).resolve().parents[2] / "pytest" / "bsa_fullquant_mxfp8_test"
    )
    module_path = pytest_dir / "qbsa_mxfp8_golden.py"
    sys.path.insert(0, str(pytest_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            f"bsa_mxfp8_pytest_golden_{abs(hash(module_path))}", module_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(pytest_dir))
        except ValueError:
            pass
    PYTEST_GOLDEN_MODULE = module
    return module


def to_cpu(value):
    return value.detach().cpu() if torch.is_tensor(value) else value


def cpu_mxfp8_quant_block_sparse_attn(
    query,
    key,
    value,
    q_descale,
    k_descale,
    v_descale,
    p_scale,
    sparse_indices,
    sparse_seq_len,
    atten_mask,
    softmax_scale,
    sparse_q_block_size,
    sparse_kv_block_size,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    block_table=None,
    metadata=None,
    max_seqlen_q=0,
    max_seqlen_kv=0,
    pa_block_stride=0,
    layout_kv="PA_BNSD",
    layout_q="TND",
    layout_sparse_indices="B_N_Qb_Kb",
    layout_out="TND",
    quant_mode=2,
    mask_mode=3,
    return_softmax_lse=False,
    sparse_count=0,
    sparse_pattern="sequential",
    block_table_pattern="sequential",
    p_scale_value=1.0,
    seed=0,
    quant_group_size=32,
    s2_base_size=512,
    **kwargs,
):
    """CPU reference implementation wrapping the MXFP8 pytest golden."""
    golden_module = load_pytest_mxfp8_golden_module()

    q_cpu = to_cpu(query)
    key_cpu = to_cpu(key)
    value_cpu = to_cpu(value)
    q_descale_cpu = to_cpu(q_descale)
    k_descale_cpu = to_cpu(k_descale)
    v_descale_cpu = to_cpu(v_descale)
    p_scale_cpu = to_cpu(p_scale)
    sparse_indices_cpu = to_cpu(sparse_indices)
    sparse_seq_len_cpu = to_cpu(sparse_seq_len)

    B = int(sparse_indices_cpu.shape[0])
    N1 = int(q_cpu.shape[1])  # TND
    N2 = int(key_cpu.shape[1])
    D = int(q_cpu.shape[-1])
    S1 = int(max_seqlen_q) if max_seqlen_q > 0 else B * int(q_cpu.shape[0]) // B
    S2 = (
        int(max_seqlen_kv)
        if max_seqlen_kv > 0
        else int(key_cpu.shape[0]) * int(key_cpu.shape[2])
    )
    block_size = int(sparse_q_block_size)

    q_lengths = [S1] * B
    kv_lengths = [S2] * B

    def _prefix(lengths):
        values = [0]
        for length in lengths:
            values.append(values[-1] + int(length))
        return torch.tensor(values, dtype=torch.int32)

    cu_seqlens_q = _prefix(q_lengths)
    cu_seqlens_kv = _prefix(kv_lengths)

    # Reconstruct dense K/V from PA format for golden
    # key_cpu: (num_blocks, N2, block_size, D) PA_BNSD
    # Need to convert to BNSD for golden: (B, N2, S2, D)
    num_blocks = int(key_cpu.shape[0])
    dense_key_bnsd = torch.zeros((B, N2, S2, D), dtype=key_cpu.dtype)
    dense_value_bnsd = torch.zeros((B, N2, S2, D), dtype=value_cpu.dtype)

    # Sequential block table
    bt_pattern = (
        block_table_pattern
        if block_table_pattern in ("sequential", "reverse", "random")
        else "sequential"
    )
    import random

    bt_rng = random.Random(seed)
    block_table_cpu = golden_module._make_reference_block_table(
        B, kv_lengths, block_size, bt_pattern, bt_rng
    )

    for b in range(B):
        for lb in range(block_table_cpu.shape[1]):
            phys = int(block_table_cpu[b, lb].item())
            start = lb * block_size
            end = min(start + block_size, S2)
            cnt = end - start
            if cnt <= 0:
                continue
            dense_key_bnsd[b, :, :cnt] = key_cpu[phys, :, :cnt]
            dense_value_bnsd[b, :, :cnt] = value_cpu[phys, :, :cnt]

    # K descale: (num_blocks, N2, block_size, D//gs//2, 2) -> BNSD (B, N2, S2, D//gs)
    k_scale_d = D // quant_group_size
    dense_k_scale_bnsd = torch.zeros((B, N2, S2, k_scale_d), dtype=torch.float32)
    # Unpack k_descale from (num_blocks, N2, bs, Dg//2, 2) to (num_blocks, N2, bs, Dg)
    k_descale_unpacked = k_descale_cpu.reshape(
        num_blocks, N2, block_size, k_scale_d // 2, 2
    )
    k_descale_flat = k_descale_unpacked.reshape(
        num_blocks, N2, block_size, k_scale_d
    ).float()
    for b in range(B):
        for lb in range(block_table_cpu.shape[1]):
            phys = int(block_table_cpu[b, lb].item())
            start = lb * block_size
            end = min(start + block_size, S2)
            cnt = end - start
            if cnt <= 0:
                continue
            dense_k_scale_bnsd[b, :, :cnt] = k_descale_flat[phys, :, :cnt]

    # V descale: (num_blocks, N2, bs//gs, D, 2) -> BNSD (B, N2, S2//gs, D)
    v_scale_bs = block_size // quant_group_size
    v_sg = math.ceil(S2 / quant_group_size)
    dense_v_scale_bnsd = torch.zeros((B, N2, v_sg, D), dtype=torch.float32)
    v_descale_unpacked = v_descale_cpu.reshape(num_blocks, N2, v_scale_bs, D, 2).float()
    for b in range(B):
        for lb in range(block_table_cpu.shape[1]):
            phys = int(block_table_cpu[b, lb].item())
            start = lb * v_scale_bs
            end = min(start + v_scale_bs, v_sg)
            cnt = end - start
            if cnt <= 0:
                continue
            # Unpack pair: (cnt, D, 2) -> (cnt*2, D)
            for i in range(cnt):
                src_idx = i
                dst_idx1 = start * 2 + i * 2
                dst_idx2 = start * 2 + i * 2 + 1
                if dst_idx1 < v_sg:
                    dense_v_scale_bnsd[b, :, dst_idx1, :] = v_descale_unpacked[
                        phys, :, src_idx, :, 0
                    ]
                if dst_idx2 < v_sg:
                    dense_v_scale_bnsd[b, :, dst_idx2, :] = v_descale_unpacked[
                        phys, :, src_idx, :, 1
                    ]

    # Q descale: (total_q, N1, Dg//2, 2) -> (total_q, N1, Dg)
    q_descale_unpacked = q_descale_cpu.reshape(B * S1, N1, k_scale_d // 2, 2).float()
    q_descale_flat = q_descale_unpacked.reshape(B * S1, N1, k_scale_d)

    # P scale: e8m0 -> fp32
    p_scale_fp32 = p_scale_cpu.float()

    # Set up the golden module's global state
    case = {
        "B": B,
        "N1": N1,
        "N2": N2,
        "D": D,
        "S1": S1,
        "S2": S2,
        "actual_seq_q": q_lengths,
        "actual_seq_kv": kv_lengths,
        "block_size": block_size,
        "mask_mode": mask_mode,
        "fp8_dtype": torch.float8_e4m3fn,
        "scale_dtype": torch.float8_e8m0fnu,
        "quant_group_size": quant_group_size,
        "layout_q": layout_q,
        "layout_kv": layout_kv,
        "layout_sparse_indices": layout_sparse_indices,
        "layout_out": layout_out,
        "kv_cache_layout": "BnNBsD",
        "is_contiguous": True,
        "p_scale_value": float(p_scale_value),
        "softmax_scale": float(softmax_scale) if softmax_scale else 1.0 / math.sqrt(D),
        "return_softmax_lse": return_softmax_lse,
        "seed": seed,
        "data_range_q": 1.0,
        "data_range_k": 1.0,
        "data_range_v": 1.0,
        "device_id": 0,
        "sparse_q_block_size": sparse_q_block_size,
        "sparse_kv_block_size": sparse_kv_block_size,
        "sparse_count": sparse_count,
        "sparse_pattern": sparse_pattern,
        "block_table_pattern": block_table_pattern,
        "quant_mode": quant_mode,
        "s2_base_size": s2_base_size,
    }
    golden_module.set_active_case(case)

    attention_out, softmax_lse = golden_module.cpu_mxfp8_golden(
        q_cpu,
        dense_key_bnsd,
        dense_value_bnsd,
        q_descale_flat,
        dense_k_scale_bnsd,
        dense_v_scale_bnsd,
        p_scale_fp32,
        q_lengths,
        kv_lengths,
        cu_seqlens_q,
        sparse_indices_cpu,
        sparse_seq_len_cpu,
    )

    if return_softmax_lse:
        return [attention_out.to(torch.bfloat16), softmax_lse]
    return [attention_out.to(torch.bfloat16)]
