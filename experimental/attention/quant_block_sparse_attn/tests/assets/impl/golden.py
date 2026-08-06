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
import math
import random
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
            f"bsa_pytest_golden_{abs(hash(module_path))}", module_path
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
    return (
        [vals[i + 1] - vals[i] for i in range(len(vals) - 1)] if len(vals) > 1 else []
    )


def infer_geometry(query, key, sparse_indices, layout_q, cu_seqlens_q, seqused_kv):
    D = int(query.shape[-1])
    if key.dim() != 4:
        raise ValueError(f"key must be 4D PA [blockNum, kvHeadNum, blockSize, headDim], got {tuple(key.shape)}")
    N2 = int(key.shape[1])

    if layout_q == "NTD":
        N1 = int(query.shape[0])
    else:  # TND
        N1 = int(query.shape[1])
    B = int(sparse_indices.shape[0]) if sparse_indices is not None else 1
    # S1 由 cu_seqlens_q 前缀和做差得到
    q_lengths = lengths_from_prefix(cu_seqlens_q)
    S1 = max(q_lengths)

    # S2 取 seqused_kv 最大值
    kv_lengths = to_list(seqused_kv)
    S2 = max(kv_lengths)

    return B, S1, N1, N2, D, q_lengths, kv_lengths, S2


def _auto_block_table(B, S2, block_size, pattern="sequential", seed=0):
    num_blocks_per_batch = math.ceil(S2 / block_size)
    block_table = torch.empty((B, num_blocks_per_batch), dtype=torch.int32)
    rng = random.Random(seed)
    for batch_idx in range(B):
        ids = list(
            range(
                batch_idx * num_blocks_per_batch, (batch_idx + 1) * num_blocks_per_batch
            )
        )
        if pattern == "reverse":
            ids.reverse()
        elif pattern == "random":
            rng.shuffle(ids)
        block_table[batch_idx] = torch.tensor(ids, dtype=torch.int32)
    return block_table


def _pa_to_dense(pa_tensor, block_table, B, S2, dim_perm, block_size):
    n2 = int(pa_tensor.shape[1])
    if pa_tensor.dim() == 4:
        d = int(pa_tensor.shape[3])
        dense = torch.zeros((B, S2, n2, d), dtype=pa_tensor.dtype)
    else:
        dense = torch.zeros((B, S2, n2), dtype=pa_tensor.dtype)
    for b in range(B):
        for logical_block in range(block_table.shape[1]):
            physical = int(block_table[b, logical_block].item())
            start = logical_block * block_size
            end = min(start + block_size, S2)
            token_count = end - start
            if token_count <= 0:
                continue
            src = pa_tensor[physical, :, :token_count]
            dense[b, start:end] = src.permute(*dim_perm)
    return dense


def cpu_quant_block_sparse_attn(
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
    cu_seqlens_q,
    seqused_kv,
    block_table=None,
    metadata=None,
    layout_kv="PA_BNSD",
    layout_q="TND",
    layout_sparse_indices="B_N_Qb_Kb",
    layout_out="TND",
    quant_mode=1,
    mask_mode=3,
    return_softmax_lse=False,
    **kwargs,
):
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
    cu_seqlens_q_cpu = to_cpu(cu_seqlens_q)
    seqused_kv_cpu = to_cpu(seqused_kv)
    block_table_cpu = to_cpu(block_table) if block_table is not None else None

    B, S1, N1, N2, D, q_lengths, kv_lengths, S2 = infer_geometry(
        q_cpu,
        key_cpu,
        sparse_indices_cpu,
        layout_q,
        cu_seqlens_q_cpu,
        seqused_kv_cpu,
    )

    sparse_q_block_size = int(sparse_q_block_size)
    sparse_kv_block_size = int(sparse_kv_block_size)

    if block_table_cpu is None:
        block_table_cpu = _auto_block_table(
            B, S2, sparse_kv_block_size, "sequential", 0
        )
    dense_key = _pa_to_dense(
        key_cpu, block_table_cpu, B, S2, (1, 0, 2), sparse_kv_block_size
    )
    dense_value = _pa_to_dense(
        value_cpu, block_table_cpu, B, S2, (1, 0, 2), sparse_kv_block_size
    )

    cu_seqlens_q_inp = cu_seqlens_q_cpu

    if k_descale_cpu.dim() == 3:
        k_scale_dense = _pa_to_dense(
            k_descale_cpu, block_table_cpu, B, S2, (1, 0), sparse_kv_block_size
        )
    elif k_descale_cpu.dim() == 1:
        k_scale_dense = (
            k_descale_cpu.unsqueeze(0).unsqueeze(0).expand(B, S2, N2).contiguous()
        )
    else:
        k_scale_dense = k_descale_cpu

    v_scale = v_descale_cpu if v_descale_cpu.dim() == 1 else v_descale_cpu.view(-1)

    case = {
        "B": B,
        "S1": S1,
        "S2": S2,
        "N1": N1,
        "N2": N2,
        "D": D,
        "layout_q": layout_q,
        "softmax_scale": float(softmax_scale),
        "sparse_q_block_size": sparse_q_block_size,
        "sparse_kv_block_size": sparse_kv_block_size,
        "mask_mode": int(mask_mode),
        "output_dtype": torch.bfloat16,
        "actlen_mode": "full",
        "S1EQS2": False,
        "seed": int(kwargs.get("seed", 0)),
        "sparse_pattern": str(kwargs.get("sparse_pattern", "sequential")),
        "block_table_pattern": "sequential",
        "sparse_count": int(kwargs.get("sparse_count", 0)),
        "p_scale_value": float(p_scale_cpu[0].item()),
        "pa_block_padding_bytes": 0,
        "layout_kv": layout_kv,
        "layout_sparse_indices": layout_sparse_indices,
        "layout_out": layout_out,
        "quant_mode": int(quant_mode),
    }

    kv_cache_storage, kv_cache_meta = golden_module.combined_kv_cache.pack_combined_kv_cache(
        dense_key,
        dense_value,
        k_scale_dense,
        block_table_cpu,
        sparse_kv_block_size,
        0,
        layout_kv,
    )

    attention_out, softmax_lse = golden_module._reference_attention(
        case,
        q_cpu,
        kv_cache_storage,
        kv_cache_meta,
        block_table_cpu,
        q_descale_cpu,
        v_scale,
        p_scale_cpu,
        sparse_indices_cpu,
        sparse_seq_len_cpu,
        cu_seqlens_q_inp,
        q_lengths,
        kv_lengths,
    )

    if return_softmax_lse:
        return [attention_out, softmax_lse]
    return [attention_out]
