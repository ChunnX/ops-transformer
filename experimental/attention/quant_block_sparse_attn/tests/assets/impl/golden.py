#!/usr/bin/python
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
PYTEST_MXFP8_GOLDEN_MODULE = None


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


def load_pytest_mxfp8_golden_module():
    global PYTEST_MXFP8_GOLDEN_MODULE
    if PYTEST_MXFP8_GOLDEN_MODULE is not None:
        return PYTEST_MXFP8_GOLDEN_MODULE
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
    PYTEST_MXFP8_GOLDEN_MODULE = module
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


def infer_geometry(
    query,
    key,
    sparse_indices,
    layout_q,
    cu_seqlens_q,
    cu_seqlens_kv,
    seqused_q,
    seqused_kv,
    max_seqlen_q=0,
    max_seqlen_kv=0,
):
    D = int(query.shape[-1])
    N2 = int(key.shape[1]) if key.dim() >= 2 else 1

    if layout_q == "BSND":
        B, S1, N1 = int(query.shape[0]), int(query.shape[1]), int(query.shape[2])
        q_lengths = to_list(seqused_q) or [S1] * B
    else:
        N1 = int(query.shape[1]) if layout_q == "TND" else int(query.shape[0])
        T = int(query.shape[0]) if layout_q == "TND" else int(query.shape[1])
        B = int(sparse_indices.shape[0]) if sparse_indices is not None else 1
        S1 = max_seqlen_q if max_seqlen_q > 0 else (T // B if B > 0 else T)
        q_lengths = to_list(seqused_q) or lengths_from_prefix(cu_seqlens_q) or [S1] * B

    kv_lengths = to_list(seqused_kv) or lengths_from_prefix(cu_seqlens_kv)
    if not kv_lengths:
        if max_seqlen_kv > 0:
            S2 = max_seqlen_kv
        elif key.dim() == 4:
            num_blocks = int(key.shape[0])
            block_size = int(key.shape[2])
            S2 = num_blocks * block_size
        else:
            S2 = int(key.shape[0])
        kv_lengths = [S2] * B if B > 0 else [S2]

    return B, S1, N1, N2, D, q_lengths, kv_lengths


def _lengths_to_prefix(lengths):
    values = [0]
    for length in lengths:
        values.append(values[-1] + int(length))
    return torch.tensor(values, dtype=torch.int32)


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


def _cpu_mxfp8_golden(
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
    layout_kv="PA_BNSD",
    layout_q="BSND",
    layout_sparse_indices="B_N_Qb_Kb",
    layout_out="TND",
    mask_mode=3,
    return_softmax_lse=False,
    **kwargs,
):
    """CPU reference implementation wrapping the MXFP8 pytest golden."""
    golden_module = load_pytest_mxfp8_golden_module()

    quant_group_size = int(kwargs.get("quant_group_size", 32))
    sparse_count = int(kwargs.get("sparse_count", 0))
    sparse_pattern = str(kwargs.get("sparse_pattern", "sequential"))
    block_table_pattern = str(kwargs.get("block_table_pattern", "sequential"))
    p_scale_value = float(kwargs.get("p_scale_value", 1.0))
    seed = int(kwargs.get("seed", 0))
    s2_base_size = int(kwargs.get("s2_base_size", 512))

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
    N1 = int(q_cpu.shape[1])
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

    cu_seqlens_q = _lengths_to_prefix(q_lengths)

    num_blocks = int(key_cpu.shape[0])
    dense_key_bnsd = torch.zeros((B, N2, S2, D), dtype=key_cpu.dtype)
    dense_value_bnsd = torch.zeros((B, N2, S2, D), dtype=value_cpu.dtype)

    bt_pattern = (
        block_table_pattern
        if block_table_pattern in ("sequential", "reverse", "random")
        else "sequential"
    )
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

    k_scale_d = D // quant_group_size
    dense_k_scale_bnsd = torch.zeros((B, N2, S2, k_scale_d), dtype=torch.float32)
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

    v_scale_bs = block_size // quant_group_size
    v_scale_bs_packed = v_scale_bs // 2
    v_sg = math.ceil(S2 / quant_group_size)
    dense_v_scale_bnsd = torch.zeros((B, N2, v_sg, D), dtype=torch.float32)
    v_descale_unpacked = v_descale_cpu.reshape(num_blocks, N2, v_scale_bs_packed, D, 2).float()
    for b in range(B):
        for lb in range(block_table_cpu.shape[1]):
            phys = int(block_table_cpu[b, lb].item())
            start = lb * v_scale_bs_packed
            end = min(start + v_scale_bs_packed, v_sg)
            cnt = end - start
            if cnt <= 0:
                continue
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

    q_descale_unpacked = q_descale_cpu.reshape(B * S1, N1, k_scale_d // 2, 2).float()
    q_descale_flat = q_descale_unpacked.reshape(B * S1, N1, k_scale_d)

    p_scale_fp32 = p_scale_cpu.float()

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
        "p_scale_value": p_scale_value,
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
        "quant_mode": 2,
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
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    block_table=None,
    metadata=None,
    max_seqlen_q=0,
    max_seqlen_kv=0,
    layout_kv="PA_BNSD",
    layout_q="BSND",
    layout_sparse_indices="B_N_Qb_Kb",
    layout_out="TND",
    quant_mode=1,
    mask_mode=3,
    return_softmax_lse=False,
    **kwargs,
):
    """CPU reference implementation wrapping the existing pytest golden."""
    if quant_mode == 2:
        return _cpu_mxfp8_golden(
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
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            seqused_q=seqused_q,
            seqused_kv=seqused_kv,
            block_table=block_table,
            metadata=metadata,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            layout_kv=layout_kv,
            layout_q=layout_q,
            layout_sparse_indices=layout_sparse_indices,
            layout_out=layout_out,
            mask_mode=mask_mode,
            return_softmax_lse=return_softmax_lse,
            **kwargs,
        )

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
    cu_seqlens_q_cpu = to_cpu(cu_seqlens_q) if cu_seqlens_q is not None else None
    cu_seqlens_kv_cpu = to_cpu(cu_seqlens_kv) if cu_seqlens_kv is not None else None
    seqused_q_cpu = to_cpu(seqused_q) if seqused_q is not None else None
    seqused_kv_cpu = to_cpu(seqused_kv) if seqused_kv is not None else None
    block_table_cpu = to_cpu(block_table) if block_table is not None else None

    B, S1, N1, N2, D, q_lengths, kv_lengths = infer_geometry(
        q_cpu,
        key_cpu,
        sparse_indices_cpu,
        layout_q,
        cu_seqlens_q_cpu,
        cu_seqlens_kv_cpu,
        seqused_q_cpu,
        seqused_kv_cpu,
        max_seqlen_q,
        max_seqlen_kv,
    )

    sparse_q_block_size = int(sparse_q_block_size)
    sparse_kv_block_size = int(sparse_kv_block_size)
    max_seqlen_q = int(max_seqlen_q) if max_seqlen_q > 0 else S1
    max_seqlen_kv = (
        int(max_seqlen_kv)
        if max_seqlen_kv > 0
        else max(kv_lengths)
        if kv_lengths
        else S1
    )
    S2 = max(kv_lengths) if kv_lengths else max_seqlen_kv

    if key_cpu.dim() == 4:
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
    else:
        dense_key = key_cpu
        dense_value = value_cpu

    if cu_seqlens_q_cpu is None and layout_q != "BSND":
        cu_seqlens_q_cpu = _lengths_to_prefix(q_lengths)
    if cu_seqlens_kv_cpu is None and layout_q != "BSND":
        cu_seqlens_kv_cpu = _lengths_to_prefix(kv_lengths)
    cu_seqlens_q_inp = cu_seqlens_q_cpu
    cu_seqlens_kv_inp = cu_seqlens_kv_cpu

    if k_descale_cpu.dim() == 3 and key_cpu.dim() == 4:
        k_scale_dense = _pa_to_dense(
            k_descale_cpu, block_table_cpu, B, S2, (1, 0), sparse_kv_block_size
        )
        if layout_q == "BSND":
            k_scale = k_scale_dense
        else:
            k_scale = k_scale_dense.reshape(B * S2, N2)
    elif k_descale_cpu.dim() == 1:
        k_scale = k_descale_cpu.unsqueeze(0).expand(B, -1, N2).contiguous()
        if layout_q != "BSND":
            k_scale = k_scale.reshape(B * S2, N2)
    else:
        k_scale = k_descale_cpu

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

    attention_out, softmax_lse = golden_module._reference_attention(
        case,
        q_cpu,
        dense_key,
        dense_value,
        q_descale_cpu,
        k_scale,
        v_scale,
        p_scale_cpu,
        sparse_indices_cpu,
        sparse_seq_len_cpu,
        cu_seqlens_q_inp,
        cu_seqlens_kv_inp,
        q_lengths,
        kv_lengths,
    )

    if return_softmax_lse:
        return [attention_out, softmax_lse]
    return [attention_out]
