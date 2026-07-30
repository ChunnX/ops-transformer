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

"""Input customization for MXFP8 quant_block_sparse_attn TTK cases.

E2E customize_inputs contract: modify tensors in-place via x.copy_(value).
"""

import math
import random
import sys
from pathlib import Path

import torch

FP8_DTYPE = torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else None
E8M0_DTYPE = torch.float8_e8m0fnu if hasattr(torch, "float8_e8m0fnu") else None

E8M0_MIN_POSITIVE = 2 ** (-127)
_EMAX_MAP = {torch.float8_e4m3fn: 8}
QUANT_GROUP_SIZE = 32


def _rand_fp8(shape, generator, data_range=1.0):
    return (torch.rand(shape, dtype=torch.float32, generator=generator) * 2 - 1).to(
        FP8_DTYPE
    ) * data_range


def get_mxfp8_per_token_group_quant_scale(tensor, group_size=32):
    """Compute per-token-group quant scale for Q/K."""
    emax = _EMAX_MAP[FP8_DTYPE]
    tensor = tensor.to(torch.float32)
    dim1, dim2, dim3, dim4 = tensor.shape
    num_groups = math.ceil(dim4 / group_size)
    pad = num_groups * group_size - dim4
    if pad > 0:
        tensor = torch.nn.functional.pad(tensor, (0, pad))
    grouped = tensor.reshape(dim1, dim2, dim3, num_groups, group_size)
    all_zero = torch.all(grouped == 0, dim=-1)
    max_vals = torch.max(torch.abs(grouped), dim=-1)[0].clamp(min=1e-12)
    shared_exp = torch.floor(torch.log2(max_vals)) - emax
    return torch.where(all_zero, torch.ones_like(shared_exp), 2**shared_exp).to(
        torch.float32
    )


def get_mxfp8_per_channel_group_quant_scale(tensor, group_size=32):
    """Compute per-channel-group quant scale for V."""
    emax = _EMAX_MAP[FP8_DTYPE]
    tensor = tensor.to(torch.float32)
    batch, seq_len, num_heads, head_dim = tensor.shape
    num_groups = math.ceil(seq_len / group_size)
    pad = num_groups * group_size - seq_len
    if pad > 0:
        tensor = torch.nn.functional.pad(tensor, (0, 0, 0, 0, 0, pad))
    grouped = tensor.reshape(batch, num_groups, group_size, num_heads, head_dim)
    all_zero = torch.all(grouped == 0, dim=2)
    max_vals = torch.max(torch.abs(grouped), dim=2)[0].clamp(min=1e-12)
    shared_exp = torch.floor(torch.log2(max_vals)) - emax
    return torch.where(all_zero, torch.ones_like(shared_exp), 2**shared_exp).to(
        torch.float32
    )


def _sanitize_e8m0_scale(scale):
    result = torch.as_tensor(scale, dtype=torch.float32).clone()
    result[~torch.isfinite(result)] = E8M0_MIN_POSITIVE
    result[result == 0] = E8M0_MIN_POSITIVE
    return result


def _fp32_to_e8m0(tensor_fp32):
    bits = tensor_fp32.float().view(torch.int32)
    return ((bits >> 23) & 0xFF).to(torch.uint8).view(E8M0_DTYPE)


def _fp32_to_e8m0_safe(scale):
    return _fp32_to_e8m0(_sanitize_e8m0_scale(scale))


def _pack_qk_scale(scale_flat):
    """Pack Q/K D-group descale: (..., Dg) -> (..., Dg//2, 2)."""
    last_dim = scale_flat.shape[-1]
    new_shape = scale_flat.shape[:-1] + (last_dim // 2, 2)
    return scale_flat.reshape(new_shape)


def _make_block_table(batch, seq_len, block_size, pattern, rng):
    block_num = math.ceil(seq_len / block_size)
    block_table = torch.empty((batch, block_num), dtype=torch.int32)
    for b in range(batch):
        ids = list(range(b * block_num, (b + 1) * block_num))
        if pattern == "reverse":
            ids.reverse()
        elif pattern == "random":
            rng.shuffle(ids)
        block_table[b] = torch.tensor(ids, dtype=torch.int32)
    return block_table


def _allowed_blocks(mask_mode, qb_idx, sqbs, skvbs, q_len, kv_len):
    block_num = math.ceil(kv_len / skvbs)
    if block_num <= 0:
        return []
    if mask_mode == 0:
        return list(range(block_num))
    if mask_mode != 3:
        raise ValueError(f"unsupported mask_mode: {mask_mode}")
    max_token = (qb_idx + 1) * sqbs - 1 + kv_len - q_len
    if max_token < 0:
        return []
    max_block = min(block_num - 1, max_token // skvbs)
    return list(range(max_block + 1))


def _select_blocks(blocks, count, pattern, rng):
    if count <= 0 or not blocks or pattern == "empty":
        return []
    if pattern in ("sequential", "dense", "causal"):
        return blocks[: min(count, len(blocks))]
    if pattern == "reverse":
        return list(reversed(blocks[-min(count, len(blocks)) :]))
    if pattern == "tail":
        selected = blocks[: max(0, min(count, len(blocks)) - 1)]
        if blocks[-1] not in selected:
            selected.append(blocks[-1])
        return selected[:count]
    if pattern == "random":
        selected = blocks[:]
        rng.shuffle(selected)
        return selected[: min(count, len(selected))]
    raise ValueError(f"unsupported sparse_pattern: {pattern}")


def _make_sparse_indices(B, N1, S1, S2, sqbs, skvbs, mask_mode, pattern, count, rng):
    qb_max = math.ceil(S1 / sqbs)
    kv_max = math.ceil(S2 / skvbs)
    indices = torch.full((B, N1, qb_max, kv_max), -1, dtype=torch.int32)
    seq_len = torch.zeros((B, N1, qb_max), dtype=torch.int32)
    for b in range(B):
        for qb in range(qb_max):
            allowed = _allowed_blocks(mask_mode, qb, sqbs, skvbs, S1, S2)
            for n in range(N1):
                selected = _select_blocks(allowed, count, pattern, rng)
                seq_len[b, n, qb] = len(selected)
                if selected:
                    indices[b, n, qb, : len(selected)] = torch.tensor(
                        selected, dtype=torch.int32
                    )
    return indices, seq_len


def _dense_to_pa(dense, block_table, block_size):
    """Convert dense BNSD to PA (num_blocks, N, block_size, D)."""
    B, S, N, D = dense.shape
    num_blocks = int(block_table.max().item()) + 1
    pa = torch.zeros((num_blocks, N, block_size, D), dtype=dense.dtype)
    for b in range(B):
        for lb in range(block_table.shape[1]):
            phys = int(block_table[b, lb].item())
            start = lb * block_size
            end = min(start + block_size, S)
            cnt = end - start
            if cnt <= 0:
                continue
            pa[phys, :, :cnt] = dense[b, start:end].permute(1, 0, 2)
    return pa


def customize_inputs(
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
    *,
    softmax_scale=1.0,
    sparse_q_block_size=128,
    sparse_kv_block_size=128,
    max_seqlen_q=0,
    max_seqlen_kv=0,
    layout_q="TND",
    layout_kv="PA_BNSD",
    mask_mode=3,
    quant_mode=2,
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
    """Fill input tensors in-place with MXFP8 test data."""
    if FP8_DTYPE is None or E8M0_DTYPE is None:
        raise RuntimeError(
            "float8_e4m3fn and float8_e8m0fnu are required for MXFP8 test data"
        )

    rng = random.Random(seed)
    generator = torch.Generator().manual_seed(seed)

    B = int(sparse_indices.shape[0])
    N1 = int(query.shape[1])  # TND only
    N2 = int(key.shape[1])
    D = int(query.shape[-1])
    S1 = max_seqlen_q if max_seqlen_q > 0 else B * int(query.shape[0]) // B
    S2 = max_seqlen_kv if max_seqlen_kv > 0 else int(key.shape[0]) * int(key.shape[2])
    block_size = sparse_q_block_size

    total_q = B * S1

    # Generate Q data and scale
    q_source = _rand_fp8((1, total_q, N1, D), generator, 1.0).squeeze(
        0
    )  # (total_q, N1, D) TND
    query.copy_(q_source)

    q_scale = get_mxfp8_per_token_group_quant_scale(
        q_source.unsqueeze(0), quant_group_size
    ).squeeze(0)
    q_scale_packed = _pack_qk_scale(q_scale)
    q_descale.copy_(_fp32_to_e8m0_safe(q_scale_packed))

    # Generate K/V data
    dense_key = (
        _rand_fp8((B, S2, N2, D), generator, 1.0).permute(0, 2, 1, 3).contiguous()
    )  # BNSD
    dense_value = (
        _rand_fp8((B, S2, N2, D), generator, 1.0).permute(0, 2, 1, 3).contiguous()
    )

    k_scale = get_mxfp8_per_token_group_quant_scale(dense_key, quant_group_size)
    v_scale = get_mxfp8_per_channel_group_quant_scale(dense_value, quant_group_size)

    block_table = _make_block_table(B, S2, block_size, "sequential", rng)

    key_pa = _dense_to_pa(dense_key, block_table, block_size)
    value_pa = _dense_to_pa(dense_value, block_table, block_size)
    key.copy_(key_pa)
    value.copy_(value_pa)

    # K descale: BNSD -> PA, then pack
    k_scale_pa = _dense_to_pa(
        k_scale.permute(0, 2, 1, 3).contiguous(), block_table, block_size
    )
    k_scale_packed = _pack_qk_scale(k_scale_pa)
    k_descale.copy_(_fp32_to_e8m0_safe(k_scale_packed))

    # V descale: per-channel-group, shape (B, N2, S2//gs, D) -> PA (num_blocks, N2, bs//gs, D, 2)
    v_scale_bnsd = v_scale.permute(0, 2, 1, 3).contiguous()  # (B, N2, S2//gs, D)
    v_scale_pa = torch.zeros(
        (int(block_table.max().item()) + 1, N2, block_size // quant_group_size, D),
        dtype=torch.float32,
    )
    for b in range(B):
        for lb in range(block_table.shape[1]):
            phys = int(block_table[b, lb].item())
            start = lb * (block_size // quant_group_size)
            end = min(start + block_size // quant_group_size, v_scale_bnsd.shape[2])
            cnt = end - start
            if cnt <= 0:
                continue
            v_scale_pa[phys, :, :cnt] = v_scale_bnsd[b, :, start:end]

    # V scale pair packing: (..., Sg, D) -> (..., Sg//2, D, 2)
    v_sg = v_scale_pa.shape[2]
    if v_sg % 2 != 0:
        v_scale_pa = torch.cat(
            [v_scale_pa, torch.full_like(v_scale_pa[:, :, :1, :], E8M0_MIN_POSITIVE)],
            dim=2,
        )
        v_sg += 1
    v_scale_packed = torch.zeros(
        (v_scale_pa.shape[0], N2, v_sg // 2, D, 2), dtype=torch.float32
    )
    v_scale_packed[..., 0] = v_scale_pa[..., ::2, :]
    v_scale_packed[..., 1] = v_scale_pa[..., 1::2, :]
    v_descale.copy_(_fp32_to_e8m0_safe(v_scale_packed))

    # P scale
    p_scale_data = torch.tensor([float(p_scale_value)], dtype=torch.float32)
    p_scale.copy_(_fp32_to_e8m0_safe(p_scale_data))

    # Sparse indices
    si, ssl = _make_sparse_indices(
        B,
        N1,
        S1,
        S2,
        sparse_q_block_size,
        sparse_kv_block_size,
        mask_mode,
        sparse_pattern,
        sparse_count,
        rng,
    )
    sparse_indices.copy_(si)
    sparse_seq_len.copy_(ssl)

    # Atten mask
    atten_mask_data = torch.tril(torch.ones(atten_mask.shape, dtype=torch.uint8)).T
    atten_mask.copy_(atten_mask_data)
