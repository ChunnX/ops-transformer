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

"""Input customization for quant_block_sparse_attn TTK cases."""

import hashlib
import math
import random

import torch

FP8_DTYPE = torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else None


def _fp8_rand(shape, low=-1.0, high=1.0, generator=None):
    return torch.empty(shape, dtype=torch.float32).uniform_(low, high, generator=generator).to(FP8_DTYPE)


def _make_lengths(batch, max_len, mode="full", rng=None):
    if mode == "full":
        return [max_len] * batch, max_len
    low = max(1, max_len // 2)
    return [rng.randint(low, max_len) for _ in range(batch)], max_len


def _prefix(lengths):
    values = [0]
    for length in lengths:
        values.append(values[-1] + int(length))
    return torch.tensor(values, dtype=torch.int32)


def _make_block_table(batch, seq_len, block_size, rng):
    block_num_per_batch = math.ceil(seq_len / block_size)
    block_table = torch.empty((batch, block_num_per_batch), dtype=torch.int32)
    for batch_idx in range(batch):
        ids = list(range(batch_idx * block_num_per_batch, (batch_idx + 1) * block_num_per_batch))
        block_table[batch_idx] = torch.tensor(ids, dtype=torch.int32)
    return block_table


def _make_sparse_indices(B, N1, S1, S2, sparse_q_block_size, sparse_kv_block_size,
                          mask_mode, sparse_pattern, sparse_count, rng):
    qb_max = math.ceil(S1 / sparse_q_block_size)
    kv_max = math.ceil(S2 / sparse_kv_block_size)
    sparse_indices = torch.full((B, N1, qb_max, kv_max), fill_value=-1, dtype=torch.int32)
    sparse_seq_len = torch.zeros((B, N1, qb_max), dtype=torch.int32)

    for b in range(B):
        for n in range(N1):
            for qb in range(qb_max):
                allowed = list(range(kv_max))
                if mask_mode == 3:
                    max_block = min(kv_max - 1, ((qb + 1) * sparse_q_block_size - 1) // sparse_kv_block_size)
                    allowed = list(range(max_block + 1))
                if sparse_pattern == "sequential":
                    selected = allowed[:min(sparse_count, len(allowed))]
                elif sparse_pattern == "random":
                    selected = rng.sample(allowed, min(sparse_count, len(allowed)))
                else:
                    selected = allowed[:min(sparse_count, len(allowed))]
                sparse_seq_len[b, n, qb] = len(selected)
                if selected:
                    sparse_indices[b, n, qb, :len(selected)] = torch.tensor(selected, dtype=torch.int32)
    return sparse_indices, sparse_seq_len


def _dense_to_pa(dense_key, dense_value, dense_k_scale, block_table, block_size):
    B, S2, N2, D = dense_key.shape
    block_num = int(block_table.max().item()) + 1
    key_pa = torch.zeros((block_num, N2, block_size, D), dtype=dense_key.dtype)
    value_pa = torch.zeros((block_num, N2, block_size, D), dtype=dense_value.dtype)
    k_scale_pa = torch.zeros((block_num, N2, block_size), dtype=torch.float32)

    for b in range(B):
        for logical_block in range(block_table.shape[1]):
            physical = int(block_table[b, logical_block].item())
            start = logical_block * block_size
            end = min(start + block_size, S2)
            token_count = end - start
            if token_count <= 0:
                continue
            key_pa[physical, :, :token_count] = dense_key[b, start:end].permute(1, 0, 2)
            value_pa[physical, :, :token_count] = dense_value[b, start:end].permute(1, 0, 2)
            k_scale_pa[physical, :, :token_count] = dense_k_scale[b, start:end].permute(1, 0)
    return key_pa, value_pa, k_scale_pa


def generate_bsa_inputs(query, key, value, q_descale, k_descale, v_descale, p_scale,
                          sparse_indices, sparse_seq_len, atten_mask,
                          softmax_scale, sparse_q_block_size, sparse_kv_block_size,
                          cu_seqlens_q=None, cu_seqlens_kv=None, seqused_q=None, seqused_kv=None,
                          block_table=None, metadata=None, max_seqlen_q=0, max_seqlen_kv=0,
                          pa_block_stride=0, layout_kv="PA_BNSD", layout_q="BSND",
                          layout_sparse_indices="B_N_Qb_Kb", layout_out="TND",
                          quant_mode=1, mask_mode=3, return_softmax_lse=False,
                          testcase_name="", **kwargs):
    """Generate proper test data for quant_block_sparse_attn.

    Fills metadata tensors and creates consistent sparse indices and block tables.
    """
    if FP8_DTYPE is None:
        raise RuntimeError("torch.float8_e4m3fn is required for quant_block_sparse_attn test data")

    seed = int(hashlib.sha256(str(testcase_name).encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    generator = torch.Generator().manual_seed(seed)

    if query.dim() == 4:
        B, S1, N1, D = query.shape
        N2 = value.shape[2]
    elif query.dim() == 3:
        if layout_q == "NTD":
            N1, _, D = query.shape
        else:
            _, N1, D = query.shape
        B = seqused_q.shape[0] if seqused_q is not None else 2
        S1 = max_seqlen_q if max_seqlen_q > 0 else 256
    else:
        B, S1, N1, D = 2, 256, 8, 128
        N2 = 2

    S2 = max_seqlen_kv if max_seqlen_kv > 0 else 512
    sparse_q_block_size = int(sparse_q_block_size)
    sparse_kv_block_size = int(sparse_kv_block_size)

    query = _fp8_rand((B, S1, N1, D), generator=generator)
    dense_key = _fp8_rand((B, S2, N2, D), generator=generator)
    dense_value = _fp8_rand((B, S2, N2, D), generator=generator)

    q_lengths, q_max = _make_lengths(B, S1, "full", rng)
    kv_lengths, kv_max = _make_lengths(B, S2, "full", rng)
    cu_seqlens_q = _prefix(q_lengths)
    cu_seqlens_kv = _prefix(kv_lengths)
    seqused_q = torch.tensor(q_lengths, dtype=torch.int32)
    seqused_kv = torch.tensor(kv_lengths, dtype=torch.int32)

    block_table = _make_block_table(B, S2, sparse_kv_block_size, rng)
    key_pa, value_pa, k_scale_pa = _dense_to_pa(
        dense_key, dense_value,
        torch.ones((B, S2, N2), dtype=torch.float32),
        block_table, sparse_kv_block_size)

    q_scale = torch.ones((B, S1, N1), dtype=torch.float32)
    k_scale = k_scale_pa
    v_scale = torch.ones((N2,), dtype=torch.float32)
    p_scale = torch.tensor([1.0], dtype=torch.float32)

    sparse_indices, sparse_seq_len = _make_sparse_indices(
        B, N1, S1, S2, sparse_q_block_size, sparse_kv_block_size,
        mask_mode, "sequential", 4, rng)

    atten_mask = torch.tril(torch.ones((2048, 2048), dtype=torch.uint8)).T

    return (query, key_pa, value_pa, q_scale, k_scale, v_scale, p_scale,
            sparse_indices, sparse_seq_len, atten_mask)
