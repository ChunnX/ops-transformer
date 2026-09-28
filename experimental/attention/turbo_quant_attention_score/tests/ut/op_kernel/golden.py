# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Simplified CPU golden for TurboQuantAttentionScore (aligned with task golden.py)."""

import math
import numpy as np

BLOCK_SIZE = 128
HEAD_DIM = 128

CENTROIDS = {
    2: [-0.1335033178, -0.04002048075, 0.04002048075, 0.1335033178],
    3: [
        -0.19020693,
        -0.1187859178,
        -0.06682205945,
        -0.02166347019,
        0.02166347019,
        0.06682205945,
        0.1187859178,
        0.19020693,
    ],
    4: [
        -0.2414890379,
        -0.1828317791,
        -0.1429702938,
        -0.1109927073,
        -0.08325428516,
        -0.05802082643,
        -0.03428063914,
        -0.01134236995,
        0.01134236995,
        0.03428063914,
        0.05802082643,
        0.08325428516,
        0.1109927073,
        0.1429702938,
        0.1828317791,
        0.2414890379,
    ],
}


def _unpack_primary(packed, bits):
    values = packed.astype(np.int32)
    if bits == 3:
        triplets = values.reshape(*values.shape[:-1], 16, 3)
        words = triplets[..., 0] | (triplets[..., 1] << 8) | (triplets[..., 2] << 16)
        idxs = np.stack([(words >> s) & 7 for s in range(0, 24, 3)], -1)
        return idxs.reshape(*packed.shape[:-1], HEAD_DIM)
    mask = (1 << bits) - 1
    idxs = np.stack([(values >> s) & mask for s in range(0, 8, bits)], -1)
    return idxs.reshape(*packed.shape[:-1], HEAD_DIM)


def _unpack_signs(packed):
    values = packed.astype(np.int32)
    bits = np.stack([(values >> s) & 1 for s in range(8)], -1)
    return bits.reshape(*packed.shape[:-1], HEAD_DIM).astype(np.float32) * 2.0 - 1.0


def calc_expect_func(
    query_rotated,
    query_qjl,
    key_cache_idx,
    key_cache_qjl,
    key_cache_norm,
    key_cache_gamma,
    block_table,
    seq_lens,
    mse_bits,
):
    bits = int(mse_bits)
    query = np.asarray(query_rotated, dtype=np.float32)
    query_residual = np.asarray(query_qjl, dtype=np.float32)
    packed_idx = np.asarray(key_cache_idx, dtype=np.uint8)
    packed_qjl = np.asarray(key_cache_qjl, dtype=np.uint8)
    norm_cache = np.asarray(key_cache_norm, dtype=np.float32)
    gamma_cache = np.asarray(key_cache_gamma, dtype=np.float32)
    table = np.asarray(block_table, dtype=np.int64)
    lengths = np.asarray(seq_lens, dtype=np.int64)

    num_tokens, q_heads, _ = query.shape
    batch = table.shape[0]
    kv_heads = packed_idx.shape[2]
    groups = q_heads // kv_heads
    max_kv_len = int(lengths.max())
    centroids = np.asarray(CENTROIDS[bits], dtype=np.float32)
    coefficient = math.sqrt(math.pi / 2.0) / HEAD_DIM
    output = np.empty((num_tokens, q_heads, max_kv_len), dtype=np.float32)

    for t in range(num_tokens):
        batch_idx = 0 if batch == 1 else t
        for h in range(q_heads):
            kv_h = h // groups
            for pos in range(max_kv_len):
                if pos >= lengths[batch_idx]:
                    output[t, h, pos] = -np.inf
                    continue
                phys = table[batch_idx, pos // BLOCK_SIZE]
                offset = pos % BLOCK_SIZE
                idx = packed_idx[phys, offset, kv_h]
                qjl = packed_qjl[phys, offset, kv_h]
                primary = centroids[_unpack_primary(idx[None, :], bits)[0]]
                signs = _unpack_signs(qjl[None, :])[0]
                main = float(np.dot(query[t, h], primary))
                corr = float(np.dot(query_residual[t, h], signs))
                n = norm_cache[phys, offset, kv_h]
                g = gamma_cache[phys, offset, kv_h]
                output[t, h, pos] = n * main + coefficient * g * corr
    return [output]
