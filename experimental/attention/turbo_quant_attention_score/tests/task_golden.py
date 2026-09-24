# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Torch NPU golden for TurboQuantAttentionScore."""

import math
import numpy as np
import torch
import torch_npu  # noqa: F401


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


def _npu_tensor(value, dtype):
    array = np.asarray(value)
    np_dtype = {
        torch.float32: np.float32,
        torch.uint8: np.uint8,
        torch.int32: np.int32,
    }[dtype]
    return (
        torch.from_numpy(array.astype(np_dtype, copy=False))
        .to("npu")
        .to(dtype)
        .contiguous()
    )


def _unpack_primary(packed, bits):
    values = packed.to(torch.int32)
    if bits == 3:
        triplets = values.reshape(*values.shape[:-1], 16, 3)
        words = triplets[..., 0] | (triplets[..., 1] << 8) | (triplets[..., 2] << 16)
        return torch.stack([(words >> s) & 7 for s in range(0, 24, 3)], -1).reshape(
            *packed.shape[:-1], HEAD_DIM
        )

    mask = (1 << bits) - 1
    return torch.stack([(values >> s) & mask for s in range(0, 8, bits)], -1).reshape(
        *packed.shape[:-1], HEAD_DIM
    )


def _unpack_signs(packed):
    values = packed.to(torch.int32)
    bits = torch.stack([(values >> s) & 1 for s in range(8)], -1)
    return bits.reshape(*packed.shape[:-1], HEAD_DIM).float().mul_(2).sub_(1)


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
    if not torch.npu.is_available():
        raise RuntimeError("Ascend NPU is required")

    bits = int(mse_bits)
    if bits not in (2, 3, 4):
        raise ValueError("mse_bits must be 2, 3, or 4")

    query = _npu_tensor(query_rotated, torch.float32)
    query_residual = _npu_tensor(query_qjl, torch.float32)
    packed_idx = _npu_tensor(key_cache_idx, torch.uint8)
    packed_qjl = _npu_tensor(key_cache_qjl, torch.uint8)
    norm_cache = _npu_tensor(key_cache_norm, torch.float32)
    gamma_cache = _npu_tensor(key_cache_gamma, torch.float32)
    table = _npu_tensor(block_table, torch.int32).long()
    lengths = _npu_tensor(seq_lens, torch.int32).long()

    num_tokens, q_heads, _ = query.shape
    batch = table.shape[0]
    kv_heads = packed_idx.shape[2]
    groups = q_heads // kv_heads
    max_kv_len = int(lengths.max().item())

    query_main = query.reshape(num_tokens, kv_heads, groups, HEAD_DIM)
    query_residual = query_residual.reshape(num_tokens, kv_heads, groups, HEAD_DIM)

    centroids = torch.tensor(CENTROIDS[bits], dtype=torch.float32, device=query.device)
    coefficient = math.sqrt(math.pi / 2.0) / HEAD_DIM
    output = torch.empty(
        (num_tokens, q_heads, max_kv_len), dtype=torch.float32, device=query.device
    )

    chunk_size = 64 if batch == 1 and num_tokens > 1 else 512

    for start in range(0, max_kv_len, chunk_size):
        end = min(start + chunk_size, max_kv_len)
        positions = torch.arange(start, end, device=query.device, dtype=torch.long)

        logical = positions // BLOCK_SIZE
        physical = table[:, logical]
        offsets = positions.remainder(BLOCK_SIZE).expand_as(physical)

        idx = packed_idx[physical, offsets]
        qjl = packed_qjl[physical, offsets]
        norm = norm_cache[physical, offsets]
        gamma = gamma_cache[physical, offsets]

        primary = centroids[_unpack_primary(idx, bits).long()]
        signs = _unpack_signs(qjl)

        if batch == 1:
            main = torch.einsum("thgd,lhd->thgl", query_main, primary[0])
            correction = torch.einsum("thgd,lhd->thgl", query_residual, signs[0])

            main *= norm[0].transpose(0, 1).unsqueeze(0).unsqueeze(2)
            correction *= gamma[0].transpose(0, 1).unsqueeze(0).unsqueeze(2)

            score = (main + coefficient * correction).reshape(
                num_tokens, q_heads, end - start
            )

            valid = positions < lengths[0]
            output[..., start:end] = score.masked_fill(~valid, -torch.inf)

        else:
            main = torch.einsum("bhgd,blhd->bhgl", query_main, primary)
            correction = torch.einsum("bhgd,blhd->bhgl", query_residual, signs)

            main *= norm.permute(0, 2, 1).unsqueeze(2)
            correction *= gamma.permute(0, 2, 1).unsqueeze(2)

            score = (main + coefficient * correction).reshape(
                num_tokens, q_heads, end - start
            )

            valid = positions.unsqueeze(0) < lengths.unsqueeze(1)
            output[..., start:end] = score.masked_fill(~valid.unsqueeze(1), -torch.inf)

    torch.npu.synchronize()
    return [output.detach().cpu().numpy()]
