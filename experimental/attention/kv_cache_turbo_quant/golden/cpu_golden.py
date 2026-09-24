# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU 复刻任务书 golden.py 的计算，不依赖 torch / torch_npu。

契约来源：任务书/9月社区任务-kv_cache_turbo_quant算子开发/golden.py
只复刻公式与打包，不改质心、不改比较方向。
"""

from __future__ import annotations

import numpy as np

CENTROIDS = {
    2: np.array(
        [-0.1335033178, -0.04002048075, 0.04002048075, 0.1335033178], dtype=np.float32
    ),
    3: np.array(
        [
            -0.19020693,
            -0.1187859178,
            -0.06682205945,
            -0.02166347019,
            0.02166347019,
            0.06682205945,
            0.1187859178,
            0.19020693,
        ],
        dtype=np.float32,
    ),
    4: np.array(
        [
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
        dtype=np.float32,
    ),
}


def pack_bits(values: np.ndarray, bit_width: int) -> np.ndarray:
    """与 golden.py `_pack_bits` 相同：每 8 个码并成 bit_width 个字节。"""
    values = np.asarray(values, dtype=np.int64)
    groups = (values.shape[-1] + 7) // 8
    if values.shape[-1] % 8:
        pad = 8 - (values.shape[-1] % 8)
        values = np.pad(values, [(0, 0)] * (values.ndim - 1) + [(0, pad)])
    values = values.reshape(*values.shape[:-1], groups, 8)
    word = values[..., 0].copy()
    for lane in range(1, 8):
        word |= values[..., lane] << (lane * bit_width)
    if bit_width == 1:
        return word.astype(np.uint8)
    parts = [(word >> (8 * byte)).astype(np.uint8) for byte in range(bit_width)]
    packed = np.stack(parts, axis=-1)
    return packed.reshape(*packed.shape[:-2], groups * bit_width)


def encode(
    kv_vectors: np.ndarray,
    rotation_matrix: np.ndarray,
    qjl_matrix: np.ndarray,
    mse_bits: int,
):
    bits = int(mse_bits)
    if bits not in CENTROIDS:
        raise ValueError("mse_bits must be 2, 3, or 4")

    x = np.asarray(kv_vectors, dtype=np.float32)
    rotation = np.asarray(rotation_matrix, dtype=np.float32)
    qjl = np.asarray(qjl_matrix, dtype=np.float32)

    norm = np.sqrt(np.sum(x * x, axis=-1))
    unit = np.where((norm > 0)[..., None], x / np.clip(norm, 1e-30, None)[..., None], 0)
    rotated = unit @ rotation.T

    centroids = CENTROIDS[bits]
    boundaries = (centroids[:-1] + centroids[1:]) * np.float32(0.5)
    indices = np.sum(rotated[..., None] > boundaries, axis=-1).astype(np.uint8)

    primary = centroids[indices.astype(np.int64)]
    residual = rotated - primary
    residual_norm = np.sqrt(np.sum(residual * residual, axis=-1))
    gamma = norm * residual_norm
    residual_unit = np.where(
        (residual_norm > 0)[..., None],
        residual / np.clip(residual_norm, 1e-30, None)[..., None],
        0,
    )
    projected = residual_unit @ qjl.T

    quant_idx = pack_bits(indices, bits)
    quant_qjl = pack_bits((projected >= 0).astype(np.uint8), 1)
    return quant_idx, quant_qjl, norm.astype(np.float32), gamma.astype(np.float32)
