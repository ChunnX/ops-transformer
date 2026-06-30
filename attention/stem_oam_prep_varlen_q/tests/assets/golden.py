#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import numpy as np
import math

__golden__ = {
    "kernel": {
        "stem_oam_prep_varlen_q": "stem_oam_prep_varlen_q_golden"
    }
}


def stem_oam_prep_varlen_q_golden(
        q,
        q_seq_lens,
        cu_seq_lens_q,
        q_scale=None,
        *,
        stemBlockSize=128,
        stemStride=16,
        **kwargs
):
    """
    Kernel golden for stem_oam_prep_varlen_q.

    Computes qflat = Weighted Group Sum of Q x q_scale, flattened in natural order.

    Parameters:
    - q:              [total_tokens, H_q, D]  FP8 (uint8 storage) or BF16/FP16
    - q_seq_lens:     [batch]                 int32
    - cu_seq_lens_q:  [batch+1]               int32
    - q_scale:        [total_tokens, H_q]     float32 (optional, required for FP8)
    - stemBlockSize:  int, default 128
    - stemStride:     int, default 16

    Returns:
        qflat: [batch, H_q, max_Qb, stemStride * D] BF16
    """
    B = stemBlockSize
    S = stemStride
    D = q.shape[-1]
    H_q = q.shape[1]
    R = B // S
    batch = len(q_seq_lens)

    q_fp32 = q.astype(np.float32)

    max_Qb = max(math.ceil(int(sl) / B) for sl in q_seq_lens)

    qflat = np.zeros((batch, H_q, max_Qb, S * D), dtype=np.float32)

    for b in range(batch):
        q_len = int(q_seq_lens[b])
        cu_off = int(cu_seq_lens_q[b])
        q_padded = math.ceil(q_len / B) * B
        num_Qb = q_padded // B

        Q_segment = q_fp32[cu_off:cu_off + q_len]
        Q_dense = np.zeros((q_padded, H_q, D), dtype=np.float32)
        Q_dense[:q_len] = Q_segment

        for h in range(H_q):
            Q_blocks = Q_dense[:num_Qb * B, h, :].reshape(num_Qb, R, S, D)

            if q_scale is not None:
                positions = np.arange(num_Qb * B).reshape(num_Qb, R, S)
                global_positions = cu_off + positions
                scales = q_scale[global_positions, h].astype(np.float32)
                Q_weighted = Q_blocks * scales[:, :, :, np.newaxis]
            else:
                Q_weighted = Q_blocks

            Q_group_sum = Q_weighted.sum(axis=1)

            for qb in range(num_Qb):
                qflat[b, h, qb, :] = Q_group_sum[qb].reshape(S * D)

    return qflat.astype(np.float16) if q.dtype == np.float16 else qflat.view(np.uint16)
