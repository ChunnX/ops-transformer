#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Special golden data generation function for stem_oam_prep_varlen_q
"""
import math
import numpy as np
import ml_dtypes
import tbetoolkits
from .registry import register_golden


@register_golden(["stem_oam_prep_varlen_q"])
def stem_oam_prep_varlen_q(context: "tbetoolkits.UniversalTestcaseStructure"):
    """
    Compute golden reference for stem_oam_prep_varlen_q.

    Args:
        context: Contains input arrays and compilation params.
                 input_arrays[0]: q          [total_tokens, H_q, D]  FP8 (uint8 storage)
                 input_arrays[1]: qscale     [total_tokens, H_q]     FP32
                 input_arrays[2]: q_seq_lens [batch]                 INT32
                 input_arrays[3]: cu_seqlens_q [batch+1]             INT32
·
                 attributes:
                 - stemBlockSize: stem_block_size (default 128)
                 - stemStride: stem_stride (default 16)

    Returns:
        qflat: [batch, H_q, max_Qb, S*D] BF16 (as numpy array)
    """
    q_np = context.input_arrays[0]
    qscale_np = context.input_arrays[3] if len(context.input_arrays) > 3 else None
    q_seq_lens_np = context.input_arrays[1]
    cu_seqlens_q_np = context.input_arrays[2]

    B = context.other_runtime_params['stemBlockSize']
    S = context.other_runtime_params['stemStride']

    D = q_np.shape[-1]
    H_q = q_np.shape[1]
    R = B // S
    batch = len(q_seq_lens_np)

    q_fp32 = q_np.astype(np.float32)
    if qscale_np is not None:
        qscale = qscale_np.astype(np.float32)
        if qscale.ndim == 1:
            qscale = qscale.reshape(-1, 1)
    else:
        qscale = None

    max_Qb = max(math.ceil(int(sl) / B) for sl in q_seq_lens_np)

    qflat = np.zeros((batch, H_q, max_Qb, S * D), dtype=np.float32)

    for b in range(batch):
        q_len = int(q_seq_lens_np[b].item())
        cu_off = int(cu_seqlens_q_np[b].item())
        q_padded = math.ceil(q_len / B) * B
        num_Qb = q_padded // B

        if num_Qb == 0:
            continue

        Q_segment = q_fp32[cu_off:cu_off + q_len]
        actual_len = Q_segment.shape[0]
        Q_dense = np.zeros((q_padded, H_q, D), dtype=np.float32)
        Q_dense[:actual_len] = Q_segment

        for h in range(H_q):
            Q_blocks = Q_dense[:num_Qb * B, h, :].reshape(num_Qb, R, S, D)

            if qscale is not None:
                positions = np.arange(num_Qb * B).reshape(num_Qb, R, S)
                global_positions = cu_off + positions
                valid_mask = global_positions < len(qscale)
                scales = np.zeros((num_Qb, R, S), dtype=np.float32)
                scales[valid_mask] = qscale[global_positions[valid_mask], h]
                Q_weighted = Q_blocks * scales[:, :, :, np.newaxis]
            else:
                Q_weighted = Q_blocks

            Q_group_sum = Q_weighted.sum(axis=1)

            for qb in range(num_Qb):
                qflat[b, h, qb, :] = Q_group_sum[qb].reshape(S * D)

    return qflat.astype(ml_dtypes.bfloat16)
