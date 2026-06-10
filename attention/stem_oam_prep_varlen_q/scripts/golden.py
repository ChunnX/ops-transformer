# ============================================================================
# Golden reference for stem_oam_prep_varlen_q
# ============================================================================
#
# Computes qflat = Weighted Group Sum of Q × qk_scale, flattened in natural order.
# qk_scale = qscale × kscale (scale fusion).
#
# This file is shared by gen_data.py (direct-invoke) and test_torch.py (PyTorch).
# ============================================================================

import torch
import math
import numpy as np


def stem_oam_prep_varlen_q_golden(q, qscale, kscale, q_seq_lens, cu_seqlens_q,
                                   B=128, S=16):
    """
    Compute golden reference for stem_oam_prep_varlen_q.

    Args:
        q:           varlen Q tensor [total_tokens, H_q, D]  (numpy BF16 or torch BF16)
        qscale:      [batch, H_q, max_seqlen_pad_stemBlockSize]  (numpy FP32 or torch FP32)
        kscale:      scalar                                   (numpy FP32 array or torch scalar)
        q_seq_lens:  [batch]                                  (numpy INT32 or torch INT32)
        cu_seqlens_q:[batch+1]                                (numpy INT32 or torch INT32)
        B:           stem_block_size (default 128)
        S:           stem_stride (default 16)

    Returns:
        qflat: [batch, H_q, max_Qb, S*D] BF16
    """
    # Convert numpy to torch if needed
    if isinstance(q, np.ndarray):
        q = torch.from_numpy(q.copy())  # BF16 numpy → torch BF16
    if isinstance(qscale, np.ndarray):
        qscale = torch.from_numpy(qscale.copy()).float()
    if isinstance(kscale, np.ndarray):
        kscale_val = float(kscale[0])
    else:
        kscale_val = float(kscale)
    if isinstance(q_seq_lens, np.ndarray):
        q_seq_lens = torch.from_numpy(q_seq_lens.copy()).int()
    if isinstance(cu_seqlens_q, np.ndarray):
        cu_seqlens_q = torch.from_numpy(cu_seqlens_q.copy()).int()

    D = q.shape[-1]       # 128
    H_q = q.shape[1]      # num Q heads
    R = B // S            # 8
    batch = len(q_seq_lens)

    # Scale fusion: qk_scale = qscale * kscale
    qk_scale = qscale * kscale_val

    # Compute max_Qb
    max_Qb = max(math.ceil(sl.item() / B) for sl in q_seq_lens)

    # Output tensor
    qflat = torch.zeros(batch, H_q, max_Qb, S * D, dtype=torch.bfloat16)

    for b in range(batch):
        q_len   = q_seq_lens[b].item()
        cu_off  = cu_seqlens_q[b].item()
        q_padded = math.ceil(q_len / B) * B
        num_Qb  = q_padded // B

        # De-page varlen: extract segment and zero-pad
        Q_segment = q[cu_off : cu_off + q_len].float()   # [q_len, H_q, D]
        Q_dense = torch.zeros(q_padded, H_q, D, dtype=torch.float32)
        Q_dense[:q_len] = Q_segment

        for h in range(H_q):
            # Reshape into stem blocks: [num_Qb, R, S, D]
            Q_blocks = Q_dense[:num_Qb * B, h, :].reshape(num_Qb, R, S, D)

            # Position indices for scale lookup
            positions = torch.arange(num_Qb * B).reshape(num_Qb, R, S)

            # Get qk_scale for these positions (clamp to available range)
            max_pos = qk_scale.shape[2] - 1
            scale_indices = positions.clamp(max=max_pos)
            scales = qk_scale[b, h, scale_indices]   # [num_Qb, R, S]

            # Weighted group sum: Q × scale, then sum over R dimension
            Q_weighted = Q_blocks * scales.unsqueeze(-1)   # [num_Qb, R, S, D]
            Q_group_sum = Q_weighted.sum(dim=1)             # [num_Qb, S, D]

            # Flatten in natural order (NO flip)
            for qb in range(num_Qb):
                qflat[b, h, qb, :] = Q_group_sum[qb].reshape(S * D).bfloat16()

    return qflat