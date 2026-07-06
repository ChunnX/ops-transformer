#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Input function for stem_oam_prep_varlen_q (TTK framework).
"""
import numpy as np
import tbetoolkits
from .registry import register_input


@register_input(["stem_oam_prep_varlen_q"])
def stem_oam_prep_varlen_q_input(context: "tbetoolkits.UniversalTestcaseStructure"):
    q = context.input_arrays[0]
    q_seq_lens = context.input_arrays[2]
    q_scale = context.input_arrays[3] if len(context.input_arrays) > 3 else None

    total_tokens = q.shape[0]
    H_q = q.shape[1]
    batch = q_seq_lens.shape[0]

    avg = total_tokens // batch
    q_seq_lens = np.full(batch, avg, dtype=np.int32)
    remainder = total_tokens - avg * batch
    if remainder > 0:
        q_seq_lens[:remainder] += 1

    cu_seq_lens_q = np.zeros(batch + 1, dtype=np.int32)
    cu_seq_lens_q[1:] = np.cumsum(q_seq_lens)

    if q_scale is None:
        q_scale = np.random.uniform(0.5, 1.5, (total_tokens, H_q)).astype(np.float32)

    return [q, q_seq_lens, cu_seq_lens_q, q_scale]
