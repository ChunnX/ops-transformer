#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Input function for stem_oam_prep_varlen_q (TTK framework).
"""
import numpy as np
import ml_dtypes
import tbetoolkits
from .registry import register_input
from typing import Tuple, Optional


@register_input(["stem_oam_prep_varlen_q"])
def stem_oam_prep_varlen_q_input(context: "tbetoolkits.UniversalTestcaseStructure") \
        -> Tuple[Tuple[Optional[np.ndarray], ...]]:
    q = context.input_arrays[0]
    q_seq_lens = context.input_arrays[1]
    q_scale = context.input_arrays[3] if len(context.input_arrays) > 3 else None

    total_tokens = q.shape[0]
    H_q = q.shape[1]
    D = q.shape[2]
    batch = q_seq_lens.shape[0]

    avg = total_tokens // batch
    q_seq_lens = np.full(batch, avg, dtype=np.int64)
    remainder = total_tokens - avg * batch
    if remainder > 0:
        q_seq_lens[:remainder] += 1

    cu_seq_lens_q = np.zeros(batch + 1, dtype=np.int64) 
    cu_seq_lens_q[1:] = np.cumsum(q_seq_lens)

    if q_scale is None:
        q_scale = np.random.uniform(0.5, 1.5, (total_tokens, H_q)).astype(np.float32)
    # 从 context 获取输入 dtype
    q_dtype = context.stc_input_dtypes[0]
    q_seq_lens_dtype = context.stc_input_dtypes[1]
    cu_seq_lens_q_dtype = context.stc_input_dtypes[2]
    q_scale_dtype = context.stc_input_dtypes[3]

    # context.input_arrays[1] = q_seq_lens
    # context.input_arrays[2] = cu_seq_lens_q
    # context.input_arrays[3] = q_scale
    q_out = q.astype(q_dtype)
    q_seq_lens_out = q_seq_lens.astype(q_seq_lens_dtype)
    cu_seq_lens_q_out = cu_seq_lens_q.astype(cu_seq_lens_q_dtype)
    q_scale_out = q_scale.astype(q_scale_dtype)
    print(f"[gen_data] total_tokens={total_tokens}, H_q={H_q}, D={D}, batch={batch}")
    print(f"[gen_data] q_seq_lens={q_seq_lens}")
    print(f"[gen_data] cu_seq_lens_q={cu_seq_lens_q}")
    print(f"[gen_data] q.shape={q.shape}, q.dtype={q.dtype}")
    print(f"[gen_data] q_scale.shape={q_scale.shape}, q_scale.dtype={q_scale.dtype}")
    print(f"[gen_data] q_scale[0:4]={q_scale[0:4].flatten()}")
    print(f"[gen_data] q[0,0,0:4]={q[0,0,0:4]}")
    return (q_out, q_seq_lens_out, cu_seq_lens_q_out, q_scale_out)
