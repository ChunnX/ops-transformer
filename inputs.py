#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# -----------------------------------------------------------------------------------------------------------
"""TTK TestSpec customize_inputs adapter for npu_fused_infer_attention_score_v2 (MXFP8 FIA).

复用 common/fia_fullquant_mxfp8_golden.py 的 MXFP8 量化函数。
生成 BNSD FP8 Q/K/V + fp32 scale + block_table。
"""

import logging
import math
import os
import sys

import torch

_ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.dirname(_ASSETS_DIR)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
from common import fia_fullquant_mxfp8_golden as golden_mod

logger = logging.getLogger(__name__)

__input__ = {
    "e2e": {
        "fia_mxfp8_wrapper.npu_fia_mxfp8": "generate_fia_mxfp8_inputs"
    }
}

_SEED_MAP = {"q": 54, "k": 3, "v": 4}


def generate_fia_mxfp8_inputs(q, k, v,
                              dequant_scale_q, dequant_scale_k, dequant_scale_v,
                              p_scale, block_table,
                              *,
                              B, N_q, N_kv, D,
                              actual_seq_q, actual_seq_kv,
                              enable_pa, kv_cache_layout, block_size,
                              sparse_mode, q_scale_layout,
                              enable_rope=False, D_rope=64,
                              enable_lse=False, graph_path=0,
                              input_layout="TND",
                              is_contiguous=True, device_id=0,
                              data_range_q=1.0, data_range_k=1.0, data_range_v=1.0,
                              p_scale_value=1.0,
                              **kwargs):
    """生成 BNSD FP8 Q/K/V + fp32 scale + block_table。"""
    actual_seq_q = list(actual_seq_q)
    actual_seq_kv = list(actual_seq_kv)
    max_sq = max(actual_seq_q) if actual_seq_q else D
    max_skv = max(actual_seq_kv) if actual_seq_kv else D

    fp8_dtype = torch.float8_e4m3fn
    group_size = 32
    p_scale_val = float(p_scale_value)

    for gkey, gval in [("B", B), ("N_q", N_q), ("N_kv", N_kv), ("D", D),
                       ("ENABLE_ROPE", enable_rope), ("D_rope", D_rope),
                       ("FP8_DTYPE", fp8_dtype), ("QUANT_GROUP_SIZE", group_size)]:
        setattr(golden_mod, gkey, gval)

    torch.manual_seed(_SEED_MAP["q"])
    q_fp16 = (torch.rand(B, N_q, max_sq, D, dtype=torch.float16) * 2 - 1) * float(data_range_q)
    torch.manual_seed(_SEED_MAP["k"])
    k_fp16 = (torch.rand(B, N_kv, max_skv, D, dtype=torch.float16) * 2 - 1) * float(data_range_k)
    torch.manual_seed(_SEED_MAP["v"])
    v_fp16 = (torch.rand(B, N_kv, max_skv, D, dtype=torch.float16) * 2 - 1) * float(data_range_v)

    quant_scale_q = golden_mod.get_mxfp8_per_token_group_quant_scale(q_fp16, fp8_dtype, group_size)
    quant_scale_k = golden_mod.get_mxfp8_per_token_group_quant_scale(k_fp16, fp8_dtype, group_size)
    quant_scale_v = golden_mod.get_mxfp8_per_channel_group_quant_scale(v_fp16, fp8_dtype, group_size)

    dequant_scale_q = quant_scale_q
    dequant_scale_k = quant_scale_k
    dequant_scale_v = quant_scale_v

    fp8_max = 448.0
    q_fp8 = golden_mod.mxfp8_per_token_group_quant(q_fp16, quant_scale_q, group_size).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    k_fp8 = golden_mod.mxfp8_per_token_group_quant(k_fp16, quant_scale_k, group_size).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    v_fp8 = golden_mod.mxfp8_per_channel_group_quant(v_fp16, quant_scale_v, group_size).clamp(-fp8_max, fp8_max).to(fp8_dtype)

    p_scale_t = torch.tensor([p_scale_val], dtype=torch.float32)

    block_table_t = None
    if enable_pa:
        block_num = sum(math.ceil(s / block_size) for s in actual_seq_kv)
        max_blocks = max(math.ceil(s / block_size) for s in actual_seq_kv)
        block_idx_list = torch.randperm(block_num, dtype=torch.int32)
        block_table_t = torch.full((B, max_blocks), -1, dtype=torch.int32)
        idx = 0
        for b in range(B):
            n_blocks = math.ceil(actual_seq_kv[b] / block_size)
            for j in range(n_blocks):
                block_table_t[b, j] = block_idx_list[idx]
                idx += 1

    return [q_fp8, k_fp8, v_fp8,
            dequant_scale_q, dequant_scale_k, dequant_scale_v,
            p_scale_t, block_table_t]
