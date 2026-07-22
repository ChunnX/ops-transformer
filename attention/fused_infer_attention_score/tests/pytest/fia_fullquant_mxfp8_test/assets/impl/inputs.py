#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# -----------------------------------------------------------------------------------------------------------
"""TTK TestSpec customize_inputs adapter for npu_fused_infer_attention_score_v2 (MXFP8 FIA).

⚠️ TTK 约定: customize_inputs 原地修改传入的 tensor (x[:] = value), 不使用返回值。
   但 MXFP8 场景需要改 dtype (float16→float8_e4m3fn), 无法原地修改。
   所以用模块级变量 _cached_inputs 缓存生成的数据, wrapper 从中取。
"""

import logging
import math
import os
import sys
from typing import List

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


def get_cached_inputs():
    """wrapper 调用前从这里取 customize_inputs 生成的真实数据。
    存在 golden_mod 上,避免模块重复加载导致缓存丢失。
    """
    return getattr(golden_mod, "_cached_mxfp8_inputs", None)


def generate_fia_mxfp8_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                              dequant_scale_q: torch.Tensor, dequant_scale_k: torch.Tensor, dequant_scale_v: torch.Tensor,
                              p_scale: torch.Tensor, block_table: torch.Tensor,
                              *,
                              B: int, N_q: int, N_kv: int, D: int,
                              actual_seq_q: List[int], actual_seq_kv: List[int],
                              enable_pa: bool, kv_cache_layout: str, block_size: int,
                              sparse_mode: int, q_scale_layout: str,
                              enable_rope: bool = False, D_rope: int = 64,
                              enable_lse: bool = False, graph_path: int = 0,
                              input_layout: str = "TND",
                              is_contiguous: bool = True, device_id: int = 0,
                              data_range_q: float = 1.0, data_range_k: float = 1.0, data_range_v: float = 1.0,
                              p_scale_value: float = 1.0,
                              **kwargs):
    """生成 BNSD FP8 Q/K/V + fp32 scale + block_table,缓存到 golden_mod 上。"""
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

    # 生成 rope 数据
    qr_bf16 = None
    kr_bf16 = None
    if enable_rope and D_rope > 0:
        torch.manual_seed(8)   # SEED_QR
        qr_bf16 = torch.randn(B, N_q, max_sq, D_rope, dtype=torch.bfloat16)
        torch.manual_seed(9)   # SEED_KR
        kr_bf16 = torch.randn(B, N_kv, max_skv, D_rope, dtype=torch.bfloat16)

    # 缓存到 golden_mod 上(不是本模块,避免 spec.py 重复加载导致缓存丢失)
    golden_mod._cached_mxfp8_inputs = [q_fp8, k_fp8, v_fp8,
                                       dequant_scale_q, dequant_scale_k, dequant_scale_v,
                                       p_scale_t, block_table_t,
                                       qr_bf16, kr_bf16]
    # 无返回值(符合 TTK 约定)
