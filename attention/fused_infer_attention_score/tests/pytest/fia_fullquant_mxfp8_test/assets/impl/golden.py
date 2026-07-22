#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# -----------------------------------------------------------------------------------------------------------
"""TTK TestSpec golden adapter for npu_fused_infer_attention_score_v2 (MXFP8 FIA).

复用 common/fia_fullquant_mxfp8_golden.py 的 cpu_mxfp8_golden 参考实现。
本文件只做签名适配:把 TTK 传入的(tensor..., *, attrs..., **kwargs)映射到
cpu_mxfp8_golden 的调用约定。
"""

import logging
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

__golden__ = {
    "e2e": {
        "fia_mxfp8_wrapper.npu_fia_mxfp8": "cpu_fia_mxfp8"
    }
}


def _apply_golden_globals(attrs):
    """把 case 属性注入 golden 模块全局变量。"""
    mapping = {
        "B": "B", "N_q": "N_q", "N_kv": "N_kv", "D": "D",
        "actual_seq_q": "ACTUAL_SEQ_Q", "actual_seq_kv": "ACTUAL_SEQ_KV",
        "enable_pa": "ENABLE_PA", "kv_cache_layout": "KV_CACHE_LAYOUT",
        "block_size": "BLOCK_SIZE", "sparse_mode": "SPARSE_MODE",
        "q_scale_layout": "Q_SCALE_LAYOUT",
        "enable_rope": "ENABLE_ROPE", "D_rope": "D_rope",
        "enable_lse": "ENABLE_LSE", "input_layout": "INPUT_LAYOUT",
        "is_contiguous": "IS_CONTIGUOUS", "device_id": "DEVICE_ID",
        "graph_path": "GRAPH_PATH",
    }
    for attr_key, golden_key in mapping.items():
        if attr_key in attrs:
            setattr(golden_mod, golden_key, attrs[attr_key])
    setattr(golden_mod, "FP8_DTYPE", torch.float8_e4m3fn)
    setattr(golden_mod, "QUANT_GROUP_SIZE", 32)
    p_scale = attrs.get("p_scale_value", 1.0)
    if isinstance(p_scale, torch.Tensor):
        p_scale = float(p_scale.item())
    setattr(golden_mod, "P_SCALE", float(p_scale))


def cpu_fia_mxfp8(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
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
                  p_scale_value: float = 1.0,
                  **kwargs):
    """CPU golden:从 golden_mod 缓存取真实 FP8 数据,调 cpu_mxfp8_golden 算参考输出。"""
    # 从 golden_mod 取 customize_inputs 缓存的真实 FP8 数据
    cached = getattr(golden_mod, "_cached_mxfp8_inputs", None)
    qr_bf16 = None
    kr_bf16 = None
    if cached is not None:
        q, k, v, dequant_scale_q, dequant_scale_k, dequant_scale_v, p_scale, block_table = cached[:8]
        if len(cached) > 8:
            qr_bf16 = cached[8]
        if len(cached) > 9:
            kr_bf16 = cached[9]

    _apply_golden_globals({
        "B": B, "N_q": N_q, "N_kv": N_kv, "D": D,
        "actual_seq_q": list(actual_seq_q), "actual_seq_kv": list(actual_seq_kv),
        "enable_pa": enable_pa, "kv_cache_layout": kv_cache_layout,
        "block_size": block_size, "sparse_mode": sparse_mode,
        "q_scale_layout": q_scale_layout,
        "enable_rope": enable_rope, "D_rope": D_rope,
        "enable_lse": enable_lse, "input_layout": input_layout,
        "is_contiguous": is_contiguous, "device_id": device_id,
        "graph_path": graph_path,
        "p_scale_value": p_scale_value,
    })

    cpu_out, cpu_lse = golden_mod.cpu_mxfp8_golden(
        q, k, v,
        dequant_scale_q, dequant_scale_k, dequant_scale_v, p_scale,
        golden_mod.ACTUAL_SEQ_Q, golden_mod.ACTUAL_SEQ_KV,
        qr_bf16, kr_bf16,
    )

    compare_layout = "TND" if enable_pa else input_layout
    cpu_out_aligned = golden_mod.convert_q_bnsd_to_layout(cpu_out, golden_mod.ACTUAL_SEQ_Q, compare_layout)

    if enable_lse:
        cpu_lse_aligned = golden_mod.convert_q_bnsd_to_layout(cpu_lse, golden_mod.ACTUAL_SEQ_Q, compare_layout)
        return [cpu_out_aligned, cpu_lse_aligned]
    return [cpu_out_aligned]
