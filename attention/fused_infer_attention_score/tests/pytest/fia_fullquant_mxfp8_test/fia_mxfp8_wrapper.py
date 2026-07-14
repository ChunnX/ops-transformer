#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# -----------------------------------------------------------------------------------------------------------
"""TTK e2e wrapper for npu_fused_infer_attention_score_v2 (MXFP8 FIA).

为什么需要 wrapper:
  虽然 npu_fused_infer_attention_score_v2 是 torch_npu 原生 API,
  但 MXFP8 场景下 NPU 调用前需要大量预处理:
    - BNSD → TND/BSND layout 转换
    - FP32 scale → e8m0fnu 打包(偶奇对/行交错)
    - PA 模式 KV Cache 预处理(block_table 散布)
    - Q scale layout AUTO→TND/N2TGD 自动选择
  这些预处理逻辑在 common/fia_fullquant_mxfp8_golden.py 的 npu_mxfp8_fa() 里,
  本 wrapper 复用该函数,避免重写。

输入约定(customize_inputs 与 golden 收到一致输入,均为 BNSD 格式):
  q, k, v: FP8 tensor, BNSD
  dequant_scale_q/k/v: FP32 scale, BNSD(分组维)
  p_scale: float32 标量 tensor
  block_table: int32 tensor (PA) 或 None
"""

import logging
import math
import os
import sys
from typing import List

import torch
import torch_npu

# 复用 common/ 里的 npu_mxfp8_fa + layout 转换 / e8m0 打包函数
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
from common import fia_fullquant_mxfp8_golden as golden_mod

logger = logging.getLogger(__name__)


def _apply_golden_globals(params):
    """把 case 参数注入 golden 模块全局变量。"""
    for k, v in params.items():
        setattr(golden_mod, k, v)


def npu_fia_mxfp8(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
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
    """单函数封装:从 customize_inputs 缓存取真实 FP8 数据,内部预处理+调原生 API。

    ⚠️ TTK 传入的 q/k/v 等 tensor 是随机占位数据(不是 customize_inputs 生成的)。
    因为 TTK customize_inputs 约定是原地修改,但 MXFP8 需改 dtype 无法原地改。
    所以 customize_inputs 把真实数据缓存到模块级变量, wrapper 从中取。
    """
    torch_npu.npu.set_device(int(device_id))

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
        logger.info("[WRAPPER] 使用 customize_inputs 缓存的真实 FP8 数据")
    else:
        logger.warning("[WRAPPER] customize_inputs 缓存为空,使用 TTK 传入的占位 tensor(可能出错)")

    p_scale_val = float(p_scale.item()) if isinstance(p_scale, torch.Tensor) else float(p_scale)

    # 注入 golden 全局变量,npu_mxfp8_fa 读这些
    _apply_golden_globals({
        "B": B, "N_q": N_q, "N_kv": N_kv, "D": D,
        "ACTUAL_SEQ_Q": list(actual_seq_q), "ACTUAL_SEQ_KV": list(actual_seq_kv),
        "ENABLE_PA": enable_pa, "KV_CACHE_LAYOUT": kv_cache_layout,
        "BLOCK_SIZE": block_size, "SPARSE_MODE": sparse_mode,
        "Q_SCALE_LAYOUT": q_scale_layout, "P_SCALE": p_scale_val,
        "ENABLE_ROPE": enable_rope, "D_rope": D_rope,
        "ENABLE_LSE": enable_lse, "FP8_DTYPE": torch.float8_e4m3fn,
        "QUANT_GROUP_SIZE": 32, "INPUT_LAYOUT": input_layout,
        "IS_CONTIGUOUS": is_contiguous, "DEVICE_ID": device_id,
        "GRAPH_PATH": graph_path,
    })

    # 直接复用 common 的 npu_mxfp8_fa(它做 layout 转换 + e8m0 打包 + PA 预处理 + 调 API)
    atten_out, lse_out = golden_mod.npu_mxfp8_fa(
        q, k, v,
        dequant_scale_q, dequant_scale_k, dequant_scale_v, p_scale,
        golden_mod.ACTUAL_SEQ_Q, golden_mod.ACTUAL_SEQ_KV,
        block_table if isinstance(block_table, torch.Tensor) else None,
        qr_bf16, kr_bf16,
    )

    return atten_out, lse_out
