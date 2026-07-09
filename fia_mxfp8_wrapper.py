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


def npu_fia_mxfp8(q, k, v,
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
                  **kwargs):
    """单函数封装:接收 BNSD FP8 + fp32 scale,内部预处理+调原生 API。

    所有参数与 fia_fullquant_mxfp8_paramset 的字段对齐,
    由 CSV attributes 传入(经 TTK param_plan 解析为 kwargs)。
    """
    torch_npu.npu.set_device(int(device_id))

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
        block_table if isinstance(block_table, torch.Tensor) and block_table.numel() > 0 else None,
        None, None,  # qr_bf16, kr_bf16 (rope 暂不支持 CSV 透传)
    )

    return atten_out, lse_out
