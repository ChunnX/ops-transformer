#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# -----------------------------------------------------------------------------------------------------------
"""TTK TestSpec compare adapter for npu_fused_infer_attention_score_v2 (MXFP8 FIA).

复用 common/result_compare_method.py 的 check_result。
返回 TTK 约定的 dict 格式: {pass, precision, error_info, metrics}

⚠️ TTK 传给 compare 的是 numpy array,不是 torch tensor。
   result_compare_method.check_result 期望 torch tensor(调 .detach().cpu().flatten()),
   所以这里做 numpy→torch 转换。
"""

import logging
import os
import sys

import numpy as np
import torch

_ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.dirname(_ASSETS_DIR)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
from common import result_compare_method

logger = logging.getLogger(__name__)


def _to_torch(x):
    """numpy array → torch tensor; torch tensor → 原样返回; None → None
    处理 bfloat16: numpy 原生不支持 bfloat16(ml_dtypes), torch.from_numpy 会失败,
    需要先 view uint8 再 reinterpret 为 bfloat16。
    """
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        # bfloat16 在 numpy 里是 ml_dtypes.bfloat16, torch.from_numpy 不支持
        if x.dtype.name == "bfloat16":
            # 通过 uint8 view + torch view 还原 bfloat16
            return torch.from_numpy(x.view(np.uint8)).view(torch.bfloat16)
        return torch.from_numpy(x)
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x)


def compare(*outputs, **kwargs):
    """对比 NPU 输出与 golden 输出。

    TTK 传参方式: [npu_out(, npu_lse), golden_out(, golden_lse)]
    当 lse 未启用时, npu_lse 可能是空数组(size=0), golden 不含 lse。
    所以输出数可能不等,需按实际非空输出配对。
    """
    # 过滤掉 None 和空数组(size=0),只保留有效输出
    valid = []
    for o in outputs:
        if o is None:
            continue
        if hasattr(o, 'size') and o.size == 0:
            continue
        if hasattr(o, 'numel') and o.numel() == 0:
            continue
        valid.append(o)

    if len(valid) < 2:
        return {"pass": False, "precision": "invalid",
                "error_info": f"not enough valid outputs: {len(valid)}"}

    # 前半 NPU,后半 golden
    half = len(valid) // 2
    npu_outputs = valid[:half]
    golden_outputs = valid[half:]

    all_pass = True
    results = []

    for idx, (npu_out, golden_out) in enumerate(zip(npu_outputs, golden_outputs)):
        label = "atten" if idx == 0 else f"out{idx}"
        npu_torch = _to_torch(npu_out)
        golden_torch = _to_torch(golden_out)
        result, fulfill_percent, max_error = result_compare_method.check_result(golden_torch, npu_torch)
        is_pass = (result == "Pass")
        if not is_pass:
            all_pass = False
        results.append({
            "pass": is_pass,
            "precision": fulfill_percent,
            "error_info": None if is_pass else f"{label}: result={result}, MaxRE={max_error}",
            "metrics": {"label": label, "result": result, "PctRlt": fulfill_percent, "MaxRE": max_error},
        })

    if len(results) == 1:
        return results[0]
    return results
