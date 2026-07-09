#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# -----------------------------------------------------------------------------------------------------------
"""TTK TestSpec compare adapter for npu_fused_infer_attention_score_v2 (MXFP8 FIA).

复用 common/result_compare_method.py 的 check_result。
"""

import logging
import os
import sys

import torch

_ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.dirname(_ASSETS_DIR)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
from common import result_compare_method

logger = logging.getLogger(__name__)


def compare(*outputs, **kwargs):
    """对比 NPU 输出与 golden 输出。

    outputs: [npu_out(, npu_lse), golden_out(, golden_lse)]
    返回 TTK 约定的 (precision_str, log_str, is_pass) 三元组。
    """
    if len(outputs) < 2 or len(outputs) % 2 != 0:
        return "invalid", "compare expects NPU outputs followed by golden outputs", False

    half = len(outputs) // 2
    npu_outputs = outputs[:half]
    golden_outputs = outputs[half:]

    all_pass = True
    precision_parts = []
    log_parts = []

    for idx, (npu_out, golden_out) in enumerate(zip(npu_outputs, golden_outputs)):
        if npu_out is None or golden_out is None:
            continue
        label = "atten" if idx == 0 else "lse"
        result, fulfill_percent, max_error = result_compare_method.check_result(golden_out, npu_out)
        is_pass = (result == "Pass")
        if not is_pass:
            all_pass = False
        precision_parts.append(f"{label}: {result} PctRlt={fulfill_percent:.4f}% MaxRE={max_error}")
        log_parts.append(f"[{label}] result={result} PctRlt={fulfill_percent:.4f}% MaxRE={max_error}")

    precision_str = " | ".join(precision_parts) if precision_parts else "no outputs"
    log_str = "\n".join(log_parts)
    return precision_str, log_str, all_pass
