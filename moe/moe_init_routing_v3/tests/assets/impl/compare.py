#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compare helpers for MoeInitRoutingV3 TTK integration.

Compares NPU outputs against golden references. expandedRowIdxOut and
expertTokensCountOrCumsumOut use exact match; expandedXOut and expandedScaleOut
use np.isclose with tolerance.
"""

import numpy as np


def _to_numpy(value):
    if value is None:
        return None
    if hasattr(value, 'detach'):
        value = value.detach().cpu()
        if value.dtype in (np.dtype('bfloat16'), np.dtype('float16')):
            value = value.to(np.float32)
        return value.numpy()
    return np.asarray(value)


def _compare_int(npu_out, golden_out):
    npu = _to_numpy(npu_out)
    golden = _to_numpy(golden_out)
    if npu is None and golden is None:
        return {"pass": True, "precision": 100.0}
    if npu is None or golden is None:
        return {"pass": False, "precision": 0.0, "error_info": "one of outputs is None"}
    npu = npu.reshape(-1)
    golden = golden.reshape(-1)
    if npu.shape != golden.shape:
        return {"pass": False, "precision": "shape_mismatch",
                "error_info": f"shape mismatch: npu={npu.shape}, golden={golden.shape}"}
    match = np.array_equal(npu, golden)
    diff_count = int(np.sum(npu != golden))
    return {"pass": bool(match), "precision": float(npu.size - diff_count) / npu.size * 100.0 if npu.size > 0 else 100.0,
            "error_info": None if match else f"mismatch count={diff_count}/{npu.size}"}


def _compare_float(npu_out, golden_out, rtol=0.001, atol=0.001):
    npu = _to_numpy(npu_out).reshape(-1).astype(np.float32)
    golden = _to_numpy(golden_out).reshape(-1).astype(np.float32)
    if npu.shape != golden.shape:
        return {"pass": False, "precision": "shape_mismatch",
                "error_info": f"shape mismatch: npu={npu.shape}, golden={golden.shape}"}
    if npu.size == 0:
        return {"pass": True, "precision": 100.0}
    close = np.isclose(npu, golden, rtol=rtol, atol=atol, equal_nan=True)
    fulfill = float(np.sum(close)) / npu.size * 100.0
    return {"pass": fulfill >= 99.5, "precision": fulfill,
            "error_info": None if fulfill >= 99.5 else f"fulfill={fulfill:.2f}%"}


def compare(*outputs, **kwargs):
    """Compare NPU outputs against golden outputs.

    Outputs order: [expandedXOut_npu, expandedRowIdxOut_npu,
                    expertTokensCountOrCumsumOut_npu, expandedScaleOut_npu,
                    expandedXOut_golden, expandedRowIdxOut_golden,
                    expertTokensCountOrCumsumOut_golden, expandedScaleOut_golden]
    """
    if len(outputs) < 2 or len(outputs) % 2 != 0:
        return {"pass": False, "precision": "invalid",
                "error_info": "compare expects NPU outputs followed by golden outputs"}
    half = len(outputs) // 2
    npu_outputs = outputs[:half]
    golden_outputs = outputs[half:]

    results = []
    for i in range(half):
        npu = npu_outputs[i]
        golden = golden_outputs[i]
        if npu is None or golden is None:
            if npu is None and golden is None:
                results.append({"pass": True, "precision": 100.0})
            else:
                results.append({"pass": False, "precision": 0.0,
                                "error_info": f"output[{i}] one is None"})
            continue
        npu_arr = _to_numpy(npu)
        if npu_arr.dtype.kind in ('i', 'u'):
            results.append(_compare_int(npu, golden))
        else:
            results.append(_compare_float(npu, golden))

    overall_pass = all(r.get("pass", False) for r in results)
    if len(results) == 1:
        return results[0]
    return {"pass": overall_pass, "results": results}
