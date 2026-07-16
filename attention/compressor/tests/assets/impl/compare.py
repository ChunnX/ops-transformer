#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Precision compare helpers for Compressor TestSpec adapters.

Comparison logic is consistent with compressor_golden.py check_result.
Compares 5 tensors: cmp_kv, kv_state_update, score_state_update,
kv_state_origin, score_state_origin.
"""

import numpy as np
import logging
import datetime
import os
import sys

_BF16_RTOL = 0.0078125
_BF16_ATOL = 0.0001
_FP16_RTOL = 0.005
_FP16_ATOL = 0.000025
_DIFF_THD = 0.005
_MAX_DIFF_HD = 10.0
_PCT_THD = 0.005
_MAX_ERROR_IDX = 10000000

_OUTPUT_NAMES = ["cmp_kv", "kv_state_update", "score_state_update",
                 "kv_state_origin", "score_state_origin"]


def as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _get_thresholds(dtype_str):
    if dtype_str == 'bfloat16':
        return _BF16_RTOL, _BF16_ATOL, _PCT_THD
    return _FP16_RTOL, _FP16_ATOL, _PCT_THD

def cal_relative_diff_np_isclose(real_data, expect_data, type_str='fp16'):
    diff = abs(float(real_data) - float(expect_data))
    result = diff / (np.abs(expect_data) + 10e-10)
    return result

def display_output_np_isclose(real_data, expect_data, start, end, expect_fp32_data=None):
    def display_inner(idx):
        j = idx + start
        diff_rate = cal_relative_diff_np_isclose(
            real_data[j], expect_data[j])

        if "inf" in str(expect_data[j]) or "nan" in str(expect_data[j]):
            diff_abs = "inf" if "inf" in str(expect_data[j]) else "nan"
            if expect_fp32_data is not None:
                print_log('%08d \t %-7s \t %-7s \t %-7s \t %-7s \t %-7s' % (
                    start + idx + 1, expect_fp32_data[j], expect_data[j], real_data[j], diff_abs, diff_rate))
            else:
                print_log('%08d \t %-7s \t %-7s \t %-7s \t %-7s' % (
                    start + idx + 1, expect_data[j], real_data[j], diff_abs, diff_rate))
        else:
            diff_abs = abs(np.float64(
                expect_data[j]) - np.float64(real_data[j]))
            if expect_fp32_data is not None:
                print_log('%08d \t %0.7f \t %0.7f \t %0.7f \t %0.7f \t %0.7f' % (
                    start + idx + 1, expect_fp32_data[j], expect_data[j], real_data[j], diff_abs, diff_rate))
            else:
                print_log('%08d \t %0.7f \t %0.7f \t %0.7f \t %0.7f' % (
                    start + idx + 1, expect_data[j], real_data[j], diff_abs, diff_rate))

    print_log(
        '---------------------------------------------------------------------------------------')
    if expect_fp32_data is not None:
        print_log(
            'Loop \t ExpFP32Out \t ExpFP16Out \t NPUOut \tFpDiff(min) \t RateDiff')
    else:
        print_log('Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff')
    print_log(
        '---------------------------------------------------------------------------------------')
    split_count = int(end - start)
    if split_count <= 20:
        for i in range(split_count + 1):
            display_inner(i)
    else:
        for i in range(10):
            display_inner(i)
        print_log('...   \t   ...   \t   ...   \t   ...    \t   ...')
        for i in range(split_count - 10 + 1, split_count + 1):
            display_inner(i)

def print_log(data=None, level='INFO'):
    print("[%s] [%s]-%s:%s - %s" % (datetime.datetime.now().strftime(
        "%Y/%m/%d %H:%M:%S"), level, os.path.basename(sys._getframe().f_back.f_code.co_filename),
                                    str(sys._getframe().f_back.f_lineno).zfill(4), data))

def display_error_output(real_data, expect_data, err_idx, relative_diff):
    print_log(
        'Error Line-----------------------------------------------------------------------------')
    print_log('Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff')
    print_log(
        '---------------------------------------------------------------------------------------')
    count = 0
    len_err = len(err_idx)
    for i in err_idx:
        count += 1
        if count < 10 or (90 < count < 100):
            print_log('%08d \t %.7f \t %.7f \t %.7f \t %.7f' % (
                i, expect_data[i], real_data[i], abs(np.float64(
                    expect_data[i]) - np.float64(real_data[i])),
                relative_diff[count - 1]))
        elif count == 10 or (count == 100 and len_err > 100):
            dot_3 = '...'
            print_log('%08s \t %07s \t %07s \t %07s \t %07s' %
                      (dot_3, dot_3, dot_3, dot_3, dot_3))
        elif count > 100:
            break

    print_log(
        'Max-RE line:---------------------------------------------------------------------------')
    max_error = max(relative_diff)
    m_idx_list = err_idx[np.where(relative_diff == max_error)]
    m_count = 0
    for m_idx in m_idx_list:
        m_count += 1
        if m_count < 4:
            print_log('%08d \t %.7f \t %.7f \t %.7f \t %.7f' % (
                m_idx, expect_data[m_idx], real_data[m_idx],
                abs(np.float64(expect_data[m_idx]) -
                    np.float64(real_data[m_idx])),
                max_error))
        else:
            break
    print_log(
        '---------------------------------------------------------------------------------------')

def _tensor_compare(npu_out, golden_out, name):
    """Compare a single tensor pair consistent with compressor_golden.py check_result.

    Logic:
      1. Empty check → Pass
      2. Size mismatch → Failed
      3. np.isclose(rtol, atol, equal_nan=True) per-element (cast to float32)
      4. Pass if fulfill_percent >= 99.5% (pct_thd=0.005)
      5. Additional fail if max relative error >= 10.0 (max_diff_hd)
    """
    npu = as_numpy(npu_out)
    golden = as_numpy(golden_out)

    real_data = npu.flatten().astype(np.float32)
    data_compe = golden.flatten().astype(np.float32)

    if real_data.size == 0 and real_data.size == data_compe.size:
        return {
            "pass": True,
            "precision": 100.0,
            "metrics": {"standard": "check_result", "name": name,
                        "pct_rlt": 100.0, "max_rel_err": 0.0, "err_count": 0},
            "error_info": None,
        }

    if real_data.size != data_compe.size:
        return {
            "pass": False,
            "precision": 0.0,
            "metrics": {"standard": "check_result", "name": name, "reason": "size mismatch",
                        "npu_size": int(real_data.size), "golden_size": int(data_compe.size)},
            "error_info": f"{name} size mismatch: npu={real_data.size}, golden={data_compe.size}",
        }

    dtype_str = str(golden.dtype).split('.')[-1]
    rtol, atol, pct_thd = _get_thresholds(dtype_str)
    if name == _OUTPUT_NAMES[3] or name == _OUTPUT_NAMES[4]:
        rtol = 0
        atol = 0
        pct_thd = 0

    diff_result = np.isclose(real_data, data_compe, rtol=rtol, atol=atol, equal_nan=True)
    err_idx = np.where(diff_result != True)[0]

    split_count = real_data.size
    fulfill_percent = float(split_count - err_idx.size) / float(split_count) * 100.0

    diff_abs = np.abs(data_compe - real_data)
    b1 = np.maximum(np.abs(real_data), np.abs(data_compe))
    b2 = float((1.0 / (1 << 14)) / _DIFF_THD)
    b = np.add(np.maximum(b1, b2), 10e-10)
    eps = 10e-10
    err_diff = diff_abs / (b + eps)
    err_diff = err_diff[err_idx]
    max_rel_err = float(np.max(err_diff[0:_MAX_ERROR_IDX])) if len(err_diff) > 0 else 0.0

    print_log(
        f"---------------------------------------check {name}------------------------------------------------")
    print_log('split_count:%s; max_diff_hd:%s;' %
              (float(split_count), _MAX_DIFF_HD))

    start = 0
    end = real_data.size - 1
    display_output_np_isclose(real_data, data_compe, start, end)
    is_pass = fulfill_percent >= (1 - pct_thd) * 100.0
    result = "pass" if is_pass else "Failed"

    if len(err_diff) > 0:
        max_error = max(err_diff[0:_MAX_ERROR_IDX])
        if max_error >= _MAX_DIFF_HD:
            result = "Failed"
            is_pass = False

    print_log(
        '---------------------------------------------------------------------------------------')
    print_log('Rtol   \t Atol   \t PctThd   \t PctRlt   \t Result')
    print_log(
        '---------------------------------------------------------------------------------------')
    print_log('%.4f    \t %.6f  \t %.2f%%   \t %.6f%%   \t %s' %
                (rtol, atol, (1 - pct_thd) * 100.0, fulfill_percent, result))
    if len(err_diff) > 0:
        print_log('Max-RelativeError is: %s. Threshold is: %s.' %
                    (max_error, _MAX_DIFF_HD))
    if result == "Failed":
        display_error_output(real_data, data_compe,
                                err_idx, err_diff[0:_MAX_ERROR_IDX])

    error_info = None
    if not is_pass:
        error_info = (
            f"{name} compare failed: fulfill_percent={fulfill_percent:.4f}%, "
            f"max_rel_err={max_rel_err:.4f}, err_count={int(err_idx.size)}"
        )

    return {
        "pass": is_pass,
        "precision": fulfill_percent,
        "diff_indices": err_idx[:1000].tolist() if len(err_idx) > 0 else [],
        "error_info": error_info,
        "metrics": {
            "standard": "check_result",
            "name": name,
            "rtol": rtol,
            "atol": atol,
            "pct_thd": (1 - pct_thd) * 100.0,
            "pct_rlt": fulfill_percent,
            "max_rel_err": max_rel_err,
            "max_diff_hd": _MAX_DIFF_HD,
            "err_count": int(err_idx.size),
        },
    }


def compare(*outputs, **kwargs):
    """Compare NPU outputs against golden references.

    Two capture modes are supported:

    state_cache captured as single tensor via output_tensor_indexes:
        outputs = [npu_cmp_kv, npu_state_cache, golden_cmp_kv, golden_state_cache]

    Fallback (no state_cache capture, NPU has 1 output):
        outputs = [npu_cmp_kv, golden_cmp_kv, golden_state_cache]
        Only cmp_kv is compared; state_cache outputs marked as N/A.
    """
    if len(outputs) < 2:
        return {"pass": False, "precision": "invalid",
                "error_info": "compare expects NPU outputs followed by golden outputs"}

    half = len(outputs) // 2
    npu_outputs = outputs[:half]
    golden_outputs = outputs[half:]

    cmp_kv_mask = kwargs.get('cmp_kv_mask', None)

    # Fallback: single NPU output (cmp_kv only) — compare cmp_kv, skip state_cache
    if half == 1:
        results = [_tensor_compare(npu_outputs[0][cmp_kv_mask], golden_outputs[0][cmp_kv_mask], "cmp_kv")]
        for i in range(1, len(golden_outputs)):
            name = _OUTPUT_NAMES[i] if i < len(_OUTPUT_NAMES) else f"output_{i}"
            results.append({
                "pass": True,
                "precision": "N/A",
                "error_info": f"{name}: NPU state_cache not captured",
            })
        return results

    npu_cmp_kv = npu_outputs[0]
    npu_state_cache = npu_outputs[1]
    cpu_cmp_kv = golden_outputs[0]
    cpu_state_cache = golden_outputs[1]

    update_kv = kwargs.get('update_kv', None)
    update_score = kwargs.get('update_score', None)

    npu_sub_outputs = [npu_cmp_kv]
    golden_sub_outputs = [cpu_cmp_kv]

    npu_sub_outputs.append(npu_state_cache[:, :, :npu_state_cache.shape[2]//2][update_kv])
    npu_sub_outputs.append(npu_state_cache[:, :, npu_state_cache.shape[2]//2:][update_score])
    npu_sub_outputs.append(npu_state_cache[:, :, :npu_state_cache.shape[2]//2][~update_kv])
    npu_sub_outputs.append(npu_state_cache[:, :, npu_state_cache.shape[2]//2:][~update_score])

    golden_sub_outputs.append(cpu_state_cache[:, :, :cpu_state_cache.shape[2]//2][update_kv])
    golden_sub_outputs.append(cpu_state_cache[:, :, cpu_state_cache.shape[2]//2:][update_score])
    golden_sub_outputs.append(cpu_state_cache[:, :, :cpu_state_cache.shape[2]//2][~update_kv])
    golden_sub_outputs.append(cpu_state_cache[:, :, cpu_state_cache.shape[2]//2:][~update_score])

    results = []
    for idx in range(len(golden_sub_outputs)):
        name = _OUTPUT_NAMES[idx]
        npu_out = npu_sub_outputs[idx]
        golden_out = golden_sub_outputs[idx]
        if npu_out is None and golden_out is None:
            results.append({
                "pass": True,
                "precision": "N/A",
                "error_info": f"{name} is None",
            })
        else:
            results.append(_tensor_compare(npu_out, golden_out, name))
    return results
