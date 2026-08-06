#!/usr/bin/python3
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

import datetime
import logging
import os
import sys

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s]-%(filename)s:%(lineno)04d - %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)


def _print_log(data=None, level='INFO'):
    print("[%s] [%s]-%s:%s - %s" % (
        datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"), level,
        os.path.basename(sys._getframe().f_back.f_code.co_filename),
        str(sys._getframe().f_back.f_lineno).zfill(4), data))


def cal_relative_diff_np_isclose(real_data, expect_data, type_str='fp16'):
    diff = abs(float(real_data) - float(expect_data))
    return diff / (np.abs(expect_data) + 10e-10)


def display_output_np_isclose(real_data, expect_data, start, end, expect_fp32_data=None):
    def display_inner(idx):
        j = idx + start
        diff_rate = cal_relative_diff_np_isclose(real_data[j], expect_data[j])
        if "inf" in str(expect_data[j]) or "nan" in str(expect_data[j]):
            diff_abs = "inf" if "inf" in str(expect_data[j]) else "nan"
            if expect_fp32_data is not None:
                _print_log('%08d \t %-7s \t %-7s \t %-7s \t %-7s \t %-7s' % (
                    start + idx + 1, expect_fp32_data[j], expect_data[j], real_data[j], diff_abs, diff_rate))
            else:
                _print_log('%08d \t %-7s \t %-7s \t %-7s \t %-7s' % (
                    start + idx + 1, expect_data[j], real_data[j], diff_abs, diff_rate))
        else:
            diff_abs = abs(np.float64(expect_data[j]) - np.float64(real_data[j]))
            if expect_fp32_data is not None:
                _print_log('%08d \t %0.7f \t %0.7f \t %0.7f \t %0.7f \t %0.7f' % (
                    start + idx + 1, expect_fp32_data[j], expect_data[j], real_data[j], diff_abs, diff_rate))
            else:
                _print_log('%08d \t %0.7f \t %0.7f \t %0.7f \t %0.7f' % (
                    start + idx + 1, expect_data[j], real_data[j], diff_abs, diff_rate))

    _print_log('---------------------------------------------------------------------------------------')
    if expect_fp32_data is not None:
        _print_log('Loop \t ExpFP32Out \t ExpFP16Out \t NPUOut \tFpDiff(min) \t RateDiff')
    else:
        _print_log('Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff')
    _print_log('---------------------------------------------------------------------------------------')
    split_count = int(end - start)
    if split_count <= 20:
        for i in range(split_count + 1):
            display_inner(i)
    else:
        for i in range(10):
            display_inner(i)
        _print_log('...   \t   ...   \t   ...   \t   ...    \t   ...')
        for i in range(split_count - 10 + 1, split_count + 1):
            display_inner(i)


def display_error_output(real_data, expect_data, err_idx, relative_diff):
    _print_log('Error Line-----------------------------------------------------------------------------')
    _print_log('Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff')
    _print_log('---------------------------------------------------------------------------------------')
    count = 0
    len_err = len(err_idx)
    for i in err_idx:
        count += 1
        if count < 10 or (90 < count < 100):
            _print_log('%08d \t %.7f \t %.7f \t %.7f \t %.7f' % (
                i, expect_data[i], real_data[i],
                abs(np.float64(expect_data[i]) - np.float64(real_data[i])),
                relative_diff[count - 1]))
        elif count == 10 or (count == 100 and len_err > 100):
            dot_3 = '...'
            _print_log('%08s \t %07s \t %07s \t %07s \t %07s' % (dot_3, dot_3, dot_3, dot_3, dot_3))
        elif count > 100:
            break

    _print_log('Max-RE line:---------------------------------------------------------------------------')
    max_error = max(relative_diff)
    m_idx_list = err_idx[np.where(relative_diff == max_error)]
    m_count = 0
    for m_idx in m_idx_list:
        m_count += 1
        if m_count < 4:
            _print_log('%08d \t %.7f \t %.7f \t %.7f \t %.7f' % (
                m_idx, expect_data[m_idx], real_data[m_idx],
                abs(np.float64(expect_data[m_idx]) - np.float64(real_data[m_idx])), max_error))
        else:
            break
    _print_log('---------------------------------------------------------------------------------------')


def check_result(expect, result, data_type, pct_thd=0.005):
    real_data = result.cpu().numpy()
    data_compe = expect.cpu().numpy()
    real_data = real_data.flatten()
    data_compe = data_compe.flatten()
    if real_data.size == 0 and real_data.size == data_compe.size:
        _print_log('The npu_output is [],and it is same as bm_output, the result of data_compare is "Pass"')
        return 100.0, "Pass"
    start = 0
    end = real_data.size - 1
    if end < start:
        end = start
    max_error = 0
    result_str = "Failed"

    if real_data.size != data_compe.size:
        _print_log('Error,the size of npu output[%s] and benchmark[%s] is not equal.' % (real_data.size, data_compe.size))
        return 0.0, result_str
    overflows_count = data_compe[np.isinf(data_compe)].size + data_compe[np.isnan(data_compe)].size
    if overflows_count > 0:
        _print_log('Overflow,size:%s,benchmark_output:%s, %s' % (
            overflows_count, data_compe[np.isinf(data_compe)][0:10], data_compe[np.isnan(data_compe)][0:10]))

    if data_type == 'bfloat16':
        diff_thd = 0.005
        max_diff_hd = 10.0
        rtol = 0.0078125
        atol = 0.0001
        max_error_idx = 10000000
    else:
        diff_thd = 0.005
        max_diff_hd = 10.0
        rtol = 0.005
        atol = 0.000025
        max_error_idx = 10000000

    split_count = int(end - start + 1) if end != start else 1
    _print_log('split_count:%s; max_diff_hd:%s;' % (float(split_count), max_diff_hd))

    if str(real_data.dtype) == 'bfloat16':
        diff_result = np.isclose(real_data.astype(np.float32), data_compe.astype(np.float32),
                                 rtol=rtol, atol=atol, equal_nan=True)
    elif str(real_data.dtype) == 'float8_e4m3fn':
        nan_mask = np.isnan(real_data)
        real_data[nan_mask] = 0
        real_data = np.frombuffer(real_data.tobytes(), dtype="uint8")
        nan_mask = np.isnan(data_compe)
        data_compe[nan_mask] = 0
        data_compe = np.frombuffer(data_compe.tobytes(), dtype="uint8")
        diff_result = np.isclose(real_data, data_compe, rtol=rtol, atol=atol, equal_nan=True)
    elif str(real_data.dtype) == 'float8_e5m2':
        nan_mask = np.isnan(real_data)
        real_data[nan_mask] = 0
        real_data[np.isposinf(real_data)] = 57344
        real_data[np.isneginf(real_data)] = -57344
        real_data = np.frombuffer(real_data.tobytes(), dtype="uint8")
        nan_mask = np.isnan(data_compe)
        data_compe[nan_mask] = 0
        data_compe[np.isposinf(data_compe)] = 57344
        data_compe[np.isneginf(data_compe)] = -57344
        data_compe = np.frombuffer(data_compe.tobytes(), dtype="uint8")
        diff_result = np.isclose(real_data, data_compe, rtol=rtol, atol=atol, equal_nan=True)
    else:
        diff_result = np.isclose(real_data, data_compe, rtol=rtol, atol=atol, equal_nan=True)
    err_idx = np.where(diff_result != np.array((True,)))[0]

    if str(data_compe.dtype) == 'bool':
        data_compe = data_compe.astype(np.int8)
        real_data = real_data.astype(np.int8)
    diff_abs = abs(data_compe - real_data)
    b1 = np.maximum(np.abs(real_data), np.abs(data_compe))
    b2 = float((1.0 / (1 << 14)) / diff_thd)
    b = np.add(np.maximum(b1, b2), 10e-10)
    eps = 10e-10
    err_diff = diff_abs / (b + eps)
    err_diff = err_diff[err_idx]

    fulfill_percent = float(split_count - err_idx.size) / float(split_count) * 100.0
    display_output_np_isclose(real_data, data_compe, start, end)
    pct_thd = (1 - pct_thd) * 100.0
    result_str = "Pass" if (fulfill_percent >= pct_thd) else "Failed"
    if len(err_diff) > 0:
        max_error = max(err_diff[0:max_error_idx])
        if max_error >= max_diff_hd:
            result_str = "Failed"
    _print_log('---------------------------------------------------------------------------------------')
    _print_log('Rtol   \t Atol   \t PctThd   \t PctRlt   \t Result')
    _print_log('---------------------------------------------------------------------------------------')
    _print_log('%.4f    \t %.6f  \t %.2f%%   \t %.6f%%   \t %s' % (rtol, atol, pct_thd, fulfill_percent, result_str))
    if len(err_diff) > 0:
        _print_log('Max-RelativeError is: %s. Threshold is: %s.' % (max_error, max_diff_hd))
    if result_str == "Failed":
        display_error_output(real_data, data_compe, err_idx, err_diff[0:max_error_idx])
    return fulfill_percent, result_str


def compare(golden_name, golden_data, real_name, real_data, src_dtype):
    compare_percent = 0
    compare_result = None
    if golden_data is not None and real_data is not None:
        print(f"--------------------------------------------------------------{golden_name} vs {real_name}-------------------------------------------------------------")
        compare_percent, compare_result = check_result(
            golden_data.to(torch.float32), real_data.cpu().to(torch.float32), src_dtype)
    return compare_percent, compare_result
