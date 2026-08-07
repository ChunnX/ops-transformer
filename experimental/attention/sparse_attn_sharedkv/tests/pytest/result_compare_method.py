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

import logging
import torch
import datetime
import os
import sys
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)
logger = logging.getLogger(__name__)


def cal_relative_diff_np_isclose(real_data, expect_data, type_str='fp16'):
    diff = abs(float(real_data) - float(expect_data))
    result = diff / (np.abs(expect_data) + 10e-10)
    return result


def display_output_np_isclose(real_data, expect_data, start, end, expect_fp32_data=None):
    def display_inner(idx):
        j = idx + start
        diff_rate = cal_relative_diff_np_isclose(real_data[j], expect_data[j])

        if "inf" in str(expect_data[j]) or "nan" in str(expect_data[j]):
            diff_abs = "inf" if "inf" in str(expect_data[j]) else "nan"
            if expect_fp32_data is not None:
                print_log('%08d \t %-7s \t %-7s \t %-7s \t %-7s \t %-7s' % (
                    start + idx, expect_fp32_data[j], expect_data[j], real_data[j], diff_abs, diff_rate))
            else:
                print_log('%08d \t %-7s \t %-7s \t %-7s \t %-7s' % (
                    start + idx, expect_data[j], real_data[j], diff_abs, diff_rate))
        else:
            diff_abs = abs(np.float64(expect_data[j]) - np.float64(real_data[j]))
            if expect_fp32_data is not None:
                print_log('%08d \t %0.7f \t %0.7f \t %0.7f \t %0.7f \t %0.7f' % (
                    start + idx, expect_fp32_data[j], expect_data[j], real_data[j], diff_abs, diff_rate))
            else:
                print_log('%08d \t %0.7f \t %0.7f \t %0.7f \t %0.7f' % (
                    start + idx, expect_data[j], real_data[j], diff_abs, diff_rate))

    print_log('---------------------------------------------------------------------------------------')
    if expect_fp32_data is not None:
        print_log('Loop \t ExpFP32Out \t ExpFP16Out \t NPUOut \tFpDiff(min) \t RateDiff')
    else:
        print_log('Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff')
    print_log('---------------------------------------------------------------------------------------')

    split_count = int(end - start)
    if split_count <= 20:
        for i in range(split_count + 1):
            display_inner(i)
    else:
        for i in range(10):
            display_inner(i)
        print_log('... \t ... \t ... \t ... \t ...')
        for i in range(split_count - 10 + 1, split_count + 1):
            display_inner(i)


def print_log(data=None, level='INFO'):
    print("[%s] [%s]-%s:%s - %s" % (datetime.datetime.now().strftime(
        "%Y/%m/%d %H:%M:%S"), level, os.path.basename(sys._getframe().f_back.f_code.co_filename),
        str(sys._getframe().f_back.f_lineno).zfill(4), data))


def display_error_output(real_data, expect_data, err_idx, relative_diff):
    print_log('Error Line-----------------------------------------------------------------------------')
    print_log('Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff')
    print_log('---------------------------------------------------------------------------------------')

    count = 0
    len_err = len(err_idx)
    for i in err_idx:
        count += 1
        if count < 10 or (90 < count < 100):
            print_log('%08d \t %.7f \t %.7f \t %.7f \t %.7f' % (
                i, expect_data[i], real_data[i],
                abs(np.float64(expect_data[i]) - np.float64(real_data[i])),
                relative_diff[count - 1]))
        elif count == 10 or (count == 100 and len_err > 100):
            dot_3 = '...'
            print_log('%08s \t %07s \t %07s \t %07s \t %07s' %
                      (dot_3, dot_3, dot_3, dot_3, dot_3))
        elif count > 100:
            break

    print_log('Max-RE line:---------------------------------------------------------------------------')
    max_error = max(relative_diff)
    m_idx_list = err_idx[np.where(relative_diff == max_error)]
    m_count = 0
    for m_idx in m_idx_list:
        m_count += 1
        if m_count < 4:
            print_log('%08d \t %.7f \t %.7f \t %.7f \t %.7f' % (
                m_idx, expect_data[m_idx], real_data[m_idx],
                abs(np.float64(expect_data[m_idx]) - np.float64(real_data[m_idx])),
                max_error))
        else:
            break
    print_log('---------------------------------------------------------------------------------------')


def display_error_distribution(err_idx, real_data, expect_data, output_shape=None):
    """打印错误在 (token, head, dim) 坐标空间中的分布"""
    if output_shape is None or len(output_shape) != 3:
        return

    S1, N1, D = output_shape
    N1D = N1 * D

    print_log('=== Error distribution by token ===')
    tokens = err_idx // N1D
    unique_tokens, counts = np.unique(tokens, return_counts=True)
    for t, c in zip(unique_tokens[:40], counts[:40]):
        print_log('  token=%d: %d errors' % (t, c))
    if len(unique_tokens) > 40:
        print_log('  ... (%d more tokens)' % (len(unique_tokens) - 40))

    print_log('=== Error distribution by head ===')
    heads = (err_idx % N1D) // D
    unique_heads, counts = np.unique(heads, return_counts=True)
    for h, c in zip(unique_heads[:40], counts[:40]):
        print_log('  head=%d: %d errors' % (h, c))
    if len(unique_heads) > 40:
        print_log('  ... (%d more heads)' % (len(unique_heads) - 40))

    print_log('=== Error distribution by dim ===')
    dims = err_idx % D
    unique_dims, counts = np.unique(dims, return_counts=True)
    for d_val, c in zip(unique_dims[:20], counts[:20]):
        print_log('  dim=%d: %d errors' % (d_val, c))
    if len(unique_dims) > 20:
        print_log('  ... (%d more dims)' % (len(unique_dims) - 20))

    print_log('=== Top-10 error tokens with most errors ===')
    sorted_idx = np.argsort(counts)[::-1]
    for rank, idx in enumerate(sorted_idx[:10]):
        t = unique_tokens[idx]
        print_log('  #%d token=%d: %d errors' % (rank + 1, t, counts[idx]))

    print_log('=== Sample errors with coordinates (first 20) ===')
    for k, idx in enumerate(err_idx[:20]):
        t = idx // N1D
        h = (idx % N1D) // D
        d = idx % D
        print_log('  idx=%d -> (token=%d, head=%d, dim=%d)  expect=%.4f  real=%.4f' % (
            idx, t, h, d, expect_data[idx], real_data[idx]))


def check_result(expect, npu_result, output_shape=None):
    """
    output_shape: 可选，形状元组如 (S1, N1, D)，提供后会打印错误坐标分布
    """
    diff_thd = 0.005
    pct_thd = 0.005
    max_diff_hd = 10
    rtol = 0.005
    atol = 0.000025
    max_error_idx = 10000000

    real_data = npu_result.cpu().to(torch.float32).numpy()
    data_compe = expect.cpu().to(torch.float32).numpy()
    real_data = real_data.flatten()
    data_compe = data_compe.flatten()

    if real_data.size == 0 and real_data.size == data_compe.size:
        print_log('The npu_output is [],and it is same as bm_output, the result of data_compare is "Pass"')
        return "Pass", 100.0, 0

    start = 0
    end = real_data.size - 1
    if end < start:
        end = start

    max_error = 0
    result = "Failed"

    if real_data.size != data_compe.size:
        print_log('Error,the size of npu output[%s] and benchmark[%s] is not equal.' % (
            real_data.size, data_compe.size))
        return result, 0.0, max_error

    overflows_count = data_compe[np.isinf(data_compe)].size + data_compe[np.isnan(data_compe)].size
    if overflows_count > 0:
        print_log('Overflow,size:%s,benchmark_output:%s, %s' % (
            overflows_count, data_compe[np.isinf(data_compe)][0:10], data_compe[np.isnan(data_compe)][0:10]))

    split_count = int(end - start + 1) if end != start else 1
    print_log('split_count:%s; max_diff_hd:%s;' % (float(split_count), max_diff_hd))

    if npu_result.dtype == torch.bfloat16:
        rtol = 0.0078125
        atol = 0.0001

    diff_result = np.isclose(real_data.astype(np.float32), data_compe.astype(np.float32),
                             rtol=rtol, atol=atol, equal_nan=True)
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
    result = "Pass" if (fulfill_percent >= pct_thd) else "Failed"

    if len(err_diff) > 0:
        max_error = max(err_diff[0:max_error_idx])
        if max_error >= max_diff_hd:
            result = "Failed"

    print_log('---------------------------------------------------------------------------------------')
    print_log('Rtol \t Atol \t PctThd \t PctRlt \t Result')
    print_log('---------------------------------------------------------------------------------------')
    print_log('%.4f \t %.6f \t %.2f%% \t %.6f%% \t %s' % (
        rtol, atol, pct_thd, fulfill_percent, result))

    if len(err_diff) > 0:
        print_log('Max-RelativeError is: %s. Threshold is: %s.' % (max_error, max_diff_hd))

    if result == "Failed":
        display_error_output(real_data, data_compe, err_idx, err_diff[0:max_error_idx])
        if output_shape is not None:
            display_error_distribution(err_idx, real_data, data_compe, output_shape)

    return result, fulfill_percent