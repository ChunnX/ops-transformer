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
import os
import sys

logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)
logger = logging.getLogger(__name__)

def cal_relative_diff_torch(real_data, expect_data):
    diff = abs(float(real_data) - float(expect_data))
    return diff / (abs(float(expect_data)) + 10e-10)


def display_output_torch(real_data, expect_data, start, end, expect_fp32_data=None):
    COL_W = 12  # 数据列宽度
    def display_inner(idx):
        j = idx + start
        real_value = float(real_data[j])
        expect_value = float(expect_data[j])
        diff_rate = cal_relative_diff_torch(real_value, expect_value)

        if not torch.isfinite(expect_data[j]).item():
            diff_abs = "inf" if torch.isinf(expect_data[j]).item() else "nan"
            if expect_fp32_data is not None:
                print_log('%08d  %*s  %*s  %*s  %*s  %*s' % (
                    start + idx + 1, COL_W, float(expect_fp32_data[j]), COL_W, expect_value,
                    COL_W, real_value, COL_W, diff_abs, COL_W, diff_rate))
            else:
                print_log('%08d  %*s  %*s  %*s  %*s' % (
                    start + idx + 1, COL_W, expect_value, COL_W, real_value,
                    COL_W, diff_abs, COL_W, diff_rate))
        else:
            diff_abs = abs(expect_value - real_value)
            if expect_fp32_data is not None:
                print_log('%08d  %*.7f  %*.7f  %*.7f  %*.7f  %*.7f' % (
                    start + idx + 1, COL_W, float(expect_fp32_data[j]), COL_W, expect_value,
                    COL_W, real_value, COL_W, diff_abs, COL_W, diff_rate))
            else:
                print_log('%08d  %*.7f  %*.7f  %*.7f  %*.7f' % (
                    start + idx + 1, COL_W, expect_value, COL_W, real_value,
                    COL_W, diff_abs, COL_W, diff_rate))

    SEP = '-' * 90
    if expect_fp32_data is not None:
        print_log(SEP)
        print_log('%-8s  %*s  %*s  %*s  %*s  %*s' % (
            'Loop', COL_W, 'ExpFP32Out', COL_W, 'ExpFP16Out', COL_W, 'NPUOut',
            COL_W, 'FpDiff(min)', COL_W, 'RateDiff'))
        print_log(SEP)
    else:
        print_log(SEP)
        print_log('%-8s  %*s  %*s  %*s  %*s' % (
            'Loop', COL_W, 'ExpectOut', COL_W, 'RealOut', COL_W, 'FpDiff', COL_W, 'RateDiff'))
        print_log(SEP)
    split_count = int(end - start)
    if split_count <= 20:
        for i in range(split_count + 1):
            display_inner(i)
    else:
        for i in range(10):
            display_inner(i)
        if expect_fp32_data is not None:
            print_log('%-8s  %*s  %*s  %*s  %*s  %*s' % ('...', COL_W, '...', COL_W, '...', COL_W, '...', COL_W, '...', COL_W, '...'))
        else:
            print_log('%-8s  %*s  %*s  %*s  %*s' % ('...', COL_W, '...', COL_W, '...', COL_W, '...', COL_W, '...'))
        for i in range(split_count - 10 + 1, split_count + 1):
            display_inner(i)

def print_log(data=None, level='INFO'):
    print("[%s]-%s:%s - %s" % (level, os.path.basename(sys._getframe().f_back.f_code.co_filename),
                               str(sys._getframe().f_back.f_lineno).zfill(4), data))

def display_error_output_torch(real_data, expect_data, err_idx, relative_diff):
    COL_W = 12
    SEP = '-' * 90
    print_log('Error Line' + '-' * 90)
    print_log('%-8s  %*s  %*s  %*s  %*s' % ('Loop', COL_W, 'ExpectOut', COL_W, 'RealOut', COL_W, 'FpDiff', COL_W, 'RateDiff'))
    print_log(SEP)
    count = 0
    len_err = int(err_idx.numel())
    for i in err_idx.tolist():
        count += 1
        if count < 10 or (90 < count < 100):
            expect_value = float(expect_data[i])
            real_value = float(real_data[i])
            print_log('%08d  %*.7f  %*.7f  %*.7f  %*.7f' % (
                i, COL_W, expect_value, COL_W, real_value,
                COL_W, abs(expect_value - real_value),
                COL_W, float(relative_diff[count - 1].item())))
        elif count == 10 or (count == 100 and len_err > 100):
            print_log('%-8s  %*s  %*s  %*s  %*s' % ('...', COL_W, '...', COL_W, '...', COL_W, '...', COL_W, '...'))
        elif count > 100:
            break

    print_log('Max-RE line:' + '-' * 87)
    max_error = float(torch.max(relative_diff).item()) if relative_diff.numel() > 0 else 0.0
    m_idx_list = err_idx[relative_diff == max_error]
    m_count = 0
    for m_idx in m_idx_list.tolist():
        m_count += 1
        if m_count < 4:
            expect_value = float(expect_data[m_idx])
            real_value = float(real_data[m_idx])
            print_log('%08d  %*.7f  %*.7f  %*.7f  %*.7f' % (
                m_idx, COL_W, expect_value, COL_W, real_value,
                COL_W, abs(expect_value - real_value), COL_W, max_error))
        else:
            break
    print_log(SEP)
# fuzz 中 precision_method == 1的精度对比方式
def check_result(expect, npu_result):
    diff_thd=0.005
    pct_thd=0.005
    max_diff_hd=10
    rtol=0.005
    atol=0.000025
    max_error_idx = 10000000

    real_data = npu_result.detach().cpu()
    data_compe = expect.detach().cpu() if isinstance(expect, torch.Tensor) else torch.as_tensor(expect)
    real_data = real_data.flatten()
    data_compe = data_compe.flatten()
    if real_data.numel() == 0 and real_data.numel() == data_compe.numel():
        print_log(
            'The npu_output is [],and it is same as bm_output, the result of data_compare is \"Pass\"')
        return "Pass", 100.0, 0
    start = 0
    end = real_data.numel() - 1
    if end < start:
        end = start
    max_error = 0
    result = "Failed"

    if real_data.numel() != data_compe.numel():
        print_log(
            'Error,the size of npu output[%s] and benchmark[%s] is not equal.' % (real_data.numel(), data_compe.numel()))
        return result, 0.0, max_error
    overflows_count = int(torch.isinf(data_compe).sum().item() + torch.isnan(data_compe).sum().item())


    if overflows_count > 0:
        print_log('Overflow,size:%s,benchmark_output:%s, %s' % (
            overflows_count, data_compe[torch.isinf(data_compe)][0:10], data_compe[torch.isnan(data_compe)][0:10]))


    split_count = int(end - start + 1) if end != start else 1
    print_log('split_count:%s; max_diff_hd:%s;' %
              (float(split_count), max_diff_hd))

    real_float = real_data.to(torch.float32)
    expect_float = data_compe.to(torch.float32)

    if npu_result.dtype == torch.bfloat16:
        rtol=0.0078125
        atol=0.0001
    diff_result = torch.isclose(real_float, expect_float, rtol=rtol, atol=atol, equal_nan=True)
    err_idx = torch.nonzero(~diff_result, as_tuple=False).flatten()

    if data_compe.dtype == torch.bool:
        data_compe = data_compe.to(torch.int8)
        real_data = real_data.to(torch.int8)
        real_float = real_data.to(torch.float32)
        expect_float = data_compe.to(torch.float32)
    diff_abs = torch.abs(expect_float - real_float)
    b1 = torch.maximum(torch.abs(real_float), torch.abs(expect_float))
    b2 = float((1.0 / (1 << 14)) / diff_thd)
    b = torch.maximum(b1, torch.tensor(b2, dtype=torch.float32)) + 10e-10
    eps = 10e-10
    err_diff = diff_abs / (b + eps)
    err_diff = err_diff[err_idx]

    fulfill_percent = float(split_count - err_idx.numel()) / \
                        float(split_count) * 100.0

    display_output_torch(real_data, data_compe, start, end)
    pct_thd = (1 - pct_thd) * 100.0
    result = "Pass" if (fulfill_percent >= pct_thd) else "Failed"
    if err_diff.numel() > 0:
        max_error = float(torch.max(err_diff[0:max_error_idx]).item())
        if max_error >= max_diff_hd:
            result = "Failed"
    COL_W = 12
    print_log('-' * 90)
    print_log('%-8s  %*s  %*s  %*s  %*s' % ('Rtol', COL_W, 'Atol', COL_W, 'PctThd', COL_W, 'PctRlt', COL_W, 'Result'))
    print_log('-' * 90)
    print_log('%.4f   %*.6f  %*.2f%%   %*.6f%%   %-s' %
                (rtol, COL_W - 2, atol, COL_W - 2, pct_thd, COL_W - 2, fulfill_percent, result))
    if err_diff.numel() > 0:
        print_log('Max-RelativeError is: %s. Threshold is: %s.' %
                    (max_error, max_diff_hd))
    if result == "Failed":
        display_error_output_torch(real_data, data_compe,
                                err_idx[0:max_error_idx], err_diff[0:max_error_idx])
    return result, fulfill_percent