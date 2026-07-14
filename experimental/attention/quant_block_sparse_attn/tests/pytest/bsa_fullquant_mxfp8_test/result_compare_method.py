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
import math
import torch

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s]-%(filename)s:%(lineno)04d - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    force=True
)
logger = logging.getLogger(__name__)

_EPS = 1e-10
_TABLE_WIDTH = 90
_BASE_HEADER = "{:>10} {:>18} {:>18} {:>18} {:>14}"
_BASE_ROW = "{:>10} {:>18} {:>18} {:>18} {:>14}"
_FP32_HEADER = "{:>10} {:>18} {:>18} {:>18} {:>18} {:>14}"
_FP32_ROW = "{:>10} {:>18} {:>18} {:>18} {:>18} {:>14}"


def cal_relative_diff_torch(real_data, expect_data):
    diff = abs(float(real_data) - float(expect_data))
    return diff / (abs(float(expect_data)) + _EPS)


def _format_float(value):
    if isinstance(value, str):
        return value
    if not math.isfinite(float(value)):
        return str(float(value))
    return f"{float(value):.7e}"


def _display_offsets(total_count):
    if total_count <= 20:
        return list(range(total_count))
    return list(range(10)) + [None] + list(range(total_count - 10, total_count))


def display_output_torch(real_data, expect_data, start, end, expect_fp32_data=None):
    has_fp32 = expect_fp32_data is not None

    logger.info('-' * (_TABLE_WIDTH + (22 if has_fp32 else 0)))
    if has_fp32:
        logger.info(_FP32_HEADER.format("Loop", "ExpFP32Out", "ExpectOut", "RealOut", "AbsDiff", "RateDiff"))
    else:
        logger.info(_BASE_HEADER.format("Loop", "ExpectOut", "RealOut", "AbsDiff", "RateDiff"))
    logger.info('-' * (_TABLE_WIDTH + (22 if has_fp32 else 0)))

    total_count = int(end - start + 1)

    def display_row(offset):
        j = start + offset
        real_value = float(real_data[j])
        expect_value = float(expect_data[j])
        diff_rate = cal_relative_diff_torch(real_value, expect_value)
        idx = j + 1

        if not torch.isfinite(expect_data[j]).item():
            diff_abs = "inf" if torch.isinf(expect_data[j]).item() else "nan"
        else:
            diff_abs = abs(expect_value - real_value)

        if has_fp32:
            logger.info(_FP32_ROW.format(
                idx, _format_float(float(expect_fp32_data[j])), _format_float(expect_value),
                _format_float(real_value), _format_float(diff_abs), _format_float(diff_rate)))
        else:
            logger.info(_BASE_ROW.format(
                idx, _format_float(expect_value), _format_float(real_value),
                _format_float(diff_abs), _format_float(diff_rate)))

    for offset in _display_offsets(total_count):
        if offset is None:
            logger.info(_FP32_ROW.format("...", "...", "...", "...", "...", "...")
                        if has_fp32 else _BASE_ROW.format("...", "...", "...", "...", "..."))
            continue
        display_row(offset)


def display_error_output_torch(real_data, expect_data, err_idx, relative_diff):
    logger.info('Error Line' + '-' * (_TABLE_WIDTH - len('Error Line')))
    logger.info(_BASE_HEADER.format("Loop", "ExpectOut", "RealOut", "AbsDiff", "RateDiff"))
    logger.info('-' * _TABLE_WIDTH)

    len_err = int(err_idx.numel())
    for offset in _display_offsets(len_err):
        if offset is None:
            logger.info(_BASE_ROW.format("...", "...", "...", "...", "..."))
            continue
        i = int(err_idx[offset].item())
        expect_value = float(expect_data[i])
        real_value = float(real_data[i])
        logger.info(_BASE_ROW.format(
            i, _format_float(expect_value), _format_float(real_value),
            _format_float(abs(expect_value - real_value)), _format_float(float(relative_diff[offset].item()))))

    logger.info('Max-RE line:' + '-' * (_TABLE_WIDTH - len('Max-RE line:')))
    max_error = float(torch.max(relative_diff).item()) if relative_diff.numel() > 0 else 0.0
    m_idx_list = err_idx[relative_diff == max_error]
    for m_count, m_idx in enumerate(m_idx_list.tolist()):
        if m_count >= 3:
            break
        expect_value = float(expect_data[m_idx])
        real_value = float(real_data[m_idx])
        logger.info(_BASE_ROW.format(
            m_idx, _format_float(expect_value), _format_float(real_value),
            _format_float(abs(expect_value - real_value)), _format_float(max_error)))
    logger.info('-' * _TABLE_WIDTH)


def check_result(expect, npu_result):
    diff_thd = 0.005
    pct_thd = 0.005
    max_diff_hd = 10
    rtol = 0.005
    atol = 0.000025

    real_data = npu_result.detach().cpu().flatten()
    expect_data = expect.detach().cpu().flatten() if isinstance(expect, torch.Tensor) else torch.as_tensor(expect).flatten()

    if real_data.numel() != expect_data.numel():
        logger.info('Error,the size of npu output[%s] and benchmark[%s] is not equal.',
                    real_data.numel(), expect_data.numel())
        return "Failed", 0.0, 0

    if real_data.numel() == 0:
        logger.info('The npu_output and benchmark are both [], the result of data_compare is "Pass"')
        return "Pass", 100.0, 0

    start = 0
    end = real_data.numel() - 1
    if end < start:
        end = start
    split_count = end - start + 1 if end != start else 1
    max_error = 0.0

    overflows_count = int(torch.isinf(expect_data).sum().item() + torch.isnan(expect_data).sum().item())
    if overflows_count > 0:
        logger.info('Overflow,size:%s,benchmark_output:%s, %s',
                    overflows_count,
                    expect_data[torch.isinf(expect_data)][0:10],
                    expect_data[torch.isnan(expect_data)][0:10])

    logger.info('split_count:%s; max_diff_hd:%s;', float(split_count), max_diff_hd)

    if expect_data.dtype == torch.bool:
        expect_data = expect_data.to(torch.int8)
        real_data = real_data.to(torch.int8)

    real_float = real_data.to(torch.float32)
    expect_float = expect_data.to(torch.float32)

    if npu_result.dtype == torch.bfloat16:
        rtol = 0.0078125
        atol = 0.0001

    diff_result = torch.isclose(real_float, expect_float, rtol=rtol, atol=atol, equal_nan=True)
    err_idx = torch.nonzero(~diff_result, as_tuple=False).flatten()

    diff_abs = torch.abs(expect_float - real_float)
    b1 = torch.maximum(torch.abs(real_float), torch.abs(expect_float))
    b2 = float((1.0 / (1 << 14)) / diff_thd)
    b = torch.maximum(b1, torch.tensor(b2, dtype=torch.float32)) + _EPS
    err_diff = (diff_abs / b)[err_idx]
    err_diff = torch.where(torch.isnan(err_diff) | torch.isinf(err_diff),
                           torch.tensor(float('inf'), dtype=torch.float32), err_diff)

    fulfill_percent = float(split_count - err_idx.numel()) / float(split_count) * 100.0

    display_output_torch(real_data, expect_data, start, end)

    pct_thd = (1 - pct_thd) * 100.0
    result = "Pass" if (fulfill_percent >= pct_thd) else "Failed"

    if err_diff.numel() > 0:
        max_error = float(torch.max(err_diff).item())
        if max_error >= max_diff_hd:
            result = "Failed"

    logger.info('-' * _TABLE_WIDTH)
    logger.info("{:>10} {:>12} {:>12} {:>14} {:>10}".format("Rtol", "Atol", "PctThd", "PctRlt", "Result"))
    logger.info('-' * _TABLE_WIDTH)
    logger.info("{:>10.4f} {:>12.6f} {:>11.2f}% {:>13.6f}% {:>10}".format(
        rtol, atol, pct_thd, fulfill_percent, result))

    if err_diff.numel() > 0:
        logger.info('Max-RelativeError is: %s. Threshold is: %s.', max_error, max_diff_hd)

    if result == "Failed":
        display_error_output_torch(real_data, expect_data, err_idx, err_diff)

    return result, fulfill_percent, max_error
