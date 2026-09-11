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

import math
import random
import logging
import torch
import datetime
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)


def cal_relative_diff_isclose(real_data, expect_data, type_str="fp16"):
    diff = abs(float(real_data) - float(expect_data))
    result = diff / (abs(float(expect_data)) + 10e-10)
    return result


def display_output_isclose(real_data, expect_data, start, end, expect_fp32_data=None):
    def display_inner(idx):
        j = idx + start
        diff_rate = cal_relative_diff_isclose(real_data[j], expect_data[j])

        if "inf" in str(expect_data[j].item()) or "nan" in str(expect_data[j].item()):
            diff_abs = (
                "inf" if "inf" in str(expect_data[j].item()) else "nan"
            )
            if expect_fp32_data is not None:
                print_log(
                    "%08d \t %-7s \t %-7s \t %-7s \t %-7s \t %-7s"
                    % (
                        start + idx + 1,
                        expect_fp32_data[j].item(),
                        expect_data[j].item(),
                        real_data[j].item(),
                        diff_abs,
                        diff_rate,
                    )
                )
            else:
                print_log(
                    "%08d \t %-7s \t %-7s \t %-7s \t %-7s"
                    % (
                        start + idx + 1,
                        expect_data[j].item(),
                        real_data[j].item(),
                        diff_abs,
                        diff_rate,
                    )
                )
        else:
            diff_abs = abs(
                float(expect_data[j].item()) - float(real_data[j].item())
            )
            if expect_fp32_data is not None:
                print_log(
                    "%08d \t %0.7f \t %0.7f \t %0.7f \t %0.7f \t %0.7f"
                    % (
                        start + idx + 1,
                        expect_fp32_data[j].item(),
                        expect_data[j].item(),
                        real_data[j].item(),
                        diff_abs,
                        diff_rate,
                    )
                )
            else:
                print_log(
                    "%08d \t %0.7f \t %0.7f \t %0.7f \t %0.7f"
                    % (
                        start + idx + 1,
                        expect_data[j].item(),
                        real_data[j].item(),
                        diff_abs,
                        diff_rate,
                    )
                )

    print_log(
        "---------------------------------------------------------------------------------------"
    )
    if expect_fp32_data is not None:
        print_log(
            "Loop \t ExpFP32Out \t ExpFP16Out \t NPUOut \tFpDiff(min) \t RateDiff"
        )
    else:
        print_log("Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff")
    print_log(
        "---------------------------------------------------------------------------------------"
    )
    split_count = int(end - start)
    if split_count <= 20:
        for i in range(split_count + 1):
            display_inner(i)
    else:
        for i in range(10):
            display_inner(i)
        print_log("...   \t   ...   \t   ...   \t   ...    \t   ...")
        for i in range(split_count - 10 + 1, split_count + 1):
            display_inner(i)


def print_log(data=None, level="INFO"):
    print(
        "[%s] [%s]-%s:%s - %s"
        % (
            datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
            level,
            os.path.basename(sys._getframe().f_back.f_code.co_filename),
            str(sys._getframe().f_back.f_lineno).zfill(4),
            data,
        )
    )


def display_error_output(real_data, expect_data, err_idx, relative_diff):
    print_log(
        "Error Line-----------------------------------------------------------------------------"
    )
    print_log("Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff")
    print_log(
        "---------------------------------------------------------------------------------------"
    )
    count = 0
    len_err = len(err_idx)
    for i in err_idx:
        count += 1
        if count < 10 or (90 < count < 100):
            print_log(
                "%08d \t %.7f \t %.7f \t %.7f \t %.7f"
                % (
                    i,
                    expect_data[i].item(),
                    real_data[i].item(),
                    abs(float(expect_data[i].item()) - float(real_data[i].item())),
                    relative_diff[count - 1],
                )
            )
        elif count == 10 or (count == 100 and len_err > 100):
            dot_3 = "..."
            print_log(
                "%08s \t %07s \t %07s \t %07s \t %07s"
                % (dot_3, dot_3, dot_3, dot_3, dot_3)
            )
        elif count > 100:
            break

    print_log(
        "Max-RE line:---------------------------------------------------------------------------"
    )
    max_error = max(relative_diff).item() if isinstance(max(relative_diff), torch.Tensor) else max(relative_diff)
    m_idx = (relative_diff == max_error).nonzero(as_tuple=False).flatten()
    m_count = 0
    for m_idx_val in m_idx:
        m_count += 1
        if m_count < 4:
            orig_idx = err_idx[m_idx_val]
            print_log(
                "%08d \t %.7f \t %.7f \t %.7f \t %.7f"
                % (
                    orig_idx.item(),
                    expect_data[orig_idx].item(),
                    real_data[orig_idx].item(),
                    abs(
                        float(expect_data[orig_idx].item())
                        - float(real_data[orig_idx].item())
                    ),
                    max_error,
                )
            )
        else:
            break
    print_log(
        "---------------------------------------------------------------------------------------"
    )


def torch_isclose_with_nan(real_data, data_compe, rtol, atol):
    nan_mask = torch.isnan(real_data) & torch.isnan(data_compe)
    real_data_clean = torch.nan_to_num(real_data, nan=0.0)
    data_compe_clean = torch.nan_to_num(data_compe, nan=0.0)
    diff_result = torch.isclose(
        real_data_clean, data_compe_clean, rtol=rtol, atol=atol
    )
    diff_result[nan_mask] = True
    return diff_result


# fuzz 中 precision_method == 1的精度对比方式
def check_result(expect, npu_result):
    diff_thd = 0.005
    pct_thd = 0.005
    max_diff_hd = 10
    rtol = 0.005
    atol = 0.000025
    max_error_idx = 10000000

    real_data = npu_result.cpu().to(torch.float).flatten()
    data_compe = expect.cpu().to(torch.float).flatten()
    if real_data.numel() == 0 and real_data.numel() == data_compe.numel():
        print_log(
            'The npu_output is [],and it is same as bm_output, the result of data_compare is "Pass"'
        )
        return "Pass", 100.0, 0
    start = 0
    end = real_data.numel() - 1
    if end < start:
        end = start
    max_error = 0
    result = "Failed"

    if real_data.numel() != data_compe.numel():
        print_log(
            "Error,the size of npu output[%s] and benchmark[%s] is not equal."
            % (real_data.numel(), data_compe.numel())
        )
        return result, 0.0, max_error
    overflows_count = (
        torch.isinf(data_compe).sum().item()
        + torch.isnan(data_compe).sum().item()
    )

    if overflows_count > 0:
        inf_vals = data_compe[torch.isinf(data_compe)][:10]
        nan_vals = data_compe[torch.isnan(data_compe)][:10]
        print_log(
            "Overflow,size:%s,benchmark_output:%s, %s"
            % (overflows_count, inf_vals.tolist(), nan_vals.tolist())
        )

    split_count = int(end - start + 1) if end != start else 1
    print_log("split_count:%s; max_diff_hd:%s;" % (float(split_count), max_diff_hd))

    if npu_result.dtype == torch.bfloat16:
        rtol = 0.0078125
        atol = 0.0001
        diff_result = torch_isclose_with_nan(
            real_data, data_compe, rtol=rtol, atol=atol
        )
    elif npu_result.dtype == torch.float8_e4m3fn:
        real_data_nan = torch.isnan(real_data)
        real_data[real_data_nan] = 0
        real_data_uint8 = real_data.contiguous().view(torch.uint8)
        data_compe_nan = torch.isnan(data_compe)
        data_compe[data_compe_nan] = 0
        data_compe_uint8 = data_compe.contiguous().view(torch.uint8)
        diff_result = torch_isclose_with_nan(
            real_data_uint8.float(), data_compe_uint8.float(), rtol=rtol, atol=atol
        )
    elif npu_result.dtype == torch.float8_e5m2:
        real_data_nan = torch.isnan(real_data)
        real_data[real_data_nan] = 0
        real_data_posinf = torch.isposinf(real_data)
        real_data[real_data_posinf] = 57344
        real_data_neginf = torch.isneginf(real_data)
        real_data[real_data_neginf] = -57344
        real_data_uint8 = real_data.contiguous().view(torch.uint8)

        data_compe_nan = torch.isnan(data_compe)
        data_compe[data_compe_nan] = 0
        data_compe_posinf = torch.isposinf(data_compe)
        data_compe[data_compe_posinf] = 57344
        data_compe_neginf = torch.isneginf(data_compe)
        data_compe[data_compe_neginf] = -57344
        data_compe_uint8 = data_compe.contiguous().view(torch.uint8)

        diff_result = torch_isclose_with_nan(
            real_data_uint8.float(), data_compe_uint8.float(), rtol=rtol, atol=atol
        )
    else:
        diff_result = torch_isclose_with_nan(
            real_data, data_compe, rtol=rtol, atol=atol
        )
    err_idx = torch.nonzero(~diff_result, as_tuple=False).flatten()

    if str(data_compe.dtype) == "bool":
        data_compe = data_compe.to(torch.int8)
        real_data = real_data.to(torch.int8)
    diff_abs = torch.abs(data_compe - real_data)
    b1 = torch.maximum(torch.abs(real_data), torch.abs(data_compe))
    b2 = float((1.0 / (1 << 14)) / diff_thd)
    b = torch.maximum(b1, torch.tensor(b2, dtype=b1.dtype)) + 10e-10
    eps = 10e-10
    err_diff = diff_abs / (b + eps)
    err_diff = err_diff[err_idx]

    fulfill_percent = float(split_count - err_idx.numel()) / float(split_count) * 100.0

    display_output_isclose(real_data, data_compe, start, end)
    pct_thd = (1 - pct_thd) * 100.0
    result = "Pass" if (fulfill_percent >= pct_thd) else "Failed"
    if err_diff.numel() > 0:
        max_error = float(err_diff[:max_error_idx].max().item())
        if max_error >= max_diff_hd:
            result = "Failed"
    print_log(
        "---------------------------------------------------------------------------------------"
    )
    print_log("Rtol   \t Atol   \t PctThd   \t PctRlt   \t Result")
    print_log(
        "---------------------------------------------------------------------------------------"
    )
    print_log(
        "%.4f    \t %.6f  \t %.2f%%   \t %.6f%%   \t %s"
        % (rtol, atol, pct_thd, fulfill_percent, result)
    )
    if err_diff.numel() > 0:
        print_log(
            "Max-RelativeError is: %s. Threshold is: %s." % (max_error, max_diff_hd)
        )
    if result == "Failed":
        display_error_output(real_data, data_compe, err_idx, err_diff[:max_error_idx])
    return result, fulfill_percent
