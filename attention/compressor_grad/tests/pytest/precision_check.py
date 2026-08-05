#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import numpy as np
import datetime
import os
import sys
import logging


np.set_printoptions(suppress=True)

logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)
logger = logging.getLogger(__name__)


# ================================================================
#  精度阈值配置
# ================================================================
#   rtol/atol:    np.isclose 的相对/绝对容差
#   pct_thd:      通过率阈值，0.005 表示 99.5% 的点位需满足 rtol/atol
#   diff_thd:     相对误差归一化分母系数（b = max(|real|,|expect|, 1/(2^14)/diff_thd)）
#   max_diff_hd:  单点最大允许相对误差阈值，超过即判 Failed
PRECISION = {
    "bfloat16": {"rtol": 0.0078125, "atol": 0.0001, "pct_thd": 0.005,
                 "diff_thd": 0.005, "max_diff_hd": 10.0},
    "float16":  {"rtol": 0.005, "atol": 0.000025, "pct_thd": 0.005,
                 "diff_thd": 0.005, "max_diff_hd": 10.0},
    "float32":  {"rtol": 0.005, "atol": 0.000025, "pct_thd": 0.005,
                 "diff_thd": 0.005, "max_diff_hd": 10.0},
}


# ================================================================
#  输出定义
# ================================================================
# 反向算子所有输出列表（顺序固定，便于汇总统计）
ALL_OUTPUTS = ["d_wkv", "d_wgate", "d_ape", "d_x"]

# 输出名 → result dict 中的 status/pct key 映射
_OUTPUT_KEY_MAP = {
    "d_wkv":   ("dwkvPct",   "dwkvStatus"),
    "d_wgate":   ("dwgatePct",   "dwgateStatus"),
    "d_ape": ("apePct",   "apeStatus"),
    "d_x":   ("dxPct",    "dxStatus"),
}


def get_output_keys(out_name):
    """返回输出名对应的 (pct_key, status_key)。"""
    return _OUTPUT_KEY_MAP[out_name]


def get_pct_thd(data_type):
    """根据 dtype 字符串返回通过率阈值 pct_thd。"""
    return PRECISION[data_type]["pct_thd"]


# ================================================================
#  日志输出
# ================================================================
def print_log(data=None, level='INFO'):
    print("[%s] [%s]-%s:%s - %s" % (datetime.datetime.now().strftime(
        "%Y/%m/%d %H:%M:%S"), level, os.path.basename(sys._getframe().f_back.f_code.co_filename),
                                     str(sys._getframe().f_back.f_lineno).zfill(4), data))


# ================================================================
#  单点位精度对比展示
# ================================================================
def cal_relative_diff_np_isclose(real_data, expect_data):
    diff = abs(float(real_data) - float(expect_data))
    result = diff / (np.abs(expect_data) + 10e-10)
    return result


def display_output_np_isclose(real_data, expect_data, start, end):
    """输出每个点位的对比结果（≤20 点位全显示，>20 显示首尾各 10）。"""
    def display_inner(idx):
        j = idx + start
        diff_rate = cal_relative_diff_np_isclose(real_data[j], expect_data[j])

        if "inf" in str(expect_data[j]) or "nan" in str(expect_data[j]):
            diff_abs = "inf" if "inf" in str(expect_data[j]) else "nan"
            print_log('%08d \t %-7s \t %-7s \t %-7s \t %-7s' % (
                start + idx + 1, expect_data[j], real_data[j], diff_abs, diff_rate))
        else:
            diff_abs = abs(np.float64(expect_data[j]) - np.float64(real_data[j]))
            print_log('%08d \t %0.7f \t %0.7f \t %0.7f \t %0.7f' % (
                start + idx + 1, expect_data[j], real_data[j], diff_abs, diff_rate))

    print_log('---------------------------------------------------------------------------------------')
    print_log('Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff')
    print_log('---------------------------------------------------------------------------------------')
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


def display_error_output(real_data, expect_data, err_idx, relative_diff):
    """输出错误点位明细 + 最大相对误差点位。"""
    print_log('Error Line-----------------------------------------------------------------------------')
    print_log('Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff')
    print_log('---------------------------------------------------------------------------------------')
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

    print_log('Max-RE line:---------------------------------------------------------------------------')
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
    print_log('---------------------------------------------------------------------------------------')


# ================================================================
#  单输出精度对比主函数
# ================================================================
def check_result(expect, result, data_type, pct_thd=0.005):
    """单输出精度对比，复用自正向 compressor_golden.check_result。

    Args:
        expect:    CPU golden 结果 (torch.Tensor)
        result:    NPU 反向算子输出 (torch.Tensor)
        data_type: dtype 字符串 ('bfloat16' / 'float16' / 'float32')
        pct_thd:   通过率阈值（默认 0.005 表示 99.5%）

    Returns:
        fulfill_percent: 满足精度要求的百分比
        result_str:      "Pass" / "Failed"
    """
    real_data = result.cpu().numpy()
    data_compe = expect.cpu().numpy()
    real_data = real_data.flatten()
    data_compe = data_compe.flatten()
    if real_data.size == 0 and real_data.size == data_compe.size:
        print_log('The npu_output is [],and it is same as bm_output, the result of data_compare is "Pass"')
        return 100.0, "Pass"
    start = 0
    end = real_data.size - 1
    if end < start:
        end = start
    max_error = 0
    result_str = "Failed"

    if real_data.size != data_compe.size:
        print_log('Error,the size of npu output[%s] and benchmark[%s] is not equal.' % (real_data.size, data_compe.size))
        return 0.0, result_str
    overflows_count = data_compe[np.isinf(data_compe)].size + data_compe[np.isnan(data_compe)].size

    if overflows_count > 0:
        print_log('Overflow,size:%s,benchmark_output:%s, %s' % (
            overflows_count, data_compe[np.isinf(data_compe)][0:10], data_compe[np.isnan(data_compe)][0:10]))

    # 检测 NPU 输出 (real_data) 中的 NaN/Inf
    real_data_f32 = real_data.astype(np.float32) if str(real_data.dtype) == 'bfloat16' else real_data
    real_nan_mask = np.isnan(real_data_f32)
    real_inf_mask = np.isinf(real_data_f32)
    real_overflow_mask = real_nan_mask | real_inf_mask
    real_overflow_count = int(real_overflow_mask.sum())
    if real_overflow_count > 0:
        print_log('NPU output has NaN/Inf, count:%s, nan:%s, inf:%s, sample_values:%s' % (
            real_overflow_count, int(real_nan_mask.sum()), int(real_inf_mask.sum()),
            real_data_f32[real_overflow_mask][0:10]))

    # 仅支持 bfloat16 / float16 / float32，阈值统一从 PRECISION 配置获取
    cfg = PRECISION[data_type]
    diff_thd = cfg["diff_thd"]
    max_diff_hd = cfg["max_diff_hd"]
    rtol = cfg["rtol"]
    atol = cfg["atol"]
    max_error_idx = 10000000

    split_count = int(end - start + 1) if end != start else 1
    print_log('split_count:%s; max_diff_hd:%s;' % (float(split_count), max_diff_hd))

    # bfloat16 需转 float32 再 isclose；float16 / float32 直接 isclose
    # 注意：使用 equal_nan=False，使 NPU 输出中的 NaN 与 CPU 期望不匹配时判为错误
    if str(real_data.dtype) == 'bfloat16':
        diff_result = np.isclose(real_data.astype(np.float32), data_compe.astype(np.float32),
                                 rtol=rtol, atol=atol, equal_nan=False)
    else:
        diff_result = np.isclose(real_data, data_compe, rtol=rtol, atol=atol, equal_nan=False)
    err_idx = np.where(diff_result != np.array((True,)))[0]

    if str(data_compe.dtype) == 'bool':
        data_compe = data_compe.astype(np.int8)
        real_data = real_data.astype(np.int8)
    diff_abs = abs(data_compe - real_data)
    b1 = np.maximum(np.abs(real_data), (np.abs(data_compe)))
    b2 = float((1.0 / (1 << 14)) / diff_thd)
    b = np.add(np.maximum(b1, b2), 10e-10)
    eps = 10e-10
    err_diff = diff_abs / (b + eps)
    err_diff = err_diff[err_idx]

    fulfill_percent = float(split_count - err_idx.size) / float(split_count) * 100.0

    # NPU 输出含 NaN/Inf 时直接判 Failed（NaN 会传染，结果不可信）
    if real_overflow_count > 0:
        result_str = "Failed"
        max_error = float('inf')
        display_output_np_isclose(real_data, data_compe, start, end)
        pct_thd = (1 - pct_thd) * 100.0
        print_log('---------------------------------------------------------------------------------------')
        print_log('Rtol   \t Atol   \t PctThd   \t PctRlt   \t Result')
        print_log('---------------------------------------------------------------------------------------')
        print_log('%.4f    \t %.6f  \t %.2f%%   \t %.6f%%   \t %s' %
                  (rtol, atol, pct_thd, fulfill_percent, result_str))
        print_log('NPU output contains NaN/Inf (count=%s). Force Failed.' % real_overflow_count)
        if len(err_diff) > 0:
            display_error_output(real_data, data_compe, err_idx, err_diff[0:max_error_idx])
        return fulfill_percent, result_str

    display_output_np_isclose(real_data, data_compe, start, end)
    pct_thd = (1 - pct_thd) * 100.0
    result_str = "Pass" if (fulfill_percent >= pct_thd) else "Failed"
    if len(err_diff) > 0:
        # 过滤 NaN/Inf 后再取 max，避免 NaN 导致 max_error 比较失效
        finite_err_diff = err_diff[np.isfinite(err_diff)]
        if len(finite_err_diff) > 0:
            max_error = float(np.max(finite_err_diff[0:max_error_idx]))
            if max_error >= max_diff_hd:
                result_str = "Failed"
    print_log('---------------------------------------------------------------------------------------')
    print_log('Rtol   \t Atol   \t PctThd   \t PctRlt   \t Result')
    print_log('---------------------------------------------------------------------------------------')
    print_log('%.4f    \t %.6f  \t %.2f%%   \t %.6f%%   \t %s' %
              (rtol, atol, pct_thd, fulfill_percent, result_str))
    if len(err_diff) > 0:
        print_log('Max-RelativeError is: %s. Threshold is: %s.' % (max_error, max_diff_hd))
    if result_str == "Failed":
        display_error_output(real_data, data_compe, err_idx, err_diff[0:max_error_idx])
    return fulfill_percent, result_str


def check_one_output(name, expect, result, data_type, enabled, total_valid, pct_thd=0.005):
    """对比单个反向输出，未开启对比的返回 SKIP 标记。

    Args:
        name:        输出名 (如 'd_ape', 'd_x', 'd_wkv', 'd_wgate')
        expect:      CPU golden 结果 (torch.Tensor)
        result:      NPU 反向算子输出 (torch.Tensor)
        data_type:   dtype 字符串 ('bfloat16' / 'float16' / 'float32')
        enabled:     是否进行精度对比
        total_valid: 该用例的有效压缩块数（0 时无有效数据，自动 SKIP）
        pct_thd:     通过率阈值

    Returns:
        dict: {diff, pct, status}
    """
    if not enabled:
        return {"diff": float("nan"), "pct": 100.0, "status": "SKIP"}
    if total_valid == 0:
        return {"diff": float("nan"), "pct": 100.0, "status": "SKIP"}

    print_log("=" * 80)
    print_log(f"check {name}:")
    print_log("--------------------------------------------------------------")
    try:
        if expect is None or result is None:
            print_log(f"{name}: expect or result is None, skip")
            return {"diff": float("nan"), "pct": 100.0, "status": "SKIP"}
        if expect.shape != result.shape:
            print_log(f"{name}: shape mismatch expect={expect.shape} result={result.shape}, flatten compare")
        fulfill_percent, result_str = check_result(expect, result, data_type, pct_thd)
        max_abs_diff = abs(expect.cpu().float().flatten() - result.cpu().float().flatten()).max().item()
        return {"diff": max_abs_diff, "pct": fulfill_percent, "status": "PASS" if result_str == "Pass" else "FAIL"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"diff": float("nan"), "pct": 0.0, "status": "ERROR"}


# ================================================================
#  结果格式化与汇总统计
# ================================================================
def format_output_status(result, out_name):
    """格式化单个输出的状态字符串（用于用例行展示）。"""
    pct_key, stat_key = get_output_keys(out_name)
    st = result.get(stat_key, "SKIP")
    pct = result.get(pct_key, 100.0)
    if st == "SKIP":
        return f"{out_name}=SKIP"
    elif st == "ERROR":
        return f"{out_name}=ERROR"
    else:
        return f"{out_name}={pct:.2f}%[{st}]"


def format_case_line(result):
    """格式化单个用例的完整结果行（含 4 个输出的对比状态）。"""
    parts = [format_output_status(result, out_name) for out_name in ALL_OUTPUTS]
    detail = "  ".join(parts)
    return f"  {result.get('status', 'N/A'):5s}  {detail}  (validBlocks={result.get('totalValid', 0)})"


def build_error_result(name):
    """构建异常用例的 result dict，所有输出标记为 ERROR。"""
    err_result = dict(name=name, status="ERROR")
    for out_name in ALL_OUTPUTS:
        _, stat_key = get_output_keys(out_name)
        err_result[stat_key] = "ERROR"
    return err_result


def print_summary(results):
    """输出汇总统计：用例通过个数 + 每个用例每个输出的对比结果（以 case 为粒度）。

    输出两部分：
      1. SUMMARY: 总体 PASS/FAIL/SKIP/ERROR 个数统计
      2. Per-Case Per-Output Summary: 以 case 为粒度的表格，
         每行一个用例，每列一个输出，明确看出每个 case 的对比结果
    """
    print("\n" + "=" * 100)
    passed  = sum(1 for r in results if r.get("status") == "PASS")
    failed  = sum(1 for r in results if r.get("status") == "FAIL")
    skipped = sum(1 for r in results if r.get("status") == "SKIP")
    errors  = sum(1 for r in results if r.get("status") == "ERROR")
    print(f"SUMMARY: {passed} PASS, {failed} FAIL, {skipped} SKIP, "
          f"{errors} ERROR out of {len(results)}")
    print("=" * 100)

    # 以 case 为粒度的每个输出对比结果
    print("\nPer-Case Per-Output Summary:")
    print("-" * 100)
    header = f"{'Case':<45}" + "".join(f"{o:<18}" for o in ALL_OUTPUTS) + "Overall"
    print(header)
    print("-" * 100)
    for r in results:
        case_name = r.get("name", "unknown")
        row = f"{case_name:<45}"
        for out_name in ALL_OUTPUTS:
            _, stat_key = get_output_keys(out_name)
            st = r.get(stat_key, "SKIP")
            if st == "SKIP":
                row += f"{'SKIP':<18}"
            elif st == "ERROR":
                row += f"{'ERROR':<18}"
            else:
                pct_key, _ = get_output_keys(out_name)
                pct = r.get(pct_key, 0.0)
                row += f"{pct:.2f}%[{st}]".ljust(18)
        row += r.get("status", "N/A")
        print(row)
    print("=" * 100)
