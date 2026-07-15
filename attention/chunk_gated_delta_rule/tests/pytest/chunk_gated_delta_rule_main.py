# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import time
import torch
import torch_npu
from cann_ops_transformer.ops import chunk_gated_delta_rule
from chunk_gated_delta_rule_benchmark import chunk_gdn_benchmark_opt
from chunk_gated_delta_rule_golden import chunk_gated_delta_rule_npu
from chunk_gated_delta_rule_torch import cgdr_torch
import torch.nn.functional as F
import numpy as np
import logging
import datetime
import os
import sys
import argparse
import gc

np.random.seed(21)
np.set_printoptions(suppress=True)
DEVICE_ID = 0
torch.npu.config.allow_internal_format = True
logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)
logger = logging.getLogger(__name__)

# ======================== 三方对比相关阈值/常量 ========================
eb_threshold = 2**(-8)
err_threshold = 2**(-8)
CV_MAX_RE = 5               # 最大相对误差
CV_AVER_RE = 1.5            # 平均相对误差
CV_RMSE = 1.5               # 均方根误差
CV_SMALL_VAL = 2            # 小值域错误占比
CV_ERR_BALANCE = 2          # 误差均衡性
MIN_ERR = 1e-3


# 定义ANSI颜色常量（新增）
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'    # 绿色
    WARNING = '\033[93m'
    FAIL = '\033[91m'       # 红色
    ENDC = '\033[0m'        # 重置颜色
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_log(data=None, level='INFO'):
    print("[%s] [%s]-%s:%s - %s" % (datetime.datetime.now().strftime(
        "%Y/%m/%d %H:%M:%S"), level, os.path.basename(sys._getframe().f_back.f_code.co_filename),
                                    str(sys._getframe().f_back.f_lineno).zfill(4), data))


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


def check_result(expect, result, data_type, pct_thd=0.005):
    real_data = result.cpu().numpy()
    data_compe = expect.cpu().numpy()
    real_data = real_data.flatten()
    data_compe = data_compe.flatten()
    if real_data.size == 0 and real_data.size == data_compe.size:
        print_log(
            'The npu_output is [],and it is same as bm_output, the result of data_compare is \"Pass\"')
        return 100.0, "Pass"
    start = 0
    end = real_data.size - 1
    if end < start:
        end = start
    max_error = 0
    result = "Failed"

    if real_data.size != data_compe.size:
        print_log(
            'Error,the size of npu output[%s] and benchmark[%s] is not equal.' % (real_data.size, data_compe.size))
        return 0.0, result
    overflows_count = data_compe[np.isinf(data_compe)].size + data_compe[np.isnan(data_compe)].size

    if overflows_count > 0:
        print_log('Overflow,size:%s,benchmark_output:%s, %s' % (
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
    print_log('split_count:%s; max_diff_hd:%s;' %
              (float(split_count), max_diff_hd))

    has_nan_inf = False
    if 'nan' in str(real_data) or 'inf' in str(real_data) or 'nan' in str(data_compe) or 'inf' in str(data_compe):
        has_nan_inf = True

    if str(real_data.dtype) == 'bfloat16':
        diff_result = np.isclose(real_data.astype(np.float32), data_compe.astype(np.float32), rtol=rtol, atol=atol,
                                 equal_nan=True)
    elif str(real_data.dtype) == 'float8_e4m3fn':
        nan_mask = np.isnan(real_data)
        real_data[nan_mask] = 0
        arr_string = real_data.tobytes()
        real_data = np.frombuffer(arr_string, dtype="uint8")
        nan_mask = np.isnan(data_compe)
        data_compe[nan_mask] = 0
        arr_string = data_compe.tobytes()
        data_compe = np.frombuffer(arr_string, dtype="uint8")
        diff_result = np.isclose(real_data, data_compe, rtol=rtol, atol=atol, equal_nan=True)
    elif str(real_data.dtype) == 'float8_e5m2':
        nan_mask = np.isnan(real_data)
        real_data[nan_mask] = 0
        nan_pos_inf = np.isposinf(real_data)
        real_data[nan_pos_inf] = 57344
        nan_neg_inf = np.isneginf(real_data)
        real_data[nan_neg_inf] = -57344

        arr_string = real_data.tobytes()
        real_data = np.frombuffer(arr_string, dtype="uint8")
        nan_mask = np.isnan(data_compe)
        data_compe[nan_mask] = 0
        nan_pos_inf = np.isposinf(data_compe)
        data_compe[nan_pos_inf] = 57344
        nan_neg_inf = np.isneginf(data_compe)
        data_compe[nan_neg_inf] = -57344

        arr_string = data_compe.tobytes()
        data_compe = np.frombuffer(arr_string, dtype="uint8")
        diff_result = np.isclose(real_data, data_compe, rtol=rtol, atol=atol, equal_nan=True)
    else:
        diff_result = np.isclose(real_data, data_compe, rtol=rtol, atol=atol, equal_nan=True)
    err_idx = np.where(diff_result != np.array((True,)))[0]
    del diff_result
    gc.collect()

    if str(data_compe.dtype) == 'bool':
        data_compe = data_compe.astype(np.int8)
        real_data = real_data.astype(np.int8)
    diff_abs = abs(data_compe - real_data)
    b1 = np.maximum(np.abs(real_data), (np.abs(data_compe)))
    b2 = float((1.0 / (1 << 14)) / diff_thd)
    b = np.add(np.maximum(b1, b2), 10e-10)
    del b1
    eps = 10e-10
    b += eps
    np.divide(diff_abs, b, out=diff_abs)
    del b
    err_diff = diff_abs[err_idx]
    del diff_abs
    gc.collect()

    fulfill_percent = float(split_count - err_idx.size) / \
        float(split_count) * 100.0

    display_output_np_isclose(real_data, data_compe, start, end)
    pct_thd = (1 - pct_thd) * 100.0
    result = "Pass" if (fulfill_percent >= pct_thd) else "Failed"
    if len(err_diff) > 0:
        max_error = max(err_diff[0:max_error_idx])
        if max_error >= max_diff_hd:
            result = "Failed"
    print_log(
        '---------------------------------------------------------------------------------------')
    print_log('Rtol   \t Atol   \t PctThd   \t PctRlt   \t Result')
    print_log(
        '---------------------------------------------------------------------------------------')
    print_log('%.4f    \t %.6f  \t %.2f%%   \t %.6f%%   \t %s' %
              (rtol, atol, pct_thd, fulfill_percent, result))
    if len(err_diff) > 0:
        print_log('Max-RelativeError is: %s. Threshold is: %s.' %
                  (max_error, max_diff_hd))
    if result == "Failed":
        display_error_output(real_data, data_compe,
                             err_idx, err_diff[0:max_error_idx])
    del real_data, data_compe, err_diff, err_idx
    gc.collect()
    return fulfill_percent, result


# ======================== 三方对比函数（golden / benchmark / npu） ========================
def get_max_re(golden: torch.Tensor, actual: torch.Tensor):
    # 最大相对误差
    abs_error = torch.abs(actual - golden) / (torch.abs(golden) + MIN_ERR)
    max_re = torch.max(abs_error.flatten())
    return max_re


def get_avg_re(golden: torch.Tensor, actual: torch.Tensor):
    # 平均相对误差
    abs_error = torch.abs(actual - golden) / (torch.abs(golden) + MIN_ERR)
    avg_re = torch.mean(abs_error)
    return avg_re


def get_rmse(golden: torch.Tensor, actual: torch.Tensor):
    # 均方根误差
    sqr_err = torch.pow((actual - golden), 2)
    rmse = torch.sqrt(torch.mean(sqr_err))
    return rmse


def get_smra(golden: torch.Tensor, actual: torch.Tensor):
    # 小值域错误占比
    abs_A = torch.abs(golden)
    mask_A = abs_A < 2**(-10)
    num_a = torch.sum(mask_A).item()

    # 统计对应位置 B 中元素绝对值大于 1e-16 的个数
    abs_B = torch.abs(golden - actual)
    mask_B = abs_B > 1e-16
    num_b = torch.sum(mask_A & mask_B).item()
    
    smra = num_b / num_a if num_a > 0 else 0
    return smra


def get_eb(golden: torch.Tensor, actual: torch.Tensor):
    # 误差均衡性
    golden_nmax = torch.clamp(torch.abs(golden), min=1)
    actual_error = actual - golden
    error_balance = torch.mean(actual_error / golden_nmax)
    return error_balance


def compare_cv(golden: torch.Tensor, golden_high_type: torch.Tensor, actual: torch.Tensor, name=None):
    golden = golden.to(torch.float32)
    golden_high_type = golden_high_type.to(torch.float32)
    actual = actual.to(torch.float32)
    # 最大相对误差
    max_re_npu = get_max_re(golden, actual)
    max_re_high_type = get_max_re(golden, golden_high_type)
    print(f"{max_re_npu=}, {max_re_high_type=}")
    # 平均相对误差
    avg_re_npu = get_avg_re(golden, actual)
    avg_re_high_type = get_avg_re(golden, golden_high_type)
    # 均方根误差
    rmse_npu = get_rmse(golden, actual)
    rmse_high_type = get_rmse(golden, golden_high_type)
    # 小值域错误占比
    smra_npu = get_smra(golden, actual)
    smra_high_type = get_smra(golden, golden_high_type)
    max_re_rate = max_re_npu / max(max_re_high_type, err_threshold)
    avg_re_rate = avg_re_npu / max(avg_re_high_type, err_threshold)
    rmse_rate = rmse_npu / max(rmse_high_type, err_threshold)
    smra_rate = smra_npu / max(smra_high_type, err_threshold)
    # 误差均衡性
    EB = get_eb(golden_high_type, actual)

    if name is not None:
        print(f"compare_cv for {name}:")
    print(f"\tmax_re_rate={max_re_rate:.3f} ({CV_MAX_RE}), max_re_high_type={max_re_high_type:.3e}")
    print(f"\tavg_re_rate={avg_re_rate:.3f} ({CV_AVER_RE}), avg_re_high_type={avg_re_high_type:.3e}")
    print(f"\trmse_rate={rmse_rate:.3f} ({CV_RMSE}), rmse_high_type={rmse_high_type:.3e}")
    print(f"\tsmra_rate={smra_rate:.3f} ({CV_SMALL_VAL}), smra_high_type={smra_high_type:.3e}")
    result = (max_re_rate < CV_MAX_RE) and (avg_re_rate < CV_AVER_RE) and (rmse_rate < CV_RMSE)
    result = result and smra_rate < CV_SMALL_VAL
    if not result:
        epsilon = 2.0**-7
        if max_re_npu < epsilon:
            print(f"\t max_re_npu={max_re_npu} less than {epsilon}.")
            result = True
    return result


def cgdr_golden(q, k, v, g, beta, scale, initial_state, actual_seq_lengths, use_float64=False, chunk_size=64):
    t0 = time.time()
    cu_seqlens = F.pad(actual_seq_lengths, (1, 0)).cumsum(dim=0)
    if use_float64:
        v = v.cpu().to(torch.float64)
    else:
        v = v.to(torch.float32)
    if g is None:
        g = torch.zeros((v.shape[0], v.shape[1])).to(v.device).to(v.dtype)
    o_golden, state_golden = chunk_gated_delta_rule_npu(
        q.unsqueeze(0).to(v.device).to(v.dtype),
        k.unsqueeze(0).to(v.device).to(v.dtype),
        v.unsqueeze(0).to(v.device).to(v.dtype),
        g.unsqueeze(0).to(v.device).to(v.dtype),
        beta.unsqueeze(0).to(v.device).to(v.dtype),
        scale=scale,
        initial_state=initial_state.transpose(-1, -2).clone().to(v.device).to(v.dtype),
        cu_seqlens=cu_seqlens.to(v.device),
        chunk_size=chunk_size,
    )
    o_golden = o_golden[0]
    state_golden = state_golden.transpose(-1, -2)
    print(f"cgdr_golden {use_float64=} time cost: {time.time() - t0} s")
    return o_golden.to(torch.float32).npu(), state_golden.to(torch.float32).npu()


def cgdr_npu(q, k, v, g, beta, scale, initial_state, actual_seq_lengths):
    o_npu, state_npu = torch.ops.cann_ops_transformer.chunk_gated_delta_rule(
    # o_npu, state_npu = torch_npu.npu_chunk_gated_delta_rule(
        q, k, v,
        beta=beta,
        initial_state=initial_state.clone(),
        actual_seq_lengths=actual_seq_lengths,
        scale=scale,
        g=g
    )
    o_npu = o_npu.to(torch.float32)
    state_npu = state_npu.to(torch.float32)
    return o_npu, state_npu


def cgdr_benchmark(q, k, v, g, beta, scale, initial_state, actual_seq_lengths, chunk_size=64):
    dtype = torch.bfloat16
    if g is None:
        g = torch.zeros((v.shape[0], v.shape[1])).to(v.device).to(torch.float32)
    o_bench, state_bench = chunk_gdn_benchmark_opt(
        q.to(dtype),
        k.to(dtype),
        v.to(dtype),
        beta.to(dtype),
        scale,
        initial_state.to(dtype),
        actual_seq_lengths,
        g,
        chunk_size=chunk_size,
    )
    o_bench = o_bench.to(torch.float32)
    state_bench = state_bench.to(torch.float32)
    return o_bench, state_bench


def run_chunk_gated_delta_rule_eager(B, seqlen, nk, nv, dk, dv, chunk_size=64,
                                     data_type=torch.bfloat16,
                                     state_data_type=torch.bfloat16,
                                     has_g=True,
                                     is_continue=True,
                                     perf_runs=0):
    torch_npu.npu.set_device(int(DEVICE_ID))
    # ======================== gen input data start =============================
    if isinstance(seqlen, (list, tuple)):
        assert len(seqlen) == B, f"seqlen array length {len(seqlen)} != B {B}"
        actual_seq_lengths = torch.tensor(seqlen, dtype=torch.int32, device="npu:%s" % DEVICE_ID)
        T = sum(seqlen)
    else:
        actual_seq_lengths = torch.tensor([seqlen] * B, dtype=torch.int32, device="npu:%s" % DEVICE_ID)
        T = B * seqlen
    q = torch.rand((T, nk, dk), dtype=data_type, device="npu:%s" % DEVICE_ID)
    k = torch.rand((T, nk, dk), dtype=data_type, device="npu:%s" % DEVICE_ID)
    v = torch.rand((T, nv, dv), dtype=data_type, device="npu:%s" % DEVICE_ID)
    if has_g:
        g = torch.rand((T, nv), dtype=torch.float32, device="npu:%s" % DEVICE_ID) * -1.0
    else:
        g = None
    beta = torch.rand((T, nv), dtype=data_type, device="npu:%s" % DEVICE_ID)
    q = torch.nn.functional.normalize(q, p=2, dim=-1)
    k = torch.nn.functional.normalize(k, p=2, dim=-1)
    scale = 1 / (dk ** 0.5)
    initial_state = torch.rand((B, nv, dv, dk), dtype=state_data_type, device="npu:%s" % DEVICE_ID)
    # ======================== gen input data finish =============================

    if perf_runs > 0:
        torch_npu.npu.synchronize()
        for _ in range(perf_runs):
            cgdr_npu(q, k, v, g, beta, scale, initial_state, actual_seq_lengths)
            torch_npu.npu.synchronize()
        print(f"perf test done, {perf_runs} runs completed")
        return True

    # # ======================== execute golden/npu ================================
    o_golden, state_golden = cgdr_torch(q, k, v, g, beta, scale, initial_state, actual_seq_lengths,
                                         use_float64=False, chunk_size=chunk_size)
    o_npu, state_npu = cgdr_npu(q, k, v, g, beta, scale, initial_state, actual_seq_lengths)
    # ======================== check result ================================
    ret = True
    data_type_str = str(data_type)
    print("--------------------------------------------------------------check result-------------------------------------------------------------")
    _, o_result = check_result(o_golden.to(torch.float32), o_npu.to(torch.float32), data_type_str)
    if o_result != "Pass":
        print("compare o failed.")
        ret = False
    print("--------------------------------------------------------------check state ouput-------------------------------------------------------------")
    _, state_result = check_result(state_golden.to(torch.float32), state_npu.to(torch.float32), data_type_str)
    if state_result != "Pass":
        print("compare state failed.")
        ret = False
    print("PASSED" if ret else "FAILED")
    return ret


def run_chunk_gated_delta_rule_eager_three_way(B, seqlen, nk, nv, dk, dv, chunk_size=64,
                                               data_type=torch.bfloat16,
                                               state_data_type=torch.bfloat16,
                                               has_g=True,
                                               is_continue=True):
    torch_npu.npu.set_device(int(DEVICE_ID))
    # ======================== gen input data start =============================
    if isinstance(seqlen, (list, tuple)):
        assert len(seqlen) == B, f"seqlen array length {len(seqlen)} != B {B}"
        actual_seq_lengths = torch.tensor(seqlen, dtype=torch.int32, device="npu:%s" % DEVICE_ID)
        T = sum(seqlen)
    else:
        actual_seq_lengths = torch.tensor([seqlen] * B, dtype=torch.int32, device="npu:%s" % DEVICE_ID)
        T = B * seqlen
    q = torch.rand((T, nk, dk), dtype=data_type, device="npu:%s" % DEVICE_ID)
    k = torch.rand((T, nk, dk), dtype=data_type, device="npu:%s" % DEVICE_ID)
    v = torch.rand((T, nv, dv), dtype=data_type, device="npu:%s" % DEVICE_ID)
    if has_g:
        g = torch.rand((T, nv), dtype=torch.float32, device="npu:%s" % DEVICE_ID) * -1.0
    else:
        g = None
    beta = torch.rand((T, nv), dtype=data_type, device="npu:%s" % DEVICE_ID)
    q = torch.nn.functional.normalize(q, p=2, dim=-1)
    k = torch.nn.functional.normalize(k, p=2, dim=-1)
    scale = 1 / (dk ** 0.5)
    initial_state = torch.rand((B, nv, dv, dk), dtype=state_data_type, device="npu:%s" % DEVICE_ID)
    # ======================== gen input data finish =============================

    # ======================== execute golden/benchmark/npu ================================
    o_golden, state_golden = cgdr_torch(q, k, v, g, beta, scale, initial_state, actual_seq_lengths,
                                         use_float64=False, chunk_size=chunk_size)
    o_bench, state_bench = cgdr_benchmark(q, k, v, g, beta, scale, initial_state, actual_seq_lengths,
                                          chunk_size=chunk_size)
    o_npu, state_npu = cgdr_npu(q, k, v, g, beta, scale, initial_state, actual_seq_lengths)
    # ======================== check result ================================
    ret = True
    if not compare_cv(o_golden, o_bench, o_npu, name="o"):
        print("compare o failed.")
        err_o = torch.abs(o_golden - o_npu).flatten()
        idx = torch.argmax(err_o)
        print(f"idx={idx}, err_o={err_o[idx]}, o_golden={o_golden.flatten()[idx]}, "
              f"o_npu={o_npu.flatten()[idx]}, o_bench={o_bench.flatten()[idx]}")
        ret = False
    if not compare_cv(state_golden, state_bench, state_npu, name="state"):
        print("compare state failed.")
        err_s = torch.abs(state_golden - state_npu).flatten()
        idx = torch.argmax(err_s)
        print(f"idx={idx}, err_s={err_s[idx]}, state_golden={state_golden.flatten()[idx]}, "
              f"state_npu={state_npu.flatten()[idx]}, state_bench={state_bench.flatten()[idx]}")
        ret = False
    print("PASSED" if ret else "FAILED")
    return ret


def run_precision_test(inputs):
    run_chunk_gated_delta_rule_eager(
        inputs['B'], inputs['seqlen'], inputs['nk'],
        inputs['nv'], inputs['dk'], inputs['dv'], inputs['chunk_size'],
        data_type=inputs['data_type'],
        state_data_type=inputs.get('state_data_type', torch.bfloat16),
        has_g=inputs.get('has_g', True),
        is_continue=inputs.get('is_continue', True)
    )
