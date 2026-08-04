# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import random
import torch
import torch_npu
import torchair
from cann_ops_transformer.ops.compressor_grad import compressor_grad
import numpy as np
import torch.nn as nn
from torch_npu.testing.testcase import TestCase, run_tests
from cpu_golden import compressor_grad_golden
import logging
import datetime
import os
 
import sys
import argparse
import math

np.random.seed(21)  # 固定随机种子
np.set_printoptions(suppress=True)

DEVICE_ID = 0
torch.npu.config.allow_internal_format = True

logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)
logger = logging.getLogger(__name__)

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

# fuzz 中precision_method == 1的精度对比方式
def check_result(expect, result, data_type, pct_thd = 0.005):
    real_data = result.cpu().numpy()
    data_compe = expect.cpu().numpy()
    real_data = real_data.flatten()
    data_compe = data_compe.flatten()
    if real_data.size == 0 and real_data.size == data_compe.size:
        print_log(
            'The npu_output is [],and it is same as bm_output, the result of data_compare is \"Pass\"')
        return  100.0, "Pass"
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
        diff_thd=0.005
        max_diff_hd=10.0
        rtol=0.0078125
        atol=0.0001
        max_error_idx = 10000000
    else:
        diff_thd=0.005
        max_diff_hd=10.0
        rtol=0.005
        atol=0.000025
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
    return fulfill_percent, result

def run_compressor_grad_eager(B, S_max, head_dim, coff, cmp_ratio, bs_combine_flag, layout,
                         S = 0, start_pos=None, seqused=None, cu_seqlens=None,
                         block_size = 128, data_type = torch.bfloat16, hidden_size = 4096,
                         x_datarange = [-10, 10], wkv_datarange = [-10, 10], wgate_datarange = [-10, 10],
                         d_cmp_kv_datarange = [-10, 10], softmax_score_datarange = [-10, 10], kv_datarange = [-10, 10]):
    torch_npu.npu.set_device(int(DEVICE_ID))
    # ======================== set input params finish ========================
    if bs_combine_flag:
        # cu_seqlens = [0, 1] # (B+1,), None时表示非BSh，否则为Th
        if cu_seqlens is None:
            print(f"Error: layout of x is [T, hidden_size], cu_seqlens is required!!!")
            return
        old_S = S
        if seqused is not None:
            S = max(seqused)
        else:
            S = 0
            for i in range(B):
                if (cu_seqlens[i + 1] - cu_seqlens[i]) > S:
                    S = cu_seqlens[i + 1] - cu_seqlens[i]
        print(f"Warning: layout of x is [T, hidden_size], S={old_S}, it is modified to S={S}!!!")
    else:
        if cu_seqlens is not None:
            print(f"Warning: layout of x is [B, S, hidden_size], but cu_seqlens is not None, it is modified to None!!!")
        cu_seqlens = None
    
    exist_start_pos = True
    if start_pos is None:
        exist_start_pos = False
        start_pos = [0] * B
    # ======================== set input params finish ========================
    # ======================== check input params start ========================
    if start_pos is not None:
        if len(start_pos) != B:
            print(f"Error: the len of start_pos is {len(start_pos)}, it should be B({B})")
            return
    if seqused is not None:
        if len(seqused) != B:
            print(f"Error: the len of seqused is {len(seqused)}, it should be B({B})")
            return
    if cu_seqlens is not None:
        if len(cu_seqlens) != (B + 1):
            print(f"Error: the len of cu_seqlens is {len(cu_seqlens)}, it should be equal to B({B}) + 1")
            return
    if bs_combine_flag:
        for i in range(B):
            if start_pos[i] + (cu_seqlens[i + 1] - cu_seqlens[i]) > S_max:
                print(f"Error: for batch {i} when shape of x is (T, hidden_size), start_pos[{i}] + (cu_seqlens[{i + 1}] - cu_seqlens[{i}]) > S_max, "
                    f"start_pos[{i}]={start_pos[i]}, cu_seqlens[{i + 1}]={cu_seqlens[i + 1]}, cu_seqlens[{i}]={cu_seqlens[i]}, S_max={S_max}")
                return
            if seqused is not None:
                if seqused[i] > (cu_seqlens[i + 1] - cu_seqlens[i]):
                    print(f"Error: for batch {i} when shape of x is (T, hidden_size), seqused[{i}] > (cu_seqlens[{i + 1}] - cu_seqlens[{i}]), "
                        f"seqused[{i}]={seqused[i]}, cu_seqlens[{i + 1}]={cu_seqlens[i + 1]}, cu_seqlens[{i}]={cu_seqlens[i]}")
                    return
    else:
        for i in range(B):
            if start_pos[i] + S > S_max:
                print(f"Error: for batch {i} when shape of x is (B, S, hidden_size), start_pos[{i}] + S > S_max, start_pos[{i}]={start_pos[i]}, S={S}, S_max={S_max}")
                return
            if seqused is not None:
                if seqused[i] > S:
                    print(f"Error: for batch {i} when shape of x is (B, S, hidden_size), seqused[{i}] > S, seqused[{i}]={seqused[i]}, S={S}")
                    return

    if start_pos is not None:
        start_pos = torch.tensor(start_pos).to(torch.int32).to("npu:%s" % DEVICE_ID)
    # other input
    if bs_combine_flag:
        x_shape = (cu_seqlens[-1], hidden_size)
        d_cmp_kv_shape = (min(cu_seqlens[-1], cu_seqlens[-1] // cmp_ratio + B), head_dim)
        softmax_score_shape = (min(cu_seqlens[-1], cu_seqlens[-1] // cmp_ratio + B), coff * cmp_ratio, head_dim)
        kv_shape = (min(cu_seqlens[-1], cu_seqlens[-1] // cmp_ratio + B), coff * cmp_ratio, head_dim)
    else:
        x_shape = (B, S, hidden_size)
        d_cmp_kv_shape = (B, math.ceil(S / cmp_ratio), head_dim)
        softmax_score_shape = (B, math.ceil(S / cmp_ratio), coff * cmp_ratio, head_dim)
        kv_shape = (B, math.ceil(S / cmp_ratio), coff * cmp_ratio, head_dim)

    x = torch.tensor(np.random.uniform(x_datarange[0], x_datarange[1], x_shape)).to(data_type)
    wkv = torch.tensor(np.random.uniform(wkv_datarange[0], wkv_datarange[1], (coff * head_dim, hidden_size))).to(data_type)
    wgate = torch.tensor(np.random.uniform(wgate_datarange[0], wgate_datarange[1], (coff * head_dim, hidden_size))).to(data_type)
    d_cmp_kv = torch.tensor(np.random.uniform(d_cmp_kv_datarange[0], d_cmp_kv_datarange[1], d_cmp_kv_shape)).to(data_type)
    softmax_score = torch.tensor(np.random.uniform(softmax_score_datarange[0], softmax_score_datarange[1], softmax_score_shape)).to(torch.float32)
    kv = torch.tensor(np.random.uniform(wgate_datarange[0], wgate_datarange[1], kv_shape)).to(torch.float32)

    print(f"start_pos={start_pos}")
    print(f"seqused={seqused}")
    print(f"cu_seqlens={cu_seqlens}")


    # ======================== gen input data finish =============================

    cpu_d_x, cpu_d_wkv, cpu_d_wgate, cpu_d_ape = compressor_grad_golden(
        x, wkv, wgate, d_cmp_kv, softmax_score, kv, cu_seqlens, seqused, start_pos, cmp_ratio, coff)
    # ======================== execute cpu finish ================================

    # ======================== execute npu start =================================
    x = x.to("npu:%s" % DEVICE_ID)
    wkv = wkv.to("npu:%s" % DEVICE_ID)
    wgate = wgate.to("npu:%s" % DEVICE_ID)
    d_cmp_kv = d_cmp_kv.to("npu:%s" % DEVICE_ID)
    softmax_score = softmax_score.to("npu:%s" % DEVICE_ID)
    kv = kv.to("npu:%s" % DEVICE_ID)


    start_pos = torch.tensor(start_pos).to(torch.int32).to("npu:%s" % DEVICE_ID)
    if cu_seqlens is not None:
        cu_seqlens = torch.tensor(cu_seqlens).to(torch.int32).to("npu:%s" % DEVICE_ID)
    if seqused is not None:
        seqused = torch.tensor(seqused).to(torch.int32).to("npu:%s" % DEVICE_ID)
    # ======================== execute npu finish ================================
    # start run custom ops
    npu_out = (
        compressor_grad(
            x,
            wkv,
            wgate,
            d_cmp_kv,
            softmax_score,
            kv,
            cmp_ratio,
            coff = coff,
            cu_seqlens = cu_seqlens,
            seqused = seqused,
            start_pos = start_pos if exist_start_pos else None
        )
    )
    print(f"x: shape {x.shape}, dtype: {x.dtype}")
    print(f"wkv: shape {wkv.shape}, dtype: {wkv.dtype}")
    print(f"wgate: shape {wgate.shape}, dtype: {wgate.dtype}")
    print(f"d_cmp_kv: shape {d_cmp_kv.shape}, dtype: {d_cmp_kv.dtype}")
    print(f"softmax_score: shape {softmax_score.shape}, dtype: {softmax_score.dtype}")
    print(f"kv: shape {kv.shape}, dtype: {kv.dtype}")
    
    check_succeed = True
    data_type = str(npu_out[0].dtype)
    results = {}
    print("--------------------------------------------------------------check result-------------------------------------------------------------")
    # _, results['d_x'] = check_result(cpu_d_x.to(torch.float32), npu_out[0].cpu().to(torch.float32), data_type)
    # _, results['d_wkv'] = check_result(cpu_d_wkv.to(torch.float32), npu_out[1].cpu().to(torch.float32), data_type)
    # _, results['d_wgate'] = check_result(cpu_d_wgate.to(torch.float32), npu_out[2].cpu().to(torch.float32), data_type)
    # _, results['d_ape'] = check_result(cpu_d_ape.to(torch.float32), npu_out[3].cpu().to(torch.float32), data_type)




# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--B', required=True, type=int, help='batch size')
#     parser.add_argument('--S_max', required=True, type=int, help='max sequence length, for gen block_table')
#     parser.add_argument('--head_dim', required=True, type=int)
#     parser.add_argument('--coff', required=True, type=int, choices=[1, 2], help='1:no overlap 2:overlap')
#     parser.add_argument('--cmp_ratio', required=True, type=int, choices=[2, 4, 8, 16, 32, 64, 128], help='compress cmp_ratio token to one')
#     parser.add_argument('--bs_combine_flag', action='store_true', help='Include this flag to set x.shape to [T, hidden_size]')
#     parser.add_argument('--S', type=int, default=0, help='cur sequence length dim of x when BSh')
#     parser.add_argument('--start_pos', type=int, nargs='*', help='start_pos of every batch, len is B')
#     parser.add_argument('--seqused', type=int, nargs='*', help="sequence which is used, len is B")
#     parser.add_argument('--cu_seqlens', type=int, nargs='*', help='when x is [T, h], it is required, len is B+1')
#     parser.add_argument('--block_size', type=int, default=128)
#     parser.add_argument('--cache_mode', type=int, default=1)
#     parser.add_argument('--data_type', type=str, choices=["bfloat16", "float16"], default="bfloat16", help='bfloat16 or float16')
#     parser.add_argument('--hidden_size', type=int, default=4096)
#     parser.add_argument('--save_state_seqlens', type=int, nargs='*', help='save to kv_state and score_state of seq id in [start_pos + seqused - save_state_seqlens, start_pos + seqused)')
#     args = parser.parse_args()

#     if args.data_type == "float16":
#         data_type = torch.float16
#     elif args.data_type == "bfloat16":
#         data_type = torch.bfloat16
#     else:
#         raise ValueError("Error: data_type only support bfloat16 and float16")
#         sys.exit(1)

#     run_compressor_eager(
#             args.B,
#             args.S_max,
#             args.head_dim,
#             args.coff,
#             args.cmp_ratio,
#             args.bs_combine_flag,
#             args.S,
#             args.start_pos,
#             args.seqused,
#             args.cu_seqlens,
#             args.block_size,
#             data_type,
#             args.hidden_size,
#             args.save_state_seqlens,
#             args.cache_mode)