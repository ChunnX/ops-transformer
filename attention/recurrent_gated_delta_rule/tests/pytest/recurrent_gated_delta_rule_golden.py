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
import numpy as np
from torch_npu.testing.testcase import TestCase, run_tests
import logging
import datetime
import os
import sys
import argparse

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

def cpu_recurrent_gated_delta_rule(
    q, k, v, state, beta, scale_value, act_seq_len, ssm_state_indices,
        num_accepted_tokens=None, g=None, gk=None):
    T, n_heads_qk, Dk = q.shape
    T, n_heads_v, Dv = v.shape
    B = act_seq_len.shape[0]
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    v = v.to(torch.float32)
    initial_state = state.to(torch.float32)
    beta = beta.to(torch.float32)
    output = torch.empty_like(v).to(torch.float32)
    g = torch.ones(T, n_heads_v).to(torch.float32) if g is None else g.to(torch.float32).exp()
    gk = torch.ones(T, n_heads_v, Dk).to(torch.float32) if gk is None else gk.to(torch.float32).exp()

    q = q * scale_value
    seq_start = 0
    for i in range(B):
        if num_accepted_tokens is None:
            init_state = initial_state[ssm_state_indices[seq_start]]
        else:
            init_state = initial_state[ssm_state_indices[seq_start + num_accepted_tokens[i] - 1]]
        for head_id in range(n_heads_v):
            S = init_state[head_id]
            for slot_id in range(seq_start, seq_start + act_seq_len[i]):
                q_i = q[slot_id][head_id // (n_heads_v // n_heads_qk)] # [Dk]
                k_i = k[slot_id][head_id // (n_heads_v // n_heads_qk)] # [Dk]
                v_i = v[slot_id][head_id] # [Dv]
                beta_i = beta[slot_id][head_id]
                alpha_i = g[slot_id][head_id]
                S = S * alpha_i
                alphak_i = gk[slot_id][head_id] # [Dk]
                S = S * alphak_i.unsqueeze(-2)

                x = (S * k_i.unsqueeze(-2)).sum(dim=-1)
                y = (v_i - x) * beta_i # [Dv]
                S_ = y[:, None] * k_i[None, :] # [Dv, Dk]
                S = S + S_ # [Dv, Dk]
                initial_state[ssm_state_indices[slot_id]][head_id] = S

                output[slot_id][head_id] = ((S * q_i.unsqueeze(-2)).sum(dim=-1)) # [Dv]
        seq_start += act_seq_len[i]
    output_golden = output
    state_golden = initial_state

    print(f"output_golden.shape: {output_golden.shape}")
    print(f"state_golden.shape: {state_golden.shape}")
    output_golden = torch.tensor(output_golden).to(q.dtype)
    return output_golden, state_golden

def rand_range(shape, data_range=[-10, 10], dtype=torch.bfloat16, device=None):
    return data_range[0] + (data_range[1] - data_range[0]) * torch.rand(shape, dtype=dtype, device=device)

def run_recurrent_gated_delta_rule_eager(B, mtp, nk, nv, dk, dv, actual_seq_lengths=None,
                         ssm_state_indices=None, has_gamma='False', has_gamma_k='False',
                         has_num_accepted_tokens='False', scale_value=None, num_accepted_tokens=None,
                         block_num=None, data_type=torch.bfloat16, query_datarange=[-10, 10], key_datarange=[-10, 10],
                         value_datarange=[-10, 10], gamma_datarange=[0, 1], gamma_k_datarange=[0, 1],
                         beta_datarange=[0, 1], state_datarange=[-10, 10]):
    torch_npu.npu.set_device(int(DEVICE_ID))
    # ======================== set input params finish ========================
    block_num = B * mtp if block_num == None else block_num
    if scale_value == None:
        scale_value = dk ** -0.5
    if actual_seq_lengths == None:
        actual_seq_lengths = [mtp] * B
    if has_num_accepted_tokens == True and num_accepted_tokens == None:
        num_accepted_tokens = torch.tensor([torch.randint(0, h, (1,)) for h in actual_seq_lengths]) + 1
    T = int(sum(actual_seq_lengths))
    actual_seq_lengths = torch.tensor(actual_seq_lengths)
    if ssm_state_indices == None:
        ssm_state_indices = torch.arange(T, dtype=torch.int32)
    # ======================== set input params finish ========================
    # ======================== check input params start ========================
    if len(actual_seq_lengths) != B:
        print(f"Error: the len of seqused is {len(actual_seq_lengths)}, it should be B({B})")
        return
    if has_num_accepted_tokens == True and len(num_accepted_tokens) != B:
        print(f"Error: the len of num_accepted_tokens is {len(num_accepted_tokens)}, it should be B({B})")
        return
    for i in range(B):
        act_seq = actual_seq_lengths[i]
        if act_seq <= 0 or act_seq > mtp:
            print(f"Error: actual_seq_lengths[{i}] is {act_seq}, it should > 0 and <= mtp({mtp})")
            return
        if has_num_accepted_tokens == True:
            accepted_token = num_accepted_tokens[i]
            if accepted_token < 1 or accepted_token > act_seq:
                print(f"Error: num_accepted_tokens[{i}] is {accepted_token}, it should >= 1 and <= actual_seq_lengths[{i}]({act_seq})")
                return
    if len(ssm_state_indices) != T:
        print(f"Error: the len of ssm_state_indices is {len(ssm_state_indices)}, it should be T({T})")
        return
    for i in range(T):
        idx = ssm_state_indices[i]
        if idx < 0 or idx > block_num:
            print(f"Error: ssm_state_indices[{i}] is {idx}, it should >= 0 and < block_num({block_num})")
            return
    # ======================== check input params finish ========================
    # ======================== gen input data start =============================
    query = rand_range((T, nk, dk), query_datarange, data_type)
    key = rand_range((T, nk, dk), key_datarange, data_type)
    value = rand_range((T, nv, dv), value_datarange, data_type)
    g = rand_range((T, nv), gamma_datarange, dtype=torch.float32) if has_gamma == True else None
    gk = rand_range((T, nv, dk), gamma_k_datarange, dtype=torch.float32) if has_gamma_k == True else None
    beta = rand_range((T, nv), beta_datarange, data_type)
    num_accepted_tokens = torch.tensor(num_accepted_tokens, dtype=torch.int32) if has_num_accepted_tokens == True else None
    state = rand_range((block_num, nv, dv, dk), state_datarange, data_type)
    act_seq_len = torch.tensor(actual_seq_lengths, dtype=torch.int32)
    ssm_state_indices = torch.tensor(ssm_state_indices, dtype=torch.int32)

    # ======================== gen input data finish =============================

    # ======================== execute cpu start =================================
    cpu_out, cpu_state_ouput = cpu_recurrent_gated_delta_rule(
        query, key, value, state, beta, scale_value, act_seq_len, ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens, g=g, gk=gk)
    # ======================== execute cpu finish ================================

    # ======================== execute npu start =================================
    query = query.to("npu:%s" % DEVICE_ID)
    key = key.to("npu:%s" % DEVICE_ID)
    value = value.to("npu:%s" % DEVICE_ID)
    state = state.to("npu:%s" % DEVICE_ID)
    beta = beta.to("npu:%s" % DEVICE_ID)
    act_seq_len = act_seq_len.to("npu:%s" % DEVICE_ID)
    ssm_state_indices = ssm_state_indices.to("npu:%s" % DEVICE_ID)
    num_accepted_tokens = num_accepted_tokens.to("npu:%s" % DEVICE_ID) if has_num_accepted_tokens == True else None
    g = g.to("npu:%s" % DEVICE_ID) if has_gamma == True else None
    gk = gk.to("npu:%s" % DEVICE_ID) if has_gamma_k == True else None

    # ======================== execute npu finish ================================
    # start run custom ops
    init_state = state.clone()
    npu_out = (
        torch_npu.npu_recurrent_gated_delta_rule(
            query, key, value, init_state, beta=beta, scale=scale_value, actual_seq_lengths=act_seq_len,
            ssm_state_indices=ssm_state_indices, num_accepted_tokens=num_accepted_tokens,g=g, gk=gk)
    )
    npu_state_out = init_state
    print(f"query: shape {query.shape}, dtype: {query.dtype}")
    print(f"key: shape {key.shape}, dtype: {key.dtype}")
    print(f"value: shape {value.shape}, dtype: {value.dtype}")
    print(f"state: shape {state.shape}, dtype: {state.dtype}")
    print(f"beta: shape {beta.shape}, dtype: {beta.dtype}")
    print(f"act_seq_len: shape {act_seq_len.shape[0]}, dtype: {act_seq_len.dtype}")
    print(f"ssm_state_indices: shape {ssm_state_indices.shape[0]}, dtype: {ssm_state_indices.dtype}")

    # 结果精度对比
    check_succeed = True
    data_type = str(npu_out.dtype)
    print("--------------------------------------------------------------check result-------------------------------------------------------------")
    check_result(cpu_out.to(torch.float32), npu_out.cpu().to(torch.float32), data_type)
    print("--------------------------------------------------------------check state ouput-------------------------------------------------------------")
    check_result(cpu_state_ouput.to(torch.float32), npu_state_out.cpu().to(torch.float32), data_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--B', required=True, type=int, help='batch size')
    parser.add_argument('--mtp', required=True, type=int, help='max sequence length, for every batch')
    parser.add_argument('--nk', required=True, type=int, help='0 < nk <= 256')
    parser.add_argument('--nv', required=True, type=int, help='nv should be a multiple of nk and 0 < nv <= 256')
    parser.add_argument('--dk', required=True, type=int, help='0 < dk <= 256')
    parser.add_argument('--dv', required=True, type=int, help='0 < dv <= 256')
    parser.add_argument('--actual_seq_lengths', type=int, nargs='*', help='sequence of every batch should not greated than mtp, len is B')
    parser.add_argument('--ssm_state_indices', type=int, nargs='*', help='map index from input sequence to state matrix, len is B')
    parser.add_argument('--has_gamma', type=str, default='False', help="whether use gamma")
    parser.add_argument('--has_gamma_k', type=str, default='False', help='whether use gamma k')
    parser.add_argument('--has_num_accepted_tokens', type=str, default='False', help='whether use num_accepted_tokens')
    parser.add_argument('--scale_value', type=float, default=None, help='query scaling factor')
    parser.add_argument('--block_num', type=int, default=None, help='block_num should not be less than the sum of actual_seq_lengths')
    parser.add_argument('--data_type', type=str, default="bfloat16", help='bfloat16')
    parser.add_argument('--query_datarange', type=list, default=[-10, 10])
    parser.add_argument('--key_datarange', type=list, default=[-10, 10])
    parser.add_argument('--value_datarange', type=list, default=[-10, 10])
    parser.add_argument('--gamma_datarange', type=list, default=[0, 1])
    parser.add_argument('--gamma_k_datarange', type=list, default=[0, 1])
    parser.add_argument('--beta_datarange', type=list, default=[0, 1])
    parser.add_argument('--state_datarange', type=list, default=[-10, 10])
    args = parser.parse_args()

    if args.data_type == "float16" or args.data_type == "FP16" or args.data_type == "fp16":
        data_type = torch.float16
    elif args.data_type == "bfloat16" or args.data_type == "BF16" or args.data_type == "bf16":
        data_type = torch.bfloat16
    else:
        raise ValueError("Error: data_type only support bfloat16 and float16")
        sys.exit(1)

    run_recurrent_gated_delta_rule_eager(
            args.B,
            args.mtp,
            args.nk,
            args.nv,
            args.dk,
            args.dv,
            args.actual_seq_lengths,
            args.ssm_state_indices,
            args.has_gamma,
            args.has_gamma_k,
            args.has_num_accepted_tokens,
            args.scale_value,
            args.block_num,
            data_type,
            args.query_datarange,
            args.key_datarange,
            args.value_datarange,
            args.gamma_datarange,
            args.gamma_k_datarange,
            args.beta_datarange,
            args.state_datarange)