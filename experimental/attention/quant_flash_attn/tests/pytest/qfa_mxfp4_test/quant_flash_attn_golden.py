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
import math
import cann_ops_transformer
from cann_ops_transformer.ops import quant_flash_attn
from cann_ops_transformer.ops import quant_flash_attn_metadata

from mx_quant_fp4_tool import mx_quantize_fp4_full, mx_dequantize_fp4_full, mxfp4_quantize_pack_last
from flash_attention_cpu_golden import attention_cpu_golden_varlen, flash_attention_cpu_golden_varlen

RAND_SEED=21

np.random.seed(RAND_SEED)           # NumPy 随机种子
torch.manual_seed(RAND_SEED)        # PyTorch CPU 随机种子
torch.cuda.manual_seed_all(RAND_SEED)  # PyTorch GPU 随机种子（如果有GPU）
random.seed(RAND_SEED)              # Python 内置 random 模块种子

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

def rand_range(shape, data_range=[-10, 10], dtype=torch.bfloat16, device=None):
    return data_range[0] + (data_range[1] - data_range[0]) * torch.rand(shape, dtype=dtype, device=device)

def randn_x(shape, x=1.0, dtype=torch.bfloat16, device=None):
    return x * torch.randn(shape, dtype=dtype, device=device)

def get_query_layout(input_layout):
    if input_layout == 'BSH' or input_layout == 'BSH_BNSD' or input_layout == 'BSH_NBSD':
        return 'BSH'
    elif input_layout == 'BSND' or input_layout == 'BSND_BNSD' or input_layout == 'BSND_NBSD':
        return 'BSND'
    elif input_layout == 'BNSD' or input_layout == 'BNSD_BSND' or input_layout == 'BNSD_NBSD':
        return 'BNSD'
    elif input_layout == 'TND' or input_layout == 'TND_NTD':
        return 'TND'
    elif input_layout == 'NTD' or input_layout == 'NTD_TND':
        return 'NTD'
    else:
        return None

def get_attn_out_layout(input_layout):
    if input_layout == 'BSH':
        return 'BSH'
    elif input_layout == 'BSND' or input_layout == 'BNSD_BSND':
        return 'BSND'
    elif input_layout == 'BNSD' or input_layout == 'BSH_BNSD' or input_layout == 'BSND_BNSD':
        return 'BNSD'
    elif input_layout == 'BSH_NBSD' or input_layout == 'BSND_NBSD' or input_layout == 'BNSD_NBSD':
        return 'NBSD'
    elif input_layout == 'TND' or input_layout == 'NTD_TND':
        return 'TND'
    elif input_layout == 'NTD' or input_layout == 'TND_NTD':
        return 'NTD'
    else:
        return None

def get_softmax_lse_layout(input_layout):
    if input_layout == 'TND' or input_layout == 'NTD_TND' or input_layout == 'NTD' or input_layout == 'TND_NTD':
        return 'TND'
    else:
        return 'BNSD'

def get_shape(layout, b, n, s, d, t):
    if layout == 'BSH':
        return (b, s, n * d)
    elif layout == 'BSND':
        return (b, s, n, d)
    elif layout == 'BNSD':
        return (b, n, s, d)
    elif layout == 'TND':
        return (t, n, d)
    elif layout == 'NTD':
        return (n, t, d)
    else:
        return None

def get_dtype(data_type):
    if data_type == 'float16':
        return torch.float16
    elif data_type == 'bfloat16':
        return torch.bfloat16
    elif data_type == 'int8':
        return torch.int8

def dtype_sizeof(data_type):
    if data_type == 'float16' or data_type == 'bfloat16':
        return 2
    elif data_type == 'int8' or data_type == 'float8':
        return 1

def get_t(b, act_seq_lens):
    if act_seq_lens is None:
        return 0
    if len(act_seq_lens) == 1:
        return b * act_seq_lens[0]
    sum = 0
    for i in range(b):
        sum += act_seq_lens[i]
    return sum

def update_act_seq_lens_for_tnd(layout, b, act_seq_lens):
    cu_seqlens = None
    if act_seq_lens is not None:
        if layout == 'TND' or layout == 'NTD':
            # 初始化前缀和数组，长度为 batch + 1
            cu_seqlens = [0] * (b + 1)
            # 兼容 tensor / 列表输入
            seq_list = act_seq_lens.tolist() if hasattr(act_seq_lens, 'tolist') else act_seq_lens
            for i in range(b):
                cu_seqlens[i + 1] = cu_seqlens[i] + seq_list[i]
    return cu_seqlens

def TO_NPU(tensor):
    if tensor is None:
        return None
    else:
        return tensor.to("npu:%s" % DEVICE_ID)

def get_act_seq_len_by_batch(b_idx, default_s, act_seq_lens):
    if act_seq_lens is None:
        return default_s
    elif len(act_seq_lens) == 1:
        return act_seq_lens[0]
    else:
        return act_seq_lens[b_idx]

def bnsd_to_bsh(bnsd_tensor):
    return bnsd_tensor.permute(0, 2, 1, 3).flatten(start_dim=2) # BNSD->BSND, 再将ND合并为H

def bnsd_to_bsnd(bnsd_tensor):
    return bnsd_tensor.permute(0, 2, 1, 3)

def bnsd_to_tnd(bnsd_tensor, b, act_seq_lens):
    if act_seq_lens is None:
        return bnsd_tensor.permute(0, 2, 1, 3).flatten(start_dim=0, end_dim=1) # 合并[start_dim, end_dim]维度
    elif len(act_seq_lens) == 1:
        # narrow是从指定维度截取数据
        return torch.narrow(bnsd_tensor, dim=2, start=0, length=act_seq_lens[0]).permute(0, 2, 1, 3).flatten(start_dim=0, end_dim=1)
    else:
        t = get_t(b, act_seq_lens)
        tnd_tensor = torch.empty(t, bnsd_tensor.shape[1], bnsd_tensor.shape[3], dtype=bnsd_tensor.dtype)
        t_idx = 0
        for i in range(b):
            if act_seq_lens[i] > 0:
                tnd_tensor[t_idx:(t_idx + act_seq_lens[i]), :, :] = bnsd_tensor[i, :, 0:act_seq_lens[i], :].permute(1, 0, 2)
                t_idx = t_idx + act_seq_lens[i]
        return tnd_tensor

def bnsd_to_ntd(bnsd_tensor, b, act_seq_lens):
    if act_seq_lens is None:
        return bnsd_tensor.permute(1, 0, 2, 3).flatten(start_dim=0, end_dim=1) # 合并[start_dim, end_dim]维度
    elif len(act_seq_lens) == 1:
        # narrow是从指定维度截取数据
        return torch.narrow(bnsd_tensor, dim=2, start=0, length=act_seq_lens[i]).permute(1, 0, 2, 3).flatten(start_dim=1, end_dim=2)
    else:
        t = get_t(b, act_seq_lens)
        ntd_tensor = torch.empty(bnsd_tensor.shape[1], t, bnsd_tensor.shape[3], dtype=bnsd_tensor.dtype)
        t_idx = 0
        for i in range(b):
            if act_seq_lens[i] > 0:
                ntd_tensor[:, t_idx:(t_idx + act_seq_lens[i]), :] = bnsd_tensor[i, :, 0:act_seq_lens[i], :]
                t_idx = t_idx + act_seq_lens[i]
        return ntd_tensor

def get_block_table(b, act_seq_lens_kv, block_size):
    s2_max = max(act_seq_lens_kv)
    max_block_num_per_batch = (s2_max + block_size - 1) // block_size
    block_table = torch.full((b, max_block_num_per_batch), -1, dtype=torch.int32)
    block_num = 0
    for i in range(b):
        b_seq = act_seq_lens_kv[i] if len(act_seq_lens_kv) > 1 else act_seq_lens_kv[0]
        block_num += (b_seq + block_size - 1) // block_size
    # page cache
    block_id_array = torch.randperm(block_num, dtype=torch.int32)
    index = 0
    for i in range(b):
        b_seq = act_seq_lens_kv[i] if len(act_seq_lens_kv) > 1 else act_seq_lens_kv[0]
        b_block_num = (b_seq + block_size - 1) // block_size
        for j in range(b_block_num):
            block_table[i][j] = block_id_array[index]
            index = index + 1
    return block_table

def page_attn_for_bnsd(bnsd_tensor, b, act_seq_lens_kv, block_table, block_size):
    block_num = int(block_table.max()) + 1
    kv_cache_bnsd_shape = (block_num, bnsd_tensor.shape[1], block_size, bnsd_tensor.shape[3])
    page_cache_tensor = torch.zeros(size=kv_cache_bnsd_shape, dtype=bnsd_tensor.dtype)
    for i in range(b):
        b_seq = act_seq_lens_kv[i] if len(act_seq_lens_kv) > 1 else act_seq_lens_kv[0]
        b_block_num = (b_seq + block_size - 1) // block_size
        for j in range(b_block_num):
            page_cache_tensor[block_table[i][j], :, :, :] = bnsd_tensor[i, :, (j*block_size):((j+1)*block_size), :]
    return page_cache_tensor

def rearrange_by_layout(bnsd_tensor, layout, b, act_seq_lens):
    if layout == "BNSD":
        return bnsd_tensor
    elif layout == "BSH":
        return bnsd_to_bsh(bnsd_tensor) # BNSD->BSND, 再将ND合并为H
    elif layout == "BSND":
        return bnsd_to_bsnd(bnsd_tensor)
    elif layout == "TND":
        return bnsd_to_tnd(bnsd_tensor, b, act_seq_lens)
    elif layout == "NTD":
        return bnsd_to_ntd(bnsd_tensor, b, act_seq_lens)
    else:
        return None

def transpose_kscale(k_scale: torch.Tensor, layout: str):
    """
    对KV侧量化scale张量做维度重排，适配不同输入布局
    :param k_scale: 原始scale张量，TND输入shape=[T, N, D]
    :param layout: 输入scale原始布局，仅支持 TND / BSND
    :return: TND输出shape=[T, N, D//2, 2]
    """
    # 空值、类型前置防护
    if k_scale is None:
        raise ValueError("k_scale cannot be None")
    if not isinstance(k_scale, torch.Tensor):
        raise TypeError("k_scale must be torch.Tensor")

    if layout == "TND":
        # 先校验维度，再解包shape
        if k_scale.dim() != 3:
            raise ValueError(f"TND layout requires 3D tensor [T,N,D], got dim={k_scale.dim()}")
        T, N, D = k_scale.shape

        # 校验D必须被2整除，防止reshape失败
        if D % 2 != 0:
            raise ValueError(f"Feature dim D={D} must be divisible by 2, split into (D//2, 2)")

        # 仅拆分特征维度D，T、N保持不变，输出 [T, N, D//2, 2]
        k_scale_out = k_scale.reshape(T, N, D // 2, 2)
        return k_scale_out

    elif layout == "BSND":
        # 未实现分支抛异常，不返回None
        B, S, N,D = k_scale.shape
        k_scale_copy = k_scale.contiguous().reshape(B, S, N, D // 2, 2)
        return  k_scale
    elif layout == "BNSD":
        # 输入 BNSD shape=[B, N, S, D]，拆分末维 D -> (D//2, 2)，输出 [B, N, S, D//2, 2]
        B, N, S, D = k_scale.shape
        if D % 2 != 0:
            raise ValueError(f"Feature dim D={D} must be divisible by 2, split into (D//2, 2)")
        k_scale_out = k_scale.contiguous().reshape(B, N, S, D // 2, 2)
        return k_scale_out
    else:
        raise ValueError(f"Unsupported layout: {layout}, only support TND、BSND and BNSD")


def transpose_qscale(q_scale: torch.Tensor, layout: str, n2: int):
    """
    对量化scale张量做维度重排，适配不同输入布局
    :param q_scale: 原始scale张量
    :param layout: 输入scale原始布局，仅支持 TND / BSND
    :param n2: 每组KV头数量，GQA总多头 N = n2 * group_size，group_size=N//n2
    :return: 重排后的scale张量
    """
    # 全局输入防护，规避崩溃
    if q_scale is None:
        raise ValueError("q_scale cannot be None")
    if not isinstance(q_scale, torch.Tensor):
        raise TypeError("q_scale must be torch.Tensor")

    if layout == "TND":
        # 原始shape: [T, N, D]
        if q_scale.dim() != 3:
            raise ValueError(f"TND layout requires 3D tensor [T,N,D], got dim={q_scale.dim()}")
        T, N, D = q_scale.shape

        if N % n2 != 0:
            raise ValueError(f"Total head N={N} must be divisible by n2={n2}")
        G = N // n2
        # 移除冗余assert N == n2 * G，整除校验后恒成立
        align_base = 2
        if D % align_base != 0:
            raise ValueError(f"Feature dim D={D} must be divisible by {align_base} for MXFP4 pack, split into (D//64, 2)")

        # Step1: 拆分多头维度 N -> [n2, G]
        q = q_scale.reshape(T, n2, G, D)
        # Step2: 拆分特征维度 D -> [D//64, 2] 适配4bit打包存储
        q = q.reshape(T, n2, G, D // 2, 2) # D//64->ascaleD//2
        # Step3: 调换维度顺序 [T,n2,G,DD,2] -> [n2,T,G,DD,2]
        q_out = q.permute(1, 0, 2, 3, 4)
        return q_out

    elif layout == "BSND":
        # 输入BSND原始shape: [B, S1, N, D]，S1为query序列长度
        # 目标输出shape: [n2, B, S1, G, D//64, 2]
        if q_scale.dim() != 4:
            raise ValueError(f"BSND layout requires 4D tensor [B,S,N,D], got dim={q_scale.dim()}")
        B, S1, N, D = q_scale.shape

        # GQA分组校验
        if N % n2 != 0:
            raise ValueError(f"Total head N={N} must be divisible by n2={n2}")
        G = N // n2
        align_base = 2
        if D % align_base != 0:
            raise ValueError(f"Feature dim D={D} must be divisible by {align_base} for MXFP4 pack")

        # Step1 拆分多头 N = n2 * G
        tmp = q_scale.reshape(B, S1, n2, G, D)
        # Step2 拆分特征维度 D -> (D//64, 2)
        tmp = tmp.reshape(B, S1, n2, G, D // 2, 2) ## D//64->ascaleD//2
        # Step3 permute调整维度顺序 B,S1,n2,G,DD,2  -> n2,B,S1,G,DD,2
        q_out = tmp.permute(2, 0, 1, 3, 4, 5)
        return q_out

    elif layout == "BNSD":
        B, N, S1, D = q_scale.shape
        q_scale_copy = q_scale.contiguous().reshape(B, N, S1, D//2, 2)
        return q_scale_copy
    else:
        raise ValueError(f"Unsupported layout: {layout}, only support TND、BSND and BNSD")

def rearrange_by_block_table(bnsd_tensor, block_table, block_size, b, act_seq_lens_kv, kv_storage_mode, kv_dtype):
    page_cache_tensor = page_attn_for_bnsd(bnsd_tensor, b, act_seq_lens_kv, block_table, block_size)
    if kv_storage_mode == "pa_bbh":
        return bnsd_to_bsh(page_cache_tensor)
    elif kv_storage_mode == "pa_bnbd":
        return page_cache_tensor
    elif kv_storage_mode == "pa_nz":
        blk_elem = 32 // dtype_sizeof(kv_dtype)
        page_cache_tensor = page_cache_tensor.reshape(page_cache_tensor.shape[0],
                                                     page_cache_tensor.shape[1],
                                                     page_cache_tensor.shape[2],
                                                     page_cache_tensor.shape[3] // blk_elem,
                                                     blk_elem).permute(0, 1, 3, 2, 4)
        return page_cache_tensor
    else:
        return None

def create_select_mask(m_shape, pre_tokens, next_tokens):
    next_masks = torch.triu(torch.ones(m_shape, dtype=torch.bool), diagonal=1+next_tokens)
    pre_masks = torch.tril(torch.ones(m_shape, dtype=torch.bool), diagonal=-1-pre_tokens)
    select_mask = next_masks + pre_masks
    return select_mask

def softmax_v1_stable(x, dim=-1):
    """
    数值稳定的Softmax实现
    通过减去最大值避免指数溢出
    """
    # 1. 减去最大值（保持数值稳定）
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_stable = x - x_max
    
    # 2. 计算指数
    exp_x = torch.exp(x_stable)
    
    # 3. 归一化
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

def mxfp_quant_per_channel(bnsd_tensor, tensor_dtype):
    quant_data = None
    scales = None
    if tensor_dtype == "fp4_e2m1":
        b, n, s, d = bnsd_tensor.shape
        packed = mx_quantize_fp4_full(bnsd_tensor.to(torch.float32), mode="baseline")
        quant_data = packed['fp4_data'].view(b, n, s, d // 2).view(dtype=torch.uint8)
        scales = packed['scales'].view(b, n, s, (d + 31) // 32).view(dtype=torch.uint8)
    return quant_data, scales

def cpu_golden_base(query, key, value,
        atten_mask=None, actual_seq_lengths=None, actual_seq_lengths_kv=None,
        query_rope=None, key_rope=None,
        scale=1.0, pre_tokens=2147483647, next_tokens=2147483647,
        sparse_mode=0, inner_precise=0,
        softmax_lse_flag=False, src_date_type=torch.float16, compute_date_type=torch.float32):
    # 注意: 不能修改输入数据
    b, n1, s1, _ = query.shape
    s2 = key.shape[2]
    # ===================== GQA扩KV头 新增代码 =====================
    hq = n1
    hkv = key.shape[1]
    if hq != hkv:
        assert hq % hkv == 0, f"Q头数{n1} 必须是KV头数{hkv}整数倍"
        gs = hq // hkv
        # 原地扩展key、value
        key = key.repeat_interleave(gs, dim=1)
        value = value.repeat_interleave(gs, dim=1)
        # 同步扩展rope key
        if key_rope is not None:
            key_rope = key_rope.repeat_interleave(gs, dim=1)
    # ==========================================================
    mask_value = float('-inf') # -1e12
    invalid_rows_out_value = 0
    invalid_rows_lse_value = float('inf')

    attn_out = torch.zeros_like(query)
    softmax_lse = torch.full((b, n1, s1, 1), float('-inf'), dtype=compute_date_type)
    for i in range(b):
        b_s1 = get_act_seq_len_by_batch(i, s1, actual_seq_lengths)
        b_s2 = get_act_seq_len_by_batch(i, s2, actual_seq_lengths_kv)
        q = query[i, :, 0:b_s1, :].to(src_date_type).to(compute_date_type)
        k = key[i, :, 0:b_s2, :].to(src_date_type).to(compute_date_type)
        v = value[i, :, 0:b_s2, :].to(src_date_type).to(compute_date_type)
        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        if query_rope is not None and key_rope is not None:
            q_r = query_rope[i, :, 0:b_s1, :].to(src_date_type).to(compute_date_type)
            k_r = key_rope[i, :, 0:b_s2, :].to(src_date_type).to(compute_date_type)
            rope_attn_scores = torch.matmul(q_r, k_r.transpose(-2, -1))
            attn_scores = attn_scores + rope_attn_scores
        attn_scores = attn_scores * scale
        # Mask
        if atten_mask is not None:
            if atten_mask.dim() == 2:
                attn_scores = attn_scores.masked_fill(atten_mask[0:b_s1, 0:b_s2], mask_value)
            elif atten_mask.dim() == 3:
                attn_scores = attn_scores.masked_fill(atten_mask[i, 0:b_s1, 0:b_s2], mask_value)
        # invalid_rows
        invalid_rows_flag = (attn_scores == mask_value).all(dim=-1)
        # Softmax
        scores_max = attn_scores.max(dim=-1, keepdim=True).values
        exp_scores = torch.exp(attn_scores - scores_max)
        scores_sum = exp_scores.sum(dim=-1, keepdim=True) + 1e-12
        attn_weights = exp_scores / scores_sum
        attn_out_tmp = torch.matmul(attn_weights, v)
        attn_out[i, :, 0:b_s1, :] = attn_out_tmp.masked_fill(invalid_rows_flag.unsqueeze(-1), invalid_rows_out_value)
        if softmax_lse_flag:
            softmax_lse_tmp = scores_max + torch.log(scores_sum)
            softmax_lse[i, :, 0:b_s1, :] = softmax_lse_tmp.masked_fill(invalid_rows_flag.unsqueeze(-1), invalid_rows_lse_value)
    return attn_out.to(src_date_type), softmax_lse.to(src_date_type) if softmax_lse_flag else None

def cpu_golden_qkv_mxfp4_attn(query, key, value,
        atten_mask=None, actual_seq_lengths=None, actual_seq_lengths_kv=None,
        query_rope=None, key_rope=None,
        scale=1.0, pre_tokens=2147483647, next_tokens=2147483647,
        sparse_mode=0, inner_precise=0,
        softmax_lse_flag=False, src_date_type=torch.float16, compute_date_type=torch.float32):
    b, n1, s1, qk_d = query.shape
    s2 = key.shape[2]
    cu_seqlens_q = [i * s1 for i in range(b + 1)]
    cu_seqlens_kv = [i * s2 for i in range(b + 1)]
    seqlens_q = actual_seq_lengths
    seqlens_kv = actual_seq_lengths_kv

    query_bsnd = bnsd_to_bsnd(query).to(compute_date_type).flatten(start_dim=0, end_dim=1)
    key_bsnd = bnsd_to_bsnd(key).to(compute_date_type).flatten(start_dim=0, end_dim=1)
    value_bsnd = bnsd_to_bsnd(value).to(compute_date_type).flatten(start_dim=0, end_dim=1)

    attn_out = attention_cpu_golden_varlen(query_bsnd, key_bsnd, value_bsnd,
        cu_seqlens_q, cu_seqlens_kv, seqlens_q, seqlens_kv,
        softmax_scale = scale, quantize=True, quantize_p=True,
        s_layout = "DN", quantize_p_mode = "global", s_dtype = "fp32", v_quant_axis = "seq_k",)

    return attn_out.reshape(b, s1, n1, qk_d).permute(0, 2, 1, 3).to(src_date_type), None

def cpu_golden_qkv_mxfp4_flash_attn(query, key, value,
        atten_mask=None, actual_seq_lengths=None, actual_seq_lengths_kv=None,
        query_rope=None, key_rope=None,
        scale=1.0, pre_tokens=2147483647, next_tokens=2147483647,
        sparse_mode=0, inner_precise=0,
        softmax_lse_flag=False, src_date_type=torch.float16, compute_date_type=torch.float32):
    b, n1, s1, qk_d = query.shape
    s2 = key.shape[2]
    cu_seqlens_q = [i * s1 for i in range(b + 1)]
    cu_seqlens_kv = [i * s2 for i in range(b + 1)]
    seqlens_q = actual_seq_lengths
    seqlens_kv = actual_seq_lengths_kv

    query_bsnd = bnsd_to_bsnd(query).to(compute_date_type).flatten(start_dim=0, end_dim=1)
    key_bsnd = bnsd_to_bsnd(key).to(compute_date_type).flatten(start_dim=0, end_dim=1)
    value_bsnd = bnsd_to_bsnd(value).to(compute_date_type).flatten(start_dim=0, end_dim=1)

    block_q = 128
    block_kv = 4096

    attn_out = flash_attention_cpu_golden_varlen(query_bsnd, key_bsnd, value_bsnd,
        cu_seqlens_q, cu_seqlens_kv, seqlens_q, seqlens_kv,
        softmax_scale = scale, quantize=True, quantize_p=True, block_q=block_q, block_kv=block_kv,
        s_layout = "DN", quantize_p_mode = "blockwise_snap_local", s_dtype = "fp16", v_quant_axis = "seq_k",) # blockwise_snap_local/blockwise/global

    return attn_out.reshape(b, s1, n1, qk_d).permute(0, 2, 1, 3).to(src_date_type), None

def compare(golden_name, golden_data, real_name, real_data, src_dtype):
    compare_percent = 0
    compare_result = None
    if golden_data is not None and real_data is not None:
        print(f"--------------------------------------------------------------{golden_name} vs {real_name}-------------------------------------------------------------")
        compare_percent, compare_result = check_result(golden_data.to(torch.float32), real_data.cpu().to(torch.float32), src_dtype)
    return compare_percent, compare_result

def run_fia_eager(b, n2, g, s1, s2, qk_d, v_d = None, rope_d = 0, input_layout = 'BSND', kv_storage_mode = 'continue', q_dtype = 'bfloat16', kv_dtype = None, out_dtype = None,
        rope_dtype = None, block_size = 0, act_seq_lens_q = None, act_seq_lens_kv = None, enable_softmax_lse = False,
        enable_mask = False, sparse_mode = 0, pre_tokens = 2147483647, next_tokens = 2147483647, enable_learnable_sink = False,
        inner_precise = 0, layout_q_descale = 'BSND',
        q_quant_mode = 3, q_scale_dtype = None, quant_block_size_qs = None,
        k_quant_mode = 3, k_scale_dtype = None, quant_block_size_ks = None,
        v_quant_mode = 3, v_scale_dtype = None, quant_block_size_vs = None):
    layout_q_descale = 'BNSD'
    if v_d is None:
        v_d = qk_d
    if kv_dtype is None:
        kv_dtype = q_dtype
    if out_dtype is None:
        out_dtype = q_dtype
    if rope_dtype is None:
        rope_dtype = q_dtype
    src_dtype = out_dtype
    if q_dtype == "int8":
        src_dtype = q_dtype

    # ======================== generate source data start ========================
    scale = 1 / math.sqrt(qk_d)
    num_heads = n2 * g
    num_key_value_heads = n2
    softmax_lse_flag = enable_softmax_lse

    query_bnsd_shape = (b, n2 * g, s1, qk_d)
    query = randn_x(query_bnsd_shape, 1.0).to(torch.float32)

    key_bnsd_shape = (b, n2, s2, qk_d)
    key = randn_x(key_bnsd_shape, 1.0).to(torch.float32)

    value_bnsd_shape = (b, n2, s2, v_d)
    value = randn_x(value_bnsd_shape, 1.0).to(torch.float32)

    query_rope_bnsd_shape = (b, n2 * g, s1, rope_d)
    query_rope = rand_range(query_rope_bnsd_shape, data_range=[-10, 10], dtype = torch.float32) if rope_d != 0 else None

    key_rope_bnsd_shape = (b, n2, s2, rope_d)
    key_rope = rand_range(key_rope_bnsd_shape, data_range=[-10, 10], dtype = torch.float32) if rope_d != 0 else None

    # 当存在 act_seq_lens_kv 且各 batch 实际长度均 < 外部 s2 时, KV 侧张量 (key/value/key_rope)
    # 的有效 S 维应基于实际序列最大长度 max(act_seq_lens_kv), 而非外部 s2. 此处将 s2 收敛到
    # max(act_seq_lens_kv) 并截断 KV 张量, 与 TTK golden (s2 = max(ACT_SEQ_LENS_KV)) 一致,
    # 确保 key/value/v_descale/k_descale 的 S 维与 AICPU 算子侧 ceil(max(seqused_kv)/64) 校验对齐.
    # 否则 v_descale 按 ceil(s2/32) 补齐会大于算子期望的 ceil(max(sk)/64), 触发 metadata 校验失败;
    # 且 value 按 s2 补 0 而 v_descale 只覆盖 max(sk) 会导致主算子 AICORE 越界 (507015).
    if act_seq_lens_kv is not None and len(act_seq_lens_kv) > 0:
        max_act_kv = max(min(sk, s2) for sk in act_seq_lens_kv)
        if max_act_kv < s2:
            s2 = max_act_kv
            key = key[:, :, :s2, :]
            value = value[:, :, :s2, :]
            if key_rope is not None:
                key_rope = key_rope[:, :, :s2, :]

    atten_mask = None
    if enable_mask:
        if sparse_mode == 0:
            mask_shape = (b, s1, s2)
            atten_mask = torch.rand(mask_shape) < 0.5
            select_mask = create_select_mask((s1, s2), pre_tokens, next_tokens)
            for i in range(b):
                atten_mask[i, :, :] = torch.where(select_mask, select_mask, atten_mask[i, :, :])
        elif sparse_mode == 1:
            mask_shape = (b, s1, s2)
            atten_mask = torch.rand(mask_shape) < 0.5
        elif sparse_mode == 2:
            mask_shape = (s1, s2)
            atten_mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1)
        elif sparse_mode == 3:
            left_up_pre_tokens = 214748647
            left_up_next_tokens = s2 - s1
            atten_mask = create_select_mask((s1, s2), left_up_pre_tokens, left_up_next_tokens)
        elif sparse_mode == 4:
            left_up_pre_tokens = s1 - s2 + pre_tokens
            left_up_next_tokens = s2 - s1 + next_tokens
            atten_mask = create_select_mask((s1, s2), left_up_pre_tokens, left_up_next_tokens)
    # ======================== generate source data finish ========================

    attn_out_layout = get_attn_out_layout(input_layout)
    # ======================== excute q/k/v mxfp4 quant golden [Flash Attention] ========================
    cpu_attn_out_mxfp4_flash, cpu_softmax_lse_mxfp4_flash = cpu_golden_qkv_mxfp4_flash_attn(query, key, value,
                                                atten_mask=atten_mask,
                                                actual_seq_lengths=act_seq_lens_q,
                                                actual_seq_lengths_kv=act_seq_lens_kv,
                                                query_rope=query_rope,
                                                key_rope=key_rope,
                                                scale=scale, pre_tokens=pre_tokens, next_tokens=next_tokens,
                                                sparse_mode=sparse_mode, inner_precise=inner_precise,
                                                softmax_lse_flag=softmax_lse_flag,
                                                src_date_type=get_dtype(src_dtype))
    cpu_attn_out_mxfp4_flash = rearrange_by_layout(cpu_attn_out_mxfp4_flash, attn_out_layout, b, act_seq_lens_q)
    if cpu_softmax_lse_mxfp4_flash is not None:
        softmax_lse_layout = get_softmax_lse_layout(input_layout)
        cpu_softmax_lse_mxfp4_flash = rearrange_by_layout(cpu_softmax_lse_mxfp4_flash, softmax_lse_layout, b, act_seq_lens_q)
    print(cpu_attn_out_mxfp4_flash.shape)

    # ======================== generate npu data start ========================
    # query dtype
    q_descale = None
    if q_dtype == "int8":
        pass
    elif q_dtype in ("fp4_e2m1", "fp8_e4m3"):
        if q_quant_mode == 5:
            query, q_descale = mxfp4_quantize_pack_last(query, quant_axis = -1, mode = "baseline")
    else:
        query = query.to(get_dtype(q_dtype))
        if query_rope is not None:
            query_rope = query_rope.to(get_dtype(rope_dtype))

    # query shape
    query_layout = get_query_layout(input_layout)
    query = rearrange_by_layout(query, query_layout, b, act_seq_lens_q)
    if query_rope is not None:
        query_rope = rearrange_by_layout(query_rope, query_layout, b, act_seq_lens_q)
    if q_descale is not None:
        q_descale = rearrange_by_layout(q_descale, query_layout, b, act_seq_lens_q)
        q_descale = transpose_qscale(q_descale, query_layout, n2)

    # kv dtype
    k_descale = None
    v_descale = None
    if kv_dtype == "int8":
        pass
    elif kv_dtype in ("fp4_e2m1", "fp8_e4m3"):
        if q_quant_mode == 5:
            key, k_descale = mxfp4_quantize_pack_last(key, quant_axis = -1, mode = "baseline")
            if act_seq_lens_kv is not None and any(sk < s2 for sk in act_seq_lens_kv):
                n_blocks_full = (s2 + 31) // 32
                n2_actual = value.shape[1]
                vd_actual = value.shape[3]
                packed_value_list = []
                v_descale_list = []

                for b_idx in range(b):
                    sk = act_seq_lens_kv[b_idx] if len(act_seq_lens_kv) > 1 else act_seq_lens_kv[0]
                    sk = min(sk, s2)
                    v_b = value[b_idx:b_idx+1, :, :sk, :]  # (1, n2, sk, v_d)
                    v_b_packed, v_b_scale = mxfp4_quantize_pack_last(v_b, quant_axis = -2, mode = "baseline")
                    # packed fp4 补 0 到 s2
                    packed_padded = torch.zeros((1, n2_actual, s2, vd_actual // 2), dtype = v_b_packed.dtype)
                    packed_padded[:, :, :sk, :] = v_b_packed
                    packed_value_list.append(packed_padded)
                    # scale 补到 n_blocks_full, padding 部分用 e8m0=127 (scale=2^0=1.0)
                    n_blocks_actual = v_b_scale.shape[2]
                    scale_padded = torch.full((1, n2_actual, n_blocks_full, vd_actual), 127, dtype = v_b_scale.dtype)
                    scale_padded[:, :, :n_blocks_actual, :] = v_b_scale
                    v_descale_list.append(scale_padded)
                value = torch.cat(packed_value_list, dim = 0)
                v_descale = torch.cat(v_descale_list, dim = 0)

            else:
                value, v_descale = mxfp4_quantize_pack_last(value, quant_axis = -2, mode = "baseline")
            # n_blocks 为奇数时 pad 一个 block (e8m0=127, scale=2^0=1.0), 使 reshape 的 //2 不丢数据
            if v_descale.shape[2] % 2 != 0:
                pad_block = torch.full(
                    (v_descale.shape[0], v_descale.shape[1], 1, v_descale.shape[3]),
                    127, dtype = v_descale.dtype)
                v_descale = torch.cat([v_descale, pad_block], dim = 2)
            v_descale = v_descale.reshape(
                    v_descale.shape[0], v_descale.shape[1], v_descale.shape[2] // 2, 2, v_descale.shape[3]
                ).transpose(-1, -2).reshape(
                    v_descale.shape[0], v_descale.shape[1], v_descale.shape[2] // 2, v_descale.shape[3] * 2)
    else:
        key = key.to(get_dtype(kv_dtype))
        value = value.to(get_dtype(kv_dtype))
        if key_rope is not None:
            key_rope = key_rope.to(get_dtype(rope_dtype))

    block_table = None
    if kv_storage_mode == "continue":
        key = rearrange_by_layout(key, query_layout, b, act_seq_lens_kv)
        value = rearrange_by_layout(value, query_layout, b, act_seq_lens_kv)
        if key_rope is not None:
            key_rope = rearrange_by_layout(key_rope, query_layout, b, act_seq_lens_kv)
        if k_descale is not None:
            aligned_seq_lens = [(x + 31) // 32 for x in act_seq_lens_kv]
            k_descale = rearrange_by_layout(k_descale, query_layout, b, aligned_seq_lens)
            k_descale = transpose_kscale(k_descale, query_layout)
        if v_descale is not None:
            aligned_seq_lens = [(x + 63) // 64 for x in act_seq_lens_kv]
            v_descale = rearrange_by_layout(v_descale, query_layout, b, aligned_seq_lens)
    else:
        block_table = get_block_table(b, act_seq_lens_kv, block_size)
        key = rearrange_by_block_table(key, block_table, block_size, b, act_seq_lens_kv, kv_storage_mode, kv_dtype)
        value = rearrange_by_block_table(value, block_table, block_size, b, act_seq_lens_kv, kv_storage_mode, kv_dtype)
        if key_rope is not None:
            key_rope = rearrange_by_block_table(key_rope, block_table, block_size, b, act_seq_lens_kv, kv_storage_mode, kv_dtype)
        if k_descale is not None:
            k_descale = rearrange_by_block_table(k_descale, block_table, block_size, b, act_seq_lens_kv, kv_storage_mode, kv_dtype)
        if v_descale is not None:
            v_descale = rearrange_by_block_table(v_descale, block_table, block_size, b, act_seq_lens_kv, kv_storage_mode, kv_dtype)
    if v_descale is not None:
        v_descale = v_descale.view(*v_descale.shape[:-1], -1, 2)

    # update act_seq_lens
    cu_seqlens_q = update_act_seq_lens_for_tnd(query_layout, b, act_seq_lens_q)
    cu_seqlens_kv = None
    if kv_storage_mode == 'continue':
        cu_seqlens_kv = update_act_seq_lens_for_tnd(query_layout, b, act_seq_lens_kv)

    # update mask
    if enable_mask:
        if sparse_mode == 2 or sparse_mode == 3 or sparse_mode == 4:
            mask_shape = (2048, 2048)
            atten_mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1)

    layout_kv = query_layout
    if kv_storage_mode == "pa_bbh":
        layout_kv = "PA_BBND"
    elif kv_storage_mode == "pa_bnbd":
        layout_kv = "PA_BNBD"
    elif kv_storage_mode == "pa_nz":
        layout_kv = "PA_NZ"

    # ======================== generate npu data finish ========================

    # ======================== excute npu start ========================
    if cu_seqlens_q is not None:
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32)
    if cu_seqlens_kv is not None:
        cu_seqlens_kv = torch.tensor(cu_seqlens_kv, dtype=torch.int32)
    if act_seq_lens_q is not None:
        act_seq_lens_q = torch.tensor(act_seq_lens_q, dtype=torch.int32)
    if act_seq_lens_kv is not None:
        act_seq_lens_kv = torch.tensor(act_seq_lens_kv, dtype=torch.int32)
    torch_npu.npu.set_device(int(DEVICE_ID))
    torch.npu.synchronize()
    metadata = quant_flash_attn_metadata(
                            num_heads_q   = num_heads,
                            num_heads_kv  = num_key_value_heads,
                            head_dim      = qk_d,
                            quant_mode    = q_quant_mode,
                            cu_seqlens_q  = TO_NPU(cu_seqlens_q),
                            cu_seqlens_kv = TO_NPU(cu_seqlens_kv),
                            seqused_q     = TO_NPU(act_seq_lens_q),
                            seqused_kv    = TO_NPU(act_seq_lens_kv),
                            v_descale     = TO_NPU(v_descale),
                            batch_size    = b,
                            max_seqlen_q  = -1,
                            max_seqlen_kv = -1,
                            mask_mode     = sparse_mode,
                            win_left      = pre_tokens,
                            win_right     = next_tokens,
                            layout_q      = query_layout,
                            layout_q_descale = layout_q_descale,
                            layout_kv     = layout_kv,
                            layout_out    = attn_out_layout)
    torch.npu.synchronize()
    npu_attn_out, npu_softmax_lse = quant_flash_attn(
                            TO_NPU(query.contiguous()),
                            TO_NPU(key.contiguous()),
                            TO_NPU(value.contiguous()),
                            TO_NPU(q_descale.contiguous()),
                            TO_NPU(k_descale.contiguous()),
                            TO_NPU(v_descale.contiguous()),
                            quant_mode=q_quant_mode,
                            block_table=TO_NPU(block_table),
                            p_scale=None,
                            cu_seqlens_q=TO_NPU(cu_seqlens_q),
                            cu_seqlens_kv=TO_NPU(cu_seqlens_kv),
                            seqused_q=TO_NPU(act_seq_lens_q),
                            seqused_kv=TO_NPU(act_seq_lens_kv),
                            sinks=None,
                            attn_mask=TO_NPU(atten_mask),
                            metadata=metadata,
                            softmax_scale=scale,
                            mask_mode=sparse_mode,
                            win_left=pre_tokens,
                            win_right=next_tokens,
                            max_seqlen_q=-1,
                            max_seqlen_kv=-1,
                            layout_q=query_layout,
                            layout_q_descale=layout_q_descale,
                            layout_kv=layout_kv,
                            layout_out=attn_out_layout,
                            return_softmax_lse=softmax_lse_flag)
    torch.npu.synchronize()
    # ======================== excute npu finish ========================
    run_percent = 0.0
    run_result = "Unknown"
    run_percent, run_result = compare('cpu_mxfp4_attn_out_flash', cpu_attn_out_mxfp4_flash, 'npu_attn_out', npu_attn_out, src_dtype)
    compare('cpu_mxfp4_softmax_flash', cpu_softmax_lse_mxfp4_flash, 'npu_softmax_lse', npu_softmax_lse, src_dtype)
    print()
    return run_percent, run_result

def parse_shape(s):
    try:
        # 支持输入 3,4 或 [3,4] 或 (3,4)
        s = s.strip('[]()')
        return tuple(map(int, s.split(',')))
    except:
        raise argparse.ArgumentTypeError("Shape 格式错误，示例：3,4 或 [3,4]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # required base params
    parser.add_argument('--b', required=True, type=int, help='batch size')
    parser.add_argument('--n2', required=True, type=int, help='head_num of value')
    parser.add_argument('--g', required=True, type=int, help='g = key_head_num / value_head_num')
    parser.add_argument('--s1', required=True, type=int, help='max sequence length of query')
    parser.add_argument('--s2', required=True, type=int, help='max sequence length of key and value')
    parser.add_argument('--qk_d', required=True, type=int, default=128, help='head_dim of query and key')
    parser.add_argument('--q_quant_mode', required=True, type=int, default=3, help='')
    parser.add_argument('--k_quant_mode', required=True, type=int, default=3, help='')
    parser.add_argument('--v_quant_mode', required=True, type=int, default=3, help='')
    parser.add_argument('--q_dtype', required=True, type=str, default="bfloat16", help='dtype of query')
    parser.add_argument('--kv_dtype', required=True, type=str, default=None, help='dtype of key and value, default is q_dtype')
    # other base params
    parser.add_argument('--v_d', type=int, default=None, help='head_dim of value, default is qk_d')
    parser.add_argument('--rope_d', type=int, default=0, help='head_dim of query_rope and key_rope, 0: no exist query_rope and key_rope')
    parser.add_argument('--input_layout', type=str, default='BSND', choices=['BSH','BSND','BNSD','BNSD_BSND','BSH_BNSD','BSND_BNSD','TND','NTD','TND_NTD','NTD_TND','BSH_NBSD','BSND_NBSD','BNSD_NBSD'], help="layout of query and attention_out")
    parser.add_argument('--layout_q_descale', type=str, default='BSND', choices=['BSND','BNSD','TND','N2TGD'], help="layout of q descale (mxfp4 unused, passthrough)")
    parser.add_argument('--kv_storage_mode', type=str, default='continue', choices=['continue','pa_bbh','pa_bnbd','pa_nz'], help="for get layout of key and value")
    parser.add_argument('--out_dtype', type=str, default=None, help='dtype of attention_out, default is q_dtype')
    parser.add_argument('--rope_dtype', type=str, default=None, help='dtype of query_rope and key_rope, default is q_dtype')
    # page attention
    parser.add_argument('--block_size', type=int, default=0, help='when paga_attention, block_size of kv cache')
    # q and kv actual_seq_lens
    parser.add_argument('--act_seq_lens_q', type=int, nargs='*', default=None, help='actual sequence of query for every batch, should not greated than s1, len is 1/B/>B')
    parser.add_argument('--act_seq_lens_kv', type=int, nargs='*', default=None, help='sequence of key and value for every batch, should not greated than s2, len is 1/B/>B')
    # lse
    parser.add_argument('--enable_softmax_lse', action='store_true', help='output softmax_lse')
    # mask
    parser.add_argument('--enable_mask', action='store_true', help='enable attention mask')
    parser.add_argument('--sparse_mode', type=int, default=0, choices=[0, 1, 2, 3, 4], help='')
    parser.add_argument('--pre_tokens', type=int, default=2147483647, help='')
    parser.add_argument('--next_tokens', type=int, default=2147483647, help='')
    # learnable sink
    parser.add_argument('--enable_learnable_sink', action='store_true', help='enable learnable_sink')
    # high precise/high pref, row invalid
    parser.add_argument('--innerPrecise', type=int, default=0, choices=[0, 1, 2, 3], help='0/1/2/3')
    # antiquant and fullquant
    parser.add_argument('--q_scale_dtype', type=str, default=None, help='dtype of query dequant scale')
    parser.add_argument('--k_scale_dtype', type=str, default=None, help='dtype of key dequant scale')
    parser.add_argument('--v_scale_dtype', type=str, default=None, help='dtype of value dequant scale')
    parser.add_argument('--quant_block_size_qs', type=int, default=None, help='')
    parser.add_argument('--quant_block_size_ks', type=int, default=None, help='')
    parser.add_argument('--quant_block_size_vs', type=int, default=None, help='')
    args = parser.parse_args()

    qfa_percent, qfa_result = run_fia_eager(args.b, args.n2, args.g, args.s1, args.s2, args.qk_d, args.v_d, args.rope_d,
        args.input_layout, args.kv_storage_mode, args.q_dtype, args.kv_dtype, args.out_dtype,
        args.rope_dtype, args.block_size, args.act_seq_lens_q, args.act_seq_lens_kv, args.enable_softmax_lse,
        args.enable_mask, args.sparse_mode, args.pre_tokens, args.next_tokens,
        args.enable_learnable_sink, args.innerPrecise, args.layout_q_descale,
        args.q_quant_mode, args.q_scale_dtype, args.quant_block_size_qs,
        args.k_quant_mode, args.k_scale_dtype, args.quant_block_size_ks,
        args.v_quant_mode, args.v_scale_dtype, args.quant_block_size_vs)