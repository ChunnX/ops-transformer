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

import os
from functools import partial
from quant_lightning_indexer_v2_golden import GeneralizedQLIV2
import pandas as pd
import numpy as np
import torch
import torch_npu
import pytest
import random
import math
import ast
import argparse

FP32_FRACTION_BITS = 23        # fp32尾数位数

HIF8_EXP_ZERO_THRESHOLD = -23  # 边界值
HIF8_EXP_DML_MIN = -22         # DML最小指数
HIF8_EXP_DML_MAX = -15         # DML最大指数
HIF8_EXP_D0 = 0                # D0指数值
HIF8_EXP_D1_BOUNDARY = 1       # D1指数值
HIF8_EXP_D2_MIN, HIF8_EXP_D2_MAX = 2, 3   # D2指数范围
HIF8_EXP_D3_MIN, HIF8_EXP_D3_MAX = 4, 7   # D3指数范围
HIF8_EXP_D4_MIN, HIF8_EXP_D4_MAX = 8, 15  # D4指数范围

HIF8_DOT_DML = 0               # DML: Denormal Low, 指数范围 -22 ~ -16, 0位尾数
HIF8_DOT_D0 = 1                # D0: 指数为0，3位尾数（最高精度）
HIF8_DOT_D1 = 2                # D1: 指数为±1，3位尾数
HIF8_DOT_D2 = 4                # D2: 指数为±2 ~ ±3，3位尾数
HIF8_DOT_D3 = 8                # D3: 指数为±4 ~ ±7，2位尾数
HIF8_DOT_D4 = 12               # D4: 指数为±8 ~ ±15，1位尾数（最低精度）
HIF8_DOT_INVALID = -1          # 无效状态

HIF8_FRAC_BITS_DML = 0         # DML档位尾数位数
HIF8_FRAC_BITS_D0 = 3          # D0档位尾数位数
HIF8_FRAC_BITS_D1 = 3          # D1档位尾数位数
HIF8_FRAC_BITS_D2 = 3          # D2档位尾数位数
HIF8_FRAC_BITS_D3 = 2          # D3档位尾数位数
HIF8_FRAC_BITS_D4 = 1          # D4档位尾数位数

HIF8_EXP_BITS_DML = 3          # DML档位指数位数
HIF8_EXP_BITS_D0 = 0           # D0档位指数位数
HIF8_EXP_BITS_D1 = 1           # D1档位指数位数
HIF8_EXP_BITS_D2 = 2           # D2档位指数位数
HIF8_EXP_BITS_D3 = 3           # D3档位指数位数
HIF8_EXP_BITS_D4 = 4           # D4档位指数位数

HIF8_ZERO = 0
HIF8_NAN = 128                 # 0b10000000, NaN
HIF8_NEG_INF = 239             # 0b11101111, -inf
HIF8_NEG_MAX = 238             # 0b11101110, 负极大值
HIF8_POS_INF = 111             # 0b01101111, +inf
HIF8_POS_MAX = 110             # 0b01101110, 正极大值

HIF8_SIGN_MASK = 128           # 0b10000000, 符号位掩码
HIF8_DOT_MASK = 120            # 0b01110000, dot值掩码
HIF8_FRAC_MASK_3BIT = 7        # 0b00000111, 3位尾数掩码（D0/D1/D2）
HIF8_FRAC_MASK_2BIT = 3        # 0b00000011, 2位尾数掩码（D3）
HIF8_FRAC_MASK_1BIT = 1        # 0b00000001, 1位尾数掩码（D4）
HIF8_EXP_MASK_DML = 7          # 0b00000111, DML指数掩码（bit0-2）
HIF8_EXP_MASK_D4 = 30          # 0b00011110, D4指数掩码（bit1-4）
HIF8_EXP_MASK_D3 = 28          # 0b00011100, D3指数掩码（bit2-4）
HIF8_EXP_MASK_D2 = 24          # 0b00011000, D2指数掩码（bit3-4）
HIF8_EXP_SIGN_MASK_D1 = 8      # 0b00001000, D1指数掩码（bit3）

HIF8_DOT_BIT_SHIFT = 3         # Dot值在HiF8中的起始位置(bit3)
HIF8_DML_EXP_OFFSET = 23       # DML指数偏移值
HIF8_OVERFLOW_SCALE = 1.25     # 溢出阈值缩放因子
HIF8_MAX_FINITE_VALUE = 32768  # 最大有限值（非饱和模式下的边界值, 2^15

SSR_T14_MASK = 16383           # 0b0011 1111 1111 1111, 14位低位掩码
SSR_F14_OFFSET = 8192          # 0b0010 0000 0000 0000, F14偏移值
SSR_DML_SHIFT = 10             # SSR舍入移位值
SSR_RESERVED_BITS = 14         # SSR舍入保留位数
HYBRID_ROUND_EXP_THRESHOLD = 4 # 混合舍入的指数分界点

def load_excel_test_cases(excel_file_path: str, sheetname: str):

    """
    从 Excel 文件加载测试用例。

    参数:
        excel_file_path (str): Excel 文件的路径。
        sheetname (str, optional): 工作表名称。若未提供，则默认 'Sheet1'。

    返回:
        list[tuple]: 测试用例元组列表，每个元组包含 20+ 个字段。
                       若失败或跳过，则返回空列表。
    """
    # 优先使用传入的 sheetname，否则尝试从环境变量获取
    if sheetname is None:
        sheetname = 'Sheet1'

    # 检查文件是否存在
    if not os.path.exists(excel_file_path):
        pytest.skip(f"Excel file not found: {excel_file_path}", allow_module_level=True)

    try:
        # 读取 Excel 文件的指定 sheet
        df = pd.read_excel(excel_file_path, sheet_name=sheetname)
        df = df.replace({np.nan: None, pd.NA: None})

        # 定义必需的列名
        required_columns = [
            'Testcase_Name', 'batch_size', 'q_seq','k_seq','q_t_size', 'k_t_size',
            'q_head_num', 'k_head_num', 'head_dim', 'block_size', 'block_num',
            'qk_dtype', 'dequant_dtype', 'actual_seq_dtype', 'cu_seqlens_q', 'cu_seqlens_k',
            'seqused_q', 'seqused_k', 'cmp_residual_k', 'max_seqlen_q', 'quant_mode',
            'layout_query', 'layout_key', 'sparse_count', 'sparse_mode', 'query_datarange',
            'key_datarange', 'weights_datarange', 'q_scale_datarange',
            'k_scale_datarange', 'cmp_ratio', 'return_value'
        ]

        # 检查是否缺少必要列
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            pytest.skip(f"Missing required columns in Excel: {missing_cols}", allow_module_level=True)

        # 构建测试用例列表
        test_cases = []
        for _, row in df.iterrows():
            test_cases.append((
                row['Testcase_Name'],
                row['batch_size'],
                row['q_seq'],
                row['k_seq'],
                row['q_t_size'],
                row['k_t_size'],
                row['q_head_num'],
                row['k_head_num'],
                row['head_dim'],
                row['block_size'],
                row['block_num'],
                row['qk_dtype'],
                row['dequant_dtype'],
                row['actual_seq_dtype'],
                row['cu_seqlens_q'],
                row['cu_seqlens_k'],
                row['seqused_q'],
                row['seqused_k'],
                row['cmp_residual_k'],
                row['max_seqlen_q'],
                row['quant_mode'],
                row['layout_query'],
                row['layout_key'],
                row['sparse_count'],
                row['sparse_mode'],
                row['query_datarange'],
                row['key_datarange'],
                row['weights_datarange'],
                row['q_scale_datarange'],
                row['k_scale_datarange'],
                row['cmp_ratio'],
                row['return_value']
            ))

        return test_cases

    except Exception as e:
        pytest.skip(f"Failed to read Excel file: {e}", allow_module_level=True)
        return None

def fp32_ta_round_to_hif8(fraction32_int, hif8_bits_num, exponent):
    if exponent == HIF8_EXP_ZERO_THRESHOLD:
        return True, 0
    hif8_value_tmp = fraction32_int >> (FP32_FRACTION_BITS - (hif8_bits_num + 1))
    if hif8_value_tmp == pow(2, hif8_bits_num + 1) - 1:
        return True, 0
    elif hif8_value_tmp == 0:
        return False, 0
    elif hif8_value_tmp % 2 == 1:
        hif8_value_tmp += 1
        return False, hif8_value_tmp >> 1
    else:
        return False, hif8_value_tmp >> 1

def fp32_ssr_round_to_hif8(fraction32_int, hif8_bits_num, exponent):
    t14_mask = SSR_T14_MASK
    if exponent == HIF8_EXP_ZERO_THRESHOLD:
        f14_values = (fraction32_int >> SSR_DML_SHIFT) + SSR_F14_OFFSET
        t14_values = fraction32_int & t14_mask
        hif8_value = 0
    else:
        hif8_value = fraction32_int >> (FP32_FRACTION_BITS - hif8_bits_num)
        f14_t14 = fraction32_int - (hif8_value << (FP32_FRACTION_BITS - hif8_bits_num))
        f14_values = f14_t14 >> (FP32_FRACTION_BITS - hif8_bits_num - SSR_RESERVED_BITS)
        t14_values = f14_t14 & t14_mask
    if f14_values >= t14_values:
        if hif8_value == pow(2, hif8_bits_num) - 1:
            return True, 0
        else:
            hif8_value += 1
            return False, hif8_value
    else:
        return False, hif8_value

def get_hif8_fraction_bits_number(exponent):
    if exponent < HIF8_EXP_DML_MIN:
        return HIF8_DOT_INVALID, HIF8_EXP_BITS_DML, HIF8_FRAC_BITS_DML
    if HIF8_EXP_DML_MIN <= exponent < HIF8_EXP_DML_MAX:
        return HIF8_DOT_DML, HIF8_EXP_BITS_DML, HIF8_FRAC_BITS_DML
    if exponent == HIF8_EXP_D0:
        return HIF8_DOT_D0, HIF8_EXP_BITS_D0, HIF8_FRAC_BITS_D0
    if abs(exponent) == HIF8_EXP_D1_BOUNDARY:
        return HIF8_DOT_D1, HIF8_EXP_BITS_D1, HIF8_FRAC_BITS_D1
    if HIF8_EXP_D2_MIN <= abs(exponent) <= HIF8_EXP_D2_MAX:
        return HIF8_DOT_D2, HIF8_EXP_BITS_D2, HIF8_FRAC_BITS_D2
    if HIF8_EXP_D3_MIN <= abs(exponent) <= HIF8_EXP_D3_MAX:
        return HIF8_DOT_D3, HIF8_EXP_BITS_D3, HIF8_FRAC_BITS_D3
    if HIF8_EXP_D4_MIN <= abs(exponent) <= HIF8_EXP_D4_MAX:
        return HIF8_DOT_D4, HIF8_EXP_BITS_D4, HIF8_FRAC_BITS_D4
    if exponent > HIF8_EXP_D4_MAX:
        return HIF8_DOT_D4, HIF8_EXP_BITS_D4, HIF8_DOT_INVALID

def cvt_float32_to_hifuint8(x, round_mode = "round", over_mode = True):
    sign = False
    sign_int_value = 0
    x_abs = math.fabs(x)
    ec = 0
    over_value = HIF8_OVERFLOW_SCALE * pow(2.0, HIF8_EXP_D4_MAX + ec)
    if x < 0.0:
        sign = True
        sign_int_value = HIF8_SIGN_MASK
    if torch.isinf(x) or x_abs >= over_value:
        if sign:
            if over_mode:
                return HIF8_NEG_INF
            else:
                return HIF8_NEG_MAX
        else:
            if over_mode:
                return HIF8_POS_INF
            else:
                return HIF8_POS_MAX
    if torch.isnan(x):
        if over_mode:
            return HIF8_NAN
        else:
            return 0
    if x_abs == 0.0:
        return 0
    exponent = math.floor(math.log2(x_abs))
    if round_mode == "hybrid":
        if abs(exponent) < HYBRID_ROUND_EXP_THRESHOLD:
            cut_bit_type = "TA"
        else:
            cut_bit_type = "SSR"
    elif round_mode == "round":
        cut_bit_type = "TA"
    elif round_mode == "storound":
        cut_bit_type = "SSR"
    else:
        cut_bit_type = "TA"
    fraction_int = int(x_abs * pow(2, FP32_FRACTION_BITS) * pow(2, -exponent) - pow(2, FP32_FRACTION_BITS))
    dot_hif8_value, exponent_hif8_bits, fraction_hif8_bits = get_hif8_fraction_bits_number(exponent)
    if cut_bit_type == "TA":
        carry_exp_status, hif8_frac_value = fp32_ta_round_to_hif8(fraction_int, fraction_hif8_bits, exponent)
    elif cut_bit_type == "SSR":
        carry_exp_status, hif8_frac_value = fp32_ssr_round_to_hif8(fraction_int, fraction_hif8_bits, exponent)
    else:
        print(f"unknown round type")
        return 0

    if carry_exp_status:
        exponent += 1
        dot_hif8_value, exponent_hif8_bits, fraction_hif8_bits_new = get_hif8_fraction_bits_number(exponent)
        fraction_hif8_bits = fraction_hif8_bits_new
    if exponent < HIF8_EXP_ZERO_THRESHOLD:
        return 0
    if exponent < 0:
        sig_exp = 1
    else:
        sig_exp = 0
    if dot_hif8_value <= 0:
        if exponent <= HIF8_EXP_ZERO_THRESHOLD:
            return 0
        else:
            return sign_int_value + exponent + HIF8_DML_EXP_OFFSET
    elif dot_hif8_value == 1:
        dot_int_value = dot_hif8_value << HIF8_DOT_BIT_SHIFT
        hif8_int_value = sign_int_value + dot_int_value + hif8_frac_value
    else:
        abs_exponent = abs(exponent)
        abs_exponent = abs_exponent - pow(2, exponent_hif8_bits - 1)
        exponent_int_value = abs_exponent << fraction_hif8_bits
        sig_exp = sig_exp << (exponent_hif8_bits - 1 + fraction_hif8_bits)
        dot_int_value = dot_hif8_value << HIF8_DOT_BIT_SHIFT
        hif8_int_value = sign_int_value + dot_int_value + sig_exp + exponent_int_value + hif8_frac_value
    return hif8_int_value

def trans_float_tensor_to_hifuint8(in_tensor, round_mode = "round", over_mode = True):
    tensor_shape = in_tensor.shape
    tensor_shape_size = in_tensor.numel()
    if tensor_shape_size == 1.0:
        tensor_shape_size = int(tensor_shape_size)
    
    out_tensor = torch.zeros(tensor_shape_size).to(torch.uint8)
    in_tensor = in_tensor.reshape(tensor_shape_size)
    for i in range(tensor_shape_size):
        out_tensor[i] = cvt_float32_to_hifuint8(in_tensor[i], round_mode, over_mode)
    out_tensor = out_tensor.view(torch.uint8)
    out_tensor = out_tensor.reshape(tensor_shape)
    return out_tensor

def qliv2_output_single(data_case):
    casename = data_case[0]
    params = data_case[1:]

    batch_size, q_seq, k_seq, q_t_size, k_t_size, q_head_num, k_head_num, head_dim, block_size, block_num, \
    qk_dtype, dequant_dtype, actual_seq_dtype, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, cmp_residual_k,\
    max_seqlen_q, quant_mode, layout_query, layout_key, sparse_count, sparse_mode, query_datarange, key_datarange,\
    weights_datarange,q_scale_datarange, k_scale_datarange, cmp_ratio, return_value = params
    if qk_dtype == 'INT8':
        qk_dtype = torch.int8
    elif qk_dtype == 'FLOAT8_E4M3FN':
        qk_dtype = torch.float8_e4m3fn

    if dequant_dtype == 'FP16':
        dequant_dtype = torch.float16
    elif dequant_dtype == 'FP32':
        dequant_dtype = torch.float32

    if actual_seq_dtype == 'INT32':
        actual_seq_dtype = torch.int32

    if query_datarange is not None:
        query_datarange = ast.literal_eval(query_datarange)
    if key_datarange is not None:
        key_datarange = ast.literal_eval(key_datarange)
    if weights_datarange is not None:
        weights_datarange = ast.literal_eval(weights_datarange)
    if q_scale_datarange is not None:
        q_scale_datarange = ast.literal_eval(q_scale_datarange)
    if k_scale_datarange is not None:
        k_scale_datarange = ast.literal_eval(k_scale_datarange)
    
    if qk_dtype == torch.uint8:
        hifp8mode = 1
        query_dtype = torch_npu.hifloat8 # ACL_HIFLOAT8
        key_dtype = torch_npu.hifloat8   # ACL_HIFLOAT8
    else:
        hifp8mode = 0

    # ======================== 核心推导：从 cu_seqlens / seqused 推导个体长度 ========================
    # 辅助函数：从前缀和 cu_seqlens [B+1] 推导个体长度 [B]
    def _cu_seqlens_to_lengths(cu_list):
        return [cu_list[i+1] - cu_list[i] for i in range(len(cu_list) - 1)]

    # Q 侧个体长度（CPU golden 用）
    if layout_query == "TND":
        # TND: 必传 cu_seqlens_q，从差分推导个体长度
        assert cu_seqlens_q is not None, "TND layout requires cu_seqlens_q"
        cu_seqlens_q = ast.literal_eval(cu_seqlens_q)
        lengths_q_list = _cu_seqlens_to_lengths(cu_seqlens_q)
    else:
        # BSND: 从 seqused_q 获取，若 None 则用 q_seq 填满
        if seqused_q is not None:
            lengths_q_list = list(ast.literal_eval(seqused_q))
        else:
            lengths_q_list = [q_seq] * batch_size

    # K 侧个体长度（CPU golden 用）
    if layout_key == "TND":
        # TND: 必传 cu_seqlens_k，从差分推导个体长度
        assert cu_seqlens_k is not None, "TND layout requires cu_seqlens_k"
        cu_seqlens_k = ast.literal_eval(cu_seqlens_k)
        lengths_k_list = _cu_seqlens_to_lengths(cu_seqlens_k)
    elif layout_key == "PA_BBND":
        # PA_BBND: 从 seqused_k 获取
        assert seqused_k is not None, f"{layout_key} layout requires seqused_k"
        lengths_k_list = list(ast.literal_eval(seqused_k))
    else:
        # BSND: 从 seqused_k 获取，若 None 则用 q_seq 填满
        if seqused_k is not None:
            lengths_k_list = list(ast.literal_eval(seqused_k))
        else:
            lengths_k_list = [k_seq] * batch_size

    # ======================== 构造 NPU 输入 tensor ========================
    # cu_seqlens tensor（仅 TND 传入）
    if layout_query == "TND":
        cu_seqlens_query = torch.tensor(cu_seqlens_q).to(actual_seq_dtype).npu()
    else:
        cu_seqlens_query = None

    if layout_key == "TND":
        cu_seqlens_key = torch.tensor(cu_seqlens_k).to(actual_seq_dtype).npu()
    else:
        cu_seqlens_key = None

    # seqused tensor
    if seqused_q is not None:
        seqused_q_tensor = torch.tensor(lengths_q_list).to(actual_seq_dtype).npu()
    else:
        seqused_q_tensor = None
    if seqused_k is not None:
        seqused_k_tensor = torch.tensor(lengths_k_list).to(actual_seq_dtype).npu()
    else:
        seqused_k_tensor = None

    # ======================== CPU golden forward 用的 actual_seq ========================
    # TND:     actual_seq 是前缀和格式，即 cu_seqlens[1:]（去掉首位 0）
    #          golden.forward 内部会 trans_tnd_actseq 差分为个体长度
    # BSND/PA: actual_seq 是个体长度，即 seqused
    # （actual_seq始终传入，CPU golden 也需要）
    if layout_query == "TND":
        actual_seq_lengths_query = torch.tensor(cu_seqlens_q[1:]).to(actual_seq_dtype).npu()
    else:
        actual_seq_lengths_query = torch.tensor(lengths_q_list).to(actual_seq_dtype).npu()

    if layout_key == "TND":
        actual_seq_lengths_key = torch.tensor(cu_seqlens_k[1:]).to(actual_seq_dtype).npu()
    else:
        actual_seq_lengths_key = torch.tensor(lengths_k_list).to(actual_seq_dtype).npu()

    # PA_BBND key 构造用的 act_seq_k 列表
    act_seq_k = lengths_k_list

    # 检查 cmp_residual_k 参数
    if (sparse_mode == 0 or cmp_ratio == 1) and cmp_residual_k is not None:
        print(f"Warning: sparse_mode={sparse_mode} or cmp_ratio={cmp_ratio}, "
              f"cmp_residual_k={cmp_residual_k}, should be None")
        print("Hint: set cmp_residual_k to None when sparse_mode==0 or cmp_ratio==1")

    # cmp_residual_k for CPU golden (always a list with zeros when cmp_ratio==1 or sparse_mode==0)
    if cmp_ratio == 1 or sparse_mode == 0:
        cmp_residual_k_for_cpu = [0] * batch_size
    else:
        cmp_residual_k_for_cpu = list(ast.literal_eval(cmp_residual_k))

    # cmp_residual_k for NPU (None when cmp_ratio==1 or sparse_mode==0, tensor otherwise)
    if cmp_ratio == 1 or sparse_mode == 0:
        cmp_residual_k_for_npu = None
    else:
        cmp_residual_k_for_npu = torch.tensor(ast.literal_eval(cmp_residual_k)).to(actual_seq_dtype).npu()

    # ======================== 构造 GeneralizedQLIV2 用于 CPU golden ========================
    # GeneralizedQLIV2 需要 act_seq 个体长度（用于 TND→BNSD 转换等）
    test_qliv2 = GeneralizedQLIV2(batch_size, q_seq, k_seq, q_t_size, k_t_size, q_head_num, k_head_num, head_dim, block_size,
                    block_num, qk_dtype, dequant_dtype, actual_seq_dtype, cu_seqlens_q, cu_seqlens_k,
                    lengths_q_list, lengths_k_list, cmp_residual_k_for_cpu, max_seqlen_q, quant_mode,
                    layout_query, layout_key, sparse_count,
                    sparse_mode, query_datarange, key_datarange, weights_datarange, q_scale_datarange,
                    k_scale_datarange, cmp_ratio, return_value)


    if layout_query == "BSND":
        query = torch.tensor(np.random.uniform(query_datarange[0], query_datarange[1],(batch_size, q_seq, q_head_num, head_dim))).to(torch.float)
        if hifp8mode == 1:
            query = trans_float_tensor_to_hifuint8(query, round_mode = "hybrid", over_mode = True).npu()
        else:
            query = query.to(qk_dtype).npu()
        
        q_scale = random.uniform(q_scale_datarange[0], q_scale_datarange[1])
        if quant_mode == 4:
            query_dequant_scale = torch.tensor([q_scale]).to(dequant_dtype).npu()
            query_dequant_scale_cpu = torch.tensor(np.random.uniform(q_scale, q_scale, (batch_size, q_seq, q_head_num))).to(dequant_dtype).npu()
        else:
            query_dequant_scale = torch.tensor(np.random.uniform(q_scale, q_scale, (batch_size, q_seq, q_head_num))).to(dequant_dtype).npu()
            query_dequant_scale_cpu = query_dequant_scale
        
        weights = torch.tensor(np.random.uniform(weights_datarange[0], weights_datarange[1], (batch_size, q_seq, q_head_num))).to(dequant_dtype).npu()

    elif layout_query == "TND":
        query = torch.tensor(np.random.uniform(query_datarange[0], query_datarange[1], (q_t_size, q_head_num, head_dim))).to(torch.float)
        if hifp8mode == 1:
            query = trans_float_tensor_to_hifuint8(query, round_mode = "hybrid", over_mode = True).npu()
        else:
            query = query.to(qk_dtype).npu()
        
        q_scale = random.uniform(q_scale_datarange[0], q_scale_datarange[1])
        if quant_mode == 4:
            query_dequant_scale = torch.tensor([q_scale]).to(dequant_dtype).npu()
            query_dequant_scale_cpu = torch.tensor(np.random.uniform(q_scale, q_scale, (q_t_size, q_head_num))).to(dequant_dtype).npu()
        else:
            query_dequant_scale = torch.tensor(np.random.uniform(q_scale, q_scale, (q_t_size, q_head_num))).to(dequant_dtype).npu()
            query_dequant_scale_cpu = query_dequant_scale
        
        weights = torch.tensor(np.random.uniform(weights_datarange[0], weights_datarange[1], (q_t_size, q_head_num))).to(dequant_dtype).npu()

    if layout_key == "BSND":
        key = torch.tensor(np.random.uniform(key_datarange[0], key_datarange[1], (batch_size, k_seq, k_head_num, head_dim))).to(torch.float)
        if hifp8mode == 1:
            key = trans_float_tensor_to_hifuint8(key, round_mode = "hybrid", over_mode = True).npu()
        else:
            key = key.to(qk_dtype).npu()
        
        k_scale = random.uniform(k_scale_datarange[0], k_scale_datarange[1])
        if quant_mode == 4:
            key_dequant_scale = torch.tensor([k_scale]).to(dequant_dtype).npu()
            key_dequant_scale_cpu = torch.tensor(np.random.uniform(k_scale, k_scale, (batch_size, k_seq, k_head_num))).to(dequant_dtype)
        else:
            key_dequant_scale = torch.tensor(np.random.uniform(k_scale_datarange[0], k_scale_datarange[1], (batch_size, k_seq, k_head_num))).to(dequant_dtype).npu()
            key_dequant_scale_cpu = key_dequant_scale.cpu()
        
        block_table = None
        cpu_result, topk_value = test_qliv2.forward(query, key, weights, query_dequant_scale_cpu, key_dequant_scale_cpu, actual_seq_lengths_query, actual_seq_lengths_key, block_table)

    elif layout_key == "TND":
        key = torch.tensor(np.random.uniform(key_datarange[0], key_datarange[1], (k_t_size, k_head_num, head_dim))).to(torch.float)
        if hifp8mode == 1:
            key = trans_float_tensor_to_hifuint8(key, round_mode = "hybrid", over_mode = True).npu()
        else:
            key = key.to(qk_dtype).npu()
        
        k_scale = random.uniform(k_scale_datarange[0], k_scale_datarange[1])
        if quant_mode == 4:
            key_dequant_scale = torch.tensor([k_scale]).to(dequant_dtype).npu()
            key_dequant_scale_cpu = torch.tensor(np.random.uniform(k_scale, k_scale, (k_t_size, k_head_num))).to(dequant_dtype)
        else:
            key_dequant_scale = torch.tensor(np.random.uniform(k_scale_datarange[0], k_scale_datarange[1], (k_t_size, k_head_num))).to(dequant_dtype).npu()
            key_dequant_scale_cpu = key_dequant_scale.cpu()
        
        block_table = None
        cpu_result, topk_value = test_qliv2.forward(query, key, weights, query_dequant_scale_cpu, key_dequant_scale_cpu, actual_seq_lengths_query, actual_seq_lengths_key, block_table)

    elif layout_key == "PA_BBND":
        # 以不同batch中最大seq为标准初始化key(bnsd)和key_dequant_scale(bns)
        k_max_s2 = math.floor(max(act_seq_k))
        k_max_block_num_per_batch = math.ceil(k_max_s2 / block_size) #遍历batch得到的最大的block num
        key_bnsd = torch.tensor(np.random.uniform(key_datarange[0], key_datarange[1],(batch_size, k_head_num, k_max_s2, head_dim))).to(torch.float)
        if hifp8mode == 1:
            key_bnsd = trans_float_tensor_to_hifuint8(key_bnsd, round_mode = "hybrid", over_mode = True)
        else:
            key_bnsd = key_bnsd.to(qk_dtype)
        
        k_scale = random.uniform(k_scale_datarange[0], k_scale_datarange[1])
        if quant_mode == 4:
            key_dequant_scale_bns = torch.tensor(np.random.uniform(k_scale, k_scale, (batch_size, k_head_num, k_max_s2))).to(dequant_dtype)
        else:
            key_dequant_scale_bns = torch.tensor(np.random.uniform(k_scale_datarange[0], k_scale_datarange[1], (batch_size, k_head_num, k_max_s2))).to(dequant_dtype)
        
        key_block_num_per_batch = []
        key_block_num_sum = 0
        for cur_act_k in act_seq_k:
            cur_cmp_act_k = math.floor(cur_act_k)
            cur_key_block_num = math.ceil(cur_cmp_act_k / block_size)
            key_block_num_per_batch.append(cur_key_block_num)
            key_block_num_sum += cur_key_block_num
        if block_num < key_block_num_sum:
            raise ValueError(f"key actual block num < needed block num")
        # 构建block table
        block_id_list = np.arange(block_num)
        block_id_list = np.random.permutation(block_id_list).astype(np.int32)
        cur_block_id = 0
        block_table = np.full((batch_size, k_max_block_num_per_batch), fill_value = -1, dtype=np.int32)
        batch_idx = 0
        for cur_block_id_threshold in key_block_num_per_batch:
            for i_block_id in range(cur_block_id_threshold):
                block_table[batch_idx][i_block_id] = block_id_list[cur_block_id]
                cur_block_id += 1
            batch_idx += 1
        # 构建PA场景的key
        # [batch_size, s2, k_head_num, head_dim] expand to [batch_size, k_max_block_num_per_batch * block_size, k_head_num, head_dim]
        key_expand = torch.zeros((batch_size, k_head_num, k_max_block_num_per_batch * block_size, head_dim), dtype = qk_dtype)
        key_expand[:,:,:k_max_s2,:] = key_bnsd
        key = torch.zeros((block_num, block_size, k_head_num, head_dim), dtype = qk_dtype) # hifp8时qk_dtype为uint8
        for i_batch in range(batch_size):
            for  i_block, cur_block_id in enumerate(block_table[i_batch]):
                block_start_pos = i_block * block_size
                if cur_block_id == -1:
                    continue
                else:
                    for i_n in range(k_head_num):
                        key[cur_block_id, :, i_n, :] = key_expand[i_batch, i_n, block_start_pos:block_start_pos+block_size,:]
        key = key.npu()
        # 构建PA场景的key_dequant_scale
        key_dequant_scale_expand = torch.zeros((batch_size, k_head_num, k_max_block_num_per_batch * block_size), dtype= dequant_dtype)
        key_dequant_scale_expand[:,:,:k_max_s2] = key_dequant_scale_bns
        key_dequant_scale_block = torch.zeros((block_num, block_size, k_head_num), dtype = dequant_dtype)
        for i_batch in range(batch_size):
            for i_block, cur_block_id in enumerate(block_table[i_batch]):
                block_start_pos = i_block * block_size
                if cur_block_id == -1:
                    continue
                else:
                    for i_n in range(k_head_num):
                        key_dequant_scale_block[cur_block_id, :, i_n] = key_dequant_scale_expand[i_batch, i_n,block_start_pos:block_start_pos+block_size]
        if quant_mode == 4:
            key_dequant_scale = torch.tensor([k_scale]).to(dequant_dtype).npu()
        else:
            key_dequant_scale = key_dequant_scale_block.npu()
        cpu_result, topk_value = test_qliv2.forward(query, key_bnsd, weights, query_dequant_scale_cpu, key_dequant_scale_bns, actual_seq_lengths_query, actual_seq_lengths_key, block_table)
        block_table = torch.from_numpy(block_table).to(dtype=torch.int32).npu()
    # ======================== metadata 构造 ========================
    # max_seqlen 从个体长度中取
    max_seqlen_q_meta = actual_seq_lengths_query.max().item()
    max_seqlen_k_meta = actual_seq_lengths_key.max().item()
    metadata = torch.ops.cann_ops_transformer.quant_lightning_indexer_metadata(
                                    cu_seqlens_q = cu_seqlens_query,
                                    cu_seqlens_k = cu_seqlens_key,
                                    seqused_q = seqused_q_tensor,
                                    seqused_k = seqused_k_tensor,
                                    cmp_residual_k = cmp_residual_k_for_npu,
                                    batch_size = batch_size,
                                    max_seqlen_q = max_seqlen_q_meta,
                                    max_seqlen_k = max_seqlen_k_meta,
                                    num_heads_q = q_head_num,
                                    num_heads_k = k_head_num,
                                    head_dim = head_dim,
                                    topk = sparse_count,
                                    quant_mode = quant_mode,
                                    mask_mode = sparse_mode,
                                    layout_q = layout_query,
                                    layout_k = layout_key,
                                    cmp_ratio = cmp_ratio)

    metadata = metadata.npu()

    if qk_dtype == torch.float8_e4m3fn:
        query = query.to(dtype=torch.float16)
        key = key.to(dtype=torch.float16)

    output_tensors = {
        "params":params,
        "cpu_result": cpu_result,
        "topk_value": topk_value,
        "query": query,
        "key":key,
        "weights": weights,
        "query_dequant_scale": query_dequant_scale,
        "key_dequant_scale": key_dequant_scale,
        "cu_seqlens_query": cu_seqlens_query,
        "cu_seqlens_key": cu_seqlens_key,
        "actual_seq_lengths_query": actual_seq_lengths_query,
        "actual_seq_lengths_key": actual_seq_lengths_key,
        "cmp_residual_k_for_npu": cmp_residual_k_for_npu,
        "block_table": block_table,
        "metadata": metadata,
        "quant_mode":quant_mode,
        "layout_query":layout_query,
        "layout_key":layout_key,
        "sparse_count":sparse_count,
        "sparse_mode":sparse_mode,
        "cmp_ratio":cmp_ratio
    }
    return  casename, output_tensors

def save_test_case(test_cases, file_path):
    print("正在保存pt文件...")
    # 创建输出目录
    os.makedirs(file_path, exist_ok=True)

    for idx, case in enumerate(test_cases):
        try:
            case_name, output_tensors = qliv2_output_single(case)
            # 生成文件名
            input_filename = f"{case_name}.pt"
            input_filepath = os.path.join(file_path, input_filename)

            # 保存数据
            torch.save(output_tensors, input_filepath)
            print(f"测试用例已保存到: {input_filepath}")

        except Exception as e:
            print(f"[失败] 生成 pt 文件失败: {case[0]} (索引: {idx})")
            print(f"错误详情: {e}")

def main():
    parser = argparse.ArgumentParser(description='qliv2_pt_save.py 接收路径参数')
    parser.add_argument('path1', type=str, help='第一个路径')
    parser.add_argument('path2', type=str, help='第二个路径')
    args = parser.parse_args()
    path1 = args.path1
    path2 = args.path2
    testcase =  load_excel_test_cases(path1, "Sheet1")
    save_test_case(testcase, path2)

if __name__ == "__main__":
    main()

