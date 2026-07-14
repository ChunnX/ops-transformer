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

import pytest
import random
import pandas as pd
from pathlib import Path
import numpy as np
import os
import math

def infer_template_run_mode(K, cmp_ratio, ori_kv_type, cmp_kv_type, cmp_mask_mode, block_size1, block_size2, block_num2):
    """
    根据K和cmp_ratio推断template_run_mode并设置默认值

    参数:
        K: sparse block count (可为None)
        cmp_ratio: 压缩率 (可为None)
        ori_kv_type: ori_kv数据类型
        cmp_kv_type: cmp_kv数据类型 (可为None)
        cmp_mask_mode: cmp mask模式 (可为None)
        block_size1: ori block大小
        block_size2: cmp block大小 (可为None)
        block_num2: cmp block总数 (可为None)

    返回:
        dict: 包含template_run_mode及相关参数的字典
    """
    template_run_mode = "SWA"

    if K is None or K == ['None'] or K == 0:
        if cmp_ratio is None or cmp_ratio == ['None']:
            template_run_mode = "SWA"
            cmp_ratio = 1
            cmp_kv_type = ori_kv_type
            cmp_mask_mode = 0
            K = 0
            block_size2 = block_size1 if block_size2 is None else block_size2
            block_num2 = 0 if block_num2 is None else block_num2
        else:
            template_run_mode = "HCA"
            cmp_ratio = int(cmp_ratio)
            cmp_kv_type = ori_kv_type if cmp_kv_type is None else cmp_kv_type
            K = 0
            block_size2 = int(block_size2) if block_size2 is not None else block_size1
            cmp_mask_mode = int(cmp_mask_mode) if cmp_mask_mode is not None else 3
            block_num2 = block_num2  # 后续计算
    else:
        template_run_mode = "CSA"
        K = int(K)
        cmp_ratio = int(cmp_ratio)
        cmp_kv_type = ori_kv_type if cmp_kv_type is None else cmp_kv_type
        block_size2 = int(block_size2) if block_size2 is not None else block_size1
        cmp_mask_mode = int(cmp_mask_mode) if cmp_mask_mode is not None else 3
        block_num2 = block_num2  # 后续计算

    return {
        'template_run_mode': template_run_mode,
        'K': K,
        'cmp_ratio': cmp_ratio,
        'cmp_kv_type': cmp_kv_type,
        'cmp_mask_mode': cmp_mask_mode,
        'block_size2': block_size2,
        'block_num2': block_num2,
    }



def gen_seqused_cmp_kv(seqused_ori_kv, cmp_ratio):
    """
    生成seqused_cmp_kv

    参数:
        seqused_ori_kv: ori_kv真实长度列表
        cmp_ratio: 压缩率

    返回:
        list: seqused_cmp_kv列表
    """
    return [math.floor(s / cmp_ratio) for s in seqused_ori_kv]

def parse_list_param(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        s = value.strip()
        if s.lower() == 'none' or s == '':
            return None
        if s.startswith('[') and s.endswith(']'):
            s = s[1:-1]
        try:
            return [int(x.strip()) for x in s.split(',') if x.strip() != '']
        except ValueError:
            return None
    return None



def gen_cu_seqlens_from_seqused(seqused):
    """
    从seqused生成cu_seqlens累加和

    参数:
        seqused: 真实长度列表

    返回:
        list: cu_seqlens累加和列表（长度B+1）
    """
    cu_seqlens = [0]
    for s in seqused:
        cu_seqlens.append(cu_seqlens[-1] + s)
    return cu_seqlens

def generate_seqused(cu_seqlens):
    """从cu_seqlens作差得到seqused"""
    return [cu_seqlens[i + 1] - cu_seqlens[i] for i in range(len(cu_seqlens) - 1)]

def fill_random_cu_len(T, SMax, B, random_seq=False):
    cu_seqlens = [0]
    S = SMax if (SMax is not None and SMax > 0) else max(1, T // B)

    can_fit = (S * B >= T) if random_seq else (S * B == T)
    S_eq = min(S, T // B)
    for i in range(B):
        remaining_batches = B - i
        remaining_T = T - cu_seqlens[-1]
        if remaining_batches <= 0:
            break
        if random_seq:
            max_len = min(S, remaining_T - (remaining_batches - 1))
            min_len = max(0, remaining_T - (remaining_batches - 1) * S)
            if max_len < min_len:
                max_len = min_len
            length = random.randint(min_len, max_len) if max_len > 0 else 0
        else:
            length = S_eq
        cu_seqlens.append(cu_seqlens[-1] + length)

    if cu_seqlens[-1] != T and can_fit:
        raise ValueError(f"fill_random_cu_len: S*B>=T但末位({cu_seqlens[-1]})!=T({T})，生成异常")
    return cu_seqlens

def fill_random_used_len(SMax, B, random_seq=False):
    used_lens = []
    for i in range(B):
        length = random.randint(0, SMax) if random_seq else SMax
        used_lens.append(length)
    return used_lens

def calc_block_num(seqused_kv, block_size):
    """
    计算block_num

    参数:
        seqused_kv: 真实长度列表
        block_size: block大小

    返回:
        int: block_num_sum
    """
    block_num_sum = 0
    for cur_act_kv in seqused_kv:
        cur_kv_block_num = math.ceil(cur_act_kv / block_size)
        block_num_sum += cur_kv_block_num
    return block_num_sum

def fill_none_params(params_dict):
    """
    填充None参数的默认值（完整封装）

    参数:
        params_dict: 参数字典

    返回:
        dict: 填充后的完整参数字典
    """
    if params_dict.get('Testcase_Name') is None and params_dict.get('testcase_name') is not None:
        params_dict['Testcase_Name'] = params_dict['testcase_name']
    layout_q = params_dict.get('layout_q')
    B = params_dict.get('B')
    S1 = params_dict.get('S1')
    S2 = params_dict.get('S2')
    q_type = params_dict.get('q_type') or torch.bfloat16
    ori_kv_type = params_dict.get('ori_kv_type') or torch.float8_e4m3fn
    block_size1 = params_dict.get('block_size1')
    actlen_mode = params_dict.get('actlen_mode', 'full')
    S1EQS2 = params_dict.get('S1EQS2')
    if S1EQS2 is None:
        S1EQS2 = params_dict.get('qk_equal_len', False)

    # 1. 获取template_run_mode（优先从params_dict获取）
    template_run_mode = params_dict.get('template_run_mode')

    if template_run_mode is None:
        # 如果未提供template_run_mode，则推断（向后兼容）
        infer_result = infer_template_run_mode(
            params_dict.get('K'),
            params_dict.get('cmp_ratio'),
            ori_kv_type,
            params_dict.get('cmp_kv_type'),
            params_dict.get('cmp_mask_mode'),
            block_size1,
            params_dict.get('block_size2'),
            params_dict.get('block_num2')
        )
        K = infer_result['K']
        cmp_ratio = infer_result['cmp_ratio']
        cmp_kv_type = infer_result['cmp_kv_type']
        cmp_mask_mode = infer_result['cmp_mask_mode']
        block_size2 = infer_result['block_size2']
        block_num2 = infer_result['block_num2']
        template_run_mode = infer_result['template_run_mode']
    else:
        # 如果提供了template_run_mode，直接使用params_dict中的参数
        K = params_dict.get('K')
        cmp_ratio = params_dict.get('cmp_ratio')
        cmp_kv_type = params_dict.get('cmp_kv_type')
        cmp_mask_mode = params_dict.get('cmp_mask_mode')
        block_size2 = params_dict.get('block_size2')
        block_num2 = params_dict.get('block_num2')

        # 处理None参数的默认值（保持一致性）
        if cmp_ratio is None:
            cmp_ratio = 1
        if cmp_kv_type is None:
            cmp_kv_type = ori_kv_type
        if cmp_mask_mode is None:
            cmp_mask_mode = 3 if cmp_ratio != 1 else 0
        if block_size2 is None:
            block_size2 = block_size1
        if K is None:
            K = 0

    # 2. 序列参数直接透传
    seqused_q = params_dict.get('seqused_q')
    cu_seqlens_q = params_dict.get('cu_seqlens_q')
    seqused_ori_kv = params_dict.get('seqused_ori_kv')
    cu_seqlens_ori_kv = params_dict.get('cu_seqlens_ori_kv')
    seqused_cmp_kv = params_dict.get('seqused_cmp_kv')
    cu_seqlens_cmp_kv = params_dict.get('cu_seqlens_cmp_kv')
    cmp_residual_kv = params_dict.get('cmp_residual_kv')

    layout_kv = params_dict.get('layout_kv')
    random_seq = (actlen_mode == 'random')

    # 2.1 Q侧: TND场景cu_seqlens_q生成
    if layout_q == "TND":
        if cu_seqlens_q is None:
            T1 = params_dict.get('T1')
            if T1 is None:
                if S1 is None:
                    raise ValueError("layout_q为TND时，cu_seqlens_q、T1和S1不能同时为None")
                T1 = S1 * B
            cu_seqlens_q = fill_random_cu_len(T1, S1, B, random_seq)
            print("cu_seqlens_q auto set to: ", cu_seqlens_q)
        T1 = cu_seqlens_q[-1]
        if params_dict.get('T1') is not None and params_dict['T1'] != T1:
            print(f"T1({params_dict['T1']})与cu_seqlens_q末位({T1})不一致，修正T1为{T1}")
        if seqused_q is None:
            seqused_q = generate_seqused(cu_seqlens_q)
            print("seqused_q auto set to: ", seqused_q)
    else:
        T1 = params_dict.get('T1')
        if seqused_q is None:
            if cu_seqlens_q is not None:
                seqused_q = generate_seqused(cu_seqlens_q)
            else:
                seqused_q = [S1] * B
            print("seqused_q auto set to: ", seqused_q)

    # 2.2 ori_kv侧: TND场景cu_seqlens_ori_kv生成
    T2 = params_dict.get('T2')
    if layout_kv == "TND":
        if cu_seqlens_ori_kv is None:
            T2 = params_dict.get('T2')
            if T2 is None:
                if S2 is None:
                    raise ValueError("layout_kv为TND时，cu_seqlens_ori_kv、T2和S2不能同时为None")
                T2 = S2 * B
            cu_seqlens_ori_kv = fill_random_cu_len(T2, S2, B, random_seq)
            print("cu_seqlens_ori_kv auto set to: ", cu_seqlens_ori_kv)
        T2 = cu_seqlens_ori_kv[-1]
        if params_dict.get('T2') is not None and params_dict['T2'] != T2:
            print(f"T2({params_dict['T2']})与cu_seqlens_ori_kv末位({T2})不一致，修正T2为{T2}")
        if seqused_ori_kv is None:
            seqused_ori_kv = generate_seqused(cu_seqlens_ori_kv)
            print("seqused_ori_kv auto set to: ", seqused_ori_kv)

    # 2.3 S1EQS2联动: Q和ori_kv对应batch的真实长度一致
    if S1EQS2 and layout_q == "TND" and layout_kv == "TND":
        if cu_seqlens_q is not None and cu_seqlens_ori_kv is not None:
            seqused_q_diffs = generate_seqused(cu_seqlens_q)
            seqused_ori_diffs = generate_seqused(cu_seqlens_ori_kv)
            if seqused_q_diffs != seqused_ori_diffs:
                cu_seqlens_ori_kv = list(cu_seqlens_q)
                T2 = T1
                print(f"S1EQS2: cu_seqlens_ori_kv同步为cu_seqlens_q: {cu_seqlens_ori_kv}")

    # 2.4 ori_kv侧: PA_BBND场景seqused_ori_kv生成
    if layout_kv == "PA_BBND":
        if seqused_ori_kv is None:
            if S2 is not None:
                seqused_ori_kv = fill_random_used_len(S2, B, random_seq)
                print("seqused_ori_kv auto set to: ", seqused_ori_kv)
            else:
                raise ValueError("layout_kv为PA_BBND时，seqused_ori_kv和S2不能同时为None")

    # 2.5 S1EQS2联动: BSND+PA_BBND场景下seqused_ori_kv同步seqused_q
    if S1EQS2 and layout_q == "BSND" and layout_kv == "PA_BBND":
        if seqused_q is not None and seqused_ori_kv is not None:
            if seqused_q != seqused_ori_kv:
                seqused_ori_kv = list(seqused_q)
                print(f"S1EQS2: seqused_ori_kv同步为seqused_q: {seqused_ori_kv}")

    # 2.6 S1EQS2联动: TND(Q)+PA_BBND(KV)场景下seqused_ori_kv同步cu_seqlens_q差值
    if S1EQS2 and layout_q == "TND" and layout_kv == "PA_BBND":
        if cu_seqlens_q is not None:
            seqused_q_diffs = generate_seqused(cu_seqlens_q)
            if seqused_ori_kv is not None:
                if seqused_ori_kv != seqused_q_diffs:
                    seqused_ori_kv = list(seqused_q_diffs)
                    print(f"S1EQS2: seqused_ori_kv同步为cu_seqlens_q差值: {seqused_ori_kv}")
            else:
                seqused_ori_kv = list(seqused_q_diffs)
                print(f"S1EQS2: seqused_ori_kv从cu_seqlens_q差值生成: {seqused_ori_kv}")

    # 2.7 S1EQS2校验: Q每batch真实长度不能大于S2
    if S1EQS2 and S2 is not None:
        if layout_q == "TND" and cu_seqlens_q is not None:
            seqused_q_diffs = generate_seqused(cu_seqlens_q)
        elif layout_q == "BSND" and seqused_q is not None:
            seqused_q_diffs = seqused_q
        else:
            seqused_q_diffs = None
        if seqused_q_diffs is not None:
            for i in range(B):
                if seqused_q_diffs[i] > S2:
                    raise ValueError(f"S1EQS2模式下batch{i}的Q真实长度({seqused_q_diffs[i]})不能大于S2({S2})")

    # 3. cmp参数 (SWA场景为None)
    if template_run_mode == "SWA":
        seqused_cmp_kv = None
        cu_seqlens_cmp_kv = None
        cmp_residual_kv = None
        T3 = None
    else:
        # cmp_kv侧: TND场景cu_seqlens_cmp_kv生成
        if layout_kv == "TND":
            if cu_seqlens_cmp_kv is None:
                T3 = params_dict.get('T3')
                S3 = S2 // cmp_ratio if (S2 is not None and cmp_ratio) else None
                if T3 is None:
                    if S3 is None:
                        raise ValueError("layout_kv为TND且非SWA时，cu_seqlens_cmp_kv、T3和S2/cmp_ratio不能同时为None")
                    T3 = S3 * B
                cu_seqlens_cmp_kv = fill_random_cu_len(T3, S3, B, random_seq)
                print("cu_seqlens_cmp_kv auto set to: ", cu_seqlens_cmp_kv)
            T3 = cu_seqlens_cmp_kv[-1]
            if params_dict.get('T3') is not None and params_dict['T3'] != T3:
                print(f"T3({params_dict['T3']})与cu_seqlens_cmp_kv末位({T3})不一致，修正T3为{T3}")
            if seqused_cmp_kv is None:
                seqused_cmp_kv = generate_seqused(cu_seqlens_cmp_kv)
                print("seqused_cmp_kv auto set to: ", seqused_cmp_kv)
        else:
            T3 = params_dict.get('T3')

        # cmp_kv侧: PA_BBND场景seqused_cmp_kv生成
        if layout_kv == "PA_BBND":
            if seqused_cmp_kv is None:
                S3 = S2 // cmp_ratio if (S2 is not None and cmp_ratio) else None
                if S3 is not None:
                    seqused_cmp_kv = fill_random_used_len(S3, B, random_seq)
                    print("seqused_cmp_kv auto set to: ", seqused_cmp_kv)
                else:
                    raise ValueError("layout_kv为PA_BBND且非SWA时，seqused_cmp_kv和S2/cmp_ratio不能同时为None")

        if cmp_residual_kv is None:
            cmp_residual_kv = [random.randint(0, cmp_ratio - 1) for _ in range(B)]

    # 4. block_num
    block_num1 = params_dict.get('block_num1')
    block_num2 = params_dict.get('block_num2')

    if block_num1 is None and seqused_ori_kv is not None:
        block_num1 = calc_block_num(seqused_ori_kv, block_size1)
    if block_num2 is None:
        if template_run_mode == "SWA":
            block_num2 = 0
        elif seqused_cmp_kv is not None:
            block_num2 = calc_block_num(seqused_cmp_kv, block_size2)

    # 5. T1已在2.1中计算

    # 构建完整参数字典
    filled_params = {
        'Testcase_Name': params_dict.get('Testcase_Name', params_dict.get('testcase_name')),
        'layout_q': layout_q,
        'layout_kv': params_dict.get('layout_kv'),
        'q_type': q_type,
        'ori_kv_type': ori_kv_type,
        'cmp_kv_type': cmp_kv_type,
        'B': B,
        'S1': S1,
        'S2': S2,
        'T1': T1,
        'T2': T2,
        'T3': T3,
        'N1': params_dict['N1'],
        'N2': params_dict['N2'],
        'D': params_dict['D'],
        'K': K,
        'block_num1': block_num1,
        'block_num2': block_num2,
        'block_size1': block_size1,
        'block_size2': block_size2,
        'seqused_q': seqused_q,
        'cu_seqlens_q': cu_seqlens_q,
        'seqused_ori_kv': seqused_ori_kv,
        'seqused_cmp_kv': seqused_cmp_kv,
        'cu_seqlens_ori_kv': cu_seqlens_ori_kv,
        'cu_seqlens_cmp_kv': cu_seqlens_cmp_kv,
        'cmp_residual_kv': cmp_residual_kv,
        'softmax_scale': params_dict['softmax_scale'],
        'cmp_ratio': cmp_ratio,
        'ori_mask_mode': params_dict['ori_mask_mode'],
        'cmp_mask_mode': cmp_mask_mode,
        'ori_win_left': params_dict['ori_win_left'],
        'ori_win_right': params_dict['ori_win_right'],
        'quant_mode': params_dict['quant_mode'],
        'tile_size': params_dict['tile_size'],
        'rope_head_dim': params_dict['rope_head_dim'],
        'template_run_mode': template_run_mode,
        'actlen_mode': actlen_mode,
        'S1EQS2': S1EQS2,
        'topk_value_mode': params_dict.get('topk_value_mode', 1),
        'return_softmax_lse': params_dict.get('return_softmax_lse', False),
    }

    return filled_params

def load_excel_test_cases(excel_file_path: str, sheetname: str):
    """
    从 Excel 文件加载测试用例。

    参数:
        excel_file_path (str): Excel 文件的路径。
        sheetname (str, optional): 工作表名称。若未提供，则默认 'Sheet1'。

    返回:
        list[]: 测试用例字典列表
    """
    if sheetname is None:
        sheetname = 'Sheet1'

    # 检查文件是否存在
    if not os.path.exists(excel_file_path):
        pytest.skip(f"Excel file not found: {excel_file_path}", allow_module_level=True)

    try:
        # 读取 Excel 文件的指定 sheet
        df = pd.read_excel(excel_file_path, sheet_name=sheetname)
        if 'testcase_name' in df.columns and 'Testcase_Name' not in df.columns:
            df = df.rename(columns={'testcase_name': 'Testcase_Name'})
        df = df.replace({np.nan: None, pd.NA: None})

        # 定义必需的列名
        required_columns = [
            "Testcase_Name", "layout_q", "layout_kv", "q_type", "ori_kv_type", "cmp_kv_type", "B", "S1", "S2", "N1",
            "N2", "D", "K", "block_size1", "block_size2", "softmax_scale", "cmp_ratio",
            "ori_mask_mode", "cmp_mask_mode", "ori_win_left", "ori_win_right", "quant_mode", "tile_size",
            "rope_head_dim", "template_run_mode", "actlen_mode", "S1EQS2",
            "topk_value_mode", "return_softmax_lse",
        ]
        optional_columns = [
            "seqused_q", "cu_seqlens_q", "seqused_ori_kv", "seqused_cmp_kv",
            "cu_seqlens_ori_kv", "cu_seqlens_cmp_kv", "cmp_residual_kv",
        ]
        # 检查是否缺少必要列
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            pytest.skip(f"Missing required columns in Excel: {missing_cols}", allow_module_level=True)

        # 构建测试用例列表
        test_cases = []
        i = 0
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            for col in optional_columns:
                if col not in row_dict:
                    row_dict[col] = None
                else:
                    row_dict[col] = parse_list_param(row_dict[col])
            test_cases.append(row_dict)
            i = i + 1

        return test_cases

    except Exception as e:
        pytest.skip(f"Failed to read Excel file: {e}", allow_module_level=True)
        return None

def save_result(params, result, fulfill_percent, result_path):
    """
    存储测试结果到表格
    """
    row_data = {
        "Testcase_Name": params.get('Testcase_Name'),
        "layout_q": params.get('layout_q'),
        "layout_kv": params.get('layout_kv'),
        "q_type": params.get('q_type'),
        "ori_kv_type": params.get('ori_kv_type'),
        "cmp_kv_type": params.get('cmp_kv_type'),
        "B": params.get('B'),
        "S1": params.get('S1'),
        "S2": params.get('S2'),
        "T1": params.get('T1'),
        "N1": params.get('N1'),
        "N2": params.get('N2'),
        "D": params.get('D'),
        "K": params.get('K'),
        "block_num1": params.get('block_num1'),
        "block_num2": params.get('block_num2'),
        "block_size1": params.get('block_size1'),
        "block_size2": params.get('block_size2'),
        "seqused_q": params.get('seqused_q'),
        "cu_seqlens_q": params.get('cu_seqlens_q'),
        "seqused_ori_kv": params.get('seqused_ori_kv'),
        "seqused_cmp_kv": params.get('seqused_cmp_kv'),
        "cu_seqlens_ori_kv": params.get('cu_seqlens_ori_kv'),
        "cu_seqlens_cmp_kv": params.get('cu_seqlens_cmp_kv'),
        "cmp_residual_kv": params.get('cmp_residual_kv'),
        "softmax_scale": params.get('softmax_scale'),
        "cmp_ratio": params.get('cmp_ratio'),
        "ori_mask_mode": params.get('ori_mask_mode'),
        "cmp_mask_mode": params.get('cmp_mask_mode'),
        "ori_win_left": params.get('ori_win_left'),
        'template_run_mode': params.get('template_run_mode'),
        'actlen_mode': params.get('actlen_mode'),
        'S1EQS2': params.get('S1EQS2'),
        'tc_prefix': params.get('tc_prefix', ' '),
        "ori_win_right": params.get('ori_win_right'),
        "quant_mode": params.get('quant_mode'),
        "tile_size": params.get('tile_size'),
        "rope_head_dim": params.get('rope_head_dim'),
        "topk_value_mode": params.get('topk_value_mode'),
        "return_softmax_lse": params.get('return_softmax_lse'),
        "result": result,
        "fulfill_percent": fulfill_percent,
    }

    # 检查文件是否存在
    if result_path.exists():
        try:
            df = pd.read_excel(result_path)
            if set(df.columns) != set(row_data.keys()):
                print("警告：变量名与Excel列名不匹配！")
                print(f"Excel列名: {list(df.columns)}")
                print(f"变量名: {list(row_data.keys())}")
                print("请检查变量名或Excel文件")
                return False
            new_df = pd.DataFrame([row_data])
            df = pd.concat([df, new_df], ignore_index=True)
        except Exception:
            print(f"警告: {result_path} 文件损坏，将重建")
            df = pd.DataFrame([row_data])
    else:
        # 文件不存在，创建新的DataFrame
        df = pd.DataFrame([row_data])

    # 保存到Excel
    df.to_excel(result_path, index=False)
