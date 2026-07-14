#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import ast
import math
import itertools
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pytest
import torch
import time
import random
 	 
str_map_dict = {
    'True': True,
    'False': False,
    'TRUE': True,
    'FALSE': False,
    'torch.bfloat16': torch.bfloat16,
    'torch.float16': torch.float16,
    'torch.float8_e4m3fn': torch.float8_e4m3fn,
}

def to_int_or_na(x):
    if pd.isna(x) or x == '':
        return pd.NA
    v = float(x)
    if not v.is_integer():
        raise ValueError(f"not a integer: {x}")
    return int(v)

def parse_to_list(x):
    if isinstance(x, str) and x.startswith('[') and x.endswith(']'):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return x  # 转换失败则保留原值
    return x

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
        df = pd.read_excel(excel_file_path,
            sheet_name=sheetname,
            converters={
                'testcase_name': lambda x: pd.NA if pd.isna(x) else str(x),
                'T1': to_int_or_na,
                'T2': to_int_or_na,
                'T3': to_int_or_na,
                'S1': to_int_or_na,
                'S2': to_int_or_na,
                'B': to_int_or_na,
                'N1': to_int_or_na,
                'N2': to_int_or_na,
                'D': to_int_or_na,
                'K': to_int_or_na,
                'ori_mask_mode': to_int_or_na,
                'cmp_mask_mode': to_int_or_na,
                'ori_win_left': to_int_or_na,
                'ori_win_right': to_int_or_na,
                'cmp_ratio': to_int_or_na,
                'block_size1': to_int_or_na,
                'block_size2': to_int_or_na,
                # 'kv_quant_mode': to_int_or_na,
            })
        df = df.replace({np.nan: None, pd.NA: None})
        df = df.applymap(parse_to_list)
        df = df.astype({
            'T1': "Int64",
            'T2': "Int64",
            'T3': "Int64",
            'S1': "Int64",
            'S2': "Int64",
            'B': "Int64",
            'N1': "Int64",
            'N2': "Int64",
            'D': "Int64",
            'K': "Int64",
            'ori_mask_mode': "Int64",
            'cmp_mask_mode': "Int64",
            'ori_win_left': "Int64",
            'ori_win_right': "Int64",
            'cmp_ratio': "Int64",
            'block_size1': "Int64",
            'block_size2': "Int64",
            # 'kv_quant_mode': "Int64",
        })
        # 处理可能为空的列
        default_None_columns = ["cu_seqlens_ori_kv","cu_seqlens_cmp_kv","T2","T3",
                                "seqused_ori_kv","seqused_cmp_kv","cmp_residual_kv",
                                "q_datarange","ori_kv_datarange","cmp_kv_datarange"]
        for key in default_None_columns:
            if key not in df.columns:
                df[key] = None
        
        if "actlen_mode" not in df.columns:
            df["actlen_mode"] = 'full'

        if "testcase_name" not in df.columns:
            # 生成用例名称
            cols_to_combine = [
                "template_mode",
                "B",
                "S1",
                "S2",
                "N1",
                "N2",
                "K",
                "layout_q",
                "layout_kv",
            ]
            
            df["testcase_name"] = df[cols_to_combine].astype(str).agg("_".join, axis=1)

        # 定义必需的列名
        required_columns = [
            "layout_q", "layout_kv", "q_type", "ori_kv_type", "cmp_kv_type", "B", "S1", "S2", "N1", "N2", "D", "K",
            "block_size1", "block_size2", "softmax_scale", "cmp_ratio",
            "ori_mask_mode", "cmp_mask_mode", "ori_win_left", "ori_win_right", "testcase_name"
        ]
        # 检查是否缺少必要列
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            pytest.skip(f"Missing required columns in Excel: {missing_cols}", allow_module_level=True)

        # 构建测试用例列表
        test_cases = []
        for _, row in df.iterrows():
            test_cases.append(row.to_dict())
        
        print(test_cases)

        return test_cases

    except Exception as e:
        pytest.skip(f"Failed to read Excel file: {e}", allow_module_level=True)
        return None

def save_result(result, fulfill_percent, params, result_path='./result/smla_result.xlsx'):
    result_path = Path(result_path)
    row_data = {
        **params,
        "result": result,
        "fulfill_percent": fulfill_percent,
    }
    # 检查文件是否存在
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if result_path.exists():
        # 读取现有数据
        df = pd.read_excel(result_path)
        
        # 检查列名是否一致
        if set(df.columns) != set(row_data.keys()):
            print("警告：变量名与Excel列名不匹配！")
            print(f"Excel列名: {list(df.columns)}")
            print(f"变量名: {list(row_data.keys())}")
            print("请检查变量名或Excel文件")
            return False
        
        # 追加新行
        new_df = pd.DataFrame([row_data])
        df = pd.concat([df, new_df], ignore_index=True)
    else:
        # 文件不存在，创建新的DataFrame
        df = pd.DataFrame([row_data])
    
    # 保存到Excel
    df.to_excel(result_path, index=False)

def generate_param_combinations(ENABLED_PARAMS, is_save_pt=False):
    """
    生成参数组合

    Args:
        ENABLED_PARAMS: 启用的参数列表

    Returns:
        list: 参数组合列表
    """
    param_combinations = []
    
    for params in ENABLED_PARAMS:
        # 确保所有参数都存在，缺失的用默认值填充
        param_values = {
            "testcase_name": params.get("testcase_name", [None] if not is_save_pt else None),
            "layout_q": params.get("layout_q", [None] if not is_save_pt else None),
            "layout_kv": params.get("layout_kv", [None] if not is_save_pt else None),
            "q_type": params.get("q_type", [None] if not is_save_pt else None),
            "ori_kv_type": params.get("ori_kv_type", [None] if not is_save_pt else None),
            "cmp_kv_type": params.get("cmp_kv_type", [None] if not is_save_pt else None),
            "B": params.get("B", [None] if not is_save_pt else None),
            "S1": params.get("S1", [None] if not is_save_pt else None),
            "S2": params.get("S2", [None] if not is_save_pt else None),
            "T1": params.get("T1", [None] if not is_save_pt else None),
            "T2": params.get("T2", [None] if not is_save_pt else None),
            "T3": params.get("T3", [None] if not is_save_pt else None),
            "N1": params.get("N1", [None] if not is_save_pt else None),
            "N2": params.get("N2", [None] if not is_save_pt else None),
            "D": params.get("D", [None] if not is_save_pt else None),
            "K1": params.get("K1", [None] if not is_save_pt else None),
            "K": params.get("K", [None] if not is_save_pt else None),
            "block_num1": params.get("block_num1", [None] if not is_save_pt else None),
            "block_num2": params.get("block_num2", [None] if not is_save_pt else None),
            "block_size1": params.get("block_size1", [None] if not is_save_pt else None),
            "block_size2": params.get("block_size2", [None] if not is_save_pt else None),
            "seqused_q": params.get("seqused_q", [None] if not is_save_pt else None),
            "cu_seqlens_q": params.get("cu_seqlens_q", [None] if not is_save_pt else None),
            "cu_seqlens_ori_kv": params.get("cu_seqlens_ori_kv", [None] if not is_save_pt else None),
            "cu_seqlens_cmp_kv": params.get("cu_seqlens_cmp_kv", [None] if not is_save_pt else None),
            # "seqused_kv": params.get("seqused_kv", [None] if not is_save_pt else None),
            "seqused_ori_kv": params.get("seqused_ori_kv", [None] if not is_save_pt else None),
            "seqused_cmp_kv": params.get("seqused_cmp_kv", [None] if not is_save_pt else None),
            "cmp_residual_kv": params.get("cmp_residual_kv", [None] if not is_save_pt else None),
            "softmax_scale": params.get("softmax_scale", [None] if not is_save_pt else None),
            "cmp_ratio": params.get("cmp_ratio", [None] if not is_save_pt else None),
            "ori_mask_mode": params.get("ori_mask_mode", [None] if not is_save_pt else None),
            "cmp_mask_mode": params.get("cmp_mask_mode", [None] if not is_save_pt else None),
            "ori_win_left": params.get("ori_win_left", [None] if not is_save_pt else None),
            "ori_win_right": params.get("ori_win_right", [None] if not is_save_pt else None),
            "ori_kv_topk_mode": params.get("ori_kv_topk_mode", ["full"] if not is_save_pt else "full"),
            "cmp_kv_topk_mode": params.get("cmp_kv_topk_mode", ["full"] if not is_save_pt else "full"),
            "ori_sparse_indices_mode": params.get("ori_sparse_indices_mode", ["full"] if not is_save_pt else "full"),
            "cmp_sparse_indices_mode": params.get("cmp_sparse_indices_mode", ["full"] if not is_save_pt else "full"),
            "actlen_mode": params.get("actlen_mode", ["full"] if not is_save_pt else "full"),
            "template_mode": params.get("template_mode", [None] if not is_save_pt else None),
            "q_datarange": params.get("q_datarange", ["[-10, 10]"] if not is_save_pt else "[-10, 10]"),
            "ori_kv_datarange": params.get("ori_kv_datarange", ["[-10, 10]"] if not is_save_pt else "[-10, 10]"),
            "cmp_kv_datarange": params.get("cmp_kv_datarange", ["[-10, 10]"] if not is_save_pt else "[-10, 10]"),
            "qk_equal_len": params.get("qk_equal_len", ["False"] if not is_save_pt else False),
            "return_softmax_lse": params.get("return_softmax_lse", [False] if not is_save_pt else False)
        }
        param_names = list(param_values.keys())
        for key, value in param_values.items():
            if isinstance(value, str) and value in str_map_dict:
                    param_values[key] = str_map_dict[value]
        if is_save_pt:
            values_lists = [[param_values[name]] for name in param_names]
        else:
            # 生成参数名和值列表
            values_lists = [param_values[name] for name in param_names]
    
        # 生成所有组合
        for combo in itertools.product(*values_lists):
            combination = dict(zip(param_names, combo))
            param_combinations.append(combination)
    return param_combinations

def generate_cu_seqlens(seqused: list) -> list:
    """
    根据seqused生成cu_seqlens
    
    参数:
        seqused: 一维list，包含n个元素，表示每个序列的长度
        
    返回:
        cu_seqlens: 一维list，长度为n+1，第一个元素为0，
                    后续每个元素是seqused对应位置及之前元素的累加和
                    
    示例:
        >>> seqused = [2, 2, 2, 2, 2]
        >>> cu_seqlens = generate_cu_seqlens(seqused)
        >>> print(cu_seqlens)
        [0, 2, 4, 6, 8, 10]
    """
    cu_seqlens = [0]  # 第一个元素总是0
    current_sum = 0
    
    for length in seqused:
        current_sum += length
        cu_seqlens.append(current_sum)
    
    return cu_seqlens

def generate_seqused(cu_seqlens: list) -> list:
    """
    根据cu_seqlens生成seqused
    
    参数:
        cu_seqlens: 一维list，长度为n+1，表示累积序列长度
        
    返回:
        seqused: 一维list，长度为n，表示每个序列的真实长度
        
    示例:
        >>> cu_seqlens = [0, 2, 4, 6, 8, 10]
        >>> generate_seqused(cu_seqlens)
        [2, 2, 2, 2, 2]
    """
    seqused = []
    for i in range(1, len(cu_seqlens)):
        seqused.append(cu_seqlens[i] - cu_seqlens[i - 1])
    return seqused

def fill_none_params(params_dict):
    """
    根据入参自动填充缺失的参数（template_mode、seqused/cu_seqlens、block_num、T1/T2/T3等）

    参数:
        params_dict: 参数字典

    返回:
        dict: 填充后的完整参数字典
    """
    layout_q = params_dict['layout_q']
    layout_kv = params_dict['layout_kv']
    B = params_dict['B']
    S1 = params_dict['S1']
    S2 = params_dict.get('S2')
    T1 = params_dict.get('T1')
    T2 = params_dict.get('T2')
    T3 = params_dict.get('T3')
    ori_kv_type = params_dict['ori_kv_type']
    block_size1 = params_dict.get('block_size1')
    actlen_mode = params_dict.get('actlen_mode', 'full')
    qk_equal_len = params_dict.get('qk_equal_len', False)
    if isinstance(qk_equal_len, str):
        qk_equal_len = str_map_dict.get(qk_equal_len, qk_equal_len == 'True')
    random_seq = (actlen_mode == 'random')

    # 1. 推断 template_mode 并填充 cmp 相关默认值
    template_mode = params_dict.get('template_mode')
    K = params_dict.get('K')
    cmp_ratio = params_dict.get('cmp_ratio')
    cmp_kv_type = params_dict.get('cmp_kv_type')
    cmp_mask_mode = params_dict.get('cmp_mask_mode')
    block_size2 = params_dict.get('block_size2')

    if template_mode is None:
        if K is None or K == ['None']:
            if cmp_ratio is None:
                template_mode = 'SWA'
            else:
                template_mode = 'CFA'
        else:
            template_mode = 'SCFA'

    if template_mode == 'SWA':
        cmp_ratio = None
        K = None
        cmp_kv_type = None
        cmp_mask_mode = None
        block_size2 = None
    else:
        cmp_ratio = int(cmp_ratio) if cmp_ratio is not None else 1
        K = int(K) if (template_mode == 'SCFA' and K is not None) else None
        cmp_kv_type = ori_kv_type if cmp_kv_type is None else cmp_kv_type
        cmp_mask_mode = 3 if cmp_mask_mode is None else cmp_mask_mode
        block_size2 = block_size1 if block_size2 is None else block_size2

    # 2. 填充 seqused / cu_seqlens
    seqused_q = params_dict.get('seqused_q')
    cu_seqlens_q = params_dict.get('cu_seqlens_q')
    seqused_ori_kv = params_dict.get('seqused_ori_kv')
    cu_seqlens_ori_kv = params_dict.get('cu_seqlens_ori_kv')
    seqused_cmp_kv = params_dict.get('seqused_cmp_kv')
    cu_seqlens_cmp_kv = params_dict.get('cu_seqlens_cmp_kv')
    cmp_residual_kv = params_dict.get('cmp_residual_kv')

    if seqused_q is not None and len(seqused_q) == 1:
        seqused_q = seqused_q * B
    if seqused_ori_kv is not None and len(seqused_ori_kv) == 1:
        seqused_ori_kv = seqused_ori_kv * B
    if seqused_cmp_kv is not None and len(seqused_cmp_kv) == 1:
        seqused_cmp_kv = seqused_cmp_kv * B
    if cmp_residual_kv is not None and len(cmp_residual_kv) == 1:
        cmp_residual_kv = cmp_residual_kv * B

    # 从 cu_seqlens 推导 seqused
    if seqused_q is None and cu_seqlens_q is not None:
        seqused_q = generate_seqused(cu_seqlens_q)
    if seqused_ori_kv is None and cu_seqlens_ori_kv is not None:
        seqused_ori_kv = generate_seqused(cu_seqlens_ori_kv)

    # qk_equal_len 联动: seqused_q 与 seqused_ori_kv
    if qk_equal_len:
        if seqused_q is None and seqused_ori_kv is not None:
            seqused_q = list(seqused_ori_kv)
        elif seqused_ori_kv is None and seqused_q is not None:
            seqused_ori_kv = list(seqused_q)

    # 独立生成 seqused_q
    if seqused_q is None:
        if random_seq:
            seqused_q = [S1 - random.randint(0, S1 - 1) for _ in range(B)]
        else:
            seqused_q = [S1 for _ in range(B)]

    # 再次 qk_equal_len 联动
    if qk_equal_len and seqused_ori_kv is None:
        seqused_ori_kv = list(seqused_q)

    # 独立生成 seqused_ori_kv
    if seqused_ori_kv is None:
        kv_full_len = S2 if S2 is not None else S1
        if random_seq:
            seqused_ori_kv = [kv_full_len - random.randint(0, math.ceil(kv_full_len / 2) - 1) for _ in range(B)]
        else:
            seqused_ori_kv = [kv_full_len for _ in range(B)]

    # 填充 cu_seqlens_q
    if cu_seqlens_q is None:
        if qk_equal_len and cu_seqlens_ori_kv is not None:
            cu_seqlens_q = list(cu_seqlens_ori_kv)
        else:
            cu_seqlens_q = generate_cu_seqlens(seqused_q)

    # 填充 cu_seqlens_ori_kv (仅 TND 需要)
    if cu_seqlens_ori_kv is None and layout_kv == 'TND':
        if qk_equal_len and cu_seqlens_q is not None:
            cu_seqlens_ori_kv = list(cu_seqlens_q)
        else:
            cu_seqlens_ori_kv = generate_cu_seqlens(seqused_ori_kv)

    # 校验 cu_seqlens 槽位 >= seqused
    for i in range(B):
        slot_len = cu_seqlens_q[i + 1] - cu_seqlens_q[i]
        if slot_len < seqused_q[i]:
            raise ValueError(f"cu_seqlens_q slot ({slot_len}) at batch {i} must >= seqused_q ({seqused_q[i]})")
    if cu_seqlens_ori_kv is not None:
        for i in range(B):
            slot_len = cu_seqlens_ori_kv[i + 1] - cu_seqlens_ori_kv[i]
            if slot_len < seqused_ori_kv[i]:
                raise ValueError(f"cu_seqlens_ori_kv slot ({slot_len}) at batch {i} must >= seqused_ori_kv ({seqused_ori_kv[i]})")

    # 3. 填充 cmp 参数 (SWA 场景为 None, 非 SWA 自动生成)
    if template_mode == 'SWA':
        seqused_cmp_kv = None
        cu_seqlens_cmp_kv = None
        cmp_residual_kv = None
    else:
        if seqused_cmp_kv is None:
            seqused_cmp_kv = [math.floor(s / cmp_ratio) for s in seqused_ori_kv]
        if cu_seqlens_cmp_kv is None and layout_kv == 'TND':
            cu_seqlens_cmp_kv = [math.floor(c / cmp_ratio) for c in cu_seqlens_ori_kv]
        if cmp_residual_kv is None:
            cmp_residual_kv = [s % cmp_ratio for s in seqused_ori_kv]
        if cu_seqlens_cmp_kv is not None:
            for i in range(B):
                slot_len = cu_seqlens_cmp_kv[i + 1] - cu_seqlens_cmp_kv[i]
                if slot_len < seqused_cmp_kv[i]:
                    raise ValueError(f"cu_seqlens_cmp_kv slot ({slot_len}) at batch {i} must >= seqused_cmp_kv ({seqused_cmp_kv[i]})")

    # 4. 计算 block_num (仅在为 None 时填充)
    block_num1 = params_dict.get('block_num1')
    block_num2 = params_dict.get('block_num2')
    if block_num1 is None and layout_kv == 'PA_BBND' and block_size1 is not None:
        block_num1 = sum(math.ceil(s / block_size1) for s in seqused_ori_kv)
    if block_num2 is None:
        if template_mode == 'SWA':
            block_num2 = 0
        elif layout_kv == 'PA_BBND' and block_size2 is not None:
            block_num2 = sum(math.ceil(s / block_size2) for s in seqused_cmp_kv)

    # 5. 计算 T1 / T2 / T3
    if T1 is None and layout_q == 'TND':
        T1 = cu_seqlens_q[-1]
    if T2 is None and layout_kv == 'TND':
        T2 = cu_seqlens_ori_kv[-1]
    if T3 is None and layout_kv == 'TND' and template_mode != 'SWA' and cu_seqlens_cmp_kv is not None:
        T3 = cu_seqlens_cmp_kv[-1]

    # 6. 数据范围与 topk 等默认值
    def _parse_datarange(v):
        if v is None:
            return [-10, 10]
        if isinstance(v, str):
            return ast.literal_eval(v)
        return v

    q_datarange = _parse_datarange(params_dict.get('q_datarange'))
    ori_kv_datarange = _parse_datarange(params_dict.get('ori_kv_datarange'))
    cmp_kv_datarange = _parse_datarange(params_dict.get('cmp_kv_datarange'))

    testcase_name = params_dict.get('testcase_name')
    if testcase_name is None:
        testcase_name = "case_" + str(int(time.time() * 1000000))

    # 构建完整参数字典
    filled_params = {
        'testcase_name': testcase_name,
        'layout_q': layout_q,
        'layout_kv': layout_kv,
        'q_type': params_dict.get('q_type'),
        'ori_kv_type': ori_kv_type,
        'cmp_kv_type': cmp_kv_type,
        'B': B,
        'S1': S1,
        'S2': S2,
        'T1': T1,
        'T2': T2,
        'T3': T3,
        'N1': params_dict.get('N1'),
        'N2': params_dict.get('N2'),
        'D': params_dict.get('D'),
        'K': K,
        'K1': params_dict.get('K1'),
        'block_num1': block_num1,
        'block_num2': block_num2,
        'block_size1': block_size1,
        'block_size2': block_size2,
        'seqused_q': seqused_q,
        'cu_seqlens_q': cu_seqlens_q,
        'seqused_ori_kv': seqused_ori_kv,
        'cu_seqlens_ori_kv': cu_seqlens_ori_kv,
        'seqused_cmp_kv': seqused_cmp_kv,
        'cu_seqlens_cmp_kv': cu_seqlens_cmp_kv,
        'cmp_residual_kv': cmp_residual_kv,
        'softmax_scale': params_dict.get('softmax_scale'),
        'cmp_ratio': cmp_ratio,
        'ori_mask_mode': params_dict.get('ori_mask_mode'),
        'cmp_mask_mode': cmp_mask_mode,
        'ori_win_left': params_dict.get('ori_win_left'),
        'ori_win_right': params_dict.get('ori_win_right'),
        'ori_kv_topk_mode': params_dict.get('ori_kv_topk_mode', 'full'),
        'cmp_kv_topk_mode': params_dict.get('cmp_kv_topk_mode', 'full'),
        'ori_sparse_indices_mode': params_dict.get('ori_sparse_indices_mode', 'full'),
        'cmp_sparse_indices_mode': params_dict.get('cmp_sparse_indices_mode', 'full'),
        'actlen_mode': actlen_mode,
        'template_mode': template_mode,
        'q_datarange': q_datarange,
        'ori_kv_datarange': ori_kv_datarange,
        'cmp_kv_datarange': cmp_kv_datarange,
        'qk_equal_len': qk_equal_len,
        'return_softmax_lse': params_dict.get('return_softmax_lse', False),
    }
    return filled_params

def generate_case_with_default_param(param_combinations):
    case_param = param_combinations.copy()
    case_param.setdefault('testcase_name', "case_" + str(int(time.time() * 1000000)))
    case_param.update({'q_datarange': ast.literal_eval(param_combinations.get('q_datarange')) if isinstance(param_combinations.get('q_datarange'), str) else (param_combinations.get('q_datarange') if param_combinations.get('q_datarange') is not None else [-10, 10])})
    case_param.update({'ori_kv_datarange': ast.literal_eval(param_combinations.get('ori_kv_datarange')) if isinstance(param_combinations.get('ori_kv_datarange'), str) else (param_combinations.get('ori_kv_datarange') if param_combinations.get('ori_kv_datarange') is not None else [-10, 10])})
    case_param.update({'cmp_kv_datarange': ast.literal_eval(param_combinations.get('cmp_kv_datarange')) if isinstance(param_combinations.get('cmp_kv_datarange'), str) else (param_combinations.get('cmp_kv_datarange') if param_combinations.get('cmp_kv_datarange') is not None else [-10, 10])})

    layout_q = case_param['layout_q']
    layout_kv = case_param['layout_kv']
    T1 = case_param['T1']
    T2 = case_param['T2']
    B = case_param['B']
    S1 = case_param['S1']
    S2 = case_param['S2']
    cmp_ratio = case_param['cmp_ratio']

    if case_param['cu_seqlens_q'] is None:
        if layout_q == "TND":
            case_param['cu_seqlens_q'] = fill_random_cu_len(T1, S1, B, param_combinations['qk_equal_len'])
            print("cu_seqlens_q auto set to: ", case_param['cu_seqlens_q'])
        elif layout_q == "BSND":
            case_param['cu_seqlens_q'] = fill_random_cu_len(B * S1, S1, B, True)
            print("cu_seqlens_q auto set to: ", case_param['cu_seqlens_q'])

    if case_param['cu_seqlens_ori_kv'] is None and layout_kv == "TND":
        case_param['cu_seqlens_ori_kv'] = fill_random_cu_len(T2, S2, B, param_combinations['qk_equal_len'])
        print("cu_seqlens_ori_kv auto set to: ", case_param['cu_seqlens_ori_kv'])

    if case_param['cu_seqlens_cmp_kv'] is None and layout_kv == "TND" and cmp_ratio is not None:
        case_param['seqused_ori_kv'] = generate_seqused(case_param['cu_seqlens_ori_kv'])
        case_param['seqused_cmp_kv'] = [x // cmp_ratio for x in case_param['seqused_ori_kv']]
        case_param['cu_seqlens_cmp_kv'] = generate_cu_seqlens(case_param['seqused_cmp_kv'])
        print("cu_seqlens_cmp_kv auto set to: ", case_param['cu_seqlens_cmp_kv'])
        case_param['T3'] = case_param['cu_seqlens_cmp_kv'][-1]

    if case_param['seqused_ori_kv'] is None and layout_kv != "TND":
        case_param['seqused_ori_kv'] = fill_random_used_len(S2, B, param_combinations['qk_equal_len'])
        print("seqused_ori_kv auto set to: ", case_param['seqused_ori_kv'])

    if 'seqused_ori_kv' not in case_param:
        case_param['seqused_ori_kv'] = None
    if 'seqused_cmp_kv' not in case_param:
        case_param['seqused_cmp_kv'] = None
    if 'cmp_residual_kv' not in case_param:
        case_param['cmp_residual_kv'] = None

    ori_block_num_per_batch = []
    ori_block_num_sum = 0
    cmp_block_num_per_batch = []
    cmp_block_num_sum = 0

    if layout_kv == "PA_BBND":
        for cur_ori_act_kv in case_param['seqused_ori_kv']:
            cur_ori_kv_block_num = math.ceil(cur_ori_act_kv / case_param['block_size1'])
            ori_block_num_per_batch.append(cur_ori_kv_block_num)
            ori_block_num_sum += cur_ori_kv_block_num
            if cmp_ratio is not None and case_param['block_size2'] is not None:
                cur_cmp_act_kv = math.floor(cur_ori_act_kv / cmp_ratio)
                cur_cmp_kv_block_num = math.ceil(cur_cmp_act_kv / case_param['block_size2'])
                cmp_block_num_per_batch.append(cur_cmp_kv_block_num)
                cmp_block_num_sum += cur_cmp_kv_block_num
    case_param['block_num1'] = ori_block_num_sum
    case_param['block_num2'] = cmp_block_num_sum
    return case_param