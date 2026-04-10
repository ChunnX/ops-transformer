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
import itertools
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pytest
import torch
 	 
str_map_dict = {
    'True': True,
    'False': False,
    'TRUE': True,
    'FALSE': False,
    'torch.bfloat16': torch.bfloat16,
    'torch.float8_e4m3fn': torch.float8_e4m3fn,
}

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

        # 处理可能为空的列
        default_None_columns = ["K1", "cu_seqlens_ori_kv","cu_seqlens_cmp_kv"]
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
                "K1",
                "K",
                "layout_q",
                "layout_kv",
            ]
            
            df["testcase_name"] = df[cols_to_combine].astype(str).agg("_".join, axis=1)

        # 定义必需的列名
        required_columns = [
            "layout_q", "layout_kv", "q_type", "ori_kv_type", "cmp_kv_type", "B", "S1", "S2", "N1", "N2", "D", "K1", "K",
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

def save_result(result, fulfill_percent, params, result_path='./result/sas_result.xlsx'):
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
            "seqused_kv": params.get("seqused_kv", [None] if not is_save_pt else None),
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
            "qk_equal_len": params.get("qk_equal_len", ["False"] if not is_save_pt else False)
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
