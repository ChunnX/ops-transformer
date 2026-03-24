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
import itertools
import torch
import result_compare_method
import check_valid_param
import kv_quant_sparse_attn_sharedkv_golden
from batch import kv_quant_sparse_attn_sharedkv_process
import math

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
        df = df.replace({np.nan: None, pd.NA: None})

        # 定义必需的列名
        required_columns = [
            "layout_q", "layout_kv", "q_type", "ori_kv_type", "cmp_kv_type", "B", "S1", "S2", "N1", "N2", "D", "K",
            "block_size1", "block_size2", "softmax_scale", "cmp_ratio",
            "ori_mask_mode", "cmp_mask_mode", "ori_win_left", "ori_win_right", "kv_quant_mode", "tile_size", "rope_head_dim", "template_run_mode",
            "actlen_mode", "S1EQS2"
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
        "case_name": params[0],
        "layout_q": params[1],
        "layout_kv": params[2],
        "q_type": params[3],
        "ori_kv_type": params[4],
        "cmp_kv_type": params[5],
        "B": params[6],
        "S1": params[7],
        "T1": params[8],
        "N1": params[11],
        "N2": params[12],
        "D": params[13],
        "K": params[14],
        "block_num1": params[15],
        "block_num2": params[16],
        "block_size1": params[17],
        "block_size2": params[18],
        "cu_seqlens_q": params[19],
        "seqused_kv": params[22],
        "softmax_scale": params[23],
        "cmp_ratio": params[24],
        "ori_mask_mode": params[25],
        "cmp_mask_mode": params[26],
        "ori_win_left": params[27],
        "ori_win_right": params[28],
        "kv_quant_mode": params[29],
        "tile_size": params[30],
        "rope_head_dim": params[31],
        "result": result,
        "fulfill_percent": fulfill_percent
    }

    # 检查文件是否存在
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

def combin_params(ENABLED_PARAMS, pytest_paramset=True):
    input_params = {}
    param_combination_set = []
    for _, params in enumerate(ENABLED_PARAMS):
        # 将params的所有字段注册为局部变量
        if pytest_paramset:
            for key, value in params.items():
                input_params[f"param_{key}"] = value
        else:
            for key, value in params.items():
                if isinstance(value, str) and value in str_map_dict:
                    value = str_map_dict[value]
                input_params[f"param_{key}"] = [value]
            # 处理表格传入的字符串 转为list
            if isinstance(input_params.get("param_template_run_mode")[0], str):
                input_params["param_template_run_mode"] = input_params["param_template_run_mode"][0].split(',')

        param_names = [
            "Testcase_Prefix", "Testcase_Number", "layout_q", "layout_kv", "q_type", "ori_kv_type", "cmp_kv_type", "B", "S1", "S2", "N1", "N2", "D", "K",
            "block_num1", "block_num2", "block_size1", "block_size2", "cu_seqlens_q", "cu_seqlens_ori_kv", "cu_seqlens_cmp_kv", "seqused_kv", "softmax_scale", "cmp_ratio",
            "ori_mask_mode", "cmp_mask_mode", "ori_win_left", "ori_win_right", "kv_quant_mode", "tile_size", "rope_head_dim",
            "ori_kv_topk_mode", "cmp_kv_topk_mode", "topk_value_mode", "template_run_mode", "actlen_mode","S1EQS2"
        ]

        param_values = [
            input_params.get("param_Testcase_Prefix", [None]),
            input_params.get("param_Testcase_Number", [None]),
            input_params["param_layout_q"],
            input_params["param_layout_kv"],
            input_params["param_q_type"],
            input_params["param_ori_kv_type"],
            input_params["param_cmp_kv_type"],
            input_params["param_B"],
            input_params["param_S1"],
            input_params["param_S2"],
            input_params["param_N1"],
            input_params["param_N2"],
            input_params["param_D"],
            input_params["param_K"],
            input_params.get("param_block_num1", [None]),
            input_params.get("param_block_num2", [None]),
            input_params["param_block_size1"],
            input_params["param_block_size2"],
            input_params.get("param_cu_seqlens_q", [None]),
            input_params.get("param_cu_seqlens_ori_kv", [None]),
            input_params.get("param_cu_seqlens_cmp_kv", [None]),
            input_params.get("param_seqused_kv", [None]),
            input_params["param_softmax_scale"],
            input_params["param_cmp_ratio"],
            input_params["param_ori_mask_mode"],
            input_params["param_cmp_mask_mode"],
            input_params["param_ori_win_left"],
            input_params["param_ori_win_right"],
            input_params["param_kv_quant_mode"],
            input_params["param_tile_size"],
            input_params["param_rope_head_dim"],
            input_params.get("param_ori_kv_topk_mode", [None]),
            input_params.get("param_cmp_kv_topk_mode", [None]),
            input_params.get("param_topk_value_mode", [None]),
            input_params["param_template_run_mode"],
            input_params["param_actlen_mode"],
            input_params["param_S1EQS2"],
        ]

        # 生成所有的组合，并转换为字典列表
        print("param_values", param_values)
        for combo in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combo))
            param_combination_set.append(param_dict)
        print(param_combination_set)
    return param_combination_set
 	 
def sas(case_id, param_combination, pt_save_path, device_id, run_npu, save_pt, result_path):   # 初始化参数和tensor
    Testcase_Prefix = param_combination['Testcase_Prefix'] if param_combination['Testcase_Prefix'] is not None else "kvquantSparseAttenShardkv"
    Testcase_Number = param_combination['Testcase_Number'] if param_combination['Testcase_Number'] is not None else case_id
    layout_q = param_combination['layout_q']
    layout_kv = param_combination['layout_kv']
    q_type = param_combination['q_type']
    ori_kv_type = param_combination['ori_kv_type']
    cmp_kv_type = param_combination['cmp_kv_type']
    B = param_combination['B']
    S1 = param_combination['S1']
    S2 = param_combination['S2']
    N1 = param_combination['N1']
    N2 = param_combination['N2']
    D = param_combination['D']
    K = param_combination['K']
    block_num1 = param_combination['block_num1']
    block_num2 = param_combination['block_num2']
    block_size1 = param_combination['block_size1']
    block_size2 = param_combination['block_size2']
    cu_seqlens_q = param_combination['cu_seqlens_q']
    cu_seqlens_ori_kv = param_combination['cu_seqlens_ori_kv']
    cu_seqlens_cmp_kv = param_combination['cu_seqlens_cmp_kv']
    seqused_kv = param_combination['seqused_kv']
    softmax_scale = param_combination['softmax_scale']
    cmp_ratio = param_combination['cmp_ratio']
    ori_mask_mode = param_combination['ori_mask_mode']
    cmp_mask_mode = param_combination['cmp_mask_mode']
    ori_win_left = param_combination['ori_win_left']
    ori_win_right = param_combination['ori_win_right']
    kv_quant_mode = param_combination['kv_quant_mode']
    tile_size = param_combination['tile_size']
    rope_head_dim = param_combination['rope_head_dim']
    ori_kv_topk_mode = param_combination['ori_kv_topk_mode']
    cmp_kv_topk_mode = param_combination['cmp_kv_topk_mode']
    topk_value_mode = param_combination.get('topk_value_mode') or 1
    template_run_mode = param_combination['template_run_mode']
    actlen_mode = param_combination['actlen_mode']
    S1EQS2 = param_combination['S1EQS2']

    ops_mode = 'prefill' if S1 > 4 else "decode"
    q_type_str = "BF16"
    if q_type == torch.float16:
        q_type_str = "FP16"

    Testcase_Name = f"{Testcase_Prefix}_{template_run_mode}_{ops_mode}_{layout_q}_{q_type_str}_{B}_{N1}_{N2}_{S1}_{S2}_{D}_{K}_{rope_head_dim}_{Testcase_Number:06d}"

    # 生成actLen
    QS = [0]
    KVS = []
    if cu_seqlens_q == None or seqused_kv == None:
        if layout_q == "TND":
            for i in range(B):
                if actlen_mode == "random":
                    tmpS1 = S1 - random.randint(0, S1 - 1)
                    QS.append(tmpS1 + QS[-1])
                    if layout_kv != "TND":
                        if S1EQS2: #  qs=kvs
                            tmpS2 = tmpS1
                        else: # qs != kvs
                            tmpS2 = S2 - random.randint(0, math.ceil(S2/2) - 1)
                        KVS.append(tmpS2)
                elif actlen_mode == "full":  ## 走该分支
                    tmpS1 = S1
                    QS.append(tmpS1 + QS[-1])
                    if layout_kv != "TND":
                        KVS.append(S2)
        elif layout_q == "BSND":
            for i in range(B):
                QS.append(S1 + QS[-1])
                if actlen_mode == "random":
                    tmpS2 = S2 - random.randint(0, math.ceil(S2/2) - 1)
                else:
                    tmpS2 = S2
                KVS.append(tmpS2)

        cu_seqlens_q = QS if cu_seqlens_q == None else cu_seqlens_q
        seqused_kv = KVS if layout_kv != "TND" else None
    T1 = cu_seqlens_q[-1] if layout_q == "TND" else None
    T2 = cu_seqlens_ori_kv[-1] if layout_kv == "TND" else None
    T3 = cu_seqlens_cmp_kv[-1] if (template_run_mode != "SWA" and layout_kv == "TND") else None
    # 生成ori_kv_topk
    if template_run_mode == "ORI_SCFA" or template_run_mode == "ALL_SCFA":
        if ori_kv_topk_mode == "full":
            if layout_q == "TND" or layout_q == "BSND":
                ori_topk_length = torch.tensor(np.random.uniform(K, K, (B, 1))).to(torch.int32)
            else:
                raise ValueError(f"layout_q is not support {layout_q}")
        elif ori_kv_topk_mode == "random":
            if layout_q == "TND" or layout_q == "BSND":
                ori_topk_length = torch.tensor(np.random.uniform(1, K, (B, 1))).to(torch.int32)
            else:
                raise ValueError(f"layout_q is not support {layout_q}")
        else:
            ori_topk_length = None
    else:
        ori_topk_length = None

    # 生成cmp_kv_topk
    if template_run_mode == "SCFA" or template_run_mode == "ALL_SCFA":
        cmp_topK = 512 if cmp_ratio == 4 else S2 // cmp_ratio
        if cmp_kv_topk_mode == "full":
            if layout_q == "TND" or layout_q == "BSND":
                cmp_topk_length = torch.tensor(np.random.uniform(cmp_topK, cmp_topK, (B, 1))).to(torch.int32)
            else:
                raise ValueError(f"layout_q is not support {layout_q}")
        elif cmp_kv_topk_mode == "random":
            if layout_q == "TND" or layout_q == "BSND":
                cmp_topk_length = torch.tensor(np.random.uniform(1, cmp_topK, (B, 1))).to(torch.int32)
            else:
                raise ValueError(f"layout_q is not support {layout_q}")
        else:
            cmp_topk_length = None
    else:
        cmp_topk_length = None

    # maxSeqLen / block_size向上取整
    ori_block_num_per_batch = []
    ori_block_num_sum = 0
    cmp_block_num_per_batch = []
    cmp_block_num_sum = 0
    if layout_kv != "TND":
        for cur_ori_act_kv in seqused_kv:
            cur_ori_kv_block_num = math.ceil(cur_ori_act_kv / block_size1)
            ori_block_num_per_batch.append(cur_ori_kv_block_num)
            ori_block_num_sum += cur_ori_kv_block_num

            cur_cmp_act_kv = math.floor(cur_ori_act_kv / cmp_ratio)
            cur_cmp_kv_block_num = math.ceil(cur_cmp_act_kv / block_size2)
            cmp_block_num_per_batch.append(cur_cmp_kv_block_num)
            cmp_block_num_sum += cur_cmp_kv_block_num
    block_num1 = ori_block_num_sum
    block_num2 = cmp_block_num_sum

    params = Testcase_Name, layout_q, layout_kv, q_type, ori_kv_type, cmp_kv_type, B, S1, T1, T2, T3, N1, N2, D, K, block_num1, \
                block_num2, block_size1, block_size2, cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv, seqused_kv, softmax_scale, cmp_ratio, ori_mask_mode, \
                cmp_mask_mode, ori_win_left, ori_win_right, kv_quant_mode, tile_size, rope_head_dim, ori_topk_length, cmp_topk_length, \
                topk_value_mode, template_run_mode

    # 输入参数的合法性校验
    try:
        check_valid_param.check_valid_param(params)
    except ValueError as e:
        pytest.skip(f"输入参数校验失败:{e}")

    # 生成测试数据及cpu golden
    test_data = kv_quant_sparse_attn_sharedkv_golden.generate_and_save_testdata(params, save_pt=save_pt, save_path=pt_save_path)
    if run_npu:
        sas_run_npu(test_data, device_id=device_id, result_path=result_path)

def sas_run_npu(test_data, device_id=0, result_path=Path('result.xlsx')):
     # 获得npu结果（测试值）
    try:
        npu_result, cpu_quant_result = kv_quant_sparse_attn_sharedkv_process.test_sas_quant_process_ci(test_data, device_id=device_id)
        result, fulfill_percent = result_compare_method.check_result(cpu_quant_result, npu_result)
    except Exception as e:
        print("NPU ERROR：", e)
        result = "NPU ERROR"
        fulfill_percent = 0

    save_result(test_data['params'], result,  fulfill_percent, result_path=result_path)

    if(result == "NPU ERROR"):
        pytest.fail(f"用例执行失败:{test_data['Testcase_Name']}")
