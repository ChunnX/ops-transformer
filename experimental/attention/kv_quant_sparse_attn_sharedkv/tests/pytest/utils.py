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
            template_run_mode = "CFA"
            cmp_ratio = int(cmp_ratio)
            cmp_kv_type = ori_kv_type if cmp_kv_type is None else cmp_kv_type
            K = 0
            block_size2 = int(block_size2) if block_size2 is not None else block_size1
            cmp_mask_mode = int(cmp_mask_mode) if cmp_mask_mode is not None else 3
            block_num2 = block_num2  # 后续计算
    else:
        template_run_mode = "SCFA"
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

def gen_cu_seqlens_and_seqused(B, S1, S2, layout_q, actlen_mode, S1EQS2):
    """
    生成cu_seqlens_q和seqused_ori_kv
    
    参数:
        B: batch size
        S1: Q序列长度
        S2: KV序列长度 (可为None)
        layout_q: Q的layout (TND/BSND)
        actlen_mode: actlen模式 (random/full)
        S1EQS2: S1是否等于S2
    
    返回:
        tuple: (cu_seqlens_q, seqused_ori_kv)
    """
    QS = [0]
    KVS = []
    
    if layout_q == "TND":
        for i in range(B):
            if actlen_mode == "random":
                tmpS1 = S1 - random.randint(0, S1 - 1)
                QS.append(tmpS1 + QS[-1])
                if S1EQS2:
                    tmpS2 = tmpS1
                else:
                    tmpS2 = S2 - random.randint(0, math.ceil(S2/2) - 1) if S2 is not None else tmpS1
                KVS.append(tmpS2)
            elif actlen_mode == "full":
                tmpS1 = S1
                QS.append(tmpS1 + QS[-1])
                KVS.append(S2 if S2 is not None else S1)
    elif layout_q == "BSND":
        for i in range(B):
            QS.append(S1 + QS[-1])
            if actlen_mode == "random":
                tmpS2 = S2 - random.randint(0, math.ceil(S2/2) - 1) if S2 is not None else S1
            else:
                tmpS2 = S2 if S2 is not None else S1
            KVS.append(tmpS2)
    
    return QS, KVS

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

def gen_cmp_residual_kv(seqused_ori_kv, cmp_ratio, cmp_mask_mode):
    """
    生成cmp_residual_kv（仅当cmp_mask_mode=3且cmp_ratio≠1）
    
    参数:
        seqused_ori_kv: ori_kv真实长度列表
        cmp_ratio: 压缩率
        cmp_mask_mode: cmp mask模式
    
    返回:
        list or None: cmp_residual_kv列表或None
    """
    if cmp_mask_mode == 3 and cmp_ratio != 1:
        return [s % cmp_ratio for s in seqused_ori_kv]
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

def calc_block_num(seqused_ori_kv, seqused_cmp_kv, block_size1, block_size2, cmp_ratio):
    """
    计算block_num1和block_num2
    
    参数:
        seqused_ori_kv: ori_kv真实长度列表
        seqused_cmp_kv: cmp_kv真实长度列表
        block_size1: ori block大小
        block_size2: cmp block大小
        cmp_ratio: 压缩率
    
    返回:
        tuple: (block_num1, block_num2)
    """
    ori_block_num_sum = 0
    cmp_block_num_sum = 0
    
    for cur_ori_act_kv in seqused_ori_kv:
        cur_ori_kv_block_num = math.ceil(cur_ori_act_kv / block_size1)
        ori_block_num_sum += cur_ori_kv_block_num
        
        cur_cmp_act_kv = math.floor(cur_ori_act_kv / cmp_ratio)
        cur_cmp_kv_block_num = math.ceil(cur_cmp_act_kv / block_size2)
        cmp_block_num_sum += cur_cmp_kv_block_num
    
    return ori_block_num_sum, cmp_block_num_sum

def fill_none_params(params_dict):
    """
    填充None参数的默认值（完整封装）
    
    参数:
        params_dict: 参数字典
    
    返回:
        dict: 填充后的完整参数字典
    """
    layout_q = params_dict['layout_q']
    B = params_dict['B']
    S1 = params_dict['S1']
    S2 = params_dict['S2']
    ori_kv_type = params_dict['ori_kv_type']
    block_size1 = params_dict['block_size1']
    actlen_mode = params_dict.get('actlen_mode', 'full')
    S1EQS2 = params_dict.get('S1EQS2', False)
    
    # 1. 推断template_run_mode及相关参数
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
    
    # 2. 生成cu_seqlens_q和seqused_ori_kv
    cu_seqlens_q = params_dict.get('cu_seqlens_q')
    seqused_ori_kv = params_dict.get('seqused_ori_kv')
    
    if cu_seqlens_q is None or seqused_ori_kv is None:
        cu_seqlens_q, seqused_ori_kv = gen_cu_seqlens_and_seqused(B, S1, S2, layout_q, actlen_mode, S1EQS2)
    
    # 3. 生成seqused_cmp_kv
    seqused_cmp_kv = params_dict.get('seqused_cmp_kv')
    if seqused_cmp_kv is None:
        seqused_cmp_kv = gen_seqused_cmp_kv(seqused_ori_kv, infer_result['cmp_ratio'])
    
    # 4. 生成cmp_residual_kv
    cmp_residual_kv = params_dict.get('cmp_residual_kv')
    if cmp_residual_kv is None:
        cmp_residual_kv = gen_cmp_residual_kv(seqused_ori_kv, infer_result['cmp_ratio'], infer_result['cmp_mask_mode'])
    
    # 5. 生成cu_seqlens_ori_kv和cu_seqlens_cmp_kv
    cu_seqlens_ori_kv = params_dict.get('cu_seqlens_ori_kv')
    if cu_seqlens_ori_kv is None:
        cu_seqlens_ori_kv = gen_cu_seqlens_from_seqused(seqused_ori_kv)
    
    cu_seqlens_cmp_kv = params_dict.get('cu_seqlens_cmp_kv')
    if cu_seqlens_cmp_kv is None:
        cu_seqlens_cmp_kv = gen_cu_seqlens_from_seqused(seqused_cmp_kv)
    
    # 6. 计算block_num
    block_num1 = params_dict.get('block_num1')
    block_num2 = params_dict.get('block_num2')
    
    if block_num1 is None or block_num2 is None:
        ori_block_num, cmp_block_num = calc_block_num(
            seqused_ori_kv, seqused_cmp_kv, block_size1, infer_result['block_size2'], infer_result['cmp_ratio']
        )
        if block_num1 is None:
            block_num1 = ori_block_num
        if block_num2 is None:
            block_num2 = cmp_block_num
    
    # 7. 计算T1
    T1 = cu_seqlens_q[-1] if layout_q == "TND" else None
    
    # 构建完整参数字典
    filled_params = {
        'Testcase_Name': params_dict.get('Testcase_Name'),
        'layout_q': layout_q,
        'layout_kv': params_dict['layout_kv'],
        'q_type': params_dict['q_type'],
        'ori_kv_type': ori_kv_type,
        'cmp_kv_type': infer_result['cmp_kv_type'],
        'B': B,
        'S1': S1,
        'T1': T1,
        'N1': params_dict['N1'],
        'N2': params_dict['N2'],
        'D': params_dict['D'],
        'K': infer_result['K'],
        'block_num1': block_num1,
        'block_num2': block_num2,
        'block_size1': block_size1,
        'block_size2': infer_result['block_size2'],
        'cu_seqlens_q': cu_seqlens_q,
        'seqused_ori_kv': seqused_ori_kv,
        'seqused_cmp_kv': seqused_cmp_kv,
        'cu_seqlens_ori_kv': cu_seqlens_ori_kv,
        'cu_seqlens_cmp_kv': cu_seqlens_cmp_kv,
        'cmp_residual_kv': cmp_residual_kv,
        'softmax_scale': params_dict['softmax_scale'],
        'cmp_ratio': infer_result['cmp_ratio'],
        'ori_mask_mode': params_dict['ori_mask_mode'],
        'cmp_mask_mode': infer_result['cmp_mask_mode'],
        'ori_win_left': params_dict['ori_win_left'],
        'ori_win_right': params_dict['ori_win_right'],
        'kv_quant_mode': params_dict['kv_quant_mode'],
        'tile_size': params_dict['tile_size'],
        'rope_head_dim': params_dict['rope_head_dim'],
        'template_run_mode': infer_result['template_run_mode'],
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
        df = df.replace({np.nan: None, pd.NA: None})

        # 定义必需的列名
        required_columns = [
            "Testcase_Name", "layout_q", "layout_kv", "q_type", "ori_kv_type", "cmp_kv_type", "B", "S1", "S2", "N1",
            "N2", "D", "K", "block_size1", "block_size2", "softmax_scale", "cmp_ratio",
            "ori_mask_mode", "cmp_mask_mode", "ori_win_left", "ori_win_right", "kv_quant_mode", "tile_size",
            "rope_head_dim", "template_run_mode", "actlen_mode", "S1EQS2",
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
        "case_name": params.get('Testcase_Name'),
        "layout_q": params.get('layout_q'),
        "layout_kv": params.get('layout_kv'),
        "q_type": params.get('q_type'),
        "ori_kv_type": params.get('ori_kv_type'),
        "cmp_kv_type": params.get('cmp_kv_type'),
        "B": params.get('B'),
        "S1": params.get('S1'),
        "T1": params.get('T1'),
        "N1": params.get('N1'),
        "N2": params.get('N2'),
        "D": params.get('D'),
        "K": params.get('K'),
        "block_num1": params.get('block_num1'),
        "block_num2": params.get('block_num2'),
        "block_size1": params.get('block_size1'),
        "block_size2": params.get('block_size2'),
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
        "ori_win_right": params.get('ori_win_right'),
        "kv_quant_mode": params.get('kv_quant_mode'),
        "tile_size": params.get('tile_size'),
        "rope_head_dim": params.get('rope_head_dim'),
        "result": result,
        "fulfill_percent": fulfill_percent,
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