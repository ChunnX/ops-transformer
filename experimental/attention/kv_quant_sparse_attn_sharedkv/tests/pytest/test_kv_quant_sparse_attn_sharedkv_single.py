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

import itertools
import torch
from kv_quant_sparse_attn_sharedkv_paramset import ENABLED_PARAMS
import result_compare_method
import check_valid_param
import kv_quant_sparse_attn_sharedkv_golden
from batch import kv_quant_sparse_attn_sharedkv_process
import pytest
import random
import pandas as pd
from pathlib import Path
import numpy as np
import math
import os
import multiprocessing as mp
import concurrent.futures
import utils

pt_save_path = "qsas_testcase"
device_id = 0
save_pt = False
result_path = Path('result.xlsx')

param_combinations = []
for params in ENABLED_PARAMS:
    param_values = {
        "Testcase_Name": params.get("Testcase_Name", [None]),
        "layout_q": params.get("layout_q"),
        "layout_kv": params.get("layout_kv"),
        "q_type": params.get("q_type"),
        "ori_kv_type": params.get("ori_kv_type"),
        "cmp_kv_type": params.get("cmp_kv_type", [None]),
        "B": params.get("B"),
        "S1": params.get("S1"),
        "S2": params.get("S2", [None]),
        "N1": params.get("N1"),
        "N2": params.get("N2"),
        "D": params.get("D"),
        "K": params.get("K", [None]),
        "block_num1": params.get("block_num1", [None]),
        "block_num2": params.get("block_num2", [None]),
        "block_size1": params.get("block_size1"),
        "block_size2": params.get("block_size2", [None]),
        "cu_seqlens_q": params.get("cu_seqlens_q", [None]),
        "seqused_ori_kv": params.get("seqused_ori_kv", [None]),
        "seqused_cmp_kv": params.get("seqused_cmp_kv", [None]),
        "cu_seqlens_ori_kv": params.get("cu_seqlens_ori_kv", [None]),
        "cu_seqlens_cmp_kv": params.get("cu_seqlens_cmp_kv", [None]),
        "cmp_residual_kv": params.get("cmp_residual_kv", [None]),
        "softmax_scale": params.get("softmax_scale"),
        "cmp_ratio": params.get("cmp_ratio", [None]),
        "ori_mask_mode": params.get("ori_mask_mode"),
        "cmp_mask_mode": params.get("cmp_mask_mode", [None]),
        "ori_win_left": params.get("ori_win_left"),
        "ori_win_right": params.get("ori_win_right"),
        "kv_quant_mode": params.get("kv_quant_mode"),
        "tile_size": params.get("tile_size"),
        "rope_head_dim": params.get("rope_head_dim"),
        "actlen_mode": params.get("actlen_mode"),
        "S1EQS2": params.get("S1EQS2", [False]),
    }

    param_names = list(param_values.keys())
    values_lists = [param_values[name] for name in param_names]

    for combo in itertools.product(*values_lists):
        combination = dict(zip(param_names, combo))
        param_combinations.append(combination)

case_id = 0
def qsas(param_combinations):
    global case_id
    Testcase_Name = param_combinations['Testcase_Name']
    layout_q = param_combinations['layout_q']
    layout_kv = param_combinations['layout_kv']
    q_type = param_combinations['q_type']
    ori_kv_type = param_combinations['ori_kv_type']
    cmp_kv_type = param_combinations['cmp_kv_type']
    B = param_combinations['B']
    S1 = param_combinations['S1']
    S2 = param_combinations['S2']
    N1 = param_combinations['N1']
    N2 = param_combinations['N2']
    D = param_combinations['D']
    K = param_combinations['K']
    block_num1 = param_combinations['block_num1']
    block_num2 = param_combinations['block_num2']
    block_size1 = param_combinations['block_size1']
    block_size2 = param_combinations['block_size2']
    cu_seqlens_q = param_combinations['cu_seqlens_q']
    seqused_ori_kv = param_combinations['seqused_ori_kv']
    seqused_cmp_kv = param_combinations['seqused_cmp_kv']
    cu_seqlens_ori_kv = param_combinations['cu_seqlens_ori_kv']
    cu_seqlens_cmp_kv = param_combinations['cu_seqlens_cmp_kv']
    cmp_residual_kv = param_combinations['cmp_residual_kv']
    softmax_scale = param_combinations['softmax_scale']
    cmp_ratio = param_combinations['cmp_ratio']
    ori_mask_mode = param_combinations['ori_mask_mode']
    cmp_mask_mode = param_combinations['cmp_mask_mode']
    ori_win_left = param_combinations['ori_win_left']
    ori_win_right = param_combinations['ori_win_right']
    kv_quant_mode = param_combinations['kv_quant_mode']
    tile_size = param_combinations['tile_size']
    rope_head_dim = param_combinations['rope_head_dim']
    actlen_mode = param_combinations['actlen_mode']
    S1EQS2 = param_combinations['S1EQS2']

    ops_mode = 'prefill' if S1 > 4 else "decode"
    q_type_str = "BF16"
    if q_type == torch.float16:
        q_type_str = "FP16"
    kv_type_str = "FP8_E4M3FN"
    if ori_kv_type == torch.uint8:
        kv_type_str = "HIF8"

    template_run_mode = "SWA"
    if K is None or K == ['None']:
        if cmp_ratio is None or cmp_ratio == ['None']:
            template_run_mode = "SWA"
            cmp_ratio = 1
            cmp_kv_type = ori_kv_type
            cmp_mask_mode = 0
            K = 0
            block_size2 = block_size1
            block_num2 = 0
        else:
            template_run_mode = "CFA"
            cmp_ratio = int(cmp_ratio)
            cmp_kv_type = ori_kv_type if cmp_kv_type is None else cmp_kv_type
            K = 0
            block_size2 = int(block_size2) if block_size2 is not None else block_size1
            cmp_mask_mode = int(cmp_mask_mode) if cmp_mask_mode is not None else 3
    else:
        template_run_mode = "SCFA"
        K = int(K)
        cmp_ratio = int(cmp_ratio)
        cmp_kv_type = ori_kv_type if cmp_kv_type is None else cmp_kv_type
        block_size2 = int(block_size2) if block_size2 is not None else block_size1
        cmp_mask_mode = int(cmp_mask_mode) if cmp_mask_mode is not None else 3

    if Testcase_Name == None:
        Testcase_Name = f"kvquantSparseAttenShardkv_{template_run_mode}_{ops_mode}_{layout_q}_{q_type_str}_{layout_kv}_{kv_type_str}_{B}_{N1}_{N2}_{S1}_{S2}_{D}_{K}_{rope_head_dim}_{case_id:06d}"

    QS = [0]
    KVS = []
    if cu_seqlens_q == None or seqused_ori_kv == None:
        if layout_q == "TND":
            for i in range(B):
                if actlen_mode == "random":
                    tmpS1 = S1 - random.randint(0, S1 - 1)
                    QS.append(tmpS1 + QS[-1])
                    if S1EQS2:
                        tmpS2 = tmpS1
                    else:
                        tmpS2 = S2 - random.randint(0, math.ceil(S2/2) - 1)
                    KVS.append(tmpS2)
                elif actlen_mode == "full":
                    tmpS1 = S1
                    QS.append(tmpS1 + QS[-1])
                    KVS.append(S2)
        elif layout_q == "BSND":
            for i in range(B):
                QS.append(S1 + QS[-1])
                if actlen_mode == "random":
                    tmpS2 = S2 - random.randint(0, math.ceil(S2/2) - 1)
                else:
                    tmpS2 = S2
                KVS.append(tmpS2)

        cu_seqlens_q = QS
        seqused_ori_kv = KVS

    if seqused_cmp_kv == None:
        seqused_cmp_kv = [math.floor(s / cmp_ratio) for s in seqused_ori_kv]

    if cmp_mask_mode == 3 and cmp_ratio != 1:
        if cmp_residual_kv == None:
            cmp_residual_kv = [s % cmp_ratio for s in seqused_ori_kv]

    if cu_seqlens_ori_kv == None:
        cu_seqlens_ori_kv = [0]
        for s in seqused_ori_kv:
            cu_seqlens_ori_kv.append(cu_seqlens_ori_kv[-1] + s)

    if cu_seqlens_cmp_kv == None:
        cu_seqlens_cmp_kv = [0]
        for s in seqused_cmp_kv:
            cu_seqlens_cmp_kv.append(cu_seqlens_cmp_kv[-1] + s)

    T1 = cu_seqlens_q[-1] if layout_q == "TND" else None

    ori_block_num_per_batch = []
    ori_block_num_sum = 0
    cmp_block_num_per_batch = []
    cmp_block_num_sum = 0
    for cur_ori_act_kv in seqused_ori_kv:
        cur_ori_kv_block_num = math.ceil(cur_ori_act_kv / block_size1)
        ori_block_num_per_batch.append(cur_ori_kv_block_num)
        ori_block_num_sum += cur_ori_kv_block_num

        cur_cmp_act_kv = math.floor(cur_ori_act_kv / cmp_ratio)
        cur_cmp_kv_block_num = math.ceil(cur_cmp_act_kv / block_size2)
        cmp_block_num_per_batch.append(cur_cmp_kv_block_num)
        cmp_block_num_sum += cur_cmp_kv_block_num
    
    if block_num1 is None:
        block_num1 = ori_block_num_sum
    if block_num2 is None:
        block_num2 = cmp_block_num_sum
    
    params = {
        'Testcase_Name': Testcase_Name,
        'layout_q': layout_q,
        'layout_kv': layout_kv,
        'q_type': q_type,
        'ori_kv_type': ori_kv_type,
        'cmp_kv_type': cmp_kv_type,
        'B': B,
        'S1': S1,
        'T1': T1,
        'N1': N1,
        'N2': N2,
        'D': D,
        'K': K,
        'block_num1': block_num1,
        'block_num2': block_num2,
        'block_size1': block_size1,
        'block_size2': block_size2,
        'cu_seqlens_q': cu_seqlens_q,
        'seqused_ori_kv': seqused_ori_kv,
        'seqused_cmp_kv': seqused_cmp_kv,
        'cu_seqlens_ori_kv': cu_seqlens_ori_kv,
        'cu_seqlens_cmp_kv': cu_seqlens_cmp_kv,
        'cmp_residual_kv': cmp_residual_kv,
        'softmax_scale': softmax_scale,
        'cmp_ratio': cmp_ratio,
        'ori_mask_mode': ori_mask_mode,
        'cmp_mask_mode': cmp_mask_mode,
        'ori_win_left': ori_win_left,
        'ori_win_right': ori_win_right,
        'kv_quant_mode': kv_quant_mode,
        'tile_size': tile_size,
        'rope_head_dim': rope_head_dim,
        'template_run_mode': template_run_mode,
    }

    # 输入参数的合法性校验
    try:
        check_valid_param.check_valid_param(params)
    except ValueError as e:
       pytest.skip(f"输入参数校验失败:{e}")

    # 生成测试数据及golden
    test_data = kv_quant_sparse_attn_sharedkv_golden.generate_and_save_testdata(params, save_pt=save_pt, save_path=pt_save_path)

    # 获得cpu结果(真值)和算子结果（测试值）
    try:
        npu_result, cpu_quant_result = kv_quant_sparse_attn_sharedkv_process.test_sas_quant_process_ci(test_data, device_id=device_id)
        result, fulfill_percent = result_compare_method.check_result(cpu_quant_result, npu_result)
    except Exception as e:
        print("NPU ERROR：", e)
        result = "NPU ERROR"
        fulfill_percent = 0

    case_id += 1

    utils.save_result(test_data['params'], result, fulfill_percent, result_path)
    
    if(result == "NPU ERROR"):
        pytest.fail(f"用例执行失败:{test_data['Testcase_Name']}")
    
@pytest.mark.ci
@pytest.mark.parametrize("param_combinations", locals()["param_combinations"])
def test_sparse_attn_sharedkv(param_combinations):
    # 线程池
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = executor.submit(qsas, param_combinations)
        # 等待并获取结果
        for future in concurrent.futures.as_completed([futures]):
            try:
                result = future.result()
            except Exception as e:
                pytest.fail(f"当前用例线程执行失败")