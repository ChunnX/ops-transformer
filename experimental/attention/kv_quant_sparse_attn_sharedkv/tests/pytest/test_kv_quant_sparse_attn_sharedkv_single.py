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

locals()["param_combinations"] = []
for _, params in enumerate(ENABLED_PARAMS):
    # 将params的所有字段注册为局部变量
    for key, value in params.items():
        locals()[f"param_{key}"] = value

    param_names = [
        "Testcase_Name", "layout_q", "layout_kv", "q_type", "ori_kv_type", "cmp_kv_type", "B", "S1", "S2", "N1", "N2", "D", "K",
        "block_num1", "block_num2", "block_size1", "block_size2", "cu_seqlens_q", "cu_seqlens_ori_kv", "cu_seqlens_cmp_kv", "seqused_kv", "softmax_scale", "cmp_ratio",
        "ori_mask_mode", "cmp_mask_mode", "ori_win_left", "ori_win_right", "kv_quant_mode", "tile_size", "rope_head_dim",
        "ori_kv_topk_mode", "cmp_kv_topk_mode", "topk_value_mode", "template_run_mode", "actlen_mode","S1EQS2"
    ]

    param_values = [
        locals()["param_Testcase_Name"],
        locals()["param_layout_q"],
        locals()["param_layout_kv"],
        locals()["param_q_type"],
        locals()["param_ori_kv_type"],
        locals()["param_cmp_kv_type"],
        locals()["param_B"],
        locals()["param_S1"],
        locals()["param_S2"],
        locals()["param_N1"],
        locals()["param_N2"],
        locals()["param_D"],
        locals()["param_K"],
        locals()["param_block_num1"],
        locals()["param_block_num2"],
        locals()["param_block_size1"],
        locals()["param_block_size2"],
        locals()["param_cu_seqlens_q"],
        locals()["param_cu_seqlens_ori_kv"],
        locals()["param_cu_seqlens_cmp_kv"],
        locals()["param_seqused_kv"],
        locals()["param_softmax_scale"],
        locals()["param_cmp_ratio"],
        locals()["param_ori_mask_mode"],
        locals()["param_cmp_mask_mode"],
        locals()["param_ori_win_left"],
        locals()["param_ori_win_right"],
        locals()["param_kv_quant_mode"],
        locals()["param_tile_size"],
        locals()["param_rope_head_dim"],
        locals()["param_ori_kv_topk_mode"],
        locals()["param_cmp_kv_topk_mode"],
        locals()["param_topk_value_mode"],
        locals()["param_template_run_mode"],
        locals()["param_actlen_mode"],
        locals()["param_S1EQS2"],
    ]

    # 生成所有的组合，并转换为字典列表
    for combo in itertools.product(*param_values):
        param_dict = dict(zip(param_names, combo))
        locals()["param_combinations"].append(param_dict)
    print(locals()["param_combinations"])

case_id = 0
def sas(param_combinations):   # 初始化参数和tensor
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
    cu_seqlens_ori_kv = param_combinations['cu_seqlens_ori_kv']
    cu_seqlens_cmp_kv = param_combinations['cu_seqlens_cmp_kv']
    seqused_kv = param_combinations['seqused_kv']
    softmax_scale = param_combinations['softmax_scale']
    cmp_ratio = param_combinations['cmp_ratio']
    ori_mask_mode = param_combinations['ori_mask_mode']
    cmp_mask_mode = param_combinations['cmp_mask_mode']
    ori_win_left = param_combinations['ori_win_left']
    ori_win_right = param_combinations['ori_win_right']
    kv_quant_mode = param_combinations['kv_quant_mode']
    tile_size = param_combinations['tile_size']
    rope_head_dim = param_combinations['rope_head_dim']
    ori_kv_topk_mode = param_combinations['ori_kv_topk_mode']
    cmp_kv_topk_mode = param_combinations['cmp_kv_topk_mode']
    topk_value_mode = param_combinations['topk_value_mode']
    template_run_mode = param_combinations['template_run_mode']
    actlen_mode = param_combinations['actlen_mode']
    S1EQS2 = param_combinations['S1EQS2']

    ops_mode = 'prefill' if S1 > 4 else "decode"
    q_type_str = "BF16"
    if q_type == torch.float16:
        q_type_str = "FP16"
    if Testcase_Name ==None :
        Testcase_Name = f"kvquantSparseAttenShardkv_{template_run_mode}_{ops_mode}_{layout_q}_{q_type_str}_{B}_{N1}_{N2}_{S1}_{S2}_{D}_{K}_{rope_head_dim}_{case_id:06d}"

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
        futures = executor.submit(sas, param_combinations)
        # 等待并获取结果
        for future in concurrent.futures.as_completed([futures]):
            try:
                result = future.result()
            except Exception as e:
                pytest.fail(f"当前用例线程执行失败")