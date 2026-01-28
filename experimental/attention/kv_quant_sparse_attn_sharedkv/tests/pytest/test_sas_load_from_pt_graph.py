#!/usr/bin/python
# -*- coding: utf-8 -*-
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import itertools
import torch
import check_result
import check_valid_param
import sparse_attn_sharedkv_process_quant
import sparse_attn_sharedkv_process_quant_graph
import pytest
import random
import pandas as pd
from pathlib import Path
import numpy as np
import math
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

pt_dir = "/npu/traindata/z00957671/testcase_SAS_SCFA_decode/"
result_path = Path('result_scfa_decode.xlsx')  # 或使用传入的result_path
device_id=0

# 读取pt_dir下所以pt文件
locals()["testcase_files"] = []
if os.path.isdir(pt_dir):
    pt_files = [f for f in os.listdir(pt_dir) if f.endswith('.pt')]
    if not pt_files:
        print(f"错误: 目录中没有找到.pt文件: {pt_dir}")
    else:
        print(f"找到 {len(pt_files)} 个测试用例文件")
        for pt_file in pt_files:  
            filepath = os.path.join(pt_dir, pt_file)
            locals()["testcase_files"].append(filepath)
else:
    print(f"错误: 输出目录不存在: {pt_dir}")
print("files:", locals()["testcase_files"])

# 固定case
# locals()["testcase_files"] = ["/home/l00947498/pytest_scfa_A5_esl_2/sas_testcase_redline/kvquantSparseAttenShardkv_cfa_readline_Decode_BF16_128_64_1_1_1028_512_640_000018.pt"]


def sas_aclgraph(testcase_files):   # 初始化参数和tensor
     # 加载测试数据
    test_data = torch.load(testcase_files, map_location="cpu")
    npu_result, cpu_quant_result = sparse_attn_sharedkv_process_quant_graph.test_sas_quant_process(test_data, device_id=device_id)

    if npu_result != None:
        result, fulfill_percent = check_result.check_result(cpu_quant_result, npu_result)
    else:
        result = "NPU ERROR"
        fulfill_percent = 0
    
    params = test_data['params']
    row_data = {
        "case_name": testcase_files,
        "layout_q": params[0],
        "layout_kv": params[1],
        "q_type": params[2],
        "ori_kv_type": params[3],
        "cmp_kv_type": params[4],
        "B": params[5],
        "S1": params[6],
        "T1": params[7],
        "N1": params[8],
        "N2": params[9],
        "D": params[10],
        "K": params[11],
        "block_num1": params[12],
        "block_num2": params[13],
        "block_size1": params[14],
        "block_size2": params[15],
        "cu_seqlens_q": params[16],
        "seqused_kv": params[17],
        "softmax_scale": params[18],
        "cmp_ratio": params[19],
        "ori_mask_mode": params[20],
        "cmp_mask_mode": params[21],
        "ori_win_left": params[22],
        "ori_win_right": params[23],
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

@pytest.mark.graph
@pytest.mark.parametrize("testcase_files", locals()["testcase_files"])
def test_sparse_attn_sharedkv(testcase_files):   # 初始化参数和tensor
    with ProcessPoolExecutor(max_workers=1) as executor:
        # 创建当前用例子进程
        future1 = executor.submit(sas_aclgraph, testcase_files)
        # 检查退出码
        for future in as_completed([future1]):
            try:
                result = future.result()
            except Exception as e:
                pytest.fail(f"❌ 当前用例子进程执行失败：{e}")
