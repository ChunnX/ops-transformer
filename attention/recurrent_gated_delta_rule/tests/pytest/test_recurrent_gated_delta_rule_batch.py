# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
from batch import recurrent_gated_delta_rule_pt_loadprocess
import pytest
import pandas as pd
from pathlib import Path
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from recurrent_gated_delta_rule_golden import check_result

TEST_INPUT_PATH = "./pt_path"
pt_dir = TEST_INPUT_PATH
result_path = Path('result.xlsx')  # 或使用传入的result_path

# 生成所有的组合，并转换为字典列表
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

def recurrent_gated_delta_rule(testcase_files):   # 初始化参数和tensor
    cpu_result, npu_result, cpu_state_result, npu_state_result, params = recurrent_gated_delta_rule_pt_loadprocess.test_recurrent_gated_delta_rule_process(testcase_files, device_id=0)
    if npu_result != None:
        data_type = str(npu_result.dtype)
        print("--------------------------------------------------------------check result-------------------------------------------------------------")
        result_percent, result= check_result(cpu_result.to(torch.float32), npu_result.cpu().to(torch.float32), data_type)
        print("--------------------------------------------------------------check kv state update-------------------------------------------------------------")
        state_result_percent, state_result= check_result(cpu_state_result.to(torch.float32), npu_state_result.cpu().to(torch.float32), data_type)
    else:
        result = "Failed"
        result_percent = 0
        state_result = "Failed"
        state_result_percent = 0

    
    row_data = {
        "Testcase_Name": Path(testcase_files).stem,
        "batch_size": params[0],
        "mtp": params[1],
        "nk": params[2],
        "nv": params[3],
        "dk": params[4],
        "dv": params[5],
        "actual_seq_lengths": params[6],
        "ssm_state_indices": params[7],
        "has_gamma": params[8],
        "has_gamma_k": params[9],
        "has_num_accepted_tokens": params[10],
        "scale_value": params[11],
        "block_num": params[12],
        "data_type": params[13],
        "query_datarange": params[14],
        "key_datarange": params[15],
        "value_datarange": params[16],
        "gamma_datarange": params[17],
        "gamma_k_datarange": params[18],
        "beta_datarange": params[19],
        "state_datarange": params[20],
        "result":result,
        "result_percent":result_percent,
        "state_result":state_result,
        "state_result_percent":state_result_percent
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

@pytest.mark.ci
@pytest.mark.parametrize("testcase_files", locals()["testcase_files"])
def test_recurrent_gated_delta_rule(testcase_files):   # 初始化参数和tensor
    with ProcessPoolExecutor(max_workers=1) as executor:
        # 创建当前用例子进程
        future1 = executor.submit(recurrent_gated_delta_rule, testcase_files)
        # 检查退出码
        for future in as_completed([future1]):
            try:
                result = future.result()
            except Exception as e:
                pytest.fail(f"❌ 当前用例子进程执行失败：{e}")
