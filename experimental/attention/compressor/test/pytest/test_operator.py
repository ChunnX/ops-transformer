# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import itertools
import torch
import torch_npu

# ******入参调用
from testcases_operator import ENABLED_PARAMS

# ******CPU侧算子逻辑实现获取golden与npu算子直调结果
import operator_single
import check_result
import check_valid_param
import pytest

locals()["param_combinations"] = []

for _, params in enumerate(ENABLED_PARAMS):
    # 将params的所有字段注册为局部变量
    for key, value in params.items():
        locals()[f"param_{key}"] = value

    # 生成所有参数组合
    param_names = [
        "batch_size", "hidden_size", "Seq_len", "head_dim", "block_size", "rope_head_dim", "cmp_ratio",
        "coff", "norm_eps", "start_p", "rotary_mode", "layout_x", "data_type", "cu_seqlens", "seqused", "start_pos"
    ]

    param_values = [
        locals()["param_batch_size"],
        locals()["param_hidden_size"],
        locals()["param_Seq_len"],
        locals()["param_head_dim"], 
        locals()["param_block_size"],
        locals()["param_rope_head_dim"],
        locals()["param_cmp_ratio"],
        locals()["param_coff"],
        locals()["param_norm_eps"],
        locals()["param_start_p"],
        locals()["param_rotary_mode"],
        locals()["param_layout_x"],
        locals()["param_data_type"],
        locals()["param_cu_seqlens"],
        locals()["param_seqused"],
        locals()["param_start_pos"],
    ]

    # 生成所有的组合，并转换为字典列表
    for combo in itertools.product(*param_values):
        param_dict = dict(zip(param_names, combo))
        locals()["param_combinations"].append(param_dict)

    @pytest.mark.ci
    @pytest.mark.parametrize("param_combinations", locals()["param_combinations"])
    def test_compressor(param_combinations):   # 初始化参数和tensor
        batch_size = param_combinations['batch_size']
        hidden_size = param_combinations['hidden_size']
        Seq_len = param_combinations['Seq_len']
        head_dim = param_combinations['head_dim']
        block_size = param_combinations['block_size']
        rope_head_dim = param_combinations['rope_head_dim']
        cmp_ratio = param_combinations['cmp_ratio']
        coff = param_combinations['coff']
        norm_eps = param_combinations['norm_eps']
        start_p = param_combinations['start_p']
        rotary_mode = param_combinations['rotary_mode']
        layout_x = param_combinations['layout_x']
        data_type = param_combinations['data_type']
        cu_seqlens = param_combinations['cu_seqlens']
        seqused = param_combinations['seqused']
        start_pos = param_combinations['start_pos']

        
        test_data = batch_size, hidden_size, Seq_len, head_dim, block_size, rope_head_dim, cmp_ratio, coff, norm_eps, \
                    start_p, rotary_mode, layout_x, data_type, cu_seqlens, seqused, start_pos

        torch_npu.npu.set_device(0)

        # 输入参数的合法性校验
        try:
            check_valid_param.check_valid_param(test_data)
        except ValueError as e:
            pytest.skip(f"输入参数校验失败:{e}")

        # 获得cpu结果(真值)和算子结果（测试值）
        cpu_result, kv_mask_result, npu_result ,cpu_kv_state, npu_kv_state, mask_cpu_kv_state, cpu_score_state, npu_score_state, mask_cpu_score_state = operator_single.output_operator(test_data)
        # 生成布尔掩码：A 中与 B 不相等的位置为 True
        # mask_cpu_kv_state = cpu_kv_state != kv_state
        # mask_cpu_score_state = cpu_score_state != score_state
        # 使用掩码筛选 A 中不相等的元素
        npu_kv_state = npu_kv_state.cpu()
        npu_score_state = npu_score_state.cpu()
        cpu_kv_state_update = cpu_kv_state[mask_cpu_kv_state]
        npu_kv_state_update = npu_kv_state[mask_cpu_kv_state]
        cpu_kv_state_origin = npu_kv_state[~mask_cpu_kv_state]
        npu_kv_state_origin = npu_kv_state[~mask_cpu_kv_state]
        cpu_score_state_update = cpu_score_state[mask_cpu_score_state]
        npu_score_state_update = npu_score_state[mask_cpu_score_state]
        cpu_score_state_origin = cpu_score_state[~mask_cpu_score_state]
        npu_score_state_origin = npu_score_state[~mask_cpu_score_state]

        # 结果精度对比
        check_succeed = True
        data_type = str(npu_result.dtype)
        print("--------------------------------------------------------------check result-------------------------------------------------------------")
        if check_result.check_result(cpu_result[kv_mask_result].to(torch.float32), npu_result.cpu()[kv_mask_result].to(torch.float32), data_type) == False:
            print(f"test_data = {test_data} check result failed")
            check_succeed = False
        print("--------------------------------------------------------------check kv state update-------------------------------------------------------------")
        if check_result.check_result(cpu_kv_state_update.to(torch.float32), npu_kv_state_update.cpu().to(torch.float32), data_type) == False:
            print(f"test_data = {test_data} check kv state update failed")
            check_succeed = False
        print("--------------------------------------------------------------check score state update-------------------------------------------------------------")
        if check_result.check_result(cpu_score_state_update.to(torch.float32), npu_score_state_update.cpu().to(torch.float32), data_type) == False:
            print(f"test_data = {test_data} check score state update failed")
            check_succeed = False
        print("--------------------------------------------------------------check kv state origin-------------------------------------------------------------")
        if check_result.check_result(cpu_kv_state_origin.to(torch.float32), npu_kv_state_origin.cpu().to(torch.float32), data_type, 0.0) == False:
            print(f"test_data = {test_data} check kv state origin failed")
            check_succeed = False
        print("--------------------------------------------------------------check score state origin-------------------------------------------------------------")
        if check_result.check_result(cpu_score_state_origin.to(torch.float32), npu_score_state_origin.cpu().to(torch.float32), data_type, 0.0) == False:
            print(f"test_data = {test_data} check score state origin failed")
            check_succeed = False

