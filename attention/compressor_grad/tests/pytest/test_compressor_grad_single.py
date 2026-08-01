# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
from test_compressor_grad_paramset import ENABLED_PARAMS
print("test_compressor_grad HYQ HYQ HYQ 111", flush=True)
# ******CPU侧算子逻辑实现获取golden与npu算子直调结果
import compressor_grad_operator_single
import pytest

locals()["param_combinations"] = []

for _, params in enumerate(ENABLED_PARAMS):
    # 将params的所有字段注册为局部变量
    for key, value in params.items():
        locals()[f"param_{key}"] = value

    # 生成所有参数组合
    param_names = [
        "batch_size", "hidden_size", "Seq_len", "head_dim", "block_num", "block_size", "cmp_ratio",
        "coff", "layout_x", "layout_d_cmp_kv", "layout_softmax_score", "layout_kv", "layout_d_x", "data_type", "cu_seqlens", "seqused", "start_pos",
        "x_datarange", "wkv_datarange", "wgate_datarange", "d_cmp_kv_datarange", "softmax_score_datarange", "kv_datarange"
    ]

    param_values = [
        locals()["param_batch_size"],
        locals()["param_hidden_size"],
        locals()["param_Seq_len"],
        locals()["param_head_dim"],
        locals()["param_block_num"], 
        locals()["param_block_size"],
        locals()["param_cmp_ratio"],
        locals()["param_coff"],
        locals()["param_layout_x"],
        locals()["param_layout_d_cmp_kv"],
        locals()["param_layout_softmax_score"],
        locals()["param_layout_kv"],
        locals()["param_layout_d_x"],
        locals()["param_data_type"],
        locals()["param_cu_seqlens"],
        locals()["param_seqused"],
        locals()["param_start_pos"],
        locals()["param_x_datarange"],
        locals()["param_wkv_datarange"],
        locals()["param_wgate_datarange"],
        locals()["param_d_cmp_kv_datarange"],
        locals()["param_softmax_score_datarange"],
        locals()["param_kv_datarange"],
    ]

    # 生成所有的组合，并转换为字典列表
    for combo in itertools.product(*param_values):
        param_dict = dict(zip(param_names, combo))
        locals()["param_combinations"].append(param_dict)

    @pytest.mark.ci
    @pytest.mark.parametrize("param_combinations", locals()["param_combinations"])
    def test_compressor_grad(param_combinations):   # 初始化参数和tensor
        print("test_compressor_grad HYQ HYQ HYQ", flush=True)
        batch_size = param_combinations['batch_size']
        hidden_size = param_combinations['hidden_size']
        Seq_len = param_combinations['Seq_len']
        head_dim = param_combinations['head_dim']
        block_num = param_combinations['block_num']
        block_size = param_combinations['block_size']
        cmp_ratio = param_combinations['cmp_ratio']
        coff = param_combinations['coff']
        layout_x = param_combinations['layout_x']
        layout_d_cmp_kv = param_combinations['layout_d_cmp_kv']
        layout_softmax_score = param_combinations['layout_softmax_score']
        layout_kv = param_combinations['layout_kv']
        layout_d_x = param_combinations['layout_d_x']
        data_type = param_combinations['data_type']
        cu_seqlens = param_combinations['cu_seqlens']
        seqused = param_combinations['seqused']
        start_pos = param_combinations['start_pos']
        x_datarange = param_combinations['x_datarange']
        wkv_datarange = param_combinations['wkv_datarange']
        wgate_datarange = param_combinations['wgate_datarange']
        d_cmp_kv_datarange = param_combinations['d_cmp_kv_datarange']
        softmax_score_datarange = param_combinations['softmax_score_datarange']
        kv_datarange = param_combinations['kv_datarange']

        test_data = batch_size, hidden_size, Seq_len, head_dim, block_size, cmp_ratio, coff, \
                    layout_x, layout_d_cmp_kv, layout_softmax_score, layout_kv, layout_d_x, data_type, cu_seqlens, seqused, start_pos, \
                    x_datarange, wkv_datarange,  wgate_datarange, d_cmp_kv_datarange, softmax_score_datarange, kv_datarange

        torch_npu.npu.set_device(0)

        # 获得cpu结果(真值)和算子结果（测试值）
        compressor_grad_operator_single.output_operator(test_data)