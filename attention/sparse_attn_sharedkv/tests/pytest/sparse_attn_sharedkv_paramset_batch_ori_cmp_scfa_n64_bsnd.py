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

import torch

# 定义测试参数组合
TEST_PARAMS = {
    "ori_cmp_scfa_g64_test_prefill_BSND_batch":{
        "layout_q": ["BSND"],
        "layout_kv": ["BSND"],
        "q_type": [torch.bfloat16],
        "ori_kv_type": [torch.bfloat16],
        "cmp_kv_type": [torch.bfloat16],
        "B": [1,5],
        "S1": [1, 16, 33, 128, 159, 166],
        "S2": [1, 16, 33, 128, 159, 166],
        "N1": [64],
        "N2": [1],
        "D": [512],
        "K1": [1, 2, 64, 129, 512],
        "K": [512],
        "softmax_scale": [0.04419417],
        "cmp_ratio": [1],
        "ori_mask_mode": [0],
        "cmp_mask_mode": [0],
        "ori_win_left": [127],
        "ori_win_right": [0],
        "ori_kv_topk_mode": ["no","full","fullK","random"],
        "cmp_kv_topk_mode": ["no","full","fullK","random"],
        "ori_sparse_indices_mode": ["full","random"],
        "cmp_sparse_indices_mode": ["full","random"]
    },
}

# 按需选择要启用的测试参数（例如默认启用所有）
ENABLED_PARAMS = [TEST_PARAMS[key] for key in TEST_PARAMS.keys()]