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

# 当前由于torchair拦截的原因，kv_type为float8_e4m3fn时只能传None
TEST_PARAMS = {
    "TND_PA": {
        "Testcase_Prefix": ["runpy_style"],
        "layout_query": ["BSND"],
        "layout_kv": ["PA_BSND"],
        "q_type": [torch.float16],
        "kv_type": ["int8"],
        "B": [1],
        "T1": [None],
        "S1": [4096],
        "S2": [4096],
        "N1": [48],
        "N2": [1],
        "D": [512],
        "K": [256],
        "scale_value": [0.041666666666666664],
        "key_quant_mode": [2],
        "value_quant_mode": [2],
        "sparse_block_size": [1],
        "tile_size": [128],
        "rope_head_dim": [64],
        "sparse_mode": [3],
        "attention_mode": [2],
        "quant_scale_repo_mode": [1],
        "block_size": [512],
        "actual_seq_q": [
            [
                4096,
            ]
        ],
        "actual_seq_kv": [
            [
                4096,
            ]
        ],
        "return_softmax_lse": [False],
        "enable_sinks": [True],
    }
}

ENABLED_PARAMS = [TEST_PARAMS[key] for key in TEST_PARAMS.keys()]
