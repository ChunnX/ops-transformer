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


TEST_PARAMS = {
    "sfaanq_fuzz_pa_rope_bf16": {
        "Testcase_Prefix": ["sFAAnQ_fuzz_pa_rope"],
        "Testcase_Number": [230],
        "layout_query": ["BSND"],
        "layout_kv": ["PA_BSND"],
        "q_type": [torch.bfloat16],
        "kv_dtype": ["hifloat8"],
        "B": [16],
        "S1": [256],
        "S2": [256],
        "N1": [16],
        "N2": [1],
        "D": [512],
        "K": [215],
        "scale_value": [0.04166666666666666],
        "key_quant_mode": [2],
        "value_quant_mode": [2],
        "sparse_block_size": [1],
        "tile_size": [128],
        "rope_head_dim": [64],
        "sparse_mode": [3],
        "attention_mode": [2],
        "quant_scale_repo_mode": [1],
        "block_size": [256],
        "block_num": [16],
        "actual_seq_q": [[211, 15, 212, 94, 114, 135, 237, 169, 131, 138, 221, 155, 235, 32, 102, 26]],
        "actual_seq_kv": [[224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 256]],
    },
}

ENABLED_PARAMS = [TEST_PARAMS["sfaanq_fuzz_pa_rope_bf16"]]
