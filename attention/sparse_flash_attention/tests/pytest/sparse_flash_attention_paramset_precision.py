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

import math
import torch

TEST_PARAMS = {
    "bsnd_norope": {
        "Testcase_Prefix": ["sfa_bsnd_norope"],
        "layout_query": ["BSND"],
        "layout_kv": ["BSND"],
        "q_type": [torch.float16],
        "kv_type": [torch.float16],
        "B": [1],
        "S1": [4],
        "S2": [128],
        "N1": [8],
        "N2": [1],
        "D": [512],
        "K": [16],
        "scale_value": [1.0 / math.sqrt(512)],
        "sparse_block_size": [1],
        "rope_head_dim": [0],
        "sparse_mode": [0],
        "attention_mode": [2],
        "return_softmax_lse": [True],
        "use_sinks": [False],
        "actual_seq_q": [[4]],
        "actual_seq_kv": [[128]],
    },
    "bsnd_with_rope": {
        "Testcase_Prefix": ["sfa_bsnd_with_rope"],
        "layout_query": ["BSND"],
        "layout_kv": ["BSND"],
        "q_type": [torch.float16],
        "kv_type": [torch.float16],
        "B": [1],
        "S1": [4],
        "S2": [128],
        "N1": [8],
        "N2": [1],
        "D": [512],
        "K": [16],
        "scale_value": [1.0 / math.sqrt(576)],
        "sparse_block_size": [1],
        "rope_head_dim": [64],
        "sparse_mode": [0],
        "attention_mode": [2],
        "return_softmax_lse": [True],
        "use_sinks": [False],
        "actual_seq_q": [[4]],
        "actual_seq_kv": [[128]],
    },
}

ENABLED_PARAMS = [
    TEST_PARAMS["bsnd_norope"],
    TEST_PARAMS["bsnd_with_rope"],
]
