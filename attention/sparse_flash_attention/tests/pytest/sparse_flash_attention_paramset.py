#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# -----------------------------------------------------------------------------------------------------------

import torch


TEST_PARAMS = {
    "runpy_style": {
        "Testcase_Prefix": ["sfa_bsmd_smoke"],
        "Testcase_Number": [1],
        "layout_query": ["BSND"],
        "layout_kv": ["BSND"],
        "q_type": [torch.float16],
        "B": [1],
        "S1": [1],
        "S2": [128],
        "N1": [8],
        "N2": [1],
        "D": [512],
        "K": [32],
        "scale_value": [0.041666666666666664],
        "sparse_block_size": [1],
        "rope_head_dim": [64],
        "sparse_mode": [3],
        "attention_mode": [2],
        "return_softmax_lse": [False],
        "block_size": [256],
        "actual_seq_q": [[1]],
        "actual_seq_kv": [[128]],
    },
}


ENABLED_PARAMS = [
    TEST_PARAMS["runpy_style"],
]
