#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import torch

TestCases = {
    "DM_Basic_FP16": {
        "B": [2], "N1": [8], "N2": [2], "S1": [128], "S2": [128], "D": [128],
        "layout_q": ["BSND"], "layout_kv": ["BSND"], "layout_out": ["BSND"],
        "Dtype": ["fp16"], "mask_mode": [0],
        "keep_prob": [0.9], "seed": [2], "offset": [0],
    },
    "DM_Basic_BF16": {
        "B": [2], "N1": [8], "N2": [2], "S1": [128], "S2": [128], "D": [128],
        "layout_q": ["BSND"], "layout_kv": ["BSND"], "layout_out": ["BSND"],
        "Dtype": ["bf16"], "mask_mode": [0],
        "keep_prob": [0.9], "seed": [2], "offset": [0],
    },
    "DM_Causal_Mask": {
        "B": [2], "N1": [8], "N2": [2], "S1": [128], "S2": [128], "D": [128],
        "layout_q": ["BNSD"], "layout_kv": ["BNSD"], "layout_out": ["BNSD"],
        "Dtype": ["fp16"], "mask_mode": [3],
        "keep_prob": [0.9], "seed": [2], "offset": [0],
    },
    "DM_Band_Mask": {
        "B": [2], "N1": [8], "N2": [2], "S1": [128], "S2": [128], "D": [128],
        "layout_q": ["BNSD"], "layout_kv": ["BNSD"], "layout_out": ["BNSD"],
        "Dtype": ["fp16"], "mask_mode": [4], "win_left": [32], "win_right": [32],
        "keep_prob": [0.9], "seed": [2], "offset": [0],
    },
    "DM_Large_S2": {
        "B": [1], "N1": [8], "N2": [2], "S1": [128], "S2": [512], "D": [128],
        "layout_q": ["BNSD"], "layout_kv": ["BNSD"], "layout_out": ["BNSD"],
        "Dtype": ["fp16"], "mask_mode": [0],
        "keep_prob": [0.9], "seed": [2], "offset": [0],
    },
    "DM_KeepProb_05": {
        "B": [2], "N1": [8], "N2": [2], "S1": [128], "S2": [128], "D": [128],
        "layout_q": ["BSND"], "layout_kv": ["BSND"], "layout_out": ["BSND"],
        "Dtype": ["fp16"], "mask_mode": [0],
        "keep_prob": [0.5], "seed": [6], "offset": [24],
    },
    "DM_Seed_Large": {
        "B": [2], "N1": [8], "N2": [2], "S1": [128], "S2": [128], "D": [128],
        "layout_q": ["BSND"], "layout_kv": ["BSND"], "layout_out": ["BSND"],
        "Dtype": ["fp16"], "mask_mode": [0],
        "keep_prob": [0.9], "seed": [4294967296], "offset": [0],
    },
    "DM_Offset_NonAligned": {
        "B": [2], "N1": [8], "N2": [2], "S1": [128], "S2": [128], "D": [128],
        "layout_q": ["BSND"], "layout_kv": ["BSND"], "layout_out": ["BSND"],
        "Dtype": ["fp16"], "mask_mode": [0],
        "keep_prob": [0.9], "seed": [2], "offset": [17],
    },
    "DM_Multi_Batch": {
        "B": [4], "N1": [8], "N2": [2], "S1": [128], "S2": [128], "D": [128],
        "layout_q": ["BSND"], "layout_kv": ["BSND"], "layout_out": ["BSND"],
        "Dtype": ["fp16"], "mask_mode": [0],
        "keep_prob": [0.9], "seed": [2], "offset": [0],
    },
    "DM_KeepProb_10_Regression": {
        "B": [2], "N1": [8], "N2": [2], "S1": [128], "S2": [128], "D": [128],
        "layout_q": ["BSND"], "layout_kv": ["BSND"], "layout_out": ["BSND"],
        "Dtype": ["fp16"], "mask_mode": [0],
        "keep_prob": [1.0], "seed": [0], "offset": [0],
    },
}
