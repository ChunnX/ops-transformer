#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Auto-generated MLA D512 non-quant test cases."""

BASE = {
    "B": [1],
    "N1": [2],
    "N2": [1],
    "S1": [1],
    "S2": [512],
    "D": [512],
    "DV": [512],
    "DRope": [64],
    "Dtype": ["fp16"],
    "layout_q": ["BNSD"],
    "layout_kv": ["BNSD"],
    "layout_out": ["BNSD"],
    "block_size": [128],
    "return_softmax_lse": [False],
    "sparse_mode": [0],
    "q_range": [(-5.0, 5.0)],
    "k_range": [(-5.0, 5.0)],
    "v_range": [(-5.0, 5.0)],
    "kv_reused": [True],
}

TestCases = {
    "case_001": {**BASE, "B": [3], "N1": [8], "S1": [4], "S2": [384], "layout_kv": ["PA_BBH"], "act_kv": [[384, 384, 384]], "variance": ["extreme"]},
    "case_002": {**BASE, "B": [3], "N1": [8], "S1": [4], "S2": [256], "layout_kv": ["PA_BBH"], "act_kv": [[256, 256, 256]], "variance": ["extreme"]},
    "case_003": {**BASE, "B": [3], "N1": [8], "S1": [4], "S2": [2048], "layout_kv": ["PA_BBH"], "act_kv": [[2048, 2048, 2048]], "variance": ["extreme"]},
    "case_004": {**BASE, "B": [3], "N1": [8], "S1": [4], "S2": [409600], "layout_kv": ["PA_BBH"], "act_kv": [[409600, 409600, 409600]], "variance": ["extreme"]},
}
