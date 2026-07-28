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
    "case_001": {**BASE, "B": [4], "N1": [8], "S1": [8], "S2": [24460], "layout_kv": ["PA_BBH"], "act_kv": [[13782,18761,17950,23938]], "variance": ["extreme"]},
    # "case_002": {**BASE, "B": [15], "N1": [128], "S1": [8], "S2": [131402], "layout_kv": ["BNSD"], "act_kv": [[131402,131402,131402,131402,131402,131402,131402,131402,131402,131402,131402,131402,131402,131402,131402]], "variance": ["uniform"]},
    # "case_003": {**BASE, "B": [4], "N1": [128], "S1": [15], "S2": [133080], "layout_kv": ["PA_NZ"], "act_kv": [[0,133080,133080,133080]], "variance": ["with_zero"]},
    # "case_004": {**BASE, "B": [13], "N1": [128], "S1": [3], "S2": [55076], "layout_kv": ["PA_NZ"], "act_kv": [[55076,55076,55076,55076,55076,55076,55076,55076,55076,55076,55076,55076,55076]], "variance": ["uniform"]},
    # "case_005": {**BASE, "B": [15], "N1": [8], "S1": [5], "S2": [53018], "layout_kv": ["PA_NZ"], "act_kv": [[53018,53018,53018,53018,53018,53018,53018,53018,53018,53018,53018,53018,53018,53018,53018]], "variance": ["uniform"]},
}
