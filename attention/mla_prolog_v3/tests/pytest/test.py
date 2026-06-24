#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# -----------------------------------------------------------------------------------------------------------

import itertools
import torch
import torch_npu
from testcases import ENABLED_PARAMS
import check_valid_param
import prologv3_golden
import pytest

param_names = [
    "batch_size", "He", "Hcq", "Hckv", "q_head_num",
    "kv_head_num", "head_dim", "rope_head_dim", "q_seq", "kv_seq",
    "block_size", "input_layout", "cache_mode", "cq_epsilon", "ckv_epsilon", "dtype",
    "weight_quant_mode", "kv_cache_quant_mode", "query_quant_mode",
    "ckvkr_repo_mode", "quant_scale_repo_mode", "tile_size",
    "qc_qr_scale", "kc_scale", "query_norm_flag", "bs_merged",
]

all_param_combinations = []
for params in ENABLED_PARAMS:
    param_values = [params[n] for n in param_names]
    for combo in itertools.product(*param_values):
        all_param_combinations.append(dict(zip(param_names, combo)))


@pytest.mark.ci
@pytest.mark.parametrize("param_combinations", all_param_combinations)
def test_mla_prolog_v3(param_combinations):
    torch_npu.npu.set_device(0)

    test_data = tuple(param_combinations.values())
    try:
        check_valid_param.validate_config(test_data)
    except ValueError as e:
        pytest.skip("param validation failed: " + str(e))

    wq = param_combinations["weight_quant_mode"]
    kvq = param_combinations["kv_cache_quant_mode"]

    if wq == 5 and not hasattr(torch, 'hifloat8'):
        pytest.skip("hifloat8 not supported in this torch version")
    if wq in (3, 4, 5):
        pytest.skip("mxfp8/fp8/hif8 not supported on A3 (ascend910_93)")

    key = (wq, kvq)
    if key not in prologv3_golden.TEST_FUNCTIONS:
        pytest.skip("golden not implemented for wq=%d kvq=%d" % key)

    expect, result = prologv3_golden.TEST_FUNCTIONS[key](test_data)
    check_valid_param.check_result(expect, result)
