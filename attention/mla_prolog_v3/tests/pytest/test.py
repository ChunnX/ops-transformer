#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import itertools
import torch
import torch_npu
from testcases import ENABLED_PARAMS
import check_valid_param
import prologv3_golden
import pytest

for _, params in enumerate(ENABLED_PARAMS):
    for key, value in params.items():
        locals()["param_" + key] = value

    param_names = [
        "batch_size", "He", "Hcq", "Hckv", "q_head_num",
        "kv_head_num", "head_dim", "rope_head_dim", "q_seq", "kv_seq",
        "block_size", "input_layout", "cache_mode", "cq_epsilon", "ckv_epsilon", "dtype",
        "weight_quant_mode", "kv_cache_quant_mode", "query_quant_mode",
        "ckvkr_repo_mode", "quant_scale_repo_mode", "tile_size",
        "qc_qr_scale", "kc_scale", "query_norm_flag",
    ]

    param_values = [
        locals()["param_" + _name] for _name in param_names
    ]

    locals()["param_combinations"] = []
    for combo in itertools.product(*param_values):
        param_dict = dict(zip(param_names, combo))
        locals()["param_combinations"].append(param_dict)

    @pytest.mark.ci
    @pytest.mark.parametrize("param_combinations", locals()["param_combinations"])
    def test_mla_prolog_v3(param_combinations):
        batch_size = param_combinations['batch_size']
        He = param_combinations['He']
        Hcq = param_combinations['Hcq']
        Hckv = param_combinations['Hckv']
        q_head_num = param_combinations['q_head_num']
        kv_head_num = param_combinations['kv_head_num']
        head_dim = param_combinations['head_dim']
        rope_head_dim = param_combinations['rope_head_dim']
        q_seq = param_combinations['q_seq']
        kv_seq = param_combinations['kv_seq']
        block_size = param_combinations['block_size']
        input_layout = param_combinations['input_layout']
        cache_mode = param_combinations['cache_mode']
        cq_epsilon = param_combinations['cq_epsilon']
        ckv_epsilon = param_combinations['ckv_epsilon']
        dtype = param_combinations['dtype']
        weight_quant_mode = param_combinations['weight_quant_mode']
        kv_cache_quant_mode = param_combinations['kv_cache_quant_mode']
        query_quant_mode = param_combinations['query_quant_mode']
        ckvkr_repo_mode = param_combinations['ckvkr_repo_mode']
        quant_scale_repo_mode = param_combinations['quant_scale_repo_mode']
        tile_size = param_combinations['tile_size']
        qc_qr_scale = param_combinations['qc_qr_scale']
        kc_scale = param_combinations['kc_scale']
        query_norm_flag = param_combinations['query_norm_flag']

        torch_npu.npu.set_device(0)

        test_data = (
            batch_size, He, Hcq, Hckv, q_head_num, kv_head_num, head_dim, rope_head_dim,
            q_seq, kv_seq, block_size, input_layout, cache_mode, cq_epsilon, ckv_epsilon, dtype,
            weight_quant_mode, kv_cache_quant_mode, query_quant_mode,
            ckvkr_repo_mode, quant_scale_repo_mode, tile_size,
            qc_qr_scale, kc_scale, query_norm_flag,
        )
        try:
            check_valid_param.validate_config(test_data)
        except ValueError as e:
            pytest.skip("param validation failed: " + str(e))

        if input_layout != "BSH":
            pytest.skip("only BSH layout is supported in golden currently")

        key = (weight_quant_mode, kv_cache_quant_mode)
        if key not in prologv3_golden.TEST_FUNCTIONS:
            pytest.skip("golden not implemented for wq=%d kvq=%d" % key)

        test_fn = prologv3_golden.TEST_FUNCTIONS[key]
        expect, result = test_fn(test_data)
        check_valid_param.check_result(expect, result)
