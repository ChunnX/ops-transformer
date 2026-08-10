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

import importlib.util
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

import check_valid_param
import quant_block_sparse_attn_golden


class _BatchCountRng:
    def __init__(self, values):
        self._values = iter(values)
        self.randint_calls = []

    def randint(self, low, high):
        self.randint_calls.append((low, high))
        value = next(self._values)
        assert low <= value <= high
        return value


def _sparse_case(sparse_mode=None):
    case = {
        "B": 2,
        "N1": 1,
        "N2": 1,
        "S1": 128,
        "S2": 384,
        "D": 128,
        "sparse_q_block_size": 128,
        "sparse_kv_block_size": 128,
        "mask_mode": 0,
        "sparse_pattern": "sequential",
    }
    if sparse_mode is not None:
        case["sparse_mode"] = sparse_mode
    return case


def test_sparse_mode_full_uses_each_batch_actual_kv_length():
    class _NoRandomRng:
        def randint(self, low, high):
            raise AssertionError("full mode must not draw a random block count")

    _, sparse_seq_len = quant_block_sparse_attn_golden._make_sparse_indices(
        _sparse_case(sparse_mode="full"),
        q_lengths=[128, 128],
        kv_lengths=[128, 300],
        rng=_NoRandomRng(),
    )

    assert sparse_seq_len[:, 0, 0].tolist() == [1, 3]


def test_batch_csv_loader_reads_sparse_mode():
    module_path = Path(__file__).parent / "batch" / "quant_block_sparse_attn_paramset_batch.py"
    spec = importlib.util.spec_from_file_location("_qbsa_batch_paramset_loader_test", module_path)
    module = importlib.util.module_from_spec(spec)
    csv_text = (
        "case_name,enable,B,N1,N2,S1,S2,D,sparse_mode\n"
        "loader_case,TRUE,2,8,1,128,256,128,random\n"
        "loader_default_case,TRUE,2,8,1,128,256,128,\n"
    )

    with patch("builtins.open", mock_open(read_data=csv_text)):
        spec.loader.exec_module(module)

    params = module.TEST_PARAMS["loader_case"]
    assert params["sparse_mode"] == ["random"]
    default_params = module.TEST_PARAMS["loader_default_case"]
    assert default_params["sparse_mode"] == ["random"]


def test_sparse_mode_random_draws_once_per_batch_with_actual_kv_upper_bound():
    rng = _BatchCountRng([0, 3])

    sparse_indices, sparse_seq_len = quant_block_sparse_attn_golden._make_sparse_indices(
        _sparse_case(sparse_mode="random"),
        q_lengths=[128, 128],
        kv_lengths=[128, 300],
        rng=rng,
    )

    assert rng.randint_calls == [(0, 1), (0, 3)]
    assert sparse_seq_len[:, 0, 0].tolist() == [0, 3]
    assert sparse_indices[0, 0, 0].tolist() == [-1, -1, -1]
    assert sparse_indices[1, 0, 0].tolist() == [0, 1, 2]


def test_sparse_mode_rejects_unknown_value():
    with pytest.raises(ValueError, match="unsupported sparse_mode"):
        quant_block_sparse_attn_golden._make_sparse_indices(
            _sparse_case(sparse_mode="unknown"),
            q_lengths=[128, 128],
            kv_lengths=[128, 300],
            rng=_BatchCountRng([]),
        )


def test_sparse_mode_validation_accepts_random():
    params = {
        "B": 1,
        "S1": 128,
        "S2": 128,
        "N1": 1,
        "N2": 1,
        "D": 128,
        "layout_q": "TND",
        "sparse_mode": "random",
    }

    check_valid_param.check_valid_param(params)


def test_sparse_mode_validation_defaults_none_to_random():
    params = {
        "B": 1,
        "S1": 128,
        "S2": 128,
        "N1": 1,
        "N2": 1,
        "D": 128,
        "layout_q": "TND",
        "sparse_mode": None,
    }

    check_valid_param.check_valid_param(params)


def test_sparse_mode_validation_rejects_unknown_value():
    params = {
        "B": 1,
        "S1": 128,
        "S2": 128,
        "N1": 1,
        "N2": 1,
        "D": 128,
        "layout_q": "TND",
        "sparse_mode": "unknown",
    }

    with pytest.raises(ValueError, match="unsupported sparse_mode"):
        check_valid_param.check_valid_param(params)


def test_normalize_params_defaults_sparse_mode_to_random():
    missing_mode = quant_block_sparse_attn_golden._normalize_params(_sparse_case())
    explicit_none_case = _sparse_case()
    explicit_none_case["sparse_mode"] = None
    none_mode = quant_block_sparse_attn_golden._normalize_params(explicit_none_case)

    assert missing_mode["sparse_mode"] == "random"
    assert none_mode["sparse_mode"] == "random"
