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

"""White-box regression tests for the MQSMLA batch assets."""

import importlib.util
from pathlib import Path

import numpy as np
import torch


def load_asset_module(stem):
    path = Path(__file__).with_name("impl") / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(f"mqsmla_batch_test_{stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMPARE_MODULE = load_asset_module("compare")
INPUTS_MODULE = load_asset_module("inputs")


def batch_kwargs(slices, seed):
    return {
        "batch_consistency_id": ((tuple(
            f"{seed}_0_{start}_{stop}_{step}" for start, stop, step in slices
        ),),),
        "batch_axis": ((0,),),
        "batch_slice_info": ((tuple(slices),),),
        "batch_seed": ((tuple(seed for _ in slices),),),
    }


def test_same_case_batch_compare_accepts_equal_slices():
    output = np.array([[1.0, 2.0], [1.0, 2.0]], dtype=np.float32)

    result = COMPARE_MODULE.compare(
        output, output.copy(), **batch_kwargs(((0, 1, 1), (1, 2, 1)), 7001)
    )

    assert result[-1] == {"pass": True, "precision": "batch_intra=PASS"}


def test_same_case_batch_compare_rejects_different_slices():
    output = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    result = COMPARE_MODULE.compare(
        output, output.copy(), **batch_kwargs(((0, 1, 1), (1, 2, 1)), 7001)
    )

    assert result[-1]["pass"] is False
    assert result[-1]["precision"] == "batch_intra=FAIL"


def test_same_case_batch_compare_rejects_different_storage_bits():
    output = np.array([[0.0], [-0.0]], dtype=np.float32)

    result = COMPARE_MODULE.compare(
        output, output.copy(), **batch_kwargs(((0, 1, 1), (1, 2, 1)), 7001)
    )

    assert result[-1]["pass"] is False
    assert result[-1]["precision"] == "batch_intra=FAIL"


def test_batch_random_context_repeats_declared_batch_slices():
    q = torch.empty((3, 1, 1, 1), dtype=torch.float32)
    fields = batch_kwargs(((0, 1, 1), (2, 3, 1)), 7001)
    fields.update({"layout_q": "BSND", "layout_kv": "BSND"})
    original_uniform = np.random.uniform

    with INPUTS_MODULE.NumpyBatchRandomContext.from_case(q, fields):
        value = np.random.uniform(-1.0, 1.0, (3, 2))

    assert np.random.uniform is original_uniform
    assert np.array_equal(value[0], value[2])


def test_non_batch_case_does_not_invent_pytest_geometry():
    q = torch.empty((1, 1, 64, 512), dtype=torch.bfloat16)
    ori_kv = torch.empty((1, 8, 1, 608), dtype=torch.float8_e4m3fn)
    cmp_kv = torch.empty((1, 2, 1, 608), dtype=torch.float8_e4m3fn)

    params = INPUTS_MODULE.INPUT_ADAPTER.build_case_params(
        q,
        ori_kv,
        cmp_kv,
        None,
        None,
        "BSND",
        "BSND",
        {},
    )

    assert params["block_size1"] is None
    assert params["block_size2"] is None
    assert "tile_size" not in params


def test_batch_input_adapter_generates_equal_full_case_slices():
    batch_size = 2
    q = torch.empty((batch_size, 1, 64, 512), dtype=torch.bfloat16)
    ori_kv = torch.empty((batch_size, 8, 1, 608), dtype=torch.float8_e4m3fn)
    cmp_kv = torch.empty((batch_size, 2, 1, 608), dtype=torch.float8_e4m3fn)
    seqused_q = torch.empty((batch_size,), dtype=torch.int32)
    seqused_ori_kv = torch.empty((batch_size,), dtype=torch.int32)
    seqused_cmp_kv = torch.empty((batch_size,), dtype=torch.int32)
    cmp_residual_kv = torch.empty((batch_size,), dtype=torch.int32)
    sinks = torch.empty((64,), dtype=torch.float32)
    testcase_name = "MQSMLA_BATCH_FULL_INPUT_001"

    INPUTS_MODULE.generate_mixed_quant_sparse_flash_mla_inputs(
        q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        seqused_q=seqused_q,
        seqused_ori_kv=seqused_ori_kv,
        seqused_cmp_kv=seqused_cmp_kv,
        cmp_residual_kv=cmp_residual_kv,
        sinks=sinks,
        testcase_name=testcase_name,
        layout_q="BSND",
        layout_kv="BSND",
        softmax_scale=0.04419417,
        cmp_ratio=4,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=127,
        ori_win_right=0,
        quant_mode=1,
        rope_head_dim=64,
        return_softmax_lse=False,
        has_ori_kv=True,
        has_cmp_kv=True,
        seqused_q_values=[1, 1],
        seqused_ori_kv_values=[8, 8],
        seqused_cmp_kv_values=[2, 2],
        cmp_residual_kv_values=[0, 0],
        input_ranges=((-10, 10), (-5, 5), (-5, 5)),
        batch_axis=((0,), (0,), (0,), None, None, None, None, None, None, None,
                    (0,), (0,), (0,), (0,), None, None, None),
        batch_slice_info=((((0, 1, 1), (1, 2, 1)),), None, None, None, None, None,
                          None, None, None, None, None, None, None, None, None, None, None),
        batch_seed=(((74123, 74123),), None, None, None, None, None, None, None,
                    None, None, None, None, None, None, None, None, None),
    )

    def storage_equal(first, second):
        return torch.equal(
            first.detach().cpu().contiguous().view(torch.uint8),
            second.detach().cpu().contiguous().view(torch.uint8),
        )

    assert storage_equal(q[0], q[1])
    assert storage_equal(ori_kv[0], ori_kv[1])
    assert storage_equal(cmp_kv[0], cmp_kv[1])
    assert seqused_q.tolist() == [1, 1]
    assert seqused_ori_kv.tolist() == [8, 8]
    assert seqused_cmp_kv.tolist() == [2, 2]
    assert cmp_residual_kv.tolist() == [0, 0]
    data = INPUTS_MODULE.INPUT_ADAPTER.load_golden_store().CASE_DATA.get(testcase_name)
    assert data is not None
    assert data["cpu_output"] is not None
