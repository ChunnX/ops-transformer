#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import pytest
import torch
import torch_npu

import npu_ops_transformer_ext


TEST_CASES = [
    {"counts": [1], "hidden": 64},
    {"counts": [3, 0, 4, 2], "hidden": 64},
    {"counts": [2, 5, 0, 9], "hidden": 128},
    {"counts": [0, 8, 7, 3, 6], "hidden": 256},
]


def _make_inputs(counts, hidden):
    token_rows = sum(counts)
    expert_num = len(counts)
    seed = token_rows * 1009 + hidden * 17 + expert_num
    torch.manual_seed(seed)

    x = torch.randn(token_rows, hidden, dtype=torch.float32)
    x = torch.clamp(x, min=-6.0, max=6.0).to(torch.bfloat16)

    quant_scale = torch.rand(expert_num, hidden, dtype=torch.float32) * 1.75 + 0.125
    local_rows = torch.cumsum(torch.tensor(counts, dtype=torch.int64), dim=0)
    return x, local_rows, quant_scale


def _reference(x, local_rows, quant_scale):
    out = torch.empty((x.shape[0], x.shape[1]), dtype=torch.int8)
    start = 0
    for expert_idx, end in enumerate(local_rows.tolist()):
        if end > start:
            scaled = x[start:end].float() * quant_scale[expert_idx].float()
            scaled = torch.clamp(scaled, min=-128.0, max=127.0)
            scaled = scaled.to(torch.float16)
            out[start:end] = torch.round(scaled).to(torch.int8)
        start = end
    return out


def _run_case(counts, hidden):
    x_cpu, local_rows_cpu, quant_scale_cpu = _make_inputs(counts, hidden)
    result = torch.ops.npu_ops_transformer_ext.moe_quant(
        x_cpu.npu(),
        local_rows_cpu.npu(),
        quant_scale_cpu.npu(),
    ).cpu()
    expected = _reference(x_cpu, local_rows_cpu, quant_scale_cpu)

    diff = (result.to(torch.int16) - expected.to(torch.int16)).abs()
    max_abs = diff.max().item() if diff.numel() else 0
    mismatch_count = torch.count_nonzero(result != expected).item()
    print(
        "[precision] token_rows={} hidden={} expert_num={} max_abs={} mismatches={}".format(
            x_cpu.shape[0], hidden, len(counts), max_abs, mismatch_count
        )
    )

    assert result.shape == expected.shape
    assert result.dtype == torch.int8
    assert torch.equal(result, expected)


def test_moe_quant_interface_exist():
    print(torch.ops.npu_ops_transformer_ext.moe_quant)
    assert hasattr(torch.ops.npu_ops_transformer_ext, "moe_quant")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("case", TEST_CASES)
def test_moe_quant_precision(case):
    _run_case(case["counts"], case["hidden"])
