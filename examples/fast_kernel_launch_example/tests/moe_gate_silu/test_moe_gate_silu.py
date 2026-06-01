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
import torch.nn.functional as F
import torch_npu

import ascend_ops


DTYPES = [torch.float16, torch.bfloat16]
SHAPE_CASES = [
    (1, 16),
    (8, 32),
    (31, 64),
    (128, 1024),
    (256, 4096),
]
TOLERANCES = {
    torch.float16: (5e-3, 5e-3),
    torch.bfloat16: (2e-2, 2e-2),
}


def _make_input(rows, hidden, dtype):
    torch.manual_seed(rows * 1009 + hidden)
    x = torch.randn(rows, hidden * 2, dtype=torch.float32)
    x = torch.clamp(x, min=-5.0, max=5.0)
    return x.to(dtype)


def _reference(x, active_rows):
    hidden = x.shape[1] // 2
    gate = x[:active_rows, :hidden].float()
    up = x[:active_rows, hidden:].float()
    return (F.silu(gate) * up).to(x.dtype)


def _max_error(actual, expected):
    actual_fp32 = actual.float()
    expected_fp32 = expected.float()
    diff = torch.abs(actual_fp32 - expected_fp32)
    max_abs = diff.max().item() if diff.numel() else 0.0
    denom = torch.clamp(torch.abs(expected_fp32), min=1e-6)
    max_rel = (diff / denom).max().item() if diff.numel() else 0.0
    return max_abs, max_rel


def _run_precision_case(rows, hidden, dtype, active_rows):
    x_cpu = _make_input(rows, hidden, dtype)
    total_rows_npu = torch.tensor([active_rows], dtype=torch.int64).npu()
    result_npu = torch.ops.ascend_ops.moe_gate_silu(x_cpu.npu(), total_rows_npu)
    result = result_npu.cpu()
    expected = _reference(x_cpu, active_rows)
    actual = result[:active_rows]
    rtol, atol = TOLERANCES[dtype]
    max_abs, max_rel = _max_error(actual, expected)
    print(
        "[precision] rows={} hidden={} active_rows={} dtype={} max_abs={:.6g} "
        "max_rel={:.6g} rtol={} atol={}".format(
            rows, hidden, active_rows, dtype, max_abs, max_rel, rtol, atol
        )
    )
    assert result.shape == (rows, hidden)
    assert result.dtype == dtype
    assert torch.allclose(actual.float(), expected.float(), rtol=rtol, atol=atol)


def test_moe_gate_silu_interface_exist():
    print(torch.ops.ascend_ops.moe_gate_silu)
    assert hasattr(torch.ops.ascend_ops, "moe_gate_silu")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", SHAPE_CASES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_moe_gate_silu_precision(shape, dtype):
    rows, hidden = shape
    _run_precision_case(rows, hidden, dtype, rows)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_moe_gate_silu_active_rows_prefix(dtype):
    _run_precision_case(rows=16, hidden=64, dtype=dtype, active_rows=7)
