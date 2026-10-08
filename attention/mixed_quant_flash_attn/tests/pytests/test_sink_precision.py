# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent precision/FD contracts, without requiring an NPU."""

import pytest
import torch

from core.cpu import run_cpu_golden, softmax_flash
from core.gen_data import generate_inputs


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("with_sink", [False, True])
def test_cpu_pv_uses_native_weights_and_fp32_sum(dtype, with_sink, monkeypatch):
    params = {
        "B": 1,
        "N1": 2,
        "N2": 1,
        "S1": 2,
        "S2": 5,
        "D": 128,
        "layout_q": "BSND",
        "layout_kv": "PA_BNBD",
        "block_size": 64,
        "quant_compute_mode": 1,
        "q_dtype": "fp16" if dtype == torch.float16 else "bf16",
        "sinks": torch.tensor([2.0, -2.0]) if with_sink else None,
        "return_softmax_lse": True,
    }
    inputs = generate_inputs(params, seed=21)
    original = torch.matmul
    operands = []

    def matmul(a, b):
        operands.append((a.clone(), b.clone()))
        return original(a, b)

    monkeypatch.setattr(torch, "matmul", matmul)
    result = run_cpu_golden(inputs)
    q, kt = operands[0]
    scores = original(q, kt) * inputs["softmax_scale"]
    sink = inputs["sinks"]
    maximum = scores.amax(-1, keepdim=True)
    if sink is not None:
        sink = sink.repeat(2).reshape(-1, 1)
        maximum = torch.maximum(maximum, sink)
    weights = torch.exp(scores - maximum)
    total = weights.sum(-1, keepdim=True)
    if sink is not None:
        total += torch.exp(sink - maximum)
    pv_weights, values = operands[1]
    torch.testing.assert_close(pv_weights, weights.to(dtype).float(), rtol=0, atol=0)
    expected = (original(weights.to(dtype).float(), values) / total).to(dtype)
    torch.testing.assert_close(
        result["attn_out"].reshape(4, 128), expected, rtol=0, atol=0
    )
    torch.testing.assert_close(
        result["softmax_lse"].transpose(1, 2).reshape(4, 1), maximum + total.log()
    )


@pytest.mark.parametrize("sink", [-1000.0, 0.0, 400.0, 1000.0])
@pytest.mark.parametrize("split_count", [1, 2, 5])
def test_fd_merge_sink_once_matches_joint_softmax(sink, split_count):
    """FD consumes normalized partials with max/sum, not an extra sink."""
    torch.manual_seed(21)
    scores = torch.randn(3, 20) * 100 + 300
    values = torch.randn(20, 8)
    outputs, totals, maxima = [], [], []
    for idx, tile in enumerate(torch.chunk(scores, split_count, dim=1)):
        rows = tile.shape[0]
        weights, total, maximum, _ = softmax_flash(
            tile,
            torch.zeros(rows, 1),
            torch.full((rows, 1), -torch.inf),
            torch.ones(rows, 1),
            True,
            torch.full((rows, 1), sink) if idx == 0 else None,
        )
        start = sum(
            part.shape[1] for part in torch.chunk(scores, split_count, dim=1)[:idx]
        )
        outputs.append(weights @ values[start : start + tile.shape[1]] / total)
        totals.append(total)
        maxima.append(maximum)
    maximum = torch.stack(maxima).amax(0)
    masses = torch.stack(
        [total * torch.exp(m - maximum) for total, m in zip(totals, maxima)]
    )
    combined = (
        torch.stack(outputs) * (masses / masses.sum(0)).unsqueeze(-1).squeeze(-2)
    ).sum(0)
    expected_max = torch.maximum(scores.amax(-1, keepdim=True), torch.tensor(sink))
    expected_weights = torch.exp(scores - expected_max)
    expected_total = expected_weights.sum(-1, keepdim=True) + torch.exp(
        sink - expected_max
    )
    torch.testing.assert_close(
        combined, expected_weights @ values / expected_total, rtol=2e-5, atol=2e-6
    )
