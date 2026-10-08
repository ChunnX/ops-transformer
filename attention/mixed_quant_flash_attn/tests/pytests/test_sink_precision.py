# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent precision/FD contracts, without requiring an NPU."""

import importlib.util
from pathlib import Path

import pytest
import torch

from core.cpu import run_cpu_golden, softmax_flash
from core.gen_data import generate_inputs


_mla_spec = importlib.util.spec_from_file_location(
    "mqfa_mla_attention_reference",
    Path(__file__).resolve().parents[3]
    / "flash_mla_with_kvcache/tests/assets/impl/attention_math.py",
)
_mla = importlib.util.module_from_spec(_mla_spec)
_mla_spec.loader.exec_module(_mla)

_fa_spec = importlib.util.spec_from_file_location(
    "mqfa_flash_attn_reference",
    Path(__file__).resolve().parents[3] / "flash_attn/tests/assets/impl/golden.py",
)
_fa = importlib.util.module_from_spec(_fa_spec)
_fa_spec.loader.exec_module(_fa)


@pytest.mark.parametrize("sink", [None, -1000.0, 0.0, 1000.0])
@pytest.mark.parametrize("mode", ["full", "tiled"])
def test_fp32_sink_masked_rows_against_flash_attn(sink, mode):
    generator = torch.Generator().manual_seed(21)
    q = torch.randn(3, 8, generator=generator)
    k = torch.randn(7, 8, generator=generator)
    v = torch.randn(7, 8, generator=generator)
    mask = torch.zeros(3, 7, dtype=torch.bool)
    mask[0] = True
    mask[1, :3] = True
    out, total, maximum = _mqfa_partition(q, k, v, 0.25, mask, sink, tile_size=3)
    lse = torch.where(total > 0, maximum + total.log(), torch.inf).squeeze(-1)
    golden = _fa.FlashAttnGolden(1, 1, head_dim=8, tile_size=3)
    if mode == "full":
        expected, expected_lse = golden._compute_full_single(q, k, v, mask, 0.25, sink)
    else:
        expected, expected_lse = golden._compute_tiled_single(q, k, v, mask, 0.25, sink)
    torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(lse, expected_lse, rtol=1e-6, atol=1e-6)
    if sink is not None:
        assert lse[0] == sink
        assert not out[0].count_nonzero()


def _mqfa_partition(q, k, v, scale, mask, sink, tile_size=512):
    """Exercise MQFA softmax state and native PV rounding independently of packing."""
    total = torch.zeros(q.shape[0], 1)
    maximum = torch.full_like(total, -torch.inf)
    acc = torch.zeros(q.shape[0], v.shape[-1])
    for start in range(0, k.shape[0], tile_size):
        stop = min(start + tile_size, k.shape[0])
        scores = (q.float() @ k[start:stop].float().T) * scale
        scores.masked_fill_(mask[:, start:stop], -torch.inf)
        weights, total, maximum, correction = softmax_flash(
            scores,
            total,
            maximum,
            torch.ones_like(total),
            start == 0,
            torch.full_like(total, sink) if start == 0 and sink is not None else None,
        )
        acc = acc * correction + weights.to(q.dtype).float() @ v[start:stop].float()
    out = acc / torch.where(total > 0, total, 1.0)
    return out, total, maximum


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("sink", [None, -1000.0, 5.0, 1000.0])
@pytest.mark.parametrize("split_count", [1, 3])
def test_native_weight_online_and_fd_against_mla(dtype, sink, split_count):
    generator = torch.Generator().manual_seed(20261009)
    q = (torch.randn(3, 8, generator=generator) * 2).to(dtype)
    k = (torch.randn(1025, 8, generator=generator) * 2).to(dtype)
    v = torch.randn(1025, 8, generator=generator).to(dtype)
    mask = torch.zeros(3, 1025, dtype=torch.bool)
    mask[0] = True  # Fully masked row, including negative dominant sink.
    mask[1, :512] = True  # First FD partition/tile has no KV mass.
    mask[2, 1024] = True  # Non-aligned masked tail.
    scale = 0.25
    partitions = torch.tensor_split(torch.arange(k.shape[0]), split_count)
    outputs, totals, maxima = [], [], []
    for idx, indices in enumerate(partitions):
        start, stop = int(indices[0]), int(indices[-1]) + 1
        local_mask = mask[:, start:stop]
        local_sink = sink if idx == 0 else None
        out, total, maximum = _mqfa_partition(
            q, k[start:stop], v[start:stop], scale, local_mask, local_sink
        )
        reference, reference_lse = _mla.attention_single(
            q,
            k[start:stop],
            v[start:stop],
            scale,
            512,
            lambda qs, qe, ks, ke: local_mask[qs:qe, ks:ke],
            local_sink,
        )
        torch.testing.assert_close(out, reference, rtol=1e-6, atol=1e-6)
        lse = torch.where(total > 0, maximum + total.log(), torch.inf).squeeze(-1)
        torch.testing.assert_close(lse, reference_lse, rtol=1e-6, atol=1e-6)
        outputs.append(out)
        totals.append(total)
        maxima.append(maximum)
    maximum = torch.stack(maxima).amax(0)
    safe_maximum = torch.where(torch.isneginf(maximum), 0.0, maximum)
    masses = torch.stack(
        [total * (m - safe_maximum).exp() for total, m in zip(totals, maxima)]
    )
    total = masses.sum(0)
    merged = (torch.stack(outputs) * masses).sum(0) / torch.where(total > 0, total, 1.0)
    lse = torch.where(total > 0, maximum + total.log(), torch.inf).squeeze(-1)
    reference, reference_lse = _mla.attention_single(
        q, k, v, scale, 512, lambda qs, qe, ks, ke: mask[qs:qe, ks:ke], sink
    )
    # Native rounding is partition dependent; use the existing operator tolerance.
    atol, rtol = (0.0001, 0.0078125) if dtype == torch.bfloat16 else (0.000025, 0.005)
    torch.testing.assert_close(merged, reference, rtol=rtol, atol=atol)
    torch.testing.assert_close(lse, reference_lse, rtol=1e-6, atol=1e-6)


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
