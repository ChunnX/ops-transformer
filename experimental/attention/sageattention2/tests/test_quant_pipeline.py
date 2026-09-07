# Copyright (c) 2026 G_W_E. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Sequence partitions, per-core slot reuse, and quantization padding."""

import pytest
import torch
import torch_npu  # noqa: F401 -- registers the NPU backend

from sageattention2 import _run, reference


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "d,seq,smooth",
    [(64, 1024, True), (128, 1025, True), (128, 4096, True), (64, 4096, False)],
)
def test_partitioned_stats_and_quantization(dtype, d, seq, smooth):
    torch.manual_seed(811)
    q = torch.randn(1, 4, seq, d, dtype=dtype)
    k = torch.randn(1, 2, seq, d, dtype=dtype)
    v = torch.randn_like(k)
    q[..., 0, :] = 0
    k[..., 0] = 0
    k[..., 1] = 0.75
    v[..., 0] = 0
    v[..., 1] = -1.25
    # 4096 rows give every AIV multiple Q/K tasks, including Q->K slot reuse.
    # 1025 also exercises uneven Stats partitions and the row-quant fallback.
    actual = [
        x.cpu()
        for x in _run(q.npu(), k.npu(), v.npu(), smooth_k=smooth, quant_only=True)
    ]
    golden = reference.quantize(q, k, v, smooth_k=smooth)
    q8, k8, v8, qs, ks, mean, vscale = actual
    for value, expected in (
        (qs[..., :seq], golden[3]),
        (ks[..., :seq], golden[4]),
        (mean, golden[5]),
        (vscale, golden[6]),
    ):
        torch.testing.assert_close(value, expected, atol=1e-6, rtol=1e-5)
    for value, expected in (
        (q8[..., :seq, :], golden[0]),
        (k8[..., :seq, :], golden[1]),
    ):
        torch.testing.assert_close(value, expected, atol=1, rtol=0)
    bits = v8[..., :seq, :].to(torch.int16)
    expected_bits = golden[2].view(torch.uint8).to(torch.int16)
    assert (bits == expected_bits).float().mean().item() >= 0.999
    assert (bits - expected_bits).abs().max().item() <= 1
    assert torch.count_nonzero(q8[..., 0, :]).item() == 0
    assert torch.count_nonzero(v8[..., 0]).item() == 0
    for quantized in (q8, k8, v8):
        assert torch.count_nonzero(quantized[..., seq:, :]).item() == 0
