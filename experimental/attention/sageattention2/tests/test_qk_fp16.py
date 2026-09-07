# Copyright (c) 2026 G_W_E. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""DEQF16 routing and mixed half-input/float-output softmax arithmetic."""

import ctypes as C
import math
import pytest
import torch
import torch_npu  # noqa: F401 -- registers the NPU backend
from sageattention2 import _library


@pytest.mark.parametrize("extreme", [False, True])
def test_fixed_deq_both_aivs(extreme):
    torch.manual_seed(159)
    q = torch.randint(-127, 128, (128, 128), dtype=torch.int8)
    k = torch.randint(-127, 128, (256, 128), dtype=torch.int8)
    if extreme:
        q.fill_(127)
        k[:128].fill_(127)
        k[128:].fill_(-127)
    tensors = [q.npu(), k.npu(), torch.empty((128, 256), device="npu", dtype=torch.float16)]
    fn = _library().sage_qk_half_debug_launch
    fn.argtypes = [C.c_void_p, C.POINTER(C.c_void_p)]
    fn.restype = None
    fn(torch.npu.current_stream().npu_stream, (C.c_void_p * 3)(*[t.data_ptr() for t in tensors]))
    expected = ((q.double() @ k.double().T) / 2048).half()
    torch.testing.assert_close(tensors[-1].cpu(), expected, atol=0, rtol=0)
    assert expected.abs().max() <= 1024


@pytest.mark.parametrize(
    "sk,qstart,sq,causal,scale,imbalance",
    [
        (512, 0, 64, False, 0.125, 1.0),
        (301, 0, 33, False, -0.125, 1.0),
        (512, 256, 283, True, 0.125, 1.0),
        (256, 0, 1, True, 0.0, 1.0),
        (512, 0, 64, False, 0.125, 1.0e6),
        (512, 0, 64, False, 0.125, 1.0e-6),
        (512, 0, 64, False, 125.0, 1.0),
        (256, 0, 64, False, 0.0, 1.0),
    ],
)
def test_half_softmax_channel_order_and_states(sk, qstart, sq, causal, scale, imbalance):
    torch.manual_seed(157)
    blocks = (sk + 255) // 256
    raw = (torch.randint(-20000, 20000, (blocks, 64, 256)).float() / 2048).half()
    ks = (torch.rand(blocks, 256) * 0.02 + 0.005) * imbalance
    qs = (torch.rand(64) * 0.02 + 0.005) / imbalance
    values = torch.empty_like(raw, device="npu")
    state = torch.empty((blocks, 3, 64), device="npu")
    probs = torch.empty_like(raw, device="npu", dtype=torch.uint8)
    tensors = [raw.npu(), ks.npu(), qs.npu(), values, state, probs]
    fn = _library().sage_half_softmax_debug_launch
    fn.argtypes = [C.c_void_p, C.POINTER(C.c_void_p)] + [C.c_uint32] * 5 + [C.c_float]
    fn.restype = None
    fn(
        torch.npu.current_stream().npu_stream,
        (C.c_void_p * 6)(*[t.data_ptr() for t in tensors]),
        blocks,
        sk,
        qstart,
        sq,
        causal,
        scale,
    )
    values, state, probs = [t.cpu() for t in (values, state, probs)]
    probs = probs.reshape(blocks, 8, 64, 32).permute(0, 2, 1, 3).reshape(blocks, 64, 256)
    m = torch.full((64,), -65504.0)
    den = torch.zeros(64)
    for b in range(blocks):
        qm = qs[: max(0, min(64, sq - qstart))].max().clamp_min(1e-30)
        km = ks[b, : min(256, sk - b * 256)].max().clamp_min(1e-30)
        common = qm.sqrt() * km.sqrt() * math.sqrt(2048 * abs(scale))
        qh = (qs * (common / qm) * (-1 if scale < 0 else 1)).half()
        kh = (ks[b] * (common / km)).half()
        x = (raw[b].float() * (qh[:, None].float() * kh[None, :].float()).half().float()).half()
        rows = torch.arange(qstart, qstart + 64)[:, None]
        cols = torch.arange(b * 256, (b + 1) * 256)[None, :]
        valid = (rows < sq) & (cols < sk)
        if causal:
            valid &= cols <= rows
        x.masked_fill_(~valid, -torch.inf)
        nm = torch.maximum(m, x.float().amax(-1))
        alpha = (m - nm).exp()
        p = (x.float() - nm[:, None]).exp() * 448
        den = den * alpha + p.sum(-1)
        m = nm
        torch.testing.assert_close(values[b], x, atol=0, rtol=0)
        for idx, expected in enumerate((m, den, alpha)):
            torch.testing.assert_close(state[b, idx], expected, atol=3e-5, rtol=3e-6)
        torch.testing.assert_close(probs[b], p.to(torch.float8_e4m3fn).view(torch.uint8), atol=0, rtol=0)
