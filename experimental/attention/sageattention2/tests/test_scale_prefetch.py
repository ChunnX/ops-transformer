# Copyright (c) 2026 G_W_E. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""K-scale prefetch tails, KV-head changes and slot reuse across query tiles."""

import pytest
import torch
import torch_npu  # noqa: F401 -- registers the NPU backend
from sageattention2 import _run, reference


@pytest.mark.parametrize("sk", [4096, 4352, 16384, 16640, 33024])
def test_scale_prefetch_long_kv_and_heads(sk):
    torch.manual_seed(sk + 193)
    b, hq, hk, sq, d = 2, 8, 2, 640, 64
    # Constant query rows let an independent one-row CPU reference cover all
    # 80 query tiles, including multiple tasks on each physical core.
    q = (torch.arange(1, b * hq + 1).reshape(b, hq, 1, 1) * 0.0078125).expand(b, hq, sq, d).half().contiguous()
    k = torch.randn(b, hk, sk, d, dtype=torch.float16)
    v = torch.randn_like(k) * 0.1
    for start in range(0, sk, 256):
        k[:, :, start : start + 256] *= 2.0 ** ((start // 256) % 5)
        v[:, :, start : start + 256] += ((start // 256) % 7) * 0.25
    inputs = [t.npu() for t in (q, k, v)]
    out, lse = _run(*inputs, return_lse=True)
    again, again_lse = _run(*inputs, return_lse=True)
    torch.testing.assert_close(again, out, atol=0, rtol=0)
    torch.testing.assert_close(again_lse, lse, atol=0, rtol=0)
    q8, k8, v8, qs, ks, m, vs = [t.cpu() for t in _run(*inputs, quant_only=True)]
    packed = (
        q8[:, :, :1, :d],
        k8[:, :, :sk, :d],
        v8[:, :, :sk, :d].view(torch.float8_e4m3fn),
        qs[:, :, :1],
        ks[:, :, :sk],
        m,
        vs,
    )
    expected, expected_lse = reference.attention(q[:, :, :1], k, v, quantized=packed, return_lse=True)
    torch.testing.assert_close(out.cpu(), expected.expand_as(q), atol=0.002, rtol=0.002)
    torch.testing.assert_close(lse.cpu(), expected_lse.expand(b, hq, sq), atol=0.0002, rtol=0.0002)
