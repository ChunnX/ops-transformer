# Copyright (c) 2026 G_W_E. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Value centering, FP32 restoration, padding and compatibility arguments."""

import pytest
import torch
import torch_npu  # noqa: F401 -- registers the NPU backend
from sageattention2 import reference, sageattn


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("smooth_k", [False, True])
@pytest.mark.parametrize(
    "sq,sk,dim,causal",
    [
        (17, 65, 48, False),
        (129, 769, 96, False),
        (256, 256, 64, True),
        (128, 4096, 128, False),
        (4096, 4096, 128, True),
    ],
)
def test_smooth_v_attention(dtype, smooth_k, sq, sk, dim, causal):
    torch.manual_seed(921)
    q = torch.randn(1, 4, sq, dim, dtype=dtype)
    k = torch.randn(1, 2, sk, dim, dtype=dtype)
    offset = torch.linspace(-16, 16, dim).view(1, 1, 1, dim)
    v = (torch.randn(1, 2, sk, dim) * 0.5 + offset).to(dtype)
    options = {
        "smooth_k": smooth_k,
        "smooth_v": True,
        "is_causal": causal,
        "return_lse": True,
    }
    actual, lse = sageattn(q.npu(), k.npu(), v.npu(), **options)
    block_k = (
        512 if dim == 128 and sq % 128 == 0 and sk % 512 == 0 and sk >= 4096 else 256
    )
    golden, golden_lse = reference.attention(q, k, v, block_k=block_k, **options)
    # Independent CPU quantization includes rounding differences before the
    # channel offset is restored and rounded to the output dtype.
    tol = 2 ** (-9 if dtype == torch.float16 else -6)
    cpu_error = (actual.cpu().float() - golden.float()).abs()
    assert (cpu_error <= tol * (1 + golden.float().abs())).float().mean() >= 0.99
    assert cpu_error.max() <= (0.1 if dtype == torch.float16 else 1.0)
    torch.testing.assert_close(lse.cpu(), golden_lse, atol=0.002, rtol=0.001)
    truth = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), is_causal=causal, enable_gqa=True
    )
    error = actual.cpu().float() - truth
    # Includes final FP16/BF16 rounding of the restored channel offset.
    assert error.square().mean().sqrt() < (0.02 if dtype == torch.float16 else 0.05)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seq,dim", [(65, 48), (513, 65), (4096, 128)])
def test_smooth_v_constant_channels(dtype, seq, dim):
    torch.manual_seed(923)
    q = torch.randn(2, 4, seq, dim, device="npu", dtype=dtype)
    k = torch.randn(2, 2, seq, dim, device="npu", dtype=dtype)
    channels = (torch.arange(dim, device="npu").float().remainder(17) - 8) / 8
    v = channels.to(dtype).view(1, 1, 1, dim).expand_as(k).contiguous()
    out = sageattn(q, k, v, smooth_v=True, is_causal=True)
    expected = channels.to(dtype).view(1, 1, 1, dim).expand_as(q)
    torch.testing.assert_close(out.cpu(), expected.cpu(), atol=0, rtol=0)


@pytest.mark.parametrize("ignored", ["fp32+fp16", "fp16", "unused", None])
def test_pv_accum_argument_is_ignored(ignored):
    torch.manual_seed(925)
    q = torch.randn(1, 2, 128, 64, device="npu", dtype=torch.float16)
    k = torch.randn(1, 1, 512, 64, device="npu", dtype=torch.float16)
    v = torch.randn_like(k)
    actual = sageattn(q, k, v, pv_accum_dtype=ignored, smooth_v=True)
    expected = sageattn(q, k, v, pv_accum_dtype="fp32", smooth_v=True)
    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=0, rtol=0)
