# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check joint 512-key normalization and both halves of the accumulated PV."""

import math

import pytest
import torch
import torch_npu  # noqa: F401 -- registers the NPU backend
from sageattention2 import reference, sageattn


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("sk", [1024, 1536, 4096])
@pytest.mark.parametrize("causal", [False, True])
def test_wide_uniform_normalization(dtype, sk, causal):
    # More than 28 query tiles also exercises per-core buffer reuse across tiles.
    sq = sk if causal else 2048
    q = torch.zeros(1, 2, sq, 128, dtype=dtype, device="npu")
    k = torch.zeros(1, 1, sk, 128, dtype=dtype, device="npu")
    v = torch.ones_like(k)
    out, lse = sageattn(q, k, v, smooth_k=False, is_causal=causal, return_lse=True)
    # Uniform P rounds to FP8 448, while the denominator retains exp(FP16 ln448).
    # Check the analytic quantized result rather than assuming exact unit output.
    p_scale = math.exp(torch.tensor(math.log(448), dtype=torch.float16).item())
    expected_out = torch.full_like(out.cpu(), 448.0 / p_scale)
    torch.testing.assert_close(out.cpu(), expected_out, atol=0, rtol=0)
    if causal:
        expected_lse = torch.arange(1, sq + 1).float().log().expand(1, 2, sq)
    else:
        expected_lse = torch.full_like(lse.cpu(), math.log(sk))
    torch.testing.assert_close(lse.cpu(), expected_lse, atol=2e-6, rtol=0)


@pytest.mark.parametrize("dominant_half", [0, 1])
def test_wide_unequal_halves(dominant_half):
    q = torch.zeros(1, 1, 128, 128, dtype=torch.float16)
    q[..., 0] = 1
    k = torch.zeros(1, 1, 4096, 128, dtype=torch.float16)
    v = torch.empty_like(k)
    for group in range(8):
        for part in range(2):
            start = group * 512 + part * 256
            k[:, :, start : start + 256, 0] = (group + 1) * (part == dominant_half)
            v[:, :, start : start + 256] = 1 if part == 0 else -1
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), scale=1.0
    )
    out, lse = sageattn(
        q.npu(),
        k.npu(),
        v.npu(),
        smooth_k=False,
        sm_scale=1.0,
        return_lse=True,
    )
    # FP8 P rounding is expected; this detects a lost half or separate maxima.
    torch.testing.assert_close(out.cpu().float(), expected, atol=0.004, rtol=0)
    # The eight groups reach logits of 8; FP16 prepared scales and scores
    # affect LSE even though the normalized output remains close to SDPA.
    _, expected_lse = reference.attention(
        q, k, v, smooth_k=False, sm_scale=1.0, return_lse=True, block_k=512
    )
    torch.testing.assert_close(lse.cpu(), expected_lse, atol=0.002, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seq", [1024, 4096])
@pytest.mark.parametrize("causal", [False, True])
def test_wide_fp32_accuracy(dtype, seq, causal):
    torch.manual_seed(719)
    q = torch.randn(1, 4, seq, 128, dtype=dtype)
    k = torch.randn(1, 2, seq, 128, dtype=dtype)
    v = torch.randn_like(k)
    out = sageattn(q.npu(), k.npu(), v.npu(), is_causal=causal).cpu().float()
    truth = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), is_causal=causal, enable_gqa=True
    )
    expected = reference.attention(
        q, k, v, is_causal=causal, pv_storage="fp32", score_mode="fp32"
    ).float()
    error = (out - truth).abs()
    ref_error = (expected - truth).abs()
    # Same independent accuracy limits as the default kernel's regression suite.
    assert error.max() <= 2 * ref_error.max() + 1e-6
    assert error.mean() <= 1.2 * ref_error.mean() + 1e-6
    assert error.square().mean().sqrt() <= 1.2 * ref_error.square().mean().sqrt() + 1e-6


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("sk", [4096, 8192])
@pytest.mark.parametrize("causal", [False, True])
def test_wide_public_dispatch(dtype, sk, causal):
    from sageattention2.aclnn import run as aclnn_run

    torch.manual_seed(723)
    q = torch.randn(1, 2, sk if causal else 128, 128, dtype=dtype, device="npu")
    k = torch.randn(1, 1, sk, 128, dtype=dtype, device="npu")
    v = torch.randn_like(k)
    public, lse = aclnn_run(q, k, v, is_causal=causal, return_lse=True)
    expected, expected_lse = reference.attention(
        q.cpu(), k.cpu(), v.cpu(), is_causal=causal, return_lse=True, block_k=512
    )
    tol = 2 ** (-9 if dtype == torch.float16 else -6)
    error = (public.cpu().float() - expected.float()).abs()
    assert (error <= tol * (1 + expected.float().abs())).float().mean() >= 0.99
    assert error.max() <= (0.1 if dtype == torch.float16 else 1.0)
    torch.testing.assert_close(lse.cpu(), expected_lse, atol=0.01, rtol=0.002)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim,layout", [(65, "HND"), (96, "NHD"), (128, "NHD")])
@pytest.mark.parametrize("causal", [False, True])
def test_wide_preprocessing_layout_accuracy(dtype, dim, layout, causal):
    # Exercise permuted V through both padded-D and strided row quantization.
    torch.manual_seed(739)
    seq = 4096 if causal else 8192
    q = torch.randn(1, 2, seq if causal else 128, dim, dtype=dtype)
    k = torch.randn(1, 1, seq, dim, dtype=dtype)
    v = torch.randn_like(k)
    inputs = [x.npu() for x in (q, k, v)]
    if layout == "NHD":
        inputs = [x.permute(0, 2, 1, 3).contiguous() for x in inputs]
    out = sageattn(*inputs, tensor_layout=layout, is_causal=causal)
    if layout == "NHD":
        out = out.permute(0, 2, 1, 3)
    out = out.cpu().float()
    truth = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), is_causal=causal, enable_gqa=True
    )
    expected = reference.attention(
        q, k, v, is_causal=causal, pv_storage="fp32", score_mode="fp32"
    ).float()
    error, ref_error = (out - truth).abs(), (expected - truth).abs()
    assert error.max() <= 2 * ref_error.max() + 1e-6
    assert error.mean() <= 1.2 * ref_error.mean() + 1e-6
    assert error.square().mean().sqrt() <= 1.2 * ref_error.square().mean().sqrt() + 1e-6
    contiguous = sageattn(q.npu(), k.npu(), v.npu(), is_causal=causal).cpu().float()
    torch.testing.assert_close(out, contiguous, atol=0, rtol=0)
