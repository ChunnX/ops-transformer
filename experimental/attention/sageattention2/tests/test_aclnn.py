# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pytest
import torch
import torch_npu  # noqa: F401 -- registers the NPU backend
from sageattention2 import reference
from sageattention2.aclnn import run


def assert_reference(actual, expected):
    if actual.dtype == torch.float32:
        torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=0.01, rtol=0.002)
        return
    tol = 2 ** (-9 if actual.dtype == torch.float16 else -6)
    error = (actual.cpu().float() - expected.float()).abs()
    assert torch.isfinite(actual).all()
    assert (error <= tol * (1 + expected.float().abs())).float().mean() >= 0.99
    assert error.max() <= (0.1 if actual.dtype == torch.float16 else 1.0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["HND", "NHD"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("return_lse", [False, True])
def test_aclnn_matches_reference(dtype, layout, causal, return_lse):
    torch.manual_seed(271)
    q = torch.randn(2, 4, 65, 96, device="npu", dtype=dtype)
    k = torch.randn(2, 2, 65, 96, device="npu", dtype=dtype)
    v = torch.randn_like(k)
    if layout == "NHD":
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
    opts = {"tensor_layout": layout, "is_causal": causal, "return_lse": return_lse}
    actual = run(q, k, v, **opts)
    expected = reference.attention(q.cpu(), k.cpu(), v.cpu(), **opts)
    if not return_lse:
        actual = (actual,)
        expected = (expected,)
    for a, e in zip(actual, expected):
        assert_reference(a, e)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("smooth_k", [False, True])
def test_fused_concurrent_streams(dtype, smooth_k):
    """Exercise idle cores, atomic stats and repeated per-call workspace reuse."""
    torch.manual_seed(273)
    inputs = []
    for sq, sk, dim in [(17, 65, 48), (128, 4096, 128)]:
        q = torch.randn(1, 4, sq, dim, device="npu", dtype=dtype)
        k = torch.randn(1, 2, sk, dim, device="npu", dtype=dtype)
        inputs.append((q, k, torch.randn_like(k)))
    options = {"smooth_k": smooth_k, "return_lse": True}
    expected = [run(*tensors, **options) for tensors in inputs]
    # Check each serial baseline against the independent CPU algorithm as well.
    for tensors, result in zip(inputs, expected):
        cpu = reference.attention(*(x.cpu() for x in tensors), **options)
        assert_reference(result[0], cpu[0])
        torch.testing.assert_close(result[1].cpu(), cpu[1], atol=2e-4, rtol=2e-4)
    torch.npu.synchronize()
    streams = [torch.npu.Stream(), torch.npu.Stream()]
    actual = []
    for _ in range(4):
        for stream, tensors, gold in zip(streams, inputs, expected):
            with torch.npu.stream(stream):
                actual.append((run(*tensors, **options), gold))
    for stream in streams:
        stream.synchronize()
    for result, gold in actual:
        for value, expected_tensor in zip(result, gold):
            # Use the existing CPU-reference LSE tolerance and dtype output
            # tolerances, allowing the
            # FP32 atomic summation order to vary across concurrent launches.
            tol = (
                2e-4
                if value.dtype == torch.float32
                else 2 ** (-9 if dtype == torch.float16 else -6)
            )
            torch.testing.assert_close(
                value.cpu(), expected_tensor.cpu(), atol=tol, rtol=tol
            )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "sq,sk,dim",
    [
        (128, 4096, 96),  # Wide with a padded head dimension.
        (127, 4096, 128),  # Query tail selects the ordinary path.
        (128, 4095, 128),  # Key tail selects the ordinary path.
        (128, 3584, 128),  # Aligned keys below the wide threshold.
        (128, 4096, 64),  # DP64 always uses the ordinary path.
    ],
)
def test_template_dispatch_boundaries(dtype, sq, sk, dim):
    """Check host specialization against the independent CPU algorithm."""
    torch.manual_seed(275)
    q = torch.randn(1, 2, sq, dim, device="npu", dtype=dtype)
    k = torch.randn(1, 1, sk, dim, device="npu", dtype=dtype)
    v = torch.randn_like(k)
    actual = run(q, k, v, return_lse=True)
    block_k = (
        512 if dim > 64 and sq % 128 == 0 and sk % 512 == 0 and sk >= 4096 else 256
    )
    expected = reference.attention(
        q.cpu(), k.cpu(), v.cpu(), return_lse=True, block_k=block_k
    )
    for value, gold in zip(actual, expected):
        assert_reference(value, gold)
