# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent FP32 SDPA checks for the revised algorithm (CPU golden, not GPU L-A)."""

import pytest
import torch
import torch_npu  # noqa: F401 -- registers the NPU backend
from sageattention2 import reference, sageattn


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("seq", [256, 1024, 4096])
def test_output_without_lse(dtype, causal, seq):
    torch.manual_seed(71)
    q = torch.randn(1, 4, seq, 128, device="npu", dtype=dtype)
    k = torch.randn(1, 2, seq, 128, device="npu", dtype=dtype)
    v = torch.randn_like(k)
    out = sageattn(q, k, v, is_causal=causal)
    with_lse, _ = sageattn(q, k, v, is_causal=causal, return_lse=True)
    # Atomic K mean reduction may change quantization-boundary rounding.
    tol = 2 ** (-9 if dtype == torch.float16 else -6)
    torch.testing.assert_close(out.cpu(), with_lse.cpu(), atol=tol, rtol=tol)
    qc, kc, vc = [t.cpu().float() for t in (q, k, v)]
    truth = torch.nn.functional.scaled_dot_product_attention(
        qc, kc, vc, is_causal=causal, enable_gqa=True
    )
    expected = reference.attention(
        q.cpu(),
        k.cpu(),
        v.cpu(),
        is_causal=causal,
        pv_storage="fp32",
        score_mode="fp32",
    ).float()
    error = (out.cpu().float() - truth).abs()
    ref_error = (expected - truth).abs()
    # L-B-style ratios using the revised CPU algorithm, not original GPU Sage.
    assert error.max() <= 2 * ref_error.max() + 1e-6
    assert error.mean() <= 1.2 * ref_error.mean() + 1e-6
    assert error.square().mean().sqrt() <= 1.2 * ref_error.square().mean().sqrt() + 1e-6
