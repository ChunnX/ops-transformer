# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""FP8 PV with FP32 accumulation and Fixpipe transport."""

import pytest
import torch
import torch_npu  # noqa: F401 -- registers the NPU backend
from sageattention2 import reference, sageattn


@pytest.mark.parametrize("seq", [256, 512, 768, 1024, 1280])
def test_pv_slot_wraparound(seq):
    # One/two/three/four/five KV blocks cover prologue, both slots, reuse,
    # and odd/even drains, with multiple query tiles per physical core.
    torch.manual_seed(seq)
    q, k, v = [torch.randn(1, 4, seq, 64, dtype=torch.float16).npu() for _ in range(3)]
    out, lse = sageattn(q, k, v, return_lse=True)
    for _ in range(3):
        again, again_lse = sageattn(q, k, v, return_lse=True)
        torch.testing.assert_close(again, out, atol=2**-9, rtol=2**-9)
        torch.testing.assert_close(again_lse, lse, atol=2e-6, rtol=2e-6)
    expected, expected_lse = reference.attention(
        q.cpu(), k.cpu(), v.cpu(), return_lse=True
    )
    # Independent CPU quantization can round boundary values differently;
    # use the public CPU-reference contract. Atomic K mean accumulation
    # permits rounding differences on repeated calls.
    error = (out.cpu().float() - expected.float()).abs()
    tol = 2**-9
    assert (error <= tol * (1 + expected.float().abs())).float().mean() >= 0.99
    assert error.max() <= 0.1
    torch.testing.assert_close(lse.cpu(), expected_lse, atol=0.0002, rtol=0.0002)
