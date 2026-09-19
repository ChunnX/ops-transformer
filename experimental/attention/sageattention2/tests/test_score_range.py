# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Regression for score magnitudes exceeding FP16 intermediate range."""

import pytest
import torch
import torch_npu  # noqa: F401 -- registers the NPU backend


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
def test_attention_score_above_fp16_range(dtype, causal):
    from sageattention2 import reference, sageattn

    # K is large but final logits remain moderate because Q is small.
    # Before Q scaling, INT32_dot * K_scale exceeds the FP16 range.
    q = torch.full((1, 2, 32, 128), 0.001, dtype=dtype)
    k = torch.full((1, 1, 32, 128), 10000.0, dtype=dtype)
    torch.manual_seed(113)
    v = torch.randn_like(k)
    packed = reference.quantize(q, k, v, smooth_k=False)
    raw = packed[0][0, 0].float() @ packed[1][0, 0].float().T
    assert (raw * packed[4][0, 0]).abs().max() > torch.finfo(torch.float16).max
    output, lse = sageattn(
        q.npu(), k.npu(), v.npu(), smooth_k=False, is_causal=causal, return_lse=True
    )
    assert torch.isfinite(output).all() and torch.isfinite(lse).all()
    expected, expected_lse = reference.attention(
        q, k, v, smooth_k=False, is_causal=causal, return_lse=True
    )
    torch.testing.assert_close(output.cpu(), expected, atol=0.002, rtol=0.01)
    torch.testing.assert_close(lse.cpu(), expected_lse, atol=0.0002, rtol=0.0002)
