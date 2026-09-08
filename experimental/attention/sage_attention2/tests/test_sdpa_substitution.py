# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Task TC-10: replace the public PyTorch SDPA call in a forward function."""

import pytest
import torch
import torch_npu  # noqa: F401 -- registers the NPU backend
from sageattention2 import sageattn


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sdpa_substitution(monkeypatch, dtype):
    torch.manual_seed(926)
    q, k, v = [torch.randn(1, 4, 256, 64, dtype=dtype, device="npu") for _ in range(3)]
    expected = sageattn(q, k, v, is_causal=True)
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", sageattn)
    output = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
    )
    torch.testing.assert_close(output.cpu(), expected.cpu(), atol=0, rtol=0)
