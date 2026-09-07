# Copyright (c) 2026 G_W_E. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import pytest
import torch
import torch_npu  # noqa: F401 -- registers the NPU backend
from sageattention2.aclnn import run
from sageattention2 import _run

pytestmark = pytest.mark.skipif(
    os.environ.get("SAGEATTN_TEST_ACLNN") != "1",
    reason="set SAGEATTN_TEST_ACLNN=1 after sourcing the installed OPP environment",
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["HND", "NHD"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("return_lse", [False, True])
def test_aclnn_matches_native(dtype, layout, causal, return_lse):
    torch.manual_seed(271)
    q = torch.randn(2, 4, 65, 96, device="npu", dtype=dtype)
    k = torch.randn(2, 2, 65, 96, device="npu", dtype=dtype)
    v = torch.randn_like(k)
    if layout == "NHD":
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
    opts = dict(tensor_layout=layout, is_causal=causal, return_lse=return_lse)
    actual = run(q, k, v, **opts)
    expected = _run(q, k, v, **opts)
    if not return_lse:
        actual = (actual,)
        expected = (expected,)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a.cpu(), e.cpu(), atol=0, rtol=0)
