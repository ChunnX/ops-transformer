# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.

import pytest
import torch


@pytest.mark.npu
def test_sage_attention2_basic_hnd_fp16():
    """One representative UT case; the full task matrix belongs to target-machine validation."""
    sage = pytest.importorskip("sageattention2")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    torch.manual_seed(1)
    q = torch.randn((1, 4, 128, 64), dtype=torch.float16, device="npu")
    k = torch.randn((1, 2, 128, 64), dtype=torch.float16, device="npu")
    v = torch.randn((1, 2, 128, 64), dtype=torch.float16, device="npu")
    out = sage.sageattn_qk_int8_pv_fp8_asc(q, k, v, tensor_layout="HND")
    assert out.shape == q.shape
    assert out.dtype == q.dtype

