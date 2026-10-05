# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Dispatcher metadata, graph capture and native launch coverage."""

import sys

import pytest
import torch
import torch_npu
from sageattention2 import (
    _ops,  # noqa: F401 -- register schema and implementations
    sageattn,
)


@pytest.mark.parametrize("layout", ["HND", "NHD"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("return_lse", [False, True])
def test_meta_layout_and_optional_lse(layout, dtype, return_lse):
    shape_q = (2, 4, 33, 64) if layout == "HND" else (2, 33, 4, 64)
    shape_k = (2, 2, 65, 64) if layout == "HND" else (2, 65, 2, 64)
    q = torch.empty(shape_q, device="meta", dtype=dtype)
    k = torch.empty(shape_k, device="meta", dtype=dtype)
    out, lse = torch.ops.sageattention2.forward(
        q, k, k, tensor_layout=layout, return_lse=return_lse
    )
    assert out.shape == q.shape and out.dtype == q.dtype and out.device.type == "meta"
    if return_lse:
        assert lse.shape == (2, 4, 33) and lse.dtype == torch.float32
    else:
        assert lse is None
    assert "cann_ops_transformer" not in sys.modules


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_public_api_requires_npu(device):
    q = torch.empty(1, 2, 33, 64, device=device, dtype=torch.float16)
    with pytest.raises(ValueError, match="requires NPU inputs"):
        sageattn(q, q, q)


def test_fake_tensor_does_not_compile_extension():
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        q = torch.empty(1, 4, 33, 64, device="npu", dtype=torch.float16)
        k = torch.empty(1, 2, 65, 64, device="npu", dtype=torch.float16)
        out, lse = sageattn(q, k, k, return_lse=True)
        assert out.shape == q.shape and out.device == q.device
        assert lse.shape == (1, 4, 33) and lse.dtype == torch.float32


def test_dispatcher_schema_and_fullgraph_eager():
    torch.manual_seed(929)
    q = torch.randn(1, 4, 33, 64, device="npu", dtype=torch.float16)
    k = torch.randn(1, 2, 65, 64, device="npu", dtype=q.dtype)
    v = torch.randn_like(k)
    expected = sageattn(q, k, v, return_lse=True)
    compiled = torch.compile(sageattn, backend="eager", fullgraph=True)
    actual = compiled(q, k, v, return_lse=True)
    for x, y in zip(actual, expected):
        torch.testing.assert_close(x, y, atol=0, rtol=0)
    # Check dispatcher aliasing and FakeTensor contract on the native implementation.
    torch.library.opcheck(
        torch.ops.sageattention2.forward.default,
        (q, k, v),
        kwargs={"return_lse": True},
        test_utils=("test_schema", "test_faketensor"),
    )


def test_private_format_is_rejected_before_aclnn():
    q = torch.randn(1, 4, 32, 64, device="npu", dtype=torch.float16)
    k = torch.randn(1, 2, 64, 64, device="npu", dtype=q.dtype)
    v = torch.randn_like(k)
    # The 950 format-cast API accepts rank-3 ND/NCL -> NZ, but not rank-4 NCHW -> NZ.
    private_q = torch_npu.npu_format_cast(q.flatten(0, 1), 29).view(q.shape)
    assert torch_npu.get_npu_format(private_q) != torch_npu.get_npu_format(q)
    with pytest.raises(RuntimeError, match="requires base ND/NCHW storage"):
        sageattn(private_q, k, v)
