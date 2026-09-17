# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Regression tests for layout/rank validation before FlashAttn shape access."""

import re

import pytest
import torch
import torch_npu  # noqa: F401
from torch._subclasses.fake_tensor import FakeTensorMode

from cann_ops_transformer.ops import flash_attn


SHAPES = {
    "TND": (16, 4, 64),
    "BSND": (1, 16, 4, 64),
    "BNSD": (1, 4, 16, 64),
    "PA_BBND": (1, 128, 4, 64),
    "PA_BNBD": (1, 4, 128, 64),
    "PA_NZ": (1, 4, 4, 128, 16),
}
LAYOUTS = [(x, x) for x in ("TND", "BSND", "BNSD")] + [
    ("TND", x) for x in ("PA_BBND", "PA_BNBD", "PA_NZ")
]


@pytest.fixture(params=["meta", "fake", "npu"])
def device(request):
    if request.param == "fake":
        with FakeTensorMode():
            yield "cpu"
    else:
        if request.param == "npu" and not torch.npu.is_available():
            pytest.skip("NPU is unavailable")
        yield request.param


@pytest.mark.parametrize("layout_q,layout_kv", LAYOUTS)
@pytest.mark.parametrize("name", ["q", "k", "v"])
@pytest.mark.parametrize("rank_change", [-1, 1])
def test_invalid_rank(device, layout_q, layout_kv, name, rank_change):
    shapes = {"q": SHAPES[layout_q], "k": SHAPES[layout_kv], "v": SHAPES[layout_kv]}
    expected = len(shapes[name])
    shapes[name] = shapes[name][:-1] if rank_change == -1 else (*shapes[name], 1)
    inputs = {
        key: torch.empty(shape, device=device, dtype=torch.float16)
        for key, shape in shapes.items()
    }
    layout = layout_q if name == "q" else layout_kv
    message = f"{name} with layout {layout} expects {expected} dims, but got {expected + rank_change} dims"
    with pytest.raises(RuntimeError, match=re.escape(message)):
        flash_attn(
            **inputs, layout_q=layout_q, layout_kv=layout_kv, layout_out=layout_q
        )


@pytest.mark.parametrize("layout_q,layout_kv", LAYOUTS)
def test_valid_meta_shape(layout_q, layout_kv):
    q = torch.empty(SHAPES[layout_q], device="meta", dtype=torch.float16)
    k = torch.empty(SHAPES[layout_kv], device="meta", dtype=torch.float16)
    # V may have a different head dimension from Q/K.
    v_shape = list(SHAPES[layout_kv])
    v_shape[-3 if layout_kv == "PA_NZ" else -1] *= 2
    v = torch.empty(v_shape, device="meta", dtype=torch.float16)
    out, _ = flash_attn(
        q, k, v, layout_q=layout_q, layout_kv=layout_kv, layout_out=layout_q
    )
    assert out.shape == (*SHAPES[layout_q][:-1], 128)


@pytest.mark.parametrize("attribute", ["layout_q", "layout_kv"])
def test_unknown_layout(device, attribute):
    q = torch.empty(SHAPES["TND"], device=device, dtype=torch.float16)
    kwargs = dict(layout_q="TND", layout_kv="TND", layout_out="TND")
    kwargs[attribute] = "UNKNOWN"
    with pytest.raises(RuntimeError, match="Unsupported .* layout"):
        flash_attn(q, q, q, **kwargs)
