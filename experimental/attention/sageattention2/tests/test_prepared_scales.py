# Copyright (c) 2026 G_W_E. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Quant emits fixed FP16 scales for batched, strided, GQA and tail inputs."""

import ctypes as C

import pytest
import torch
import torch_npu  # noqa: F401 -- registers the NPU backend

from sageattention2 import _Params, _library, reference


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "b,hq,hk,sq,sk,d,scale,smooth",
    [
        (1, 2, 1, 64, 64, 64, 0.125, False),
        (2, 8, 2, 128, 512, 128, 128**-0.5, True),
        (1, 4, 1, 129, 4353, 96, 0.125, True),
        (1, 4, 2, 257, 513, 128, -0.125, True),
        (1, 4, 2, 65, 301, 64, 0.0, True),
    ],
)
def test_prepared_scales(dtype, b, hq, hk, sq, sk, d, scale, smooth):
    torch.manual_seed(709)
    q = torch.randn(b, hq, sq, d, dtype=dtype, device="npu")
    k = torch.randn(b, hk, sk, d, dtype=dtype, device="npu")
    v = torch.randn_like(k)
    qp = (sq + 127) // 128 * 128
    kp = (sk + 511) // 512 * 512
    dp = 64 if d <= 64 else 128

    def alloc(shape, dt):
        return torch.empty(shape, device="npu", dtype=dt)

    qs = alloc((b, hq, qp), torch.float32)
    ks = alloc((b, hk, kp), torch.float32)
    qh = torch.full_like(qs, float("nan"), dtype=torch.float16)
    kh = torch.full_like(ks, float("nan"), dtype=torch.float16)
    tensors = [
        q,
        k,
        v,
        alloc((b, hk, dp), torch.float32),
        alloc((b, hk, dp), torch.float32),
        alloc((b, hq, qp, dp), torch.int8),
        alloc((b, hk, kp, dp), torch.int8),
        alloc((b, hk, kp, dp), torch.uint8),
        qs,
        ks,
        None,
        None,
        qh,
        kh,
    ]
    p = _Params(
        b,
        hq,
        hk,
        sq,
        sk,
        d,
        dp,
        qp,
        kp,
        torch.npu.get_device_properties(q.device).cube_core_num,
        int(dtype == torch.bfloat16),
        0,
        int(smooth),
        0,
        scale,
        448.0,
        *[(C.c_uint64 * 3)(*t.stride()[:3]) for t in (q, k, v)],
    )
    ptrs = (C.c_void_p * 14)(*[t.data_ptr() if t is not None else 0 for t in tensors])
    fn = _library().sage_launch_stages
    fn.argtypes = [C.c_void_p, C.POINTER(_Params), C.POINTER(C.c_void_p), C.c_int]
    fn.restype = C.c_int
    assert fn(torch.npu.current_stream().npu_stream, C.byref(p), ptrs, 3) == 0
    qs, ks, actual_q, actual_k = [t.cpu() for t in (qs, ks, qh, kh)]
    # Use device-produced raw scales to isolate packing from quantization.
    expected_q, expected_k = reference.prepare_scales(
        qs[:, :, :sq], ks[:, :, :sk], p.smScale
    )
    for actual, expected in [
        (actual_q[:, :, :sq], expected_q),
        (actual_k[:, :, :sk], expected_k),
    ]:
        delta = (
            actual.contiguous().view(torch.int16).int()
            - expected.contiguous().view(torch.int16).int()
        )
        assert delta.abs().max() <= 1
    assert torch.isfinite(actual_q).all() and torch.isfinite(actual_k).all()
    assert (actual_q[:, :, sq:] == 0).all() and (actual_k[:, :, sk:] == 0).all()
    assert (qs[:, :, sq:] == 1).all() and (ks[:, :, sk:] == 1).all()
