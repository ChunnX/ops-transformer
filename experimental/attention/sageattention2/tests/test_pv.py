# Copyright (c) 2026 G_W_E. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""FP8 PV with FP32 accumulation and Fixpipe transport."""

import ctypes
import pytest
import torch
import torch_npu  # noqa: F401 -- registers the NPU backend
from sageattention2 import _library, _run, reference


@pytest.mark.parametrize("extreme", [False, True])
def test_pv_fixpipe_rounding_and_both_aivs(extreme):
    storage = torch.float32
    torch.manual_seed(131)
    levels = torch.tensor([16.0, 32.0, 64.0, 128.0, 256.0, 448.0])
    p = levels[torch.randint(6, (128, 256))]
    v = levels[torch.randint(6, (256, 128))] * (torch.randint(2, (256, 128)) * 2 - 1)
    if extreme:
        p.fill_(448)
        v.fill_(448)
    # Byte inputs avoid depending on PyTorch NPU FP8 operator support.
    tensors = [
        p.to(torch.float8_e4m3fn).view(torch.uint8).npu(),
        v.to(torch.float8_e4m3fn).view(torch.uint8).npu(),
        torch.empty((128, 128), dtype=storage, device="npu"),
    ]
    fn = _library().sage_pv_fp32_debug_launch
    fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    fn.restype = None
    fn(torch.npu.current_stream().npu_stream, (ctypes.c_void_p * 3)(*[t.data_ptr() for t in tensors]))
    torch.npu.synchronize()
    exact = p.double() @ v.double()
    assert exact.abs().max() > torch.finfo(torch.float16).max
    assert torch.isfinite(tensors[-1]).all()
    torch.testing.assert_close(tensors[-1].cpu(), exact.to(storage), atol=0, rtol=0)


@pytest.mark.parametrize("seq", [256, 512, 768, 1024, 1280])
def test_pv_slot_wraparound(seq):
    # One/two/three/four/five KV blocks cover prologue, both slots, reuse,
    # and odd/even drains, with multiple query tiles per physical core.
    torch.manual_seed(seq)
    q, k, v = [torch.randn(1, 4, seq, 64, dtype=torch.float16).npu() for _ in range(3)]
    out, lse = _run(q, k, v, return_lse=True)
    for _ in range(3):
        again, again_lse = _run(q, k, v, return_lse=True)
        torch.testing.assert_close(again, out, atol=0, rtol=0)
        torch.testing.assert_close(again_lse, lse, atol=0, rtol=0)
    quant = [x.cpu() for x in _run(q, k, v, quant_only=True)]
    q8, k8, v8, qs, ks, m, vs = quant
    packed = (
        q8[:, :, :seq, :64],
        k8[:, :, :seq, :64],
        v8[:, :, :seq, :64].view(torch.float8_e4m3fn),
        qs[:, :, :seq],
        ks[:, :, :seq],
        m,
        vs,
    )
    expected, expected_lse = reference.attention(q.cpu(), k.cpu(), v.cpu(), quantized=packed, return_lse=True)
    torch.testing.assert_close(out.cpu(), expected, atol=0.002, rtol=0.002)
    torch.testing.assert_close(lse.cpu(), expected_lse, atol=0.0002, rtol=0.0002)
