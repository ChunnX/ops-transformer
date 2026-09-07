# Copyright (c) 2026 G_W_E. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pytest
import torch
import torch_npu  # noqa: F401 -- registers the NPU backend
from sageattention2 import sageattn_qk_int8_pv_fp8_asc, _run
from sageattention2 import reference


def assert_quantized_close(actual, expected):
    # Revised per-token CPU reference, NOT the original GPU L-A golden.
    tol = 2 ** (-9 if expected.dtype == torch.float16 else -6)
    error = (actual.float() - expected.float()).abs()
    assert torch.isfinite(actual).all()
    assert (error <= tol + tol * expected.float().abs()).float().mean() >= 0.99
    assert error.max() <= (0.1 if expected.dtype == torch.float16 else 1.0)


@pytest.mark.parametrize(
    "d,seq,smooth",
    [
        (64, 16, False),
        (48, 17, True),
        (128, 65, True),
        (128, 256, True),
        (64, 80, True),
    ],
)
def test_quant(d, seq, smooth):
    torch.manual_seed(123)
    q, k, v = [torch.randn(1, 2, seq, d, dtype=torch.float16) for _ in range(3)]
    actual = _run(*[t.npu() for t in (q, k, v)], smooth_k=smooth, quant_only=True)
    torch.npu.synchronize()
    q8, k8, v8, qs, ks, m, vs = [t.cpu() for t in actual]
    g = reference.quantize(q, k, v, smooth_k=smooth)
    torch.testing.assert_close(qs[:, :, :seq], g[3], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(ks[:, :, :seq], g[4], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(m[..., :d], g[5], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(vs[..., :d], g[6], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(q8[:, :, :seq, :d], g[0], atol=1, rtol=0)
    torch.testing.assert_close(k8[:, :, :seq, :d], g[1], atol=1, rtol=0)
    # Check both the fraction of matching E4M3 codes and the largest code gap.
    bits = v8[:, :, :seq, :d].to(torch.int16)
    expected = g[2].view(torch.uint8).to(torch.int16)
    assert (bits == expected).float().mean().item() >= 0.999
    assert (bits - expected).abs().max().item() <= 1


@pytest.mark.parametrize(
    "sq,sk,d,causal,smooth",
    [(16, 64, 64, False, False), (17, 65, 48, False, True), (65, 65, 128, True, True)],
)
def test_attention(sq, sk, d, causal, smooth):
    torch.manual_seed(12)
    q = torch.randn(1, 2, sq, d, dtype=torch.float16)
    k = torch.randn(1, 1, sk, d, dtype=torch.float16)
    v = torch.randn_like(k)
    options = dict(is_causal=causal, smooth_k=smooth, return_lse=True)
    o, lse = sageattn_qk_int8_pv_fp8_asc(*[t.npu() for t in (q, k, v)], **options)
    torch.npu.synchronize()
    expected, expected_lse = reference.attention(q, k, v, **options)
    torch.testing.assert_close(o.cpu(), expected, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(lse.cpu(), expected_lse, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["HND", "NHD"])
@pytest.mark.parametrize(
    "d,sq,sk,scale", [(1, 1, 1, 0.0), (64, 33, 97, -0.125), (96, 129, 131, None)]
)
def test_layout_dtype_tail(dtype, layout, d, sq, sk, scale):
    torch.manual_seed(17)
    # Slicing exercises noncontiguous outer dimensions with a contiguous D.
    q = torch.randn(2, 4, sq * 2, d, dtype=dtype)[:, :, ::2]
    k = torch.randn(2, 2, sk * 2, d, dtype=dtype)[:, :, ::2]
    v = torch.randn_like(k)
    if layout == "NHD":
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
    opts = dict(
        tensor_layout=layout, sm_scale=scale, return_lse=True, pv_accum_dtype="fp32"
    )

    # Slice on NPU so the test really passes tensors with storage gaps.
    def device_strided(t):
        storage = torch.empty((*t.shape[:-1], d * 2), dtype=dtype, device="npu")
        # Preserve a contiguous D while leaving a gap between outer rows.
        result = storage[..., :d]
        result.copy_(t)
        return result

    device_inputs = [device_strided(t) for t in (q, k, v)]
    actual, al = sageattn_qk_int8_pv_fp8_asc(*device_inputs, **opts)
    torch.npu.synchronize()
    expected, el = reference.attention(
        q, k, v, tensor_layout=layout, sm_scale=scale, return_lse=True, v_max=448.0
    )
    assert_quantized_close(actual.cpu(), expected)
    # Quantization can shift LSE too. Isolate the attention kernel below.
    torch.testing.assert_close(al.cpu(), el, atol=0.01, rtol=0.002)
    packed = [
        t.cpu()
        for t in _run(
            *device_inputs, tensor_layout=layout, quant_only=True, pv_accum_dtype="fp32"
        )
    ]
    quantized = (
        packed[0][:, :, :sq, :d],
        packed[1][:, :, :sk, :d],
        packed[2][:, :, :sk, :d].view(torch.float8_e4m3fn),
        packed[3][:, :, :sq],
        packed[4][:, :, :sk],
        packed[5][..., :d],
        packed[6][..., :d],
    )
    same, same_lse = reference.attention(
        q,
        k,
        v,
        tensor_layout=layout,
        sm_scale=scale,
        return_lse=True,
        quantized=quantized,
    )
    torch.testing.assert_close(
        actual.cpu(), same, atol=0.002, rtol=0.01 if dtype == torch.bfloat16 else 0.002
    )
    torch.testing.assert_close(al.cpu(), same_lse, atol=5e-5, rtol=2e-5)


@pytest.mark.parametrize(
    "bad",
    [
        {"qk_quant_gran": "per_thread"},
        {"qk_quant_gran": "per_warp"},
        {"dropout_p": 0.1},
        {"attn_mask": torch.ones(1)},
    ],
)
def test_reject_unsupported(bad):
    q = torch.zeros(1, 1, 4, 64, dtype=torch.float16)
    with pytest.raises((ValueError, NotImplementedError)):
        sageattn_qk_int8_pv_fp8_asc(q, q, q, **bad)


def test_zero_and_repeated_stream():
    q = torch.zeros(1, 2, 17, 64, dtype=torch.float16, device="npu")
    k = torch.zeros(1, 1, 65, 64, dtype=torch.float16, device="npu")
    v = torch.ones_like(k)
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        q.record_stream(stream)
        k.record_stream(stream)
        v.record_stream(stream)
        stream.wait_stream(torch.npu.default_stream())
        outputs = [
            sageattn_qk_int8_pv_fp8_asc(q, k, v, return_lse=True) for _ in range(3)
        ]
    stream.synchronize()
    for out, lse in outputs:
        torch.testing.assert_close(
            out.cpu(), torch.ones_like(out.cpu()), atol=0.003, rtol=0.003
        )
        torch.testing.assert_close(
            lse.cpu(),
            torch.full_like(lse.cpu(), __import__("math").log(65)),
            atol=5e-4,
            rtol=0,
        )


@pytest.mark.parametrize(
    "sq,sk,d,causal",
    [(513, 513, 128, True), (129, 769, 64, False), (1024, 1024, 128, True)],
)
def test_streaming_blocks(sq, sk, d, causal):
    torch.manual_seed(12)
    q = torch.randn(1, 2, sq, d, dtype=torch.float16)
    k = torch.randn(1, 1, sk, d, dtype=torch.float16)
    v = torch.randn_like(k)
    device_inputs = [t.npu() for t in (q, k, v)]
    out, lse = sageattn_qk_int8_pv_fp8_asc(
        *device_inputs, is_causal=causal, return_lse=True
    )
    expected = reference.attention(q, k, v, is_causal=causal)
    assert_quantized_close(out.cpu(), expected)
    packed = [t.cpu() for t in _run(*device_inputs, quant_only=True)]
    quantized = (
        packed[0][:, :, :sq, :d],
        packed[1][:, :, :sk, :d],
        packed[2][:, :, :sk, :d].view(torch.float8_e4m3fn),
        packed[3][:, :, :sq],
        packed[4][:, :, :sk],
        packed[5][..., :d],
        packed[6][..., :d],
    )
    same, same_lse = reference.attention(
        q, k, v, is_causal=causal, return_lse=True, quantized=quantized
    )
    torch.testing.assert_close(out.cpu(), same, atol=0.002, rtol=0.002)
    torch.testing.assert_close(lse.cpu(), same_lse, atol=5e-5, rtol=2e-5)
