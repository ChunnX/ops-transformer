# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pytest
import torch
import torch_npu  # noqa: F401 -- registers the NPU backend
from sageattention2 import reference, sageattn_qk_int8_pv_fp8_asc


def assert_quantized_close(actual, expected):
    # Revised per-token CPU reference, NOT the original GPU L-A golden.
    tol = 2 ** (-9 if expected.dtype == torch.float16 else -6)
    error = (actual.float() - expected.float()).abs()
    assert torch.isfinite(actual).all()
    assert (error <= tol + tol * expected.float().abs()).float().mean() >= 0.99
    assert error.max() <= (0.1 if expected.dtype == torch.float16 else 1.0)


@pytest.mark.parametrize(
    "sq,sk,d,causal,smooth",
    [(16, 64, 64, False, False), (17, 65, 48, False, True), (65, 65, 128, True, True)],
)
def test_attention(sq, sk, d, causal, smooth):
    torch.manual_seed(12)
    q = torch.randn(1, 2, sq, d, dtype=torch.float16)
    k = torch.randn(1, 1, sk, d, dtype=torch.float16)
    v = torch.randn_like(k)
    options = {"is_causal": causal, "smooth_k": smooth, "return_lse": True}
    o, lse = sageattn_qk_int8_pv_fp8_asc(*[t.npu() for t in (q, k, v)], **options)
    torch.npu.synchronize()
    expected, expected_lse = reference.attention(q, k, v, **options)
    torch.testing.assert_close(o.cpu(), expected, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(lse.cpu(), expected_lse, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["HND", "NHD"])
@pytest.mark.parametrize(
    "d,sq,sk,scale",
    [(1, 1, 1, 0.0), (64, 33, 97, -0.125), (96, 129, 131, None),
     (1, 129, 137, None), (17, 65, 97, None), (63, 127, 129, None),
     (65, 129, 131, None), (127, 193, 257, None)],
)
def test_layout_dtype_tail(dtype, layout, d, sq, sk, scale):
    torch.manual_seed(17)
    # Slicing exercises noncontiguous outer dimensions with a contiguous D.
    q = torch.randn(2, 4, sq * 2, d, dtype=dtype)[:, :, ::2]
    k = torch.randn(2, 2, sk * 2, d, dtype=dtype)[:, :, ::2]
    v = torch.randn_like(k)
    if layout == "NHD":
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
    opts = {
        "tensor_layout": layout,
        "sm_scale": scale,
        "return_lse": True,
        "pv_accum_dtype": "fp32",
    }

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
        q, k, v, tensor_layout=layout, sm_scale=scale, return_lse=True
    )
    assert_quantized_close(actual.cpu(), expected)
    # CPU and device quantization can also shift LSE slightly.
    torch.testing.assert_close(al.cpu(), el, atol=0.01, rtol=0.002)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["HND", "NHD"])
def test_batched_copy_broadcast_kv(dtype, layout):
    torch.manual_seed(19)
    query = torch.randn(2, 4, 129, 65, dtype=dtype)
    key = torch.randn(2, 2, 1, 65, dtype=dtype)
    value = torch.randn_like(key)
    device_inputs = [query.npu(), key.npu().expand(2, 2, 137, 65), value.npu().expand(2, 2, 137, 65)]
    inputs = [query, key.expand(2, 2, 137, 65), value.expand(2, 2, 137, 65)]
    if layout == "NHD":
        device_inputs = [tensor.transpose(1, 2) for tensor in device_inputs]
        inputs = [tensor.transpose(1, 2) for tensor in inputs]
    sequence_axis = 2 if layout == "HND" else 1
    assert device_inputs[1].stride(sequence_axis) == 0
    assert device_inputs[2].stride(sequence_axis) == 0
    options = {"tensor_layout": layout, "return_lse": True}
    actual, actual_lse = sageattn_qk_int8_pv_fp8_asc(*device_inputs, **options)
    expected, expected_lse = reference.attention(*inputs, **options)
    assert_quantized_close(actual.cpu(), expected)
    torch.testing.assert_close(actual_lse.cpu(), expected_lse, atol=0.01, rtol=0.002)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("smooth_k", [False, True])
@pytest.mark.parametrize(
    "dim,sq,sk",
    [(1, 1, 1), (17, 65, 97), (48, 128, 192), (63, 257, 511),
     (64, 513, 769), (65, 513, 513), (96, 512, 512),
     (127, 513, 769), (128, 513, 769)],
)
def test_contiguous_quant_batch_padding_tail(dtype, smooth_k, dim, sq, sk, record_property):
    torch.manual_seed(955)
    query = torch.randn(2, 8, sq, dim, dtype=dtype)
    key = torch.randn(2, 4, sk, dim, dtype=dtype)
    value = torch.randn_like(key)
    query[:, :, ::17].zero_()
    key[:, -1].zero_()
    options = {"smooth_k": smooth_k, "return_lse": True}
    actual, actual_lse = sageattn_qk_int8_pv_fp8_asc(
        query.npu(), key.npu(), value.npu(), **options
    )
    actual = actual.cpu()
    expected, expected_lse = reference.attention(query, key, value, **options)
    assert_quantized_close(actual, expected)
    torch.testing.assert_close(actual_lse.cpu(), expected_lse, atol=0.01, rtol=0.002)
    truth = torch.nn.functional.scaled_dot_product_attention(
        query.float(), key.float(), value.float(), enable_gqa=True
    )
    relative_l2 = (actual.float() - truth).norm() / truth.norm().clamp_min(1e-12)
    record_property("relative_l2_vs_fp32_sdpa", relative_l2.item())
    assert relative_l2 <= 0.05


@pytest.mark.parametrize(
    "bad,message",
    [
        ({"qk_quant_gran": "invalid"}, "qk_quant_gran must be"),
        ({"dropout_p": 0.1}, "dropout is unsupported"),
        ({"attn_mask": torch.ones(1)}, "attn_mask is unsupported"),
    ],
)
def test_reject_unsupported(bad, message):
    q = torch.zeros(1, 1, 4, 64, dtype=torch.float16, device="npu")
    with pytest.raises(ValueError, match=message):
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
    _, expected_lse = reference.attention(q, k, v, is_causal=causal, return_lse=True)
    torch.testing.assert_close(lse.cpu(), expected_lse, atol=0.01, rtol=0.002)
