"""Batched LSE stores and asynchronous smooth-K mean correction."""

import math

import pytest
import torch
import torch_npu
from sageattention2 import reference, sageattn


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("smooth_k", [False, True], ids=["plain", "smooth"])
@pytest.mark.parametrize(
    "sq,sk,dim,layout",
    [(65, 97, 48, "HND"), (129, 257, 96, "NHD"),
     (1024, 1024, 128, "HND"), (1024, 4096, 128, "HND")],
    ids=["dp64_tail", "dp128_tail_nhd", "narrow_repeated", "wide_repeated"],
)
def test_lse_async(dtype, smooth_k, sq, sk, dim, layout):
    torch.manual_seed(71)
    query = torch.randn(1, 4, sq, dim, dtype=dtype)
    key = (torch.randn(1, 2, sk, dim) + 0.25).to(dtype)
    value = torch.randn_like(key)
    inputs = [tensor.npu() for tensor in (query, key, value)]
    if layout == "NHD":
        inputs = [tensor.transpose(1, 2) for tensor in inputs]
    outputs = [sageattn(*inputs, tensor_layout=layout, smooth_k=smooth_k, return_lse=True)
               for _ in range(16)]
    torch.npu.synchronize()
    positions = sorted({position for position in (0, 63, 64, 127, 128, sq - 1) if position < sq})
    expected, expected_lse = reference.attention(
        query[:, :, positions], key, value, smooth_k=smooth_k, return_lse=True
    )
    tolerance = 2 ** (-9 if dtype == torch.float16 else -6)
    for actual, actual_lse in outputs:
        actual = actual.cpu()
        if layout == "NHD":
            actual = actual.transpose(1, 2)
        assert torch.isfinite(actual).all()
        actual_lse = actual_lse.cpu()
        assert torch.isfinite(actual_lse).all()
        error = (actual[:, :, positions].float() - expected.float()).abs()
        assert (error <= tolerance * (1 + expected.float().abs())).float().mean() >= 0.99
        assert error.max() <= (0.1 if dtype == torch.float16 else 1.0)
        torch.testing.assert_close(actual_lse[:, :, positions], expected_lse, atol=0.002, rtol=0.001)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("scale", [None, 0.0, -0.125, 0.375])
@pytest.mark.parametrize("seq,dim,causal", [(65, 48, False), (4096, 128, True)])
def test_lse_constant_key_mean(dtype, scale, seq, dim, causal):
    query = (torch.arange(1, 5).reshape(1, 4, 1, 1) * 0.0625).expand(1, 4, seq, dim).to(dtype).contiguous()
    key = torch.tensor([0.5, -1.0], dtype=dtype).reshape(1, 2, 1, 1).expand(1, 2, seq, dim).contiguous()
    value = torch.zeros_like(key)
    actual, actual_lse = sageattn(
        *[tensor.npu() for tensor in (query, key, value)],
        sm_scale=scale, is_causal=causal, smooth_k=True, return_lse=True,
    )
    sm_scale = 1 / math.sqrt(dim) if scale is None else scale
    mean = key.float().mean(-2, keepdim=True).repeat_interleave(2, dim=1)
    expected_lse = (query.float() * mean).sum(-1) * sm_scale
    expected_lse += torch.arange(1, seq + 1).float().log() if causal else math.log(seq)
    torch.testing.assert_close(actual.cpu(), torch.zeros_like(query), atol=0, rtol=0)
    torch.testing.assert_close(actual_lse.cpu(), expected_lse, atol=0.0002, rtol=0.0002)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_lse_broadcast_query_stride(dtype):
    torch.manual_seed(73)
    query = torch.randn(1, 4, 1, 96, dtype=dtype)
    key = (torch.randn(1, 2, 257, 96) + 0.25).to(dtype)
    value = torch.zeros_like(key)
    device_query = query.npu().expand(1, 4, 129, 96)
    assert device_query.stride(2) == 0
    actual, actual_lse = sageattn(device_query, key.npu(), value.npu(), return_lse=True)
    _, expected_lse = reference.attention(query, key, value, return_lse=True)
    torch.testing.assert_close(actual.cpu(), torch.zeros_like(actual.cpu()), atol=0, rtol=0)
    torch.testing.assert_close(actual_lse.cpu(), expected_lse.expand(1, 4, 129), atol=0.002, rtol=0.001)
