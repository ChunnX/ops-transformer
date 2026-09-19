# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""MXFP8 group scales, zero padding, cache fallback and compatibility behavior."""

import math

import pytest
import torch
import torch_npu  # noqa: F401 -- registers the NPU backend
from sageattention2 import reference, sageattn


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "seq,dim", [(1, 1), (31, 17), (33, 48), (65, 65), (513, 96), (4096, 128)]
)
def test_mxfp8_group_scales_and_padding(dtype, seq, dim):
    # Each sequence group has a different power-of-two scale. 3.9 rounds to
    # 4.0 under E8M0/E4M3, providing an analytic reference independent of the
    # CPU quantizer. Distinct head/channel signs detect scale-layout errors.
    q = torch.zeros(1, 4, 128, dim, dtype=dtype)
    k = torch.zeros(1, 2, seq, dim, dtype=dtype)
    exponent = (torch.arange(seq) // 32).remainder(7) - 3
    magnitude = torch.exp2(exponent.float()).view(1, 1, seq, 1)
    sign = (1 - 2 * (torch.arange(2 * dim).reshape(1, 2, 1, dim) % 2)).float()
    v = (3.9 * magnitude * sign).to(dtype)
    actual, lse = sageattn(q.npu(), k.npu(), v.npu(), smooth_k=False, return_lse=True)
    p_ratio = 448.0 / math.exp(torch.tensor(math.log(448), dtype=torch.float16).item())
    decoded = 4.0 * magnitude * sign
    expected = (
        (decoded.mean(-2, keepdim=True) * p_ratio)
        .repeat_interleave(2, dim=1)
        .expand_as(q)
        .to(dtype)
    )
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
    torch.testing.assert_close(
        lse.cpu(), torch.full_like(lse.cpu(), math.log(seq)), atol=2e-6, rtol=0
    )
    v8, scales = reference.quantize_v_mx(v)
    reconstructed = v8.float() * scales.repeat_interleave(32, dim=-2)[..., :seq, :]
    torch.testing.assert_close(reconstructed, decoded, atol=0, rtol=0)


@pytest.mark.parametrize("seq", [32768, 65536, 131072])
def test_mxfp8_scale_cache_capacity(seq):
    # D128 needs 128/256/512 KiB of scales per head: first fits the L1 cache,
    # the latter two exercise per-tile scale loads through the same MX path.
    q = torch.zeros(1, 4, 128, 128, dtype=torch.float16)
    k = torch.zeros(1, 2, seq, 128, dtype=torch.float16)
    powers = torch.exp2(((torch.arange(seq) // 32) % 9 - 4).float()).view(1, 1, seq, 1)
    heads = torch.tensor([1.0, -2.0]).view(1, 2, 1, 1)
    v = (powers * heads).expand_as(k).contiguous().half()
    actual = sageattn(q.npu(), k.npu(), v.npu(), smooth_k=False)
    p_ratio = 448.0 / math.exp(torch.tensor(math.log(448), dtype=torch.float16).item())
    expected = (
        (v.float().mean(-2, keepdim=True) * p_ratio)
        .repeat_interleave(2, dim=1)
        .expand_as(q)
        .half()
    )
    torch.testing.assert_close(actual.cpu(), expected, atol=0.004, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("accum", ["fp32", "fp32+fp16", "fp32+fp32", None])
def test_smooth_v_warns_and_is_ignored(dtype, accum):
    torch.manual_seed(925)
    q = torch.randn(1, 2, 33, 65, device="npu", dtype=dtype)
    k = torch.randn(1, 1, 65, 65, device="npu", dtype=dtype)
    v = torch.randn_like(k) + 4
    with pytest.warns(UserWarning, match="smooth_v is ignored"):
        actual, lse = sageattn(
            q, k, v, smooth_v=True, pv_accum_dtype=accum, return_lse=True
        )
    expected, expected_lse = sageattn(
        q, k, v, smooth_v=False, pv_accum_dtype="fp32", return_lse=True
    )
    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=0, rtol=0)
    torch.testing.assert_close(lse.cpu(), expected_lse.cpu(), atol=0, rtol=0)


@pytest.mark.parametrize("entry", ["sageattn", "sageattn_qk_int8_pv_fp8_asc"])
@pytest.mark.parametrize("gran", ["per_thread", "per_warp"])
@pytest.mark.parametrize("accum", ["fp32", "fp32+fp16", "fp32+fp32", None, "invalid"])
def test_gpu_options_warn_and_fall_back(entry, gran, accum):
    import sageattention2

    torch.manual_seed(927)
    # A short sequence has deterministic K stats, so fallback must be exact.
    q = torch.randn(1, 4, 33, 64, device="npu", dtype=torch.float16)
    k = torch.randn(1, 2, 65, 64, device="npu", dtype=q.dtype)
    v = torch.randn_like(k)
    with pytest.warns(UserWarning) as caught:
        actual, lse = getattr(sageattention2, entry)(
            q, k, v, qk_quant_gran=gran, pv_accum_dtype=accum, return_lse=True
        )
    messages = [str(w.message) for w in caught]
    assert any(
        "qk_quant_gran=" in m and "falls back to per_token" in m for m in messages
    )
    assert any(
        "pv_accum_dtype=" in m and "falls back to fp32+fp32" in m for m in messages
    ) == (accum not in ("fp32+fp32", "fp32"))
    expected, expected_lse = sageattn(q, k, v, return_lse=True)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(lse, expected_lse, atol=0, rtol=0)


@pytest.mark.parametrize("entry", ["sageattn", "sageattn_qk_int8_pv_fp8_asc"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fp32_two_level_default_and_legacy_alias(entry, dtype):
    import warnings
    import sageattention2

    torch.manual_seed(928)
    q = torch.randn(1, 4, 33, 64, device="npu", dtype=dtype)
    k = torch.randn(1, 2, 65, 64, device="npu", dtype=dtype)
    v = torch.randn_like(k)
    call = getattr(sageattention2, entry)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        expected, expected_lse = call(q, k, v, return_lse=True)
        for accum in ("fp32+fp32", "fp32"):
            actual, lse = call(q, k, v, pv_accum_dtype=accum, return_lse=True)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            torch.testing.assert_close(lse, expected_lse, atol=0, rtol=0)
    assert not [w for w in caught if issubclass(w.category, UserWarning)]
