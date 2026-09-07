# ---------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------------------------------------

"""Native ACLNN tests for first-axis-strided TurboQuant cache views."""

import numpy as np
import pytest
import torch

from test_npu_turbo_quant_sparse_flash_attention import (
    INT64_MAX,
    build_case,
    check,
    golden,
)


def strided_cache(cache, gap_bytes, storage_offset=0):
    page_bytes = cache[0].numel()
    stride0 = page_bytes + gap_bytes
    storage = torch.full(
        (storage_offset + (cache.shape[0] - 1) * stride0 + page_bytes + 64,),
        113,
        dtype=torch.int8,
        device=cache.device,
    )
    view = storage.as_strided(
        cache.shape, (stride0, *cache.stride()[1:]), storage_offset
    )
    view.copy_(cache)
    return view, storage


def call_sfa(case, sparse_block_size=1, return_lse=True, value=None):
    return torch.ops.custom.npu_turbo_quant_sparse_flash_attention(
        case["q"],
        case["kv"],
        case["kv"] if value is None else value,
        case["si"],
        key_dequant_scale=None,
        value_dequant_scale=None,
        block_table=case["bt"],
        actual_seq_lengths_query=case["asq"],
        actual_seq_lengths_kv=case["ask"],
        scale_value=case["scale_value"],
        key_quant_mode=3,
        value_quant_mode=3,
        sparse_block_size=sparse_block_size,
        layout_query="TND",
        layout_kv=case["layout_kv"],
        sparse_mode=0,
        pre_tokens=INT64_MAX,
        next_tokens=INT64_MAX,
        attention_mode=2,
        quant_scale_repo_mode=1,
        tile_size=128,
        rope_head_dim=64,
        return_softmax_lse=return_lse,
    )


def make_case(dtype, layout, heads, sparse_block_size):
    case = build_case(
        3,
        [257, 0, 145],
        64,
        17001,
        dtype,
        "normal",
        q_len=[2, 1, 0],
        query_heads=heads,
        layout_kv=layout,
        shuffle_blocks=True,
        sparse_mode=0,
    )
    block_count = (257 + sparse_block_size - 1) // sparse_block_size
    chosen = np.unique(
        np.linspace(0, block_count - 1, min(64, block_count), dtype=np.int32)
    )[::-1].copy()
    indices = np.full((3, 64), -1, dtype=np.int32)
    indices[:2, : chosen.size] = chosen
    case["si"] = torch.from_numpy(indices).reshape(3, 1, 64).npu()
    expanded = np.asarray(
        [
            block * sparse_block_size + offset
            for block in chosen
            for offset in range(sparse_block_size)
            if block * sparse_block_size + offset < 257
        ],
        dtype=np.int32,
    )
    case["idx"] = np.full((3, expanded.size), -1, dtype=np.int32)
    case["idx"][:2] = expanded
    case["nvalid"] = [expanded.size, expanded.size, 0]
    return case


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["PA_BSND", "PA_BNBD"])
@pytest.mark.parametrize("heads", [8, 128])
@pytest.mark.parametrize("sparse_block_size", [1, 2, 16])
@pytest.mark.parametrize("spacing", ["step2", "step3_offset", "padding_offset"])
def test_strided_cache_matches_contiguous_and_golden(
    dtype, layout, heads, sparse_block_size, spacing
):
    case = make_case(dtype, layout, heads, sparse_block_size)
    expected = golden(case)
    return_lse = heads == 128 or sparse_block_size != 1
    reference = call_sfa(case, sparse_block_size, return_lse)
    page_bytes = case["kv"][0].numel()
    gap, offset = {
        "step2": (page_bytes, 0),
        "step3_offset": (2 * page_bytes, 17),
        "padding_offset": (256, 32),
    }[spacing]
    case["kv"], storage = strided_cache(case["kv"], gap, offset)
    assert not case["kv"].is_contiguous()
    before = storage.clone()
    result = call_sfa(case, sparse_block_size, return_lse)
    for actual, contiguous in zip(result, reference):
        torch.testing.assert_close(actual.cpu(), contiguous.cpu(), rtol=0, atol=0)
    passed, detail = check(result[0].float().cpu().numpy(), expected, dtype)
    assert passed, detail
    assert torch.equal(storage, before)
    assert torch.count_nonzero(result[0][2]).item() == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["PA_BSND", "PA_BNBD"])
@pytest.mark.parametrize("tokens", [0, 1])
def test_strided_cache_empty_sequences(dtype, layout, tokens):
    case = build_case(
        2, [17, 0], 16, 17002, dtype, "normal", q_len=[0, tokens], layout_kv=layout
    )
    case["kv"], storage = strided_cache(case["kv"], 256, 19)
    output, maximum, total = call_sfa(case)
    torch.npu.synchronize()
    assert output.shape == (tokens, 8, 512)
    assert torch.count_nonzero(output).item() == 0
    assert maximum.shape == total.shape == (1, tokens, 8)
    assert torch.equal(maximum, torch.full_like(maximum, -2e38))
    assert torch.count_nonzero(total).item() == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["PA_BSND", "PA_BNBD"])
@pytest.mark.parametrize("tokens", [0, 1])
@pytest.mark.parametrize("which", ["key", "value"])
@pytest.mark.parametrize("kind", ["token_stride", "byte_stride", "overlap"])
def test_rejects_page_internal_noncontiguity(dtype, layout, tokens, which, kind):
    case = build_case(
        1, 257, 16, 17003, dtype, "normal", q_len=tokens, layout_kv=layout
    )
    strides = list(case["kv"].stride())
    if kind == "token_stride":
        strides[2 if layout == "PA_BNBD" else 1] += 1
        strides[0] *= 2
    elif kind == "byte_stride":
        strides = [stride * 2 for stride in strides]
    else:
        strides[0] -= 1
    bad = torch.empty_strided(
        case["kv"].shape, strides, dtype=torch.int8, device=case["kv"].device
    )
    value = case["kv"]
    if which == "key":
        case["kv"] = bad
    else:
        value = bad
    with pytest.raises(RuntimeError, match="axis 0|overlap"):
        call_sfa(case, value=value)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["PA_BSND", "PA_BNBD"])
def test_single_page_ignores_singleton_strides(dtype, layout):
    case = build_case(1, 17, 16, 17004, dtype, "normal", layout_kv=layout)
    reference = call_sfa(case)
    strides = list(case["kv"].stride())
    strides[0] = 0
    strides[1 if layout == "PA_BNBD" else 2] = 7
    case["kv"] = case["kv"].as_strided(case["kv"].shape, strides)
    for actual, expected in zip(call_sfa(case), reference):
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["PA_BSND", "PA_BNBD"])
def test_page_stride_beyond_dma_limit(dtype, layout):
    torch.npu.empty_cache()
    free_bytes, _ = torch.npu.mem_get_info()
    stride0 = (1 << 31) + 49664
    if free_bytes < stride0 + (3 << 30):
        pytest.skip("Large-stride test needs 2 GiB plus 3 GiB of device headroom")
    case = build_case(
        1, 256, 2, 17005, dtype, "normal", layout_kv=layout, shuffle_blocks=True
    )
    case["si"] = torch.tensor([[[0, 128]]], dtype=torch.int32, device=case["kv"].device)
    reference = call_sfa(case)
    page_bytes = case["kv"][0].numel()
    storage = torch.empty(
        (32 + stride0 + page_bytes,), dtype=torch.int8, device=case["kv"].device
    )
    cache = storage.as_strided(
        case["kv"].shape, (stride0, *case["kv"].stride()[1:]), 32
    )
    # Copy individual pages so fixture setup does not require a large-stride DMA.
    for page in range(2):
        cache[page].copy_(case["kv"][page])
    case["kv"] = cache
    for actual, expected in zip(call_sfa(case), reference):
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
