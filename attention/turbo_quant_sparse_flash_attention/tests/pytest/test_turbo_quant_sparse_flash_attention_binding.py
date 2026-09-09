# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Standard binding contract, Meta/FakeTensor, and ACL graph replay tests."""

import pytest
import torch
import torch_npu  # noqa: F401
import cann_ops_transformer


def metadata_inputs(dtype, layout, tokens=3, device="meta"):
    query = torch.empty((tokens, 8, 576), dtype=dtype, device=device)
    shape = (4, 1, 128, 386) if layout == "PA_BNBD" else (4, 128, 1, 386)
    cache = torch.empty(shape, dtype=torch.int8, device=device)
    indices = torch.empty((tokens, 1, 65), dtype=torch.int32, device=device)
    kwargs = dict(
        block_table=torch.empty((3, 2), dtype=torch.int32, device=device),
        actual_seq_lengths_query=torch.empty(3, dtype=torch.int32, device=device),
        actual_seq_lengths_kv=torch.empty(3, dtype=torch.int32, device=device),
        layout_kv=layout,
    )
    return [query, cache, cache, indices], kwargs


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["PA_BSND", "PA_BNBD"])
@pytest.mark.parametrize("tokens", [0, 3])
@pytest.mark.parametrize("lse", [False, True])
def test_meta_without_native_load(dtype, layout, tokens, lse, monkeypatch):
    from cann_ops_transformer.op_builder import OpBuilder

    def forbid_load(*args, **kwargs):
        raise AssertionError("Meta must not compile or load the native binding")

    monkeypatch.setattr(OpBuilder, "load", forbid_load)
    inputs, kwargs = metadata_inputs(dtype, layout, tokens)
    output, maximum, total = cann_ops_transformer.turbo_quant_sparse_flash_attention(
        *inputs, **kwargs, return_softmax_lse=lse
    )
    assert output.shape == (tokens, 8, 512) and output.dtype == dtype
    assert maximum.shape == total.shape == ((1, tokens, 8) if lse else (0,))
    assert maximum.dtype == total.dtype == torch.float32
    assert all(t.device.type == "meta" for t in (output, maximum, total))


@pytest.mark.parametrize(
    "override",
    [
        {"layout_query": "BSND"},
        {"layout_kv": "PA_BNSD"},
        {"tile_size": 64},
        {"rope_head_dim": 32},
        {"sparse_block_size": 3},
        {"sparse_mode": 1},
        {"key_quant_mode": 1},
        {"attention_mode": 0},
    ],
)
@pytest.mark.parametrize("tokens", [0, 3])
def test_meta_rejects_invalid_attributes(override, tokens):
    inputs, kwargs = metadata_inputs(torch.float16, "PA_BSND", tokens)
    kwargs.update(override)
    with pytest.raises(RuntimeError):
        cann_ops_transformer.turbo_quant_sparse_flash_attention(*inputs, **kwargs)


def test_meta_strided_cache_and_device_checks():
    inputs, kwargs = metadata_inputs(torch.bfloat16, "PA_BNBD")
    cache = torch.empty_strided(
        (4, 1, 128, 386), (100000, 49408, 386, 1), dtype=torch.int8, device="meta"
    )
    inputs[1] = inputs[2] = cache
    assert cann_ops_transformer.turbo_quant_sparse_flash_attention(*inputs, **kwargs)[
        0
    ].shape == (3, 8, 512)
    inputs[1] = torch.empty_strided(
        cache.shape, (100000, 49408, 387, 1), dtype=cache.dtype, device="meta"
    )
    with pytest.raises(RuntimeError, match="axis 0"):
        cann_ops_transformer.turbo_quant_sparse_flash_attention(*inputs, **kwargs)
    inputs[1] = cache
    kwargs["key_dequant_scale"] = torch.ones(1)
    with pytest.raises(RuntimeError, match="device"):
        cann_ops_transformer.turbo_quant_sparse_flash_attention(*inputs, **kwargs)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fake_tensor(dtype):
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        inputs, kwargs = metadata_inputs(dtype, "PA_BNBD", device="npu:0")
        output = cann_ops_transformer.turbo_quant_sparse_flash_attention(
            *inputs, **kwargs
        )[0]
        assert output.device == torch.device("npu:0") and output.dtype == dtype
        assert output.shape == (3, 8, 512)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["PA_BSND", "PA_BNBD"])
def test_acl_graph_replays_changed_inputs(dtype, layout):
    from test_npu_turbo_quant_sparse_flash_attention import build_case
    from test_npu_turbo_quant_sparse_flash_attention_strided import strided_cache

    case = build_case(
        3,
        [257, 0, 145],
        65,
        19001,
        dtype,
        "normal",
        q_len=[2, 1, 0],
        layout_kv=layout,
        shuffle_blocks=True,
    )
    case["kv"], storage = strided_cache(case["kv"], 257, 131)
    before = storage.clone()

    def call():
        return cann_ops_transformer.turbo_quant_sparse_flash_attention(
            case["q"],
            case["kv"],
            case["kv"],
            case["si"],
            block_table=case["bt"],
            actual_seq_lengths_query=case["asq"],
            actual_seq_lengths_kv=case["ask"],
            scale_value=case["scale_value"],
            layout_kv=layout,
            return_softmax_lse=True,
        )

    for _ in range(3):
        call()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = call()
    for zero_kv in (False, True, False):
        case["q"].mul_(0.5)
        case["ask"].copy_(
            torch.tensor(
                [0 if zero_kv else 257, 0, 145],
                dtype=torch.int32,
                device=case["q"].device,
            )
        )
        expected = call()
        graph.replay()
        torch.npu.synchronize()
        for actual, eager in zip(captured, expected):
            torch.testing.assert_close(actual, eager, rtol=0, atol=0)
        if zero_kv:
            assert torch.count_nonzero(captured[0]).item() == 0
            assert torch.count_nonzero(captured[2]).item() == 0
    torch.testing.assert_close(storage, before, rtol=0, atol=0)


def test_legacy_alias_calls_standard_operator():
    import custom_ops  # noqa: F401  optional compatibility package is tested explicitly
    from test_npu_turbo_quant_sparse_flash_attention import build_case

    case = build_case(1, 129, 65, 19002, torch.float16, "uniform", layout_kv="PA_BNBD")
    args = (case["q"], case["kv"], case["kv"], case["si"])
    kwargs = dict(
        block_table=case["bt"],
        actual_seq_lengths_query=case["asq"],
        actual_seq_lengths_kv=case["ask"],
        layout_kv="PA_BNBD",
    )
    standard = cann_ops_transformer.turbo_quant_sparse_flash_attention(*args, **kwargs)
    for alias in (
        torch.ops.custom.npu_turbo_quant_sparse_flash_attention,
        torch_npu.npu_turbo_quant_sparse_flash_attention,
    ):
        for actual, expected in zip(alias(*args, **kwargs), standard):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
