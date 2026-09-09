# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Execute TTK assets through the real standard interface and verify the byte-decoding golden."""

import importlib.util
from pathlib import Path

import pytest
import torch
import cann_ops_transformer

_path = Path(__file__).resolve().parents[1] / "assets" / "spec.py"
_spec = importlib.util.spec_from_file_location("tq_sfa_test_spec", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
SPEC = _module.TurboQuantSparseFlashAttentionSpec


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["PA_BSND", "PA_BNBD"])
@pytest.mark.parametrize("block_size", [16, 1024])
@pytest.mark.parametrize("sparse_block", [1, 2, 4, 8, 16])
def test_assets_quantized_bytes(dtype, layout, block_size, sparse_block):
    from test_npu_turbo_quant_sparse_flash_attention_strided import strided_cache

    q = torch.empty((3, 8, 576), dtype=dtype)
    shape = (4, 1, block_size, 386) if layout == "PA_BNBD" else (4, block_size, 1, 386)
    kv = torch.empty(shape, dtype=torch.int8)
    indices = torch.empty((3, 1, 65), dtype=torch.int32)
    kwargs = dict(
        block_table=torch.empty((3, 2), dtype=torch.int32),
        actual_seq_lengths_query=torch.empty(3, dtype=torch.int32),
        actual_seq_lengths_kv=torch.empty(3, dtype=torch.int32),
        layout_kv=layout,
        sparse_block_size=sparse_block,
        sparse_mode=0,
        scale_value=512**-0.5,
        return_softmax_lse=True,
    )
    SPEC.customize_inputs(
        q,
        kv,
        kv,
        indices,
        **kwargs,
        testcase_name=f"assets-{block_size}-{sparse_block}",
    )
    expected = SPEC.golden(q, kv, kv, indices, **kwargs)
    cache, storage = strided_cache(kv.npu(), 257, 131)
    before = storage.clone()
    npu_kwargs = {
        name: value.npu() if isinstance(value, torch.Tensor) else value
        for name, value in kwargs.items()
    }
    actual = cann_ops_transformer.turbo_quant_sparse_flash_attention(
        q.npu(), cache, cache, indices.npu(), **npu_kwargs
    )
    torch.npu.synchronize()
    metrics = SPEC.compare(*actual, *expected)
    assert all(result["pass"] for result in metrics), metrics
    torch.testing.assert_close(storage, before, rtol=0, atol=0)
    faulty = (actual[0] + 1, *actual[1:])
    assert not SPEC.compare(*faulty, *expected)[0]["pass"], (
        "comparison must reject a corrupted NPU output"
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["PA_BSND", "PA_BNBD"])
@pytest.mark.parametrize("kv_length", [8192, 32768])
def test_long_kv_against_two_references(dtype, layout, kv_length):
    from test_npu_turbo_quant_sparse_flash_attention import build_case, golden, check

    case = build_case(
        1, kv_length, 129, 19003, dtype, "normal", layout_kv=layout, shuffle_blocks=True
    )
    kwargs = dict(
        block_table=case["bt"],
        actual_seq_lengths_query=case["asq"],
        actual_seq_lengths_kv=case["ask"],
        layout_kv=layout,
        scale_value=case["scale_value"],
    )
    args = (case["q"], case["kv"], case["kv"], case["si"])
    expected = SPEC.golden(*args, **kwargs)
    torch.testing.assert_close(
        expected[0], torch.from_numpy(golden(case)), rtol=1e-12, atol=1e-12
    )
    actual = cann_ops_transformer.turbo_quant_sparse_flash_attention(*args, **kwargs)
    torch.npu.synchronize()
    ok, description = check(actual[0].float().cpu().numpy(), expected[0].numpy(), dtype)
    assert ok, description
