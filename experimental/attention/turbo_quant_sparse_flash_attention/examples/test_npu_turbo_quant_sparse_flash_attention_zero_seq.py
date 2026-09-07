# ---------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------------------------------------

"""Empty-query and mixed-batch tests through the native ACLNN extension."""

import numpy as np
import pytest
import torch

from test_npu_turbo_quant_sparse_flash_attention import (
    DN,
    INT64_MAX,
    build_case,
    check,
    golden,
)


def call_sfa(case, return_lse, sparse_mode):
    return torch.ops.custom.npu_turbo_quant_sparse_flash_attention(
        case["q"],
        case["kv"],
        case["kv"],
        case["si"],
        key_dequant_scale=None,
        value_dequant_scale=None,
        block_table=case["bt"],
        actual_seq_lengths_query=case["asq"],
        actual_seq_lengths_kv=case["ask"],
        scale_value=case["scale_value"],
        key_quant_mode=3,
        value_quant_mode=3,
        sparse_block_size=1,
        layout_query="TND",
        layout_kv=case["layout_kv"],
        sparse_mode=sparse_mode,
        pre_tokens=INT64_MAX,
        next_tokens=INT64_MAX,
        attention_mode=2,
        quant_scale_repo_mode=1,
        tile_size=128,
        rope_head_dim=64,
        return_softmax_lse=return_lse,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["PA_BSND", "PA_BNBD"])
@pytest.mark.parametrize("return_lse", [False, True])
@pytest.mark.parametrize("heads", [1, 8, 128])
@pytest.mark.parametrize("kv_lengths", [[0, 0, 0], [17, 0, 29]])
def test_empty_query(dtype, layout, return_lse, heads, kv_lengths):
    case = build_case(
        3,
        kv_lengths,
        32,
        12001,
        dtype,
        "normal",
        q_len=0,
        query_heads=heads,
        layout_kv=layout,
    )
    output, maximum, total = call_sfa(case, return_lse, 3)
    torch.npu.synchronize()
    assert output.shape == (0, heads, DN) and output.dtype == dtype
    assert maximum.shape == total.shape == ((1, 0, heads) if return_lse else (0,))
    assert maximum.dtype == total.dtype == torch.float32
    assert output.numel() == maximum.numel() == total.numel() == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["PA_BSND", "PA_BNBD"])
@pytest.mark.parametrize("return_lse", [False, True])
@pytest.mark.parametrize("sparse_mode", [0, 3])
@pytest.mark.parametrize("heads", [8, 128])
@pytest.mark.parametrize("pattern", ["edges", "all_zero_kv", "pipeline_tail"])
def test_mixed_empty_sequences(dtype, layout, return_lse, sparse_mode, heads, pattern):
    if pattern == "edges":
        q_lengths, kv_lengths = [0, 2, 0, 1, 0], [0, 17, 29, 0, 32]
    elif pattern == "all_zero_kv":
        q_lengths, kv_lengths = [1, 0, 2, 0], [0, 0, 0, 0]
    else:
        # More query rows than A2 AICs force tasks to share a pipeline, with empty tails.
        q_lengths = [0] + [1] * 65 + [0]
        kv_lengths = [17] + [17 if i % 2 == 0 else 0 for i in range(64)] + [0, 0]
    case = build_case(
        len(q_lengths),
        kv_lengths,
        32,
        12002,
        dtype,
        "normal",
        q_len=q_lengths,
        query_heads=heads,
        layout_kv=layout,
        shuffle_blocks=True,
        sparse_mode=sparse_mode,
    )
    expected = golden(case)
    output, maximum, total = call_sfa(case, return_lse, sparse_mode)
    actual = output.float().cpu().numpy()
    passed, detail = check(actual, expected, dtype)
    assert passed, detail
    assert np.isfinite(actual).all()
    zero_rows = np.repeat(np.asarray(kv_lengths) == 0, q_lengths)
    np.testing.assert_array_equal(actual[zero_rows], 0)
    if return_lse:
        maximum, total = maximum.cpu(), total.cpu()
        assert maximum.shape == total.shape == (1, sum(q_lengths), heads)
        assert torch.equal(total[:, zero_rows], torch.zeros_like(total[:, zero_rows]))
        assert torch.equal(
            maximum[:, zero_rows], torch.full_like(maximum[:, zero_rows], -2e38)
        )
        assert torch.isfinite(maximum).all() and torch.isfinite(total).all()
        assert (total[:, ~zero_rows] > 0).all()
    else:
        assert maximum.shape == total.shape == (0,)


@pytest.mark.parametrize(
    "bad_input", ["heads", "sparse_dtype", "sparse_tokens", "block_columns"]
)
def test_empty_query_does_not_bypass_validation(bad_input):
    case = build_case(1, 0, 32, 12003, torch.float16, "normal", q_len=0)
    if bad_input == "heads":
        case["q"] = case["q"][:, :0]
    elif bad_input == "sparse_dtype":
        case["si"] = case["si"].long()
    elif bad_input == "sparse_tokens":
        case["si"] = torch.zeros((1, 1, 32), dtype=torch.int32).npu()
    else:
        case["bt"] = case["bt"][:, :0]
    with pytest.raises(RuntimeError):
        call_sfa(case, True, 3)
