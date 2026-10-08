#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""A2 SparseFlashAttention Split-KV numerical and graph regression tests."""

import math
import os
from pathlib import Path

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")


@pytest.fixture(scope="module")
def sfa_operator():
    extension_dir = os.environ.get("SFA_TEST_EXTENSION_DIR")
    if not extension_dir:
        return torch_npu.npu_sparse_flash_attention

    from torch.utils.cpp_extension import load

    root = Path(extension_dir)
    npu_root = Path(torch_npu.__file__).parent
    cann_root = Path(os.environ["ASCEND_HOME_PATH"])
    module = load(
        name="sfa_ordinary_lse_compat_probe",
        sources=[
            str(root / "sparse_flash_attention.cpp"),
            str(root / "extension_hash.cpp"),
        ],
        extra_include_paths=[
            str(cann_root / "include"),
            str(cann_root / "include/aclnnop"),
            str(npu_root / "include"),
            str(npu_root / "include/third_party/hccl/inc"),
            str(npu_root / "include/third_party/op-plugin"),
            str(root),
        ],
        extra_cflags=["-O2"],
        extra_ldflags=[
            f"-L{cann_root}/lib64",
            "-lascendcl",
            f"-L{npu_root}/lib",
            "-ltorch_npu",
        ],
        verbose=True,
    )

    def run(q, k, v, indices, scale, **kwargs):
        return module.sparse_flash_attention(
            q,
            k,
            v,
            indices,
            scale,
            kwargs["block_table"],
            kwargs["actual_seq_lengths_query"],
            kwargs["actual_seq_lengths_kv"],
            kwargs.get("query_rope"),
            kwargs.get("key_rope"),
            None,
            1,
            "TND",
            "PA_BSND",
            0,
            9223372036854775807,
            9223372036854775807,
            2,
            kwargs["return_softmax_lse"],
        )

    return run


def _run_reference_case(
    sfa_operator,
    query_count,
    batch_count,
    topk,
    valid_count,
    replay_graph,
    return_lse,
    heads,
    dtype=torch.bfloat16,
    use_rope=True,
    row_valid_counts=None,
    seed=20260929,
    actual_kv_length=4096,
    query_counts=None,
):
    if "910B" not in torch.npu.get_device_name(0):
        pytest.skip("This test requires Ascend 910B")

    generator = torch.Generator().manual_seed(seed + query_count + batch_count)
    kv_length, page_size = 4096, 128
    if query_counts is None:
        query_counts = [1] * (batch_count - 1) + [query_count - batch_count + 1]
    query_ends = [sum(query_counts[: idx + 1]) for idx in range(batch_count)]
    query = torch.randn(query_count, heads, 512, generator=generator).to(dtype).npu()
    query_rope = (
        torch.randn(query_count, heads, 64, generator=generator).to(dtype).npu()
        if use_rope
        else None
    )
    key = torch.randn(batch_count * kv_length, 512, generator=generator).to(dtype).npu()
    key_rope = (
        torch.randn(batch_count * kv_length, 64, generator=generator).to(dtype).npu()
        if use_rope
        else None
    )
    key_pages = key.reshape(-1, page_size, 1, 512)
    rope_pages = key_rope.reshape(-1, page_size, 1, 64) if use_rope else None
    if row_valid_counts is None:
        row_valid_counts = [valid_count] * query_count
    selected = torch.stack(
        [
            torch.nn.functional.pad(
                torch.randperm(actual_kv_length, generator=generator)[:count],
                (0, topk - count),
                value=-1,
            )
            for count in row_valid_counts
        ]
    ).int()
    indices = selected.unsqueeze(1).npu()
    block_table = torch.arange(batch_count * kv_length // page_size, dtype=torch.int32)
    block_table = block_table.reshape(batch_count, kv_length // page_size).npu()
    query_lengths = torch.tensor(query_ends, dtype=torch.int32).npu()
    kv_lengths = torch.full((batch_count,), actual_kv_length, dtype=torch.int32).npu()
    scale = 1.0 / math.sqrt(576 if use_rope else 512)

    def run_operator():
        return sfa_operator(
            query,
            key_pages,
            key_pages,
            indices,
            scale,
            block_table=block_table,
            actual_seq_lengths_query=query_lengths,
            actual_seq_lengths_kv=kv_lengths,
            query_rope=query_rope,
            key_rope=rope_pages,
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=0,
            attention_mode=2,
            return_softmax_lse=return_lse,
        )

    if replay_graph:
        run_operator()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual = run_operator()
        graph.replay()
    else:
        actual = run_operator()
    torch.npu.synchronize()
    actual, max_out, sum_out = actual
    assert actual.dtype == dtype
    assert torch.isfinite(actual).all()

    expected = []
    expected_lse = []
    expected_max = []
    expected_sum = []
    for row in range(query_count):
        count = row_valid_counts[row]
        if count == 0:
            expected.append(torch.zeros_like(actual[row], dtype=torch.float32))
            expected_lse.append(torch.full((heads,), -math.inf, device=query.device))
            expected_max.append(torch.zeros(heads, device=query.device))
            expected_sum.append(torch.zeros(heads, device=query.device))
            continue
        batch_idx = next(idx for idx, end in enumerate(query_ends) if row < end)
        kv_positions = batch_idx * kv_length + indices[row, 0, :count].long()
        q = query[row].float()
        k = key[kv_positions].float()
        if use_rope:
            q = torch.cat((q, query_rope[row].float()), dim=-1)
            k = torch.cat((k, key_rope[kv_positions].float()), dim=-1)
        scores = torch.matmul(q, k.T).mul_(scale)
        weights = scores.softmax(dim=-1)
        row_max = scores.max(dim=-1).values
        expected_max.append(row_max)
        expected_sum.append((scores - row_max[:, None]).exp().sum(dim=-1))
        expected_lse.append(scores.logsumexp(dim=-1))
        expected.append(torch.matmul(weights, key[kv_positions].float()))
    reference = torch.stack(expected)
    relative_l2 = (actual.float() - reference).norm() / reference.norm().clamp_min(
        1e-20
    )
    tolerance = 0.02 if dtype == torch.bfloat16 else 0.005
    assert relative_l2.item() < tolerance, f"relative L2 error: {relative_l2.item()}"
    for row, count in enumerate(row_valid_counts):
        if count == 0:
            assert torch.count_nonzero(actual[row]).item() == 0
        else:
            row_error = (actual[row].float() - reference[row]).norm() / reference[
                row
            ].norm()
            assert row_error.item() < tolerance, (
                f"row={row}, relative L2={row_error.item()}"
            )
    if return_lse:
        assert tuple(max_out.shape) == (1, query_count, heads)
        assert max_out.dtype == sum_out.dtype == torch.float32
        torch.testing.assert_close(
            max_out[0], torch.stack(expected_max), rtol=0.001, atol=0.03
        )
        torch.testing.assert_close(
            sum_out[0], torch.stack(expected_sum), rtol=tolerance, atol=0.03
        )
        for row, count in enumerate(row_valid_counts):
            if count == 0:
                assert torch.count_nonzero(max_out[0, row]).item() == 0
                assert torch.count_nonzero(sum_out[0, row]).item() == 0
        lse = max_out[0] + sum_out[0].log()
        torch.testing.assert_close(
            lse, torch.stack(expected_lse), rtol=0.001, atol=0.03
        )


@pytest.mark.parametrize(
    "query_count,batch_count",
    [(1, 1), (4, 1), (4, 2), (4, 4), (8, 1), (8, 2), (8, 4), (8, 8)],
)
@pytest.mark.parametrize(
    "topk,valid_count", [(2048, 256), (1024, 1024), (1536, 1536), (2048, 2048)]
)
@pytest.mark.parametrize("replay_graph", [False, True])
@pytest.mark.parametrize("return_lse", [False, True])
@pytest.mark.parametrize("heads", [8, 64])
def test_split_kv_arch22(
    sfa_operator,
    query_count,
    batch_count,
    topk,
    valid_count,
    replay_graph,
    return_lse,
    heads,
):
    _run_reference_case(
        sfa_operator,
        query_count,
        batch_count,
        topk,
        valid_count,
        replay_graph,
        return_lse,
        heads,
    )


@pytest.mark.parametrize("heads", [1, 2, 4, 8, 16, 32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("use_rope", [False, True], ids=["no_rope", "rope64"])
@pytest.mark.parametrize("return_lse", [False, True])
@pytest.mark.parametrize("replay_graph", [False, True])
@pytest.mark.parametrize("row_kind", ["full", "tail", "empty", "mixed"])
def test_general_heads_dtype_rope(
    sfa_operator, heads, dtype, use_rope, return_lse, replay_graph, row_kind
):
    counts = {
        "full": [2048] * 4,
        "tail": [513] * 4,
        "empty": [0] * 4,
        "mixed": [2048, 513, 0, 1],
    }[row_kind]
    _run_reference_case(
        sfa_operator,
        4,
        2,
        2048,
        counts[0],
        replay_graph,
        return_lse,
        heads,
        dtype,
        use_rope,
        counts,
    )


@pytest.mark.parametrize("heads", [8, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("use_rope", [False, True], ids=["no_rope", "rope64"])
@pytest.mark.parametrize("return_lse", [False, True])
@pytest.mark.parametrize("replay_graph", [False, True])
def test_large_query_fallback(
    sfa_operator, heads, dtype, use_rope, return_lse, replay_graph
):
    _run_reference_case(
        sfa_operator,
        32,
        4,
        2048,
        0,
        replay_graph,
        return_lse,
        heads,
        dtype,
        use_rope,
        [2048, 513, 0, 1] * 8,
    )


@pytest.mark.parametrize("seed", [7, 101, 20261008])
@pytest.mark.parametrize("heads", [1, 8, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("use_rope", [False, True], ids=["no_rope", "rope64"])
def test_padding_random_seeds(sfa_operator, seed, heads, dtype, use_rope):
    _run_reference_case(
        sfa_operator,
        4,
        2,
        2048,
        2048,
        False,
        True,
        heads,
        dtype,
        use_rope,
        [2048, 511, 0, 1],
        seed,
    )


@pytest.mark.parametrize("heads", [1, 8, 128])
@pytest.mark.parametrize("replay_graph", [False, True])
@pytest.mark.parametrize("valid_count", [0, 1, 511, 512, 513, 1023, 1024, 1025, 2047])
def test_padding_block_boundaries(sfa_operator, heads, replay_graph, valid_count):
    _run_reference_case(
        sfa_operator, 1, 1, 2048, valid_count, replay_graph, True, heads
    )


@pytest.mark.parametrize("heads", [1, 2, 4, 128])
@pytest.mark.parametrize("replay_graph", [False, True])
@pytest.mark.parametrize(
    "topk,counts", [(128, [128, 1, 0, 127]), (2048, [256, 1, 0, 255])]
)
def test_short_kv_and_empty_query_batch(
    sfa_operator, heads, replay_graph, topk, counts
):
    _run_reference_case(
        sfa_operator,
        4,
        3,
        topk,
        counts[0],
        replay_graph,
        True,
        heads,
        row_valid_counts=counts,
        actual_kv_length=256,
        query_counts=[0, 1, 3],
    )
