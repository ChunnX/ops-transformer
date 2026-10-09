#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""A5 SparseFlashAttention Split-KV output and graph regression tests."""

import math

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from sfa_test_operator import get_sfa_operator
from bench_sfa_split_kv_sweeps_arch35 import cases


@pytest.mark.parametrize(
    ("query_dtype", "kv_dtype"),
    [(torch.float16, torch.bfloat16), (torch.bfloat16, torch.float16)],
    ids=["fp16-bf16", "bf16-fp16"],
)
@pytest.mark.parametrize("sparse_mode", [0, 3])
def test_mixed_query_kv_types_rejected(query_dtype, kv_dtype, sparse_mode):
    if "950" not in torch.npu.get_device_name(0):
        pytest.skip("This regression targets Ascend 950")
    query = torch.zeros((1, 8, 512), dtype=query_dtype, device="npu")
    query_rope = torch.zeros((1, 8, 64), dtype=query_dtype, device="npu")
    key = torch.zeros((32, 128, 1, 512), dtype=kv_dtype, device="npu")
    key_rope = torch.zeros((32, 128, 1, 64), dtype=kv_dtype, device="npu")
    indices = torch.arange(2048, dtype=torch.int32, device="npu").reshape(1, 1, -1)
    block_table = torch.arange(32, dtype=torch.int32, device="npu").reshape(1, -1)
    query_lengths = torch.tensor([1], dtype=torch.int32, device="npu")
    kv_lengths = torch.tensor([4096], dtype=torch.int32, device="npu")
    # OpDef checks the complete dtype combination before launching a kernel.
    with pytest.raises(RuntimeError, match="dtype or format.*inconsistent.*OpDef"):
        torch_npu.npu_sparse_flash_attention(
            query,
            key,
            key,
            indices,
            1.0 / math.sqrt(576),
            block_table=block_table,
            actual_seq_lengths_query=query_lengths,
            actual_seq_lengths_kv=kv_lengths,
            query_rope=query_rope,
            key_rope=key_rope,
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=sparse_mode,
            attention_mode=2,
            return_softmax_lse=False,
        )


@pytest.mark.parametrize(
    ("batch_count", "query_count", "topk_count", "valid_count"),
    [(1, q, 2048, valid) for q in (1, 4) for valid in (256, 2048, "mixed")]
    + [(1, q, topk, topk) for q in (1, 4) for topk in (256, 512, 768, 1000, 1024, 1536)]
    + [(1, q, topk, topk) for q, topk in ((2, 768), (3, 1024), (8, 2048), (16, 2048))]
    + [
        (b, q, 2048, valid)
        for b, q, valid in ((2, 2, 2048), (2, 4, 2048), (3, 4, 2048), (2, 4, "mixed"))
    ]
    + [(2, 4, topk, topk) for topk in (512, 1024, 1536)]
    + [(3, 4, 1024, 1024)],
)
@pytest.mark.parametrize("replay_graph", [False, True])
@pytest.mark.parametrize("sparse_mode", [0, 3])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("return_lse", [False, True], ids=["no-lse", "lse"])
@pytest.mark.parametrize("rope_dim", [0, 64], ids=["no-rope", "rope"])
def test_split_kv_pa_bsnd(
    batch_count,
    query_count,
    topk_count,
    valid_count,
    replay_graph,
    sparse_mode,
    dtype,
    return_lse,
    rope_dim,
    heads=8,
    kv_length=None,
    page_size=128,
):
    if "950" not in torch.npu.get_device_name(0):
        pytest.skip("This tiling is only selected on Ascend 950")
    mixed_rows = valid_count in ("mixed", "mixed-empty")
    if mixed_rows and query_count == 1:
        pytest.skip("Mixed split/fallback requires multiple query rows")

    seed = 1536 if mixed_rows else valid_count
    generator = torch.Generator().manual_seed(
        20260928 + query_count + seed + (batch_count - 1) * 1000
    )
    kv_length = max(4096, 2 * topk_count) if kv_length is None else kv_length
    queries_per_batch, remainder = divmod(query_count, batch_count)
    query_counts = [
        queries_per_batch + (index < remainder) for index in range(batch_count)
    ]
    query_ends = [sum(query_counts[: index + 1]) for index in range(batch_count)]
    query = torch.randn(query_count, heads, 512, generator=generator).to(dtype).npu()
    query_rope = (
        torch.randn(query_count, heads, rope_dim, generator=generator).to(dtype).npu()
        if rope_dim
        else None
    )
    key_flat = (
        torch.randn(batch_count * kv_length, 512, generator=generator).to(dtype).npu()
    )
    rope_flat = (
        torch.randn(batch_count * kv_length, rope_dim, generator=generator)
        .to(dtype)
        .npu()
        if rope_dim
        else None
    )
    key_pages = key_flat.reshape(-1, page_size, 1, 512)
    rope_pages = rope_flat.reshape(-1, page_size, 1, rope_dim) if rope_dim else None
    index_rows = []
    for row in range(query_count):
        row_count = (
            (topk_count if row == 0 else (0 if valid_count == "mixed-empty" else 256))
            if mixed_rows
            else seed
        )
        selected = torch.randperm(kv_length - query_count, generator=generator)[
            :row_count
        ].int()
        index_rows.append(
            torch.nn.functional.pad(selected, (0, topk_count - row_count), value=-1)
        )
    indices = torch.stack(index_rows).unsqueeze(1).npu()
    block_table = torch.arange(batch_count * kv_length // page_size, dtype=torch.int32)
    block_table = block_table.reshape(batch_count, kv_length // page_size).npu()
    query_lengths = torch.tensor(query_ends, dtype=torch.int32).npu()
    kv_lengths = torch.full((batch_count,), kv_length, dtype=torch.int32).npu()
    scale = 1.0 / math.sqrt(512 + rope_dim)

    sfa_operator = get_sfa_operator()

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
            sparse_mode=sparse_mode,
            attention_mode=2,
            return_softmax_lse=return_lse,
        )

    if replay_graph:
        run_operator()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            outputs = run_operator()
        graph.replay()
    else:
        outputs = run_operator()

    torch.npu.synchronize()
    actual, actual_max, actual_sum = outputs
    assert actual.dtype == dtype
    checked_rows = query_count
    assert torch.isfinite(actual[:checked_rows]).all()
    if return_lse:
        assert actual_max.shape == actual_sum.shape == (1, query_count, heads)
        assert actual_max.dtype == actual_sum.dtype == torch.float32
        assert torch.isfinite(actual_max[:, :checked_rows]).all()
        assert torch.isfinite(actual_sum[:, :checked_rows]).all()
    expected, expected_max, expected_sum = [], [], []
    for row in range(checked_rows):
        batch_index = next(index for index, end in enumerate(query_ends) if row < end)
        selected = indices[row, 0, :topk_count].long()
        # The documented -1 suffix terminates the valid prefix; it is not an
        # instruction to skip one index and resume attention after the sentinel.
        invalid = (selected < 0).nonzero()
        if invalid.numel():
            selected = selected[: invalid[0, 0].item()]
        if selected.numel() == 0:
            expected.append(torch.zeros_like(query[row], dtype=torch.float32))
            expected_max.append(torch.zeros(heads, dtype=torch.float32, device="npu"))
            expected_sum.append(torch.zeros(heads, dtype=torch.float32, device="npu"))
            continue
        kv_positions = batch_index * kv_length + selected
        q = query[row].float()
        k = key_flat[kv_positions].float()
        if rope_dim:
            q = torch.cat((q, query_rope[row].float()), dim=-1)
            k = torch.cat((k, rope_flat[kv_positions].float()), dim=-1)
        scores = torch.matmul(q, k.T) * scale
        weights = scores.softmax(dim=-1)
        expected.append(torch.matmul(weights, key_flat[kv_positions].float()))
        maximum = scores.max(dim=-1).values
        expected_max.append(maximum)
        expected_sum.append((scores - maximum[:, None]).exp().sum(dim=-1))
    reference = torch.stack(expected)
    difference = actual[:checked_rows].float() - reference
    relative_l2 = difference.norm() / reference.norm().clamp_min(1e-12)
    tolerance = 0.02 if dtype == torch.bfloat16 else 0.005
    assert relative_l2.item() < tolerance, f"relative L2 error: {relative_l2.item()}"
    if return_lse:
        # QK accumulation is not bitwise identical to the FP32 reference.
        # Check both interface statistics, not only attentionOut/logsumexp.
        stats_tolerance = 0.02 if dtype == torch.bfloat16 else 0.005
        torch.testing.assert_close(
            actual_max[0, :checked_rows],
            torch.stack(expected_max),
            rtol=stats_tolerance,
            atol=stats_tolerance,
        )
        torch.testing.assert_close(
            actual_sum[0, :checked_rows],
            torch.stack(expected_sum),
            rtol=stats_tolerance,
            atol=stats_tolerance,
        )


@pytest.mark.parametrize(
    "heads",
    [1, 2, 3, 4, 7, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 66, 95, 96, 97, 127, 128],
)
@pytest.mark.parametrize(
    ("batch_count", "query_count", "topk_count"),
    [(1, 1, 256), (1, 1, 2048), (2, 4, 2048)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("sparse_mode", [0, 3])
@pytest.mark.parametrize("replay_graph", [False, True])
@pytest.mark.parametrize("return_lse", [False, True], ids=["no-lse", "lse"])
@pytest.mark.parametrize("rope_dim", [0, 64], ids=["no-rope", "rope"])
def test_split_kv_head_counts(
    heads,
    batch_count,
    query_count,
    topk_count,
    dtype,
    sparse_mode,
    replay_graph,
    return_lse,
    rope_dim,
):
    test_split_kv_pa_bsnd(
        batch_count,
        query_count,
        topk_count,
        topk_count,
        replay_graph,
        sparse_mode,
        dtype,
        return_lse,
        rope_dim,
        heads,
    )


@pytest.mark.parametrize("heads", range(1, 129))
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("rope_dim", [0, 64], ids=["no-rope", "rope"])
def test_split_kv_all_head_counts(heads, dtype, rope_dim):
    test_split_kv_pa_bsnd(1, 1, 2048, 2048, False, 0, dtype, True, rope_dim, heads)


@pytest.mark.parametrize("heads", [1, 3, 8, 64, 65, 97, 128])
@pytest.mark.parametrize(
    ("batch_count", "query_count", "topk_count"),
    [(1, 1, 128), (1, 1, 4096), (2, 40, 2048)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("sparse_mode", [0, 3])
@pytest.mark.parametrize("replay_graph", [False, True])
@pytest.mark.parametrize("return_lse", [False, True], ids=["no-lse", "lse"])
@pytest.mark.parametrize("rope_dim", [0, 64], ids=["no-rope", "rope"])
def test_non_fd_template(
    heads,
    batch_count,
    query_count,
    topk_count,
    dtype,
    sparse_mode,
    replay_graph,
    return_lse,
    rope_dim,
):
    test_split_kv_pa_bsnd(
        batch_count,
        query_count,
        topk_count,
        topk_count,
        replay_graph,
        sparse_mode,
        dtype,
        return_lse,
        rope_dim,
        heads,
    )


@pytest.mark.parametrize("heads", [1, 3, 8, 64, 65, 97, 128])
@pytest.mark.parametrize(
    "valid_count",
    [0, 1, 63, 64, 65, 127, 128, 129, 255, 256, 257, 2047, "mixed", "mixed-empty"],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("rope_dim", [0, 64], ids=["no-rope", "rope"])
@pytest.mark.parametrize("sparse_mode", [0, 3])
@pytest.mark.parametrize("replay_graph", [False, True])
@pytest.mark.parametrize("return_lse", [False, True], ids=["no-lse", "lse"])
def test_sparse_padding(
    heads, valid_count, dtype, rope_dim, sparse_mode, replay_graph, return_lse
):
    test_split_kv_pa_bsnd(
        2,
        4,
        2048,
        valid_count,
        replay_graph,
        sparse_mode,
        dtype,
        return_lse,
        rope_dim,
        heads,
    )


@pytest.mark.parametrize("heads", [1, 8, 65, 128])
@pytest.mark.parametrize("valid_count", [0, 129, "mixed-empty"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("rope_dim", [0, 64], ids=["no-rope", "rope"])
@pytest.mark.parametrize("sparse_mode", [0, 3])
@pytest.mark.parametrize("replay_graph", [False, True])
@pytest.mark.parametrize("return_lse", [False, True], ids=["no-lse", "lse"])
def test_non_fd_sparse_padding(
    heads, valid_count, dtype, rope_dim, sparse_mode, replay_graph, return_lse
):
    test_split_kv_pa_bsnd(
        2,
        40,
        2048,
        valid_count,
        replay_graph,
        sparse_mode,
        dtype,
        return_lse,
        rope_dim,
        heads,
    )


@pytest.mark.parametrize(
    "sweep,value,config",
    list(cases()),
    ids=[f"{sweep}-{value}" for sweep, value, _ in cases()],
)
def test_benchmark_configuration_accuracy(sweep, value, config):
    test_split_kv_pa_bsnd(
        config["batch_count"],
        config["query_count"],
        config["topk_count"],
        config["topk_count"],
        True,
        config["sparse_mode"],
        config["dtype"],
        config["return_lse"],
        config["rope_dim"],
        config["heads"],
        config["kv_length"],
        config["page_size"],
    )
