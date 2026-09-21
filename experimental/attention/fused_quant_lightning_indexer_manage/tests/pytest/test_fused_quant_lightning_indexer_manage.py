#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Core correctness tests for FusedQuantLightningIndexerManage."""

from __future__ import annotations

import importlib
import os

import pytest
import torch
import torch_npu

BLOCK = 128
HEAD_DIM = 128
TOPK = 2048
MISS_CAPACITY = 32768
SOURCE_CAPACITY = 16384


def _device():
    device_id = int(os.environ.get("DEVICE_ID", "0"))
    device = torch.device(f"npu:{device_id}")
    torch_npu.npu.set_device(device)
    return device


def _load_operator():
    op = getattr(torch_npu, "npu_fused_quant_lightning_indexer_manage", None)
    if op is not None:
        return op
    errors = []
    for module_name in (
        "cann_ops_transformer_custom.ops",
        "cann_ops_transformer.ops",
    ):
        try:
            module = importlib.import_module(module_name)
            op = getattr(module, "fused_quant_lightning_indexer_manage", None)
            if op is not None:
                return op
        except (ImportError, RuntimeError, AttributeError) as error:
            errors.append(f"{module_name}: {error}")
    raise RuntimeError(
        "fused_quant_lightning_indexer_manage PyTorch extension is unavailable: "
        + "; ".join(errors)
    )


def _make_case(
    device,
    heads,
    seed,
    *,
    source_capacity=SOURCE_CAPACITY,
    query_counts=(1, 4),
):
    torch.manual_seed(seed)
    query_counts = tuple(query_counts)
    batch = len(query_counts)
    total_queries = sum(query_counts)
    query_ends = torch.tensor(
        torch.tensor(query_counts, dtype=torch.int32).cumsum(0).tolist(),
        dtype=torch.int32,
        device=device,
    )

    blocks_per_request = source_capacity // BLOCK
    physical_blocks = batch * blocks_per_request
    query_bf16 = torch.randn(
        total_queries, heads, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    key_bf16 = torch.randn(
        physical_blocks,
        BLOCK,
        1,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    weights_storage = torch.randn(
        total_queries, heads + 128, dtype=torch.bfloat16, device=device
    )
    weights = weights_storage[:, 128:]
    query, query_scale = torch_npu.npu_dynamic_quant(
        query_bf16, dst_type=torch.float8_e4m3fn
    )
    key_dense, key_scale_dense = torch_npu.npu_dynamic_quant(
        key_bf16, dst_type=torch.float8_e4m3fn
    )
    query = query.reshape(total_queries, heads, HEAD_DIM).contiguous()
    query_scale = query_scale.reshape(total_queries, heads).float().contiguous()

    key_stride = BLOCK * HEAD_DIM + 256
    key_storage = torch.empty(
        physical_blocks * key_stride,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    key = torch.as_strided(
        key_storage,
        size=(physical_blocks, BLOCK, 1, HEAD_DIM),
        stride=(key_stride, HEAD_DIM, HEAD_DIM, 1),
    )
    key.copy_(key_dense)

    scale_stride = BLOCK + 8
    scale_storage = torch.empty(
        physical_blocks * scale_stride, dtype=torch.float32, device=device
    )
    key_scale = torch.as_strided(
        scale_storage,
        size=(physical_blocks, BLOCK, 1),
        stride=(scale_stride, 1, 1),
    )
    key_scale.copy_(key_scale_dense.reshape(physical_blocks, BLOCK, 1).float())

    block_table = torch.arange(
        physical_blocks, dtype=torch.int32, device=device
    ).reshape(batch, blocks_per_request)
    actual_key = torch.full(
        (batch,), source_capacity, dtype=torch.int32, device=device
    )
    offload_key = actual_key.clone()
    cache_tokens = torch.full((batch,), 8192, dtype=torch.int32, device=device)
    states = torch.full((batch,), -3, dtype=torch.int32, device=device)
    entries = torch.arange(batch, dtype=torch.int32, device=device).roll(1)
    pool = torch.full(
        (max(4, batch), source_capacity),
        -1,
        dtype=torch.int32,
        device=device,
    )
    outputs = (
        torch.full((total_queries, 1, TOPK), -7, dtype=torch.int32, device=device),
        torch.full((total_queries, 1, TOPK), -7, dtype=torch.int32, device=device),
        torch.full((total_queries,), -7, dtype=torch.int32, device=device),
        torch.full((batch, MISS_CAPACITY), -7, dtype=torch.int32, device=device),
        torch.full((batch, MISS_CAPACITY), -7, dtype=torch.int32, device=device),
        torch.full((batch,), -7, dtype=torch.int32, device=device),
    )
    return {
        "query": query,
        "key": key,
        "weights": weights,
        "query_scale": query_scale,
        "key_scale": key_scale,
        "block_table": block_table,
        "query_ends": query_ends,
        "actual_key": actual_key,
        "offload_key": offload_key,
        "cache_tokens": cache_tokens,
        "states": states,
        "entries": entries,
        "pool": pool,
        "outputs": outputs,
        "query_counts": query_counts,
    }


def _launch(case):
    result = _load_operator()(
        case["weights"],
        case["query_scale"],
        case["query"],
        case["key_scale"],
        case["key"],
        case["block_table"],
        case["query_ends"],
        case["actual_key"],
        case["offload_key"],
        case["cache_tokens"],
        case["states"],
        case["entries"],
        case["pool"],
        *case["outputs"],
    )
    assert result is None


def _run(case):
    _launch(case)
    torch_npu.npu.synchronize()


def _clear_outputs(case):
    for output in case["outputs"]:
        output.fill_(-7)


def _request_ranges(case):
    start = 0
    for request, count in enumerate(case["query_counts"]):
        end = start + count
        yield request, start, end
        start = end


def _official_topk(case, request, row, end):
    op = getattr(torch_npu, "npu_quant_lightning_indexer", None)
    if op is None:
        raise RuntimeError("official npu_quant_lightning_indexer is unavailable")
    visible = int(case["actual_key"][request]) - (end - 1 - row)
    output = op(
        query=case["query"][row : row + 1],
        key=case["key"],
        weights=case["weights"][row : row + 1],
        query_dequant_scale=case["query_scale"][row : row + 1],
        key_dequant_scale=case["key_scale"],
        actual_seq_lengths_query=torch.ones(
            (1,), dtype=torch.int32, device=case["query"].device
        ),
        actual_seq_lengths_key=torch.tensor(
            [visible], dtype=torch.int32, device=case["query"].device
        ),
        block_table=case["block_table"][request : request + 1],
        query_quant_mode=0,
        key_quant_mode=0,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=TOPK,
        sparse_mode=3,
    )
    if isinstance(output, torch.Tensor):
        return output.reshape(-1).cpu()
    return next(
        tensor.reshape(-1).cpu()
        for tensor in output
        if isinstance(tensor, torch.Tensor)
        and tensor.dtype == torch.int32
        and tensor.numel() == TOPK
    )


def _configure_first_fill(case, budget):
    prefix = (case["pool"].shape[1] - 256) // BLOCK * BLOCK
    case["actual_key"].fill_(prefix + 256)
    case["offload_key"].fill_(prefix)
    case["cache_tokens"].fill_(budget)
    case["states"].fill_(-2)
    case["pool"].fill_(-1)
    _clear_outputs(case)
    return prefix


def _assert_sparse_pool(row, prefix, budget):
    resident = row >= 0
    assert int(resident.sum()) == budget
    assert set(row[resident].cpu().tolist()) == set(range(budget))
    assert bool((row[prefix:] == -1).all())


@pytest.mark.ci
@pytest.mark.parametrize("heads", (32, 64), ids=("heads32", "heads64"))
def test_non_offload(heads):
    case = _make_case(_device(), heads, seed=7 + heads)
    pointers = [case["pool"].data_ptr()]
    pointers.extend(output.data_ptr() for output in case["outputs"])
    _run(case)

    topk_src, topk_dst, route_miss, _, _, request_miss = case["outputs"]
    assert torch.equal(topk_src, topk_dst)
    assert bool((route_miss == 0).all())
    assert bool((request_miss == 0).all())
    assert pointers == [case["pool"].data_ptr()] + [
        output.data_ptr() for output in case["outputs"]
    ]
    identity = torch.arange(case["pool"].shape[1], dtype=torch.int32)
    for request, start, end in _request_ranges(case):
        pool_row = int(case["entries"][request])
        assert torch.equal(case["pool"][pool_row].cpu(), identity)
        for row in range(start, end):
            assert set(topk_src[row, 0].cpu().tolist()) == set(
                _official_topk(case, request, row, end).tolist()
            )


@pytest.mark.ci
@pytest.mark.parametrize(
    "heads,budget",
    ((32, 8192), (64, 12288)),
    ids=("heads32-c8192", "heads64-c12288"),
)
def test_first_fill(heads, budget):
    case = _make_case(_device(), heads, seed=100 + heads)
    prefix = _configure_first_fill(case, budget)
    _run(case)

    topk_src, topk_dst, route_miss, miss_src, miss_dst, request_miss = case["outputs"]
    assert bool((request_miss == budget).all())
    for request, start, end in _request_ranges(case):
        pool_row = case["pool"][int(case["entries"][request])]
        _assert_sparse_pool(pool_row, prefix, budget)
        sources = miss_src[request, :budget].cpu().tolist()
        slots = miss_dst[request, :budget].cpu().tolist()
        assert len(set(sources)) == budget
        assert slots == list(range(budget))
        for source, slot in zip(sources, slots):
            assert int(pool_row[source]) == slot
        for row in range(start, end):
            golden = _official_topk(case, request, row, end)
            actual = topk_src[row, 0].cpu()
            assert set(actual.tolist()) == set(golden.tolist())
            count = int(route_miss[row])
            assert count == sum(int(token) < prefix for token in golden.tolist())
            assert bool((actual[:count] < prefix).all())
            assert bool((actual[count:] >= prefix).all())
            for source, slot in zip(actual.tolist(), topk_dst[row, 0].cpu().tolist()):
                expected = int(pool_row[source]) if source < prefix else budget + source % 256
                assert slot == expected


@pytest.mark.ci
@pytest.mark.parametrize(
    "heads,budget",
    ((32, 8192), (64, 12288)),
    ids=("heads32-c8192", "heads64-c12288"),
)
def test_steady(heads, budget):
    case = _make_case(_device(), heads, seed=200 + heads)
    prefix = _configure_first_fill(case, budget)
    _run(case)

    first_topk = case["outputs"][0].cpu()
    expected_misses = []
    for request, start, end in _request_ranges(case):
        row = case["pool"][int(case["entries"][request])]
        selected = {
            int(source)
            for route in range(start, end)
            for source in first_topk[route, 0].tolist()
            if 0 <= int(source) < prefix
        }
        misses = set(sorted(selected)[:4])
        replacements = [
            source
            for source in range(prefix)
            if int(row[source]) == -1 and source not in selected
        ][: len(misses)]
        assert len(misses) == len(replacements) == 4
        for source, replacement in zip(sorted(misses), replacements):
            slot = int(row[source])
            row[source] = -1
            row[replacement] = slot
        expected_misses.append(misses)

    case["states"].fill_(-1)
    _clear_outputs(case)
    _run(case)
    topk_src, _, route_miss, miss_src, _, request_miss = case["outputs"]
    for request, start, end in _request_ranges(case):
        misses = expected_misses[request]
        assert int(request_miss[request]) == len(misses)
        assert set(miss_src[request, : len(misses)].cpu().tolist()) == misses
        row_pool = case["pool"][int(case["entries"][request])]
        _assert_sparse_pool(row_pool, prefix, budget)
        assert all(int(row_pool[source]) >= 0 for source in misses)
        for row in range(start, end):
            golden = _official_topk(case, request, row, end)
            assert set(topk_src[row, 0].cpu().tolist()) == set(golden.tolist())
            assert int(route_miss[row]) == sum(
                int(source) in misses for source in golden.tolist()
            )


@pytest.mark.ci
def test_invalid_request_is_atomic():
    case = _make_case(_device(), 32, seed=301)
    _configure_first_fill(case, 9000)
    case["pool"].copy_(
        torch.arange(
            case["pool"].numel(), dtype=torch.int32, device=case["pool"].device
        ).reshape_as(case["pool"])
    )
    before = case["pool"].clone()
    _run(case)
    assert torch.equal(case["pool"], before)
    assert bool((case["outputs"][2] == -1).all())
    assert bool((case["outputs"][5] == -1).all())


@pytest.mark.ci
def test_mixed_request_pool_row_isolation():
    case = _make_case(_device(), 32, seed=401, query_counts=(1, 2, 4))
    budget = 8192
    prefix = _configure_first_fill(case, budget)
    case["states"].copy_(
        torch.tensor([-3, -2, -1], dtype=torch.int32, device=case["states"].device)
    )
    steady_row = int(case["entries"][2])
    case["pool"][steady_row, :budget] = torch.arange(
        budget, dtype=torch.int32, device=case["pool"].device
    )
    unused_before = case["pool"][3].clone()
    _run(case)

    assert int(case["outputs"][5][0]) == 0
    assert int(case["outputs"][5][1]) == budget
    assert int(case["outputs"][5][2]) >= 0
    identity = torch.arange(case["pool"].shape[1], dtype=torch.int32)
    assert torch.equal(case["pool"][int(case["entries"][0])].cpu(), identity)
    _assert_sparse_pool(case["pool"][int(case["entries"][1])], prefix, budget)
    _assert_sparse_pool(case["pool"][steady_row], prefix, budget)
    assert torch.equal(case["pool"][3], unused_before)


@pytest.mark.ci
def test_262144_token_boundary():
    case = _make_case(
        _device(),
        32,
        seed=501,
        source_capacity=1 << 18,
        query_counts=(1,),
    )
    _run(case)
    assert int(case["outputs"][0].max()) >= 1 << 17
    assert torch.equal(case["outputs"][0], case["outputs"][1])
    assert int(case["outputs"][5][0]) == 0


@pytest.mark.ci
@pytest.mark.graph
def test_graph_replay():
    case = _make_case(_device(), 32, seed=601, query_counts=(1, 2))
    _run(case)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        _launch(case)
    torch_npu.npu.synchronize()

    case["pool"].fill_(-1)
    _clear_outputs(case)
    graph.replay()
    torch_npu.npu.synchronize()
    graph_outputs = tuple(output.cpu().clone() for output in case["outputs"])
    graph_pool = case["pool"].cpu().clone()

    case["pool"].fill_(-1)
    _clear_outputs(case)
    _run(case)
    eager_outputs = tuple(output.cpu() for output in case["outputs"])
    assert torch.equal(graph_pool, case["pool"].cpu())
    assert torch.equal(graph_outputs[2], eager_outputs[2])
    assert torch.equal(graph_outputs[5], eager_outputs[5])
    for route in range(case["query"].shape[0]):
        graph_pairs = sorted(
            zip(graph_outputs[0][route, 0].tolist(), graph_outputs[1][route, 0].tolist())
        )
        eager_pairs = sorted(
            zip(eager_outputs[0][route, 0].tolist(), eager_outputs[1][route, 0].tolist())
        )
        assert graph_pairs == eager_pairs
