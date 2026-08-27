# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License for the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Pytest test cases for PoolKeyIndexer TorchAir (GE graph) mode.

覆盖 GE 图模式专属路径(与 eager 用例 test_pool_key_indexer.py 互补):

  1. tiling 值校验跳过: 图模式下 actual_seq_q / actual_seq_k /
     pool_tail_k 是 device tensor, tiling 期数值不可见(GetData 为 null)。
     host 侧跳过值校验(非递减/末项=T1/取值范围等), 由 kernel 运行期从
     GM 读值(GetActualSeqLen)。覆盖 TND 与 PA_BBND 布局。
  2. PA key 0 轴非连续: converter(graph_convert_pool_key_indexer.py)从
     转换期 tensor 元数据读 stride(0) 下传 key_stride0 属性(与 eager 路径
     csrc 直读 at::Tensor::stride(0) 对齐); 属性缺失时回退 tiling context
     运行时 stride(框架填 TensorV2 view)。

比对策略(与 eager 用例一致):
  - indices(第一输出): 逐行有效索引(>=0)排序集合比对, 100% 行通过为唯一
    精度门禁;
  - values(第二输出): 逐行有效值排序数值比对, 仅打印日志不作为判据。

依赖: torchair(缺失则整文件 skip); NPU 设备(不可用则 skip)。
"""

import os
import sys

import pytest

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")
torchair = pytest.importorskip("torchair")

if not torch.npu.is_available():
    pytest.skip("NPU device is not available", allow_module_level=True)

sys.path.insert(0, os.path.dirname(__file__))

from result_compare import compare_indices, compare_values  # noqa: E402
from pool_key_indexer_reference import pool_key_indexer_reference  # noqa: E402

# 复用 eager 用例的输入构造/PA 打包/非连续视图 helper(同目录)
from test_pool_key_indexer import (  # noqa: E402
    HEAD_DIM,
    N2,
    _build_bsnd_inputs,
    _build_tnd_inputs,
    _make_actual_seq_noncum,
    _make_pa_key_non_contig,
    _build_block_table,
    _pack_pa_key,
)

# Import torch extension (registers torch.ops.cann_ops_transformer.pool_key_indexer)
from cann_ops_transformer import pool_key_indexer  # noqa: E402,F401


# ---------------------------------------------------------------------------
# GE graph mode helpers
# ---------------------------------------------------------------------------


class _PoolKeyIndexerModule(torch.nn.Module):
    """Wrapper calling the torch.ops path (converter interception point)."""

    def __init__(self, **op_kwargs):
        super().__init__()
        self._op_kwargs = op_kwargs

    def forward(
        self,
        query,
        pool_key,
        weights,
        pool_tail_k,
        actual_seq_q=None,
        actual_seq_k=None,
        block_table=None,
    ):
        return torch.ops.cann_ops_transformer.pool_key_indexer(
            query,
            pool_key,
            weights,
            pool_tail_k,
            actual_seq_q=actual_seq_q,
            actual_seq_k=actual_seq_k,
            block_table=block_table,
            **self._op_kwargs,
        )


def _compile_ge_graph(module):
    """Compile with the torchair GE graph backend (fullgraph, static shape)."""
    from torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    torch._dynamo.reset()
    return torch.compile(module, fullgraph=True, backend=npu_backend, dynamic=False)


def _run_ge_graph(
    query,
    pool_key,
    weights,
    pool_tail_k,
    *,
    actual_seq_q=None,
    actual_seq_k=None,
    block_table=None,
    layout_q="BSND",
    layout_k="BSND",
    topk=128,
    pool_size=1,
    mask_mode=0,
    quant_mode=-1,
    return_value=False,
    device_id=0,
    pool_key_npu_fn=None,
):
    """Run one pool_key_indexer call through the GE graph mode and return CPU outputs."""
    dev = f"npu:{device_id}"
    q = query.to(dev)
    k = pool_key_npu_fn(dev) if pool_key_npu_fn is not None else pool_key.to(dev)
    w = weights.to(dev)
    ptk = pool_tail_k.to(dev)
    asq = actual_seq_q.to(dev) if actual_seq_q is not None else None
    ask = actual_seq_k.to(dev) if actual_seq_k is not None else None
    bt = block_table.to(dev) if block_table is not None else None

    model = _compile_ge_graph(
        _PoolKeyIndexerModule(
            layout_q=layout_q,
            layout_k=layout_k,
            topk=topk,
            pool_size=pool_size,
            mask_mode=mask_mode,
            quant_mode=quant_mode,
            return_value=return_value,
        )
    )
    indices, values = model(q, k, w, ptk, asq, ask, bt)
    torch.npu.synchronize()
    return indices.cpu(), values.cpu()


def _compare_and_assert(
    tag,
    layout_q,
    layout_k,
    pool_size,
    mask_mode,
    g_indices,
    g_values,
    npu_indices,
    npu_values,
    return_value,
    dtype,
):
    """Shared assertion block (indices gate + values shape/log)."""
    ok_idx = compare_indices(g_indices, npu_indices)
    assert ok_idx, (
        f"[{tag}] GE graph mode indices set comparison failed "
        f"({layout_q}/{layout_k} ps={pool_size} mask={mask_mode})"
    )
    if return_value:
        assert npu_values.shape == g_values.shape, (
            f"[{tag}] values shape mismatch: golden {g_values.shape} "
            f"vs npu {npu_values.shape}"
        )
        compare_values(g_values, npu_values, dtype, label=f"{tag}-values")
    else:
        assert npu_values.numel() == 0 or npu_values.shape[-1] == 0, (
            f"[{tag}] expected empty values when return_value=False, "
            f"got shape {npu_values.shape}"
        )
    print(f"[{tag}] GE graph mode PASSED")


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def test_ge_graph_bsnd_baseline():
    """BSND + causal + pooling: graph-mode baseline (was already working)."""
    torch.manual_seed(42)
    dtype = torch.float16
    batch, q_seq, k_seq, n1 = 2, 64, 256, 8
    topk, pool_size, mask_mode = 128, 16, 3
    return_value = True

    query, pool_key, weights, pool_tail_k = _build_bsnd_inputs(
        batch, q_seq, k_seq, n1, pool_size, dtype, seed=42
    )

    g_indices, g_values = pool_key_indexer_reference(
        query,
        pool_key,
        weights,
        pool_tail_k,
        layout_q="BSND",
        layout_k="BSND",
        topk=topk,
        pool_size=pool_size,
        mask_mode=mask_mode,
        return_value=return_value,
    )

    npu_indices, npu_values = _run_ge_graph(
        query,
        pool_key,
        weights,
        pool_tail_k,
        layout_q="BSND",
        layout_k="BSND",
        topk=topk,
        pool_size=pool_size,
        mask_mode=mask_mode,
        quant_mode=-1,
        return_value=return_value,
    )

    _compare_and_assert(
        "bsnd_baseline",
        "BSND",
        "BSND",
        pool_size,
        mask_mode,
        g_indices,
        g_values,
        npu_indices,
        npu_values,
        return_value,
        dtype,
    )


def test_ge_graph_tnd():
    """TND + no mask: actual_seq_q/k are device tensors in graph mode.

    tiling-side value validation is skipped (data invisible at tiling time);
    the kernel reads prefix sums from GM at run time. Previously this failed
    with "TND: actual_seq_q data is null, tiling sink not supported".
    """
    torch.manual_seed(7)
    dtype = torch.bfloat16
    q_lens, k_lens, n1 = [32, 48], [128, 192], 8
    topk, pool_size, mask_mode = 64, 16, 0
    return_value = True

    query, pool_key, weights, pool_tail_k, actual_seq_q, actual_seq_k = (
        _build_tnd_inputs(q_lens, k_lens, n1, pool_size, dtype, seed=7)
    )

    g_indices, g_values = pool_key_indexer_reference(
        query,
        pool_key,
        weights,
        pool_tail_k,
        actual_seq_q=actual_seq_q,
        actual_seq_k=actual_seq_k,
        layout_q="TND",
        layout_k="TND",
        topk=topk,
        pool_size=pool_size,
        mask_mode=mask_mode,
        return_value=return_value,
    )

    npu_indices, npu_values = _run_ge_graph(
        query,
        pool_key,
        weights,
        pool_tail_k,
        actual_seq_q=actual_seq_q,
        actual_seq_k=actual_seq_k,
        layout_q="TND",
        layout_k="TND",
        topk=topk,
        pool_size=pool_size,
        mask_mode=mask_mode,
        quant_mode=-1,
        return_value=return_value,
    )

    _compare_and_assert(
        "tnd",
        "TND",
        "TND",
        pool_size,
        mask_mode,
        g_indices,
        g_values,
        npu_indices,
        npu_values,
        return_value,
        dtype,
    )


def test_ge_graph_bsnd_pa():
    """BSND query + PA_BBND key: actual_seq_k + block_table as device tensors.

    Previously failed at tiling with "actual_seq_k data is null, tiling sink
    not supported" (graph mode). Includes unequal pool counts per batch
    (-1 block_table slots + padded tail block).
    """
    torch.manual_seed(11)
    dtype = torch.float16
    batch, q_seq, n1 = 2, 32, 8
    k_lens = [256, 128]  # unequal -> -1 slots + tail padding
    topk, pool_size, mask_mode = 128, 16, 3
    pa_block_size = 16
    return_value = True

    k_pool_lens = [kl // pool_size for kl in k_lens]
    query, pool_key, weights, pool_tail_k = _build_bsnd_inputs(
        batch, q_seq, k_lens, n1, pool_size, dtype, seed=11
    )
    actual_seq_q = None  # PA + BSND: must not be provided
    actual_seq_k = _make_actual_seq_noncum(k_pool_lens)
    pool_key = pool_key.reshape(-1, pool_key.shape[-2], pool_key.shape[-1])
    pool_key, pa_block_size, _ = _pack_pa_key(
        pool_key, k_pool_lens, pool_size, pa_block_size
    )
    block_table, _ = _build_block_table(batch, k_pool_lens, pool_size, pa_block_size)

    g_indices, g_values = pool_key_indexer_reference(
        query,
        pool_key,
        weights,
        pool_tail_k,
        actual_seq_k=actual_seq_k,
        block_table=block_table,
        layout_q="BSND",
        layout_k="PA_BBND",
        topk=topk,
        pool_size=pool_size,
        mask_mode=mask_mode,
        return_value=return_value,
    )

    npu_indices, npu_values = _run_ge_graph(
        query,
        pool_key,
        weights,
        pool_tail_k,
        actual_seq_q=actual_seq_q,
        actual_seq_k=actual_seq_k,
        block_table=block_table,
        layout_q="BSND",
        layout_k="PA_BBND",
        topk=topk,
        pool_size=pool_size,
        mask_mode=mask_mode,
        quant_mode=-1,
        return_value=return_value,
    )

    _compare_and_assert(
        "bsnd_pa",
        "BSND",
        "PA_BBND",
        pool_size,
        mask_mode,
        g_indices,
        g_values,
        npu_indices,
        npu_values,
        return_value,
        dtype,
    )


def test_ge_graph_tnd_pa():
    """TND query + PA_BBND key: both actual_seq tensors as device tensors.

    Previously failed at tiling (TND actual_seq_q data null). Covers the
    combined TND + PA + causal path with block_size 32.
    """
    torch.manual_seed(13)
    dtype = torch.float16
    q_lens = [16, 16]
    k_lens = [256, 256]
    n1 = 8
    topk, pool_size, mask_mode = 128, 16, 3
    pa_block_size = 32
    return_value = True

    k_pool_lens = [kl // pool_size for kl in k_lens]
    query, _, weights, pool_tail_k, actual_seq_q, _ = _build_tnd_inputs(
        q_lens, k_lens, n1, pool_size, dtype, seed=13
    )
    actual_seq_k = _make_actual_seq_noncum(k_pool_lens)
    pool_key, pa_block_size, _ = _pack_pa_key(
        torch.randn(sum(k_pool_lens), N2, HEAD_DIM, dtype=dtype),
        k_pool_lens,
        pool_size,
        pa_block_size,
    )
    block_table, _ = _build_block_table(
        len(q_lens), k_pool_lens, pool_size, pa_block_size
    )

    g_indices, g_values = pool_key_indexer_reference(
        query,
        pool_key,
        weights,
        pool_tail_k,
        actual_seq_q=actual_seq_q,
        actual_seq_k=actual_seq_k,
        block_table=block_table,
        layout_q="TND",
        layout_k="PA_BBND",
        topk=topk,
        pool_size=pool_size,
        mask_mode=mask_mode,
        return_value=return_value,
    )

    npu_indices, npu_values = _run_ge_graph(
        query,
        pool_key,
        weights,
        pool_tail_k,
        actual_seq_q=actual_seq_q,
        actual_seq_k=actual_seq_k,
        block_table=block_table,
        layout_q="TND",
        layout_k="PA_BBND",
        topk=topk,
        pool_size=pool_size,
        mask_mode=mask_mode,
        quant_mode=-1,
        return_value=return_value,
    )

    _compare_and_assert(
        "tnd_pa",
        "TND",
        "PA_BBND",
        pool_size,
        mask_mode,
        g_indices,
        g_values,
        npu_indices,
        npu_values,
        return_value,
        dtype,
    )


def test_ge_graph_bsnd_pa_key_noncontig():
    """BSND + PA_BBND + key 0-axis non-contiguous (key_stride0 attr path).

    The physical blocks are embedded in the even rows of a 2x buffer so the
    view stride(0) = 2 * blockSize * N2 * D. In GE graph mode the converter
    reads stride(0) at conversion time and passes it as the key_stride0
    attribute (falls back to the framework-reported runtime stride when the
    attribute is unavailable); the host tiling consumes the attribute first.
    Golden consumes the same strided CPU view directly.
    """
    torch.manual_seed(17)
    dtype = torch.float16
    batch, q_seq, n1 = 2, 32, 8
    k_lens = [256, 256]
    topk, pool_size, mask_mode = 128, 16, 0
    pa_block_size = 16
    return_value = True

    k_pool_lens = [kl // pool_size for kl in k_lens]
    query, pool_key, weights, pool_tail_k = _build_bsnd_inputs(
        batch, q_seq, k_lens, n1, pool_size, dtype, seed=17
    )
    actual_seq_q = None  # PA + BSND
    actual_seq_k = _make_actual_seq_noncum(k_pool_lens)
    pool_key = pool_key.reshape(-1, pool_key.shape[-2], pool_key.shape[-1])
    pool_key, pa_block_size, pa_block_num = _pack_pa_key(
        pool_key, k_pool_lens, pool_size, pa_block_size
    )
    block_table, _ = _build_block_table(batch, k_pool_lens, pool_size, pa_block_size)

    # 0-axis non-contiguous view: CPU view for golden, NPU builder for graph
    pool_key, pool_key_npu_fn = _make_pa_key_non_contig(
        pool_key, pa_block_size, pa_block_num, dtype
    )
    assert (
        not pool_key.is_contiguous()
        and pool_key.stride(0) == 2 * pa_block_size * N2 * HEAD_DIM
    )

    g_indices, g_values = pool_key_indexer_reference(
        query,
        pool_key,
        weights,
        pool_tail_k,
        actual_seq_k=actual_seq_k,
        block_table=block_table,
        layout_q="BSND",
        layout_k="PA_BBND",
        topk=topk,
        pool_size=pool_size,
        mask_mode=mask_mode,
        return_value=return_value,
    )

    npu_indices, npu_values = _run_ge_graph(
        query,
        pool_key,
        weights,
        pool_tail_k,
        actual_seq_q=actual_seq_q,
        actual_seq_k=actual_seq_k,
        block_table=block_table,
        layout_q="BSND",
        layout_k="PA_BBND",
        topk=topk,
        pool_size=pool_size,
        mask_mode=mask_mode,
        quant_mode=-1,
        return_value=return_value,
        pool_key_npu_fn=pool_key_npu_fn,
    )

    _compare_and_assert(
        "bsnd_pa_key_noncontig",
        "BSND",
        "PA_BBND",
        pool_size,
        mask_mode,
        g_indices,
        g_values,
        npu_indices,
        npu_values,
        return_value,
        dtype,
    )


def test_ge_graph_bsnd_mask0_pools_lt_sparse_race():
    """BSND + mask=0 + pools < sparseCount (tail-append race regression).

    Regression for the arch22 ExpandAndAppendIndices pipeline race: with
    mask_mode=0 and pool count < sparseCount (topk // pool_size), the
    "-1 cleanup" branch (validExpand < topk) and the tail-token append
    branch execute back-to-back in the same row. Without the V->S hard
    sync after the tail-area Duplicate(-1) and the MTE3->V wait before the
    next row's overwrite of the shared UB buffer, the scalar tail writes
    were intermittently overwritten (GE graph mode: up to 96/128 rows lost
    their tail tokens; eager was masked by launch timing).

    This case pins the exact failing combination and repeats the compiled
    graph call REPEAT times to catch the intermittent failure.
    """
    REPEAT = 20
    torch.manual_seed(23)
    dtype = torch.float16
    batch, q_seq, n1 = 2, 64, 8
    pools = 4  # < sparseCount (224 // 16 = 14)
    k_seq = pools * 16
    topk, pool_size, mask_mode = 224, 16, 0
    return_value = False

    query, pool_key, weights, pool_tail_k = _build_bsnd_inputs(
        batch, q_seq, k_seq, n1, pool_size, dtype, seed=23
    )
    # deterministic non-zero tails so the regression cannot pass with tail==0
    pool_tail_k = torch.tensor([5, 13], dtype=torch.int64)
    assert bool((pool_tail_k > 0).all())

    g_indices, g_values = pool_key_indexer_reference(
        query,
        pool_key,
        weights,
        pool_tail_k,
        layout_q="BSND",
        layout_k="BSND",
        topk=topk,
        pool_size=pool_size,
        mask_mode=mask_mode,
        return_value=return_value,
    )

    npu_indices, npu_values = _run_ge_graph(
        query,
        pool_key,
        weights,
        pool_tail_k,
        layout_q="BSND",
        layout_k="BSND",
        topk=topk,
        pool_size=pool_size,
        mask_mode=mask_mode,
        quant_mode=-1,
        return_value=return_value,
    )
    _compare_and_assert(
        "bsnd_mask0_pools_lt_sparse",
        "BSND",
        "BSND",
        pool_size,
        mask_mode,
        g_indices,
        g_values,
        npu_indices,
        npu_values,
        return_value,
        dtype,
    )

    # Repeat the cached graph to stress the row-loop pipeline race:
    # every call must match the golden row set exactly.
    from result_compare import compare_indices as _cmp

    dev = "npu:0"
    model = _compile_ge_graph(
        _PoolKeyIndexerModule(
            layout_q="BSND",
            layout_k="BSND",
            topk=topk,
            pool_size=pool_size,
            mask_mode=mask_mode,
            quant_mode=-1,
            return_value=return_value,
        )
    )
    q = query.to(dev)
    k = pool_key.to(dev)
    w = weights.to(dev)
    ptk = pool_tail_k.to(dev)
    for rep in range(REPEAT):
        idx, _ = model(q, k, w, ptk)
        torch.npu.synchronize()
        assert _cmp(g_indices, idx.cpu()), (
            f"race regression: repeat {rep}/{REPEAT} indices mismatch "
            f"(mask=0 pools={pools} < sparse={topk // pool_size})"
        )
    print(
        f"[bsnd_mask0_pools_lt_sparse] race stress: {REPEAT}/{REPEAT} repeats all matched"
    )


def test_ge_graph_tnd_mask0_pools_lt_sparse_race():
    """TND + mask=0 + pools < sparseCount (race regression, TND variant).

    Same race pattern as the BSND variant, but with TND layout and prefix-sum
    actual_seq tensors (which are device tensors in graph mode, exercising the
    tiling value-validation skip path as well).
    """
    REPEAT = 20
    torch.manual_seed(29)
    dtype = torch.bfloat16
    q_lens = [32, 48]
    k_lens = [64, 32]  # pools = 4, 2 < sparseCount (224 // 16 = 14)
    n1 = 8
    topk, pool_size, mask_mode = 224, 16, 0
    return_value = True

    query, pool_key, weights, pool_tail_k, actual_seq_q, actual_seq_k = (
        _build_tnd_inputs(q_lens, k_lens, n1, pool_size, dtype, seed=29)
    )
    pool_tail_k = torch.tensor([5, 13], dtype=torch.int64)
    assert bool((pool_tail_k > 0).all())

    g_indices, g_values = pool_key_indexer_reference(
        query,
        pool_key,
        weights,
        pool_tail_k,
        actual_seq_q=actual_seq_q,
        actual_seq_k=actual_seq_k,
        layout_q="TND",
        layout_k="TND",
        topk=topk,
        pool_size=pool_size,
        mask_mode=mask_mode,
        return_value=return_value,
    )

    npu_indices, npu_values = _run_ge_graph(
        query,
        pool_key,
        weights,
        pool_tail_k,
        actual_seq_q=actual_seq_q,
        actual_seq_k=actual_seq_k,
        layout_q="TND",
        layout_k="TND",
        topk=topk,
        pool_size=pool_size,
        mask_mode=mask_mode,
        quant_mode=-1,
        return_value=return_value,
    )
    _compare_and_assert(
        "tnd_mask0_pools_lt_sparse",
        "TND",
        "TND",
        pool_size,
        mask_mode,
        g_indices,
        g_values,
        npu_indices,
        npu_values,
        return_value,
        dtype,
    )

    from result_compare import compare_indices as _cmp

    dev = "npu:0"
    model = _compile_ge_graph(
        _PoolKeyIndexerModule(
            layout_q="TND",
            layout_k="TND",
            topk=topk,
            pool_size=pool_size,
            mask_mode=mask_mode,
            quant_mode=-1,
            return_value=return_value,
        )
    )
    q = query.to(dev)
    k = pool_key.to(dev)
    w = weights.to(dev)
    ptk = pool_tail_k.to(dev)
    asq = actual_seq_q.to(dev)
    ask = actual_seq_k.to(dev)
    for rep in range(REPEAT):
        idx, _ = model(q, k, w, ptk, asq, ask)
        torch.npu.synchronize()
        assert _cmp(g_indices, idx.cpu()), (
            f"race regression (TND): repeat {rep}/{REPEAT} indices mismatch"
        )
    print(
        f"[tnd_mask0_pools_lt_sparse] race stress: {REPEAT}/{REPEAT} repeats all matched"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
