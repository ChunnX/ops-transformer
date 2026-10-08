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
        sources=[str(root / "sparse_flash_attention.cpp"), str(root / "extension_hash.cpp")],
        extra_include_paths=[
            str(cann_root / "include"), str(cann_root / "include/aclnnop"),
            str(npu_root / "include"), str(npu_root / "include/third_party/hccl/inc"),
            str(npu_root / "include/third_party/op-plugin"), str(root),
        ],
        extra_cflags=["-O2"],
        extra_ldflags=[f"-L{cann_root}/lib64", "-lascendcl", f"-L{npu_root}/lib", "-ltorch_npu"],
        verbose=True,
    )

    def run(q, k, v, indices, scale, **kwargs):
        return module.sparse_flash_attention(
            q, k, v, indices, scale, kwargs["block_table"], kwargs["actual_seq_lengths_query"],
            kwargs["actual_seq_lengths_kv"], kwargs["query_rope"], kwargs["key_rope"], None,
            1, "TND", "PA_BSND", 0, 9223372036854775807, 9223372036854775807, 2,
            kwargs["return_softmax_lse"],
        )

    return run


@pytest.mark.parametrize(
    "query_count,batch_count",
    [(1, 1), (4, 1), (4, 2), (4, 4), (8, 1), (8, 2), (8, 4), (8, 8)],
)
@pytest.mark.parametrize("topk,valid_count", [(2048, 256), (1024, 1024), (1536, 1536), (2048, 2048)])
@pytest.mark.parametrize("replay_graph", [False, True])
@pytest.mark.parametrize("return_lse", [False, True])
@pytest.mark.parametrize("heads", [8, 64])
def test_split_kv_arch22(sfa_operator, query_count, batch_count, topk, valid_count, replay_graph, return_lse, heads):
    if "910B" not in torch.npu.get_device_name(0):
        pytest.skip("This test requires Ascend 910B")

    generator = torch.Generator().manual_seed(20260929 + query_count + batch_count)
    kv_length, page_size = 4096, 128
    query_counts = [1] * (batch_count - 1) + [query_count - batch_count + 1]
    query_ends = [sum(query_counts[:idx + 1]) for idx in range(batch_count)]
    query = torch.randn(query_count, heads, 512, generator=generator).bfloat16().npu()
    query_rope = torch.randn(query_count, heads, 64, generator=generator).bfloat16().npu()
    key = torch.randn(batch_count * kv_length, 512, generator=generator).bfloat16().npu()
    key_rope = torch.randn(batch_count * kv_length, 64, generator=generator).bfloat16().npu()
    key_pages = key.reshape(-1, page_size, 1, 512)
    rope_pages = key_rope.reshape(-1, page_size, 1, 64)
    selected = torch.stack(
        [torch.randperm(kv_length, generator=generator)[:valid_count] for _ in range(query_count)]
    ).int()
    indices = torch.nn.functional.pad(selected, (0, topk - valid_count), value=-1).unsqueeze(1).npu()
    block_table = torch.arange(batch_count * kv_length // page_size, dtype=torch.int32)
    block_table = block_table.reshape(batch_count, kv_length // page_size).npu()
    query_lengths = torch.tensor(query_ends, dtype=torch.int32).npu()
    kv_lengths = torch.full((batch_count,), kv_length, dtype=torch.int32).npu()
    scale = 1.0 / math.sqrt(576)

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
    assert torch.isfinite(actual).all()
    if valid_count < topk:
        return

    expected = []
    expected_lse = []
    for row in range(query_count):
        batch_idx = next(idx for idx, end in enumerate(query_ends) if row < end)
        kv_positions = batch_idx * kv_length + indices[row, 0].long()
        q = torch.cat((query[row].float(), query_rope[row].float()), dim=-1)
        k = torch.cat((key[kv_positions].float(), key_rope[kv_positions].float()), dim=-1)
        scores = torch.matmul(q, k.T).mul_(scale)
        weights = scores.softmax(dim=-1)
        expected_lse.append(scores.logsumexp(dim=-1))
        expected.append(torch.matmul(weights, key[kv_positions].float()))
    reference = torch.stack(expected)
    relative_l2 = (actual.float() - reference).norm() / reference.norm()
    assert relative_l2.item() < 0.02, f"relative L2 error: {relative_l2.item()}"
    if return_lse:
        assert tuple(max_out.shape) == (1, query_count, heads)
        assert max_out.dtype == sum_out.dtype == torch.float32
        lse = max_out[0] + sum_out[0].log()
        torch.testing.assert_close(lse, torch.stack(expected_lse), rtol=0.001, atol=0.03)
