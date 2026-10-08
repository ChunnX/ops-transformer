#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""A2 G8/G64 packed-C8 Split-KV output, LSE, fallback and graph regressions."""

import importlib
import math
import os
from pathlib import Path

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")


@pytest.fixture(scope="module")
def quant_operator():
    extension_dir = os.environ.get("SFA_TEST_EXTENSION_DIR")
    if not extension_dir:
        importlib.import_module("cann_ops_transformer")
        return torch.ops.cann_ops_transformer.kv_quant_sparse_flash_attention

    from torch.utils.cpp_extension import load

    root = Path(extension_dir)
    npu_root = Path(torch_npu.__file__).parent
    cann_root = Path(os.environ["ASCEND_HOME_PATH"])
    module = load(
        name="sfa_c8_lse_compat_probe",
        sources=[str(root / "kv_quant_sparse_flash_attention.cpp"), str(root / "extension_hash.cpp")],
        extra_include_paths=[
            str(cann_root / "include"), str(cann_root / "include/aclnnop"),
            str(npu_root / "include"), str(npu_root / "include/third_party/hccl/inc"),
            str(npu_root / "include/third_party/op-plugin"), str(root),
        ],
        extra_cflags=["-O2"],
        extra_ldflags=[f"-L{cann_root}/lib64", "-lascendcl", f"-L{npu_root}/lib", "-ltorch_npu"],
        verbose=True,
    )

    def run(q, kv, value, indices, scale, key_mode, value_mode, **kwargs):
        return module.npu_kv_quant_sparse_flash_attention(
            q, kv, value, indices, None, None, kwargs["block_table"],
            kwargs["actual_seq_lengths_query"], kwargs["actual_seq_lengths_kv"],
            scale, key_mode, value_mode, 1, "TND", "PA_BSND", 0,
            9223372036854775807, 9223372036854775807, 2, 1, 128, 64, None, None,
            None, kwargs["return_softmax_lse"],
        )

    return run


@pytest.mark.parametrize("query_count,batch_count", [(1, 1), (4, 1), (4, 2), (8, 4), (16, 1)])
@pytest.mark.parametrize("topk,valid_count", [(512, 512), (1024, 1024), (1536, 1536), (2048, 2048), (2048, 256), (2048, 0)])
@pytest.mark.parametrize("return_lse", [False, True])
@pytest.mark.parametrize("replay_graph", [False, True])
@pytest.mark.parametrize("heads", [8, 64])
def test_split_kv_c8_arch22(quant_operator, query_count, batch_count, topk, valid_count, return_lse, replay_graph, heads):
    if "910B" not in torch.npu.get_device_name(0):
        pytest.skip("This test requires Ascend 910B")
    if valid_count == 0 and not (heads == 64 and return_lse):
        pytest.skip("Empty local KV shard regression targets G64 DCP LSE calls")

    generator = torch.Generator().manual_seed(20261006 + query_count + batch_count)
    kv_length, page_size = 4096, 128
    query_counts = [1] * (batch_count - 1) + [query_count - batch_count + 1]
    query_ends = [sum(query_counts[:idx + 1]) for idx in range(batch_count)]
    query_cpu = torch.randn(query_count, heads, 576, generator=generator).bfloat16()
    keys = torch.randint(-32, 33, (batch_count * kv_length, 512), generator=generator, dtype=torch.int8)
    rope = torch.randn(batch_count * kv_length, 64, generator=generator).bfloat16()
    scales = torch.rand(batch_count * kv_length, 4, generator=generator).mul_(0.04).add_(0.01)
    packed = torch.cat((keys.view(torch.uint8), rope.view(torch.uint8), scales.view(torch.uint8)), dim=-1)
    packed = packed.view(torch.int8).reshape(-1, page_size, 1, 656).npu()
    query = query_cpu.npu()
    selected = torch.stack([torch.randperm(kv_length, generator=generator)[:valid_count] for _ in range(query_count)])
    indices = torch.nn.functional.pad(selected.int(), (0, topk - valid_count), value=-1).unsqueeze(1).npu()
    table = torch.arange(batch_count * kv_length // page_size, dtype=torch.int32)
    table = table.reshape(batch_count, kv_length // page_size).npu()
    q_lengths = torch.tensor(query_ends, dtype=torch.int32).npu()
    kv_lengths = torch.full((batch_count,), kv_length, dtype=torch.int32).npu()
    scale = 1.0 / math.sqrt(576)

    def run_operator():
        return quant_operator(
            query, packed, packed, indices, scale, 2, 2, block_table=table,
            actual_seq_lengths_query=q_lengths, actual_seq_lengths_kv=kv_lengths,
            sparse_block_size=1, layout_query="TND", layout_kv="PA_BSND",
            sparse_mode=0, attention_mode=2, quant_scale_repo_mode=1,
            tile_size=128, rope_head_dim=64, return_softmax_lse=return_lse,
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
    out, max_out, sum_out = [tensor.cpu() for tensor in actual]
    assert torch.isfinite(out).all()
    if valid_count == 0:
        assert torch.count_nonzero(out) == 0
        assert torch.count_nonzero(sum_out) == 0
        assert torch.isneginf(max_out + sum_out.log()).all()
        return
    dequantized = (keys.float().reshape(-1, 4, 128) * scales.unsqueeze(-1)).reshape(-1, 512)
    references, lse_references = [], []
    for row in range(query_count):
        batch = next(idx for idx, end in enumerate(query_ends) if row < end)
        positions = selected[row] + batch * kv_length
        k = torch.cat((dequantized[positions], rope[positions].float()), dim=-1)
        scores = (query_cpu[row].float() @ k.T) * scale
        references.append(scores.softmax(dim=-1) @ dequantized[positions])
        lse_references.append(scores.logsumexp(dim=-1))
    reference = torch.stack(references)
    relative_l2 = (out.float() - reference).norm() / reference.norm()
    assert relative_l2.item() < 0.02, f"relative L2 error: {relative_l2.item()}"
    if return_lse:
        assert max_out.shape == sum_out.shape == (1, query_count, heads)
        assert max_out.dtype == sum_out.dtype == torch.float32
        torch.testing.assert_close(max_out[0] + sum_out[0].log(), torch.stack(lse_references), rtol=0.001, atol=0.03)
        if query_count == 4 and batch_count == 2 and topk == 2048 and not replay_graph:
            # DCP combines attention outputs from disjoint KV shards using LSE.
            shard_outputs, shard_lses = [], []
            for shard in indices.chunk(2, dim=-1):
                shard_out, shard_max, shard_sum = quant_operator(
                    query, packed, packed, shard.contiguous(), scale, 2, 2,
                    block_table=table, actual_seq_lengths_query=q_lengths,
                    actual_seq_lengths_kv=kv_lengths, return_softmax_lse=True,
                )
                shard_outputs.append(shard_out.float().cpu())
                shard_lses.append((shard_max + shard_sum.log())[0].cpu())
            lses = torch.stack(shard_lses)
            weights = lses.softmax(dim=0).unsqueeze(-1)
            merged = (torch.stack(shard_outputs) * weights).sum(dim=0)
            assert ((merged - reference).norm() / reference.norm()).item() < 0.02
            torch.testing.assert_close(lses.logsumexp(dim=0), torch.stack(lse_references), rtol=0.001, atol=0.03)
    else:
        assert max_out.numel() == sum_out.numel() == 0
