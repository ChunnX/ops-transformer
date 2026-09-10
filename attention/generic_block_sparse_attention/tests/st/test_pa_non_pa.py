# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Run against an installed build: pytest -q attention/generic_block_sparse_attention/tests/st/test_pa_non_pa.py"""
import math
import os

import pytest
import torch

pytest.importorskip("torch_npu")
cann_ops_transformer = pytest.importorskip(os.environ.get("GBSA_TEST_MODULE", "cann_ops_transformer"))


def _reference(q, k, v, cu_q, cu_kv, used_q, used_kv, indices, counts):
    q, k, v = q.float(), k.float(), v.float()
    out = torch.zeros_like(q, dtype=torch.float32)
    group = q.shape[1] // k.shape[1]
    for b, (nq, nkv) in enumerate(zip(used_q, used_kv)):
        for token in range(nq):
            qt = cu_q[b] + token
            end = nkv - nq + token + 1
            for head in range(k.shape[1]):
                selected = []
                for block in indices[head, qt, : counts[head, qt]].tolist():
                    if block >= 0:
                        selected.extend(range(block * 128, min((block + 1) * 128, end)))
                if not selected:
                    continue
                rows = torch.tensor(selected) + cu_kv[b]
                query = q[qt, head * group : (head + 1) * group].float()
                prob = torch.softmax(query @ k[rows, head].float().T / math.sqrt(128), dim=-1)
                out[qt, head * group : (head + 1) * group] = prob @ v[rows, head].float()
    return out


@pytest.mark.parametrize("dtype_name,out_dtype_name", [
    ("float16", "float16"), ("bfloat16", "bfloat16"),
    ("float8_e4m3fn", "float16"), ("float8_e4m3fn", "bfloat16"),
])
@pytest.mark.parametrize("topk,padded,kv_heads,group,batch", [
    (1, False, 1, 1, 1), (11, False, 2, 4, 2),
    (12, True, 1, 8, 2), (16, True, 2, 4, 2),
])
def test_pa_tnd_equivalence(dtype_name, out_dtype_name, topk, padded, kv_heads, group, batch):
    if not torch.npu.is_available():
        pytest.skip("NPU is unavailable")
    torch.npu.set_device(int(os.environ.get("GBSA_TEST_DEVICE", "0")))
    is_a5 = "950" in torch.npu.get_device_name()
    if dtype_name == "float8_e4m3fn" and not is_a5:
        pytest.skip("FP8 requires A5")
    dtype = getattr(torch, dtype_name)
    quant = 5 if dtype_name.startswith("float8") else 0
    out_dtype = getattr(torch, out_dtype_name)
    precision = 1 if is_a5 else 0
    torch.manual_seed(103)
    cu_q = [0, 2, 5][: batch + 1]
    cu_kv = [0, 131, 1686][: batch + 1]
    used_q = ([1, 2] if padded else [2, 3])[:batch]
    used_kv = ([129, 1541] if padded else [131, 1555])[:batch]
    q = torch.randn(cu_q[-1], kv_heads * group, 128).to(dtype)
    k = torch.randn(cu_kv[-1], kv_heads, 128).to(dtype)
    v = torch.randn_like(k.float()).to(dtype)
    idx = torch.zeros(kv_heads, cu_q[-1], topk, dtype=torch.int32)
    counts = torch.full((kv_heads, cu_q[-1]), topk, dtype=torch.int32)
    for b in range(batch):
        # Unsorted, duplicate blocks and out-of-range entries exercise causal filtering.
        pattern = ([0] if topk == 1 else [1, 0, -1, 1, 3, 2, 11, 4, 8, 7, 6, 5, 10, 9, 20, 0])[:topk]
        idx[:, cu_q[b] : cu_q[b + 1]] = torch.tensor(pattern, dtype=torch.int32)
    if topk > 1:
        idx[:, 0] = -1  # A nonzero raw count with no surviving blocks.
        counts[:, -1] = topk - 1  # Last slot must be ignored.
    # Permute physical pages and retain dim0 holes to cover the existing PA view path.
    max_pages = 13
    pages = batch * max_pages
    permutation = torch.randperm(pages)
    table = permutation.reshape(batch, max_pages).to(torch.int32)
    paged_k = torch.zeros(pages, 128, kv_heads, 128, dtype=dtype)
    paged_v = torch.zeros_like(paged_k)
    for b in range(batch):
        for logical in range(math.ceil((cu_kv[b + 1] - cu_kv[b]) / 128)):
            begin = cu_kv[b] + logical * 128
            end = min(begin + 128, cu_kv[b + 1])
            physical = table[b, logical].item()
            paged_k[physical, : end - begin] = k[begin:end]
            paged_v[physical, : end - begin] = v[begin:end]
    def npu(x):
        return x.to("npu")
    def with_holes(x):
        storage = torch.empty((x.shape[0] * 2, *x.shape[1:]), dtype=x.dtype, device="npu")
        storage[::2].copy_(npu(x))
        return storage[::2]
    ops = cann_ops_transformer.ops
    common = dict(cu_seqlens_q=npu(torch.tensor(cu_q, dtype=torch.int64)),
                  softmax_precision=precision, quant_mode=quant)
    if padded:
        common["seqused_q"] = npu(torch.tensor(used_q, dtype=torch.int32))
    used_kv_npu = npu(torch.tensor(used_kv, dtype=torch.int32))
    idx_npu, counts_npu = npu(idx), npu(counts)
    q_npu = npu(q)
    results = []
    for layout in ("PA_BBND", "TND"):
        lengths = dict(common, layout_kv=layout)
        if layout == "PA_BBND":
            lengths["seqused_kv"] = used_kv_npu
            kn, vn = with_holes(paged_k), with_holes(paged_v)
            block_table = npu(table)
        else:
            lengths["cu_seqlens_kv"] = npu(torch.tensor(cu_kv, dtype=torch.int64))
            if padded:
                lengths["seqused_kv"] = used_kv_npu
            # ACLNN must make dim0-strided TND views contiguous.
            kn, vn = with_holes(k), with_holes(v)
            block_table = None
        metadata = ops.generic_block_sparse_attention_metadata(
            idx_npu, counts_npu, kv_heads * group, kv_heads, 128, [1, 128], **lengths)
        out, _ = ops.generic_block_sparse_attention(
            q_npu, kn, vn, idx_npu, counts_npu, [1, 128], metadata=metadata,
            block_table=block_table, attention_out_dtype=out_dtype, **lengths)
        results.append(out.float().cpu())
    active = [cu_q[b] + t for b in range(batch) for t in range(used_q[b])]
    ref = _reference(q, k, v, cu_q, cu_kv, used_q, used_kv, idx, counts)
    torch.testing.assert_close(results[0][active], results[1][active], rtol=0, atol=2e-3)
    tol = 0.08 if quant else (0.02 if dtype == torch.bfloat16 else 0.006)
    torch.testing.assert_close(results[1][active], ref[active], rtol=tol, atol=tol)
