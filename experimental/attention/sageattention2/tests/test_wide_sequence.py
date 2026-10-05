# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Full-sequence wide regression and acceptance-shape coverage.

Reduced-head cases check every output against the CPU algorithm. P-01--P-07
execute the full acceptance shape, compare both APIs and LSE on every element,
and check the first/last query heads against CPU over the complete sequence.
"""

import pytest
import torch
import torch_npu  # noqa: F401
from sageattention2 import reference, sageattn
from sageattention2.aclnn import run as aclnn_run


def check_wide(q, k, v, causal, layout, sampled_heads, record_property):
    torch.set_num_threads(8)
    b, hq, seq, dim = q.shape
    hk = k.shape[1]
    assert dim == 128 and seq % 128 == 0 and k.shape[2] % 512 == 0
    assert k.shape[2] >= 4096  # Same predicate as the production tiler.
    record_property("kernel_path", "wide N512")
    record_property("shape", str((b, hq, hk, seq, k.shape[2], dim)))
    record_property("dtype", str(q.dtype))
    record_property("causal", str(causal))
    record_property("layout", layout)
    record_property("cpu_coverage", "first/last head, all batches and rows" if sampled_heads else "all elements")
    inputs = [x.npu() for x in (q, k, v)]
    if layout == "NHD":
        inputs = [x.transpose(1, 2).contiguous() for x in inputs]
    options = dict(tensor_layout=layout, is_causal=causal, smooth_k=True)
    public, lse = sageattn(*inputs, return_lse=True, **options)
    native, native_lse = aclnn_run(*inputs, return_lse=True, **options)
    plain = sageattn(*inputs, **options)
    public, native, plain = [x.cpu() for x in (public, native, plain)]
    lse, native_lse = lse.cpu(), native_lse.cpu()
    tol = 2 ** (-9 if q.dtype == torch.float16 else -6)
    for value in (public, native, plain, lse, native_lse):
        assert torch.isfinite(value).all()
    # Independent stats reductions may round boundary tokens differently.
    torch.testing.assert_close(public, native, atol=tol, rtol=tol)
    torch.testing.assert_close(public, plain, atol=tol, rtol=tol)
    torch.testing.assert_close(lse, native_lse, atol=0.002, rtol=0.001)
    if layout == "NHD":
        public = public.transpose(1, 2)
    heads = [0, hq - 1] if sampled_heads else list(range(hq))
    errors, golds, lse_errors = [], [], []
    for h in heads:
        kh = h // (hq // hk)
        gold, gold_lse = reference.attention(
            q[:, h:h+1], k[:, kh:kh+1], v[:, kh:kh+1],
            is_causal=causal, smooth_k=True, return_lse=True, block_k=512,
        )
        actual = public[:, h:h+1].float()
        errors.append((actual - gold.float()).abs())
        golds.append(gold.float())
        lse_errors.append((lse[:, h:h+1] - gold_lse).abs())
        torch.testing.assert_close(lse[:, h:h+1], gold_lse, atol=0.002, rtol=0.001)
    error, gold = torch.cat(errors, 1), torch.cat(golds, 1)
    matched = (error <= tol * (1 + gold.abs())).float().mean().item()
    record_property("reference", "CPU per-token QK / MXFP8 PV, block_k=512")
    record_property("matched_ratio", matched)
    record_property("max_abs", error.max().item())
    record_property("rmse", error.square().mean().sqrt().item())
    record_property("lse_max_abs", torch.cat(lse_errors, 1).max().item())
    assert matched >= 0.99
    assert error.max() <= (0.1 if q.dtype == torch.float16 else 1.0)


@pytest.mark.parametrize("seq", [4096, 8192, 16384])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
def test_wide_full_sequence(seq, dtype, causal, record_property):
    torch.manual_seed(951 + seq + int(causal))
    # Multiple queries per core, query-boundary ring reuse, GQA and both dies.
    b, hq, hk = (2, 2, 1) if seq == 4096 else (1, 2, 1)
    q = torch.randn(b, hq, seq, 128, dtype=dtype)
    k = torch.randn(b, hk, seq, 128, dtype=dtype)
    v = torch.randn_like(k)
    check_wide(q, k, v, causal, "NHD" if causal else "HND", False, record_property)


# Field order matches benchmarks/cases.py: name, seq, batch, hk.
# All acceptance cases use Hq=32 and D=128.
PERFORMANCE_SHAPES = [
    ("P-01", 4096, 1, 32, torch.float16, False),
    ("P-02", 8192, 1, 32, torch.float16, False),
    ("P-03", 8192, 1, 32, torch.float16, True),
    ("P-04", 8192, 2, 32, torch.float16, False),
    ("P-05", 4096, 1, 8, torch.float16, False),
    ("P-06", 4096, 1, 32, torch.bfloat16, False),
    ("P-07", 16384, 1, 32, torch.float16, False),
]


@pytest.mark.parametrize("sq,hq", [(128, 64), (1024, 4)],
                         ids=["many_heads", "many_query_tiles"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_wide_pair_odd_query_tail(sq, hq, dtype, record_property):
    # More than 28 query tiles force per-core transitions within/across heads.
    # Nine groups leave one unpaired output at each query boundary and drain.
    # Causal fronts use the square shapes in test_wide_full_sequence above.
    sk = 4608
    torch.manual_seed(975 + sq + sk)
    q = torch.randn(1, hq, sq, 128, dtype=dtype)
    k = torch.randn(1, 1, sk, 128, dtype=dtype)
    v = torch.randn_like(k)
    check_wide(q, k, v, False, "HND", False, record_property)


@pytest.mark.parametrize("name,seq,b,hk,dtype,causal", PERFORMANCE_SHAPES, ids=[c[0] for c in PERFORMANCE_SHAPES])
def test_wide_acceptance_shape(name, seq, b, hk, dtype, causal, record_property):
    torch.manual_seed(960 + seq)
    record_property("performance_case", name)
    q = torch.randn(b, 32, seq, 128, dtype=dtype)
    k = torch.randn(b, hk, seq, 128, dtype=dtype)
    v = torch.randn_like(k)
    check_wide(q, k, v, causal, "HND", True, record_property)
