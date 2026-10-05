# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Generated native-parameter matrix against quantized CPU and FP32 SDPA references."""

import pytest
import torch
import torch_npu  # noqa: F401 -- registers the NPU backend
from sageattention2 import reference, sageattn_qk_int8_pv_fp8_asc
from task_cases import generate_cases


@pytest.mark.filterwarnings("error:.*falls back to.*:UserWarning")
@pytest.mark.parametrize("case", list(generate_cases()), ids=lambda c: c["id"])
def test_task_document_matrix(case, record_property):
    torch.set_num_threads(8)
    s = case["shape"]
    b, hq, hk = s["batch_size"], s["num_qo_heads"], s["num_kv_heads"]
    sq, sk, d = s["qo_len"], s["kv_len"], s["head_dim"]
    dtype = torch.float16 if case["dtype"] == "float16" else torch.bfloat16
    generator = torch.Generator().manual_seed(case["seed"])
    layout = case["tensor_layout"]

    def make(heads, seq):
        shape = (b, seq, heads, d) if layout == "NHD" else (b, heads, seq, d)
        if case["distribution"] == "uniform":
            return (2 * torch.rand(shape, generator=generator) - 1).to(dtype)
        return torch.randn(shape, generator=generator).to(dtype)

    q, k, v = make(hq, sq), make(hk, sk), make(hk, sk)
    gran = case["qk_quant_gran"]
    options = {
        "tensor_layout": layout,
        "is_causal": case["is_causal"],
        "qk_quant_gran": gran,
        "sm_scale": case["sm_scale"],
        "pv_accum_dtype": case["pv_accum_dtype"],
        "smooth_k": case["smooth_k"],
        "return_lse": case["return_lse"],
    }
    if case["expect"] == "error":
        with pytest.raises(ValueError, match=case["error_match"]):
            sageattn_qk_int8_pv_fp8_asc(q.npu(), k.npu(), v.npu(), **options)
        return
    result = sageattn_qk_int8_pv_fp8_asc(q.npu(), k.npu(), v.npu(), **options)
    actual, lse = result if case["return_lse"] else (result, None)
    actual = actual.cpu()
    assert actual.shape == q.shape and actual.dtype == dtype
    assert torch.isfinite(actual).all()
    gold_options = {
        key: options[key]
        for key in ("is_causal", "sm_scale", "smooth_k", "return_lse")
    }
    qc, kc, vc = (
        (q, k, v) if layout != "NHD" else tuple(t.transpose(1, 2) for t in (q, k, v))
    )
    golden = reference.attention(qc, kc, vc, **gold_options)
    golden, gold_lse = golden if lse is not None else (golden, None)
    if layout == "NHD":
        golden = golden.transpose(1, 2)
    error = (actual.float() - golden.float()).abs()
    tolerance = 2 ** (-9 if dtype == torch.float16 else -6)
    matched = (error <= tolerance * (1 + golden.float().abs())).float().mean().item()
    record_property("reference", "per-token QK / MXFP8 PV CPU algorithm")
    record_property("matched_ratio", matched)
    max_abs = error.max().item()
    record_property("max_abs", max_abs)
    limit = 0.1 if dtype == torch.float16 else 1.0
    record_property("original_max_abs_limit", limit)
    record_property("strict_max_abs_passed", str(max_abs <= limit).lower())
    record_property("effective_max_abs_limit", limit)
    assert matched >= 0.99
    assert max_abs <= limit
    if lse is not None:
        assert tuple(lse.shape) == (b, hq, sq)
        torch.testing.assert_close(lse.cpu(), gold_lse, atol=0.002, rtol=0.001)
    if case["suite"] == "accuracy":
        truth = torch.nn.functional.scaled_dot_product_attention(
            qc.float(),
            kc.float(),
            vc.float(),
            is_causal=case["is_causal"],
            scale=case["sm_scale"],
            enable_gqa=True,
        )
        base_options = dict(gold_options, return_lse=False)
        baseline = reference.attention(
            qc, kc, vc, score_mode="fp32", **base_options
        ).float()
        measured = actual if layout != "NHD" else actual.transpose(1, 2)
        err = (measured.float() - truth).abs()
        ref = (baseline - truth).abs()
        for name, reduce, limit in (
            ("max", lambda x: x.max(), 2.0),
            ("mean", lambda x: x.mean(), 1.2),
            ("rmse", lambda x: x.square().mean().sqrt(), 1.2),
        ):
            value, ref_value = reduce(err).item(), reduce(ref).item()
            record_property(name + "_vs_sdpa", value)
            record_property(name + "_error_ratio", value / max(ref_value, 1e-30))
            assert value <= limit * ref_value + 1e-6
