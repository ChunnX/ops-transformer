#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import json
import logging
from pathlib import Path

import pytest
import torch
import torch_npu


logger = logging.getLogger(__name__)

CASE_FILE = Path(__file__).with_name("case803.json")
QUERY_VALUE = 0.0
KV_QUANT_VALUE = -2.0
ANTIQUANT_SCALE = 1.0
ATTN_RTOL = 1e-2
ATTN_ATOL = 1e-2
LSE_RTOL = 1e-3
LSE_ATOL = 1e-3
QUERY_DTYPE = torch.float16
# The V5 template is HIFLOAT8 + sparse_mode=2. The v1 torch_npu API has no
# key/value dtype override, so the pytest path uses native FP8 tensors and a
# dense mask-free call while preserving the case shape, layout, heads and seqs.
V1_SPARSE_MODE = 0


def _load_case():
    with CASE_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def _item(case, name):
    for item in case["input"]:
        if item["name"] == name:
            return item
    raise KeyError(f"{name} is not found in {CASE_FILE}")


def _attr(case, name):
    return _item(case, name)["value"]


def _shape(case, name):
    item = _item(case, name)
    if "shape" in item:
        return tuple(item["shape"])
    return tuple(item["shapes"][0])


def _full_npu(shape, value, dtype):
    try:
        return torch.full(shape, value, dtype=dtype, device="npu")
    except (RuntimeError, TypeError, NotImplementedError):
        return torch.full(shape, value, dtype=dtype).npu()


def _build_inputs(case):
    query_shape = _shape(case, "input0")
    key_shape = _shape(case, "input1")

    return {
        "query": _full_npu(query_shape, QUERY_VALUE, QUERY_DTYPE),
        "key": _full_npu(key_shape, KV_QUANT_VALUE, torch.float8_e4m3fn),
        "value": _full_npu(_shape(case, "input2"), KV_QUANT_VALUE, torch.float8_e4m3fn),
        "atten_mask": None,
        "dequant_scale1": _full_npu(_shape(case, "input5"), 0.0, torch.float32),
        "quant_scale1": _full_npu(_shape(case, "input6"), 1.0, torch.float32),
        "key_antiquant_scale": _full_npu(_shape(case, "input15"), ANTIQUANT_SCALE, QUERY_DTYPE),
        "value_antiquant_scale": _full_npu(_shape(case, "input17"), ANTIQUANT_SCALE, QUERY_DTYPE),
        "actual_seq_lengths": [int(x) for x in _attr(case, "actualseqlengths")],
        "actual_seq_lengths_kv": [int(x) for x in _attr(case, "actualseqlengthskv")],
    }


def _expected_attention(case, actual_seq_lengths):
    expect = torch.zeros(tuple(case["output"][0]["shape"]), dtype=torch.float32)
    for batch_idx, q_len in enumerate(actual_seq_lengths):
        expect[batch_idx, :q_len, :] = KV_QUANT_VALUE * ANTIQUANT_SCALE
    return expect


def _expected_softmax_lse(case, actual_seq_lengths, actual_seq_lengths_kv):
    expect = torch.zeros(tuple(case["output"][1]["shape"]), dtype=torch.float32)
    for batch_idx, q_len in enumerate(actual_seq_lengths):
        kv_len = actual_seq_lengths_kv[batch_idx]
        valid_counts = torch.full((q_len,), kv_len, dtype=torch.float32)
        expect[batch_idx, :, :q_len, 0] = torch.log(valid_counts).view(1, q_len)
    return expect


def _valid_attention_mask(expect_shape, actual_seq_lengths):
    mask = torch.zeros(expect_shape, dtype=torch.bool)
    for batch_idx, q_len in enumerate(actual_seq_lengths):
        mask[batch_idx, :q_len, :] = True
    return mask


def _valid_lse_mask(expect_shape, actual_seq_lengths):
    mask = torch.zeros(expect_shape, dtype=torch.bool)
    for batch_idx, q_len in enumerate(actual_seq_lengths):
        mask[batch_idx, :, :q_len, :] = True
    return mask


def _assert_close(name, result, expect, rtol, atol, mask=None):
    if mask is not None:
        result = result[mask]
        expect = expect[mask]

    diff = (result - expect).abs()
    max_diff = diff.max().item() if diff.numel() else 0.0
    mean_diff = diff.mean().item() if diff.numel() else 0.0
    rmse = torch.sqrt(torch.mean(diff * diff)).item() if diff.numel() else 0.0
    logger.info("%s max_diff=%s mean_diff=%s rmse=%s", name, max_diff, mean_diff, rmse)

    close = torch.isclose(result, expect, rtol=rtol, atol=atol, equal_nan=True)
    if not torch.all(close):
        bad_idx = torch.nonzero(~close.flatten()).flatten()[:20]
        logger.info("%s first bad indices: %s", name, bad_idx)
        logger.info("%s result values: %s", name, result.flatten()[bad_idx])
        logger.info("%s expect values: %s", name, expect.flatten()[bad_idx])
    assert torch.all(close)


def test_gqa_pseudo_quant_case803():
    if not torch.npu.is_available():
        pytest.skip("case803 requires an available NPU device")

    case = _load_case()
    torch.npu.set_device(0)

    inputs = _build_inputs(case)
    result, softmax_lse = torch_npu.npu_fused_infer_attention_score(
        inputs["query"],
        inputs["key"],
        inputs["value"],
        atten_mask=inputs["atten_mask"],
        actual_seq_lengths=inputs["actual_seq_lengths"],
        actual_seq_lengths_kv=inputs["actual_seq_lengths_kv"],
        dequant_scale1=inputs["dequant_scale1"],
        quant_scale1=inputs["quant_scale1"],
        key_antiquant_scale=inputs["key_antiquant_scale"],
        value_antiquant_scale=inputs["value_antiquant_scale"],
        num_heads=int(_attr(case, "numheads")),
        scale=float(_attr(case, "scalevalue")),
        pre_tokens=int(_attr(case, "pretokens")),
        next_tokens=int(_attr(case, "nexttokens")),
        input_layout=str(_attr(case, "inputlayout")),
        num_key_value_heads=int(_attr(case, "numkeyvalueheads")),
        sparse_mode=V1_SPARSE_MODE,
        inner_precise=int(_attr(case, "innerprecise")),
        block_size=int(_attr(case, "blocksize")),
        antiquant_mode=int(_attr(case, "antiquant_mode")),
        softmax_lse_flag=bool(_attr(case, "softmax_lse_flag")),
        key_antiquant_mode=int(_attr(case, "k_antiquant_mode")),
        value_antiquant_mode=int(_attr(case, "v_antiquant_mode")),
    )
    torch.npu.synchronize()

    expect_attention = _expected_attention(case, inputs["actual_seq_lengths"])
    result_attention = result.cpu().float()
    assert result_attention.shape == expect_attention.shape
    assert result.dtype == QUERY_DTYPE
    _assert_close(
        "attention_out",
        result_attention,
        expect_attention,
        ATTN_RTOL,
        ATTN_ATOL,
        mask=_valid_attention_mask(expect_attention.shape, inputs["actual_seq_lengths"]),
    )

    expect_lse = _expected_softmax_lse(
        case,
        inputs["actual_seq_lengths"],
        inputs["actual_seq_lengths_kv"],
    )
    result_lse = softmax_lse.cpu().float()
    assert result_lse.shape == expect_lse.shape
    _assert_close(
        "softmax_lse",
        result_lse,
        expect_lse,
        LSE_RTOL,
        LSE_ATOL,
        mask=_valid_lse_mask(expect_lse.shape, inputs["actual_seq_lengths"]),
    )
