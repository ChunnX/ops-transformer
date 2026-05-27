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

CASE_FILE = Path(__file__).with_name("case0382.json")
FP8_E4M3FN_DTYPE_ENUM = 24
QUERY_VALUE = -0.001
KV_FP8_VALUE = 0.25
ATTN_RTOL = 0.05
ATTN_ATOL = 0.05
LSE_RTOL = 1e-3
LSE_ATOL = 1e-3


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


def _upper_range(case, name):
    return float(_item(case, name)["data_range"][1])


def _full_npu(shape, value, dtype):
    try:
        return torch.full(shape, value, dtype=dtype, device="npu")
    except (RuntimeError, TypeError, NotImplementedError):
        return torch.full(shape, value, dtype=dtype).npu()


def _make_block_table(actual_seq_lens_kv, block_size, max_block_num):
    block_table = torch.full((len(actual_seq_lens_kv), max_block_num), -1, dtype=torch.int32)
    block_idx = 0
    for batch_idx, kv_len in enumerate(actual_seq_lens_kv):
        block_num = (kv_len + block_size - 1) // block_size
        if block_num > 0:
            block_table[batch_idx, :block_num] = torch.arange(
                block_idx, block_idx + block_num, dtype=torch.int32
            )
            block_idx += block_num
    return block_table


def _nz_cache_shape(case):
    # case0382 is a PA + NZ template. The logical dense key/value shapes are also
    # present in the JSON, but the executable PA cache is the rank-5 NZ tensor.
    return _shape(case, "input21")


def _build_inputs(case):
    actual_seq_lens_kv = [int(x) for x in _attr(case, "actualseqlengthskv")]
    block_size = int(_attr(case, "blocksize"))
    kv_cache_shape = _nz_cache_shape(case)
    block_table = _make_block_table(actual_seq_lens_kv, block_size, _shape(case, "input12")[1])

    antiquant_scale = _upper_range(case, "input15")
    quant_scale = _upper_range(case, "input8")

    query = _full_npu(_shape(case, "input0"), QUERY_VALUE, torch.float16)
    key = _full_npu(kv_cache_shape, KV_FP8_VALUE, torch.float8_e4m3fn)
    value = _full_npu(kv_cache_shape, KV_FP8_VALUE, torch.float8_e4m3fn)
    pse_shift = _full_npu(_shape(case, "input3"), 0.0, torch.float16)
    atten_mask = _full_npu(_shape(case, "input4"), False, torch.bool)
    quant_scale_out = _full_npu(_shape(case, "input8"), quant_scale, torch.float32)
    key_antiquant_scale = _full_npu(_shape(case, "input15"), antiquant_scale, torch.float16)
    value_antiquant_scale = _full_npu(_shape(case, "input17"), antiquant_scale, torch.float16)

    return {
        "query": query,
        "key": key,
        "value": value,
        "pse_shift": pse_shift,
        "atten_mask": atten_mask,
        "actual_seq_lens_kv": actual_seq_lens_kv,
        "quant_scale_out": quant_scale_out,
        "block_table": block_table.npu(),
        "key_antiquant_scale": key_antiquant_scale,
        "value_antiquant_scale": value_antiquant_scale,
        "antiquant_scale": antiquant_scale,
        "quant_scale": quant_scale,
    }


def _expected_attention(case, actual_seq_lens_kv, antiquant_scale, quant_scale):
    out_shape = tuple(case["output"][0]["shape"])
    attention_value = KV_FP8_VALUE * antiquant_scale * quant_scale
    expect = torch.full(out_shape, attention_value, dtype=torch.float32)
    valid_batch = torch.tensor(actual_seq_lens_kv, dtype=torch.bool)
    expect[~valid_batch, :, :, :] = 0.0
    return expect.to(torch.float8_e4m3fn).float()


def _expected_softmax_lse(case, actual_seq_lens_kv, antiquant_scale):
    out_shape = tuple(case["output"][1]["shape"])
    head_dim = _shape(case, "input0")[-1]
    scale_value = float(_attr(case, "scalevalue"))
    score = head_dim * QUERY_VALUE * KV_FP8_VALUE * antiquant_scale * scale_value

    seq_lens = torch.tensor(actual_seq_lens_kv, dtype=torch.float32)
    lse = torch.log(seq_lens.clamp_min(1.0)).view(-1, 1, 1, 1) + score
    lse = lse.expand(out_shape).clone()
    lse[seq_lens == 0, :, :, :] = float("-inf")
    return lse


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


def test_gqa_pseudo_quant_case0382():
    if not torch.npu.is_available():
        pytest.skip("case0382 requires an available NPU device")

    case = _load_case()
    torch.npu.set_device(0)

    inputs = _build_inputs(case)
    result, softmax_lse = torch_npu.npu_fused_infer_attention_score_v2(
        inputs["query"],
        inputs["key"],
        inputs["value"],
        pse_shift=inputs["pse_shift"],
        atten_mask=inputs["atten_mask"],
        actual_seq_qlen=None,
        actual_seq_kvlen=inputs["actual_seq_lens_kv"],
        block_table=inputs["block_table"],
        dequant_scale_key=inputs["key_antiquant_scale"],
        dequant_scale_value=inputs["value_antiquant_scale"],
        quant_scale_out=inputs["quant_scale_out"],
        num_query_heads=int(_attr(case, "numheads")),
        num_key_value_heads=int(_attr(case, "numkeyvalueheads")),
        softmax_scale=float(_attr(case, "scalevalue")),
        pre_tokens=int(_attr(case, "pretokens")),
        next_tokens=int(_attr(case, "nexttokens")),
        input_layout=str(_attr(case, "inputlayout")),
        sparse_mode=int(_attr(case, "sparsemode")),
        block_size=int(_attr(case, "blocksize")),
        query_quant_mode=int(_attr(case, "query_quant_mode")),
        key_quant_mode=int(_attr(case, "k_antiquant_mode")),
        value_quant_mode=int(_attr(case, "v_antiquant_mode")),
        inner_precise=int(_attr(case, "innerprecise")),
        return_softmax_lse=bool(_attr(case, "softmax_lse_flag")),
        out_dtype=FP8_E4M3FN_DTYPE_ENUM,
    )
    torch.npu.synchronize()

    expect_attention = _expected_attention(
        case,
        inputs["actual_seq_lens_kv"],
        inputs["antiquant_scale"],
        inputs["quant_scale"],
    )
    result_attention = result.cpu().float()
    assert result_attention.shape == expect_attention.shape
    assert result.dtype == torch.float8_e4m3fn
    _assert_close("attention_out", result_attention, expect_attention, ATTN_RTOL, ATTN_ATOL)

    expect_lse = _expected_softmax_lse(case, inputs["actual_seq_lens_kv"], inputs["antiquant_scale"])
    result_lse = softmax_lse.cpu().float()
    assert result_lse.shape == expect_lse.shape
    valid_lse = torch.tensor(inputs["actual_seq_lens_kv"], dtype=torch.bool).view(-1, 1, 1, 1).expand_as(expect_lse)
    _assert_close("softmax_lse", result_lse, expect_lse, LSE_RTOL, LSE_ATOL, mask=valid_lse)
