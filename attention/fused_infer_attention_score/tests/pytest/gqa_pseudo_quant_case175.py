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
import hashlib
import logging
import math
import os
from pathlib import Path

import pytest
import torch
import torch._dynamo
import torch.nn as nn
import torch_npu
import torchair
from torchair.configs.compiler_config import CompilerConfig


logger = logging.getLogger(__name__)

CASE_FILE = Path(__file__).with_name("case175.json")
KV_VALUE = 2.0
FP8_E8M0_ONE = 127
FP8_E8M0_MIN = 125
FP8_E8M0_RANGE = 5
GROUP_SIZE = 32
QUERY_DTYPE = torch.bfloat16
ATTN_RTOL = 1e-2
ATTN_ATOL = 1e-2
LSE_RTOL = 1e-3
LSE_ATOL = 1e-3
DEFAULT_REPEAT_TIMES = 40
REPEAT_TIMES_ENV = "CASE175_REPEAT_TIMES"


class Case175GraphWrapper(nn.Module):
    def __init__(self, keyword_inputs):
        super().__init__()
        self.actual_seq_qlen = keyword_inputs["actual_seq_qlen"]
        self.actual_seq_kvlen = keyword_inputs["actual_seq_kvlen"]
        self.num_query_heads = keyword_inputs["num_query_heads"]
        self.num_key_value_heads = keyword_inputs["num_key_value_heads"]
        self.softmax_scale = keyword_inputs["softmax_scale"]
        self.pre_tokens = keyword_inputs["pre_tokens"]
        self.next_tokens = keyword_inputs["next_tokens"]
        self.input_layout = keyword_inputs["input_layout"]
        self.sparse_mode = keyword_inputs["sparse_mode"]
        self.block_size = keyword_inputs["block_size"]
        self.query_quant_mode = keyword_inputs["query_quant_mode"]
        self.key_quant_mode = keyword_inputs["key_quant_mode"]
        self.value_quant_mode = keyword_inputs["value_quant_mode"]
        self.inner_precise = keyword_inputs["inner_precise"]
        self.return_softmax_lse = keyword_inputs["return_softmax_lse"]
        self.key_dtype = keyword_inputs["key_dtype"]
        self.value_dtype = keyword_inputs["value_dtype"]
        self.dequant_scale_key_dtype = keyword_inputs["dequant_scale_key_dtype"]
        self.dequant_scale_value_dtype = keyword_inputs["dequant_scale_value_dtype"]

    def forward(self, query, key, value, dequant_scale_key, dequant_scale_value):
        return torch_npu.npu_fused_infer_attention_score_v2(
            query,
            key,
            value,
            actual_seq_qlen=self.actual_seq_qlen,
            actual_seq_kvlen=self.actual_seq_kvlen,
            dequant_scale_key=dequant_scale_key,
            dequant_scale_value=dequant_scale_value,
            num_query_heads=self.num_query_heads,
            num_key_value_heads=self.num_key_value_heads,
            softmax_scale=self.softmax_scale,
            pre_tokens=self.pre_tokens,
            next_tokens=self.next_tokens,
            input_layout=self.input_layout,
            sparse_mode=self.sparse_mode,
            block_size=self.block_size,
            query_quant_mode=self.query_quant_mode,
            key_quant_mode=self.key_quant_mode,
            value_quant_mode=self.value_quant_mode,
            inner_precise=self.inner_precise,
            return_softmax_lse=self.return_softmax_lse,
            key_dtype=self.key_dtype,
            value_dtype=self.value_dtype,
            dequant_scale_key_dtype=self.dequant_scale_key_dtype,
            dequant_scale_value_dtype=self.dequant_scale_value_dtype,
        )


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


def _to_npu(tensor):
    try:
        return tensor.to("npu")
    except (RuntimeError, TypeError, NotImplementedError):
        return tensor.npu()


def _to_fp4_e2m1(tensor):
    return torch_npu.npu_dtype_cast(tensor, torch_npu.float4_e2m1fn_x2)


def _make_query(shape):
    data = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
    data = ((data % 17) - 8.0) / 256.0
    return data.to(QUERY_DTYPE).float()


def _make_fp8_e8m0_scale(shape, offset):
    data = torch.arange(math.prod(shape), dtype=torch.int64).reshape(shape)
    data = FP8_E8M0_MIN + ((data + offset) % FP8_E8M0_RANGE)
    return data.to(torch.uint8)


def _build_inputs(case):
    batch = _batch_size(case)
    query_cpu = _make_query(_shape(case, "input0"))
    key_cpu = torch.full(_shape(case, "input1"), KV_VALUE, dtype=torch.float32)
    value_cpu = torch.full(_shape(case, "input2"), KV_VALUE, dtype=torch.float32)
    dequant_scale_key_cpu = _make_fp8_e8m0_scale(_shape(case, "input15"), 0)
    dequant_scale_value_cpu = _make_fp8_e8m0_scale(_shape(case, "input17"), 2)
    raw_actual_seq_kvlen = [int(x) for x in _attr(case, "actualseqlengthskv")]
    assert len(raw_actual_seq_kvlen) >= batch

    key_npu = _to_fp4_e2m1(_to_npu(key_cpu.to(QUERY_DTYPE)))
    value_npu = _to_fp4_e2m1(_to_npu(value_cpu.to(QUERY_DTYPE)))

    return {
        "query": _to_npu(query_cpu.to(QUERY_DTYPE)),
        "key": key_npu,
        "value": value_npu,
        "dequant_scale_key": _to_npu(dequant_scale_key_cpu),
        "dequant_scale_value": _to_npu(dequant_scale_value_cpu),
        "query_cpu": query_cpu,
        "key_cpu": key_cpu,
        "value_cpu": value_cpu,
        "dequant_scale_key_cpu": dequant_scale_key_cpu,
        "dequant_scale_value_cpu": dequant_scale_value_cpu,
        "actual_seq_kvlen": raw_actual_seq_kvlen,
    }


def _batch_size(case):
    return _shape(case, "input0")[0]


def _decode_fp8_e8m0(scale):
    exponent = scale.to(torch.float32) - FP8_E8M0_ONE
    return torch.pow(torch.full_like(exponent, 2.0), exponent)


def _expand_group_scale(raw_scale, batch, num_heads, seq_len, head_dim):
    group_count = head_dim // GROUP_SIZE
    assert head_dim % GROUP_SIZE == 0
    assert tuple(raw_scale.shape) == (1, batch, num_heads, seq_len, group_count)
    scale = _decode_fp8_e8m0(raw_scale)
    scale = scale[0].permute(0, 2, 1, 3).contiguous()
    scale = scale.repeat_interleave(GROUP_SIZE, dim=-1)
    assert tuple(scale.shape) == (batch, seq_len, num_heads, head_dim)
    return scale


def _cpu_reference(case, inputs):
    assert str(_attr(case, "inputlayout")) == "BSH"

    batch, q_seq, hidden = _shape(case, "input0")
    key_batch, kv_seq, kv_hidden = _shape(case, "input1")
    num_heads = int(_attr(case, "numheads"))
    num_kv_heads = int(_attr(case, "numkeyvalueheads"))
    assert batch == key_batch
    assert hidden == kv_hidden
    assert hidden % num_heads == 0
    assert num_heads % num_kv_heads == 0

    head_dim = hidden // num_heads
    query = inputs["query_cpu"].reshape(batch, q_seq, num_heads, head_dim).float()
    key = inputs["key_cpu"].reshape(batch, kv_seq, num_kv_heads, head_dim).float()
    value = inputs["value_cpu"].reshape(batch, kv_seq, num_kv_heads, head_dim).float()
    key = key * _expand_group_scale(inputs["dequant_scale_key_cpu"], batch, num_kv_heads, kv_seq, head_dim)
    value = value * _expand_group_scale(inputs["dequant_scale_value_cpu"], batch, num_kv_heads, kv_seq, head_dim)

    output = torch.empty((batch, q_seq, num_heads, head_dim), dtype=torch.float32)
    softmax_lse = torch.empty(tuple(case["output"][1]["shape"]), dtype=torch.float32)
    actual_seq_kvlen = inputs["actual_seq_kvlen"]
    softmax_scale = float(_attr(case, "scalevalue"))

    for batch_idx in range(batch):
        kv_len = int(actual_seq_kvlen[batch_idx])
        assert 0 < kv_len <= kv_seq
        for head_idx in range(num_heads):
            kv_head_idx = head_idx // (num_heads // num_kv_heads)
            scores = torch.matmul(
                query[batch_idx, :, head_idx, :],
                key[batch_idx, :kv_len, kv_head_idx, :].transpose(0, 1),
            )
            scores = scores * softmax_scale
            probs = torch.softmax(scores, dim=-1)
            output[batch_idx, :, head_idx, :] = torch.matmul(
                probs,
                value[batch_idx, :kv_len, kv_head_idx, :],
            )
            softmax_lse[batch_idx, head_idx, :, 0] = torch.logsumexp(scores, dim=-1)

    return output.reshape(tuple(case["output"][0]["shape"])), softmax_lse


def _get_repeat_times():
    raw_repeat_times = os.getenv(REPEAT_TIMES_ENV, str(DEFAULT_REPEAT_TIMES))
    try:
        repeat_times = int(raw_repeat_times)
    except ValueError:
        pytest.fail(f"{REPEAT_TIMES_ENV} must be a positive integer, got {raw_repeat_times}")
    if repeat_times <= 0:
        pytest.fail(f"{REPEAT_TIMES_ENV} must be a positive integer, got {repeat_times}")
    return repeat_times


def _tensor_hash(tensor):
    data = tensor.contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()[:16]


def _assert_close(name, result, expect, rtol, atol):
    diff = (result - expect).abs()
    max_diff = diff.max().item() if diff.numel() else 0.0
    mean_diff = diff.mean().item() if diff.numel() else 0.0
    rmse = torch.sqrt(torch.mean(diff * diff)).item() if diff.numel() else 0.0
    rel = diff / torch.clamp(expect.abs(), min=1e-12)
    max_rel_diff = rel.max().item() if rel.numel() else 0.0

    close = torch.isclose(result, expect, rtol=rtol, atol=atol, equal_nan=True)
    bad_count = int((~close).sum().item())
    logger.info(
        "%s compared=%s bad_count=%s max_diff=%s max_rel_diff=%s mean_diff=%s rmse=%s",
        name, diff.numel(), bad_count, max_diff, max_rel_diff, mean_diff, rmse,
    )
    if not torch.all(close):
        bad_idx = torch.nonzero(~close.flatten()).flatten()[:20]
        logger.info("%s first bad indices: %s", name, bad_idx)
        logger.info("%s result values: %s", name, result.flatten()[bad_idx])
        logger.info("%s expect values: %s", name, expect.flatten()[bad_idx])
    assert torch.all(close)
    return {
        "bad_count": bad_count,
        "max_diff": max_diff,
        "max_rel_diff": max_rel_diff,
        "mean_diff": mean_diff,
        "rmse": rmse,
    }


def _torch_npu_dtype_name(value):
    dtype_names = (
        "float4_e2m1fn_x2",
        "float8_e8m0fnu",
    )
    for dtype_name in dtype_names:
        if value == getattr(torch_npu, dtype_name):
            return f"torch_npu.{dtype_name}"
    return str(value)


def _format_torch_api_input(name, value):
    if isinstance(value, torch.Tensor):
        return f"shape={tuple(value.shape)} dtype={value.dtype}"
    if name.endswith("_dtype"):
        return f"value={value} dtype={_torch_npu_dtype_name(value)}"
    return f"value={value}"


def _log_torch_api_inputs(api_name, api_inputs):
    logger.info("case175 %s inputs:", api_name)
    for name, value in api_inputs.items():
        logger.info("  %s: %s", name, _format_torch_api_input(name, value))


def _make_torch_api_inputs(case, inputs):
    positional_inputs = {
        "query": inputs["query"],
        "key": inputs["key"],
        "value": inputs["value"],
    }
    keyword_inputs = {
        "actual_seq_qlen": None,
        "actual_seq_kvlen": inputs["actual_seq_kvlen"],
        "dequant_scale_key": inputs["dequant_scale_key"],
        "dequant_scale_value": inputs["dequant_scale_value"],
        "num_query_heads": int(_attr(case, "numheads")),
        "num_key_value_heads": int(_attr(case, "numkeyvalueheads")),
        "softmax_scale": float(_attr(case, "scalevalue")),
        "pre_tokens": int(_attr(case, "pretokens")),
        "next_tokens": int(_attr(case, "nexttokens")),
        "input_layout": str(_attr(case, "inputlayout")),
        "sparse_mode": int(_attr(case, "sparsemode")),
        "block_size": int(_attr(case, "blocksize")),
        "query_quant_mode": int(_attr(case, "query_quant_mode")),
        "key_quant_mode": int(_attr(case, "k_antiquant_mode")),
        "value_quant_mode": int(_attr(case, "v_antiquant_mode")),
        "inner_precise": int(_attr(case, "innerprecise")),
        "return_softmax_lse": bool(_attr(case, "softmax_lse_flag")),
        "key_dtype": torch_npu.float4_e2m1fn_x2,
        "value_dtype": torch_npu.float4_e2m1fn_x2,
        "dequant_scale_key_dtype": torch_npu.float8_e8m0fnu,
        "dequant_scale_value_dtype": torch_npu.float8_e8m0fnu,
    }
    return positional_inputs, keyword_inputs


def _run_npu_case(case, inputs):
    api_name = "torch_npu.npu_fused_infer_attention_score_v2"
    positional_inputs, keyword_inputs = _make_torch_api_inputs(case, inputs)
    _log_torch_api_inputs(api_name, {**positional_inputs, **keyword_inputs})

    result, softmax_lse = torch_npu.npu_fused_infer_attention_score_v2(
        positional_inputs["query"],
        positional_inputs["key"],
        positional_inputs["value"],
        **keyword_inputs,
    )
    torch.npu.synchronize()
    return result, softmax_lse


def _setup_case175_graph_model(case, inputs):
    torch.npu.set_compile_mode(jit_compile=True)
    _, keyword_inputs = _make_torch_api_inputs(case, inputs)
    model = Case175GraphWrapper(keyword_inputs).npu()

    config = CompilerConfig()
    config.experimental_config.keep_inference_input_mutations = True
    config.experimental_config.frozen_parameter = True
    config.experimental_config.tiling_schedule_optimize = True

    npu_backend = torchair.get_npu_backend(compiler_config=config)
    return torch.compile(model, fullgraph=True, backend=npu_backend, dynamic=False)


def _mark_static_graph_inputs(inputs):
    torch._dynamo.mark_static(inputs["query"])
    torch._dynamo.mark_static(inputs["key"])
    torch._dynamo.mark_static(inputs["value"])
    torch._dynamo.mark_static(inputs["dequant_scale_key"])
    torch._dynamo.mark_static(inputs["dequant_scale_value"])


def _run_npu_case_graph(case, inputs, compiled_model):
    api_name = "torch.compile(torch_npu.npu_fused_infer_attention_score_v2)"
    positional_inputs, keyword_inputs = _make_torch_api_inputs(case, inputs)
    _log_torch_api_inputs(api_name, {**positional_inputs, **keyword_inputs})
    _mark_static_graph_inputs(inputs)

    result, softmax_lse = compiled_model(
        positional_inputs["query"],
        positional_inputs["key"],
        positional_inputs["value"],
        keyword_inputs["dequant_scale_key"],
        keyword_inputs["dequant_scale_value"],
    )
    torch.npu.synchronize()
    return result, softmax_lse


def _verify_one_run(case, inputs, expect_attention, expect_lse, run_idx, runner):
    result, softmax_lse = runner(case, inputs)
    result_attention = result.cpu().float()
    assert result_attention.shape == expect_attention.shape
    assert result.dtype == QUERY_DTYPE
    attention_metrics = _assert_close(
        f"run{run_idx:02d}_attention_out",
        result_attention,
        expect_attention,
        ATTN_RTOL,
        ATTN_ATOL,
    )

    result_lse = softmax_lse.cpu().float()
    assert result_lse.shape == expect_lse.shape
    lse_metrics = _assert_close(
        f"run{run_idx:02d}_softmax_lse",
        result_lse,
        expect_lse,
        LSE_RTOL,
        LSE_ATOL,
    )
    return {
        "attention": result_attention,
        "softmax_lse": result_lse,
        "attention_metrics": attention_metrics,
        "lse_metrics": lse_metrics,
        "attention_hash": _tensor_hash(result_attention),
        "lse_hash": _tensor_hash(result_lse),
    }


def _summarize_metric(records, metric_name, value_name):
    values = [record[metric_name][value_name] for record in records]
    return min(values), max(values), sum(values) / len(values)


def _log_repeat_summary(records):
    attn_hashes = {record["attention_hash"] for record in records}
    lse_hashes = {record["lse_hash"] for record in records}
    post_warmup_attn_hashes = {record["attention_hash"] for record in records[1:]}
    post_warmup_lse_hashes = {record["lse_hash"] for record in records[1:]}
    logger.info(
        "case175 repeat summary runs=%s attention_unique_hashes=%s softmax_lse_unique_hashes=%s "
        "post_warmup_attention_unique_hashes=%s post_warmup_softmax_lse_unique_hashes=%s",
        len(records),
        len(attn_hashes),
        len(lse_hashes),
        len(post_warmup_attn_hashes),
        len(post_warmup_lse_hashes),
    )
    for metric_name, log_name in (("attention_metrics", "attention"), ("lse_metrics", "softmax_lse")):
        for value_name in ("bad_count", "max_diff", "max_rel_diff", "rmse"):
            min_value, max_value, mean_value = _summarize_metric(records, metric_name, value_name)
            logger.info(
                "case175 %s %s min=%s max=%s mean=%s",
                log_name,
                value_name,
                min_value,
                max_value,
                mean_value,
            )


def run_gqa_pseudo_quant_case175(repeat_times=None):
    if not torch.npu.is_available():
        pytest.skip("case175 requires an available NPU device")

    case = _load_case()
    torch.npu.set_device(0)

    inputs = _build_inputs(case)
    assert len(inputs["actual_seq_kvlen"]) >= _batch_size(case)
    assert inputs["key"].shape[-1] * 2 == _shape(case, "input1")[-1]
    assert inputs["value"].shape[-1] * 2 == _shape(case, "input2")[-1]
    expect_attention, expect_lse = _cpu_reference(case, inputs)

    repeat_times = _get_repeat_times() if repeat_times is None else int(repeat_times)
    if repeat_times <= 0:
        pytest.fail(f"repeat_times must be a positive integer, got {repeat_times}")
    records = []
    first_attention = None
    first_lse = None
    runner = _run_npu_case
    for run_idx in range(1, repeat_times + 1):
        record = _verify_one_run(case, inputs, expect_attention, expect_lse, run_idx, runner)
        if first_attention is None:
            first_attention = record["attention"]
            first_lse = record["softmax_lse"]
            attention_vs_first = 0.0
            lse_vs_first = 0.0
        else:
            attention_vs_first = (record["attention"] - first_attention).abs().max().item()
            lse_vs_first = (record["softmax_lse"] - first_lse).abs().max().item()
        record["attention_vs_first_max"] = attention_vs_first
        record["lse_vs_first_max"] = lse_vs_first
        logger.info(
            "case175 run=%s/%s attention_hash=%s softmax_lse_hash=%s "
            "attention_vs_first_max=%s softmax_lse_vs_first_max=%s",
            run_idx,
            repeat_times,
            record["attention_hash"],
            record["lse_hash"],
            attention_vs_first,
            lse_vs_first,
        )
        records.append(record)

    _log_repeat_summary(records)
    return records


def test_gqa_pseudo_quant_case175():
    run_gqa_pseudo_quant_case175()


def run_gqa_pseudo_quant_case175_graph(repeat_times=None):
    if not torch.npu.is_available():
        pytest.skip("case175 graph mode requires an available NPU device")

    case = _load_case()
    torch.npu.set_device(0)

    inputs = _build_inputs(case)
    assert len(inputs["actual_seq_kvlen"]) >= _batch_size(case)
    assert inputs["key"].shape[-1] * 2 == _shape(case, "input1")[-1]
    assert inputs["value"].shape[-1] * 2 == _shape(case, "input2")[-1]
    expect_attention, expect_lse = _cpu_reference(case, inputs)

    compiled_model = _setup_case175_graph_model(case, inputs)
    runner = lambda current_case, current_inputs: _run_npu_case_graph(
        current_case, current_inputs, compiled_model
    )

    repeat_times = _get_repeat_times() if repeat_times is None else int(repeat_times)
    if repeat_times <= 0:
        pytest.fail(f"repeat_times must be a positive integer, got {repeat_times}")
    records = []
    first_attention = None
    first_lse = None
    for run_idx in range(1, repeat_times + 1):
        record = _verify_one_run(case, inputs, expect_attention, expect_lse, run_idx, runner)
        if first_attention is None:
            first_attention = record["attention"]
            first_lse = record["softmax_lse"]
            attention_vs_first = 0.0
            lse_vs_first = 0.0
        else:
            attention_vs_first = (record["attention"] - first_attention).abs().max().item()
            lse_vs_first = (record["softmax_lse"] - first_lse).abs().max().item()
        record["attention_vs_first_max"] = attention_vs_first
        record["lse_vs_first_max"] = lse_vs_first
        logger.info(
            "case175 graph run=%s/%s attention_hash=%s softmax_lse_hash=%s "
            "attention_vs_first_max=%s softmax_lse_vs_first_max=%s",
            run_idx,
            repeat_times,
            record["attention_hash"],
            record["lse_hash"],
            attention_vs_first,
            lse_vs_first,
        )
        records.append(record)

    _log_repeat_summary(records)
    return records


def test_gqa_pseudo_quant_case175_graph():
    run_gqa_pseudo_quant_case175_graph()
