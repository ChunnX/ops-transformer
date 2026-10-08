# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import pytest
import torch

from core.cpu import run_cpu_golden, softmax_flash, tforward
from core.case_loader import normalize_params
from core.gen_data import generate_inputs
from testcase.functional_stc import TestCases


def _base_params(s2, sinks=None):
    return {
        "B": 1,
        "N1": 2,
        "N2": 1,
        "S1": 2,
        "S2": s2,
        "D": 64,
        "layout_q": "BNSD",
        "layout_kv": "PA_BNBD",
        "layout_attn_out": "BNSD",
        "block_size": 64,
        "quant_compute_mode": 1,
        "seqused_q": [2],
        "seqused_kv": [s2],
        "max_seqlen_q": 2,
        "max_seqlen_kv": s2,
        "return_softmax_lse": True,
        "sinks": sinks,
    }


def _layout_params(layout_q, layout_kv, b, n1, n2, s1, s2, sinks=None,
                   quant_compute_mode=1, seqused_q=None, seqused_kv=None):
    seqused_q = seqused_q or [s1] * b
    seqused_kv = seqused_kv or [s2] * b
    params = _base_params(s2, sinks=sinks)
    params.update({
        "B": b,
        "N1": n1,
        "N2": n2,
        "S1": s1,
        "layout_q": layout_q,
        "layout_kv": layout_kv,
        "layout_attn_out": layout_q,
        "quant_compute_mode": quant_compute_mode,
        "seqused_q": seqused_q,
        "seqused_kv": seqused_kv,
        "max_seqlen_q": max(seqused_q, default=0),
        "max_seqlen_kv": max(seqused_kv, default=0),
    })
    if layout_q == "TND":
        cumulative = [0]
        for length in seqused_q:
            cumulative.append(cumulative[-1] + length)
        params["cu_seqlens_q"] = cumulative
    return params


def _scale_output_for_layout(scale, layout_q):
    if layout_q == "BSND":
        scale = scale.transpose(1, 2)
    return scale.unsqueeze(-1)


def test_generate_empty_kv_and_fia_zero_output_golden():
    inputs = generate_inputs(_base_params(0), seed=1)

    assert inputs["k"].numel() == 0
    assert inputs["v"].numel() == 0
    assert tuple(inputs["block_table"].shape) == (1, 0)

    result = run_cpu_golden(inputs)
    assert torch.count_nonzero(result["attn_out"]) == 0
    assert torch.isposinf(result["softmax_lse"]).all()


def test_physical_empty_kv_without_seqused_kv_uses_fia_lse_contract():
    inputs = generate_inputs(_base_params(0), seed=1)
    inputs["seqused_kv"] = None

    result = run_cpu_golden(inputs)

    assert torch.count_nonzero(result["attn_out"]) == 0
    assert torch.isposinf(result["softmax_lse"]).all()


def test_empty_kv_with_sink_has_sink_lse_and_zero_output():
    sinks = torch.tensor([0.25, -0.5], dtype=torch.float32)
    inputs = generate_inputs(_base_params(0, sinks=sinks), seed=1)
    result = run_cpu_golden(inputs)

    assert torch.count_nonzero(result["attn_out"]) == 0
    expected_lse = sinks.reshape(1, 2, 1).expand(1, 2, 2)
    torch.testing.assert_close(result["softmax_lse"], expected_lse)


def test_physical_empty_kv_with_nonzero_declared_length_keeps_infinite_lse():
    params = _base_params(2048, sinks=torch.tensor([0.25, -0.5]))
    params["N2"] = 0
    params["seqused_kv"] = [2048]
    inputs = generate_inputs(params, seed=1)

    result = run_cpu_golden(inputs)

    assert inputs["k"].numel() == 0
    assert torch.count_nonzero(result["attn_out"]) == 0
    assert torch.isposinf(result["softmax_lse"]).all()


def test_zero_batch_query_is_generated_and_has_empty_outputs():
    raw = {
        key: value[0] if isinstance(value, list) else value
        for key, value in TestCases["b270_empty_q_b"].items()
    }
    inputs = generate_inputs(normalize_params(raw), seed=21)
    result = run_cpu_golden(inputs)

    assert tuple(inputs["q"].shape) == (0, 1, 8, 128)
    assert tuple(inputs["block_table"].shape) == (0, 0)
    assert tuple(result["attn_out"].shape) == (0, 1, 8, 128)
    assert tuple(result["softmax_lse"].shape) == (0, 8, 1)


def test_generated_sink_changes_normal_attention_normalization():
    no_sink_inputs = generate_inputs(_base_params(2), seed=7)
    sink_inputs = dict(no_sink_inputs)
    sink_inputs["sinks"] = torch.full((2,), 20.0, dtype=torch.float32)

    no_sink = run_cpu_golden(no_sink_inputs)
    with_sink = run_cpu_golden(sink_inputs)

    sink_logits = sink_inputs["sinks"].reshape(1, 2, 1)
    expected_lse = torch.logaddexp(no_sink["softmax_lse"], sink_logits)
    expected_out = no_sink["attn_out"].float() * torch.exp(
        no_sink["softmax_lse"] - expected_lse
    ).unsqueeze(-1)
    torch.testing.assert_close(with_sink["softmax_lse"], expected_lse)
    torch.testing.assert_close(
        with_sink["attn_out"].float(), expected_out, rtol=2e-2, atol=2e-2
    )


def test_standard_no_sink_preserves_legacy_golden_path():
    inputs = generate_inputs(_base_params(2), seed=23)
    legacy_out, x_max, x_sum = tforward(**inputs)
    result = run_cpu_golden(inputs)

    torch.testing.assert_close(result["attn_out"], legacy_out, rtol=0, atol=0)
    torch.testing.assert_close(
        result["softmax_lse"], torch.log(x_sum) + x_max, rtol=0, atol=0
    )


def test_empty_sink_sentinel_is_treated_as_absent():
    inputs = generate_inputs(_base_params(2, sinks="empty"), seed=5)
    without_sink = dict(inputs)
    without_sink["sinks"] = None

    assert tuple(inputs["sinks"].shape) == (0,)
    actual = run_cpu_golden(inputs)
    expected = run_cpu_golden(without_sink)
    torch.testing.assert_close(actual["attn_out"], expected["attn_out"])
    torch.testing.assert_close(actual["softmax_lse"], expected["softmax_lse"])


@pytest.mark.parametrize(
    "layout_q, sinks",
    [("BNSD", None), ("BSND", "empty"), ("TND", True)],
)
def test_generate_empty_query_layout_matrix(layout_q, sinks):
    params = _layout_params(
        layout_q, "PA_BNBD", b=2, n1=2, n2=1, s1=0, s2=2, sinks=sinks
    )
    inputs = generate_inputs(params, seed=3)
    result = run_cpu_golden(inputs)

    assert result["attn_out"].shape == inputs["q"].shape
    assert result["attn_out"].numel() == 0
    expected_lse_shape = (0, 2) if layout_q == "TND" else (2, 2, 0)
    assert tuple(result["softmax_lse"].shape) == expected_lse_shape


def test_partial_empty_kv_batch_only_initializes_that_batch():
    params = _base_params(2)
    params.update({"B": 2, "seqused_q": [2, 2], "seqused_kv": [0, 2]})
    inputs = generate_inputs(params, seed=11)
    result = run_cpu_golden(inputs)

    assert torch.count_nonzero(result["attn_out"][0]) == 0
    assert torch.isposinf(result["softmax_lse"][0]).all()
    assert torch.isfinite(result["softmax_lse"][1]).all()


def test_tnd_middle_empty_batch_preserves_token_offsets():
    params = _base_params(2)
    params.update({
        "B": 3,
        "layout_q": "TND",
        "layout_attn_out": "TND",
        "cu_seqlens_q": [0, 1, 1, 3],
        "seqused_q": [1, 0, 2],
        "seqused_kv": [2, 0, 2],
    })
    inputs = generate_inputs(params, seed=13)
    result = run_cpu_golden(inputs)

    assert tuple(inputs["q"].shape) == (3, 2, 64)
    assert tuple(result["attn_out"].shape) == (3, 2, 64)
    assert tuple(result["softmax_lse"].shape) == (3, 2)
    assert torch.isfinite(result["softmax_lse"]).all()


def test_return_softmax_lse_false_returns_empty_placeholder():
    params = _base_params(0)
    params["return_softmax_lse"] = False
    inputs = generate_inputs(params, seed=17)
    result = run_cpu_golden(inputs)

    assert result["softmax_lse"].dtype == torch.float32
    assert tuple(result["softmax_lse"].shape) == (0,)


def test_quant_mode2_short_kv_tail_cpu_golden():
    params = _base_params(16)
    params.update({
        "layout_q": "BSND",
        "layout_kv": "PA_BBND",
        "layout_attn_out": "BSND",
        "block_size": 64,
        "quant_compute_mode": 2,
    })
    inputs = generate_inputs(params, seed=29)
    result = run_cpu_golden(inputs)

    assert inputs["v_descale"].shape[-1] == 1
    assert tuple(result["attn_out"].shape) == tuple(inputs["q"].shape)
    assert not torch.isnan(result["attn_out"].float()).any()
    assert not torch.isnan(result["softmax_lse"]).any()


def test_causal_invalid_rows_have_zero_output_and_positive_infinite_lse():
    params = _base_params(2)
    params.update({
        "S1": 4,
        "seqused_q": [4],
        "max_seqlen_q": 4,
        "mask_mode": 3,
    })
    inputs = generate_inputs(params, seed=31)
    result = run_cpu_golden(inputs)

    assert torch.count_nonzero(result["attn_out"][:, :, :2, :]) == 0
    assert torch.isposinf(result["softmax_lse"][:, :, :2]).all()
    assert torch.isfinite(result["softmax_lse"][:, :, 2:]).all()
    assert not torch.isnan(result["attn_out"].float()).any()
    assert not torch.isnan(result["softmax_lse"]).any()


def test_softmax_flash_keeps_multiple_fully_masked_blocks_empty():
    block_sum = torch.zeros((1, 1), dtype=torch.float32)
    block_max = torch.full((1, 1), -torch.inf, dtype=torch.float32)
    block_exp = torch.ones((1, 1), dtype=torch.float32)
    masked_logits = torch.full((1, 2), -torch.inf, dtype=torch.float32)

    weights, block_sum, block_max, block_exp = softmax_flash(
        masked_logits, block_sum, block_max, block_exp, is_first=True
    )
    weights, block_sum, block_max, block_exp = softmax_flash(
        masked_logits, block_sum, block_max, block_exp, is_first=False
    )

    assert torch.count_nonzero(weights) == 0
    assert torch.count_nonzero(block_sum) == 0
    assert torch.isneginf(block_max).all()
    torch.testing.assert_close(block_exp, torch.ones_like(block_exp))


@pytest.mark.parametrize("sink_values", [[-2.0, 8.0], [5.0, -4.0]])
def test_chunked_sink_softmax_matches_flash_attn_reference(sink_values):
    logits = torch.tensor(
        [[1.0, -torch.inf, 3.0, 0.5, 2.0, -1.0, 4.0],
         [-2.0, 1.0, 0.0, 3.0, -torch.inf, 2.0, 1.5]],
        dtype=torch.float32,
    )
    values = torch.arange(21, dtype=torch.float32).reshape(7, 3) / 7
    sinks = torch.tensor(sink_values, dtype=torch.float32).reshape(2, 1)

    # Same full-row formula as flash_attn's tsoftmax.
    row_max = torch.maximum(logits.max(dim=-1, keepdim=True).values, sinks)
    exp_logits = torch.exp(logits - row_max)
    row_sum = exp_logits.sum(dim=-1, keepdim=True) + torch.exp(sinks - row_max)
    expected_out = exp_logits @ values / row_sum

    block_sum = torch.zeros((2, 1), dtype=torch.float32)
    block_max = torch.full((2, 1), -torch.inf, dtype=torch.float32)
    block_exp = torch.ones((2, 1), dtype=torch.float32)
    out = torch.zeros((2, 3), dtype=torch.float32)
    for block_idx, (start, end) in enumerate(((0, 3), (3, 5), (5, 7))):
        weights, block_sum, block_max, block_exp = softmax_flash(
            logits[:, start:end],
            block_sum,
            block_max,
            block_exp,
            is_first=block_idx == 0,
            sink=sinks if block_idx == 0 else None,
        )
        block_out = weights @ values[start:end]
        out = block_out if block_idx == 0 else out * block_exp + block_out

    torch.testing.assert_close(block_max, row_max)
    torch.testing.assert_close(block_sum, row_sum)
    torch.testing.assert_close(out / block_sum, expected_out)


@pytest.mark.parametrize(
    "sinks, error",
    [
        ([0.0], "sinks length must be 0 or num_heads_q"),
        (torch.zeros((1, 2)), "sinks must be 1-D"),
    ],
)
def test_generate_inputs_rejects_invalid_sink_shape(sinks, error):
    with pytest.raises(ValueError, match=error):
        generate_inputs(_base_params(2, sinks=sinks), seed=19)


@pytest.mark.parametrize(
    "layout_q, layout_kv, b, n1, n2, quant_mode, sinks",
    [
        ("BNSD", "PA_BNBD", 1, 2, 1, 1, None),
        ("BNSD", "PA_BBND", 2, 6, 2, 2, "empty"),
        ("BSND", "PA_BBND", 3, 4, 1, 1, None),
        ("BSND", "PA_NZ", 2, 6, 2, 2, True),
        ("TND", "PA_NZ", 2, 2, 2, 1, "empty"),
        ("TND", "PA_BNBD", 3, 6, 2, 2, True),
    ],
)
def test_physical_empty_kv_layout_head_and_sink_matrix(
        layout_q, layout_kv, b, n1, n2, quant_mode, sinks):
    params = _layout_params(
        layout_q, layout_kv, b, n1, n2, s1=2, s2=0,
        sinks=sinks, quant_compute_mode=quant_mode,
    )
    inputs = generate_inputs(params, seed=37)
    result = run_cpu_golden(inputs)

    assert inputs["k"].numel() == 0
    assert inputs["v"].numel() == 0
    assert tuple(inputs["block_table"].shape) == (b, 0)
    assert torch.count_nonzero(result["attn_out"]) == 0
    if inputs["sinks"] is None or inputs["sinks"].numel() == 0:
        assert torch.isposinf(result["softmax_lse"]).all()
    elif layout_q == "TND":
        expected_lse = inputs["sinks"].reshape(1, n1).expand(b * 2, n1)
        torch.testing.assert_close(result["softmax_lse"], expected_lse)
    else:
        expected_lse = inputs["sinks"].reshape(1, n1, 1).expand(b, n1, 2)
        torch.testing.assert_close(result["softmax_lse"], expected_lse)


@pytest.mark.parametrize(
    "layout_q, layout_kv, quant_mode, with_sink",
    [
        ("BNSD", "PA_BNBD", 1, False),
        ("BNSD", "PA_BBND", 2, True),
        ("BSND", "PA_BBND", 1, False),
        ("BSND", "PA_NZ", 2, True),
        ("TND", "PA_NZ", 1, False),
        ("TND", "PA_BNBD", 2, True),
    ],
)
def test_partial_empty_kv_batch_layout_and_sink_matrix(
        layout_q, layout_kv, quant_mode, with_sink):
    sinks = torch.linspace(-0.5, 0.5, 6) if with_sink else None
    params = _layout_params(
        layout_q, layout_kv, b=3, n1=6, n2=2, s1=3, s2=3,
        sinks=sinks, quant_compute_mode=quant_mode,
        seqused_q=[2, 1, 3], seqused_kv=[0, 3, 0],
    )
    inputs = generate_inputs(params, seed=41)
    result = run_cpu_golden(inputs)

    if layout_q == "TND":
        empty_rows = torch.cat((result["attn_out"][:2], result["attn_out"][3:]))
        empty_lse = torch.cat((result["softmax_lse"][:2], result["softmax_lse"][3:]))
        assert torch.count_nonzero(empty_rows) == 0
        assert torch.isfinite(result["softmax_lse"][2]).all()
        if with_sink:
            torch.testing.assert_close(empty_lse, sinks.reshape(1, 6).expand(5, 6))
        else:
            assert torch.isposinf(empty_lse).all()
        return

    assert torch.count_nonzero(result["attn_out"][0]) == 0
    assert torch.count_nonzero(result["attn_out"][2]) == 0
    assert torch.isfinite(result["softmax_lse"][1, :, 0]).all()
    assert torch.isposinf(result["softmax_lse"][1, :, 1:]).all()
    if with_sink:
        expected_batch0 = sinks.reshape(6, 1).expand(6, 2)
        expected_batch2 = sinks.reshape(6, 1).expand(6, 3)
        torch.testing.assert_close(result["softmax_lse"][0, :, :2], expected_batch0)
        torch.testing.assert_close(result["softmax_lse"][2], expected_batch2)
        assert torch.isposinf(result["softmax_lse"][0, :, 2]).all()
    else:
        assert torch.isposinf(result["softmax_lse"][0]).all()
        assert torch.isposinf(result["softmax_lse"][2]).all()


@pytest.mark.parametrize(
    "layout_q, layout_kv, n1, n2, quant_mode",
    [
        ("BNSD", "PA_BNBD", 2, 2, 1),
        ("BNSD", "PA_BBND", 4, 1, 2),
        ("BSND", "PA_BBND", 6, 2, 1),
        ("BSND", "PA_NZ", 2, 2, 2),
        ("TND", "PA_NZ", 4, 1, 1),
        ("TND", "PA_BNBD", 6, 2, 2),
    ],
)
def test_sink_normalization_invariant_matrix(
        layout_q, layout_kv, n1, n2, quant_mode):
    params = _layout_params(
        layout_q, layout_kv, b=2, n1=n1, n2=n2, s1=3, s2=5,
        quant_compute_mode=quant_mode, seqused_kv=[5, 3],
    )
    inputs = generate_inputs(params, seed=43)
    sink_inputs = dict(inputs)
    sinks = torch.linspace(-0.75, 0.75, n1)
    sink_inputs["sinks"] = sinks

    no_sink = run_cpu_golden(inputs)
    with_sink = run_cpu_golden(sink_inputs)
    sink_logits = sinks.reshape(1, n1) if layout_q == "TND" else sinks.reshape(1, n1, 1)
    expected_lse = torch.logaddexp(no_sink["softmax_lse"], sink_logits)
    scale = torch.exp(no_sink["softmax_lse"] - expected_lse)
    expected_out = no_sink["attn_out"].float() * _scale_output_for_layout(scale, layout_q)

    torch.testing.assert_close(with_sink["softmax_lse"], expected_lse)
    torch.testing.assert_close(
        with_sink["attn_out"].float(), expected_out, rtol=2e-2, atol=2e-2
    )


@pytest.mark.parametrize(
    "case_name",
    ["b270_sink_n1_6", "b270_sink_bnsd_g1", "b270_sink_tnd_g8"],
)
def test_stc_sink_ranges_materially_change_output_and_lse(case_name):
    raw = {
        key: value[0] if isinstance(value, list) else value
        for key, value in TestCases[case_name].items()
    }
    params = normalize_params(raw)
    params["return_softmax_lse"] = True
    with_sink_inputs = generate_inputs(params, seed=21)
    no_sink_inputs = dict(with_sink_inputs)
    no_sink_inputs["sinks"] = None

    no_sink = run_cpu_golden(no_sink_inputs)
    with_sink = run_cpu_golden(with_sink_inputs)

    out_delta = torch.linalg.vector_norm(
        with_sink["attn_out"].float() - no_sink["attn_out"].float()
    )
    out_norm = torch.linalg.vector_norm(no_sink["attn_out"].float())
    finite = torch.isfinite(no_sink["softmax_lse"]) & torch.isfinite(with_sink["softmax_lse"])
    lse_delta = (
        with_sink["softmax_lse"][finite] - no_sink["softmax_lse"][finite]
    ).abs()

    assert out_delta / out_norm.clamp_min(1e-12) > 0.1
    assert lse_delta.numel() > 0
    assert lse_delta.max() > 1.0
