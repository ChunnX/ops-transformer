#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pytest
import torch

from .transform import (
    ActualInputSemanticOracle,
    BatchCaseTransformer,
    InvalidTransformError,
)


def make_unbounded_swa_case(q_length, ori_length):
    return {
        "params": {},
        "metadata_input": {"batch_size": 1},
        "op_input": {
            "q": torch.zeros((q_length, 1, 1)),
            "q_descale": None,
            "cu_seqlens_q": torch.tensor([0, q_length], dtype=torch.int32),
            "seqused_q": torch.tensor([q_length], dtype=torch.int32),
            "seqused_ori_kv": torch.tensor([ori_length], dtype=torch.int32),
            "seqused_cmp_kv": None,
            "cmp_kv": None,
            "sinks": torch.zeros((1,)),
            "ori_mask_mode": 4,
            "cmp_mask_mode": 0,
            "ori_win_left": -1,
            "ori_win_right": -1,
            "layout_q": "TND",
            "layout_kv": "PA_BBND",
        },
    }


def test_unbounded_mode4_rejects_token_split_that_shortens_kv():
    baseline = make_unbounded_swa_case(q_length=2, ori_length=4096)
    derived = make_unbounded_swa_case(q_length=1, ori_length=4095)

    assert ActualInputSemanticOracle._windows(baseline, 0, 0)["ori"] == (0, 4096)
    assert ActualInputSemanticOracle._windows(derived, 0, 0)["ori"] == (0, 4095)
    with pytest.raises(InvalidTransformError, match="different mask window"):
        ActualInputSemanticOracle.validate_mapped_tokens(
            baseline, derived, [(0, 0, 0, 0)]
        )


def make_independent_cmp_case(
    ori_length=10, cmp_length=4, residual=1, ori_mode=3, cmp_mode=3
):
    q_length = 4
    return {
        "B": 1,
        "layout_q": "BSND",
        "layout_kv": "BSND",
        "params": {},
        "cpu_output": torch.zeros((1, q_length, 1, 1)),
        "metadata_input": {
            "B": 1,
            "N1": 1,
            "N2": 1,
            "D": 1,
            "max_seqlen_q": q_length,
            "max_seqlen_ori_kv": ori_length,
            "max_seqlen_cmp_kv": cmp_length,
            "layout_q": "BSND",
            "layout_kv": "BSND",
        },
        "input": {
            "q": torch.arange(q_length, dtype=torch.float32).reshape(1, -1, 1, 1),
            "ori_kv": torch.arange(ori_length, dtype=torch.float32).reshape(
                1, -1, 1, 1
            ),
            "cmp_kv": torch.arange(cmp_length, dtype=torch.float32).reshape(
                1, -1, 1, 1
            ),
            "cu_seqlens_q": None,
            "seqused_q": torch.tensor([q_length], dtype=torch.int32),
            "seqused_ori_kv": torch.tensor([ori_length], dtype=torch.int32),
            "seqused_cmp_kv": torch.tensor([cmp_length], dtype=torch.int32),
            "cmp_residual_kv": (
                torch.tensor([residual], dtype=torch.int32)
                if residual is not None
                else None
            ),
            "cmp_ratio": 4,
            "sinks": torch.zeros((1,)),
            "softmax_scale": 1.0,
            "ori_mask_mode": ori_mode,
            "cmp_mask_mode": cmp_mode,
            "ori_win_left": -1,
            "ori_win_right": -1,
            "layout_q": "BSND",
            "layout_kv": "BSND",
        },
    }


@pytest.mark.parametrize("ori_length", [1, 100])
@pytest.mark.parametrize(
    "cmp_mode,residual,expected", [(0, None, 4), (3, None, 3), (3, 1, 4)]
)
def test_cmp_window_uses_its_own_length_and_residual(
    ori_length, cmp_mode, residual, expected
):
    case = make_independent_cmp_case(
        ori_length=ori_length, cmp_mode=cmp_mode, residual=residual
    )
    assert ActualInputSemanticOracle._windows(case, 0, 2)["cmp"] == (0, expected)


@pytest.mark.parametrize("ori_mode,cmp_mode", [(3, 3), (0, 3), (3, 0)])
@pytest.mark.parametrize("transform_kind", ["partition", "shorter", "longer"])
def test_position_alignment_keeps_kv_contexts_independent(
    ori_mode, cmp_mode, transform_kind
):
    # A short unmasked ori context must not block cmp's causal alignment.
    ori_length = 1 if ori_mode == 0 else 10
    baseline = make_independent_cmp_case(
        ori_length=ori_length, ori_mode=ori_mode, cmp_mode=cmp_mode
    )
    transformer = BatchCaseTransformer(baseline)
    if transform_kind == "partition":
        cases = transformer.token_partition(0, [2, 2])
        deltas = [-2, 0]
    else:
        target_length = 2 if transform_kind == "shorter" else 6
        cases = [transformer.shape_change_common_prefix(0, 1, target_length - 1)]
        deltas = [target_length - 4]

    for case, delta in zip(cases, deltas):
        tensors = case.input_data["input"]
        assert tensors["seqused_ori_kv"].tolist() == [
            ori_length + delta if ori_mode == 3 else ori_length
        ]
        cmp_length, residual = divmod(17 + delta, 4) if cmp_mode == 3 else (4, 1)
        assert tensors["seqused_cmp_kv"].tolist() == [cmp_length]
        assert tensors["cmp_residual_kv"].tolist() == [residual]
        assert case.transform_meta["semantic_oracle"]["pass"]


@pytest.mark.parametrize("invalid_context", ["ori", "cmp"])
def test_position_alignment_rejects_each_negative_context(invalid_context):
    case = make_independent_cmp_case(
        ori_length=1 if invalid_context == "ori" else 10,
        cmp_length=0 if invalid_context == "cmp" else 4,
        residual=1,
    )
    transformer = BatchCaseTransformer(case)
    expected = (
        "negative ori KV" if invalid_context == "ori" else "negative restored cmp KV"
    )
    with pytest.raises(InvalidTransformError, match=expected):
        transformer.token_partition(0, [2, 2])
