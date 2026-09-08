#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU regressions for independent ori/cmp test-case length defaults."""

import pytest

from utils import generate_case_with_default_param


def make_case(**overrides):
    case = {
        "layout_q": "TND",
        "layout_kv": "TND",
        "B": 2,
        "T1": 4,
        "T2": 64,
        "T3": None,
        "S1": 2,
        "S2": 32,
        "cu_seqlens_q": [0, 2, 4],
        "cu_seqlens_ori_kv": [0, 32, 64],
        "cu_seqlens_cmp_kv": None,
        "seqused_ori_kv": None,
        "seqused_cmp_kv": None,
        "cmp_residual_kv": None,
        "cmp_ratio": 4,
        "block_size1": 16,
        "block_size2": 16,
        "random_seq": False,
    }
    case.update(overrides)
    return case


def test_explicit_used_lengths_and_padded_t3_are_preserved():
    source = make_case(
        seqused_ori_kv=[20, 25],
        seqused_cmp_kv=[3, 9],
        T3=20,
        cmp_residual_kv=[1, 2],
    )
    result = generate_case_with_default_param(source)

    assert result["seqused_ori_kv"] == [20, 25]
    assert result["seqused_cmp_kv"] == [3, 9]
    assert result["cu_seqlens_cmp_kv"] == [0, 3, 12]
    assert result["T3"] == 20
    assert result["cmp_residual_kv"] == [1, 2]
    assert source["cu_seqlens_cmp_kv"] is None


@pytest.mark.parametrize("ori_lengths", [[32, 32], [11, 53]])
def test_t3_only_generates_cmp_lengths_independently(ori_lengths):
    result = generate_case_with_default_param(
        make_case(seqused_ori_kv=ori_lengths, T3=11)
    )

    assert result["seqused_cmp_kv"] == [5, 6]
    assert result["cu_seqlens_cmp_kv"] == [0, 5, 11]
    assert result["T3"] == 11
    assert result["cmp_residual_kv"] is None


def test_cmp_cumulative_lengths_supply_only_missing_fields():
    result = generate_case_with_default_param(
        make_case(cu_seqlens_cmp_kv=[0, 5, 18], T3=24)
    )

    assert result["cu_seqlens_cmp_kv"] == [0, 5, 18]
    assert result["seqused_cmp_kv"] == [5, 13]
    assert result["T3"] == 24


def test_cmp_cumulative_storage_does_not_replace_explicit_used_lengths():
    result = generate_case_with_default_param(
        make_case(cu_seqlens_cmp_kv=[0, 5, 18], seqused_cmp_kv=[3, 9])
    )

    assert result["cu_seqlens_cmp_kv"] == [0, 5, 18]
    assert result["seqused_cmp_kv"] == [3, 9]
    assert result["T3"] == 18


@pytest.mark.parametrize("layout", ["BSND", "PA_BBND"])
def test_non_tnd_explicit_cmp_lengths_are_independent(layout):
    result = generate_case_with_default_param(
        make_case(layout_kv=layout, seqused_cmp_kv=[33, 1])
    )

    assert result["seqused_cmp_kv"] == [33, 1]
    assert result["cmp_residual_kv"] is None
    if layout == "PA_BBND":
        assert result["block_num1"] == 4
        assert result["block_num2"] == 4


def test_legacy_missing_cmp_lengths_keep_shorthand_without_inferred_residual():
    result = generate_case_with_default_param(make_case(seqused_ori_kv=[17, 22]))

    assert result["seqused_ori_kv"] == [17, 22]
    assert result["seqused_cmp_kv"] == [4, 5]
    assert result["cu_seqlens_cmp_kv"] == [0, 4, 9]
    assert result["T3"] == 9
    assert result["cmp_residual_kv"] is None
