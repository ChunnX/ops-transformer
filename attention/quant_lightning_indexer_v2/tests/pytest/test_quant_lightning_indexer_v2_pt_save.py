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

import pandas as pd
import pytest

from batch.quant_lightning_indexer_v2_pt_save import (
    PARAM_COLUMNS,
    _normalize_redline_row,
    load_excel_test_cases,
)


def _redline_row(**overrides):
    row = {
        "Testcase_Name": "QLI_L0_11_57_1_1248_14336_128_BSND_BSND_fp4_e2m1_000010",
        "B": 11,
        "Q_S": 1248,
        "KV_S": 14336,
        "Q_N": 57,
        "KV_N": 1,
        "D": 128,
        "block_size": None,
        "topk": 6502,
        "max_seqlen_q": 1163,
        "mask_mode": 0,
        "cmp_ratio": 2,
        "quant_mode": 5,
        "return_value": 0,
        "q_layout": "BSND",
        "k_layout": "BSND",
        "q_shape": "11,1248,57,128",
        "q_dtype": "fp4_e2m1",
        "q_datarange": "1,1",
        "k_shape": "11,14336,1,128",
        "k_dtype": "fp4_e2m1",
        "k_datarange": "2,2",
        "weight_dtype": "FP32",
        "weight_datarange": "1,2",
        "q_descale_dtype": "FP8_E8M0",
        "q_descale_datarange": "2**-31,2**4",
        "k_descale_dtype": "FP8_E8M0",
        "k_descale_datarange": "2**-31,2**4",
        "cu_seqlens_q_dtype": None,
        "cu_seqlens_q_datarange": None,
        "cu_seqlens_k_dtype": None,
        "cu_seqlens_k_datarange": None,
        "seqused_q_dtype": "INT32",
        "seqused_q_datarange": "1163,716,723,723,291,150,723,723,723,723,723",
        "seqused_k_dtype": None,
        "seqused_k_datarange": None,
        "cmp_residual_k_dtype": None,
        "cmp_residual_k_datarange": None,
        "output_idx_offset_dtype": "INT32",
        "output_idx_offset": "2,10",
    }
    row.update(overrides)
    return row


def _as_params(case):
    return dict(zip(("Testcase_Name", *PARAM_COLUMNS), case))


def test_normalize_redline_mxfp4_bsnd_case():
    params = _as_params(_normalize_redline_row(pd.Series(_redline_row())))

    assert params["q_t_size"] is None
    assert params["k_t_size"] is None
    assert params["block_num"] is None
    assert params["qk_dtype"] == "FLOAT4_E2M1"
    assert params["dequant_dtype"] == "FLOAT8_E8M0"
    assert params["actual_seq_dtype"] == "INT32"
    assert params["seqused_q"] == [1163, 716, 723, 723, 291, 150, 723, 723, 723, 723, 723]
    assert params["q_scale_datarange"] == [2**-31, 2**4]
    assert params["output_idx_offset"] == "[2, 10]"


def test_normalize_redline_tnd_pa_case():
    row = _redline_row(
        Testcase_Name="QLI_L0_2_48_1_6748_14852_128_TND_PA_BBND_FP8_E4M3FN_000004",
        B=2,
        Q_S=6748,
        KV_S=14852,
        Q_N=48,
        block_size=240,
        topk=3100,
        max_seqlen_q=6748,
        mask_mode=3,
        cmp_ratio=8,
        quant_mode=3,
        return_value=1,
        q_layout="TND",
        k_layout="PA_BBND",
        q_shape="12350,48,128",
        q_dtype="FP8_E4M3FN",
        q_datarange="-20,20",
        k_shape="63,240,1,128",
        k_dtype="FP8_E4M3FN",
        k_datarange="0.01,1",
        weight_datarange="-1,1",
        cu_seqlens_q_dtype="INT32",
        cu_seqlens_q_datarange="0,5602,12350",
        seqused_q_datarange="5118,6748",
        seqused_k_dtype="INT32",
        seqused_k_datarange="14851,160",
        cmp_residual_k_dtype="INT32",
        cmp_residual_k_datarange="7,2",
        output_idx_offset="0,10",
    )
    params = _as_params(_normalize_redline_row(pd.Series(row)))

    assert params["q_t_size"] == 12350
    assert params["k_t_size"] is None
    assert params["block_num"] == 63
    assert params["qk_dtype"] == "FLOAT8_E4M3FN"
    assert params["cu_seqlens_q"] == [0, 5602, 12350]
    assert params["seqused_k"] == [14851, 160]
    assert params["cmp_residual_k"] == [7, 2]


def test_load_excel_detects_redline_schema(tmp_path, monkeypatch):
    excel_path = tmp_path / "redline.xlsx"
    excel_path.touch()
    monkeypatch.setattr(pd, "read_excel", lambda *_args, **_kwargs: pd.DataFrame([_redline_row()]))

    cases = load_excel_test_cases(str(excel_path), "Sheet1")

    assert len(cases) == 1
    assert cases[0][0].endswith("000010")


def test_tnd_total_size_must_match_cu_seqlens():
    row = _redline_row(
        q_layout="TND",
        q_shape="20,4,128",
        cu_seqlens_q_dtype="INT32",
        cu_seqlens_q_datarange="0,5,11",
    )

    with pytest.raises(ValueError, match="does not match tensor total size"):
        _normalize_redline_row(pd.Series(row))
