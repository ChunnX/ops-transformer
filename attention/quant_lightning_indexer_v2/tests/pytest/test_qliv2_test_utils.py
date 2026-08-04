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

import pickle
from pathlib import Path

import pandas as pd
import pytest

from qliv2_test_utils import (
    PARAM_NAMES,
    QliV2CaseSelector,
    QliV2ResultWriter,
    ensure_comparison_passed,
    restore_float8_block_fusion,
)


def test_case_selector_preserves_requested_order(tmp_path):
    for name in ("case_10.pt", "case_2.pt", "case_1.pt"):
        (tmp_path / name).touch()

    natural = QliV2CaseSelector.resolve(tmp_path)
    assert [Path(path).name for path in natural] == ["case_1.pt", "case_2.pt", "case_10.pt"]

    indexed = QliV2CaseSelector.resolve(tmp_path, case_indexes="3,1,2-2")
    assert [Path(path).name for path in indexed] == ["case_10.pt", "case_1.pt", "case_2.pt"]

    named = QliV2CaseSelector.resolve(tmp_path, case_names="case_2,case_1.pt")
    assert [Path(path).name for path in named] == ["case_2.pt", "case_1.pt"]


def test_case_selector_rejects_ambiguous_or_invalid_selection(tmp_path):
    (tmp_path / "case_1.pt").touch()
    with pytest.raises(ValueError, match="cannot be specified together"):
        QliV2CaseSelector.resolve(tmp_path, case_names="case_1", case_indexes="1")
    with pytest.raises(ValueError, match="out of range"):
        QliV2CaseSelector.resolve(tmp_path, case_indexes="2")


def test_result_writer_uses_readable_name_and_migrates_legacy_result(tmp_path):
    params = list(range(len(PARAM_NAMES)))
    params[PARAM_NAMES.index("qk_dtype")] = "INT8"
    params[PARAM_NAMES.index("layout_query")] = "BSND"
    params[PARAM_NAMES.index("layout_key")] = "PA_BBND"
    name = QliV2ResultWriter.case_name(params)
    assert name == QliV2ResultWriter.case_name(params)
    assert name == (
        "QLI_B0_S11_S22_N15_N26_D7_BSND_PA_BBND_INT8_"
        "QM20_SM24_CR30_K23_RV31"
    )
    assert QliV2ResultWriter.case_name(
        params,
        explicit_name="quant li/default:a5 v2",
    ) == "quant_li_default_a5_v2"

    row = QliV2ResultWriter.row(name, params, "Pass", 100.0)
    output = tmp_path / "result.xlsx"
    legacy = pd.DataFrame([{key: value for key, value in row.items() if key != "return_value"}])
    legacy.to_excel(output, index=False)

    QliV2ResultWriter.append(output, row)
    result = pd.read_excel(output)
    assert list(result.columns) == list(row.keys())
    assert len(result) == 2
    assert result.iloc[1]["return_value"] == params[PARAM_NAMES.index("return_value")]


def test_comparison_failure_raises_serializable_assertion():
    ensure_comparison_passed("case_pass", "Pass", 100.0)

    with pytest.raises(AssertionError, match="case_index_fail.*index result=Failed") as caught:
        ensure_comparison_passed("case_index_fail", "Failed", 97.5)
    restored = pickle.loads(pickle.dumps(caught.value))
    assert str(restored) == str(caught.value)

    with pytest.raises(AssertionError, match="case_value_fail.*value result=Failed"):
        ensure_comparison_passed("case_value_fail", "Pass", 100.0, "Failed", 90.0)


def test_restore_float8_block_fusion_preserves_scale_bytes():
    import torch

    key = torch.tensor([1.0, -2.0], dtype=torch.float16).reshape(1, 1, 1, 2)
    scale = torch.tensor([1.25], dtype=torch.float32).reshape(1, 1, 1)
    key_bytes = key.to(torch.float8_e4m3fn).view(torch.uint8).reshape(1, -1)
    scale_bytes = scale.view(torch.uint8).reshape(1, -1)
    expected_bytes = torch.cat((key_bytes, scale_bytes), dim=1)

    legacy = torch.zeros(expected_bytes.shape, dtype=torch.float16)
    restored_legacy = restore_float8_block_fusion(
        {
            "key": key,
            "key_dequant_scale": scale,
            "blockFusion": legacy,
        }
    )
    assert restored_legacy.dtype == torch.float8_e4m3fn
    assert torch.equal(restored_legacy.view(torch.uint8), expected_bytes)

    restored_packed = restore_float8_block_fusion(
        {"blockFusion": expected_bytes}
    )
    assert restored_packed.dtype == torch.float8_e4m3fn
    assert torch.equal(restored_packed.view(torch.uint8), expected_bytes)
