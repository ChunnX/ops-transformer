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

"""BSND/BNSD feature cases using the existing MXFP8 CSV and golden pipeline."""

import csv
from pathlib import Path

import pytest


CASE_FILE = Path(__file__).with_name("qbsa_mxfp8_layout_stc.csv")


def _case_names():
    with CASE_FILE.open(encoding="utf-8-sig", newline="") as stream:
        return [
            row["name"]
            for row in csv.DictReader(stream)
            if row["enable"].strip().lower() in ("true", "1", "yes")
        ]


@pytest.mark.ci
@pytest.mark.parametrize("case_name", _case_names())
def test_qbsa_mxfp8_layout(case_name, tmp_path):
    import qbsa_mxfp8_golden as golden

    case = golden.load_test_cases(str(CASE_FILE), use_csv=True)[case_name]
    result = golden.run_one_case(
        case_name,
        case,
        {"gen", "cpu", "npu", "compare"},
        case_name,
        str(tmp_path),
    )
    if result["status"] != "Pass":
        pytest.fail(f"case failed: {case_name}, result={result}")
