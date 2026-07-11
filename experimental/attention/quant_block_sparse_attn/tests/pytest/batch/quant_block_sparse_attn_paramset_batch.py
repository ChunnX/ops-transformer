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

import os

EXCEL_COLUMNS = [
    "case_name", "enable",
    "B", "N1", "N2", "S1", "S2", "D",
    "sparse_q_block_size", "sparse_kv_block_size",
    "sparse_count", "sparse_pattern",
    "block_table_pattern",
    "layout_q", "layout_kv", "layout_sparse_indices", "layout_out",  "output_dtype",
    "actlen_mode", "quant_mode", "mask_mode",
    "scale_pattern", "p_scale_value", "softmax_scale",
    "return_softmax_lse", "seed",
]

PA_BLOCK_PADDING_BYTES = 0

EXCEL_FILENAME = "cases.csv"

_INT_COLS = {"B", "N1", "N2", "S1", "S2", "D",
             "sparse_q_block_size", "sparse_kv_block_size",
             "sparse_count", "quant_mode", "mask_mode", "seed"}
_FLOAT_COLS = {"p_scale_value", "softmax_scale"}
_BOOL_COLS = {"return_softmax_lse"}


def _get_excel_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), EXCEL_FILENAME)


def _get_dtype_map():
    import torch as _torch
    return {
        "bfloat16": _torch.bfloat16,
        "float16": _torch.float16,
        "float32": _torch.float32,
    }


def _parse_bool(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().upper() == "TRUE"
    return bool(val)


def _parse_int(val):
    if val is None:
        return None
    return int(val)


def _parse_float(val):
    if val is None:
        return None
    return float(val)


def load_cases_from_csv(csv_path=None):
    import csv
    if csv_path is None:
        csv_path = _get_excel_path()
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return {}
    header = [c.strip() for c in rows[0]]
    dtype_map = _get_dtype_map()
    import torch as _torch
    cases = {}
    for row in rows[1:]:
        record = dict(zip(header, row))
        case_name = record.get("case_name")
        if case_name is None:
            continue
        case_name = str(case_name).strip()
        if not case_name:
            continue

        if not _parse_bool(record.get("enable", False)):
            continue

        params = {}
        params["Testcase_Name"] = [case_name]
        for col in EXCEL_COLUMNS:
            if col in ("case_name", "enable"):
                continue
            val = record.get(col)
            if val == "":
                val = None
            if col in _INT_COLS:
                val = _parse_int(val)
            elif col in _FLOAT_COLS:
                val = _parse_float(val)
            elif col in _BOOL_COLS:
                val = _parse_bool(val)
            elif col == "output_dtype":
                val = dtype_map.get(str(val).strip(), _torch.bfloat16) if val is not None else _torch.bfloat16
            params[col] = [val]

        params["pa_block_padding_bytes"] = [PA_BLOCK_PADDING_BYTES]
        params["S1EQS2"] = [params["S1"][0] == params["S2"][0]]
        cases[case_name] = params
    return cases


_excel_path = _get_excel_path()
TEST_PARAMS = load_cases_from_csv(_excel_path)
ENABLED_PARAMS = [TEST_PARAMS[key] for key in TEST_PARAMS.keys()]
