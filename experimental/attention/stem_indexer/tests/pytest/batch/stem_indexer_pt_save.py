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

import argparse
import ast
import glob
import os
import re
import sys

import pandas as pd
import torch
import torch_npu

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTEST_DIR = os.path.dirname(CURRENT_DIR)
if PYTEST_DIR not in sys.path:
    sys.path.insert(0, PYTEST_DIR)

import stem_indexer_golden
import custom_ops  # 注册 torch.ops.custom.npu_stem_indexer(_metadata)


LIST_COLUMNS = {
    "q_seq_lens",
    "kv_seq_lens",
    "num_prompt_tokens",
}
INT_COLUMNS = {
    "batch_size",
    "q_heads",
    "kv_heads",
    "stem_block_size",
    "stem_stride",
    "initial_blocks",
    "window_size",
}
REQUIRED_COLUMNS = [
    "case_id",
    "testcase_name",
    "expected_result",
    "batch_size",
    "q_heads",
    "kv_heads",
    "q_seq_lens",
    "kv_seq_lens",
    "num_prompt_tokens",
    "causal",
    "alpha",
    "stem_block_size",
    "stem_stride",
    "initial_blocks",
    "window_size",
    "qflat_dtype",
    "kflat_dtype",
    "vbias_dtype",
    "special_setting",
]


def parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes")


def parse_list(value):
    if isinstance(value, list):
        return [int(item) for item in value]
    if isinstance(value, int):
        return [value]
    parsed = ast.literal_eval(str(value))
    if isinstance(parsed, int):
        return [parsed]
    return [int(item) for item in parsed]


def normalize_cell(value):
    if pd.isna(value):
        return ""
    return value


def row_to_case(row):
    case = {}
    for column in REQUIRED_COLUMNS:
        value = normalize_cell(row[column])
        if column in LIST_COLUMNS:
            case[column] = parse_list(value)
        elif column in INT_COLUMNS:
            case[column] = int(value)
        elif column == "causal":
            case[column] = parse_bool(value)
        elif column == "alpha":
            case[column] = float(value)
        elif column == "expected_result":
            case[column] = str(value).strip().upper()
        else:
            case[column] = str(value).strip()
    return case


def find_excel_files(path_pattern):
    paths = sorted(glob.glob(path_pattern))
    if not paths and os.path.isfile(path_pattern):
        paths = [path_pattern]
    return [path for path in paths if path.endswith((".xlsx", ".xls")) and not os.path.basename(path).startswith("~$")]


def load_excel_test_cases(path_pattern, sheet_name):
    excel_files = find_excel_files(path_pattern)
    if not excel_files:
        raise FileNotFoundError(f"No Excel testcase file found from path: {path_pattern}")

    test_cases = []
    for excel_file in excel_files:
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
        if missing_columns:
            raise ValueError(f"{excel_file} missing columns: {missing_columns}")
        for _, row in df.iterrows():
            if pd.isna(row["case_id"]):
                continue
            test_cases.append(row_to_case(row))
    return test_cases


def to_cpu_tensor(tensor):
    return tensor.detach().cpu() if hasattr(tensor, "detach") else tensor


def build_metadata(case, inputs):
    if case["expected_result"] == "FAIL":
        return inputs["metadata"]

    q_seq_lens = inputs["q_seq_lens"].npu()
    kv_seq_lens = inputs["kv_seq_lens"].npu()
    attrs = stem_indexer_golden.get_metadata_attrs(case)
    metadata = torch.ops.custom.npu_stem_indexer_metadata(
        q_seq_lens,
        kv_seq_lens,
        case["q_heads"],
        case["kv_heads"],
        **attrs,
    )
    torch_npu.npu.synchronize()
    return to_cpu_tensor(metadata)


def sanitize_case_name(case):
    name = f"{case['case_id']}_{case['testcase_name']}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def build_pt_payload(case):
    inputs = stem_indexer_golden.build_case_inputs(case)
    metadata = build_metadata(case, inputs)
    expected_indices = None
    expected_seq_len = None
    if case["expected_result"] == "PASS":
        expected_indices, expected_seq_len = stem_indexer_golden.stem_indexer_golden(case, inputs)

    return {
        "case": case,
        "qflat": inputs["qflat"],
        "kflat": inputs["kflat"],
        "vbias": inputs["vbias"],
        "q_seq_lens": inputs["q_seq_lens"],
        "kv_seq_lens": inputs["kv_seq_lens"],
        "num_prompt_tokens": inputs["num_prompt_tokens"],
        "metadata": metadata,
        "expected_sparse_indices": expected_indices,
        "expected_sparse_seq_len": expected_seq_len,
    }


def save_test_cases(test_cases, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    saved_count = 0
    for case in test_cases:
        try:
            payload = build_pt_payload(case)
            output_path = os.path.join(output_dir, f"{sanitize_case_name(case)}.pt")
            torch.save(payload, output_path)
            saved_count += 1
            print(f"Saved StemIndexer testcase pt: {output_path}")
        except Exception as err:
            print(f"[FAILED] Generate pt for {case.get('case_id', '<unknown>')}: {err}")
            raise
    print(f"Saved {saved_count} StemIndexer testcase pt files.")


def main():
    parser = argparse.ArgumentParser(description="Generate StemIndexer pt cases from Excel.")
    parser.add_argument("excel_path", type=str, help="Excel file path or glob pattern.")
    parser.add_argument("pt_output_dir", type=str, help="Output directory for pt files.")
    parser.add_argument("--sheet-name", type=str, default="Sheet1")
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()

    torch_npu.npu.set_device(args.device_id)
    test_cases = load_excel_test_cases(args.excel_path, args.sheet_name)
    save_test_cases(test_cases, args.pt_output_dir)


if __name__ == "__main__":
    main()
